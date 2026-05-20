"""§6 entry — Phase2 시작 시 호출하는 build-or-load 진입점.

Skill-wise vector DB ``B_t^{(m)}`` 의 lifecycle (spec §14):

  1. session_dir 에 캐시 파일(``skill_wise_vector_db.npz``) 이 있으면 그대로 로드.
  2. 없으면 ``phase1_trained_vla_path`` 로 encoder + ``phase1_dataset_path``
     LeRobot dataset 으로 §6 Step 3-4 (re-embedding) 수행 → SkillVectorDB 생성·
     저장 → 반환.

이름은 ``skill_wise`` — spec §14 의 ``B_t^{(m)} = P_phase1 ∪ D_phase2,t`` 에 따라
같은 DB 가 phase2 진행 중 누적된다. 처음 build 시점엔 P_phase1 만 들어있지만 그
파일 자체가 phase1 전용은 아니다.

``phase1_trained_vla_path`` 는 원래 §6 Step 1-2 로 Phase1 raw 에서 학습·freeze
된 encoder 경로지만, 아직 학습 안 된 동안엔 임시로 pretrained VLA 체크포인트
경로를 넣어 파이프라인 동작을 검증할 수 있다 (``PretrainedVLAStateEncoder``).
나중에 Phase1-trained encoder 가 준비되면 같은 인자에 그 경로만 바꿔 swap.

테스트에서는 ``raw_dataset`` / ``encoder`` 를 직접 주입해 LeRobot·VLA 의존성을
우회할 수 있다 (실제 LeRobot dataset 이나 GPU 가 없어도 검증 가능).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from method3.phase2_mi_selection.vector_db import SkillVectorDB
from method3.reembedding.seed_builder import (
    ReembeddingConfig,
    build_phase1_vector_db,
)
from method3.reembedding.vla_encoder import VLAStateEncoder


# skill-wise vector DB 캐시 파일명. spec §14 의 ``B_t^{(m)} = P_phase1 ∪ D_phase2,t``
# 가 같은 DB 에 누적되므로 "phase1" 한정으로 오해되지 않도록 skill-wise 로 명명.
_DEFAULT_VECTOR_DB_FILENAME = "skill_wise_vector_db.npz"


def build_or_load_phase1_vector_db(
    session_dir: str | Path,
    *,
    phase1_trained_vla_path: str | Path | None = None,
    phase1_dataset_path: str | Path | None = None,
    vector_db_filename: str = _DEFAULT_VECTOR_DB_FILENAME,
    rebuild: bool = False,
    encoder_kwargs: dict | None = None,
    reembedding_config: ReembeddingConfig | None = None,
    # 테스트·커스텀 주입용:
    raw_dataset=None,
    encoder: VLAStateEncoder | None = None,
    observation_loader: Callable[[dict], object] | None = None,
) -> SkillVectorDB:
    """§6 — session 의 P_phase1 vector DB 를 로드하거나 새로 구축한다.

    Args:
        session_dir: vector DB 캐시 위치 (results/session_*/).
        phase1_trained_vla_path: §6 Step 1-2 의 frozen VLA encoder 체크포인트.
            아직 미학습이면 pretrained VLA 경로를 임시로 넣어도 된다.
        phase1_dataset_path: Phase1 raw dataset (LeRobot repo_id 또는 local path).
        vector_db_filename: 캐시 파일명 (session_dir 내).
        rebuild: True → 캐시 있어도 다시 구축.
        encoder_kwargs: ``PretrainedVLAStateEncoder`` 에 넘길 추가 인자
            (device, autocast_dtype, state_dim, family 등).
        reembedding_config: §6 re-embedding 파라미터 (dct_coeffs 등).
        raw_dataset / encoder / observation_loader: 직접 주입 — 주면
            ``phase1_*`` 경로 인자를 무시하고 그대로 사용 (테스트·커스텀용).

    Returns:
        Phase2 가 reference buffer 로 쓸 SkillVectorDB (``P_phase1``).
    """
    vdb_path = Path(session_dir) / vector_db_filename

    # ── 1. 캐시 hit ────────────────────────────────────────────────
    if vdb_path.exists() and not rebuild:
        db = SkillVectorDB()
        db.load(vdb_path)
        print(f"[reembed] cached vector DB → {vdb_path} "
              f"({db.total_size()} entries, {len(db.skill_ids())} skills)")
        return db

    # ── 2. miss → re-embed ─────────────────────────────────────────
    # raw_dataset / encoder / loader 우선순위: 직접 주입 > 경로 인자.
    own_encoder = False
    try:
        if raw_dataset is None:
            if phase1_dataset_path is None:
                raise ValueError(
                    "phase1_dataset_path is required when no cached vector DB "
                    "exists and raw_dataset is not supplied")
            from method3.reembedding.lerobot_adapter import LeRobotPhase1RawAdapter
            raw_dataset = LeRobotPhase1RawAdapter(phase1_dataset_path)

        if encoder is None:
            if phase1_trained_vla_path is None:
                raise ValueError(
                    "phase1_trained_vla_path is required when no cached vector "
                    "DB exists and encoder is not supplied")
            from method3.reembedding.pretrained_encoder import (
                PretrainedVLAStateEncoder,
            )
            encoder = PretrainedVLAStateEncoder(
                phase1_trained_vla_path, **(encoder_kwargs or {}))
            own_encoder = True

        if observation_loader is None:
            observation_loader = getattr(raw_dataset, "load_observation", None)
            if observation_loader is None:
                raise ValueError(
                    "observation_loader is required (raw_dataset has no "
                    "`load_observation` method to fall back on)")

        # reembedding_config 가 caller 에서 명시 안 됐으면 phase2_config.yaml 의
        # reembedding 섹션을 *자동* 로드 (subgoal_filter_radius_m 같은 옵션이
        # 어떤 caller 든 자동 적용되도록).
        if reembedding_config is None:
            try:
                from method3.config import load_phase2_config
                from pathlib import Path as _P
                _yaml = _P(__file__).resolve().parents[2] / "pipeline_config" / "phase2_config.yaml"
                if _yaml.exists():
                    _acq = load_phase2_config(_yaml, phase1_raw_dir="", phase2_raw_dir="")
                    reembedding_config = _acq.reembedding
                    print(f"[reembed] reembedding_config ← {_yaml} "
                          f"(subgoal_filter_radius_m={reembedding_config.subgoal_filter_radius_m})")
            except Exception as _e:
                print(f"[reembed] yaml fallback failed ({_e}); using defaults")
        print(f"[reembed] building skill-wise vector DB (initial = P_phase1) ... "
              f"(raw_dataset={len(raw_dataset)} entries → {vdb_path})")
        db = build_phase1_vector_db(
            raw_dataset, encoder, observation_loader,
            reembedding_config or ReembeddingConfig(),
        )
    finally:
        # 임시로 만든 encoder 만 close (외부에서 주입한 encoder 는 caller 가 관리)
        if own_encoder and encoder is not None and hasattr(encoder, "close"):
            encoder.close()

    vdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(vdb_path)
    print(f"[reembed] saved → {vdb_path} "
          f"({db.total_size()} entries, {len(db.skill_ids())} skills)")
    return db
