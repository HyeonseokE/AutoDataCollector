"""Server-side setup helper for method3 phase2 useful-OOD acquisition.

Replaces the old ``preselective_filter.integration.setup_preselective_filter``
(IG·AC + FaissBufferStore) with the new spec
``final_method3_spec_useful_ood_updated §11-§13`` server-side stack:

  - SkillVectorDB                        — accumulating vector DB (§14)
  - Phase2MISelector + Phase2MIConfig    — M_MI computation + Useful-OOD rule
  - LeRobotVLAInformativenessScorer (옵셔널) — U_VLA (§12), if vla_informativeness.enabled
  - VLAKeyExtractor (preselective_filter.vectorDB.vla_embedding) — shared encoder

The encoder is still the frozen VLA pulled in via ``make_vla_key_extractor``
(pi0 / pi05 / smolvla / groot auto-dispatch) — that path is intentionally
preserved because it is the *only* piece of preselective_filter that survives
the refactor (everything else is method3-native).

Persistence model — the server-local SkillVectorDB lives in a sidecar npz file
so the server can be restarted without losing the buffer (옛 FaissBufferStore
의 server-local persistence 와 동등). On first boot the file does not exist
→ a fresh empty DB.

Returns ``Method3ServerStack`` so server.py just unpacks one bundle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from method3.phase2_mi_selection import (
    LeRobotVLAInformativenessScorer,
    Phase2MIConfig,
    Phase2MISelector,
    SkillVectorDB,
)
from method3.config import load_phase2_config, resolve_repo_path


@dataclass
class Method3ServerStack:
    """Bundle returned by ``setup_method3_phase2_server``."""

    selector: Phase2MISelector
    encoder: Any                     # VLAKeyExtractor — frozen VLA, share-able
    vla_scorer: Optional[LeRobotVLAInformativenessScorer]  # None if disabled
    db: SkillVectorDB
    db_path: Path                    # sidecar npz for persistence
    config: Phase2MIConfig           # for Ready() display
    encoder_lock_payload: dict       # passthrough metadata for debug


def _load_phase2_yaml(recording_cfg: dict, phase2_yaml_path: Optional[Path]) -> dict:
    """Read phase2_config.yaml (raw dict). recording_cfg 의 ``skill_planner_transport``
    / ``preselective_filter`` legacy block 은 무시 — 새 acquisition 설정은 phase2 쪽."""
    if phase2_yaml_path is None or not phase2_yaml_path.exists():
        return {}
    try:
        with open(phase2_yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[method3_setup] failed to read {phase2_yaml_path}: {e}")
        return {}


def setup_method3_phase2_server(
    recording_cfg: dict,
    *,
    phase2_yaml: Optional[str | Path] = None,
    db_save_dir: Optional[str | Path] = None,
    db_filename: str = "server_skill_wise_vector_db.npz",
) -> Optional[Method3ServerStack]:
    """Build the method3-phase2 server-side stack from yaml.

    Args:
        recording_cfg: recording_config_ws*.yaml 의 dict (encoder checkpoint 등
            기존 preselective_filter 영역에서 가져오던 키 — 호환 위해 유지).
        phase2_yaml: ``pipeline_config/phase2_config.yaml`` 경로 (mi_selection +
            vla_informativeness + skill_planner_transport 블록 source).
            None 이면 server 호스트의 기본 위치 시도.
        db_save_dir: server-local SkillVectorDB persistence 경로 root. None 이면
            ``./grpc_server/buffer/`` 사용.
        db_filename: 캐시 파일명.

    Returns:
        Method3ServerStack 또는 None (preselective_filter / encoder 가 yaml 에서
        비활성화돼 있으면).
    """
    # 2026-05-21: 옛 phase2_server_infer_settings.yaml 를 phase2_config.yaml 으로
    # 통합. server 운영 설정 (transport/policy/selector/buffer/remote) 도 phase2_config
    # 의 top-level 섹션에서 읽음. VLA checkpoint 는 phase2_config.yaml 의
    # ``phase1_trained_vla_path`` 가 *유일한 SoT* — client/server mismatch 방지.
    #
    # ⚠️ phase2_yaml=None 일 때 *반드시* default phase2_config.yaml 으로 fallback.
    # server.py main() 이 --phase2-yaml 인자를 안 전달해도 ph2_raw 가 통합된 server
    # 섹션을 읽어야 server boot 가능 (없으면 ckpt None → setup return None → fail).
    psf_raw: dict = {}
    if phase2_yaml is None:
        _default_ph2 = Path(__file__).resolve().parent.parent / "pipeline_config" / "phase2_config.yaml"
        if _default_ph2.exists():
            phase2_yaml = _default_ph2
            print(f"[method3_setup] phase2_yaml default ← {phase2_yaml}")
    ph2_yaml_path = Path(phase2_yaml) if phase2_yaml else None
    ph2_raw = _load_phase2_yaml(recording_cfg, ph2_yaml_path)
    psf_raw = dict(
        ph2_raw.get("skill_planner_transport")
        or recording_cfg.get("preselective_filter")
        or {}
    )
    # 서버 운영 설정 — phase2_config.yaml 의 top-level 섹션에서 머지.
    t = ph2_raw.get("transport") or {}
    psf_raw.setdefault("transport_address", t.get("address"))
    psf_raw.setdefault("transport_timeout_s", t.get("timeout_s"))
    psf_raw.setdefault("debug_verbose", t.get("debug_verbose", False))
    if "policy" not in psf_raw and "policy" in ph2_raw:
        psf_raw["policy"] = ph2_raw["policy"]
    if "selector" not in psf_raw and "selector" in ph2_raw:
        psf_raw["selector"] = ph2_raw["selector"]
    buf = ph2_raw.get("buffer") or {}
    if db_save_dir is None and buf.get("save_dir"):
        db_save_dir = buf["save_dir"]
    if buf.get("filename"):
        db_filename = buf["filename"]

    if not psf_raw:
        return None

    # VLA checkpoint 의 *유일한 source* — phase2_config.yaml.phase1_trained_vla_path.
    # 옛 policy.checkpoint 키는 phase2_config 의 phase1_trained_vla_path 와 mismatch
    # race 가 잦아 *제거*. policy 안의 다른 키 (family/device/autocast_dtype/state_weight)
    # 는 server runtime config 로 유지.
    policy_cfg = psf_raw.get("policy") or {}
    # repo-relative path 도 허용 (local/remote portable yaml 의도).
    ckpt = resolve_repo_path(
        ph2_raw.get("phase1_trained_vla_path") or policy_cfg.get("checkpoint")
    )
    if not ckpt:
        print("[method3_setup] no VLA checkpoint — set phase1_trained_vla_path in phase2_config.yaml; disabling")
        return None

    # 1) VLA encoder (frozen) — shared dep, encode + (옵션) denoise loss
    from method3.vectorDB.vla_embedding import make_vla_key_extractor
    encoder = make_vla_key_extractor(
        checkpoint=str(ckpt),
        device=str(policy_cfg.get("device", "cuda")),
        autocast_dtype=str(policy_cfg.get("autocast_dtype", "bfloat16")),
        debug_verbose=bool(psf_raw.get("debug_verbose", False)),
        family=policy_cfg.get("family"),
    )

    # 2) Phase2MISelector + SkillVectorDB (with persistence)
    if phase2_yaml is None:
        phase2_yaml = Path(__file__).resolve().parent.parent / "pipeline_config" / "phase2_config.yaml"
    phase2_yaml = Path(phase2_yaml)
    try:
        acq = load_phase2_config(phase2_yaml, phase1_raw_dir="", phase2_raw_dir="")
        phase2_mi = acq.phase2_mi
    except Exception as e:
        print(f"[method3_setup] phase2_config.yaml load failed ({e}); using defaults")
        phase2_mi = Phase2MIConfig()

    db = SkillVectorDB()
    if db_save_dir is None:
        db_save_dir = Path(__file__).resolve().parent / "buffer"
    db_save_dir = Path(db_save_dir)
    db_save_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_save_dir / db_filename
    if db_path.exists():
        try:
            db.load(db_path)
            print(f"[method3_setup] loaded server-side DB ← {db_path} "
                  f"({db.total_size()} entries over {len(db.skill_ids())} skills)")
        except Exception as e:
            print(f"[method3_setup] DB load failed ({e}); starting fresh")
            db = SkillVectorDB()

    # 2026-05-21: 옛 architecture 는 client 가 P_phase1 build → npz cache 보관.
    # grpc mode 에서는 *score 계산이 server* 라 client 의 npz 가 사용 안 됨
    # (= server 의 db 는 empty 시작 → useful-OOD 의 기준 분포 P_phase1 부재).
    # 이제 *server 가 직접 P_phase1 build* — cache 비어있고 yaml 에 dataset path
    # 있으면 자체 re-embed. server cache npz 영구 저장 (= 다음 부팅 즉시 load).
    # yaml.server_build_phase1: false 면 server 가 자체 build 안 함 (= client 가
    # build + scp). default true (= 옛 동작).
    _server_build_enabled = bool(ph2_raw.get("server_build_phase1", True))
    if db.total_size() == 0 and _server_build_enabled:
        phase1_ds_path = ph2_raw.get("phase1_dataset_path")
        if phase1_ds_path:
            try:
                print(f"[method3_setup] server-side P_phase1 build start "
                      f"(dataset={phase1_ds_path})")
                from method3.reembedding.lerobot_adapter import LeRobotPhase1RawAdapter
                from method3.reembedding.seed_builder import (
                    ReembeddingConfig, build_phase1_vector_db,
                )
                raw_ds = LeRobotPhase1RawAdapter(str(phase1_ds_path))
                re_cfg_raw = ph2_raw.get("reembedding") or {}
                _sg_r = re_cfg_raw.get("subgoal_filter_radius_m")
                re_cfg = ReembeddingConfig(
                    skip_invalid=bool(re_cfg_raw.get("skip_invalid", False)),
                    show_progress=bool(re_cfg_raw.get("show_progress", True)),
                    frame_stride=int(re_cfg_raw.get("frame_stride", 1)),
                    batch_size=int(re_cfg_raw.get("batch_size", 32)),
                    subgoal_filter_radius_m=(None if _sg_r is None else float(_sg_r)),
                    subgoal_filter_min_keep=int(re_cfg_raw.get("subgoal_filter_min_keep", 30)),
                )
                db = build_phase1_vector_db(
                    raw_ds, encoder, raw_ds.load_observation, re_cfg,
                    g_seed_buffer=None,   # G_seed 는 client session 의 npz — 별도 RPC 로
                                          # 받아야. 일단 fallback (skill.goal_position).
                )
                db.save(db_path)
                print(f"[method3_setup] server-side P_phase1 saved ← {db_path} "
                      f"({db.total_size()} entries over {len(db.skill_ids())} skills)")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[method3_setup] server-side P_phase1 build FAILED ({e}); "
                      f"db remains empty — acquisition 진행 시 IngestEpisode 로만 누적")
        else:
            print(f"[method3_setup] phase1_dataset_path 미설정 — db empty 시작")
    elif db.total_size() == 0 and not _server_build_enabled:
        print(f"[method3_setup] server_build_phase1=false — client 가 build 후 scp 예정. "
              f"db empty 시작 (= cache 도착하면 다음 부팅에 load).")

    selector = Phase2MISelector(vector_db=db, config=phase2_mi)

    # 3) VLA-side informativeness (옵션) — yaml.vla_informativeness.enabled 일 때
    vla_scorer = None
    ph2_raw = _load_phase2_yaml(recording_cfg, phase2_yaml)
    vla_cfg = ph2_raw.get("vla_informativeness") or {}
    if vla_cfg.get("enabled", False):
        scorer_mode = str(vla_cfg.get("scorer_type", "default")).lower()
        try:
            _sigma = vla_cfg.get("sigma")
            # mode="dct" 는 R=1 single-step, candidate.dct_target inject.
            # mode="default" 는 기존 R-stochastic frame-level chunk denoise.
            vla_scorer = LeRobotVLAInformativenessScorer(
                policy=encoder.policy,
                R=int(vla_cfg.get("R", 8)),
                agg=str(vla_cfg.get("agg", "mean")),
                mode=scorer_mode,
                sigma=None if _sigma is None else float(_sigma),
            )
            print(
                f"[method3_setup] U_VLA scorer ENABLED (mode={scorer_mode}, "
                f"R={'1' if scorer_mode == 'dct' else vla_scorer.R}, "
                f"sigma={vla_scorer.sigma})"
            )
        except Exception as e:
            print(f"[method3_setup] U_VLA setup failed ({e}); selection falls back to argmax M_MI")
            vla_scorer = None

    print(
        f"[method3_setup] Phase2MISelector ready — tau_MI={phase2_mi.tau_MI}, "
        f"k_nn_a={phase2_mi.k_nn_a}, vla_scorer={'on' if vla_scorer else 'off'}"
    )
    return Method3ServerStack(
        selector=selector,
        encoder=encoder,
        vla_scorer=vla_scorer,
        db=db,
        db_path=db_path,
        config=phase2_mi,
        encoder_lock_payload={"checkpoint": str(ckpt)},
    )
