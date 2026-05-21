"""Method3 config loaders — pipeline_config/phase{1,2}_config.yaml 파싱.

Method3 의 설정은 두 YAML 파일로 분리한다 (recording_config 와 같은 방식):

    pipeline_config/phase1_config.yaml  — Phase1 subgoal seeding (§4-5)
    pipeline_config/phase2_config.yaml  — Phase2 MI + 전환 + re-embedding (§6-§16)

phase1_config 는 recording 파이프라인(execution_forward_and_reset)이 직접 읽어
Phase1 selector 를 구성하고, phase2_config 는 ``load_phase2_config`` 가
``Method3AcquisitionConfig`` 로 변환해 acquisition 오케스트레이터가 쓴다.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from method3.acquisition.orchestrator import Method3AcquisitionConfig
from method3.phase2_mi_selection.mi_selector import Phase2MIConfig
from method3.phase_control.phase1_readiness import Phase1ReadinessConfig
from method3.phase_control.phase_controller import PhaseControllerConfig
from method3.reembedding.seed_builder import ReembeddingConfig


# Project root = AutoDataCollector/ (method3/config.py 의 부모의 부모)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(p: str | Path | None) -> str | None:
    """repo-relative path 를 ``_REPO_ROOT`` 기준 absolute 로 resolve.

    yaml 의 artifact path 가 local/remote 양쪽 portable 하도록:
      * absolute path → 그대로 (이미 운영자가 절대경로로 명시)
      * repo-relative path → ``_REPO_ROOT/p`` 로 join
      * None / "" → 그대로 (None)

    HF repo_id 같이 *path 아닌 식별자* 는 caller 가 미리 분기해야 한다
    (예: ``phase1_dataset_path`` 는 repo_id 도 받을 수 있으므로 resolve
    적용 안 함). 본 helper 는 *path 인 것이 보장된 키* 에만 사용한다.
    """
    if p is None or p == "":
        return None if p is None else p
    pp = Path(str(p))
    if pp.is_absolute():
        return str(pp)
    return str(_REPO_ROOT / pp)


def load_phase1_config(path: str | Path) -> dict:
    """phase1_config.yaml → Phase1 subgoal 설정 dict.

    recording_config 의 ``perturbation.subgoal`` 와 같은 shape 의 dict 를
    반환한다 (``subgoal:`` 키로 감싸져 있으면 그 안을, 아니면 전체를).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("subgoal") or raw


def load_phase2_config(
    path: str | Path,
    phase1_raw_dir: str | Path,
    phase2_raw_dir: str | Path,
) -> Method3AcquisitionConfig:
    """phase2_config.yaml → ``Method3AcquisitionConfig`` (문서 §6-§16).

    Args:
        path: phase2_config.yaml 경로.
        phase1_raw_dir / phase2_raw_dir: D_phase{1,2}_raw 디렉터리 — 세션에
            종속된 런타임 경로이므로 YAML 이 아니라 인자로 받는다.

    Returns:
        오케스트레이터(Method3Acquisition)가 그대로 쓰는 설정 객체.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # §4.2 K — re-embedding 과 Phase2 가 공유한다 (같은 metric space).
    dct = int(raw.get("dct_coeffs", 3))
    tau_ready = float(raw.get("tau_ready", 0.7))

    controller = PhaseControllerConfig(
        budget=int(raw.get("budget", 100)),
        phase1_min=int(raw.get("phase1_min", 20)),
        phase1_max=int(raw.get("phase1_max", 50)),
        tau_ready=tau_ready,
        saturation_window=int(raw.get("saturation_window", 10)),
        enable_phase2_early_stop=bool(raw.get("enable_phase2_early_stop", True)),
    )

    rd = raw.get("readiness") or {}
    readiness = Phase1ReadinessConfig(
        tau_ready=tau_ready,
        k_min=int(rd.get("k_min", 3)),
        radius_k=int(rd.get("radius_k", 5)),
        radius_quantile=float(rd.get("radius_quantile", 0.7)),
    )

    # selector.action_horizon 은 Phase1 re-embed (lerobot_adapter) 와 Phase2 candidate
    # gen (curobo_candidate_gen) 의 *공통* H — yaml 의 단일 SoT (spec §7.3: H=50).
    # 옛 키 ``selector.chunk_size`` 는 deprecated alias.
    # selector.use_dct_target 은 method3 DCT paradigm 의 z-space 전환 flag.
    sel = raw.get("selector") or {}
    action_horizon = int(sel.get("action_horizon", sel.get("chunk_size", 50)))
    use_dct_target = bool(sel.get("use_dct_target", False))
    # path-like 항목은 repo-relative 도 허용하도록 resolve.
    skill_dct_parquet = resolve_repo_path(sel.get("skill_dct_parquet"))

    mi = raw.get("mi_selection") or {}
    phase2_mi = Phase2MIConfig(
        dct_coeffs=dct,
        k_nn_a=int(mi.get("k_nn_a", 5)),
        q_quantile=float(mi.get("q_quantile", 0.5)),
        k_min=int(mi.get("k_min", 3)),
        radius_k=int(mi.get("radius_k", 5)),
        radius_quantile=float(mi.get("radius_quantile", 0.7)),
        s_min_a=float(mi.get("s_min_a", 1e-3)),
        eps=float(mi.get("eps", 1e-6)),
        beta=float(mi.get("beta", 1.0)),
        # YAML 의 'lambda' 는 파이썬 예약어 → dataclass 필드는 lambda_.
        lambda_=float(mi.get("lambda", mi.get("lambda_", 1.0))),
        # §13.2 — Useful-OOD constraint. accept_threshold 는 deprecated alias 로
        # tau_MI 가 없을 때만 fallback (Phase2MIConfig.__post_init__ 가 매핑).
        tau_MI=float(mi.get("tau_MI", mi.get("accept_threshold", 0.0))),
        amb_agg=str(mi.get("amb_agg", "mean")),
        min_covered_windows=int(mi.get("min_covered_windows", 1)),
        debug_verbose=bool(mi.get("debug_verbose", False)),
        use_dct_target=use_dct_target,
        arm_dof=mi.get("arm_dof", 5),
        full_dof=mi.get("full_dof", 6),
        dct_L0=int(mi.get("dct_L0", 50)),
    )

    re = raw.get("reembedding") or {}
    _sg_r = re.get("subgoal_filter_radius_m")
    reembedding = ReembeddingConfig(
        dct_coeffs=dct,
        skip_invalid=bool(re.get("skip_invalid", False)),
        show_progress=bool(re.get("show_progress", True)),
        frame_stride=int(re.get("frame_stride", 1)),
        batch_size=int(re.get("batch_size", 32)),
        subgoal_filter_radius_m=(None if _sg_r is None else float(_sg_r)),
        subgoal_filter_min_keep=int(re.get("subgoal_filter_min_keep", 30)),
        decode_workers=int(re.get("decode_workers", 1)),
        action_horizon=action_horizon,
        use_dct_target=use_dct_target,
        skill_dct_parquet=skill_dct_parquet,
    )

    return Method3AcquisitionConfig(
        phase1_raw_dir=phase1_raw_dir,
        phase2_raw_dir=phase2_raw_dir,
        phase_controller=controller,
        readiness=readiness,
        phase2_mi=phase2_mi,
        reembedding=reembedding,
        verbose=bool(raw.get("verbose", False)),
    )
