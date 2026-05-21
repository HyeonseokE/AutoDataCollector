"""Curobo TrajectoryCandidate → Phase2Candidate adapter — useful_ood_updated §7.3.

skills_lerobot 의 ``_skill_candidate_selector`` hook 자리에서 curobo 의 K-batch
가 이미 ``list[TrajectoryCandidate]`` 를 만들어 넘긴다. 본 모듈은 그 리스트를
mi_selector 가 이해하는 ``list[Phase2Candidate]`` 로 변환하는 thin adapter 다.

각 ``TrajectoryCandidate`` 의 ``waypoints (N, dof)`` 를 다음처럼 매핑한다::

    ξ 의 시간 길이 T = candidate trajectory 의 step 수 (= N).
    H 는 한 시점에서 미리 보는 action prediction horizon (spec §7.3, H=50).

    각 시점 τ ∈ [1, T] 에 대해::
        proprios[τ]      = waypoints[τ]                              (T, dof)
        action_chunks[τ] = waypoints[τ : τ + H]                       (T, H, dof)
        state_keys[τ]    = [φ_VLA(o_τ, I); proprios[τ]]               (T, D_e)
                          — o_τ 는 *현재* observation 을 모든 τ 에 공유
                          (no forward dynamics 환경의 근사).

Boundary 처리::
    N < H  : 마지막 waypoint 로 *waypoints* 자체를 length-H 까지 pad → T = 1
             (단일 시점, action_chunk 는 hold-last).
    N ≥ H  : stride=1 sliding window. 끝부분의 action chunk 가 trajectory 끝을
             넘으면 *그 시점의 action chunk 만* 마지막 waypoint 로 pad.
             T = N (모든 시점 평가, spec §7.3 의 τ=1..T 직역).

본 모듈은 curobo / skills_lerobot 의존성을 import 하지 않는다 (TrajectoryCandidate
와 동일한 ``waypoints/algo/seed`` 만 duck-typing 으로 쓰기 때문에 mock 도 가능).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from method3.dct.transform import traj_to_dct
from method3.phase2_mi_selection.mi_selector import Phase2Candidate
from method3.phase2_mi_selection.joint_servo_conversion import JointServoConverter
from method3.reembedding.vla_encoder import VLAStateEncoder


# Module-level cache — calibration JSON parse 는 once per process.
_CACHED_CONVERTER: dict[str, JointServoConverter] = {}


def _get_converter(path: str | None) -> JointServoConverter | None:
    """Lazy load + cache JointServoConverter. path None 이면 None 반환."""
    if not path:
        return None
    if path in _CACHED_CONVERTER:
        return _CACHED_CONVERTER[path]
    try:
        conv = JointServoConverter.from_calibration(path)
        _CACHED_CONVERTER[path] = conv
        return conv
    except Exception as e:
        print(f"  [curobo_candidate_gen] failed to load calibration {path!r}: {e}", flush=True)
        return None


@dataclass
class CurobogenConfig:
    """waypoints → (T, H) chunking 파라미터.

    spec §7.3:
      ξ = {(S_τ, A_{τ:τ+H-1})}_{τ=1}^{T} — T 는 trajectory 의 시간 길이,
      H 는 한 시점의 action prediction horizon.
    """

    action_horizon: int = 50  # H — spec §7.3: A_{τ:τ+H-1} ∈ ℝ^{H×6}, H=50
                              # (smolvla chunk_size 와 일치).
    fail_safe_min_waypoints: int = 2  # waypoints 가 이보다 작으면 skip
    max_T_eval: int | None = None     # T cost cap (운영용, spec 외).
                                       # None=비활성 (spec 직역, 모든 시점 평가).
                                       # int 면 T_raw > max_T_eval 일 때 균일 간격
                                       # sub-sample 해서 max_T_eval 개 시점만 평가.
    # method3 DCT paradigm — candidate 의 skill 단위 DCT feature 차원.
    # smolvla chunk_size 와 일치하도록 50 default.
    dct_L0: int = 50
    # candidate waypoints 는 *arm-only* (curobo so101 = 5 arm joints). gripper
    # 는 비교에서 제외 — DB action descriptor / state_key 는 mi_selector 에서
    # arm dof 만큼 slice 한다 (Phase2MIConfig.arm_dof / full_dof).
    # 5 이외의 값을 주면 dof 가 그 이상이면 truncate, 미만이면 last-value padding.
    dct_target_dof: int = 5
    # A.3 paradigm — curobo joint radians → lerobot servo positions (±100)
    # 변환. None 이면 변환 비활성 (legacy: candidate 는 radians space, DB 는
    # servo space → unit mismatch). path 가 주어지면 first-use 시 lazy load.
    servo_calibration_file: str | None = None


def _chunk_waypoints(
    waypoints: np.ndarray,
    action_horizon: int,
    max_T_eval: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``waypoints (N, dof)`` → ``(proprios (T, dof), action_chunks (T, H, dof))``.

    stride=1 sliding window. T 는 N 으로부터 자동 계산.

    Boundary 처리 (stride=1 sliding, 모든 시점 평가):
      - T = N (= waypoints 의 길이). N < H 이든 N ≥ H 이든 *모든 시점 τ ∈ [0, N)*
        에 대해 length-H action chunk 를 만든다.
      - 각 시점의 action chunk 가 trajectory 끝을 넘으면 (τ + H − 1 > N − 1) *그
        chunk 만* 끝점 hold pad (= "부족하면 pad"). trajectory 자체를 줄이거나
        늘리지 않는다.

      예시:
        N=10, H=50  →  T=10, action_chunks=(10, 50, 6).
                       각 τ ∈ [0,10): wp[τ:τ+50] 인데 N=10 이라 모든 chunk 가
                       앞 (10−τ) 개 실제 waypoint + 뒤 (40+τ) 개 끝점 hold.
        N=60, H=50  →  T=60, action_chunks=(60, 50, 6).
                       τ ≤ 9: 실제 waypoint 만으로 chunk 채워짐.
                       τ ≥ 10: 끝부분 hold pad 가 일부 섞임.

    Args:
        waypoints: (N, dof) — candidate trajectory.
        action_horizon: H — action prediction horizon.
        max_T_eval: T cost cap. None 이면 비활성 (T=N 모두 평가). int 면 sub-sample.
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2:
        raise ValueError(f"waypoints must be (N, dof); got {wp.shape}")
    N, dof = wp.shape
    H = int(action_horizon)
    if H < 1:
        raise ValueError(f"action_horizon must be >= 1; got {H}")
    if N < 1:
        raise ValueError("waypoints must have at least 1 row")

    # stride=1 sliding 으로 모든 N 시점 평가. 부족분은 끝점 hold pad.
    # wp_ext: trajectory 끝에 H−1 개 끝점 복제 → N<H 이든 N≥H 이든 모든 시점
    # τ ∈ [0, N) 에서 length-H chunk 추출 가능. T = N 일관.
    tail_pad = np.tile(wp[-1:, :], (H - 1, 1))
    wp_ext = np.concatenate([wp, tail_pad], axis=0)        # (N+H−1, dof)

    T_raw = N
    # T cost cap — sub-sample 시점 (운영 옵션, spec 외).
    if max_T_eval is not None and T_raw > int(max_T_eval):
        taus = np.linspace(0, T_raw - 1, int(max_T_eval), dtype=np.int64)
    else:
        taus = np.arange(T_raw, dtype=np.int64)

    proprios = wp[taus].copy()                              # (T, dof)
    action_chunks = np.stack([wp_ext[t:t + H] for t in taus])  # (T, H, dof)
    return proprios, action_chunks


def candidates_from_trajectory_list(
    trajectories: Iterable[Any],
    *,
    skill_id: str,
    seed_subgoal: np.ndarray,
    current_observation: Any,
    instruction: str,
    encoder: Optional[VLAStateEncoder],
    config: CurobogenConfig | None = None,
    batch_observations: Any = None,
    robot_state: np.ndarray | None = None,
) -> list[Phase2Candidate]:
    """TrajectoryCandidate-list 를 Phase2Candidate-list 로 변환.

    Args:
        trajectories: ``waypoints`` attribute 를 가진 객체들 (curobo
            TrajectoryCandidate 또는 동등한 duck-typed object).
        skill_id: 현재 skill m.
        seed_subgoal: G_seed^{(m)} 의 anchor g (3,).
        current_observation: rollout 시작 시점 raw obs. encoder 가 쓸 형식
            (image array 또는 cam-name → ndarray dict).
        instruction: task instruction (자연어).
        encoder: state_keys 계산용 frozen VLA encoder. None 이면 state_keys
            는 proprio 만 사용 (mock / debug).
        config: chunking 파라미터.
        batch_observations: LeRobot family-aware batch dict 가 미리 준비됐다면
            그대로 candidate.observations 에 넣어 vla_scorer 가 사용. 없으면
            current_observation 을 그대로 넣는다 (default batch_builder 가
            그걸 batch 로 해석할 수 있을 때만 U_VLA 활성).

    Returns:
        list[Phase2Candidate].
    """
    cfg = config or CurobogenConfig()
    g = np.asarray(seed_subgoal, dtype=np.float64).reshape(3)

    # Hoist VLA backbone call — all candidates share the same observation+instruction.
    # 1 backbone forward per request (not per candidate × per tau).
    _shared_e_vla: np.ndarray | None = None
    if encoder is not None and current_observation is not None:
        try:
            # zero_state — DB build 와 동일 dim (full proprio = arm+gripper = 6).
            # encoder 가 _fuses_state=False 면 state arg 가 key 에 concat 되므로
            # DB 와 *반드시 같은 dim* 이어야 한다.
            _zero_state = np.zeros(6, dtype=np.float64)
            _shared_e_vla = np.asarray(
                encoder.encode(current_observation, instruction, _zero_state),
                dtype=np.float64,
            ).reshape(-1)
        except Exception as _e:
            import traceback as _tb
            print(f"  [curobo_candidate_gen] shared encoder.encode failed: {type(_e).__name__}: {_e}", flush=True)
            _tb.print_exc()
            _shared_e_vla = None

    out: list[Phase2Candidate] = []
    for traj in trajectories:
        wp = getattr(traj, "waypoints", None)
        if wp is None:
            continue
        wp_arr = np.asarray(wp, dtype=np.float64)
        if wp_arr.ndim != 2 or wp_arr.shape[0] < cfg.fail_safe_min_waypoints:
            continue

        # waypoints dof 가 target dof 보다 작으면 constant (last value) padding
        # — curobo arm-only (5) vs full action (6) mismatch 보정.
        # *chunking 전* 에 pad — proprios / action_chunks / dct_target / state_keys
        # 모두 full dof 로 일관되게 만들어 DB 와 차원 정합.
        _wp_full = wp_arr
        if cfg.dct_target_dof and wp_arr.shape[1] < cfg.dct_target_dof:
            _pad = np.tile(wp_arr[:, -1:], (1, cfg.dct_target_dof - wp_arr.shape[1]))
            _wp_full = np.concatenate([wp_arr, _pad], axis=1)
        elif cfg.dct_target_dof and wp_arr.shape[1] > cfg.dct_target_dof:
            _wp_full = wp_arr[:, : cfg.dct_target_dof]

        # A.3 paradigm — curobo joint radians → lerobot servo positions (±100).
        # 이 변환이 있어야 candidate.dct_target / proprios / action_chunks 가
        # DB (servo space) 와 같은 unit. converter 없으면 *radians 그대로* (unit
        # mismatch, action descriptor ΔH_A 의미 약화).
        _converter = _get_converter(cfg.servo_calibration_file)
        if _converter is not None:
            try:
                _wp_full = _converter.radians_to_normalized(_wp_full)
            except Exception as e:
                print(f"  [curobo_candidate_gen] joint→servo conversion failed: {e}", flush=True)

        proprios, action_chunks = _chunk_waypoints(
            _wp_full, cfg.action_horizon, cfg.max_T_eval)
        T = proprios.shape[0]
        # skill 단위 DCT feature — candidate 의 전체 waypoints 를 한 skill 로
        # 보고 (L0, dof) DCT 로 변환 (paradigm step [3]).
        dct_target = traj_to_dct(_wp_full, L0=cfg.dct_L0)

        # state_keys[τ] = [e_vla; proprios[τ]] — spec §7.3 직역.
        # A.3 후 proprios 는 *servo space* (curobo joint radians → servo 변환됨)
        # 이므로 DB build pattern (proprio=observation.state) 과 unit 통일.
        # robot_state 인자는 *legacy* (모든 τ 동일 → no forward dynamics).
        # proprios[τ] 가 candidate trajectory 의 *τ-step 후 expected state* 라
        # τ별 다양성 살아남고 ΔH_A|S 측정이 의미 있게 됨.
        if _shared_e_vla is not None:
            state_keys = np.stack([
                np.concatenate([_shared_e_vla, proprios[tau]])
                for tau in range(T)
            ])
        else:
            state_keys = proprios.copy()

        out.append(Phase2Candidate(
            skill_id=str(skill_id),
            state_keys=state_keys,
            action_chunks=action_chunks,
            seed_subgoal=g,
            payload={"algo": getattr(traj, "algo", ""), "cost": getattr(traj, "cost", None)},
            observations=batch_observations if batch_observations is not None else current_observation,
            instruction=str(instruction),
            proprios=proprios,
            dct_target=dct_target,
        ))
    return out
