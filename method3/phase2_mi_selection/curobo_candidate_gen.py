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
from method3.reembedding.vla_encoder import VLAStateEncoder


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
    # candidate waypoints 가 *arm-only* (예: curobo so101 = 5 arm joints) 인데
    # DB action_descriptor 가 *full action (5 arm + 1 gripper = 6)* 으로 빌드
    # 됐다면 dim mismatch (250 vs 300). 이 값으로 *target full dof* 명시,
    # waypoints 의 dof 가 부족하면 constant (마지막 값) padding 으로 채워서
    # DCT 변환 → DB 와 같은 (L0 × full_dof) z-space.
    dct_target_dof: int = 6


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

    out: list[Phase2Candidate] = []
    for traj in trajectories:
        wp = getattr(traj, "waypoints", None)
        if wp is None:
            continue
        wp_arr = np.asarray(wp, dtype=np.float64)
        if wp_arr.ndim != 2 or wp_arr.shape[0] < cfg.fail_safe_min_waypoints:
            continue

        proprios, action_chunks = _chunk_waypoints(
            wp_arr, cfg.action_horizon, cfg.max_T_eval)
        T = proprios.shape[0]
        # skill 단위 DCT feature — candidate 의 전체 waypoints 를 한 skill 로
        # 보고 (L0, dof) DCT 로 변환 (paradigm step [3]).
        # waypoints dof 가 target dof 보다 작으면 constant (last value) padding
        # — curobo arm-only (5) vs full action (6) mismatch 보정.
        _wp_full = wp_arr
        if cfg.dct_target_dof and wp_arr.shape[1] < cfg.dct_target_dof:
            _pad = np.tile(wp_arr[:, -1:], (1, cfg.dct_target_dof - wp_arr.shape[1]))
            _wp_full = np.concatenate([wp_arr, _pad], axis=1)
        elif cfg.dct_target_dof and wp_arr.shape[1] > cfg.dct_target_dof:
            _wp_full = wp_arr[:, : cfg.dct_target_dof]
        dct_target = traj_to_dct(_wp_full, L0=cfg.dct_L0)

        # state_keys = [φ_VLA(o_τ, I); p_τ]. o_τ 는 모든 τ 에서 current_obs 공유.
        if encoder is not None and current_observation is not None:
            try:
                e_vla = np.asarray(
                    encoder.encode(current_observation, instruction),
                    dtype=np.float64,
                ).reshape(-1)
                state_keys = np.stack([
                    np.concatenate([e_vla, proprios[tau]])
                    for tau in range(T)
                ])
            except Exception:
                # encoder 실패 시 proprio 만으로 state_keys (degraded).
                state_keys = proprios.copy()
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
