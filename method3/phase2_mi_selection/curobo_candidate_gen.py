"""Curobo TrajectoryCandidate → Phase2Candidate adapter — useful_ood_updated §7.3.

skills_lerobot 의 ``_skill_candidate_selector`` hook 자리에서 curobo 의 K-batch
가 이미 ``list[TrajectoryCandidate]`` 를 만들어 넘긴다. 본 모듈은 그 리스트를
mi_selector 가 이해하는 ``list[Phase2Candidate]`` 로 변환하는 thin adapter 다.

각 ``TrajectoryCandidate`` 의 ``waypoints (N, dof)`` 를 다음처럼 매핑한다::

    proprios[τ]     = waypoints[τ · H]                          (T, dof)
    action_chunks[τ] = waypoints[τ · H : τ · H + H]              (T, H, dof)
    state_keys[τ]   = [φ_VLA(o_τ, I); proprios[τ]]               (T, D_e)
                       — o_τ 는 *현재* observation 을 모든 τ 에 공유 (no forward
                       dynamics 환경의 근사).
    observations    = caller-provided LeRobot batch dict 또는 raw current_obs.
    instruction     = task instruction (skill 의 자연어).
    seed_subgoal    = G_seed^{(m)} 에서 정한 anchor g (caller).

T·H > N 이면 마지막 waypoint 로 pad. ``T``, ``H`` 는 caller 가 결정.

본 모듈은 curobo / skills_lerobot 의존성을 import 하지 않는다 (TrajectoryCandidate
와 동일한 ``waypoints/algo/seed`` 만 duck-typing 으로 쓰기 때문에 mock 도 가능).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from method3.phase2_mi_selection.mi_selector import Phase2Candidate
from method3.reembedding.vla_encoder import VLAStateEncoder


@dataclass
class CurobogenConfig:
    """waypoints → (T, H) chunking 파라미터."""

    n_windows: int = 3      # T
    action_horizon: int = 12  # H
    fail_safe_min_waypoints: int = 2  # waypoints 가 이보다 작으면 skip


def _chunk_waypoints(
    waypoints: np.ndarray,
    n_windows: int,
    action_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``waypoints (N, dof)`` → ``(proprios (T, dof), action_chunks (T, H, dof))``.

    부족분은 마지막 waypoint 로 pad. step stride = H (no overlap).
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2:
        raise ValueError(f"waypoints must be (N, dof); got {wp.shape}")
    N, dof = wp.shape
    needed = n_windows * action_horizon
    if N < needed:
        pad = np.tile(wp[-1:, :], (needed - N, 1))
        wp = np.concatenate([wp, pad], axis=0)
    # window stride H — non-overlapping.
    action_chunks = np.stack([
        wp[tau * action_horizon : tau * action_horizon + action_horizon]
        for tau in range(n_windows)
    ])  # (T, H, dof)
    proprios = action_chunks[:, 0, :].copy()  # (T, dof) — window 시작 joint state
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
            wp_arr, cfg.n_windows, cfg.action_horizon)

        # state_keys = [φ_VLA(o_τ, I); p_τ]. o_τ 는 모든 τ 에서 current_obs 공유.
        if encoder is not None and current_observation is not None:
            try:
                e_vla = np.asarray(
                    encoder.encode(current_observation, instruction),
                    dtype=np.float64,
                ).reshape(-1)
                state_keys = np.stack([
                    np.concatenate([e_vla, proprios[tau]])
                    for tau in range(cfg.n_windows)
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
        ))
    return out
