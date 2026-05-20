"""Canonical preview trajectory for Phase1 subgoal scoring.

문서 final_method3_spec §5.4 — 각 subgoal 후보까지 canonical interpolation
trajectory ``ξ_j = InterpPlan(S_t, g'_j)`` 를 만들고, §5.3 — subgoal 근처
마지막 20% 구간 ``T_end`` 만 평가한다. Cartesian 직선 EE 보간을 쓴다.
"""
from __future__ import annotations

import numpy as np


def interp_plan(
    current_ee: np.ndarray,
    goal: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """현재 EE 위치 → goal 의 직선 EE 보간 (문서 §5.4 InterpPlan).

    Args:
        current_ee: 현재 end-effector 위치 (3,) xyz.
        goal: subgoal (3,) xyz.
        n_points: preview state 개수 T (>= 2). 양 끝점 포함.

    Returns:
        (n_points, 3) preview EE 위치. ``traj[0]==current_ee``, ``traj[-1]==goal``.
    """
    current_ee = np.asarray(current_ee, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    alphas = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
    return current_ee[None, :] + alphas * (goal - current_ee)[None, :]


def last_segment(trajectory: np.ndarray, end_fraction: float) -> np.ndarray:
    """preview trajectory 의 마지막 ``end_fraction`` 구간 ``T_end`` 반환 (문서 §5.3).

    ``start_idx = int((1 - end_fraction) * T)`` 에서 슬라이스. 최소 1개 보장.
    """
    traj = np.asarray(trajectory)
    n = traj.shape[0]
    if not 0.0 < end_fraction <= 1.0:
        raise ValueError(f"end_fraction must be in (0, 1], got {end_fraction}")
    start_idx = min(int((1.0 - end_fraction) * n), n - 1)
    return traj[start_idx:]
