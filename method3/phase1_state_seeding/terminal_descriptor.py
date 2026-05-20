"""Geometric terminal-region descriptor — 문서 final_method3_spec §5.3 φ_goal.

§5.3 — preview state 를 descriptor 로 변환한다: ``ĥ_τ = φ_goal(Ŝ_τ, g', m)``.
초기 구현은 VLA embedding 대신 geometric descriptor 를 쓴다.

§5.3 예시는 ``[q_τ, x_EE, x_EE - g', m]`` 이지만, Cartesian 직선 preview 에는
joint ``q`` 가 없고(per-waypoint IK 미사용) buffer 가 skill 별로 분리돼
``m`` 이 불필요하므로, 핵심만 남겨 ``ĥ = [x_EE, x_EE - g']`` (6-dim) 을 쓴다.
요구사항은 preview state 와 buffer state 가 **같은 encoder** 를 쓰는 것이다.
"""
from __future__ import annotations

import numpy as np

# descriptor = concat(x_EE (3), x_EE - goal (3))
DESCRIPTOR_DIM = 6


def state_descriptor(ee_xyz: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """``ĥ = [x_EE, x_EE - g']`` ∈ R^6 — preview·buffer 공용 φ_goal (문서 §5.3).

    Args:
        ee_xyz: end-effector 위치 (3,) xyz.
        goal: 해당 state 가 향하는 subgoal (3,) xyz.

    Returns:
        (DESCRIPTOR_DIM,) float64 descriptor.
    """
    ee = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    g = np.asarray(goal, dtype=np.float64).reshape(3)
    return np.concatenate([ee, ee - g])
