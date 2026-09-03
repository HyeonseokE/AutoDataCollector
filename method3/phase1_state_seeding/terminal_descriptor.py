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

# [GOAL-RELATIVE 2026-08-25] descriptor 에서 절대 좌표 x_EE 를 뺀다.
# 기존 6-dim `[x_EE, x_EE - g]` 는 절대 좌표 절반이 섞여 있어, 버퍼에 20개
# layout 의 절대 위치가 함께 쌓이면 novelty 가 "어느 레이아웃인가"에 지배됐다
# (레이아웃 간 수십 cm ≫ 후보 간 ≤5cm). 그 결과 argmax 가 이전 레이아웃들에서
# 멀어지는 한 방향으로만 쏠렸다 (실측: offset 크기가 반경의 93%, 방향 일관성 0.45).
# 목표 상대분만 남기면 novelty 가 "이 skill 에서 어떤 상대 접근을 이미 써봤나"를
# 재게 되어 레이아웃 위치가 개입하지 않는다.
# 절대 공간의 state coverage 는 layout seed 20 개가 담당한다.
# Phase2 는 이 필드를 쓰지 않는다 (seed_anchor 는 entry.subgoal 절대 xyz 사용).
# descriptor = x_EE - goal (3)
DESCRIPTOR_DIM = 3


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
    return ee - g
