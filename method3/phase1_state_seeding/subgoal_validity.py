"""Subgoal position validity constraints for Phase1 (문서 §4.2 reachable/safe).

문서 §4.2 — ``G_valid = {g' | reachable(g'), safe(g'), is_transit(g')}``.
싸고 안전한 검사만 둔다: 로봇 reach (kinematics) + z-floor/ceiling + workspace
AABB. 후보 생성 단계의 rejection sampling 과 선택 단계의 필터가 **같은
술어**를 쓴다 (single source of truth).

scene-object collision 과 IK feasibility 검사는 후속으로 남긴다 — 비싼
holding-phase IK 게이트는 ``subgoal_selector.select_subgoal`` 의 ``feasibility_fn``
2차 필터로 분리돼 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ReachabilityConfig:
    """subgoal 위치 제약 — 각 항목이 None 이면 해당 검사 비활성.

    모두 None 이면 (reach 술어만 적용되거나) 아무 제약 없음 → 기존 동작과 동일.
    """

    z_min: float | None = None   # subgoal z 하한 (m) — 테이블/하향 충돌 회피
    z_max: float | None = None   # subgoal z 상한 (m)
    x_bounds: tuple[float, float] | None = None  # workspace AABB x (min, max)
    y_bounds: tuple[float, float] | None = None  # workspace AABB y (min, max)


def make_subgoal_validity_fn(
    config: ReachabilityConfig,
    is_reachable: Callable[[np.ndarray], bool] | None = None,
) -> Callable[[np.ndarray], bool]:
    """subgoal 후보 (3,) xyz 의 valid 여부 술어를 만든다 (문서 §4.2 reachable/safe).

    아래 검사를 **모두** 통과해야 valid:
      - ``is_reachable(xyz)``   — 로봇 reach (예: kinematics.is_position_reachable)
      - ``z_min ≤ z ≤ z_max``   — 높이 하한/상한
      - ``x_bounds`` / ``y_bounds`` — workspace AABB

    Args:
        config: 위치 제약 파라미터.
        is_reachable: 로봇 reach 술어. None 이면 reach 검사 생략.

    Returns:
        ``callable(xyz) -> bool``.
    """
    cfg = config

    def valid(xyz) -> bool:
        p = np.asarray(xyz, dtype=np.float64).reshape(3)
        if is_reachable is not None and not bool(is_reachable(p)):
            return False
        if cfg.z_min is not None and p[2] < cfg.z_min:
            return False
        if cfg.z_max is not None and p[2] > cfg.z_max:
            return False
        if cfg.x_bounds is not None and not (cfg.x_bounds[0] <= p[0] <= cfg.x_bounds[1]):
            return False
        if cfg.y_bounds is not None and not (cfg.y_bounds[0] <= p[1] <= cfg.y_bounds[1]):
            return False
        return True

    return valid
