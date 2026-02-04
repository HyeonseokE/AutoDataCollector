"""
Base Workspace - Kinematics 기반 중심 Workspace

모든 태스크별 workspace는 이것을 상속하여 추가 제약을 건다.

계층 구조:
    BaseWorkspace (Kinematics 기반)
    ├── ResetWorkspace (reset_execution/workspace.py)
    ├── ForwardWorkspace (forward_execution/workspace.py)
    └── ...

좌표계:
    - World frame: 외부 좌표계 (x_min_world 제약 적용)
    - Base_link frame: 로봇 기준 좌표계 (reach limits 적용)
"""

import math
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot_cap.kinematics.engine import KinematicsEngine
    from lerobot_cap.transforms import FrameTransformer


class BaseWorkspace:
    """
    Kinematics 기반 기본 Workspace.

    두 가지 좌표계의 제약을 모두 처리:
    1. World frame: x >= x_min_world (전방 제한)
    2. Base_link frame: reach limits (원형 도달 범위)

    Attributes:
        min_reach: 최소 도달 거리 (m, base_link frame)
        max_reach: 최대 도달 거리 (m, base_link frame)
        z_floor: 바닥 높이 (m)
        x_min_world: World frame X 최소값 (m)
    """

    def __init__(
        self,
        kinematics_engine: Optional["KinematicsEngine"] = None,
        frame_transformer: Optional["FrameTransformer"] = None,
        min_reach: float = 0.05,
        max_reach: float = 0.407,
        z_floor: float = -0.02,  # 캘리브레이션 오차 허용 (-2cm)
        x_min_world: float = 0.12,
    ):
        """
        Initialize BaseWorkspace.

        Args:
            kinematics_engine: KinematicsEngine 인스턴스 (reach limits 자동 로드)
            frame_transformer: FrameTransformer 인스턴스 (world↔base 변환)
            min_reach: 최소 도달 거리 (kinematics_engine 없을 때 사용)
            max_reach: 최대 도달 거리 (kinematics_engine 없을 때 사용)
            z_floor: 바닥 높이
            x_min_world: World frame X 최소값 (기본 0.12m)
        """
        if kinematics_engine is not None:
            self.min_reach = kinematics_engine.min_reach
            self.max_reach = kinematics_engine.max_reach
            self._kinematics = kinematics_engine
        else:
            self.min_reach = min_reach
            self.max_reach = max_reach
            self._kinematics = None

        self._transformer = frame_transformer
        self.z_floor = z_floor
        self.x_min_world = x_min_world

    def is_reachable(
        self,
        position_world: np.ndarray,
        margin: float = 0.01,  # 1cm 안전 마진
    ) -> bool:
        """
        도달 가능 여부 검사 (world frame 입력).

        검사 순서:
        1. World frame 제약: x >= x_min_world
        2. World → Base_link 변환
        3. Base_link frame 제약: reach limits (원형)

        Args:
            position_world: [x, y, z] 위치 (world frame, meters)
            margin: 안전 여유 (meters)

        Returns:
            True if reachable
        """
        x_w, y_w, z_w = position_world

        # [1] World frame 제약: x >= x_min_world
        if x_w < self.x_min_world:
            return False

        # [2] World → Base_link 변환
        if self._transformer is not None:
            position_base = self._transformer.transform_position(
                position_world, from_frame="world"
            )
        else:
            # 변환기 없으면 동일 프레임 가정
            position_base = np.array(position_world)

        # [3] Base_link frame에서 reach 체크
        return self._check_reach_base(position_base, margin)

    def _check_reach_base(
        self,
        position_base: np.ndarray,
        margin: float = 0.01,  # 1cm 안전 마진
    ) -> bool:
        """
        Base_link frame에서 reach limits 검사.

        Args:
            position_base: [x, y, z] 위치 (base_link frame, meters)
            margin: 안전 여유 (meters)

        Returns:
            True if within reach
        """
        # Use kinematics engine if available
        if self._kinematics is not None:
            return self._kinematics.is_position_reachable(position_base, margin)

        # Otherwise use simple reach check
        x, y, z = position_base

        # Z floor check
        if z < self.z_floor:
            return False

        # Horizontal distance (XY plane)
        horizontal_distance = math.sqrt(x * x + y * y)

        # 3D distance
        distance_3d = math.sqrt(x * x + y * y + z * z)

        # Check reach limits
        max_horizontal = self.max_reach - margin
        min_horizontal = self.min_reach + margin

        is_within_horizontal = min_horizontal <= horizontal_distance <= max_horizontal
        is_within_3d = distance_3d <= (self.max_reach - margin)

        return is_within_horizontal and is_within_3d

    def get_reach_limits(self) -> tuple:
        """
        Reach 범위 반환.

        Returns:
            (min_reach, max_reach) in meters
        """
        return (self.min_reach, self.max_reach)

    def __repr__(self) -> str:
        return (f"BaseWorkspace(min_reach={self.min_reach:.3f}, "
                f"max_reach={self.max_reach:.3f}, x_min_world={self.x_min_world:.3f})")


# Singleton instance for convenience (initialized lazily)
_base_workspace: Optional[BaseWorkspace] = None


def get_base_workspace(
    kinematics_engine: Optional["KinematicsEngine"] = None,
    frame_transformer: Optional["FrameTransformer"] = None,
) -> BaseWorkspace:
    """
    BaseWorkspace 싱글톤 인스턴스 반환.

    Args:
        kinematics_engine: KinematicsEngine (첫 호출 시 필요)
        frame_transformer: FrameTransformer (첫 호출 시 필요)

    Returns:
        BaseWorkspace instance
    """
    global _base_workspace

    if _base_workspace is None:
        _base_workspace = BaseWorkspace(kinematics_engine, frame_transformer)

    return _base_workspace
