"""
Multi-Robot Workspace Manager

Manages workspace constraints and collision avoidance for multiple robots.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class RobotWorkspace:
    """단일 로봇의 작업 공간 정의"""
    robot_id: int
    # 작업 공간 범위 (로봇 프레임 기준, meters)
    x_range: Tuple[float, float] = (0.05, 0.30)
    y_range: Tuple[float, float] = (-0.15, 0.15)
    z_range: Tuple[float, float] = (0.0, 0.20)
    # 안전 높이
    safe_height: float = 0.15
    approach_height: float = 0.15

    def is_reachable(self, position: List[float]) -> bool:
        """위치가 로봇의 도달 가능 범위 내인지 확인"""
        x, y, z = position[:3]
        return (
            self.x_range[0] <= x <= self.x_range[1] and
            self.y_range[0] <= y <= self.y_range[1] and
            self.z_range[0] <= z <= self.z_range[1]
        )

    def clamp_position(self, position: List[float]) -> List[float]:
        """위치를 작업 공간 범위 내로 제한"""
        x, y, z = position[:3]
        return [
            max(self.x_range[0], min(x, self.x_range[1])),
            max(self.y_range[0], min(y, self.y_range[1])),
            max(self.z_range[0], min(z, self.z_range[1])),
        ]


@dataclass
class MultiRobotWorkspace:
    """
    멀티 로봇 작업 공간 관리자

    여러 로봇의 작업 공간을 관리하고 충돌 회피를 지원합니다.
    """
    robot_workspaces: Dict[int, RobotWorkspace] = field(default_factory=dict)
    # 글로벌 작업 공간 (world frame)
    global_x_range: Tuple[float, float] = (-0.3, 0.3)
    global_y_range: Tuple[float, float] = (-0.3, 0.3)
    global_z_range: Tuple[float, float] = (0.0, 0.25)
    # 로봇 간 최소 거리
    min_robot_distance: float = 0.10  # 10cm

    def add_robot(self, robot_id: int, workspace: RobotWorkspace = None):
        """로봇 작업 공간 추가"""
        if workspace is None:
            workspace = RobotWorkspace(robot_id=robot_id)
        self.robot_workspaces[robot_id] = workspace

    def get_workspace(self, robot_id: int) -> Optional[RobotWorkspace]:
        """로봇 작업 공간 반환"""
        return self.robot_workspaces.get(robot_id)

    def is_collision_free(
        self,
        robot_positions: Dict[int, List[float]],
    ) -> bool:
        """
        여러 로봇의 현재 위치가 충돌 없는지 확인

        Args:
            robot_positions: {robot_id: [x, y, z]}

        Returns:
            bool: 충돌 없으면 True
        """
        positions = list(robot_positions.values())
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(
                    np.array(positions[i][:3]) - np.array(positions[j][:3])
                )
                if dist < self.min_robot_distance:
                    return False
        return True

    def validate_robot_positions(
        self,
        robot_id: int,
        object_positions: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        로봇의 객체 위치를 작업 공간 범위 내로 검증/조정

        Args:
            robot_id: 로봇 ID
            object_positions: {name: {"position": [x,y,z], ...}}

        Returns:
            검증된 객체 위치
        """
        workspace = self.get_workspace(robot_id)
        if workspace is None:
            return object_positions

        validated = {}
        for name, info in object_positions.items():
            if info is None:
                validated[name] = None
                continue

            pos = info.get("position", info) if isinstance(info, dict) else info

            if workspace.is_reachable(pos):
                validated[name] = info
            else:
                # 범위 내로 조정
                clamped = workspace.clamp_position(pos)
                print(f"[Workspace] Robot {robot_id}: {name} position adjusted "
                      f"from {pos} to {clamped}")
                if isinstance(info, dict):
                    validated[name] = {**info, "position": clamped}
                else:
                    validated[name] = {"position": clamped, "gripper_offset": 0.02}

        return validated


# 기본 작업 공간 설정 (SO-101 로봇용)
DEFAULT_WORKSPACE_CONFIG = {
    2: RobotWorkspace(
        robot_id=2,
        x_range=(0.05, 0.28),
        y_range=(-0.15, 0.15),
        z_range=(0.0, 0.18),
    ),
    3: RobotWorkspace(
        robot_id=3,
        x_range=(0.05, 0.28),
        y_range=(-0.15, 0.15),
        z_range=(0.0, 0.18),
    ),
}


def create_default_multi_workspace() -> MultiRobotWorkspace:
    """기본 멀티 로봇 작업 공간 생성"""
    workspace = MultiRobotWorkspace()
    for robot_id, robot_ws in DEFAULT_WORKSPACE_CONFIG.items():
        workspace.add_robot(robot_id, robot_ws)
    return workspace
