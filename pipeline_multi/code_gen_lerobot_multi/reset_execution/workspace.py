"""
Multi-Robot Reset Workspace Manager

Manages workspace constraints for reset operations,
including random position generation for data augmentation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class ResetWorkspaceConfig:
    """리셋 작업 공간 설정"""
    # 작업 공간 범위 (world frame, meters)
    x_range: Tuple[float, float] = (0.10, 0.25)
    y_range: Tuple[float, float] = (-0.10, 0.10)
    z_range: Tuple[float, float] = (0.0, 0.05)  # 테이블 위
    # 객체 간 최소 거리
    min_object_distance: float = 0.05  # 5cm
    # 그리퍼 최대 열림 폭 (grippable 판단용)
    gripper_max_open_width: float = 0.10  # 10cm


@dataclass
class MultiRobotResetWorkspace:
    """
    멀티 로봇 리셋 작업 공간 관리자

    랜덤 위치 생성 및 충돌 회피를 지원합니다.
    """
    config: ResetWorkspaceConfig = field(default_factory=ResetWorkspaceConfig)
    # 로봇별 작업 영역 (선택적)
    robot_zones: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=dict)

    def set_robot_zone(
        self,
        robot_id: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
    ):
        """로봇별 전용 영역 설정"""
        self.robot_zones[robot_id] = (x_range, y_range)

    def generate_random_position(
        self,
        robot_id: int = None,
        existing_positions: List[List[float]] = None,
        object_height: float = 0.02,
        max_attempts: int = 100,
    ) -> Optional[List[float]]:
        """
        충돌 없는 랜덤 위치 생성

        Args:
            robot_id: 로봇 ID (로봇별 영역 사용 시)
            existing_positions: 기존 배치된 객체 위치들
            object_height: 객체 높이 (z 좌표)
            max_attempts: 최대 시도 횟수

        Returns:
            [x, y, z] 또는 None (실패 시)
        """
        if existing_positions is None:
            existing_positions = []

        # 로봇별 영역 사용
        if robot_id is not None and robot_id in self.robot_zones:
            x_range, y_range = self.robot_zones[robot_id]
        else:
            x_range = self.config.x_range
            y_range = self.config.y_range

        for _ in range(max_attempts):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = object_height

            new_pos = [x, y, z]

            # 기존 객체와의 거리 확인
            is_valid = True
            for existing in existing_positions:
                dist = np.sqrt((x - existing[0])**2 + (y - existing[1])**2)
                if dist < self.config.min_object_distance:
                    is_valid = False
                    break

            if is_valid:
                return new_pos

        return None

    def is_grippable(self, bbox_size_m: List[float]) -> bool:
        """
        객체가 그리퍼로 집을 수 있는지 확인

        Args:
            bbox_size_m: [width, height] in meters

        Returns:
            bool: 집을 수 있으면 True
        """
        if bbox_size_m is None or len(bbox_size_m) < 2:
            return True  # 정보 없으면 True로 가정

        min_dim = min(bbox_size_m[0], bbox_size_m[1])
        return min_dim <= self.config.gripper_max_open_width

    def classify_objects(
        self,
        object_infos: Dict[str, Dict],
    ) -> Tuple[List[str], List[str]]:
        """
        객체를 grippable/non-grippable로 분류

        Args:
            object_infos: {name: {"bbox_size_m": [w, h], ...}}

        Returns:
            Tuple[grippable_names, non_grippable_names]
        """
        grippable = []
        non_grippable = []

        for name, info in object_infos.items():
            if info is None:
                continue

            bbox = info.get("bbox_size_m", None)
            if self.is_grippable(bbox):
                grippable.append(name)
            else:
                non_grippable.append(name)

        return grippable, non_grippable


def generate_random_positions(
    object_names: List[str],
    object_infos: Dict[str, Dict] = None,
    robot_id: int = None,
    workspace: MultiRobotResetWorkspace = None,
) -> Dict[str, List[float]]:
    """
    여러 객체에 대한 랜덤 위치 생성

    Args:
        object_names: 객체 이름 목록
        object_infos: 객체 정보 (높이 등)
        robot_id: 로봇 ID
        workspace: 작업 공간 관리자

    Returns:
        Dict[str, List[float]]: {name: [x, y, z]}
    """
    if workspace is None:
        workspace = MultiRobotResetWorkspace()

    if object_infos is None:
        object_infos = {}

    positions = {}
    existing = []

    for name in object_names:
        info = object_infos.get(name, {})

        # 객체 높이 결정
        if info and "position" in info:
            z = info["position"][2]
        else:
            z = 0.02  # 기본 높이

        pos = workspace.generate_random_position(
            robot_id=robot_id,
            existing_positions=existing,
            object_height=z,
        )

        if pos is not None:
            positions[name] = pos
            existing.append(pos)
        else:
            print(f"[Warning] Could not generate random position for {name}")
            # 기존 위치 사용 또는 기본값
            if info and "position" in info:
                positions[name] = info["position"]
            else:
                positions[name] = [0.15, 0.0, z]

    return positions


def generate_random_positions_multi_robot(
    robot_object_infos: Dict[int, Dict[str, Dict]],
    workspace: MultiRobotResetWorkspace = None,
) -> Dict[int, Dict[str, List[float]]]:
    """
    멀티 로봇용 랜덤 위치 생성

    각 로봇의 객체에 대해 독립적으로 랜덤 위치를 생성합니다.

    Args:
        robot_object_infos: {robot_id: {object_name: info}}
        workspace: 작업 공간 관리자

    Returns:
        Dict[int, Dict[str, List[float]]]: {robot_id: {name: [x,y,z]}}
    """
    if workspace is None:
        workspace = MultiRobotResetWorkspace()

    results = {}

    for robot_id, object_infos in robot_object_infos.items():
        object_names = list(object_infos.keys())
        results[robot_id] = generate_random_positions(
            object_names=object_names,
            object_infos=object_infos,
            robot_id=robot_id,
            workspace=workspace,
        )

    return results
