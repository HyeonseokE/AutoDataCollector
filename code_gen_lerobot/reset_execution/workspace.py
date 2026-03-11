"""
Reset Workspace - Reset 태스크용 추가 제약

BaseWorkspace를 상속하고 더 좁은 범위로 제한.

주요 기능:
1. Grippable 객체 분류 (그리퍼로 잡을 수 있는지 판단)
2. 랜덤 타겟 위치 생성 (충돌 회피, 초기 위치와 다른 위치)
3. 워크스페이스 자동 경계 계산 (IK 그리드 샘플링)
4. 이미지 위 워크스페이스 시각화
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot_cap.workspace import BaseWorkspace

if TYPE_CHECKING:
    from lerobot_cap.kinematics.engine import KinematicsEngine


# SO-101 그리퍼 사양
GRIPPER_MAX_OPEN_WIDTH = 0.10  # 10cm - 그리퍼 최대 열림 폭 (bbox 추정 오차 고려)


class ResetWorkspace(BaseWorkspace):
    """
    Reset 태스크 전용 Workspace.

    BaseWorkspace를 상속하고 추가 제약:
    - 더 좁은 XY 범위 (안정적인 배치를 위해)
    - Z는 테이블 표면에 고정
    - 객체 간 최소 거리
    - 초기 위치에서 최소 이동 거리
    """

    def __init__(
        self,
        kinematics_engine: Optional["KinematicsEngine"] = None,
        frame_transformer=None,
        x_range_world: Tuple[float, float] = (0.14, 0.32),
        y_range_world: Tuple[float, float] = (-0.25, 0.08),
        z_fixed_world: float = 0.01,
        min_object_distance: float = 0.07,
        min_displacement_from_inital: float = 0.07,
        gripper_max_width: float = GRIPPER_MAX_OPEN_WIDTH,
    ):
        """
        Initialize ResetWorkspace.

        Args:
            kinematics_engine: KinematicsEngine 인스턴스
            frame_transformer: FrameTransformer 인스턴스
            x_range_world: X 범위 (min, max) - world frame
            y_range_world: Y 범위 (min, max) - world frame
            z_fixed_world: 고정 Z 높이 (테이블 표면, world frame)
            min_object_distance: 객체 간 최소 거리
            min_displacement_from_inital: 초기 위치에서 최소 이동 거리
            gripper_max_width: 그리퍼 최대 열림 폭
        """
        super().__init__(kinematics_engine, frame_transformer)

        # Reset 전용 제약 (world frame)
        self.x_min_world, self.x_max_world = x_range_world
        self.y_min_world, self.y_max_world = y_range_world
        self.z_fixed_world = z_fixed_world
        self.min_object_distance = min_object_distance
        self.min_displacement_from_inital = min_displacement_from_inital
        self.gripper_max_width = gripper_max_width

    def is_valid(self, position_world: np.ndarray) -> bool:
        """
        Reset용 유효성 검사.

        1. 기본 검사 (Kinematics reach)
        2. Reset 추가 제약 (XY 범위)

        Args:
            position_world: [x, y, z] 위치 (world frame)

        Returns:
            True if valid for reset
        """
        # 1. 기본 검사 (BaseWorkspace)
        if not self.is_reachable(position_world):
            return False

        # 2. Reset 추가 제약
        x, y = position_world[0], position_world[1]
        return (self.x_min_world <= x <= self.x_max_world and
                self.y_min_world <= y <= self.y_max_world)

    def generate_random_position(
        self,
        obstacles: List[dict],
        initial_pos: Optional[List[float]] = None,
        obj_radius: float = 0.015,
        max_attempts: int = 100,
    ) -> Optional[List[float]]:
        """
        단일 객체용 랜덤 위치 생성.

        Args:
            obstacles: 피해야 할 위치들 [{"center": [x,y], "radius": r}, ...]
            initial_pos: 초기 위치 (이 위치에서 min_displacement_from_inital 이상 떨어져야 함)
            obj_radius: 객체 반경
            max_attempts: 최대 시도 횟수

        Returns:
            [x, y, z] 또는 None (실패 시)
        """
        for _ in range(max_attempts):
            # 랜덤 위치 생성
            x = np.random.uniform(self.x_min_world, self.x_max_world)
            y = np.random.uniform(self.y_min_world, self.y_max_world)
            z = self.z_fixed_world

            candidate = [x, y, z]

            # 조건 1: 기본 도달 가능 여부
            if not self.is_reachable(np.array(candidate)):
                continue

            # 조건 2: 초기 위치에서 충분히 떨어졌는지
            if initial_pos is not None:
                displacement = np.sqrt(
                    (x - initial_pos[0])**2 + (y - initial_pos[1])**2
                )
                if displacement < self.min_displacement_from_inital:
                    continue

            # 조건 3: 장애물과 겹치지 않는지
            collision = False
            for occ in obstacles:
                dist = np.sqrt(
                    (x - occ["center"][0])**2 +
                    (y - occ["center"][1])**2
                )
                if dist < occ["radius"] + obj_radius:
                    collision = True
                    break

            if collision:
                continue

            return candidate

        return None

    def __repr__(self) -> str:
        return (f"ResetWorkspace(x_world=[{self.x_min_world:.2f}, {self.x_max_world:.2f}], "
                f"y_world=[{self.y_min_world:.2f}, {self.y_max_world:.2f}], z_world={self.z_fixed_world:.2f})")


# ============================================================
# Helper Functions
# ============================================================

def is_grippable(
    bbox_size_m: Tuple[float, float],
    gripper_max_width: float = GRIPPER_MAX_OPEN_WIDTH,
) -> bool:
    """
    그리퍼로 잡을 수 있는 물체인지 판단.

    Args:
        bbox_size_m: (width, height) in meters
        gripper_max_width: 그리퍼 최대 열림 폭

    Returns:
        True if object can be gripped
    """
    if bbox_size_m is None:
        return True

    width, height = bbox_size_m
    min_dim = min(width, height)
    return min_dim < gripper_max_width


def classify_objects(
    detections: Dict[str, dict],
    gripper_max_width: float = GRIPPER_MAX_OPEN_WIDTH,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    객체를 grippable / non-grippable(obstacle)로 분류.

    Args:
        detections: {name: {"position": [x,y,z], "bbox_size_m": [w,h], ...}}
        gripper_max_width: 그리퍼 최대 열림 폭

    Returns:
        (grippable_objects, obstacle_objects)
    """
    grippable = {}
    obstacles = {}

    for name, info in detections.items():
        if info is None:
            continue

        bbox_size = info.get("bbox_size_m")

        if is_grippable(bbox_size, gripper_max_width):
            grippable[name] = info
        else:
            obstacles[name] = info

    return grippable, obstacles


def generate_random_positions(
    grippable_objects: Dict[str, dict],
    obstacle_objects: Dict[str, dict],
    initial_positions: Dict[str, List[float]],
    workspace: ResetWorkspace = None,
    seed: int = None,
    max_attempts: int = 100,
) -> Dict[str, List[float]]:
    """
    랜덤 위치 생성 (핵심 알고리즘).

    조건:
    1. workspace 범위 내 (base reachability + reset 제약)
    2. 모든 장애물(non-grippable)과 겹치지 않음
    3. 이미 배치된 다른 grippable 객체와 min_distance 유지
    4. 초기 위치에서 min_displacement_from_inital 이상 떨어져야 함

    Args:
        grippable_objects: 이동할 객체들
        obstacle_objects: 장애물 객체들 (위치 고정)
        initial_positions: 객체들의 초기 위치
        workspace: ResetWorkspace 인스턴스
        seed: 재현성을 위한 random seed
        max_attempts: 위치당 최대 시도 횟수

    Returns:
        {object_name: [x, y, z]} 랜덤 타겟 위치
    """
    if workspace is None:
        workspace = ResetWorkspace()

    if seed is not None:
        np.random.seed(seed)

    # 장애물 위치 리스트 - 모든 검출 객체 포함 (grippable + non-grippable)
    occupied_positions = []

    # 1) Non-grippable 객체 추가 (위치 고정, 이동 안 함)
    for name, info in obstacle_objects.items():
        if info is None:
            continue
        pos = info.get("position", info) if isinstance(info, dict) else info
        if pos is None:
            continue
        size = info.get("bbox_size_m", [0.05, 0.05]) if isinstance(info, dict) else [0.05, 0.05]
        if size is None:
            size = [0.05, 0.05]
        occupied_positions.append({
            "name": name,
            "center": [pos[0], pos[1]],
            "radius": max(size) / 2 + workspace.min_object_distance,
            "is_fixed": True,  # 고정 장애물
        })

    # 2) Grippable 객체의 현재 위치도 추가 (이동 전까지 장애물로 취급)
    for name, info in grippable_objects.items():
        if info is None:
            continue
        pos = info.get("position", info) if isinstance(info, dict) else info
        if pos is None:
            continue
        size = info.get("bbox_size_m", [0.03, 0.03]) if isinstance(info, dict) else [0.03, 0.03]
        if size is None:
            size = [0.03, 0.03]
        occupied_positions.append({
            "name": name,
            "center": [pos[0], pos[1]],
            "radius": max(size) / 2 + workspace.min_object_distance,
            "is_fixed": False,  # 이동 가능 객체
        })

    # 결과 저장
    target_positions = {}

    for obj_name, obj_info in grippable_objects.items():
        if obj_info is None:
            continue

        # 초기 위치 가져오기 (forward 시작 전 위치)
        initial_info = initial_positions.get(obj_name)
        if initial_info is None:
            initial_pos = None
        elif isinstance(initial_info, dict) and "position" in initial_info:
            initial_pos = initial_info["position"]
        elif isinstance(initial_info, (list, tuple)) and len(initial_info) >= 3:
            initial_pos = list(initial_info[:3])
        else:
            initial_pos = None

        # 객체 크기
        if isinstance(obj_info, dict):
            obj_size = obj_info.get("bbox_size_m", [0.03, 0.03])
        else:
            obj_size = [0.03, 0.03]
        if obj_size is None:
            obj_size = [0.03, 0.03]
        obj_radius = max(obj_size) / 2

        # 이 객체의 현재 위치를 장애물 목록에서 제거 (자기 자신과 충돌 방지)
        obstacles_for_this_obj = [
            occ for occ in occupied_positions if occ["name"] != obj_name
        ]

        # 랜덤 위치 생성
        position = workspace.generate_random_position(
            obstacles=obstacles_for_this_obj,
            initial_pos=initial_pos,
            obj_radius=obj_radius,
            max_attempts=max_attempts,
        )

        if position is not None:
            target_positions[obj_name] = position
            # 이 객체의 현재 위치를 제거하고 새 위치로 업데이트
            occupied_positions = [occ for occ in occupied_positions if occ["name"] != obj_name]
            occupied_positions.append({
                "name": obj_name,
                "center": [position[0], position[1]],
                "radius": obj_radius + workspace.min_object_distance,
                "is_fixed": False,
            })
        else:
            # Fallback
            print(f"[Warning] Could not find valid position for '{obj_name}'")
            if initial_pos is not None:
                offset = np.random.uniform(-0.03, 0.03, 2)
                fallback_pos = [
                    np.clip(initial_pos[0] + offset[0], workspace.x_min_world, workspace.x_max_world),
                    np.clip(initial_pos[1] + offset[1], workspace.y_min_world, workspace.y_max_world),
                    workspace.z_fixed_world,
                ]
            else:
                fallback_pos = [
                    (workspace.x_min_world + workspace.x_max_world) / 2,
                    (workspace.y_min_world + workspace.y_max_world) / 2,
                    workspace.z_fixed_world,
                ]
            target_positions[obj_name] = fallback_pos

    return target_positions


# ============================================================
# Workspace Bounds & Visualization
# ============================================================

def compute_workspace_bounds(
    workspace: BaseWorkspace = None,
    z_table: float = 0.01,
    step: float = 0.01,
    x_scan: Tuple[float, float] = (0.05, 0.45),
    y_scan: Tuple[float, float] = (-0.35, 0.20),
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    IK 그리드 샘플링으로 로봇별 도달 가능 workspace 경계를 자동 계산.

    Args:
        workspace: BaseWorkspace 인스턴스 (None이면 기본 생성)
        z_table: 테이블 표면 Z 높이 (meters)
        step: 스캔 간격 (meters)
        x_scan: X 스캔 범위 (min, max)
        y_scan: Y 스캔 범위 (min, max)

    Returns:
        ((x_min, x_max), (y_min, y_max)) in meters
    """
    if workspace is None:
        workspace = BaseWorkspace()

    valid_x, valid_y = [], []
    for x in np.arange(x_scan[0], x_scan[1], step):
        for y in np.arange(y_scan[0], y_scan[1], step):
            if workspace.is_reachable(np.array([x, y, z_table])):
                valid_x.append(x)
                valid_y.append(y)

    if not valid_x:
        # Fallback to hardcoded defaults
        return (0.14, 0.32), (-0.25, 0.08)

    return (min(valid_x), max(valid_x)), (min(valid_y), max(valid_y))


def draw_workspace_on_image(
    image: np.ndarray,
    workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    coord_transformer,
    alpha: float = 0.3,
    robot_id: int = 3,
) -> np.ndarray:
    """
    로봇 워크스페이스를 원형(min/max reach) + x_min 제약으로 이미지에 시각화.

    시각적 요소:
    - 로봇 base 중심으로 min_reach / max_reach 시안 원호 경계
    - x_min_world 시안 수직선
    - convex hull 바깥 어둡게 마스킹

    Args:
        image: BGR 이미지 (numpy array)
        workspace_bounds: ((x_min, x_max), (y_min, y_max)) in meters (프롬프트 전달용)
        coord_transformer: CoordinateTransformer 인스턴스
        alpha: 미사용 (호환성 유지)
        robot_id: 로봇 ID (프레임 설정 로드용)

    Returns:
        시각화된 이미지 (numpy array, copy)
    """
    import cv2
    import json

    img_h, img_w = image.shape[:2]
    result = image.copy()

    # ── 로봇 base 위치 로드 (World frame, cm) ──
    robot_x_cm, robot_y_cm = 5.1, -25.1  # defaults
    base_rotation_matrix = None

    try:
        frame_config_path = (
            Path(__file__).parent.parent.parent
            / f"robot_configs/world2robot_matrices/robot{robot_id}_matrix.json"
        )
        if frame_config_path.exists():
            with open(frame_config_path, 'r') as f:
                frame_config = json.load(f)
            frames_world = frame_config.get("frames", {}).get("world", {})
            if "translation" in frames_world:
                t = frames_world["translation"]
                robot_x_cm = t[0] * 100
                robot_y_cm = t[1] * 100
            raw_transform = frame_config.get("_raw_transform", {})
            base_rotation_matrix = raw_transform.get("rotation_matrix")
    except Exception:
        pass

    robot_x_m = robot_x_cm / 100.0
    robot_y_m = robot_y_cm / 100.0

    # ── Workspace 파라미터 ──
    ws = BaseWorkspace()
    min_reach_cm = ws.min_reach * 100
    max_reach_cm = ws.max_reach * 100
    x_min_world_cm = ws.x_min_world * 100

    # 그리드 스캔 범위 (로봇 base 중심)
    grid_step_cm = 1.5
    x_min_scan = max(x_min_world_cm, robot_x_cm - max_reach_cm)
    x_max_scan = robot_x_cm + max_reach_cm
    y_min_scan = robot_y_cm - max_reach_cm
    y_max_scan = robot_y_cm + max_reach_cm

    def is_reachable_from_base(x_m, y_m):
        if x_m < ws.x_min_world:
            return False
        dx = x_m - robot_x_m
        dy = y_m - robot_y_m
        dist = np.sqrt(dx * dx + dy * dy)
        margin = 0.01
        return (ws.min_reach + margin) <= dist <= (ws.max_reach - margin)

    # ── 도달 가능 픽셀 수집 (convex hull 마스킹용) ──
    valid_pixels = []
    for x_cm in np.arange(x_min_scan, x_max_scan, grid_step_cm):
        for y_cm in np.arange(y_min_scan, y_max_scan, grid_step_cm):
            try:
                u, v = coord_transformer.world_to_pixel(x_cm, y_cm)
                if not (0 <= u < img_w and 0 <= v < img_h):
                    continue
                if is_reachable_from_base(x_cm / 100.0, y_cm / 100.0):
                    valid_pixels.append((u, v))
            except Exception:
                continue

    # ── 바깥 영역 마스킹 (valid 영역 바깥 어둡게) ──
    if valid_pixels:
        ws_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        hull = cv2.convexHull(np.array(valid_pixels))
        cv2.fillConvexPoly(ws_mask, hull, 255)
        # 마스크 바깥을 어둡게
        result[ws_mask == 0] = (result[ws_mask == 0] * 0.5).astype(np.uint8)

    COLOR_CYAN = (255, 255, 0)

    # ── Min reach 원 ──
    for angle in range(0, 360, 10):
        rad = np.radians(angle)
        px = robot_x_cm + min_reach_cm * np.cos(rad)
        py = robot_y_cm + min_reach_cm * np.sin(rad)
        try:
            pu, pv = coord_transformer.world_to_pixel(px, py)
            if 0 <= pu < img_w and 0 <= pv < img_h:
                cv2.circle(result, (pu, pv), 2, COLOR_CYAN, -1)
        except Exception:
            pass

    # ── Max reach 원 ──
    for angle in range(0, 360, 3):
        rad = np.radians(angle)
        px = robot_x_cm + max_reach_cm * np.cos(rad)
        py = robot_y_cm + max_reach_cm * np.sin(rad)
        try:
            pu, pv = coord_transformer.world_to_pixel(px, py)
            if 0 <= pu < img_w and 0 <= pv < img_h:
                cv2.circle(result, (pu, pv), 2, COLOR_CYAN, -1)
        except Exception:
            pass

    # ── x_min_world 수직선 ──
    for y_cm in np.arange(y_min_scan, y_max_scan, 2):
        try:
            pu, pv = coord_transformer.world_to_pixel(x_min_world_cm, y_cm)
            if 0 <= pu < img_w and 0 <= pv < img_h:
                cv2.circle(result, (pu, pv), 2, COLOR_CYAN, -1)
        except Exception:
            pass

    # ── Workspace bounds 라벨 (프롬프트에 전달되는 값) ──
    (x_min, x_max), (y_min, y_max) = workspace_bounds
    label = f"WS: x=[{x_min:.2f},{x_max:.2f}] y=[{y_min:.2f},{y_max:.2f}]"
    cv2.putText(result, label, (10, img_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return result


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("ResetWorkspace Test")
    print("=" * 60)

    # Create workspace
    ws = ResetWorkspace()
    print(f"Workspace: {ws}")
    print(f"Base reach: {ws.get_reach_limits()}")
    print()

    # Test positions
    test_positions = [
        [0.15, 0.0, 0.01],   # Valid
        [0.10, 0.0, 0.01],   # Out of x_range
        [0.17, 0.10, 0.01],  # Out of y_range
        [0.50, 0.0, 0.01],   # Out of reach
    ]

    for pos in test_positions:
        reachable = ws.is_reachable(np.array(pos))
        valid = ws.is_valid(np.array(pos))
        print(f"  {pos} -> reachable={reachable}, valid={valid}")

    print()

    # Test random position generation
    grippable = {
        "green block": {"position": [0.15, 0.05, 0.01], "bbox_size_m": [0.03, 0.03]},
        "red cube": {"position": [0.18, -0.02, 0.01], "bbox_size_m": [0.025, 0.025]},
    }

    obstacles = {
        "blue dish": {"position": [0.20, -0.03, 0.01], "bbox_size_m": [0.12, 0.10]},
    }

    initial_positions = {
        "green block": [0.15, 0.05, 0.01],
        "red cube": [0.18, -0.02, 0.01],
    }

    print(f"Grippable: {list(grippable.keys())}")
    print(f"Obstacles: {list(obstacles.keys())}")

    random_targets = generate_random_positions(
        grippable_objects=grippable,
        obstacle_objects=obstacles,
        initial_positions=initial_positions,
        workspace=ws,
        seed=42,
    )

    print("\nGenerated positions:")
    for name, pos in random_targets.items():
        initial = initial_positions.get(name, [0, 0, 0])
        displacement = np.sqrt((pos[0] - initial[0])**2 + (pos[1] - initial[1])**2)
        print(f"  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] (disp: {displacement*100:.1f}cm)")
