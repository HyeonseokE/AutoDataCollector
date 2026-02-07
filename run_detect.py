#!/usr/bin/env python3
"""
Object Detection Module for LeRobot Pipeline

자연어로 객체를 찾아 World 좌표계 기준 위치를 반환합니다.
run_forward_and_reset 파이프라인에서 사용되는 핵심 모듈입니다.

Main function:
    run_realtime_detection() - 객체 검출 및 위치 반환

Usage (in pipeline):
    from run_detect import run_realtime_detection

    positions = run_realtime_detection(
        queries=["red cup", "blue box"],
        timeout=10.0,
        unit="m",
        visualize=False,
        return_extended=True,
    )
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple
import numpy as np


def draw_workspace_overlay(
    image: np.ndarray,
    transformer,
    workspace=None,
    grid_step_cm: float = 1.0,
    alpha: float = 0.15,
    robot_id: int = 3,
) -> np.ndarray:
    """
    Workspace overlay를 이미지에 그립니다.

    Valid 영역: 초록색 (투명도 높음)
    Invalid 영역: 빨간색 (투명도 높음)
    경계: 점선
    Robot base: 파란색 원 (이미지 내 보이면 표시)

    Args:
        image: 원본 이미지 (BGR)
        transformer: CoordinateTransformer 인스턴스 (world_to_pixel 메서드 필요)
        workspace: BaseWorkspace 인스턴스 (is_reachable 메서드 필요)
        grid_step_cm: 그리드 간격 (cm)
        alpha: 투명도 (0.0 ~ 1.0)
        robot_id: 로봇 ID (2 또는 3) - 프레임 설정 로드용

    Returns:
        overlay가 그려진 이미지
    """
    import cv2
    import json

    if transformer is None or not transformer.is_ready:
        return image

    # Workspace 없으면 기본 생성
    if workspace is None:
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from lerobot_cap.workspace import BaseWorkspace
            workspace = BaseWorkspace()
        except ImportError:
            return image

    result = image.copy()
    overlay_valid = np.zeros_like(image)
    overlay_invalid = np.zeros_like(image)

    # Robot frame config 로드하여 robot base 위치 획득
    # Base_link origin의 World frame 좌표 = -R^T @ t (역변환)
    # Transformer가 내부적으로 Y축을 반전하므로, World frame Y 그대로 전달
    robot_x_cm = 5.1   # default (World frame)
    robot_y_cm = -25.1  # default (World frame, transformer가 내부에서 +25.1로 반전)

    try:
        frame_config_path = PROJECT_ROOT / f"robot_configs/world2robot_matrices/robot{robot_id}_matrix.json"
        if frame_config_path.exists():
            with open(frame_config_path, 'r') as f:
                frame_config = json.load(f)

            # 방법 1: frames.world.translation 직접 사용 (Base origin in World frame)
            frames_world = frame_config.get("frames", {}).get("world", {})
            if "translation" in frames_world:
                translation = frames_world["translation"]
                robot_x_cm = translation[0] * 100  # m -> cm
                robot_y_cm = translation[1] * 100  # m -> cm (World frame Y 그대로)
            else:
                # 방법 2: _raw_transform에서 역변환 계산
                raw_transform = frame_config.get("_raw_transform", {})
                R = np.array(raw_transform.get("rotation_matrix", np.eye(3).tolist()))
                t = np.array(raw_transform.get("translation", [0.05, -0.25, 0]))
                base_in_world = -R.T @ t
                robot_x_cm = base_in_world[0] * 100
                robot_y_cm = base_in_world[1] * 100
    except Exception as e:
        print(f"[Warning] Failed to load frame config: {e}")
        pass  # 기본값 사용

    # 실제 workspace 기반으로 그리드 범위 계산
    min_reach_cm = workspace.min_reach * 100  # 5cm
    max_reach_cm = workspace.max_reach * 100  # 40.7cm
    x_min_world_cm = workspace.x_min_world * 100  # 12cm

    # Robot base 실제 위치 (World frame, m) - workspace 거리 계산용
    # robot_y_cm은 이제 World frame Y 좌표 그대로 (반전 없음)
    robot_x_m = robot_x_cm / 100.0
    robot_y_m = robot_y_cm / 100.0

    # 유효 범위 계산 (detection world frame)
    # Robot base 중심으로 reach 범위 + x_min_world 제약
    x_min = max(x_min_world_cm, robot_x_cm - max_reach_cm)
    x_max = robot_x_cm + max_reach_cm
    y_min = robot_y_cm - max_reach_cm
    y_max = robot_y_cm + max_reach_cm

    # 경계 좌표 수집
    boundary_points = []

    def check_reachable_from_robot_base(x_world_m, y_world_m):
        """
        Robot base 기준으로 도달 가능 여부 검사.

        Args:
            x_world_m, y_world_m: World frame 좌표 (meters)

        Returns:
            True if reachable
        """
        # [1] World frame 제약: x >= x_min_world
        if x_world_m < workspace.x_min_world:
            return False

        # [2] Robot base로부터의 거리 계산 (XY 평면)
        dx = x_world_m - robot_x_m
        dy = y_world_m - robot_y_m
        distance = np.sqrt(dx * dx + dy * dy)

        # [3] Reach 범위 체크 (margin 포함)
        margin = 0.01  # 1cm 안전 마진
        if distance < workspace.min_reach + margin:
            return False
        if distance > workspace.max_reach - margin:
            return False

        return True

    # 그리드 순회
    for x_cm in np.arange(x_min, x_max + grid_step_cm, grid_step_cm):
        prev_valid = None
        for y_cm in np.arange(y_min, y_max + grid_step_cm, grid_step_cm):
            try:
                # Detection world frame 좌표 → 픽셀 좌표
                u, v = transformer.world_to_pixel(x_cm, y_cm)

                # 이미지 범위 체크
                if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
                    prev_valid = None
                    continue

                # Workspace 유효성 검사 (meters)
                # Grid 좌표는 World frame (frame_config 기준), transformer가 내부에서 Y 반전
                x_world_m = x_cm / 100.0
                y_world_m = y_cm / 100.0  # World frame Y 좌표 그대로
                is_valid = check_reachable_from_robot_base(x_world_m, y_world_m)

                # 경계 감지
                if prev_valid is not None and prev_valid != is_valid:
                    boundary_points.append((u, v))

                prev_valid = is_valid

                # 작은 원으로 그리드 점 표시 (BGR 형식)
                if is_valid:
                    cv2.circle(overlay_valid, (u, v), 3, (0, 255, 0), -1)  # Green (BGR)
                else:
                    cv2.circle(overlay_invalid, (u, v), 3, (0, 0, 255), -1)  # Red (BGR)

            except Exception:
                prev_valid = None
                continue

    # Y 방향으로도 경계 탐지
    for y_cm in np.arange(y_min, y_max + grid_step_cm, grid_step_cm):
        prev_valid = None
        for x_cm in np.arange(x_min, x_max + grid_step_cm, grid_step_cm):
            try:
                u, v = transformer.world_to_pixel(x_cm, y_cm)

                if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
                    prev_valid = None
                    continue

                # Grid 좌표는 World frame (frame_config 기준)
                x_world_m = x_cm / 100.0
                y_world_m = y_cm / 100.0  # World frame Y 좌표 그대로
                is_valid = check_reachable_from_robot_base(x_world_m, y_world_m)

                if prev_valid is not None and prev_valid != is_valid:
                    boundary_points.append((u, v))

                prev_valid = is_valid

            except Exception:
                prev_valid = None
                continue

    # Overlay 블렌딩
    mask_valid = overlay_valid.sum(axis=2) > 0
    mask_invalid = overlay_invalid.sum(axis=2) > 0

    result[mask_valid] = cv2.addWeighted(
        result[mask_valid], 1 - alpha,
        overlay_valid[mask_valid], alpha,
        0
    )
    result[mask_invalid] = cv2.addWeighted(
        result[mask_invalid], 1 - alpha,
        overlay_invalid[mask_invalid], alpha,
        0
    )

    # 색상 정의 (BGR 형식)
    COLOR_BLUE = (255, 0, 0)      # Blue in BGR
    COLOR_RED = (0, 0, 255)       # Red in BGR
    COLOR_GREEN = (0, 255, 0)     # Green in BGR
    COLOR_YELLOW = (0, 255, 255)  # Yellow in BGR
    COLOR_ORANGE = (0, 80, 255)   # Dark Orange in BGR - robot base frame (더 진한 주황)
    COLOR_CYAN = (255, 255, 0)    # Cyan in BGR - workspace bounds 통일 색상

    # 주석용 작은 폰트 설정
    FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_SMALL = 0.3
    FONT_THICKNESS_SMALL = 1

    # 좌표축 그리기 헬퍼 함수
    def draw_coordinate_axes(img, origin_u, origin_v, origin_x_cm, origin_y_cm,
                              axis_length_cm, color_x, color_y, line_thickness=1,
                              rotation_matrix=None):
        """
        좌표축 (X, Y) 시각화

        Args:
            rotation_matrix: 2x2 또는 3x3 회전 행렬 (None이면 World frame 방향 사용)
                            Base frame의 축 방향을 World frame에서 표현할 때 사용
        """
        # 기본 축 방향 (World frame 기준)
        x_dir = np.array([1.0, 0.0])
        y_dir = np.array([0.0, 1.0])

        # 회전 행렬이 있으면 축 방향 변환
        # rotation_matrix는 World→Base 변환이므로, Base 축을 World에서 표현하려면 R^T 사용
        if rotation_matrix is not None:
            R = np.array(rotation_matrix)
            if R.shape[0] >= 2 and R.shape[1] >= 2:
                # R^T의 첫 번째 열 = Base X축의 World 방향
                # R^T의 두 번째 열 = Base Y축의 World 방향
                R_2d = R[:2, :2]
                x_dir = R_2d.T[:, 0]  # Base X in World
                y_dir = R_2d.T[:, 1]  # Base Y in World

        # X축 끝점 계산
        x_end_x_cm = origin_x_cm + axis_length_cm * x_dir[0]
        x_end_y_cm = origin_y_cm + axis_length_cm * x_dir[1]
        try:
            x_end_u, x_end_v = transformer.world_to_pixel(x_end_x_cm, x_end_y_cm)
            if 0 <= x_end_u < img.shape[1] and 0 <= x_end_v < img.shape[0]:
                cv2.arrowedLine(img, (origin_u, origin_v), (x_end_u, x_end_v),
                               color_x, line_thickness, tipLength=0.15)
                cv2.putText(img, "X", (x_end_u + 2, x_end_v),
                           FONT_SMALL, FONT_SCALE_SMALL, color_x, FONT_THICKNESS_SMALL)
        except:
            pass

        # Y축 끝점 계산
        y_end_x_cm = origin_x_cm + axis_length_cm * y_dir[0]
        y_end_y_cm = origin_y_cm + axis_length_cm * y_dir[1]
        try:
            y_end_u, y_end_v = transformer.world_to_pixel(y_end_x_cm, y_end_y_cm)
            if 0 <= y_end_u < img.shape[1] and 0 <= y_end_v < img.shape[0]:
                cv2.arrowedLine(img, (origin_u, origin_v), (y_end_u, y_end_v),
                               color_y, line_thickness, tipLength=0.15)
                cv2.putText(img, "Y", (y_end_u + 2, y_end_v),
                           FONT_SMALL, FONT_SCALE_SMALL, color_y, FONT_THICKNESS_SMALL)
        except:
            pass

        # Origin 점 표시
        cv2.circle(img, (origin_u, origin_v), 2, color_x, -1)

    def draw_frame_label(img, origin_u, origin_v, label, color, offset_x=5, offset_y=-5):
        """프레임 라벨 표시"""
        cv2.putText(img, label, (origin_u + offset_x, origin_v + offset_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # World origin (0, 0) 시각화 - 좌표축 형태
    try:
        origin_u, origin_v = transformer.world_to_pixel(0, 0)
        if 0 <= origin_u < image.shape[1] and 0 <= origin_v < image.shape[0]:
            draw_coordinate_axes(result, origin_u, origin_v, 0, 0,
                                axis_length_cm=3, color_x=COLOR_YELLOW, color_y=COLOR_YELLOW,
                                line_thickness=1, rotation_matrix=None)
            draw_frame_label(result, origin_u, origin_v, "World", COLOR_YELLOW,
                           offset_x=5, offset_y=-8)
    except Exception:
        pass

    # min_reach, max_reach 라벨 위치 저장용
    min_reach_label_pos = None
    max_reach_label_pos = None
    x_min_label_pos = None

    # Robot base 회전 행렬 로드
    base_rotation_matrix = None
    try:
        frame_config_path = PROJECT_ROOT / f"robot_configs/world2robot_matrices/robot{robot_id}_matrix.json"
        if frame_config_path.exists():
            with open(frame_config_path, 'r') as f:
                frame_config = json.load(f)
            raw_transform = frame_config.get("_raw_transform", {})
            base_rotation_matrix = raw_transform.get("rotation_matrix")
    except Exception as e:
        print(f"[Warning] Failed to load rotation matrix: {e}")

    # Robot base 위치 시각화 - 좌표축 형태 (회전 적용)
    try:
        robot_u, robot_v = transformer.world_to_pixel(robot_x_cm, robot_y_cm)
        if 0 <= robot_u < image.shape[1] and 0 <= robot_v < image.shape[0]:
            draw_coordinate_axes(result, robot_u, robot_v, robot_x_cm, robot_y_cm,
                                axis_length_cm=3, color_x=COLOR_ORANGE, color_y=COLOR_ORANGE,
                                line_thickness=1, rotation_matrix=base_rotation_matrix)
            draw_frame_label(result, robot_u, robot_v, "Robot Base", COLOR_ORANGE,
                           offset_x=-55, offset_y=-8)  # 왼쪽으로 이동

            # Min reach circle (Cyan - workspace bounds)
            for angle in range(0, 360, 15):
                rad = np.radians(angle)
                px = robot_x_cm + min_reach_cm * np.cos(rad)
                py = robot_y_cm + min_reach_cm * np.sin(rad)
                try:
                    pu, pv = transformer.world_to_pixel(px, py)
                    if 0 <= pu < image.shape[1] and 0 <= pv < image.shape[0]:
                        cv2.circle(result, (pu, pv), 2, COLOR_CYAN, -1)
                        # 오른쪽 방향(angle=0)에 라벨 위치 저장
                        if angle == 0:
                            min_reach_label_pos = (pu + 5, pv)
                except:
                    pass

            # Max reach circle (Cyan - workspace bounds)
            # Legend 영역 정의 (오른쪽 상단)
            legend_x_start = image.shape[1] - 120
            legend_y_end = 115

            max_reach_visible_points = []
            for angle in range(0, 360, 5):  # 더 촘촘하게 (10 -> 5)
                rad = np.radians(angle)
                px = robot_x_cm + max_reach_cm * np.cos(rad)
                py = robot_y_cm + max_reach_cm * np.sin(rad)
                try:
                    pu, pv = transformer.world_to_pixel(px, py)
                    if 0 <= pu < image.shape[1] and 0 <= pv < image.shape[0]:
                        # Legend 영역이면 점 그리지 않음
                        if pu > legend_x_start and pv < legend_y_end:
                            continue
                        cv2.circle(result, (pu, pv), 2, COLOR_CYAN, -1)
                        max_reach_visible_points.append((pu, pv, angle))
                except:
                    pass
            # 이미지 내 보이는 점 중 가장 아래쪽 점에 라벨 (legend와 겹치지 않게)
            if max_reach_visible_points:
                bottommost = max(max_reach_visible_points, key=lambda p: p[1])
                max_reach_label_pos = (bottommost[0] + 5, bottommost[1] - 5)
    except Exception:
        pass  # robot base가 이미지 밖이면 표시 안함

    # x_min_world 선 그리기 (세로선) - Cyan으로 통일
    x_min_pixels = []
    try:
        for y_cm in np.arange(y_min, y_max, 2):
            pu, pv = transformer.world_to_pixel(x_min_world_cm, y_cm)
            if 0 <= pu < image.shape[1] and 0 <= pv < image.shape[0]:
                cv2.circle(result, (pu, pv), 2, COLOR_CYAN, -1)
                x_min_pixels.append((pu, pv))
        # x_min 라벨 위치: 가장 위쪽 점의 오른쪽 (화면 내 보이도록)
        if x_min_pixels:
            top_point = min(x_min_pixels, key=lambda p: p[1])
            label_x = max(5, top_point[0] + 5)  # 최소 5px 이상
            x_min_label_pos = (label_x, top_point[1] + 10)
    except Exception:
        pass

    # Workspace bounds 라벨 표시 (작은 폰트)
    if min_reach_label_pos and 0 <= min_reach_label_pos[0] < image.shape[1]:
        cv2.putText(result, f"min_reach({min_reach_cm:.0f}cm)",
                    min_reach_label_pos, FONT_SMALL, FONT_SCALE_SMALL, COLOR_CYAN, FONT_THICKNESS_SMALL)

    if max_reach_label_pos and 0 <= max_reach_label_pos[0] < image.shape[1]:
        cv2.putText(result, f"max_reach({max_reach_cm:.0f}cm)",
                    max_reach_label_pos, FONT_SMALL, FONT_SCALE_SMALL, COLOR_CYAN, FONT_THICKNESS_SMALL)

    if x_min_label_pos and 0 <= x_min_label_pos[0] < image.shape[1]:
        cv2.putText(result, f"x_min({x_min_world_cm:.0f}cm)",
                    x_min_label_pos, FONT_SMALL, FONT_SCALE_SMALL, COLOR_CYAN, FONT_THICKNESS_SMALL)

    # 범례 추가 - 텍스트 색상으로 의미 매핑
    legend_x = image.shape[1] - 110
    cv2.putText(result, "Valid", (legend_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREEN, 1)
    cv2.putText(result, "Invalid", (legend_x, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, 1)
    cv2.putText(result, "World Frame", (legend_x, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)
    cv2.putText(result, "Robot Base", (legend_x, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ORANGE, 1)
    cv2.putText(result, "WS Bounds", (legend_x, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CYAN, 1)

    return result


def run_realtime_detection(
    queries: list,
    timeout: float = 10.0,
    unit: str = "m",
    config_path: str = "config/config.yaml",
    return_last_frame: bool = False,
    visualize: bool = True,
    return_extended: bool = False,
    robot_id: int = 3,
    external_camera=None,
    frame_callback=None,
    skip_workspace_filter: bool = False,
    early_exit: bool = False,
):
    """
    실시간 객체 검출 (시각화/비시각화 모드 통합)

    Args:
        queries: 찾을 객체 리스트
        timeout: 검출 유지 시간 (초) - 이 시간 동안 best confidence 결과 수집
        unit: 출력 단위 ("m" 또는 "cm")
        config_path: object_detection 설정 경로
        return_last_frame: True면 (positions, last_frame, last_vis_image) 튜플 반환
        visualize: True면 GUI 창 표시, False면 headless 모드
        return_extended: True면 확장 정보 반환 (bbox_size_m, grippable 포함)
        robot_id: 로봇 ID (2 또는 3) - workspace 시각화용
        external_camera: 외부에서 전달받은 카메라 인스턴스 (공유 모드)
                        - RealSenseD435 또는 get_frames() 메서드를 가진 객체
                        - 전달 시 내부에서 카메라를 생성/종료하지 않음
        frame_callback: 프레임마다 호출될 콜백 함수 (Recording 연동용)
                       - callback(color_frame, depth_frame) 형태
                       - Detection 중에도 Recording 가능하게 함
        skip_workspace_filter: True면 workspace 범위 필터링 건너뜀 (멀티 로봇용)
        early_exit: True면 모든 객체 검출 즉시 종료 (skill용), False면 timeout까지 대기 (파이프라인용)

    Note:
        Workspace overlay는 항상 표시됩니다.
        skip_workspace_filter=False일 때, 검출된 객체가 workspace 범위 밖이면 필터링됩니다.

    Returns:
        return_extended=False, return_last_frame=False:
            {query: (x, y, z) or None} 딕셔너리
        return_extended=False, return_last_frame=True:
            (positions, last_frame, last_vis_image)
        return_extended=True:
            {query: {"position": [x,y,z], "bbox_size_m": [w,h], "confidence": float} or None}
    """
    import cv2
    import numpy as np

    # object_detection 모듈 import
    sys.path.insert(0, str(PROJECT_ROOT / "object_detection"))

    try:
        from object_detection.camera import RealSenseD435
        from object_detection.detection import GroundingDINODetector
        from object_detection.localization import CoordinateTransformer
    except ImportError as e:
        print(f"[Error] Could not import object_detection modules: {e}")
        return {}

    mode_str = "Real-time" if visualize else "Headless"
    print(f"\n{'='*60}")
    print(f"{mode_str} Object Detection")
    print(f"{'='*60}")
    print(f"  Queries: {queries}")
    print(f"  Timeout: {timeout}s")
    print(f"  Unit: {unit}")
    print(f"  Visualize: {visualize}")
    print(f"  External Camera: {'Yes (shared)' if external_camera else 'No (internal)'}")
    if visualize:
        print(f"  Press 'q' to quit early, 's' to save current detection")
    print(f"{'='*60}\n")

    # 컴포넌트 초기화
    # external_camera가 전달되면 공유 모드, 아니면 내부에서 생성
    owns_camera = external_camera is None

    if external_camera:
        print("[System] Using external (shared) camera...")
        camera = external_camera
    else:
        print("[System] Initializing camera...")
        camera = RealSenseD435(width=640, height=480, fps=30)
        camera.start()
        # 카메라 안정화 대기 (warm-up) - 내부 생성 시에만
        print("[System] Waiting for camera warm-up...")
        time.sleep(1.5)

    print("[System] Loading detector (this may take a moment)...")
    detector = GroundingDINODetector(
        box_threshold=0.4,
        text_threshold=0.4,
        device="cuda"
    )
    detector.load_model()

    print("[System] Loading calibration...")
    calibration_file = PROJECT_ROOT / "robot_configs" / "pix2world_matrices" / "pix2world_transform_data.npz"
    transformer = CoordinateTransformer()
    if calibration_file.exists():
        transformer.load_calibration(str(calibration_file))
        transformer.set_camera_intrinsics(camera.get_intrinsics())
        print("[System] Calibration loaded")
    else:
        print("[Warning] No calibration found! Robot coordinates will be unavailable.")

    # Workspace 로드 (FrameTransformer 포함)
    print("[System] Loading workspace with frame transformer...")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from lerobot_cap.workspace import BaseWorkspace
    from lerobot_cap.transforms import FrameTransformer

    # FrameTransformer 생성 (robot_id에 맞는 config 로드)
    frame_config_path = PROJECT_ROOT / f"robot_configs/world2robot_matrices/robot{robot_id}_matrix.json"
    if frame_config_path.exists():
        frame_transformer = FrameTransformer(str(frame_config_path))
        print(f"[System] Frame transformer loaded: {frame_config_path.name}")
    else:
        frame_transformer = None
        print(f"[Warning] Frame config not found: {frame_config_path}")

    workspace = BaseWorkspace(frame_transformer=frame_transformer)
    print(f"[System] Workspace loaded: x_min={workspace.x_min_world}m, reach=[{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m")

    # 쿼리 문자열 생성 (Grounding DINO 형식)
    query_string = ". ".join(queries) + "."

    # 결과 저장용
    last_positions = {q: None for q in queries}
    last_confidences = {q: 0.0 for q in queries}
    last_bbox_sizes = {q: None for q in queries}  # bbox 크기 (meters)
    last_pixel_coords = {q: None for q in queries}  # 픽셀 좌표 (cx, cy)
    last_bbox_pixels = {q: None for q in queries}  # bbox 픽셀 좌표 (x1, y1, x2, y2)
    last_frame = None  # 마지막 프레임 저장
    last_vis_image = None  # 마지막 검출 시각화 이미지

    # 카메라 intrinsics (bbox 실제 크기 계산용)
    intrinsics = camera.get_intrinsics()
    fx = intrinsics.get('fx', 600)  # focal length x

    # 타이머 시작
    start_time = time.time()
    last_progress_log = start_time

    # 백그라운드 스레드 감지 (메인 스레드가 아니면 시각화 비활성화)
    import threading
    is_main_thread = threading.current_thread() is threading.main_thread()
    if visualize and not is_main_thread:
        print("[Warning] Running in background thread - visualization disabled (cv2.waitKey issue)")
        visualize = False

    print(f"\n[Detection] Starting real-time detection... (timeout={timeout}s)")

    try:
        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed

            # 타임아웃 체크
            if remaining <= 0:
                print(f"\n[Timeout] Detection completed ({elapsed:.1f}s)")
                break

            # Early quit: 모든 객체가 검출되면 즉시 종료 (early_exit=True일 때만)
            if early_exit:
                all_found = all(pos is not None for pos in last_positions.values())
                if all_found:
                    print(f"\n[Early Exit] All {len(queries)} objects detected ({elapsed:.1f}s)")
                    break

            # 1초마다 진행 상황 로그 출력
            if time.time() - last_progress_log >= 1.0:
                found_count = sum(1 for v in last_positions.values() if v is not None)
                print(f"[Detection] {elapsed:.1f}s / {timeout:.1f}s | Found: {found_count}/{len(queries)}")
                last_progress_log = time.time()

            # 프레임 획득
            color, depth = camera.get_frames()
            if color is None:
                continue

            # 프레임 콜백 호출 (Recording 연동용)
            if frame_callback is not None:
                try:
                    frame_callback(color, depth)
                except Exception as e:
                    print(f"[Warning] Frame callback error: {e}")

            # 마지막 프레임 저장
            last_frame = color.copy()

            # 객체 탐지
            vis_image, detections = detector.detect_and_draw(color, query_string)

            # 각 검출된 객체에 대해 매 프레임 좌표 계산 및 시각화
            for det in detections:
                # confidence threshold 체크 (0.4 이상만)
                if det.confidence < 0.4:
                    continue

                # 매칭되는 쿼리 찾기
                matched_query = None
                for q in queries:
                    if q.lower() in det.label.lower() or det.label.lower() in q.lower():
                        matched_query = q
                        break

                if matched_query:
                    cx, cy = det.center

                    # 로봇 좌표 계산 (매 프레임)
                    if transformer.is_ready:
                        depth_m = camera.get_depth_at_pixel(cx, cy, depth)
                        # 3D 캘리브레이션이 있으면 depth 사용, 없으면 2D fallback
                        world_coords_cm = transformer.pixel_depth_to_world(cx, cy, depth_m)

                        if unit == "m":
                            world_coords = (
                                world_coords_cm[0] / 100.0,
                                world_coords_cm[1] / 100.0,
                                world_coords_cm[2] / 100.0
                            )
                        else:
                            world_coords = (
                                world_coords_cm[0],
                                world_coords_cm[1],
                                world_coords_cm[2]
                            )

                        # Workspace 범위 체크 (is_reachable는 meters 단위 필요)
                        position_m = np.array([
                            world_coords_cm[0] / 100.0,
                            world_coords_cm[1] / 100.0,
                            world_coords_cm[2] / 100.0
                        ])

                        # Workspace 범위 체크 (skip_workspace_filter=True면 건너뜀)
                        if not skip_workspace_filter and not workspace.is_reachable(position_m):
                            # Workspace 밖 - 검출 결과 무시 (디버그 로그 추가)
                            # 디버그: 왜 필터링되었는지 출력
                            frame_info = workspace._transformer.get_frame_info("world") if workspace._transformer else None
                            if frame_info:
                                robot_pos = frame_info["robot_position"]
                                dx = position_m[0] - robot_pos[0]
                                dy = position_m[1] - robot_pos[1]
                                dist_from_robot = np.sqrt(dx*dx + dy*dy)
                                print(f"[DEBUG] {matched_query} filtered: world=({position_m[0]:.3f}, {position_m[1]:.3f}, {position_m[2]:.3f})m, "
                                      f"robot=({robot_pos[0]:.3f}, {robot_pos[1]:.3f})m, "
                                      f"dist={dist_from_robot:.3f}m (reach=[{workspace.min_reach:.3f}, {workspace.max_reach:.3f}]m)")
                            else:
                                print(f"[DEBUG] {matched_query} filtered: world=({position_m[0]:.3f}, {position_m[1]:.3f}, {position_m[2]:.3f})m (no frame info)")
                            continue

                        # 더 높은 confidence일 때 결과 저장
                        if det.confidence > last_confidences.get(matched_query, 0):
                            last_positions[matched_query] = world_coords
                            last_confidences[matched_query] = det.confidence

                            # 실시간 검출 로그 출력
                            print(f"[Detection] Found '{matched_query}': "
                                  f"pos=[{world_coords[0]:.4f}, {world_coords[1]:.4f}, {world_coords[2]:.4f}]{unit}, "
                                  f"conf={det.confidence:.3f}")

                            # 픽셀 좌표 저장 (Judge용)
                            last_pixel_coords[matched_query] = (int(cx), int(cy))

                            # bbox 실제 크기 계산 (depth 기반)
                            if det.bbox is not None and depth_m is not None and depth_m > 0:
                                x1, y1, x2, y2 = det.bbox
                                # bbox 픽셀 좌표 저장
                                last_bbox_pixels[matched_query] = (int(x1), int(y1), int(x2), int(y2))

                                width_px = x2 - x1
                                height_px = y2 - y1
                                # 실제 크기 = (픽셀 크기 * depth) / focal_length
                                width_m = (width_px * depth_m) / fx
                                height_m = (height_px * depth_m) / fx
                                last_bbox_sizes[matched_query] = (width_m, height_m)

                        # bbox 중심점에 마커 표시
                        cv2.circle(vis_image, (int(cx), int(cy)), 5, (0, 255, 0), -1)

                        # 좌표 정보를 bbox 중심점 바로 옆에 (x, y, z) 포맷으로 표시 (항상 cm 단위)
                        coord_text = f"({world_coords_cm[0]:.1f}, {world_coords_cm[1]:.1f}, {world_coords_cm[2]:.1f})cm"
                        text_x = int(cx) + 10
                        text_y = int(cy) + 5

                        # 배경 박스 (가독성)
                        (text_w, text_h), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(vis_image, (text_x - 2, text_y - text_h - 2),
                                     (text_x + text_w + 2, text_y + 4), (0, 0, 0), -1)

                        # 좌표 텍스트
                        cv2.putText(vis_image, coord_text,
                                   (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Workspace overlay 그리기 (항상 표시)
            vis_image = draw_workspace_overlay(vis_image, transformer, workspace, robot_id=robot_id)

            # 검출 시각화 이미지 저장 (좌표 포함, UI 요소 제외)
            last_vis_image = vis_image.copy()

            # GUI 모드일 때만 화면 표시
            if visualize:
                # 타이머 및 상태 표시 (UI 요소 - 저장에서 제외)
                timer_text = f"Time: {remaining:.1f}s"
                cv2.putText(vis_image, timer_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                query_text = f"Query: {', '.join(queries)}"
                cv2.putText(vis_image, query_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 검출된 객체 수 표시
                found_count = sum(1 for v in last_positions.values() if v is not None)
                found_text = f"Found: {found_count}/{len(queries)}"
                cv2.putText(vis_image, found_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 진행 바
                bar_width = 200
                bar_height = 10
                progress = elapsed / timeout
                cv2.rectangle(vis_image, (10, 100), (10 + bar_width, 100 + bar_height), (100, 100, 100), -1)
                cv2.rectangle(vis_image, (10, 100), (10 + int(bar_width * progress), 100 + bar_height), (0, 255, 0), -1)

                # 화면 표시
                cv2.imshow("Object Detection - Press 'q' to quit", vis_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[User] Quit requested")
                    break
                elif key == ord('s'):
                    # 현재 검출 결과 저장/출력
                    print("\n[Snapshot] Current detections:")
                    for q, pos in last_positions.items():
                        if pos:
                            print(f"  '{q}': ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) {unit}")
                        else:
                            print(f"  '{q}': Not found")

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        if visualize:
            cv2.destroyAllWindows()
        # 내부에서 생성한 카메라만 종료 (외부 공유 카메라는 유지)
        if owns_camera:
            camera.stop()
            print("[System] Cleanup complete (camera stopped)")
        else:
            print("[System] Cleanup complete (shared camera kept alive)")

    # Z값 처리: 3D 캘리브레이션이 없으면 기본값 사용
    # (3D 캘리브레이션이 있으면 pixel_depth_to_world에서 이미 정확한 Z 반환)
    if transformer.transform_matrix_3d is None:
        # 2D 캘리브레이션만 있는 경우: Z를 테이블 높이(1cm)로 고정
        for q in queries:
            if last_positions.get(q) is not None:
                pos = last_positions[q]
                last_positions[q] = (pos[0], pos[1], 0.01)

    # Z값 보정: z > 50cm인 경우 9cm로 하드코딩 (depth 센서 오류 보정)
    for q in queries:
        if last_positions.get(q) is not None:
            pos = last_positions[q]
            if pos[2] > 0.50:  # z > 50cm
                print(f"[Z-FIX] {q} z={pos[2]*100:.1f}cm > 50cm, forcing to 9cm")
                last_positions[q] = (pos[0], pos[1], 0.09)

    # 확장 정보 반환 모드
    if return_extended:
        from code_gen_lerobot.reset_execution.workspace import is_grippable

        def calculate_gripper_offset(bbox_size_m, position_world, default_offset=0.02):
            """
            Gripper offset 크기 계산.

            SO-101 비대칭 그리퍼 (고정 손가락 + 가동 손가락) 충돌 방지를 위한 오프셋.

            Returns:
                gripper_offset in meters
            """
            # 고정 4.5cm offset 사용
            FIXED_OFFSET = 0.045  # 4.5cm

            return FIXED_OFFSET

        extended_results = {}
        for q in queries:
            pos = last_positions.get(q)
            if pos is None:
                extended_results[q] = None
            else:
                bbox_size = last_bbox_sizes.get(q)
                grippable = is_grippable(bbox_size) if bbox_size else True

                # grippable한 물체만 gripper_offset 계산
                if grippable:
                    gripper_offset = calculate_gripper_offset(bbox_size, pos)
                else:
                    gripper_offset = 0.0  # 잡을 수 없는 물체는 offset 불필요

                extended_results[q] = {
                    "position": list(pos),
                    "bbox_size_m": list(bbox_size) if bbox_size else None,
                    "confidence": last_confidences.get(q, 0.0),
                    "grippable": grippable,
                    "gripper_offset": gripper_offset,
                    # Judge용 픽셀 좌표 정보
                    "pixel_coords": last_pixel_coords.get(q),  # (cx, cy)
                    "bbox_pixels": last_bbox_pixels.get(q),  # (x1, y1, x2, y2)
                }

        if return_last_frame:
            return extended_results, last_frame, last_vis_image
        return extended_results

    if return_last_frame:
        return last_positions, last_frame, last_vis_image
    return last_positions


if __name__ == "__main__":
    print("Object Detection Module for LeRobot Pipeline")
    print("=" * 60)
    print("This module is designed to be imported, not run directly.")
    print()
    print("Usage:")
    print("    from run_detect import run_realtime_detection")
    print()
    print("    positions = run_realtime_detection(")
    print('        queries=["red cup", "blue box"],')
    print("        timeout=10.0,")
    print("        unit='m',")
    print("        visualize=False,")
    print("        return_extended=True,")
    print("    )")
    print()
    print("For pipeline execution, use:")
    print("    ./run_forward_and_reset.sh")
    print("=" * 60)
