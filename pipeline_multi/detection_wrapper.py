"""
Shared Detection Manager for Multi-Robot Pipeline

공유 카메라로 감지를 한 번 실행하고, 각 로봇 프레임으로 변환한 결과를 반환합니다.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.lerobot_cap.transforms import FrameTransformer
from src.lerobot_cap.workspace import BaseWorkspace


class SharedDetectionManager:
    """
    공유 감지 관리자

    1. 카메라로 감지 1회 실행
    2. 각 로봇의 FrameTransformer로 world → robot 프레임 변환
    3. 각 로봇의 Workspace에서 도달 가능한 객체만 필터링

    Returns:
        {robot_id: {object_name: {"position": [...], ...}}}
    """

    def __init__(
        self,
        robot_ids: List[int],
        camera_manager: Optional[Any] = None,
        verbose: bool = True,
    ):
        """
        Args:
            robot_ids: 로봇 ID 목록
            camera_manager: 공유 카메라 매니저 (None이면 내부에서 생성)
            verbose: 상세 로그 출력
        """
        self.robot_ids = robot_ids
        self.camera_manager = camera_manager
        self.verbose = verbose

        # 로봇별 FrameTransformer 및 Workspace 초기화
        self.frame_transformers: Dict[int, FrameTransformer] = {}
        self.workspaces: Dict[int, BaseWorkspace] = {}

        # 정상 검출 캐시 (z > 0.30m 오류 검출 시 사용)
        self._valid_detection_cache: Dict[str, Dict] = {}

        self._initialize_robot_frames()

    def _initialize_robot_frames(self):
        """로봇별 프레임 변환기 및 워크스페이스 초기화"""
        for robot_id in self.robot_ids:
            frame_config_path = PROJECT_ROOT / f"robot_configs/world2robot_matrices/robot{robot_id}_matrix.json"

            if frame_config_path.exists():
                transformer = FrameTransformer(str(frame_config_path))
                self.frame_transformers[robot_id] = transformer

                # 로봇별 Workspace 생성
                workspace = BaseWorkspace(frame_transformer=transformer)
                self.workspaces[robot_id] = workspace

                if self.verbose:
                    print(f"[SharedDetection] Robot {robot_id}: Frame transformer loaded")
            else:
                print(f"[Warning] Robot {robot_id}: Frame config not found: {frame_config_path}")
                self.frame_transformers[robot_id] = None
                self.workspaces[robot_id] = BaseWorkspace()

    def run_shared_detection(
        self,
        queries: List[str],
        timeout: float = 10.0,
        visualize: bool = False,
        external_camera: Optional[Any] = None,
    ) -> Dict[str, Dict]:
        """
        공유 감지 실행

        Args:
            queries: 검출할 객체 이름 목록
            timeout: 감지 타임아웃 (초)
            visualize: 시각화 창 표시 여부
            external_camera: 외부 카메라 인스턴스 (레코딩 공유용)

        Returns:
            검출된 객체 위치 (world frame)
            {object_name: {"position": [x,y,z], ...}}
        """
        from run_detect import run_realtime_detection

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SharedDetection] Running detection for {len(self.robot_ids)} robots")
            print(f"  Queries: {queries}")
            print(f"  Timeout: {timeout}s")
            print(f"{'='*60}")

        # 감지 실행 (world frame 좌표 반환)
        # robot_id는 시각화용으로만 사용 (첫 번째 로봇 기준)
        # skip_workspace_filter=True: 멀티 로봇에서는 모든 객체를 검출하고 나중에 각 로봇별로 필터링
        detected_positions = run_realtime_detection(
            queries=queries,
            timeout=timeout,
            unit="m",
            return_extended=True,
            visualize=visualize,
            robot_id=self.robot_ids[0] if self.robot_ids else 3,
            external_camera=external_camera,
            skip_workspace_filter=True,
        )

        # Z > 0.30m (30cm) 필터링: 오류 검출만 캐시값으로 대체
        MAX_Z_HEIGHT = 0.30
        for obj_name, obj_data in list(detected_positions.items()):
            if obj_data is not None:
                z = obj_data["position"][2]
                if z <= MAX_Z_HEIGHT:
                    # 정상 검출: 캐시에 저장
                    self._valid_detection_cache[obj_name] = obj_data.copy()
                else:
                    # 오류 검출: 캐시값 사용 (없으면 None)
                    cached = self._valid_detection_cache.get(obj_name)
                    if cached is not None:
                        if self.verbose:
                            print(f"  [Filter] {obj_name}: z={z:.3f}m > {MAX_Z_HEIGHT}m, using cached value")
                        detected_positions[obj_name] = cached.copy()
                    else:
                        if self.verbose:
                            print(f"  [Filter] {obj_name}: z={z:.3f}m > {MAX_Z_HEIGHT}m, no cache, filtered out")
                        detected_positions[obj_name] = None

        if self.verbose:
            found = sum(1 for v in detected_positions.values() if v is not None)
            print(f"[SharedDetection] Detected {found}/{len(queries)} objects (after z-filter)")

        return detected_positions

    def filter_reachable_positions(
        self,
        detected_positions: Dict[str, Dict],
        robot_id: int,
    ) -> Dict[str, Dict]:
        """
        특정 로봇에서 도달 가능한 위치만 필터링

        Args:
            detected_positions: 검출된 위치 (world frame)
            robot_id: 로봇 ID

        Returns:
            도달 가능한 객체만 포함된 딕셔너리
        """
        if robot_id not in self.workspaces:
            return detected_positions

        workspace = self.workspaces[robot_id]
        reachable = {}

        for obj_name, obj_data in detected_positions.items():
            if obj_data is None:
                reachable[obj_name] = None
                continue

            position = np.array(obj_data["position"])

            if workspace.is_reachable(position):
                reachable[obj_name] = obj_data
                if self.verbose:
                    print(f"  [Robot {robot_id}] {obj_name}: REACHABLE at {position}")
            else:
                reachable[obj_name] = None
                if self.verbose:
                    print(f"  [Robot {robot_id}] {obj_name}: OUT OF REACH at {position}")

        return reachable

    def get_robot_specific_positions(
        self,
        detected_positions: Dict[str, Dict],
    ) -> Dict[int, Dict[str, Dict]]:
        """
        각 로봇별 도달 가능한 위치 반환

        Args:
            detected_positions: 공유 감지 결과 (world frame)

        Returns:
            {robot_id: {object_name: {"position": [...], ...}}}
        """
        robot_positions = {}

        for robot_id in self.robot_ids:
            robot_positions[robot_id] = self.filter_reachable_positions(
                detected_positions, robot_id
            )

        return robot_positions

    def run_detection_for_all_robots(
        self,
        queries: List[str],
        timeout: float = 10.0,
        visualize: bool = False,
        external_camera: Optional[Any] = None,
        filter_by_reachability: bool = False,
    ) -> Dict[int, Dict[str, Dict]]:
        """
        전체 파이프라인: 감지 + 로봇별 결과 분배

        Args:
            queries: 검출할 객체 이름 목록
            timeout: 감지 타임아웃 (초)
            visualize: 시각화 창 표시 여부
            external_camera: 외부 카메라 인스턴스
            filter_by_reachability: 도달 불가능한 위치 필터링 여부

        Returns:
            {robot_id: {object_name: {"position": [...], ...}}}
        """
        # 1. 공유 감지 실행
        detected_positions = self.run_shared_detection(
            queries=queries,
            timeout=timeout,
            visualize=visualize,
            external_camera=external_camera,
        )

        # 2. 로봇별 결과 분배
        if filter_by_reachability:
            return self.get_robot_specific_positions(detected_positions)
        else:
            # 모든 로봇에 동일한 감지 결과 전달
            return {robot_id: detected_positions.copy() for robot_id in self.robot_ids}

    def transform_to_robot_frame(
        self,
        position_world: np.ndarray,
        robot_id: int,
    ) -> Optional[np.ndarray]:
        """
        World 프레임 위치를 로봇 base_link 프레임으로 변환

        Args:
            position_world: World 프레임 위치 [x, y, z]
            robot_id: 로봇 ID

        Returns:
            base_link 프레임 위치 [x, y, z] 또는 None (변환기 없는 경우)
        """
        transformer = self.frame_transformers.get(robot_id)
        if transformer is None:
            return None

        return transformer.transform_position(position_world, from_frame="world")

    def get_workspace(self, robot_id: int) -> Optional[BaseWorkspace]:
        """로봇별 Workspace 반환"""
        return self.workspaces.get(robot_id)

    def get_frame_transformer(self, robot_id: int) -> Optional[FrameTransformer]:
        """로봇별 FrameTransformer 반환"""
        return self.frame_transformers.get(robot_id)

    def assign_objects_to_robots(
        self,
        detected_positions: Dict[str, Dict],
        object_names: List[str] = None,
    ) -> Dict[int, str]:
        """
        각 로봇에게 가장 가까운 물체를 할당

        거리 기반으로 각 로봇이 담당할 물체를 결정합니다.
        물체 수와 로봇 수가 같아야 합니다.

        Args:
            detected_positions: 검출된 위치 (world frame)
                               {object_name: {"position": [x,y,z], ...}}
            object_names: 할당할 물체 이름 목록 (None이면 detected_positions의 모든 물체)

        Returns:
            {robot_id: assigned_object_name}
        """
        if object_names is None:
            # None이 아닌 물체만 선택
            object_names = [name for name, data in detected_positions.items() if data is not None]

        if len(object_names) != len(self.robot_ids):
            if self.verbose:
                print(f"[Warning] Object count ({len(object_names)}) != Robot count ({len(self.robot_ids)})")
            # 물체가 더 많으면 로봇 수만큼만 사용
            object_names = object_names[:len(self.robot_ids)]

        # 각 로봇의 base 위치 가져오기
        robot_bases = {}
        for robot_id in self.robot_ids:
            transformer = self.frame_transformers.get(robot_id)
            if transformer and transformer.has_frame("world"):
                frame_info = transformer.get_frame_info("world")
                robot_bases[robot_id] = np.array(frame_info["robot_position"][:2])  # XY만
            else:
                robot_bases[robot_id] = np.array([0, 0])

        # 물체 위치
        object_positions_xy = {}
        for name in object_names:
            if detected_positions.get(name):
                pos = detected_positions[name]["position"]
                object_positions_xy[name] = np.array(pos[:2])  # XY만

        if self.verbose:
            print(f"\n[Object Assignment] Assigning {len(object_names)} objects to {len(self.robot_ids)} robots")
            for rid, base in robot_bases.items():
                print(f"  Robot {rid} base: ({base[0]:.3f}, {base[1]:.3f})")
            for name, pos in object_positions_xy.items():
                print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f})")

        # 거리 행렬 계산
        assignments = {}
        remaining_objects = list(object_names)
        remaining_robots = list(self.robot_ids)

        # Greedy 할당: 가장 가까운 (로봇, 물체) 쌍부터 할당
        while remaining_robots and remaining_objects:
            min_dist = float('inf')
            best_robot = None
            best_object = None

            for robot_id in remaining_robots:
                for obj_name in remaining_objects:
                    if obj_name not in object_positions_xy:
                        continue
                    dist = np.linalg.norm(robot_bases[robot_id] - object_positions_xy[obj_name])
                    if dist < min_dist:
                        min_dist = dist
                        best_robot = robot_id
                        best_object = obj_name

            if best_robot is not None and best_object is not None:
                assignments[best_robot] = best_object
                remaining_robots.remove(best_robot)
                remaining_objects.remove(best_object)
                if self.verbose:
                    print(f"  -> Robot {best_robot} assigned to '{best_object}' (dist={min_dist:.3f}m)")

        return assignments
