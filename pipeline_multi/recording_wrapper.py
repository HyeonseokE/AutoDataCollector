"""
Multi-Robot Recording Manager

로봇별 별도 데이터셋을 생성하고 관리합니다.
각 로봇은 독립적인 DatasetRecorder 인스턴스를 가집니다.

전략: 로봇별 별도 데이터셋
  - Robot 2 → local/robot2_dataset_<timestamp>
  - Robot 3 → local/robot3_dataset_<timestamp>
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from record_dataset.recorder import DatasetRecorder
from .multi_robot_config import RobotConfig


class MultiRobotRecordingManager:
    """
    멀티로봇 레코딩 관리자

    각 로봇별로 별도의 DatasetRecorder 인스턴스를 관리합니다.
    기존 LeRobot 스키마를 수정 없이 호환됩니다.

    Usage:
        manager = MultiRobotRecordingManager(robot_configs, fps=30)
        manager.start_episodes(task="pick up the red cup")

        # 각 로봇 실행 후
        manager.end_episode(robot_id=2)
        manager.end_episode(robot_id=3)

        manager.finalize_all()
    """

    def __init__(
        self,
        robot_configs: List[RobotConfig],
        fps: int = 30,
        config_yaml: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            robot_configs: 로봇 설정 목록
            fps: 레코딩 FPS
            config_yaml: 카메라 설정 YAML 경로
            verbose: 상세 로그 출력
        """
        self.robot_configs = robot_configs
        self.fps = fps
        self.config_yaml = config_yaml or "pipeline_config/recording_config.yaml"
        self.verbose = verbose

        # 타임스탬프 (동일 세션의 모든 로봇 동일)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 로봇별 DatasetRecorder
        self.recorders: Dict[int, DatasetRecorder] = {}

        # 초기화 상태
        self._initialized = False
        self._finalized = False

    def initialize(self) -> bool:
        """
        모든 로봇의 DatasetRecorder 초기화

        Returns:
            성공 여부
        """
        if self._initialized:
            if self.verbose:
                print("[MultiRobotRecording] Already initialized")
            return True

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[MultiRobotRecording] Initializing recorders for {len(self.robot_configs)} robots")
            print(f"  Session: {self.session_timestamp}")
            print(f"  FPS: {self.fps}")
            print(f"{'='*60}")

        success = True

        for robot_config in self.robot_configs:
            if not robot_config.enabled:
                continue

            robot_id = robot_config.robot_id
            repo_id = f"{robot_config.dataset_repo_id}_{self.session_timestamp}"

            try:
                if self.verbose:
                    print(f"\n[Robot {robot_id}] Creating dataset: {repo_id}")

                recorder = DatasetRecorder(
                    repo_id=repo_id,
                    fps=self.fps,
                    config_yaml=self.config_yaml,
                )

                self.recorders[robot_id] = recorder

                if self.verbose:
                    print(f"[Robot {robot_id}] Recorder initialized successfully")

            except Exception as e:
                print(f"[Robot {robot_id}] Failed to initialize recorder: {e}")
                success = False

        self._initialized = success

        if self.verbose:
            print(f"\n[MultiRobotRecording] Initialization {'complete' if success else 'failed'}")
            print(f"  Active recorders: {list(self.recorders.keys())}")

        return success

    def get_recorder(self, robot_id: int) -> Optional[DatasetRecorder]:
        """특정 로봇의 Recorder 반환"""
        return self.recorders.get(robot_id)

    def start_episode(self, robot_id: int, task: str) -> bool:
        """
        특정 로봇의 에피소드 시작

        Args:
            robot_id: 로봇 ID
            task: 태스크 설명

        Returns:
            성공 여부
        """
        recorder = self.recorders.get(robot_id)
        if recorder is None:
            print(f"[MultiRobotRecording] Robot {robot_id} not found")
            return False

        try:
            recorder.start_episode(task)
            return True
        except Exception as e:
            print(f"[Robot {robot_id}] Failed to start episode: {e}")
            return False

    def start_episodes(self, task: str) -> Dict[int, bool]:
        """
        모든 로봇의 에피소드 동시 시작

        Args:
            task: 태스크 설명 (모든 로봇 동일)

        Returns:
            {robot_id: success}
        """
        results = {}

        if self.verbose:
            print(f"\n[MultiRobotRecording] Starting episodes: '{task}'")

        for robot_id in self.recorders:
            results[robot_id] = self.start_episode(robot_id, task)

        return results

    def end_episode(self, robot_id: int, discard: bool = False) -> Optional[Dict]:
        """
        특정 로봇의 에피소드 종료

        Args:
            robot_id: 로봇 ID
            discard: 에피소드 버리기 여부

        Returns:
            에피소드 정보 또는 None
        """
        recorder = self.recorders.get(robot_id)
        if recorder is None:
            print(f"[MultiRobotRecording] Robot {robot_id} not found")
            return None

        try:
            return recorder.end_episode(discard=discard)
        except Exception as e:
            print(f"[Robot {robot_id}] Failed to end episode: {e}")
            return None

    def end_episodes(self, discard_dict: Optional[Dict[int, bool]] = None) -> Dict[int, Dict]:
        """
        모든 로봇의 에피소드 동시 종료

        Args:
            discard_dict: {robot_id: discard_flag} (None이면 모두 저장)

        Returns:
            {robot_id: episode_info}
        """
        results = {}
        discard_dict = discard_dict or {}

        if self.verbose:
            print(f"\n[MultiRobotRecording] Ending episodes")

        for robot_id in self.recorders:
            discard = discard_dict.get(robot_id, False)
            results[robot_id] = self.end_episode(robot_id, discard=discard)

        return results

    def finalize(self, robot_id: int) -> bool:
        """
        특정 로봇의 데이터셋 완료 처리

        Args:
            robot_id: 로봇 ID

        Returns:
            성공 여부
        """
        recorder = self.recorders.get(robot_id)
        if recorder is None:
            return False

        try:
            recorder.finalize()
            return True
        except Exception as e:
            print(f"[Robot {robot_id}] Failed to finalize: {e}")
            return False

    def finalize_all(self) -> Dict[int, bool]:
        """
        모든 로봇의 데이터셋 완료 처리

        Returns:
            {robot_id: success}
        """
        if self._finalized:
            if self.verbose:
                print("[MultiRobotRecording] Already finalized")
            return {rid: True for rid in self.recorders}

        results = {}

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[MultiRobotRecording] Finalizing all datasets")
            print(f"{'='*60}")

        for robot_id in self.recorders:
            results[robot_id] = self.finalize(robot_id)

        self._finalized = True

        if self.verbose:
            print(f"\n[MultiRobotRecording] Finalization complete")
            for robot_id, success in results.items():
                status = "SUCCESS" if success else "FAILED"
                recorder = self.recorders.get(robot_id)
                path = recorder._dataset.root if recorder and recorder._dataset else "N/A"
                print(f"  Robot {robot_id}: {status} - {path}")

        return results

    def get_stats(self) -> Dict[int, Dict[str, Any]]:
        """모든 로봇의 레코딩 통계 반환"""
        return {
            robot_id: recorder.get_stats()
            for robot_id, recorder in self.recorders.items()
        }

    def is_recording(self, robot_id: int) -> bool:
        """특정 로봇이 레코딩 중인지 확인"""
        recorder = self.recorders.get(robot_id)
        return recorder.is_recording if recorder else False

    def any_recording(self) -> bool:
        """하나라도 레코딩 중인지 확인"""
        return any(r.is_recording for r in self.recorders.values())

    def all_recording(self) -> bool:
        """모든 로봇이 레코딩 중인지 확인"""
        return all(r.is_recording for r in self.recorders.values())

    def get_robot_ids(self) -> List[int]:
        """레코더가 있는 로봇 ID 목록"""
        return list(self.recorders.keys())

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto finalize"""
        if self.any_recording():
            # 레코딩 중인 에피소드 종료
            for robot_id in self.recorders:
                if self.is_recording(robot_id):
                    self.end_episode(robot_id, discard=exc_type is not None)

        self.finalize_all()
        return False


def create_recording_manager(
    robot_configs: List[RobotConfig],
    fps: int = 30,
    config_yaml: Optional[str] = None,
    auto_initialize: bool = True,
    verbose: bool = True,
) -> MultiRobotRecordingManager:
    """
    MultiRobotRecordingManager 생성 헬퍼 함수

    Args:
        robot_configs: 로봇 설정 목록
        fps: 레코딩 FPS
        config_yaml: 카메라 설정 YAML
        auto_initialize: 자동 초기화 여부
        verbose: 상세 로그

    Returns:
        초기화된 MultiRobotRecordingManager
    """
    manager = MultiRobotRecordingManager(
        robot_configs=robot_configs,
        fps=fps,
        config_yaml=config_yaml,
        verbose=verbose,
    )

    if auto_initialize:
        manager.initialize()

    return manager
