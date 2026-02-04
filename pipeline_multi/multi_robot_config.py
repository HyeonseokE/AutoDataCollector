"""
Multi-Robot Configuration

멀티로봇 파이프라인을 위한 설정 dataclass 정의
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class RobotConfig:
    """개별 로봇 설정"""
    robot_id: int

    # Frame transformation config
    frames_file: Optional[str] = None

    # Dataset recording config
    dataset_repo_id: Optional[str] = None

    # Robot-specific settings
    enabled: bool = True

    def __post_init__(self):
        # 기본 frames_file 경로 설정
        if self.frames_file is None:
            self.frames_file = f"robot_configs/world2robot_matrices/robot{self.robot_id}_matrix.json"

        # 기본 dataset_repo_id 설정
        if self.dataset_repo_id is None:
            self.dataset_repo_id = f"local/robot{self.robot_id}_dataset"

    def get_frames_path(self, base_path: str = ".") -> Path:
        """Get full path to frames file"""
        return Path(base_path) / self.frames_file

    def get_dataset_repo_id_with_timestamp(self) -> str:
        """Generate timestamped dataset repo ID"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.dataset_repo_id}_{timestamp}"


@dataclass
class MultiRobotConfig:
    """멀티로봇 파이프라인 전체 설정"""

    # 로봇 목록
    robots: List[RobotConfig] = field(default_factory=list)

    # 공유 카메라 설정
    camera_config_yaml: str = "pipeline_config/recording_config.yaml"

    # LLM 설정
    llm_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    use_server: bool = False

    # 실행 설정
    reset_mode: str = "original"  # "original" or "random"
    skip_judge: bool = False
    skip_reset: bool = False

    # 레코딩 설정
    record_dataset: bool = True
    recording_fps: int = 30

    # 감지 설정
    detection_timeout: float = 15.0
    visualize_detection: bool = False

    # 병렬 실행 설정
    parallel_timeout: float = 120.0  # 각 단계의 최대 실행 시간 (초)

    # 결과 저장 경로
    save_dir: str = "./pipeline_multi/results_multi"

    @classmethod
    def from_robot_ids(
        cls,
        robot_ids: List[int],
        **kwargs
    ) -> "MultiRobotConfig":
        """로봇 ID 목록으로 설정 생성"""
        robots = [RobotConfig(robot_id=rid) for rid in robot_ids]
        return cls(robots=robots, **kwargs)

    def get_robot(self, robot_id: int) -> Optional[RobotConfig]:
        """특정 로봇 설정 가져오기"""
        for robot in self.robots:
            if robot.robot_id == robot_id:
                return robot
        return None

    def get_enabled_robots(self) -> List[RobotConfig]:
        """활성화된 로봇 목록"""
        return [r for r in self.robots if r.enabled]

    def get_robot_ids(self) -> List[int]:
        """활성화된 로봇 ID 목록"""
        return [r.robot_id for r in self.get_enabled_robots()]

    def validate(self) -> bool:
        """설정 유효성 검사"""
        if not self.robots:
            raise ValueError("At least one robot must be configured")

        # 중복 ID 검사
        ids = [r.robot_id for r in self.robots]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate robot IDs found")

        # frames 파일 존재 확인
        for robot in self.get_enabled_robots():
            frames_path = robot.get_frames_path()
            if not frames_path.exists():
                raise FileNotFoundError(f"Frames file not found: {frames_path}")

        return True


def create_default_multi_robot_config(
    robot_ids: List[int] = [2, 3],
    **kwargs
) -> MultiRobotConfig:
    """기본 멀티로봇 설정 생성 헬퍼 함수"""
    return MultiRobotConfig.from_robot_ids(robot_ids, **kwargs)
