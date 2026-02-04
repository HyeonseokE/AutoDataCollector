"""
Multi-Robot Pipeline Package

병렬로 여러 로봇을 제어하고 데이터셋을 레코딩하는 파이프라인
"""

from .multi_robot_config import RobotConfig, MultiRobotConfig
from .detection_wrapper import SharedDetectionManager
from .execution_wrapper import ParallelExecutionManager
from .recording_wrapper import MultiRobotRecordingManager
from .multi_robot_pipeline import MultiRobotPipeline
from .results_saver import MultiRobotResultsSaver

__all__ = [
    "RobotConfig",
    "MultiRobotConfig",
    "SharedDetectionManager",
    "ParallelExecutionManager",
    "MultiRobotRecordingManager",
    "MultiRobotPipeline",
    "MultiRobotResultsSaver",
]
