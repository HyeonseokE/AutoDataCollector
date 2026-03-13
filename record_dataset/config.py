"""
Configuration constants for LeRobot Dataset Recording.

SO-101 로봇의 데이터셋 스키마 및 기본 설정값을 정의합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# =============================================================================
# Robot Configuration (공식 LeRobot 형식과 동일)
# =============================================================================

# 공식 LeRobot SO101 로봇 타입 (so101_follower.py 참조)
ROBOT_TYPE = "so101_follower"

# Joint names for SO-101 (5 arm joints + 1 gripper)
# 공식 형식: "{motor}.pos" (예: "shoulder_pan.pos")
# so101_follower.py의 observation_features/action_features와 동일
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# 공식 LeRobot 형식의 joint names (motor.pos 형태)
JOINT_NAMES = [f"{motor}.pos" for motor in MOTOR_NAMES]

NUM_JOINTS = 6  # 5 arm + 1 gripper

# =============================================================================
# Recording Configuration
# =============================================================================

# Default recording FPS (LeRobot standard)
DEFAULT_FPS = 30

# Control loop frequency (current system)
CONTROL_HZ = 50

# Frame skip ratio for FPS synchronization
# 50Hz control -> 30Hz recording means record every ~1.67 frames
FRAME_SKIP_RATIO = CONTROL_HZ / DEFAULT_FPS

# =============================================================================
# Camera Configuration
# =============================================================================

@dataclass
class CameraConfigRecord:
    """카메라 레코딩 설정

    LeRobot 공식 설정 방식과 호환:
    --robot.cameras="{ front: {type: intelrealsense, index_or_path: /dev/video0, ...}}"
    """
    name: str              # 카메라 식별자 (예: "realsense", "innomaker")
    type: str              # "realsense" 또는 "opencv"
    enabled: bool = True   # 레코딩 활성화 여부
    width: int = 640
    height: int = 480
    fps: int = 30

    # 공통 설정 (LeRobot 호환)
    index_or_path: Optional[str] = None  # 장치 경로 (예: "/dev/video7") 또는 인덱스

    # OpenCV 전용 설정 (레거시 호환)
    device_path: Optional[str] = None  # 예: "/dev/video7" (index_or_path 대체 가능)
    device_index: Optional[int] = None
    fourcc: Optional[str] = "MJPG"

    # RealSense 전용 설정
    serial_number: Optional[str] = None
    enable_depth: bool = False

    def get_device_path(self) -> Optional[str]:
        """장치 경로 반환 (index_or_path 또는 device_path)"""
        return self.index_or_path or self.device_path

    def to_feature_key(self) -> str:
        """LeRobot 데이터셋 feature 키"""
        return f"observation.images.{self.name}"

    def to_feature_schema(self) -> Dict[str, Any]:
        """LeRobot 데이터셋 feature 스키마"""
        return {
            "dtype": "video",
            "shape": (self.height, self.width, 3),
            "names": ["height", "width", "channels"],
        }


# 기본 카메라 설정
DEFAULT_CAMERAS = [
    CameraConfigRecord(
        name="realsense",
        type="realsense",
        enabled=True,
        width=640,
        height=480,
        fps=30,
    ),
    CameraConfigRecord(
        name="innomaker",
        type="opencv",
        enabled=True,
        device_path="/dev/video7",
        width=640,
        height=480,
        fps=30,
        fourcc="MJPG",
    ),
]


def get_enabled_cameras() -> List[CameraConfigRecord]:
    """활성화된 카메라만 반환"""
    return [cam for cam in DEFAULT_CAMERAS if cam.enabled]


# =============================================================================
# Skill Features Configuration
# =============================================================================

# Observation feature keys (FK 기반 EE 자세 등)
OBSERVATION_FEATURE_KEYS = [
    "observation.ee_pos.robot_xyzrpy",
    "observation.ee_pos.world_xyzrpy",
    "observation.gripper_binary",
]


# Skill feature keys
SKILL_FEATURE_KEYS = [
    "skill.natural_language",
    "skill.type",
    "skill.progress",
    "skill.goal_position.joint",
    "skill.goal_position.world_xyzrpy",
    "skill.goal_position.robot_xyzrpy",
    "skill.goal_position.gripper",
]


def load_skill_features_from_yaml(yaml_path: str = None) -> Dict[str, bool]:
    """
    YAML에서 skill feature enabled 설정 로드.

    Args:
        yaml_path: recording_config.yaml 경로 (None이면 기본 경로)

    Returns:
        Dict[str, bool]: 각 skill feature의 enabled 여부
                         없으면 전부 True (기본값)
    """
    import yaml
    from pathlib import Path

    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "pipeline_config" / "recording_config.yaml"
    else:
        yaml_path = Path(yaml_path)

    # 기본값: 전부 True
    defaults = {key: True for key in SKILL_FEATURE_KEYS}

    if not yaml_path.exists():
        return defaults

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        sf = data.get("skill_features")
        if not sf:
            return defaults

        gp = sf.get("goal_position", {})
        # goal_position이 bool이면 하위 전체에 적용
        if isinstance(gp, bool):
            gp = {"joint": gp, "world_xyzrpy": gp, "robot_xyzrpy": gp, "gripper": gp}

        return {
            "skill.natural_language": sf.get("natural_language", True),
            "skill.type": sf.get("type", True),
            "skill.progress": sf.get("progress", True),
            "skill.goal_position.joint": gp.get("joint", True),
            "skill.goal_position.world_xyzrpy": gp.get("world_xyzrpy", True),
            "skill.goal_position.robot_xyzrpy": gp.get("robot_xyzrpy", True),
            "skill.goal_position.gripper": gp.get("gripper", True),
        }
    except Exception as e:
        print(f"[Config] Warning: Failed to load skill_features from {yaml_path}: {e}")
        return defaults


def load_observation_features_from_yaml(yaml_path: str = None) -> Dict[str, bool]:
    """
    YAML에서 observation feature enabled 설정 로드.

    Args:
        yaml_path: recording_config.yaml 경로 (None이면 기본 경로)

    Returns:
        Dict[str, bool]: 각 observation feature의 enabled 여부
                         없으면 전부 False (기본값 — 명시적 활성화 필요)
    """
    import yaml
    from pathlib import Path

    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "pipeline_config" / "recording_config.yaml"
    else:
        yaml_path = Path(yaml_path)

    # 기본값: 전부 False (기존 동작과 호환 — 명시적으로 켜야 함)
    defaults = {key: False for key in OBSERVATION_FEATURE_KEYS}

    if not yaml_path.exists():
        return defaults

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        of = data.get("observation_features")
        if not of:
            return defaults

        ep = of.get("ee_pos", {})
        # ee_pos가 bool이면 하위 전체에 적용
        if isinstance(ep, bool):
            ep = {"robot_xyzrpy": ep, "world_xyzrpy": ep}

        return {
            "observation.ee_pos.robot_xyzrpy": ep.get("robot_xyzrpy", False),
            "observation.ee_pos.world_xyzrpy": ep.get("world_xyzrpy", False),
            "observation.gripper_binary": of.get("gripper_binary", False),
        }
    except Exception as e:
        print(f"[Config] Warning: Failed to load observation_features from {yaml_path}: {e}")
        return defaults



def get_camera_feature_keys() -> List[str]:
    """활성화된 카메라 feature 키 목록"""
    return [cam.to_feature_key() for cam in get_enabled_cameras()]


# =============================================================================
# Dataset Features Schema (LeRobot v3.0 format)
# =============================================================================

def build_dataset_features(
    cameras: List[CameraConfigRecord] = None,
    skill_enabled: Dict[str, bool] = None,
    obs_enabled: Dict[str, bool] = None,
) -> Dict[str, Any]:
    """
    데이터셋 features 스키마 빌드

    Args:
        cameras: 카메라 설정 리스트 (None이면 DEFAULT_CAMERAS 사용)
        skill_enabled: skill feature별 enabled 여부 (None이면 전부 True)
        obs_enabled: observation feature별 enabled 여부 (None이면 전부 False)

    Returns:
        LeRobot 데이터셋 features 딕셔너리
    """
    if cameras is None:
        cameras = get_enabled_cameras()

    if skill_enabled is None:
        skill_enabled = {key: True for key in SKILL_FEATURE_KEYS}

    if obs_enabled is None:
        obs_enabled = {key: False for key in OBSERVATION_FEATURE_KEYS}


    features = {
        # Robot state: current joint positions (normalized -100 to +100)
        "observation.state": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },

        # Action: target joint positions (normalized -100 to +100)
        "action": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": JOINT_NAMES,
        },
    }

    # 카메라별 이미지 feature 추가
    for cam in cameras:
        features[cam.to_feature_key()] = cam.to_feature_schema()

    # Observation features (FK 기반 EE 자세 등, enabled인 것만 추가)
    obs_schemas = {
        "observation.ee_pos.robot_xyzrpy": {
            "dtype": "float32", "shape": (6,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        "observation.ee_pos.world_xyzrpy": {
            "dtype": "float32", "shape": (6,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        "observation.gripper_binary": {
            "dtype": "float32", "shape": (1,),
            "names": None,
        },
    }

    for key, schema in obs_schemas.items():
        if obs_enabled.get(key, False):
            features[key] = schema


    # Skill-level subgoal labels (enabled인 것만 추가)
    skill_schemas = {
        "skill.natural_language": {"dtype": "string", "shape": (1,), "names": None},
        "skill.type": {"dtype": "string", "shape": (1,), "names": None},
        "skill.progress": {"dtype": "float32", "shape": (1,), "names": None},
        "skill.goal_position.joint": {"dtype": "float32", "shape": (NUM_JOINTS,), "names": JOINT_NAMES},
        "skill.goal_position.world_xyzrpy": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "roll", "pitch", "yaw"]},
        "skill.goal_position.robot_xyzrpy": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "roll", "pitch", "yaw"]},
        "skill.goal_position.gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper.pos"]},
    }

    for key, schema in skill_schemas.items():
        if skill_enabled.get(key, True):
            features[key] = schema

    return features


# 기본 features (공식 LeRobot 형식)
# names 필드는 공식 형식: ["shoulder_pan.pos", "shoulder_lift.pos", ...]
DATASET_FEATURES = {
    # Robot state: current joint positions (normalized -100 to +100)
    # 공식 형식: names에 "{motor}.pos" 형태 사용
    "observation.state": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,  # ["shoulder_pan.pos", "shoulder_lift.pos", ...]
    },

    # Action: target joint positions (normalized -100 to +100)
    # 공식 형식: names에 "{motor}.pos" 형태 사용
    "action": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,  # ["shoulder_pan.pos", "shoulder_lift.pos", ...]
    },

    # Camera images (RGB) - 멀티 카메라
    "observation.images.realsense": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.innomaker": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },

    # Skill-level subgoal labels
    "skill.natural_language": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "skill.type": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "skill.progress": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "skill.goal_position.joint": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,
    },
    "skill.goal_position.world_xyzrpy": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "skill.goal_position.robot_xyzrpy": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "skill.goal_position.gripper": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper.pos"],
    },
}

# 레거시 호환 (기존 코드에서 사용)
DATASET_FEATURES_LEGACY = {
    "observation.state": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,
    },
    "action": {
        "dtype": "float32",
        "shape": (NUM_JOINTS,),
        "names": JOINT_NAMES,
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
}

# =============================================================================
# Dynamic Camera Loading from YAML
# =============================================================================

def load_cameras_from_yaml(yaml_path: str = None) -> List[CameraConfigRecord]:
    """
    YAML 파일에서 카메라 설정을 동적으로 로드

    Args:
        yaml_path: recording_config.yaml 경로 (None이면 기본 경로)

    Returns:
        List[CameraConfigRecord]: 카메라 설정 리스트
    """
    import yaml
    from pathlib import Path

    if yaml_path is None:
        # 기본 경로: pipeline_config/recording_config.yaml
        yaml_path = Path(__file__).parent.parent / "pipeline_config" / "recording_config.yaml"
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        print(f"[Config] Warning: {yaml_path} not found, using DEFAULT_CAMERAS")
        return DEFAULT_CAMERAS.copy()

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        cameras_data = data.get("cameras", [])
        if not cameras_data:
            print(f"[Config] Warning: No cameras defined in {yaml_path}, using DEFAULT_CAMERAS")
            return DEFAULT_CAMERAS.copy()

        cameras = []
        for cam_data in cameras_data:
            cam = CameraConfigRecord(
                name=cam_data.get("name", "camera"),
                type=cam_data.get("type", "opencv"),
                enabled=cam_data.get("enabled", True),
                width=cam_data.get("width", 640),
                height=cam_data.get("height", 480),
                fps=cam_data.get("fps", 30),
                index_or_path=cam_data.get("index_or_path"),  # LeRobot 호환
                device_path=cam_data.get("device_path"),      # 레거시 호환
                device_index=cam_data.get("device_index"),
                fourcc=cam_data.get("fourcc", "MJPG"),
                serial_number=cam_data.get("serial_number"),
                enable_depth=cam_data.get("enable_depth", False),
            )
            cameras.append(cam)

        print(f"[Config] Loaded {len(cameras)} camera(s) from {yaml_path}")
        for cam in cameras:
            status = "enabled" if cam.enabled else "disabled"
            print(f"  - {cam.name} ({cam.type}): {cam.width}x{cam.height}@{cam.fps}fps [{status}]")

        return cameras

    except Exception as e:
        print(f"[Config] Error loading {yaml_path}: {e}, using DEFAULT_CAMERAS")
        return DEFAULT_CAMERAS.copy()


def create_camera_manager_from_config(yaml_path: str = None):
    """
    YAML 설정에서 MultiCameraManager 생성

    Args:
        yaml_path: recording_config.yaml 경로

    Returns:
        MultiCameraManager 인스턴스
    """
    import sys
    from pathlib import Path

    # cameras 모듈 경로 추가
    cameras_path = Path(__file__).parent.parent / "cameras"
    if str(cameras_path.parent) not in sys.path:
        sys.path.insert(0, str(cameras_path.parent))

    from cameras import MultiCameraManager, RealSenseCameraConfig, OpenCVCameraConfig

    # YAML에서 카메라 설정 로드
    camera_configs = load_cameras_from_yaml(yaml_path)

    # cameras 모듈용 config 객체로 변환
    configs = []
    for cam in camera_configs:
        if not cam.enabled:
            continue

        if cam.type == "realsense":
            configs.append(RealSenseCameraConfig(
                name=cam.name,
                width=cam.width,
                height=cam.height,
                fps=cam.fps,
                serial_number=cam.serial_number,
                enable_depth=cam.enable_depth,
            ))
        else:  # opencv
            # index_or_path 우선, 없으면 device_path 사용
            device = cam.get_device_path() or "/dev/video0"
            configs.append(OpenCVCameraConfig(
                name=cam.name,
                device_path=device,
                device_index=cam.device_index,
                width=cam.width,
                height=cam.height,
                fps=cam.fps,
                fourcc=cam.fourcc,
            ))

    return MultiCameraManager(configs)


def build_features_from_yaml(yaml_path: str = None) -> Dict[str, Any]:
    """
    YAML 설정 기반으로 LeRobot dataset features 생성

    Args:
        yaml_path: recording_config.yaml 경로

    Returns:
        LeRobot dataset features dict
    """
    cameras = load_cameras_from_yaml(yaml_path)
    enabled_cameras = [cam for cam in cameras if cam.enabled]
    skill_enabled = load_skill_features_from_yaml(yaml_path)
    obs_enabled = load_observation_features_from_yaml(yaml_path)
    return build_dataset_features(enabled_cameras, skill_enabled, obs_enabled)


# =============================================================================
# Image Configuration
# =============================================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CHANNELS = 3

# =============================================================================
# Storage Configuration
# =============================================================================

# Default local cache path (follows HuggingFace convention)
# Actual path: ~/.cache/huggingface/lerobot/{repo_id}
DEFAULT_ROOT = None  # Will use HF_LEROBOT_HOME

# Chunk settings for large datasets
DEFAULT_CHUNKS_SIZE = 1000  # Files per chunk directory
DEFAULT_DATA_FILES_SIZE_MB = 500  # Max parquet file size
DEFAULT_VIDEO_FILES_SIZE_MB = 500  # Max video file size

# =============================================================================
# Normalization Range
# =============================================================================

# Current system uses -100 to +100 normalized range
NORM_MIN = -100.0
NORM_MAX = 100.0
