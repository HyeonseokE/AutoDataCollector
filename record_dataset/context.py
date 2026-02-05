"""
Global Recording Context

생성된 코드 수정 없이 LeRobotSkills에 레코딩 콜백을 주입하기 위한 전역 컨텍스트.

비동기 카메라 캡처:
- 백그라운드 스레드에서 연속 캡처 (60fps)
- 제어 루프는 최신 이미지만 즉시 가져감 (블로킹 없음!)
- 50Hz 제어 루프 달성 가능

통합 카메라 지원:
- 공식 LeRobot 방식과 동일하게 MultiCameraManager만 사용
- 카메라 수와 관계없이 동일한 코드 경로

Usage:
    from record_dataset.context import RecordingContext
    from record_dataset.config import create_camera_manager_from_config

    camera_manager = create_camera_manager_from_config()
    camera_manager.connect_all()

    RecordingContext.setup(recorder=recorder, camera_manager=camera_manager)
    exec(generated_code)
    RecordingContext.clear()
"""

import time
from typing import Optional, Any, Dict, TYPE_CHECKING
from threading import Lock

import numpy as np

if TYPE_CHECKING:
    from .recorder import DatasetRecorder

# MultiCameraManager import
try:
    import sys
    from pathlib import Path
    cameras_path = Path(__file__).parent.parent / "cameras"
    if str(cameras_path.parent) not in sys.path:
        sys.path.insert(0, str(cameras_path.parent))
    from cameras import MultiCameraManager
    HAS_MULTI_CAMERA = True
except ImportError:
    HAS_MULTI_CAMERA = False
    MultiCameraManager = None

# AsyncCameraCapture import
try:
    from .async_camera import AsyncCameraCapture
    HAS_ASYNC_CAMERA = True
except ImportError:
    HAS_ASYNC_CAMERA = False
    AsyncCameraCapture = None


class RecordingContext:
    """
    전역 레코딩 컨텍스트 (Singleton)

    LeRobotSkills가 생성될 때 이 컨텍스트에서 콜백을 자동으로 획득합니다.

    비동기 카메라 캡처:
    - 백그라운드 스레드에서 연속 캡처 (60fps)
    - 제어 루프는 블로킹 없이 최신 이미지만 가져감
    - 50Hz 제어 루프 달성 가능!

    통합 카메라 접근 (공식 LeRobot 방식):
    - MultiCameraManager만 사용 (단일/다중 카메라 구분 없음)
    - 카메라 수와 관계없이 동일한 코드 경로
    """

    _instance: Optional['RecordingContext'] = None
    _lock = Lock()

    # Class-level state
    _recorder: Optional['DatasetRecorder'] = None
    _camera_manager: Any = None  # MultiCameraManager (통합)
    _async_capture: Any = None  # AsyncCameraCapture (비동기 캡처)
    _is_active: bool = False
    _target_fps: int = 30
    _control_hz: int = 50

    # FPS synchronization
    _frame_skip_ratio: float = 1.0
    _step_counter: int = 0
    _last_record_step: int = -1
    _start_time: Optional[float] = None

    # Statistics
    _recorded_frames: int = 0
    _skipped_frames: int = 0
    _camera_errors: int = 0

    # Skill-level subgoal info
    _current_skill_label: Optional[str] = None
    _current_skill_type: Optional[str] = None
    _skill_start_time: Optional[float] = None
    _skill_duration: Optional[float] = None
    _current_goal_joint: Optional[np.ndarray] = None
    _current_goal_world_xyzrpy: Optional[np.ndarray] = None
    _current_goal_robot_xyzrpy: Optional[np.ndarray] = None
    _current_goal_gripper: Optional[float] = None
    _skill_start_state: Optional[np.ndarray] = None

    def __init__(
        self,
        recorder: 'DatasetRecorder',
        camera_manager: Any,
        target_fps: int = 30,
        control_hz: int = 50,
    ):
        """Context manager 방식으로 사용할 때의 초기화"""
        self._init_recorder = recorder
        self._init_camera_manager = camera_manager
        self._init_target_fps = target_fps
        self._init_control_hz = control_hz

    def __enter__(self):
        """Context manager entry"""
        RecordingContext.setup(
            recorder=self._init_recorder,
            camera_manager=self._init_camera_manager,
            target_fps=self._init_target_fps,
            control_hz=self._init_control_hz,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        RecordingContext.clear()
        return False

    @classmethod
    def setup(
        cls,
        recorder: 'DatasetRecorder',
        camera_manager: Any,
        target_fps: int = 30,
        control_hz: int = 50,
        use_async_capture: bool = True,  # 비동기 캡처 사용 여부
    ) -> None:
        """
        레코딩 컨텍스트 설정 (비동기 카메라 캡처 지원)

        Args:
            recorder: DatasetRecorder 인스턴스
            camera_manager: MultiCameraManager 인스턴스
            target_fps: 목표 레코딩 FPS
            control_hz: 제어 루프 주파수
            use_async_capture: 비동기 캡처 사용 (True 권장, 50Hz 달성 가능)
        """
        with cls._lock:
            cls._recorder = recorder
            cls._camera_manager = camera_manager
            cls._target_fps = target_fps
            cls._control_hz = control_hz
            cls._frame_skip_ratio = control_hz / target_fps
            cls._is_active = True

            # Reset counters
            cls._step_counter = 0
            cls._last_record_step = -1
            cls._start_time = None
            cls._recorded_frames = 0
            cls._skipped_frames = 0
            cls._camera_errors = 0

            # 비동기 카메라 캡처 시작
            if use_async_capture and HAS_ASYNC_CAMERA and camera_manager is not None:
                try:
                    cls._async_capture = AsyncCameraCapture(
                        camera_manager=camera_manager,
                        capture_fps=60,  # 제어 루프(50Hz)보다 빠르게 캡처
                    )
                    cls._async_capture.start()
                    print(f"[RecordingContext] ✓ Async camera capture enabled (60fps)")
                except Exception as e:
                    print(f"[RecordingContext] Warning: Failed to start async capture: {e}")
                    print(f"[RecordingContext] Falling back to sync capture")
                    cls._async_capture = None
            else:
                cls._async_capture = None
                if use_async_capture:
                    print(f"[RecordingContext] Async capture disabled (using sync mode)")

            print(f"[RecordingContext] Setup complete")
            print(f"  Target FPS: {target_fps}, Control Hz: {control_hz}")
            print(f"  Frame skip ratio: {cls._frame_skip_ratio:.2f}")
            if cls._camera_manager is not None and hasattr(cls._camera_manager, 'camera_names'):
                print(f"  Cameras: {cls._camera_manager.camera_names}")

    @classmethod
    def clear(cls) -> None:
        """레코딩 컨텍스트 해제"""
        with cls._lock:
            # 비동기 캡처 스레드 정지
            if cls._async_capture is not None:
                try:
                    cls._async_capture.stop()
                except Exception as e:
                    print(f"[RecordingContext] Warning: Failed to stop async capture: {e}")
                cls._async_capture = None

            if cls._is_active:
                print(f"[RecordingContext] Cleared")
                print(f"  Recorded frames: {cls._recorded_frames}")
                print(f"  Skipped frames: {cls._skipped_frames}")
                if cls._camera_errors > 0:
                    print(f"  Camera errors: {cls._camera_errors}")

            cls._recorder = None
            cls._camera_manager = None
            cls._is_active = False
            cls._step_counter = 0
            cls._last_record_step = -1
            cls._start_time = None

    @classmethod
    def is_active(cls) -> bool:
        """컨텍스트가 활성 상태인지 확인"""
        return cls._is_active and cls._recorder is not None

    @classmethod
    def set_skill_label(cls, label: str) -> None:
        """현재 실행 중인 스킬의 자연어 라벨 설정 (레거시 호환)"""
        with cls._lock:
            cls._current_skill_label = label

    @classmethod
    def clear_skill_label(cls) -> None:
        """스킬 라벨 해제 (레거시 호환)"""
        with cls._lock:
            cls._current_skill_label = None

    @classmethod
    def get_skill_label(cls) -> Optional[str]:
        """현재 스킬 라벨 반환"""
        return cls._current_skill_label

    @classmethod
    def set_skill_info(
        cls,
        label: str,
        skill_type: str,
        duration: float,
        goal_joint: np.ndarray,
        goal_world_xyzrpy: np.ndarray,
        goal_robot_xyzrpy: np.ndarray,
        goal_gripper: float,
        start_state: Optional[np.ndarray] = None,
    ) -> None:
        """스킬 정보 설정 (스킬 시작 시 호출)"""
        with cls._lock:
            cls._current_skill_label = label
            cls._current_skill_type = skill_type
            cls._skill_start_time = time.time()
            cls._skill_duration = duration
            cls._current_goal_joint = np.asarray(goal_joint, dtype=np.float32)
            cls._current_goal_world_xyzrpy = np.asarray(goal_world_xyzrpy, dtype=np.float32)
            cls._current_goal_robot_xyzrpy = np.asarray(goal_robot_xyzrpy, dtype=np.float32)
            cls._current_goal_gripper = float(goal_gripper)
            cls._skill_start_state = np.asarray(start_state, dtype=np.float32) if start_state is not None else None
            if cls._is_active:
                print(f"[RecordingContext] Skill: {skill_type} - {label}")

    @classmethod
    def clear_skill_info(cls) -> None:
        """스킬 정보 해제 (스킬 종료 시 호출)"""
        with cls._lock:
            cls._current_skill_label = None
            cls._current_skill_type = None
            cls._skill_start_time = None
            cls._skill_duration = None
            cls._current_goal_joint = None
            cls._current_goal_world_xyzrpy = None
            cls._current_goal_robot_xyzrpy = None
            cls._current_goal_gripper = None
            cls._skill_start_state = None

    @classmethod
    def get_skill_progress(cls, current_state: Optional[np.ndarray] = None) -> float:
        """스킬 진행률 반환 (0.0 ~ 1.0, state-based with time-based fallback)

        State-based: progress = 1.0 - (||goal - current|| / ||goal - start||)
        Fallback to time-based (elapsed / duration) when start_state is None.
        """
        # State-based progress (primary)
        if (
            current_state is not None
            and cls._skill_start_state is not None
            and cls._current_goal_joint is not None
        ):
            goal = cls._current_goal_joint
            start = cls._skill_start_state
            current = np.asarray(current_state, dtype=np.float32)
            start_to_goal = np.linalg.norm(goal - start)
            if start_to_goal < 1e-6:
                return 1.0  # already at goal
            current_to_goal = np.linalg.norm(goal - current)
            progress = 1.0 - (current_to_goal / start_to_goal)
            return float(np.clip(progress, 0.0, 1.0))

        # Time-based fallback
        if cls._skill_start_time is None or cls._skill_duration is None:
            return 0.0
        elapsed = time.time() - cls._skill_start_time
        return min(elapsed / cls._skill_duration, 1.0)

    @classmethod
    def get_skill_info(cls, current_state: Optional[np.ndarray] = None) -> dict:
        """현재 스킬 정보 반환

        Args:
            current_state: 현재 로봇 상태 (6 joints, normalized). State-based progress 계산에 사용.
        """
        return {
            "label": cls._current_skill_label or "",
            "type": cls._current_skill_type or "",
            "progress": cls.get_skill_progress(current_state=current_state),
            "goal_joint": cls._current_goal_joint if cls._current_goal_joint is not None else np.zeros(6, dtype=np.float32),
            "goal_world_xyzrpy": cls._current_goal_world_xyzrpy if cls._current_goal_world_xyzrpy is not None else np.zeros(6, dtype=np.float32),
            "goal_robot_xyzrpy": cls._current_goal_robot_xyzrpy if cls._current_goal_robot_xyzrpy is not None else np.zeros(6, dtype=np.float32),
            "goal_gripper": cls._current_goal_gripper if cls._current_goal_gripper is not None else 0.0,
        }

    @classmethod
    def reset_episode(cls) -> None:
        """에피소드 시작 시 카운터 리셋"""
        with cls._lock:
            cls._step_counter = 0
            cls._last_record_step = -1
            cls._start_time = time.time()
            cls._recorded_frames = 0
            cls._skipped_frames = 0

    @classmethod
    def should_record(cls) -> bool:
        """
        현재 스텝에서 레코딩해야 하는지 판단 (FPS 동기화)

        Returns:
            레코딩 여부
        """
        if not cls._is_active or cls._recorder is None:
            return False

        if not cls._recorder.is_recording:
            return False

        # FPS 동기화
        target_frame = int(cls._step_counter / cls._frame_skip_ratio)
        return target_frame > cls._last_record_step

    @classmethod
    def record_step(
        cls,
        state: np.ndarray,
        action: np.ndarray,
    ) -> bool:
        """
        제어 루프에서 호출되는 레코딩 함수 (통합 방식)

        Args:
            state: 현재 로봇 상태 (6 joints, normalized)
            action: 타겟 액션 (6 joints, normalized)

        Returns:
            레코딩 성공 여부
        """
        if not cls._is_active or cls._recorder is None:
            return False

        if not cls._recorder.is_recording:
            cls._step_counter += 1
            return False

        # FPS 동기화 체크
        if not cls.should_record():
            cls._step_counter += 1
            cls._skipped_frames += 1
            return False

        try:
            # 데이터 타입 확인
            state = np.asarray(state, dtype=np.float32)
            action = np.asarray(action, dtype=np.float32)

            # Shape 확인 및 조정
            if state.shape == (5,):
                # 5 arm joints만 있으면 gripper 추가
                state = np.concatenate([state, [0.0]], dtype=np.float32)
            if action.shape == (5,):
                action = np.concatenate([action, [0.0]], dtype=np.float32)

            # 통합 방식: MultiCameraManager로 모든 카메라에서 캡처
            images = cls._capture_images()
            if not images:
                cls._step_counter += 1
                cls._camera_errors += 1
                return False

            # 멀티 카메라 레코딩 (통합) + 스킬 라벨
            cls._recorder.record_frame_multi(
                observation=state,
                action=action,
                images=images,
                skill_label=cls._current_skill_label,
            )

            # 상태 업데이트
            cls._last_record_step = int(cls._step_counter / cls._frame_skip_ratio)
            cls._recorded_frames += 1
            cls._step_counter += 1

            return True

        except Exception as e:
            print(f"[RecordingContext] Error recording: {e}")
            cls._step_counter += 1
            return False

    @classmethod
    def _capture_images(cls) -> Optional[Dict[str, np.ndarray]]:
        """
        카메라 이미지 캡처 (비동기 또는 동기)

        비동기 캡처가 활성화된 경우:
        - 백그라운드 스레드에서 캡처된 최신 이미지를 즉시 반환 (블로킹 없음!)
        - 제어 루프가 50Hz로 동작 가능

        동기 캡처 (fallback):
        - async_read_all()을 직접 호출 (블로킹 발생, ~25ms)

        Returns:
            Dict[str, np.ndarray]: {camera_name: image} 형태
            예: {"realsense": img1, "innomaker": img2}
        """
        if cls._camera_manager is None:
            # 카메라 없이 테스트용 더미 이미지
            print("[RecordingContext] Warning: No camera manager, using dummy images")
            return {"dummy": np.zeros((480, 640, 3), dtype=np.uint8)}

        try:
            # 비동기 캡처 사용 (블로킹 없음!)
            if cls._async_capture is not None:
                images = cls._async_capture.get_latest_images()

                if not images:
                    # 아직 캡처된 이미지가 없음 (초기 상태)
                    print("[RecordingContext] No images in async buffer yet")
                    return None

                return images

            # Fallback: 동기 캡처 (블로킹 발생)
            else:
                images = cls._camera_manager.async_read_all()

                if not images:
                    print("[RecordingContext] No images captured from cameras")
                    return None

                return images

        except TimeoutError as e:
            print(f"[RecordingContext] Camera timeout: {e}")
            return None
        except Exception as e:
            print(f"[RecordingContext] Camera error: {e}")
            return None

    @classmethod
    def get_callback(cls):
        """
        LeRobotSkills에서 사용할 콜백 함수 반환

        Returns:
            콜백 함수 또는 None
        """
        if not cls._is_active:
            return None

        def callback(state: np.ndarray, action: np.ndarray):
            """Skills 제어 루프에서 호출되는 콜백"""
            cls.record_step(state, action)

        return callback

    @classmethod
    def get_stats(cls) -> dict:
        """레코딩 통계 반환"""
        elapsed = time.time() - cls._start_time if cls._start_time else 0
        stats = {
            "is_active": cls._is_active,
            "recorded_frames": cls._recorded_frames,
            "skipped_frames": cls._skipped_frames,
            "camera_errors": cls._camera_errors,
            "total_steps": cls._step_counter,
            "target_fps": cls._target_fps,
            "control_hz": cls._control_hz,
            "elapsed_time": elapsed,
            "effective_fps": cls._recorded_frames / elapsed if elapsed > 0 else 0,
        }

        if cls._camera_manager is not None and hasattr(cls._camera_manager, 'camera_names'):
            stats["cameras"] = cls._camera_manager.camera_names

        return stats
