"""
MultiArmRecorder: External 50Hz unified recording for multi-arm setups.

Instead of each LeRobotSkills instance recording independently via callbacks,
this recorder runs an external 50Hz loop that:
  1. Reads both arms' joint positions → concat to 12-axis state
  2. Reads both arms' action targets → concat to 12-axis action
  3. Captures images from all cameras (shared RealSense + per-arm Innomaker)
  4. Records per-arm skill info (left_skill.*, right_skill.*)
  5. Writes unified frames to a single LeRobot dataset

This follows the ALOHA format: concat left/right state and action.

Usage:
    recorder = MultiArmRecorder(multi_arm_skills, dataset_recorder, camera_manager)
    recorder.start()
    # ... execute skills ...
    recorder.stop()
"""

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from .config import (
    CONTROL_HZ,
    DEFAULT_FPS,
    NUM_JOINTS,
    MULTI_ARM_NUM_JOINTS,
)


class MultiArmRecorder:
    """
    External 50Hz recording loop for bi-arm (ALOHA-style) datasets.

    Runs in a background thread, sampling both arms at CONTROL_HZ (50Hz)
    and writing frames at target_fps (30Hz) via frame-skip synchronization.

    Args:
        multi_arm: MultiArmSkills instance (provides left_arm/right_arm access).
        recorder: DatasetRecorder instance (initialized with multi-arm features).
        camera_manager: MultiCameraManager for image capture.
        target_fps: Recording FPS (default: 30).
        control_hz: Sampling frequency (default: 50).
    """

    def __init__(
        self,
        multi_arm,
        recorder,
        camera_manager=None,
        target_fps: int = DEFAULT_FPS,
        control_hz: int = CONTROL_HZ,
    ):
        self.multi_arm = multi_arm
        self.recorder = recorder
        self.camera_manager = camera_manager
        self.target_fps = target_fps
        self.control_hz = control_hz
        self.frame_skip_ratio = control_hz / target_fps

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Frame counting
        self._step_counter = 0
        self._last_record_step = -1
        self._recorded_frames = 0
        self._errors = 0

        # Per-arm skill info (updated externally via set_skill_info)
        self._left_skill_info: Dict[str, Any] = self._default_skill_info("standby")
        self._right_skill_info: Dict[str, Any] = self._default_skill_info("standby")

        # Per-arm last state/action (updated via callbacks from skill execution)
        # State is set by callback to avoid concurrent serial port access.
        self._left_state: Optional[np.ndarray] = None
        self._right_state: Optional[np.ndarray] = None
        self._left_action: Optional[np.ndarray] = None
        self._right_action: Optional[np.ndarray] = None

    @staticmethod
    def _default_skill_info(skill_type: str = "standby") -> Dict[str, Any]:
        """Default skill info for idle/standby arm."""
        return {
            "type": skill_type,
            "natural_language": skill_type,
            "verification_question": "",
            "progress": 0.0,
            "goal_joint": np.zeros(NUM_JOINTS, dtype=np.float32),
            "goal_xyzrpy": np.zeros(6, dtype=np.float32),
            "goal_gripper": np.float32(0.0),
        }

    def set_left_skill_info(self, info: Dict[str, Any]):
        """Update left arm's current skill recording info."""
        with self._lock:
            self._left_skill_info = info

    def set_right_skill_info(self, info: Dict[str, Any]):
        """Update right arm's current skill recording info."""
        with self._lock:
            self._right_skill_info = info

    def set_left_state(self, state: np.ndarray):
        """Update left arm's current state (from callback, avoids serial access)."""
        with self._lock:
            self._left_state = np.asarray(state, dtype=np.float32)

    def set_right_state(self, state: np.ndarray):
        """Update right arm's current state (from callback, avoids serial access)."""
        with self._lock:
            self._right_state = np.asarray(state, dtype=np.float32)

    def set_left_action(self, action: np.ndarray):
        """Update left arm's current action target."""
        with self._lock:
            self._left_action = np.asarray(action, dtype=np.float32)

    def set_right_action(self, action: np.ndarray):
        """Update right arm's current action target."""
        with self._lock:
            self._right_action = np.asarray(action, dtype=np.float32)

    # ─────────────────────────────────────────────
    # Recording loop
    # ─────────────────────────────────────────────

    def start(self):
        """Start the background recording thread."""
        if self._running:
            return
        self._running = True
        self._step_counter = 0
        self._last_record_step = -1
        self._recorded_frames = 0
        self._errors = 0
        self._thread = threading.Thread(
            target=self._recording_loop,
            name="multi_arm_recorder",
            daemon=True,
        )
        self._thread.start()
        print(f"[MultiArmRecorder] Started ({self.control_hz}Hz → {self.target_fps}fps)")

    def stop(self):
        """Stop the recording thread and wait for it to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print(f"[MultiArmRecorder] Stopped (frames={self._recorded_frames}, errors={self._errors})")

    def _recording_loop(self):
        """Main 50Hz sampling loop running in background thread."""
        period = 1.0 / self.control_hz
        next_time = time.monotonic()

        while self._running:
            loop_start = time.monotonic()

            # Check if we should record this step (FPS sync)
            target_frame = int(self._step_counter / self.frame_skip_ratio)
            should_record = target_frame > self._last_record_step

            if should_record and self.recorder.is_recording:
                try:
                    self._record_frame()
                    self._last_record_step = target_frame
                    self._recorded_frames += 1
                except Exception as e:
                    self._errors += 1
                    if self._errors <= 5:
                        print(f"[MultiArmRecorder] Frame error: {e}")

            self._step_counter += 1

            # Timing: sleep until next period
            next_time += period
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _record_frame(self):
        """Record a single unified frame from both arms + cameras."""
        left_arm = self.multi_arm.left_arm
        right_arm = self.multi_arm.right_arm

        # Use callback-provided state (avoids concurrent serial access),
        # fall back to direct read only if callback hasn't fired yet
        with self._lock:
            left_state = self._left_state if self._left_state is not None else self._read_state(left_arm)
            right_state = self._right_state if self._right_state is not None else self._read_state(right_arm)
            left_action = self._left_action if self._left_action is not None else left_state.copy()
            right_action = self._right_action if self._right_action is not None else right_state.copy()
            left_skill = dict(self._left_skill_info)
            right_skill = dict(self._right_skill_info)

        # Concat to 12-axis
        state_12 = np.concatenate([left_state, right_state])

        action_12 = np.concatenate([left_action, right_action])

        # Capture images from all cameras
        images = self._capture_images()

        # Build frame dict
        frame = {
            "observation.state": state_12.astype(np.float32),
            "action": action_12.astype(np.float32),
            "task": self.recorder._current_task,
        }

        # Add images
        if images:
            for cam_name, img in images.items():
                frame[f"observation.images.{cam_name}"] = img

        # Add per-arm skill features
        for prefix, skill, arm_state in [("left", left_skill, left_state), ("right", right_skill, right_state)]:
            frame[f"{prefix}_skill.type"] = skill.get("type", "standby")
            frame[f"{prefix}_skill.natural_language"] = skill.get("natural_language", "standby")
            frame[f"{prefix}_skill.verification_question"] = skill.get("verification_question", "")
            # Dynamic progress: 1.0 - (||goal - current|| / ||goal - start||)
            progress = skill.get("progress", 0.0)
            start = skill.get("start_state")
            goal = skill.get("goal_joint")
            if start is not None and goal is not None:
                start_to_goal = np.linalg.norm(goal - start)
                if start_to_goal > 1e-6:
                    current_to_goal = np.linalg.norm(goal - arm_state)
                    progress = float(np.clip(1.0 - current_to_goal / start_to_goal, 0.0, 1.0))
                else:
                    progress = 1.0
            frame[f"{prefix}_skill.progress"] = np.asarray([progress], dtype=np.float32)
            frame[f"{prefix}_skill.goal_position.joint"] = np.asarray(
                skill.get("goal_joint", np.zeros(NUM_JOINTS)), dtype=np.float32
            )
            frame[f"{prefix}_skill.goal_position.robot_xyzrpy"] = np.asarray(
                skill.get("goal_xyzrpy", np.zeros(6)), dtype=np.float32
            )
            frame[f"{prefix}_skill.goal_position.gripper"] = np.asarray(
                [skill.get("goal_gripper", 0.0)], dtype=np.float32
            )

        # Add per-arm observation extras
        self._add_observation_extras(frame, left_arm, right_arm, left_state, right_state,
                                     left_action, right_action)

        # Write to dataset
        self.recorder.add_frame(frame)

    def _read_state(self, arm) -> np.ndarray:
        """Read 6-axis joint state from a LeRobotSkills arm."""
        if arm.robot is not None:
            try:
                positions = arm.robot.read_positions()
                if positions is not None:
                    return np.asarray(positions, dtype=np.float32)
            except Exception:
                pass
        return np.zeros(NUM_JOINTS, dtype=np.float32)

    def _capture_images(self) -> Optional[Dict[str, np.ndarray]]:
        """Capture from MultiCameraManager."""
        if self.camera_manager is None:
            return None
        try:
            return self.camera_manager.async_read_all()
        except Exception:
            return None

    def _add_observation_extras(
        self,
        frame: dict,
        left_arm,
        right_arm,
        left_state: np.ndarray,
        right_state: np.ndarray,
        left_action: np.ndarray,
        right_action: np.ndarray,
    ):
        """Add per-arm EE pose, gripper binary, radian state/action."""
        from skills.skills_lerobot import _compute_ee_xyzrpy

        for prefix, arm, state, action in [
            ("left", left_arm, left_state, left_action),
            ("right", right_arm, right_state, right_action),
        ]:
            # EE pose via FK
            if arm.kinematics is not None and arm.calibration_limits is not None:
                try:
                    joints_rad = arm.calibration_limits.normalized_to_radians(state[:5])
                    ee_xyzrpy = _compute_ee_xyzrpy(arm.kinematics, joints_rad)
                    frame[f"observation.ee_pos.{prefix}_robot_xyzrpy"] = ee_xyzrpy
                except Exception:
                    frame[f"observation.ee_pos.{prefix}_robot_xyzrpy"] = np.zeros(6, dtype=np.float32)
            else:
                frame[f"observation.ee_pos.{prefix}_robot_xyzrpy"] = np.zeros(6, dtype=np.float32)

            # Gripper binary from actual state (state[5] = gripper joint, >0 = open)
            gripper_pos = float(state[5]) if len(state) > 5 else 0.0
            gripper_binary = 1.0 if gripper_pos > 0 else 0.0
            frame[f"{prefix}_observation.gripper_binary"] = np.asarray([gripper_binary], dtype=np.float32)

            # Radian state/action (5 arm joints)
            if arm.calibration_limits is not None:
                try:
                    rad_state = arm.calibration_limits.normalized_to_radians(state[:5])
                    rad_action = arm.calibration_limits.normalized_to_radians(action[:5])
                    # Pad to 6 (include gripper as-is)
                    frame[f"observation.radian.{prefix}_state"] = np.append(
                        rad_state, state[5]
                    ).astype(np.float32)
                    frame[f"observation.radian.{prefix}_action"] = np.append(
                        rad_action, action[5]
                    ).astype(np.float32)
                except Exception:
                    frame[f"observation.radian.{prefix}_state"] = np.zeros(NUM_JOINTS, dtype=np.float32)
                    frame[f"observation.radian.{prefix}_action"] = np.zeros(NUM_JOINTS, dtype=np.float32)
            else:
                frame[f"observation.radian.{prefix}_state"] = np.zeros(NUM_JOINTS, dtype=np.float32)
                frame[f"observation.radian.{prefix}_action"] = np.zeros(NUM_JOINTS, dtype=np.float32)

    # ─────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def recorded_frames(self) -> int:
        return self._recorded_frames

    def get_stats(self) -> Dict[str, int]:
        return {
            "recorded_frames": self._recorded_frames,
            "errors": self._errors,
            "total_steps": self._step_counter,
        }
