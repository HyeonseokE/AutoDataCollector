#!/usr/bin/env python3
"""
LeRobot Primitive Skills

Skill-based control for SO-101 LeRobot.
Ported from Franka skills.py for LeRobot hardware.

Usage:
    from skills.skills_lerobot import LeRobotSkills

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml")
    skills.connect()

    # Pick and Place
    skills.execute_pick_and_place(
        pick_position=[0.2, 0.0, 0.05],
        place_position=[0.2, 0.1, 0.05],
    )

    skills.disconnect()
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot_cap.hardware import FeetechController
from lerobot_cap.hardware.calibration import MotorCalibration
from lerobot_cap.kinematics import KinematicsEngine, load_calibration_limits
from lerobot_cap.planning import TrajectoryPlanner
from lerobot_cap.compensation import AdaptiveCompensator
from lerobot_cap.transforms import FrameTransformer
from lerobot_cap.workspace import BaseWorkspace

# Recording context for skill-level subgoal labeling
try:
    from record_dataset.context import RecordingContext
    HAS_RECORDING_CONTEXT = True
except ImportError:
    HAS_RECORDING_CONTEXT = False
    RecordingContext = None


def _compute_ee_xyzrpy(kinematics, joints_rad: np.ndarray) -> np.ndarray:
    """FK로 EE의 xyzrpy 계산"""
    pos, R = kinematics.forward_kinematics(joints_rad)
    # Rotation matrix to euler (ZYX convention)
    pitch = np.arcsin(-R[2, 0])
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.array([pos[0], pos[1], pos[2], roll, pitch, yaw], dtype=np.float32)


class LeRobotSkills:
    """
    Primitive skills for SO-101 LeRobot.

    Class Attributes:
        _last_instance: 마지막으로 생성된 인스턴스 (후처리에서 skill_sequence 접근용)

    Provides high-level manipulation primitives:
    - move_to_position: Move end-effector to target position
    - gripper_open / gripper_close: Gripper control
    - execute_pick: Pick object at position
    - execute_place: Place object at position
    - execute_pick_and_place: Full pick-and-place sequence

    Args:
        robot_config: Path to robot configuration YAML file
        frame: Coordinate frame for positions ('base_link' or 'world')
        gripper_open_pos: Gripper position for open state (normalized, -100 to +100)
        gripper_close_pos: Gripper position for closed state (normalized)
        movement_duration: Default duration for movements (seconds)
        use_compensation: Enable adaptive compensation
        use_deceleration: Enable end deceleration
        verbose: Print debug information
        pick_offset: Fixed offset from object top for pick/place (meters, default 3cm)
                   Default is [-0.04, 0, 0] (4cm). Can be customized per task.
    """

    _last_instance = None

    def __init__(
        self,
        robot_config: str = "robot_configs/robot/so101_robot3.yaml",
        frame: str = "world",
        gripper_open_pos: float = 100.0,
        gripper_close_pos: float = -100.0,
        movement_duration: float = 3.0,
        use_compensation: bool = True,
        use_deceleration: bool = True,
        verbose: bool = True,
        pick_offset: float = 0.015,  # Pick/place offset from object top (meters, 1.5cm)
        recording_callback: callable = None,  # LeRobot dataset recording callback
        camera=None,  # Shared camera instance for object detection (RealSenseD435)
    ):
        self.robot_config_path = Path(robot_config)
        self.frame = frame
        self.gripper_open_pos = gripper_open_pos
        self.gripper_close_pos = gripper_close_pos
        self.movement_duration = movement_duration
        self.use_compensation = use_compensation
        self.use_deceleration = use_deceleration
        self.verbose = verbose
        self.pick_offset = pick_offset  # Fixed offset from object top for pick/place
        self.skill_sequence = []  # 실행된 스킬 시퀀스 기록 (후처리 라벨링용)
        LeRobotSkills._last_instance = self  # 후처리에서 접근 가능하도록

        # LeRobot dataset recording callback
        # Auto-acquire from RecordingContext if not explicitly provided
        if recording_callback is not None:
            self.recording_callback = recording_callback
        else:
            self.recording_callback = self._get_recording_callback_from_context()

        # Will be initialized on connect()
        self.config = None
        self.robot: Optional[FeetechController] = None
        self.kinematics: Optional[KinematicsEngine] = None
        self.planner: Optional[TrajectoryPlanner] = None
        self.calibration_limits = None
        self.frame_transformer: Optional[FrameTransformer] = None
        self.compensator: Optional[AdaptiveCompensator] = None
        self.initial_state: Optional[np.ndarray] = None
        self.initial_state_gripper: float = gripper_open_pos
        self.free_state: Optional[np.ndarray] = None
        self.free_state_gripper: float = gripper_open_pos


        # Current state
        self.current_gripper_pos = gripper_open_pos
        self.is_connected = False

        # Base workspace for reachability validation
        self.workspace = None

        # Shared camera for object detection (optional)
        self.camera = camera

        # Deceleration parameters
        self.decel_start = 0.7  # 40% 지점부터 감속 시작
        self.decel_strength = 0.5  # 최종 속도 25%

        # Last movement error (for reporting)
        self.last_error: Optional[dict] = None

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _get_recording_callback_from_context(self):
        """
        RecordingContext에서 레코딩 콜백을 자동 획득.

        생성된 코드가 LeRobotSkills를 생성할 때 recording_callback 파라미터를
        전달하지 않아도, 전역 RecordingContext가 설정되어 있으면 자동으로
        콜백을 주입합니다.

        Returns:
            콜백 함수 또는 None
        """
        try:
            from record_dataset.context import RecordingContext
            if RecordingContext.is_active():
                callback = RecordingContext.get_callback()
                if callback is not None:
                    self._log("[LeRobotSkills] Recording callback acquired from context")
                return callback
        except ImportError:
            pass  # record_dataset not available
        except Exception as e:
            pass  # Context not set up
        return None

    def _extract_robot_id(self) -> str:
        """Extract robot ID from config path (e.g., 'robot3' from 'so101_robot3.yaml')."""
        stem = self.robot_config_path.stem  # e.g., "so101_robot3"
        if "_" in stem:
            return stem.split("_")[-1]  # e.g., "robot3"
        return stem

    def connect(self) -> bool:
        """
        Initialize and connect to robot hardware.

        Returns:
            True if connection successful
        """
        self._log(f"\n{'='*60}")
        self._log("LeRobotSkills: Initializing...")
        self._log(f"{'='*60}")

        # Load configuration
        if not self.robot_config_path.exists():
            print(f"Error: Config file not found: {self.robot_config_path}")
            return False

        with open(self.robot_config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize kinematics
        urdf_path = self.config.get("kinematics", {}).get("urdf_path", "assets/urdf/so101.urdf")
        ee_frame = self.config.get("kinematics", {}).get("end_effector_frame", "gripper_frame_link")

        ik_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]

        self._log(f"  Loading URDF: {urdf_path}")

        # Default kinematics engine (gripper_frame_link)
        self.kinematics = KinematicsEngine(
            str(urdf_path),
            end_effector_frame=ee_frame,
            joint_names=ik_joint_names,
        )

        # Load calibration limits
        calibration_file = self.config.get("calibration_file")
        if calibration_file and Path(calibration_file).exists():
            self.calibration_limits = load_calibration_limits(
                calibration_file,
                joint_names=ik_joint_names,
            )
            self._log(f"  Calibration limits loaded: {calibration_file}")

        # Initialize trajectory planners (default + TCP)
        self.planner = TrajectoryPlanner(
            self.kinematics,
            max_velocity=1.0,
            max_acceleration=2.0,
            interpolation_points=50,
            calibration_limits=self.calibration_limits,
        )

        # Initialize frame transformer
        frames_file = self.config.get("frames_file")
        if frames_file and Path(frames_file).exists():
            self.frame_transformer = FrameTransformer(frames_file)
            self._log(f"  Frame transformer loaded: {frames_file}")

        # Load calibration for motor control
        calibration_by_id = {}
        if calibration_file and Path(calibration_file).exists():
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
            for name, data in calib_data.items():
                motor_id = data.get('motor_id', data.get('id'))
                calibration_by_id[motor_id] = MotorCalibration(
                    motor_id=motor_id,
                    model=data.get('model', 'sts3215'),
                    drive_mode=data.get('drive_mode', 0),
                    homing_offset=data.get('homing_offset', 0),
                    range_min=data.get('range_min', 0),
                    range_max=data.get('range_max', 4095),
                )

        # Initialize motor controller
        motor_ids = [self.config["motors"][f"motor_{i}"]["id"] for i in range(1, 7)]

        self._log(f"  Connecting to robot...")
        self.robot = FeetechController(
            port=self.config.get("port"),
            baudrate=self.config.get("baudrate", 1000000),
            motor_ids=motor_ids,
            calibration=calibration_by_id,
        )

        connected = self.robot.connect()
        assert connected, "Failed to connect to robot hardware. Check: 1) Power on 2) USB connected 3) Correct port"

        # Enable torque
        self.robot.enable_torque()

        # Extract robot_id from config path (e.g., "so101_robot2.yaml" -> "robot2")
        import re
        robot_id_match = re.search(r'robot(\d+)', str(self.robot_config_path))
        robot_id = f"robot{robot_id_match.group(1)}" if robot_id_match else "robot3"

        # Load initial state (robot-specific)
        initial_state_path = Path(f"robot_configs/initial_state/{robot_id}_initial_state.json")
        if initial_state_path.exists():
            with open(initial_state_path, 'r') as f:
                initial_data = json.load(f)
            self.initial_state = np.array(initial_data["initial_state_normalized"])
            self.initial_state_gripper = initial_data.get("gripper_normalized", self.gripper_open_pos)
            self._log(f"  Initial state loaded: {initial_state_path}")
        else:
            self._log(f"  Warning: Initial state not found: {initial_state_path}")

        # Load free state (robot-specific)
        free_state_path = Path(f"robot_configs/free_state/{robot_id}_free_state.json")
        if free_state_path.exists():
            with open(free_state_path, 'r') as f:
                free_data = json.load(f)
            self.free_state = np.array(free_data["initial_state_normalized"])
            self.free_state_gripper = free_data.get("gripper_normalized", self.gripper_open_pos)
            self._log(f"  Free state loaded: {free_state_path}")
        else:
            self._log(f"  Warning: Free state not found: {free_state_path}")

        # Initialize compensator
        if self.use_compensation:
            compensation_file = self.config.get("compensation_file")
            if compensation_file and Path(compensation_file).exists():
                self.compensator = AdaptiveCompensator.from_config(
                    config_path=compensation_file,
                    target_z=0.1,  # Will be updated per movement
                )
                self._log(f"  Compensation config: {compensation_file}")

        # Initialize workspace with kinematics engine and frame transformer
        self.workspace = BaseWorkspace(self.kinematics, self.frame_transformer)
        self._log(f"  Workspace: reach [{self.workspace.min_reach:.3f}, {self.workspace.max_reach:.3f}]m")

        # Initialize current_gripper_pos with actual position (for correct action recording)
        self.current_gripper_pos = float(self.robot.read_positions(normalize=True)[5])

        # Register kinematics with RecordingContext for FK-based observation features
        if HAS_RECORDING_CONTEXT and RecordingContext.is_active():
            RecordingContext.set_kinematics(
                kinematics=self.kinematics,
                calibration_limits=self.calibration_limits,
                frame_transformer=self.frame_transformer,
            )

        self.is_connected = True
        self._log(f"\n{'='*60}")
        self._log("LeRobotSkills: Ready")
        self._log(f"{'='*60}\n")

        return True

    def disconnect(self):
        """Disconnect from robot hardware."""
        if self.robot:
            try:
                self.robot.disable_torque()
                self.robot.disconnect()
            except:
                pass
        self.is_connected = False
        self._log("LeRobotSkills: Disconnected")

    # ========== Utility Functions ==========

    def _normalized_to_radians(self, norm_positions: np.ndarray) -> np.ndarray:
        """Convert normalized (-100 to +100) to radians."""
        if self.calibration_limits is not None:
            return self.calibration_limits.normalized_to_radians(np.asarray(norm_positions))
        else:
            joint_lower = self.kinematics.joint_limits_lower
            joint_upper = self.kinematics.joint_limits_upper
            joint_center = (joint_lower + joint_upper) / 2
            joint_range = (joint_upper - joint_lower) / 2
            return joint_center + (np.asarray(norm_positions) / 100.0) * joint_range

    def _radians_to_normalized(self, rad_positions: np.ndarray) -> np.ndarray:
        """Convert radians to normalized (-100 to +100)."""
        if self.calibration_limits is not None:
            return self.calibration_limits.radians_to_normalized(np.asarray(rad_positions))
        else:
            joint_lower = self.kinematics.joint_limits_lower
            joint_upper = self.kinematics.joint_limits_upper
            joint_center = (joint_lower + joint_upper) / 2
            joint_range = (joint_upper - joint_lower) / 2
            return (np.asarray(rad_positions) - joint_center) / joint_range * 100.0


    def _transform_pos_world2robot(self, position: np.ndarray) -> np.ndarray:
        """Transform position from specified frame to base_link."""
        if self.frame != "base_link" and self.frame_transformer:
            if self.frame_transformer.has_frame(self.frame):
                return self.frame_transformer.transform_position(position, self.frame)
        return position

    def _compute_goal_xyzrpy(self, joints_rad: np.ndarray, kinematics=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        목표 joint로부터 world/robot xyzrpy 계산

        Returns:
            (world_xyzrpy, robot_xyzrpy) 각각 (6,) 배열
        """
        kin = kinematics if kinematics else self.kinematics
        robot_xyzrpy = _compute_ee_xyzrpy(kin, joints_rad)

        # robot(base_link) frame -> world frame 역변환 (position만)
        if self.frame != "base_link" and self.frame_transformer and self.frame_transformer.has_frame(self.frame):
            T = self.frame_transformer.frames[self.frame]["T_base_from_frame"]
            T_inv = np.linalg.inv(T)
            p_base = np.array([robot_xyzrpy[0], robot_xyzrpy[1], robot_xyzrpy[2], 1.0])
            world_pos = (T_inv @ p_base)[:3]
        else:
            world_pos = robot_xyzrpy[:3]

        world_xyzrpy = np.array([world_pos[0], world_pos[1], world_pos[2],
                                  robot_xyzrpy[3], robot_xyzrpy[4], robot_xyzrpy[5]], dtype=np.float32)
        return world_xyzrpy, robot_xyzrpy

    def _set_skill_recording(
        self,
        label: str,
        skill_type: str,
        goal_joint_5: np.ndarray,
        goal_gripper: float,
        kinematics=None,
        target_name: str = None,
        position: list = None,
        verification_question: str = None,
    ) -> None:
        """스킬 레코딩 정보 설정 (헬퍼)"""
        self.skill_sequence.append({
            "label": label,
            "type": skill_type,
            "target_name": target_name,
            "position": [round(p, 4) for p in position] if position else None,
        })

        if not HAS_RECORDING_CONTEXT or not RecordingContext.is_active():
            return

        start_state = self.robot.read_positions(normalize=True).copy()
        goal_arm_normalized = self._radians_to_normalized(goal_joint_5)
        goal_joint_6 = np.concatenate([goal_arm_normalized, [goal_gripper]])
        world_xyzrpy, robot_xyzrpy = self._compute_goal_xyzrpy(goal_joint_5, kinematics)

        RecordingContext.set_skill_info(
            label=label,
            skill_type=skill_type,
            goal_joint=goal_joint_6,
            goal_world_xyzrpy=world_xyzrpy,
            goal_robot_xyzrpy=robot_xyzrpy,
            goal_gripper=goal_gripper,
            start_state=start_state,
            verification_question=verification_question,
        )

    def _clear_skill_recording(self) -> None:
        """스킬 레코딩 정보 해제 (헬퍼)"""
        if HAS_RECORDING_CONTEXT and RecordingContext.is_active():
            RecordingContext.clear_skill_info()

    def _apply_end_deceleration(self, t_normalized: float) -> float:
        """Apply time warping to slow down at the end of trajectory.

        Uses ease-out curve: f(t) = 1 - (1-t)^n
        - decel_strength = 2: quadratic ease-out (smooth)
        - decel_strength = 3: cubic ease-out (noticeable)
        - decel_strength = 4: quartic ease-out (strong)
        """
        if t_normalized <= self.decel_start:
            return t_normalized
        else:
            # Normalize t to [0, 1] within the deceleration region
            t = (t_normalized - self.decel_start) / (1.0 - self.decel_start)

            # Ease-out: f(t) = 1 - (1-t)^n
            remaining_progress = 1.0 - (1.0 - t) ** self.decel_strength

            return self.decel_start + remaining_progress * (1.0 - self.decel_start)

    def _get_current_state(
        self,
        kinematics: Optional['KinematicsEngine'] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read current robot state.

        Args:
            kinematics: KinematicsEngine to use for FK calculation.
                       If None, uses default self.kinematics (gripper_frame_link).

        Returns:
            Tuple of (normalized_arm, radians_arm, ee_position)
        """
        pos_all = self.robot.read_positions(normalize=True)
        pos_arm_norm = pos_all[:5]
        # Note: self.current_gripper_pos는 목표값(action)으로 유지, 실제값으로 덮어쓰지 않음

        pos_arm_rad = self._normalized_to_radians(pos_arm_norm)

        # Use specified kinematics or default
        kin = kinematics if kinematics is not None else self.kinematics
        ee_pos = kin.get_ee_position(pos_arm_rad)

        return pos_arm_norm, pos_arm_rad, ee_pos

    def get_current_ee_position(self) -> np.ndarray:
        """Get current end-effector position in base_link frame."""
        _, _, ee_pos = self._get_current_state()
        return ee_pos

    def _calculate_error(
        self,
        target_position: np.ndarray,
        actual_position: np.ndarray,
        target_wrist_roll_rad: Optional[float] = None,
        actual_wrist_roll_rad: Optional[float] = None,
    ) -> dict:
        """
        Calculate position and orientation error.

        Args:
            target_position: Target [x, y, z] in meters
            actual_position: Actual [x, y, z] in meters
            target_wrist_roll_rad: Target wrist roll angle in radians (optional)
            actual_wrist_roll_rad: Actual wrist roll angle in radians (optional)

        Returns:
            Error dictionary with x, y, z errors (mm), orientation errors (deg), and total (mm)
        """
        # Position errors in mm
        pos_error = (actual_position - target_position) * 1000  # to mm
        total_error = np.linalg.norm(pos_error)

        error = {
            "x_mm": pos_error[0],
            "y_mm": pos_error[1],
            "z_mm": pos_error[2],
            "total_mm": total_error,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
        }

        # Wrist roll (yaw) error if available
        if target_wrist_roll_rad is not None and actual_wrist_roll_rad is not None:
            yaw_error_rad = actual_wrist_roll_rad - target_wrist_roll_rad
            # Normalize to [-pi, pi]
            while yaw_error_rad > np.pi:
                yaw_error_rad -= 2 * np.pi
            while yaw_error_rad < -np.pi:
                yaw_error_rad += 2 * np.pi
            error["yaw_deg"] = np.degrees(yaw_error_rad)

        return error

    def _print_error(self, error: dict, label: str = ""):
        """Print error in a formatted way."""
        if label:
            self._log(f"\n  [{label}] Error:")
        else:
            self._log(f"\n  Error:")
        self._log(f"    Position: x={error['x_mm']:+.2f}mm, y={error['y_mm']:+.2f}mm, z={error['z_mm']:+.2f}mm")
        self._log(f"    Orientation: r={error['roll_deg']:+.1f}°, p={error['pitch_deg']:+.1f}°, y={error['yaw_deg']:+.1f}°")
        self._log(f"    Total position error: {error['total_mm']:.2f}mm")


########################   Low-level Motion Execution   ########################

    def _execute_move_to_known_pose(
        self,
        start_normalized: np.ndarray,
        end_normalized: np.ndarray,
        duration: float,
        description: str = "",
        start_gripper: Optional[float] = None,
        end_gripper: Optional[float] = None,
    ) -> bool:
        """6축(arm 5축 + optional gripper) Joint-space 단순 보간 이동.

        목적: 목표 joint 값이 이미 알려진 경우의 단순 이동
        사용처: rotate_90degree(), move_to_initial_state(), move_to_free_state()
        특징: Compensation 없음, Hold phase 없음, cosine smoothing 보간

        Args:
            start_normalized: 시작 arm 관절값 (5축)
            end_normalized: 목표 arm 관절값 (5축)
            duration: 이동 시간 (초)
            description: 로그 설명
            start_gripper: 시작 gripper 값 (None이면 현재값 유지)
            end_gripper: 목표 gripper 값 (None이면 현재값 유지)
        """
        self._log(f"  {description}")

        # Gripper 설정: None이면 현재값 고정
        if start_gripper is None:
            start_gripper = self.current_gripper_pos
        if end_gripper is None:
            end_gripper = start_gripper  # 변화 없음

        num_points = 50
        start_time = time.time()

        for i in range(num_points):
            elapsed = time.time() - start_time
            expected_time = (i / (num_points - 1)) * duration

            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)

            # Cosine smoothing
            alpha = i / (num_points - 1)
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2

            # Arm + Gripper interpolation
            arm_normalized = np.clip(
                start_normalized + smooth_alpha * (end_normalized - start_normalized),
                -99.0, 99.0
            )
            gripper_pos = start_gripper + smooth_alpha * (end_gripper - start_gripper)

            full_normalized = np.concatenate([arm_normalized, [gripper_pos]])
            self.robot.write_positions(full_normalized, normalize=True)

            # LeRobot dataset recording callback
            if self.recording_callback is not None:
                try:
                    state_full = self.robot.read_positions(normalize=True)  # 6축 실제 값
                    self.recording_callback(state_full.copy(), full_normalized.copy())
                except Exception as _rec_e:
                    if not getattr(self, '_rec_err_logged', False):
                        print(f"\n[Recording] Callback error: {_rec_e}")
                        self._rec_err_logged = True

            if self.verbose:
                progress = (i + 1) / num_points
                filled = int(30 * progress)
                print(f"\r  [{'=' * filled}{'-' * (30 - filled)}] {progress*100:5.1f}%", end="", flush=True)

        if self.verbose:
            print()

        # Gripper 상태 업데이트
        self.current_gripper_pos = end_gripper
        time.sleep(0.2)  # Settle time
        return True

    def _execute_move_gripper_pose(self, position: float, duration: float = 1.5):
        """
        Gripper 단독 제어 (1-DoF). 
        IK 기반 Arm 움직임(5-DoF)과 독립적. Arm은 현재 위치 유지.

        목적: Gripper만 목표 위치로 이동 (Arm 5축은 고정)
        사용처: gripper_open(), gripper_close(), execute_pick_object(), execute_place_object()
        특징: Arm 고정, linear interpolation, recording callback 지원

        Note:
            - IK 기반 Arm 제어(_execute_trajectory)와 완전히 독립적
            - Arm 5축은 현재 상태 그대로 유지하고 Gripper만 움직임

        Args:
            position: 목표 gripper 위치 (normalized, -100 ~ +100)
            duration: 이동 시간 (초)
        """
        current_arm_norm, _, _ = self._get_current_state()
        start_gripper = self.current_gripper_pos

        # Control loop with recording (50Hz)
        num_steps = int(duration * 50)
        start_time = time.time()

        for i in range(num_steps):
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            # Gripper interpolation (linear)
            alpha = min(elapsed / duration, 1.0)
            current_gripper = start_gripper + alpha * (position - start_gripper)

            # Build full command (arm stays at current position)
            full_normalized = np.concatenate([current_arm_norm, [current_gripper]])

            # Send command
            self.robot.write_positions(full_normalized, normalize=True)

            # LeRobot dataset recording callback
            if self.recording_callback is not None:
                try:
                    state_full = self.robot.read_positions(normalize=True)  # 6축 실제 값
                    self.recording_callback(state_full.copy(), full_normalized.copy())
                except Exception as _rec_e:
                    if not getattr(self, '_rec_err_logged', False):
                        print(f"\n[Recording] Callback error: {_rec_e}")
                        self._rec_err_logged = True

            # Wait for next control step
            next_time = start_time + (i + 1) * (duration / num_steps)
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Final position
        full_normalized = np.concatenate([current_arm_norm, [position]])
        self.robot.write_positions(full_normalized, normalize=True)
        self.current_gripper_pos = position


    def _execute_trajectory(
        self,
        trajectory,
        target_position: np.ndarray,
        description: str = "",
        kinematics: Optional['KinematicsEngine'] = None,
    ) -> bool:
        """Cartesian-space trajectory 실행 (IK 계산된 경로 추종).

        목적: XYZ 목표 위치로 이동 (IK로 계산된 trajectory 사용)
        사용처: move_to_position() → execute_pick_object(), execute_place_object()
        특징: Compensation 적용, Hold phase 있음 (목표 도달 확인), deceleration 적용

        Args:
            trajectory: Planner가 계산한 trajectory
            target_position: 목표 위치 (base_link frame)
            description: 로그 설명
            kinematics: FK용 KinematicsEngine
        """
        self._log(f"  {description}")

        duration = trajectory.duration
        start_time = time.time()

        POSITION_TOLERANCE = 0.007  # 7mm (fail threshold = 7mm * 3 = 21mm)
        MAX_TOTAL_TIME = duration + 2.0
        SETTLE_TIME = 0.2

        target_reached = False
        reach_time = None

        while True:
            elapsed = time.time() - start_time

            # Read current state (use specified kinematics for TCP mode)
            actual_norm, actual_rad, current_ee = self._get_current_state(kinematics)
            position_error = np.linalg.norm(target_position - current_ee)

            # Determine command
            if elapsed < duration:
                # Trajectory phase
                if self.use_deceleration:
                    t_normalized = elapsed / duration
                    progress = self._apply_end_deceleration(t_normalized)
                    warped_time = progress * duration
                    arm_positions_rad = trajectory.get_state_at_time(warped_time)
                else:
                    arm_positions_rad = trajectory.get_state_at_time(elapsed)

                arm_normalized = self._radians_to_normalized(arm_positions_rad)
                phase = "Traj"
            else:
                # Hold phase
                arm_normalized = self._radians_to_normalized(trajectory.joint_positions[-1])
                phase = "Hold"

            # Apply compensation
            if self.use_compensation and self.compensator:
                arm_normalized = self.compensator.compensate(actual_norm, arm_normalized)

            # Send command
            arm_normalized = np.clip(arm_normalized, -99.0, 99.0)
            full_normalized = np.concatenate([arm_normalized, [self.current_gripper_pos]])
            self.robot.write_positions(full_normalized, normalize=True)

            # LeRobot dataset recording callback (Traj phase only, skip Hold phase)
            if self.recording_callback is not None and phase == "Traj":
                try:
                    state_full = self.robot.read_positions(normalize=True)  # 6축 실제 값
                    self.recording_callback(state_full.copy(), full_normalized.copy())
                except Exception as _rec_e:
                    if not getattr(self, '_rec_err_logged', False):
                        print(f"\n[Recording] Callback error: {_rec_e}")
                        self._rec_err_logged = True

            # Progress display
            if self.verbose:
                progress = min(elapsed / duration, 1.0)
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r  [{bar}] {phase} err:{position_error*1000:6.1f}mm", end="", flush=True)

            # Check target reached
            if position_error < POSITION_TOLERANCE:
                if reach_time is None:
                    reach_time = time.time()
                elif time.time() - reach_time > SETTLE_TIME:
                    target_reached = True
                    break
            else:
                reach_time = None

            # Timeout
            if elapsed > MAX_TOTAL_TIME:
                if self.verbose:
                    print(f"\n  Timeout after {MAX_TOTAL_TIME:.1f}s")
                break

            time.sleep(0.02)  # 50Hz

        if self.verbose:
            if target_reached:
                print(f"\r  [{'=' * 30}] Done (err: {position_error*1000:.1f}mm)    ")
            else:
                print(f"\r  [{'=' * 30}] Timeout (err: {position_error*1000:.1f}mm)")

        # Calculate and store final error (use specified kinematics for TCP mode)
        _, final_rad, final_ee = self._get_current_state(kinematics)
        self.last_error = self._calculate_error(
            target_position=target_position,
            actual_position=final_ee,
        )
        self._print_error(self.last_error, description)

        return target_reached or position_error < POSITION_TOLERANCE * 4  # 20mm 허용
    
    # ========== Robot Tool skills ==========

    def detect_objects(
        self,
        queries: List[str],
        timeout: float = 5.0,
        visualize: bool = True,
    ) -> Dict[str, Dict]:
        """
        실시간 객체 검출 스킬

        코드 실행 중 객체 위치를 실시간으로 검출합니다.
        RealSense 카메라를 사용하여 World frame 좌표를 반환합니다.

        Args:
            queries: 검출할 객체 이름 리스트 ["red part", "pink part"]
            timeout: 검출 타임아웃 (초)
            visualize: True면 검출 창 표시

        Returns:
            Dict[str, Dict]: 검출 결과
            {
                "red part": {"position": [x, y, z], ...},
                "pink part": {"position": [x, y, z], ...},
            }
            검출 실패한 객체는 None

        Example:
            # 코드 실행 중 객체 검출
            positions = skills.detect_objects(["red part", "pink part"])
            red_pos = positions["red part"]["position"]

            # Pick 후 재검출
            skills.execute_pick_object(red_pos)
            positions = skills.detect_objects(["pink part"])  # 업데이트된 위치
        """
        from pathlib import Path
        import sys

        PROJECT_ROOT = Path(__file__).parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            from run_detect import run_realtime_detection
        except ImportError as e:
            self._log(f"[detect_objects] Error: Could not import run_detect: {e}")
            return {q: None for q in queries}

        # robot_id 추출 (robot config path에서)
        robot_id = self._extract_robot_id_int()

        # 카메라 소스 결정: self.camera 또는 global shared_camera
        camera_to_use = self.camera
        if camera_to_use is None:
            # global shared_camera 확인 (pipeline에서 주입됨)
            import builtins
            camera_to_use = getattr(builtins, 'shared_camera', None)

        self._log(f"\n[detect_objects] Starting detection...")
        self._log(f"  Queries: {queries}")
        self._log(f"  Timeout: {timeout}s")
        self._log(f"  Robot ID: {robot_id}")
        self._log(f"  Using shared camera: {camera_to_use is not None}")

        try:
            # run_realtime_detection 호출
            # external_camera가 있으면 공유 모드, 없으면 내부에서 생성
            results = run_realtime_detection(
                queries=queries,
                timeout=timeout,
                unit="m",
                return_extended=True,
                visualize=visualize,
                robot_id=robot_id,
                external_camera=camera_to_use,
                skip_workspace_filter=True,  # 멀티로봇: workspace 필터링은 나중에
                early_exit=True,  # skill에서는 검출 즉시 종료
            )

            # 결과 로깅
            found_count = sum(1 for v in results.values() if v is not None)
            self._log(f"[detect_objects] Found {found_count}/{len(queries)} objects")
            for name, info in results.items():
                if info:
                    pos = info["position"]
                    self._log(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                else:
                    self._log(f"  {name}: NOT FOUND")

            return results

        except Exception as e:
            self._log(f"[detect_objects] Error: {e}")
            import traceback
            traceback.print_exc()
            return {q: None for q in queries}
        
    # ========== Primitive Skills ==========
    def move_to_initial_state(self, duration: Optional[float] = None, skill_description: Optional[str] = None, verification_question: Optional[str] = None) -> bool:
        """
        Move robot to recorded initial (home) state including gripper.

        Args:
            duration: Movement duration (uses default if None)

        Returns:
            True if movement successful
        """
        if self.initial_state is None:
            print("Error: Initial state not loaded")
            return False

        duration = duration or self.movement_duration

        # Set skill recording info
        goal_joint_rad = self._normalized_to_radians(self.initial_state)
        self._set_skill_recording(
            label=skill_description or "move to initial state",
            skill_type="move_initial",
            goal_joint_5=goal_joint_rad,
            goal_gripper=self.initial_state_gripper,
            verification_question=verification_question,
        )

        self._log(f"\nMoving to Initial State...")
        current_arm_norm, _, _ = self._get_current_state()

        try:
            arm_dist = np.max(np.abs(current_arm_norm - self.initial_state))
            gripper_dist = abs(self.current_gripper_pos - self.initial_state_gripper)
            if arm_dist < 5.0 and gripper_dist < 5.0:
                self._log(f"  Already at initial state (arm: {arm_dist:.1f}, gripper: {gripper_dist:.1f})")
            else:
                self._execute_move_to_known_pose(
                    current_arm_norm, self.initial_state,
                    duration, "Moving to Initial State",
                    start_gripper=self.current_gripper_pos,
                    end_gripper=self.initial_state_gripper,
                )
            return True
        finally:
            self._clear_skill_recording()

    def move_to_position(
        self,
        position: Union[List[float], np.ndarray],
        duration: Optional[float] = None,
        maintain_wrist_roll: bool = True,
        maintain_pitch: bool = False,
        target_pitch: Optional[float] = None,
        target_name: Optional[str] = None,
        skill_description: Optional[str] = None,
        verification_question: Optional[str] = None,
    ) -> bool:
        """
        Move end-effector to target position with orientation constraints.

        For 5-DOF robots like SO-101, full 6-DOF orientation control is not possible.
        This function provides two orientation constraints:
        1. maintain_wrist_roll: Maintain wrist_roll joint to prevent gripper yaw rotation
        2. maintain_pitch / target_pitch: Control gripper pitch angle during movement

        Args:
            position: Target [x, y, z] in meters (in self.frame coordinate)
            duration: Movement duration (uses default if None)
            maintain_wrist_roll: Maintain wrist_roll joint during movement (default: True)
            maintain_pitch: Maintain current gripper pitch during movement (default: False).
                           Ignored if target_pitch is specified.
            target_pitch: Specific pitch angle to achieve (radians). If specified,
                         overrides maintain_pitch. Use for restoring saved pitch at place.
            target_name: Name of the target object for subgoal labeling (optional).
                        예: "blue dish", "yellow dice"
                        Used for recording skill-level subgoal labels.

        Returns:
            True if movement successful
        """
        position = np.array(position)
        duration = duration or self.movement_duration

        # Get current state first
        current_arm_norm, current_joints, current_ee = self._get_current_state()

        active_planner = self.planner

        # Transform to robot_base_link frame
        target_position = self._transform_pos_world2robot(position)

        self._log(f"\nMoving to position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] ({self.frame})")
        if self.frame != "base_link":
            self._log(f"  -> base_link: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")

        # 1st filter: Geometric reachability check (fast O(1) check)
        kinematics = active_planner.kinematics
        if not kinematics.is_position_reachable(target_position):
            self._log(f"ERROR: Position [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}] is outside reachable workspace")
            self._log(f"  Reach limits: [{kinematics.min_reach:.3f}m, {kinematics.max_reach:.3f}m]")
            return False

        # Maintain wrist_roll at current value (joint index 4)
        fixed_joints_list = None
        if maintain_wrist_roll:
            fixed_joints_list = [4]  # wrist_roll is joint index 4
            # current_joints[4] already has the current wrist_roll value
            self._log(f"  Fixing wrist_roll at {np.degrees(current_joints[4]):.1f}°")

        # Handle pitch constraint: explicit target_pitch takes precedence over maintain_pitch
        ik_target_pitch = None
        if target_pitch is not None:
            # Explicit target pitch specified (e.g., restoring saved pitch at place)
            ik_target_pitch = target_pitch
            self._log(f"  Target pitch: {np.degrees(target_pitch):.1f}°")
        elif maintain_pitch:
            # Maintain current pitch (legacy behavior)
            current_pitch = active_planner.kinematics.get_gripper_pitch(current_joints)
            ik_target_pitch = current_pitch
            self._log(f"  Maintaining pitch at {np.degrees(current_pitch):.1f}°")

        # Plan trajectory with position-only IK but fixed wrist_roll and optional pitch constraint
        trajectory, ik_info = active_planner.plan_to_position_multi(
            target_position,
            current_joints,
            duration,
            num_random_samples=10,
            verbose=False,
            fixed_joints=fixed_joints_list,
            target_pitch=ik_target_pitch,
        )
        self._log(f"  Multi-IK: {ik_info['num_valid']}/{ik_info['num_solutions']} valid solutions")
        if ik_info.get("selected_pitch") is not None:
            self._log(f"  Selected pitch: {np.degrees(ik_info['selected_pitch']):.1f}°")

        if not trajectory.ik_converged:
            if ik_target_pitch is not None:
                # IK failed with pitch constraint - retry with relaxed pitch range (±10°)
                PITCH_TOLERANCE_DEG = 10.0
                pitch_tolerance_rad = np.radians(PITCH_TOLERANCE_DEG)
                original_pitch = ik_target_pitch

                self._log(f"  WARNING: IK failed with pitch={np.degrees(original_pitch):.1f}°. "
                         f"Retrying within ±{PITCH_TOLERANCE_DEG}° range...")

                # Try multiple pitch values within ±10° range
                # Order: 0, ±1, ±2, ... ±10 degrees from original
                pitch_offsets_deg = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8, -9, 9, -10, 10]
                best_trajectory = None
                best_ik_info = None
                best_pitch_offset = None

                for offset_deg in pitch_offsets_deg:
                    test_pitch = original_pitch + np.radians(offset_deg)

                    trajectory_retry, ik_info_retry = active_planner.plan_to_position_multi(
                        target_position,
                        current_joints,
                        duration,
                        num_random_samples=10,
                        verbose=False,
                        fixed_joints=fixed_joints_list,
                        target_pitch=test_pitch,
                    )

                    if trajectory_retry.ik_converged and ik_info_retry['num_valid'] > 0:
                        best_trajectory = trajectory_retry
                        best_ik_info = ik_info_retry
                        best_pitch_offset = offset_deg
                        break  # Found valid solution, use it

                if best_trajectory is not None:
                    # Found valid solution within ±5° range
                    trajectory = best_trajectory
                    ik_info = best_ik_info
                    self._log(f"  Retry Multi-IK: {ik_info['num_valid']}/{ik_info['num_solutions']} valid solutions")
                    if ik_info.get("selected_pitch") is not None:
                        actual_pitch = np.degrees(ik_info['selected_pitch'])
                        self._log(f"  Relaxed pitch: {np.degrees(original_pitch):.1f}° → {actual_pitch:.1f}° (offset: {best_pitch_offset:+d}°)")
                else:
                    # No valid solution within ±10° range - position unreachable, abort pipeline
                    error_msg = (f"IK failed within ±{PITCH_TOLERANCE_DEG}° pitch range. "
                                f"Position [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] unreachable "
                                f"with pitch={np.degrees(original_pitch):.1f}°")
                    self._log(f"ERROR: {error_msg}")
                    raise RuntimeError(f"[Pipeline Abort] {error_msg}")
            else:
                print(f"Warning: IK did not converge. Position may be unreachable.")

        # Update compensator target z
        if self.compensator:
            self.compensator = AdaptiveCompensator.from_config(
                config_path=self.config.get("compensation_file"),
                target_z=target_position[2],
            )

        # Set skill recording info (after trajectory planning)
        label = skill_description or (f"move {target_name}" if target_name else "move to position")
        goal_joint_rad = trajectory.joint_positions[-1]
        self._set_skill_recording(
            label=label,
            skill_type="move",
            goal_joint_5=goal_joint_rad,
            goal_gripper=self.current_gripper_pos,
            kinematics=active_planner.kinematics,
            target_name=target_name,
            position=position.tolist() if hasattr(position, 'tolist') else list(position),
            verification_question=verification_question,
        )

        # Execute trajectory with active kinematics for correct error measurement
        try:
            return self._execute_trajectory(
                trajectory=trajectory,
                target_position=target_position,
                description=f"Moving to [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]",
                kinematics=active_planner.kinematics,
            )
        finally:
            self._clear_skill_recording()

    def gripper_open(self, duration: float = 2.0, ratio: float = 1.0, skill_description: Optional[str] = None, verification_question: Optional[str] = None):
        """
        Open gripper with recording support.

        Args:
            duration: Movement duration in seconds (default: 1.5)
            ratio: Open ratio (0.0 = closed, 1.0 = fully open, default: 1.0)
        """
        GRIPPER_MAX_RATIO = 0.25
        clamped_ratio = min(ratio, GRIPPER_MAX_RATIO)
        target_pos = self.gripper_close_pos + (self.gripper_open_pos - self.gripper_close_pos) * clamped_ratio
        current_arm_norm, current_arm_rad, _ = self._get_current_state()

        self._set_skill_recording(
            label=skill_description or "open gripper",
            skill_type="gripper_open",
            goal_joint_5=current_arm_rad,
            goal_gripper=target_pos,
            verification_question=verification_question,
        )

        try:
            self._log(f"Gripper: Opening to {clamped_ratio*100:.0f}% (pos={target_pos:.0f})...")
            self._execute_move_gripper_pose(target_pos, duration=duration)
            self._log(f"Gripper: Open ({clamped_ratio*100:.0f}%)")
        finally:
            self._clear_skill_recording()

    def gripper_close(self, duration: float = 1.5, skill_description: Optional[str] = None, verification_question: Optional[str] = None):
        """
        Close gripper with recording support.

        Args:
            duration: Movement duration in seconds (default: 1.5)
        """
        GRIPPER_CLOSE_RATIO = 1.0
        target_pos = self.gripper_open_pos + (self.gripper_close_pos - self.gripper_open_pos) * GRIPPER_CLOSE_RATIO
        current_arm_norm, current_arm_rad, _ = self._get_current_state()

        self._set_skill_recording(
            label=skill_description or "close gripper",
            skill_type="gripper_close",
            goal_joint_5=current_arm_rad,
            goal_gripper=target_pos,
            verification_question=verification_question,
        )

        try:
            self._log(f"Gripper: Closing to 95% (pos={target_pos:.0f})...")
            self._execute_move_gripper_pose(target_pos, duration=duration)
            self._log("Gripper: Closed (95%)")
        finally:
            self._clear_skill_recording()
    
    def rotate_90degree(self, direction: int = 1, duration: float = 2.0, skill_description: Optional[str] = None, verification_question: Optional[str] = None) -> bool:
        """
        Rotate gripper (wrist_roll) by 90 degrees.

        Args:
            direction: 1 for clockwise, -1 for counter-clockwise
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        self._log(f"\n{'='*60}")
        self._log(f"ROTATE 90 DEGREE ({'CW' if direction > 0 else 'CCW'})")
        self._log(f"{'='*60}")

        # Get current state
        current_arm_norm, current_joints, current_ee = self._get_current_state()

        # Wrist roll is joint index 4 (0-indexed)
        wrist_roll_idx = 4
        target_wrist_roll_rad = current_joints[wrist_roll_idx] + direction * (np.pi / 2)

        # Check joint limits
        if self.calibration_limits:
            lower = self.calibration_limits.lower_limits_radians[wrist_roll_idx]
            upper = self.calibration_limits.upper_limits_radians[wrist_roll_idx]
            if target_wrist_roll_rad < lower or target_wrist_roll_rad > upper:
                self._log(f"  Warning: Target wrist roll {np.degrees(target_wrist_roll_rad):.1f}° exceeds limits")
                self._log(f"           Limits: [{np.degrees(lower):.1f}°, {np.degrees(upper):.1f}°]")
                target_wrist_roll_rad = np.clip(target_wrist_roll_rad, lower, upper)
                self._log(f"           Clamped to: {np.degrees(target_wrist_roll_rad):.1f}°")

        # Create target joints
        target_joints = current_joints.copy()
        target_joints[wrist_roll_idx] = target_wrist_roll_rad

        # Set skill recording info
        dir_str = "clockwise" if direction > 0 else "counter-clockwise"
        self._set_skill_recording(
            label=skill_description or f"rotate gripper {dir_str}",
            skill_type="rotate",
            goal_joint_5=target_joints,
            goal_gripper=self.current_gripper_pos,
            verification_question=verification_question,
        )

        # Convert to normalized
        start_normalized = current_arm_norm
        end_normalized = self._radians_to_normalized(target_joints)

        self._log(f"  Wrist roll: {np.degrees(current_joints[wrist_roll_idx]):.1f}° -> {np.degrees(target_wrist_roll_rad):.1f}°")

        try:
            # Execute joint trajectory
            success = self._execute_move_to_known_pose(
                start_normalized=start_normalized,
                end_normalized=end_normalized,
                duration=duration,
                description=f"Rotating wrist 90° {'CW' if direction > 0 else 'CCW'}",
            )

            # Calculate and store error
            _, final_rad, final_ee = self._get_current_state()
            self.last_error = self._calculate_error(
                target_position=current_ee,
                actual_position=final_ee,
                target_wrist_roll_rad=target_wrist_roll_rad,
                actual_wrist_roll_rad=final_rad[wrist_roll_idx],
            )
            self._print_error(self.last_error, "Rotate 90°")

            self._log("ROTATE 90 DEGREE: Complete")
            return success
        finally:
            self._clear_skill_recording()

    def move_to_free_state(self, duration: Optional[float] = None, skill_description: Optional[str] = None, verification_question: Optional[str] = None) -> bool:
        """
        Move robot to recorded free state for safe parking.

        This state is designed for when the robot is idle -
        it puts the arm in a relaxed, safe position.

        Args:
            duration: Movement duration (uses default if None)

        Returns:
            True if movement successful
        """
        if self.free_state is None:
            print("Error: Free state not loaded")
            return False

        duration = duration or self.movement_duration

        # Set skill recording info
        goal_joint_rad = self._normalized_to_radians(self.free_state)
        self._set_skill_recording(
            label=skill_description or "move to free state",
            skill_type="move_free",
            goal_joint_5=goal_joint_rad,
            goal_gripper=self.free_state_gripper,
            verification_question=verification_question,
        )

        self._log(f"\n{'='*60}")
        self._log("MOVE TO FREE STATE")
        self._log(f"{'='*60}")

        current_arm_norm, _, current_ee = self._get_current_state()

        try:
            arm_dist = np.max(np.abs(current_arm_norm - self.free_state))
            gripper_dist = abs(self.current_gripper_pos - self.free_state_gripper)
            if arm_dist < 5.0 and gripper_dist < 5.0:
                self._log(f"  Already at free state (arm: {arm_dist:.1f}, gripper: {gripper_dist:.1f})")
                return True

            self._log(f"  Target: {self.free_state}")
            self._log(f"  Gripper: {self.free_state_gripper:.1f}")

            success = self._execute_move_to_known_pose(
                current_arm_norm, self.free_state,
                duration, "Moving to Free State",
                start_gripper=self.current_gripper_pos,
                end_gripper=self.free_state_gripper,
            )

            _, final_rad, final_ee = self._get_current_state()
            free_state_rad = self._normalized_to_radians(self.free_state)
            expected_ee = self.kinematics.get_ee_position(free_state_rad)

            self.last_error = self._calculate_error(
                target_position=expected_ee,
                actual_position=final_ee,
            )
            self._print_error(self.last_error, "Free State")

            self._log("MOVE TO FREE STATE: Complete")
            return success
        finally:
            self._clear_skill_recording()

    def execute_pick_object(
        self,
        object_position: Union[List[float], np.ndarray],
        object_name: Optional[str] = None,
        skill_description: Optional[str] = None,
        verification_question: Optional[str] = None,
    ) -> bool:
        """
        Execute pick at object position (called from pick_approach position).

        TCP moves to object's 2/3 height point, then closes gripper.
        Assumes table surface is at z=0, so object_position[2] equals object height.

        Args:
            object_position: Object top surface position [x, y, z] in meters (from 3D detection)
            object_name: Name of the object being picked for subgoal labeling (optional).
                        예: "yellow dice", "red cup"

        Returns:
            True if successful
        """
        object_position = np.array(object_position)
        object_height = object_position[2]

        MIN_PICK_Z = 0.015  # Minimum pick height (1.5cm) — gripper ground margin
        pick_z = max(object_height - self.pick_offset, MIN_PICK_Z)
        pick_position = [object_position[0], object_position[1], pick_z]

        self._log(f"\n[Execute Pick Object]")
        self._log(f"  Object height: {object_height*100:.1f}cm")
        if object_height - self.pick_offset < MIN_PICK_Z:
            self._log(f"  [Pick Z-Fix] {(object_height - self.pick_offset)*100:.1f}cm < min {MIN_PICK_Z*100:.1f}cm, clamping to {MIN_PICK_Z*100:.1f}cm")
        self._log(f"  Pick point: {pick_z*100:.1f}cm ({self.pick_offset*100:.1f}cm from top)")

        # Pitch 보상: approach→pick 하강 시 pitch 변화로 인한 그리퍼 끝점 XY 드리프트 보정
        GRIPPER_TIP_LENGTH = 0.05  # gripper_frame_link → 실제 접촉점 거리 (m)
        _, approach_joints, _ = self._get_current_state()
        pick_joints, ik_ok = self.kinematics.inverse_kinematics_position_only(
            np.array(pick_position), initial_guess=approach_joints,
        )
        if ik_ok:
            tip_local = np.array([0, 0, GRIPPER_TIP_LENGTH])  # gripper Z-axis 방향
            approach_pos, approach_rot = self.kinematics.forward_kinematics(approach_joints)
            pick_pos_fk, pick_rot = self.kinematics.forward_kinematics(pick_joints)
            approach_tip = approach_pos + approach_rot @ tip_local
            pick_tip = pick_pos_fk + pick_rot @ tip_local
            tip_drift = pick_tip[:2] - approach_tip[:2]  # XY 밀림량

            if np.linalg.norm(tip_drift) > 0.002:  # 2mm 이상 밀림 시만 보상
                pick_position = [
                    pick_position[0] - tip_drift[0],
                    pick_position[1] - tip_drift[1],
                    pick_position[2],
                ]
                self._log(f"  [Pitch Compensation] tip_drift=({tip_drift[0]*1000:.1f}, {tip_drift[1]*1000:.1f})mm")

        # Move to pick position (skill recording handled inside)
        pick_label = f"pick {object_name}" if object_name else None
        # pick_z를 먼저 저장 (place에서 참조, pick 실패 시에도 crash 방지)
        self._pick_z = pick_z

        if not self.move_to_position(pick_position, target_name=pick_label, skill_description=skill_description):
            print("Error: Failed to reach pick position")
            return False

        # Close gripper (skill recording handled inside)
        grasp_desc = f"grasp {object_name}" if object_name else None
        self.gripper_close(skill_description=grasp_desc)

        # Store current pitch for place operation
        _, current_joints, _ = self._get_current_state()
        self._saved_pitch = self.kinematics.get_gripper_pitch(current_joints)
        self._log(f"  Saved pitch: {np.degrees(self._saved_pitch):.1f}°")

        self._log("[Execute Pick Object] Complete")
        return True

    def execute_place_object(
        self,
        place_position: Union[List[float], np.ndarray],
        is_table: bool = True,
        gripper_open_ratio: float = 1.0,
        target_name: Optional[str] = None,
        skill_description: Optional[str] = None,
        verification_question: Optional[str] = None,
    ) -> bool:
        """
        Execute place at target position (called from place_approach position).

        Moves to position where object bottom touches target surface, then opens gripper.

        Args:
            place_position: Target position [x, y, z] in meters
            is_table: If True, place on table (z=0). If False, place on object at place_position[2].
            gripper_open_ratio: How much to open gripper (0.0-1.0, default: 0.3 = 30%)
            target_name: Name of the target for subgoal labeling (optional).
                        예: "blue dish", "table"

        Returns:
            True if successful
        """
        place_position = np.array(place_position)
        target_surface_height = 0.0 if is_table else place_position[2]

        MIN_PLACE_Z = 0.015  # Minimum place height (1.5cm) — ground margin

        if is_table:
            # Placing on table: use target z (object's own height) as reference
            # place_position[2] = object's own height when on table
            # place_z = object height - pick_offset (same as how we'd pick it from table)
            place_z = max(place_position[2] - self.pick_offset, MIN_PLACE_Z)
        else:
            # Placing on another object: use saved pick_z offset from surface
            pick_z = getattr(self, '_pick_z', self.pick_offset)
            place_z = target_surface_height + pick_z
            if place_z < MIN_PLACE_Z:
                self._log(f"  [Place Z-Fix] {place_z*100:.1f}cm < min {MIN_PLACE_Z*100:.0f}cm, clamping to {MIN_PLACE_Z*100:.0f}cm")
                place_z = MIN_PLACE_Z

        final_position = [place_position[0], place_position[1], place_z]

        saved_pitch = getattr(self, '_saved_pitch', None)

        pick_z = getattr(self, '_pick_z', self.pick_offset)
        self._log(f"\n[Execute Place Object]")
        self._log(f"  Target surface: z={target_surface_height*100:.1f}cm")
        self._log(f"  Place point: z={place_z*100:.1f}cm (pick_z={pick_z*100:.1f}cm, min={MIN_PLACE_Z*100:.0f}cm)")
        if saved_pitch is not None:
            self._log(f"  Restoring pitch: {np.degrees(saved_pitch):.1f}°")

        # Move to place position (skill recording handled inside)
        place_label = f"place on {target_name}" if target_name else None
        if not self.move_to_position(final_position,
                                     target_pitch=saved_pitch,
                                     target_name=place_label,
                                     skill_description=skill_description):
            print("Error: Failed to reach place position")
            return False

        # Open gripper (skill recording handled inside)
        release_desc = f"release object on {target_name}" if target_name else None
        self.gripper_open(ratio=gripper_open_ratio, skill_description=release_desc)

        # Clear saved state
        self._pick_z = None
        self._saved_pitch = None

        self._log("[Execute Place Object] Complete")
        return True

    def execute_press(
        self,
        position: Union[List[float], np.ndarray],
        press_depth: float = 0.01,
        contact_height: float = 0.02,
        press_duration: float = 0.5,
        hold_time: float = 0.3,
        max_press_torque: int = 400,
        duration: Optional[float] = None,
        target_name: Optional[str] = None,
        skill_description: Optional[str] = None,
        verification_question: Optional[str] = None,
    ) -> bool:
        """
        Press a target (button, switch) with 2-phase descent and torque limiting.

        Call after moving to approach position with gripper closed.
        Phase 1: descend to contact_height at normal speed.
        Phase 2: press down press_depth with limited torque (slow).
        Hold, then retract to contact_height.

        Args:
            position: target [x, y, z] in current frame (meters)
            press_depth: extra depth below contact surface (meters, default 1cm)
            contact_height: estimated surface height (meters, default 2cm)
            press_duration: time for pressing phase (seconds, default 0.5)
            hold_time: time to hold pressed state (seconds, default 0.3)
            max_press_torque: torque limit during press (0-1000, default 400)
            duration: descent/retract movement time (seconds, None=default)
            target_name: target label for recording
            skill_description: skill label for recording
        """
        from skills.press import press
        return press(
            self,
            position=position,
            press_depth=press_depth,
            contact_height=contact_height,
            press_duration=press_duration,
            hold_time=hold_time,
            max_press_torque=max_press_torque,
            duration=duration,
            target_name=target_name,
            skill_description=skill_description,
        )

    def execute_push(
        self,
        start_position: Union[List[float], np.ndarray],
        end_position: Union[List[float], np.ndarray],
        push_height: float = 0.01,
        run_up_distance: float = 0.03,
        approach_height: float = 0.20,
        duration: Optional[float] = None,
        object_name: Optional[str] = None,
        skill_description: Optional[str] = None,
        verification_question: Optional[str] = None,
    ) -> bool:
        """
        Push an object in a straight line (Cartesian linear path).

        Call after closing gripper and moving to approach position above start.
        Internally: descends to pre-contact (run-up offset behind start),
        moves linearly through start to end, then retreats to approach_height.

        Args:
            start_position: contact point [x, y, z] (interaction point, e.g. object edge)
            end_position: push end [x, y, z] in current frame (meters)
            push_height: EE height during push (meters, default 1cm)
            run_up_distance: pre-contact offset behind start (meters, default 3cm)
            approach_height: retreat height after push (meters, default 20cm)
            duration: push movement time (seconds, None=auto based on distance)
            object_name: object label for recording
            skill_description: skill label for recording
        """
        from skills.push_object import push_object
        return push_object(
            self,
            start_position=start_position,
            end_position=end_position,
            push_height=push_height,
            run_up_distance=run_up_distance,
            approach_height=approach_height,
            duration=duration,
            object_name=object_name,
            skill_description=skill_description,
        )

    # ========== High-Level Skills: Each high-level skill is composed of primitive skills ==========
    # def execute_pick(
    #     self,
    #     position: Union[List[float], np.ndarray],
    #     approach_height: float = 0.05,
    #     duration: Optional[float] = None,
    # ) -> bool:
    #     """
    #     Execute pick operation at specified position.

    #     Sequence:
    #     1. Open gripper
    #     2. Move to approach position (above target)
    #     3. Move down to pick position
    #     4. Close gripper
    #     5. Move back to approach position

    #     Args:
    #         position: Pick position [x, y, z] in meters
    #         approach_height: Height above position for approach (meters)
    #         duration: Movement duration per segment

    #     Returns:
    #         True if successful
    #     """
    #     position = np.array(position)
    #     duration = duration or self.movement_duration

    #     self._log(f"\n{'='*60}")
    #     self._log(f"EXECUTE PICK at [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    #     self._log(f"{'='*60}")

    #     # Save current wrist_roll to maintain throughout pick operation
    #     _, current_joints, _ = self._get_current_state()
    #     target_wrist_roll = current_joints[4]
    #     self._log(f"  Target wrist_roll for pick: {np.degrees(target_wrist_roll):.1f}°")

    #     # 1. Open gripper
    #     self.gripper_open()

    #     # 2. Move to approach position (fix wrist_roll to keep gripper orientation)
    #     approach_pos = position.copy()
    #     approach_pos[2] += approach_height
    #     if not self.move_to_position(approach_pos, duration, maintain_wrist_roll=True, target_wrist_roll_rad=target_wrist_roll):
    #         print("Error: Failed to reach approach position")
    #         return False

    #     # 3. Move down to pick position
    #     if not self.move_to_position(position, duration * 0.5, maintain_wrist_roll=True, target_wrist_roll_rad=target_wrist_roll):
    #         print("Error: Failed to reach pick position")
    #         return False

    #     # 4. Close gripper
    #     self.gripper_close()

    #     # 5. Move back to approach position
    #     if not self.move_to_position(approach_pos, duration * 0.5, maintain_wrist_roll=True, target_wrist_roll_rad=target_wrist_roll):
    #         print("Error: Failed to lift after pick")
    #         return False

    #     self._log("PICK: Complete")
    #     return True

    # def execute_place(
    #     self,
    #     position: Union[List[float], np.ndarray],
    #     approach_height: float = 0.05,
    #     duration: Optional[float] = None,
    # ) -> bool:
    #     """
    #     Execute place operation at specified position.

    #     Sequence:
    #     1. Move to approach position (above target)
    #     2. Move down to place position
    #     3. Open gripper
    #     4. Move back to approach position

    #     Args:
    #         position: Place position [x, y, z] in meters
    #         approach_height: Height above position for approach (meters)
    #         duration: Movement duration per segment

    #     Returns:
    #         True if successful
    #     """
    #     position = np.array(position)
    #     duration = duration or self.movement_duration

    #     self._log(f"\n{'='*60}")
    #     self._log(f"EXECUTE PLACE at [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    #     self._log(f"{'='*60}")

    #     # Save current wrist_roll to maintain throughout place operation
    #     _, current_joints, _ = self._get_current_state()
    #     target_wrist_roll = current_joints[4]
    #     self._log(f"  Target wrist_roll for place: {np.degrees(target_wrist_roll):.1f}°")

    #     # 1. Move to approach position (fix wrist_roll to keep gripper orientation)
    #     approach_pos = position.copy()
    #     approach_pos[2] += approach_height
    #     if not self.move_to_position(approach_pos, duration, maintain_wrist_roll=True, target_wrist_roll_rad=target_wrist_roll):
    #         print("Error: Failed to reach approach position")
    #         return False

    #     # 2. Move down to place position
    #     if not self.move_to_position(position, duration * 0.5, maintain_wrist_roll=True, target_wrist_roll_rad=target_wrist_roll):
    #         print("Error: Failed to reach place position")
    #         return False

    #     # 3. Open gripper
    #     self.gripper_open()

    #     # 4. Move back to approach position
    #     if not self.move_to_position(approach_pos, duration * 0.5, maintain_wrist_roll=True, target_wrist_roll_rad=target_wrist_roll):
    #         print("Error: Failed to lift after place")
    #         return False

    #     self._log("PLACE: Complete")
    #     return True

    # def execute_pick_and_place(
    #     self,
    #     pick_position: Union[List[float], np.ndarray],
    #     place_position: Union[List[float], np.ndarray],
    #     approach_height: float = 0.05,
    #     duration: Optional[float] = None,
    #     return_to_initial: bool = True,
    # ) -> bool:
    #     """
    #     Execute full pick-and-place sequence.

    #     Sequence:
    #     0. Move to initial state
    #     1. Rotate gripper 90 degrees
    #     2. Execute pick (open gripper, approach, descend, close gripper, lift)
    #     3. Execute place (approach, descend, open gripper, lift)
    #     4. Return to initial state
    #     5. Move to free state (safe parking)

    #     Args:
    #         pick_position: Pick position [x, y, z] in meters
    #         place_position: Place position [x, y, z] in meters
    #         approach_height: Height above positions for approach
    #         duration: Movement duration per segment
    #         return_to_initial: Return to initial state after placing

    #     Returns:
    #         True if successful
    #     """
    #     pick_pos = np.array(pick_position)
    #     place_pos = np.array(place_position)
    #     duration = duration or self.movement_duration

    #     self._log(f"\n{'='*60}")
    #     self._log("EXECUTE PICK AND PLACE")
    #     self._log(f"  Pick:  [{pick_pos[0]:.3f}, {pick_pos[1]:.3f}, {pick_pos[2]:.3f}]")
    #     self._log(f"  Place: [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")
    #     self._log(f"{'='*60}")

    #     # 0. Move to initial state
    #     self._log("\n[Step 0] Moving to initial state...")
    #     if not self.move_to_initial_state(duration):
    #         print("Warning: Could not move to initial state, continuing...")

    #     # 1. Rotate gripper 90 degrees
    #     self._log("\n[Step 1] Rotating gripper 90 degrees...")
    #     self.rotate_90degree(direction=1, duration=2.0)

    #     # 2. Execute pick
    #     self._log("\n[Step 2] Executing pick...")
    #     if not self.execute_pick(pick_pos, approach_height, duration):
    #         print("Error: Pick operation failed")
    #         return False

    #     # 3. Execute place
    #     self._log("\n[Step 3] Executing place...")
    #     if not self.execute_place(place_pos, approach_height, duration):
    #         print("Error: Place operation failed")
    #         return False

    #     # 4. Return to initial state
    #     if return_to_initial:
    #         self._log("\n[Step 4] Returning to initial state...")
    #         self.move_to_initial_state(duration)

    #     # 5. Move to free state (safe parking)
    #     self._log("\n[Step 5] Moving to free state...")
    #     self.move_to_free_state(duration)

    #     self._log(f"\n{'='*60}")
    #     self._log("PICK AND PLACE: Complete")
    #     self._log(f"{'='*60}\n")

    #     return True

    def _extract_robot_id_int(self) -> int:
        """Extract robot ID as integer from config path."""
        import re
        config_name = self.robot_config_path.stem
        match = re.search(r'robot(\d+)', config_name)
        if match:
            return int(match.group(1))
        return 3  # default


# ========== Main (for testing) ==========

def main():
    """Test LeRobotSkills with a simple pick-and-place demo."""
    import argparse

    parser = argparse.ArgumentParser(description="LeRobot Skills Demo")
    parser.add_argument("--config", type=str, default="robot_configs/robot/so101_robot3.yaml",
                        help="Robot configuration file")
    parser.add_argument("--pick-x", type=float, default=0.15, help="Pick X position")
    parser.add_argument("--pick-y", type=float, default=0.05, help="Pick Y position")
    parser.add_argument("--pick-z", type=float, default=0.02, help="Pick Z position")
    parser.add_argument("--place-x", type=float, default=0.15, help="Place X position")
    parser.add_argument("--place-y", type=float, default=-0.05, help="Place Y position")
    parser.add_argument("--place-z", type=float, default=0.02, help="Place Z position")
    parser.add_argument("--approach-height", type=float, default=0.05,
                        help="Approach height above pick/place positions")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Movement duration per segment")
    parser.add_argument("--frame", type=str, default="world",
                        help="Coordinate frame (base_link or world)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")

    args = parser.parse_args()

    pick_position = [args.pick_x, args.pick_y, args.pick_z]
    place_position = [args.place_x, args.place_y, args.place_z]

    print(f"\n{'='*60}")
    print("LeRobot Skills Demo: Pick and Place")
    print(f"{'='*60}")
    print(f"  Config: {args.config}")
    print(f"  Frame:  {args.frame}")
    print(f"  Pick:   {pick_position}")
    print(f"  Place:  {place_position}")
    print(f"  Approach Height: {args.approach_height}m")
    print(f"  Duration: {args.duration}s per segment")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[DRY RUN] Would execute pick-and-place sequence")
        return

    # Initialize skills
    skills = LeRobotSkills(
        robot_config=args.config,
        frame=args.frame,
        movement_duration=args.duration,
    )

    try:
        # Connect
        if not skills.connect():
            print("Failed to connect to robot")
            return

        # Execute pick and place using primitive skills
        print("\n[Step 0] Moving to initial state...")
        skills.move_to_initial_state()

        print("\n[Step 1] Rotating gripper 90 degrees...")
        skills.rotate_90degree(direction=1)

        # Calculate approach positions
        pick_approach = [pick_position[0], pick_position[1], pick_position[2] + args.approach_height]
        place_approach = [place_position[0], place_position[1], place_position[2] + args.approach_height]

        # PICK sequence (with gripper offset)
        print("\n[Step 2] Executing PICK sequence...")
        print("  2-1. Opening gripper...")
        skills.gripper_open()

        print("  2-2. Moving to pick approach position...")
        skills.move_to_position(pick_approach)

        print("  2-3. Descending to pick position...")
        skills.move_to_position(pick_position)

        print("  2-4. Closing gripper...")
        skills.gripper_close()

        print("  2-5. Lifting object (maintain pitch)...")
        skills.move_to_position(pick_approach, maintain_pitch=True)

        # PLACE sequence (maintain_pitch while holding object)
        print("\n[Step 3] Executing PLACE sequence...")
        print("  3-1. Moving to place approach position (maintain pitch)...")
        skills.move_to_position(place_approach, maintain_pitch=True)

        print("  3-2. Descending to place position (maintain pitch)...")
        skills.move_to_position(place_position, maintain_pitch=True)

        print("  3-3. Opening gripper...")
        skills.gripper_open()

        print("  3-4. Retracting...")
        skills.move_to_position(place_approach)

        # Cleanup
        print("\n[Step 4] Returning to initial state...")
        skills.move_to_initial_state()

        print("\n[Step 5] Moving to free state...")
        skills.move_to_free_state()

        print("\nDemo completed successfully!")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        skills.disconnect()


if __name__ == "__main__":
    main()
