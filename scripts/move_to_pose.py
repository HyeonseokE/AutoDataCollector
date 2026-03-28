#!/usr/bin/env python3
"""
Move to EE Pose Script

Move the robot end-effector to a target position.

Usage:
    python scripts/move_to_pose.py --x 0.2 --y 0.0 --z 0.3
    python scripts/move_to_pose.py --config configs/robot/so101.yaml --x 0.15 --y 0.1 --z 0.25
    python scripts/move_to_pose.py --x 0.2 --y 0.0 --z 0.3 --via-initial-state

This is the core functionality that combines:
- Hardware control (from lerobot_ros2)
- IK computation (from LeRobot/Pinocchio)
- Trajectory planning
- Safe execution

New feature: --via-initial-state
    When enabled, robot first moves to a recorded "safe" initial state,
    then computes IK from that position to reach the target.
    This avoids IK failures from extreme starting positions.
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lerobot_cap.hardware import FeetechController
from lerobot_cap.kinematics import KinematicsEngine, load_calibration_limits
from lerobot_cap.planning import TrajectoryPlanner
from lerobot_cap.compensation import AdaptiveCompensator


def extract_robot_id(config_path: str) -> str:
    """Extract robot ID from config path (e.g., 'so101_robot2.yaml' -> 'robot2')."""
    match = re.search(r'robot(\d+)', str(config_path))
    return f"robot{match.group(1)}" if match else "robot3"


def load_config(config_path: str) -> dict:
    """Load robot configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_initial_state(robot_id: str, initial_state_path: str = None) -> dict:
    """Load recorded initial state from JSON file.

    Args:
        robot_id: Robot identifier (e.g., 'robot2', 'robot3')
        initial_state_path: Optional custom path to initial state file

    Returns:
        Initial state data dict or None if not found
    """
    if initial_state_path:
        path = Path(initial_state_path)
    else:
        # Default: configs/initial_state/{robot_id}_initial_state.json
        path = PROJECT_ROOT / "robot_configs" / "initial_state" / f"{robot_id}_initial_state.json"

    if not path.exists():
        return None

    with open(path, 'r') as f:
        return json.load(f)


def interpolate_joint_trajectory(
    start_normalized: np.ndarray,
    end_normalized: np.ndarray,
    num_points: int = 50,
) -> np.ndarray:
    """
    Generate linear interpolation trajectory in joint space.

    Args:
        start_normalized: Starting joint positions (normalized -100 to +100)
        end_normalized: Ending joint positions (normalized -100 to +100)
        num_points: Number of interpolation points

    Returns:
        Array of shape (num_points, num_joints) with interpolated positions
    """
    trajectory = np.zeros((num_points, len(start_normalized)))

    for i in range(num_points):
        alpha = i / (num_points - 1)  # 0 to 1
        # Smooth interpolation using cosine
        smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
        trajectory[i] = start_normalized + smooth_alpha * (end_normalized - start_normalized)

    return trajectory


def apply_end_deceleration(t_normalized: float, decel_start: float = 0.7, decel_strength: float = 2.0) -> float:
    """
    Apply time warping to slow down at the end of trajectory.

    Uses a cubic curve that ensures smooth velocity transition:
    - Velocity = 1 at decel_start (no sudden acceleration)
    - Velocity → 0 at the end (robot stops smoothly)

    Args:
        t_normalized: Normalized time (0 to 1)
        decel_start: When to start decelerating (0.7 = last 30%)
        decel_strength: Not used in cubic mode, kept for API compatibility

    Returns:
        Warped position progress (0 to 1), slower near the end
    """
    if t_normalized <= decel_start:
        return t_normalized
    else:
        # Map to 0-1 range within decel zone
        t = (t_normalized - decel_start) / (1.0 - decel_start)

        # Cubic curve: f(t) = -t³ + t² + t
        # Velocity: f'(t) = -3t² + 2t + 1
        #   f'(0) = 1  (smooth start, no jump)
        #   f'(1) = 0  (stops at end)
        remaining_progress = -t**3 + t**2 + t

        return decel_start + remaining_progress * (1.0 - decel_start)


def main():
    parser = argparse.ArgumentParser(description="Move robot to target EE position")
    parser.add_argument("--config", type=str, default="robot_configs/robot/so101_robot3.yaml",
                        help="Robot configuration file")
    parser.add_argument("--urdf", type=str, default=None,
                        help="URDF file path (overrides config)")
    parser.add_argument("--x", type=float, required=True, help="Target X position (meters)")
    parser.add_argument("--y", type=float, required=True, help="Target Y position (meters)")
    parser.add_argument("--z", type=float, required=True, help="Target Z position (meters)")
    parser.add_argument("--roll", type=float, default=None, help="Target roll (degrees)")
    parser.add_argument("--pitch", type=float, default=None, help="Target pitch (degrees)")
    parser.add_argument("--yaw", type=float, default=None, help="Target yaw (degrees)")
    parser.add_argument("--gripper-down", action="store_true",
                        help="Set orientation so gripper points downward (default forward)")
    parser.add_argument("--orientation-tolerance", type=float, default=0.3,
                        help="Orientation tolerance in radians (default 0.3 = ~17 deg)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Movement duration (seconds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only, don't execute")
    parser.add_argument("--via-initial-state", action="store_true",
                        help="Move to initial state first, then to target (safer IK)")
    parser.add_argument("--initial-state-file", type=str, default=None,
                        help="Path to initial state JSON file (auto-detected from robot config if not specified)")
    parser.add_argument("--initial-state-duration", type=float, default=2.0,
                        help="Duration for moving to initial state (seconds)")

    # End deceleration parameters
    parser.add_argument("--decel-start", type=float, default=0.7,
                        help="When to start decelerating (0.7 = last 30%%, default: 0.7)")
    parser.add_argument("--decel-strength", type=float, default=2.0,
                        help="Deceleration strength (2.0=moderate, 3.0=strong, default: 2.0)")
    parser.add_argument("--no-decel", action="store_true",
                        help="Disable end deceleration")

    # Multi-solution IK parameters
    parser.add_argument("--multi-ik", action="store_true",
                        help="Use multi-solution IK solver (tries multiple initial guesses)")
    parser.add_argument("--multi-ik-samples", type=int, default=10,
                        help="Number of random samples for multi-IK (default: 10)")
    parser.add_argument("--multi-ik-verbose", action="store_true",
                        help="Print multi-IK search progress")

    # Adaptive compensation parameters
    parser.add_argument("--no-compensation", action="store_true",
                        help="Disable adaptive direction compensation (enabled by default)")
    parser.add_argument("--compensation-factor", type=float, default=None,
                        help="Override adaptive compensation factor (auto-selects based on z if not set)")

    # Logging parameters
    parser.add_argument("--log-dir", type=str, default="logs/move_to_pose",
                        help="Directory to save log files (default: logs/move_to_pose)")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable logging to file")

    # Coordinate frame parameters
    parser.add_argument("--frame", type=str, default="base_link",
                        help="Coordinate frame for target position (default: base_link, or 'world')")

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config: {config_path}")
        config = load_config(str(config_path))
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = {
            "port": "/dev/ttyACM0",
            "baudrate": 1000000,
            "motors": {f"motor_{i}": {"id": i} for i in range(1, 7)},
        }

    # URDF path
    urdf_path = args.urdf
    if urdf_path is None:
        urdf_path = config.get("kinematics", {}).get("urdf_path", "assets/urdf/so101.urdf")

    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}")
        print("Please provide a valid URDF file.")
        sys.exit(1)

    # Target position (in specified frame)
    target_position_input = np.array([args.x, args.y, args.z])

    # Coordinate frame
    if args.frame != "base_link":
        print(f"\nWarning: frame '{args.frame}' is not supported. Using base_link.")
    target_position = target_position_input
    print(f"\nTarget EE position: x={args.x:.3f}, y={args.y:.3f}, z={args.z:.3f}")

    # Target orientation (optional)
    # By default, use position-only IK (robot finds natural orientation)
    # Orientation can be constrained with --gripper-down or --roll/--pitch/--yaw
    target_orientation = None
    use_orientation = False

    if args.gripper_down:
        # Gripper pointing down (-Z world direction)
        # EE Z-axis = [0, 0, -1] (down)
        # EE X-axis = [1, 0, 0] (forward)
        # EE Y-axis = [0, -1, 0] (right, for right-hand rule)
        target_orientation = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=float)
        use_orientation = True
        print("Orientation: gripper pointing DOWN (constrained)")
    elif args.roll is not None or args.pitch is not None or args.yaw is not None:
        # Custom orientation from RPY (in degrees)
        # Convention (like aircraft):
        # - Roll: rotate around gripper pointing axis (positive = clockwise when viewed from behind)
        # - Pitch: positive = gripper tilts UP, negative = gripper tilts DOWN
        # - Yaw: positive = gripper rotates LEFT, negative = gripper rotates RIGHT
        roll = np.radians(args.roll or 0)
        pitch = -np.radians(args.pitch or 0)  # Negate to match aircraft convention
        yaw = np.radians(args.yaw or 0)

        # Rotation matrices (world frame)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Base orientation: gripper pointing forward (from zero joints FK)
        # EE frame at zero: X=[0,0,-1], Y=[0,1,0], Z=[1,0,0]
        base_orientation = np.array([
            [0,  0,  1],
            [0,  1,  0],
            [-1, 0,  0]
        ], dtype=float)

        # Apply RPY: first rotate the base orientation, then EE aligns to it
        # ZYX order (yaw around Z, then pitch around Y, then roll around X)
        world_rotation = Rz @ Ry @ Rx
        target_orientation = world_rotation @ base_orientation
        use_orientation = True
        print(f"Orientation: roll={args.roll or 0}°, pitch={args.pitch or 0}°, yaw={args.yaw or 0}° (constrained)")
    else:
        # Default: position-only IK (no orientation constraint)
        # Robot will find a natural orientation to reach the target position
        print("Orientation: FREE (position-only IK, natural arm configuration)")

    # Initialize kinematics
    print(f"\nInitializing kinematics...")
    ee_frame = config.get("kinematics", {}).get("end_effector_frame", "gripper_frame_link")

    # Exclude gripper from IK - gripper should be controlled separately
    ik_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        # "gripper" is excluded - it's for grasping, not positioning
    ]
    kinematics = KinematicsEngine(str(urdf_path), end_effector_frame=ee_frame, joint_names=ik_joint_names)

    # Load calibration limits (actual robot physical limits)
    calibration_limits = None
    calibration_file = config.get("calibration_file")
    if calibration_file and Path(calibration_file).exists():
        try:
            calibration_limits = load_calibration_limits(
                calibration_file,
                joint_names=ik_joint_names,
            )
            print(f"Calibration limits loaded: {calibration_file}")
            print(f"  Using ACTUAL robot limits (from calibration)")
        except Exception as e:
            print(f"Warning: Could not load calibration limits: {e}")
            print(f"  Falling back to URDF theoretical limits")
    else:
        print(f"  Using URDF theoretical limits (no calibration file)")

    # Initialize planner
    planner = TrajectoryPlanner(
        kinematics,
        max_velocity=1.0,
        max_acceleration=2.0,
        interpolation_points=50,
        calibration_limits=calibration_limits,
    )

    if args.dry_run:
        # Dry run - just plan without hardware
        print("\n[DRY RUN] Planning trajectory without hardware...")

        # Use zero as current position
        current_joints = np.zeros(kinematics.num_joints)

        # Compute current EE position
        current_ee = kinematics.get_ee_position(current_joints)
        print(f"Current EE position (from zero joints): {current_ee}")

        # Plan trajectory
        if use_orientation:
            trajectory = planner.plan_to_pose(target_position, target_orientation, current_joints, args.duration, args.orientation_tolerance)
        else:
            trajectory = planner.plan_to_position(target_position, current_joints, args.duration)

        # # Check IK convergence and display warning
        # if not trajectory.ik_converged or (trajectory.expected_position_error and trajectory.expected_position_error > 0.01):
        #     error_mm = trajectory.expected_position_error * 1000 if trajectory.expected_position_error else 0
        #     print("\n" + "!" * 60)
        #     print("!!" + " " * 14 + "IK CONVERGENCE FAILED" + " " * 15 + "!!")
        #     print("!" * 60)
        #     print(f"!  Target position may be UNREACHABLE or at workspace limit")
        #     print(f"!  Expected position error: {error_mm:.1f} mm")
        #     if use_orientation:
        #         print(f"!  Tip: Try without orientation constraint (omit roll/pitch/yaw)")
        #     print("!" * 60)

        # # Check joint limits and display warning
        # if not trajectory.joints_within_limits and trajectory.joint_limit_violations:
        #     print("\n" + "#" * 60)
        #     print("##" + " " * 12 + "JOINT LIMITS EXCEEDED" + " " * 13 + "##")
        #     print("#" * 60)
        #     print("#  IK solution requires joints BEYOND physical limits!")
        #     print("#  Robot CANNOT reach the target position from current pose.")
        #     print("#")
        #     for idx, name, val, limit in trajectory.joint_limit_violations:
        #         direction = "below min" if val < limit else "above max"
        #         overflow = abs(val - limit)
        #         print(f"#  [Joint {idx+1}] {name}: {np.degrees(val):.1f}° ({direction} {np.degrees(limit):.1f}° by {np.degrees(overflow):.1f}°)")
        #     print("#")
        #     print("#  Suggestion: Move robot to a different starting position")
        #     print("#              or choose a different target.")
        #     print("#" * 60)

        print(f"\nTrajectory planned:")
        print(f"  Duration: {trajectory.duration:.2f}s")
        print(f"  Points: {trajectory.num_points}")

        # Show final IK result
        final_joints = trajectory.joint_positions[-1]
        final_pos, final_rot = kinematics.forward_kinematics(final_joints)
        error = np.linalg.norm(target_position - final_pos)

        # Extract gripper direction (Z-axis of EE frame)
        gripper_direction = final_rot[:, 2]

        print(f"\nFinal joint positions (radians): {final_joints}")
        print(f"Final EE position: {final_pos}")
        print(f"Final gripper direction: {gripper_direction}")
        print(f"  (X={gripper_direction[0]:.2f} forward, Y={gripper_direction[1]:.2f} left, Z={gripper_direction[2]:.2f} up)")
        print(f"Position error: {error * 1000:.2f} mm")

        return

    # Initialize hardware
    print("\nInitializing hardware...")

    # Load calibration
    import json
    from lerobot_cap.hardware.calibration import MotorCalibration

    calibration_file = config.get("calibration_file")
    calibration_by_id = {}

    if calibration_file and Path(calibration_file).exists():
        try:
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
            print(f"Calibration loaded: {len(calibration_by_id)} motors from {calibration_file}")
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}")
    else:
        print(f"Warning: Calibration file not found: {calibration_file}")

    # Motor IDs
    motor_ids = [config["motors"][f"motor_{i}"]["id"] for i in range(1, 7)]

    # Create controller
    robot = FeetechController(
        port=config.get("port"),
        baudrate=config.get("baudrate", 1000000),
        motor_ids=motor_ids,
        calibration=calibration_by_id,
    )

    try:
        # Connect
        if not robot.connect():
            print("Error: Failed to connect to robot")
            sys.exit(1)

        # Enable torque
        robot.enable_torque()

        # Read current position (all 6 motors)
        current_normalized_all = robot.read_positions(normalize=True)

        # Separate arm joints (1-5) and gripper (6)
        current_normalized_arm = current_normalized_all[:5]  # First 5 joints for IK
        current_gripper_normalized = current_normalized_all[5]  # Gripper position

        # Convert normalized (-100 to +100) to radians
        # Use CALIBRATION range (actual robot limits), not URDF range!
        if calibration_limits is not None:
            # Use calibration-based conversion (correct!)
            print("Using CALIBRATION range for normalized <-> radians conversion")
            current_joints = calibration_limits.normalized_to_radians(current_normalized_arm)
        else:
            # Fallback to URDF range (may have mismatch!)
            print("WARNING: Using URDF range for conversion (calibration not loaded)")
            joint_lower = kinematics.joint_limits_lower
            joint_upper = kinematics.joint_limits_upper
            joint_center = (joint_lower + joint_upper) / 2
            joint_range = (joint_upper - joint_lower) / 2
            current_joints = joint_center + (current_normalized_arm / 100.0) * joint_range

        print(f"Current arm positions (normalized): {current_normalized_arm}")
        print(f"Current arm positions (radians): {current_joints}")
        print(f"Current gripper position: {current_gripper_normalized:.1f}")

        current_ee = kinematics.get_ee_position(current_joints)
        print(f"Current EE position: {current_ee}")

        # Helper functions for trajectory execution
        import time as time_module

        def normalized_to_radians(norm_positions):
            """Convert normalized (-100 to +100) to radians."""
            if calibration_limits is not None:
                return calibration_limits.normalized_to_radians(np.asarray(norm_positions))
            else:
                # Fallback to URDF range
                _joint_lower = kinematics.joint_limits_lower
                _joint_upper = kinematics.joint_limits_upper
                _joint_center = (_joint_lower + _joint_upper) / 2
                _joint_range = (_joint_upper - _joint_lower) / 2
                return _joint_center + (np.asarray(norm_positions) / 100.0) * _joint_range

        def radians_to_normalized(rad_positions):
            """Convert radians to normalized (-100 to +100)."""
            if calibration_limits is not None:
                return calibration_limits.radians_to_normalized(np.asarray(rad_positions))
            else:
                # Fallback to URDF range
                _joint_lower = kinematics.joint_limits_lower
                _joint_upper = kinematics.joint_limits_upper
                _joint_center = (_joint_lower + _joint_upper) / 2
                _joint_range = (_joint_upper - _joint_lower) / 2
                return (np.asarray(rad_positions) - _joint_center) / _joint_range * 100.0

        def execute_joint_trajectory(trajectory_normalized, gripper_pos, duration, description=""):
            """Execute a joint-space trajectory."""
            print(f"\n{description}")
            num_points = len(trajectory_normalized)
            start_time = time_module.time()

            for i, target_norm in enumerate(trajectory_normalized):
                elapsed = time_module.time() - start_time
                expected_time = (i / (num_points - 1)) * duration if num_points > 1 else 0

                # Wait if ahead of schedule
                if elapsed < expected_time:
                    time_module.sleep(expected_time - elapsed)

                # Send command (with safety clamp)
                arm_normalized = np.clip(target_norm, -99.0, 99.0)
                full_normalized = np.concatenate([arm_normalized, [gripper_pos]])
                robot.write_positions(full_normalized, normalize=True)

                # Progress display
                progress = (i + 1) / num_points
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r  [{bar}] {progress*100:5.1f}%", end="", flush=True)

            print()  # Newline after progress bar

        # ================================================================
        # PHASE 0: Move to Initial State (if --via-initial-state enabled)
        # ================================================================
        ik_initial_guess = current_joints  # Default: use current position

        if args.via_initial_state:
            # Extract robot_id from config path for auto-detection
            robot_id = extract_robot_id(args.config)
            initial_state_data = load_initial_state(robot_id, args.initial_state_file)

            if initial_state_data is None:
                expected_path = args.initial_state_file or f"robot_configs/initial_state/{robot_id}_initial_state.json"
                print(f"\nError: Initial state file not found: {expected_path}")
                print(f"Run 'python scripts/record_initial_state.py --robot {robot_id[-1]}' first to record a safe position.")
                sys.exit(1)

            initial_state_normalized = np.array(initial_state_data["initial_state_normalized"])
            print(f"\n{'='*60}")
            print("PHASE 0: Moving to Initial State (safe IK starting point)")
            print(f"{'='*60}")
            print(f"  Initial State (normalized): {initial_state_normalized}")
            print(f"  Current position (normalized): {current_normalized_arm}")

            # Check if already at initial state
            distance_to_initial = np.max(np.abs(current_normalized_arm - initial_state_normalized))

            if distance_to_initial < 5.0:
                print(f"  Already near initial state (max diff: {distance_to_initial:.1f})")
            else:
                print(f"  Distance to initial state: {distance_to_initial:.1f} (normalized units)")

                # Generate joint-space trajectory: current -> initial state
                trajectory_to_initial = interpolate_joint_trajectory(
                    current_normalized_arm,
                    initial_state_normalized,
                    num_points=50,
                )

                # Execute trajectory to initial state
                execute_joint_trajectory(
                    trajectory_to_initial,
                    current_gripper_normalized,
                    args.initial_state_duration,
                    description="Executing: Current -> Initial State"
                )

                # Brief settle time
                time_module.sleep(0.3)

                # Update current position after move
                current_normalized_arm = robot.read_positions(normalize=True)[:5]
                current_joints = normalized_to_radians(current_normalized_arm)
                print(f"  Arrived at Initial State: {current_normalized_arm}")

            # Use Initial State as IK starting point
            ik_initial_guess = normalized_to_radians(initial_state_normalized)
            initial_ee = kinematics.get_ee_position(ik_initial_guess)
            print(f"  Initial State EE position: {initial_ee}")
            print(f"{'='*60}")

        # ================================================================
        # PHASE 1: Plan trajectory from Initial State to Target
        # ================================================================
        ik_mode = "Multi-IK" if args.multi_ik else ("Initial State" if args.via_initial_state else "current position")
        print(f"\nPlanning trajectory (IK mode: {ik_mode})...")

        ik_info = None
        if use_orientation:
            trajectory = planner.plan_to_pose(target_position, target_orientation, ik_initial_guess, args.duration, args.orientation_tolerance)
        elif args.multi_ik:
            # Multi-solution IK: try multiple initial guesses and select valid solution
            trajectory, ik_info = planner.plan_to_position_multi(
                target_position,
                ik_initial_guess,
                args.duration,
                num_random_samples=args.multi_ik_samples,
                verbose=args.multi_ik_verbose,
            )
            print(f"  Multi-IK: {ik_info['num_attempts']} attempts, {ik_info['num_solutions']} solutions, {ik_info['num_valid']} valid")
            if ik_info['num_valid'] > 0:
                print(f"  Selected: {ik_info.get('selected', 'N/A')}")
            elif ik_info['num_solutions'] > 0:
                print(f"  WARNING: No valid solutions found! Using best invalid solution.")
        else:
            trajectory = planner.plan_to_position(target_position, ik_initial_guess, args.duration)

        # # Check IK convergence and display warning
        # if not trajectory.ik_converged or (trajectory.expected_position_error and trajectory.expected_position_error > 0.01):
        #     error_mm = trajectory.expected_position_error * 1000 if trajectory.expected_position_error else 0
        #     print("\n" + "!" * 60)
        #     print("!!" + " " * 14 + "IK CONVERGENCE FAILED" + " " * 15 + "!!")
        #     print("!" * 60)
        #     print(f"!  Target position may be UNREACHABLE or at workspace limit")
        #     print(f"!  Expected position error: {error_mm:.1f} mm")
        #     if use_orientation:
        #         print(f"!  Tip: Try without orientation constraint (omit roll/pitch/yaw)")
        #     print("!" * 60 + "\n")

        # # Check joint limits and ABORT if exceeded
        # if not trajectory.joints_within_limits and trajectory.joint_limit_violations:
        #     print("\n" + "#" * 60)
        #     print("##" + " " * 12 + "JOINT LIMITS EXCEEDED" + " " * 13 + "##")
        #     print("#" * 60)
        #     print("#  IK solution requires joints BEYOND physical limits!")
        #     print("#  Robot CANNOT reach the target position from current pose.")
        #     print("#")
        #     for idx, name, val, limit in trajectory.joint_limit_violations:
        #         direction = "below min" if val < limit else "above max"
        #         overflow = abs(val - limit)
        #         print(f"#  [Joint {idx+1}] {name}: {np.degrees(val):.1f}° ({direction} {np.degrees(limit):.1f}° by {np.degrees(overflow):.1f}°)")
        #     print("#")
        #     print("#  EXECUTION ABORTED - Move robot to a different starting")
        #     print("#  position or choose a different target.")
        #     print("#" * 60 + "\n")
        #
        #     # Abort - cleanup will be done in finally block
        #     raise SystemExit(1)

        print(f"Trajectory: {trajectory.num_points} points, {trajectory.duration:.2f}s")

        # ================================================================
        # PHASE 2: Execute trajectory to target
        # ================================================================
        print(f"\nExecuting trajectory to target...")

        start_time = time_module.time()
        duration = trajectory.duration

        # Position-based completion settings
        POSITION_TOLERANCE = 0.005  # 5mm tolerance for success
        MAX_TOTAL_TIME = duration + 3.0  # Max time = trajectory duration + 3s hold
        SETTLE_TIME = 0.3  # Time to wait after reaching target

        # End deceleration parameters (from CLI args)
        DECEL_ENABLED = not args.no_decel
        DECEL_START = args.decel_start  # When to start decelerating (0.7 = last 30%)
        DECEL_STRENGTH = args.decel_strength  # How aggressively to decelerate

        # Adaptive compensation parameters
        COMPENSATION_ENABLED = not args.no_compensation
        compensator = None
        if COMPENSATION_ENABLED:
            # Try to load robot-specific compensation config
            compensation_file = config.get("compensation_file")
            if compensation_file and Path(compensation_file).exists():
                compensator = AdaptiveCompensator.from_config(
                    config_path=compensation_file,
                    target_z=args.z,
                    override_factor=args.compensation_factor,
                )
                print(f"  Compensation config: {compensation_file}")
                print(f"  Robot ID: {compensator.robot_id}")
            else:
                # Fallback to legacy (hardcoded values)
                compensator = AdaptiveCompensator(
                    target_z=args.z,
                    override_factor=args.compensation_factor,
                    gravity_lut_path="robot_configs/motor_calibration/gravity_lut.json",
                )
                print(f"  Compensation: using legacy defaults (no config file)")
            comp_info = compensator.get_info()
            print(f"  Adaptive compensation: ON (factor={comp_info['factor']:.2f}, z={args.z:.2f}m)")
            print(f"    shoulder_lift: {comp_info['shoulder_lift_comp']:.2f} units")
            print(f"    elbow_flex:    {comp_info['elbow_flex_comp']:.2f} units")
        else:
            print(f"  Adaptive compensation: OFF")

        if DECEL_ENABLED:
            print(f"  End deceleration: ON (start={DECEL_START:.0%}, strength={DECEL_STRENGTH:.1f})")
        else:
            print(f"  End deceleration: OFF")

        def get_current_state():
            """Read current joint positions and EE position."""
            pos_normalized = robot.read_positions(normalize=True)[:5]
            pos_rad = normalized_to_radians(pos_normalized)
            ee_pos = kinematics.get_ee_position(pos_rad)
            return pos_normalized, pos_rad, ee_pos

        success = True
        target_reached = False
        reach_time = None

        try:
            # Check for invalid IK solution (values outside normalized range)
            target_rad = trajectory.joint_positions[-1]
            target_norm = radians_to_normalized(target_rad)
            if np.any(np.abs(target_norm) > 100):
                print("\n" + "!" * 60)
                print("WARNING: IK solution outside valid range!")
                print("The target may be unreachable from the current position.")
                if not args.via_initial_state:
                    print("TIP: Try using --via-initial-state option for safer IK.")
                else:
                    print("Even with Initial State, target may be unreachable.")
                print(f"Target normalized: {target_norm}")
                print("!" * 60 + "\n")

            while True:
                elapsed = time_module.time() - start_time

                # Read current state
                actual_norm, actual_rad, current_ee = get_current_state()
                position_error = np.linalg.norm(target_position - current_ee)

                # Determine command
                if elapsed < duration:
                    # Phase 1: Trajectory execution with optional end deceleration
                    if DECEL_ENABLED:
                        # Apply time warping for slower approach at the end
                        t_normalized = elapsed / duration  # 0 to 1
                        progress = apply_end_deceleration(t_normalized, DECEL_START, DECEL_STRENGTH)
                        warped_time = progress * duration
                        arm_positions_rad = trajectory.get_state_at_time(warped_time)
                    else:
                        # No deceleration, use elapsed time directly
                        arm_positions_rad = trajectory.get_state_at_time(elapsed)
                    arm_normalized = radians_to_normalized(arm_positions_rad)
                    phase = "Traj"
                else:
                    # Phase 2: Position hold (maintain final trajectory position)
                    arm_normalized = radians_to_normalized(trajectory.joint_positions[-1])
                    phase = "Hold"

                # Apply adaptive compensation (feedforward correction for backlash/gravity)
                if COMPENSATION_ENABLED and compensator is not None:
                    arm_normalized = compensator.compensate(actual_norm, arm_normalized)

                # Safety clamp and send command
                arm_normalized = np.clip(arm_normalized, -99.0, 99.0)
                full_normalized = np.concatenate([arm_normalized, [current_gripper_normalized]])
                robot.write_positions(full_normalized, normalize=True)

                # Progress display
                if elapsed < duration:
                    progress = elapsed / duration
                else:
                    progress = 1.0

                bar_len = 30
                filled = int(bar_len * min(progress, 1.0))
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r[{bar}] {phase} err:{position_error*1000:6.1f}mm", end="", flush=True)

                # Check if target reached
                if position_error < POSITION_TOLERANCE:
                    if reach_time is None:
                        reach_time = time_module.time()
                    elif time_module.time() - reach_time > SETTLE_TIME:
                        target_reached = True
                        break
                else:
                    reach_time = None  # Reset if moved away

                # Timeout check
                if elapsed > MAX_TOTAL_TIME:
                    print(f"\n\nTimeout after {MAX_TOTAL_TIME:.1f}s")
                    break

                time_module.sleep(0.02)  # 50Hz control rate

            # Final status
            if target_reached:
                print(f"\r[{'=' * 30}] Target reached!        ")
            else:
                print(f"\r[{'=' * 30}] Timeout (target not reached)")

        except Exception as e:
            print(f"\nExecution error: {e}")
            success = False

        if success:
            time_module.sleep(0.3)  # Brief settle time

            # Verify final position
            final_normalized_all = robot.read_positions(normalize=True)
            final_normalized_arm = final_normalized_all[:5]
            final_joints = normalized_to_radians(final_normalized_arm)
            final_ee = kinematics.get_ee_position(final_joints)
            cartesian_error = np.linalg.norm(target_position - final_ee)

            # Calculate joint-space error (IK target vs actual)
            joint_error_rad = target_rad - final_joints
            joint_error_deg = np.degrees(joint_error_rad)
            joint_error_norm = target_norm - final_normalized_arm
            joint_error_rms = np.sqrt(np.mean(joint_error_rad ** 2))  # RMS in radians
            joint_error_max = np.max(np.abs(joint_error_deg))  # Max error in degrees

            # Expected EE from IK (what IK thought the EE would be)
            expected_ee = kinematics.get_ee_position(target_rad)
            ik_model_error = np.linalg.norm(target_position - expected_ee)

            print(f"\n{'='*60}")
            print("FINAL RESULT")
            print(f"{'='*60}")

            # Cartesian error
            print(f"\n[Cartesian Error]")
            print(f"  Target EE:     [{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}] m")
            print(f"  Actual EE:     [{final_ee[0]:.4f}, {final_ee[1]:.4f}, {final_ee[2]:.4f}] m")
            print(f"  EE Error:      {cartesian_error * 1000:.2f} mm")

            # Joint error
            print(f"\n[Joint Error] (IK target - Actual)")
            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
            for i, name in enumerate(joint_names):
                print(f"  {name:15s}: {joint_error_deg[i]:+6.2f}° ({joint_error_norm[i]:+5.1f} norm)")
            print(f"  RMS Error:     {np.degrees(joint_error_rms):.2f}°")
            print(f"  Max Error:     {joint_error_max:.2f}°")

            # IK model accuracy
            print(f"\n[IK Model Check]")
            print(f"  Expected EE (from IK): [{expected_ee[0]:.4f}, {expected_ee[1]:.4f}, {expected_ee[2]:.4f}] m")
            print(f"  IK Residual Error:     {ik_model_error * 1000:.2f} mm")

            print(f"\n{'='*60}")
            error = cartesian_error  # For backward compatibility

            if target_reached:
                result_status = "SUCCESS"
                print("SUCCESS: Target position reached!")
            elif error < POSITION_TOLERANCE * 2:
                result_status = "CLOSE"
                print("CLOSE: Nearly reached target position.")
            else:
                result_status = "INCOMPLETE"
                print("INCOMPLETE: Target not reached within tolerance.")

            # Save log file
            if not args.no_log:
                log_dir = Path(args.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Detailed JSON log
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "target_position": {
                        "x": float(target_position[0]),
                        "y": float(target_position[1]),
                        "z": float(target_position[2])
                    },
                    "actual_position": {
                        "x": float(final_ee[0]),
                        "y": float(final_ee[1]),
                        "z": float(final_ee[2])
                    },
                    "cartesian_error_mm": float(cartesian_error * 1000),
                    "joint_error": {
                        joint_names[i]: {
                            "error_deg": float(joint_error_deg[i]),
                            "error_norm": float(joint_error_norm[i]),
                            "target_rad": float(target_rad[i]),
                            "actual_rad": float(final_joints[i])
                        } for i in range(5)
                    },
                    "joint_error_rms_deg": float(np.degrees(joint_error_rms)),
                    "joint_error_max_deg": float(joint_error_max),
                    "ik_residual_mm": float(ik_model_error * 1000),
                    "expected_ee_from_ik": {
                        "x": float(expected_ee[0]),
                        "y": float(expected_ee[1]),
                        "z": float(expected_ee[2])
                    },
                    "result": result_status,
                    "settings": {
                        "duration": args.duration,
                        "decel_enabled": not args.no_decel,
                        "compensation_enabled": not args.no_compensation,
                        "via_initial_state": args.via_initial_state
                    }
                }

                json_path = log_dir / f"log_{timestamp}.json"
                with open(json_path, 'w') as f:
                    json.dump(log_data, f, indent=2)

                # Append to summary CSV
                csv_path = log_dir / "summary.csv"
                csv_exists = csv_path.exists()

                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not csv_exists:
                        # Write header
                        writer.writerow([
                            'timestamp', 'target_x', 'target_y', 'target_z',
                            'actual_x', 'actual_y', 'actual_z',
                            'cartesian_error_mm', 'joint_rms_deg', 'joint_max_deg',
                            'ik_residual_mm', 'result',
                            'shoulder_pan_err', 'shoulder_lift_err', 'elbow_flex_err',
                            'wrist_flex_err', 'wrist_roll_err'
                        ])
                    writer.writerow([
                        timestamp,
                        f"{target_position[0]:.4f}", f"{target_position[1]:.4f}", f"{target_position[2]:.4f}",
                        f"{final_ee[0]:.4f}", f"{final_ee[1]:.4f}", f"{final_ee[2]:.4f}",
                        f"{cartesian_error * 1000:.2f}",
                        f"{np.degrees(joint_error_rms):.2f}",
                        f"{joint_error_max:.2f}",
                        f"{ik_model_error * 1000:.2f}",
                        result_status,
                        f"{joint_error_deg[0]:.2f}", f"{joint_error_deg[1]:.2f}",
                        f"{joint_error_deg[2]:.2f}", f"{joint_error_deg[3]:.2f}",
                        f"{joint_error_deg[4]:.2f}"
                    ])

                print(f"\nLog saved: {json_path}")
                print(f"Summary:   {csv_path}")
        else:
            print("Movement failed or was stopped")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except SystemExit as e:
        # Handle abort from joint limits check
        pass  # Cleanup will happen in finally

    finally:
        # Clean up (check if robot is still connected)
        try:
            robot.disable_torque()
            robot.disconnect()
        except:
            pass  # Already disconnected or not initialized


if __name__ == "__main__":
    main()
