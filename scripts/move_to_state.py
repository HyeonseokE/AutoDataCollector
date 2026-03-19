#!/usr/bin/env python3
"""
Move Robot to Recorded State (initial / free)

Moves robot to a previously recorded joint-space state using smooth interpolation.

Usage:
    python scripts/move_to_state.py 0 initial          # robot0 -> initial state
    python scripts/move_to_state.py 0 free              # robot0 -> free state
    python scripts/move_to_state.py 0 initial free      # robot0 -> initial -> free (sequential)
    python scripts/move_to_state.py 0 initial --duration 3.0
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lerobot_cap.hardware import FeetechController
from lerobot_cap.hardware.calibration import MotorCalibration


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_robot_id(config_path: str) -> str:
    match = re.search(r'robot(\d+)', str(config_path))
    return f"robot{match.group(1)}" if match else "robot0"


def load_state(robot_id: str, state_type: str) -> dict:
    """Load recorded state JSON file."""
    path = PROJECT_ROOT / "robot_configs" / f"{state_type}_state" / f"{robot_id}_{state_type}_state.json"
    if not path.exists():
        print(f"Error: State file not found: {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        return json.load(f)


def interpolate_trajectory(start: np.ndarray, end: np.ndarray, num_points: int = 50) -> np.ndarray:
    """Cosine-smoothed joint-space interpolation."""
    trajectory = np.zeros((num_points, len(start)))
    for i in range(num_points):
        alpha = i / (num_points - 1)
        smooth = (1 - np.cos(alpha * np.pi)) / 2
        trajectory[i] = start + smooth * (end - start)
    return trajectory


def execute_trajectory(robot, trajectory: np.ndarray, gripper_start: float,
                       gripper_end: float, duration: float, description: str):
    """Execute a joint trajectory with progress display."""
    print(f"\n  {description}")
    num_points = len(trajectory)
    start_time = time.time()

    for i, target in enumerate(trajectory):
        elapsed = time.time() - start_time
        expected = (i / (num_points - 1)) * duration if num_points > 1 else 0

        if elapsed < expected:
            time.sleep(expected - elapsed)

        # Interpolate gripper
        progress = i / (num_points - 1) if num_points > 1 else 1.0
        gripper = gripper_start + progress * (gripper_end - gripper_start)

        arm = np.clip(target, -99.0, 99.0)
        full = np.concatenate([arm, [gripper]])
        robot.write_positions(full, normalize=True)

        # Progress bar
        bar_len = 30
        filled = int(bar_len * (i + 1) / num_points)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {(i+1)/num_points*100:5.1f}%", end="", flush=True)

    print()


def move_to_state(robot, robot_id: str, state_type: str, duration: float):
    """Move robot to a recorded state."""
    state_data = load_state(robot_id, state_type)
    target_arm = np.array(state_data["initial_state_normalized"])
    target_gripper = float(state_data["gripper_normalized"])

    print(f"\n{'='*60}")
    print(f"  MOVE TO {state_type.upper()} STATE ({robot_id})")
    print(f"{'='*60}")
    print(f"  Target arm:     {target_arm}")
    print(f"  Target gripper: {target_gripper:.1f}")

    # Read current position
    current = robot.read_positions(normalize=True)
    current_arm = current[:5]
    current_gripper = float(current[5])

    print(f"  Current arm:    {current_arm}")
    print(f"  Current gripper:{current_gripper:.1f}")

    # Check if already there
    arm_dist = np.max(np.abs(current_arm - target_arm))
    gripper_dist = abs(current_gripper - target_gripper)

    if arm_dist < 5.0 and gripper_dist < 5.0:
        print(f"  Already at {state_type} state (arm diff: {arm_dist:.1f}, gripper diff: {gripper_dist:.1f})")
        return

    print(f"  Distance: arm={arm_dist:.1f}, gripper={gripper_dist:.1f}")

    # Generate and execute trajectory
    trajectory = interpolate_trajectory(current_arm, target_arm, num_points=50)
    execute_trajectory(
        robot, trajectory,
        current_gripper, target_gripper,
        duration,
        f"Moving to {state_type} state ({duration:.1f}s)..."
    )

    # Verify
    time.sleep(0.3)
    final = robot.read_positions(normalize=True)
    final_arm = final[:5]
    error = np.max(np.abs(final_arm - target_arm))
    print(f"  Done. Max error: {error:.1f} normalized units")


def main():
    parser = argparse.ArgumentParser(description="Move robot to recorded state")
    parser.add_argument("robot_id", type=int, help="Robot ID (e.g., 0, 2, 3)")
    parser.add_argument("states", type=str, nargs='+', choices=["initial", "free"],
                        help="State(s) to move to, in order (e.g., 'initial free')")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Movement duration per state (seconds, default: 2.0)")
    parser.add_argument("--config", type=str, default=None,
                        help="Robot config file (auto-detected if not specified)")

    args = parser.parse_args()

    # Resolve config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = PROJECT_ROOT / "robot_configs" / "robot" / f"so101_robot{args.robot_id}.yaml"

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    robot_id = extract_robot_id(str(config_path))

    # Load calibration
    calibration_file = config.get("calibration_file")
    calibration_by_id = {}

    if calibration_file and Path(calibration_file).exists():
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
        for name, data in calib_data.items():
            motor_id = data.get('motor_id', data.get('id'))
            if motor_id is None:
                continue
            calibration_by_id[motor_id] = MotorCalibration(
                motor_id=motor_id,
                model=data.get('model', 'sts3215'),
                drive_mode=data.get('drive_mode', 0),
                homing_offset=data.get('homing_offset', 0),
                range_min=data.get('range_min', 0),
                range_max=data.get('range_max', 4095),
            )
    else:
        print(f"Error: Calibration file not found: {calibration_file}")
        sys.exit(1)

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
        if not robot.connect():
            print("Error: Failed to connect to robot")
            sys.exit(1)

        robot.enable_torque()

        # Execute each state in order
        for state_type in args.states:
            move_to_state(robot, robot_id, state_type, args.duration)
            if state_type != args.states[-1]:
                time.sleep(0.5)  # Brief pause between states

        print(f"\n{'='*60}")
        print("  All movements complete!")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        try:
            robot.disable_torque()
            robot.disconnect()
        except:
            pass


if __name__ == "__main__":
    main()
