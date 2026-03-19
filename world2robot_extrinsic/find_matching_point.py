#!/usr/bin/env python3
"""
Interactive point collection for world-to-robot extrinsic calibration.

This script collects ONE matching point pair (world_point, robot_base_point)
per execution. Run multiple times to accumulate points.

Usage:
    python find_matching_point.py --robot 2 --x 0.1 --y 0.0 --z 0.05

Run 10+ times with different world coordinates to collect enough points.
To start fresh, delete the JSON file: rm robot{N}_matching_points.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot_cap.hardware.feetech import FeetechController
from lerobot_cap.hardware.calibration import load_calibration
from lerobot_cap.kinematics.engine import KinematicsEngine
from lerobot_cap.kinematics.calibration_limits import load_calibration_limits


def load_robot_config(robot_num: int) -> dict:
    """Load robot configuration from YAML file."""
    config_path = Path(__file__).parent.parent / f"robot_configs/robot/so101_robot{robot_num}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Robot config not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_current_tcp_position(
    controller: FeetechController,
    kinematics: KinematicsEngine,
    calibration_limits,
) -> np.ndarray:
    """Read current joint positions and compute TCP position via FK."""
    # Read normalized positions from motors
    normalized_positions = controller.read_positions(normalize=True)

    # Convert to radians (first 5 joints, excluding gripper)
    joint_radians = calibration_limits.normalized_to_radians(normalized_positions[:5])

    # Compute FK to get TCP position
    tcp_position = kinematics.get_ee_position(joint_radians)

    return tcp_position


def load_existing_points(output_path: Path) -> list:
    """Load existing points from JSON file."""
    if output_path.exists():
        with open(output_path, 'r') as f:
            data = json.load(f)
            return data.get("points", [])
    return []


def save_points(output_path: Path, robot_num: int, points: list):
    """Save points to JSON file."""
    output_data = {
        "robot_id": f"robot{robot_num}",
        "updated_at": datetime.now().isoformat(),
        "num_points": len(points),
        "points": points,
        "_description": {
            "world": "World 좌표계 기준 위치 [x, y, z] (미터)",
            "robot_base": "로봇 Base 좌표계 기준 TCP 위치 [x, y, z] (미터)",
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Collect ONE matching point pair for world-to-robot extrinsic calibration"
    )
    parser.add_argument(
        "--robot", type=int, required=True,
        help="Robot number (e.g., 0, 2, 3)"
    )
    parser.add_argument(
        "--x", type=float, required=True,
        help="World X coordinate (meters)"
    )
    parser.add_argument(
        "--y", type=float, required=True,
        help="World Y coordinate (meters)"
    )
    parser.add_argument(
        "--z", type=float, required=True,
        help="World Z coordinate (meters)"
    )
    args = parser.parse_args()

    # Output filename (in matching_points subfolder)
    matching_points_dir = Path(__file__).parent / "matching_points"
    matching_points_dir.mkdir(exist_ok=True)
    output_path = matching_points_dir / f"robot{args.robot}_matching_points.json"

    # Load existing points (accumulate)
    matching_points = load_existing_points(output_path)

    world_point = np.array([args.x, args.y, args.z])

    print("=" * 60)
    print(f"World-to-Robot 매칭점 수집: Robot {args.robot}")
    print("=" * 60)
    print()
    print(f"현재 수집된 점: {len(matching_points)}개")
    print(f"출력 파일: {output_path}")
    print()
    print(f"입력된 World 좌표: ({args.x:.4f}, {args.y:.4f}, {args.z:.4f})")
    print()

    # Load robot configuration
    print(f"로봇 {args.robot} 초기화 중...")
    config = load_robot_config(args.robot)

    # Initialize hardware
    calibration_path = Path(config['calibration_file'])
    calibration_by_name = load_calibration(calibration_path.parent, calibration_path.name)
    # Convert calibration keys from "motor_1" to integer 1
    calibration = {calib.motor_id: calib for calib in calibration_by_name.values()}
    motor_ids = [m['id'] for m in config['motors'].values()]

    controller = FeetechController(
        port=config['port'],
        baudrate=config['baudrate'],
        motor_ids=motor_ids,
        calibration=calibration,
    )

    if not controller.connect():
        print("모터 연결 실패!")
        return 1

    # Initialize kinematics
    kinematics = KinematicsEngine(
        urdf_path=config['kinematics']['urdf_path'],
        end_effector_frame=config['kinematics']['end_effector_frame'],
    )

    # Load calibration limits for normalized-to-radians conversion
    calibration_limits = load_calibration_limits(
        config['calibration_file'],
        joint_names=['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'],
    )

    # Disable torque for manual positioning
    print()
    print("토크 비활성화 - 로봇을 수동으로 움직일 수 있습니다")
    controller.disable_torque()

    print()
    print("=" * 60)
    print(f"World 좌표 ({args.x:.4f}, {args.y:.4f}, {args.z:.4f})에 해당하는")
    print("위치로 로봇 그리퍼를 이동시키고 Enter를 누르세요.")
    print("(취소하려면 'q' + Enter)")
    print("=" * 60)
    print()

    try:
        # Show real-time TCP position while waiting
        print("실시간 TCP 위치 표시 중...")
        print()

        import select

        while True:
            # Read current TCP position
            tcp_pos = get_current_tcp_position(
                controller, kinematics, calibration_limits
            )

            # Display in-place (using carriage return)
            sys.stdout.write(
                f"\r  현재 TCP (base): x={tcp_pos[0]:+.4f}, y={tcp_pos[1]:+.4f}, z={tcp_pos[2]:+.4f}   "
            )
            sys.stdout.flush()

            # Check for input (non-blocking)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                user_input = sys.stdin.readline().strip()
                if user_input.lower() == 'q':
                    print("\n\n취소됨. 저장하지 않고 종료합니다.")
                    controller.disconnect()
                    return 0
                break

            time.sleep(0.05)  # 20 Hz update rate

    except KeyboardInterrupt:
        print("\n\n중단됨")
        controller.disconnect()
        return 1

    print()
    print()

    # Record final TCP position
    tcp_position = get_current_tcp_position(
        controller, kinematics, calibration_limits
    )

    # Store matching pair
    point_data = {
        "world": world_point.tolist(),
        "robot_base": tcp_position.tolist(),
        "timestamp": datetime.now().isoformat(),
    }
    matching_points.append(point_data)

    # Save to file
    save_points(output_path, args.robot, matching_points)

    print("=" * 60)
    print("기록 완료!")
    print("=" * 60)
    print()
    print(f"  World:      ({args.x:+.4f}, {args.y:+.4f}, {args.z:+.4f})")
    print(f"  Robot Base: ({tcp_position[0]:+.4f}, {tcp_position[1]:+.4f}, {tcp_position[2]:+.4f})")
    print()
    print(f"총 {len(matching_points)}개 점 수집됨 -> {output_path}")
    print()

    if len(matching_points) >= 10:
        print("10개 이상 수집 완료! 변환행렬을 계산할 수 있습니다:")
        print(f"  ./calculate_world2robot_transform_matrix.sh {args.robot}")
    else:
        remaining = 10 - len(matching_points)
        print(f"권장: {remaining}개 더 수집하세요 (최소 10개 권장)")

    print()

    # Cleanup
    controller.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
