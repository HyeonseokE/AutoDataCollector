#!/usr/bin/env python3
"""
Show coordinate transformation from world to base_link.

Usage:
    python scripts/show_transform.py --x 0.25 --y 0.0 --z 0.15
    python scripts/show_transform.py --x 0.25 --y 0.0 --z 0.15 --robot 3
    python scripts/show_transform.py --x 0.25 --y 0.0 --z 0.15 --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot_cap.transforms import FrameTransformer


def main():
    parser = argparse.ArgumentParser(description="Show world to base_link coordinate transformation")
    parser.add_argument("--x", type=float, required=True, help="World X position (meters)")
    parser.add_argument("--y", type=float, required=True, help="World Y position (meters)")
    parser.add_argument("--z", type=float, required=True, help="World Z position (meters)")
    parser.add_argument("--robot", type=int, choices=[2, 3], default=None,
                        help="Robot number (2 or 3). If not specified, shows both.")
    parser.add_argument("--all", action="store_true", help="Show transformation for all robots")

    args = parser.parse_args()

    world_position = np.array([args.x, args.y, args.z])

    robots = []
    if args.robot:
        robots = [args.robot]
    else:
        robots = [2, 3]

    print("=" * 60)
    print("World → Base_link 좌표 변환")
    print("=" * 60)
    print(f"\nWorld 좌표 입력: x={args.x:.4f}, y={args.y:.4f}, z={args.z:.4f}")
    print()

    for robot_num in robots:
        frames_file = f"robot_configs/world2robot_matrices/robot{robot_num}_matrix.json"

        if not Path(frames_file).exists():
            print(f"[Robot {robot_num}] 설정 파일 없음: {frames_file}")
            continue

        tf = FrameTransformer(frames_file)

        if not tf.has_frame("world"):
            print(f"[Robot {robot_num}] 'world' 프레임 미정의")
            continue

        frame_info = tf.get_frame_info("world")
        base_position = tf.transform_position(world_position, "world")

        print(f"[Robot {robot_num}]")
        print(f"  로봇 위치 (world 기준): {frame_info['robot_position']}")
        print(f"  로봇 회전 (world 기준): {frame_info['robot_rpy']} deg")
        print(f"  ────────────────────────────────────")
        print(f"  World:     ({args.x:+.4f}, {args.y:+.4f}, {args.z:+.4f})")
        print(f"  Base_link: ({base_position[0]:+.4f}, {base_position[1]:+.4f}, {base_position[2]:+.4f})")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
