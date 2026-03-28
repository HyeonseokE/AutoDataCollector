#!/usr/bin/env python3
"""
Detect & Move Test Script
물체를 감지하고 해당 위치로 로봇을 이동하는 간소화 테스트

Usage:
    python scripts/test_detect_and_move.py --robot 2 --query "chocolate pie"
    python scripts/test_detect_and_move.py --robot 2 --query "red plate" --approach-height 0.15
    python scripts/test_detect_and_move.py --robot 2 --query "blue cup" --dry-run
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from object_detection.camera.realsense import RealSenseD435
from object_detection.detection.grounding_detector import GroundingDINODetector
from pix2robot_calibrator import Pix2RobotCalibrator
from skills.skills_lerobot import LeRobotSkills


def main():
    parser = argparse.ArgumentParser(description="Detect object and move robot to it")
    parser.add_argument("--robot", type=int, default=2, choices=[2, 3], help="Robot number")
    parser.add_argument("--query", type=str, required=True, help="Object to detect (e.g., 'red cup')")
    parser.add_argument("--approach-height", type=float, default=0.20, help="Approach height above object (meters)")
    parser.add_argument("--dry-run", action="store_true", help="Detect only, don't move robot")
    parser.add_argument("--save-image", type=str, default=None, help="Save detection image to path")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="Detection box threshold")
    args = parser.parse_args()

    robot_config = f"robot_configs/robot/so101_robot{args.robot}.yaml"
    pix2robot_file = f"robot_configs/pix2robot_matrices/robot{args.robot}_pix2robot_data.npz"

    # ========== 1. Camera ==========
    print("\n[1/5] Camera initialization...")
    camera = RealSenseD435()
    camera.start()
    intrinsics = camera.get_intrinsics()
    print(f"  Intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")

    # Warm up (skip first few frames)
    for _ in range(10):
        camera.get_frames()

    color, depth = camera.get_frames()
    if color is None:
        print("ERROR: Failed to capture image")
        camera.stop()
        return 1
    print(f"  Captured: {color.shape[1]}x{color.shape[0]}")

    # ========== 2. Detection ==========
    print(f"\n[2/5] Detecting '{args.query}'...")
    detector = GroundingDINODetector(box_threshold=args.box_threshold)
    detector.load_model()

    detections = detector.detect(color, args.query)
    if not detections:
        print(f"  No '{args.query}' detected!")
        camera.stop()
        return 1

    # Pick highest confidence
    best = max(detections, key=lambda d: d.confidence)
    cx, cy = best.center
    x1, y1, x2, y2 = best.bbox
    print(f"  Found: '{best.label}' conf={best.confidence:.2f} center=({cx}, {cy}) bbox=({x1},{y1},{x2},{y2})")

    # Save detection image
    if args.save_image:
        vis = color.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{best.label} {best.confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imwrite(args.save_image, vis)
        print(f"  Saved: {args.save_image}")

    # ========== 3. Pixel → Robot ==========
    print(f"\n[3/5] Pixel → Robot coordinate transform...")
    pix2robot = Pix2RobotCalibrator(robot_id=args.robot)
    if not pix2robot.load(pix2robot_file):
        print("ERROR: Failed to load Pix2Robot calibration")
        camera.stop()
        return 1

    depth_m = camera.get_depth_at_pixel(cx, cy, depth)
    print(f"  Pixel: ({cx}, {cy}), depth={depth_m:.3f}m")

    obj_depth = depth_m if depth_m > 0.05 else None
    world_pos = np.array(pix2robot.pixel_to_robot(cx, cy, depth_m=obj_depth))

    # Z clamping (table objects should be near z=0)
    Z_MAX = 0.15
    Z_DEFAULT = 0.02
    if world_pos[2] < 0 or world_pos[2] > Z_MAX:
        print(f"  [Z-FIX] z={world_pos[2]*100:.1f}cm out of range, clamping to {Z_DEFAULT*100:.0f}cm")
        world_pos[2] = Z_DEFAULT

    print(f"  Robot position: [{world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}] m")

    camera.stop()

    # ========== 4. Dry-run check ==========
    if args.dry_run:
        print(f"\n[DRY RUN] Would move robot{args.robot} to:")
        print(f"  World:    [{world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}]")
        print(f"  Approach: [{world_pos[0]:.4f}, {world_pos[1]:.4f}, {args.approach_height:.4f}]")
        return 0

    # ========== 5. Move Robot ==========
    print(f"\n[4/5] Connecting robot{args.robot}...")
    skills = LeRobotSkills(
        robot_config=robot_config,
        frame="world",
        verbose=True,
    )
    skills.connect()

    try:
        # Move to initial state
        print(f"\n[5/5] Moving to detected object...")
        skills.move_to_initial_state()
        skills.gripper_open()

        # Approach above object
        approach_pos = [world_pos[0], world_pos[1], args.approach_height]
        print(f"\n  Approach: {approach_pos}")
        success = skills.move_to_position(
            approach_pos,
            target_name=args.query,
            skill_description=f"approach above {args.query}",
        )

        if success:
            print(f"\n  Successfully reached above '{args.query}'!")
            # Descend to object
            descend_pos = [world_pos[0], world_pos[1], world_pos[2] + 0.02]
            print(f"  Descend: {descend_pos}")
            skills.move_to_position(
                descend_pos,
                target_name=args.query,
                skill_description=f"descend to {args.query}",
            )
        else:
            print(f"\n  Failed to reach approach position")

        # Return to initial
        skills.move_to_initial_state()

    finally:
        skills.disconnect()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
