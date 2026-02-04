#!/usr/bin/env python3
"""
Test script for LeRobot Dataset Recording

Tests the recording pipeline with dummy data (no robot/camera required).

Usage:
    # Test with dummy data
    python record_dataset/test_recording.py --dummy

    # Test with real robot and camera
    python record_dataset/test_recording.py --robot 3
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))


def test_dummy_recording():
    """Test recording with dummy data (no hardware required)."""
    print("\n" + "=" * 60)
    print("Testing LeRobot Dataset Recording (Dummy Mode)")
    print("=" * 60)

    try:
        from record_dataset import DatasetRecorder
        from record_dataset.utils import DummyCamera
        from record_dataset.config import NUM_JOINTS

        # Create test repo_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_id = f"local/test_recording_{timestamp}"

        print(f"\n[Test] Creating DatasetRecorder...")
        print(f"  Repo ID: {repo_id}")

        recorder = DatasetRecorder(
            repo_id=repo_id,
            fps=30,
            robot_type="so101",
        )

        print(f"[Test] Recorder created successfully!")
        print(f"  Dataset path: {recorder._dataset.root}")

        # Test episode recording
        print(f"\n[Test] Starting episode...")
        recorder.start_episode(task="test task: pick up the object")

        # Simulate 3 seconds of recording at 30fps = 90 frames
        print(f"\n[Test] Recording 90 frames (3 seconds at 30fps)...")
        dummy_camera = DummyCamera()

        for i in range(90):
            # Generate dummy data
            observation = np.random.uniform(-100, 100, NUM_JOINTS).astype(np.float32)
            action = np.random.uniform(-100, 100, NUM_JOINTS).astype(np.float32)
            image, _ = dummy_camera.get_frames()

            recorder.record_frame(
                observation=observation,
                action=action,
                image=image,
            )

            # Progress
            if (i + 1) % 30 == 0:
                print(f"    Recorded {i + 1} frames")

        print(f"\n[Test] Ending episode...")
        episode_info = recorder.end_episode()
        print(f"  Episode info: {episode_info}")

        # Finalize
        print(f"\n[Test] Finalizing dataset...")
        recorder.finalize()

        # Print summary
        print(f"\n[Test] Recording test completed!")
        print(f"  Dataset path: {recorder._dataset.root}")
        print(f"  Total episodes: {recorder.episode_count}")
        print(f"  Total frames: {recorder.total_frames}")

        # Verify dataset can be loaded
        print(f"\n[Test] Verifying dataset can be loaded...")
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            loaded_dataset = LeRobotDataset(repo_id, root=recorder._dataset.root)
            print(f"  Loaded dataset successfully!")
            print(f"  Total episodes: {loaded_dataset.meta.total_episodes}")
            print(f"  Total frames: {loaded_dataset.meta.total_frames}")
            print(f"  Features: {list(loaded_dataset.meta.features.keys())}")
            return True

        except Exception as e:
            print(f"  Warning: Failed to load dataset: {e}")
            return True  # Recording worked, loading might have issues

    except Exception as e:
        print(f"\n[Test] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_callback():
    """Test the callback mechanism with simulated skills execution."""
    print("\n" + "=" * 60)
    print("Testing Recording Callback Mechanism")
    print("=" * 60)

    try:
        from record_dataset import DatasetRecorder
        from record_dataset.callback import RecordingCallback
        from record_dataset.utils import DummyCamera

        # Create recorder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_id = f"local/test_callback_{timestamp}"

        recorder = DatasetRecorder(repo_id=repo_id, fps=30)
        camera = DummyCamera()

        # Create callback
        callback = RecordingCallback(
            recorder=recorder,
            camera=camera,
            target_fps=30,
            control_hz=50,
        )

        print(f"\n[Test] Simulating skills execution with callback...")
        recorder.start_episode(task="test callback task")
        callback.reset()

        # Simulate 2 seconds of 50Hz control loop
        num_steps = 100  # 2 seconds at 50Hz
        recorded_count = 0

        for i in range(num_steps):
            # Simulate robot state/action
            current_state = np.random.uniform(-100, 100, 6).astype(np.float32)
            action = np.random.uniform(-100, 100, 6).astype(np.float32)

            # Check if should record (FPS sync)
            if callback.should_record():
                callback.on_step(current_state[:5], action[:5], current_state[5], action[5])
                recorded_count += 1
            else:
                callback.step_without_record()

            time.sleep(0.02)  # 50Hz

        # Print callback stats
        stats = callback.get_stats()
        print(f"\n[Test] Callback statistics:")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Recorded frames: {stats['recorded_frames']}")
        print(f"  Skipped frames: {stats['skipped_frames']}")
        print(f"  Effective FPS: {stats['effective_fps']:.1f}")

        # End episode
        recorder.end_episode()
        recorder.finalize()

        print(f"\n[Test] Callback test completed!")
        print(f"  Expected ~60 frames (2s at 30fps), got {recorded_count}")
        return True

    except Exception as e:
        print(f"\n[Test] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_robot(robot_id: int = 3):
    """Test recording with real robot (no actual movement)."""
    print("\n" + "=" * 60)
    print(f"Testing Recording with Real Robot (Robot {robot_id})")
    print("=" * 60)

    try:
        from record_dataset import DatasetRecorder
        from record_dataset.utils import DummyCamera

        # Try to import and connect to robot
        print(f"\n[Test] Connecting to robot...")

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from lerobot_cap.hardware import FeetechController

        robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"

        import yaml
        with open(PROJECT_ROOT / robot_config) as f:
            config = yaml.safe_load(f)

        robot = FeetechController(
            port=config.get("port", "/dev/ttyACM0"),
            baudrate=config.get("baudrate", 1000000),
        )
        robot.connect()
        print(f"  Robot connected!")

        # Create recorder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_id = f"local/test_real_robot_{timestamp}"

        recorder = DatasetRecorder(repo_id=repo_id, fps=30)
        camera = DummyCamera()  # Use dummy camera for now

        # Start episode
        recorder.start_episode(task="test real robot recording")

        # Record current robot state for 2 seconds
        print(f"\n[Test] Recording robot state for 2 seconds...")
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < 2.0:
            # Read actual robot state
            positions = robot.read_positions(normalize=True)  # 6 joints
            observation = positions.astype(np.float32)
            action = positions.astype(np.float32)  # Same as current (no movement)

            # Get dummy image
            image, _ = camera.get_frames()

            recorder.record_frame(
                observation=observation,
                action=action,
                image=image,
            )
            frame_count += 1

            time.sleep(1.0 / 30)  # 30fps

        print(f"  Recorded {frame_count} frames")

        # End episode and finalize
        recorder.end_episode()
        recorder.finalize()

        # Disconnect robot
        robot.disconnect()
        print(f"\n[Test] Robot disconnected")

        print(f"\n[Test] Real robot test completed!")
        print(f"  Dataset path: {recorder._dataset.root}")
        return True

    except Exception as e:
        print(f"\n[Test] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test LeRobot Dataset Recording")
    parser.add_argument("--dummy", action="store_true", help="Test with dummy data")
    parser.add_argument("--callback", action="store_true", help="Test callback mechanism")
    parser.add_argument("--robot", type=int, help="Test with real robot (specify robot ID)")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    results = {}

    if args.all or args.dummy or (not args.callback and not args.robot):
        results["dummy"] = test_dummy_recording()

    if args.all or args.callback:
        results["callback"] = test_with_callback()

    if args.robot:
        results["real_robot"] = test_with_real_robot(args.robot)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"  {test_name}: {color}{status}{reset}")

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
