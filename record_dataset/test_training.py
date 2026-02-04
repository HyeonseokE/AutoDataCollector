#!/usr/bin/env python3
"""
Test LeRobot Training with Recorded Dataset

1. 더미 데이터로 2개 에피소드 데이터셋 생성
2. 데이터셋 로드 및 검증
3. Diffusion Policy로 간단한 학습 테스트

Usage:
    python record_dataset/test_training.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))


def create_dummy_dataset():
    """2개 에피소드의 더미 데이터셋 생성"""
    print("\n" + "=" * 60)
    print("Step 1: Creating dummy dataset (2 episodes)")
    print("=" * 60)

    from record_dataset import DatasetRecorder
    from record_dataset.utils import DummyCamera

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_id = f"local/training_test_{timestamp}"

    print(f"  Repo ID: {repo_id}")

    recorder = DatasetRecorder(repo_id=repo_id, fps=30, robot_type="so101")
    camera = DummyCamera()

    # Episode 1: Pick task (100 frames, ~3.3 seconds)
    print("\n  Recording Episode 1 (pick task)...")
    recorder.start_episode(task="pick up the yellow dice")

    for i in range(100):
        # 시뮬레이션된 pick trajectory
        t = i / 100.0
        # Arm moves down, grabs, moves up
        base_pos = np.array([0.0, 30.0, -20.0, 10.0, 0.0], dtype=np.float32)
        if t < 0.3:  # Move down
            offset = np.array([0.0, -20.0 * (t / 0.3), 10.0 * (t / 0.3), 0.0, 0.0])
        elif t < 0.5:  # Grab
            offset = np.array([0.0, -20.0, 10.0, 0.0, 0.0])
        else:  # Move up
            offset = np.array([0.0, -20.0 + 20.0 * ((t - 0.5) / 0.5), 10.0 - 10.0 * ((t - 0.5) / 0.5), 0.0, 0.0])

        state = base_pos + offset
        # Action is slightly ahead of state (simulating tracking delay)
        action = state + np.random.uniform(-2, 2, 5).astype(np.float32)
        # Add gripper
        gripper = 50.0 if t < 0.35 else -50.0  # Open then close
        state_full = np.concatenate([state, [gripper]])
        action_full = np.concatenate([action, [gripper]])

        image, _ = camera.get_frames()

        recorder.record_frame(
            observation=state_full,
            action=action_full,
            image=image,
        )

    recorder.end_episode()
    print(f"    Episode 1: 100 frames recorded")

    # Episode 2: Place task (80 frames, ~2.7 seconds)
    print("  Recording Episode 2 (place task)...")
    recorder.start_episode(task="place the dice on the plate")

    for i in range(80):
        t = i / 80.0
        # Arm moves to target, releases, moves back
        base_pos = np.array([10.0, 10.0, -10.0, 5.0, 0.0], dtype=np.float32)
        if t < 0.4:  # Move to target
            offset = np.array([20.0 * (t / 0.4), 10.0 * (t / 0.4), -20.0 * (t / 0.4), 0.0, 0.0])
        elif t < 0.6:  # Release
            offset = np.array([20.0, 10.0, -20.0, 0.0, 0.0])
        else:  # Move back
            offset = np.array([20.0 - 20.0 * ((t - 0.6) / 0.4), 10.0 - 10.0 * ((t - 0.6) / 0.4), -20.0 + 20.0 * ((t - 0.6) / 0.4), 0.0, 0.0])

        state = base_pos + offset
        action = state + np.random.uniform(-2, 2, 5).astype(np.float32)
        gripper = -50.0 if t < 0.55 else 50.0  # Closed then open
        state_full = np.concatenate([state, [gripper]])
        action_full = np.concatenate([action, [gripper]])

        image, _ = camera.get_frames()

        recorder.record_frame(
            observation=state_full,
            action=action_full,
            image=image,
        )

    recorder.end_episode()
    print(f"    Episode 2: 80 frames recorded")

    # Finalize
    recorder.finalize()

    print(f"\n  Dataset created!")
    print(f"    Total episodes: {recorder.episode_count}")
    print(f"    Total frames: {recorder.total_frames}")
    print(f"    Path: {recorder._dataset.root}")

    return repo_id, str(recorder._dataset.root)


def verify_dataset(repo_id: str, root: str):
    """데이터셋 로드 및 검증"""
    print("\n" + "=" * 60)
    print("Step 2: Verifying dataset")
    print("=" * 60)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root)

    print(f"  Loaded successfully!")
    print(f"    Total episodes: {dataset.meta.total_episodes}")
    print(f"    Total frames: {dataset.meta.total_frames}")
    print(f"    FPS: {dataset.meta.fps}")
    print(f"    Features: {list(dataset.meta.features.keys())}")

    # Sample check
    sample = dataset[0]
    print(f"\n  Sample 0:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"    {key}: {type(value).__name__}")

    return dataset


def test_training(dataset):
    """간단한 학습 테스트"""
    print("\n" + "=" * 60)
    print("Step 3: Testing training loop")
    print("=" * 60)

    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Features
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(f"  Input features: {list(input_features.keys())}")
    print(f"  Output features: {list(output_features.keys())}")

    # Create policy config
    # Use smaller model for testing
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # Smaller config for faster testing
        diffusion_step_embed_dim=32,
        down_dims=[64, 128],  # Correct parameter name
        n_action_steps=8,
        n_obs_steps=2,
    )

    print(f"\n  Creating DiffusionPolicy...")
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)

    # Preprocessors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)

    # Delta timestamps for Diffusion Policy
    delta_timestamps = {
        "observation.images.front": [i / dataset.meta.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset.meta.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset.meta.fps for i in cfg.action_delta_indices],
    }

    print(f"  Delta timestamps: {delta_timestamps}")

    # Recreate dataset with delta timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset_with_delta = LeRobotDataset(
        dataset.meta.repo_id,
        root=str(dataset.root),
        delta_timestamps=delta_timestamps,
    )

    # DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset_with_delta,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # For testing
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Training loop (5 steps)
    print(f"\n  Running 5 training steps...")
    policy.train()

    for step, batch in enumerate(dataloader):
        if step >= 5:
            break

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Preprocess
        batch = preprocessor(batch)

        # Forward
        loss, output_dict = policy.forward(batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"    Step {step + 1}: loss = {loss.item():.4f}")

    print(f"\n  Training test completed successfully!")
    return True


def main():
    print("\n" + "=" * 60)
    print("LeRobot Training Pipeline Test")
    print("=" * 60)

    try:
        # Step 1: Create dummy dataset
        repo_id, root = create_dummy_dataset()

        # Step 2: Verify dataset
        dataset = verify_dataset(repo_id, root)

        # Step 3: Test training
        success = test_training(dataset)

        print("\n" + "=" * 60)
        if success:
            print("ALL TESTS PASSED!")
            print("=" * 60)
            print(f"\nThe recording pipeline is compatible with LeRobot training.")
            print(f"Dataset location: {root}")
            return 0
        else:
            print("TESTS FAILED!")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
