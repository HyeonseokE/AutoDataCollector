"""Dataset loader for subgoal prediction.

Loads a LeRobot v3.0 dataset and extracts per-episode samples:
    Input:  (o₀, instruction, s₀)
    Output: (subgoals [N_max, 7], valid_mask [N_max])

Each subgoal is (x, y, z, roll, pitch, yaw, gripper) in robot base frame.
"""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.subgoal_predictor.config import ModelConfig


class SubgoalDataset(Dataset):
    """Extracts per-episode subgoal sequences from a LeRobot dataset."""

    def __init__(
        self,
        dataset_path: str,
        config: ModelConfig,
        episode_indices: list[int] | None = None,
        norm_stats: dict | None = None,
        image_transform=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.image_transform = image_transform

        # Load metadata
        with open(self.dataset_path / "meta" / "info.json") as f:
            self.info = json.load(f)

        # Load task instruction
        tasks_df = pd.read_parquet(self.dataset_path / "meta" / "tasks.parquet")
        self.instruction = tasks_df.index[0]  # task text is the index

        # Load all parquet files and extract subgoal sequences
        self.episodes = self._load_episodes(episode_indices)

        # Compute or apply normalization stats
        if norm_stats is None:
            self.norm_stats = self._compute_norm_stats()
        else:
            self.norm_stats = norm_stats

        # Video path for frame extraction
        self.video_path = (
            self.dataset_path
            / "videos"
            / "observation.images.realsense"
            / "chunk-000"
            / "file-000.mp4"
        )

    def _load_episodes(self, episode_indices: list[int] | None) -> list[dict]:
        """Load per-episode data: first frame info + subgoal sequence."""
        data_dir = self.dataset_path / "data" / "chunk-000"
        parquet_files = sorted(data_dir.glob("*.parquet"))

        episodes = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            ep_idx = int(df["episode_index"].iloc[0])

            if episode_indices is not None and ep_idx not in episode_indices:
                continue

            # First frame data
            first_row = df.iloc[0]
            s0 = np.array(first_row["observation.state"], dtype=np.float32)
            global_frame_idx = int(first_row["index"])  # for video extraction

            # Extract subgoal transitions (where skill.type changes)
            mask = df["skill.type"] != df["skill.type"].shift(1)
            transitions = df[mask]

            subgoals = []
            for _, row in transitions.iterrows():
                robot_xyzrpy = np.array(
                    row["skill.goal_position.robot_xyzrpy"], dtype=np.float32
                )
                gripper = np.float32(row["skill.goal_position.gripper"])
                # Binarize gripper: open (>0) → 1.0, closed (<=0) → 0.0
                gripper_binary = np.float32(1.0 if gripper > 0 else 0.0)
                goal = np.concatenate([robot_xyzrpy, [gripper_binary]])
                subgoals.append(goal)

            episodes.append(
                {
                    "episode_index": ep_idx,
                    "s0": s0,
                    "subgoals": np.stack(subgoals),  # (N, 7)
                    "n_subgoals": len(subgoals),
                    "video_frame_idx": global_frame_idx,
                }
            )

        return episodes

    def _compute_norm_stats(self) -> dict:
        """Compute mean/std of pose dimensions (first 6) across all subgoals."""
        all_poses = np.concatenate(
            [ep["subgoals"][:, :6] for ep in self.episodes], axis=0
        )
        all_states = np.stack([ep["s0"] for ep in self.episodes], axis=0)
        return {
            "pose_mean": all_poses.mean(axis=0).astype(np.float32),
            "pose_std": all_poses.std(axis=0).astype(np.float32).clip(min=1e-6),
            "state_mean": all_states.mean(axis=0).astype(np.float32),
            "state_std": all_states.std(axis=0).astype(np.float32).clip(min=1e-6),
        }

    def _extract_frame(self, global_frame_idx: int) -> np.ndarray:
        """Extract a single frame from the video file."""
        container = av.open(str(self.video_path))
        stream = container.streams.video[0]

        # Seek to approximate position
        fps = stream.average_rate
        target_pts = int(global_frame_idx * stream.time_base.denominator / float(fps))
        container.seek(target_pts, stream=stream)

        frame_count = 0
        for frame in container.decode(video=0):
            current_idx = frame.pts * float(fps) * frame.time_base
            if int(round(current_idx)) >= global_frame_idx or frame_count > 0:
                img = frame.to_ndarray(format="rgb24")
                container.close()
                return img
            frame_count += 1

        container.close()
        raise RuntimeError(f"Could not extract frame {global_frame_idx}")

    def _normalize_pose(self, pose: np.ndarray) -> np.ndarray:
        """Normalize xyzrpy (first 6 dims) to zero-mean unit-variance."""
        normalized = pose.copy()
        normalized[:, :6] = (pose[:, :6] - self.norm_stats["pose_mean"]) / self.norm_stats["pose_std"]
        return normalized

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        n = ep["n_subgoals"]
        n_max = self.config.n_max

        # Image (o₀)
        image = self._extract_frame(ep["video_frame_idx"])
        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            # Default: resize to config size and normalize to [0,1]
            from torchvision.transforms import functional as F
            from PIL import Image

            image = Image.fromarray(image)
            image = F.resize(image, list(self.config.image_size))
            image = F.to_tensor(image)  # (3, H, W), [0, 1]

        # State (s₀) — normalized
        s0 = self._normalize_state(ep["s0"])

        # Subgoals — pad to N_max
        subgoals_raw = ep["subgoals"]  # (N, 7)
        subgoals_normed = self._normalize_pose(subgoals_raw)

        subgoals_padded = np.zeros((n_max, 7), dtype=np.float32)
        valid_mask = np.zeros(n_max, dtype=np.float32)

        subgoals_padded[:n] = subgoals_normed[:n]
        valid_mask[:n] = 1.0

        return {
            "image": image,  # (3, H, W)
            "instruction": self.instruction,
            "state": torch.from_numpy(s0),  # (6,)
            "subgoals": torch.from_numpy(subgoals_padded),  # (N_max, 7)
            "valid_mask": torch.from_numpy(valid_mask),  # (N_max,)
            "n_subgoals": n,
        }


def build_datasets(
    dataset_path: str,
    config: ModelConfig,
    image_transform=None,
) -> tuple["SubgoalDataset", "SubgoalDataset"]:
    """Build train/val datasets with shared normalization stats."""
    # Load full dataset to get episode list
    full_ds = SubgoalDataset(dataset_path, config, image_transform=image_transform)
    n_episodes = len(full_ds.episodes)
    all_indices = [ep["episode_index"] for ep in full_ds.episodes]

    # Split
    rng = np.random.RandomState(config.seed)
    rng.shuffle(all_indices)
    n_val = max(1, int(n_episodes * config.val_ratio))
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    # Compute norm stats from train set only
    train_ds = SubgoalDataset(
        dataset_path, config, episode_indices=train_indices,
        image_transform=image_transform,
    )
    norm_stats = train_ds.norm_stats

    val_ds = SubgoalDataset(
        dataset_path, config, episode_indices=val_indices,
        norm_stats=norm_stats, image_transform=image_transform,
    )

    return train_ds, val_ds
