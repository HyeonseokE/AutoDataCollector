#!/usr/bin/env python3
"""
Convert a LeRobotDataset episode into the same .npz format that the RTC
inference pipeline emits, so the existing chunk visualizer can render
demo trajectories side-by-side with model outputs.

Usage:
    python -m chunk_visualizer.dataset_to_chunks \
        --repo-id skkuprism/test_pick_red_place_blue_50epi \
        --episode 1 \
        --chunk-size 50 \
        --output outputs/action_chunks/demo_ep1.npz
"""

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION


def episode_actions(repo_id: str, episode: int, root: str | None = None) -> tuple[np.ndarray, list[str], float]:
    ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
    ep_frames = ds.hf_dataset.filter(lambda x: x["episode_index"] == episode)
    action_col = ep_frames.select_columns(ACTION)
    actions = np.stack([np.asarray(action_col[i][ACTION], dtype=np.float32) for i in range(len(ep_frames))])
    feature_names = list(ds.features[ACTION]["names"])
    fps = float(ds.fps)
    return actions, feature_names, fps


def split_into_chunks(actions: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    if chunk_size <= 0:
        return [actions]
    n = len(actions)
    return [actions[i : i + chunk_size] for i in range(0, n, chunk_size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--root", type=str, default=None, help="Optional local cache root for the dataset")
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="Split the episode into pseudo-chunks of this length. 0 = single chunk")
    ap.add_argument("--output", required=True, help="Output .npz path")
    args = ap.parse_args()

    print(f"Loading {args.repo_id} episode {args.episode} ...")
    actions, features, fps = episode_actions(args.repo_id, args.episode, root=args.root)
    print(f"  frames: {len(actions)} | features: {features} | fps: {fps}")

    chunks = split_into_chunks(actions, args.chunk_size)
    print(f"  split into {len(chunks)} chunks of size ~{args.chunk_size}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {
        "action_features": np.array(features, dtype=object),
        "fps": np.float64(fps),
        "timestamps": np.arange(len(chunks), dtype=np.float64) * (args.chunk_size / fps),
    }
    for i, ch in enumerate(chunks):
        save_kwargs[f"chunk_{i}"] = ch
    np.savez(out_path, **save_kwargs)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
