#!/usr/bin/env python3
"""
H6 실험: 학습에 쓰인 데이터셋의 in-distribution observation 으로 정책을
오프라인에서 돌려 action chunk 를 저장.

배포 추론(RTC on, 실제 카메라)과 비교해 지그재그가 OOD 때문인지 확인.

Usage:
    PYTHONPATH=src:scripts python -m chunk_visualizer.offline_policy_eval \
        --policy-path skkuprism/pi05_test_pick_place_10K \
        --dataset-repo-id skkuprism/test_pick_red_place_blue_50epi \
        --episode 1 \
        --num-chunks 15 \
        --output outputs/action_chunks/offline_eval_ep1.npz \
        --device cuda
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def load_policy(policy_path: str, device: str):
    config = PreTrainedConfig.from_pretrained(policy_path)
    config.pretrained_path = policy_path
    policy_class = get_policy_class(config.type)
    if config.type in ("pi05", "pi0"):
        config.compile_model = False
    policy = policy_class.from_pretrained(policy_path, config=config)
    policy = policy.to(device)
    policy.eval()
    return policy, config


def episode_frame_indices(dataset: LeRobotDataset, episode: int) -> list[int]:
    """Return dataset-level indices belonging to `episode`."""
    idxs = []
    for i in range(len(dataset)):
        if int(dataset.hf_dataset[i]["episode_index"]) == episode:
            idxs.append(i)
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--num-chunks", type=int, default=15,
                    help="How many observations to sample from the episode")
    ap.add_argument("--stride", type=int, default=0,
                    help="If >0, pick frames at this stride; else evenly spaced")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading policy: {args.policy_path}")
    policy, policy_cfg = load_policy(args.policy_path, args.device)
    print(f"  type={policy_cfg.type} chunk_size={policy_cfg.chunk_size}")

    print(f"Loading dataset: {args.dataset_repo_id}")
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    delta_ts = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_ts)
    print(f"  frames={len(dataset)} episodes={dataset.num_episodes} fps={dataset.fps}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    ep_idxs = episode_frame_indices(dataset, args.episode)
    if not ep_idxs:
        raise RuntimeError(f"Episode {args.episode} has no frames")
    print(f"  episode {args.episode}: {len(ep_idxs)} frames")

    # Pick observation frame indices within the episode
    if args.stride > 0:
        picks = ep_idxs[:: args.stride][: args.num_chunks]
    else:
        # evenly spaced
        picks = [ep_idxs[i] for i in np.linspace(0, len(ep_idxs) - 1, args.num_chunks, dtype=int)]
    print(f"  running policy on {len(picks)} observations")

    action_features = list(dataset.features["action"]["names"])
    chunks = []
    with torch.no_grad():
        for k, idx in enumerate(picks):
            sample = dataset[idx]
            # add batch dim (torch tensors / strings)
            batched = {}
            for key, val in sample.items():
                if isinstance(val, torch.Tensor):
                    batched[key] = val.unsqueeze(0)
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    batched[key] = torch.tensor([val])
                elif isinstance(val, str):
                    batched[key] = [val]
                elif isinstance(val, np.ndarray):
                    batched[key] = torch.as_tensor(val).unsqueeze(0)
                else:
                    batched[key] = [val]
            processed = preprocessor(batched)
            action_chunk = policy.predict_action_chunk(processed)     # (1, chunk_size, max_action_dim)
            action_chunk = postprocessor(action_chunk)                # denormalize back to joint space
            arr = action_chunk[0, :, : len(action_features)].detach().float().cpu().numpy()
            chunks.append(arr)
            print(f"    [{k+1}/{len(picks)}] frame_idx={idx}  chunk shape={arr.shape}  "
                  f"range=[{arr.min():.2f}, {arr.max():.2f}]")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "action_features": np.array(action_features, dtype=object),
        "fps": np.float64(dataset.fps),
        "timestamps": np.arange(len(chunks), dtype=np.float64) * (policy_cfg.chunk_size / float(dataset.fps)),
    }
    for i, ch in enumerate(chunks):
        save_kwargs[f"chunk_{i}"] = ch
    np.savez(out_path, **save_kwargs)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
