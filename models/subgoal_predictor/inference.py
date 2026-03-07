"""Inference and evaluation for Subgoal Prediction Model."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.subgoal_predictor.config import ModelConfig
from models.subgoal_predictor.dataset import SubgoalDataset
from models.subgoal_predictor.model import SubgoalPredictor


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: stack tensors, collect instruction strings."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
        "state": torch.stack([b["state"] for b in batch]),
        "subgoals": torch.stack([b["subgoals"] for b in batch]),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
        "n_subgoals": [b["n_subgoals"] for b in batch],
    }


def load_model(checkpoint_path: str, device: torch.device) -> tuple[SubgoalPredictor, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config
    cfg_dict = checkpoint["config"]
    config = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})

    model = SubgoalPredictor(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_stats = {k: np.array(v, dtype=np.float32) for k, v in checkpoint["norm_stats"].items()}

    return model, norm_stats, config


def denormalize_pose(pose: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Denormalize xyzrpy from zero-mean unit-variance back to original scale."""
    denorm = pose.copy()
    denorm[..., :6] = pose[..., :6] * norm_stats["pose_std"] + norm_stats["pose_mean"]
    return denorm


@torch.no_grad()
def evaluate_model(
    model: SubgoalPredictor,
    dataset: SubgoalDataset,
    norm_stats: dict,
    device: torch.device,
) -> dict:
    """Evaluate model on dataset, compute per-subgoal and per-dimension errors."""
    model.eval()

    all_pose_errors = []  # (N_samples, N_subgoals, 6)
    all_gripper_correct = []
    all_valid_correct = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    results = []

    for batch in loader:
        images = batch["image"].to(device)
        states = batch["state"].to(device)
        instructions = batch["instruction"]
        gt_subgoals = batch["subgoals"].numpy()[0]  # (N_max, 7)
        gt_valid = batch["valid_mask"].numpy()[0]  # (N_max,)
        n_subgoals = batch["n_subgoals"][0]

        # Forward
        out = model(images, instructions, states)
        pred_pose = out["pose"].cpu().numpy()[0]  # (N_max, 6)
        pred_gripper_logits = out["gripper"].cpu().numpy()[0, :, 0]  # (N_max,)
        pred_valid_logits = out["valid"].cpu().numpy()[0, :, 0]  # (N_max,)

        # Binary predictions
        pred_gripper = (pred_gripper_logits > 0).astype(float)
        pred_valid = (pred_valid_logits > 0).astype(float)

        # Denormalize poses for interpretable errors
        gt_pose_denorm = denormalize_pose(gt_subgoals[:, :6], norm_stats)
        pred_pose_denorm = denormalize_pose(pred_pose, norm_stats)

        # Compute errors for valid subgoals only
        for i in range(n_subgoals):
            pose_error = np.abs(pred_pose_denorm[i] - gt_pose_denorm[i])
            all_pose_errors.append(pose_error)

            gt_grip = gt_subgoals[i, 6]
            all_gripper_correct.append(pred_gripper[i] == gt_grip)

        # Valid mask accuracy (all slots)
        all_valid_correct.extend((pred_valid == gt_valid).tolist())

        results.append({
            "n_subgoals": n_subgoals,
            "gt_pose": gt_pose_denorm[:n_subgoals],
            "pred_pose": pred_pose_denorm[:n_subgoals],
            "gt_gripper": gt_subgoals[:n_subgoals, 6],
            "pred_gripper": pred_gripper[:n_subgoals],
            "gt_valid": gt_valid,
            "pred_valid": pred_valid,
        })

    all_pose_errors = np.array(all_pose_errors)  # (total_subgoals, 6)

    # Aggregate metrics
    dim_names = ["x", "y", "z", "roll", "pitch", "yaw"]
    metrics = {
        "n_episodes": len(dataset),
        "n_subgoals_total": len(all_pose_errors),
        "gripper_accuracy": np.mean(all_gripper_correct) * 100,
        "valid_mask_accuracy": np.mean(all_valid_correct) * 100,
        "pose_mae_total": np.mean(all_pose_errors),
    }

    for i, name in enumerate(dim_names):
        metrics[f"mae_{name}"] = np.mean(all_pose_errors[:, i])

    # Position (xyz) and rotation (rpy) MAE
    metrics["mae_xyz"] = np.mean(all_pose_errors[:, :3])
    metrics["mae_rpy"] = np.mean(all_pose_errors[:, 3:])

    # Convert xyz MAE to mm for readability
    metrics["mae_xyz_mm"] = metrics["mae_xyz"] * 1000

    return metrics, results


def print_metrics(metrics: dict):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes: {metrics['n_episodes']}, Total subgoals: {metrics['n_subgoals_total']}")
    print()
    print("Position Error (xyz):")
    print(f"  x:   {metrics['mae_x']*1000:6.2f} mm")
    print(f"  y:   {metrics['mae_y']*1000:6.2f} mm")
    print(f"  z:   {metrics['mae_z']*1000:6.2f} mm")
    print(f"  avg: {metrics['mae_xyz_mm']:6.2f} mm")
    print()
    print("Rotation Error (rpy):")
    print(f"  roll:  {np.degrees(metrics['mae_roll']):6.2f} deg")
    print(f"  pitch: {np.degrees(metrics['mae_pitch']):6.2f} deg")
    print(f"  yaw:   {np.degrees(metrics['mae_yaw']):6.2f} deg")
    print(f"  avg:   {np.degrees(metrics['mae_rpy']):6.2f} deg")
    print()
    print(f"Gripper Accuracy:    {metrics['gripper_accuracy']:6.2f}%")
    print(f"Valid Mask Accuracy: {metrics['valid_mask_accuracy']:6.2f}%")
    print("=" * 50)


def print_episode_details(results: list, max_episodes: int = 3):
    """Print detailed predictions for a few episodes."""
    print("\n" + "=" * 50)
    print("SAMPLE EPISODE DETAILS")
    print("=" * 50)

    for ep_idx, r in enumerate(results[:max_episodes]):
        print(f"\nEpisode {ep_idx} ({r['n_subgoals']} subgoals):")
        print(f"{'#':>2} {'GT xyz':^24} {'Pred xyz':^24} {'Err mm':>8} {'GT_g':>5} {'Pr_g':>5}")
        print("-" * 75)

        for i in range(r["n_subgoals"]):
            gt = r["gt_pose"][i]
            pr = r["pred_pose"][i]
            err_mm = np.linalg.norm(gt[:3] - pr[:3]) * 1000
            gt_g = "open" if r["gt_gripper"][i] > 0.5 else "close"
            pr_g = "open" if r["pred_gripper"][i] > 0.5 else "close"
            match = "OK" if gt_g == pr_g else "X"

            print(f"{i:>2} [{gt[0]:6.3f},{gt[1]:6.3f},{gt[2]:6.3f}] "
                  f"[{pr[0]:6.3f},{pr[1]:6.3f},{pr[2]:6.3f}] "
                  f"{err_mm:8.1f} {gt_g:>5} {pr_g:>5} {match}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Subgoal Prediction Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--details", type=int, default=3, help="Number of episodes to show details")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, norm_stats, config = load_model(args.checkpoint, device)
    print(f"Model: d_model={config.d_model}, n_layers={config.n_decoder_layers}")

    # Load dataset
    full_ds = SubgoalDataset(args.dataset, config, norm_stats=norm_stats)
    n_episodes = len(full_ds.episodes)
    all_indices = [ep["episode_index"] for ep in full_ds.episodes]

    rng = np.random.RandomState(config.seed)
    shuffled = all_indices.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(n_episodes * config.val_ratio))
    val_indices = shuffled[:n_val]
    train_indices = shuffled[n_val:]

    if args.split == "train":
        eval_indices = train_indices
    elif args.split == "val":
        eval_indices = val_indices
    else:
        eval_indices = all_indices

    eval_ds = SubgoalDataset(args.dataset, config, episode_indices=eval_indices, norm_stats=norm_stats)
    print(f"Evaluating on {args.split} set: {len(eval_ds)} episodes")

    # Evaluate
    metrics, results = evaluate_model(model, eval_ds, norm_stats, device)

    print_metrics(metrics)
    if args.details > 0:
        print_episode_details(results, args.details)


if __name__ == "__main__":
    main()
