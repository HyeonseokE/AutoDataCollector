"""Training script for Subgoal Prediction Model."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from models.subgoal_predictor.config import ModelConfig
from models.subgoal_predictor.dataset import SubgoalDataset, build_datasets
from models.subgoal_predictor.loss import SubgoalLoss
from models.subgoal_predictor.model import SubgoalPredictor


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: stack tensors, collect instruction strings."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
        "state": torch.stack([b["state"] for b in batch]),
        "subgoals": torch.stack([b["subgoals"] for b in batch]),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
    }


def train_one_epoch(
    model: SubgoalPredictor,
    loader: DataLoader,
    criterion: SubgoalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    # Keep SigLIP in eval mode (frozen BatchNorm, etc.)
    model.siglip.eval()

    total_losses = {"total": 0, "pose": 0, "gripper": 0, "valid": 0}
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        states = batch["state"].to(device)
        targets = batch["subgoals"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        instructions = batch["instruction"]

        pred = model(images, instructions, states)
        losses = criterion(pred, targets, valid_mask)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    model: SubgoalPredictor,
    loader: DataLoader,
    criterion: SubgoalLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_losses = {"total": 0, "pose": 0, "gripper": 0, "valid": 0}
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        states = batch["state"].to(device)
        targets = batch["subgoals"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        instructions = batch["instruction"]

        pred = model(images, instructions, states)
        losses = criterion(pred, targets, valid_mask)

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


def train(config: ModelConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # --- Datasets ---
    print(f"Loading dataset from {config.dataset_path}")
    train_ds, val_ds = build_datasets(config.dataset_path, config)
    print(f"Train: {len(train_ds)} episodes, Val: {len(val_ds)} episodes")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # --- Model ---
    print("Building model...")
    model = SubgoalPredictor(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- Optimizer & Scheduler ---
    criterion = SubgoalLoss(config)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=config.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )

    # --- Output dir ---
    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Save norm stats for inference
    norm_stats_serializable = {
        k: v.tolist() for k, v in train_ds.norm_stats.items()
    }
    with open(output_path / "norm_stats.json", "w") as f:
        json.dump(norm_stats_serializable, f, indent=2)

    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # --- Training loop ---
    best_val_loss = float("inf")
    history = []

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Pose':>8} {'Grip':>8} {'Valid':>8} {'LR':>10} {'Time':>6}")
    print("-" * 76)

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_losses = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses["total"],
                "val_loss": val_losses["total"],
                "val_pose": val_losses["pose"],
                "val_gripper": val_losses["gripper"],
                "val_valid": val_losses["valid"],
                "lr": lr,
            }
        )

        # Save best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "config": vars(config),
                    "norm_stats": norm_stats_serializable,
                },
                output_path / "best.pt",
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{epoch:>6d} {train_losses['total']:>10.4f} {val_losses['total']:>10.4f} "
                f"{val_losses['pose']:>8.4f} {val_losses['gripper']:>8.4f} {val_losses['valid']:>8.4f} "
                f"{lr:>10.2e} {dt:>5.1f}s"
            )

    # Save final
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "val_loss": val_losses["total"],
            "config": vars(config),
            "norm_stats": norm_stats_serializable,
        },
        output_path / "final.pt",
    )

    # Save history
    with open(output_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {output_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train Subgoal Prediction Model")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = ModelConfig(dataset_path=args.dataset)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.n_layers is not None:
        config.n_decoder_layers = args.n_layers
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed

    train(config)


if __name__ == "__main__":
    main()
