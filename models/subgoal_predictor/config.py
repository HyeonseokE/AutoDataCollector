"""Configuration for Subgoal Prediction Model."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    # --- Dataset ---
    dataset_path: str = ""
    n_max: int = 10  # max subgoals per episode
    pose_dim: int = 6  # xyzrpy
    goal_dim: int = 7  # xyzrpy + gripper
    val_ratio: float = 0.2
    image_size: tuple[int, int] = (384, 384)  # SigLIP input

    # --- Model ---
    d_model: int = 256
    n_decoder_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    siglip_model: str = "google/siglip-base-patch16-384"

    # --- Training ---
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 500
    batch_size: int = 4
    lambda_grip: float = 1.0
    lambda_valid: float = 0.1
    huber_delta: float = 1.0
    warmup_epochs: int = 20
    seed: int = 42

    # --- Paths ---
    output_dir: str = "models/checkpoints"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
