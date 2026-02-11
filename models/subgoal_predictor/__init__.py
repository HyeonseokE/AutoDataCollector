"""
Subgoal Predictor Model

Open-loop subgoal sequence prediction from initial observation.

Architecture:
    - SigLIP (frozen): Vision-Language encoder
    - Transformer Decoder: Predicts N subgoals in parallel
    - Output: (xyz, rpy, gripper) × N_max

Input:  (o₀, I, s₀) - initial image, instruction, robot state
Output: [(x,y,z,r,p,y,grip), ...] × N subgoals

Training:
    python -m models.subgoal_predictor.train --dataset <path> --epochs 500

Inference:
    ./models/subgoal_predictor/run_inference.sh --execute

Best config: d_model=64, n_layers=1 (170K params)
Val MAE: 2.86mm position, 1.09deg rotation
"""

from models.subgoal_predictor.config import ModelConfig
from models.subgoal_predictor.dataset import SubgoalDataset, build_datasets
from models.subgoal_predictor.model import SubgoalPredictor
from models.subgoal_predictor.loss import SubgoalLoss

__all__ = [
    "ModelConfig",
    "SubgoalDataset",
    "build_datasets",
    "SubgoalPredictor",
    "SubgoalLoss",
]
