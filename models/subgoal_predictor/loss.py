"""Loss functions for subgoal prediction.

L_total = L_pose + λ_grip * L_grip + λ_valid * L_valid

- L_pose:  Huber loss on xyzrpy (6D), masked by valid_gt
- L_grip:  BCE loss on gripper (binary), masked by valid_gt
- L_valid: BCE loss on valid mask prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.subgoal_predictor.config import ModelConfig


class SubgoalLoss(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lambda_grip = config.lambda_grip
        self.lambda_valid = config.lambda_valid
        self.huber = nn.HuberLoss(reduction="none", delta=config.huber_delta)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred: dict with 'pose' (B, N, 6), 'gripper' (B, N, 1), 'valid' (B, N, 1)
            targets: (B, N, 7) — first 6 = normalized xyzrpy, last 1 = gripper binary
            valid_mask: (B, N) — 1 for real subgoals, 0 for padding

        Returns:
            dict with 'total', 'pose', 'gripper', 'valid' losses
        """
        gt_pose = targets[:, :, :6]  # (B, N, 6)
        gt_gripper = targets[:, :, 6:7]  # (B, N, 1)
        mask = valid_mask.unsqueeze(-1)  # (B, N, 1)
        n_valid = valid_mask.sum().clamp(min=1)

        # Pose loss: Huber on xyzrpy, masked
        pose_loss = self.huber(pred["pose"], gt_pose)  # (B, N, 6)
        pose_loss = (pose_loss * mask).sum() / n_valid

        # Gripper loss: BCE, masked
        gripper_loss = F.binary_cross_entropy_with_logits(
            pred["gripper"], gt_gripper, reduction="none"
        )
        gripper_loss = (gripper_loss * mask).sum() / n_valid

        # Valid mask loss: BCE on all slots
        valid_loss = F.binary_cross_entropy_with_logits(
            pred["valid"].squeeze(-1),
            valid_mask,
            reduction="mean",
        )

        total = pose_loss + self.lambda_grip * gripper_loss + self.lambda_valid * valid_loss

        return {
            "total": total,
            "pose": pose_loss,
            "gripper": gripper_loss,
            "valid": valid_loss,
        }
