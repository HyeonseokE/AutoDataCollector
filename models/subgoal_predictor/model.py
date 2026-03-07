"""Subgoal Prediction Model.

Architecture:
    SigLIP (frozen) → vision patches (196, D_sig) + text embedding (1, D_sig)
    State MLP       → state token (1, D)
    Transformer Decoder with learned subgoal queries → (N_max, 7) + valid mask
"""

import torch
import torch.nn as nn
from transformers import SiglipModel, SiglipProcessor

from models.subgoal_predictor.config import ModelConfig


class StateEncoder(nn.Module):
    """Encodes robot joint state s₀ ∈ ℝ⁶ → (1, D)."""

    def __init__(self, state_dim: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state).unsqueeze(1)  # (B, 1, D)


class SubgoalDecoder(nn.Module):
    """Transformer decoder with learned subgoal queries.

    Queries cross-attend to context tokens (vision patches + text + state).
    Self-attention captures inter-subgoal dependencies.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.queries = nn.Parameter(
            torch.randn(1, config.n_max, config.d_model) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.n_decoder_layers
        )

        # Output heads
        self.pose_head = nn.Linear(config.d_model, config.pose_dim)  # xyzrpy
        self.gripper_head = nn.Linear(config.d_model, 1)  # binary
        self.valid_head = nn.Linear(config.d_model, 1)  # binary

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            context: (B, L, D) — vision patches + text token + state token

        Returns:
            pose:    (B, N_max, 6)
            gripper: (B, N_max, 1)  — logits
            valid:   (B, N_max, 1)  — logits
        """
        B = context.size(0)
        queries = self.queries.expand(B, -1, -1)  # (B, N_max, D)

        decoded = self.decoder(queries, context)  # (B, N_max, D)

        return {
            "pose": self.pose_head(decoded),
            "gripper": self.gripper_head(decoded),
            "valid": self.valid_head(decoded),
        }


class SubgoalPredictor(nn.Module):
    """Full model: SigLIP (frozen) + projection + state encoder + decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # SigLIP vision-language encoder (frozen)
        self.siglip = SiglipModel.from_pretrained(config.siglip_model)
        self.processor = SiglipProcessor.from_pretrained(config.siglip_model)
        for param in self.siglip.parameters():
            param.requires_grad = False
        self.siglip.eval()

        siglip_dim = self.siglip.config.vision_config.hidden_size

        # Projection layers: SigLIP dim → d_model
        self.vision_proj = nn.Linear(siglip_dim, config.d_model)
        self.text_proj = nn.Linear(siglip_dim, config.d_model)

        # State encoder
        self.state_encoder = StateEncoder(state_dim=6, d_model=config.d_model)

        # Subgoal decoder
        self.decoder = SubgoalDecoder(config)

    @torch.no_grad()
    def _encode_siglip(
        self, images: torch.Tensor, instructions: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract vision patches and text embedding from SigLIP.

        Args:
            images: (B, 3, H, W) in [0, 1]
            instructions: list of B strings

        Returns:
            vision_tokens: (B, N_patches, D_sig)
            text_token:    (B, 1, D_sig)
        """
        device = images.device

        # Process text
        text_inputs = self.processor(
            text=instructions, padding=True, return_tensors="pt"
        ).to(device)

        # SigLIP forward
        outputs = self.siglip(
            pixel_values=images,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
        )

        vision_tokens = outputs.vision_model_output.last_hidden_state  # (B, N_patches, D)
        text_embedding = outputs.text_model_output.pooler_output  # (B, D)

        return vision_tokens, text_embedding.unsqueeze(1)

    def forward(
        self,
        images: torch.Tensor,
        instructions: list[str],
        states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images:       (B, 3, H, W) in [0, 1]
            instructions: list of B strings
            states:       (B, 6) normalized joint state

        Returns:
            pose:    (B, N_max, 6)
            gripper: (B, N_max, 1)  logits
            valid:   (B, N_max, 1)  logits
        """
        # Frozen SigLIP encoding
        vision_tokens, text_token = self._encode_siglip(images, instructions)

        # Project to d_model
        vision_tokens = self.vision_proj(vision_tokens)  # (B, N_patches, D)
        text_token = self.text_proj(text_token)  # (B, 1, D)

        # State encoding
        state_token = self.state_encoder(states)  # (B, 1, D)

        # Build context: [vision_patches, text, state]
        context = torch.cat([vision_tokens, text_token, state_token], dim=1)

        # Decode subgoals
        return self.decoder(context)

    def predict(
        self,
        images: torch.Tensor,
        instructions: list[str],
        states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Inference: returns denormalized pose + binary gripper + valid mask."""
        self.eval()
        with torch.no_grad():
            out = self.forward(images, instructions, states)
        return {
            "pose": out["pose"],  # still normalized — caller denormalizes
            "gripper": (torch.sigmoid(out["gripper"]) > 0.5).float(),
            "valid": (torch.sigmoid(out["valid"]) > 0.5).float(),
        }
