"""SmolVLA → preselective_filter.PolicyAdapter implementation.

Decisions reflected:
- #2a (z pooling)    : mean over (batch, chunk) → (expert_hidden_size,)
- #2b (L_FM MC)      : N_b=8 batched single forward with (ε, t) sampling
- #2c (sample batch) : batch=M one denoising chain for AC_model mode set
- #4  (context sim)  : c_m = mean-pool prefix_embs → (expert_hidden_size,)

Implementation notes:
- forward_fm runs one transformer forward at batch=N_b. (ε, t) per row.
- prefix_embs (for c_m) captured by monkey-patching embed_prefix.
- suffix_out (for z) captured by forward hook on action_out_proj.
- Both captures populate self._captured during the forward call, then
  read out and cleared.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from preselective_filter.types import ActionChunk, Context, FMOutput


@dataclass
class SmolVLAAdapterConfig:
    n_fm_mc_samples: int = 8   # N_b — (ε, t) Monte Carlo samples for L_FM
    z_pool: str = "mean"       # decision #2a — only "mean" supported in v1
    device: str = "cuda"
    debug_verbose: bool = False  # P1 logging — per-call timing + L_FM range


class SmolVLAAdapter:
    """Wraps a loaded SmolVLAPolicy to satisfy preselective_filter.PolicyAdapter.

    Caller is responsible for loading the policy and its processor; this
    class only orchestrates per-call data prep, batched forward, and
    hidden-state capture.
    """

    def __init__(
        self,
        policy: Any,                  # lerobot.policies.smolvla.SmolVLAPolicy
        preprocessor: Any,            # PolicyProcessorPipeline (input side)
        config: SmolVLAAdapterConfig | None = None,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.config = config or SmolVLAAdapterConfig()

        if self.config.z_pool != "mean":
            raise NotImplementedError(
                f"z_pool='{self.config.z_pool}' not supported; only 'mean' in v1"
            )

        self._captured: dict[str, torch.Tensor] = {}
        self._install_capture_hooks()

    # ------------------------------------------------------------------
    # Capture installation
    # ------------------------------------------------------------------
    def _install_capture_hooks(self) -> None:
        """Hook prefix_embs (via embed_prefix wrap) and suffix_out (via
        action_out_proj forward hook)."""
        model = self.policy.model  # VLAFlowMatching

        # 1. Monkey-patch embed_prefix to capture prefix_embs
        original_embed_prefix = model.embed_prefix

        def patched_embed_prefix(*args, **kwargs):
            prefix_embs, pad_masks, att_masks = original_embed_prefix(*args, **kwargs)
            self._captured["prefix_embs"] = prefix_embs.detach()
            return prefix_embs, pad_masks, att_masks

        model.embed_prefix = patched_embed_prefix

        # 2. Forward hook on action_out_proj — input[0] is suffix_out
        def hook(_module, inputs, _output) -> None:
            self._captured["suffix_out"] = inputs[0].detach()

        model.action_out_proj.register_forward_hook(hook)

    # ------------------------------------------------------------------
    # PolicyAdapter.forward_fm
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_fm(self, context: Context, action_chunk: ActionChunk) -> FMOutput:
        """Returns (L_FM averaged over N_b MC samples, mean-pooled z, mean-pooled c_m)."""
        device = self.config.device
        N_b = self.config.n_fm_mc_samples
        t_start = time.perf_counter() if self.config.debug_verbose else 0.0

        # 1. Build a batch with a single sample, then preprocess (tokenize, etc.)
        batch = self._build_single_sample_batch(context, action_chunk)
        batch = self.preprocessor(batch)

        # 2. Replicate every tensor along batch dim to N_b
        batch_nb = {k: self._repeat_to_n(v, N_b) for k, v in batch.items()}

        # 3. Sample N_b independent (ε, t) pairs
        actions = batch_nb[_ACTION_KEY]  # (N_b, chunk_size, max_action_dim)
        noise = torch.randn_like(actions, device=device)            # ε
        time = torch.rand(N_b, device=device, dtype=actions.dtype)  # t ~ U(0,1)

        # 4. Forward (reduction="none" → per-sample loss (N_b,))
        self._captured.clear()
        per_sample_loss, _ = self.policy.forward(
            batch_nb, noise=noise, time=time, reduction="none"
        )
        l_fm = float(per_sample_loss.mean().item())

        # 5. Pool captured hiddens
        suffix_out = self._captured["suffix_out"]   # (N_b, chunk_size, H_expert)
        z = suffix_out.mean(dim=(0, 1)).float().cpu().numpy()

        prefix_embs = self._captured["prefix_embs"]  # (N_b, prefix_len, H_expert)
        c_m = prefix_embs.mean(dim=(0, 1)).float().cpu().numpy()

        if self.config.debug_verbose:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            print(
                f"[preselective_filter]   forward_fm N_b={N_b}: {elapsed_ms:.1f}ms "
                f"L_FM={l_fm:.4f} z_dim={z.shape[0]} c_m_dim={c_m.shape[0]}"
            )

        return FMOutput(l_fm=l_fm, z=z, c_m=c_m)

    # ------------------------------------------------------------------
    # PolicyAdapter.sample_actions
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self, context: Context, n_samples: int,
    ) -> list[ActionChunk]:
        """Returns n_samples action chunks ~ π₀(·|context) via batched denoising."""
        device = self.config.device
        t_start = time.perf_counter() if self.config.debug_verbose else 0.0

        # 1. Build single-sample batch + preprocess
        batch = self._build_single_sample_batch(context, action_chunk=None)
        batch = self.preprocessor(batch)

        # 2. Replicate to batch=n_samples
        batch_n = {k: self._repeat_to_n(v, n_samples) for k, v in batch.items()}

        # 3. Extract inputs in the form VLAFlowMatching.sample_actions expects
        images, img_masks = self.policy.prepare_images(batch_n)
        state = self.policy.prepare_state(batch_n)
        lang_tokens = batch_n[_LANG_TOKENS_KEY]
        lang_masks = batch_n[_LANG_MASK_KEY]

        # 4. Independent noise per row → one denoising chain → (n_samples, chunk, dim)
        cfg = self.policy.model.config
        noise = torch.randn(
            n_samples, cfg.chunk_size, cfg.max_action_dim, device=device,
        )
        chunks = self.policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
        )

        # 5. Strip padding to real action_dim (max_action_dim is zero-padded)
        chunks = chunks[..., : cfg.max_action_dim]
        out = [c.float().cpu().numpy() for c in chunks]

        if self.config.debug_verbose:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            print(
                f"[preselective_filter]   sample_actions M={n_samples}: "
                f"{elapsed_ms:.1f}ms chunk_shape={out[0].shape}"
            )

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_single_sample_batch(
        self, context: Context, action_chunk: ActionChunk | None,
    ) -> dict[str, Any]:
        """Assemble the raw (pre-processor) batch dict for SmolVLA.

        Caller passes only the salient fields; the preprocessor pipeline
        adds language tokens, normalization, device placement.
        """
        batch: dict[str, Any] = {
            _STATE_KEY: torch.as_tensor(context.state, device=self.config.device),
            _TASK_KEY: context.instruction,
        }
        # Observation images — caller supplies them in context.observation;
        # exact key naming depends on the policy's image_features config.
        # The integration layer is responsible for shaping `context.observation`
        # to match the policy's expected image keys.
        if isinstance(context.observation, dict):
            for k, v in context.observation.items():
                batch[k] = torch.as_tensor(v, device=self.config.device)
        else:
            # Single ndarray case: assume the policy has exactly one image feature.
            image_keys = list(self.policy.config.image_features.keys())
            if len(image_keys) != 1:
                raise ValueError(
                    "context.observation must be a dict when the policy has "
                    f"multiple image features: {image_keys}"
                )
            batch[image_keys[0]] = torch.as_tensor(
                context.observation, device=self.config.device,
            )

        if action_chunk is not None:
            batch[_ACTION_KEY] = torch.as_tensor(
                action_chunk, device=self.config.device, dtype=torch.float32,
            )
        return batch

    @staticmethod
    def _repeat_to_n(value: Any, n: int) -> Any:
        """Repeat the first (batch) dim of a tensor n times; passthrough non-tensors."""
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value.expand(n)
            return value.expand(n, *value.shape[1:]).clone() if value.shape[0] == 1 \
                else value.repeat_interleave(n // value.shape[0], dim=0)
        return value


# Lerobot batch-dict keys (lazy import to avoid pulling lerobot at module
# import time when the adapter isn't used).
def _lerobot_keys() -> tuple[str, str, str, str, str]:
    from lerobot.utils.constants import (
        ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE,
    )
    return ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, "task"


_ACTION_KEY, _STATE_KEY, _LANG_TOKENS_KEY, _LANG_MASK_KEY, _TASK_KEY = _lerobot_keys()
