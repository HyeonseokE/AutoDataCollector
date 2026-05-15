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
    autocast_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
                                      # bf16 halves activation memory at the
                                      # cost of slight numeric drift; the
                                      # final MSE is still computed in fp32.


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
        noise = torch.randn_like(actions, device=device)                  # ε
        time_samples = torch.rand(N_b, device=device, dtype=actions.dtype)  # t ~ U(0,1)

        # 4. Forward (reduction="none" → per-sample loss (N_b,))
        self._captured.clear()
        per_sample_loss, _ = self.policy.forward(
            batch_nb, noise=noise, time=time_samples, reduction="none"
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
    # PolicyAdapter.forward_fm_batched  — K candidates × N_b MC in one forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_fm_batched(
        self, context: Context, action_chunks: list[ActionChunk],
    ) -> list[FMOutput]:
        """Batched L_FM across K action candidates that share the same context.

        Key optimization vs naive batching: prefix (images + language + state)
        is computed ONCE on batch=1 and expanded as a memory-view across the
        K · N_b sample dim. The vision encoder is therefore called once
        instead of K · N_b times — saves both VRAM and compute when the
        context is identical across candidates (which is always true in a
        per-step skill-acquirer call).
        """
        K = len(action_chunks)
        if K == 0:
            return []
        if K == 1:
            return [self.forward_fm(context, action_chunks[0])]

        import torch.nn.functional as F
        from contextlib import nullcontext
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        device = self.config.device
        N_b = self.config.n_fm_mc_samples
        KN = K * N_b
        t_start = time.perf_counter() if self.config.debug_verbose else 0.0

        ac_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                    "float32": torch.float32}.get(self.config.autocast_dtype,
                                                  torch.bfloat16)
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=ac_dtype)
            if ac_dtype != torch.float32 and device != "cpu" else nullcontext()
        )

        # Release PyTorch's caching allocator pool to defragment GPU memory.
        # Without this, repeat invocations leave 700MB+ "reserved-but-unallocated"
        # blocks that prevent curobo's next plan_batch from finding a contiguous
        # allocation. Cheap (microseconds), and the cache rebuilds on demand.
        if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
            torch.cuda.empty_cache()

        # ---- 1. Build a single-row batch with ONLY the shared context
        #         (no action). Preprocess once to get normalized state +
        #         tokenized language + normalized images.
        base_batch = self._build_single_sample_batch(context, action_chunk=None)
        base_batch = self.preprocessor(base_batch)

        model = self.policy.model

        # ---- 2. Prepare images/state/language on batch=1.
        images_1, img_masks_1 = self.policy.prepare_images(base_batch)
        state_1 = self.policy.prepare_state(base_batch)
        lang_tokens_1 = base_batch[_LANG_TOKENS_KEY]
        lang_masks_1 = base_batch[_LANG_MASK_KEY]

        # ---- 3. Compute prefix once under autocast. embed_prefix is
        #         monkey-patched to capture prefix_embs into self._captured.
        self._captured.clear()
        with autocast_ctx:
            prefix_embs_1, prefix_pad_masks_1, prefix_att_masks_1 = model.embed_prefix(
                images_1, img_masks_1, lang_tokens_1, lang_masks_1, state=state_1,
            )
        # (1, P, H), (1, P), (1, P)

        # ---- 4. Pre-normalize the K action chunks. We run the preprocessor
        #         per-candidate (cheap — it's just normalization + concat,
        #         no model forward), then stack into (K, T, D).
        per_cand_actions: list[torch.Tensor] = []
        for ac in action_chunks:
            b = self._build_single_sample_batch(context, ac)
            b = self.preprocessor(b)
            per_cand_actions.append(b[_ACTION_KEY])
        actions_k = torch.cat(per_cand_actions, dim=0)              # (K, T, D)
        # Each candidate's action is shared across its N_b MC samples.
        actions_kn = actions_k.repeat_interleave(N_b, dim=0)        # (K·N_b, T, D)

        # ---- 5. Independent (ε, t) per row.
        noise = torch.randn_like(actions_kn, device=device)
        time_samples = torch.rand(KN, device=device, dtype=actions_kn.dtype)

        # ---- 6. Build x_t and u_t (mirrors VLAFlowMatching.forward).
        time_expanded = time_samples[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions_kn
        u_t = noise - actions_kn

        # ---- 7. Embed suffix per-row. (K·N_b, T, H_expert)
        with autocast_ctx:
            suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(
                x_t, time_samples,
            )

        # ---- 8. Expand prefix to (K·N_b, P, H) as a memory view — NO copy.
        prefix_embs_kn = prefix_embs_1.expand(KN, *prefix_embs_1.shape[1:])
        prefix_pad_masks_kn = prefix_pad_masks_1.expand(KN, *prefix_pad_masks_1.shape[1:])
        prefix_att_masks_kn = prefix_att_masks_1.expand(KN, *prefix_att_masks_1.shape[1:])

        # ---- 9. Concat masks, run the main expert transformer under autocast.
        pad_masks = torch.cat([prefix_pad_masks_kn, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_kn, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        with autocast_ctx:
            (_, suffix_out), _ = model.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_kn, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            suffix_out = suffix_out[:, -model.config.chunk_size:]
        # Loss/projection back in fp32 for numerical stability.
        suffix_out_f32 = suffix_out.to(dtype=torch.float32)
        v_t = model.action_out_proj(suffix_out_f32)

        # ---- 10. Per-row MSE → per-candidate L_FM
        losses = F.mse_loss(u_t, v_t, reduction="none")
        # Strip padded action_dim — same as VLAFlowMatching.forward.
        losses = losses[:, :, : model.config.max_action_dim]
        per_sample_loss = losses.mean(dim=(1, 2))                   # (K·N_b,)
        loss_k = per_sample_loss.view(K, N_b).mean(dim=1)           # (K,)

        # ---- 11. z from suffix_out: (K·N_b, T, H) → (K, H_expert)
        z_k = suffix_out_f32.view(K, N_b, *suffix_out_f32.shape[1:]) \
            .mean(dim=(1, 2)).float().cpu().numpy()

        # ---- 12. c_m is shared (prefix is K-independent). Mean-pool prefix
        #         over (batch=1, prefix_len) → (H,) and broadcast to K.
        c_m_shared = prefix_embs_1.mean(dim=(0, 1)).float().cpu().numpy()  # (H,)

        if self.config.debug_verbose:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            l_min = float(loss_k.min().item())
            l_max = float(loss_k.max().item())
            print(
                f"[preselective_filter]   forward_fm_batched K={K} N_b={N_b}: "
                f"{elapsed_ms:.1f}ms L_FM range=[{l_min:.4f}, {l_max:.4f}] "
                f"z_dim={z_k.shape[1]} c_m_dim={c_m_shared.shape[0]} "
                f"(prefix shared, 1× encoder pass)"
            )

        result = [
            FMOutput(
                l_fm=float(loss_k[k].item()), z=z_k[k], c_m=c_m_shared,
            )
            for k in range(K)
        ]
        # Free SmolVLA's transient activation cache so curobo's next plan_batch
        # call has contiguous memory available.
        if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
            torch.cuda.empty_cache()
        return result

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

        # Defragment caching allocator before allocating the denoising batch
        # so curobo's next plan_batch finds a contiguous block.
        if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
            torch.cuda.empty_cache()

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

        # Free denoising activations so curobo's next plan_batch sees a clean cache.
        if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
            torch.cuda.empty_cache()
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
        state = torch.as_tensor(
            context.state, device=self.config.device, dtype=torch.float32,
        )
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (D,) → (1, D) so _repeat_to_n can expand
        batch: dict[str, Any] = {
            _STATE_KEY: state,
            _TASK_KEY: context.instruction,
        }
        # Observation images — caller supplies them in context.observation;
        # exact key naming depends on the policy's image_features config.
        # The integration layer is responsible for shaping `context.observation`
        # to match the policy's expected image keys.
        if isinstance(context.observation, dict):
            for k, v in context.observation.items():
                batch[k] = torch.as_tensor(
                    v, device=self.config.device, dtype=torch.float32,
                )
        else:
            # Single ndarray case: assume the policy has exactly one image feature.
            image_keys = list(self.policy.config.image_features.keys())
            if len(image_keys) != 1:
                raise ValueError(
                    "context.observation must be a dict when the policy has "
                    f"multiple image features: {image_keys}"
                )
            batch[image_keys[0]] = torch.as_tensor(
                context.observation, device=self.config.device, dtype=torch.float32,
            )

        if action_chunk is not None:
            ac = torch.as_tensor(
                action_chunk, device=self.config.device, dtype=torch.float32,
            )
            if ac.dim() == 2:
                ac = ac.unsqueeze(0)  # (T, D) → (1, T, D) so _repeat_to_n can expand
            batch[_ACTION_KEY] = ac
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
