"""Backbone-last-feature key extractor — produces the vector-DB key.

A pretrained VLA is used ONLY as a frozen feature extractor (no flow-matching
loss, no action sampling). For a context (images + instruction + robot state)
it returns the FAISS key.

WHAT FEATURE — the *backbone's last feature*
--------------------------------------------
A VLA runs: vision/language encoders → backbone NN → VL joint feature → action
head. The key must be that **VL joint feature fed into the action head** — the
backbone's *last* feature — NOT the raw ``embed_prefix`` input embeddings.

Concretely the backbone is a transformer; its last-layer output over the
prefix (image+language[+state]) tokens is the conditioning the action head /
action expert reads. We run the backbone on the prefix only and mean-pool that
last-layer output. (We do not use the action-token feature right before the
final ``action_out_proj`` linear, because it depends on the noisy action and is
therefore not a deterministic key.)

PER FAMILY — dispatch by the user-provided checkpoint path
----------------------------------------------------------
- smolvla : prefix output of ``vlm_with_expert``; state IS already fused into
            the prefix (``state_proj``) → key = normalize(VL+state feature).
- pi0/pi05: prefix output of ``paligemma_with_expert``; prefix is VL-only
            (pi05 discretizes state into the prompt, has no ``state_proj``) →
            key = concat(normalize(VL feature), state_weight*normalize(state)).
- groot   : Eagle backbone ``backbone_features``; VL-only → same concat as pi0.

Each block is L2-normalized so a low-dim state block is not drowned by the
high-dim VL block; ``state_weight`` tunes their relative influence.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any

import numpy as np

VLAFamily = str  # one of: "smolvla", "pi0", "pi05", "groot"


def _lerobot_lang_keys() -> tuple[str, str, str]:
    """(state_key, lang_tokens_key, lang_mask_key) — lazy lerobot import."""
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )
    return OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


def _l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (float(np.linalg.norm(v)) + eps)


def _infer_family(checkpoint: str) -> VLAFamily:
    """Map a checkpoint path / HF repo id to a VLA family.

    Branch is purely on the user-provided checkpoint string. ``pi05`` is
    checked before ``pi0`` because the former is a substring-superset case.
    Pass ``family=`` to ``make_vla_key_extractor`` to override this heuristic.
    """
    s = str(checkpoint).lower()
    if "smolvla" in s:
        return "smolvla"
    if "groot" in s or "gr00t" in s:
        return "groot"
    if "pi05" in s or "pi0.5" in s or "pi0_5" in s or "pi0p5" in s:
        return "pi05"
    if "pi0" in s or "pi-0" in s:
        return "pi0"
    raise ValueError(
        f"cannot infer VLA family from checkpoint '{checkpoint}'. "
        "Pass family= explicitly (smolvla | pi0 | pi05 | groot)."
    )


# --------------------------------------------------------------------------
# Abstract interface
# --------------------------------------------------------------------------
class VLAKeyExtractor(ABC):
    """Frozen-VLA → FAISS key. Thread-unsafe; call ``encode`` serially."""

    @abstractmethod
    def encode(
        self, observation: Any, instruction: str, state: np.ndarray,
    ) -> np.ndarray:
        """Return the FAISS key for one context.

        observation : raw {camera_name: HxWxC array} dict.
        instruction : task text.
        state       : continuous robot proprioception (1-D float array).
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int | None:
        """Key vector dimension. None until the first ``encode`` call."""

    @abstractmethod
    def close(self) -> None:
        """Release GPU memory held by the policy."""


# --------------------------------------------------------------------------
# Shared torch/lerobot machinery
# --------------------------------------------------------------------------
class _TorchVLAExtractor(VLAKeyExtractor):
    """Common base — policy loading, image mapping, autocast, key assembly.

    Subclasses implement ``_load_policy`` and ``_backbone_feature`` (the
    masked-mean-pooled backbone last feature for one context). ``_fuses_state``
    declares whether that feature already contains proprioception.
    """

    _fuses_state: bool = False

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        autocast_dtype: str = "bfloat16",
        state_weight: float = 1.0,
        debug_verbose: bool = False,
    ) -> None:
        import torch

        self._torch = torch
        self.checkpoint = checkpoint
        self.device = device
        self.state_weight = float(state_weight)
        self.debug_verbose = debug_verbose

        self._state_key, self._lang_tokens_key, self._lang_mask_key = (
            _lerobot_lang_keys()
        )
        self._ac_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(autocast_dtype, torch.bfloat16)
        self._embedding_dim: int | None = None

        self.policy = self._load_policy()
        self._init_image_spec()

    # ---- subclass hooks --------------------------------------------------
    @abstractmethod
    def _load_policy(self) -> Any:
        """Load and return the frozen policy (eval mode, on device)."""

    @abstractmethod
    def _backbone_feature(
        self, observation: Any, instruction: str, state: np.ndarray,
    ) -> np.ndarray:
        """Backbone last feature for one context, mean-pooled to a 1-D vector."""

    # ---- shared helpers --------------------------------------------------
    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def _init_image_spec(self) -> None:
        """Policy image-feature keys + target (C,H,W) for raw-frame mapping."""
        cfg = self.policy.config
        self._image_keys = list(getattr(cfg, "image_features", {}) or {})
        self._image_target_shape = (
            tuple(cfg.image_features[self._image_keys[0]].shape)
            if self._image_keys else None
        )

    def _autocast(self):
        torch = self._torch
        if self._ac_dtype != torch.float32 and str(self.device) != "cpu":
            return torch.amp.autocast("cuda", dtype=self._ac_dtype)
        return nullcontext()

    def _map_raw_images(self, raw_imgs: dict | None) -> dict[str, np.ndarray]:
        """Map raw {camera: HxWxC uint8} frames onto the policy image-feature
        keys, resized to (C,H,W) float32 in [0,1] with a leading batch dim.
        """
        if not self._image_keys or not raw_imgs or self._image_target_shape is None:
            return {}
        import cv2  # lazy

        C, H, W = (int(x) for x in self._image_target_shape)
        out: dict[str, np.ndarray] = {}
        items = list(raw_imgs.items())
        for i, key in enumerate(self._image_keys):
            if i >= len(items):
                break
            frame = items[i][1]
            if frame is None:
                continue
            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.shape[2] == 3:          # HWC
                if arr.shape[:2] != (H, W):
                    arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
                arr = arr.transpose(2, 0, 1)
            elif arr.ndim == 3 and arr.shape[0] == 3:        # CHW
                if arr.shape[1:] != (H, W):
                    hwc = cv2.resize(
                        arr.transpose(1, 2, 0), (W, H),
                        interpolation=cv2.INTER_AREA,
                    )
                    arr = hwc.transpose(2, 0, 1)
            else:
                continue
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)
            out[key] = arr[None, ...]                        # (1, C, H, W)
        return out

    def _build_batch(
        self, observation: Any, instruction: str, state: np.ndarray,
    ) -> dict[str, Any]:
        """Raw context → a batch dict the policy preprocessor accepts."""
        torch = self._torch
        s = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0)  # (D,) → (1, D)
        batch: dict[str, Any] = {self._state_key: s, "task": instruction}
        raw = observation if isinstance(observation, dict) else None
        for k, v in self._map_raw_images(raw).items():
            batch[k] = torch.as_tensor(v, device=self.device, dtype=torch.float32)
        return batch

    def _masked_mean(self, feats: Any, pad_mask: Any) -> np.ndarray:
        """Mean-pool feats (1, L, H) over valid tokens given pad_mask (1, L)."""
        torch = self._torch
        f = feats[0].to(torch.float32)                       # (L, H)
        if pad_mask is None:
            return f.mean(dim=0).cpu().numpy()
        m = pad_mask[0].to(torch.float32)                    # (L,)
        denom = m.sum().clamp(min=1.0)
        return ((f * m[:, None]).sum(dim=0) / denom).cpu().numpy()

    def encode(
        self, observation: Any, instruction: str, state: np.ndarray,
    ) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            vl = self._backbone_feature(observation, instruction, state)

        vl_unit = _l2_normalize(vl)
        if self._fuses_state:
            # Proprioception is already inside the backbone feature.
            key = np.ascontiguousarray(vl_unit, dtype=np.float32)
            state_dim = 0
        else:
            state_unit = _l2_normalize(state) * self.state_weight
            key = np.ascontiguousarray(
                np.concatenate([vl_unit, state_unit]), dtype=np.float32,
            )
            state_dim = state_unit.shape[0]

        if self._embedding_dim is None:
            self._embedding_dim = int(key.shape[0])
            if self.debug_verbose:
                print(
                    f"[vla_embedding] family={type(self).__name__} "
                    f"key dim={self._embedding_dim} "
                    f"(VL={vl_unit.shape[0]} + state={state_dim}, "
                    f"state_weight={self.state_weight})"
                )
        return key

    def close(self) -> None:
        torch = self._torch
        self.policy = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------
# pi0 / pi05
# --------------------------------------------------------------------------
class _PI0Extractor(_TorchVLAExtractor):
    """pi0 / pi05 — prefix output of the PaliGemma+expert backbone (VL only).

    pi05 has no ``state_proj``; continuous proprioception is concatenated
    externally by the base ``encode`` (``_fuses_state = False``).
    """

    _fuses_state = False

    def _load_policy(self) -> Any:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        policy = PI05Policy.from_pretrained(self.checkpoint).to(self.device).eval()
        self.preprocessor, _ = make_pre_post_processors(
            policy.config, dataset_stats=None,
        )
        return policy

    def _backbone_feature(self, observation, instruction, state):
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        torch = self._torch
        batch = self._build_batch(observation, instruction, state)
        batch = self.preprocessor(batch)

        model = self.policy.model
        images, img_masks = self.policy._preprocess_images(batch)
        tokens = batch[self._lang_tokens_key]
        masks = batch[self._lang_mask_key]

        with self._autocast():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, tokens, masks,
            )
            # Match the weight dtype, mirroring PI05Pytorch.forward.
            lm = model.paligemma_with_expert.paligemma.model.language_model
            if lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att_4d = model._prepare_attention_masks_4d(att_2d)
            lm.config._attn_implementation = "eager"

            # Run the backbone on the prefix only → last-layer prefix output.
            (prefix_out, _suffix), _pkv = model.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
            )
        return self._masked_mean(prefix_out, prefix_pad_masks)


# --------------------------------------------------------------------------
# SmolVLA
# --------------------------------------------------------------------------
class _SmolVLAExtractor(_TorchVLAExtractor):
    """SmolVLA — prefix output of the SmolVLM+expert backbone.

    SmolVLA's ``embed_prefix`` fuses image + language + state (via
    ``state_proj``); the backbone feature already carries proprioception, so
    no external state concat (``_fuses_state = True``).
    """

    _fuses_state = True

    def _load_policy(self) -> Any:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(self.checkpoint).to(self.device)
        policy.eval()
        self.preprocessor, _ = make_pre_post_processors(
            policy.config, dataset_stats=None,
        )
        return policy

    def _backbone_feature(self, observation, instruction, state):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        torch = self._torch
        batch = self._build_batch(observation, instruction, state)
        batch = self.preprocessor(batch)

        model = self.policy.model
        images, img_masks = self.policy.prepare_images(batch)
        proc_state = self.policy.prepare_state(batch)
        tokens = batch[self._lang_tokens_key]
        masks = batch[self._lang_mask_key]

        with self._autocast():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, tokens, masks, state=proc_state,
            )
            ref_w = model.vlm_with_expert.get_vlm_model().text_model
            if next(ref_w.parameters()).dtype == torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            # Run the backbone on the prefix only → last-layer prefix output.
            outputs_embeds, _pkv = model.vlm_with_expert.forward(
                attention_mask=att_2d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                fill_kv_cache=False,
            )
        prefix_out = outputs_embeds[0]
        return self._masked_mean(prefix_out, prefix_pad_masks)


# --------------------------------------------------------------------------
# GR00T (Isaac-GR00T N1.5)
# --------------------------------------------------------------------------
class _GrootExtractor(_TorchVLAExtractor):
    """GR00T — Eagle backbone ``backbone_features`` (VL only).

    GR00T's action head has its own state encoder; the Eagle backbone is
    VL-only, so continuous state is concatenated externally
    (``_fuses_state = False``).
    """

    _fuses_state = False

    def _load_policy(self) -> Any:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        policy = GrootPolicy.from_pretrained(self.checkpoint).to(self.device)
        policy.eval()
        self.preprocessor, _ = make_pre_post_processors(
            policy.config, dataset_stats=None,
        )
        return policy

    def _backbone_feature(self, observation, instruction, state):
        batch = self._build_batch(observation, instruction, state)
        batch = self.preprocessor(batch)

        groot = self.policy._groot_model
        with self._autocast():
            backbone_inputs, _action_inputs = groot.prepare_input(batch)
            backbone_out = groot.backbone(backbone_inputs)
        feats = backbone_out["backbone_features"]            # (1, n, H)
        mask = backbone_out.get("backbone_attention_mask")   # (1, n)
        return self._masked_mean(feats, mask)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
_EXTRACTORS: dict[VLAFamily, type[_TorchVLAExtractor]] = {
    "pi0": _PI0Extractor,
    "pi05": _PI0Extractor,
    "smolvla": _SmolVLAExtractor,
    "groot": _GrootExtractor,
}


def make_vla_key_extractor(
    checkpoint: str,
    device: str = "cuda",
    autocast_dtype: str = "bfloat16",
    state_weight: float = 1.0,
    debug_verbose: bool = False,
    family: VLAFamily | None = None,
) -> VLAKeyExtractor:
    """Build the right ``VLAKeyExtractor`` for a checkpoint.

    Dispatch is by the checkpoint path/repo-id string (see ``_infer_family``);
    pass ``family`` to override the heuristic for an ambiguously-named
    local checkpoint.
    """
    fam = family or _infer_family(checkpoint)
    if fam not in _EXTRACTORS:
        raise ValueError(
            f"unsupported VLA family '{fam}' "
            f"(known: {sorted(_EXTRACTORS)})"
        )
    if debug_verbose:
        print(
            f"[vla_embedding] checkpoint='{checkpoint}' → family='{fam}' "
            f"→ {_EXTRACTORS[fam].__name__}"
        )
    return _EXTRACTORS[fam](
        checkpoint=checkpoint,
        device=device,
        autocast_dtype=autocast_dtype,
        state_weight=state_weight,
        debug_verbose=debug_verbose,
    )
