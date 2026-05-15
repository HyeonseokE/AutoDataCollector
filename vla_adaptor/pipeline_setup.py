"""Pipeline integration entry point.

Parses the `preselective_filter:` yaml section, loads the SmolVLA policy +
its preprocessor, constructs JsonlBufferStore, builds Selector, and
returns it for the caller to inject into skills (mirroring the existing
_setup_skill_perturbation_on_skills pattern in
execution_forward_and_reset.py).

Expected yaml schema:

    preselective_filter:
      enabled: false
      policy:
        checkpoint: "CoRL2026-CSI/smol_CaP_pnp_10fps"   # HF hub or local
        device: "cuda"
      buffer:
        root: "./results/<session>/preselective_buffer"  # per-session
      selector:
        alpha: 0.5
        lam: 0.5
        n_vla_samples: 8
        max_modes: null               # None = identity (decision (a))
        context_k: 8
      adapter:
        n_fm_mc_samples: 8            # N_b — decision #2b
        z_pool: "mean"                # decision #2a
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from preselective_filter import Selector, SelectorConfig

from .jsonl_buffer import JsonlBufferStore
from .smolvla_adapter import SmolVLAAdapter, SmolVLAAdapterConfig


def setup_preselective_filter(
    recording_config: dict | None,
) -> Selector | None:
    """Builds a Selector from a parsed recording_config dict.

    Returns None when:
      - recording_config is None or empty
      - preselective_filter section is missing
      - preselective_filter.enabled is false
    """
    if not recording_config:
        return None
    section = recording_config.get("preselective_filter") or {}
    # Phase-split aware: enabled if either forward or reset is on.
    # Legacy `enabled: <bool>` still supported as fallback.
    en_fwd = section.get("enabled_forward")
    en_reset = section.get("enabled_reset")
    if en_fwd is None and en_reset is None:
        if not section.get("enabled", False):
            return None
    elif not (bool(en_fwd) or bool(en_reset)):
        return None

    policy_cfg = section.get("policy") or {}
    buffer_cfg = section.get("buffer") or {}
    selector_cfg = section.get("selector") or {}
    adapter_cfg = section.get("adapter") or {}
    debug_verbose = bool(section.get("debug_verbose", False))

    # ---- Load SmolVLA policy + preprocessor via lerobot factory ----
    checkpoint = policy_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "preselective_filter.policy.checkpoint is required when enabled"
        )
    device = policy_cfg.get("device", "cuda")
    print(
        f"[preselective_filter] loading SmolVLA checkpoint='{checkpoint}' device={device} ..."
    )
    policy, preprocessor = _load_smolvla(checkpoint, device)
    print("[preselective_filter] SmolVLA loaded")

    # ---- Build adapter ----
    adapter = SmolVLAAdapter(
        policy=policy,
        preprocessor=preprocessor,
        config=SmolVLAAdapterConfig(
            n_fm_mc_samples=int(adapter_cfg.get("n_fm_mc_samples", 8)),
            z_pool=str(adapter_cfg.get("z_pool", "mean")),
            device=device,
            debug_verbose=debug_verbose,
            autocast_dtype=str(adapter_cfg.get("autocast_dtype", "bfloat16")),
        ),
    )

    # ---- Build buffer ----
    buffer_root = buffer_cfg.get("root")
    if not buffer_root:
        raise ValueError(
            "preselective_filter.buffer.root is required when enabled"
        )
    buffer = JsonlBufferStore(root=Path(buffer_root), debug_verbose=debug_verbose)
    print(f"[preselective_filter] buffer root={buffer_root}")

    # ---- Build selector ----
    sel_config = SelectorConfig(
        alpha=float(selector_cfg.get("alpha", 0.5)),
        lam=float(selector_cfg.get("lam", 0.5)),
        n_vla_samples=int(selector_cfg.get("n_vla_samples", 8)),
        max_modes=selector_cfg.get("max_modes"),  # None or int
        context_k=int(selector_cfg.get("context_k", 8)),
        debug_verbose=debug_verbose,
    )
    selector = Selector(policy=adapter, buffer=buffer, config=sel_config)
    print(
        f"[preselective_filter] Selector ready: "
        f"α={sel_config.alpha} λ={sel_config.lam} "
        f"M={sel_config.n_vla_samples} k={sel_config.context_k} "
        f"N_b={adapter.config.n_fm_mc_samples} debug_verbose={debug_verbose}"
    )
    return selector


def teardown_preselective_filter(selector: Selector | None) -> None:
    """Release adapter resources at session end.

    JsonlBufferStore is filesystem-backed and needs no explicit teardown.
    The SmolVLA policy releases its GPU memory when the process exits or
    when del'd; this function is a hook for future cleanup needs.
    """
    if selector is None:
        return
    # If the policy adapter ever holds explicit resources, release here.
    adapter = getattr(selector, "policy", None)
    close = getattr(adapter, "close", None)
    if callable(close):
        try:
            close()
        except Exception as e:  # pragma: no cover
            print(f"[preselective_filter] teardown adapter.close() failed: {e}")


# ----------------------------------------------------------------------
# SmolVLA loader (kept private; depends on lerobot specifics)
# ----------------------------------------------------------------------
def _load_smolvla(checkpoint: str, device: str) -> tuple[Any, Any]:
    """Load a SmolVLAPolicy + its preprocessor pipeline from a HF hub id or
    local path. Returns (policy_on_device, preprocessor)."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
    preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=None)
    return policy, preprocessor
