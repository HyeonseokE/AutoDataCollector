"""Pipeline integration entry point — retrieval-augmented (FAISS + VLA key).

Parses the `preselective_filter:` yaml section, loads a frozen VLA key
extractor (backbone-last-feature → FAISS key), constructs a FaissBufferStore
vector DB, builds the IG·AC Selector, and returns ``(selector, extractor)``.

The VLA is used ONLY as a frozen encoder — its backbone's last feature (the
VL joint feature feeding the action head). The family (smolvla / pi0 / pi05 /
groot) is dispatched from the checkpoint path; see ``vla_embedding``.

Expected yaml schema:

    preselective_filter:
      enabled_forward: true             # forward-only — no reset phase
      policy:
        checkpoint: "lerobot/pi05_base"   # HF hub or local VLA checkpoint
        device: "cuda"
        family: null                      # optional override of path dispatch
        state_weight: 1.0                 # continuous-state block weight in key
      buffer:
        root: "./results/<session>/preselective_buffer"
      selector:
        context_k: 8        # FAISS kNN size for AC
        chunk_size: 50      # action-chunk resampling horizon
"""
from __future__ import annotations

from pathlib import Path

from preselective_filter import Selector, SelectorConfig

from ..vectorDB.faiss_buffer import FaissBufferStore
from ..vectorDB.vla_embedding import VLAKeyExtractor, make_vla_key_extractor


def setup_preselective_filter(
    recording_config: dict | None,
    session_dir: str | None = None,
) -> tuple[Selector, VLAKeyExtractor] | None:
    """Build (Selector, VLAKeyExtractor) from a parsed recording_config dict.

    Buffer location precedence:
      1. explicit ``preselective_filter.buffer.root`` in the yaml — a fixed
         path (e.g. a persistent buffer shared across sessions, or a gRPC
         server's buffer).
      2. else ``<session_dir>/preselective_buffer`` — the buffer lives inside
         the run's auto-generated session folder (the normal local case).
      3. else ``./results/preselective_buffer`` — last-resort fallback when
         neither is given (e.g. a server started without a session).

    Returns None when:
      - recording_config is None or empty
      - preselective_filter section is missing
      - preselective_filter is disabled (enabled_forward false/absent)

    preselective_filter is forward-only — there is no enabled_reset knob.
    """
    if not recording_config:
        return None
    section = recording_config.get("preselective_filter") or {}
    # enabled_forward (legacy fallback: `enabled`).
    if not bool(section.get("enabled_forward", section.get("enabled", False))):
        return None

    policy_cfg = section.get("policy") or {}
    buffer_cfg = section.get("buffer") or {}
    selector_cfg = section.get("selector") or {}
    debug_verbose = bool(section.get("debug_verbose", False))

    # ---- Load frozen VLA key extractor (backbone-last-feature) ----
    checkpoint = policy_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "preselective_filter.policy.checkpoint is required when enabled"
        )
    device = policy_cfg.get("device", "cuda")
    print(f"[preselective_filter] loading frozen VLA key extractor '{checkpoint}' ...")
    extractor = make_vla_key_extractor(
        checkpoint=checkpoint,
        device=device,
        autocast_dtype=str(policy_cfg.get("autocast_dtype", "bfloat16")),
        state_weight=float(policy_cfg.get("state_weight", 1.0)),
        debug_verbose=debug_verbose,
        family=policy_cfg.get("family"),
    )
    print("[preselective_filter] VLA key extractor ready")

    # ---- Build FAISS vector-DB buffer ----
    explicit_root = buffer_cfg.get("root")
    if explicit_root:
        resolved_root, src = Path(explicit_root), "explicit buffer.root"
    elif session_dir:
        resolved_root, src = Path(session_dir) / "preselective_buffer", "session-scoped"
    else:
        resolved_root, src = Path("./results/preselective_buffer"), "fallback (no session_dir)"
    buffer = FaissBufferStore(root=resolved_root, debug_verbose=debug_verbose)
    print(f"[preselective_filter] FAISS buffer root={resolved_root}  [{src}]")

    # ---- Build selector ----
    sel_config = SelectorConfig(
        context_k=int(selector_cfg.get("context_k", 8)),
        debug_verbose=debug_verbose,
    )
    selector = Selector(buffer=buffer, config=sel_config)
    print(
        f"[preselective_filter] Selector ready (FAISS retrieval): "
        f"k={sel_config.context_k} debug_verbose={debug_verbose}"
    )
    return selector, extractor


def teardown_preselective_filter(
    selector: Selector | None,
    extractor: VLAKeyExtractor | None = None,
) -> None:
    """Release the VLA extractor's GPU memory at session end.

    The FaissBufferStore is filesystem-backed and needs no teardown.
    """
    if extractor is not None:
        try:
            extractor.close()
        except Exception as e:  # pragma: no cover
            print(f"[preselective_filter] extractor.close() failed: {e}")
