"""IG(ξ) = Ñ_B — buffer novelty (buffer-only; no VLA term).

  N_B(ξ) = min ‖vec(A_ξ) - vec(A_i)‖₂ over the skill buffer B_t^(m)
  IG     = within-set min-max normalization of N_B (higher = more novel)

The VLA flow-matching uncertainty term U_π₀ has been removed; IG is now a
pure buffer-distance signal computed directly on the candidate action chunk.
"""
from __future__ import annotations

import numpy as np

from ..types import ActionChunk, BufferEntry


def minmax_norm(values: list[float], eps: float = 1e-8) -> list[float]:
    """Within-set min-max normalization, eps-stabilized denominator.

    Norm(Q^j) = (Q^j - min Q) / (max Q - min Q + eps)
    """
    arr = np.asarray(values, dtype=np.float64)
    lo = float(arr.min())
    rng = float(arr.max() - lo)
    return [float((v - lo) / (rng + eps)) for v in arr]


def compute_buffer_novelty(
    action_chunk: ActionChunk,
    buffer_entries: list[BufferEntry],
) -> float:
    """N_B(ξ) = min ‖vec(A_ξ) - vec(A_i)‖₂ over B_t^(m).

    Distance is the raw Euclidean norm between flattened action chunks —
    joint-space magnitude is meaningful here, so (unlike the old z-space
    term) no L2 direction normalization is applied.

    Empty buffer → ValueError. Caller (Selector) handles cold-start.
    """
    if not buffer_entries:
        raise ValueError("buffer_entries is empty; caller must handle cold-start")
    a = np.asarray(action_chunk, dtype=np.float64).reshape(-1)

    best = float("inf")
    for entry in buffer_entries:
        ea = np.asarray(entry.action_chunk, dtype=np.float64).reshape(-1)
        if ea.shape != a.shape:
            raise ValueError(
                f"action_chunk shape mismatch: query={a.shape}, entry={ea.shape}"
            )
        d = float(np.linalg.norm(a - ea))
        if d < best:
            best = d
    return best


def compute_ig(novelty_values: list[float], eps: float = 1e-8) -> list[float]:
    """IG = within-set min-max normalized novelty (higher = more novel)."""
    return minmax_norm(novelty_values, eps)
