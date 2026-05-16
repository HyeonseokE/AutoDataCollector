"""AC(ξ) = ÃC_buffer — buffer consistency (buffer-only; no VLA term).

  M_B(x_m) = action chunks of the context-kNN neighbours of x_m in B_t^(m)
  D_buffer(ξ) = min ‖vec(A_ξ) - mode_r‖² over M_B(x_m)
  AC = 1 - within-set min-max normalization of D_buffer (higher = more consistent)

The VLA action-mode term AC_model has been removed; AC is now a pure
buffer-precedent signal: how close the candidate is to actions chosen
before in similar-context (similar robot state) situations.
"""
from __future__ import annotations

import numpy as np

from ..types import ActionChunk
from .information_gain import minmax_norm


def build_mode_set(action_chunks: list[ActionChunk]) -> list[ActionChunk]:
    """Mode set = the raw neighbour action chunks (identity, no clustering)."""
    return list(action_chunks)


def compute_mode_distance(
    candidate_action: ActionChunk,
    mode_set: list[ActionChunk],
) -> float:
    """D = min_{c ∈ mode_set} ‖vec(A_ξ) - c‖².

    Empty mode_set → ValueError. Caller (Selector) handles cold-start.
    """
    if not mode_set:
        raise ValueError("mode_set is empty; caller must handle cold-start")
    a = np.asarray(candidate_action, dtype=np.float64).reshape(-1)

    best = float("inf")
    for mode in mode_set:
        mv = np.asarray(mode, dtype=np.float64).reshape(-1)
        if mv.shape != a.shape:
            raise ValueError(
                f"action shape mismatch: candidate={a.shape}, mode={mv.shape}"
            )
        diff = a - mv
        d = float(np.dot(diff, diff))
        if d < best:
            best = d
    return best


def compute_ac(d_buffer_values: list[float], eps: float = 1e-8) -> list[float]:
    """AC = 1 - within-set min-max normalized buffer distance.

    Low D_buffer (close to a precedent) → high AC (consistent).
    """
    d_norm = minmax_norm(d_buffer_values, eps)
    return [1.0 - d for d in d_norm]
