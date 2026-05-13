"""IG(ξ) = Ũ_π₀^α · Ñ_B^(m)^(1-α).

  U_π₀(ξ)    = single-step FM loss              (PolicyAdapter.forward.l_fm)
  N_B^(m)(ξ) = min ‖z̄(ξ) - z̄(ξ_i)‖₂ over B_t^(m)
"""
from __future__ import annotations

import numpy as np

from .types import BufferEntry


def compute_buffer_novelty(z: np.ndarray, buffer_entries: list[BufferEntry]) -> float:
    """N_B^(m)(ξ_m^j). Buffer empty → caller decides warm-start semantics."""
    raise NotImplementedError


def compute_ig(
    u_pi0_values: list[float],
    n_buffer_values: list[float],
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> tuple[list[float], list[float], list[float]]:
    """Within-set min-max normalize → geometric mean.

    Returns (u_norm, n_norm, ig) per candidate.
    """
    raise NotImplementedError
