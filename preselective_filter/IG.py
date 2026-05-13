"""IG(ξ) = Ũ_π₀^α · Ñ_B^(m)^(1-α).

  U_π₀(ξ)    = single-step FM loss              (PolicyAdapter.forward_fm.l_fm)
  N_B^(m)(ξ) = min ‖z̄(ξ) - z̄(ξ_i)‖₂ over B_t^(m), z̄ = L2-normalized z
"""
from __future__ import annotations

import numpy as np

from .types import BufferEntry


def minmax_norm(values: list[float], eps: float = 1e-8) -> list[float]:
    """Within-set min-max normalization, eps-stabilized denominator.

    Spec:  Norm(Q^j) = (Q^j - min Q) / (max Q - min Q + eps)
    """
    arr = np.asarray(values, dtype=np.float64)
    lo = float(arr.min())
    rng = float(arr.max() - lo)
    return [float((v - lo) / (rng + eps)) for v in arr]


def compute_buffer_novelty(
    z: np.ndarray,
    buffer_entries: list[BufferEntry],
    eps: float = 1e-8,
) -> float:
    """N_B^(m)(ξ_m^j) = min ‖z̄(ξ) - z̄(ξ_i)‖₂ over B_t^(m).

    Empty buffer → ValueError. Caller (Selector) handles warm-start.
    """
    if not buffer_entries:
        raise ValueError("buffer_entries is empty; caller must handle warm-start")
    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    z_bar = z_arr / (float(np.linalg.norm(z_arr)) + eps)

    best = float("inf")
    for entry in buffer_entries:
        ez = np.asarray(entry.z, dtype=np.float64).reshape(-1)
        if ez.shape != z_bar.shape:
            raise ValueError(
                f"z shape mismatch: query={z_bar.shape}, entry={ez.shape}"
            )
        ez_bar = ez / (float(np.linalg.norm(ez)) + eps)
        d = float(np.linalg.norm(z_bar - ez_bar))
        if d < best:
            best = d
    return best


def compute_ig(
    u_pi0_values: list[float],
    n_buffer_values: list[float],
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> tuple[list[float], list[float], list[float]]:
    """Within-set min-max normalize → geometric mean.

    Returns (u_norm, n_norm, ig) per candidate.
    """
    if len(u_pi0_values) != len(n_buffer_values):
        raise ValueError(
            f"length mismatch: u={len(u_pi0_values)}, n={len(n_buffer_values)}"
        )
    u_norm = minmax_norm(u_pi0_values, eps)
    n_norm = minmax_norm(n_buffer_values, eps)
    ig = [
        float((u ** alpha) * (n ** (1.0 - alpha)))
        for u, n in zip(u_norm, n_norm)
    ]
    return u_norm, n_norm, ig
