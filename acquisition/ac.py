"""AC(ξ) = AC_model^λ · AC_buffer^(1-λ).

  AC_model  = 1 - MinMaxNorm(L_FM^j)
  AC_buffer = 1 - MinMaxNorm(D_buffer-AC^j)
  D_buffer-AC^j = min ‖vec(A_ξ) - c_r‖² over M_B(x_m)
"""
from __future__ import annotations

from .types import ActionChunk


def compute_buffer_ac_distance(
    candidate_action: ActionChunk,
    expert_mode_set: list[ActionChunk],
) -> float:
    """D_buffer-AC^j. Empty expert set → caller decides warm-start semantics."""
    raise NotImplementedError


def compute_ac(
    l_fm_values: list[float],
    d_buffer_ac_values: list[float],
    lam: float = 0.5,
    eps: float = 1e-8,
) -> tuple[list[float], list[float], list[float]]:
    """Within-set min-max normalize, invert (low → high), then geometric mean.

    Returns (ac_model, ac_buffer, ac) per candidate.
    """
    raise NotImplementedError
