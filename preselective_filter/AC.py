"""AC(ξ) = AC_model^λ · AC_buffer^(1-λ).

  AC_model  = 1 - MinMaxNorm(D_model-AC^j)
  AC_buffer = 1 - MinMaxNorm(D_buffer-AC^j)

  D_*-AC^j = min ‖vec(A_ξ) - mode_r‖² over the respective mode set
    model  : π₀ sampled actions → M_π₀(x_m)
    buffer : context-kNN neighbor actions → M_B(x_m)
"""
from __future__ import annotations

import numpy as np

from .IG import minmax_norm
from .types import ActionChunk


def build_mode_set(
    action_chunks: list[ActionChunk],
    max_modes: int | None = None,
) -> list[ActionChunk]:
    """Construct mode set from action samples.

    max_modes=None → identity (raw samples as modes).
    Integer max_modes → clustering (not in v1: requires scipy/sklearn,
    see LIMITATIONS.md).
    """
    if max_modes is None:
        return list(action_chunks)
    raise NotImplementedError(
        "Clustering-based mode reduction requires scipy/sklearn and is "
        "intentionally out of v1 (module independence). "
        "Pass max_modes=None for identity mode set."
    )


def compute_mode_distance(
    candidate_action: ActionChunk,
    mode_set: list[ActionChunk],
) -> float:
    """D = min_{c ∈ mode_set} ‖vec(A_ξ) - c‖².

    Empty mode_set → ValueError. Caller (Selector) handles warm-start.
    """
    if not mode_set:
        raise ValueError("mode_set is empty; caller must handle warm-start")
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


def compute_ac(
    d_model_ac_values: list[float],
    d_buffer_ac_values: list[float],
    lam: float = 0.5,
    eps: float = 1e-8,
) -> tuple[list[float], list[float], list[float]]:
    """Within-set min-max normalize, invert (low D → high AC), geometric mean.

    Returns (ac_model, ac_buffer, ac) per candidate.
    """
    if len(d_model_ac_values) != len(d_buffer_ac_values):
        raise ValueError(
            f"length mismatch: model={len(d_model_ac_values)}, "
            f"buffer={len(d_buffer_ac_values)}"
        )
    d_model_norm = minmax_norm(d_model_ac_values, eps)
    d_buffer_norm = minmax_norm(d_buffer_ac_values, eps)
    ac_model = [1.0 - d for d in d_model_norm]
    ac_buffer = [1.0 - d for d in d_buffer_norm]
    ac = [
        float((m ** lam) * (b ** (1.0 - lam)))
        for m, b in zip(ac_model, ac_buffer)
    ]
    return ac_model, ac_buffer, ac
