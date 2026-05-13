"""Method 3 — Skill-wise Pre-selective Acquisition: data types.

Spec: 구현내용정리_modelAC_updated_nuance_final.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

SkillId: TypeAlias = int | str
Observation: TypeAlias = np.ndarray
State: TypeAlias = np.ndarray
Instruction: TypeAlias = str
ActionChunk: TypeAlias = np.ndarray  # (H, action_dim)


@dataclass(frozen=True)
class Context:
    """x_m = (O_m, S_m, I, m)."""

    observation: Observation
    state: State
    instruction: Instruction
    skill_id: SkillId


@dataclass(frozen=True)
class Candidate:
    """One ξ_m^j ∈ Ξ_m. payload is an opaque pass-through for callers."""

    skill_id: SkillId
    action_chunk: ActionChunk
    payload: object | None = None


@dataclass(frozen=True)
class FMOutput:
    """One π₀ forward result.

    l_fm : single-step flow-matching loss  ‖v_π₀(A_t, t, c_m) - u_t‖²
           (Diff-DAgger style: MC-averaged over N_b (ε, t) samples; adapter detail)
    z    : action-head hidden feature      h_π₀^FM(O, S, I, A_ξ)   — for IG buffer-side kNN
    c_m  : context conditioning            [h_VL(O,I), h_prop(S)]   — for AC_buffer cosine kNN
    """

    l_fm: float
    z: np.ndarray
    c_m: np.ndarray


@dataclass(frozen=True)
class BufferEntry:
    """One past selected candidate stored in B_t."""

    context: Context
    action_chunk: ActionChunk
    z: np.ndarray
    context_embedding: np.ndarray


@dataclass(frozen=True)
class ScoreReport:
    """Per-candidate raw + normalized + final scores."""

    candidate_index: int
    # IG raws
    u_pi0_raw: float        # U_π₀(ξ)  (L_FM — pure novelty signal)
    n_buffer_raw: float     # N_B^(m)(ξ) — z-space kNN distance in B_t^(m)
    # AC raws (both: candidate ↔ nearest mode in respective mode set)
    d_model_ac_raw: float   # min ‖vec(A_ξ) - p_r‖² over M_π₀(x_m)
    d_buffer_ac_raw: float  # min ‖vec(A_ξ) - c_r‖² over M_B(x_m)
    # IG normalized
    u_pi0_norm: float       # Ũ_π₀
    n_buffer_norm: float    # Ñ_B
    # AC normalized (inverted: high score = action-consistent)
    ac_model: float         # 1 - MinMaxNorm(D_model-AC)
    ac_buffer: float        # 1 - MinMaxNorm(D_buffer-AC)
    # Finals
    ig: float               # Ũ^α · Ñ^(1-α)
    ac: float               # AC_model^λ · AC_buffer^(1-λ)
    score: float            # IG · AC


@dataclass(frozen=True)
class Selection:
    """Result of one selector call.

    argmax IG·AC, no thresholds → chosen_* are always non-None when
    candidates is non-empty. chosen_z / chosen_context_embedding are
    captured during select() so the caller can pass them to add_to_buffer()
    without re-running π₀ forward.
    """

    chosen_index: int
    chosen_candidate: Candidate
    chosen_z: np.ndarray
    chosen_context_embedding: np.ndarray
    reports: list[ScoreReport]
