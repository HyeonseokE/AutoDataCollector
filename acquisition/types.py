"""Method 3 acquisition module — data types.

Mirrors ours_method/구현내용정리.md.
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
    z    : action-head hidden feature      h_π₀^FM(O, S, I, A_ξ)
    """

    l_fm: float
    z: np.ndarray


@dataclass(frozen=True)
class BufferEntry:
    """One past selected candidate stored in B_t."""

    context: Context
    action_chunk: ActionChunk
    z: np.ndarray


@dataclass(frozen=True)
class ScoreReport:
    """Per-candidate raw + normalized + final scores."""

    candidate_index: int
    u_pi0_raw: float        # U_π₀(ξ)  (= L_FM)
    n_buffer_raw: float     # N_B^(m)(ξ)
    d_ac_buffer_raw: float  # D_buffer-AC^j
    u_pi0_norm: float       # Ũ_π₀
    n_buffer_norm: float    # Ñ_B
    ac_model: float         # 1 - MinMaxNorm(L_FM)
    ac_buffer: float        # 1 - MinMaxNorm(D_buffer-AC)
    ig: float               # Ũ^α · Ñ^(1-α)
    ac: float               # AC_model^λ · AC_buffer^(1-λ)
    score: float            # IG · AC


@dataclass(frozen=True)
class Selection:
    """Result of one selector call. chosen_* are None if all candidates fail thresholds."""

    chosen_index: int | None
    chosen_candidate: Candidate | None
    reports: list[ScoreReport]
