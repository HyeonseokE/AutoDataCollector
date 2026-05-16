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
    """x_m = (O_m, S_m, I, m).

    ``key_embedding`` is the frozen-VLA key: the backbone's last feature (the
    VL joint feature feeding the action head, mean-pooled) plus, for families
    that do not fuse state into the backbone, the continuous proprioception.
    Used as the vector-DB retrieval key. None until a VLAKeyExtractor fills
    it; the FAISS buffer needs it for nearest_by_context and for storing an
    appended entry's key.
    """

    observation: Observation
    state: State
    instruction: Instruction
    skill_id: SkillId
    key_embedding: np.ndarray | None = None


@dataclass(frozen=True)
class Candidate:
    """One ξ_m^j ∈ Ξ_m. payload is an opaque pass-through for callers."""

    skill_id: SkillId
    action_chunk: ActionChunk
    payload: object | None = None


@dataclass(frozen=True)
class BufferEntry:
    """One past selected candidate stored in B_t.

    Buffer-only design: an entry is fully described by its context (whose
    ``state`` is the context-kNN key) and the executed ``action_chunk``
    (the feature used for both novelty and consistency distances). No VLA-
    derived latents are stored.
    """

    context: Context
    action_chunk: ActionChunk


@dataclass(frozen=True)
class ScoreReport:
    """Per-candidate raw + normalized + final scores (buffer-only)."""

    candidate_index: int
    novelty_raw: float       # N_B(ξ) — min L2 dist to skill buffer (action space)
    ig: float                # within-K min-max normalized novelty (higher = more novel)
    consistency_raw: float   # D_buffer(ξ) — min sq-dist to context-kNN neighbour actions
    ac: float                # 1 - MinMaxNorm(D_buffer) (higher = more consistent)
    score: float             # IG · AC


@dataclass(frozen=True)
class Selection:
    """Result of one selector call.

    argmax IG·AC, no thresholds → chosen_* are always non-None when
    candidates is non-empty.
    """

    chosen_index: int
    chosen_candidate: Candidate
    reports: list[ScoreReport]
