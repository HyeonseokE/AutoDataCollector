"""Method 3 — external dependency Protocols."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import ActionChunk, BufferEntry, Context, FMOutput, SkillId


class PolicyAdapter(Protocol):
    """Pretrained VLA π₀ wrapper.

    forward_fm     → L_FM + hidden feature z + context embedding c_m
    sample_actions → action chunks ~ π₀(·|ctx), for AC_model mode set M_π₀(x_m)
    """

    def forward_fm(self, context: Context, action_chunk: ActionChunk) -> FMOutput: ...

    def forward_fm_batched(
        self, context: Context, action_chunks: list[ActionChunk],
    ) -> list[FMOutput]:
        """K-candidate-batched forward_fm.

        Folds K independent forward_fm calls into a single (K · N_b) forward
        to amortize GPU launch overhead. Returns one FMOutput per input chunk
        in the same order. Implementations may fall back to per-candidate
        calls when K == 1 or batched execution is not beneficial.
        """
        ...

    def sample_actions(self, context: Context, n_samples: int) -> list[ActionChunk]: ...


class BufferStore(Protocol):
    """Append-only B_t.

    query_skill        → B_t^(m), for IG buffer-side z-kNN
    nearest_by_context → N_k(x_m; B_t^(m)), for AC_buffer expert mode set M_B(x_m).
                         Distance metric is implementation-defined; the canonical
                         choice (decision #4(b)) is cosine on context_embedding.
    """

    def append(self, entry: BufferEntry) -> None: ...

    def query_skill(self, skill_id: SkillId) -> list[BufferEntry]: ...

    def nearest_by_context(
        self,
        context: Context,
        context_embedding: np.ndarray,
        skill_id: SkillId,
        k: int,
    ) -> list[BufferEntry]: ...
