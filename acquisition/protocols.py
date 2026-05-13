"""Method 3 acquisition module — external dependency Protocols."""
from __future__ import annotations

from typing import Protocol

from .types import ActionChunk, BufferEntry, Context, FMOutput, SkillId


class PolicyAdapter(Protocol):
    """Pretrained VLA π₀ wrapper.

    Returns L_FM and the action-head hidden feature z in one forward.
    Random t and noise A_0 are handled internally by the implementation.
    """

    def forward(self, context: Context, action_chunk: ActionChunk) -> FMOutput: ...


class BufferStore(Protocol):
    """Append-only B_t.

    query_skill        → B_t^(m)
    nearest_by_context → N_k(x_m; B_t^(m)) for AC_buffer
    """

    def append(self, entry: BufferEntry) -> None: ...

    def query_skill(self, skill_id: SkillId) -> list[BufferEntry]: ...

    def nearest_by_context(
        self,
        context: Context,
        skill_id: SkillId,
        k: int,
    ) -> list[BufferEntry]: ...
