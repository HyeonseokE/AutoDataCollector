"""Method 3 — external dependency Protocols (buffer-only)."""
from __future__ import annotations

from typing import Protocol

from .types import BufferEntry, Context, SkillId


class BufferStore(Protocol):
    """Append-only B_t.

    query_skill        → B_t^(m), the full per-skill buffer (IG novelty).
    nearest_by_context → N_k(x_m; B_t^(m)), context-kNN neighbour set used as
                         the expert action mode set for AC_buffer. The context
                         key is the robot ``state`` carried by ``context``.
    """

    def append(self, entry: BufferEntry) -> None: ...

    def query_skill(self, skill_id: SkillId) -> list[BufferEntry]: ...

    def nearest_by_context(
        self,
        context: Context,
        skill_id: SkillId,
        k: int,
    ) -> list[BufferEntry]: ...
