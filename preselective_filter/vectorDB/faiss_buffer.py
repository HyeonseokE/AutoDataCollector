"""FAISS-backed BufferStore — vector DB for retrieval-augmented selection.

Each skill gets:
  - a durable jsonl shard  <root>/<skill_id>.jsonl   (source of truth)
  - an in-memory FAISS IndexFlatL2 over the key embeddings, rebuilt from the
    jsonl on first access (so the index file never goes out of sync).

Key   = frozen-VLA joint embedding (Context.key_embedding): mean-pooled
        embed_prefix output = VL feature + proprioception.
Value = the executed action chunk (BufferEntry.action_chunk).

jsonl line schema:
    {
        "skill_id": "move_to",
        "instruction": "...",
        "state": [base64 float32], "state_shape": [...],
        "action_chunk": [base64 float32], "action_chunk_shape": [...],
        "key_embedding": [base64 float32], "key_embedding_shape": [...],
    }

IndexFlatL2 (exact L2) is used — the per-skill buffer is well under the
million-entry scale where ANN (HNSW) would be needed.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from preselective_filter.types import BufferEntry, Context, SkillId


def _b64encode(arr: np.ndarray) -> str:
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def _b64decode(s: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = base64.b64decode(s.encode("ascii"))
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float64)


def _serialize_entry(entry: BufferEntry) -> dict:
    ctx = entry.context
    if ctx.key_embedding is None:
        raise ValueError(
            "BufferEntry.context.key_embedding is None — a VLAKeyExtractor must "
            "fill it before the entry can be stored in the FAISS buffer."
        )
    return {
        "skill_id": ctx.skill_id,
        "instruction": ctx.instruction,
        "state": _b64encode(ctx.state),
        "state_shape": list(np.asarray(ctx.state).shape),
        "action_chunk": _b64encode(entry.action_chunk),
        "action_chunk_shape": list(np.asarray(entry.action_chunk).shape),
        "key_embedding": _b64encode(ctx.key_embedding),
        "key_embedding_shape": list(np.asarray(ctx.key_embedding).shape),
    }


def _deserialize_entry(obj: dict) -> BufferEntry:
    state = _b64decode(obj["state"], tuple(obj["state_shape"]))
    action_chunk = _b64decode(obj["action_chunk"], tuple(obj["action_chunk_shape"]))
    key_embedding = _b64decode(
        obj["key_embedding"], tuple(obj["key_embedding_shape"]),
    )
    ctx = Context(
        observation=np.zeros(0, dtype=np.float32),
        state=state,
        instruction=obj["instruction"],
        skill_id=obj["skill_id"],
        key_embedding=key_embedding,
    )
    return BufferEntry(context=ctx, action_chunk=action_chunk)


class _SkillShard:
    """In-memory FAISS index + entry list for one skill, backed by a jsonl."""

    def __init__(self, path: Path) -> None:
        import faiss

        self._faiss = faiss
        self.path = path
        self.entries: list[BufferEntry] = []
        self.index = None          # faiss.IndexFlatL2, created lazily on first vec
        self._dim: int | None = None
        self._load()

    def _new_index(self, dim: int):
        self._dim = dim
        self.index = self._faiss.IndexFlatL2(dim)

    def _add_vec(self, emb: np.ndarray) -> None:
        v = np.ascontiguousarray(emb, dtype=np.float32).reshape(1, -1)
        if self.index is None:
            self._new_index(int(v.shape[1]))
        if v.shape[1] != self._dim:
            raise ValueError(
                f"key embedding dim {v.shape[1]} != index dim {self._dim}"
            )
        self.index.add(v)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = _deserialize_entry(json.loads(line))
                self.entries.append(entry)
                self._add_vec(entry.context.key_embedding)

    def append(self, entry: BufferEntry) -> None:
        line = json.dumps(_serialize_entry(entry), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.entries.append(entry)
        self._add_vec(entry.context.key_embedding)

    def search(self, query: np.ndarray, k: int) -> list[BufferEntry]:
        if self.index is None or not self.entries:
            return []
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self._dim:
            return []
        top_k = min(k, len(self.entries))
        _dist, idx = self.index.search(q, top_k)
        return [self.entries[i] for i in idx[0] if 0 <= i < len(self.entries)]

    def count(self) -> int:
        return len(self.entries)


class FaissBufferStore:
    """BufferStore backed by per-skill jsonl shards + in-memory FAISS indices."""

    def __init__(self, root: str | Path, debug_verbose: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.debug_verbose = debug_verbose
        self._shards: dict[str, _SkillShard] = {}

    def _safe(self, skill_id: SkillId) -> str:
        return str(skill_id).replace("/", "_").replace(" ", "_")

    def _shard(self, skill_id: SkillId) -> _SkillShard:
        safe = self._safe(skill_id)
        shard = self._shards.get(safe)
        if shard is None:
            shard = _SkillShard(self.root / f"{safe}.jsonl")
            self._shards[safe] = shard
        return shard

    def summary(self) -> dict[str, int]:
        """{skill_id: entry_count} over every jsonl shard on disk."""
        out: dict[str, int] = {}
        if not self.root.exists():
            return out
        for p in sorted(self.root.glob("*.jsonl")):
            out[p.stem] = self._shard(p.stem).count()
        return out

    def append(self, entry: BufferEntry) -> None:
        shard = self._shard(entry.context.skill_id)
        shard.append(entry)
        if self.debug_verbose:
            print(
                f"[preselective_filter]   faiss.append skill={entry.context.skill_id} "
                f"→ {shard.path.name} (now {len(shard.entries)} entries)"
            )

    def query_skill(self, skill_id: SkillId) -> list[BufferEntry]:
        entries = list(self._shard(skill_id).entries)
        if self.debug_verbose:
            print(
                f"[preselective_filter]   faiss.query_skill({skill_id}) "
                f"→ {len(entries)} entries"
            )
        return entries

    def nearest_by_context(
        self,
        context: Context,
        skill_id: SkillId,
        k: int,
    ) -> list[BufferEntry]:
        """Top-k entries by FAISS L2 search on Context.key_embedding."""
        if context.key_embedding is None:
            if self.debug_verbose:
                print(
                    "[preselective_filter]   faiss.nearest_by_context: "
                    "context.key_embedding is None → []"
                )
            return []
        result = self._shard(skill_id).search(context.key_embedding, k)
        if self.debug_verbose:
            print(
                f"[preselective_filter]   faiss.nearest_by_context(skill={skill_id}, "
                f"k={k}) → {len(result)} entries"
            )
        return result
