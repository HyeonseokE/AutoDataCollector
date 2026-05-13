"""JSONL-backed BufferStore — one file per skill (decision #3 + #4(b)).

Layout:
    <root>/
    ├── move_to.jsonl
    ├── grasp.jsonl
    └── ...

Each line in <skill_id>.jsonl is a JSON object:
    {
        "skill_id": "move_to",
        "instruction": "...",
        "state": [base64 float32],
        "observation_meta": {...},
        "action_chunk": [base64 float32, shape: [H, action_dim]],
        "z":            [base64 float32, shape: [D_z]],
        "context_embedding": [base64 float32, shape: [D_ctx]],
    }

nearest_by_context uses cosine distance on stored context_embedding
(decision #4(b)). Observation images are NOT stored verbatim in the jsonl
(too large) — only their derived c_m is retained.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from preselective_filter.types import (
    ActionChunk,
    BufferEntry,
    Context,
    SkillId,
)


def _b64encode(arr: np.ndarray) -> str:
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def _b64decode(s: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = base64.b64decode(s.encode("ascii"))
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float64)


def _serialize_entry(entry: BufferEntry) -> dict:
    ctx = entry.context
    return {
        "skill_id": ctx.skill_id,
        "instruction": ctx.instruction,
        "state": _b64encode(ctx.state),
        "state_shape": list(np.asarray(ctx.state).shape),
        "action_chunk": _b64encode(entry.action_chunk),
        "action_chunk_shape": list(np.asarray(entry.action_chunk).shape),
        "z": _b64encode(entry.z),
        "z_shape": list(np.asarray(entry.z).shape),
        "context_embedding": _b64encode(entry.context_embedding),
        "context_embedding_shape": list(np.asarray(entry.context_embedding).shape),
    }


def _deserialize_entry(obj: dict) -> BufferEntry:
    state = _b64decode(obj["state"], tuple(obj["state_shape"]))
    action_chunk = _b64decode(obj["action_chunk"], tuple(obj["action_chunk_shape"]))
    z = _b64decode(obj["z"], tuple(obj["z_shape"]))
    context_embedding = _b64decode(
        obj["context_embedding"], tuple(obj["context_embedding_shape"]),
    )
    # Observation is not persisted (too large); use empty placeholder.
    # The selector never reads context.observation for IG/AC; only metadata
    # such as skill_id and instruction matter post-commit.
    ctx = Context(
        observation=np.zeros(0, dtype=np.float32),
        state=state,
        instruction=obj["instruction"],
        skill_id=obj["skill_id"],
    )
    return BufferEntry(
        context=ctx,
        action_chunk=action_chunk,
        z=z,
        context_embedding=context_embedding,
    )


class JsonlBufferStore:
    """BufferStore implementation backed by per-skill jsonl files."""

    def __init__(self, root: str | Path, debug_verbose: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.debug_verbose = debug_verbose

    def _path(self, skill_id: SkillId) -> Path:
        # skill_id may contain characters unsafe for filenames; sanitize lightly.
        safe = str(skill_id).replace("/", "_").replace(" ", "_")
        return self.root / f"{safe}.jsonl"

    def _count_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("rb") as f:
            return sum(1 for _ in f)

    def append(self, entry: BufferEntry) -> None:
        path = self._path(entry.context.skill_id)
        line = json.dumps(_serialize_entry(entry), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.debug_verbose:
            total = self._count_lines(path)
            print(
                f"[preselective_filter]   buffer.append skill={entry.context.skill_id} "
                f"→ {path.name} (now {total} entries)"
            )

    def query_skill(self, skill_id: SkillId) -> list[BufferEntry]:
        path = self._path(skill_id)
        if not path.exists():
            if self.debug_verbose:
                print(
                    f"[preselective_filter]   buffer.query_skill({skill_id}) → 0 (no file)"
                )
            return []
        out: list[BufferEntry] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(_deserialize_entry(json.loads(line)))
        if self.debug_verbose:
            print(
                f"[preselective_filter]   buffer.query_skill({skill_id}) → {len(out)} entries"
            )
        return out

    def nearest_by_context(
        self,
        context: Context,
        context_embedding: np.ndarray,
        skill_id: SkillId,
        k: int,
    ) -> list[BufferEntry]:
        """Top-k entries by cosine distance on context_embedding."""
        entries = self.query_skill(skill_id)
        if not entries:
            return []

        q = np.asarray(context_embedding, dtype=np.float64).reshape(-1)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        # Stack stored embeddings, normalize, compute cosine distance.
        E = np.stack(
            [np.asarray(e.context_embedding, dtype=np.float64).reshape(-1)
             for e in entries],
            axis=0,
        )
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        cos_sim = E_norm @ q_norm           # (N,)
        cos_dist = 1.0 - cos_sim            # smaller = closer

        # argpartition for top-k smallest distances
        n = len(entries)
        top_k = min(k, n)
        idx = np.argpartition(cos_dist, top_k - 1)[:top_k]
        # Sort the top_k indices by distance ascending
        idx = idx[np.argsort(cos_dist[idx])]
        result = [entries[i] for i in idx]
        if self.debug_verbose:
            top_d = float(cos_dist[idx[0]])
            print(
                f"[preselective_filter]   buffer.nearest_by_context(skill={skill_id}, k={k}) "
                f"→ {len(result)} entries (top_cosine_dist={top_d:.4f})"
            )
        return result
