"""Ingest recorded demonstrations into the FAISS vector DB.

Builds the episodic-memory vector DB from real demonstrations. For every
timestep t of a (successful, forward-phase) episode it stores one entry:

    key_t   = frozen-VLA backbone embedding of the context at t
              (image O_t + instruction I + state S_t)
    value_t = the raw demonstrated action chunk A_{t:t+H-1}

Chunking is delegated to LeRobot's own ``delta_timestamps`` mechanism — the
exact code path used to build a pi0/pi05/smolvla training dataset — so the
(S_t -> A_{t:t+H-1}) pairs are identical to training by construction:
end-of-episode tails are last-frame-held and flagged in ``action_is_pad``.

Two entry points share this one ingestion path:
  - online  : the recording pipeline calls ``ingest_episode`` on a forward
              episode the moment its task judge returns TRUE.
  - offline : the ``__main__`` CLI ingests every forward episode of a given
              dataset — for debugging / verifying DB construction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from preselective_filter.types import BufferEntry, Context
from preselective_filter.vectorDB.faiss_buffer import FaissBufferStore
from preselective_filter.vectorDB.vla_embedding import (
    VLAKeyExtractor,
    make_vla_key_extractor,
)

# v1 buffer bucket — must match the skill_id the live IG·AC selector queries
# with, otherwise retrieval finds nothing. Single bucket for now.
DEFAULT_SKILL_ID = "move_to"


def _episode_bounds(dataset, ep_idx: int) -> tuple[int, int]:
    ep = dataset.meta.episodes[ep_idx]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def all_episode_indices(dataset) -> list[int]:
    """Episode indices of a dataset (``dataset.episodes`` is None = 'all')."""
    eps = getattr(dataset, "episodes", None)
    if eps is not None:
        return list(eps)
    return list(range(int(dataset.meta.total_episodes)))


def _frame_images(item: dict) -> dict[str, np.ndarray]:
    """Pull observation.images.* frames from a dataset item as numpy arrays."""
    out: dict[str, np.ndarray] = {}
    for k, v in item.items():
        if k.startswith("observation.images"):
            out[k] = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    return out


def _frame_instruction(item: dict, dataset, ep_idx: int, fallback: str) -> str:
    task = item.get("task")
    if isinstance(task, str) and task:
        return task
    tasks = dataset.meta.episodes[ep_idx].get("tasks")
    if isinstance(tasks, (list, tuple)) and tasks:
        return str(tasks[0])
    return fallback


def ingest_frame(
    extractor: VLAKeyExtractor,
    buffer: FaissBufferStore,
    *,
    images: dict,
    instruction: str,
    state,
    action_chunk,
    skill_id: str = DEFAULT_SKILL_ID,
) -> None:
    """Encode one demo frame and append its (key, value) buffer entry.

    key   = extractor.encode(images, instruction, state)  — backbone feature
    value = action_chunk (raw demonstrated A_{t:t+H-1})

    Shared by the offline/local ``ingest_episode`` loop and the gRPC server's
    per-streamed-frame ``IngestEpisode`` handler.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    key = extractor.encode(images, instruction, state)
    ctx = Context(
        observation=np.zeros(0, dtype=np.float32),
        state=state,
        instruction=instruction,
        skill_id=skill_id,
        key_embedding=key,
    )
    buffer.append(BufferEntry(context=ctx, action_chunk=action_chunk))


def ingest_episode(
    dataset,
    ep_idx: int,
    extractor: VLAKeyExtractor,
    buffer: FaissBufferStore,
    *,
    skill_id: str = DEFAULT_SKILL_ID,
    instruction_fallback: str = "",
    debug_verbose: bool = False,
) -> int:
    """Ingest one episode: append (key_t, value_t) for every timestep t.

    ``dataset`` must have been built with action ``delta_timestamps`` so each
    item's ``action`` is already the (H, action_dim) chunk. Returns the number
    of frames appended.
    """
    ep_start, ep_end = _episode_bounds(dataset, ep_idx)
    appended = 0
    for global_idx in range(ep_start, ep_end):
        item = dataset[global_idx]
        instruction = _frame_instruction(item, dataset, ep_idx, instruction_fallback)
        ingest_frame(
            extractor, buffer,
            images=_frame_images(item),
            instruction=instruction,
            state=item["observation.state"],
            action_chunk=item["action"],
            skill_id=skill_id,
        )
        appended += 1

    if debug_verbose:
        print(
            f"[demo_ingest] episode {ep_idx}: appended {appended} frames "
            f"(skill_id={skill_id})"
        )
    return appended


def open_chunked_dataset(dataset_root: str | Path, chunk_size: int):
    """Open a recorded LeRobot dataset with action delta_timestamps set, so
    every item's ``action`` is the training-identical (H, action_dim) chunk.
    """
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    root = Path(dataset_root)
    repo_id = root.name
    fps = LeRobotDatasetMetadata(repo_id, root=root).fps
    delta = {"action": [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta)


def ingest_dataset(
    dataset_root: str | Path,
    extractor: VLAKeyExtractor,
    buffer: FaissBufferStore,
    *,
    chunk_size: int,
    episode_indices: list[int] | None = None,
    skill_id: str = DEFAULT_SKILL_ID,
    debug_verbose: bool = False,
) -> dict[int, int]:
    """Ingest every (or selected) episode of a recorded dataset.

    Returns {episode_index: frames_appended}.
    """
    dataset = open_chunked_dataset(dataset_root, chunk_size)
    eps = episode_indices if episode_indices is not None else all_episode_indices(dataset)
    print(f"[demo_ingest] dataset={dataset_root}  episodes={len(eps)}  H={chunk_size}")
    result: dict[int, int] = {}
    for ep_idx in eps:
        result[ep_idx] = ingest_episode(
            dataset, ep_idx, extractor, buffer,
            skill_id=skill_id, debug_verbose=debug_verbose,
        )
    total = sum(result.values())
    print(f"[demo_ingest] done — {total} entries from {len(eps)} episodes")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demo_ingest",
        description="Ingest recorded demonstrations into a FAISS vector DB "
                    "(offline / debug).",
    )
    parser.add_argument("dataset", help="path to a recorded LeRobot dataset root")
    parser.add_argument(
        "--checkpoint", required=True,
        help="frozen VLA checkpoint (HF id or local path) for key embedding",
    )
    parser.add_argument(
        "--buffer-root", required=True,
        help="FAISS buffer root to write the vector DB into",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="action chunk size H (must match the policy's chunk_size)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skill-id", default=DEFAULT_SKILL_ID)
    parser.add_argument("--family", default=None,
                        help="override VLA family dispatch (smolvla|pi0|pi05|groot)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="ingest only the first N episodes (quick check)")
    args = parser.parse_args(argv)

    extractor = make_vla_key_extractor(
        checkpoint=args.checkpoint, device=args.device,
        family=args.family, debug_verbose=True,
    )
    buffer = FaissBufferStore(root=Path(args.buffer_root), debug_verbose=True)
    try:
        episode_indices = None
        if args.max_episodes is not None:
            ds_probe = open_chunked_dataset(args.dataset, args.chunk_size)
            episode_indices = all_episode_indices(ds_probe)[: args.max_episodes]
        ingest_dataset(
            args.dataset, extractor, buffer,
            chunk_size=args.chunk_size, skill_id=args.skill_id,
            episode_indices=episode_indices, debug_verbose=True,
        )
    finally:
        extractor.close()

    totals = buffer.summary()
    print(f"[demo_ingest] buffer summary: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
