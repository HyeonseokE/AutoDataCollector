"""Schedule iterators for the multi-episode pipeline.

Folder name (``episode_NN``) corresponds to **execution / save order**
in BOTH modes:

  seed_major   ep_NN = batch_index × episodes_per_seed + slot + 1
               (sequential by batch then slot — matches execution order)
  round_robin  ep_NN = execution_idx + 1
               (sequential by execution order — `ls` matches run order)

The (seed, slot) tuple for each folder is recovered from its
``batch_info.json`` (``batch_seed_index``, ``slot``) — folder name alone
no longer encodes it in round_robin. cleanup_dataset_for_resume only
needs a sorted-by-name traversal (= save order) regardless of mode.

seed_major (legacy):
    seed 0 slot 0, seed 0 slot 1, ..., seed 0 slot K-1,
    seed 1 slot 0, seed 1 slot 1, ..., seed 1 slot K-1, ...
    → ep_01, ep_02, ..., ep_NK  (sequential)

round_robin (balanced, default):
    seed 0 slot 0, seed 1 slot 0, ..., seed N-1 slot 0,
    seed 0 slot 1, seed 1 slot 1, ..., seed N-1 slot 1, ...
    → ep_01, ep_02, ..., ep_NK  (sequential in execution order)

Each yield is ``(execution_idx, batch_index, slot, episode_num, is_round_last)``:

  execution_idx  — 0-based step in execution order
  batch_index    — 0-based seed index (from schedule)
  slot           — 0-based within-seed slot (from schedule)
  episode_num    — 1-based folder NN = execution_idx + 1 (in both modes)
  is_round_last  — True at "round" boundary: seed_major = last slot of a seed,
                   round_robin = last seed of a round. Hook checks fire here.
"""

from __future__ import annotations

from typing import Iterator


def schedule_iter(
    num_episodes: int,
    episodes_per_seed: int,
    num_seeds: int,
    mode: str,
) -> Iterator[tuple[int, int, int, int, bool]]:
    """Yield (execution_idx, batch_index, slot, episode_num, is_round_last).

    ``num_episodes`` is the cap; iteration stops when this many episodes
    have been yielded (handles ragged cases where num_episodes < seeds×eps).
    """
    if mode not in ("seed_major", "round_robin"):
        raise ValueError(f"unknown schedule_mode: {mode!r}")
    if num_seeds <= 0 or episodes_per_seed <= 0:
        return

    yielded = 0
    if mode == "seed_major":
        for batch_index in range(num_seeds):
            for slot in range(episodes_per_seed):
                if yielded >= num_episodes:
                    return
                episode_num = batch_index * episodes_per_seed + slot + 1
                is_round_last = (slot == episodes_per_seed - 1)
                yield yielded, batch_index, slot, episode_num, is_round_last
                yielded += 1
    else:  # round_robin
        for slot in range(episodes_per_seed):
            for batch_index in range(num_seeds):
                if yielded >= num_episodes:
                    return
                # Folder name = execution order so `ls` matches run order
                # AND extending episodes_per_seed later just appends ep_(K+1)
                # without renaming any existing folder.
                episode_num = yielded + 1
                is_round_last = (batch_index == num_seeds - 1)
                yield yielded, batch_index, slot, episode_num, is_round_last
                yielded += 1


def save_order_key(batch_index: int, slot: int, mode: str) -> tuple[int, int]:
    """[deprecated] Sort key recovering save order from (batch_index, slot).

    No longer needed — folder name (``episode_NN``) now equals execution /
    save order in BOTH schedule modes, so sorted-name traversal is correct
    everywhere. Kept for backward-compat with callers from when round_robin
    used the seed_major folder formula.
    """
    if mode == "round_robin":
        return (slot, batch_index)
    return (batch_index, slot)
