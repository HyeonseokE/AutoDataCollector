#!/usr/bin/env python3
"""Recover a contaminated subgoal_buffer.npz from a partially-migrated session.

Background
----------
A session collected at ``episodes_per_seed=N1`` then extended to ``N2`` via
folder rename ONLY (no buffer migration) leaves the buffer with OLD continuous
episode IDs that don't match the new folder layout. If a resume then ran on
top of this, the buffer now contains a MIX of:

  * Run A entries  — OLD-layout (continuous 1..N1) IDs, appended first.
  * Today's resume — NEW-layout (slot-based) IDs, appended after.

The two portions can overlap numerically (e.g., both produce ``episode_04``,
but they refer to different physical episodes).

This script recovers by:

  1. Detecting the boundary between Run A and today's portion per skill via
     the first monotonicity break in the episode_id sequence (Run A and the
     resume each append in ascending episode order; the boundary is where
     the running max drops back).
  2. Applying the OLD→NEW mapping to the Run A portion only.
  3. Leaving today's portion unchanged.

Conditions for valid recovery
-----------------------------
  * Run A appended in ascending episode order      (`run_multiple_episodes`).
  * Today's resume appended in ascending order     (`resume_multiple_episodes`).
  * No interleaving — sessions ran sequentially.

If these hold, recovery is exact for every entry. If a skill shows multiple
drops (retries within a session) the script SKIPS that skill and warns.

Default mode is dry-run with a per-skill diff preview. Pass ``--apply`` to
commit; the previous file is copied to ``subgoal_buffer.npz.before-recover``.

Usage
-----
    python scripts/recover_subgoal_buffer.py <session_dir> \\
        --old-eps-per-seed 3 --new-eps-per-seed 5 --num-seeds 10
    # add --apply to write.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


def parse_episode_int(s) -> int:
    return int(str(s).split("_")[-1])


def build_mapping(old_eps_per_seed: int, new_eps_per_seed: int, num_seeds: int) -> dict[str, str]:
    """Build old-continuous → new-slot-based ``episode_id`` mapping.

    For each (seed, slot) the canonical truth, the new episode number is
    ``seed * new_eps_per_seed + slot + 1`` while the old was
    ``seed * old_eps_per_seed + slot + 1``.
    """
    if new_eps_per_seed < old_eps_per_seed:
        raise ValueError("new_eps_per_seed must be ≥ old_eps_per_seed")
    mapping: dict[str, str] = {}
    for seed in range(num_seeds):
        for slot in range(old_eps_per_seed):
            old = seed * old_eps_per_seed + slot + 1
            new = seed * new_eps_per_seed + slot + 1
            mapping[f"episode_{old:02d}"] = f"episode_{new:02d}"
    return mapping


def detect_all_drops(ids: list[int]) -> list[int]:
    """Return every index where the running max drops."""
    drops: list[int] = []
    max_seen = -1
    for i, x in enumerate(ids):
        if x < max_seen:
            drops.append(i)
        if x > max_seen:
            max_seen = x
    return drops


def recover_skill_array(
    eps_arr: np.ndarray,
    mapping: dict[str, str],
    old_layout_max: int,
) -> tuple[np.ndarray, dict]:
    """Recover a single ``<skill>::episode`` array.

    Returns ``(new_arr, diag_dict)``. ``diag_dict`` carries enough info to
    print a clear per-skill audit line.
    """
    ids = [parse_episode_int(e) for e in eps_arr]
    N = len(ids)
    drops = detect_all_drops(ids)

    diag = {
        "n_entries": N,
        "drops": drops,
        "boundary": None,
        "status": "ok",
        "rewritten": 0,
        "kept": 0,
        "unmapped": 0,
        "run_a_max": None,
        "today_max": None,
    }
    if N == 0:
        diag["status"] = "empty"
        return eps_arr.copy(), diag

    # ── case A: no drop  ── all monotonic. Either uncontaminated OR overlap-
    # free (Run A ≤ old_layout_max followed by today > old_layout_max). Decide
    # per-entry by value range against old_layout_max.
    if not drops:
        new: list[str] = []
        for e in eps_arr:
            s = str(e)
            if parse_episode_int(s) <= old_layout_max and s in mapping:
                tgt = mapping[s]
                new.append(tgt)
                if tgt != s:
                    diag["rewritten"] += 1
                else:
                    diag["kept"] += 1
            else:
                new.append(s)
                diag["kept"] += 1
        diag["status"] = "no_drop_value_based"
        return np.array(new, dtype=eps_arr.dtype), diag

    # ── case B: multiple drops  ── suspicious (retries / multiple sessions).
    # Refuse to guess.
    if len(drops) > 1:
        diag["status"] = "multiple_drops_suspicious"
        return eps_arr.copy(), diag

    # ── case C: exactly one drop  ── clean Run A | today boundary.
    B = drops[0]
    diag["boundary"] = B
    diag["run_a_max"] = max(ids[:B])
    diag["today_max"] = max(ids[B:])

    # sanity: Run A's max must fit in the old layout space
    if diag["run_a_max"] > old_layout_max:
        diag["status"] = (
            f"pre_boundary_max_{diag['run_a_max']}_gt_old_layout_max_{old_layout_max}"
        )
        return eps_arr.copy(), diag

    new = []
    for i, e in enumerate(eps_arr):
        s = str(e)
        if i < B:
            tgt = mapping.get(s)
            if tgt is None:
                new.append(s)
                diag["unmapped"] += 1
            else:
                new.append(tgt)
                if tgt != s:
                    diag["rewritten"] += 1
                else:
                    diag["kept"] += 1
        else:
            new.append(s)
            diag["kept"] += 1
    return np.array(new, dtype=eps_arr.dtype), diag


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session_dir", type=Path,
                    help="session folder containing subgoal_buffer.npz")
    ap.add_argument("--old-eps-per-seed", type=int, required=True)
    ap.add_argument("--new-eps-per-seed", type=int, required=True)
    ap.add_argument("--num-seeds", type=int, required=True)
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry-run with diff preview)")
    args = ap.parse_args()

    buf_path = args.session_dir / "subgoal_buffer.npz"
    if not buf_path.exists():
        sys.exit(f"buffer not found: {buf_path}")

    old_layout_max = args.num_seeds * args.old_eps_per_seed
    new_layout_max = args.num_seeds * args.new_eps_per_seed
    mapping = build_mapping(args.old_eps_per_seed, args.new_eps_per_seed, args.num_seeds)

    print(f"session : {args.session_dir}")
    print(f"layout  : {args.old_eps_per_seed}×{args.num_seeds} = {old_layout_max}  "
          f"→  {args.new_eps_per_seed}×{args.num_seeds} = {new_layout_max}")
    print(f"mapping : {sum(1 for k,v in mapping.items() if k!=v)} non-identity pairs")
    sample_changing = [(k, v) for k, v in sorted(mapping.items()) if k != v][:6]
    print(f"          e.g. {sample_changing}")
    print()

    z = dict(np.load(buf_path, allow_pickle=False))
    skills = sorted({k.split("::", 1)[0] for k in z.keys() if "::" in k})

    new_z = dict(z)
    tot = {"rewritten": 0, "kept": 0, "unmapped": 0}
    diags: list[tuple[str, dict]] = []

    for sk in skills:
        eps_arr = z.get(f"{sk}::episode")
        if eps_arr is None:
            continue
        new_arr, diag = recover_skill_array(eps_arr, mapping, old_layout_max)
        new_z[f"{sk}::episode"] = new_arr
        for k in ("rewritten", "kept", "unmapped"):
            tot[k] += diag.get(k, 0)
        diags.append((sk, diag))

        drop_str = (
            f"@{diag['drops'][0]}" if len(diag["drops"]) == 1
            else (f"@{diag['drops']}" if diag["drops"] else "none")
        )
        print(f"  [{sk:<16}] N={diag['n_entries']:3d}  drops={drop_str:<10}  "
              f"status={diag['status']:<28}  rewrite={diag['rewritten']:3d}  "
              f"keep={diag['kept']:3d}  unmapped={diag['unmapped']:3d}")
        if diag["boundary"] is not None:
            print(f"      Run A IDs ≤{diag['run_a_max']:>2}  @ [0..{diag['boundary']})   |   "
                  f"today IDs ≤{diag['today_max']:>2}  @ [{diag['boundary']}..{diag['n_entries']})")

    suspicious = [
        sk for sk, d in diags
        if d["status"] not in ("ok", "no_drop_value_based", "empty")
    ]
    print()
    print(f"summary: rewritten={tot['rewritten']}, kept={tot['kept']}, unmapped={tot['unmapped']}")
    if suspicious:
        print(f"WARN  : {len(suspicious)} skills SKIPPED due to anomalies — review manually:")
        for sk in suspicious:
            print(f"          - {sk}")
        print(f"        (these skills' arrays are left unchanged in the output)")

    if not args.apply:
        print()
        print("[DRY-RUN] no file written. Re-run with --apply to commit.")
        return

    backup = buf_path.with_suffix(".npz.before-recover")
    shutil.copy2(buf_path, backup)
    print()
    print(f"backup    : {backup}")

    tmp = buf_path.with_suffix(".npz.tmp")
    np.savez(tmp, **new_z)
    tmp.replace(buf_path)
    print(f"recovered : {buf_path}")


if __name__ == "__main__":
    main()
