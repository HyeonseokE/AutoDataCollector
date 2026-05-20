#!/usr/bin/env python3
"""Rebuild ``<session_dir>/judge_results/`` from per-episode source-of-truth.

For each ``episode_NN/forward/judge_result.jpg`` in the session, copy it to
``judge_results/judge_result_epNN.jpg`` using the **current** folder layout's
NN. Stale files in ``judge_results/`` whose NN no longer corresponds to a
folder are removed.

This is idempotent and safe to run after any folder rename (e.g., after the
30→50 layout migration that didn't include judge_results in its rename step).
The per-episode ``judge_result.jpg`` files are preserved by the folder mv;
this script just reconstructs the consolidated view consistently.

Default mode is dry-run with a per-file plan. Pass ``--apply`` to commit.

Usage
-----
    python scripts/rebuild_judge_results.py <session_dir>
    python scripts/rebuild_judge_results.py <session_dir> --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


EP_RE = re.compile(r"^episode_(\d+)$")
JR_RE = re.compile(r"^judge_result_ep(\d+)\.jpg$")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="commit changes (default: dry-run)")
    args = ap.parse_args()

    sd = args.session_dir
    if not sd.is_dir():
        sys.exit(f"not a directory: {sd}")

    jr_dir = sd / "judge_results"
    jr_dir.mkdir(exist_ok=True)

    # 1. plan COPIES: for each episode_NN that has forward/judge_result.jpg,
    #    target = judge_results/judge_result_epNN.jpg
    plan_copy: list[tuple[Path, Path]] = []
    eps_with_source: set[int] = set()
    for ep_dir in sorted(sd.glob("episode_*")):
        m = EP_RE.match(ep_dir.name)
        if not m:
            continue
        nn = int(m.group(1))
        src = ep_dir / "forward" / "judge_result.jpg"
        if not src.exists():
            continue
        dst = jr_dir / f"judge_result_ep{nn:02d}.jpg"
        eps_with_source.add(nn)
        # skip if dst already a byte-identical copy of src
        if dst.exists():
            try:
                if dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime == src.stat().st_mtime:
                    continue   # already aligned
            except OSError:
                pass
        plan_copy.append((src, dst))

    # 2. plan DELETES: stale judge_result_epNN.jpg with no matching folder
    plan_delete: list[Path] = []
    valid_eps = {int(EP_RE.match(p.name).group(1))
                 for p in sd.glob("episode_*") if EP_RE.match(p.name)}
    for f in sorted(jr_dir.glob("judge_result_ep*.jpg")):
        m = JR_RE.match(f.name)
        if not m:
            continue
        nn = int(m.group(1))
        if nn not in valid_eps:
            plan_delete.append(f)

    # 3. print plan
    print(f"session_dir : {sd}")
    print(f"folders     : {len(valid_eps)} episodes")
    print(f"sources     : {len(eps_with_source)} with forward/judge_result.jpg")
    print(f"plan        : {len(plan_copy)} copy/overwrite, {len(plan_delete)} delete")
    print()
    if plan_copy:
        print("would copy:")
        for src, dst in plan_copy[:8]:
            print(f"  {src.relative_to(sd)}  →  {dst.relative_to(sd)}")
        if len(plan_copy) > 8:
            print(f"  ... and {len(plan_copy) - 8} more")
    if plan_delete:
        print()
        print("would delete (no matching folder):")
        for f in plan_delete[:8]:
            print(f"  {f.relative_to(sd)}")
        if len(plan_delete) > 8:
            print(f"  ... and {len(plan_delete) - 8} more")

    if not args.apply:
        print()
        print("[DRY-RUN] no changes. Re-run with --apply to commit.")
        return

    print()
    for src, dst in plan_copy:
        shutil.copy2(src, dst)
        print(f"  copied   : {dst.name}")
    for f in plan_delete:
        f.unlink()
        print(f"  deleted  : {f.name}")
    print(f"\ndone: {len(plan_copy)} copied, {len(plan_delete)} deleted")


if __name__ == "__main__":
    main()
