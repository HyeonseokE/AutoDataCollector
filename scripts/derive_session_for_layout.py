"""Derive a new session folder from an existing one with a different
``episodes_per_seed`` layout.

The source session is copied to ``--dest``; then each episode folder inside the
destination is re-numbered so the new name encodes the **same** (seed, slot)
under the new ``episodes_per_seed = new_episodes // num_seeds``.

Use this when you want a 50-episode variant of a session that's currently laid
out as 100/10, or any other layout where the underlying (seed, slot) truth
should be preserved.

Mapping (single source of truth — each episode's ``batch_info.json``):

    new_ep_num = (batch_seed_index - 1) * new_eps_per_seed + slot + 1

Two-phase rename guarantees no src/dst collisions:

    1. Every ep_NN  → __migrate_tmp__NN
    2. Every __migrate_tmp__NN → new_ep_NN

Compared to ``migrate_session_episodes_per_seed.py``, this script:
  * Works **across sessions** (source ≠ dest).
  * Does **not** care about old_episodes / old_eps_per_seed — batch_info is the
    only source of truth for (seed, slot).
  * Supports **both** up- and down-migration (50/10 → 100/10, 100/10 → 50/10,
    30/10 → 50/10 …) as long as every existing episode still fits the new
    layout (slot < new_eps_per_seed).

Defaults to **dry-run**; pass ``--apply`` to commit.

Example:
    python -m scripts.derive_session_for_layout \\
        --source results/session_20260519_225321 \\
        --dest   results/session_20260519_225321_50 \\
        --new-episodes 50 --num-seeds 10

    # then re-run with --apply.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_TMP_PREFIX = "__migrate_tmp__"


@dataclass(frozen=True)
class DeriveMove:
    old_ep: int
    new_ep: int
    seed_idx: int  # 0-based
    slot: int  # 0-based

    def label(self) -> str:
        return (
            f"ep_{self.old_ep:02d} → ep_{self.new_ep:02d}"
            f"  (seed_{self.seed_idx + 1:02d}, slot {self.slot})"
        )


def _read_seed_slot(batch_info_path: Path) -> tuple[int, int]:
    bi = json.loads(batch_info_path.read_text())
    if "batch_seed_index" in bi:
        seed_idx = int(bi["batch_seed_index"]) - 1
    elif "batch_index" in bi:
        seed_idx = int(bi["batch_index"])  # legacy 0-based
    else:
        raise KeyError(f"{batch_info_path} has neither batch_seed_index nor batch_index")
    if "slot" not in bi:
        raise KeyError(f"{batch_info_path} has no slot")
    return seed_idx, int(bi["slot"])


def plan_moves(
    session_dir: Path,
    new_episodes: int,
    num_seeds: int,
) -> list[DeriveMove]:
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    if new_episodes % num_seeds:
        raise ValueError(
            f"new_episodes ({new_episodes}) must be divisible by num_seeds ({num_seeds})"
        )
    new_eps_per_seed = new_episodes // num_seeds

    ep_dirs = sorted(session_dir.glob("episode_*"))
    moves: list[DeriveMove] = []
    errors: list[str] = []
    for ep_dir in ep_dirs:
        try:
            old_ep = int(ep_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        bi_path = ep_dir / "batch_info.json"
        if not bi_path.exists():
            errors.append(f"  ep_{old_ep:02d}: batch_info.json missing — cannot re-number")
            continue
        try:
            seed_idx, slot = _read_seed_slot(bi_path)
        except KeyError as e:
            errors.append(f"  ep_{old_ep:02d}: {e}")
            continue
        if seed_idx < 0 or seed_idx >= num_seeds:
            errors.append(
                f"  ep_{old_ep:02d}: batch_seed_index out of range "
                f"(seed_idx={seed_idx + 1}, num_seeds={num_seeds})"
            )
            continue
        if slot < 0 or slot >= new_eps_per_seed:
            errors.append(
                f"  ep_{old_ep:02d}: slot {slot} ≥ new_eps_per_seed {new_eps_per_seed} — "
                f"cannot fit into new layout (seed has too few slots)"
            )
            continue
        new_ep = seed_idx * new_eps_per_seed + slot + 1
        moves.append(DeriveMove(old_ep, new_ep, seed_idx, slot))

    if errors:
        raise RuntimeError("layout incompatibility:\n" + "\n".join(errors))

    # collision detection (after both phases, every new_ep must be unique)
    new_eps = [m.new_ep for m in moves]
    dup = {n for n in new_eps if new_eps.count(n) > 1}
    if dup:
        raise RuntimeError(f"duplicate new_ep numbers: {sorted(dup)}")

    return moves


def print_plan(
    moves: list[DeriveMove],
    source: Path,
    dest: Path,
    new_episodes: int,
    num_seeds: int,
) -> None:
    new_eps_per_seed = new_episodes // num_seeds
    print("=" * 70)
    print(f"  Derive session — {source.name} → {dest.name}")
    print("=" * 70)
    print(f"  source: {source}")
    print(f"  dest  : {dest}")
    print(f"  new layout: {new_episodes} episodes, {num_seeds} seeds → eps/seed = {new_eps_per_seed}")
    print()
    print(f"  {'#':>3}  Action")
    print(f"  {'-' * 3}  {'-' * 56}")
    no_op = 0
    moves_sorted = sorted(moves, key=lambda m: m.old_ep)
    for i, m in enumerate(moves_sorted, 1):
        tag = "(no-op)" if m.old_ep == m.new_ep else ""
        if m.old_ep == m.new_ep:
            no_op += 1
        print(f"  {i:>3}. {m.label()}  {tag}")
    print()
    print(f"  Total renames planned: {len(moves)}  (no-ops: {no_op})")
    occupied = {m.new_ep for m in moves}
    empty_after = [
        seed_idx * new_eps_per_seed + s + 1
        for seed_idx in range(num_seeds)
        for s in range(new_eps_per_seed)
        if (seed_idx * new_eps_per_seed + s + 1) not in occupied
    ]
    if empty_after:
        print(f"  Empty slots after derive (resume 시 채워질 자리, {len(empty_after)}개):")
        # group by seed
        by_seed: dict[int, list[int]] = {}
        for ep in empty_after:
            seed_idx = (ep - 1) // new_eps_per_seed
            by_seed.setdefault(seed_idx, []).append(ep)
        for seed_idx in sorted(by_seed):
            eps = by_seed[seed_idx]
            print(f"    seed_{seed_idx + 1:02d}: {', '.join(f'ep_{e:02d}' for e in eps)}")


def two_phase_rename(session_dir: Path, moves: Iterable[DeriveMove]) -> None:
    """Phase 1: ep_NN → __migrate_tmp__NN.  Phase 2: __migrate_tmp__NN → ep_<new>.

    judge_results/judge_result_ep{NN}.* 도 같은 매핑으로 동반 rename. judge 파일명
    의 NN 은 source 에서 episode_{NN} 폴더와 동기화돼 있다고 전제 (이전 마이그레이션
    툴이 둘을 같이 옮겼다면 항상 정합).
    """
    moves_list = list(moves)
    judge_dir = session_dir / "judge_results"
    has_judge_dir = judge_dir.is_dir()

    # phase 1: ep_NN → tmp  (episode 폴더 + judge_results 파일)
    for m in moves_list:
        src = session_dir / f"episode_{m.old_ep:02d}"
        tmp = session_dir / f"{_TMP_PREFIX}{m.old_ep:02d}"
        if not src.exists():
            print(f"  [skip phase1] ep_{m.old_ep:02d} src missing")
            continue
        if tmp.exists():
            raise RuntimeError(f"unexpected tmp leftover: {tmp}")
        src.rename(tmp)

    if has_judge_dir:
        for m in moves_list:
            for j_src in sorted(judge_dir.glob(f"judge_result_ep{m.old_ep:02d}.*")):
                suffix = j_src.name[len(f"judge_result_ep{m.old_ep:02d}"):]
                j_tmp = judge_dir / f"{_TMP_PREFIX}judge_{m.old_ep:02d}{suffix}"
                if j_tmp.exists():
                    raise RuntimeError(f"unexpected judge tmp leftover: {j_tmp}")
                j_src.rename(j_tmp)

    # phase 2: tmp → ep_<new>  (episode 폴더 + judge_results 파일)
    for m in moves_list:
        tmp = session_dir / f"{_TMP_PREFIX}{m.old_ep:02d}"
        dst = session_dir / f"episode_{m.new_ep:02d}"
        if not tmp.exists():
            continue
        if dst.exists():
            raise RuntimeError(
                f"unexpected dst collision after phase1: {dst} already exists"
            )
        tmp.rename(dst)
        if m.old_ep != m.new_ep:
            print(f"  [renamed] ep_{m.old_ep:02d} → ep_{m.new_ep:02d}")

    if has_judge_dir:
        judge_moved = 0
        for m in moves_list:
            for j_tmp in sorted(judge_dir.glob(f"{_TMP_PREFIX}judge_{m.old_ep:02d}.*")):
                suffix = j_tmp.name[len(f"{_TMP_PREFIX}judge_{m.old_ep:02d}"):]
                j_dst = judge_dir / f"judge_result_ep{m.new_ep:02d}{suffix}"
                if j_dst.exists():
                    raise RuntimeError(
                        f"unexpected judge dst collision: {j_dst} already exists"
                    )
                j_tmp.rename(j_dst)
                if m.old_ep != m.new_ep:
                    judge_moved += 1
        if judge_moved:
            print(f"  [renamed-judge] {judge_moved} judge_results files synced")


def update_session_config(session_dir: Path, num_episodes: int, num_seeds: int, dry_run: bool) -> None:
    config_path = session_dir / "session_config.json"
    if not config_path.exists():
        print(f"  [warn] session_config.json not found at {config_path}")
        return
    cfg = json.loads(config_path.read_text())
    old_cfg = {k: cfg.get(k) for k in ("num_episodes", "num_random_seeds", "episodes_per_seed")}
    new_cfg = {
        "num_episodes": num_episodes,
        "num_random_seeds": num_seeds,
        "episodes_per_seed": num_episodes // num_seeds,
    }
    print()
    print(f"  session_config.json update:")
    for k, v in new_cfg.items():
        marker = "*" if old_cfg.get(k) != v else " "
        print(f"   {marker} {k}: {old_cfg.get(k)} → {v}")
    if dry_run:
        print(f"  (dry-run — not writing)")
        return
    cfg.update(new_cfg)
    config_path.write_text(json.dumps(cfg, indent=2))
    print(f"  session_config.json written.")


def copy_source_to_dest(source: Path, dest: Path) -> None:
    if dest.exists():
        raise RuntimeError(f"dest already exists: {dest}  (rm it first or pick another name)")
    print(f"  copying {source} → {dest} ...")
    shutil.copytree(source, dest)
    print(f"  copy done.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, type=Path, help="Source session_dir")
    parser.add_argument("--dest", required=True, type=Path, help="Destination session_dir (must not exist)")
    parser.add_argument("--new-episodes", type=int, required=True, help="Target total episodes")
    parser.add_argument("--num-seeds", type=int, required=True, help="Number of random seeds (unchanged)")
    parser.add_argument("--apply", action="store_true", help="Copy + rename + update config (default: dry-run)")
    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()
    if not source.is_dir():
        print(f"error: source dir not found: {source}", file=sys.stderr)
        return 2

    # plan against the source (it holds the batch_info ground truth)
    try:
        moves = plan_moves(source, args.new_episodes, args.num_seeds)
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print_plan(moves, source, dest, args.new_episodes, args.num_seeds)

    print()
    if not args.apply:
        # also show what session_config will become (dry-run)
        update_session_config(source, args.new_episodes, args.num_seeds, dry_run=True)
        print()
        print("  DRY RUN — no copy, no rename. Re-run with --apply to commit.")
        return 0

    # apply: copy source → dest, rename inside dest, update dest's session_config
    copy_source_to_dest(source, dest)
    two_phase_rename(dest, moves)
    update_session_config(dest, args.new_episodes, args.num_seeds, dry_run=False)
    print()
    print(f"  done. derived session ready at: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
