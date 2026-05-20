"""Migrate a session folder from (N1 episodes, S seeds) → (N2 episodes, S seeds).

Use case: a session was collected with `episodes_per_seed = N1 // S` (e.g. 30/10
= 3). The user now wants to extend to N2 / S (e.g. 100/10 = 10) keeping the same
S physical seeds. Each existing episode's `(seed, slot)` truth — recorded in
`batch_info.json` — must be preserved while the on-disk folder number shifts so
that `episode_idx // new_eps_per_seed == seed_idx` for the new scheme.

Mapping:
    new_ep_num = seed_idx * new_eps_per_seed + slot + 1
                 with seed_idx = (old_ep_num - 1) // old_eps_per_seed
                      slot     = (old_ep_num - 1) % old_eps_per_seed

mv order matters — must go from the largest old_ep_num downward so destination
slots are always free at the time of the move.

By default this is a DRY RUN. Pass `--apply` to actually rename folders and
update `session_config.json`.

Example:
    python -m scripts.migrate_session_episodes_per_seed \
        --session results/session_20260519_225321 \
        --old-episodes 30 --new-episodes 100 --num-seeds 10
    # then add --apply to commit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MigrationMove:
    """A single planned folder rename."""

    old_ep: int
    new_ep: int
    seed_idx: int  # 0-based
    slot: int  # 0-based
    src: Path
    dst: Path

    def label(self) -> str:
        return (
            f"ep_{self.old_ep:02d} → ep_{self.new_ep:02d}"
            f"  (seed_{self.seed_idx + 1:02d}, slot {self.slot})"
        )


def _compute_new_ep_num(old_ep: int, old_eps_per_seed: int, new_eps_per_seed: int) -> tuple[int, int, int]:
    idx0 = old_ep - 1
    seed_idx = idx0 // old_eps_per_seed
    slot = idx0 % old_eps_per_seed
    new_ep = seed_idx * new_eps_per_seed + slot + 1
    return new_ep, seed_idx, slot


def plan_moves(
    session_dir: Path,
    old_episodes: int,
    new_episodes: int,
    num_seeds: int,
) -> list[MigrationMove]:
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    if old_episodes % num_seeds:
        raise ValueError(
            f"old_episodes ({old_episodes}) must be divisible by num_seeds ({num_seeds})"
        )
    if new_episodes % num_seeds:
        raise ValueError(
            f"new_episodes ({new_episodes}) must be divisible by num_seeds ({num_seeds})"
        )
    if new_episodes < old_episodes:
        raise ValueError("new_episodes must be ≥ old_episodes (this script only extends)")

    old_eps_per_seed = old_episodes // num_seeds
    new_eps_per_seed = new_episodes // num_seeds

    moves: list[MigrationMove] = []
    # iterate largest → smallest so destinations are always empty
    for old_ep in range(old_episodes, 0, -1):
        new_ep, seed_idx, slot = _compute_new_ep_num(old_ep, old_eps_per_seed, new_eps_per_seed)
        src = session_dir / f"episode_{old_ep:02d}"
        dst = session_dir / f"episode_{new_ep:02d}"
        moves.append(MigrationMove(old_ep, new_ep, seed_idx, slot, src, dst))
    return moves


def _read_batch_info_seed_slot(ep_dir: Path) -> tuple[int | None, int | None]:
    """Read (seed_idx_0based, slot) from batch_info.json, or (None, None) if absent."""
    bi_path = ep_dir / "batch_info.json"
    if not bi_path.exists():
        return None, None
    try:
        bi = json.loads(bi_path.read_text())
    except json.JSONDecodeError:
        return None, None
    if "batch_seed_index" in bi:
        seed_idx = bi["batch_seed_index"] - 1  # 1-based → 0-based
    else:
        seed_idx = bi.get("batch_index")  # legacy
    slot = bi.get("slot")
    return seed_idx, slot


def verify_moves_against_batch_info(moves: Iterable[MigrationMove]) -> list[str]:
    """Sanity check — recorded (seed, slot) in batch_info must match the computed mapping."""
    mismatches: list[str] = []
    for m in moves:
        if not m.src.exists():
            continue  # missing src — flagged separately
        bi_seed, bi_slot = _read_batch_info_seed_slot(m.src)
        if bi_seed is None or bi_slot is None:
            continue  # no batch_info to verify against (will be a separate warning)
        if bi_seed != m.seed_idx or bi_slot != m.slot:
            mismatches.append(
                f"ep_{m.old_ep:02d}: batch_info says (seed={bi_seed + 1}, slot={bi_slot}) "
                f"but mapping says (seed={m.seed_idx + 1}, slot={m.slot})"
            )
    return mismatches


def print_plan(
    moves: list[MigrationMove],
    session_dir: Path,
    old_episodes: int,
    new_episodes: int,
    num_seeds: int,
) -> None:
    print("=" * 70)
    print(f"  Migration plan — {session_dir.name}")
    print("=" * 70)
    print(f"  Old layout: {old_episodes} episodes, {num_seeds} seeds → eps/seed = {old_episodes // num_seeds}")
    print(f"  New layout: {new_episodes} episodes, {num_seeds} seeds → eps/seed = {new_episodes // num_seeds}")
    print(f"  Empty new slots to be collected later: {new_episodes - old_episodes}")
    print()
    print(f"  {'#':>3}  Action")
    print(f"  {'-' * 3}  {'-' * 56}")
    no_op = 0
    for i, m in enumerate(moves, 1):
        if m.src == m.dst:
            tag = "(no-op)"
            no_op += 1
        else:
            tag = ""
        exists = "✓" if m.src.exists() else "✗ MISSING"
        print(f"  {i:>3}. {m.label()}  {tag}  [src {exists}]")
    print()
    print(f"  Total moves planned: {len(moves)}  (no-ops: {no_op})")
    new_eps_per_seed = new_episodes // num_seeds
    print(f"  Empty slots after migration (will be filled by resume):")
    for seed_idx in range(num_seeds):
        old_eps_per_seed = old_episodes // num_seeds
        empty_start = seed_idx * new_eps_per_seed + old_eps_per_seed + 1
        empty_end = (seed_idx + 1) * new_eps_per_seed
        print(f"    seed_{seed_idx + 1:02d}: ep_{empty_start:02d}..ep_{empty_end:02d}  ({old_eps_per_seed + 1}..{new_eps_per_seed} 의 slot)")


def collect_judge_moves(session_dir: Path, moves: list[MigrationMove]) -> list[tuple[Path, Path]]:
    """Collect judge_results/judge_result_ep{NN}.* renames matching episode moves.

    `judge_result_ep{NN}.*` 파일은 episode 폴더와 동일한 NN 을 가져야 정합.
    episode rename 의 (old_ep → new_ep) 매핑을 그대로 judge_results 에 적용한다.
    glob 으로 확장자 무관 (.jpg, .json, .txt 등) 매칭.
    """
    judge_dir = session_dir / "judge_results"
    if not judge_dir.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for m in moves:
        if m.old_ep == m.new_ep:
            continue
        for src in sorted(judge_dir.glob(f"judge_result_ep{m.old_ep:02d}.*")):
            suffix = src.name[len(f"judge_result_ep{m.old_ep:02d}"):]  # ".jpg" 같은 나머지
            dst = judge_dir / f"judge_result_ep{m.new_ep:02d}{suffix}"
            pairs.append((src, dst))
    return pairs


def apply_folder_moves(moves: list[MigrationMove]) -> None:
    """episode 폴더만 mv (큰 번호 → 작은 번호 순서로 충돌 회피)."""
    for m in moves:
        if m.src == m.dst:
            continue
        if not m.src.exists():
            print(f"  [skip folder] {m.label()} — src missing")
            continue
        if m.dst.exists():
            raise RuntimeError(
                f"destination {m.dst} already exists; aborting to avoid overwrite "
                f"(this shouldn't happen if mv order is correct — old→new descending)"
            )
        shutil.move(str(m.src), str(m.dst))
        print(f"  [moved] {m.label()}")


def apply_judge_moves(session_dir: Path, moves: list[MigrationMove]) -> None:
    """judge_results/judge_result_ep{NN}.* 만 episode 매핑과 동일하게 mv.

    episode 폴더 mv 와 동일한 순서 (큰 번호 → 작은) 라 충돌 회피 패턴 동일.
    """
    judge_pairs = collect_judge_moves(session_dir, moves)
    if not judge_pairs:
        return
    print()
    print(f"  judge_results sync ({len(judge_pairs)} files):")
    for src, dst in judge_pairs:
        if not src.exists():
            print(f"  [skip judge] {src.name} — src missing")
            continue
        if dst.exists():
            raise RuntimeError(
                f"judge_results destination {dst} already exists; aborting "
                f"(verify the mv order is correct)"
            )
        shutil.move(str(src), str(dst))
        print(f"  [moved-judge] {src.name} → {dst.name}")


def apply_buffer_rewrite(session_dir: Path, moves: list[MigrationMove]) -> None:
    """subgoal_buffer.npz 의 ``episode_id`` 를 episode 매핑과 동일하게 치환.

    폴더·judge_results rename 과 짝을 맞춰 buffer 도 새 layout 으로 따라가지
    않으면, 다음 resume 의 ``retain_episodes`` reconcile 이 stale id 라 판단해
    buffer 의 Run A 부분을 drop 한다.
    """
    buf_path = session_dir / "subgoal_buffer.npz"
    if not buf_path.exists():
        return
    try:
        # 지연 import: migration script 가 method3 의존성을 강제하지 않도록.
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from method3.phase1_state_seeding.subgoal_buffer import SubgoalBuffer
    except ImportError as e:
        print(f"  [skip buffer] cannot import SubgoalBuffer: {e}")
        return

    mapping = {
        f"episode_{m.old_ep:02d}": f"episode_{m.new_ep:02d}"
        for m in moves if m.old_ep != m.new_ep
    }
    if not mapping:
        return
    buf = SubgoalBuffer(buffer_file=buf_path)
    buf.load()
    counts = buf.rewrite_episodes(mapping)
    print(
        f"\n  subgoal_buffer rewrite: rewritten={counts['rewritten']}, "
        f"kept={counts['kept']}, unmapped={counts['unmapped']}"
    )


def apply_moves(moves: list[MigrationMove], session_dir: Path) -> None:
    """폴더 + judge_results + subgoal_buffer 모두 atomic 하게 변환 (정상 경로)."""
    apply_folder_moves(moves)
    apply_judge_moves(session_dir, moves)
    apply_buffer_rewrite(session_dir, moves)


def update_session_config(session_dir: Path, num_episodes: int, num_seeds: int, dry_run: bool) -> None:
    config_path = session_dir / "session_config.json"
    if not config_path.exists():
        print(f"  [warn] session_config.json not found at {config_path}")
        return
    cfg = json.loads(config_path.read_text())
    old_cfg = {k: cfg.get(k) for k in ("num_episodes", "num_random_seeds", "episodes_per_seed")}
    eps_per_seed = num_episodes // num_seeds
    new_cfg = {
        "num_episodes": num_episodes,
        "num_random_seeds": num_seeds,
        "episodes_per_seed": eps_per_seed,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, type=Path, help="Path to session_YYYYMMDD_HHMMSS directory")
    parser.add_argument("--old-episodes", type=int, required=True, help="Currently collected total episodes")
    parser.add_argument("--new-episodes", type=int, required=True, help="Target total episodes after extension")
    parser.add_argument("--num-seeds", type=int, required=True, help="Number of random seeds (unchanged across migration)")
    parser.add_argument("--apply", action="store_true", help="Actually perform the moves and config update (default: dry-run)")
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help=(
            "Backfill mode: leave episode folders + session_config untouched, "
            "only sync judge_results/judge_result_ep{NN}.* file names to match the "
            "new layout. Use when a previous migration moved folders but skipped "
            "judge_results. Skips batch_info verify (intentionally — folders are "
            "already at the post-migration layout)."
        ),
    )
    args = parser.parse_args()

    session_dir = args.session.resolve()
    if not session_dir.is_dir():
        print(f"error: session dir not found: {session_dir}", file=sys.stderr)
        return 2

    try:
        moves = plan_moves(session_dir, args.old_episodes, args.new_episodes, args.num_seeds)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print_plan(moves, session_dir, args.old_episodes, args.new_episodes, args.num_seeds)

    mismatches: list[str] = []
    if args.judge_only:
        print()
        print("  [judge-only] skipping batch_info verification — folders left as-is, only judge_results synced.")
    else:
        mismatches = verify_moves_against_batch_info(moves)
        print()
        if mismatches:
            print("  [WARN] batch_info ↔ mapping mismatches:")
            for msg in mismatches:
                print(f"    - {msg}")
            print("  → fix mapping or batch_info before applying.")
        else:
            print("  [ok] batch_info ↔ mapping consistent for all existing episodes.")

        missing = [m for m in moves if not m.src.exists()]
        if missing:
            print(f"  [warn] {len(missing)} src episode(s) missing on disk (will be skipped).")

    # judge_results 동반 마이그레이션 계획 출력
    judge_pairs = collect_judge_moves(session_dir, moves)
    if judge_pairs:
        print()
        print(f"  judge_results sync planned ({len(judge_pairs)} files):")
        for src, dst in judge_pairs[:5]:
            print(f"    {src.name} → {dst.name}")
        if len(judge_pairs) > 5:
            print(f"    … (+{len(judge_pairs) - 5} more)")
    else:
        if (session_dir / "judge_results").is_dir():
            print()
            print("  judge_results sync: 0 files (already aligned or no matching names).")

    if not args.judge_only:
        update_session_config(session_dir, args.new_episodes, args.num_seeds, dry_run=not args.apply)

    if not args.apply:
        print()
        print("  DRY RUN — no changes made. Re-run with --apply to commit.")
        return 0

    if mismatches:
        print()
        print("  refusing to --apply with batch_info mismatches.", file=sys.stderr)
        return 1

    print()
    if args.judge_only:
        print("  applying judge_results sync only (folders + session_config untouched)...")
        apply_judge_moves(session_dir, moves)
    else:
        print("  applying moves (largest old → smallest)...")
        apply_moves(moves, session_dir)
    print("  done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
