#!/usr/bin/env python3
"""Convert gripper column in existing LeRobot datasets from RANGE_M100_100 to RANGE_0_100.

ADC previously recorded all motors (including the gripper) on the -100..+100
RANGE_M100_100 scale. After migrating to RANGE_0_100 for the gripper (matching
upstream LeRobot), existing datasets have gripper columns in the old scheme and
are incompatible with models trained on upstream conventions.

Conversion: new_gripper = (old_gripper + 100) / 2   → -100↦0, 0↦50, +100↦100

Only `observation.state[5]` and `action[5]` are modified. Other columns —
including `observation.state.radian_urdf0` and `action.radian_urdf0` — represent
physical joint angles and are unchanged.

Usage:
    # Dry run (stats only)
    python scripts/convert_gripper_normalization.py \
        --repo-id skkuprism/cap_pnp_100ep --revision v3.0 --dry-run

    # Apply locally, write to ./converted/<repo>_converted/
    python scripts/convert_gripper_normalization.py \
        --repo-id skkuprism/cap_pnp_100ep --revision v3.0

    # Push converted result to HF as a new tag (requires HF login)
    python scripts/convert_gripper_normalization.py \
        --repo-id skkuprism/cap_pnp_100ep --revision v3.0 \
        --push-tag v3.1_gripper_0_100
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


GRIPPER_IDX = 5  # last column of 6D observation.state / action


def _convert_column(arr_list: list) -> list:
    """Apply (x + 100) / 2 to the gripper element of every 6D row."""
    out = []
    for row in arr_list:
        new = list(row)
        new[GRIPPER_IDX] = (float(new[GRIPPER_IDX]) + 100.0) / 2.0
        out.append(new)
    return out


def _convert_parquet(src: Path, dst: Path, columns_to_fix: list[str]) -> dict:
    """Rewrite one parquet file with gripper columns converted. Return stats."""
    table = pq.read_table(src)
    stats = {"rows": len(table), "columns_modified": []}

    new_columns = {}
    for name in table.column_names:
        col = table.column(name)
        if name in columns_to_fix:
            data = col.to_pylist()
            converted = _convert_column(data)
            new_columns[name] = pa.array(converted, type=col.type)
            stats["columns_modified"].append(name)
        else:
            new_columns[name] = col

    new_table = pa.table(new_columns)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst, compression="zstd")
    return stats


def _sample_stats(local_dir: Path, col: str, limit: int = 2000) -> dict:
    """Sample gripper values from the first data shard to show before/after ranges."""
    data_files = sorted((local_dir / "data").rglob("*.parquet"))
    if not data_files:
        return {"error": "no data files"}
    t = pq.read_table(data_files[0], columns=[col])
    arr = np.array(t.column(col).slice(0, limit).to_pylist())
    if arr.ndim == 2:
        return {
            "file": str(data_files[0].relative_to(local_dir)),
            "sampled": len(arr),
            "gripper_min": float(arr[:, GRIPPER_IDX].min()),
            "gripper_max": float(arr[:, GRIPPER_IDX].max()),
            "gripper_mean": float(arr[:, GRIPPER_IDX].mean()),
        }
    return {"error": f"unexpected shape {arr.shape}"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="HF dataset repo id")
    p.add_argument("--revision", default="main", help="Source revision/tag")
    p.add_argument("--output-dir", default=None,
                   help="Local output directory. Default: ./converted/<repo_tag>")
    p.add_argument("--dry-run", action="store_true",
                   help="Only report gripper ranges, don't convert")
    p.add_argument("--push-tag", default=None,
                   help="After conversion, push result to HF as this tag (requires login)")
    p.add_argument("--force", action="store_true",
                   help="Run conversion even if gripper already looks like [0, 100] range")
    args = p.parse_args()

    columns_to_fix = ["observation.state", "action"]

    print(f"Downloading {args.repo_id}@{args.revision} ...")
    local_dir = Path(snapshot_download(
        repo_id=args.repo_id, revision=args.revision,
        repo_type="dataset",
    ))

    print(f"\n=== BEFORE (sampled from first shard) ===")
    before_stats = {}
    for col in columns_to_fix:
        s = _sample_stats(local_dir, col)
        before_stats[col] = s
        print(f"  {col}[{GRIPPER_IDX}]: {s}")

    # Guard: if gripper values are already in the [0, 100] range, the dataset
    # was recorded with upstream LeRobot directly and does NOT need conversion.
    # Only ADC-collected datasets (values in [-100, +100] with negatives) need it.
    min_val = min(s.get("gripper_min", 0) for s in before_stats.values())
    max_val = max(s.get("gripper_max", 0) for s in before_stats.values())
    if min_val >= 0.0:
        print(f"\n[SKIP] gripper values already in [0, 100] range "
              f"(min={min_val:.2f}, max={max_val:.2f}). "
              f"This dataset was likely recorded with upstream LeRobot directly — "
              f"no conversion needed. Use --force to override.")
        if not getattr(args, "force", False):
            return

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        tag = args.repo_id.replace("/", "_") + "_" + args.revision
        out_dir = Path(f"./converted/{tag}")

    if out_dir.exists():
        print(f"\nOutput dir exists, removing: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"\nConverting → {out_dir}")
    total_rows = 0
    for src in sorted(local_dir.rglob("*")):
        rel = src.relative_to(local_dir)
        dst = out_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if src.suffix == ".parquet" and str(rel).startswith("data/"):
            stats = _convert_parquet(src, dst, columns_to_fix)
            total_rows += stats["rows"]
            print(f"  data: {rel}  rows={stats['rows']}  fixed={stats['columns_modified']}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(f"\nConverted {total_rows} frames across data shards.")

    print("\n=== AFTER (sampled from first shard) ===")
    for col in columns_to_fix:
        s = _sample_stats(out_dir, col)
        print(f"  {col}[{GRIPPER_IDX}]: {s}")

    if args.push_tag:
        print(f"\nPushing converted dataset to {args.repo_id}@{args.push_tag} ...")
        api = HfApi()
        # HF tags are read-only pointers; we need a branch to hold the commit.
        # Strategy: create a branch of the same name, upload, then tag its HEAD.
        branch_name = args.push_tag
        try:
            api.create_branch(
                repo_id=args.repo_id, branch=branch_name,
                revision=args.revision, repo_type="dataset",
                exist_ok=True,
            )
            print(f"  branch '{branch_name}' ready (based on {args.revision})")
        except Exception as e:
            print(f"  create_branch error: {e}")
            raise

        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=branch_name,
            create_pr=False,
            commit_message=f"Convert gripper from RANGE_M100_100 to RANGE_0_100 (post-process from {args.revision})",
        )
        try:
            api.create_tag(
                repo_id=args.repo_id, tag=args.push_tag, repo_type="dataset",
                revision=branch_name,
                tag_message=f"Gripper normalization aligned to upstream LeRobot RANGE_0_100",
            )
            print(f"  ✓ tag '{args.push_tag}' created")
        except Exception as e:
            print(f"  (tag may already exist: {e})")
        print(f"  ✓ pushed.")

    print(f"\nDone. Local output: {out_dir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
