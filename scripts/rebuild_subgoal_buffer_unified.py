"""subgoal_buffer.npz 를 paradigm 일관화 namespace 로 재생성.

기존 buffer: transit + move_initial 만 entry (skill_0~N 일부)
새 buffer:  모든 set_skill_info 호출 단위 entry (VDB 와 동일 namespace).

Source: phase1 dataset (LeRobot v3) 의 skill.natural_language run-length
        + observation.ee_pos.robot_xyzrpy + skill.goal_position.robot_xyzrpy.

Usage:
    python -m scripts.rebuild_subgoal_buffer_unified \\
        --dataset CoRL2026-CSI/table3/stateseeding+random_pert \\
        --out results/completed_logs/table3/stateseeding+random_pert/subgoal_buffer.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _resolve_dataset_root(repo_id_or_path: str) -> Path:
    p = Path(repo_id_or_path)
    if p.exists() and (p / "meta").exists():
        return p
    from lerobot.utils.constants import HF_LEROBOT_HOME
    return Path(HF_LEROBOT_HOME) / repo_id_or_path


def _run_length(seq):
    out = []
    if not seq:
        return out
    s = 0
    prev = seq[0]
    for i in range(1, len(seq)):
        if seq[i] != prev:
            out.append((s, i, prev))
            s = i
            prev = seq[i]
    out.append((s, len(seq), prev))
    return out


def rebuild(dataset_path: Path, out_path: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from method3.phase1_state_seeding.subgoal_buffer import (
        SubgoalBuffer, SubgoalBufferEntry,
    )
    from method3.phase1_state_seeding.subgoal_selector import (
        Phase1SubgoalSelector, Phase1SubgoalConfig,
    )

    root = _resolve_dataset_root(str(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no data parquet under {root}/data/")

    needed = [
        "index",
        "skill.natural_language", "skill.type",
        "skill.goal_position.robot_xyzrpy",
        "observation.ee_pos.robot_xyzrpy",
    ]
    schema = pq.read_schema(data_files[0]).names
    cols = [c for c in needed if c in schema]
    missing = [c for c in needed if c not in cols]
    if missing:
        raise RuntimeError(f"dataset missing columns: {missing}")
    tbl = pa.concat_tables([pq.read_table(f, columns=cols) for f in data_files])
    tbl = tbl.sort_by("index")
    nl = [str(v) if v else "" for v in tbl["skill.natural_language"].to_pylist()]
    skill_types = [str(v) if v else "move" for v in tbl["skill.type"].to_pylist()]
    goal_xyzrpy = np.asarray(
        tbl["skill.goal_position.robot_xyzrpy"].to_pylist(), dtype=np.float64
    )
    ee_xyzrpy = np.asarray(
        tbl["observation.ee_pos.robot_xyzrpy"].to_pylist(), dtype=np.float64
    )

    ep_meta = []
    for f in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        ep_meta.extend(pq.read_table(
            f, columns=["episode_index", "dataset_from_index", "dataset_to_index"]
        ).to_pylist())
    ep_meta.sort(key=lambda r: int(r["episode_index"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = SubgoalBuffer()
    buf.set_file(out_path)
    sel = Phase1SubgoalSelector(buf, Phase1SubgoalConfig())

    total_entries = 0
    for ep in ep_meta:
        ei = int(ep["episode_index"])
        f0 = int(ep["dataset_from_index"])
        f1 = int(ep["dataset_to_index"])
        ep_id = f"episode_{ei + 1:02d}"

        ep_nl = nl[f0:f1]
        ep_st = skill_types[f0:f1]
        ep_xyz = goal_xyzrpy[f0:f1]
        ep_ee = ee_xyzrpy[f0:f1]

        segments = _run_length(ep_nl)
        for skill_idx, (s, e, seg_nl) in enumerate(segments):
            try:
                start_ee = np.asarray(ep_ee[s, :3], dtype=np.float64).reshape(3)
                goal = np.asarray(ep_xyz[s, :3], dtype=np.float64).reshape(3)
            except Exception:
                continue
            try:
                end_keys, h_star = sel._terminal_region(start_ee, goal)
            except Exception:
                continue
            buf.append(SubgoalBufferEntry(
                skill_id=f"skill_{skill_idx}",
                subgoal=goal,
                terminal_region_key=h_star,
                end_state_keys=end_keys,
                episode_id=ep_id,
                start_t=f0 + s,
                end_t=f0 + e,
                success_flag=True,
                planner_type="InterpPlan",
                phase="phase1",
                natural_language=seg_nl,
                skill_type=ep_st[s] if s < len(ep_st) else "",
            ))
            total_entries += 1

    buf.save()
    sks = sorted(buf._skills.keys(),
                 key=lambda x: int(x.split("_", 1)[1]) if "_" in x else 0)
    print(f"[rebuild] saved {total_entries} entries across {len(sks)} skills")
    for sk in sks:
        ents = buf._skills[sk]
        print(f"  {sk}: n={len(ents)}  e.g. nl='{ents[0].natural_language[:40]}' "
              f"type='{ents[0].skill_type}'")
    print(f"[rebuild] file: {out_path}")
    return total_entries


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    n = rebuild(Path(args.dataset), Path(args.out))
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
