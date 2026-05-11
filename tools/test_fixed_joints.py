"""Verify that fixed_joint_indices truly holds the listed joints constant.

For SO-101 we expect wrist_roll (joint index 4) to stay at the start value
across every waypoint of every candidate trajectory, with no wiggle.

Without fixed_joint_indices, RRTConnect tends to wiggle wrist_roll by 0.05~0.3
rad on average (random tree expansion). With ``fixed_joint_indices=[4]``,
the wrist_roll column of every waypoint must equal the start value exactly.

Usage:
    conda activate mplib_env
    python tools/test_fixed_joints.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perturbation.skill_level import (
    PlannerEnsemble,
    PlannerEnsembleConfig,
)

URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"


def planner_with(fixed_indices):
    cfg = PlannerEnsembleConfig(
        enabled=True,
        algorithms=("RRTConnect", "PRMstar", "BITstar"),
        planning_time=0.5,
        waypoint_density=0.02,
        workspace=dict(
            table_enabled=True, table_z=0.0, table_thickness=0.10,
            table_size=(2.0, 2.0),
        ),
        fixed_joint_indices=tuple(fixed_indices),
    )
    return PlannerEnsemble(urdf=URDF, config=cfg)


def stats_for_joint(planner, joint_idx, n_trials=6, seed_base=42):
    rng = np.random.default_rng(seed_base)
    start = np.array([0.0, -0.3, 0.4, -0.1, 0.5])   # non-zero wrist_roll so motion *could* drift
    goal = np.array([0.3, -0.4, 0.5, -0.2, 0.5])   # same wrist_roll
    deltas = []
    candidates = []
    for _ in range(n_trials):
        c = planner.plan(start, goal, rng=rng)
        if c is None:
            continue
        candidates.append(c)
        wrist_roll_track = c.waypoints[:, joint_idx]
        deltas.append(np.max(np.abs(wrist_roll_track - start[joint_idx])))
    return candidates, deltas


def main() -> int:
    print("=" * 72)
    print("WITHOUT fixed_joint_indices (baseline; wrist_roll free to wiggle)")
    print("=" * 72)
    p_free = planner_with([])
    cands_free, deltas_free = stats_for_joint(p_free, joint_idx=4)
    for c, d in zip(cands_free, deltas_free):
        print(f"  {c.algo:<12} wp={c.waypoints.shape[0]:<5} "
              f"max|wrist_roll − start|: {d*1000:.2f} mrad")
    print(f"  → mean = {np.mean(deltas_free)*1000:.2f} mrad, "
          f"max = {np.max(deltas_free)*1000:.2f} mrad")
    print()

    print("=" * 72)
    print("WITH fixed_joint_indices=[4]  (wrist_roll held at start value)")
    print("=" * 72)
    p_fix = planner_with([4])
    cands_fix, deltas_fix = stats_for_joint(p_fix, joint_idx=4)
    for c, d in zip(cands_fix, deltas_fix):
        print(f"  {c.algo:<12} wp={c.waypoints.shape[0]:<5} "
              f"max|wrist_roll − start|: {d*1000:.2f} mrad")
    print(f"  → mean = {np.mean(deltas_fix)*1000:.2f} mrad, "
          f"max = {np.max(deltas_fix)*1000:.2f} mrad")
    print()

    print("=" * 72)
    print("SANITY: other joints (0..3) STILL planned over")
    print("=" * 72)
    # Cumulative distance traversed in active dims for the first candidate
    if cands_fix:
        wp = cands_fix[0].waypoints
        for j in range(wp.shape[1]):
            track = wp[:, j]
            traveled = float(np.sum(np.abs(np.diff(track))))
            print(f"  joint {j} total travel: {traveled:.4f} rad")

    ok = np.max(deltas_fix) < 1e-6
    print()
    print("RESULT:", "PASS — wrist_roll truly fixed" if ok else "FAIL — wrist_roll still moved")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
