"""Sanity test for PlannerEnsemble + ParallelEnsemble.

Single-process: a few RNG-driven plans → expect varying algos / seeds / waypoints.
Parallel:        plan_batch(n=8) with persistent worker pool → expect ~4× speedup
                 over single-process at the same n.

Usage:
    conda activate mplib_env
    python tools/test_planner_ensemble.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perturbation.skill_level import (
    PlannerEnsemble,
    PlannerEnsembleConfig,
    ParallelEnsemble,
)

URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"


def main() -> int:
    if not URDF.exists():
        print(f"[FAIL] URDF not found: {URDF}")
        return 1

    cfg = PlannerEnsembleConfig(
        enabled=True,
        algorithms=("RRTConnect", "PRMstar", "BITstar", "KPIECE1"),
        planning_time=0.5,
        waypoint_density=0.02,
    )

    # ── Single-process: 4 plan calls, RNG-driven algo selection ─────────────
    print("=" * 72)
    print("SINGLE-PROCESS  PlannerEnsemble.plan() x 4")
    print("=" * 72)
    ens = PlannerEnsemble(urdf=URDF, config=cfg)
    print(f"  dof: {ens.dof}, joint_limits shape: {ens.joint_limits.shape}")

    rng = np.random.default_rng(42)
    start = np.zeros(ens.dof)
    goal = np.array([0.3, -0.4, 0.5, -0.2, 0.0, 0.0])[: ens.dof]

    seq_t0 = time.time()
    for i in range(4):
        t = ens.plan(start, goal, rng=rng)
        if t is None:
            print(f"  trial {i+1}: FAILED")
            continue
        print(f"  trial {i+1}: algo={t.algo:<12} seed={t.seed:<11} "
              f"wp={t.waypoints.shape[0]:<5} cost={t.cost:.3f} plan={t.plan_time_s:.3f}s")
    seq_wall = time.time() - seq_t0
    print(f"  TOTAL wall: {seq_wall:.2f}s")
    print()

    # ── Parallel: 8 candidates via persistent pool ──────────────────────────
    print("=" * 72)
    print("PARALLEL  ParallelEnsemble.plan_batch(n=8) — persistent worker pool")
    print("=" * 72)
    pool_t0 = time.time()
    pool = ParallelEnsemble(urdf=URDF, config=cfg, n_workers=4)
    pool_init = time.time() - pool_t0

    rng_par = np.random.default_rng(42)
    batch_t0 = time.time()
    cands = pool.plan_batch(start, goal, n=8, rng=rng_par)
    batch_wall = time.time() - batch_t0
    pool.close()

    print(f"  pool init (one-time worker setup): {pool_init:.2f}s")
    print(f"  plan_batch wall: {batch_wall:.2f}s for {len(cands)}/8 candidates")
    if cands:
        algos_seen = sorted(set(c.algo for c in cands))
        wp_counts = [c.waypoints.shape[0] for c in cands]
        plan_times = [c.plan_time_s for c in cands]
        print(f"  algos seen: {algos_seen}")
        print(f"  waypoint counts: min={min(wp_counts)} max={max(wp_counts)} "
              f"unique={len(set(wp_counts))}")
        print(f"  plan times (s): min={min(plan_times):.3f} max={max(plan_times):.3f}")
    print()

    # ── Speedup vs single-process equivalent ────────────────────────────────
    seq_per_plan = seq_wall / 4
    par_per_plan = batch_wall / max(len(cands), 1)
    print("=" * 72)
    print("SPEEDUP")
    print("=" * 72)
    print(f"  single-process per-plan: {seq_per_plan:.3f}s")
    print(f"  parallel    per-plan: {par_per_plan:.3f}s "
          f"(worker setup amortized: {pool_init / 4:.3f}s charged once)")
    if par_per_plan > 0:
        print(f"  effective speedup: {seq_per_plan / par_per_plan:.2f}×")
    print()

    return 0 if cands else 1


if __name__ == "__main__":
    sys.exit(main())
