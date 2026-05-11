"""CuroboBackend.plan_batch sanity — 1 direct + N-1 via-point candidates.

Reports per-candidate cost, waypoint count, plan latency, and verifies the
via-point sampling produces qualitatively different EE arcs.

Usage:
    conda activate lerobot_cap
    python tools/test_curobo_backend.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perturbation.skill_level import get_curobo_backend

URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
CFG_PATH = PROJECT_ROOT / "robot_configs" / "curobo" / "so101_robot0.yml"


def main() -> int:
    CuroboBackend, CuroboBackendConfig = get_curobo_backend()
    cfg = CuroboBackendConfig(
        enabled=True,
        robot_cfg_path=str(CFG_PATH),
        num_trajopt_seeds=4,
        num_ik_seeds=16,
        use_cuda_graph=True,    # backend now sets curobo.runtime.cuda_graph_reset=True
        max_batch_size=4,
        fixed_joint_indices=(4,),
        arm_joint_count=5,
    )

    print("[init] Loading CuroboBackend...")
    t0 = time.time()
    backend = CuroboBackend(urdf=str(URDF), config=cfg)
    print(f"[init] ready in {time.time()-t0:.2f}s, dof={backend.dof}")
    print(f"[init] joint_names={backend.joint_names}")

    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.3, -0.4, 0.5, -0.2, 0.0])
    rng = np.random.default_rng(42)

    print(f"\n[task] start={start.tolist()}")
    print(f"[task] goal ={goal.tolist()}\n")

    print("=" * 64)
    print("plan_batch(n=4)  — call 1 (graph compile + plan)")
    print("=" * 64)
    t0 = time.time()
    cands = backend.plan_batch(start, goal, n=4, rng=rng)
    wall1 = time.time() - t0

    print(f"\nbatch wall: {wall1:.3f}s, {len(cands)}/4 succeeded\n")
    print(f"{'algo':<22} {'wp':<6} {'cost':<8} {'plan_s':<8} {'wrist_roll_dev':<14}")
    print("-" * 60)
    for c in cands:
        # wrist_roll = column index 4 (the locked joint)
        wr_dev_rad = float(np.max(np.abs(c.waypoints[:, 4] - start[4]))) \
            if c.waypoints.shape[1] > 4 else 0.0
        print(f"{c.algo:<22} {c.waypoints.shape[0]:<6} {c.cost:<8.3f} "
              f"{c.plan_time_s:<8.3f} {wr_dev_rad*1000:>8.2f} mrad")

    print()
    print("=" * 64)
    print("plan_batch(n=4)  — call 2 (graph cached, should be ~10× faster)")
    print("=" * 64)
    t0 = time.time()
    cands2 = backend.plan_batch(start, goal, n=4, rng=np.random.default_rng(99))
    wall2 = time.time() - t0
    print(f"batch wall: {wall2:.3f}s, {len(cands2)}/4 succeeded")
    if wall1 > 0:
        print(f"speedup vs call 1: {wall1/wall2:.2f}×")

    print()
    if len(cands) >= 2:
        costs = np.array([c.cost for c in cands])
        unique_costs = len(np.unique(np.round(costs, 2)))
        print(f"cost min={costs.min():.3f} max={costs.max():.3f} "
              f"unique≈{unique_costs}/{len(costs)}")
        if unique_costs >= 2:
            print("  ✓ candidates ARE diverse (cost spread)")
        else:
            print("  ⚠ candidates look identical — via-point sampling ineffective")

    # wrist_roll lock — should be sub-mrad across all candidates
    max_wr_dev = max(
        (float(np.max(np.abs(c.waypoints[:, 4] - start[4])))
         for c in cands if c.waypoints.shape[1] > 4),
        default=0.0,
    )
    print(f"max wrist_roll deviation across all candidates: {max_wr_dev*1000:.3f} mrad "
          f"({'✓ locked' if max_wr_dev < 1e-6 else '⚠ NOT locked'})")

    return 0 if cands else 1


if __name__ == "__main__":
    sys.exit(main())
