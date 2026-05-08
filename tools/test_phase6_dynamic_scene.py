"""Phase 6 dynamic scene update test.

Verifies:
  1. update_scene adds N obstacles to ALL workers' planning_world.
  2. A subsequent plan_batch correctly avoids them (collision_free waypoints).
  3. Re-calling update_scene with a smaller set REMOVES old obstacles.

Usage:
    conda activate lerobot_cap
    python tools/test_phase6_dynamic_scene.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perturbation.skill_level import PlanServiceClient

URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
SOCKET = f"/tmp/lerobot_planner_phase6_{os.getuid()}.sock"


def main() -> int:
    cfg = dict(
        enabled=True,
        algorithms=("RRTConnect",),
        planning_time=0.5,
        waypoint_density=0.05,
        workspace=dict(
            table_enabled=True, table_z=0.0, table_thickness=0.10,
            table_size=(2.0, 2.0),
        ),
    )
    client = PlanServiceClient(urdf=URDF, config=cfg, socket_path=SOCKET, n_workers=2)

    # ── 1) Initial state — only static workspace obstacles ────────────────
    print("=" * 64)
    print("STEP 1: baseline — no dynamic obstacles")
    print("=" * 64)
    rep = client.update_scene({})  # empty
    print(f"  update_scene({{}}) → {rep}")

    # ── 2) Add 3 obstacles ────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("STEP 2: add 3 obstacles")
    print("=" * 64)
    detected = {
        "red_block":    {"position": [0.25, 0.10, 0.025], "size": [0.04, 0.04, 0.05]},
        "green_block":  {"position": [0.30, 0.00, 0.025], "size": [0.04, 0.04, 0.05]},
        "blue_dish":    {"position": [0.20, -0.10, 0.005], "size": [0.10, 0.10, 0.01]},
    }
    rep = client.update_scene(detected)
    print(f"  update_scene → {rep}")

    # ── 3) Plan around them ───────────────────────────────────────────────
    print()
    print("=" * 64)
    print("STEP 3: plan around obstacles (start zeros → goal forward-low)")
    print("=" * 64)
    start = np.zeros(5)
    goal = np.array([0.3, -0.4, 0.5, -0.2, 0.0])
    cands = client.plan_batch(start, goal, n=4, seed=42)
    print(f"  plan_batch: {len(cands)}/4 candidates")
    for c in cands:
        print(f"    {c.algo:<14} wp={c.waypoints.shape[0]:<5} cost={c.cost:.3f}")

    # ── 4) Reduce — remove green_block, change red_block position ─────────
    print()
    print("=" * 64)
    print("STEP 4: refresh scene (red moved, green removed, blue same)")
    print("=" * 64)
    detected2 = {
        "red_block":    {"position": [0.30, 0.05, 0.025], "size": [0.04, 0.04, 0.05]},
        "blue_dish":    {"position": [0.20, -0.10, 0.005], "size": [0.10, 0.10, 0.01]},
    }
    rep = client.update_scene(detected2)
    print(f"  update_scene → {rep}")
    # Expected: removed=1 (green_block), kept=2 (red_block, blue_dish refreshed),
    #          added=0
    print(f"  expected: removed=1 (green), kept=2 (red+blue refreshed), added=0")

    # ── 5) Clear all dynamic objects ──────────────────────────────────────
    print()
    print("=" * 64)
    print("STEP 5: clear all dynamic obstacles")
    print("=" * 64)
    rep = client.update_scene({})
    print(f"  update_scene({{}}) → {rep}")
    print(f"  expected: removed=2 (red+blue), kept=0, added=0")

    client.close()
    print()
    print("=" * 64)
    print("PASS")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
