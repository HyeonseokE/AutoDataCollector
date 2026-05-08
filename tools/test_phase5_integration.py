"""Phase 5 integration smoke test (no robot motion).

Verifies the wiring without actually moving the robot:
  1. Read recording_config_ws1.yaml.
  2. Build a PlanServiceClient from the perturbation.skill section.
  3. Daemon auto-spawns; plan_batch returns trajectories.
  4. Manually simulate the swap inside move_to_position by calling
     time_parameterize_trajectory and constructing a Trajectory dataclass —
     proves the pieces compose without any motion.

Usage:
    conda activate lerobot_cap
    python tools/test_phase5_integration.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CFG_PATH = PROJECT_ROOT / "pipeline_config" / "recording_config_ws1.yaml"
URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
SOCKET = f"/tmp/lerobot_planner_phase5_{os.getuid()}.sock"


def main() -> int:
    if not CFG_PATH.exists():
        print(f"[FAIL] config not found: {CFG_PATH}")
        return 1
    if not URDF.exists():
        print(f"[FAIL] URDF not found: {URDF}")
        return 1

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    skill_raw = (cfg.get("perturbation") or {}).get("skill") or {}
    print(f"[config] perturbation.skill.enabled = {skill_raw.get('enabled')}")
    print(f"[config] algorithms = {skill_raw.get('algorithms')}")
    print(f"[config] n_candidates = {skill_raw.get('n_candidates')}")
    print(f"[config] workspace.table_z = "
          f"{(skill_raw.get('workspace') or {}).get('table_z')}")
    print()

    # Build the planner config the same way execution_forward_and_reset does
    planner_cfg = dict(
        enabled=True,
        algorithms=tuple(skill_raw.get("algorithms",
            ("RRTConnect", "PRMstar", "BITstar", "KPIECE1"))),
        planning_time=float(skill_raw.get("planning_time", 0.5)),
        waypoint_density=float(skill_raw.get("waypoint_density", 0.02)),
        workspace=skill_raw.get("workspace") or None,
    )
    ws = planner_cfg.get("workspace")
    if isinstance(ws, dict):
        if ws.get("table_size") is not None:
            ws["table_size"] = tuple(ws["table_size"])
        if ws.get("table_mount_links") is not None:
            ws["table_mount_links"] = tuple(ws["table_mount_links"])

    from perturbation.skill_level import PlanServiceClient
    print("[1/4] Spawning daemon + init...")
    t0 = time.time()
    client = PlanServiceClient(
        urdf=URDF, config=planner_cfg, socket_path=SOCKET,
        n_workers=int(skill_raw.get("n_workers", 4)),
    )
    print(f"      ready in {time.time()-t0:.2f}s")

    print("\n[2/4] plan_batch from current_joints (zeros) → goal_joints")
    start = np.zeros(5)
    goal = np.array([0.3, -0.4, 0.5, -0.2, 0.0])
    n = int(skill_raw.get("n_candidates", 4))
    t0 = time.time()
    cands = client.plan_batch(start, goal, n=n, seed=42)
    print(f"      plan_batch wall: {time.time()-t0:.2f}s, "
          f"{len(cands)}/{n} candidates")
    for c in cands:
        print(f"        {c.algo:<14} wp={c.waypoints.shape[0]:<4} cost={c.cost:.3f}")

    print("\n[3/4] Pick a candidate (RNG-driven), build Trajectory dataclass")
    rng = np.random.default_rng(99)
    chosen = cands[int(rng.integers(0, len(cands)))]
    from lerobot_cap.planning.interpolation import time_parameterize_trajectory
    from lerobot_cap.planning.trajectory import Trajectory

    new_joints = np.asarray(chosen.waypoints, dtype=float)
    new_ts, _ = time_parameterize_trajectory(new_joints, max_velocity=1.0,
                                             max_acceleration=2.0)
    traj = Trajectory(
        joint_positions=new_joints,
        timestamps=new_ts,
        ik_converged=True,
        target_position=None,
    )
    print(f"      built Trajectory: {traj.num_points} points, "
          f"duration={traj.duration:.2f}s, algo={chosen.algo}")

    print("\n[4/4] Cleanup")
    client.close()
    print("      daemon shutdown sent")
    print()
    print("=" * 64)
    print("PASS — config → client → daemon → plan → Trajectory all wire.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
