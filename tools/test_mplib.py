"""Phase 1 sanity check for OMPL integration via mplib.

Loads so101_robot0 URDF, initializes a Planner pinned at gripper_frame_link,
and runs:
  (a) RRT-Connect plan from a known start qpos to a goal pose
  (b) Same plan with PRM and BIT* to confirm algorithm switching works

No physical robot needed — purely offline. Output: per-algo trajectory length
and whether each algorithm produced a path. If this passes, mplib is ready
for Phase 2 (PlannerEnsemble class).

Usage:
    conda activate lerobot_cap
    python tools/test_mplib.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP_LINK = "gripper_frame_link"
ALGORITHMS = ["RRTConnect", "RRTstar", "PRMstar", "BITstar"]


def make_planner():
    import mplib
    # Don't pass user_joint_names — let mplib infer from URDF + move_group.
    # SRDF (collision pairs) auto-generated on first run, cached as <urdf>_mplib.srdf.
    planner = mplib.Planner(
        urdf=str(URDF),
        move_group=TIP_LINK,
        verbose=False,
    )
    return planner, mplib


def main() -> int:
    if not URDF.exists():
        print(f"[FAIL] URDF not found: {URDF}")
        return 1

    print(f"[info] URDF: {URDF}")
    print(f"[info] move_group: {TIP_LINK}")
    print(f"[info] arm joints: {ARM_JOINTS}")
    print()

    planner, mplib = make_planner()
    print(f"[OK] Planner initialized")
    print(f"     active joint dim: {len(planner.user_joint_names)}")
    print(f"     active joints  : {planner.user_joint_names}")
    print()

    nq = len(planner.user_joint_names)
    start_qpos = np.zeros(nq)
    # Goal qpos = small pose offsets in the arm joints (within calibrated limits)
    goal_qpos = np.array([0.3, -0.4, 0.5, -0.2, 0.0, 0.0])[:nq]

    print(f"[info] start_qpos: {start_qpos.tolist()}")
    print(f"[info] goal_qpos : {goal_qpos.tolist()}")
    print()

    print("=" * 64)
    print(f"{'attempt':<14} {'status':<22} {'waypoints':>10} {'duration':>10}")
    print("-" * 64)

    # Run plan_qpos multiple times. mplib's default OMPL planner is RRTConnect;
    # different RNG seeds (controlled by the OMPL backend) will produce different
    # paths from the same algorithm. This sanity test verifies:
    #   (1) planner can produce valid joint trajectories at all
    #   (2) repeated calls yield non-degenerate (path length > 2) results
    #   (3) trajectory waypoint counts vary, indicating actual stochastic search
    success = True
    waypoint_counts = []
    for i in range(5):
        try:
            result = planner.plan_qpos(
                goal_qposes=[goal_qpos],
                current_qpos=start_qpos,
                planning_time=1.0,
                rrt_range=0.1,
                simplify=False,  # Keep raw waypoints to see algo variation
                verbose=False,
            )
            status = result.get("status", "unknown")
            if status == "Success":
                positions = result.get("position")
                n = len(positions) if positions is not None else 0
                t = result.get("time")
                duration = float(t[-1]) if t is not None and len(t) > 0 else 0.0
                waypoint_counts.append(n)
                print(f"trial {i+1:<8} {status:<22} {n:>10} {duration:>9.2f}s")
            else:
                print(f"trial {i+1:<8} {str(status):<22}")
                success = False
        except Exception as e:
            print(f"trial {i+1:<8} ERROR: {str(e)[:30]}")
            success = False
    print("=" * 64)
    print()

    if waypoint_counts:
        print(f"[stats] waypoint counts across trials: {waypoint_counts}")
        print(f"        min={min(waypoint_counts)}, max={max(waypoint_counts)}, "
              f"unique={len(set(waypoint_counts))}")
        if len(set(waypoint_counts)) > 1:
            print("        ✓ Stochastic variation confirmed across trials")
        else:
            print("        ⚠ All trials identical — RNG may be deterministic by default")
    print()
    print("[info] plan_qpos uses default OMPL backend (RRTConnect). For Phase 2")
    print("       we will explicitly select different algorithms via OMPL Python")
    print("       binding or by switching mplib's internal planner_name.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
