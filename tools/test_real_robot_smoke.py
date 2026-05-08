"""Real-robot smoke test for skill-level OMPL perturbation.

Connects to robot0, attaches a PlanServiceClient with skill perturbation
ENABLED, and performs ONE short transit:

    move_to_initial_state()    ← starting baseline (no perturbation)
    move_to_position(target)   ← perturbed transit (OMPL-replaced trajectory)
    move_to_initial_state()    ← return to safe pose (no perturbation)

The middle move_to_position has is_transit=True + RNG attached + client
attached, so per skill_perturbation logic, OMPL will be invoked.

Safety:
  * Single short transit. No pick/place. No table contact.
  * target chosen at modest reach + safe z (≥10cm above table).
  * Cartesian-fallback is automatic on any daemon error.
  * If fingertip dives below z=2cm during execution, the controller halts
    on its own (existing motor protection).

PRE-FLIGHT CHECK BEFORE RUNNING:
  - Workspace is clear of objects in front of robot0.
  - Robot is powered, motors responsive (test with `! ls /dev/ttyACM*`).
  - You are watching the arm in person and can hit e-stop.

Usage:
    conda activate lerobot_cap
    python tools/test_real_robot_smoke.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from perturbation.skill_level import PlanServiceClient
from skills.skills_lerobot import LeRobotSkills

URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
SOCKET = f"/tmp/lerobot_planner_smoke_{os.getuid()}.sock"


def main() -> int:
    print("=" * 72)
    print("REAL-ROBOT SMOKE TEST — skill-level OMPL perturbation")
    print("=" * 72)
    print()
    print("This script will:")
    print("  1. Connect to robot0 (/dev/ttyACM1)")
    print("  2. Move to initial state (no perturbation)")
    print("  3. Run ONE perturbed transit to a safe forward target")
    print("  4. Return to initial state and disconnect")
    print()
    print("Safety: workspace must be clear; you must be watching the arm.")
    print()
    response = input("Type 'GO' to proceed (or anything else to abort): ").strip()
    if response != "GO":
        print("Aborted.")
        return 0

    # ── Skills setup ────────────────────────────────────────────────────────
    print("\n[1/5] Connecting to robot0...")
    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot0.yaml",
        frame="base_link",
        verbose=True,
    )
    skills.connect()

    # ── Plan service ────────────────────────────────────────────────────────
    print("\n[2/5] Spawning OMPL daemon...")
    cfg = dict(
        enabled=True,
        algorithms=("RRTConnect", "PRMstar", "BITstar"),  # skip KPIECE1 (wild paths)
        planning_time=0.5,
        waypoint_density=0.02,
        workspace=dict(
            table_enabled=True, table_z=0.0, table_thickness=0.10,
            table_size=(2.0, 2.0), table_mount_links=("base_link",),
        ),
    )
    client = PlanServiceClient(
        urdf=URDF, config=cfg, socket_path=SOCKET, n_workers=2,
    )
    skills.set_skill_planner_client(client, n_candidates=4)
    skills.set_perturbation_rng(0)  # required for skill perturbation to fire

    # ── Move to initial (no perturbation — start baseline) ──────────────────
    print("\n[3/5] Moving to initial state (baseline, no perturbation)...")
    # Detach perturbation temporarily for this baseline move
    saved_client = skills._skill_planner_client
    skills._skill_planner_client = None
    skills.move_to_initial_state()
    skills._skill_planner_client = saved_client
    time.sleep(0.5)

    # ── Perturbed transit ───────────────────────────────────────────────────
    print("\n[4/5] Running PERTURBED transit to safe forward target...")
    # Safe target: 25cm forward, 0 lateral, 20cm up — well clear of table.
    target = [0.25, 0.0, 0.20]
    print(f"  target: {target}")
    print(f"  expecting [Skill Perturbation] log line indicating OMPL trajectory")
    ok = skills.move_to_position(
        target,
        target_name="smoke_test_target",
        skill_description="OMPL perturbation smoke test transit",
        verification_question="Did the gripper reach the safe forward position?",
        is_transit=True,
    )
    print(f"  move_to_position returned: {ok}")

    # ── Return to safe pose (no perturbation) ──────────────────────────────
    print("\n[5/5] Returning to initial state...")
    skills._skill_planner_client = None
    skills.move_to_initial_state()
    skills._skill_planner_client = saved_client

    # ── Cleanup ─────────────────────────────────────────────────────────────
    print("\n[cleanup] Disconnecting...")
    skills.disconnect()
    client.close()
    print()
    print("=" * 72)
    print("DONE — review the [Skill Perturbation] line above to confirm OMPL fired")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
