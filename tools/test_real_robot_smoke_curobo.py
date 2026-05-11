"""Real-robot smoke test for skill-level **CUROBO** perturbation.

Mirror of ``test_real_robot_smoke.py`` but using the GPU-accelerated
curobo backend instead of OMPL subprocess daemon. Connects to robot0,
attaches a CuroboBackend with skill perturbation ENABLED, and performs
TWO short transits so we can observe the graph-compile-vs-cached
wall-time difference live:

    move_to_initial_state()          ← baseline (perturbation detached)
    move_to_position(target_A)       ← perturbed transit (1st — graph compile)
    move_to_position(target_B)       ← perturbed transit (2nd — graph cached)
    move_to_initial_state()          ← return to safe pose (no perturbation)

Both perturbed moves go through the exact production path
(``skills_lerobot.move_to_position`` with ``is_transit=True``), which calls
``client.plan_batch(start, goal, n, seed=...)``. CuroboBackend duck-types
that signature so the same call site serves both backends.

Safety
------
* Two short transits, modest reach, z ≥ 10cm above table.
* Cartesian-fallback automatic on any planner error.
* Existing motor protection halts on dive-below-z=2cm.

PRE-FLIGHT
----------
* Workspace clear of objects in front of robot0.
* Robot powered, motors responsive.
* Curobo config exists at robot_configs/curobo/so101_robot0.yml
  (generated once by the curobo build script).
* You are watching the arm and can hit e-stop.

Usage:
    conda activate lerobot_cap
    python tools/test_real_robot_smoke_curobo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from perturbation.skill_level import get_curobo_backend
from skills.skills_lerobot import LeRobotSkills

URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
CUROBO_CFG = PROJECT_ROOT / "robot_configs" / "curobo" / "so101_robot0.yml"


def main() -> int:
    print("=" * 72)
    print("REAL-ROBOT SMOKE TEST — skill-level CUROBO (GPU) perturbation")
    print("=" * 72)
    print()
    print("This script will:")
    print("  1. Connect to robot0 (/dev/ttyACM1)")
    print("  2. Initialize CuroboBackend (≈5s warmup + 1st-plan graph compile)")
    print("  3. Move to initial state (no perturbation)")
    print("  4. Perturbed transit to target A  ← graph COMPILE expected (~6s plan)")
    print("  5. Perturbed transit to target B  ← graph CACHED expected (<0.2s plan)")
    print("  6. Return to initial state and disconnect")
    print()
    print("Safety: workspace must be clear; you must be watching the arm.")
    print()
    response = input("Type 'GO' to proceed (or anything else to abort): ").strip()
    if response != "GO":
        print("Aborted.")
        return 0

    if not CUROBO_CFG.exists():
        print(f"[ERROR] curobo config missing: {CUROBO_CFG}")
        print("Generate it once via the curobo robot model build script.")
        return 1

    # ── Skills setup ────────────────────────────────────────────────────────
    print("\n[1/6] Connecting to robot0...")
    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot0.yaml",
        frame="base_link",
        verbose=True,
    )
    skills.connect()

    # ── Curobo backend init ────────────────────────────────────────────────
    print("\n[2/6] Initializing CuroboBackend (GPU warmup)...")
    CuroboBackend, CuroboBackendConfig = get_curobo_backend()
    cb_cfg = CuroboBackendConfig(
        enabled=True,
        robot_cfg_path=str(CUROBO_CFG),
        num_trajopt_seeds=4,
        num_ik_seeds=16,
        use_cuda_graph=True,        # 1st plan_batch pays graph-capture cost
        max_batch_size=4,
        fixed_joint_indices=(4,),   # lock wrist_roll (parity w/ OMPL & cartesian)
        arm_joint_count=5,
        via_offset_mag=0.10,
        junction_smooth_k=5,
    )
    t0 = time.time()
    backend = CuroboBackend(urdf=str(URDF), config=cb_cfg)
    print(f"  CuroboBackend ready in {time.time()-t0:.2f}s (dof={backend.dof})")

    skills.set_skill_planner_client(backend, n_candidates=4)
    skills.set_perturbation_rng(0)

    # ── Move to initial (no perturbation) ──────────────────────────────────
    print("\n[3/6] Moving to initial state (baseline, no perturbation)...")
    saved_client = skills._skill_planner_client
    skills._skill_planner_client = None
    skills.move_to_initial_state()
    skills._skill_planner_client = saved_client
    time.sleep(0.5)

    # ── Perturbed transit #1 (expect graph-compile latency on plan) ────────
    print("\n[4/6] PERTURBED transit #1 — target A (graph COMPILE expected)...")
    target_a = [0.25, 0.0, 0.20]
    print(f"  target_a: {target_a}")
    t0 = time.time()
    ok_a = skills.move_to_position(
        target_a,
        target_name="curobo_smoke_target_A",
        skill_description="CUROBO perturbation smoke transit #1 (graph compile)",
        verification_question="Did the gripper reach the safe forward position?",
        is_transit=True,
    )
    wall_a = time.time() - t0
    print(f"  move_to_position #1 returned ok={ok_a}, wall={wall_a:.2f}s")
    time.sleep(0.5)

    # ── Perturbed transit #2 (graph cached — should be ≥10× faster on plan) ─
    print("\n[5/6] PERTURBED transit #2 — target B (graph CACHED expected)...")
    target_b = [0.20, -0.10, 0.18]
    print(f"  target_b: {target_b}")
    t0 = time.time()
    ok_b = skills.move_to_position(
        target_b,
        target_name="curobo_smoke_target_B",
        skill_description="CUROBO perturbation smoke transit #2 (graph cached)",
        verification_question="Did the gripper reach the second safe position?",
        is_transit=True,
    )
    wall_b = time.time() - t0
    print(f"  move_to_position #2 returned ok={ok_b}, wall={wall_b:.2f}s")

    # ── Return to safe pose ────────────────────────────────────────────────
    print("\n[6/6] Returning to initial state (no perturbation)...")
    skills._skill_planner_client = None
    skills.move_to_initial_state()
    skills._skill_planner_client = saved_client

    # ── Cleanup ────────────────────────────────────────────────────────────
    print("\n[cleanup] Disconnecting...")
    skills.disconnect()

    print()
    print("=" * 72)
    print("DONE — review the [Skill Perturbation] log lines for both transits")
    print(f"  transit #1 (graph compile): wall={wall_a:.2f}s")
    print(f"  transit #2 (graph cached) : wall={wall_b:.2f}s")
    if wall_a > 0 and wall_b > 0:
        # Note: wall_a/wall_b includes motion execution, not just plan time.
        # Graph caching only affects the plan portion (~6s vs ~0.2s).
        print(f"  (motion-execution time is roughly equal between the two;")
        print(f"   the difference reflects plan-time graph caching)")
    print("=" * 72)
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
