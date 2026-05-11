"""Real-robot smoke test for skill-level **CUROBO** perturbation (K_max=2).

Mirror of ``test_real_robot_smoke.py`` but using the GPU-accelerated
curobo backend with **multi-via mode** (max_vias_per_candidate=2).
Connects to robot0, attaches a CuroboBackend with skill perturbation
ENABLED, and performs FOUR short transits so we observe (a) the graph-
compile-vs-cached wall-time difference and (b) the random K∈{1,2}
mix in production:

    move_to_initial_state()          ← baseline (perturbation detached)
    move_to_position(target_A)       ← perturbed #1 (graph compile)
    move_to_position(target_B)       ← perturbed #2 (graph cached)
    move_to_position(target_C)       ← perturbed #3 (different mix)
    move_to_position(target_D)       ← perturbed #4 (different mix)
    move_to_initial_state()          ← return to safe pose

With K_max=2 and 4 candidates per transit, slot 0 is always direct
(K=0) and slots 1-3 each sample K∈{1,2} uniformly — so across 4
transits we expect to see a mix of "curobo:direct", "curobo:via1_s*",
and "curobo:via2_s*" lines in the [Skill Perturbation] log.

Both perturbed moves go through the exact production path
(``skills_lerobot.move_to_position`` with ``is_transit=True``), which calls
``client.plan_batch(start, goal, n, seed=...)``. CuroboBackend duck-types
that signature so the same call site serves both backends.

Safety
------
* Short transits, modest reach, z ≥ 10cm above table.
* Cartesian-fallback automatic on any planner error.
* Existing motor protection halts on dive-below-z=2cm.
* K=2 (two-via) paths produce wider arcs (wp up to ~180) and may
  swing further from the start-goal line than single-via paths —
  workspace must be CLEAR for the full arc envelope, not just the
  straight line.

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
    print("This script will (max_vias_per_candidate=2):")
    print("  1. Connect to robot0 (/dev/ttyACM1)")
    print("  2. Initialize CuroboBackend (≈5s warmup + 1st-plan graph compile)")
    print("  3. Move to initial state (no perturbation)")
    print("  4. Perturbed transit A  ← graph COMPILE expected (~6s plan)")
    print("  5. Perturbed transit B  ← graph CACHED (<0.2s plan), random K mix")
    print("  6. Perturbed transit C  ← cached, different K mix")
    print("  7. Perturbed transit D  ← cached, different K mix")
    print("  8. Return to initial state and disconnect")
    print()
    print("With K_max=2 you should see 'curobo:direct' AND 'curobo:via1_s*' AND")
    print("'curobo:via2_s*' across the 4 transits — confirms multi-via mode active.")
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
        max_vias_per_candidate=2,   # K_max=2 — random K ∈ {1, 2} mix
    )
    t0 = time.time()
    backend = CuroboBackend(urdf=str(URDF), config=cb_cfg)
    print(f"  CuroboBackend ready in {time.time()-t0:.2f}s (dof={backend.dof})")

    skills.set_skill_planner_client(backend, n_candidates=4)
    skills.set_perturbation_rng(0)

    # ── Move to initial (no perturbation) ──────────────────────────────────
    print("\n[3/8] Moving to initial state (baseline, no perturbation)...")
    saved_client = skills._skill_planner_client
    skills._skill_planner_client = None
    skills.move_to_initial_state()
    skills._skill_planner_client = saved_client
    time.sleep(0.5)

    # ── Four perturbed transits (target list chosen to stay in the
    #    pre-cleared workspace envelope — same z window throughout) ─────────
    targets = [
        ("A", [0.25,  0.00, 0.20]),
        ("B", [0.20, -0.10, 0.18]),
        ("C", [0.22, +0.10, 0.19]),
        ("D", [0.25,  0.00, 0.22]),
    ]
    walls: list[float] = []
    oks: list[bool] = []
    for step_idx, (tag, tgt) in enumerate(targets, start=4):
        note = (
            "graph COMPILE expected"
            if step_idx == 4
            else f"graph CACHED — observe random K∈(0,1,2) mix"
        )
        print(f"\n[{step_idx}/8] PERTURBED transit #{step_idx-3} — target {tag} ({note})")
        print(f"  target_{tag.lower()}: {tgt}")
        t0 = time.time()
        ok = skills.move_to_position(
            tgt,
            target_name=f"curobo_smoke_target_{tag}",
            skill_description=f"CUROBO K_max=2 smoke transit #{step_idx-3}",
            verification_question=f"Did the gripper reach safe position {tag}?",
            is_transit=True,
        )
        wall = time.time() - t0
        oks.append(ok)
        walls.append(wall)
        print(f"  move_to_position #{step_idx-3} returned ok={ok}, wall={wall:.2f}s")
        time.sleep(0.3)

    # ── Return to safe pose ────────────────────────────────────────────────
    print("\n[8/8] Returning to initial state (no perturbation)...")
    skills._skill_planner_client = None
    skills.move_to_initial_state()
    skills._skill_planner_client = saved_client

    # ── Cleanup ────────────────────────────────────────────────────────────
    print("\n[cleanup] Disconnecting...")
    skills.disconnect()

    print()
    print("=" * 72)
    print("DONE — review the [Skill Perturbation] log lines for each transit:")
    print("  expect a mix of 'curobo:direct', 'curobo:via1_s*', 'curobo:via2_s*'")
    for i, (tag, _) in enumerate(targets):
        marker = "compile" if i == 0 else "cached"
        print(f"  transit #{i+1} ({tag}, {marker}): wall={walls[i]:.2f}s")
    if len(walls) >= 2 and walls[0] > 0:
        amortized = sum(walls[1:]) / max(len(walls) - 1, 1)
        print(f"  amortized cached wall (txns 2..N): {amortized:.2f}s")
    print("=" * 72)
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
