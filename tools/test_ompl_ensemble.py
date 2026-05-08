"""POC: Sequential vs parallel OMPL ensemble planning.

Confirms two things:
  1. We can pick OMPL algorithms directly (RRTConnect / PRMstar / BITstar / KPIECE1)
     while reusing mplib's URDF-based collision checker for state validity.
  2. multiprocessing.Pool gives near-linear speedup across algorithms.

Usage:
    conda activate mplib_env
    python tools/test_ompl_ensemble.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
SRDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0_mplib.srdf"

ALGORITHMS = ["RRTConnect", "PRMstar", "BITstar", "KPIECE1"]
PLANNING_TIME = 1.0
SEED = 42


def _build_state_space(joint_limits: np.ndarray):
    """Build OMPL real-vector state space matching the move_group joint limits."""
    from ompl import base as ob
    n = len(joint_limits)
    space = ob.RealVectorStateSpace(n)
    bounds = ob.RealVectorBounds(n)
    for i, (lo, hi) in enumerate(joint_limits):
        bounds.setLow(i, float(lo))
        bounds.setHigh(i, float(hi))
    space.setBounds(bounds)
    return space


def _make_planner(name: str, si):
    """Instantiate OMPL planner by name."""
    from ompl import geometric as og
    return getattr(og, name)(si)


def _plan_one(args):
    """Worker: run one (algorithm, seed) plan call. Designed for Pool.map.

    Each subprocess loads mplib + ompl independently. Returns dict.
    """
    algo, seed, start_qpos, goal_qpos = args
    import mplib
    from ompl import base as ob, util as ou

    t0 = time.time()
    planner_mp = mplib.Planner(urdf=str(URDF), move_group="gripper_frame_link", verbose=False)
    move_idx = planner_mp.move_group_joint_indices
    joint_limits = planner_mp.joint_limits[move_idx]

    space = _build_state_space(joint_limits)
    si = ob.SpaceInformation(space)

    def is_valid(state):
        # Pad up to full robot qpos using current planner state
        q = np.array([state[i] for i in range(len(joint_limits))])
        full = planner_mp.pad_move_group_qpos(q)
        planner_mp.robot.set_qpos(full, True)
        return len(planner_mp.planning_world.check_collision()) == 0

    si.setStateValidityChecker(is_valid)
    si.setup()

    pdef = ob.ProblemDefinition(si)
    s0 = space.allocState()
    sg = space.allocState()
    for i in range(len(joint_limits)):
        s0[i] = float(start_qpos[i])
        sg[i] = float(goal_qpos[i])
    pdef.setStartAndGoalStates(s0, sg)

    planner = _make_planner(algo, si)
    planner.setProblemDefinition(pdef)
    # Seed via OMPL's RNG (constant across the planner's internal random calls)
    ou.RNG.setSeed(seed)
    planner.setup()

    setup_time = time.time() - t0
    plan_t0 = time.time()
    solved = planner.solve(PLANNING_TIME)
    plan_time = time.time() - plan_t0

    if bool(solved):
        path = pdef.getSolutionPath()
        n = path.getStateCount()
    else:
        n = 0

    return {
        "algo": algo,
        "seed": seed,
        "solved": bool(solved),
        "waypoints": n,
        "setup_s": round(setup_time, 2),
        "plan_s": round(plan_time, 2),
        "wall_s": round(time.time() - t0, 2),
    }


def main() -> int:
    if not URDF.exists():
        print(f"[FAIL] URDF not found: {URDF}")
        return 1

    print(f"[info] URDF: {URDF}")
    print(f"[info] algorithms: {ALGORITHMS}")
    print(f"[info] planning_time per call: {PLANNING_TIME}s")
    print()

    # Define start/goal — must be inside calibrated joint limits (clipped per-process)
    start_qpos = np.zeros(5)
    goal_qpos = np.array([0.3, -0.4, 0.5, -0.2, 0.0])
    args_list = [(algo, SEED + i, start_qpos, goal_qpos) for i, algo in enumerate(ALGORITHMS)]

    # ── Sequential ──
    print("=" * 72)
    print("SEQUENTIAL")
    print("=" * 72)
    seq_t0 = time.time()
    seq_results = [_plan_one(a) for a in args_list]
    seq_wall = time.time() - seq_t0
    for r in seq_results:
        print(f"  {r['algo']:<14} solved={r['solved']!s:<5} "
              f"wp={r['waypoints']:<5} setup={r['setup_s']}s plan={r['plan_s']}s wall={r['wall_s']}s")
    print(f"  TOTAL wall: {seq_wall:.2f}s")
    print()

    # ── Parallel ──
    print("=" * 72)
    print(f"PARALLEL (multiprocessing.Pool, {len(ALGORITHMS)} workers)")
    print("=" * 72)
    par_t0 = time.time()
    with mp.Pool(processes=len(ALGORITHMS)) as pool:
        par_results = pool.map(_plan_one, args_list)
    par_wall = time.time() - par_t0
    for r in par_results:
        print(f"  {r['algo']:<14} solved={r['solved']!s:<5} "
              f"wp={r['waypoints']:<5} setup={r['setup_s']}s plan={r['plan_s']}s wall={r['wall_s']}s")
    print(f"  TOTAL wall: {par_wall:.2f}s")
    print()

    # ── Speedup ──
    print("=" * 72)
    if seq_wall > 0:
        speedup = seq_wall / par_wall
        print(f"  speedup: {speedup:.2f}× ({seq_wall:.2f}s → {par_wall:.2f}s)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
