"""Sanity test — curobo MotionPlanner with SO-101 URDF.

Verifies the auto-generated robot config (robot_configs/curobo/so101_robot0.yml)
loads and plans a single trajectory. Mirrors the franka.yml smoke test but with
our 5-DoF arm. Output reports waypoint count, duration, success rate, and plan
latency for comparison against the OMPL backend.

Usage:
    conda activate lerobot_cap
    python tools/test_curobo_so101.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


CFG_PATH = PROJECT_ROOT / "robot_configs" / "curobo" / "so101_robot0.yml"


def main() -> int:
    if not CFG_PATH.exists():
        print(f"[FAIL] curobo robot config missing: {CFG_PATH}")
        return 1

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    import inspect
    print("plan_cspace signature:")
    print(inspect.signature(MotionPlanner.plan_cspace))
    print()

    print("[init] Creating MotionPlannerCfg from SO-101 config...")
    t0 = time.time()
    cfg = MotionPlannerCfg.create(
        robot=str(CFG_PATH),
        num_trajopt_seeds=4,
        num_ik_seeds=16,
        random_seed=42,
        use_cuda_graph=False,   # disabled: lets us swap plan_pose / plan_cspace freely
    )
    planner = MotionPlanner(cfg)
    planner.warmup(enable_graph=False, num_warmup_iterations=3)
    print(f"[init] ready in {time.time() - t0:.2f}s")
    print(f"[info] joint_names: {planner.joint_names}")
    print(f"[info] tool_frames: {planner.tool_frames}")

    # Start: default joint state (typically zeros)
    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    print(f"[info] start qpos shape: {q_start.position.shape}")
    print(f"[info] start qpos: {q_start.position.cpu().numpy().tolist()}")

    # Goal: 5-joint qpos matching the OMPL smoke test target.
    # plan_cspace takes a JointState goal (no IK needed).
    import numpy as np
    goal_q = torch.tensor(
        [[0.3, -0.4, 0.5, -0.2, 0.0, 0.0]],   # 5 arm + 1 gripper
        device="cuda", dtype=torch.float32,
    )
    goal_state = JointState.from_position(goal_q, joint_names=planner.joint_names)
    print(f"[info] goal qpos: {goal_q.cpu().numpy().tolist()}")

    print("\n[plan] Running plan_cspace...")
    n_trials = 5
    for i in range(n_trials):
        t0 = time.time()
        result = planner.plan_cspace(goal_state, q_start)
        wall = time.time() - t0
        if result is not None and bool(result.success.any()):
            interp = result.get_interpolated_plan()
            n = interp.position.shape[-2]
            dt = planner.trajopt_solver.config.interpolation_dt
            print(f"  trial {i+1}: ✓ wp={n:<4} duration={n*dt:.2f}s wall={wall*1000:.1f}ms")
        else:
            err = str(getattr(result, "status", "no status"))
            print(f"  trial {i+1}: ✗ wall={wall*1000:.1f}ms  status={err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
