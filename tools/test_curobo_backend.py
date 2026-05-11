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

    # Parallelism sanity — confirm GPU is being used and not just CPU.
    import torch
    torch.cuda.synchronize()
    # Time three pieces explicitly so the breakdown is visible:
    print()
    print("[parallelism breakdown — 3rd call]")
    # IK retry loop (sequential): measure isolated
    start_full = backend._to_full_qpos(start)
    goal_full = backend._to_full_qpos(goal)
    s_ee, s_q = backend._compute_ee_xyz_quat(start_full)
    g_ee, g_q = backend._compute_ee_xyz_quat(goal_full)
    rng_t = np.random.default_rng(11)
    via_xyzs = [backend._sample_via_xyz(s_ee, g_ee, k, rng_t) for k in range(3)]

    # (a) Old path: 3 separate IK calls (each goalset=5 internally).
    t0 = time.time()
    via_qs_seq = [backend._ik_solve(v, s_q, g_q, seed_q=start_full) for v in via_xyzs]
    torch.cuda.synchronize()
    ik_seq_time = time.time() - t0
    print(f"  3-via IK SEQUENTIAL (3 calls × goalset=5):    {ik_seq_time*1000:.1f}ms")

    # (b) New path: 1 batched IK call with batch=3, goalset=5 (used by plan_batch).
    t0 = time.time()
    via_qs = backend._ik_solve_batched(via_xyzs, s_q, g_q, seed_q=start_full)
    torch.cuda.synchronize()
    ik_time = time.time() - t0
    print(f"  3-via IK BATCHED   (1 call,  batch=3×goalset=5): {ik_time*1000:.1f}ms "
          f"({ik_seq_time/max(ik_time,1e-6):.1f}× vs sequential)")

    # plan_cspace merged seg1+seg2 — one batched GPU call of size 2N
    # (this is exactly what plan_batch uses internally).
    via_qs_filled = [vq if vq is not None else (start_full + goal_full)/2 for vq in via_qs]
    N = backend._batch_size
    merged_start = [start_full]*N + [goal_full] + via_qs_filled    # seg1 ⊕ seg2 starts
    merged_goal = [goal_full] + via_qs_filled + [goal_full]*N      # seg1 ⊕ seg2 goals
    # pad to 2N if needed
    pad = backend._cspace_batch - len(merged_start)
    merged_start += [start_full]*pad
    merged_goal += [start_full]*pad
    t0 = time.time()
    _ = backend._plan_cspace_batch(merged_start, merged_goal)
    torch.cuda.synchronize()
    merged_time = time.time() - t0
    print(f"  plan_cspace MERGED seg1+seg2 (batch={backend._cspace_batch}, 1 GPU call): "
          f"{merged_time*1000:.1f}ms")
    print(f"  total plan stage (1 merged call): {merged_time*1000:.1f}ms "
          f"+ IK batched {ik_time*1000:.1f}ms = {(merged_time+ik_time)*1000:.1f}ms")

    print()
    # Use the SECOND call (graph cached, no warmup confound) for diversity stats
    cands_to_score = cands2 if cands2 else cands
    if len(cands_to_score) >= 2:
        costs = np.array([c.cost for c in cands_to_score])
        unique_costs = len(np.unique(np.round(costs, 2)))
        print(f"costs={[f'{c:.3f}' for c in costs]}")
        print(f"  unique≈{unique_costs}/{len(costs)}  "
              f"(spread {costs.max()-costs.min():.3f} rad)")
        if unique_costs >= 3:
            print("  ✓ diversity strong (3+ distinct paths) — Fix 5 effective")
        elif unique_costs >= 2:
            print("  ⚠ diversity moderate (only 2 distinct) — some via still IK-failed")
        else:
            print("  ✗ candidates identical — via-IK collapsed onto midpoint")

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
