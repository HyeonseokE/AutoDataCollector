"""End-to-end test of PlanServiceClient ↔ daemon.

Run from ``lerobot_cap`` (numpy 2.x). Daemon will auto-spawn in ``mplib_env``
(numpy 1.26) — verifying the env-split design.

Steps:
  1. Construct PlanServiceClient → daemon spawns, ``init`` succeeds.
  2. ``ping`` round-trip.
  3. ``plan_batch(n=8)`` returns 8 candidates.
  4. Inspect TrajectoryCandidate fields (waypoints, algo, seed, cost).
  5. Verify second client call reuses the live daemon (no respawn).
  6. ``close`` shuts daemon down cleanly.

Usage:
    conda activate lerobot_cap   # or any env with numpy + python
    python tools/test_plan_daemon.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perturbation.skill_level import PlanServiceClient

URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
SOCKET = f"/tmp/lerobot_planner_test_{os.getuid()}.sock"


def banner(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def main() -> int:
    if not URDF.exists():
        print(f"[FAIL] URDF not found: {URDF}")
        return 1

    cfg = dict(
        enabled=True,
        algorithms=("RRTConnect", "PRMstar", "BITstar", "KPIECE1"),
        planning_time=0.5,
        waypoint_density=0.02,
        workspace=dict(
            table_enabled=True,
            table_z=0.0,
            table_thickness=0.10,
            table_size=(2.0, 2.0),
        ),
    )

    # ── 1) Construct client (autospawn daemon) ──────────────────────────────
    banner("STEP 1: spawn daemon + init")
    t0 = time.time()
    client = PlanServiceClient(
        urdf=URDF, config=cfg, socket_path=SOCKET, n_workers=4,
    )
    spawn_t = time.time() - t0
    print(f"  spawn + init: {spawn_t:.2f}s")

    # ── 2) ping ─────────────────────────────────────────────────────────────
    banner("STEP 2: ping")
    t0 = time.time()
    pong = client.ping()
    print(f"  ping → {pong!r}  ({(time.time()-t0)*1000:.1f}ms)")
    assert pong == "pong"

    # ── 3) plan_batch ───────────────────────────────────────────────────────
    banner("STEP 3: plan_batch(n=8)")
    start = np.zeros(5)
    goal = np.array([0.3, -0.4, 0.5, -0.2, 0.0])
    t0 = time.time()
    cands = client.plan_batch(start, goal, n=8, seed=42)
    batch_t = time.time() - t0
    print(f"  plan_batch wall: {batch_t:.2f}s for {len(cands)}/8 candidates")

    if not cands:
        print("  [FAIL] no candidates returned")
        client.close()
        return 1

    # ── 4) Inspect candidates ───────────────────────────────────────────────
    print()
    print(f"  {'algo':<14} {'seed':<12} {'wp':<6} {'cost':<8} {'plan_s':<8}")
    print("  " + "-" * 50)
    for c in cands:
        print(f"  {c.algo:<14} {c.seed:<12} {c.waypoints.shape[0]:<6} "
              f"{c.cost:<8.3f} {c.plan_time_s:<8.4f}")

    algos_seen = sorted(set(c.algo for c in cands))
    print(f"\n  algorithms seen: {algos_seen}")
    print(f"  waypoint count range: {min(c.waypoints.shape[0] for c in cands)}"
          f" – {max(c.waypoints.shape[0] for c in cands)}")
    assert len(algos_seen) >= 2, "expected algorithm variation across batch"

    # ── 5) Reuse daemon — second client should NOT respawn ──────────────────
    banner("STEP 5: second client reuses live daemon (no respawn)")
    t0 = time.time()
    # autospawn is True but daemon is alive, so it should just connect+init.
    client2 = PlanServiceClient(
        urdf=URDF, config=cfg, socket_path=SOCKET, n_workers=4,
    )
    reuse_t = time.time() - t0
    print(f"  reuse-connect + re-init: {reuse_t:.2f}s")
    print(f"  ping → {client2.ping()!r}")
    # Important: client2 should NOT own the daemon (it was already running)
    print(f"  client2._owns_daemon = {client2._owns_daemon}  (expected False)")
    client2.close()  # close socket only, daemon stays up

    # ── 6) Final shutdown ───────────────────────────────────────────────────
    banner("STEP 6: shutdown daemon (owned by client1)")
    client.close()
    print("  daemon shutdown sent")

    # Verify socket cleaned up
    time.sleep(0.5)
    if os.path.exists(SOCKET):
        print(f"  [WARN] socket file still exists: {SOCKET}")
    else:
        print(f"  socket file cleaned up: {SOCKET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
