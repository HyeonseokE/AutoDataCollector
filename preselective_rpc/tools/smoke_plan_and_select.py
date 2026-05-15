"""Smoke test for PreselectiveAcquirer.PlanAndSelect + CommitToBuffer.

Synthetic context (no robot, no cameras). Validates the full RPC path:
  1. encode/decode of numpy + image dict
  2. server-side curobo plan_batch (K candidates)
  3. server-side SmolVLA forward_fm_batched + sample_actions
  4. selector.select → trajectory pickle round-trip
  5. selection_id → CommitToBuffer flushes / drops correctly

Server must be running:
  python -u -m preselective_rpc.server --host 127.0.0.1 --port 50061 \
      --recording-config pipeline_config/recording_config_ws3.yaml \
      --urdf assets/urdf/so101_robot4.urdf
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preselective_rpc.client import PreselectiveClient


def _make_synthetic_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    # SO-101 6-DOF (5 arm + gripper). Small motion between two reachable poses.
    start_qpos = np.array([0.0, -0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal_qpos = np.array([0.4, -0.3, 0.8, 0.2, 0.0, 0.0], dtype=np.float32)
    state = start_qpos.copy()
    # Two cameras matching ws3.yaml (top + left_wrist), 480x640x3 RGB uint8.
    images = {
        "top": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        "left_wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
    }
    return start_qpos, goal_qpos, state, images


def main() -> int:
    addr = "127.0.0.1:50061"
    print(f"[smoke] connecting to {addr} ...")
    client = PreselectiveClient(addr, timeout_s=120.0)

    # 0. Ready — confirm server up + read baseline buffer total
    info = client.ready()
    baseline = info["buffer_total"]
    print(f"[smoke] server up: device={info['device']}, "
          f"baseline buffer_total={baseline}, selector={info['selector_summary']}")

    # 1. PlanAndSelect with synthetic context (K=4 to keep it fast on local 8GB GPU)
    start_qpos, goal_qpos, state, images = _make_synthetic_inputs(seed=42)
    print(f"\n[smoke] PlanAndSelect K=4 ...")
    t0 = time.perf_counter()
    resp = client.plan_and_select(
        skill_id="smoke_test",  # distinct shard so we don't pollute move_to
        start_qpos=start_qpos,
        goal_qpos=goal_qpos,
        state=state,
        images=images,
        instruction="smoke test — pick up the red block",
        n_candidates=4,
        seed=12345,
        is_transit=True,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"[smoke] RTT: {elapsed:.0f}ms  used_fallback={resp['used_fallback']}")
    if resp["used_fallback"]:
        print("[smoke] server returned fallback (curobo found 0 cands?). aborting.")
        client.close()
        return 1

    traj = resp["trajectory"]
    print(f"[smoke] chosen_index={resp['chosen_index']}  "
          f"selection_id={resp['selection_id'][:12]}...")
    print(f"[smoke] trajectory: waypoints.shape={traj['waypoints'].shape} "
          f"dtype={traj['waypoints'].dtype} "
          f"algo={traj['algo']} cost={traj['cost']:.3f} seed={traj['seed']}")
    print(f"[smoke] times: {traj['times'].shape} "
          f"[{traj['times'].min():.3f}..{traj['times'].max():.3f}]s")
    # Show first 3 reports of score breakdown
    import json
    reports = json.loads(resp["score_report_json"])
    print(f"[smoke] score reports ({len(reports)}):")
    for r in reports[:4]:
        marker = "  <-- CHOSEN" if r["i"] == resp["chosen_index"] else ""
        print(f"  idx={r['i']}: ig={r['ig']:.3f} ac={r['ac']:.3f} "
              f"score={r['score']:.4f}{marker}")

    # 2. PlanAndSelect again (different seed) — should also work, server caches another selection
    print(f"\n[smoke] PlanAndSelect K=4 (second call, different seed) ...")
    t0 = time.perf_counter()
    resp2 = client.plan_and_select(
        skill_id="smoke_test",
        start_qpos=start_qpos,
        goal_qpos=goal_qpos,
        state=state,
        images=images,
        instruction="smoke test — pick up the red block",
        n_candidates=4,
        seed=99999,
        is_transit=True,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"[smoke] RTT: {elapsed:.0f}ms  chosen_index={resp2['chosen_index']}  "
          f"selection_id={resp2['selection_id'][:12]}...")

    # 3. Commit with judge_true=True — both selections should be appended
    print(f"\n[smoke] CommitToBuffer judge_true=True ...")
    commit = client.commit(judge_true=True, episode_id="smoke_ep_1")
    print(f"[smoke] commit response: {commit}")

    # 4. Verify buffer grew
    info_after = client.ready()
    delta = info_after["buffer_total"] - baseline
    print(f"\n[smoke] buffer_total: {baseline} → {info_after['buffer_total']} (Δ={delta})")
    print(f"[smoke] buffer_per_skill: {info_after['buffer_per_skill']}")

    # 5. One more plan + commit(judge_true=False) — should NOT append
    print(f"\n[smoke] PlanAndSelect again, then CommitToBuffer judge_true=False ...")
    resp3 = client.plan_and_select(
        skill_id="smoke_test",
        start_qpos=start_qpos, goal_qpos=goal_qpos, state=state, images=images,
        instruction="smoke", n_candidates=4, seed=77777, is_transit=True,
    )
    print(f"[smoke] another selection_id={resp3['selection_id'][:12]}...")
    commit_drop = client.commit(judge_true=False, episode_id="smoke_ep_2_dropped")
    print(f"[smoke] drop response: {commit_drop}")

    info_final = client.ready()
    delta_final = info_final["buffer_total"] - info_after["buffer_total"]
    print(f"\n[smoke] buffer_total after drop: {info_after['buffer_total']} → "
          f"{info_final['buffer_total']} (Δ={delta_final}, expected 0)")

    # 6. Pass / fail summary
    ok = (delta == 2) and (delta_final == 0) and (not resp["used_fallback"])
    print("\n" + ("=" * 60))
    print(f"[smoke] {'PASS ✓' if ok else 'FAIL ✗'}")
    print(f"  - 2 selections committed (judge_true): {'OK' if delta == 2 else f'FAIL (got Δ={delta})'}")
    print(f"  - 0 selections after drop: {'OK' if delta_final == 0 else f'FAIL (got Δ={delta_final})'}")
    print(f"  - fallback never triggered: {'OK' if not resp['used_fallback'] else 'FAIL'}")
    print("=" * 60)

    client.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
