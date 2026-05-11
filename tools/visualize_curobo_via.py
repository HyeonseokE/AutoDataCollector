"""Visualize curobo plan_batch candidates as EE-space arcs.

Generates one figure with three panels — 3D, top-down (xy), side (xz) —
showing each candidate trajectory as an EE-position polyline. Colors are
keyed to algo name, so you can see at a glance which slot picked which
K mix (direct vs via1 vs via2).

Typical use:
    conda activate lerobot_cap
    python tools/visualize_curobo_via.py
    # → results/curobo_viz_<timestamp>.png

Override defaults via CLI:
    python tools/visualize_curobo_via.py \\
        --max_vias 2 --n 4 --seed 7 \\
        --start 0,0,0,0,0 --goal 0.3,-0.4,0.5,-0.2,0.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
CUROBO_CFG = PROJECT_ROOT / "robot_configs" / "curobo" / "so101_robot0.yml"


def _parse_qpos(text: str, expect_dim: int = 5) -> np.ndarray:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != expect_dim:
        raise SystemExit(
            f"qpos must be {expect_dim} comma-separated floats; got {len(parts)} ({text!r})"
        )
    return np.array([float(p) for p in parts])


def _ee_xyz_along_trajectory(backend, waypoints: np.ndarray) -> np.ndarray:
    """FK every waypoint in a (T, dof) trajectory → (T, 3) EE xyz."""
    # Pad to full DoF if caller only supplied arm joints.
    T = waypoints.shape[0]
    qpos_list: list[np.ndarray] = []
    for t in range(T):
        qpos_list.append(backend._to_full_qpos(waypoints[t]))
    pairs = backend._compute_ee_xyz_quat_batch(qpos_list)
    xyz = np.stack([p[0] for p in pairs], axis=0)
    return xyz


def _plot(
    out_path: Path,
    start_ee: np.ndarray,
    goal_ee: np.ndarray,
    candidate_paths: list[dict],
    title_suffix: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)

    fig = plt.figure(figsize=(15, 5))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 3, 2)
    ax_xz = fig.add_subplot(1, 3, 3)

    cmap = plt.get_cmap("tab10")
    for i, cand in enumerate(candidate_paths):
        c = cmap(i % 10)
        xyz = cand["xyz"]
        label = f"{cand['algo']}  (wp={xyz.shape[0]}, cost={cand['cost']:.2f})"
        for ax in (ax3d,):
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=c, lw=2, label=label, alpha=0.85)
        ax_xy.plot(xyz[:, 0], xyz[:, 1], color=c, lw=2, label=label, alpha=0.85)
        ax_xz.plot(xyz[:, 0], xyz[:, 2], color=c, lw=2, label=label, alpha=0.85)

    # Start / goal markers on every panel.
    for ax, dims in ((ax3d, (0, 1, 2)), (ax_xy, (0, 1)), (ax_xz, (0, 2))):
        s_args = [start_ee[d] for d in dims]
        g_args = [goal_ee[d] for d in dims]
        ax.scatter(*s_args, color="green", s=120, marker="o", edgecolors="black", zorder=5, label="start")
        ax.scatter(*g_args, color="red", s=120, marker="X", edgecolors="black", zorder=5, label="goal")

    # Straight line (start→goal) for reference, on every panel.
    line = np.stack([start_ee, goal_ee], axis=0)
    line_kw = dict(color="gray", lw=1.5, linestyle="--", alpha=0.5, label="start→goal line")
    ax3d.plot(line[:, 0], line[:, 1], line[:, 2], **line_kw)
    ax_xy.plot(line[:, 0], line[:, 1], **line_kw)
    ax_xz.plot(line[:, 0], line[:, 2], **line_kw)

    ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
    ax3d.set_title("3D")
    ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("Top-down (xy)")
    ax_xy.set_aspect("equal", adjustable="datalim"); ax_xy.grid(True, alpha=0.3)
    ax_xz.set_xlabel("x (m)"); ax_xz.set_ylabel("z (m)")
    ax_xz.set_title("Side (xz)")
    ax_xz.set_aspect("equal", adjustable="datalim"); ax_xz.grid(True, alpha=0.3)

    # One shared legend below the figure.
    handles, labels = ax_xy.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"curobo plan_batch candidates — EE arcs   {title_suffix}",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="n_candidates to ask plan_batch for")
    ap.add_argument("--max_vias", type=int, default=2, help="K_max — max vias per candidate (0/1/2)")
    ap.add_argument("--start", type=str, default="0,0,0,0,0", help="start qpos (5 floats)")
    ap.add_argument("--goal", type=str, default="0.3,-0.4,0.5,-0.2,0.0", help="goal qpos (5 floats)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--via_offset_mag", type=float, default=0.10)
    ap.add_argument("--out", type=str, default=None,
                    help="output PNG path (default: results/curobo_viz_<ts>.png)")
    args = ap.parse_args()

    start_qpos = _parse_qpos(args.start)
    goal_qpos = _parse_qpos(args.goal)

    from perturbation.skill_level import get_curobo_backend
    CuroboBackend, CuroboBackendConfig = get_curobo_backend()
    cfg = CuroboBackendConfig(
        enabled=True,
        robot_cfg_path=str(CUROBO_CFG),
        num_trajopt_seeds=4,
        num_ik_seeds=16,
        use_cuda_graph=True,
        max_batch_size=max(args.n, 4),
        fixed_joint_indices=(4,),
        arm_joint_count=5,
        via_offset_mag=float(args.via_offset_mag),
        junction_smooth_k=5,
        max_vias_per_candidate=int(args.max_vias),
    )
    print(f"[init] CuroboBackend (K_max={args.max_vias}, n_candidates={args.n})")
    t0 = time.time()
    backend = CuroboBackend(urdf=str(URDF), config=cfg)
    print(f"[init] ready in {time.time()-t0:.2f}s")

    rng = np.random.default_rng(args.seed)
    # Warm-up call (graph compile) — discard result; subsequent call is cached.
    print("[warmup] graph compile call...")
    _ = backend.plan_batch(start_qpos, goal_qpos, n=args.n, rng=rng)

    # Real call for visualization
    rng2 = np.random.default_rng(args.seed + 1)
    print("[plan] cached call for viz...")
    t0 = time.time()
    cands = backend.plan_batch(start_qpos, goal_qpos, n=args.n, rng=rng2)
    wall = time.time() - t0
    print(f"[plan] wall={wall*1000:.0f}ms, {len(cands)}/{args.n} succeeded")

    # FK on start, goal, and every candidate trajectory.
    start_full = backend._to_full_qpos(start_qpos)
    goal_full = backend._to_full_qpos(goal_qpos)
    (start_ee, _), (goal_ee, _) = backend._compute_ee_xyz_quat_batch([start_full, goal_full])

    candidate_paths: list[dict] = []
    for c in cands:
        xyz = _ee_xyz_along_trajectory(backend, c.waypoints)
        candidate_paths.append({
            "algo": c.algo,
            "cost": c.cost,
            "xyz": xyz,
        })
        print(f"  {c.algo:<22} wp={c.waypoints.shape[0]:<4} cost={c.cost:.3f}")

    if args.out:
        out_path = Path(args.out)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "results" / f"curobo_viz_{ts}.png"

    title = (
        f"K_max={args.max_vias}, n={args.n}, seed={args.seed+1}, "
        f"plan={wall*1000:.0f}ms"
    )
    _plot(out_path, start_ee, goal_ee, candidate_paths, title_suffix=title)

    backend.close()
    return 0 if cands else 1


if __name__ == "__main__":
    sys.exit(main())
