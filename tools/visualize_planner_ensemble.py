"""Visualize OMPL planner ensemble candidates.

Generates N candidate trajectories from the same (start, goal), forward-
kinematics each waypoint to EE position, and plots them so the qualitative
mode differences between algorithms are visible.

Three plots:
  (1) 3D EE path  — every candidate as a polyline, colored by algorithm.
  (2) EE z over arc-length — shows how each algo handles approach/lift.
  (3) Joint trajectories — one subplot per arm joint, joint angle over time.

Run inside ``mplib_env`` (needs mplib for FK + ompl for planning).

Usage::
    conda activate mplib_env
    python tools/visualize_planner_ensemble.py
    python tools/visualize_planner_ensemble.py --n 12 --seed 7 \\
        --start "0 0 0 0 0" --goal "0.3 -0.4 0.5 -0.2 0"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perturbation.skill_level import (
    PlannerEnsemble,
    PlannerEnsembleConfig,
)


URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
RESULTS = PROJECT_ROOT / "tools" / "results"

# Colors per algorithm (consistent across plots)
ALGO_COLORS = {
    "RRTConnect": "tab:blue",
    "PRMstar":    "tab:orange",
    "BITstar":    "tab:green",
    "KPIECE1":    "tab:red",
    "RRTstar":    "tab:purple",
    "FMT":        "tab:brown",
}


def fk_path(planner_ensemble: PlannerEnsemble, waypoints: np.ndarray) -> np.ndarray:
    """Forward-kinematics every waypoint → EE xyz. Returns (N, 3)."""
    p_mp = planner_ensemble._planner_mp
    pin = p_mp.robot.get_pinocchio_model()
    tip_idx = p_mp.user_link_names.index(planner_ensemble.cfg.move_group_link)
    out = np.empty((waypoints.shape[0], 3), dtype=float)
    for i, wp in enumerate(waypoints):
        full = p_mp.pad_move_group_qpos(wp)
        p_mp.robot.set_qpos(full, True)
        out[i] = pin.get_link_pose(tip_idx).p
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="Candidate count (passes to plan().")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="0 0 0 0 0",
                    help="Start qpos (5 floats space-separated, radians)")
    ap.add_argument("--goal",  default="0.3 -0.4 0.5 -0.2 0",
                    help="Goal qpos (5 floats space-separated, radians)")
    ap.add_argument("--planning_time", type=float, default=0.5)
    ap.add_argument("--out", default=str(RESULTS / "planner_ensemble.png"))
    args = ap.parse_args()

    start = np.array([float(x) for x in args.start.split()])
    goal  = np.array([float(x) for x in args.goal.split()])

    if not URDF.exists():
        print(f"[FAIL] URDF not found: {URDF}")
        return 1
    RESULTS.mkdir(parents=True, exist_ok=True)

    cfg = PlannerEnsembleConfig(
        enabled=True,
        algorithms=("RRTConnect", "PRMstar", "BITstar", "KPIECE1"),
        planning_time=args.planning_time,
        waypoint_density=0.01,
        workspace=dict(
            table_enabled=True, table_z=0.0, table_thickness=0.10,
            table_size=(2.0, 2.0),
        ),
    )
    ens = PlannerEnsemble(urdf=URDF, config=cfg)
    print(f"[ensemble] dof={ens.dof}, joint_limits.shape={ens.joint_limits.shape}")
    print(f"[task] start={start.tolist()}")
    print(f"[task] goal ={goal.tolist()}")

    rng = np.random.default_rng(args.seed)
    candidates = []
    for i in range(args.n):
        c = ens.plan(start, goal, rng=rng)
        if c is None:
            print(f"  trial {i+1}: FAILED")
            continue
        candidates.append(c)
        print(f"  trial {i+1}: {c.algo:<12} seed={c.seed:<11} "
              f"wp={c.waypoints.shape[0]:<5} cost={c.cost:.3f}")

    if not candidates:
        print("[FAIL] no candidates produced")
        return 1

    # Compute EE paths via FK
    ee_paths = []
    for c in candidates:
        ee = fk_path(ens, c.waypoints)
        ee_paths.append((c, ee))

    # ── Build figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    # (1) 3D EE path
    ax1 = fig.add_subplot(gs[:, 0], projection="3d")
    seen_legend = set()
    for c, ee in ee_paths:
        color = ALGO_COLORS.get(c.algo, "gray")
        label = c.algo if c.algo not in seen_legend else None
        seen_legend.add(c.algo)
        ax1.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-", color=color,
                 alpha=0.55, linewidth=1.4, label=label)
        ax1.scatter([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], color="k",
                    s=20, marker="o")
        ax1.scatter([ee[-1, 0]], [ee[-1, 1]], [ee[-1, 2]], color="k",
                    s=30, marker="^")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_zlabel("z (m)")
    ax1.set_title(f"3D EE paths (●=start, ▲=goal)\n{len(ee_paths)} candidates")
    ax1.legend(loc="upper left", fontsize=9)

    # (2) EE z over arc-length
    ax2 = fig.add_subplot(gs[0, 1:])
    for c, ee in ee_paths:
        seg = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        s_norm = s / max(s[-1], 1e-9)
        color = ALGO_COLORS.get(c.algo, "gray")
        ax2.plot(s_norm, ee[:, 2] * 1000, "-", color=color, alpha=0.6,
                 linewidth=1.4)
    ax2.set_xlabel("arc-length (normalized)")
    ax2.set_ylabel("EE z (mm)")
    ax2.set_title("EE z profile over the path (lower = closer to table)")
    ax2.axhline(0, color="k", linestyle=":", linewidth=0.8, alpha=0.6,
                label="table z=0")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # (3) Joint trajectories — one row, 5 subplots (one per arm joint)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll"]
    # Re-grid: 5 subplots across the bottom row
    gs2 = fig.add_gridspec(2, 5, left=0.42, right=0.98, top=0.46, bottom=0.06,
                           hspace=0.35, wspace=0.35)
    for j, jname in enumerate(joint_names):
        ax = fig.add_subplot(gs2[1, j])
        for c, _ in ee_paths:
            color = ALGO_COLORS.get(c.algo, "gray")
            t = np.linspace(0, 1, c.waypoints.shape[0])
            ax.plot(t, np.degrees(c.waypoints[:, j]), "-", color=color,
                    alpha=0.55, linewidth=1.0)
        ax.axhline(np.degrees(start[j]), color="g", linestyle=":",
                   linewidth=0.8, alpha=0.7)
        ax.axhline(np.degrees(goal[j]),  color="r", linestyle=":",
                   linewidth=0.8, alpha=0.7)
        ax.set_title(jname, fontsize=9)
        ax.set_xlabel("path frac", fontsize=8)
        ax.set_ylabel("deg" if j == 0 else "")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"OMPL planner ensemble — {len(ee_paths)} candidates\n"
        f"start={start.tolist()}  goal={goal.tolist()}",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\n[plot] saved: {out_path}")

    # Summary stats per algorithm
    print()
    print("=" * 56)
    print(f"{'algo':<12} {'count':>5} {'wp(min)':>8} {'wp(max)':>8} {'cost(med)':>10}")
    print("-" * 56)
    by_algo = {}
    for c, _ in ee_paths:
        by_algo.setdefault(c.algo, []).append(c)
    for algo, cs in sorted(by_algo.items()):
        wps = [c.waypoints.shape[0] for c in cs]
        costs = [c.cost for c in cs]
        print(f"{algo:<12} {len(cs):>5} {min(wps):>8} {max(wps):>8} {np.median(costs):>10.3f}")
    print("=" * 56)

    return 0


if __name__ == "__main__":
    sys.exit(main())
