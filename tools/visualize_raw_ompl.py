"""Visualize RAW OMPL waypoints (no post-processing) per algorithm.

Key difference from visualize_planner_ensemble.py:
  - waypoint_density=0 → no _densify() linear interpolation
  - One trajectory per algorithm (deterministic comparison, not RNG batch)
  - Markers at every raw waypoint to expose corners / jitter

Use this to see what each OMPL planner actually produces before any smoothing.

Usage:
    conda activate mplib_env
    python tools/visualize_raw_ompl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perturbation.skill_level import (
    PlannerEnsemble,
    PlannerEnsembleConfig,
)

URDF = PROJECT_ROOT / "assets" / "urdf" / "so101_robot0.urdf"
RESULTS = PROJECT_ROOT / "tools" / "results"

ALGORITHMS = ("RRTConnect", "PRMstar", "BITstar", "KPIECE1")
ALGO_COLORS = {
    "RRTConnect": "tab:blue",
    "PRMstar":    "tab:orange",
    "BITstar":    "tab:green",
    "KPIECE1":    "tab:red",
}


def fk_path(planner_ensemble: PlannerEnsemble, waypoints: np.ndarray) -> np.ndarray:
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
    if not URDF.exists():
        print(f"[FAIL] URDF not found: {URDF}")
        return 1
    RESULTS.mkdir(parents=True, exist_ok=True)

    cfg = PlannerEnsembleConfig(
        enabled=True,
        algorithms=ALGORITHMS,
        planning_time=0.5,
        waypoint_density=0.0,   # ← no densify, raw OMPL output only
        workspace=dict(
            table_enabled=True, table_z=0.0, table_thickness=0.10,
            table_size=(2.0, 2.0),
        ),
    )
    ens = PlannerEnsemble(urdf=URDF, config=cfg)

    start = np.zeros(5)
    goal = np.array([0.3, -0.4, 0.5, -0.2, 0.0])

    rng = np.random.default_rng(42)
    cand_per_algo = {}
    for algo in ALGORITHMS:
        c = ens.plan(start, goal, rng=rng, algorithm=algo, seed=42)
        if c is None:
            print(f"  {algo}: FAILED")
            continue
        cand_per_algo[algo] = c
        ee = fk_path(ens, c.waypoints)
        cand_per_algo[algo] = (c, ee)
        print(f"  {algo:<12} raw_wp={c.waypoints.shape[0]:<5} cost={c.cost:.3f}")

    # ── Build figure: 4 algorithms × 3 plots (3D, EE z, joints) ─────────────
    fig = plt.figure(figsize=(20, 14))

    # Top: 3D paths overlay (all 4 in one)
    ax3d = fig.add_subplot(3, 4, (1, 4), projection="3d")
    for algo, (c, ee) in cand_per_algo.items():
        color = ALGO_COLORS[algo]
        ax3d.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-", color=color, alpha=0.7,
                  linewidth=1.2, label=f"{algo} (wp={c.waypoints.shape[0]})")
        ax3d.scatter(ee[:, 0], ee[:, 1], ee[:, 2], color=color, s=20,
                     edgecolors="k", linewidths=0.4, alpha=0.85, zorder=5)
    ax3d.scatter([0, 0], [0, 0], [0, 0], c="red", s=50)  # robot base marker
    ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
    ax3d.set_title(
        "RAW OMPL waypoints — each dot is one planner-emitted state\n"
        "(no _densify, no simplify, no smoothing)",
        fontsize=11,
    )
    ax3d.legend(loc="upper left", fontsize=10)

    # Middle row: EE z over arc-length, per algorithm (one subplot each)
    for col, algo in enumerate(ALGORITHMS):
        if algo not in cand_per_algo:
            continue
        c, ee = cand_per_algo[algo]
        ax = fig.add_subplot(3, 4, 5 + col)
        seg = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        s_norm = s / max(s[-1], 1e-9)
        color = ALGO_COLORS[algo]
        ax.plot(s_norm, ee[:, 2] * 1000, "-", color=color, alpha=0.7,
                linewidth=1.2)
        ax.scatter(s_norm, ee[:, 2] * 1000, color=color, s=18,
                   edgecolors="k", linewidths=0.4, zorder=5)
        ax.axhline(0, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("arc-length (norm.)", fontsize=9)
        ax.set_ylabel("EE z (mm)" if col == 0 else "")
        ax.set_title(f"{algo} — wp={c.waypoints.shape[0]}, cost={c.cost:.2f}",
                     fontsize=10)
        ax.grid(True, alpha=0.3)

    # Bottom row: shoulder_lift joint angle (most informative single joint)
    # over path fraction, per algorithm — exposes corners/jitter clearly
    for col, algo in enumerate(ALGORITHMS):
        if algo not in cand_per_algo:
            continue
        c, _ = cand_per_algo[algo]
        ax = fig.add_subplot(3, 4, 9 + col)
        wp = c.waypoints
        t = np.linspace(0, 1, wp.shape[0])
        # plot all 5 joints with thin lines, shoulder_lift bold
        for j, (jname, lw, alpha) in enumerate([
            ("shoulder_pan", 0.8, 0.45),
            ("shoulder_lift", 1.6, 0.95),
            ("elbow_flex", 0.8, 0.45),
            ("wrist_flex", 0.8, 0.45),
            ("wrist_roll", 0.8, 0.45),
        ]):
            ax.plot(t, np.degrees(wp[:, j]), "-", color=ALGO_COLORS[algo],
                    linewidth=lw, alpha=alpha)
            ax.scatter(t, np.degrees(wp[:, j]), color=ALGO_COLORS[algo],
                       s=8, alpha=0.6, zorder=5)
        ax.set_xlabel("path frac", fontsize=9)
        ax.set_ylabel("joint deg" if col == 0 else "")
        ax.set_title(f"{algo} — joints (5 lines, shoulder_lift bold)",
                     fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Raw OMPL outputs — no smoothing/simplify/densify applied",
        fontsize=13,
    )
    plt.tight_layout()
    out_path = RESULTS / "raw_ompl_per_algo.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\n[plot] saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
