"""Visualize closed-loop refinement convergence per seed from run6 log."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path("/tmp/touch_seeds_run6.log")
OUT_PNG = Path("/home/lerobot/AutoDataCollector/results/session_20260519_225321/seed_refinement_convergence.png")
TOL_MM = 3.0

# Hand-extracted from run6 log (parsing CR-separated trajectory garbage is brittle).
DATA: dict[int, list[float]] = {
    1:  [7.67, 13.35, 20.82, 28.94],
    2:  [2.98],
    3:  [4.16,  6.77, 19.96, 18.49],
    4:  [5.20,  5.04,  5.40,  5.40],
    5:  [2.36],
    6:  [10.09, 10.25, 13.44, 14.92],
    7:  [3.27,  3.26,  3.20,  3.20],
    8:  [2.94],
    9:  [6.62,  7.55, 11.02, 16.24],
    10: [2.92],
}

# Compare against open-loop baseline (run2: z=0 no overshoot, run4: z=-8mm only).
BASELINE_RUN2_TARGET = {
    1: 12.48, 2: 6.97, 3: 11.72, 4: 9.40, 5: 11.68,
    6: 9.12,  7: 9.54, 8: 10.85, 9: 10.22, 10: 8.75,
}

def main() -> int:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Closed-loop EE refinement — robot4 (ws3) seed touch\n"
        "Damping 0.5  |  Max 3 refinements  |  Tolerance 3mm  |  z floor=-8mm",
        fontsize=13, fontweight="bold",
    )

    # ── (Left) Per-iteration convergence ─────────────────────────────────
    cmap = plt.get_cmap("tab10")
    converged_seeds = []
    diverged_seeds = []
    stuck_seeds = []
    for sid, errs in DATA.items():
        xs = list(range(len(errs)))
        color = cmap((sid - 1) % 10)
        ax1.plot(xs, errs, "o-", color=color, label=f"seed_{sid:02d}", linewidth=2, markersize=8)
        # Classification: converged on initial, diverged with refinement, or stuck.
        if errs[0] <= TOL_MM:
            converged_seeds.append(sid)
        elif len(errs) > 1 and errs[-1] > errs[0] * 1.5:
            diverged_seeds.append(sid)
        else:
            stuck_seeds.append(sid)
    ax1.axhline(TOL_MM, color="green", linestyle="--", linewidth=1.5, label=f"tol = {TOL_MM} mm")
    ax1.set_xlabel("Refinement iteration (0 = initial open-loop)")
    ax1.set_ylabel("|EE - target|  (mm)")
    ax1.set_title("Per-seed error progression")
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(["initial", "#1", "#2", "#3"])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", ncol=2, fontsize=8)
    ax1.set_ylim(0, 30)

    # ── (Right) Best-of-iterations vs open-loop baseline ─────────────────
    seeds = sorted(DATA.keys())
    best_each = [min(DATA[s]) for s in seeds]
    baseline = [BASELINE_RUN2_TARGET[s] for s in seeds]
    x = np.arange(len(seeds))
    w = 0.38
    ax2.bar(x - w/2, baseline, w, label="Open-loop (no overshoot)", color="#888888", edgecolor="black")
    ax2.bar(x + w/2, best_each, w,
            label="Best closed-loop iter (damped 0.5)", color="#2a9df4", edgecolor="black")
    ax2.axhline(TOL_MM, color="green", linestyle="--", linewidth=1.5, label=f"tol = {TOL_MM} mm")
    for i, (b, e) in enumerate(zip(baseline, best_each)):
        if e <= TOL_MM:
            ax2.text(i + w/2, e + 0.4, "✓", ha="center", fontsize=14, color="green", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"s{s:02d}" for s in seeds])
    ax2.set_ylabel("|EE - target|  (mm)")
    ax2.set_title("Best result vs naive open-loop baseline")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper right")

    # ── Footer summary ────────────────────────────────────────────────────
    summary = (
        f"converged @ initial (≤3mm): {converged_seeds}   "
        f"refinement diverged: {diverged_seeds}   "
        f"stuck plateau: {stuck_seeds}"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10, fontstyle="italic")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Saved: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
