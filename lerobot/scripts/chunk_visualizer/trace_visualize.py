#!/usr/bin/env python3
"""Visualize a recorded EE trace npz from the chunk_visualizer toolkit.

The default camera view matches:
``results/session_20260604_131637/ee_trace.mp4``
which was rendered with ``--eye 0 -0.05 0.15``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import get_cmap  # noqa: E402


def load_trace(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    episode = np.asarray(data["episode"], dtype=np.int32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(episode):
        raise ValueError(f"Bad trace arrays in {npz_path}")
    return xyz, episode


def equal_bounds(xyz: np.ndarray, pad: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    lo = xyz.min(axis=0) - pad
    hi = xyz.max(axis=0) + pad
    center = (lo + hi) / 2.0
    half = float((hi - lo).max()) / 2.0 + pad
    return center - half, center + half


def plot_trace(
    xyz: np.ndarray,
    episode: np.ndarray,
    output: str | Path,
    elev: float = 31.4,
    azim: float = -159.9,
    title: str | None = None,
) -> None:
    eps = sorted(np.unique(episode).tolist())
    cmap = get_cmap("tab20", len(eps))
    color_by_ep = {ep: cmap(i) for i, ep in enumerate(eps)}
    lo, hi = equal_bounds(xyz)

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0], [0], [0], c="black", s=120, marker="*", zorder=10, label="Origin")

    for ep in eps:
        pts = xyz[episode == ep]
        if len(pts) == 0:
            continue
        color = color_by_ep[ep]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=1.6, alpha=0.95, label=f"ep {ep}")
        ax.scatter(*pts[0], color=color, marker="o", s=22, zorder=5)
        ax.scatter(*pts[-1], color=color, marker="x", s=28, zorder=5)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title or f"Recorded EE trace — {len(eps)} episodes")
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved matching-view trace image to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Recorded ee_trace.npz")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--elev", type=float, default=31.4, help="Camera elevation")
    parser.add_argument("--azim", type=float, default=-159.9, help="Camera azimuth")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    xyz, episode = load_trace(args.npz)
    plot_trace(xyz, episode, args.output, elev=args.elev, azim=args.azim, title=args.title)


if __name__ == "__main__":
    main()
