#!/usr/bin/env python3
"""
Episode-level EE trajectory visualizer for LeRobot datasets.

Draws one or more whole episodes from a HuggingFace LeRobot dataset, where
each episode is rendered with its own deterministic color (`tab20[ep_idx % 20]`).
Skill labels are NOT required, so this works on teleop datasets, CaP-collected
datasets, or any other LeRobot v3.0 dataset.

Episode spec syntax:
    "5"        single episode (5)
    "1:10"     inclusive range (1, 2, 3, ..., 10)
    "0,3,7"    explicit list
    "0:4,9"    mixed: range + extras

Usage:
    python -m episode_visualizer.visualize \
        --repo-id skkuprism/cap_pnp_100ep \
        --revision v3.0 \
        --episodes 1:5 \
        --urdf assets/urdf/so101_robot2.urdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from matplotlib.cm import get_cmap

# Reuse FK and rendering primitives from the chunk visualizer.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chunk_visualizer.simple_fk import SimpleFK  # noqa: E402
from chunk_visualizer.visualize import (  # noqa: E402
    _VIEWS_3D,
    _compute_axis_limits,
    _matplotlib_color_to_hex,
    compute_ee_trajectory,
    plot_ee_displacement,
    plot_joint_angles,
)


# ---------------------------------------------------------------------------
# Episode spec parsing
# ---------------------------------------------------------------------------

def parse_episodes_spec(spec: str) -> list[int]:
    """Parse a CLI episode specification into a sorted, deduplicated list.

    Accepts:
        "5"          → [5]
        "1:10"       → [1, 2, ..., 10] (inclusive both ends)
        "0,3,7"      → [0, 3, 7]
        "0:4,9,12:14"→ [0, 1, 2, 3, 4, 9, 12, 13, 14]
    """
    episodes: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            lo, hi = token.split(":", 1)
            episodes.update(range(int(lo), int(hi) + 1))
        else:
            episodes.add(int(token))
    if not episodes:
        raise ValueError(f"Empty episode spec: {spec!r}")
    return sorted(episodes)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_episodes_meta(repo_id: str, revision: str) -> tuple[dict, "pq.Table"]:
    """Download info.json + the episodes metadata parquet."""
    info_path = hf_hub_download(
        repo_id=repo_id, filename="meta/info.json",
        repo_type="dataset", revision=revision,
    )
    with open(info_path) as f:
        info = json.load(f)
    eps_path = hf_hub_download(
        repo_id=repo_id, filename="meta/episodes/chunk-000/file-000.parquet",
        repo_type="dataset", revision=revision,
    )
    return info, pq.read_table(eps_path)


def load_episode_joints(
    repo_id: str,
    revision: str,
    episode: int,
    joint_key: str,
    info: dict,
    eps_table: "pq.Table",
) -> np.ndarray:
    """Return (T, num_joints) joint array (radians) for one episode."""
    eps_idx_col = eps_table.column("episode_index").to_pylist()
    if episode not in eps_idx_col:
        raise ValueError(
            f"Episode {episode} not found. Available: {min(eps_idx_col)}..{max(eps_idx_col)}"
        )
    row = eps_idx_col.index(episode)
    chunk = eps_table.column("data/chunk_index")[row].as_py()
    file = eps_table.column("data/file_index")[row].as_py()
    from_idx = eps_table.column("dataset_from_index")[row].as_py()
    to_idx = eps_table.column("dataset_to_index")[row].as_py()

    data_rel = info["data_path"].format(chunk_index=chunk, file_index=file)
    data_path = hf_hub_download(
        repo_id=repo_id, filename=data_rel,
        repo_type="dataset", revision=revision,
    )
    table = pq.read_table(data_path, columns=[joint_key])
    sliced = table.slice(from_idx, to_idx - from_idx)
    joints = np.array(sliced.column(joint_key).to_pylist(), dtype=np.float64)
    if joints.ndim != 2:
        raise ValueError(f"Expected 2D array from {joint_key}, got {joints.shape}")
    return joints


# ---------------------------------------------------------------------------
# Plotting (episode-aware: deterministic color = tab20[ep % 20])
# ---------------------------------------------------------------------------

_PALETTE_NAME = "tab20"
_PALETTE_SIZE = 20


def episode_color(ep_idx: int):
    """Deterministic per-episode color. Same episode → same RGBA across runs."""
    cmap = get_cmap(_PALETTE_NAME, _PALETTE_SIZE)
    return cmap(ep_idx % _PALETTE_SIZE)


def _draw_episodes_on_axis(
    ax,
    ee_trajectories: list[np.ndarray],
    episodes: list[int],
    initial_ee: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    box_aspect: np.ndarray,
    step_label_stride: int = 50,
):
    for traj, ep in zip(ee_trajectories, episodes):
        color = episode_color(ep)
        ax.plot(
            traj[:, 0], traj[:, 1], traj[:, 2],
            color=color, linewidth=1.6, alpha=0.85, label=f"ep{ep:02d}",
        )
        ax.scatter(*traj[0], color=color, marker="o", s=28, zorder=5)
        ax.scatter(*traj[-1], color=color, marker="x", s=28, zorder=5)

        if step_label_stride and step_label_stride > 0:
            n = len(traj)
            idxs = list(range(0, n, step_label_stride))
            if idxs and idxs[-1] != n - 1:
                idxs.append(n - 1)
            for k in idxs:
                ax.text(
                    traj[k, 0], traj[k, 1], traj[k, 2],
                    f"{k}",
                    color=color, fontsize=6, alpha=0.9, zorder=6,
                )

    ax.scatter(0, 0, 0, color="black", marker="*", s=200, zorder=10, label="Origin")
    ax.scatter(
        *initial_ee, facecolors="none", edgecolors="red", marker="D",
        s=150, linewidths=2, zorder=10, label="Initial EE",
    )
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(box_aspect)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")


def plot_episode_trajectory_3d(
    ee_trajectories: list[np.ndarray],
    episodes: list[int],
    title: str,
    output_path: str | None = None,
    step_label_stride: int = 50,
):
    initial_ee = ee_trajectories[0][0]
    lo, hi, box_aspect = _compute_axis_limits(ee_trajectories)

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    for idx, (name, elev, azim) in enumerate(_VIEWS_3D):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        _draw_episodes_on_axis(
            ax, ee_trajectories, episodes,
            initial_ee, lo, hi, box_aspect,
            step_label_stride=step_label_stride,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=2)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D trajectory figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_episode_trajectory_3d_html(
    ee_trajectories: list[np.ndarray],
    episodes: list[int],
    title: str,
    output_path: str,
    step_label_stride: int = 50,
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly is not installed. Run: pip install plotly")
        return

    initial_ee = ee_trajectories[0][0]
    fig = go.Figure()

    for traj, ep in zip(ee_trajectories, episodes):
        color = _matplotlib_color_to_hex(episode_color(ep))
        n = len(traj)
        if step_label_stride and step_label_stride > 0:
            label_idxs = set(range(0, n, step_label_stride))
            label_idxs.add(n - 1)
        else:
            label_idxs = set()
        text_labels = [str(k) if k in label_idxs else "" for k in range(n)]
        hover = [
            f"ep{ep:02d} · frame {k}<br>x={traj[k,0]:.4f}<br>y={traj[k,1]:.4f}<br>z={traj[k,2]:.4f}"
            for k in range(n)
        ]
        group = f"ep{ep}"
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode="lines+markers+text",
                line=dict(color=color, width=3),
                marker=dict(size=2, color=color),
                text=text_labels,
                textfont=dict(color=color, size=9),
                textposition="top center",
                hovertext=hover, hoverinfo="text",
                name=f"ep{ep:02d}",
                legendgroup=group, showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
                mode="markers",
                marker=dict(size=6, color=color, symbol="circle", line=dict(color="black", width=1)),
                legendgroup=group, showlegend=False,
                hovertext=f"ep{ep:02d} start", hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
                mode="markers",
                marker=dict(size=6, color=color, symbol="x"),
                legendgroup=group, showlegend=False,
                hovertext=f"ep{ep:02d} end (frame {n - 1})", hoverinfo="text",
            )
        )

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=8, color="black", symbol="diamond"),
        name="Origin (0,0,0)", hovertext="Origin (robot base)", hoverinfo="text",
    ))
    fig.add_trace(go.Scatter3d(
        x=[initial_ee[0]], y=[initial_ee[1]], z=[initial_ee[2]],
        mode="markers",
        marker=dict(size=8, color="red", symbol="diamond-open", line=dict(color="red", width=3)),
        name="Initial EE",
        hovertext=f"Initial EE ({initial_ee[0]:.3f}, {initial_ee[1]:.3f}, {initial_ee[2]:.3f})",
        hoverinfo="text",
    ))

    lo, hi, _ = _compute_axis_limits(ee_trajectories)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X (m)", range=[lo[0], hi[0]]),
            yaxis=dict(title="Y (m)", range=[lo[1], hi[1]]),
            zaxis=dict(title="Z (m)", range=[lo[2], hi[2]]),
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved interactive HTML to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _spec_to_tag(episodes: list[int]) -> str:
    """Compress an episode list into a filesystem-friendly tag.
    [1] → 'ep01'. [1,2,...,10] → 'ep01-10'. [0,3,7] → 'ep00_03_07'.
    Long lists get truncated to 'epN-M_Mlist' for sanity.
    """
    if len(episodes) == 1:
        return f"ep{episodes[0]:02d}"
    contiguous = episodes == list(range(episodes[0], episodes[-1] + 1))
    if contiguous:
        return f"ep{episodes[0]:02d}-{episodes[-1]:02d}"
    if len(episodes) <= 8:
        return "ep" + "_".join(f"{e:02d}" for e in episodes)
    return f"ep{episodes[0]:02d}-{episodes[-1]:02d}_{len(episodes)}eps"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="skkuprism/cap_pnp_100ep", help="HF dataset repo id")
    p.add_argument("--revision", default="v3.0", help="HF dataset revision/tag")
    p.add_argument(
        "--episodes", default="0",
        help='Episode spec: "5" / "1:10" / "0,3,7" / "0:4,9"',
    )
    p.add_argument(
        "--urdf",
        default=str(Path(__file__).resolve().parents[3] / "assets/urdf/so101_robot2.urdf"),
    )
    p.add_argument(
        "--joint-key", default="action.radian_urdf0",
        help="Joint field (radians, URDF 0° reference). "
             "Common alternatives: observation.state.radian_urdf0",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--step-label-stride", type=int, default=50)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    episodes = parse_episodes_spec(args.episodes)
    print(f"Loading {len(episodes)} episode(s) from {args.repo_id}@{args.revision}: {episodes}")

    info, eps_table = _load_episodes_meta(args.repo_id, args.revision)
    fk = SimpleFK(args.urdf)

    ee_trajectories: list[np.ndarray] = []
    joint_segments: list[np.ndarray] = []  # for joint_angles plot (deg)
    for ep in episodes:
        joints_rad = load_episode_joints(
            args.repo_id, args.revision, ep, args.joint_key, info, eps_table,
        )
        joints_deg = np.degrees(joints_rad)
        ee = compute_ee_trajectory(joints_deg, fk)
        ee_trajectories.append(ee)
        joint_segments.append(joints_deg)
        print(f"  ep{ep:02d}: {len(joints_rad)} frames")

    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"outputs/episode_viz/{args.repo_id.replace('/', '_')}_{args.revision}/{_spec_to_tag(episodes)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    title = (
        f"Episode EE trajectory — {args.repo_id}@{args.revision} "
        f"({len(episodes)} episode{'s' if len(episodes) > 1 else ''}: {_spec_to_tag(episodes)})"
    )

    plot_episode_trajectory_3d(
        ee_trajectories=ee_trajectories,
        episodes=episodes,
        title=title,
        output_path=str(out_dir / "ee_trajectory_3d.png"),
        step_label_stride=args.step_label_stride,
    )
    plot_episode_trajectory_3d_html(
        ee_trajectories=ee_trajectories,
        episodes=episodes,
        title=title,
        output_path=str(out_dir / "ee_trajectory_3d.html"),
        step_label_stride=args.step_label_stride,
    )
    # Reuse chunk_visualizer's per-segment plots (legend says "chunk N" but
    # here each "chunk" is one full episode — color picked by cmap order).
    plot_ee_displacement(
        ee_trajectories=ee_trajectories,
        fps=args.fps,
        output_path=str(out_dir / "ee_displacement.png"),
    )
    plot_joint_angles(
        chunks=joint_segments,
        action_features=None,
        fps=args.fps,
        output_path=str(out_dir / "joint_angles.png"),
    )

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
