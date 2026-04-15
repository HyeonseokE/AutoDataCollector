#!/usr/bin/env python3
"""
Skill-segmented EE trajectory visualizer for LeRobot datasets.

Loads a single episode from a HuggingFace LeRobot dataset, splits the joint
trajectory at skill boundaries (where the chosen skill label changes between
consecutive frames), runs Forward Kinematics on each segment, and plots EE
(end-effector) trajectories in 3D — one color per skill.

Mirrors `chunk_visualizer/visualize.py` but the color boundary is the skill
label transition rather than a fixed-size action chunk.

Usage:
    python -m skill_visualizer.visualize \
        --repo-id skkuprism/cap_pnp_100ep \
        --revision v3.0 \
        --episode 0 \
        --urdf assets/urdf/so101_robot2.urdf \
        --output outputs/skill_viz/ep00.png
"""

from __future__ import annotations

import argparse
import io
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
# Dataset loading
# ---------------------------------------------------------------------------

def _load_episode_meta(repo_id: str, revision: str, episode: int) -> dict:
    """Read meta/info.json + meta/episodes/* and resolve which parquet shard
    holds the requested episode, plus its [from, to) row range."""
    info_path = hf_hub_download(
        repo_id=repo_id, filename="meta/info.json",
        repo_type="dataset", revision=revision,
    )
    with open(info_path) as f:
        info = json.load(f)

    # Episodes metadata is stored as parquet under meta/episodes/chunk-XXX/file-YYY.parquet.
    # For datasets of this size (≤ a few hundred episodes) a single file is sufficient.
    eps_path = hf_hub_download(
        repo_id=repo_id, filename="meta/episodes/chunk-000/file-000.parquet",
        repo_type="dataset", revision=revision,
    )
    eps_table = pq.read_table(eps_path)
    eps_idx = eps_table.column("episode_index").to_pylist()
    if episode not in eps_idx:
        raise ValueError(
            f"Episode {episode} not found. Available range: "
            f"{min(eps_idx)}..{max(eps_idx)} ({len(eps_idx)} episodes)"
        )
    row = eps_idx.index(episode)
    return {
        "info": info,
        "data_chunk": eps_table.column("data/chunk_index")[row].as_py(),
        "data_file": eps_table.column("data/file_index")[row].as_py(),
        "from_idx": eps_table.column("dataset_from_index")[row].as_py(),
        "to_idx": eps_table.column("dataset_to_index")[row].as_py(),
    }


def load_episode_frames(
    repo_id: str,
    revision: str,
    episode: int,
    joint_key: str,
    skill_key: str,
) -> dict:
    """Download the parquet shard containing `episode` and return joint+skill arrays.

    Returns:
        dict with keys:
            joints_rad: (T, num_joints) float radians
            skill_labels: (T,) python list of label strings
            episode: int
            num_frames: int
    """
    meta = _load_episode_meta(repo_id, revision, episode)
    data_template = meta["info"]["data_path"]
    data_rel = data_template.format(
        chunk_index=meta["data_chunk"], file_index=meta["data_file"]
    )

    data_path = hf_hub_download(
        repo_id=repo_id, filename=data_rel,
        repo_type="dataset", revision=revision,
    )
    table = pq.read_table(data_path, columns=[joint_key, skill_key, "episode_index"])

    # Slice this episode's rows. dataset_from/to_index is global; within a single
    # parquet shard those correspond to row offsets directly in v3.0.
    from_idx = meta["from_idx"]
    to_idx = meta["to_idx"]
    sliced = table.slice(from_idx, to_idx - from_idx)

    joints_rad = np.array(sliced.column(joint_key).to_pylist(), dtype=np.float64)
    skill_labels = sliced.column(skill_key).to_pylist()
    if joints_rad.ndim != 2:
        raise ValueError(
            f"Expected 2D joint array from {joint_key}, got shape {joints_rad.shape}"
        )

    return {
        "joints_rad": joints_rad,
        "skill_labels": skill_labels,
        "episode": episode,
        "num_frames": len(joints_rad),
    }


# ---------------------------------------------------------------------------
# Skill segmentation
# ---------------------------------------------------------------------------

def segment_by_skill(
    joints: np.ndarray, labels: list,
) -> tuple[list[np.ndarray], list[str], list[tuple[int, int]]]:
    """Split joint trajectory at points where consecutive labels differ.

    Returns:
        segments: list of (segment_len, num_joints) arrays
        seg_labels: skill label at each segment's start
        seg_ranges: (start, end) indices for each segment (end is exclusive)
    """
    if len(labels) != len(joints):
        raise ValueError(
            f"labels ({len(labels)}) and joints ({len(joints)}) length mismatch"
        )

    boundaries = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.append(i)
    boundaries.append(len(labels))

    segments, seg_labels, seg_ranges = [], [], []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        segments.append(joints[s:e])
        seg_labels.append(labels[s])
        seg_ranges.append((s, e))
    return segments, seg_labels, seg_ranges


# ---------------------------------------------------------------------------
# Plotting (skill-aware: legend shows skill name + step range)
# ---------------------------------------------------------------------------

def _short_label(text: str, max_len: int = 60) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _draw_skills_on_axis(
    ax,
    ee_trajectories: list[np.ndarray],
    seg_labels: list[str],
    seg_ranges: list[tuple[int, int]],
    cmap,
    initial_ee: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    box_aspect: np.ndarray,
    step_label_stride: int = 25,
):
    """Draw skill-segmented EE trajectories with per-skill legend entries."""
    for i, (traj, label, (s, e)) in enumerate(zip(ee_trajectories, seg_labels, seg_ranges)):
        color = cmap(i)
        legend_text = f"[{s}-{e - 1}] {_short_label(label)}"
        ax.plot(
            traj[:, 0], traj[:, 1], traj[:, 2],
            color=color, linewidth=1.6, alpha=0.85, label=legend_text,
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
                    f"{s + k}",
                    color=color, fontsize=6, alpha=0.9, zorder=6,
                )

    ax.scatter(0, 0, 0, color="black", marker="*", s=200, zorder=10, label="Origin")
    ax.scatter(
        *initial_ee, facecolors="none", edgecolors="red", marker="D",
        s=150, linewidths=2, zorder=10, label="Initial EE",
    )

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(box_aspect)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


def plot_skill_trajectory_3d(
    ee_trajectories: list[np.ndarray],
    seg_labels: list[str],
    seg_ranges: list[tuple[int, int]],
    title: str,
    output_path: str | None = None,
    step_label_stride: int = 25,
):
    """Render a 4-view 3D EE-trajectory plot, one color per skill segment."""
    cmap = get_cmap("tab20", max(len(ee_trajectories), 2))
    initial_ee = ee_trajectories[0][0]
    lo, hi, box_aspect = _compute_axis_limits(ee_trajectories)

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    for idx, (name, elev, azim) in enumerate(_VIEWS_3D):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        _draw_skills_on_axis(
            ax, ee_trajectories, seg_labels, seg_ranges,
            cmap, initial_ee, lo, hi, box_aspect,
            step_label_stride=step_label_stride,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved skill trajectory figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive HTML plot (one trace per skill, hover shows skill text)
# ---------------------------------------------------------------------------

def plot_skill_trajectory_3d_html(
    ee_trajectories: list[np.ndarray],
    seg_labels: list[str],
    seg_ranges: list[tuple[int, int]],
    title: str,
    output_path: str,
    step_label_stride: int = 25,
):
    """Interactive 3D EE-trajectory plot, one color per skill segment.

    Mouse-drag rotates, scroll zooms, shift-drag pans. Legend entries (skill
    names) toggle individual segments. Hover shows skill text + frame index.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly is not installed. Run: pip install plotly")
        return

    cmap = get_cmap("tab20", max(len(ee_trajectories), 2))
    initial_ee = ee_trajectories[0][0]

    fig = go.Figure()

    for i, (traj, label, (s, e)) in enumerate(zip(ee_trajectories, seg_labels, seg_ranges)):
        color = _matplotlib_color_to_hex(cmap(i))
        n = len(traj)
        if step_label_stride and step_label_stride > 0:
            label_idxs = set(range(0, n, step_label_stride))
            label_idxs.add(n - 1)
        else:
            label_idxs = set()
        text_labels = [str(s + k) if k in label_idxs else "" for k in range(n)]
        short = _short_label(label, 80)
        hover = [
            f"skill {i}: {short}<br>frame {s + k}<br>"
            f"x={traj[k, 0]:.4f}<br>y={traj[k, 1]:.4f}<br>z={traj[k, 2]:.4f}"
            for k in range(n)
        ]
        legend_name = f"[{s}-{e - 1}] {short}"
        group = f"skill{i}"
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
                name=legend_name,
                legendgroup=group,
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
                mode="markers",
                marker=dict(size=6, color=color, symbol="circle", line=dict(color="black", width=1)),
                legendgroup=group, showlegend=False,
                hovertext=f"skill {i} start (frame {s})", hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
                mode="markers",
                marker=dict(size=6, color=color, symbol="x"),
                legendgroup=group, showlegend=False,
                hovertext=f"skill {i} end (frame {e - 1})", hoverinfo="text",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers",
            marker=dict(size=8, color="black", symbol="diamond"),
            name="Origin (0,0,0)",
            hovertext="Origin (robot base)", hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[initial_ee[0]], y=[initial_ee[1]], z=[initial_ee[2]],
            mode="markers",
            marker=dict(size=8, color="red", symbol="diamond-open", line=dict(color="red", width=3)),
            name="Initial EE",
            hovertext=f"Initial EE ({initial_ee[0]:.3f}, {initial_ee[1]:.3f}, {initial_ee[2]:.3f})",
            hoverinfo="text",
        )
    )

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

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="skkuprism/cap_pnp_100ep", help="HF dataset repo id")
    p.add_argument("--revision", default="v3.0", help="HF dataset revision/tag")
    p.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    p.add_argument(
        "--urdf",
        default=str(Path(__file__).resolve().parents[3] / "assets/urdf/so101_robot2.urdf"),
        help="URDF path for FK",
    )
    p.add_argument(
        "--joint-key", default="action.radian_urdf0",
        choices=["action.radian_urdf0", "observation.state.radian_urdf0"],
        help="Joint field to compute EE from (radians, URDF 0° reference)",
    )
    p.add_argument(
        "--skill-key", default="skill.natural_language",
        help="Field whose value transitions define skill segment boundaries",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Directory to write all figures into. "
             "Defaults to outputs/skill_viz/{repo}_{rev}/ep{NN}/",
    )
    p.add_argument("--step-label-stride", type=int, default=25)
    p.add_argument("--fps", type=float, default=30.0,
                   help="Sampling rate (Hz) for time-axis on joint/displacement plots")
    args = p.parse_args()

    print(f"Loading episode {args.episode} from {args.repo_id}@{args.revision} ...")
    ep = load_episode_frames(
        repo_id=args.repo_id, revision=args.revision, episode=args.episode,
        joint_key=args.joint_key, skill_key=args.skill_key,
    )
    print(f"  frames: {ep['num_frames']}  joint dim: {ep['joints_rad'].shape[1]}")

    # rad → deg (SimpleFK takes degrees)
    joints_deg = np.degrees(ep["joints_rad"])

    segments, seg_labels, seg_ranges = segment_by_skill(joints_deg, ep["skill_labels"])
    print(f"  detected {len(segments)} skill segment(s) using key={args.skill_key}")
    for i, (lab, rng) in enumerate(zip(seg_labels, seg_ranges)):
        print(f"    [{i}] frames {rng[0]:>4}–{rng[1] - 1:<4}  {_short_label(lab, 80)}")

    print(f"Loading FK from {args.urdf} ...")
    fk = SimpleFK(args.urdf)
    ee_trajectories = [compute_ee_trajectory(seg, fk) for seg in segments]

    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"outputs/skill_viz/{args.repo_id.replace('/', '_')}_{args.revision}/ep{args.episode:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"Skill-segmented EE trajectory — {args.repo_id}@{args.revision} "
        f"ep{args.episode}  ({len(segments)} skills, {ep['num_frames']} frames)"
    )

    # 1) Static 4-view 3D PNG
    plot_skill_trajectory_3d(
        ee_trajectories=ee_trajectories,
        seg_labels=seg_labels,
        seg_ranges=seg_ranges,
        title=title,
        output_path=str(out_dir / "ee_trajectory_3d.png"),
        step_label_stride=args.step_label_stride,
    )
    # 2) Interactive 3D HTML
    plot_skill_trajectory_3d_html(
        ee_trajectories=ee_trajectories,
        seg_labels=seg_labels,
        seg_ranges=seg_ranges,
        title=title,
        output_path=str(out_dir / "ee_trajectory_3d.html"),
        step_label_stride=args.step_label_stride,
    )
    # 3) Per-skill EE displacement (reuses chunk_visualizer's function — color per segment)
    plot_ee_displacement(
        ee_trajectories=ee_trajectories,
        fps=args.fps,
        output_path=str(out_dir / "ee_displacement.png"),
    )
    # 4) Joint angles colored per skill
    plot_joint_angles(
        chunks=segments,
        action_features=None,
        fps=args.fps,
        output_path=str(out_dir / "joint_angles.png"),
    )

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
