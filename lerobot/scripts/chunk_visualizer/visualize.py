#!/usr/bin/env python3
"""
Action chunk EE trajectory visualizer.

Loads saved action chunks (.npz), computes Forward Kinematics for each step,
and plots EE (end-effector) trajectories in 3D space using matplotlib.

Usage:
    python -m chunk_visualizer.visualize \
        --npz outputs/action_chunks/chunks_20260412_151957.npz \
        --urdf assets/urdf/so101_robot2.urdf

    # Or via the shell script:
    bash scripts/visualize_chunks.sh outputs/action_chunks/chunks_20260412_151957.npz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap


def _parse_action_features(arr) -> list[str]:
    """Extract action feature names from a numpy array that may be 0-d object (dict) or 1-d string array."""
    if arr.ndim == 0:
        obj = arr.item()
        if isinstance(obj, dict):
            return list(obj.keys())
        if isinstance(obj, (list, tuple)):
            return list(obj)
        return [str(obj)]
    return list(arr)


def load_chunks(npz_path: str) -> dict:
    """Load action chunks from .npz file."""
    data = np.load(npz_path, allow_pickle=True)

    chunks = []
    i = 0
    while f"chunk_{i}" in data:
        chunks.append(data[f"chunk_{i}"])
        i += 1

    return {
        "chunks": chunks,
        "timestamps": data["timestamps"] if "timestamps" in data else None,
        "inference_delays": data["inference_delays"] if "inference_delays" in data else None,
        "action_features": _parse_action_features(data["action_features"]) if "action_features" in data else None,
        "fps": float(data["fps"]) if "fps" in data else 30.0,
    }


def compute_ee_trajectory(chunk: np.ndarray, fk) -> np.ndarray:
    """Compute EE positions for each step in a chunk via FK.

    Args:
        chunk: (num_steps, num_joints) joint angles in degrees
        fk: SimpleFK instance (or any object with forward_position method)

    Returns:
        (num_steps, 3) array of EE (x, y, z) positions
    """
    positions = []
    for step in range(chunk.shape[0]):
        joint_angles = chunk[step]
        pos = fk.forward_position(joint_angles)
        positions.append(pos)
    return np.array(positions)


_VIEWS_3D = [
    ("Isometric", 25, 45),
    ("Top (XY)", 90, -90),
    ("Front (XZ)", 0, -90),
    ("Side (YZ)", 0, 0),
]


def _compute_axis_limits(ee_trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute shared axis limits covering all trajectories and the origin, with a small margin."""
    all_points = np.concatenate(ee_trajectories + [np.zeros((1, 3))], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    spans = maxs - mins
    # Avoid zero-span axes (e.g., planar motion)
    spans = np.where(spans < 1e-6, 1e-3, spans)
    margin = spans * 0.1
    lo = mins - margin
    hi = maxs + margin
    return lo, hi, hi - lo


def _draw_trajectories_on_axis(
    ax,
    ee_trajectories: list[np.ndarray],
    cmap,
    initial_ee: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    box_aspect: np.ndarray,
    step_label_stride: int = 10,
):
    """Draw all chunk trajectories plus origin and initial-EE markers onto a single 3D axis."""
    for i, traj in enumerate(ee_trajectories):
        color = cmap(i)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(*traj[0], color=color, marker="o", s=25, zorder=5)
        ax.scatter(*traj[-1], color=color, marker="x", s=25, zorder=5)

        # Step-index labels (every `step_label_stride` steps + final step)
        if step_label_stride and step_label_stride > 0:
            n = len(traj)
            idxs = list(range(0, n, step_label_stride))
            if idxs[-1] != n - 1:
                idxs.append(n - 1)
            for s in idxs:
                ax.text(
                    traj[s, 0], traj[s, 1], traj[s, 2],
                    f"{s}",
                    color=color, fontsize=6, alpha=0.9, zorder=6,
                )

    # Reference points: origin (robot base) and initial EE pose
    ax.scatter(0, 0, 0, color="black", marker="*", s=200, zorder=10, label="Origin")
    ax.scatter(
        *initial_ee,
        facecolors="none",
        edgecolors="red",
        marker="D",
        s=150,
        linewidths=2,
        zorder=10,
        label="Initial EE",
    )

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(box_aspect)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


def plot_ee_trajectory_3d(
    ee_trajectories: list[np.ndarray],
    output_path: str | None = None,
    step_label_stride: int = 10,
):
    """Plot EE trajectories in 3D from multiple viewpoints.

    Renders a 2x2 grid of axes (isometric, top, front, side) sharing the same
    limits so chunk trajectories can be compared across viewpoints. Robot-base
    origin and the very first EE position are drawn with distinct markers.
    Step index labels are placed every `step_label_stride` steps per chunk.
    """
    cmap = get_cmap("tab20", len(ee_trajectories))
    initial_ee = ee_trajectories[0][0]
    lo, hi, box_aspect = _compute_axis_limits(ee_trajectories)

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    fig.suptitle(f"EE Trajectory — {len(ee_trajectories)} chunks (4 views)", fontsize=13)

    for idx, (name, elev, azim) in enumerate(_VIEWS_3D):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        _draw_trajectories_on_axis(
            ax, ee_trajectories, cmap, initial_ee, lo, hi, box_aspect,
            step_label_stride=step_label_stride,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D trajectory figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_ee_trajectory_single_view(
    ee_trajectories: list[np.ndarray],
    output_path: str | None = None,
    step_label_stride: int = 10,
    elev: float = 31.4,
    azim: float = -159.9,
    title: str | None = None,
):
    """Plot EE trajectories from one fixed 3D viewpoint.

    The default camera angles match the custom eye view used by
    ``scripts/render_ee_trace_video.py --eye 0 -0.05 0.15`` for
    ``results/session_20260604_131637/ee_trace.mp4``.
    """
    cmap = get_cmap("tab20", len(ee_trajectories))
    initial_ee = ee_trajectories[0][0]
    lo, hi, box_aspect = _compute_axis_limits(ee_trajectories)

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    _draw_trajectories_on_axis(
        ax,
        ee_trajectories,
        cmap,
        initial_ee,
        lo,
        hi,
        box_aspect,
        step_label_stride=step_label_stride,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title or f"EE Trajectory — {len(ee_trajectories)} chunks")
    ax.legend(loc="upper left", fontsize=8)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved matching-view 3D trajectory figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def _matplotlib_color_to_hex(rgba) -> str:
    """Convert an RGBA tuple (matplotlib cmap output) to a #rrggbb hex string."""
    r, g, b = (int(round(c * 255)) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_ee_trajectory_3d_html(
    ee_trajectories: list[np.ndarray],
    output_path: str,
    step_label_stride: int = 10,
):
    """Save an interactive 3D EE-trajectory plot as a self-contained HTML file.

    Mouse-drag rotates, scroll zooms, shift-drag pans. Legend entries toggle
    individual chunks. Origin (robot base) and the very first EE position are
    drawn with distinct markers. Every point carries chunk+step info in the
    hover text; every `step_label_stride` steps gets a visible number label.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly is not installed. Run: pip install plotly")
        return

    cmap = get_cmap("tab20", len(ee_trajectories))
    initial_ee = ee_trajectories[0][0]

    fig = go.Figure()

    for i, traj in enumerate(ee_trajectories):
        color = _matplotlib_color_to_hex(cmap(i))
        n = len(traj)
        # Text shown on the 3D plot itself: step number every `stride` steps
        if step_label_stride and step_label_stride > 0:
            label_idxs = set(range(0, n, step_label_stride))
            label_idxs.add(n - 1)
        else:
            label_idxs = set()
        text_labels = [str(s) if s in label_idxs else "" for s in range(n)]
        # Hover text always shows chunk + step + coords
        hover = [
            f"chunk {i} · step {s}<br>x={traj[s,0]:.4f}<br>y={traj[s,1]:.4f}<br>z={traj[s,2]:.4f}"
            for s in range(n)
        ]
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
                name=f"chunk {i}",
                legendgroup=f"chunk{i}",
                showlegend=True,
            )
        )
        # Per-chunk start (circle) and end (x) — larger markers on top
        fig.add_trace(
            go.Scatter3d(
                x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
                mode="markers",
                marker=dict(size=6, color=color, symbol="circle", line=dict(color="black", width=1)),
                legendgroup=f"chunk{i}", showlegend=False,
                hovertext=f"chunk {i} start (step 0)", hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
                mode="markers",
                marker=dict(size=6, color=color, symbol="x"),
                legendgroup=f"chunk{i}", showlegend=False,
                hovertext=f"chunk {i} end (step {n-1})", hoverinfo="text",
            )
        )

    # Reference: robot base origin
    fig.add_trace(
        go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers",
            marker=dict(size=8, color="black", symbol="diamond"),
            name="Origin (0,0,0)",
            hovertext="Origin (robot base)", hoverinfo="text",
        )
    )
    # Reference: initial EE pose
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

    # Equal-aspect axes
    lo, hi, _ = _compute_axis_limits(ee_trajectories)
    fig.update_layout(
        title=f"EE Trajectory — {len(ee_trajectories)} chunks (interactive)",
        scene=dict(
            xaxis=dict(title="X (m)", range=[lo[0], hi[0]]),
            yaxis=dict(title="Y (m)", range=[lo[1], hi[1]]),
            zaxis=dict(title="Z (m)", range=[lo[2], hi[2]]),
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved interactive 3D trajectory to {output_path}")
    print("  Open in a browser to rotate/zoom/pan:")
    print(f"    firefox {output_path} &")
    print(f"    # or: xdg-open {output_path}")


def plot_ee_displacement(
    ee_trajectories: list[np.ndarray],
    fps: float = 30.0,
    output_path: str | None = None,
):
    """Plot per-chunk EE displacement (from each chunk's start) over global time."""
    cmap = get_cmap("tab20", len(ee_trajectories))

    fig, ax_disp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    global_step = 0
    for i, traj in enumerate(ee_trajectories):
        color = cmap(i)
        steps = np.arange(global_step, global_step + len(traj))
        time_s = steps / fps
        disp = np.linalg.norm(traj - traj[0], axis=1)
        ax_disp.plot(time_s, disp, color=color, linewidth=1.2, label=f"chunk {i}")
        global_step += len(traj)

    ax_disp.set_xlabel("Time (s)")
    ax_disp.set_ylabel("Displacement from chunk start (m)")
    ax_disp.set_title(f"Per-chunk EE displacement ({len(ee_trajectories)} chunks)")
    ax_disp.legend(fontsize=6, ncol=6, loc="upper left")
    ax_disp.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved displacement figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_joint_angles(
    chunks: list[np.ndarray],
    action_features: list[str] | None = None,
    fps: float = 30.0,
    output_path: str | None = None,
):
    """Plot raw joint angles over time, colored by chunk.

    Args:
        chunks: list of (num_steps, num_joints) arrays
        action_features: joint names
        fps: action execution frequency
        output_path: if given, save figure
    """
    num_joints = chunks[0].shape[1]
    joint_names = action_features or [f"joint_{j}" for j in range(num_joints)]
    cmap = get_cmap("tab20", len(chunks))

    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2 * num_joints), sharex=True)
    if num_joints == 1:
        axes = [axes]

    global_step = 0
    for i, chunk in enumerate(chunks):
        color = cmap(i)
        steps = np.arange(global_step, global_step + len(chunk))
        time_s = steps / fps
        for j, ax in enumerate(axes):
            ax.plot(time_s, chunk[:, j], color=color, linewidth=1.0, alpha=0.8)
        global_step += len(chunk)

    for j, ax in enumerate(axes):
        ax.set_ylabel(joint_names[j], fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"Joint angles over time ({len(chunks)} chunks)")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved joint plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize saved action chunks as EE trajectories")
    parser.add_argument("--npz", type=str, required=True, help="Path to .npz file with saved chunks")
    parser.add_argument(
        "--urdf",
        type=str,
        default="assets/urdf/so101_robot2.urdf",
        help="Path to robot URDF file",
    )
    parser.add_argument(
        "--ee-frame",
        type=str,
        default="gripper_frame_link",
        help="End-effector frame name in URDF",
    )
    parser.add_argument(
        "--joint-names",
        type=str,
        nargs="+",
        default=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        help="Joint names for FK (order must match action features)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Save figures to this directory instead of showing")
    parser.add_argument("--no-fk", action="store_true", help="Skip FK, only plot joint angles")
    parser.add_argument(
        "--single-view",
        action="store_true",
        help="Also render a fixed-view EE trajectory image matching ee_trace.mp4 by default",
    )
    parser.add_argument("--view-elev", type=float, default=31.4, help="Single-view elevation angle")
    parser.add_argument("--view-azim", type=float, default=-159.9, help="Single-view azimuth angle")
    parser.add_argument(
        "--step-label-stride",
        type=int,
        default=1,
        help="Label every Nth step on the 3D plots (0 disables labels). Default: 1 (every step)",
    )
    args = parser.parse_args()

    # Load chunks
    print(f"Loading chunks from {args.npz}")
    data = load_chunks(args.npz)
    chunks = data["chunks"]
    print(f"  Loaded {len(chunks)} chunks, shape={chunks[0].shape}")
    print(f"  Action features: {data['action_features']}")
    print(f"  FPS: {data['fps']}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Plot joint angles
    joint_out = str(output_dir / "joint_angles.png") if output_dir else None
    plot_joint_angles(
        chunks,
        action_features=data["action_features"],
        fps=data["fps"],
        output_path=joint_out,
    )

    # FK + EE trajectory
    if not args.no_fk:
        print(f"Computing FK with URDF: {args.urdf}")
        from chunk_visualizer.simple_fk import SimpleFK

        fk = SimpleFK(
            urdf_path=args.urdf,
            ee_frame=args.ee_frame,
            joint_names=args.joint_names,
        )
        print(f"  Chain: {' -> '.join(j.name for j in fk.chain)}")
        print(f"  Actuated joints: {[j.name for j in fk.actuated_joints]}")

        ee_trajectories = []
        for i, chunk in enumerate(chunks):
            traj = compute_ee_trajectory(chunk, fk)
            ee_trajectories.append(traj)
            print(f"  Chunk {i}: EE range x=[{traj[:,0].min():.4f}, {traj[:,0].max():.4f}] "
                  f"y=[{traj[:,1].min():.4f}, {traj[:,1].max():.4f}] "
                  f"z=[{traj[:,2].min():.4f}, {traj[:,2].max():.4f}]")

        ee_3d_out = str(output_dir / "ee_trajectory_3d.png") if output_dir else None
        ee_disp_out = str(output_dir / "ee_displacement.png") if output_dir else None
        plot_ee_trajectory_3d(
            ee_trajectories, output_path=ee_3d_out, step_label_stride=args.step_label_stride
        )
        if args.single_view:
            single_out = str(output_dir / "ee_trajectory_matching_view.png") if output_dir else None
            plot_ee_trajectory_single_view(
                ee_trajectories,
                output_path=single_out,
                step_label_stride=args.step_label_stride,
                elev=args.view_elev,
                azim=args.view_azim,
                title=f"EE Trajectory — matching ee_trace.mp4 view ({args.view_elev:.1f}, {args.view_azim:.1f})",
            )
        plot_ee_displacement(ee_trajectories, fps=data["fps"], output_path=ee_disp_out)

        if output_dir:
            ee_html_out = str(output_dir / "ee_trajectory_3d.html")
            plot_ee_trajectory_3d_html(
                ee_trajectories, output_path=ee_html_out, step_label_stride=args.step_label_stride
            )

    print("Done.")


if __name__ == "__main__":
    main()
