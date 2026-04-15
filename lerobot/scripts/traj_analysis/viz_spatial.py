"""3D Spatial Overlay: EE 궤적 중첩 시각화.

- line_overlay: 라인 alpha-blending 으로 밀도 표현
- voxel_density: 3D 히스토그램 밀도 스캐터
- interactive: plotly HTML (회전/줌)

모든 시각화에 다음 마커가 함께 표시됨:
- 로봇 base frame 원점 (0, 0, 0): 검은 별 (양 데이터셋 공통)
- 각 에피소드의 초기 EE 위치: human=초록, auto=주황 (라인 색과 명확히 구분)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


HUMAN_COLOR = "#1f77b4"   # blue  — human 라인
AUTO_COLOR = "#d62728"    # red   — auto 라인

HUMAN_INIT_COLOR = "#2ca02c"  # green  — human 초기 EE
AUTO_INIT_COLOR = "#ff7f0e"   # orange — auto 초기 EE
ORIGIN_COLOR = "#000000"      # black  — 로봇 base frame 원점


def _auto_alpha(n: int, scale: float = 2.0, cap: float = 0.35) -> float:
    """경로 수에 맞춰 per-line alpha 자동 조절."""
    return float(min(cap, scale / max(np.sqrt(n), 1.0)))


def _initial_points(trajs: List[np.ndarray]) -> np.ndarray:
    """각 에피소드의 첫 EE 위치 → (N, 3)."""
    if not trajs:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack([t[0] for t in trajs if len(t) > 0])


# ─────────────────────────────────────────────
# matplotlib helpers
# ─────────────────────────────────────────────

def _draw_origin_mpl(ax) -> None:
    """로봇 base frame 원점 (0,0,0) 표시."""
    ax.scatter([0], [0], [0], c=ORIGIN_COLOR, s=160, marker="*",
               edgecolors="white", linewidths=1.2, zorder=10,
               label="origin (0,0,0)")


def _draw_initial_ee_mpl(ax, init_pts: np.ndarray, color: str, label: str) -> None:
    if len(init_pts) == 0:
        return
    ax.scatter(init_pts[:, 0], init_pts[:, 1], init_pts[:, 2],
               c=color, s=45, marker="o", edgecolors="black", linewidths=0.5,
               alpha=0.85, zorder=9, label=label)


# ─────────────────────────────────────────────
# 1) 라인 오버레이 (matplotlib)
# ─────────────────────────────────────────────

def plot_line_overlay(
    human_xyz: List[np.ndarray],
    auto_xyz: List[np.ndarray],
    output_path: str,
    title: str = "EE Trajectory Overlay (x, y, z)",
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    a_human = _auto_alpha(len(human_xyz))
    a_auto = _auto_alpha(len(auto_xyz))

    for t in auto_xyz:
        ax.plot(t[:, 0], t[:, 1], t[:, 2], c=AUTO_COLOR, alpha=a_auto, lw=0.6)
    for t in human_xyz:
        ax.plot(t[:, 0], t[:, 1], t[:, 2], c=HUMAN_COLOR, alpha=a_human, lw=0.6)

    # 마커: 원점 + 초기 EE
    _draw_origin_mpl(ax)
    h_init = _initial_points(human_xyz)
    a_init = _initial_points(auto_xyz)
    _draw_initial_ee_mpl(ax, h_init, HUMAN_INIT_COLOR, f"human init EE (n={len(h_init)})")
    _draw_initial_ee_mpl(ax, a_init, AUTO_INIT_COLOR, f"auto init EE (n={len(a_init)})")

    # 라인 범례 (더미)
    ax.plot([], [], [], c=HUMAN_COLOR, lw=2, label=f"human path (n={len(human_xyz)})")
    ax.plot([], [], [], c=AUTO_COLOR, lw=2, label=f"auto path (n={len(auto_xyz)})")
    ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
# 2) Voxel density (matplotlib)
# ─────────────────────────────────────────────

def _voxel_density(xyz_all: np.ndarray, bins: int) -> Tuple[np.ndarray, Tuple]:
    """(N, 3) → voxel count 히스토그램."""
    H, edges = np.histogramdd(xyz_all, bins=bins)
    return H, edges


def plot_voxel_density(
    human_xyz: List[np.ndarray],
    auto_xyz: List[np.ndarray],
    output_path: str,
    bins: int = 50,
    title: str = "EE Voxel Density (log count)",
) -> None:
    """3D 공간을 voxel 로 나누고 log(count) 로 투명도/크기 조절."""
    fig = plt.figure(figsize=(12, 6))

    for idx, (xyz_list, color, init_color, label) in enumerate(
        [
            (human_xyz, HUMAN_COLOR, HUMAN_INIT_COLOR, "human"),
            (auto_xyz, AUTO_COLOR, AUTO_INIT_COLOR, "auto"),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        if not xyz_list:
            ax.set_title(f"{label} (empty)")
            continue
        all_pts = np.concatenate(xyz_list, axis=0)
        H, edges = _voxel_density(all_pts, bins)
        xc = 0.5 * (edges[0][:-1] + edges[0][1:])
        yc = 0.5 * (edges[1][:-1] + edges[1][1:])
        zc = 0.5 * (edges[2][:-1] + edges[2][1:])
        XX, YY, ZZ = np.meshgrid(xc, yc, zc, indexing="ij")
        mask = H > 0
        log_count = np.log1p(H[mask])
        s = 4.0 + 40.0 * (log_count / log_count.max())
        a = 0.15 + 0.70 * (log_count / log_count.max())
        ax.scatter(XX[mask], YY[mask], ZZ[mask],
                   c=color, s=s, alpha=a, linewidths=0)

        # 마커: 원점 + 초기 EE
        _draw_origin_mpl(ax)
        init = _initial_points(xyz_list)
        _draw_initial_ee_mpl(ax, init, init_color, f"init EE (n={len(init)})")

        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(f"{label}  (voxels visited = {int(mask.sum())}/{bins**3})")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
# 3) Interactive (plotly)
# ─────────────────────────────────────────────

def plot_interactive_3d(
    human_xyz: List[np.ndarray],
    auto_xyz: List[np.ndarray],
    output_path: str,
    title: str = "Interactive EE Trajectory Overlay",
) -> None:
    fig = go.Figure()

    # 라인 (auto 먼저 → 시각적으로 human 이 위에)
    for i, t in enumerate(auto_xyz):
        fig.add_trace(go.Scatter3d(
            x=t[:, 0], y=t[:, 1], z=t[:, 2],
            mode="lines",
            line=dict(color=AUTO_COLOR, width=2),
            opacity=0.20,
            showlegend=(i == 0),
            name=f"auto path (n={len(auto_xyz)})",
            hoverinfo="skip",
        ))
    for i, t in enumerate(human_xyz):
        fig.add_trace(go.Scatter3d(
            x=t[:, 0], y=t[:, 1], z=t[:, 2],
            mode="lines",
            line=dict(color=HUMAN_COLOR, width=2),
            opacity=0.20,
            showlegend=(i == 0),
            name=f"human path (n={len(human_xyz)})",
            hoverinfo="skip",
        ))

    # 초기 EE 위치 — 데이터셋별 색 구분
    h_init = _initial_points(human_xyz)
    a_init = _initial_points(auto_xyz)
    if len(a_init):
        fig.add_trace(go.Scatter3d(
            x=a_init[:, 0], y=a_init[:, 1], z=a_init[:, 2],
            mode="markers",
            marker=dict(color=AUTO_INIT_COLOR, size=5,
                        line=dict(color="black", width=1)),
            name=f"auto init EE (n={len(a_init)})",
            hovertemplate="auto init<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        ))
    if len(h_init):
        fig.add_trace(go.Scatter3d(
            x=h_init[:, 0], y=h_init[:, 1], z=h_init[:, 2],
            mode="markers",
            marker=dict(color=HUMAN_INIT_COLOR, size=5,
                        line=dict(color="black", width=1)),
            name=f"human init EE (n={len(h_init)})",
            hovertemplate="human init<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        ))

    # 로봇 base frame 원점
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(color=ORIGIN_COLOR, size=10, symbol="diamond",
                    line=dict(color="white", width=2)),
        name="origin (0,0,0)",
        hovertemplate="base frame origin<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
