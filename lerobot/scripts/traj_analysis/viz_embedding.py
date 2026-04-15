"""UMAP / t-SNE 2D embedding visualization.

- Step-level: 모든 timestep 을 개별 점으로 (길이 정규화 X)
- Trajectory-level: 에피소드 1개 = 점 1개 (100-step flatten)
- PNG + HTML 두 버전
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP


HUMAN_COLOR = "#1f77b4"
AUTO_COLOR = "#d62728"


def _embed(X: np.ndarray, method: str, random_state: int = 42) -> np.ndarray:
    """(N, D) → (N, 2)."""
    X = StandardScaler().fit_transform(X)
    if method == "umap":
        return UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    random_state=random_state, verbose=False).fit_transform(X)
    elif method == "tsne":
        perplexity = min(30, max(5, (len(X) - 1) // 3))
        return TSNE(n_components=2, perplexity=perplexity,
                    random_state=random_state, init="pca", learning_rate="auto").fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")


def _scatter_png(
    emb: np.ndarray, labels: np.ndarray, output_path: str,
    title: str, level: str, marker_alpha: float, marker_size: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    mask_h = labels == 0
    mask_a = labels == 1
    ax.scatter(emb[mask_a, 0], emb[mask_a, 1], c=AUTO_COLOR, s=marker_size,
               alpha=marker_alpha, label=f"auto  (n={mask_a.sum()})", linewidths=0)
    ax.scatter(emb[mask_h, 0], emb[mask_h, 1], c=HUMAN_COLOR, s=marker_size,
               alpha=marker_alpha, label=f"human (n={mask_h.sum()})", linewidths=0)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(f"{title}  ({level})")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _scatter_html(
    emb: np.ndarray, labels: np.ndarray, ep_ids: np.ndarray,
    output_path: str, title: str,
) -> None:
    fig = go.Figure()
    mask_h = labels == 0
    mask_a = labels == 1
    fig.add_trace(go.Scatter(
        x=emb[mask_a, 0], y=emb[mask_a, 1], mode="markers",
        marker=dict(color=AUTO_COLOR, size=4, opacity=0.5),
        name=f"auto (n={mask_a.sum()})",
        text=[f"ep={e}" for e in ep_ids[mask_a]],
        hoverinfo="text",
    ))
    fig.add_trace(go.Scatter(
        x=emb[mask_h, 0], y=emb[mask_h, 1], mode="markers",
        marker=dict(color=HUMAN_COLOR, size=4, opacity=0.5),
        name=f"human (n={mask_h.sum()})",
        text=[f"ep={e}" for e in ep_ids[mask_h]],
        hoverinfo="text",
    ))
    fig.update_layout(title=title, xaxis_title="dim 1", yaxis_title="dim 2",
                      margin=dict(l=40, r=20, t=60, b=40))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)


def step_level_embed_and_plot(
    human_stack: np.ndarray, auto_stack: np.ndarray,
    human_ep_ids: np.ndarray, auto_ep_ids: np.ndarray,
    output_dir: Path, feature_name: str, max_points: int = 20000,
) -> dict:
    """모든 timestep 대상. 너무 많으면 random subsample."""
    rng = np.random.default_rng(42)

    # max_points 로 서브샘플 (각 라벨 독립)
    if len(human_stack) > max_points:
        idx = rng.choice(len(human_stack), max_points, replace=False)
        human_stack = human_stack[idx]
        human_ep_ids = human_ep_ids[idx]
    if len(auto_stack) > max_points:
        idx = rng.choice(len(auto_stack), max_points, replace=False)
        auto_stack = auto_stack[idx]
        auto_ep_ids = auto_ep_ids[idx]

    X = np.concatenate([human_stack, auto_stack], axis=0)
    labels = np.concatenate([
        np.zeros(len(human_stack), dtype=np.int32),
        np.ones(len(auto_stack), dtype=np.int32),
    ])
    ep_ids = np.concatenate([human_ep_ids, auto_ep_ids])

    coords = {}
    for method in ("umap", "tsne"):
        emb = _embed(X, method)
        coords[method] = emb
        title = f"{method.upper()} — {feature_name}"
        png = output_dir / f"step_level__{method}__{feature_name}.png"
        html = output_dir / f"step_level__{method}__{feature_name}.html"
        _scatter_png(emb, labels, str(png), title, "step-level",
                     marker_alpha=0.35, marker_size=6)
        _scatter_html(emb, labels, ep_ids, str(html), f"{title} (step-level)")
    return {"human_points": int((labels == 0).sum()),
            "auto_points": int((labels == 1).sum())}


def trajectory_level_embed_and_plot(
    human_flat: np.ndarray, auto_flat: np.ndarray,
    human_ep_ids: np.ndarray, auto_ep_ids: np.ndarray,
    output_dir: Path, feature_name: str,
) -> dict:
    """에피소드 1개 = 점 1개."""
    X = np.concatenate([human_flat, auto_flat], axis=0)
    labels = np.concatenate([
        np.zeros(len(human_flat), dtype=np.int32),
        np.ones(len(auto_flat), dtype=np.int32),
    ])
    ep_ids = np.concatenate([human_ep_ids, auto_ep_ids])

    coords = {}
    for method in ("umap", "tsne"):
        emb = _embed(X, method)
        coords[method] = emb
        title = f"{method.upper()} — {feature_name}"
        png = output_dir / f"trajectory_level__{method}__{feature_name}.png"
        html = output_dir / f"trajectory_level__{method}__{feature_name}.html"
        _scatter_png(emb, labels, str(png), title, "trajectory-level",
                     marker_alpha=0.8, marker_size=60)
        _scatter_html(emb, labels, ep_ids, str(html), f"{title} (trajectory-level)")
    return {"coords_umap": coords["umap"], "coords_tsne": coords["tsne"],
            "labels": labels}
