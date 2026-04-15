"""정량 지표 — DTW, MMD, energy distance, 경로 길이, UMAP hull area."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from fastdtw import fastdtw
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def _basic_stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def episode_length_stats(trajs: List[np.ndarray]) -> dict:
    return _basic_stats(np.array([len(t) for t in trajs], dtype=np.float32))


def path_length_xyz(trajs_xyz: List[np.ndarray]) -> dict:
    """EE 경로의 누적 거리."""
    lens = []
    for t in trajs_xyz:
        if len(t) < 2:
            lens.append(0.0)
            continue
        diffs = np.diff(t, axis=0)
        lens.append(float(np.sqrt((diffs ** 2).sum(axis=1)).sum()))
    return _basic_stats(np.array(lens, dtype=np.float32))


def intra_dtw_mean(trajs: List[np.ndarray], max_pairs: int = 200, seed: int = 42) -> dict:
    """같은 데이터셋 내 에피소드 쌍의 DTW 거리 평균 (다양성 지표)."""
    n = len(trajs)
    if n < 2:
        return {"mean": 0.0, "std": 0.0, "num_pairs": 0}
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(max_pairs):
        i, j = rng.choice(n, 2, replace=False)
        pairs.append((int(i), int(j)))
    dists = []
    for i, j in pairs:
        a = trajs[i].astype(np.float64)
        b = trajs[j].astype(np.float64)
        d, _ = fastdtw(a, b, dist=2)
        # 길이 정규화
        dists.append(d / (len(a) + len(b)))
    dists = np.array(dists)
    return {"mean": float(dists.mean()), "std": float(dists.std()),
            "num_pairs": len(pairs)}


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None,
            max_samples: int = 2000, seed: int = 42) -> float:
    """Maximum Mean Discrepancy (RBF kernel) 의 제곱 추정치."""
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]
    XY = np.concatenate([X, Y], axis=0)
    if gamma is None:
        # 중간 거리 기반 median heuristic
        D = cdist(XY, XY, "euclidean")
        med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        gamma = 1.0 / (2 * med ** 2 + 1e-9)

    def rbf(A, B):
        D2 = cdist(A, B, "sqeuclidean")
        return np.exp(-gamma * D2)

    Kxx = rbf(X, X); np.fill_diagonal(Kxx, 0.0)
    Kyy = rbf(Y, Y); np.fill_diagonal(Kyy, 0.0)
    Kxy = rbf(X, Y)
    m, n = len(X), len(Y)
    mmd2 = Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.sum() / (m * n)
    return float(mmd2)


def energy_distance(X: np.ndarray, Y: np.ndarray,
                    max_samples: int = 2000, seed: int = 42) -> float:
    """Székely energy distance."""
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]
    A = cdist(X, Y, "euclidean").mean()
    B = cdist(X, X, "euclidean").mean()
    C = cdist(Y, Y, "euclidean").mean()
    return float(2 * A - B - C)


def hull_area_2d(coords: np.ndarray) -> float:
    """2D 좌표의 convex hull 면적."""
    if len(coords) < 3:
        return 0.0
    try:
        hull = ConvexHull(coords)
        return float(hull.volume)  # 2D 에서 volume == area
    except Exception:
        return 0.0


def summarize(
    label: str,
    trajs_state_rad: List[np.ndarray],
    trajs_action_rad: List[np.ndarray],
    trajs_xyz: List[np.ndarray],
) -> dict:
    return {
        "label": label,
        "num_episodes": len(trajs_state_rad),
        "episode_length": episode_length_stats(trajs_state_rad),
        "path_length_ee": path_length_xyz(trajs_xyz),
        "intra_dtw_state_rad": intra_dtw_mean(trajs_state_rad),
    }
