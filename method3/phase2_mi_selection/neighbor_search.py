"""L2 neighbor-search primitives for Phase2 MI scoring.

action-space (``d_a``) 와 state-space (``d_s``) 거리는 모두 Euclidean L2 로
둔다 (문서가 별도 metric 을 지정하지 않음). 아래 세 가지가 §8·§9·§16 의
공통 building block 이다:

  ``knn_mean_distance``  — query 와 buffer 간 kNN 평균 거리 (d_k)
  ``mean_nn_distance``   — buffer 내부 평균 nearest-neighbor 거리 (d̄_NN)
  ``min_distance``       — query 와 buffer 간 최소 거리 (d_min, §9.3)
"""
from __future__ import annotations

import numpy as np


def knn_mean_distance(query: np.ndarray, keys: np.ndarray, k: int) -> float:
    """``d_k(q, B)`` — query 와 buffer 간 kNN 평균 L2 거리 (문서 §8/§9).

    Args:
        query: descriptor (D,).
        keys: buffer descriptor 들 (N, D), N >= 1.
        k: nearest-neighbor 개수. N 보다 크면 N 으로 clamp.
    """
    keys = np.asarray(keys, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    if keys.ndim != 2 or keys.shape[0] == 0:
        raise ValueError("keys must be a non-empty (N, D) array")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    dists = np.linalg.norm(keys - query[None, :], axis=1)
    kk = min(k, dists.shape[0])
    return float(np.partition(dists, kk - 1)[:kk].mean())


def mean_nn_distance(keys: np.ndarray) -> float:
    """``d̄_NN(B)`` — buffer 내부 평균 nearest-neighbor 거리 (문서 §8/§9).

    각 ``z_i`` 에 대해 ``min_{l != i} d(z_i, z_l)`` 의 평균. buffer 가 보통
    어느 정도 촘촘한지(typical spacing)를 나타내며 novelty scale 정규화 분모.

    Args:
        keys: buffer descriptor 들 (N, D), N >= 2.
    """
    keys = np.asarray(keys, dtype=np.float64)
    if keys.ndim != 2 or keys.shape[0] < 2:
        raise ValueError(f"need a (N>=2, D) array, got {keys.shape}")
    diff = keys[:, None, :] - keys[None, :, :]          # (N, N, D)
    dist = np.linalg.norm(diff, axis=2)                  # (N, N)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min(axis=1).mean())


def min_distance(query: np.ndarray, keys: np.ndarray) -> float:
    """``d_min(q, B)`` — query 와 buffer 간 최소 L2 거리 (문서 §9.3).

    Args:
        query: descriptor (D,).
        keys: buffer descriptor 들 (N, D), N >= 1.
    """
    keys = np.asarray(keys, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    if keys.ndim != 2 or keys.shape[0] == 0:
        raise ValueError("keys must be a non-empty (N, D) array")
    return float(np.linalg.norm(keys - query[None, :], axis=1).min())
