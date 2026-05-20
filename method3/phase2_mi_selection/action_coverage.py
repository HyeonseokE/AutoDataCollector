"""Action coverage gain ΔH_A — 문서 final_method3_spec §8.

후보 trajectory 가 skill buffer 의 action coverage 를 얼마나 확장하는지:

  d_k^a(z_τ^a, B_A^{(m)})                          — action-space kNN 거리
  d̄_NN^a(B_A^{(m)})                                — buffer 내부 typical spacing
  n_a(z_τ^a) = [log(d_k^a / (d̄_NN^a + ε))]_+       — per-window action novelty
  ΔH_A = mean_{τ∈T_q} n_a(z_τ^a)                   — top-q% window novelty 평균

n_a > 0 이면 후보 action 이 기존 action support 바깥(novel)이라는 뜻이다.
"""
from __future__ import annotations

import math

import numpy as np

from method3.phase2_mi_selection.neighbor_search import (
    knn_mean_distance,
    mean_nn_distance,
)


def action_novelty(
    descriptor: np.ndarray,
    buffer_descriptors: np.ndarray,
    nn_scale: float,
    k: int,
    eps: float = 1e-6,
) -> float:
    """``n_a(z_τ^a) = [log(d_k^a / (d̄_NN^a + ε))]_+`` — point-level novelty (§8).

    Args:
        descriptor: 후보 window 의 action descriptor ``z_τ^a`` (D_z,).
        buffer_descriptors: ``B_A^{(m)}`` — buffer action descriptor (N, D_z).
        nn_scale: ``d̄_NN^a`` — buffer 내부 typical spacing (정규화 분모).
        k: action-space kNN 의 k.
        eps: 분모 안정화 상수.
    """
    d_knn = knn_mean_distance(descriptor, buffer_descriptors, k)
    # d_knn==0 (buffer 와 정확히 일치) → log(0)=-inf → [·]_+=0; ValueError 단락.
    if d_knn <= 0.0:
        return 0.0
    return max(math.log(d_knn / (nn_scale + eps)), 0.0)


def top_quantile_mean(values: list[float], q: float) -> float:
    """action novelty 가 높은 상위 ``q%`` window 의 평균 (문서 §8 T_q).

    상위 window 수 = ``max(1, ceil(q · T))`` — q 가 작아도 최소 1개.
    """
    if not values:
        raise ValueError("values is empty")
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    arr = np.sort(np.asarray(values, dtype=np.float64))[::-1]   # 내림차순
    n_top = max(1, math.ceil(q * arr.shape[0]))
    return float(arr[:n_top].mean())


def action_coverage_gain(
    candidate_descriptors: np.ndarray,
    buffer_descriptors: np.ndarray,
    k: int,
    q: float,
    eps: float = 1e-6,
) -> float:
    """``ΔH_A(D_t, ξ)`` — trajectory-level action coverage gain (문서 §8).

    Args:
        candidate_descriptors: 후보 trajectory 의 window 별 ``z_τ^a`` (T, D_z).
        buffer_descriptors: ``B_A^{(m)}`` — skill buffer action descriptor
            (N, D_z), N >= 2 (d̄_NN 계산에 필요).
        k: action-space kNN 의 k.
        q: top-quantile 비율 (0, 1].
        eps: 분모 안정화 상수.

    Returns:
        ΔH_A — 상위 q% window novelty 평균.
    """
    cand = np.asarray(candidate_descriptors, dtype=np.float64)
    if cand.ndim != 2 or cand.shape[0] == 0:
        raise ValueError(f"candidate_descriptors must be (T>=1, D_z), got {cand.shape}")
    nn_scale = mean_nn_distance(buffer_descriptors)             # d̄_NN^a
    novelties = [
        action_novelty(z, buffer_descriptors, nn_scale, k, eps) for z in cand
    ]
    return top_quantile_mean(novelties, q)
