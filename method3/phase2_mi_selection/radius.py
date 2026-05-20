"""State-neighborhood radius ρ_m — 문서 final_method3_spec §16.

skill ``m`` 의 covered-state 판정(§9.1)에 쓰는 local neighborhood scale 을
buffer 에서 자동으로 잡는다:

  r_i^{(m)} = d_k^s(e_i, E^{(m)})                  — 각 point 의 kNN 거리
  ρ_m = Quantile_{0.7}({r_i^{(m)}})                — 그 분포의 0.7 quantile

즉 skill 별 embedding 밀도에 맞춰 radius 를 적응적으로 정한다.
"""
from __future__ import annotations

import numpy as np


def state_neighborhood_radius(
    state_keys: np.ndarray,
    k: int,
    quantile: float = 0.7,
) -> float:
    """``ρ_m`` — skill buffer 의 kNN 거리 분포 quantile (문서 §16).

    Args:
        state_keys: ``E^{(m)}`` — skill buffer 의 state key 들 (N, D_e), N >= 2.
        k: 각 point 의 kNN 거리에 쓰는 k. N-1 보다 크면 clamp.
        quantile: radius 로 쓸 분위수 (기본 0.7, 문서 §16).

    Returns:
        ρ_m — covered-state 판정용 radius.
    """
    keys = np.asarray(state_keys, dtype=np.float64)
    if keys.ndim != 2 or keys.shape[0] < 2:
        raise ValueError(f"need a (N>=2, D) array, got {keys.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    n = keys.shape[0]
    diff = keys[:, None, :] - keys[None, :, :]      # (N, N, D)
    dist = np.linalg.norm(diff, axis=2)             # (N, N)
    np.fill_diagonal(dist, np.inf)                  # 자기 자신 제외
    kk = min(k, n - 1)
    # 각 row 에서 가장 가까운 kk개 거리의 평균 = r_i.
    r = np.partition(dist, kk - 1, axis=1)[:, :kk].mean(axis=1)
    return float(np.quantile(r, quantile))
