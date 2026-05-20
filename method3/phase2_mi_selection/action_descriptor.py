"""DCT action descriptor ψ — 문서 final_method3_spec §4.2 / §7.3.

action chunk ``A_{τ:τ+H-1} ∈ R^{H×6}`` 를 시간축 DCT 로 변환하고 앞쪽 K개의
저주파 성분만 남겨 flatten 한다:

  C_τ = DCT_time(A)              — 시간축 DCT-II
  C̃_τ = C_τ[:K, :]              — 앞쪽 K개 저주파 성분 (DC 포함)
  z_τ^a = ψ(A) = vec(C̃_τ)       — flatten → R^{K·6}

K=3 이면 ``z^a ∈ R^18``. K 는 temporal resolution hyperparameter — 너무
작으면 contact timing 차이를 잃고, 너무 크면 jitter 를 novelty 로 과대평가한다.
``dct_energy_optimal_k`` 로 DCT energy preservation ratio 기준 K 를 고를 수 있다.

scipy 의존을 피하려고 DCT-II 행렬을 numpy 로 직접 만든다.
"""
from __future__ import annotations

import numpy as np


def _dct_ii_matrix(n: int) -> np.ndarray:
    """길이 ``n`` 시간축의 DCT-II 행렬 ``D`` (n, n).

    ``C = D @ x`` 가 DCT-II: ``C_k = Σ_t x_t cos(π (t+0.5) k / n)``.
    """
    t = np.arange(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    return np.cos(np.pi * (t[None, :] + 0.5) * k[:, None] / n)


def dct_time(action_chunk: np.ndarray) -> np.ndarray:
    """action chunk 의 시간축 DCT-II 계수 ``C_τ`` (문서 §4.2).

    Args:
        action_chunk: ``A_{τ:τ+H-1}`` — (H, action_dim).

    Returns:
        (H, action_dim) DCT 계수. row 0 = DC(가장 느린 성분).
    """
    a = np.asarray(action_chunk, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"action_chunk must be (H, action_dim), got {a.shape}")
    return _dct_ii_matrix(a.shape[0]) @ a


def dct_action_descriptor(action_chunk: np.ndarray, n_coeffs: int) -> np.ndarray:
    """``z^a = ψ(A) = vec(C[:K])`` — DCT action descriptor (문서 §4.2).

    Args:
        action_chunk: ``A_{τ:τ+H-1}`` — (H, action_dim).
        n_coeffs: K — 남길 저주파 성분 개수 (1 <= K <= H).

    Returns:
        (n_coeffs * action_dim,) flatten 된 action descriptor.
    """
    coeffs = dct_time(action_chunk)
    if not 1 <= n_coeffs <= coeffs.shape[0]:
        raise ValueError(
            f"n_coeffs must be in [1, H={coeffs.shape[0]}], got {n_coeffs}"
        )
    return coeffs[:n_coeffs, :].reshape(-1)


def dct_energy_optimal_k(action_chunk: np.ndarray, eta: float) -> int:
    """DCT energy preservation ratio 가 ``eta`` 이상이 되는 최소 K (문서 §4.2).

    ``K = min{ K' : Σ_{k<K'}‖C_k‖² / Σ_k‖C_k‖² >= eta }``.

    Args:
        action_chunk: ``A_{τ:τ+H-1}`` — (H, action_dim).
        eta: 보존할 energy 비율 (0, 1].

    Returns:
        조건을 만족하는 최소 K (>= 1).
    """
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
    coeffs = dct_time(action_chunk)
    energy = np.sum(coeffs ** 2, axis=1)             # (H,) per-coefficient energy
    total = float(energy.sum())
    if total <= 0.0:
        return 1
    ratio = np.cumsum(energy) / total
    return int(np.searchsorted(ratio, eta) + 1)
