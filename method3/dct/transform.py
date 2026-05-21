"""skill trajectory ↔ DCT feature 변환 (DCT-II orthonormal).

사용자 명시 paradigm:
  - VLA 학습 target = skill 단위 traj 를 DCT 로 변환한 feature (L0=50 차원).
  - candidate selection 시점에도 동일 변환 (skill 단위 DCT) 으로 후보를
    DCT 피쳐로 만듦.

두 함수만 노출:

    traj_to_dct(traj, L0=50)        →  (L0, dof)
    dct_to_traj(coeffs, T_target)   →  (T_target, dof)

resample 은 cubic interpolation. DCT 는 scipy.fft.dct(type=2, norm='ortho').
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct
from scipy.interpolate import CubicSpline


def _resample_cubic(traj: np.ndarray, target_len: int) -> np.ndarray:
    """``traj (T, dof)`` 를 cubic interpolation 으로 ``target_len`` 길이로 resample.

    T < 2 이면 cubic 불가 → 단일 시점을 target_len 번 반복하거나, T=2 이면
    linear fallback. T >= 4 일 때 cubic spline 이 사용된다 (scipy default).
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"traj must be (T, dof); got {traj.shape}")
    T, dof = traj.shape
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1; got {target_len}")
    if T == 0:
        raise ValueError("traj must have at least 1 row")
    if T == 1:
        return np.tile(traj, (target_len, 1))
    t_src = np.linspace(0.0, 1.0, T)
    t_dst = np.linspace(0.0, 1.0, target_len)
    if T == 2:
        # cubic spline needs >= 2 points but degrades to linear with 2 — explicit.
        out = np.empty((target_len, dof), dtype=np.float64)
        for d in range(dof):
            out[:, d] = np.interp(t_dst, t_src, traj[:, d])
        return out
    cs = CubicSpline(t_src, traj, axis=0, bc_type="natural")
    return cs(t_dst)


def traj_to_dct(traj: np.ndarray, L0: int = 50) -> np.ndarray:
    """``traj (T, dof)`` → ``(L0, dof)`` DCT-II orthonormal coefficients.

    Step:
      1. cubic resample T → L0.
      2. DCT-II (per-joint, axis=time) with norm='ortho'.

    Args:
        traj: (T, dof) skill 단위 trajectory.
        L0: 고정 차원 (default 50, smolvla chunk_size 와 일치).

    Returns:
        (L0, dof) DCT coefficients.
    """
    resampled = _resample_cubic(traj, L0)
    return dct(resampled, type=2, norm="ortho", axis=0)


def dct_to_traj(coeffs: np.ndarray, T_target: int) -> np.ndarray:
    """``coeffs (L0, dof)`` → ``(T_target, dof)`` trajectory 복원.

    Step:
      1. IDCT (DCT-III with norm='ortho') → (L0, dof) resampled trajectory.
      2. cubic resample L0 → T_target.

    Args:
        coeffs: (L0, dof) DCT-II coefficients.
        T_target: 복원할 trajectory 길이.

    Returns:
        (T_target, dof) trajectory.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 2:
        raise ValueError(f"coeffs must be (L0, dof); got {coeffs.shape}")
    resampled = idct(coeffs, type=2, norm="ortho", axis=0)
    return _resample_cubic(resampled, T_target)
