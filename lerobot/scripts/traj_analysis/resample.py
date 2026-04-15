"""Trajectory resampling utilities.

- Step-level 분석용: fps 서브샘플링 (30 → 10fps)
- Trajectory-level 분석용: 고정 길이 선형 보간
"""
from __future__ import annotations

from typing import List

import numpy as np


def subsample_fps(trajs: List[np.ndarray], stride: int) -> List[np.ndarray]:
    """매 stride 프레임마다 1개 유지 (30fps→10fps 면 stride=3)."""
    if stride <= 1:
        return [t.copy() for t in trajs]
    return [t[::stride].copy() for t in trajs]


def resample_fixed_length(trajs: List[np.ndarray], T_target: int) -> np.ndarray:
    """각 (T_i, D) 를 (T_target, D) 로 선형 보간 → stack 해서 (N, T_target, D).

    한 점짜리(degenerate) 궤적은 값 복제로 처리.
    """
    if not trajs:
        return np.empty((0, T_target, 0), dtype=np.float32)

    D = trajs[0].shape[1]
    out = np.empty((len(trajs), T_target, D), dtype=np.float32)
    for i, traj in enumerate(trajs):
        T = len(traj)
        if T == 0:
            out[i] = 0
            continue
        if T == 1:
            out[i] = np.broadcast_to(traj[0], (T_target, D))
            continue
        src_idx = np.linspace(0, T - 1, T_target, dtype=np.float64)
        lo = np.floor(src_idx).astype(np.int64)
        hi = np.clip(lo + 1, 0, T - 1)
        frac = (src_idx - lo).astype(np.float32)[:, None]
        out[i] = traj[lo] * (1 - frac) + traj[hi] * frac
    return out


def flatten_for_embedding(resampled: np.ndarray) -> np.ndarray:
    """(N, T, D) → (N, T*D) — UMAP/t-SNE 입력용."""
    N = resampled.shape[0]
    return resampled.reshape(N, -1)


def concat_all_steps(trajs: List[np.ndarray]) -> np.ndarray:
    """Step-level 임베딩용: 모든 timestep 을 하나로 concat.

    Returns:
        (sum_T, D) ndarray
    """
    if not trajs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(trajs, axis=0)
