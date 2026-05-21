"""Conditional action ambiguity increase ΔH_A|S — 문서 final_method3_spec §9.

후보가 유사 state 에서 action ambiguity 를 얼마나 키우는지:

  §9.1 covered window — state-neighbor 가 k_min 개 이상인 window
       N_ρ^s(e_τ) = {i ∈ B_t^{(m)} : d_s(e_τ, e_i) < ρ_m}
  §9.2 local action support A_{e_τ} = {z_i^a : i ∈ N_ρ^s(e_τ)}
  §9.3 Δh_A|S(e_τ, z_τ^a) = [log(d_min^a / (s_a + ε))]_+
       s_a = max(d̄_NN^a(A_{e_τ}), s_min^a)
  §9.4 ΔH_A|S = Agg_{τ∈T_covered} Δh_A|S   (mean 기본, max 는 보수적 ablation)

covered window 가 없으면 후보는 under-covered — MI selector 로 신뢰성 있게
평가하지 않고 state-seeding branch 로 보낸다 (§9.1). 이 모듈은 covered window
수를 함께 반환하여 호출부(selector)가 그 분기를 판단하게 한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from method3.phase2_mi_selection.neighbor_search import mean_nn_distance, min_distance


@dataclass(frozen=True)
class ConditionalAmbiguityReport:
    """``conditional_ambiguity`` 결과 — §9.4 값 + covered 통계."""

    delta_h_a_given_s: float   # ΔH_A|S — covered window aggregation
    n_covered: int             # |T_covered(ξ)|
    n_windows: int             # |T(ξ)|

    @property
    def covered_ratio(self) -> float:
        """``R_cov(ξ) = |T_covered| / |T|`` (문서 §15.1)."""
        return self.n_covered / self.n_windows if self.n_windows else 0.0


def covered_windows(
    candidate_keys: np.ndarray,
    db_state_keys: np.ndarray,
    radius: float,
    k_min: int = 3,
) -> list[tuple[int, np.ndarray]]:
    """covered window 와 그 state-neighbor index 목록 (문서 §9.1).

    Args:
        candidate_keys: 후보 trajectory 의 window 별 state key ``e_τ`` (T, D_e).
        db_state_keys: ``B_t^{(m)}`` 의 state key (N, D_e).
        radius: ρ_m — state-neighborhood radius.
        k_min: covered 판정 최소 neighbor 수 (문서 §9.1 기본 3).

    Returns:
        ``[(window_idx, neighbor_indices), ...]`` — covered window 만.
    """
    cand = np.asarray(candidate_keys, dtype=np.float64)
    db = np.asarray(db_state_keys, dtype=np.float64)
    if cand.ndim != 2:
        raise ValueError(f"candidate_keys must be (T, D_e), got {cand.shape}")
    out: list[tuple[int, np.ndarray]] = []
    if db.ndim != 2 or db.shape[0] == 0:
        return out
    for tau in range(cand.shape[0]):
        dists = np.linalg.norm(db - cand[tau][None, :], axis=1)
        neighbors = np.flatnonzero(dists < radius)
        if neighbors.shape[0] >= k_min:
            out.append((tau, neighbors))
    return out


def conditional_ambiguity(
    candidate_keys: np.ndarray,
    candidate_descriptors: np.ndarray,
    db_state_keys: np.ndarray,
    db_action_descriptors: np.ndarray,
    radius: float,
    k_min: int = 3,
    s_min: float = 1e-3,
    eps: float = 1e-6,
    agg: str = "mean",
) -> ConditionalAmbiguityReport:
    """``ΔH_A|S(D_t, ξ)`` — trajectory-level conditional ambiguity (문서 §9).

    Args:
        candidate_keys: window 별 state key ``e_τ`` (T, D_e).
        candidate_descriptors: window 별 action descriptor ``z_τ^a`` (T, D_z).
        db_state_keys: ``B_t^{(m)}`` 의 state key (N, D_e).
        db_action_descriptors: ``B_t^{(m)}`` 의 action descriptor (N, D_z).
        radius: ρ_m — state-neighborhood radius.
        k_min: covered 판정 최소 neighbor 수 (문서 §9.1 기본 3).
        s_min: ``s_a`` scale floor — local support 가 너무 좁을 때 penalty 폭주 방지.
        eps: 분모 안정화 상수.
        agg: covered window aggregation — ``"mean"``(기본) 또는 ``"max"``.

    Returns:
        ConditionalAmbiguityReport. covered window 가 0개면 ΔH_A|S=0.0,
        n_covered=0 — 호출부가 under-covered 분기를 판단한다.
    """
    if agg not in ("mean", "max"):
        raise ValueError(f"agg must be 'mean' or 'max', got {agg!r}")
    cand_z = np.asarray(candidate_descriptors, dtype=np.float64)
    db_z = np.asarray(db_action_descriptors, dtype=np.float64)
    n_windows = int(np.asarray(candidate_keys).shape[0])

    covered = covered_windows(candidate_keys, db_state_keys, radius, k_min)
    # use_dct_target=True 면 cand_z 가 *skill-atomic single window* (1, D_z) —
    # 모든 τ 에 동일 row broadcast. legacy frame-level path 면 cand_z.shape[0]==T.
    _cand_n = cand_z.shape[0]
    deltas: list[float] = []
    for tau, neighbors in covered:
        support = db_z[neighbors]                          # §9.2 A_{e_τ}
        _cz_row = cand_z[tau if tau < _cand_n else 0]
        d_min = min_distance(_cz_row, support)             # §9.3 d_min^a
        # covered → |support| >= k_min >= 2 이므로 d̄_NN 계산 가능.
        s_a = max(mean_nn_distance(support), s_min)        # §9.3 s_a
        if d_min <= 0.0:
            deltas.append(0.0)                             # log(0) 단락
        else:
            deltas.append(max(math.log(d_min / (s_a + eps)), 0.0))   # §9.3

    if not deltas:
        return ConditionalAmbiguityReport(0.0, 0, n_windows)
    value = float(np.mean(deltas) if agg == "mean" else np.max(deltas))
    return ConditionalAmbiguityReport(value, len(deltas), n_windows)
