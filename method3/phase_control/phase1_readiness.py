"""Phase1 saturation — Phase2-readiness — 문서 final_method3_spec §15.1.

Phase1 은 state novelty 를 많이 만드는 것이 목적이 아니라, Phase2 에서 local
action consistency 를 평가할 수 있을 만큼 state support 를 만드는 것이 목적이다.
따라서 "현재 buffer 가 Phase2 후보의 H(A|S) proxy 를 계산할 수 있을 만큼
covered 됐는가" 를 기준으로 종료한다.

  R_cov(ξ) = |T_covered(ξ)| / |T(ξ)|                      — 후보별 covered ratio
  R̄_cov^(m) = mean_{ξ∈Ξ_probe^(m)} R_cov(ξ)                — skill 평균
  R_ready = (1/M) Σ_m R̄_cov^(m)                            — 전체 readiness
  Phase1 종료: R_ready > τ_ready   (기본 τ_ready = 0.7)

covered window 판정은 §9.1 과 동일한 ``covered_windows`` 를 재사용한다. probe
candidate 의 state key ``e_τ`` 와 buffer key ``e_i`` 는 호출부가 계산해 넘긴다
(Phase1 중에는 §7 의 임시 key 일 수 있음 — 이 모듈은 encoder-agnostic).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from method3.phase2_mi_selection.conditional_ambiguity import covered_windows
from method3.phase2_mi_selection.radius import state_neighborhood_radius


@dataclass
class Phase1ReadinessConfig:
    """Phase1 readiness 평가 파라미터 (문서 §15.1, §16)."""

    tau_ready: float = 0.7        # §15.1 — Phase1 종료 임계값
    k_min: int = 3                # §9.1 — covered window 최소 neighbor 수
    radius_k: int = 5             # §16 — ρ_m 계산용 kNN k
    radius_quantile: float = 0.7  # §16 — ρ_m quantile


@dataclass(frozen=True)
class SkillProbe:
    """skill ``m`` 의 readiness 입력 — probe 후보 + 현재 buffer (문서 §15.1).

    Phase1 중에도 실제 실행 없이 Phase2 후보 generator 로 만든 probe trajectory
    들의 window state key 를 ``probe_candidate_keys`` 에 담는다.
    """

    skill_id: str
    probe_candidate_keys: list[np.ndarray]   # Ξ_probe^(m) — 각 (T_i, D_e)
    db_state_keys: np.ndarray                # B_t^(m) 의 state key (N, D_e)


@dataclass(frozen=True)
class Phase1ReadinessReport:
    """``evaluate_phase1_readiness`` 결과 (문서 §15.1)."""

    r_ready: float                     # R_ready — 전체 readiness score
    per_skill: dict[str, float]        # R̄_cov^(m) — skill 별 평균 covered ratio
    is_ready: bool                     # R_ready > τ_ready


def candidate_covered_ratio(
    candidate_keys: np.ndarray,
    db_state_keys: np.ndarray,
    radius: float,
    k_min: int = 3,
) -> float:
    """``R_cov(ξ) = |T_covered(ξ)| / |T(ξ)|`` — 후보 trajectory covered ratio (§15.1).

    Args:
        candidate_keys: 후보 trajectory 의 window state key (T, D_e).
        db_state_keys: 비교할 buffer 의 state key (N, D_e).
        radius: ρ_m — state-neighborhood radius.
        k_min: covered 판정 최소 neighbor 수 (§9.1 기본 3).

    Returns:
        covered window 비율 [0, 1]. window 가 0개면 0.0.
    """
    n_windows = int(np.asarray(candidate_keys).shape[0])
    if n_windows == 0:
        return 0.0
    covered = covered_windows(candidate_keys, db_state_keys, radius, k_min)
    return len(covered) / n_windows


def evaluate_phase1_readiness(
    skill_probes: list[SkillProbe],
    config: Phase1ReadinessConfig | None = None,
) -> Phase1ReadinessReport:
    """Phase2-readiness score 를 평가한다 (문서 §15.1).

    skill 마다 buffer 에서 ρ_m 을 추정(§16)하고, probe 후보들의 covered ratio
    평균 ``R̄_cov^(m)`` 을 구한 뒤, skill 평균 ``R_ready`` 를 계산한다.

    Args:
        skill_probes: skill 당 하나의 ``SkillProbe`` (M개 skill).
        config: readiness 파라미터.

    Returns:
        Phase1ReadinessReport. buffer 가 2개 미만인 skill 은 covered 불가로
        ``R̄_cov^(m) = 0`` 처리한다 (cold-start).
    """
    cfg = config or Phase1ReadinessConfig()
    per_skill: dict[str, float] = {}
    for probe in skill_probes:
        db = np.asarray(probe.db_state_keys, dtype=np.float64)
        if db.ndim != 2 or db.shape[0] < 2:
            # buffer 가 너무 작아 ρ_m 추정 불가 → 아무 window 도 covered 안 됨.
            per_skill[str(probe.skill_id)] = 0.0
            continue
        radius = state_neighborhood_radius(db, cfg.radius_k, cfg.radius_quantile)
        ratios = [
            candidate_covered_ratio(c, db, radius, cfg.k_min)
            for c in probe.probe_candidate_keys
        ]
        per_skill[str(probe.skill_id)] = float(np.mean(ratios)) if ratios else 0.0

    r_ready = float(np.mean(list(per_skill.values()))) if per_skill else 0.0
    return Phase1ReadinessReport(
        r_ready=r_ready,
        per_skill=per_skill,
        is_ready=r_ready > cfg.tau_ready,
    )
