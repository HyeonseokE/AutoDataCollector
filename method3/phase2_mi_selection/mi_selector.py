"""Phase2 MI-style candidate selector — 문서 final_method3_spec §11-12.

후보 trajectory 들을 받아 MI-style score 로 평가·선택한다:

  §11  Q2(ξ) = β·ΔH_A(D_t, ξ) − λ·ΔH_A|S(D_t, ξ)
       ξ* = argmax Q2
  §12  배치 내부 정규화 Q̃2 = (Q2 − μ_Q) / (σ_Q + ε)
       accept: Q̃2(ξ*) > τ_Q̃   (Option B — normalized score threshold)
  §14  accepted 후보는 skill-wise vector DB 에 window 별 entry 로 누적한다.

ΔH_A 는 action_coverage(§8), ΔH_A|S 는 conditional_ambiguity(§9) 에서 온다.
covered window 가 부족한 후보는 under-covered 로 표시해 accept 에서 제외한다
(§9.1 — MI score 를 신뢰성 있게 평가할 수 없는 후보). 후보는 모두 Phase1 seed
anchor 주변에서 생성되므로 (phase1_seed_anchor_logic) 정상 경로에선 드물다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from method3.phase2_mi_selection.action_coverage import action_coverage_gain
from method3.phase2_mi_selection.action_descriptor import dct_action_descriptor
from method3.phase2_mi_selection.conditional_ambiguity import conditional_ambiguity
from method3.phase2_mi_selection.radius import state_neighborhood_radius
from method3.phase2_mi_selection.vector_db import SkillVectorDB, VectorDBEntry


@dataclass(frozen=True)
class Phase2Candidate:
    """후보 skill trajectory segment ``ξ ∈ Ξ_t^{(m)}(g)`` (§7.3, phase1_seed_anchor_logic).

    Phase2 후보는 Phase1 seed ``g ∈ G_seed^{(m)}`` 를 anchor 로 그 주변에서
    생성된 action variation 이다 — Phase2 는 새 subgoal 을 탐색하지 않는다.
    state_keys 는 이미 계산된 retrieval key ``e_τ = [φ_VLA(o_τ,I); p_τ]`` 이다
    (VLA embedding 추출은 이 라이브러리 밖에서 수행 — 호출부가 채워 넘긴다).
    action descriptor ``z_τ^a`` 는 selector 가 action_chunks 에서 DCT 로 계산한다.
    """

    skill_id: str
    state_keys: np.ndarray         # (T, D_e) — window 별 e_τ
    action_chunks: np.ndarray      # (T, H, action_dim) — window 별 A_{τ:τ+H-1}
    seed_subgoal: np.ndarray | None = None  # anchor — 이 후보가 perturb 한 Phase1 seed g
    payload: object | None = None  # caller pass-through (raw pointer 등)


@dataclass
class Phase2MIConfig:
    """Phase2 MI scoring 파라미터 (문서 §8-12, §16)."""

    dct_coeffs: int = 3           # §4.2 K — DCT action descriptor 저주파 성분 수
    k_nn_a: int = 5               # §8 action-space kNN 의 k
    q_quantile: float = 0.5       # §8 ΔH_A top-quantile 비율
    k_min: int = 3                # §9.1 covered window 최소 neighbor 수
    radius_k: int = 5             # §16 ρ_m 계산용 kNN k
    radius_quantile: float = 0.7  # §16 ρ_m quantile
    s_min_a: float = 1e-3         # §9.3 local support scale floor s_min^a
    eps: float = 1e-6             # log/정규화 분모 안정화
    beta: float = 1.0             # §11 ΔH_A 가중
    lambda_: float = 1.0          # §11 ΔH_A|S 가중
    accept_threshold: float = 0.0  # §12 τ_Q̃ — normalized score accept 기준
    amb_agg: str = "mean"         # §9.4 covered aggregation: "mean" | "max"
    min_covered_windows: int = 1  # §9.1 T_min — 미만이면 under-covered
    debug_verbose: bool = False


@dataclass(frozen=True)
class Phase2ScoreReport:
    """후보 하나의 Phase2 점수 리포트."""

    candidate_index: int
    delta_h_a: float            # ΔH_A — action coverage gain (§8)
    delta_h_a_given_s: float    # ΔH_A|S — conditional ambiguity increase (§9)
    q2: float                   # §11 Q2 = β·ΔH_A − λ·ΔH_A|S
    q2_norm: float              # §12 Q̃2 — 배치 정규화 후 채워진다
    covered_ratio: float        # §15.1 R_cov(ξ)
    under_covered: bool         # covered window < T_min → MI 평가 신뢰 불가


@dataclass
class Phase2Selection:
    """``select`` 결과 — argmax Q2 + Option B accept 판정 (§11-12)."""

    chosen_index: int
    chosen_candidate: Phase2Candidate
    accepted: bool
    reports: list[Phase2ScoreReport] = field(default_factory=list)


class Phase2MISelector:
    """MI-style score 로 후보 trajectory 를 평가·선택한다 (문서 §11-12).

    Usage::

        sel = selector.select(candidates)          # argmax Q2 + accept 판정
        if sel.accepted:
            selector.accept_to_buffer(sel.chosen_candidate)
    """

    def __init__(
        self,
        vector_db: SkillVectorDB,
        config: Phase2MIConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.db = vector_db
        self.cfg = config or Phase2MIConfig()
        self._rng = rng if rng is not None else np.random.default_rng()

    def _dbg(self, msg: str) -> None:
        if self.cfg.debug_verbose:
            print(f"[Phase2-MI][debug] {msg}")

    def action_descriptors(self, candidate: Phase2Candidate) -> np.ndarray:
        """후보 window 별 DCT action descriptor ``z_τ^a`` (T, K·action_dim)."""
        return np.stack([
            dct_action_descriptor(a, self.cfg.dct_coeffs)
            for a in np.asarray(candidate.action_chunks, dtype=np.float64)
        ])

    def score_one(self, index: int, candidate: Phase2Candidate) -> Phase2ScoreReport:
        """후보 하나의 ΔH_A·ΔH_A|S·Q2 (정규화 전) 계산 (문서 §8-11)."""
        cfg = self.cfg
        cand_z = self.action_descriptors(candidate)
        db_keys = self.db.state_keys(candidate.skill_id)
        db_z = self.db.action_descriptors(candidate.skill_id)
        n_windows = cand_z.shape[0]

        # Cold-start — buffer 가 2개 미만이면 d̄_NN/radius 를 계산할 수 없다.
        # ΔH_A·ΔH_A|S 를 0 으로 두고 under-covered 로 표시한다 (Phase1 seed 가
        # 아직 안 들어온 비정상 경로의 안전망; 정상 Phase2 는 seed 후 시작).
        if db_z.ndim != 2 or db_z.shape[0] < 2:
            self._dbg(f"skill={candidate.skill_id} | COLD-START (buffer<2)")
            return Phase2ScoreReport(index, 0.0, 0.0, 0.0, 0.0, 0.0, True)

        # §8 — action coverage gain ΔH_A.
        delta_a = action_coverage_gain(
            cand_z, db_z, cfg.k_nn_a, cfg.q_quantile, cfg.eps)

        # §16 — skill buffer 에서 state-neighborhood radius ρ_m 자동 추정.
        radius = state_neighborhood_radius(
            db_keys, cfg.radius_k, cfg.radius_quantile)

        # §9 — conditional ambiguity increase ΔH_A|S (covered window 만).
        amb = conditional_ambiguity(
            candidate.state_keys, cand_z, db_keys, db_z,
            radius, cfg.k_min, cfg.s_min_a, cfg.eps, cfg.amb_agg)

        under = amb.n_covered < cfg.min_covered_windows
        q2 = cfg.beta * delta_a - cfg.lambda_ * amb.delta_h_a_given_s   # §11
        self._dbg(
            f"skill={candidate.skill_id} cand#{index} | "
            f"ΔH_A={delta_a:.4f} ΔH_A|S={amb.delta_h_a_given_s:.4f} "
            f"Q2={q2:.4f} covered={amb.n_covered}/{n_windows} "
            f"{'UNDER-COVERED' if under else ''}"
        )
        return Phase2ScoreReport(
            candidate_index=index,
            delta_h_a=delta_a,
            delta_h_a_given_s=amb.delta_h_a_given_s,
            q2=q2,
            q2_norm=0.0,                       # select() 에서 배치 정규화 후 채움
            covered_ratio=amb.covered_ratio,
            under_covered=under,
        )

    def select(self, candidates: list[Phase2Candidate]) -> Phase2Selection:
        """후보 batch 를 평가하고 argmax Q2 + Option B accept 판정 (문서 §11-12).

        Args:
            candidates: 현재 skill ``m`` 의 후보 trajectory set ``Ξ_t^{(m)}``.

        Returns:
            Phase2Selection. ``accepted`` 는 정규화 score Q̃2(ξ*) 가
            ``accept_threshold`` 를 넘고 후보가 under-covered 가 아닐 때 True.
        """
        if not candidates:
            raise ValueError("candidates is empty")
        cfg = self.cfg
        reports = [self.score_one(i, c) for i, c in enumerate(candidates)]

        # §12 — 배치 내부 정규화: Q̃2 = (Q2 − μ_Q) / (σ_Q + ε).
        q2 = np.array([r.q2 for r in reports], dtype=np.float64)
        mu, sigma = float(q2.mean()), float(q2.std())
        q2_norm = (q2 - mu) / (sigma + cfg.eps)
        reports = [
            Phase2ScoreReport(
                r.candidate_index, r.delta_h_a, r.delta_h_a_given_s, r.q2,
                float(q2_norm[i]), r.covered_ratio, r.under_covered)
            for i, r in enumerate(reports)
        ]

        # §11 — argmax Q2 (정규화는 단조 변환이므로 argmax 동일).
        chosen = int(np.argmax(q2))
        # §12 Option B — normalized score 가 threshold 초과 + covered 신뢰 가능.
        accepted = (
            bool(q2_norm[chosen] > cfg.accept_threshold)
            and not reports[chosen].under_covered
        )
        self._dbg(
            f"batch K={len(candidates)} | chose cand#{chosen} "
            f"Q2={q2[chosen]:.4f} Q̃2={q2_norm[chosen]:.4f} "
            f"accepted={accepted}"
        )
        return Phase2Selection(
            chosen_index=chosen,
            chosen_candidate=candidates[chosen],
            accepted=accepted,
            reports=reports,
        )

    def accept_to_buffer(
        self,
        candidate: Phase2Candidate,
        ref: dict | None = None,
        meta: dict | None = None,
    ) -> int:
        """accepted 후보를 skill-wise vector DB 에 window 별로 누적한다 (문서 §14).

        후보 trajectory 의 각 window 가 ``(e_τ, z_τ^a, ref, meta)`` entry 가 된다.

        Args:
            candidate: accepted 된 후보 trajectory.
            ref: raw dataset pointer (episode_id/start_t/end_t 등). 모든 window
                공통 — 호출부가 알면 넘긴다.
            meta: phase/subgoal/score 등 metadata.

        Returns:
            추가된 entry 수 (= window 수 T).
        """
        cand_z = self.action_descriptors(candidate)
        keys = np.asarray(candidate.state_keys, dtype=np.float64)
        base_meta = {"phase": "phase2", **(meta or {})}
        for tau in range(cand_z.shape[0]):
            self.db.append(VectorDBEntry(
                skill_id=str(candidate.skill_id),
                state_key=keys[tau],
                action_descriptor=cand_z[tau],
                ref=dict(ref or {}),
                meta={**base_meta, "window": tau},
            ))
        self._dbg(
            f"accepted skill={candidate.skill_id} → +{cand_z.shape[0]} entries, "
            f"buffer total={self.db.total_size()}"
        )
        return int(cand_z.shape[0])
