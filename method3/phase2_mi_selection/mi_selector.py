"""Phase2 MI-side usefulness + Useful-OOD selector — final_method3_spec_useful_ood_updated §11-13.

후보 trajectory 들을 받아 두 축으로 평가·선택한다::

  §11  M_MI(D_t, ξ) = β·ΔH_A(D_t, ξ) − λ·ΔH_A|S(D_t, ξ)            ← buffer-side
  §12  U_VLA(ξ) = Agg_τ [(1/R) Σ_r L_denoise^{(r)}(o_τ,I,p_τ,A; π_θ^{(1)})] ← model-side
  §13  ξ* = argmax U_VLA(ξ)   s.t.   M̃_MI(D_t, ξ) ≥ τ_MI            (Useful-OOD rule)

M_MI 는 candidate batch 안에서 정규화한다::

       M̃_MI = (M_MI − μ_M) / (σ_M + ε)         (§13.2)

기본값 ``τ_MI = 0`` 은 *batch 평균 이상* 의 MI-useful 후보만 통과시킨다는 의미.
이로써 Harmful OOD (high U_VLA + low M_MI) 와 Redundant ID (모두 낮음) 가 제거되고,
Useful OOD (high U_VLA + high M_MI) 가 우선 선택된다.

covered window 가 부족한 후보는 ``under_covered=True`` 로 마크되어 통과에서 제외된다
(§9.1 — MI score 를 신뢰성 있게 평가할 수 없는 후보). 후보는 Phase1 seed anchor 주변
에서만 생성되므로 (`phase1_seed_anchor_logic`) 정상 경로에선 드물다.

§14 — accepted 후보는 skill-wise vector DB 에 window 별 entry 로 누적한다.

ΔH_A 는 action_coverage(§8), ΔH_A|S 는 conditional_ambiguity(§9) 에서 온다.
U_VLA 는 ``vla_informativeness.VLAInformativenessScorer`` 가 책임지며 selector 는
값만 받는다 — VLA policy 의존성이 selector 핵심 경로에 들어가지 않게 분리.

Backward-compat: ``vla_scorer=None`` 으로 ``select`` 를 호출하면 stage 2 가
``argmax M_MI among eligible`` 로 fallback (이전 ``argmax Q2`` 동작과 동등).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from method3.phase2_mi_selection.action_coverage import action_coverage_gain
from method3.phase2_mi_selection.action_descriptor import dct_action_descriptor
from method3.phase2_mi_selection.conditional_ambiguity import (
    conditional_ambiguity,
    reset_amb_timing,
    get_amb_timing,
)
from method3.phase2_mi_selection.radius import state_neighborhood_radius
from method3.phase2_mi_selection.vector_db import SkillVectorDB, VectorDBEntry

if TYPE_CHECKING:
    from method3.phase2_mi_selection.vla_informativeness import (
        VLAInformativenessScorer,
    )


@dataclass(frozen=True)
class Phase2Candidate:
    """후보 skill trajectory segment ``ξ ∈ Ξ_t^{(m)}(g)`` (§7.3, phase1_seed_anchor_logic).

    Phase2 후보는 Phase1 seed ``g ∈ G_seed^{(m)}`` 를 anchor 로 그 주변에서
    생성된 action variation 이다 — Phase2 는 새 subgoal 을 탐색하지 않는다.
    state_keys 는 이미 계산된 retrieval key ``e_τ = [φ_VLA(o_τ,I); p_τ]`` 이다
    (VLA embedding 추출은 이 라이브러리 밖에서 수행 — 호출부가 채워 넘긴다).
    action descriptor ``z_τ^a`` 는 selector 가 action_chunks 에서 DCT 로 계산한다.

    §12 U_VLA 채점을 활성화하려면 raw observation/instruction/proprio 도 필요하다
    (옵셔널 — 없어도 selector 의 MI 경로는 동작). LeRobotVLAInformativenessScorer
    의 default batch_builder 는 ``observations`` 가 이미 LeRobot batch dict 라면
    그대로 사용한다 — caller 가 family-aware 로 미리 채워 넣을 수 있게 둔다.
    """

    skill_id: str
    state_keys: np.ndarray         # (T, D_e) — window 별 e_τ
    action_chunks: np.ndarray      # (T, H, action_dim) — window 별 A_{τ:τ+H-1}
    seed_subgoal: np.ndarray | None = None  # anchor — 이 후보가 perturb 한 Phase1 seed g
    payload: object | None = None  # caller pass-through (raw pointer 등)
    # §12 U_VLA 용 raw 입력 — vla_scorer 를 안 쓰면 무시. observations 는 LeRobot
    # policy.forward 가 받는 batch dict 그대로 (caller 가 family-aware 로 채움)
    # 또는 family-별 batch_builder 가 해석할 수 있는 임의 구조.
    observations: object | None = None
    instruction: str | None = None
    proprios: np.ndarray | None = None  # (T, P) — window 별 proprio
    # method3 DCT paradigm — skill 단위 후보 trajectory 의 DCT_50 feature.
    # (L0, action_dim). vla_dct_uncertainty.DCTDenoiseUncertainty 가 사용.
    dct_target: np.ndarray | None = None


@dataclass
class Phase2MIConfig:
    """Phase2 scoring · Useful-OOD selection 파라미터 (문서 §8-13, §16)."""

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
    tau_MI: float = 0.0           # §13.2 — Useful-OOD constraint: M̃_MI ≥ τ_MI
    accept_threshold: Optional[float] = None  # deprecated alias of tau_MI (yaml 호환)
    amb_agg: str = "mean"         # §9.4 covered aggregation: "mean" | "max"
    min_covered_windows: int = 1  # §9.1 T_min — 미만이면 under-covered
    debug_verbose: bool = False
    # method3 DCT paradigm — True 면 action_descriptors 가 candidate.dct_target
    # (skill-unit (L0, dof) DCT) 을 그대로 z-space 로 사용. 기본 False 는
    # 기존 frame-level chunk → truncated DCT descriptor 경로 (backward compat).
    use_dct_target: bool = False
    # Arm-only 비교 — DB 도 candidate 도 arm-only (gripper 축 제외) 로 통일됨
    # (DB build path 의 skill_segment_adapter 가 gripper 축을 사전 제외).
    # 따라서 *기본은 slicing 비활성* (arm_dof == full_dof).
    # legacy: DB 가 full_dof (arm+gripper) 로 빌드된 경우만 arm_dof < full_dof
    # 로 두면 score_one 이 비교 시점에 db_z / db_keys 를 slice 한다.
    #   - db_z: (N, L0*full_dof) → (N, L0, full_dof) → [:,:,:arm_dof] → (N, L0*arm_dof)
    #   - db_keys: (N, D_vl + full_dof) → [:, :-(full_dof-arm_dof)]
    arm_dof: Optional[int] = 5
    full_dof: Optional[int] = 5  # = arm_dof → slicing 비활성 (DB 가 이미 arm-only)
    dct_L0: int = 50  # DB 빌드 시 L0 — db_z reshape 용

    def __post_init__(self) -> None:
        # 기존 yaml 들이 accept_threshold 만 지정하던 호환 경로를 보존.
        if self.accept_threshold is not None:
            self.tau_MI = float(self.accept_threshold)


@dataclass(frozen=True)
class Phase2ScoreReport:
    """후보 하나의 Phase2 점수 리포트.

    ``q2`` / ``q2_norm`` 은 신규 spec 의 ``M_MI`` / ``M̃_MI`` 와 같은 값이다 —
    필드명은 호환을 위해 보존하고 의미만 §11 (MI-side usefulness) 로 재정의.
    ``u_vla`` 는 ``vla_scorer`` 가 주어졌을 때만 채워지는 model-side informativeness.
    """

    candidate_index: int
    delta_h_a: float            # ΔH_A — action coverage gain (§8)
    delta_h_a_given_s: float    # ΔH_A|S — conditional ambiguity increase (§9)
    q2: float                   # §11 M_MI = β·ΔH_A − λ·ΔH_A|S
    q2_norm: float              # §13.2 M̃_MI — batch 정규화
    covered_ratio: float        # §15.1 R_cov(ξ)
    under_covered: bool         # covered window < T_min → MI 평가 신뢰 불가
    u_vla: float = 0.0          # §12 U_VLA — vla_scorer 가 있을 때만 의미


@dataclass
class Phase2Selection:
    """``select`` 결과 — Useful-OOD rule (§13.2) 의 판정.

    ``accepted`` 는 ``M̃_MI ≥ τ_MI`` (+ covered) 를 만족하는 후보가 1개 이상 있고
    그중 max U_VLA 후보가 정상 선택됐을 때 True. eligible 이 비어 있으면
    ``chosen_index`` 는 argmax M_MI 로 fallback 되고 ``accepted=False``.
    """

    chosen_index: int
    chosen_candidate: Phase2Candidate
    accepted: bool
    reports: list[Phase2ScoreReport] = field(default_factory=list)
    u_vla_chosen: Optional[float] = None  # 선택된 후보의 U_VLA — vla_scorer 없으면 None
    eligible_indices: list[int] = field(default_factory=list)  # §13.2 stage 1 통과한 후보


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
        """후보의 action descriptor.

        paradigm step [6] — ``cfg.use_dct_target=True`` 면 candidate 의 skill
        단위 DCT feature ``dct_target (L0, dof)`` 를 flatten 한 (1, L0·dof)
        를 single-window z 로 사용 (skill-atomic representation).

        Backward compat — False (default) 면 frame-level action_chunk 의
        truncated DCT descriptor (T, K·action_dim) 반환.
        """
        if self.cfg.use_dct_target:
            z = candidate.dct_target
            if z is None:
                raise ValueError(
                    "use_dct_target=True 인데 candidate.dct_target 이 None — "
                    "curobo_candidate_gen 의 dct_target 필드 채움이 필요합니다."
                )
            arr = np.asarray(z, dtype=np.float64).reshape(1, -1)
            return arr
        return np.stack([
            dct_action_descriptor(a, self.cfg.dct_coeffs)
            for a in np.asarray(candidate.action_chunks, dtype=np.float64)
        ])

    def score_one(self, index: int, candidate: Phase2Candidate) -> Phase2ScoreReport:
        """후보 하나의 ΔH_A·ΔH_A|S·Q2 (정규화 전) 계산 (문서 §8-11)."""
        cfg = self.cfg
        import time as _tmod
        _acc = getattr(self, "_t_acc", None)
        _t = _tmod.perf_counter()
        cand_z = self.action_descriptors(candidate)
        db_keys = self.db.state_keys(candidate.skill_id)
        db_z = self.db.action_descriptors(candidate.skill_id)
        # Arm-only 비교 — DB 는 full_dof (arm+gripper) 로 저장, candidate 는
        # arm_dof 만. 비교 시점에 DB 의 gripper 축 제거.
        if (cfg.use_dct_target and cfg.arm_dof and cfg.full_dof
                and cfg.arm_dof < cfg.full_dof):
            L0 = cfg.dct_L0
            full = cfg.full_dof
            arm = cfg.arm_dof
            # db_z: (N, L0*full) → reshape (N, L0, full) → arm slice → (N, L0*arm)
            if db_z.ndim == 2 and db_z.shape[1] == L0 * full:
                db_z = db_z.reshape(-1, L0, full)[:, :, :arm].reshape(db_z.shape[0], -1)
            # db_keys: (N, D_vl + full) → drop last (full-arm) cols (proprio 끝 축).
            if db_keys.ndim == 2 and db_keys.shape[1] > (full - arm):
                db_keys = db_keys[:, :-(full - arm)]
        n_windows = cand_z.shape[0]
        if _acc is not None:
            _acc["db"] += _tmod.perf_counter() - _t
            _t = _tmod.perf_counter()

        # Cold-start — buffer 가 2개 미만이면 d̄_NN/radius 를 계산할 수 없다.
        # ΔH_A·ΔH_A|S 를 0 으로 두고 under-covered 로 표시한다 (Phase1 seed 가
        # 아직 안 들어온 비정상 경로의 안전망; 정상 Phase2 는 seed 후 시작).
        if db_z.ndim != 2 or db_z.shape[0] < 2:
            self._dbg(f"skill={candidate.skill_id} | COLD-START (buffer<2)")
            return Phase2ScoreReport(index, 0.0, 0.0, 0.0, 0.0, 0.0, True)

        # §8 — action coverage gain ΔH_A.
        delta_a = action_coverage_gain(
            cand_z, db_z, cfg.k_nn_a, cfg.q_quantile, cfg.eps)
        if _acc is not None:
            _acc["cov"] += _tmod.perf_counter() - _t
            _t = _tmod.perf_counter()

        # §16 — state-neighborhood radius ρ_m. db_keys 는 한 plan_and_select
        # 내 모든 후보가 동일하므로 skill 별 1회만 계산하고 cache. 128× 중복
        # 호출이 select 의 최대 병목이었다 (timing rad≈18.5s → 1회 ≈0.15s).
        _rc = getattr(self, "_radius_cache", None)
        if _rc is not None and _rc[0] == candidate.skill_id:
            radius = _rc[1]
        else:
            radius = state_neighborhood_radius(
                db_keys, cfg.radius_k, cfg.radius_quantile)
            self._radius_cache = (candidate.skill_id, radius)
        if _acc is not None:
            _acc["rad"] += _tmod.perf_counter() - _t
            _t = _tmod.perf_counter()

        # 진단 — index==0 일 때 한 번만 출력. why under_covered? 거리 분포 + radius.
        if index == 0:
            try:
                _ck = np.asarray(candidate.state_keys, dtype=np.float64)
                _q = _ck[0:1]
                _d2db = np.linalg.norm(db_keys - _q, axis=1)
                _ck_norm = np.linalg.norm(_ck, axis=1)
                _db_norm = np.linalg.norm(db_keys, axis=1)
                print(f"[diagnose] skill={candidate.skill_id} cand#0 "
                      f"state_keys={_ck.shape} (norm μ={_ck_norm.mean():.3f}) "
                      f"db_keys={db_keys.shape} (norm μ={_db_norm.mean():.3f}) "
                      f"dist_cand0_to_db[min,p50,p70,max]=({_d2db.min():.4f},"
                      f"{np.quantile(_d2db,0.5):.4f},"
                      f"{np.quantile(_d2db,0.7):.4f},{_d2db.max():.4f}) "
                      f"radius={radius:.4f} "
                      f"→ covered_iff: ∃ db within radius ({(_d2db<=radius).sum()}/{len(_d2db)})",
                      flush=True)
            except Exception as _e:
                print(f"[diagnose] failed: {_e}", flush=True)

        # §9 — conditional ambiguity increase ΔH_A|S (covered window 만).
        # db_z pairwise distance — §9.3 s_a(mean_nn) 용. db_z 는 한
        # plan_and_select 내 동일하므로 skill 별 1회만 계산하고 cache
        # (radius cache 와 동일 패턴 — covered window 128× 중복 제거).
        _pc = getattr(self, "_pdist_cache", None)
        if _pc is not None and _pc[0] == candidate.skill_id:
            db_z_pdist = _pc[1]
        else:
            db_z_pdist = np.linalg.norm(
                db_z[:, None, :] - db_z[None, :, :], axis=2)
            self._pdist_cache = (candidate.skill_id, db_z_pdist)
        amb = conditional_ambiguity(
            candidate.state_keys, cand_z, db_keys, db_z,
            radius, cfg.k_min, cfg.s_min_a, cfg.eps, cfg.amb_agg,
            db_z_pdist=db_z_pdist)
        if _acc is not None:
            _acc["amb"] += _tmod.perf_counter() - _t

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

    def select(
        self,
        candidates: list[Phase2Candidate],
        vla_scorer: "VLAInformativenessScorer | None" = None,
    ) -> Phase2Selection:
        """후보 batch 를 Useful-OOD rule (§13.2) 로 평가·선택한다.

        3-stage:
          1. M_MI(ξ) 계산 + batch 정규화 (M̃_MI).
          2. eligible := {i : M̃_MI[i] ≥ τ_MI ∧ not under_covered[i]}.
          3. vla_scorer 가 있으면  ξ* = argmax_{i∈eligible} U_VLA(ξ_i).
             없으면 (backward-compat)  ξ* = argmax_{i∈eligible} M_MI(ξ_i).

        eligible 이 비면 ``accepted=False``, ``chosen`` 은 argmax M_MI 로 fallback.

        Args:
            candidates: 현재 skill ``m`` 의 후보 trajectory set ``Ξ_t^{(m)}(g)``.
            vla_scorer: §12 ``U_VLA`` 계산기. None 이면 stage 3 는 M_MI 로 fallback.

        Returns:
            Phase2Selection.
        """
        if not candidates:
            raise ValueError("candidates is empty")
        cfg = self.cfg
        import time as _t_mod
        _t0 = _t_mod.perf_counter()
        self._t_acc = {"db": 0.0, "cov": 0.0, "rad": 0.0, "amb": 0.0}
        self._radius_cache = None  # plan_and_select 마다 invalidate
        self._pdist_cache = None   # db_z pairwise distance cache invalidate
        reset_amb_timing()  # amb 내부 timing (state/zdist/loop) reset
        reports = [self.score_one(i, c) for i, c in enumerate(candidates)]
        _t_mmi = _t_mod.perf_counter()

        # Stage 1 — §11 M_MI 계산 + §13.2 batch 정규화 M̃_MI.
        m_mi = np.array([r.q2 for r in reports], dtype=np.float64)
        mu, sigma = float(m_mi.mean()), float(m_mi.std())
        m_mi_norm = (m_mi - mu) / (sigma + cfg.eps)

        # U_VLA 채점 — vla_scorer 가 주어졌을 때만. eligible 후보에만 호출해
        # 비용 절약 (cost-aware: stage 1 통과한 것에만 R-stochastic eval 수행).
        u_vla = np.zeros(len(candidates), dtype=np.float64)

        # Stage 2 — eligible: M̃_MI ≥ τ_MI ∧ not under_covered.
        eligible = [
            i for i, r in enumerate(reports)
            if m_mi_norm[i] >= cfg.tau_MI and not r.under_covered
        ]

        if vla_scorer is not None and eligible:
            # GPU batched U_VLA — eligible 후보를 batch 단위로 묶어 한 번에
            # forward (candidate 1개씩 forward 하던 것 대비 큰 속도 이득).
            # score_batch 미지원 scorer 는 per-candidate score 로 fallback.
            elig_cands = [candidates[i] for i in eligible]
            _score_batch = getattr(vla_scorer, "score_batch", None)
            if callable(_score_batch):
                u_scores = _score_batch(elig_cands)
            else:
                u_scores = [vla_scorer.score(c) for c in elig_cands]
            for j, i in enumerate(eligible):
                u_vla[i] = float(u_scores[j])

        _t_uvla = _t_mod.perf_counter()
        _a = self._t_acc
        _at = get_amb_timing()
        print(
            f"[timing:select] M_MI({len(candidates)}cand)="
            f"{(_t_mmi - _t0) * 1000:.0f}ms "
            f"[db={_a['db'] * 1000:.0f} cov={_a['cov'] * 1000:.0f} "
            f"rad={_a['rad'] * 1000:.0f} amb={_a['amb'] * 1000:.0f}"
            f"(state={_at['state'] * 1000:.0f} zdist={_at['zdist'] * 1000:.0f} "
            f"loop={_at['loop'] * 1000:.0f})]  "
            f"U_VLA({len(eligible)}elig)={(_t_uvla - _t_mmi) * 1000:.0f}ms",
            flush=True,
        )

        # report 에 정규화 score · U_VLA 를 채워 dataclass 재생성.
        reports = [
            Phase2ScoreReport(
                candidate_index=r.candidate_index,
                delta_h_a=r.delta_h_a,
                delta_h_a_given_s=r.delta_h_a_given_s,
                q2=r.q2,
                q2_norm=float(m_mi_norm[i]),
                covered_ratio=r.covered_ratio,
                under_covered=r.under_covered,
                u_vla=float(u_vla[i]),
            )
            for i, r in enumerate(reports)
        ]

        # Stage 3 — eligible 안에서 max U_VLA (없으면 fallback to max M_MI).
        if eligible:
            if vla_scorer is not None:
                chosen = max(eligible, key=lambda i: u_vla[i])
                rule = "argmax U_VLA s.t. M̃_MI≥τ_MI"
            else:
                chosen = max(eligible, key=lambda i: m_mi[i])
                rule = "argmax M_MI s.t. M̃_MI≥τ_MI  (vla_scorer=None)"
            accepted = True
        else:
            # fallback — eligible 이 비면 argmax M_MI 반환하되 accept=False.
            chosen = int(np.argmax(m_mi))
            accepted = False
            rule = "fallback argmax M_MI (no eligible)"

        u_chosen: Optional[float] = float(u_vla[chosen]) if vla_scorer is not None else None
        self._dbg(
            f"batch K={len(candidates)} eligible={len(eligible)} | "
            f"chose cand#{chosen} M_MI={m_mi[chosen]:.4f} M̃_MI={m_mi_norm[chosen]:.4f} "
            f"U_VLA={u_chosen if u_chosen is not None else 'n/a'} "
            f"accepted={accepted} rule='{rule}'"
        )
        return Phase2Selection(
            chosen_index=chosen,
            chosen_candidate=candidates[chosen],
            accepted=accepted,
            reports=reports,
            u_vla_chosen=u_chosen,
            eligible_indices=list(eligible),
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
