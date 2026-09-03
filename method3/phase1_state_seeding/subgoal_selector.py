"""Phase1 buffer-aware subgoal selector — 문서 final_method3_spec §4-5.

§4.2 — 여러 subgoal 후보를 만들고 각 후보까지 canonical preview trajectory 를
생성한 뒤, 후보가 도달하는 terminal state region 이 skill-wise subgoal buffer
``B_{g,t}^{(m)}`` 기준으로 얼마나 덜 커버됐는지(novel) 평가하여 argmax 선택한다.

§5.4 scoring:
  ξ_j = InterpPlan(S_t, g'_j)              — canonical preview (canonical_preview)
  T_end = last 20% of preview              — subgoal 근처 구간 (§5.3)
  ĥ_τ = φ_goal(Ŝ_τ, g'_j, m)               — geometric descriptor (terminal_descriptor)
  s_g = max(d̄_NN^g, s_min)                 — minimum scale floor
  n_g(ĥ_τ) = [log(d_k^g / (s_g + ε))]_+    — point novelty
  G_S^goal(g'_j) = mean_{τ∈T_end} n_g(ĥ_τ) — subgoal score
  g* = argmax G_S^goal

§5.5 — 선택된 subgoal 의 canonical trajectory 가 성공 실행되면, executed
trajectory 의 T_end descriptor 평균 ``h*`` 을 ``SubgoalBufferEntry`` 로 buffer
에 추가한다 (TRUE episode 만; raw dataset ingest 와 동일 시점).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from method3.phase1_state_seeding.canonical_preview import interp_plan, last_segment
from method3.phase1_state_seeding.subgoal_candidates import sample_subgoal_candidates
from method3.phase1_state_seeding.subgoal_buffer import (
    SubgoalBuffer,
    SubgoalBufferEntry,
    knn_mean_distance,
    mean_nn_distance,
)
from method3.phase1_state_seeding.subgoal_validity import (
    ReachabilityConfig,
    make_subgoal_validity_fn,
)
from method3.phase1_state_seeding.terminal_descriptor import state_descriptor

# [POISSON-DISK 2026-08-26] 최소 분리거리 계수. r_min = 이 값 · R · (1/(n+1))^(1/3)
# (R = clip_factor·sigma, n = 해당 skill 버퍼 크기). 0.75 는 더미 실험에서
# 거리 분포를 균등 수준(0.805 vs 0.794)으로 유지하면서 최소 분리 0.162R 을
# 확보한 값이다. 0 으로 두면 모든 후보가 통과해 순수 랜덤(=A1)이 된다.
POISSON_R_MIN_FACTOR = 0.75

# module-level optional import — _on_skill_stamp 의 hot path 라 매 호출 import
# 회피. record_dataset 가 install 안 된 환경 (테스트) 에서는 None.
try:
    from record_dataset.context import RecordingContext as _RecordingContext
except Exception:
    _RecordingContext = None


def _record_ctx_kinematics():
    """RecordingContext 의 글로벌 kinematics — _on_skill_stamp lazy lookup."""
    if _RecordingContext is None:
        return None
    return _RecordingContext._kinematics


@dataclass
class Phase1SubgoalConfig:
    """Phase1 buffer-aware subgoal scoring 파라미터 (문서 §5)."""

    sigma: float = 0.05          # candidate 분포 폭 파라미터 (metres)
    clip_factor: float = 2.0     # 후보 최대 반경 = clip_factor * sigma
    n_candidates: int = 32       # K — subgoal 후보 개수
    k_nn: int = 5                # buffer kNN novelty 의 k
    end_fraction: float = 0.2    # §5.3 T_end — 평가할 preview 마지막 구간 비율
    preview_points: int = 20     # §5.4 InterpPlan preview state 개수 T
    eps: float = 1e-6            # log scale 분모 안정화
    s_min: float = 0.01          # §5.4 minimum scale floor s_min^g (metres)
    min_buffer_size: int = 8     # 미만이면 cold-start (랜덤 후보 폴백);
                                 # 실제 임계값은 max(min_buffer_size, k_nn, 2)
    candidate_dist: str = "uniform_ball"  # "uniform_ball" | "gaussian" | "hemisphere"
    # §4.2 hemisphere dist 의 방향 기준점 (로봇 좌표 원점, 보통 base frame 의
    # (0,0,0)). 후보가 nominal goal → robot_origin 쪽 반구로만 생성된다.
    robot_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_resample: int = 200      # valid 후보 rejection 재샘플 상한 (문서 §4.2)
    # subgoal 위치 제약 (reachable/safe, 문서 §4.2) — 후보 생성·선택 공통.
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)
    # skill_id 별 부분 override. 예: ``{"move_initial": {"sigma": 0.02}}`` 면
    # ``select_subgoal(skill_id="move_initial", ...)`` 호출에 한해 sigma 만 교체
    # (나머지는 전역 값 유지). 지원 키: ``sigma`` / ``clip_factor`` /
    # ``n_candidates``. 비어 있거나 해당 skill_id 가 없으면 전역 값 그대로 사용.
    # home 자세처럼 카메라 FOV 가림을 피해야 하는 skill 에 좁은 분포를 주는 용도.
    per_skill_overrides: dict = field(default_factory=dict)
    debug_verbose: bool = False  # True → 버퍼/판단/선택 디버그 로그를 터미널에 출력


@dataclass(frozen=True)
class SubgoalScoreReport:
    """후보 하나에 대한 점수 리포트 (디버깅/로깅용)."""

    candidate_index: int
    goal: np.ndarray
    gain: float          # G_S^goal — subgoal-side state novelty (§5.4)


@dataclass(frozen=True)
class SubgoalSelection:
    """``select_subgoal`` 결과."""

    chosen_goal: np.ndarray
    chosen_index: int
    cold_start: bool                  # True 면 buffer 부족으로 랜덤 폴백
    reports: list[SubgoalScoreReport]


@dataclass(frozen=True)
class _PendingMove:
    """현재 episode 에서 실행된 transit move 하나 (flush 까지 staging).

    skill_id(ordinal)는 flush 시점에 _pending 리스트 인덱스로 부여한다.
    natural_language / skill_type 은 metadata.
    """

    start_ee: np.ndarray
    goal: np.ndarray
    episode_id: str
    start_t: int
    end_t: int
    natural_language: str = ""
    skill_type: str = ""


class Phase1SubgoalSelector:
    """argmax G_S^goal — buffer-aware subgoal 선택기 (문서 §5).

    Usage (move_to_position 안에서)::

        sel = selector.select_subgoal(current_ee, nominal_goal, skill_id, rng)
        target_position = sel.chosen_goal
    """

    def __init__(
        self,
        buffer: SubgoalBuffer,
        config: Phase1SubgoalConfig | None = None,
        reachable_fn: Callable[[np.ndarray], bool] | None = None,
        kinematics=None,
    ) -> None:
        """
        Args:
            buffer: skill-wise subgoal buffer ``B_{g,t}^{(m)}`` (§5).
            config: scoring 파라미터.
            reachable_fn: ``callable(xyz) -> bool`` — reachable/safe 후보 필터
                (§4.2). ``select_subgoal`` 의 인자로 호출별 override 가능.
            kinematics: forward-kinematics 엔진. ``_on_skill_stamp`` callback 이
                start_state(joint) → start_ee 변환에 사용. None 이면 callback
                내부 staging skip. 주입 안 하면 RecordingContext._kinematics 로
                lazy lookup.
        """
        self.buffer = buffer
        self.cfg = config or Phase1SubgoalConfig()
        self._reachable_fn = reachable_fn
        self._kinematics = kinematics
        # 현재 에피소드에서 실행된 skill move 의 staging. episode TRUE 판정 시
        # 에만 buffer 에 commit 된다 (flush_episode). paradigm 일관화 — 모든
        # set_skill_info 호출 단위로 staging (transit + interaction). VDB
        # 의 NL run-length segment 와 동일 namespace.
        self._pending: list[_PendingMove] = []

    # ANSI color — module-local literal to avoid NameError 위험. GREEN 으로
    # 버퍼 카운트 변화 로그를 시각화 (사용자 요청).
    _BUF_GREEN = "\033[92m"
    _BUF_END = "\033[0m"

    def _dbg(self, msg: str, *, buffer_event: bool = False) -> None:
        """debug_verbose 가 켜져 있으면 디버그 로그를 터미널에 출력.

        ``buffer_event=True`` 면 본문 전체를 GREEN 으로 감싼다 — staged/flush/
        discard 같은 buffer 카운트 변화 로그를 한눈에 식별. 일반 debug 로그는
        기본색 유지.
        """
        if not self.cfg.debug_verbose:
            return
        if buffer_event:
            print(f"{self._BUF_GREEN}[Subgoal-Phase1][debug] {msg}{self._BUF_END}")
        else:
            print(f"[Subgoal-Phase1][debug] {msg}")

    def _terminal_region(
        self, start_ee: np.ndarray, goal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """문서 §5.3/§5.5 — canonical preview 의 T_end descriptor set + 평균.

        Returns:
            ``(end_keys, h)`` — end_keys 는 ``{h^τ}`` (T_end, D), h 는 평균 ``h``.
        """
        preview = interp_plan(start_ee, goal, self.cfg.preview_points)    # §5.4
        end_states = last_segment(preview, self.cfg.end_fraction)         # §5.3
        end_keys = np.stack([state_descriptor(s, goal) for s in end_states])
        return end_keys, end_keys.mean(axis=0)

    def select_subgoal(
        self,
        current_ee: np.ndarray,
        nominal_goal: np.ndarray,
        rng: np.random.Generator,
        reachable_fn: Callable[[np.ndarray], bool] | None = None,
        feasibility_fn: Callable[[np.ndarray], bool] | None = None,
        skill_type: str = "",
    ) -> SubgoalSelection:
        """문서 §4.2·§5.4 — 가장 novel 한 subgoal 을 골라 반환한다.

        Args:
            current_ee: 현재 end-effector 위치 (3,) xyz — preview 시작점.
            nominal_goal: 현재 nominal subgoal ``g`` (3,) xyz.
            rng: 재현 가능한 numpy Generator.
            reachable_fn: 이 호출에 한해 생성자 ``reachable_fn`` 을 덮어쓰는
                reachable/safe 후보 필터.
            feasibility_fn: ``callable(xyz) -> bool`` — 2차 필터. geometric
                reachable 을 통과한 후보에 한해 호출되며(생성/resample 경로엔
                개입하지 않음), False 인 후보는 scoring·cold-start 양쪽에서
                제외된다. holding-phase IK feasibility 같은 비싼 검사를 valid
                후보 집합에만 적용하기 위한 것. None 이면 기존 동작과 동일.

        Returns:
            SubgoalSelection. cold-start(buffer 부족) 또는 valid 후보 0개면
            랜덤/ nominal 폴백.
        """
        cfg = self.cfg
        # skill_id = ordinal key — 이 transit move 가 flush 시 받을 _pending
        # 인덱스 = 현재 len(_pending). B_{g}^(m) 를 이 안정적 순번으로 조회.
        skill_id = f"skill_{len(self._pending)}"
        current_ee = np.asarray(current_ee, dtype=np.float64).reshape(3)
        nominal_goal = np.asarray(nominal_goal, dtype=np.float64).reshape(3)
        reach_fn = reachable_fn if reachable_fn is not None else self._reachable_fn

        # §4.2 — reachable/safe 합성 술어: 로봇 reach + z-floor/ceiling + workspace.
        valid_fn = make_subgoal_validity_fn(cfg.reachability, is_reachable=reach_fn)

        # per-skill override (있으면 이 호출에 한해 sigma/clip_factor/n_candidates
        # 만 교체; 없으면 전역 값 그대로). FOV 가림 회피용 좁은 분포 같은 skill-
        # 별 튜닝을 코드 분기 없이 config 만으로 가능하게 한다.
        # per_skill_overrides 는 skill.type 단위 튜닝 (move_initial/move/...) —
        # buffer 분할 키(skill_id=nl)가 아니라 skill_type 으로 lookup 한다.
        # skill_type 미전달 시 skill_id 로 폴백 (구 호출부 호환).
        _ov = (cfg.per_skill_overrides or {}).get(str(skill_type or skill_id), {})
        eff_sigma = float(_ov.get("sigma", cfg.sigma))
        eff_clip = float(_ov.get("clip_factor", cfg.clip_factor))
        eff_n = int(_ov.get("n_candidates", cfg.n_candidates))

        # §4.2 — 후보 생성. candidate 0 = nominal. 생성 단계에서 invalid 후보는
        # rejection resampling 으로 valid 후보로 채운다.
        candidates = sample_subgoal_candidates(
            nominal_goal,
            eff_n,
            eff_sigma,
            eff_clip,
            rng,
            include_nominal=True,
            dist=cfg.candidate_dist,
            valid_fn=valid_fn,
            max_resample=cfg.max_resample,
            robot_origin=cfg.robot_origin,
        )

        # §4.2 — valid 후보만 남긴다 (생성 rejection 의 안전망).
        valid_idx = np.array(
            [j for j in range(len(candidates)) if bool(valid_fn(candidates[j]))],
            dtype=int,
        )
        if valid_idx.size == 0:
            self._dbg(f"skill={skill_id} | no reachable candidate → nominal")
            return SubgoalSelection(nominal_goal, 0, cold_start=False, reports=[])

        # §4.2b — feasibility 필터 (예: holding-phase IK) 는 후보당 비싸므로
        # valid 후보 전체를 미리 검사하지 않고 **lazy** 하게 적용한다. 후보를
        # 우선순위(cold-start=랜덤 / scored=gain 내림차순)로 훑으며 처음 통과하는
        # 후보에서 멈춘다. gain 내림차순 검사의 결과는 "feasible 후보 중 argmax
        # G_S^goal" 과 동일하지만, 검사 횟수는 보통 1~수 회로 줄어든다.
        # feasibility_fn=None 이면 feas 가 항상 True → 검사 비용 0, 동작 불변.
        feas = feasibility_fn if feasibility_fn is not None else (lambda _xyz: True)

        # §5.6 — buffer 가 너무 작아 novelty 를 신뢰할 수 없으면 cold-start:
        # 랜덤 후보 폴백. 임계값 = max(min_buffer_size, k_nn, 2): kNN 은 k개
        # 이웃이 있어야 의미가 있고, mean_nn_distance 는 N>=2 필요.
        keys = self.buffer.query_skill(skill_id)
        if keys.shape[0] < max(cfg.min_buffer_size, cfg.k_nn, 2):
            pool = valid_idx[valid_idx != 0]   # nominal 제외 → perturbation 효과
            if pool.size == 0:
                pool = valid_idx
            # 1차: 기존과 동일한 단일 랜덤 draw. feasible 이면 그대로 채택
            # (feasibility_fn=None 이면 항상 여기서 끝나 기존 동작과 동일).
            first = int(rng.choice(pool))
            chosen = -1
            n_checked = 1
            if bool(feas(candidates[first])):
                chosen = first
            else:
                # 1차가 infeasible → 나머지 후보를 랜덤 순서로 lazy 검사.
                for x in rng.permutation(pool[pool != first]):
                    n_checked += 1
                    if bool(feas(candidates[int(x)])):
                        chosen = int(x)
                        break
            if chosen >= 0:
                # COLD-START 도 trace 에 저장 (사용자 요청: 모든 selection 기록)
                trace_path = getattr(self, "_trace_file", None)
                if trace_path is not None:
                    try:
                        import json as _json, time as _time
                        from pathlib import Path as _Path
                        _Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
                        rec = {
                            "ts": _time.time(),
                            "skill": str(skill_id),
                            "buffer_total_for_skill": int(keys.shape[0]),
                            "n_candidates": int(len(candidates)),
                            "n_valid": int(valid_idx.size),
                            "candidates": [
                                {"idx": int(j), "xyz": [float(v) for v in candidates[j]]}
                                for j in valid_idx.tolist()
                            ],
                            "selected": {
                                "idx": int(chosen),
                                "xyz": [float(v) for v in candidates[chosen]],
                                "cold_start": True,
                                "feas_checks": int(n_checked),
                            },
                            "episode_id": getattr(self, "_current_episode_id", ""),
                            "skill_order": getattr(self, "_current_skill_order", -1),
                        }
                        with open(trace_path, "a", encoding="utf-8") as f:
                            f.write(_json.dumps(rec) + "\n")
                    except Exception as e:
                        print(f"[Subgoal-Phase1][trace] cold-start save failed: {e}")
                CYAN = "\033[1;96m"; RESET = "\033[0m"
                self._dbg(
                    f"skill={skill_id} | COLD-START buffer N={keys.shape[0]} "
                    f"< min={cfg.min_buffer_size} → "
                    f"{CYAN}random cand#{chosen}{RESET} (feas checks={n_checked})"
                )
                return SubgoalSelection(
                    candidates[chosen], chosen, cold_start=True, reports=[],
                )
            self._dbg(
                f"skill={skill_id} | COLD-START — no feasible candidate "
                f"(all {pool.size} rejected by feasibility) → nominal"
            )
            return SubgoalSelection(nominal_goal, 0, cold_start=False, reports=[])

        # §5.4 — buffer 내부 기준 거리 + minimum scale floor: s_g = max(d̄_NN, s_min).
        scale = max(mean_nn_distance(keys), cfg.s_min)

        # §5.4 — valid 후보 전체에 G_S^goal (gain) 을 계산한다. gain 은 기하
        # 연산이라 싸므로 전부 구해 ranking 에 쓰고, 비싼 feasibility 검사만
        # 아래에서 gain 내림차순으로 lazy 하게 한다.
        reports: list[SubgoalScoreReport] = []
        for j in valid_idx:
            goal = candidates[j]
            end_keys, _ = self._terminal_region(current_ee, goal)         # §5.3 T_end
            novelties = []
            for h_tau in end_keys:                                        # §5.3 ĥ_τ
                d_knn = knn_mean_distance(h_tau, keys, cfg.k_nn)           # §5.4 d_k^g
                # §5.4 n_g = [log(d_k^g / (s_g + ε))]_+. d_knn==0 (query 가
                # buffer 와 정확히 일치) 이면 log(0)=-inf → [·]_+=0; math.log(0)
                # 은 ValueError 이므로 직접 0 으로 단락한다.
                if d_knn <= 0.0:
                    novelties.append(0.0)
                else:
                    novelties.append(max(math.log(d_knn / (scale + cfg.eps)), 0.0))
            gain = float(np.mean(novelties))                              # §5.4 G_S^goal
            reports.append(SubgoalScoreReport(int(j), goal, gain))

        # [POISSON-DISK 2026-08-26] argmax G_S^goal 를 최소분리 제약 + 랜덤
        # 채택으로 바꾼다.
        #
        # 왜: 후보는 반경 R = clip_factor·sigma 구 안에 갇혀 있는데, argmax 는
        # "버퍼에서 가장 먼 후보"라 구조적으로 구 껍질만 고른다 (더미 실험:
        # 거리/R 중앙 0.938, 경계>0.95R 45%; 실측 sort_by_color A2 는 0.956/54%).
        # sort_by_color 는 에피소드당 섭동 스킬이 13 종이라 R 급 섭동이 연쇄되어
        # 수집 성공률이 23% 까지 떨어졌다 (push_button 은 2 종이라 97%).
        # 게다가 argmax 는 "가장 먼 하나"만 볼 뿐 실제 분리를 보장하지 않아
        # 최근접 분리가 0.048R 까지 내려간다 — 겹치는 subgoal 이 실제로 생긴다.
        #
        # 대신 "최근접 버퍼 점까지 거리 >= r_min" 인 후보들 중 랜덤으로 뽑는다
        # (poisson-disk / blue-noise). 더미 실험: 거리/R 중앙 0.805(균등 0.794 와
        # 동등), 경계>0.95R 16%, 최소 분리 0.162R — 겹침 방지는 오히려 3 배 낫다.
        # r_min 은 버퍼가 찰수록 자동 축소해 후보 고갈을 막고, 전부 기각되면
        # gain 최대(= 기존 argmax)로 폴백한다.
        # [SCALE FIX 2026-08-26] r_min 은 descriptor 공간에서 재야 한다.
        # d_sep 는 terminal descriptor 거리인데 초기 구현은 offset 공간의
        # R = clip_factor·sigma 를 썼다. 두 공간의 스케일이 달라 실측에서는
        # 후보 1 개만 통과했고(선택==argmax 63/67) poisson 경로가 사실상 죽었다.
        # 버퍼 자신의 해상도 s_g (= max(d̄_NN, s_min), §5.4 에서 이미 계산) 를
        # 기준으로 삼으면 공간이 일치하고 skill 마다 자동으로 맞춰진다.
        _n_buf = int(keys.shape[0])
        _r_min = POISSON_R_MIN_FACTOR * float(scale)
        _sep = {}
        for _ri, _rep in enumerate(reports):
            _h, _ = self._terminal_region(current_ee, _rep.goal)
            _sep[_ri] = float(min(
                float(np.min(np.linalg.norm(keys - _h_tau, axis=1)))
                for _h_tau in _h))
        _ok = [i for i, d in _sep.items() if d >= _r_min]
        if _ok:
            ranked = [int(i) for i in rng.permutation(np.asarray(_ok))]
            ranked += sorted((i for i in range(len(reports)) if i not in set(_ok)),
                             key=lambda i: reports[i].gain, reverse=True)
        else:
            ranked = sorted(range(len(reports)), key=lambda i: reports[i].gain,
                            reverse=True)
        for n_checked, ri in enumerate(ranked, start=1):
            best_idx = reports[ri].candidate_index
            if bool(feas(candidates[best_idx])):
                self._log_selection(skill_id, keys.shape[0], scale, candidates,
                                    valid_idx, reports, best_idx,
                                    reports[ri].gain, n_checked, n_checked - 1)
                return SubgoalSelection(
                    candidates[best_idx], best_idx, cold_start=False,
                    reports=reports,
                )

        self._dbg(
            f"skill={skill_id} | no feasible candidate "
            f"(all {valid_idx.size} valid rejected by feasibility) → nominal"
        )
        return SubgoalSelection(nominal_goal, 0, cold_start=False, reports=reports)

    def _log_selection(self, skill_id, n_buffer, scale, candidates, valid_idx,
                       reports, best_idx, best_gain, n_feas_checks,
                       best_rank) -> None:
        """판단 기준(s_g, gain 분포)과 선택 결과를 *cyan* 강조 로그로 출력
        + (옵션) jsonl trace file 에 영구 저장. 사용자 요청 — 분석용.

        ``n_feas_checks`` 는 lazy feasibility 검사 횟수, ``best_rank`` 는 선택된
        후보의 gain 내림차순 등수(0 = 최고 gain 후보가 그대로 feasible).
        """
        # ── 1. trace file save (선택된 수치 + 모든 candidate scores) ──
        # session 폴더 에 .jsonl 로 append. 사용자 요청: episode/skill 순서별
        # subgoal 후보 수치 + 선정된 수치 보존.
        trace_path = getattr(self, "_trace_file", None)
        if trace_path is not None:
            try:
                import json as _json, time as _time
                from pathlib import Path as _Path
                _Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "ts": _time.time(),
                    "skill": str(skill_id),
                    "buffer_total_for_skill": int(n_buffer),
                    "scale_s_g": float(scale),
                    "n_candidates": int(len(candidates)),
                    "n_valid": int(valid_idx.size),
                    "candidates": [
                        {"idx": int(r.candidate_index),
                         "gain": float(r.gain),
                         "xyz": [float(v) for v in r.goal]}
                        for r in reports
                    ],
                    "selected": {
                        "idx": int(best_idx),
                        "gain": float(best_gain),
                        "xyz": [float(v) for v in candidates[best_idx]],
                        "gain_rank": int(best_rank),
                        "feas_checks": int(n_feas_checks),
                        "cold_start": False,
                    },
                    "episode_id": getattr(self, "_current_episode_id", ""),
                    "skill_order": getattr(self, "_current_skill_order", -1),
                }
                with open(trace_path, "a", encoding="utf-8") as f:
                    f.write(_json.dumps(rec) + "\n")
            except Exception as e:
                print(f"[Subgoal-Phase1][trace] save failed: {e}")
        # ── 2. cyan 강조 log (terminal) ──
        if not self.cfg.debug_verbose:
            return
        CYAN = "\033[1;96m"; RESET = "\033[0m"; DIM = "\033[2;37m"
        gains = [r.gain for r in reports]
        order = sorted(range(len(reports)), key=lambda i: reports[i].gain,
                       reverse=True)
        top = "  ".join(
            f"cand#{reports[i].candidate_index}({reports[i].gain:.3f})"
            for i in order[:4]
        )
        g = candidates[best_idx]
        self._dbg(f"skill={skill_id} | buffer N={n_buffer}  s_g={scale:.4f}")
        self._dbg(
            f"  K={len(candidates)} valid={valid_idx.size} | "
            f"gain min={min(gains):.3f} mean={float(np.mean(gains)):.3f} "
            f"max={max(gains):.3f}"
        )
        self._dbg(
            f"  {CYAN}CHOSEN cand#{best_idx} gain={best_gain:.4f}{RESET} "
            f"(gain-rank {best_rank}, feas checks={n_feas_checks}) "
            f"{CYAN}goal=[{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}]{RESET}"
        )
        self._dbg(f"  {DIM}top: {top}{RESET}")

    def set_trace_file(self, path) -> None:
        """jsonl trace 출력 경로 설정 — None 이면 trace 비활성."""
        self._trace_file = str(path) if path else None

    def set_current_context(self, episode_id: str = "", skill_order: int = -1) -> None:
        """trace 의 *episode + skill_order* 필드 채우기 위해 호출자가 알려줌."""
        self._current_episode_id = str(episode_id)
        self._current_skill_order = int(skill_order)

    def _on_skill_stamp(
        self,
        *,
        skill_call_index: int,
        label: str,
        skill_type: str,
        goal_joint,
        goal_robot_xyzrpy,
        goal_gripper,
        start_state,
    ) -> None:
        """``RecordingContext.set_skill_info`` 가 발동시키는 callback.

        method3 paradigm 일관화 — 모든 set_skill_info 호출 단위로 staging.
        transit 과 interaction 모두에서 발동되어 subgoal_buffer 의 ordinal
        이 VDB (build_skill_dct 의 episode-내 skill_index) 와 정합한다.

        perturbation 적용은 별개 — ``select_subgoal`` 이 transit 단계에서만
        호출되어 chosen_goal 을 override. 여기서는 staging entry 만 등록.
        """
        if goal_robot_xyzrpy is None or start_state is None:
            return
        kin = self._kinematics
        if kin is None:
            # lazy lookup — selector init 시점에 kinematics 가 None 이면 record
            # context 의 글로벌 인스턴스를 사용.
            kin = _record_ctx_kinematics()
        if kin is None:
            return  # FK 불가 — staging skip (callback 등록 시점 mis-config)
        try:
            _ss = np.asarray(start_state, dtype=np.float64).reshape(-1)
            start_ee = np.asarray(
                kin.get_ee_position(_ss[:5]), dtype=np.float64
            ).reshape(3)
        except Exception:
            return
        try:
            goal = np.asarray(goal_robot_xyzrpy, dtype=np.float64).reshape(-1)[:3]
        except Exception:
            return
        self.stage_executed(
            start_ee=start_ee,
            goal=goal,
            episode_id=getattr(self, "_current_episode_id", "") or "",
            start_t=-1,
            end_t=-1,
            natural_language=label or "",
            skill_type=skill_type or "",
        )

    def stage_executed(
        self,
        start_ee: np.ndarray,
        goal: np.ndarray,
        episode_id: str = "",
        start_t: int = -1,
        end_t: int = -1,
        natural_language: str = "",
        skill_type: str = "",
    ) -> None:
        """실행된 transit move 를 episode pending 에 staging 한다 (문서 §5.5).

        §5.5 의 buffer update 는 episode 가 TRUE 로 판정될 때만 일어난다
        (raw dataset ingest 와 동일한 시점). 따라서 move 시점엔 buffer 가 아니라
        episode pending 에 move 정보를 모아두고, ``flush_episode`` (TRUE) /
        ``discard_episode`` (FALSE·UNCERTAIN) 가 확정/폐기한다.

        Args:
            start_ee: 그 move 시작 시점의 end-effector 위치 (3,).
            goal: 그 move 가 향했던 (선택된) subgoal ``g*`` (3,).
            episode_id: §5.2 raw dataset pointer — episode 식별자. 파이프라인이
                알면 넘기고, 모르면 기본값("") 으로 둔다.
            start_t: §5.2 raw dataset pointer — 구간 시작 time index.
            end_t: §5.2 raw dataset pointer — 구간 끝 time index.
        """
        self._pending.append(_PendingMove(
            start_ee=np.asarray(start_ee, dtype=np.float64).reshape(3),
            goal=np.asarray(goal, dtype=np.float64).reshape(3),
            episode_id=str(episode_id),
            start_t=int(start_t),
            end_t=int(end_t),
            natural_language=str(natural_language),
            skill_type=str(skill_type),
        ))
        # staging 요약 — 항상 초록색으로 간단히 출력.
        _mv = self._pending[-1]
        _g = _mv.goal
        print(
            f"{self._BUF_GREEN}[Subgoal] staged skill_{len(self._pending) - 1}  "
            f"{_mv.skill_type or '?'}  \"{(_mv.natural_language or '')[:34]}\"  "
            f"g=[{_g[0]:.3f}, {_g[1]:.3f}, {_g[2]:.3f}]  "
            f"(episode pending={len(self._pending)}){self._BUF_END}"
        )

    def flush_episode(self, episode_id: str = "") -> None:
        """episode TRUE 판정 → staged subgoal 들을 buffer 에 commit + 영속화 (문서 §5.5).

        각 staged move 에 대해 canonical trajectory 의 T_end descriptor 평균
        ``h* = mean_{τ∈T_end} φ_goal(S_τ, g*, m)`` 을 계산해 §5.2 형식의
        ``SubgoalBufferEntry`` 로 ``B_{g}^{(m)}`` 에 추가한다. raw dataset ingest
        와 **동일한 시점**(judge=TRUE)에 호출하므로, 다중 에피소드 run 이 중간에
        중단돼도 마지막 TRUE 에피소드까지 보존된다.

        Args:
            episode_id: 이 episode 의 canonical 식별자. 주어지면 commit 되는 모든
                entry 의 ``episode_id`` 를 이 값으로 stamp 한다 (episode lifecycle —
                나중에 episode 단위로 buffer 를 정리/재취득할 수 있도록). 비워두면
                staging 시점의 ``_PendingMove.episode_id`` 를 그대로 쓴다.
        """
        if not self._pending:
            return
        for i, mv in enumerate(self._pending):
            end_keys, h_star = self._terminal_region(mv.start_ee, mv.goal)   # §5.5
            self.buffer.append(SubgoalBufferEntry(
                skill_id=f"skill_{i}",
                subgoal=mv.goal,
                terminal_region_key=h_star,
                end_state_keys=end_keys,
                episode_id=(str(episode_id) if episode_id else mv.episode_id),
                start_t=mv.start_t,
                end_t=mv.end_t,
                success_flag=True,
                planner_type="InterpPlan",
                phase="phase1",
                natural_language=mv.natural_language,
                skill_type=mv.skill_type,
            ))
        n = len(self._pending)
        self._pending.clear()
        self.buffer.save()   # 파일 미바인딩 시 no-op
        # episode commit 요약 — 항상 초록색으로 간단히 출력.
        # file 미바인딩이면 save() 가 no-op 이므로 "saved" 표기 안 함.
        _fp = self.buffer.file_path()
        _persisted = f"saved → {_fp}" if _fp is not None else "MEMORY-ONLY"
        print(
            f"{self._BUF_GREEN}[Subgoal] episode TRUE → +{n} subgoals committed "
            f"(skill_0..skill_{n - 1}), buffer total={self.buffer.total_size()}, "
            f"{_persisted}{self._BUF_END}"
        )

    def discard_episode(self) -> None:
        """episode FALSE/UNCERTAIN 판정 → staged subgoal 폐기 (문서 §5.5).

        §5.5 — 실패한 trajectory 는 subgoal coverage seed 로 쓰지 않는다.
        """
        if self._pending and self.cfg.debug_verbose:
            self._dbg(
                f"discard episode → {len(self._pending)} staged subgoals dropped",
                buffer_event=True,
            )
        self._pending.clear()
