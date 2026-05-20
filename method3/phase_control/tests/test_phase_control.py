"""Method3 phase control — unit tests (문서 final_method3_spec §15).

§15.1 readiness / §15.2 saturation / §15.3 transition·budget 를 격리 검증한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from method3.phase_control.phase1_readiness import (
    Phase1ReadinessConfig,
    SkillProbe,
    candidate_covered_ratio,
    evaluate_phase1_readiness,
)
from method3.phase_control.phase2_saturation import Phase2SaturationTracker
from method3.phase_control.phase_controller import (
    PhaseController,
    PhaseControllerConfig,
)


def _line_db(n: int) -> np.ndarray:
    """1축 정수 간격 buffer (0..n-1) — radius 가 예측 가능하게 잡힌다."""
    return np.arange(n, dtype=float).reshape(n, 1)


# ─────────────────────────────────────────────────────────────
# §15.1  Phase1 readiness
# ─────────────────────────────────────────────────────────────
class TestPhase1Readiness:
    def test_covered_ratio_all_covered(self):
        # db 원점 12개, 후보 window 전부 원점 → radius 안에 12개 → 전부 covered.
        db = np.zeros((12, 2))
        cand = np.zeros((4, 2))
        assert candidate_covered_ratio(cand, db, radius=1.0, k_min=3) == 1.0

    def test_covered_ratio_none_covered(self):
        db = np.zeros((12, 2))
        cand = np.full((4, 2), 100.0)
        assert candidate_covered_ratio(cand, db, radius=1.0, k_min=3) == 0.0

    def test_covered_ratio_partial(self):
        db = np.zeros((12, 2))
        cand = np.array([[0.0, 0.0], [0.0, 0.0], [100.0, 0.0], [100.0, 0.0]])
        assert candidate_covered_ratio(cand, db, radius=1.0, k_min=3) == 0.5

    def test_covered_ratio_empty_candidate(self):
        assert candidate_covered_ratio(np.empty((0, 2)), np.zeros((5, 2)),
                                       radius=1.0) == 0.0

    def test_readiness_score_is_skill_mean(self):
        # skill A 후보는 buffer 안(covered), skill B 후보는 멀다(uncovered).
        db = _line_db(20)
        probes = [
            SkillProbe("A", [np.full((4, 1), 10.0)], db),     # R̄_cov = 1.0
            SkillProbe("B", [np.full((4, 1), 100.0)], db),    # R̄_cov = 0.0
        ]
        rep = evaluate_phase1_readiness(probes, Phase1ReadinessConfig(tau_ready=0.7))
        assert rep.per_skill["A"] == pytest.approx(1.0)
        assert rep.per_skill["B"] == pytest.approx(0.0)
        assert rep.r_ready == pytest.approx(0.5)
        assert rep.is_ready is False                          # 0.5 > 0.7 거짓

    def test_readiness_ready_when_all_covered(self):
        db = _line_db(20)
        probes = [
            SkillProbe("A", [np.full((3, 1), 8.0)], db),
            SkillProbe("B", [np.full((3, 1), 12.0)], db),
        ]
        rep = evaluate_phase1_readiness(probes, Phase1ReadinessConfig(tau_ready=0.7))
        assert rep.r_ready == pytest.approx(1.0)
        assert rep.is_ready is True

    def test_readiness_cold_start_small_buffer(self):
        # buffer 가 2개 미만이면 covered 불가 → R̄_cov = 0.
        probes = [SkillProbe("A", [np.zeros((4, 1))], np.zeros((1, 1)))]
        rep = evaluate_phase1_readiness(probes)
        assert rep.per_skill["A"] == 0.0
        assert rep.is_ready is False


# ─────────────────────────────────────────────────────────────
# §15.2  Phase2 saturation
# ─────────────────────────────────────────────────────────────
class TestPhase2Saturation:
    def test_window_mean_none_before_window_full(self):
        t = Phase2SaturationTracker(window=3)
        t.record(1.0)
        t.record(1.0)
        assert t.window_mean() is None        # 2 < W=3
        t.record(1.0)
        assert t.window_mean() == pytest.approx(1.0)

    def test_window_mean_uses_last_w(self):
        t = Phase2SaturationTracker(window=3)
        for v in [5.0, 5.0, 5.0, 0.0, 0.0, 0.0]:
            t.record(v)
        assert t.window_mean() == pytest.approx(0.0)   # 최근 3개만

    def test_is_saturated_negative_window_mean(self):
        t = Phase2SaturationTracker(window=3)
        for v in [-1.0, -2.0, -3.0]:
            t.record(v)
        assert t.is_saturated() is True

    def test_not_saturated_positive_window_mean(self):
        t = Phase2SaturationTracker(window=3)
        for v in [1.0, -0.5, 1.0]:
            t.record(v)
        assert t.is_saturated() is False      # 평균 > 0

    def test_not_saturated_before_window_full(self):
        t = Phase2SaturationTracker(window=5)
        t.record(-10.0)
        assert t.is_saturated() is False      # accepted < W → 판단 불가

    def test_rejects_bad_window(self):
        with pytest.raises(ValueError):
            Phase2SaturationTracker(window=0)


# ─────────────────────────────────────────────────────────────
# §15.3  Phase controller
# ─────────────────────────────────────────────────────────────
class TestPhaseController:
    def _cfg(self, **kw):
        base = dict(budget=100, phase1_min=20, phase1_max=50, tau_ready=0.7,
                    saturation_window=3)
        base.update(kw)
        return PhaseControllerConfig(**base)

    def test_starts_in_phase1(self):
        c = PhaseController(self._cfg())
        assert c.phase == "phase1" and c.t == 0

    def test_record_episode_increments_t(self):
        c = PhaseController(self._cfg())
        for _ in range(5):
            c.record_episode()
        assert c.t == 5

    def test_no_transition_before_phase1_min(self):
        c = PhaseController(self._cfg())
        for _ in range(10):
            c.record_episode()
        assert c.maybe_transition(r_ready=0.95) is False   # t=10 < B1_min=20
        assert c.phase == "phase1"

    def test_transition_on_readiness_after_min(self):
        c = PhaseController(self._cfg())
        for _ in range(20):
            c.record_episode()
        assert c.maybe_transition(r_ready=0.95) is True    # t=20, R_ready>τ
        assert c.phase == "phase2"
        assert c.status().phase1_episodes == 20

    def test_no_transition_when_readiness_low(self):
        c = PhaseController(self._cfg())
        for _ in range(20):
            c.record_episode()
        assert c.maybe_transition(r_ready=0.5) is False    # R_ready < τ, t<max
        assert c.phase == "phase1"

    def test_forced_transition_at_phase1_max(self):
        c = PhaseController(self._cfg())
        for _ in range(50):
            c.record_episode()
        assert c.maybe_transition(r_ready=0.0) is True     # t=50 = B1_max 강제
        assert c.phase == "phase2"

    def test_should_stop_at_budget(self):
        c = PhaseController(self._cfg(budget=30, phase1_min=5, phase1_max=10))
        for _ in range(30):
            c.record_episode()
        assert c.should_stop() is True

    def test_phase2_early_stop_on_saturation(self):
        c = PhaseController(self._cfg(budget=100, phase1_min=20, phase1_max=50,
                                      saturation_window=3))
        for _ in range(20):
            c.record_episode()
        c.maybe_transition(r_ready=0.95)                   # → phase2
        for _ in range(3):
            c.record_phase2_accept(-1.0)                   # Q̄̃2^(W) < 0
        assert c.should_stop() is True

    def test_phase2_early_stop_disabled(self):
        c = PhaseController(self._cfg(budget=100, phase1_min=20, phase1_max=50,
                                      saturation_window=3,
                                      enable_phase2_early_stop=False))
        for _ in range(20):
            c.record_episode()
        c.maybe_transition(r_ready=0.95)
        for _ in range(3):
            c.record_phase2_accept(-1.0)
        assert c.should_stop() is False                    # early stop 꺼짐

    def test_config_rejects_bad_guards(self):
        with pytest.raises(ValueError):
            PhaseControllerConfig(budget=100, phase1_min=60, phase1_max=50)
        with pytest.raises(ValueError):
            PhaseControllerConfig(budget=40, phase1_min=20, phase1_max=50)

    def test_full_acquisition_loop(self):
        # Phase1 → readiness 충족 시 전환 → Phase2 → budget 소진까지.
        cfg = self._cfg(budget=30, phase1_min=5, phase1_max=10, saturation_window=3)
        c = PhaseController(cfg)
        while not c.should_stop():
            if c.phase == "phase1":
                c.record_episode()
                c.maybe_transition(r_ready=0.9)            # t>=5 에서 전환
            else:
                c.record_episode()
                c.record_phase2_accept(1.0)               # 양수 → early stop 없음
        assert c.t == 30
        assert c.phase == "phase2"
        assert c.status().phase1_episodes == 5            # t=5 에서 전환됨
