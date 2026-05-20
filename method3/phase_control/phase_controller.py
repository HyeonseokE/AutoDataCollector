"""Acquisition phase controller — fixed budget + adaptive transition — §15.3.

논문 Method 는 전체 episode 수를 고정하지 않고 각 phase 가 목적을 달성했는지로
종료하지만, 실험에서는 baseline 과 공정 비교를 위해 전체 budget ``B`` 를 고정하되
Phase1↔Phase2 경계는 adaptive 하게 둔다.

  Phase1 transition: (t ≥ B_{1,min} ∧ R_ready > τ_ready) ∨ (t ≥ B_{1,max})
  Phase2 stop:       t = B ∨ Q̄̃2^(W) < 0

readiness ``R_ready`` 는 ``phase1_readiness.evaluate_phase1_readiness`` 가,
saturation ``Q̄̃2^(W)`` 는 ``phase2_saturation.Phase2SaturationTracker`` 가
계산한다. 이 컨트롤러는 그 신호로 budget guard 와 phase 전환을 관리하는
상태기계다.
"""
from __future__ import annotations

from dataclasses import dataclass

from method3.phase_control.phase2_saturation import Phase2SaturationTracker


@dataclass
class PhaseControllerConfig:
    """Acquisition 컨트롤러 파라미터 (문서 §15.3)."""

    budget: int = 100                       # B — 전체 acquisition budget
    phase1_min: int = 20                    # B_{1,min} — Phase1 최소 guard
    phase1_max: int = 50                    # B_{1,max} — Phase1 최대 guard
    tau_ready: float = 0.7                  # §15.1 readiness 임계값
    saturation_window: int = 10             # §15.2 W
    enable_phase2_early_stop: bool = True    # False → 공정비교용 (t=B 까지만)

    def __post_init__(self) -> None:
        if not 1 <= self.phase1_min <= self.phase1_max <= self.budget:
            raise ValueError(
                "must satisfy 1 <= phase1_min <= phase1_max <= budget, got "
                f"min={self.phase1_min}, max={self.phase1_max}, B={self.budget}"
            )


@dataclass(frozen=True)
class PhaseStatus:
    """컨트롤러 상태 스냅샷 (로깅/디버깅용)."""

    phase: str                              # "phase1" | "phase2"
    t: int                                  # 완료된 acquisition episode 수
    budget: int                             # B
    phase1_episodes: int | None             # B_1 — Phase2 전환 시점에 고정
    saturation_window_mean: float | None    # Q̄̃2^(W) — accepted < W 면 None
    done: bool                              # should_stop()


class PhaseController:
    """Phase1↔Phase2 전환과 종료를 관리하는 acquisition 상태기계 (문서 §15.3).

    Usage (acquisition loop)::

        ctrl = PhaseController(cfg)
        while not ctrl.should_stop():
            if ctrl.phase == "phase1":
                run_phase1_episode()
                ctrl.record_episode()
                ctrl.maybe_transition(measured_r_ready)
            else:
                sel = run_phase2_episode()
                ctrl.record_episode()
                if sel.accepted:
                    ctrl.record_phase2_accept(sel.reports[sel.chosen_index].q2_norm)
    """

    def __init__(self, config: PhaseControllerConfig | None = None) -> None:
        self.cfg = config or PhaseControllerConfig()
        self.phase = "phase1"
        self.t = 0                                   # 완료된 episode 수
        self._phase1_episodes: int | None = None     # B_1 (전환 시 고정)
        self._sat = Phase2SaturationTracker(self.cfg.saturation_window)

    def record_episode(self) -> None:
        """acquisition episode 하나 완료 — episode 카운터 ``t`` 증가."""
        self.t += 1

    def record_phase2_accept(self, q2_norm: float) -> None:
        """Phase2 에서 후보가 accept 되면 그 정규화 score ``Q̃2(ξ*)`` 를 기록한다."""
        self._sat.record(q2_norm)

    def _phase1_transition_met(self, r_ready: float) -> bool:
        """§15.3 — (t ≥ B_{1,min} ∧ R_ready > τ_ready) ∨ (t ≥ B_{1,max})."""
        c = self.cfg
        readiness_ok = self.t >= c.phase1_min and r_ready > c.tau_ready
        max_guard = self.t >= c.phase1_max
        return readiness_ok or max_guard

    def maybe_transition(self, r_ready: float = 0.0) -> bool:
        """Phase1 transition rule 을 검사해 만족하면 Phase2 로 전환한다 (문서 §15.3).

        Args:
            r_ready: 측정된 readiness score. 측정하지 않은 episode 면 0.0 을
                넘겨 max-guard(``t ≥ B_{1,max}``)만 적용한다.

        Returns:
            이 호출에서 Phase2 로 전환했으면 True.
        """
        if self.phase == "phase1" and self._phase1_transition_met(r_ready):
            self.phase = "phase2"
            self._phase1_episodes = self.t
            return True
        return False

    def should_stop(self) -> bool:
        """acquisition 종료 여부 — §15.3 ``t = B ∨ Q̄̃2^(W) < 0``."""
        if self.t >= self.cfg.budget:
            return True
        if self.phase == "phase2" and self.cfg.enable_phase2_early_stop:
            return self._sat.is_saturated()
        return False

    def status(self) -> PhaseStatus:
        """현재 상태 스냅샷."""
        return PhaseStatus(
            phase=self.phase,
            t=self.t,
            budget=self.cfg.budget,
            phase1_episodes=self._phase1_episodes,
            saturation_window_mean=self._sat.window_mean(),
            done=self.should_stop(),
        )
