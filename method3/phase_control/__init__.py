"""Method3 phase control — Phase saturation & transition (문서 §15).

acquisition 을 Phase1(state coverage seeding)↔Phase2(MI selection) 두 단계로
운영하며, 각 phase 가 목적을 달성했는지로 전환·종료를 판단한다. 파일별 역할:

    phase1_readiness.py   — §15.1  Phase2-readiness R_cov/R̄_cov^(m)/R_ready
    phase2_saturation.py  — §15.2  MI-style gain saturation Q̄̃2^(W)
    phase_controller.py   — §15.3  fixed budget + adaptive transition 상태기계
"""
from method3.phase_control.phase1_readiness import (
    Phase1ReadinessConfig,
    Phase1ReadinessReport,
    SkillProbe,
    candidate_covered_ratio,
    evaluate_phase1_readiness,
)
from method3.phase_control.phase2_saturation import Phase2SaturationTracker
from method3.phase_control.phase_controller import (
    PhaseController,
    PhaseControllerConfig,
    PhaseStatus,
)

__all__ = [
    # §15.1 Phase1 readiness
    "Phase1ReadinessConfig",
    "Phase1ReadinessReport",
    "SkillProbe",
    "candidate_covered_ratio",
    "evaluate_phase1_readiness",
    # §15.2 Phase2 saturation
    "Phase2SaturationTracker",
    # §15.3 phase controller
    "PhaseController",
    "PhaseControllerConfig",
    "PhaseStatus",
]
