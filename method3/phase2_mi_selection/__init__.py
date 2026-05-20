"""Method3 Phase2 — Useful-OOD Acquisition (final_method3_spec_useful_ood_updated §7-14).

Phase1 이 만든 state support 안에서, 후보 trajectory 를 두 축으로 평가한다::

    M_MI(ξ) = β·ΔH_A − λ·ΔH_A|S         (§11 buffer-side usefulness)
    U_VLA(ξ) = R-stochastic denoise loss (§12 model-side informativeness)
    ξ* = argmax U_VLA(ξ)  s.t.  M̃_MI ≥ τ_MI    (§13.2 Useful-OOD rule)

이로써 ① VLA 가 낯설지만(high U_VLA) ② buffer 입장에서도 의미 있는(high M_MI)
"Useful OOD" 후보를 우선 수집한다. Harmful OOD 와 Redundant ID 는 M̃_MI constraint
가 자동 제거. 파일별 역할:

    action_descriptor.py     — §4.2  DCT action descriptor ψ → z^a
    vector_db.py             — §3/§7.2  skill-wise vector DB B_t^{(m)}
    neighbor_search.py       — §8/§9  L2 kNN/meanNN/min 거리 primitive
    radius.py                — §16  state-neighborhood radius ρ_m
    action_coverage.py       — §8  action coverage gain ΔH_A
    conditional_ambiguity.py — §9  covered-state + conditional ambiguity ΔH_A|S
    mi_selector.py           — §11 M_MI + §13 Useful-OOD selection rule
    vla_informativeness.py   — §12 U_VLA (R-stochastic denoise loss)
"""
from method3.phase2_mi_selection.action_coverage import (
    action_coverage_gain,
    action_novelty,
    top_quantile_mean,
)
from method3.phase2_mi_selection.action_descriptor import (
    dct_action_descriptor,
    dct_energy_optimal_k,
    dct_time,
)
from method3.phase2_mi_selection.conditional_ambiguity import (
    ConditionalAmbiguityReport,
    conditional_ambiguity,
    covered_windows,
)
from method3.phase2_mi_selection.mi_selector import (
    Phase2Candidate,
    Phase2MIConfig,
    Phase2MISelector,
    Phase2ScoreReport,
    Phase2Selection,
)
from method3.phase2_mi_selection.neighbor_search import (
    knn_mean_distance,
    mean_nn_distance,
    min_distance,
)
from method3.phase2_mi_selection.radius import state_neighborhood_radius
from method3.phase2_mi_selection.vector_db import SkillVectorDB, VectorDBEntry
from method3.phase2_mi_selection.vla_informativeness import (
    ActionMagnitudeScorer,
    ConstantScorer,
    LeRobotBatchBuilder,
    LeRobotVLAInformativenessScorer,
    VLAInformativenessScorer,
    make_default_scorer,
)

__all__ = [
    # §4.2 DCT action descriptor
    "dct_time",
    "dct_action_descriptor",
    "dct_energy_optimal_k",
    # §3/§7.2 skill-wise vector DB
    "SkillVectorDB",
    "VectorDBEntry",
    # §8/§9 neighbor-search primitives
    "knn_mean_distance",
    "mean_nn_distance",
    "min_distance",
    # §16 radius
    "state_neighborhood_radius",
    # §8 action coverage gain
    "action_novelty",
    "top_quantile_mean",
    "action_coverage_gain",
    # §9 conditional ambiguity
    "covered_windows",
    "conditional_ambiguity",
    "ConditionalAmbiguityReport",
    # §11/§13 MI + Useful-OOD selector
    "Phase2Candidate",
    "Phase2MIConfig",
    "Phase2MISelector",
    "Phase2ScoreReport",
    "Phase2Selection",
    # §12 U_VLA scorers + batch builder
    "VLAInformativenessScorer",
    "ConstantScorer",
    "ActionMagnitudeScorer",
    "LeRobotVLAInformativenessScorer",
    "LeRobotBatchBuilder",
    "make_default_scorer",
]
