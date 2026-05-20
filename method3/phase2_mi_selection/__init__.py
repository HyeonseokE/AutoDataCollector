"""Method3 Phase2 — MI-based Consistent Diversity Acquisition (문서 §7-14).

Phase1 이 만든 state support 안에서, 후보 trajectory 를 MI-style score
``Q2 = β·ΔH_A − λ·ΔH_A|S`` 로 평가하여 action diversity 는 키우되 유사 state
에서 action ambiguity 는 키우지 않는 trajectory 만 선별 수집한다. 파일별 역할:

    action_descriptor.py     — §4.2  DCT action descriptor ψ → z^a
    vector_db.py             — §3/§7.2  skill-wise vector DB B_t^{(m)}
    neighbor_search.py       — §8/§9  L2 kNN/meanNN/min 거리 primitive
    radius.py                — §16  state-neighborhood radius ρ_m
    action_coverage.py       — §8  action coverage gain ΔH_A
    conditional_ambiguity.py — §9  covered-state + conditional ambiguity ΔH_A|S
    mi_selector.py           — §11-12  Q2 score + 배치 정규화 + accept + argmax
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
    # §11-12 MI selector
    "Phase2Candidate",
    "Phase2MIConfig",
    "Phase2MISelector",
    "Phase2ScoreReport",
    "Phase2Selection",
]
