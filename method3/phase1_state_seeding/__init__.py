"""Method3 Phase1 — State Coverage Seeding (문서 final_method3_spec §4-5).

subgoal diversity 로 state coverage ``H(S)`` 를 키우고 path 는 canonical 하게
유지하는 Phase1 구현. 파일별 역할:

    subgoal_candidates.py  — §4.2  K개 subgoal 후보 G 생성 (uniform_ball/gaussian)
    subgoal_validity.py    — §4.2  G_valid 술어 (reachable/safe/AABB)
    canonical_preview.py   — §5.4  InterpPlan canonical preview + T_end 슬라이스
    terminal_descriptor.py — §5.3  φ_goal geometric descriptor ĥ
    subgoal_buffer.py      — §5    skill-wise subgoal buffer B_{g,t}^{(m)} + 거리
    subgoal_selector.py    — §5.4  argmax G_S^goal 선택기 (+ TRUE-only buffer update)
    legacy_blob.py         — ablation baseline (legacy 3D Gaussian blob)
"""
from method3.phase1_state_seeding.canonical_preview import interp_plan, last_segment
from method3.phase1_state_seeding.legacy_blob import (
    SubgoalPerturbation,
    SubgoalPerturbationConfig,
    TRANSIT_SKILL_TYPES,
)
from method3.phase1_state_seeding.subgoal_buffer import (
    SubgoalBuffer,
    SubgoalBufferEntry,
    knn_mean_distance,
    mean_nn_distance,
)
from method3.phase1_state_seeding.subgoal_candidates import sample_subgoal_candidates
from method3.phase1_state_seeding.subgoal_selector import (
    Phase1SubgoalConfig,
    Phase1SubgoalSelector,
    SubgoalScoreReport,
    SubgoalSelection,
)
from method3.phase1_state_seeding.subgoal_validity import (
    ReachabilityConfig,
    make_subgoal_validity_fn,
)
from method3.phase1_state_seeding.terminal_descriptor import (
    DESCRIPTOR_DIM,
    state_descriptor,
)

__all__ = [
    # legacy 3D Gaussian blob (ablation baseline)
    "SubgoalPerturbation",
    "SubgoalPerturbationConfig",
    "TRANSIT_SKILL_TYPES",
    # buffer-aware subgoal scoring (§4-5)
    "Phase1SubgoalSelector",
    "Phase1SubgoalConfig",
    "SubgoalScoreReport",
    "SubgoalSelection",
    "SubgoalBuffer",
    "SubgoalBufferEntry",
    "ReachabilityConfig",
    "make_subgoal_validity_fn",
    "sample_subgoal_candidates",
    "interp_plan",
    "last_segment",
    "state_descriptor",
    "DESCRIPTOR_DIM",
    "knn_mean_distance",
    "mean_nn_distance",
]
