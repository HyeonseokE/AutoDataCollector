"""Acquisition environment ports — 문서 final_method3_spec §2.

``Method3Acquisition`` 오케스트레이터가 의존하는 hardware/model/environment-facing
연산의 Protocol. 오케스트레이터는 method3 라이브러리(phase1·phase2·storage·
reembedding·phase_control)를 조합하는 **순수 로직**이고, 로봇 실행·VLA 모델·후보
생성처럼 실제 시스템에 닿는 부분은 이 Protocol 뒤로 분리한다.

실제 시스템(예: execution_forward_and_reset)이 ``AcquisitionEnvironment`` 를
구현해 오케스트레이터에 주입한다. 테스트는 in-memory fake 를 주입한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from method3.phase2_mi_selection.mi_selector import Phase2Candidate
from method3.phase_control.phase1_readiness import SkillProbe
from method3.reembedding.vla_encoder import VLAStateEncoder
from method3.storage.raw_dataset import RawDatasetEntry, RawTrajectoryDataset


@dataclass(frozen=True)
class Phase1EpisodeResult:
    """Phase1 acquisition episode 한 번의 결과 (문서 §4)."""

    judged_true: bool                       # task judge 결과 (로깅/통계용)
    raw_entries: list[RawDatasetEntry] = field(default_factory=list)  # D_phase1_raw 적재분


@dataclass(frozen=True)
class Phase2EpisodeResult:
    """accept 된 Phase2 후보를 실제 실행한 결과 (문서 §14)."""

    raw_entries: list[RawDatasetEntry] = field(default_factory=list)  # D_phase2_raw 적재분


@runtime_checkable
class AcquisitionEnvironment(Protocol):
    """오케스트레이터가 위임하는 hardware/model/environment 연산 (문서 §2).

    오케스트레이터는 이 6개 연산만 환경에 위임하고, phase 전환·budget·readiness·
    re-embedding·MI scoring 등 모든 method3 로직은 직접 소유한다.
    """

    def run_phase1_episode(self) -> Phase1EpisodeResult:
        """Phase1 episode 한 번 실행 — subgoal-seeded canonical trajectory.

        subgoal 선택·subgoal buffer 갱신은 skill layer 안에서 이미 일어난다.
        이 호출은 judge 결과와 D_phase1_raw 에 적재할 raw window 들을 돌려준다.
        """
        ...

    def probe_readiness_candidates(self) -> list[SkillProbe]:
        """Phase2-readiness 측정용 probe 후보 집합 (문서 §15.1).

        skill 당 하나의 ``SkillProbe`` — probe trajectory 후보들과 비교 buffer.
        Phase1 중에는 §7 의 임시 key 를 써도 된다 (환경 재량).
        """
        ...

    def build_phase2_encoder(
        self, phase1_raw: RawTrajectoryDataset,
    ) -> VLAStateEncoder:
        """§6 Step 1-2 — Phase1 raw 로 VLA encoder 를 학습·freeze 하여 반환한다.

        Phase1↔Phase2 전환 시 한 번 호출된다. 학습은 ML job 이므로 환경이 맡는다.
        """
        ...

    def resolve_observation(self, observation_ref: dict) -> np.ndarray:
        """raw dataset 의 ``observation_ref`` 포인터를 observation 배열로 푼다 (§4)."""
        ...

    def collect_seed_subgoals(self) -> dict[str, np.ndarray]:
        """Phase1 종료 시점의 skill-wise seed subgoal 집합 ``G_seed^{(m)}`` 를 돌려준다.

        Phase1 subgoal buffer ``B_{g}^{(m)}`` 가 모은 seed subgoal anchor 들 —
        ``skill_id → (N_m, 3)`` 배열. Phase2 는 새 subgoal 을 탐색하지 않고 이
        anchor 주변에서만 action variation 후보를 만든다 (phase1_seed_anchor_logic).
        Phase1↔Phase2 전환 시 한 번 호출된다.
        """
        ...

    def generate_phase2_candidates(
        self, encoder: VLAStateEncoder, seed_subgoals: dict[str, np.ndarray],
    ) -> list[Phase2Candidate]:
        """seed anchor 주변 Phase2 후보 trajectory set ``Ξ_t^{(m)}(g)`` 를 생성한다.

        Phase2 는 새 subgoal/state region 을 탐색하지 않는다 — ``seed_subgoals``
        의 각 Phase1 seed ``g ∈ G_seed^{(m)}`` 를 anchor 로 그 주변에서 skill-level
        trajectory perturbation 으로 action variation 후보만 만든다
        (phase1_seed_anchor_logic). 생성한 후보의 ``seed_subgoal`` 필드에는 anchor
        로 쓴 seed 를 채워 넘긴다.
        state key ``e_τ`` 는 §6 의 동일 encoder 로 추출해야 하므로 주입받는다.
        """
        ...

    def execute_phase2_candidate(
        self, candidate: Phase2Candidate,
    ) -> Phase2EpisodeResult:
        """accept 된 Phase2 후보를 실제 실행하고 raw window 들을 돌려준다 (§14)."""
        ...
