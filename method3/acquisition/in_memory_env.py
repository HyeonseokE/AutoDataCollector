"""In-memory acquisition environment — 하드웨어 없는 reference 구현 (문서 §2).

``AcquisitionEnvironment`` 포트의 in-memory 시뮬레이션 구현. 로봇·VLA 모델·후보
generator 없이 합성 데이터로 두-단계 acquisition 루프 전체를 돌릴 수 있다.

용도:
  - method3 오케스트레이션 로직을 하드웨어 없이 검증 (단위 테스트 / CI)
  - ``demo_offline`` 의 dry-run 환경
  - 실제 ``AcquisitionEnvironment`` 구현 시 포트별 입출력 형상의 참조

``ready_after`` episode 부터 readiness probe 가 covered(R_ready=1.0)로 바뀐다.
None 이면 영원히 uncovered → Phase1 은 B_{1,max} 에서 강제 전환된다.
"""
from __future__ import annotations

import numpy as np

from method3.acquisition.ports import Phase1EpisodeResult, Phase2EpisodeResult
from method3.phase2_mi_selection.mi_selector import Phase2Candidate
from method3.phase_control.phase1_readiness import SkillProbe
from method3.reembedding.vla_encoder import MeanPoolStateEncoder
from method3.storage.raw_dataset import RawDatasetEntry

# 합성 데이터 차원 — seed key = encode(out_dim) + proprioception, 후보와 동일.
_P, _H, _A, _OUT = 7, 12, 6, 8


def _make_raw_entry(skill: str, phase: str, episode: str, t: int) -> RawDatasetEntry:
    """§3.1 필드를 모두 채운 합성 RawDatasetEntry."""
    rng = np.random.default_rng(abs(hash((skill, episode, t))) % (2 ** 31))
    return RawDatasetEntry(
        episode_id=episode, phase=phase, skill_id=skill,
        instruction=f"do {skill}",
        subgoal=np.array([0.3, 0.0, 0.2]),
        time_index=t,
        observation_ref={"frame": t, "episode": episode},
        proprioception=rng.normal(0, 1, _P),
        action_chunk=rng.normal(0, 1, (_H, _A)),
        raw_action_sequence=rng.normal(0, 1, (_H, _A)),
        planner_type="InterpPlan",
        success_flag=True, validity_flag=True,
    )


def _line_db(n: int) -> np.ndarray:
    """probe 비교용 buffer — radius 가 예측 가능한 1축 정수 격자."""
    return np.arange(n, dtype=float).reshape(n, 1)


class InMemoryAcquisitionEnvironment:
    """모든 hardware/model 연산을 in-memory 로 시뮬레이션하는 환경 (문서 §2)."""

    def __init__(self, ready_after: int | None = None) -> None:
        self.ready_after = ready_after
        self._phase1_count = 0
        self._rng = np.random.default_rng(0)
        self._seed_subgoals: dict[str, list] = {}   # G_seed^{(m)} 누적 (phase1)

    def run_phase1_episode(self) -> Phase1EpisodeResult:
        self._phase1_count += 1
        entries = [
            _make_raw_entry("reach", "phase1", f"p1_ep{self._phase1_count}", t)
            for t in range(2)                       # episode 당 2 window
        ]
        # phase1_seed_anchor_logic — episode 마다 seed subgoal 하나를 G_seed 에 적재.
        seed = np.array([0.3, 0.0, 0.2]) + self._rng.normal(0, 0.05, 3)
        self._seed_subgoals.setdefault("reach", []).append(seed)
        return Phase1EpisodeResult(judged_true=True, raw_entries=entries)

    def probe_readiness_candidates(self) -> list[SkillProbe]:
        db = _line_db(20)
        covered = self.ready_after is not None and self._phase1_count >= self.ready_after
        probe_x = 10.0 if covered else 100.0        # 10=covered, 100=uncovered
        return [SkillProbe("reach", [np.full((4, 1), probe_x)], db)]

    def build_phase2_encoder(self, phase1_raw) -> MeanPoolStateEncoder:
        return MeanPoolStateEncoder(out_dim=_OUT)

    def resolve_observation(self, observation_ref: dict) -> np.ndarray:
        return np.full((3, 4), float(observation_ref.get("frame", 0)))

    def collect_seed_subgoals(self) -> dict[str, np.ndarray]:
        # G_seed^{(m)} — Phase1 이 모은 skill-wise seed subgoal anchor 집합.
        return {m: np.stack(gs) for m, gs in self._seed_subgoals.items() if gs}

    def generate_phase2_candidates(
        self, encoder, seed_subgoals: dict[str, np.ndarray],
    ) -> list[Phase2Candidate]:
        # phase1_seed_anchor_logic — 새 subgoal 탐색 없이, G_seed 의 각 seed 를
        # anchor 로 그 주변에서만 action variation 후보를 만든다.
        seeds = seed_subgoals.get("reach")
        if seeds is None or len(seeds) == 0:
            return []
        cands = []
        for c in range(3):                          # 후보마다 action 분포 다르게
            anchor = seeds[c % len(seeds)]          # G_seed 의 seed 를 anchor 로
            keys = np.stack([
                np.concatenate([
                    encoder.encode(self._rng.normal(0, 1, (3, 4)), "phase2"),
                    self._rng.normal(0, 1, _P),
                ])
                for _ in range(3)
            ])
            chunks = self._rng.normal(float(c), 1.0, (3, _H, _A))
            cands.append(Phase2Candidate("reach", keys, chunks, seed_subgoal=anchor))
        return cands

    def execute_phase2_candidate(self, candidate) -> Phase2EpisodeResult:
        return Phase2EpisodeResult(
            raw_entries=[_make_raw_entry("reach", "phase2", "p2_ep", 0)])
