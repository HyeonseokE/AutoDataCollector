"""Phase2 candidate generator — Protocol + reference impl (phase1_seed_anchor_logic).

스펙 §7.3 와 phase1_seed_anchor_logic 에 따라 Phase2 후보 trajectory 는 Phase1
seed subgoal ``g ∈ G_seed^{(m)}`` 주변에서 skill-level trajectory perturbation
으로 생성된다. 본 모듈은 그 정의를 라이브러리 인터페이스로 노출한다:

  ``Phase2CandidateGenerator.generate(seed_subgoal, skill_id, current_state,
                                     encoder, K) -> list[Phase2Candidate]``

실제 trajectory 생성기는 라이브러리 밖이다 (하드웨어에선 curobo skill-level
perturbation, 시뮬에선 InterpPlan 변형 등). 본 모듈은:

- Protocol 정의 → 구현체가 이 모양만 맞추면 ``Method3Acquisition`` /
  ``Phase2MISelector`` 에 그대로 꽂힌다.
- ``MockCandidateGenerator`` — encoder + RNG 만으로 합성 candidate 를 만드는
  reference (테스트·offline 검증용).

production 어댑터 (예: ``method3/phase2_mi_selection/curobo_candidate_gen.py``)
는 별도 파일로 분리한다 — curobo runtime / preselective_filter.GrpcPlannerClient
와 연결하는 것이 본질적으로 하드웨어 의존이라 본 라이브러리에 들어오면 안 됨.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from method3.phase2_mi_selection.mi_selector import Phase2Candidate
from method3.reembedding.vla_encoder import VLAStateEncoder


@runtime_checkable
class Phase2CandidateGenerator(Protocol):
    """seed-anchored Phase2 candidate generator (phase1_seed_anchor_logic).

    각 호출은 ``seed_subgoal`` anchor 주변에서 K 개의 trajectory 변형을 만들고,
    각각을 ``Phase2Candidate`` 로 감싸 반환한다. candidate 의 ``state_keys`` 는
    rollout 시작 시점 ``current_state`` (observation + proprio) 와 trajectory
    의 per-window proprio 로부터 frozen ``encoder`` 를 통해 추출된다.
    """

    def generate(
        self,
        seed_subgoal: np.ndarray,
        skill_id: str,
        current_state: dict,
        encoder: VLAStateEncoder,
        K: int,
    ) -> list[Phase2Candidate]:
        """seed 주변 K 개 후보.

        Args:
            seed_subgoal: ``G_seed^{(m)}`` 의 한 anchor g — (3,).
            skill_id: 현재 skill m.
            current_state: rollout 시작 시점 정보. 최소 ``observation`` (raw
                image dict 또는 array) 과 ``proprioception`` (1-D array) 키
                포함. 실제 생성기 (curobo 등) 가 더 많은 필드를 요구할 수 있음.
            encoder: state key 추출에 쓸 frozen VLA encoder
                (P_phase1 구축과 **같은 instance**, §6 핵심 원칙).
            K: 생성할 후보 수.

        Returns:
            K 개 ``Phase2Candidate``. 각 candidate 의 ``seed_subgoal`` 필드는
            입력 ``seed_subgoal`` 과 동일하게 채워진다.
        """
        ...


class MockCandidateGenerator:
    """오프라인·테스트용 mock generator — curobo 없이 합성 candidate 생성.

    Phase2 파이프라인 (orchestrator → mi_selector → accept) 의 end-to-end 검증
    용. 실제 trajectory feasibility / robot constraints 는 모사하지 않는다.

    state_keys: encoder 에 합성 obs 를 K·T 번 통과시킨 뒤 proprio (seed 근처
    난수) 와 concat. action_chunks: seed 근처 좌표 변동 + Gaussian noise.
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator | None = None,
        action_dim: int = 6,
        action_horizon: int = 50,
        n_steps: int = 3,
        proprio_dim: int = 7,
        obs_shape: tuple[int, ...] = (3, 4),
        n_windows: int | None = None,  # deprecated alias for n_steps
    ) -> None:
        # n_windows 는 옛 이름. 의미가 *T = trajectory 시간 길이* 로 갱신되어
        # n_steps 로 rename — backward compat 위해 옛 키 받으면 fallback.
        if n_windows is not None:
            n_steps = n_windows
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._action_dim = int(action_dim)
        self._H = int(action_horizon)
        self._T = int(n_steps)
        self._proprio_dim = int(proprio_dim)
        self._obs_shape = obs_shape

    def generate(
        self,
        seed_subgoal,
        skill_id,
        current_state,
        encoder,
        K,
    ) -> list[Phase2Candidate]:
        seed = np.asarray(seed_subgoal, dtype=np.float64).reshape(3)
        instruction = str(current_state.get("instruction", ""))
        out: list[Phase2Candidate] = []
        for c in range(int(K)):
            # state_keys (T, D_e): per-window VLA embedding + proprio.
            state_keys = []
            for _ in range(self._T):
                obs = self._rng.normal(0, 1, self._obs_shape)
                e_vla = np.asarray(encoder.encode(obs, instruction), dtype=np.float64).reshape(-1)
                p = self._rng.normal(0, 0.1, self._proprio_dim)
                state_keys.append(np.concatenate([e_vla, p]))
            # action_chunks (T, H, action_dim): seed 근처 + Gaussian.
            chunks = self._rng.normal(
                loc=float(c) * 0.1, scale=1.0,
                size=(self._T, self._H, self._action_dim))
            out.append(Phase2Candidate(
                skill_id=str(skill_id),
                state_keys=np.stack(state_keys),
                action_chunks=chunks,
                seed_subgoal=seed,
            ))
        return out
