"""curobo_candidate_gen.candidates_from_trajectory_list — dct_target 채움 검증."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from method3.dct.transform import traj_to_dct
from method3.phase2_mi_selection.curobo_candidate_gen import (
    CurobogenConfig,
    candidates_from_trajectory_list,
)


def _fake_traj(N: int, dof: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(waypoints=rng.normal(size=(N, dof)), algo="curobo", cost=0.1)


def test_dct_target_shape_50_5():
    # arm-only DCT paradigm: dct_target_dof=5 (gripper 축 제외) → shape (L0, 5).
    trajs = [_fake_traj(40, 6, 0), _fake_traj(80, 6, 1)]
    cands = candidates_from_trajectory_list(
        trajs,
        skill_id="move",
        seed_subgoal=np.zeros(3),
        current_observation=None,
        instruction="",
        encoder=None,
        config=CurobogenConfig(action_horizon=50, dct_L0=50),
    )
    assert len(cands) == 2
    for c in cands:
        assert c.dct_target is not None
        assert c.dct_target.shape == (50, 5)


def test_dct_target_matches_direct_transform():
    # candidate 의 dct_target 이 traj_to_dct(waypoints[:, :5]) 와 동일해야 한다.
    # arm-only: production 코드가 dct_target_dof=5 로 truncate 하므로
    # expected 도 arm dof(5) 로 맞춰 비교한다.
    rng = np.random.default_rng(42)
    wp = rng.normal(size=(35, 6))
    traj = SimpleNamespace(waypoints=wp, algo="curobo", cost=0.0)
    cands = candidates_from_trajectory_list(
        [traj],
        skill_id="move",
        seed_subgoal=np.zeros(3),
        current_observation=None,
        instruction="",
        encoder=None,
        config=CurobogenConfig(action_horizon=50, dct_L0=50),
    )
    expected = traj_to_dct(wp[:, :5], L0=50)
    np.testing.assert_allclose(cands[0].dct_target, expected, atol=1e-10)


def test_dct_L0_configurable():
    # CurobogenConfig.dct_L0 변경 시 shape 추적. arm-only → (L0, 5).
    trajs = [_fake_traj(20, 6, 0)]
    cands = candidates_from_trajectory_list(
        trajs,
        skill_id="move",
        seed_subgoal=np.zeros(3),
        current_observation=None,
        instruction="",
        encoder=None,
        config=CurobogenConfig(dct_L0=30),
    )
    assert cands[0].dct_target.shape == (30, 5)
