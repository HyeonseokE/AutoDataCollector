"""LeRobotVLAInformativenessScorer mode='dct' + alias DCTDenoiseUncertainty."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from method3.phase2_mi_selection.mi_selector import Phase2Candidate
from method3.phase2_mi_selection.vla_dct_uncertainty import DCTDenoiseUncertainty
from method3.phase2_mi_selection.vla_informativeness import (
    LeRobotVLAInformativenessScorer,
)


class _FakePolicy:
    """forward(batch, reduction, **kwargs) → (loss_tensor, info_dict) mock."""

    def __init__(self, loss_value: float = 0.42):
        self.loss_value = loss_value
        self.calls: list[dict] = []
        self.training = False

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def forward(self, batch, reduction="mean", **kwargs):
        self.calls.append({
            "action": batch.get("action"),
            "kwargs": kwargs,
            "reduction": reduction,
        })
        return torch.tensor(self.loss_value), {"loss": self.loss_value}


def _make_candidate(dct_target=None):
    return Phase2Candidate(
        skill_id="move",
        state_keys=np.zeros((1, 8)),
        action_chunks=np.zeros((1, 50, 6)),
        observations={"_batch": True},
        dct_target=dct_target,
    )


def _identity_builder(candidate):
    return {"observation.state": torch.zeros(1, 6)}


# ─────────────────────────────────────────────
# mode='dct' on LeRobotVLAInformativenessScorer
# ─────────────────────────────────────────────


def test_dct_mode_requires_dct_target():
    scorer = LeRobotVLAInformativenessScorer(
        policy=_FakePolicy(), batch_builder=_identity_builder, mode="dct"
    )
    with pytest.raises(ValueError, match="dct_target"):
        scorer.score(_make_candidate(dct_target=None))


def test_dct_mode_action_replaced_with_dct_target():
    policy = _FakePolicy(loss_value=0.0)
    scorer = LeRobotVLAInformativenessScorer(
        policy=policy, batch_builder=_identity_builder, mode="dct"
    )
    z = np.arange(50 * 6, dtype=np.float32).reshape(50, 6)
    scorer.score(_make_candidate(dct_target=z))
    call = policy.calls[0]
    assert call["action"] is not None
    assert call["action"].shape == (1, 50, 6)
    np.testing.assert_allclose(call["action"].numpy()[0], z, atol=1e-6)


def test_dct_mode_single_step_forced():
    # R=1 강제 (param 으로 R 더 큰 값을 줘도 1 회만 forward).
    policy = _FakePolicy()
    scorer = LeRobotVLAInformativenessScorer(
        policy=policy, batch_builder=_identity_builder, mode="dct", R=99
    )
    z = np.zeros((50, 6))
    scorer.score(_make_candidate(dct_target=z))
    assert len(policy.calls) == 1


def test_dct_mode_sigma_passed_as_time():
    policy = _FakePolicy()
    scorer = LeRobotVLAInformativenessScorer(
        policy=policy, batch_builder=_identity_builder, mode="dct", sigma=0.7
    )
    z = np.zeros((50, 6))
    scorer.score(_make_candidate(dct_target=z))
    kwargs = policy.calls[0]["kwargs"]
    assert "time" in kwargs
    np.testing.assert_allclose(kwargs["time"].numpy(), [0.7], atol=1e-6)


def test_dct_mode_sigma_none_omits_time():
    policy = _FakePolicy()
    scorer = LeRobotVLAInformativenessScorer(
        policy=policy, batch_builder=_identity_builder, mode="dct"
    )
    z = np.zeros((50, 6))
    scorer.score(_make_candidate(dct_target=z))
    assert "time" not in policy.calls[0]["kwargs"]


def test_unknown_mode_raises():
    scorer = LeRobotVLAInformativenessScorer(
        policy=_FakePolicy(), batch_builder=_identity_builder, mode="bogus"
    )
    with pytest.raises(ValueError, match="unknown mode"):
        scorer.score(_make_candidate(dct_target=np.zeros((50, 6))))


# ─────────────────────────────────────────────
# default mode (backward compat) — R 그대로
# ─────────────────────────────────────────────


def test_default_mode_runs_R_times():
    policy = _FakePolicy()
    scorer = LeRobotVLAInformativenessScorer(
        policy=policy, batch_builder=_identity_builder, mode="default", R=5
    )
    scorer.score(_make_candidate(dct_target=None))  # default mode 는 dct_target 불필요
    assert len(policy.calls) == 5


def test_default_mode_does_not_swap_action():
    policy = _FakePolicy()
    scorer = LeRobotVLAInformativenessScorer(
        policy=policy, batch_builder=_identity_builder, mode="default", R=1
    )
    scorer.score(_make_candidate(dct_target=None))
    # _identity_builder 가 batch 에 action 안 넣음 → default mode 도 안 넣어야 함.
    assert policy.calls[0]["action"] is None


# ─────────────────────────────────────────────
# Backward-compat alias DCTDenoiseUncertainty
# ─────────────────────────────────────────────


def test_alias_returns_scorer_with_dct_mode():
    scorer = DCTDenoiseUncertainty(policy=_FakePolicy(), batch_builder=_identity_builder)
    assert isinstance(scorer, LeRobotVLAInformativenessScorer)
    assert scorer.mode == "dct"
    assert scorer.R == 1


def test_alias_with_sigma():
    policy = _FakePolicy()
    scorer = DCTDenoiseUncertainty(policy=policy, batch_builder=_identity_builder, sigma=0.5)
    z = np.zeros((50, 6))
    scorer.score(_make_candidate(dct_target=z))
    kwargs = policy.calls[0]["kwargs"]
    np.testing.assert_allclose(kwargs["time"].numpy(), [0.5], atol=1e-6)


def test_policy_eval_mode_restored():
    policy = _FakePolicy()
    policy.training = True
    scorer = LeRobotVLAInformativenessScorer(
        policy=policy, batch_builder=_identity_builder, mode="dct"
    )
    z = np.zeros((50, 6))
    scorer.score(_make_candidate(dct_target=z))
    assert policy.training is True
