"""DCTDenoiseUncertainty — single-step denoise loss scorer."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from method3.phase2_mi_selection.mi_selector import Phase2Candidate
from method3.phase2_mi_selection.vla_dct_uncertainty import DCTDenoiseUncertainty


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
        # action 자리에 z_cand 가 들어갔는지 확인 위해 batch 저장.
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
        observations={"_batch": True},  # builder pass-through
        dct_target=dct_target,
    )


def _identity_builder(candidate):
    # observations 가 이미 batch dict → 그대로 사용.
    return {"observation.state": torch.zeros(1, 6)}


def test_dct_target_required():
    scorer = DCTDenoiseUncertainty(policy=_FakePolicy(), batch_builder=_identity_builder)
    with pytest.raises(ValueError, match="dct_target"):
        scorer.score(_make_candidate(dct_target=None))


def test_returns_policy_loss():
    policy = _FakePolicy(loss_value=0.123)
    scorer = DCTDenoiseUncertainty(policy=policy, batch_builder=_identity_builder)
    z = np.random.default_rng(0).normal(size=(50, 6))
    u = scorer.score(_make_candidate(dct_target=z))
    assert u == pytest.approx(0.123)


def test_action_replaced_with_dct_target():
    policy = _FakePolicy(loss_value=0.0)
    scorer = DCTDenoiseUncertainty(policy=policy, batch_builder=_identity_builder)
    z = np.arange(50 * 6, dtype=np.float32).reshape(50, 6)
    scorer.score(_make_candidate(dct_target=z))
    call = policy.calls[0]
    assert call["action"] is not None
    assert call["action"].shape == (1, 50, 6)
    np.testing.assert_allclose(call["action"].numpy()[0], z, atol=1e-6)


def test_single_step_R_equals_1():
    # paradigm 핵심: 단 한 번의 forward 만 호출 (R=1).
    policy = _FakePolicy()
    scorer = DCTDenoiseUncertainty(policy=policy, batch_builder=_identity_builder)
    z = np.zeros((50, 6))
    scorer.score(_make_candidate(dct_target=z))
    assert len(policy.calls) == 1


def test_sigma_passed_as_time_kwarg():
    policy = _FakePolicy()
    scorer = DCTDenoiseUncertainty(
        policy=policy, batch_builder=_identity_builder, sigma=0.7
    )
    z = np.zeros((50, 6))
    scorer.score(_make_candidate(dct_target=z))
    kwargs = policy.calls[0]["kwargs"]
    assert "time" in kwargs
    np.testing.assert_allclose(kwargs["time"].numpy(), [0.7], atol=1e-6)


def test_sigma_none_omits_time_kwarg():
    policy = _FakePolicy()
    scorer = DCTDenoiseUncertainty(policy=policy, batch_builder=_identity_builder)
    z = np.zeros((50, 6))
    scorer.score(_make_candidate(dct_target=z))
    assert "time" not in policy.calls[0]["kwargs"]


def test_policy_eval_mode_restored():
    policy = _FakePolicy()
    policy.training = True
    scorer = DCTDenoiseUncertainty(policy=policy, batch_builder=_identity_builder)
    z = np.zeros((50, 6))
    scorer.score(_make_candidate(dct_target=z))
    # was_training=True → restored after score().
    assert policy.training is True
