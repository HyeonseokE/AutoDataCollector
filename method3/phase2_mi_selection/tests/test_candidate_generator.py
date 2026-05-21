"""Phase2 candidate generator — Protocol conformance + MockCandidateGenerator tests."""
from __future__ import annotations

import numpy as np

from method3.phase2_mi_selection.candidate_generator import (
    MockCandidateGenerator,
    Phase2CandidateGenerator,
)
from method3.phase2_mi_selection.mi_selector import Phase2Candidate
from method3.reembedding.vla_encoder import MeanPoolStateEncoder


class TestPhase2CandidateGenerator:
    def test_mock_satisfies_protocol(self):
        gen = MockCandidateGenerator()
        assert isinstance(gen, Phase2CandidateGenerator)

    def test_generates_K_candidates(self):
        gen = MockCandidateGenerator(rng=np.random.default_rng(42))
        enc = MeanPoolStateEncoder(out_dim=8)
        seed = np.array([0.3, 0.0, 0.2])
        cands = gen.generate(seed, "move", {"instruction": "do"}, enc, K=4)
        assert len(cands) == 4
        assert all(isinstance(c, Phase2Candidate) for c in cands)

    def test_each_candidate_carries_seed_anchor(self):
        gen = MockCandidateGenerator(rng=np.random.default_rng(0))
        enc = MeanPoolStateEncoder(out_dim=8)
        seed = np.array([0.5, -0.1, 0.3])
        cands = gen.generate(seed, "move_and_close", {"instruction": "do"}, enc, K=3)
        for c in cands:
            assert np.allclose(c.seed_subgoal, seed)

    def test_state_keys_dim_consistent(self):
        # state_keys 의 D_e = VLA out_dim + proprio dim.
        gen = MockCandidateGenerator(rng=np.random.default_rng(0), proprio_dim=7)
        enc = MeanPoolStateEncoder(out_dim=8)
        cands = gen.generate(
            np.zeros(3), "move", {"instruction": ""}, enc, K=2)
        for c in cands:
            assert c.state_keys.shape[1] == 8 + 7  # VLA + proprio

    def test_action_chunks_shape(self):
        gen = MockCandidateGenerator(
            rng=np.random.default_rng(0), action_dim=6, action_horizon=12, n_steps=3)
        enc = MeanPoolStateEncoder(out_dim=8)
        cands = gen.generate(np.zeros(3), "move", {"instruction": ""}, enc, K=2)
        for c in cands:
            assert c.action_chunks.shape == (3, 12, 6)
