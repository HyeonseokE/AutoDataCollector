"""Smoke tests for Selector with mock PolicyAdapter and BufferStore.

Run:  python -m preselective_filter.tests.test_selector_smoke
"""
from __future__ import annotations

import numpy as np

from preselective_filter import (
    BufferEntry,
    Candidate,
    Context,
    FMOutput,
    Selector,
    SelectorConfig,
)


class MockPolicy:
    def __init__(self, action_dim=7, chunk_len=4, hidden_dim=16, ctx_dim=12, seed=0):
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.hidden_dim = hidden_dim
        self.ctx_dim = ctx_dim
        self.rng = np.random.default_rng(seed)

    def forward_fm(self, context, action_chunk):
        a = np.asarray(action_chunk).reshape(-1)
        l_fm = float(np.linalg.norm(a)) * 0.1 + float(self.rng.uniform(0, 0.01))
        z = self.rng.standard_normal(self.hidden_dim).astype(np.float64)
        c_m = self.rng.standard_normal(self.ctx_dim).astype(np.float64)
        return FMOutput(l_fm=l_fm, z=z, c_m=c_m)

    def sample_actions(self, context, n_samples):
        return [
            self.rng.standard_normal((self.chunk_len, self.action_dim))
            for _ in range(n_samples)
        ]


class MockBuffer:
    def __init__(self):
        self.entries: list[BufferEntry] = []

    def append(self, entry):
        self.entries.append(entry)

    def query_skill(self, skill_id):
        return [e for e in self.entries if e.context.skill_id == skill_id]

    def nearest_by_context(self, context, context_embedding, skill_id, k):
        same = [e for e in self.entries if e.context.skill_id == skill_id]
        return same[-k:]


def _make_context(skill_id=0):
    return Context(
        observation=np.zeros((3, 224, 224), dtype=np.float32),
        state=np.zeros(7, dtype=np.float32),
        instruction="test",
        skill_id=skill_id,
    )


def _make_candidates(K=4, action_dim=7, chunk_len=4, seed=0):
    rng = np.random.default_rng(seed)
    return [
        Candidate(
            skill_id=0,
            action_chunk=rng.standard_normal((chunk_len, action_dim)),
            payload={"idx": i},
        )
        for i in range(K)
    ]


def test_cold_start():
    policy = MockPolicy(seed=1)
    buffer = MockBuffer()
    selector = Selector(policy, buffer, SelectorConfig())

    ctx = _make_context()
    cands = _make_candidates(K=4)
    sel = selector.select(ctx, cands)

    assert 0 <= sel.chosen_index < 4
    assert sel.chosen_candidate is cands[sel.chosen_index]
    assert len(sel.reports) == 4
    for r in sel.reports:
        assert r.n_buffer_norm == 1.0, f"n_buffer_norm should neutralize: {r.n_buffer_norm}"
        assert r.ac_buffer == 1.0, f"ac_buffer should neutralize: {r.ac_buffer}"
    print("[OK] cold_start")


def test_warm_start():
    policy = MockPolicy(seed=2)
    buffer = MockBuffer()
    selector = Selector(policy, buffer, SelectorConfig(n_vla_samples=4))
    ctx = _make_context()

    for trial in range(3):
        cands = _make_candidates(K=4, seed=trial)
        sel = selector.select(ctx, cands)
        selector.add_to_buffer(ctx, sel)

    assert len(buffer.entries) == 3

    cands = _make_candidates(K=4, seed=99)
    sel = selector.select(ctx, cands)
    assert 0 <= sel.chosen_index < 4

    ac_buffer_vals = [r.ac_buffer for r in sel.reports]
    assert not all(v == 1.0 for v in ac_buffer_vals), "ac_buffer should vary in warm-start"
    print(f"[OK] warm_start (chose {sel.chosen_index}, "
          f"score={sel.reports[sel.chosen_index].score:.4f})")


def test_report_shape_and_finite():
    policy = MockPolicy(seed=3)
    buffer = MockBuffer()
    # Pre-populate buffer with synthetic entries
    pre_rng = np.random.default_rng(10)
    for _ in range(5):
        buffer.append(BufferEntry(
            context=_make_context(),
            action_chunk=pre_rng.standard_normal((4, 7)),
            z=pre_rng.standard_normal(policy.hidden_dim),
            context_embedding=pre_rng.standard_normal(policy.ctx_dim),
        ))

    selector = Selector(policy, buffer, SelectorConfig())
    ctx = _make_context()
    cands = _make_candidates(K=8)
    sel = selector.select(ctx, cands)

    assert len(sel.reports) == 8
    for r in sel.reports:
        for field, value in vars(r).items():
            if isinstance(value, float):
                assert np.isfinite(value), f"non-finite {field}={value}"
    print("[OK] report_shape_and_finite")


def test_argmax_deterministic_with_seed():
    def run_once(seed):
        policy = MockPolicy(seed=seed)
        buffer = MockBuffer()
        selector = Selector(policy, buffer, SelectorConfig())
        ctx = _make_context()
        cands = _make_candidates(K=4, seed=seed)
        return selector.select(ctx, cands).chosen_index

    a, b = run_once(42), run_once(42)
    assert a == b, f"non-deterministic: {a} vs {b}"
    print(f"[OK] deterministic (chosen = {a})")


def test_score_formula_consistency():
    """Verify score == ig * ac for every candidate report."""
    policy = MockPolicy(seed=4)
    buffer = MockBuffer()
    # Warm up buffer
    pre_rng = np.random.default_rng(20)
    for _ in range(3):
        buffer.append(BufferEntry(
            context=_make_context(),
            action_chunk=pre_rng.standard_normal((4, 7)),
            z=pre_rng.standard_normal(policy.hidden_dim),
            context_embedding=pre_rng.standard_normal(policy.ctx_dim),
        ))
    selector = Selector(policy, buffer, SelectorConfig())
    ctx = _make_context()
    cands = _make_candidates(K=4)
    sel = selector.select(ctx, cands)
    for r in sel.reports:
        assert abs(r.score - r.ig * r.ac) < 1e-9, \
            f"score mismatch: {r.score} vs {r.ig * r.ac}"
    # Verify Selection carries chosen z and context_embedding
    assert sel.chosen_z.shape == (policy.hidden_dim,)
    assert sel.chosen_context_embedding.shape == (policy.ctx_dim,)
    # add_to_buffer round-trip
    n_before = len(buffer.entries)
    selector.add_to_buffer(ctx, sel)
    assert len(buffer.entries) == n_before + 1
    last = buffer.entries[-1]
    assert last.context_embedding.shape == (policy.ctx_dim,)
    print("[OK] score_formula_consistency")


if __name__ == "__main__":
    test_cold_start()
    test_warm_start()
    test_report_shape_and_finite()
    test_argmax_deterministic_with_seed()
    test_score_formula_consistency()
    print("\nAll smoke tests passed.")
