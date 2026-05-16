"""Smoke tests for the buffer-only Selector with a mock BufferStore.

Run:  python -m preselective_filter.tests.test_selector_smoke
"""
from __future__ import annotations

import numpy as np

from preselective_filter import (
    BufferEntry,
    Candidate,
    Context,
    Selector,
    SelectorConfig,
)


class MockBuffer:
    def __init__(self):
        self.entries: list[BufferEntry] = []

    def append(self, entry):
        self.entries.append(entry)

    def query_skill(self, skill_id):
        return [e for e in self.entries if e.context.skill_id == skill_id]

    def nearest_by_context(self, context, skill_id, k):
        same = [e for e in self.entries if e.context.skill_id == skill_id]
        return same[-k:]


def _make_context(skill_id=0):
    return Context(
        observation=np.zeros(0, dtype=np.float32),
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
    buffer = MockBuffer()
    selector = Selector(buffer, SelectorConfig(), rng=np.random.default_rng(1))

    ctx = _make_context()
    cands = _make_candidates(K=4)
    sel = selector.select(ctx, cands)

    assert 0 <= sel.chosen_index < 4
    assert sel.chosen_candidate is cands[sel.chosen_index]
    assert len(sel.reports) == 4
    for r in sel.reports:
        assert r.ig == 1.0, f"ig should neutralize on empty buffer: {r.ig}"
        assert r.ac == 1.0, f"ac should neutralize on empty buffer: {r.ac}"
    print("[OK] cold_start")


def test_warm_start():
    buffer = MockBuffer()
    selector = Selector(buffer, SelectorConfig(), rng=np.random.default_rng(2))
    ctx = _make_context()

    for trial in range(3):
        cands = _make_candidates(K=4, seed=trial)
        sel = selector.select(ctx, cands)
        selector.add_to_buffer(ctx, sel)

    assert len(buffer.entries) == 3

    cands = _make_candidates(K=4, seed=99)
    sel = selector.select(ctx, cands)
    assert 0 <= sel.chosen_index < 4

    ig_vals = [r.ig for r in sel.reports]
    ac_vals = [r.ac for r in sel.reports]
    assert not all(v == 1.0 for v in ig_vals), "ig should vary in warm-start"
    assert not all(v == 1.0 for v in ac_vals), "ac should vary in warm-start"
    print(f"[OK] warm_start (chose {sel.chosen_index}, "
          f"score={sel.reports[sel.chosen_index].score:.4f})")


def test_report_shape_and_finite():
    buffer = MockBuffer()
    pre_rng = np.random.default_rng(10)
    for _ in range(5):
        buffer.append(BufferEntry(
            context=_make_context(),
            action_chunk=pre_rng.standard_normal((4, 7)),
        ))

    selector = Selector(buffer, SelectorConfig(), rng=np.random.default_rng(3))
    ctx = _make_context()
    cands = _make_candidates(K=8)
    sel = selector.select(ctx, cands)

    assert len(sel.reports) == 8
    for r in sel.reports:
        for field, value in vars(r).items():
            if isinstance(value, float):
                assert np.isfinite(value), f"non-finite {field}={value}"
    print("[OK] report_shape_and_finite")


def test_cold_start_rng_deterministic():
    def run_once(seed):
        buffer = MockBuffer()
        selector = Selector(buffer, SelectorConfig(), rng=np.random.default_rng(seed))
        ctx = _make_context()
        cands = _make_candidates(K=4, seed=seed)
        return selector.select(ctx, cands).chosen_index

    a, b = run_once(42), run_once(42)
    assert a == b, f"non-deterministic cold-start RNG: {a} vs {b}"
    print(f"[OK] cold_start_rng_deterministic (chose {a})")


def test_score_formula_consistency():
    """Verify score == ig * ac for every candidate report."""
    buffer = MockBuffer()
    pre_rng = np.random.default_rng(20)
    for _ in range(3):
        buffer.append(BufferEntry(
            context=_make_context(),
            action_chunk=pre_rng.standard_normal((4, 7)),
        ))
    selector = Selector(buffer, SelectorConfig(), rng=np.random.default_rng(4))
    ctx = _make_context()
    cands = _make_candidates(K=4)
    sel = selector.select(ctx, cands)
    for r in sel.reports:
        assert abs(r.score - r.ig * r.ac) < 1e-9, \
            f"score mismatch: {r.score} vs {r.ig * r.ac}"

    # add_to_buffer round-trip
    n_before = len(buffer.entries)
    selector.add_to_buffer(ctx, sel)
    assert len(buffer.entries) == n_before + 1
    last = buffer.entries[-1]
    assert last.action_chunk.shape == (4, 7)
    print("[OK] score_formula_consistency")


if __name__ == "__main__":
    test_cold_start()
    test_warm_start()
    test_report_shape_and_finite()
    test_cold_start_rng_deterministic()
    test_score_formula_consistency()
    print("\nAll smoke tests passed.")
