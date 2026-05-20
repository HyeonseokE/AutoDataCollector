"""Method3 Phase2 — MI-based candidate selection unit tests.

문서 final_method3_spec §7-14 의 각 단계를 격리 검증한다.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from method3.phase2_mi_selection.action_coverage import (
    action_coverage_gain,
    action_novelty,
    top_quantile_mean,
)
from method3.phase2_mi_selection.action_descriptor import (
    dct_action_descriptor,
    dct_energy_optimal_k,
    dct_time,
)
from method3.phase2_mi_selection.conditional_ambiguity import (
    conditional_ambiguity,
    covered_windows,
)
from method3.phase2_mi_selection.mi_selector import (
    Phase2Candidate,
    Phase2MIConfig,
    Phase2MISelector,
)
from method3.phase2_mi_selection.neighbor_search import (
    knn_mean_distance,
    mean_nn_distance,
    min_distance,
)
from method3.phase2_mi_selection.radius import state_neighborhood_radius
from method3.phase2_mi_selection.vector_db import SkillVectorDB, VectorDBEntry


# ─────────────────────────────────────────────────────────────
# §4.2  DCT action descriptor
# ─────────────────────────────────────────────────────────────
class TestDCTActionDescriptor:
    def test_dct_time_shape(self):
        a = np.random.default_rng(0).normal(0, 1, (50, 6))
        assert dct_time(a).shape == (50, 6)

    def test_dct_rejects_non_2d(self):
        with pytest.raises(ValueError):
            dct_time(np.zeros((4, 5, 6)))

    def test_constant_signal_only_dc_component(self):
        # 상수 action chunk → DC(row 0) 만 nonzero, 나머지 ≈ 0.
        a = np.full((40, 6), 0.7)
        c = dct_time(a)
        assert np.all(np.abs(c[0]) > 1.0)
        np.testing.assert_allclose(c[1:], 0.0, atol=1e-9)

    def test_descriptor_dim_is_k_times_action_dim(self):
        a = np.random.default_rng(1).normal(0, 1, (50, 6))
        z = dct_action_descriptor(a, n_coeffs=3)
        assert z.shape == (3 * 6,)

    def test_descriptor_rejects_k_out_of_range(self):
        a = np.zeros((10, 6))
        with pytest.raises(ValueError):
            dct_action_descriptor(a, n_coeffs=0)
        with pytest.raises(ValueError):
            dct_action_descriptor(a, n_coeffs=11)

    def test_energy_optimal_k_constant_signal_is_one(self):
        # 상수 신호는 DC 만으로 energy 100% → 어떤 eta 든 K=1.
        a = np.full((30, 6), 0.5)
        assert dct_energy_optimal_k(a, eta=0.99) == 1

    def test_energy_optimal_k_monotone_in_eta(self):
        a = np.random.default_rng(2).normal(0, 1, (50, 6))
        k_low = dct_energy_optimal_k(a, eta=0.5)
        k_high = dct_energy_optimal_k(a, eta=0.95)
        assert 1 <= k_low <= k_high <= 50


# ─────────────────────────────────────────────────────────────
# §8/§9  neighbor-search primitives
# ─────────────────────────────────────────────────────────────
class TestNeighborSearch:
    def test_knn_mean_distance_picks_k_nearest(self):
        keys = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        assert knn_mean_distance(np.zeros(2), keys, k=1) == pytest.approx(0.0)
        assert knn_mean_distance(np.zeros(2), keys, k=2) == pytest.approx(2.5)

    def test_knn_clamps_k(self):
        knn_mean_distance(np.zeros(2), np.array([[1.0, 0.0]]), k=9)  # 예외 없음

    def test_knn_rejects_empty(self):
        with pytest.raises(ValueError):
            knn_mean_distance(np.zeros(2), np.empty((0, 2)), k=1)

    def test_mean_nn_distance_uniform_spacing(self):
        keys = np.array([[0.0], [1.0], [2.0]])
        assert mean_nn_distance(keys) == pytest.approx(1.0)

    def test_mean_nn_distance_rejects_under_two(self):
        with pytest.raises(ValueError):
            mean_nn_distance(np.zeros((1, 3)))

    def test_min_distance(self):
        keys = np.array([[0.0, 0.0], [10.0, 0.0]])
        assert min_distance(np.array([2.0, 0.0]), keys) == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────
# §3/§7.2  skill-wise vector DB
# ─────────────────────────────────────────────────────────────
class TestSkillVectorDB:
    def _entry(self, skill, e, z):
        return VectorDBEntry(skill_id=skill, state_key=np.asarray(e, float),
                             action_descriptor=np.asarray(z, float))

    def test_append_and_query_partitioned_by_skill(self):
        db = SkillVectorDB()
        db.append(self._entry("reach", [1, 2], [0, 0, 0]))
        db.append(self._entry("reach", [3, 4], [1, 1, 1]))
        db.append(self._entry("grasp", [5, 6], [2, 2, 2]))
        assert db.size("reach") == 2
        assert db.size("grasp") == 1
        assert db.total_size() == 3
        assert set(db.skill_ids()) == {"reach", "grasp"}

    def test_state_keys_and_action_descriptors_matrices(self):
        db = SkillVectorDB()
        db.append(self._entry("reach", [1, 2], [0, 0, 0]))
        db.append(self._entry("reach", [3, 4], [1, 1, 1]))
        assert db.state_keys("reach").shape == (2, 2)
        assert db.action_descriptors("reach").shape == (2, 3)

    def test_empty_skill_returns_empty(self):
        db = SkillVectorDB()
        assert db.state_keys("never").shape == (0, 0)
        assert db.query_skill("never") == []


# ─────────────────────────────────────────────────────────────
# §16  state-neighborhood radius ρ_m
# ─────────────────────────────────────────────────────────────
class TestRadius:
    def test_radius_is_quantile_of_knn_distances(self):
        # 1축 등간격 0..9 → 각 point 의 1-NN 거리 = 1 → 모든 r_i=1 → ρ=1.
        keys = np.arange(10, dtype=float).reshape(10, 1)
        assert state_neighborhood_radius(keys, k=1, quantile=0.7) == pytest.approx(1.0)

    def test_radius_rejects_under_two(self):
        with pytest.raises(ValueError):
            state_neighborhood_radius(np.zeros((1, 3)), k=1)

    def test_radius_grows_with_quantile(self):
        keys = np.random.default_rng(0).normal(0, 1, (50, 4))
        lo = state_neighborhood_radius(keys, k=3, quantile=0.3)
        hi = state_neighborhood_radius(keys, k=3, quantile=0.9)
        assert hi >= lo


# ─────────────────────────────────────────────────────────────
# §8  action coverage gain ΔH_A
# ─────────────────────────────────────────────────────────────
class TestActionCoverage:
    def test_action_novelty_zero_when_inside_buffer(self):
        buf = np.array([[0.0, 0.0], [10.0, 10.0]])
        # query 가 buffer point 와 정확히 일치 → d_knn=0 → novelty 0.
        assert action_novelty(np.zeros(2), buf, nn_scale=14.14, k=1) == 0.0

    def test_action_novelty_positive_when_far(self):
        buf = np.array([[0.0, 0.0], [1.0, 1.0]])
        scale = mean_nn_distance(buf)
        n = action_novelty(np.array([100.0, 100.0]), buf, scale, k=1)
        assert n > 0.0

    def test_top_quantile_mean(self):
        assert top_quantile_mean([1, 2, 3, 4], q=0.5) == pytest.approx(3.5)
        assert top_quantile_mean([1, 2, 3, 4], q=0.25) == pytest.approx(4.0)
        assert top_quantile_mean([5.0], q=1.0) == pytest.approx(5.0)

    def test_top_quantile_mean_rejects_bad_input(self):
        with pytest.raises(ValueError):
            top_quantile_mean([], q=0.5)
        with pytest.raises(ValueError):
            top_quantile_mean([1.0], q=0.0)

    def test_coverage_gain_higher_for_novel_candidate(self):
        rng = np.random.default_rng(0)
        buf = rng.normal(0.0, 0.05, (30, 6))           # 원점 부근 밀집 buffer
        near = rng.normal(0.0, 0.05, (5, 6))           # buffer 와 유사
        far = rng.normal(5.0, 0.05, (5, 6))            # buffer 에서 멀리
        g_near = action_coverage_gain(near, buf, k=3, q=0.5)
        g_far = action_coverage_gain(far, buf, k=3, q=0.5)
        assert g_far > g_near


# ─────────────────────────────────────────────────────────────
# §9  covered-state + conditional ambiguity ΔH_A|S
# ─────────────────────────────────────────────────────────────
class TestConditionalAmbiguity:
    def test_covered_windows_threshold(self):
        db = np.zeros((5, 2))                          # 원점에 5개
        cand = np.array([[0.0, 0.0], [100.0, 0.0]])    # w0 가깝다, w1 멀다
        cov = covered_windows(cand, db, radius=1.0, k_min=3)
        assert [c[0] for c in cov] == [0]              # w0 만 covered
        assert cov[0][1].shape[0] == 5

    def test_covered_windows_empty_db(self):
        cov = covered_windows(np.zeros((3, 2)), np.empty((0, 2)), radius=1.0)
        assert cov == []

    def test_no_covered_window_reports_zero(self):
        db_keys = np.zeros((5, 2))
        db_z = np.random.default_rng(0).normal(0, 1, (5, 4))
        cand_keys = np.full((3, 2), 100.0)             # buffer 에서 전부 멀다
        cand_z = np.zeros((3, 4))
        rep = conditional_ambiguity(cand_keys, cand_z, db_keys, db_z,
                                    radius=1.0, k_min=3)
        assert rep.n_covered == 0
        assert rep.delta_h_a_given_s == 0.0
        assert rep.covered_ratio == 0.0

    def test_ambiguity_zero_when_action_inside_support(self):
        db_keys = np.zeros((5, 2))                     # 같은 state 영역
        db_z = np.zeros((5, 4))                        # local action support = 원점
        cand_keys = np.zeros((1, 2))                   # covered
        cand_z = np.zeros((1, 4))                      # support 안 (일치)
        rep = conditional_ambiguity(cand_keys, cand_z, db_keys, db_z,
                                    radius=1.0, k_min=3)
        assert rep.n_covered == 1
        assert rep.delta_h_a_given_s == pytest.approx(0.0)

    def test_ambiguity_positive_when_action_outside_support(self):
        rng = np.random.default_rng(1)
        db_keys = np.zeros((6, 2))
        db_z = rng.normal(0.0, 0.05, (6, 4))           # 좁은 local support
        cand_keys = np.zeros((1, 2))                   # covered
        cand_z = np.full((1, 4), 10.0)                 # support 바깥
        rep = conditional_ambiguity(cand_keys, cand_z, db_keys, db_z,
                                    radius=1.0, k_min=3)
        assert rep.n_covered == 1
        assert rep.delta_h_a_given_s > 0.0

    def test_max_agg_ge_mean_agg(self):
        rng = np.random.default_rng(2)
        db_keys = np.zeros((6, 2))
        db_z = rng.normal(0.0, 0.05, (6, 4))
        cand_keys = np.zeros((3, 2))
        cand_z = rng.normal(2.0, 0.5, (3, 4))
        mean_rep = conditional_ambiguity(cand_keys, cand_z, db_keys, db_z,
                                         radius=1.0, k_min=3, agg="mean")
        max_rep = conditional_ambiguity(cand_keys, cand_z, db_keys, db_z,
                                        radius=1.0, k_min=3, agg="max")
        assert max_rep.delta_h_a_given_s >= mean_rep.delta_h_a_given_s

    def test_rejects_bad_agg(self):
        with pytest.raises(ValueError):
            conditional_ambiguity(np.zeros((1, 2)), np.zeros((1, 4)),
                                  np.zeros((3, 2)), np.zeros((3, 4)),
                                  radius=1.0, agg="median")


# ─────────────────────────────────────────────────────────────
# §11-14  Phase2 MI selector
# ─────────────────────────────────────────────────────────────
def _candidate(skill_id, rng, key_center=0.0, action_loc=0.0,
               T=4, De=8, H=12, A=6):
    """합성 후보 — state key 는 key_center 부근, action chunk 는 action_loc 중심."""
    state_keys = key_center + rng.normal(0.0, 0.05, (T, De))
    action_chunks = rng.normal(action_loc, 0.3, (T, H, A))
    return Phase2Candidate(skill_id, state_keys, action_chunks)


def _seed_db(selector, skill_id, rng, n_seed=6):
    """selector.accept_to_buffer 로 buffer 를 seed — 차원 일관성 보장."""
    for _ in range(n_seed):
        selector.accept_to_buffer(_candidate(skill_id, rng))


class TestPhase2MISelector:
    def _cfg(self, **kw):
        base = dict(dct_coeffs=3, k_nn_a=3, k_min=3, radius_k=3)
        base.update(kw)
        return Phase2MIConfig(**base)

    def test_action_descriptors_shape(self):
        sel = Phase2MISelector(SkillVectorDB(), self._cfg())
        cand = _candidate("reach", np.random.default_rng(0))
        z = sel.action_descriptors(cand)
        assert z.shape == (4, 3 * 6)               # (T, K·action_dim)

    def test_cold_start_empty_buffer(self):
        # buffer 가 비면 ΔH_A·ΔH_A|S 계산 불가 → under-covered, accept 안 함.
        sel = Phase2MISelector(SkillVectorDB(), self._cfg())
        out = sel.select([_candidate("reach", np.random.default_rng(0))])
        assert out.accepted is False
        assert out.reports[0].under_covered is True
        assert out.reports[0].q2 == 0.0

    def test_accept_to_buffer_adds_window_entries(self):
        db = SkillVectorDB()
        sel = Phase2MISelector(db, self._cfg())
        cand = _candidate("reach", np.random.default_rng(0), T=5)
        n = sel.accept_to_buffer(cand, ref={"episode_id": "ep_1"})
        assert n == 5
        assert db.size("reach") == 5               # window 당 1 entry
        assert db.query_skill("reach")[0].meta["phase"] == "phase2"

    def test_select_returns_report_per_candidate(self):
        rng = np.random.default_rng(0)
        db = SkillVectorDB()
        sel = Phase2MISelector(db, self._cfg())
        _seed_db(sel, "reach", rng)
        cands = [_candidate("reach", rng) for _ in range(4)]
        out = sel.select(cands)
        assert len(out.reports) == 4
        # §12 — 배치 정규화 Q̃2 의 평균은 0.
        assert float(np.mean([r.q2_norm for r in out.reports])) == pytest.approx(
            0.0, abs=1e-9)
        assert 0 <= out.chosen_index < 4

    def test_select_picks_argmax_q2(self):
        rng = np.random.default_rng(1)
        db = SkillVectorDB()
        sel = Phase2MISelector(db, self._cfg())
        _seed_db(sel, "reach", rng)
        cands = [_candidate("reach", rng) for _ in range(5)]
        out = sel.select(cands)
        q2 = [r.q2 for r in out.reports]
        assert out.chosen_index == int(np.argmax(q2))

    def test_novel_action_candidate_scores_higher_coverage(self):
        # action chunk 가 seed buffer 와 멀수록 ΔH_A 가 크다 (§8).
        rng = np.random.default_rng(2)
        db = SkillVectorDB()
        sel = Phase2MISelector(db, self._cfg())
        _seed_db(sel, "reach", rng)                       # buffer ≈ action_loc 0
        near = _candidate("reach", rng, action_loc=0.0)
        far = _candidate("reach", rng, action_loc=8.0)
        r_near = sel.score_one(0, near)
        r_far = sel.score_one(1, far)
        assert r_far.delta_h_a > r_near.delta_h_a

    def test_under_covered_candidate_not_accepted(self):
        # state key 가 buffer 영역에서 멀면 covered window 0 → accept 안 함.
        rng = np.random.default_rng(3)
        db = SkillVectorDB()
        sel = Phase2MISelector(db, self._cfg(min_covered_windows=1))
        _seed_db(sel, "reach", rng)                       # buffer state ≈ 0
        out = sel.select([_candidate("reach", rng, key_center=50.0)])
        assert out.reports[0].under_covered is True
        assert out.accepted is False

    def test_select_rejects_empty(self):
        sel = Phase2MISelector(SkillVectorDB(), self._cfg())
        with pytest.raises(ValueError):
            sel.select([])

    def test_q2_combines_coverage_and_ambiguity(self):
        # Q2 = β·ΔH_A − λ·ΔH_A|S — 부호 규약 검증.
        rng = np.random.default_rng(4)
        db = SkillVectorDB()
        sel = Phase2MISelector(db, self._cfg(beta=2.0, lambda_=3.0))
        _seed_db(sel, "reach", rng)
        r = sel.score_one(0, _candidate("reach", rng))
        expected = 2.0 * r.delta_h_a - 3.0 * r.delta_h_a_given_s
        assert r.q2 == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────
# §3  vector DB 영속화
# ─────────────────────────────────────────────────────────────
class TestVectorDBPersistence:
    def _entry(self, skill, e, z, ref, meta):
        return VectorDBEntry(skill_id=skill, state_key=np.asarray(e, float),
                             action_descriptor=np.asarray(z, float),
                             ref=ref, meta=meta)

    def test_save_load_round_trip(self, tmp_path):
        path = tmp_path / "vector_db.npz"
        db = SkillVectorDB()
        db.append(self._entry("reach", [0.1, 0.2], [1, 2, 3],
                               ref={"episode_id": "ep_1", "entry_index": 4},
                               meta={"phase": "phase2", "window": 0}))
        db.append(self._entry("reach", [0.3, 0.4], [4, 5, 6],
                               ref={"episode_id": "ep_2"}, meta={"phase": "phase2"}))
        db.append(self._entry("grasp", [0.5, 0.6], [7, 8, 9],
                               ref={}, meta={"phase": "phase1"}))
        db.save(path)
        assert path.exists()

        restored = SkillVectorDB()
        restored.load(path)
        np.testing.assert_array_equal(restored.state_keys("reach"),
                                      db.state_keys("reach"))
        np.testing.assert_array_equal(restored.action_descriptors("reach"),
                                      db.action_descriptors("reach"))
        assert restored.size("grasp") == 1

    def test_ref_and_meta_dicts_round_trip(self, tmp_path):
        path = tmp_path / "vdb.npz"
        db = SkillVectorDB()
        db.append(self._entry("reach", [1.0, 1.0], [0.0, 0.0],
                               ref={"episode_id": "ep_42", "entry_index": 7},
                               meta={"phase": "phase2", "window": 3, "score": 0.5}))
        db.save(path)
        restored = SkillVectorDB()
        restored.load(path)
        e = restored.query_skill("reach")[0]
        assert e.ref == {"episode_id": "ep_42", "entry_index": 7}
        assert e.meta == {"phase": "phase2", "window": 3, "score": 0.5}

    def test_npz_suffix_appended(self, tmp_path):
        db = SkillVectorDB()
        db.append(self._entry("reach", [0.0], [0.0], ref={}, meta={}))
        db.save(tmp_path / "vdb")                       # 확장자 없음
        assert (tmp_path / "vdb.npz").exists()

    def test_load_missing_file_is_noop(self, tmp_path):
        db = SkillVectorDB()
        db.load(tmp_path / "does_not_exist.npz")        # 예외 없이 no-op
        assert db.total_size() == 0

    def test_empty_db_save_writes_nothing(self, tmp_path):
        path = tmp_path / "empty.npz"
        SkillVectorDB().save(path)                      # entry 0개 → no-op
        assert not path.exists()
