"""Method3 Phase1 — buffer-aware subgoal scoring unit tests.

문서 final_method3_spec §4-5 의 각 단계를 격리 검증한다.
파이프라인(skills_lerobot) 의존 없이 순수 함수/클래스만 테스트한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from method3.phase1_state_seeding.canonical_preview import interp_plan, last_segment
from method3.phase1_state_seeding.subgoal_buffer import (
    SubgoalBuffer,
    SubgoalBufferEntry,
    knn_mean_distance,
    mean_nn_distance,
)
from method3.phase1_state_seeding.subgoal_candidates import sample_subgoal_candidates
from method3.phase1_state_seeding.subgoal_selector import (
    Phase1SubgoalConfig,
    Phase1SubgoalSelector,
)
from method3.phase1_state_seeding.subgoal_validity import (
    ReachabilityConfig,
    make_subgoal_validity_fn,
)
from method3.phase1_state_seeding.terminal_descriptor import (
    DESCRIPTOR_DIM,
    state_descriptor,
)

# preview 시작 end-effector 위치 — select_subgoal/_terminal_region 의
# current_ee·start_ee 인자 (대부분 테스트에서 고정값으로 충분).
START_EE = np.array([0.30, 0.0, 0.30])


def _entry(skill_id, key, subgoal=None) -> SubgoalBufferEntry:
    """terminal_region_key 만 정해 §5.2 entry 를 만든다 (합성 buffer 용).

    end_state_keys 는 key 1개짜리 set(평균=key)으로 둔다.
    """
    key = np.asarray(key, dtype=np.float64)
    return SubgoalBufferEntry(
        skill_id=str(skill_id),
        subgoal=np.zeros(3) if subgoal is None else np.asarray(subgoal, float),
        terminal_region_key=key,
        end_state_keys=key.reshape(1, -1),
    )


def _commit_one(sel, skill_id, start_ee, goal):
    """한 에피소드(TRUE 판정)처럼 stage 후 flush — buffer 에 1개 commit."""
    sel.stage_executed(skill_id, start_ee, goal)
    sel.flush_episode()


# ─────────────────────────────────────────────────────────────
# §4.2  candidate generation
# ─────────────────────────────────────────────────────────────
class TestSampleSubgoalCandidates:
    def test_returns_requested_count_and_shape(self):
        cands = sample_subgoal_candidates(
            np.array([0.3, 0.0, 0.2]), 16, 0.05, 2.0, np.random.default_rng(0),
        )
        assert cands.shape == (16, 3)

    def test_include_nominal_places_nominal_at_index_0(self):
        nominal = np.array([0.3, -0.1, 0.25])
        cands = sample_subgoal_candidates(
            nominal, 8, 0.05, 2.0, np.random.default_rng(0), include_nominal=True,
        )
        np.testing.assert_allclose(cands[0], nominal)

    def test_offsets_stay_within_clip_radius(self):
        nominal = np.array([0.3, 0.0, 0.2])
        cands = sample_subgoal_candidates(nominal, 200, 0.05, 2.0,
                                          np.random.default_rng(1))
        offsets = np.linalg.norm(cands - nominal, axis=1)
        assert np.all(offsets <= 0.05 * 2.0 + 1e-9)

    def test_deterministic_for_same_seed(self):
        nominal = np.array([0.3, 0.0, 0.2])
        a = sample_subgoal_candidates(nominal, 10, 0.05, 2.0, np.random.default_rng(7))
        b = sample_subgoal_candidates(nominal, 10, 0.05, 2.0, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_rejects_non_positive_count(self):
        with pytest.raises(ValueError):
            sample_subgoal_candidates(np.zeros(3), 0, 0.05, 2.0, np.random.default_rng(0))

    def test_uniform_ball_offsets_within_radius(self):
        nominal = np.array([0.3, 0.0, 0.2])
        cands = sample_subgoal_candidates(nominal, 300, 0.05, 2.0,
                                          np.random.default_rng(2), dist="uniform_ball")
        offsets = np.linalg.norm(cands - nominal, axis=1)
        assert np.all(offsets <= 0.05 * 2.0 + 1e-9)

    def test_uniform_ball_spreads_wider_than_gaussian(self):
        # Phase1 의도 — uniform_ball 후보가 gaussian 보다 평균적으로 더 멀리 퍼진다.
        nominal = np.array([0.3, 0.0, 0.2])
        ub = sample_subgoal_candidates(nominal, 500, 0.05, 2.0,
                                       np.random.default_rng(0),
                                       include_nominal=False, dist="uniform_ball")
        gs = sample_subgoal_candidates(nominal, 500, 0.05, 2.0,
                                       np.random.default_rng(0),
                                       include_nominal=False, dist="gaussian")
        ub_r = np.linalg.norm(ub - nominal, axis=1).mean()
        gs_r = np.linalg.norm(gs - nominal, axis=1).mean()
        assert ub_r > gs_r

    def test_gaussian_dist_still_supported(self):
        cands = sample_subgoal_candidates(np.zeros(3), 50, 0.05, 2.0,
                                          np.random.default_rng(0), dist="gaussian")
        assert cands.shape == (50, 3)

    def test_rejects_unknown_dist(self):
        with pytest.raises(ValueError):
            sample_subgoal_candidates(np.zeros(3), 8, 0.05, 2.0,
                                      np.random.default_rng(0), dist="poisson")

    def test_default_dist_is_uniform_ball(self):
        # 기본 분포가 uniform_ball — dist 명시 없이도 공 안에 골고루 분포.
        cands = sample_subgoal_candidates(np.zeros(3), 400, 0.05, 2.0,
                                          np.random.default_rng(3))
        offsets = np.linalg.norm(cands, axis=1)
        assert offsets[1:].mean() > 0.55 * (0.05 * 2.0)   # uniform-by-volume ≈ 0.75R

    def test_valid_fn_rejection_in_generation(self):
        # valid_fn (z>=0.15) → 모든 non-nominal 후보가 제약을 만족 (문서 §4.2).
        nominal = np.array([0.3, 0.0, 0.2])
        v = make_subgoal_validity_fn(ReachabilityConfig(z_min=0.15))
        cands = sample_subgoal_candidates(nominal, 40, 0.05, 2.0,
                                          np.random.default_rng(0),
                                          valid_fn=v, max_resample=500)
        assert np.all(cands[1:, 2] >= 0.15)   # nominal(idx 0) 제외 전부 z>=0.15

    def test_valid_fn_exhaustion_does_not_hang(self):
        # valid_fn 이 전부 거부 → max_resample 후 종료, 배열 shape 유지.
        cands = sample_subgoal_candidates(np.zeros(3), 8, 0.05, 2.0,
                                          np.random.default_rng(0),
                                          valid_fn=lambda p: False, max_resample=10)
        assert cands.shape == (8, 3)

    # --- hemisphere dist (§4.2 — robot 원점 쪽 반구) ---
    def test_hemisphere_offsets_within_radius(self):
        nominal = np.array([0.3, 0.0, 0.2])
        cands = sample_subgoal_candidates(nominal, 300, 0.05, 2.0,
                                          np.random.default_rng(4),
                                          dist="hemisphere")
        offsets = np.linalg.norm(cands - nominal, axis=1)
        assert np.all(offsets <= 0.05 * 2.0 + 1e-9)

    def test_hemisphere_samples_only_toward_origin(self):
        # 모든 non-nominal offset 이 d=(origin-nominal) 과 같은 반구 (offset·d >= 0).
        nominal = np.array([0.3, -0.1, 0.25])
        cands = sample_subgoal_candidates(nominal, 400, 0.05, 2.0,
                                          np.random.default_rng(5),
                                          include_nominal=False, dist="hemisphere")
        d = np.zeros(3) - nominal                       # 기본 robot_origin (0,0,0)
        dots = (cands - nominal) @ d
        assert np.all(dots >= -1e-12)

    def test_hemisphere_respects_custom_robot_origin(self):
        # robot_origin 을 +x 쪽에 두면 offset 의 x 성분이 항상 >= 0.
        nominal = np.array([0.3, 0.0, 0.2])
        origin = np.array([1.0, 0.0, 0.2])              # d = +x
        cands = sample_subgoal_candidates(nominal, 300, 0.05, 2.0,
                                          np.random.default_rng(6),
                                          include_nominal=False, dist="hemisphere",
                                          robot_origin=origin)
        assert np.all(cands[:, 0] - nominal[0] >= -1e-12)

    def test_hemisphere_degenerate_origin_falls_back(self):
        # nominal == robot_origin → 방향 불능 → uniform_ball 폴백, 크래시 없음.
        cands = sample_subgoal_candidates(np.zeros(3), 8, 0.05, 2.0,
                                          np.random.default_rng(0),
                                          dist="hemisphere", robot_origin=np.zeros(3))
        assert cands.shape == (8, 3)

    def test_selector_accepts_hemisphere_config(self):
        # Phase1SubgoalConfig 가 hemisphere dist + robot_origin 을 받아
        # select_subgoal 이 크래시 없이 후보를 고른다 (cold-start 경로).
        cfg = Phase1SubgoalConfig(candidate_dist="hemisphere",
                                  robot_origin=(0.0, 0.0, 0.0), n_candidates=16)
        sel = Phase1SubgoalSelector(SubgoalBuffer(), cfg)
        out = sel.select_subgoal(START_EE, np.array([0.3, 0.0, 0.2]),
                                 "reach", np.random.default_rng(0))
        assert out.chosen_goal.shape == (3,)


# ─────────────────────────────────────────────────────────────
# §4.2  reachable / safe — subgoal 위치 제약
# ─────────────────────────────────────────────────────────────
class TestReachability:
    def test_z_min_floor(self):
        v = make_subgoal_validity_fn(ReachabilityConfig(z_min=0.05))
        assert v([0.3, 0.0, 0.10])
        assert not v([0.3, 0.0, 0.02])

    def test_z_max_ceiling(self):
        v = make_subgoal_validity_fn(ReachabilityConfig(z_max=0.20))
        assert v([0.3, 0.0, 0.10])
        assert not v([0.3, 0.0, 0.30])

    def test_xy_bounds(self):
        v = make_subgoal_validity_fn(ReachabilityConfig(
            x_bounds=(0.2, 0.4), y_bounds=(-0.1, 0.1)))
        assert v([0.3, 0.0, 0.1])
        assert not v([0.5, 0.0, 0.1])    # x 범위 밖
        assert not v([0.3, 0.5, 0.1])    # y 범위 밖

    def test_composes_is_reachable(self):
        v = make_subgoal_validity_fn(
            ReachabilityConfig(z_min=0.0), is_reachable=lambda p: p[0] > 0.0,
        )
        assert v([0.3, 0.0, 0.1])
        assert not v([-0.3, 0.0, 0.1])   # is_reachable False
        assert not v([0.3, 0.0, -0.1])   # z_min False

    def test_permissive_default_allows_all(self):
        v = make_subgoal_validity_fn(ReachabilityConfig())
        assert v([100.0, -50.0, 9.0])

    def test_selector_respects_reachability_z_min(self):
        # 선택기에 z_min 제약을 주면 선택된 subgoal 이 이를 만족해야 한다.
        cfg = Phase1SubgoalConfig(
            n_candidates=20, min_buffer_size=4, k_nn=2,
            reachability=ReachabilityConfig(z_min=0.18),
        )
        sel = Phase1SubgoalSelector(SubgoalBuffer(), cfg)
        out = sel.select_subgoal(START_EE, np.array([0.3, 0.0, 0.2]), "move",
                                 np.random.default_rng(0))
        assert out.chosen_goal[2] >= 0.18 - 1e-9


# ─────────────────────────────────────────────────────────────
# §5.3  geometric state descriptor  ĥ = [x_EE, x_EE - g']  (6-dim)
# ─────────────────────────────────────────────────────────────
class TestStateDescriptor:
    def test_dim_is_six(self):
        assert DESCRIPTOR_DIM == 6

    def test_concat_ee_and_ee_minus_goal(self):
        ee = np.array([0.3, 0.1, 0.2])
        goal = np.array([0.3, 0.0, 0.0])
        d = state_descriptor(ee, goal)
        assert d.shape == (DESCRIPTOR_DIM,)
        np.testing.assert_allclose(d[:3], ee)
        np.testing.assert_allclose(d[3:], ee - goal)

    def test_is_float64(self):
        d = state_descriptor([0.3, 0.1, 0.2], [0.0, 0.0, 0.0])
        assert d.dtype == np.float64


# ─────────────────────────────────────────────────────────────
# §5.4  canonical preview trajectory (InterpPlan + T_end)
# ─────────────────────────────────────────────────────────────
class TestPreview:
    def test_interp_plan_shape_and_endpoints(self):
        cur = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 2.0, 3.0])
        plan = interp_plan(cur, goal, 10)
        assert plan.shape == (10, 3)
        np.testing.assert_allclose(plan[0], cur)
        np.testing.assert_allclose(plan[-1], goal)

    def test_interp_plan_is_straight_line(self):
        cur = np.zeros(3)
        goal = np.array([1.0, 0.0, 0.0])
        plan = interp_plan(cur, goal, 5)
        np.testing.assert_allclose(plan[:, 0], np.linspace(0.0, 1.0, 5))

    def test_interp_plan_rejects_under_two_points(self):
        with pytest.raises(ValueError):
            interp_plan(np.zeros(3), np.ones(3), 1)

    def test_last_segment_returns_tail_fraction(self):
        traj = np.arange(30, dtype=float).reshape(10, 3)
        seg = last_segment(traj, 0.2)
        assert seg.shape == (2, 3)                 # int(0.8*10)=8 → traj[8:]
        np.testing.assert_array_equal(seg, traj[-2:])

    def test_last_segment_guarantees_one_point(self):
        traj = np.arange(15, dtype=float).reshape(5, 3)
        seg = last_segment(traj, 0.01)             # 매우 작은 비율도 최소 1개
        assert seg.shape[0] >= 1

    def test_last_segment_rejects_invalid_fraction(self):
        traj = np.arange(15, dtype=float).reshape(5, 3)
        with pytest.raises(ValueError):
            last_segment(traj, 0.0)


# ─────────────────────────────────────────────────────────────
# §5.4  buffer distances
# ─────────────────────────────────────────────────────────────
class TestBufferDistances:
    def test_knn_mean_distance_picks_k_nearest(self):
        keys = np.array([[0.0] * 3, [1.0] * 3, [2.0] * 3, [10.0] * 3])
        query = np.zeros(3)
        assert knn_mean_distance(query, keys, k=1) == pytest.approx(0.0)
        assert knn_mean_distance(query, keys, k=2) == pytest.approx(np.sqrt(3) / 2)

    def test_knn_clamps_k_to_buffer_size(self):
        keys = np.array([[0.0] * 3, [1.0] * 3])
        knn_mean_distance(np.zeros(3), keys, k=10)  # k>N → clamp, 예외 없음

    def test_knn_rejects_empty_buffer(self):
        with pytest.raises(ValueError):
            knn_mean_distance(np.zeros(3), np.empty((0, 3)), k=1)

    def test_mean_nn_distance_uniform_spacing(self):
        keys = np.zeros((3, 3))
        keys[:, 0] = [0.0, 1.0, 2.0]      # 1축 0,1,2 → 각 NN 거리 1 → 평균 1
        assert mean_nn_distance(keys) == pytest.approx(1.0)

    def test_mean_nn_distance_rejects_under_two(self):
        with pytest.raises(ValueError):
            mean_nn_distance(np.zeros((1, 3)))

    def test_mean_nn_distance_chunked_matches_naive(self):
        keys = np.random.default_rng(42).normal(0.0, 0.1, size=(300, 3))
        diff = keys[:, None, :] - keys[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        assert mean_nn_distance(keys) == pytest.approx(float(d.min(axis=1).mean()),
                                                       rel=1e-9)

    def test_mean_nn_distance_chunked_multiple_chunks(self):
        # chunk 경계를 여러 번 넘는 크기에서도 naive 와 동일 (chunking 회귀).
        keys = np.random.default_rng(0).normal(0.0, 0.1, size=(2500, 3))
        diff = keys[:, None, :] - keys[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        assert mean_nn_distance(keys) == pytest.approx(float(d.min(axis=1).mean()),
                                                       rel=1e-9)


# ─────────────────────────────────────────────────────────────
# §5  SubgoalBuffer — §5.2 entry 적재·영속화
# ─────────────────────────────────────────────────────────────
class TestSubgoalBuffer:
    def test_append_query_size(self):
        buf = SubgoalBuffer()
        assert buf.size("move") == 0
        buf.append(_entry("move", np.arange(DESCRIPTOR_DIM, dtype=float)))
        buf.append(_entry("move", np.arange(DESCRIPTOR_DIM, dtype=float) + 1))
        assert buf.size("move") == 2
        assert buf.query_skill("move").shape == (2, DESCRIPTOR_DIM)

    def test_query_unknown_skill_returns_empty(self):
        assert SubgoalBuffer().query_skill("never_seen").shape == (0, DESCRIPTOR_DIM)

    def test_append_rejects_wrong_key_dim(self):
        with pytest.raises(ValueError):
            SubgoalBuffer().append(_entry("move", np.zeros(3)))   # 3 != DESCRIPTOR_DIM

    def test_append_rejects_wrong_end_keys_shape(self):
        bad = SubgoalBufferEntry(
            skill_id="move", subgoal=np.zeros(3),
            terminal_region_key=np.zeros(DESCRIPTOR_DIM),
            end_state_keys=np.zeros((4, 3)),          # D=3 != DESCRIPTOR_DIM
        )
        with pytest.raises(ValueError):
            SubgoalBuffer().append(bad)

    def test_total_size_and_skill_ids(self):
        buf = SubgoalBuffer()
        buf.append(_entry("move", np.zeros(DESCRIPTOR_DIM)))
        buf.append(_entry("move_and_close", np.ones(DESCRIPTOR_DIM)))
        assert buf.total_size() == 2
        assert set(buf.skill_ids()) == {"move", "move_and_close"}

    def test_save_load_round_trip_keys(self, tmp_path):
        path = tmp_path / "subgoal_buffer.npz"
        buf = SubgoalBuffer(buffer_file=path)
        for i in range(5):
            buf.append(_entry("move", np.arange(DESCRIPTOR_DIM, dtype=float) + i))
        for i in range(3):
            buf.append(_entry("move_and_close",
                              np.arange(DESCRIPTOR_DIM, dtype=float) + i * 2))
        buf.save()
        assert path.exists()

        restored = SubgoalBuffer(buffer_file=path)
        restored.load()
        np.testing.assert_array_equal(restored.query_skill("move"),
                                      buf.query_skill("move"))
        np.testing.assert_array_equal(restored.query_skill("move_and_close"),
                                      buf.query_skill("move_and_close"))

    def test_save_load_round_trip_spec_5_2_metadata(self, tmp_path):
        # §5.2 — subgoal / raw pointer(episode_id, start_t, end_t) / end_state_keys
        # 가 영속화·복원된다.
        path = tmp_path / "buf.npz"
        buf = SubgoalBuffer(buffer_file=path)
        buf.append(SubgoalBufferEntry(
            skill_id="reach",
            subgoal=np.array([0.31, -0.02, 0.18]),
            terminal_region_key=np.arange(DESCRIPTOR_DIM, dtype=float),
            end_state_keys=np.arange(3 * DESCRIPTOR_DIM, dtype=float).reshape(3, -1),
            episode_id="ep_0032", start_t=120, end_t=170,
        ))
        buf.save()

        restored = SubgoalBuffer(buffer_file=path)
        restored.load()
        e = restored.entries("reach")[0]
        np.testing.assert_allclose(e.subgoal, [0.31, -0.02, 0.18])
        assert e.episode_id == "ep_0032"
        assert e.start_t == 120 and e.end_t == 170
        assert e.end_state_keys.shape == (3, DESCRIPTOR_DIM)
        assert e.success_flag is True and e.planner_type == "InterpPlan"
        assert e.phase == "phase1"

    def test_buffer_file_appends_npz_suffix(self, tmp_path):
        buf = SubgoalBuffer(buffer_file=tmp_path / "buf")
        assert buf.file_path().suffix == ".npz"
        buf.append(_entry("move", np.zeros(DESCRIPTOR_DIM)))
        buf.save()
        assert (tmp_path / "buf.npz").exists()
        restored = SubgoalBuffer(buffer_file=tmp_path / "buf")
        restored.load()
        assert restored.size("move") == 1

    def test_set_file_binds_path_later(self, tmp_path):
        # 메모리 전용 시작 → set_file 로 경로 바인딩 후 save (세션 디렉터리
        # 가 나중에 정해지는 파이프라인 시나리오).
        buf = SubgoalBuffer()
        assert buf.file_path() is None
        buf.append(_entry("move", np.ones(DESCRIPTOR_DIM)))
        buf.save()  # no-op — 파일 미바인딩
        path = tmp_path / "session_x" / "subgoal_buffer.npz"
        buf.set_file(path)
        buf.save()
        assert path.exists()

    def test_memory_only_buffer_save_is_noop(self):
        buf = SubgoalBuffer()
        buf.append(_entry("move", np.zeros(DESCRIPTOR_DIM)))
        buf.save()  # 예외 없이 no-op
        assert buf.file_path() is None


# ─────────────────────────────────────────────────────────────
#       Phase1SubgoalSelector — end to end
# ─────────────────────────────────────────────────────────────
class TestPhase1SubgoalSelector:
    def _cfg(self, **kw):
        base = dict(n_candidates=8, k_nn=1, min_buffer_size=4)
        base.update(kw)
        return Phase1SubgoalConfig(**base)

    def test_cold_start_falls_back_to_random_candidate(self):
        # buffer 비어있음 → cold_start, 변형 후보(idx != 0) 선택
        sel = Phase1SubgoalSelector(SubgoalBuffer(), self._cfg())
        out = sel.select_subgoal(START_EE, np.array([0.3, 0.0, 0.2]), "move",
                                 np.random.default_rng(0))
        assert out.cold_start is True
        assert out.reports == []
        assert out.chosen_index != 0

    def test_no_reachable_candidate_falls_back_to_nominal(self):
        sel = Phase1SubgoalSelector(SubgoalBuffer(), self._cfg(),
                                    reachable_fn=lambda xyz: False)
        nominal = np.array([0.3, 0.0, 0.2])
        out = sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(0))
        assert out.cold_start is False
        np.testing.assert_allclose(out.chosen_goal, nominal)

    def test_gain_increases_with_distance_from_buffer(self):
        # buffer 가 한 영역에 밀집 → 후보 gain 이 그 영역에서 멀수록 높다 (§5.4).
        nominal = np.array([0.3, 0.0, 0.2])
        cfg = self._cfg(n_candidates=24, k_nn=3, min_buffer_size=4)
        buf = SubgoalBuffer()
        sel = Phase1SubgoalSelector(buf, cfg)
        rng = np.random.default_rng(123)
        for _ in range(20):
            _commit_one(sel, "move", START_EE, nominal + rng.normal(0.0, 0.003, 3))
        out = sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(2))

        assert out.cold_start is False
        dists = np.array([np.linalg.norm(r.goal - nominal) for r in out.reports])
        gains = np.array([r.gain for r in out.reports])
        assert np.corrcoef(dists, gains)[0, 1] > 0.8        # 거리-gain 양의 상관
        assert out.chosen_index == int(np.argmax(gains))    # §5.4 argmax

    def test_deterministic_for_same_seed(self):
        buf = SubgoalBuffer()
        for i in range(10):
            buf.append(_entry("move", np.arange(DESCRIPTOR_DIM, dtype=float) + i * 0.5))
        cfg = self._cfg()
        a = Phase1SubgoalSelector(buf, cfg).select_subgoal(
            START_EE, np.array([0.3, 0.0, 0.2]), "move", np.random.default_rng(5))
        b = Phase1SubgoalSelector(buf, cfg).select_subgoal(
            START_EE, np.array([0.3, 0.0, 0.2]), "move", np.random.default_rng(5))
        np.testing.assert_array_equal(a.chosen_goal, b.chosen_goal)
        assert a.chosen_index == b.chosen_index

    def test_per_call_reachable_fn_overrides_constructor(self):
        # 호출 시 '전부 unreachable' → valid 후보 0개 → nominal 폴백.
        sel = Phase1SubgoalSelector(SubgoalBuffer(), self._cfg())
        nominal = np.array([0.3, 0.0, 0.2])
        out = sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(0),
                                 reachable_fn=lambda xyz: False)
        np.testing.assert_allclose(out.chosen_goal, nominal)

    def test_stage_does_not_grow_buffer_until_flush(self):
        buf = SubgoalBuffer()
        sel = Phase1SubgoalSelector(buf, self._cfg())
        sel.stage_executed("move", START_EE, np.array([0.3, 0.0, 0.2]))
        sel.stage_executed("move", START_EE, np.array([0.3, 0.0, 0.25]))
        assert buf.size("move") == 0          # staged, 아직 buffer 미반영
        sel.flush_episode()                   # episode TRUE
        assert buf.size("move") == 2          # flush → commit

    def test_flush_episode_commits_terminal_descriptor(self):
        buf = SubgoalBuffer()
        sel = Phase1SubgoalSelector(buf, self._cfg())
        goal = np.array([0.3, 0.0, 0.2])
        _commit_one(sel, "move", START_EE, goal)
        assert buf.size("move") == 1
        # §5.5 — buffer key = canonical preview 의 T_end descriptor 평균 h*.
        _, h_star = sel._terminal_region(START_EE, goal)
        np.testing.assert_allclose(buf.query_skill("move")[0], h_star)

    def test_flush_records_spec_5_2_entry_fields(self):
        # §5.2 — flush 가 raw pointer 까지 갖춘 entry 를 적재한다.
        buf = SubgoalBuffer()
        sel = Phase1SubgoalSelector(buf, self._cfg())
        goal = np.array([0.3, 0.0, 0.2])
        sel.stage_executed("move", START_EE, goal,
                           episode_id="ep_0007", start_t=120, end_t=170)
        sel.flush_episode()
        e = buf.entries("move")[0]
        np.testing.assert_allclose(e.subgoal, goal)
        assert e.terminal_region_key.shape == (DESCRIPTOR_DIM,)
        assert e.end_state_keys.ndim == 2
        assert e.end_state_keys.shape[1] == DESCRIPTOR_DIM
        assert e.episode_id == "ep_0007"
        assert e.start_t == 120 and e.end_t == 170
        assert e.success_flag is True
        assert e.planner_type == "InterpPlan"
        assert e.phase == "phase1"

    def test_discard_episode_drops_staged_subgoals(self):
        buf = SubgoalBuffer()
        sel = Phase1SubgoalSelector(buf, self._cfg())
        sel.stage_executed("move", START_EE, np.array([0.3, 0.0, 0.2]))
        sel.discard_episode()
        sel.flush_episode()                   # pending 비었으므로 no-op
        assert buf.size("move") == 0

    def test_flush_episode_persists_to_file(self, tmp_path):
        # TRUE flush 시점에 .npz 로 즉시 영속화 (raw dataset ingest 와 동일 시점).
        path = tmp_path / "subgoal_buffer.npz"
        sel = Phase1SubgoalSelector(SubgoalBuffer(buffer_file=path), self._cfg())
        sel.stage_executed("move", START_EE, np.array([0.3, 0.0, 0.2]))
        sel.flush_episode()
        assert path.exists()

    def test_query_exactly_matching_buffer_does_not_crash(self):
        """RCA 회귀 — query descriptor 가 buffer 의 k개 이상 entry 와 정확히
        일치하면 d_knn=0. §5.4 의 [log(d/(s+ε))]_+ 는 0 이어야 하나 math.log(0)
        은 ValueError. select_subgoal 이 크래시 없이 novelty=0 으로 처리한다."""
        nominal = np.array([0.3, 0.0, 0.2])
        cfg = self._cfg(min_buffer_size=4, k_nn=3)
        buf = SubgoalBuffer()
        # candidate 0 = nominal: 그 canonical preview 의 T_end descriptor 들을
        # 그대로 buffer 에 적재 → query d_knn=0.
        end_states = last_segment(
            interp_plan(START_EE, nominal, cfg.preview_points), cfg.end_fraction)
        for s in end_states:
            for _ in range(4):                       # k_nn=3 이웃 모두 exact-match
                buf.append(_entry("move", state_descriptor(s, nominal)))
        for i in range(5):                           # scale > 0 되도록 떨어진 점
            buf.append(_entry("move", state_descriptor(nominal + 0.1 * (i + 1), nominal)))
        sel = Phase1SubgoalSelector(buf, cfg)
        out = sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(0))
        assert out.cold_start is False
        gains = {r.candidate_index: r.gain for r in out.reports}
        assert gains[0] == pytest.approx(0.0, abs=1e-9)   # nominal 이미 covered

    def test_cold_start_resolves_after_enough_commits(self):
        cfg = self._cfg(min_buffer_size=5, k_nn=1)
        sel = Phase1SubgoalSelector(SubgoalBuffer(), cfg)
        nominal = np.array([0.3, 0.0, 0.2])
        assert sel.select_subgoal(START_EE, nominal, "move",
                                  np.random.default_rng(0)).cold_start is True
        rng = np.random.default_rng(1)
        for _ in range(6):
            _commit_one(sel, "move", START_EE, nominal + rng.normal(0, 0.02, 3))
        assert sel.select_subgoal(START_EE, nominal, "move",
                                  np.random.default_rng(0)).cold_start is False

    def test_cold_start_threshold_respects_k_nn(self):
        # min_buffer_size 가 k_nn 보다 작아도 실제 임계값은 k_nn 으로 올라간다.
        cfg = self._cfg(min_buffer_size=2, k_nn=5)
        sel = Phase1SubgoalSelector(SubgoalBuffer(), cfg)
        nominal = np.array([0.3, 0.0, 0.2])
        rng = np.random.default_rng(1)
        for _ in range(4):                       # N=4 < k_nn=5 → 아직 cold-start
            _commit_one(sel, "move", START_EE, nominal + rng.normal(0, 0.02, 3))
        assert sel.select_subgoal(START_EE, nominal, "move",
                                  np.random.default_rng(0)).cold_start is True
        _commit_one(sel, "move", START_EE, nominal + np.array([0.03, 0.0, 0.0]))  # N=5
        assert sel.select_subgoal(START_EE, nominal, "move",
                                  np.random.default_rng(0)).cold_start is False

    def test_candidate_dist_config_is_used(self):
        for d in ("uniform_ball", "gaussian"):
            sel = Phase1SubgoalSelector(SubgoalBuffer(), self._cfg(candidate_dist=d))
            out = sel.select_subgoal(START_EE, np.array([0.3, 0.0, 0.2]), "move",
                                     np.random.default_rng(0))
            assert out.chosen_goal.shape == (3,)


class TestDebugVerbose:
    """debug_verbose 옵션 — 켜면 버퍼/판단/선택 로그를 stdout 에 출력."""

    def _cfg(self, **kw):
        base = dict(n_candidates=8, k_nn=3, min_buffer_size=4)
        base.update(kw)
        return Phase1SubgoalConfig(**base)

    def test_debug_verbose_emits_buffer_and_selection_logs(self, capsys):
        sel = Phase1SubgoalSelector(SubgoalBuffer(), self._cfg(debug_verbose=True))
        nominal = np.array([0.3, 0.0, 0.2])
        rng = np.random.default_rng(1)
        for _ in range(8):
            _commit_one(sel, "move", START_EE, nominal + rng.normal(0, 0.03, 3))
        sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(0))

        out = capsys.readouterr().out
        assert "[Subgoal-Phase1][debug]" in out
        assert "staged skill=move" in out
        assert "flush episode" in out
        assert "CHOSEN cand#" in out
        assert "gain" in out

    def test_debug_verbose_off_is_silent(self, capsys):
        sel = Phase1SubgoalSelector(SubgoalBuffer(), self._cfg(debug_verbose=False))
        nominal = np.array([0.3, 0.0, 0.2])
        rng = np.random.default_rng(1)
        for _ in range(8):
            _commit_one(sel, "move", START_EE, nominal + rng.normal(0, 0.03, 3))
        sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(0))
        assert "[Subgoal-Phase1][debug]" not in capsys.readouterr().out

    def test_debug_verbose_logs_cold_start(self, capsys):
        sel = Phase1SubgoalSelector(SubgoalBuffer(), self._cfg(debug_verbose=True))
        sel.select_subgoal(START_EE, np.array([0.3, 0.0, 0.2]), "move",
                           np.random.default_rng(0))
        assert "COLD-START" in capsys.readouterr().out


class TestPerSkillOverrides:
    """skill_id 별 sigma/clip_factor/n_candidates 부분 override."""

    def _cfg(self, **kw):
        base = dict(n_candidates=8, k_nn=1, min_buffer_size=4,
                    sigma=0.05, clip_factor=2.0)
        base.update(kw)
        return Phase1SubgoalConfig(**base)

    def _radii(self, nominal, reports):
        return np.array([np.linalg.norm(r.goal - nominal) for r in reports])

    def test_override_skill_radius_is_narrower(self):
        # 같은 nominal · 같은 시드인데 skill_id 별 sigma override 가 적용된 호출은
        # 후보 반경 분포가 명백히 좁아진다 (전역 R=0.10 vs override R=0.04).
        cfg = self._cfg(per_skill_overrides={
            "move_initial": {"sigma": 0.02, "clip_factor": 2.0}
        })
        buf = SubgoalBuffer()
        for i in range(8):
            buf.append(_entry("move_initial",
                              np.arange(DESCRIPTOR_DIM, dtype=float) + i * 0.3))
            buf.append(_entry("move",
                              np.arange(DESCRIPTOR_DIM, dtype=float) + i * 0.3))
        sel = Phase1SubgoalSelector(buf, cfg)
        nominal = np.array([0.3, 0.0, 0.2])

        global_out = sel.select_subgoal(START_EE, nominal, "move",
                                        np.random.default_rng(0))
        override_out = sel.select_subgoal(START_EE, nominal, "move_initial",
                                          np.random.default_rng(0))
        global_max = float(self._radii(nominal, global_out.reports).max())
        override_max = float(self._radii(nominal, override_out.reports).max())
        assert override_max <= 0.04 + 1e-9          # override clip = 2 × 0.02
        assert global_max > override_max * 1.5       # 명백히 더 넓음

    def test_unknown_skill_falls_back_to_global(self):
        cfg = self._cfg(per_skill_overrides={"move_initial": {"sigma": 0.02}})
        buf = SubgoalBuffer()
        for i in range(8):
            buf.append(_entry("other_skill",
                              np.arange(DESCRIPTOR_DIM, dtype=float) + i * 0.3))
        sel = Phase1SubgoalSelector(buf, cfg)
        nominal = np.array([0.3, 0.0, 0.2])
        out = sel.select_subgoal(START_EE, nominal, "other_skill",
                                 np.random.default_rng(0))
        # 전역 sigma=0.05, clip=2.0 → R=0.10. 후보 반경 최대 ≤ 0.10 + 수치오차.
        radii = self._radii(nominal, out.reports)
        assert float(radii.max()) <= 0.10 + 1e-9
        assert float(radii.max()) > 0.04            # override 가 안 먹은 증거

    def test_empty_overrides_unchanged_behavior(self):
        # 빈 dict → 기존 동작과 완전 동일 (회귀 방지).
        cfg_a = self._cfg(per_skill_overrides={})
        cfg_b = self._cfg()  # default: 빈 dict
        buf = SubgoalBuffer()
        for i in range(8):
            buf.append(_entry("move",
                              np.arange(DESCRIPTOR_DIM, dtype=float) + i * 0.3))
        nominal = np.array([0.3, 0.0, 0.2])
        a = Phase1SubgoalSelector(buf, cfg_a).select_subgoal(
            START_EE, nominal, "move", np.random.default_rng(7))
        b = Phase1SubgoalSelector(buf, cfg_b).select_subgoal(
            START_EE, nominal, "move", np.random.default_rng(7))
        np.testing.assert_array_equal(a.chosen_goal, b.chosen_goal)
        assert a.chosen_index == b.chosen_index


class TestEpisodeSimulation:
    """move_to_position 의 실제 사용 계약을 모사한 select→commit 루프 테스트."""

    def test_repeated_select_commit_loop_is_stable(self):
        cfg = Phase1SubgoalConfig(n_candidates=24, k_nn=5, min_buffer_size=12)
        buf = SubgoalBuffer()
        sel = Phase1SubgoalSelector(buf, cfg)
        nominal = np.array([0.30, 0.0, 0.20])
        rng = np.random.default_rng(7)

        chosen_goals = []
        prev_size = 0
        for _ in range(40):                   # 각 반복 = 한 에피소드(TRUE)
            out = sel.select_subgoal(START_EE, nominal, "move", rng,
                                     reachable_fn=lambda x: True)
            for r in out.reports:
                assert np.isfinite(r.gain) and r.gain >= 0.0
            _commit_one(sel, "move", START_EE, out.chosen_goal)
            assert buf.size("move") == prev_size + 1   # episode 당 정확히 +1
            prev_size = buf.size("move")
            chosen_goals.append(tuple(np.round(out.chosen_goal, 4)))

        assert len(set(chosen_goals)) > 1   # 여러 subgoal 을 탐색

    def test_buffer_aware_avoids_already_covered_region(self):
        cfg = Phase1SubgoalConfig(n_candidates=32, k_nn=5, min_buffer_size=8, sigma=0.06)
        buf = SubgoalBuffer()
        sel = Phase1SubgoalSelector(buf, cfg)
        nominal = np.array([0.30, 0.0, 0.20])

        rng = np.random.default_rng(5)
        for _ in range(30):
            _commit_one(sel, "move", START_EE, nominal + rng.normal(0, 0.005, 3))

        out = sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(0),
                                 reachable_fn=lambda xyz: True)
        assert out.cold_start is False
        gains = {r.candidate_index: r.gain for r in out.reports}
        assert gains[0] == pytest.approx(min(gains.values()))   # nominal 최소
        assert out.chosen_index == max(gains, key=gains.get)    # argmax
        assert out.chosen_index != 0

    def test_only_true_episodes_grow_buffer(self):
        cfg = Phase1SubgoalConfig(n_candidates=16, k_nn=3, min_buffer_size=4)
        buf = SubgoalBuffer()
        sel = Phase1SubgoalSelector(buf, cfg)
        nominal = np.array([0.30, 0.0, 0.20])
        rng = np.random.default_rng(0)

        # episode 1 (TRUE): 3 transit move → flush
        for _ in range(3):
            out = sel.select_subgoal(START_EE, nominal, "move", rng,
                                     reachable_fn=lambda x: True)
            sel.stage_executed("move", START_EE, out.chosen_goal)
        sel.flush_episode()
        assert buf.size("move") == 3

        # episode 2 (FALSE): 2 transit move → discard
        for _ in range(2):
            out = sel.select_subgoal(START_EE, nominal, "move", rng,
                                     reachable_fn=lambda x: True)
            sel.stage_executed("move", START_EE, out.chosen_goal)
        sel.discard_episode()
        assert buf.size("move") == 3          # FALSE 에피소드는 buffer 미반영

        # episode 3 (TRUE): 1 transit move → flush
        out = sel.select_subgoal(START_EE, nominal, "move", rng,
                                 reachable_fn=lambda x: True)
        sel.stage_executed("move", START_EE, out.chosen_goal)
        sel.flush_episode()
        assert buf.size("move") == 4


# ─────────────────────────────────────────────────────────────
# §4.2b  feasibility_fn — 2차 필터 (holding-phase IK 등)
# ─────────────────────────────────────────────────────────────
class TestFeasibilityFilter:
    """select_subgoal 의 feasibility_fn 2차 필터 검증."""

    def _populated(self):
        cfg = Phase1SubgoalConfig(n_candidates=24, k_nn=3, min_buffer_size=4)
        buf = SubgoalBuffer()
        sel = Phase1SubgoalSelector(buf, cfg)
        nominal = np.array([0.30, 0.0, 0.10])
        rng = np.random.default_rng(0)
        for _ in range(10):
            _commit_one(sel, "move", START_EE, nominal + rng.normal(0, 0.03, 3))
        return sel, nominal

    def test_none_preserves_legacy_behavior(self):
        sel, nominal = self._populated()
        a = sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(1))
        b = sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(1),
                               feasibility_fn=None)
        assert a.chosen_index == b.chosen_index

    def test_rejected_candidates_excluded_from_selection(self):
        # feasibility_fn 이 거른 후보는 chosen 이 될 수 없고, 선택된 후보는
        # feasible 후보들 중 gain 최대여야 한다 (lazy gain-내림차순 검사).
        # reports 는 전체 valid 후보를 담는다 (feasible 만이 아님).
        sel, nominal = self._populated()
        feas = lambda xyz: xyz[0] <= 0.30          # x>0.30 후보 전부 탈락
        out = sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(1),
                                 feasibility_fn=feas)
        assert out.chosen_goal[0] <= 0.30 + 1e-9
        feasible = [r for r in out.reports if r.goal[0] <= 0.30 + 1e-9]
        chosen = next(r for r in out.reports
                      if r.candidate_index == out.chosen_index)
        assert chosen.gain == pytest.approx(max(r.gain for r in feasible))

    def test_all_rejected_falls_back_to_nominal(self):
        sel, nominal = self._populated()
        out = sel.select_subgoal(START_EE, nominal, "move", np.random.default_rng(1),
                                 feasibility_fn=lambda xyz: False)
        assert out.chosen_index == 0
        assert np.allclose(out.chosen_goal, nominal)
        assert out.cold_start is False

    def test_cold_start_pool_respects_feasibility(self):
        cfg = Phase1SubgoalConfig(n_candidates=24, min_buffer_size=8)
        sel = Phase1SubgoalSelector(SubgoalBuffer(), cfg)
        nominal = np.array([0.30, 0.0, 0.10])
        feas = lambda xyz: xyz[0] <= 0.30
        for seed in range(20):
            out = sel.select_subgoal(START_EE, nominal, "move",
                                     np.random.default_rng(seed), feasibility_fn=feas)
            assert out.cold_start is True
            assert out.chosen_goal[0] <= 0.30 + 1e-9
