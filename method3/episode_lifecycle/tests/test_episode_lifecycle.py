"""episode_lifecycle — unit tests (episode 단위 삭제·재취득 정합성).

epsiode 를 삭제하고 resume 으로 재취득할 때, stale entry 가 store 에 남지 않고
생존 에피소드 집합과 정합을 유지하는지 검증한다.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from method3.episode_lifecycle import (
    EpisodeReconciler,
    EpisodeScopedStore,
    episode_id,
    episode_num,
)
from method3.episode_lifecycle.migrate_legacy_buffer import migrate_session_buffer
from method3.phase1_state_seeding.subgoal_buffer import SubgoalBuffer, SubgoalBufferEntry
from method3.storage.raw_dataset import RawDatasetEntry, RawTrajectoryDataset


# ─────────────────────────────────────────────────────────────
# episode_ref — canonical 식별자
# ─────────────────────────────────────────────────────────────
class TestEpisodeRef:
    def test_episode_id_zero_pads_to_folder_convention(self):
        assert episode_id(39) == "episode_39"
        assert episode_id(5) == "episode_05"
        assert episode_id(100) == "episode_100"

    def test_roundtrip(self):
        for n in (1, 5, 39, 100):
            assert episode_num(episode_id(n)) == n


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def _buffer_entry(skill: str, ep: str) -> SubgoalBufferEntry:
    return SubgoalBufferEntry(
        skill_id=skill,
        subgoal=np.array([0.3, 0.0, 0.2]),
        terminal_region_key=np.zeros(6),
        end_state_keys=np.zeros((4, 6)),
        episode_id=ep,
        start_t=0,
        end_t=4,
        success_flag=True,
        planner_type="InterpPlan",
        phase="phase1",
    )


def _filled_buffer(*eps: str) -> SubgoalBuffer:
    buf = SubgoalBuffer()
    for ep in eps:
        buf.append(_buffer_entry("move", ep))
    return buf


def _raw_entry(ep: str, marker: float) -> RawDatasetEntry:
    # subgoal 에 marker 를 심어 npz 가 올바르게 매칭되는지 확인할 수 있게 한다.
    return RawDatasetEntry(
        episode_id=ep, phase="phase1", skill_id="move",
        instruction="do", subgoal=np.full(3, marker), time_index=0,
        observation_ref={}, proprioception=np.zeros(7),
        action_chunk=np.zeros((12, 6)), raw_action_sequence=np.zeros((12, 6)),
    )


# ─────────────────────────────────────────────────────────────
# SubgoalBuffer — episode 단위 제거
# ─────────────────────────────────────────────────────────────
class TestSubgoalBufferEpisodeOps:
    def test_remove_episode_drops_matching_entries(self):
        buf = _filled_buffer("episode_01", "episode_02", "episode_02", "episode_03")
        assert buf.remove_episode("episode_02") == 2
        assert buf.total_size() == 2

    def test_remove_absent_episode_is_noop(self):
        buf = _filled_buffer("episode_01")
        assert buf.remove_episode("episode_99") == 0
        assert buf.total_size() == 1

    def test_retain_episodes_keeps_only_given(self):
        buf = _filled_buffer("episode_01", "episode_02", "episode_03")
        assert buf.retain_episodes({"episode_01", "episode_03"}) == 1
        assert buf.total_size() == 2

    def test_retain_empty_clears_all(self):
        buf = _filled_buffer("episode_01", "episode_02")
        assert buf.retain_episodes(set()) == 2
        assert buf.total_size() == 0

    def test_retain_keeps_untagged_legacy_entries(self):
        # episode_id plumbing 이전 buffer — untagged("") entry 는 reconcile 이 보존.
        buf = SubgoalBuffer()
        buf.append(_buffer_entry("move", ""))             # legacy untagged
        buf.append(_buffer_entry("move", "episode_02"))
        dropped = buf.retain_episodes({"episode_05"})      # 둘 다 keep 집합 밖
        assert dropped == 1                                # tagged 만 제거
        assert buf.total_size() == 1                       # untagged 는 보존

    def test_satisfies_episode_scoped_store_protocol(self):
        assert isinstance(SubgoalBuffer(), EpisodeScopedStore)

    def test_remove_persists_to_npz(self, tmp_path):
        buf = _filled_buffer("episode_01", "episode_02")
        buf.set_file(tmp_path / "subgoal_buffer.npz")
        buf.save()
        buf.remove_episode("episode_01")          # auto-save
        reloaded = SubgoalBuffer()
        reloaded.set_file(tmp_path / "subgoal_buffer.npz")
        reloaded.load()
        assert reloaded.total_size() == 1
        assert reloaded.entries("move")[0].episode_id == "episode_02"


# ─────────────────────────────────────────────────────────────
# RawTrajectoryDataset — episode 단위 제거
# ─────────────────────────────────────────────────────────────
class TestRawDatasetEpisodeOps:
    def test_remove_episode(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "d")
        ds.append(_raw_entry("episode_01", 1.0))
        ds.append(_raw_entry("episode_02", 2.0))
        ds.append(_raw_entry("episode_02", 2.0))
        assert ds.remove_episode("episode_02") == 2
        assert len(ds) == 1
        assert ds.get(0).episode_id == "episode_01"

    def test_append_after_remove_has_no_npz_collision(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "d")
        ds.append(_raw_entry("episode_01", 1.0))   # 000000.npz
        ds.append(_raw_entry("episode_02", 2.0))   # 000001.npz
        ds.remove_episode("episode_01")            # 000000.npz 삭제
        ds.append(_raw_entry("episode_03", 3.0))   # len=1 → 000001 충돌 회피 → 000002
        assert len(ds) == 2
        # npz 덮어쓰기가 없었으면 marker 가 보존된다.
        by_ep = {e.episode_id: float(e.subgoal[0]) for e in ds.entries()}
        assert by_ep == {"episode_02": 2.0, "episode_03": 3.0}

    def test_retain_episodes(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "d")
        for i, ep in enumerate(("episode_01", "episode_02", "episode_03"), 1):
            ds.append(_raw_entry(ep, float(i)))
        assert ds.retain_episodes({"episode_02"}) == 2
        assert [e.episode_id for e in ds.entries()] == ["episode_02"]

    def test_survives_reopen(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "d")
        ds.append(_raw_entry("episode_01", 1.0))
        ds.append(_raw_entry("episode_02", 2.0))
        ds.remove_episode("episode_01")
        reopened = RawTrajectoryDataset(tmp_path / "d")
        assert len(reopened) == 1
        assert reopened.get(0).episode_id == "episode_02"
        assert float(reopened.get(0).subgoal[0]) == 2.0

    def test_satisfies_episode_scoped_store_protocol(self, tmp_path):
        assert isinstance(RawTrajectoryDataset(tmp_path / "d"), EpisodeScopedStore)


# ─────────────────────────────────────────────────────────────
# EpisodeReconciler — 여러 store 일괄 정리
# ─────────────────────────────────────────────────────────────
class TestEpisodeReconciler:
    def test_remove_episode_across_stores(self, tmp_path):
        buf = _filled_buffer("episode_01", "episode_02")
        ds = RawTrajectoryDataset(tmp_path / "d")
        ds.append(_raw_entry("episode_01", 1.0))
        ds.append(_raw_entry("episode_02", 2.0))
        result = EpisodeReconciler([buf, ds]).remove_episode("episode_01")
        assert sum(result.values()) == 2          # buffer 1 + raw dataset 1
        assert buf.total_size() == 1
        assert len(ds) == 1

    def test_retain_episodes_across_stores(self):
        buf = _filled_buffer("episode_01", "episode_02", "episode_03")
        EpisodeReconciler([buf]).retain_episodes({"episode_02"})
        assert buf.total_size() == 1

    def test_none_stores_are_skipped(self):
        buf = _filled_buffer("episode_01")
        result = EpisodeReconciler([None, buf]).remove_episode("episode_01")
        assert sum(result.values()) == 1


# ─────────────────────────────────────────────────────────────
# flush_episode — episode_id stamping (단계0 plumbing)
# ─────────────────────────────────────────────────────────────
class TestFlushEpisodeStamping:
    def test_flush_stamps_episode_id_on_all_entries(self):
        from method3.phase1_state_seeding.subgoal_selector import (
            Phase1SubgoalConfig,
            Phase1SubgoalSelector,
        )
        sel = Phase1SubgoalSelector(SubgoalBuffer(), Phase1SubgoalConfig())
        sel.stage_executed(np.zeros(3), np.array([0.3, 0.0, 0.2]))
        sel.stage_executed(np.zeros(3), np.array([0.3, 0.1, 0.2]))
        sel.flush_episode(episode_id="episode_07")
        # ordinal 키 — 2개 staged → skill_0, skill_1.
        committed = sel.buffer.entries("skill_0") + sel.buffer.entries("skill_1")
        assert len(committed) == 2
        assert all(e.episode_id == "episode_07" for e in committed)

    def test_flush_without_episode_id_keeps_staged_value(self):
        from method3.phase1_state_seeding.subgoal_selector import (
            Phase1SubgoalConfig,
            Phase1SubgoalSelector,
        )
        sel = Phase1SubgoalSelector(SubgoalBuffer(), Phase1SubgoalConfig())
        sel.stage_executed(np.zeros(3), np.array([0.3, 0.0, 0.2]),
                           episode_id="episode_03")
        sel.flush_episode()                       # episode_id 미지정
        assert sel.buffer.entries("skill_0")[0].episode_id == "episode_03"


# ─────────────────────────────────────────────────────────────
# resume reconcile — 사용자 시나리오 (폴더 삭제 → resume)
# ─────────────────────────────────────────────────────────────
class TestResumeReconcileScenario:
    """삭제된 에피소드 폴더 → resume → buffer reconcile 의 end-to-end 로직."""

    def test_deleted_true_episode_dropped_then_reacquired(self):
        # ep1·2·3 모두 TRUE 로 commit. 사용자가 episode_02 폴더를 삭제.
        buf = _filled_buffer("episode_01", "episode_02", "episode_03")
        # cleanup_dataset_for_resume 가 kept_true=[1,3] 산출 → _finalize 가 reconcile.
        keep_ids = {episode_id(n) for n in [1, 3]}
        dropped = EpisodeReconciler([buf]).retain_episodes(keep_ids)
        assert sum(dropped.values()) == 1              # episode_02 stale 제거
        assert buf.total_size() == 2
        # 재취득되면 fresh episode_02 가 새로 들어온다.
        buf.append(_buffer_entry("move", "episode_02"))
        assert buf.total_size() == 3

    def test_false_episode_resume_is_clean_noop(self):
        # FALSE 에피소드는 discard_episode 로 애초에 buffer 에 없음 (scenario A).
        # resume reconcile 은 그 episode 에 아무 영향도 주지 않는다.
        buf = _filled_buffer("episode_01", "episode_03")   # ep2 는 FALSE → 없음
        keep_ids = {episode_id(n) for n in [1, 3]}
        dropped = EpisodeReconciler([buf]).retain_episodes(keep_ids)
        assert sum(dropped.values()) == 0
        assert buf.total_size() == 2


# ─────────────────────────────────────────────────────────────
# legacy buffer 마이그레이션 — episode_id 소급 태깅
# ─────────────────────────────────────────────────────────────
class TestMigrateLegacyBuffer:
    @staticmethod
    def _make_session(root, episodes):
        """episodes: [(ep_num, judge, {skill: staged_count})] → 세션 폴더 생성."""
        for ep_num, judge, skill_counts in episodes:
            fwd = root / f"episode_{ep_num:02d}" / "forward"
            fwd.mkdir(parents=True)
            (root / f"episode_{ep_num:02d}" / "batch_info.json").write_text(
                json.dumps({"judge": judge}))
            lines = [
                f"[Subgoal-Phase1][debug] staged skill={s} → episode pending=1"
                for s, cnt in skill_counts.items() for _ in range(cnt)
            ]
            (fwd / "forward_log.txt").write_text("\n".join(lines))

    @staticmethod
    def _save_buffer(root, skill_counts):
        buf = SubgoalBuffer()
        for skill, n in skill_counts.items():
            for _ in range(n):
                buf.append(_buffer_entry(skill, ""))      # untagged
        buf.set_file(root / "subgoal_buffer.npz")
        buf.save()

    def test_migrates_untagged_buffer(self, tmp_path):
        self._make_session(tmp_path, [
            (1, "TRUE",  {"move_and_open": 1, "move": 2, "move_and_close": 1}),
            (2, "TRUE",  {"move_and_open": 1, "move": 2, "move_and_close": 1}),
            (3, "FALSE", {"move_and_open": 1, "move": 2, "move_and_close": 1}),
        ])
        # FALSE 는 flush 안 됨 → buffer 는 TRUE 2개 분량.
        self._save_buffer(tmp_path, {"move_and_open": 2, "move": 4, "move_and_close": 2})

        report = migrate_session_buffer(tmp_path)
        assert report["true_episodes"] == [1, 2]
        assert report["tagged"] == 8

        reloaded = SubgoalBuffer()
        reloaded.set_file(tmp_path / "subgoal_buffer.npz")
        reloaded.load()
        assert [e.episode_id for e in reloaded.entries("move_and_open")] == \
               ["episode_01", "episode_02"]
        assert [e.episode_id for e in reloaded.entries("move")] == \
               ["episode_01", "episode_01", "episode_02", "episode_02"]

    def test_gap_episode_handled(self, tmp_path):
        # ep02 가 move_and_open 을 staging 안 함 (184406 의 ep13 과 동일 케이스).
        self._make_session(tmp_path, [
            (1, "TRUE", {"move_and_open": 1, "move": 2}),
            (2, "TRUE", {"move": 2}),                       # move_and_open 0 — 갭
            (3, "TRUE", {"move_and_open": 1, "move": 2}),
        ])
        self._save_buffer(tmp_path, {"move_and_open": 2, "move": 6})

        migrate_session_buffer(tmp_path)
        reloaded = SubgoalBuffer()
        reloaded.set_file(tmp_path / "subgoal_buffer.npz")
        reloaded.load()
        # move_and_open 은 ep01·ep03 만 (ep02 는 갭).
        assert [e.episode_id for e in reloaded.entries("move_and_open")] == \
               ["episode_01", "episode_03"]

    def test_aborts_on_count_mismatch(self, tmp_path):
        # staged 로그(2)와 buffer entry(5) 불일치 → 잘못된 태깅 대신 abort.
        self._make_session(tmp_path, [(1, "TRUE", {"move": 2})])
        self._save_buffer(tmp_path, {"move": 5})
        with pytest.raises(ValueError, match="불일치"):
            migrate_session_buffer(tmp_path)

    def test_backup_created(self, tmp_path):
        self._make_session(tmp_path, [(1, "TRUE", {"move": 2})])
        self._save_buffer(tmp_path, {"move": 2})
        migrate_session_buffer(tmp_path)
        assert (tmp_path / "subgoal_buffer.npz.pre_migration_backup").exists()
