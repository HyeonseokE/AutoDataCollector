"""Phase2SubgoalReplay — seed-aware mapping unit tests.

원래 cycle-mapping (eps[(N-1) % len(eps)]) 은 seed-blind 라 round_robin schedule
에서 Phase2 ep_69 (seed 9 reset) 가 Phase1 ep_01 (seed 1) subgoal 을 replay 해
approach 와 pick descent 가 40 cm 어긋나는 사고가 났다. 본 모듈은 seed-aware
매핑 (entry stamp + num_seeds backfill) 이 그 mismatch 를 차단함을 검증한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from method3.phase1_state_seeding.subgoal_buffer import (
    SubgoalBuffer,
    SubgoalBufferEntry,
)
from method3.phase1_state_seeding.terminal_descriptor import DESCRIPTOR_DIM
from method3.phase2_mi_selection.subgoal_replay import Phase2SubgoalReplay


def _entry(skill_id, episode_id, subgoal_xy, *, skill_type="", seed_index=-1):
    """skill_id/episode_id/subgoal/seed_index 만 정해서 minimal entry 생성."""
    g = np.array([subgoal_xy[0], subgoal_xy[1], 0.15], dtype=np.float64)
    key = np.zeros(DESCRIPTOR_DIM, dtype=np.float64)
    return SubgoalBufferEntry(
        skill_id=str(skill_id),
        subgoal=g,
        terminal_region_key=key,
        end_state_keys=key.reshape(1, -1),
        episode_id=str(episode_id),
        start_t=0,
        end_t=0,
        skill_type=str(skill_type),
        seed_index=int(seed_index),
    )


def _build_buffer(tmp_path: Path, entries: list[SubgoalBufferEntry]) -> Path:
    """주어진 entries 로 npz buffer 를 만들어 경로 반환."""
    buf = SubgoalBuffer()
    buf.set_file(tmp_path / "subgoal_buffer.npz")
    for e in entries:
        buf.append(e)
    buf.save()
    return tmp_path / "subgoal_buffer.npz"


# ─────────────────────────────────────────────────────────────
# 1) round-trip 으로 seed_index 가 npz 에 보존되는가
# ─────────────────────────────────────────────────────────────
class TestBufferRoundTrip:
    def test_seed_index_roundtrip(self, tmp_path):
        path = _build_buffer(tmp_path, [
            _entry("skill_0", "episode_01", (0.1, 0.2), seed_index=0),
            _entry("skill_0", "episode_02", (0.3, 0.2), seed_index=1),
            _entry("skill_1", "episode_01", (0.2, -0.1), seed_index=0),
        ])
        buf = SubgoalBuffer()
        buf.set_file(path)
        buf.load()
        e0 = buf.entries("skill_0")
        # 정렬 보장 안 됨 → episode_id 로 lookup
        by_ep = {str(e.episode_id): e for e in e0}
        assert by_ep["episode_01"].seed_index == 0
        assert by_ep["episode_02"].seed_index == 1

    def test_legacy_buffer_load_default_neg1(self, tmp_path):
        """seedidx 키 없는 legacy npz 도 load 시 seed_index=-1 로 채워진다."""
        # 새 schema 로 일단 저장한 뒤 ::seedidx 만 빼고 다시 저장
        path = _build_buffer(tmp_path, [
            _entry("skill_0", "episode_01", (0.1, 0.2), seed_index=5),
        ])
        with np.load(path, allow_pickle=False) as data:
            keys_no_seed = {k: data[k] for k in data.files if "::seedidx" not in k}
        np.savez(path, **keys_no_seed)
        buf = SubgoalBuffer()
        buf.set_file(path)
        buf.load()
        e = buf.entries("skill_0")[0]
        assert e.seed_index == -1


# ─────────────────────────────────────────────────────────────
# 2) seed-aware lookup — entry stamp 우선
# ─────────────────────────────────────────────────────────────
class TestSeedAwareLookup:
    def test_stamped_seed_matches(self, tmp_path):
        """entry 에 seed_index 가 stamp 되어 있으면 그 seed pool 에서 선택."""
        path = _build_buffer(tmp_path, [
            _entry("skill_0", "episode_01", (0.10, 0.20), seed_index=0),
            _entry("skill_0", "episode_09", (0.30, 0.20), seed_index=8),
            _entry("skill_0", "episode_29", (0.31, 0.18), seed_index=8),
        ])
        replay = Phase2SubgoalReplay(path)
        replay.set_episode("episode_69", seed_index=8)
        # episode_69 는 buffer 에 없음 → seed-aware rotation 으로 ep_09 또는 ep_29
        assert replay._episode_id in ("episode_09", "episode_29")

    def test_rotation_uses_all_pool_entries(self, tmp_path):
        """같은 seed 의 pool 에 여러 episode 가 있으면 rotation 으로 분산 사용."""
        path = _build_buffer(tmp_path, [
            _entry("skill_0", "episode_09", (0.30, 0.20), seed_index=8),
            _entry("skill_0", "episode_29", (0.31, 0.18), seed_index=8),
            _entry("skill_0", "episode_49", (0.29, 0.20), seed_index=8),
        ])
        replay = Phase2SubgoalReplay(path)
        seen = set()
        for ep_n in (69, 89, 109):  # 동일 seed=8 의 3번 호출
            replay.set_episode(f"episode_{ep_n:02d}", seed_index=8)
            seen.add(replay._episode_id)
        # rotation 이라면 3 호출에 3 ep 모두 등장
        assert seen == {"episode_09", "episode_29", "episode_49"}

    def test_exact_match_takes_priority(self, tmp_path):
        """buffer 에 정확히 같은 episode_id 가 있으면 seed_index 무시하고 exact."""
        path = _build_buffer(tmp_path, [
            _entry("skill_0", "episode_69", (0.30, 0.20), seed_index=8),
            _entry("skill_0", "episode_09", (0.99, 0.99), seed_index=8),
        ])
        replay = Phase2SubgoalReplay(path)
        replay.set_episode("episode_69", seed_index=8)
        assert replay._episode_id == "episode_69"


# ─────────────────────────────────────────────────────────────
# 3) backfill — legacy buffer + num_seeds → seed-aware 동작
# ─────────────────────────────────────────────────────────────
class TestBackfill:
    def test_round_robin_backfill(self, tmp_path):
        """seed_index stamp 없는 legacy buffer 도 num_seeds 로 backfill 동작."""
        # 모든 entry 의 seed_index=-1 — legacy 시나리오
        path = _build_buffer(tmp_path, [
            _entry("skill_0", f"episode_{n:02d}", (0.2, 0.2), seed_index=-1)
            for n in range(1, 21)  # ep_01..ep_20 (20 seeds 모두 1번씩)
        ])
        replay = Phase2SubgoalReplay(
            path, num_seeds=20, schedule_mode="round_robin",
        )
        # ep_69 의 round_robin seed = (69-1) % 20 = 8 → ep_09 (backfilled seed=8)
        replay.set_episode("episode_69", seed_index=8)
        assert replay._episode_id == "episode_09"

    def test_seed_major_backfill(self, tmp_path):
        """seed_major: ep_num // episodes_per_seed 로 seed 추정."""
        # episodes_per_seed=5 → ep_01..05 → seed_0, ep_06..10 → seed_1, ...
        path = _build_buffer(tmp_path, [
            _entry("skill_0", f"episode_{n:02d}", (0.2, 0.2), seed_index=-1)
            for n in range(1, 21)
        ])
        replay = Phase2SubgoalReplay(
            path, num_seeds=4, schedule_mode="seed_major",
            episodes_per_seed=5,
        )
        # ep_22 → seed (22-1) // 5 = 4 — out of range (num_seeds=4 means 0..3)
        # 그러나 ep_22 의 seed_index 인자로 0 을 주면 seed 0 pool (ep_01..05) 에서 선택.
        replay.set_episode("episode_22", seed_index=0)
        assert replay._episode_id in {
            f"episode_{n:02d}" for n in range(1, 6)
        }


# ─────────────────────────────────────────────────────────────
# 4) legacy cycle fallback — num_seeds 도 없고 seed_index 도 없을 때만
# ─────────────────────────────────────────────────────────────
class TestLegacyFallback:
    def test_falls_back_with_warning(self, tmp_path, capsys):
        """num_seeds 없음 + seed_index<0 → 옛 cycle 매핑 (경고 출력)."""
        path = _build_buffer(tmp_path, [
            _entry("skill_0", f"episode_{n:02d}", (0.2, 0.2), seed_index=-1)
            for n in range(1, 11)
        ])
        replay = Phase2SubgoalReplay(path)  # num_seeds=None → backfill 없음
        replay.set_episode("episode_69")     # seed_index=-1 default
        out = capsys.readouterr().out
        assert "SEED-BLIND" in out or "seed_index 미전달" in out

    def test_falls_back_when_seed_pool_empty(self, tmp_path, capsys):
        """seed_index 가 buffer pool 에 없으면 legacy fallback (경고)."""
        path = _build_buffer(tmp_path, [
            _entry("skill_0", "episode_01", (0.1, 0.2), seed_index=0),
        ])
        replay = Phase2SubgoalReplay(path)
        replay.set_episode("episode_69", seed_index=42)  # 42 는 buffer 에 없음
        out = capsys.readouterr().out
        assert "SEED-BLIND" in out


# ─────────────────────────────────────────────────────────────
# 5) regression — RCA 시나리오: seed 9 reset 가 seed 1 subgoal 안 받게
# ─────────────────────────────────────────────────────────────
class TestRcaRegression:
    def test_ep_69_seed_9_does_not_get_seed_1(self, tmp_path):
        """session_20260524_233421 의 실제 mismatch 시나리오 회귀.

        round_robin 20 seeds, Phase1 ep_01..ep_68 commit. Phase2 ep_69 reset
        seed=8 (0-based). 옛 코드는 ep_01 (seed 0) replay; 새 코드는 seed 8
        pool 에서 — ep_09/ep_29/ep_49 중 하나.
        """
        entries = []
        for n in range(1, 69):
            seed = (n - 1) % 20
            entries.append(_entry(
                "skill_0", f"episode_{n:02d}", (0.2 + seed*0.01, 0.2),
                seed_index=seed,
            ))
        path = _build_buffer(tmp_path, entries)
        replay = Phase2SubgoalReplay(path, num_seeds=20)
        replay.set_episode("episode_69", seed_index=8)
        # seed 8 pool = {ep_09, ep_29, ep_49} (round_robin)
        assert replay._episode_id in {"episode_09", "episode_29", "episode_49"}
        assert replay._episode_id != "episode_01"  # 옛 코드의 잘못된 매핑
