"""Method3 storage layer — unit tests (문서 final_method3_spec §3).

raw trajectory dataset 의 append·복원·pointer 왕복을 격리 검증한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from method3.storage.raw_dataset import (
    RawDatasetEntry,
    RawTrajectoryDataset,
    resolve_pointer,
)


def _entry(skill="reach", phase="phase1", episode="ep_0", t=0, H=12, A=6, P=7):
    """§3.1 필드를 모두 채운 합성 RawDatasetEntry."""
    rng = np.random.default_rng(t + 1)
    return RawDatasetEntry(
        episode_id=episode,
        phase=phase,
        skill_id=skill,
        instruction="pick up the cube",
        subgoal=np.array([0.3, 0.0, 0.2]) + t * 0.01,
        time_index=t,
        observation_ref={"dataset": "lerobot_ds", "episode": episode, "frame": t},
        proprioception=rng.normal(0, 1, P),
        action_chunk=rng.normal(0, 1, (H, A)),
        raw_action_sequence=rng.normal(0, 1, (H * 2, A)),
        planner_type="InterpPlan",
        success_flag=True,
        validity_flag=True,
        environment_metadata={"table_z": 0.0},
        object_metadata={"cube": [0.3, 0.0, 0.05]},
    )


class TestRawTrajectoryDataset:
    def test_append_returns_running_index(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "D_phase1_raw")
        assert ds.append(_entry(t=0)) == 0
        assert ds.append(_entry(t=1)) == 1
        assert len(ds) == 2

    def test_get_round_trips_numeric_arrays(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "D_phase1_raw")
        e = _entry(t=3)
        ds.append(e)
        got = ds.get(0)
        np.testing.assert_allclose(got.subgoal, e.subgoal)
        np.testing.assert_allclose(got.proprioception, e.proprioception)
        np.testing.assert_allclose(got.action_chunk, e.action_chunk)
        np.testing.assert_allclose(got.raw_action_sequence, e.raw_action_sequence)

    def test_get_round_trips_metadata(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "D_phase1_raw")
        e = _entry(skill="grasp", phase="phase2", episode="ep_9", t=5)
        ds.append(e)
        got = ds.get(0)
        assert got.episode_id == "ep_9"
        assert got.phase == "phase2"
        assert got.skill_id == "grasp"
        assert got.instruction == "pick up the cube"
        assert got.time_index == 5
        assert got.observation_ref == {"dataset": "lerobot_ds",
                                       "episode": "ep_9", "frame": 5}
        assert got.planner_type == "InterpPlan"
        assert got.success_flag is True and got.validity_flag is True
        assert got.environment_metadata == {"table_z": 0.0}
        assert got.object_metadata == {"cube": [0.3, 0.0, 0.05]}

    def test_entries_returns_all(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "D_phase1_raw")
        for t in range(4):
            ds.append(_entry(t=t))
        entries = ds.entries()
        assert len(entries) == 4
        assert [e.time_index for e in entries] == [0, 1, 2, 3]

    def test_skill_entries_filter(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "D_phase1_raw")
        ds.append(_entry(skill="reach", t=0))
        ds.append(_entry(skill="grasp", t=1))
        ds.append(_entry(skill="reach", t=2))
        reach = ds.skill_entries("reach")
        assert len(reach) == 2
        assert all(e.skill_id == "reach" for e in reach)

    def test_reopen_existing_dataset_continues(self, tmp_path):
        d = tmp_path / "D_phase1_raw"
        ds = RawTrajectoryDataset(d)
        ds.append(_entry(t=0))
        ds.append(_entry(t=1))

        reopened = RawTrajectoryDataset(d)          # 같은 디렉터리 재오픈
        assert len(reopened) == 2
        assert reopened.append(_entry(t=2)) == 2    # 이어서 append
        assert len(reopened) == 3
        assert RawTrajectoryDataset(d).get(2).time_index == 2

    def test_pointer_resolve_round_trip(self, tmp_path):
        # §3 — vector DB ref 가 raw dataset entry 를 복원할 수 있어야 한다 (§4).
        ds = RawTrajectoryDataset(tmp_path / "D_phase1_raw")
        ds.append(_entry(t=0))
        ds.append(_entry(skill="grasp", episode="ep_7", t=1))
        ptr = ds.pointer(1)
        assert ptr["entry_index"] == 1
        assert ptr["skill_id"] == "grasp"
        resolved = resolve_pointer(ptr)
        assert resolved.episode_id == "ep_7"
        assert resolved.time_index == 1
        np.testing.assert_allclose(resolved.action_chunk, ds.get(1).action_chunk)

    def test_index_jsonl_one_line_per_entry(self, tmp_path):
        # append-only — entry 마다 index.jsonl 에 정확히 1줄.
        d = tmp_path / "D_phase1_raw"
        ds = RawTrajectoryDataset(d)
        for t in range(5):
            ds.append(_entry(t=t))
        lines = (d / "index.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5
