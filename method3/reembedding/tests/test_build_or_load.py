"""build_or_load_phase1_vector_db — unit tests (§6 entry).

LeRobot 데이터셋과 실제 VLA encoder 의존 없이 testable 하도록 ``raw_dataset`` /
``encoder`` 를 직접 주입한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from method3.phase2_mi_selection.vector_db import SkillVectorDB
from method3.reembedding.build_or_load import build_or_load_phase1_vector_db
from method3.reembedding.vla_encoder import MeanPoolStateEncoder
from method3.storage.raw_dataset import RawDatasetEntry


# ─────────────────────────────────────────────────────────────
# fixtures — synthetic RawTrajectoryDataset-like
# ─────────────────────────────────────────────────────────────
class _SyntheticRawDataset:
    """RawTrajectoryDataset-compatible — seed_builder 의 minimal contract."""

    def __init__(self, n_entries=6, skills=("move_and_open", "move", "move_and_close")):
        self._entries = []
        for i in range(n_entries):
            self._entries.append(RawDatasetEntry(
                episode_id=f"episode_{(i // 3) + 1:02d}",
                phase="phase1",
                skill_id=skills[i % len(skills)],
                instruction="pick up the red block",
                subgoal=np.array([0.3, 0.0, 0.2]),
                time_index=i,
                observation_ref={"frame": i},
                proprioception=np.full(7, float(i)),
                action_chunk=np.full((12, 6), float(i)),
                raw_action_sequence=np.full((12, 6), float(i)),
            ))

    def __len__(self):
        return len(self._entries)

    def get(self, idx):
        return self._entries[idx]

    def pointer(self, idx):
        return {"entry_index": idx, "episode_id": self._entries[idx].episode_id}

    def load_observation(self, ref):
        # stub — MeanPool 은 어떤 형상이든 받음.
        return np.full((3, 4), float(ref.get("frame", 0)))


# ─────────────────────────────────────────────────────────────
class TestBuildOrLoad:
    def test_builds_when_missing(self, tmp_path):
        ds = _SyntheticRawDataset(n_entries=6)
        enc = MeanPoolStateEncoder(out_dim=8)
        db = build_or_load_phase1_vector_db(
            tmp_path, raw_dataset=ds, encoder=enc,
            observation_loader=ds.load_observation,
        )
        assert isinstance(db, SkillVectorDB)
        assert db.total_size() == 6
        # 캐시 파일 생성됨
        assert (tmp_path / "phase1_vector_db.npz").exists()

    def test_returns_cached_when_present(self, tmp_path):
        ds = _SyntheticRawDataset(n_entries=6)
        enc = MeanPoolStateEncoder(out_dim=8)
        # 첫 호출 → 빌드
        db1 = build_or_load_phase1_vector_db(
            tmp_path, raw_dataset=ds, encoder=enc,
            observation_loader=ds.load_observation,
        )
        # 두 번째 호출 → 캐시 hit (raw_dataset/encoder 안 줘도 작동)
        db2 = build_or_load_phase1_vector_db(tmp_path)
        assert db2.total_size() == db1.total_size()
        assert sorted(db2.skill_ids()) == sorted(db1.skill_ids())

    def test_rebuild_flag_forces_rebuild(self, tmp_path):
        ds = _SyntheticRawDataset(n_entries=6)
        enc = MeanPoolStateEncoder(out_dim=8)
        build_or_load_phase1_vector_db(
            tmp_path, raw_dataset=ds, encoder=enc,
            observation_loader=ds.load_observation,
        )
        # rebuild=True 면 raw_dataset/encoder 다시 요구
        ds2 = _SyntheticRawDataset(n_entries=9)
        db = build_or_load_phase1_vector_db(
            tmp_path, raw_dataset=ds2, encoder=enc,
            observation_loader=ds2.load_observation, rebuild=True,
        )
        assert db.total_size() == 9

    def test_missing_inputs_raises(self, tmp_path):
        # 캐시 없는데 raw_dataset/dataset_path 모두 없음
        with pytest.raises(ValueError, match="dataset_path"):
            build_or_load_phase1_vector_db(tmp_path)

    def test_missing_encoder_raises(self, tmp_path):
        ds = _SyntheticRawDataset(n_entries=3)
        # raw_dataset 만 주고 encoder/vla_path 둘 다 안 줌
        with pytest.raises(ValueError, match="vla_path"):
            build_or_load_phase1_vector_db(tmp_path, raw_dataset=ds)

    def test_skill_partitioning_preserved(self, tmp_path):
        # entry 들이 skill 별로 분리되는지 — SkillVectorDB 의 기본 contract.
        ds = _SyntheticRawDataset(n_entries=6)
        enc = MeanPoolStateEncoder(out_dim=8)
        db = build_or_load_phase1_vector_db(
            tmp_path, raw_dataset=ds, encoder=enc,
            observation_loader=ds.load_observation,
        )
        # 6 entries, 3 skills round-robin → skill 당 2 entries
        skills = sorted(db.skill_ids())
        assert skills == ["move", "move_and_close", "move_and_open"]
        for s in skills:
            assert db.size(s) == 2

    def test_meta_carries_phase1_marker(self, tmp_path):
        # vector DB entry 의 meta 가 §6 규약대로 phase=phase1 태깅 돼 있는지.
        ds = _SyntheticRawDataset(n_entries=3)
        enc = MeanPoolStateEncoder(out_dim=8)
        db = build_or_load_phase1_vector_db(
            tmp_path, raw_dataset=ds, encoder=enc,
            observation_loader=ds.load_observation,
        )
        entries = db.query_skill("move_and_open")
        assert all(e.meta.get("phase") == "phase1" for e in entries)
        assert all(e.meta.get("accepted_by") == "phase1_seed" for e in entries)
