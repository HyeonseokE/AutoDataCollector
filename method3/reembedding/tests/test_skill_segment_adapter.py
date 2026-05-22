"""LeRobotPhase1SkillSegmentAdapter — schema + sidecar wire-up."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from method3.storage.raw_dataset import RawDatasetEntry


@pytest.fixture
def fake_sidecar(tmp_path):
    rng = np.random.default_rng(0)
    records = []
    for i in range(3):
        target = rng.normal(size=(50, 6)).astype(np.float64)
        records.append({
            "episode_id": f"episode_{i+1:02d}",
            "skill_index": 0,
            "skill_type": "move",
            "instruction": "do",
            "frame_start": i * 30,
            "frame_end": i * 30 + 20,
            "dct_shape": [50, 6],
            "dct_target": target.flatten().tolist(),
        })
    path = tmp_path / "sidecar.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    return path


def test_adapter_uses_segment_count(fake_sidecar, monkeypatch):
    """__len__ 이 segment 수, get() 이 RawDatasetEntry 반환하는지 — base 의 lerobot
    dataset 부분은 mock 해 격리."""
    from method3.reembedding import lerobot_skill_segment_adapter as mod

    # _bulk_read_actions + LeRobotPhase1RawAdapter 모두 stub.
    class _FakeBase:
        def __init__(self):
            self._dataset_path = "/fake/dataset"
            self._observation_key = "observation.images.top"
            self._dataset = self  # _dataset[idx] 로 접근

        def __len__(self):
            return 200  # fake dataset 길이 — _ds_len = len(self._base._dataset) 에서 사용

        def __getitem__(self, idx):
            # frame stub.
            return {}

        def _extract_proprio(self, frame):
            return np.zeros(6)

        def _extract_subgoal(self, frame):
            return np.array([0.1, 0.2, 0.3])

        def load_observation(self, ref):
            return {"top": np.zeros((224, 224, 3), dtype=np.uint8)}

    fake_actions = np.arange(200 * 6, dtype=np.float64).reshape(200, 6)
    monkeypatch.setattr(mod.LeRobotPhase1RawAdapter, "__init__",
                        lambda self, *a, **k: None)

    adapter = mod.LeRobotPhase1SkillSegmentAdapter.__new__(
        mod.LeRobotPhase1SkillSegmentAdapter)
    adapter._base = _FakeBase()
    from method3.dct.skill_dataset import load_dct_targets
    adapter.segments = load_dct_targets(fake_sidecar)
    adapter._dataset_path = adapter._base._dataset_path
    adapter._raw_actions = fake_actions

    assert len(adapter) == 3
    entry = adapter.get(0)
    assert isinstance(entry, RawDatasetEntry)
    # skill_id = "skill_{skill_index}" (ordinal key), skill.type 은 pointer() 에서 확인.
    assert entry.skill_id == "skill_0"
    assert entry.episode_id == "episode_01"
    assert entry.time_index == 0
    # action_chunk = raw_actions[frame_start:frame_end][:, :5] — arm-only (gripper 제외).
    assert entry.action_chunk.shape == (20, 5)
    np.testing.assert_allclose(entry.action_chunk, fake_actions[0:20, :5])

    entry2 = adapter.get(2)
    # frame_start = 60, frame_end = 80
    assert entry2.action_chunk.shape == (20, 5)
    np.testing.assert_allclose(entry2.action_chunk, fake_actions[60:80, :5])


def test_pointer_includes_skill_meta(fake_sidecar, monkeypatch):
    from method3.reembedding import lerobot_skill_segment_adapter as mod
    monkeypatch.setattr(mod.LeRobotPhase1RawAdapter, "__init__",
                        lambda self, *a, **k: None)

    class _FakeBase:
        _dataset_path = "/fake/dataset"
        _observation_key = "observation.images.top"

    adapter = mod.LeRobotPhase1SkillSegmentAdapter.__new__(
        mod.LeRobotPhase1SkillSegmentAdapter)
    adapter._base = _FakeBase()
    from method3.dct.skill_dataset import load_dct_targets
    adapter.segments = load_dct_targets(fake_sidecar)
    adapter._dataset_path = adapter._base._dataset_path
    adapter._raw_actions = np.zeros((200, 6))

    p = adapter.pointer(1)
    assert p["skill_type"] == "move"
    assert p["frame_end"] == 50  # ep 2: frame_start=30, end=50
    assert p["global_idx"] == 30
