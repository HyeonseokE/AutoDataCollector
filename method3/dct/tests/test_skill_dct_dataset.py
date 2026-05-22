"""SkillDCTDataset — base wrap + action 교체 + language prefix."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from method3.dct.skill_dct_dataset import SkillDCTDataset


class _FakeBaseDataset:
    """LeRobotDataset 의 frame-단위 __getitem__ 만 mimic 한 mock."""

    def __init__(self):
        # segment 1: frame 0..30, segment 2: frame 30..45.
        self.features = {"observation.images.top", "observation.state", "action"}

    def __getitem__(self, idx):
        # 모든 frame 에 대해 deterministic mock.
        return {
            "observation.images.top": torch.full((3, 224, 224), float(idx)),
            "observation.state": torch.full((6,), float(idx)),
            "action": torch.zeros(50, 6),
            "task": "pick and place",
            "actions_id_pad": torch.zeros(50, dtype=torch.bool),
        }


@pytest.fixture
def sidecar_parquet(tmp_path):
    rng = np.random.default_rng(0)
    target1 = rng.normal(size=(50, 6))
    target2 = rng.normal(size=(50, 6))
    records = [
        {
            "episode_id": "episode_01",
            "skill_index": 0,
            "skill_type": "move_initial",
            "instruction": "pick and place",
            "frame_start": 0,
            "frame_end": 30,
            "dct_shape": [50, 6],
            "dct_target": target1.flatten().tolist(),
        },
        {
            "episode_id": "episode_01",
            "skill_index": 1,
            "skill_type": "gripper_close",
            "instruction": "pick and place",
            "frame_start": 30,
            "frame_end": 45,
            "dct_shape": [50, 6],
            "dct_target": target2.flatten().tolist(),
        },
    ]
    path = tmp_path / "skill_dct.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    return path, target1, target2


def test_length_matches_segment_count(sidecar_parquet):
    path, _, _ = sidecar_parquet
    ds = SkillDCTDataset(_FakeBaseDataset(), path)
    assert len(ds) == 2


def test_action_replaced_with_dct_target(sidecar_parquet):
    path, target1, target2 = sidecar_parquet
    ds = SkillDCTDataset(_FakeBaseDataset(), path)

    item0 = ds[0]
    assert item0["action"].shape == (50, 6)
    np.testing.assert_allclose(item0["action"].numpy(), target1, atol=1e-6)

    item1 = ds[1]
    np.testing.assert_allclose(item1["action"].numpy(), target2, atol=1e-6)


def test_task_prefixed_with_skill_type(sidecar_parquet):
    path, _, _ = sidecar_parquet
    ds = SkillDCTDataset(_FakeBaseDataset(), path)
    assert ds[0]["task"] == "move_initial: pick and place"
    assert ds[1]["task"] == "gripper_close: pick and place"


def test_actions_pad_all_valid(sidecar_parquet):
    path, _, _ = sidecar_parquet
    ds = SkillDCTDataset(_FakeBaseDataset(), path)
    item = ds[0]
    pad = item["actions_id_pad"]
    assert pad.shape == (50,)
    assert not pad.any()


def test_observation_columns_preserved(sidecar_parquet):
    path, _, _ = sidecar_parquet
    ds = SkillDCTDataset(_FakeBaseDataset(), path)
    item0 = ds[0]
    item1 = ds[1]
    assert "observation.images.top" in item0
    assert "observation.state" in item0
    # frame_start=0 vs 30 의 base 값 차이가 보존돼야 함 (mock 이 frame_idx 를
    # obs.state 에 그대로 넣음).
    assert torch.equal(item0["observation.state"], torch.zeros(6))
    assert torch.equal(item1["observation.state"], torch.full((6,), 30.0))


def test_segment_meta_attached(sidecar_parquet):
    path, _, _ = sidecar_parquet
    ds = SkillDCTDataset(_FakeBaseDataset(), path)
    meta = ds[0]["_skill_segment_meta"]
    assert meta["episode_id"] == "episode_01"
    assert meta["skill_index"] == 0
    assert meta["skill_type"] == "move_initial"
    assert meta["frame_start"] == 0
    assert meta["frame_end"] == 30


def test_base_attribute_delegated(sidecar_parquet):
    path, _, _ = sidecar_parquet
    ds = SkillDCTDataset(_FakeBaseDataset(), path)
    # base.features 가 wrapper 를 통해 접근 가능해야 lerobot trainer 가
    # dataset.features 등을 직접 호출할 때 동작.
    assert hasattr(ds, "features")
    assert "observation.images.top" in ds.features
