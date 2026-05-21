"""skill_dataset — _run_length_segments + sidecar parquet I/O."""
from __future__ import annotations

import numpy as np
import pytest

from method3.dct.skill_dataset import (
    SkillSegment,
    _run_length_segments,
    build_dct_targets,
    load_dct_targets,
)


class TestRunLengthSegments:
    def test_empty(self):
        assert _run_length_segments([]) == []

    def test_single(self):
        assert _run_length_segments(["a"]) == [(0, 1, "a")]

    def test_all_same(self):
        assert _run_length_segments(["a", "a", "a"]) == [(0, 3, "a")]

    def test_alternating(self):
        assert _run_length_segments(["a", "b", "a"]) == [
            (0, 1, "a"), (1, 2, "b"), (2, 3, "a"),
        ]

    def test_typical_episode(self):
        # 9 skills/episode 시나리오 (사용자 description 의 typical Phase1 episode).
        s = (
            ["move_initial"] * 30
            + ["move_free"] * 30
            + ["move_and_open"] * 35
            + ["gripper_open"] * 20
            + ["move"] * 100
            + ["gripper_close"] * 15
            + ["move_and_close"] * 35
            + ["move"] * 28
        )
        segs = _run_length_segments(s)
        assert len(segs) == 8
        assert segs[0] == (0, 30, "move_initial")
        assert segs[-1] == (sum(len(g) for g in [
            ["move_initial"] * 30,
            ["move_free"] * 30,
            ["move_and_open"] * 35,
            ["gripper_open"] * 20,
            ["move"] * 100,
            ["gripper_close"] * 15,
            ["move_and_close"] * 35,
        ]), 293, "move")
        # frame indices align to total length.
        assert segs[-1][1] == len(s)


def test_sidecar_roundtrip(tmp_path):
    """build_dct_targets 의 parquet 출력을 load_dct_targets 가 정확히 복원."""
    # build_dct_targets 를 직접 부르려면 LeRobot dataset 이 필요 — 그 대신
    # parquet writer 부분만 검증한다: SkillSegment 합성 → parquet 직접 write →
    # load 결과 비교. load_dct_targets 의 shape/dtype 복원 정합성만 확인.
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    seg1_target = rng.normal(size=(50, 6))
    seg2_target = rng.normal(size=(50, 6))
    records = [
        {
            "episode_id": "episode_01",
            "skill_index": 0,
            "skill_type": "move_initial",
            "instruction": "pick and place",
            "frame_start": 0,
            "frame_end": 30,
            "dct_shape": [50, 6],
            "dct_target": seg1_target.flatten().tolist(),
        },
        {
            "episode_id": "episode_01",
            "skill_index": 1,
            "skill_type": "gripper_close",
            "instruction": "pick and place",
            "frame_start": 30,
            "frame_end": 45,
            "dct_shape": [50, 6],
            "dct_target": seg2_target.flatten().tolist(),
        },
    ]
    table = pa.Table.from_pylist(records)
    parquet_path = tmp_path / "skill_dct.parquet"
    pq.write_table(table, parquet_path)

    loaded = load_dct_targets(parquet_path)
    assert len(loaded) == 2
    assert loaded[0].episode_id == "episode_01"
    assert loaded[0].skill_index == 0
    assert loaded[0].skill_type == "move_initial"
    assert loaded[0].frame_start == 0
    assert loaded[0].frame_end == 30
    assert loaded[0].dct_target.shape == (50, 6)
    np.testing.assert_allclose(loaded[0].dct_target, seg1_target)
    np.testing.assert_allclose(loaded[1].dct_target, seg2_target)
    assert loaded[1].skill_type == "gripper_close"


def test_skill_segment_immutable():
    # @dataclass(frozen=True) — 사용자 immutability rule 준수.
    seg = SkillSegment(
        episode_id="episode_01",
        skill_index=0,
        skill_type="move",
        instruction="",
        frame_start=0,
        frame_end=10,
        dct_target=np.zeros((50, 6)),
    )
    with pytest.raises((AttributeError, Exception)):
        seg.skill_index = 1  # type: ignore
