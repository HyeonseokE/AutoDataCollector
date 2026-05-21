"""LeRobot dataset → skill-segment 단위 RawTrajectoryDataset adapter (DCT paradigm).

기존 ``LeRobotPhase1RawAdapter`` 는 한 entry = 한 frame (frame-level sliding
H-window chunk). DCT paradigm 의 P_phase1 은 한 entry = 한 skill segment 라
별도 adapter 가 필요하다.

핵심 매핑:
  RawDatasetEntry        ← skill segment 단위
  ───────────────────────────────────────────────
  skill_id               = skill_type
  instruction            = sidecar parquet 의 instruction (skill.natural_language
                            우선, 없으면 episode task)
  subgoal                = segment 시작 frame 의 skill.goal_position.robot_xyzrpy[:3]
  time_index             = skill_index (in episode)
  observation_ref        = pointer to segment 시작 frame
  proprioception         = segment 시작 frame 의 proprio
  action_chunk           = (T_skill, dof) raw action sequence — 길이 가변.
                            seed_builder 의 use_dct_target 분기가 이걸 traj_to_dct
                            로 (L0, dof) DCT_50 feature 로 변환.

video decode 는 ``load_observation`` 에서만 (segment 시작 frame 1개) → 297
segment ≈ 297 frame decode. action 은 raw parquet 직접 access 로 bulk 추출.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from method3.dct.skill_dataset import (
    _resolve_dataset_root,
    load_dct_targets,
)
from method3.reembedding.lerobot_adapter import LeRobotPhase1RawAdapter
from method3.storage.raw_dataset import RawDatasetEntry


class LeRobotPhase1SkillSegmentAdapter:
    """skill-segment 단위 RawTrajectoryDataset adapter (DCT paradigm).

    Composition: 내부에 ``LeRobotPhase1RawAdapter`` 를 가지고 image / proprio /
    subgoal 추출은 그대로 위임. action 만 sidecar parquet + raw data parquet
    bulk read 로 segment 단위 채움.

    Usage::

        adapter = LeRobotPhase1SkillSegmentAdapter(
            "CoRL2026-CSI/pnp_phase1_30_table2",
            skill_dct_parquet="results/skill_dct/pnp_phase1_30_table2.parquet",
        )
        db = build_phase1_vector_db(adapter, encoder, adapter.load_observation,
                                    ReembeddingConfig(use_dct_target=True))
    """

    def __init__(
        self,
        repo_id_or_path: str | Path,
        skill_dct_parquet: str | Path,
        *,
        proprio_key: str | None = None,
        observation_key: str | None = None,
        skill_id_default: str = "move",
        video_backend: str = "pyav",
    ) -> None:
        self._base = LeRobotPhase1RawAdapter(
            repo_id_or_path,
            action_horizon=2,        # sliding window 무시 (skill-segment adapter)
            proprio_key=proprio_key,
            observation_key=observation_key,
            skill_id_default=skill_id_default,
            video_backend=video_backend,
        )
        self.segments = load_dct_targets(skill_dct_parquet)
        self._dataset_path = self._base._dataset_path
        # raw action bulk pre-read — global frame index 정렬.
        self._raw_actions = self._bulk_read_actions(repo_id_or_path)

    @staticmethod
    def _bulk_read_actions(repo_id_or_path: str | Path) -> np.ndarray:
        """모든 frame 의 action column 만 raw parquet 에서 직접 read (video skip)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        root = _resolve_dataset_root(repo_id_or_path)
        data_files = sorted((root / "data").rglob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No data parquet under {root}/data/")
        tbl = pa.concat_tables(
            [pq.read_table(f, columns=["action", "index"]) for f in data_files]
        )
        tbl = tbl.sort_by("index")
        return np.array(tbl["action"].to_pylist(), dtype=np.float64)

    # ─────────────────────────────────────────────
    # RawTrajectoryDataset interface
    # ─────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.segments)

    def get(self, idx: int) -> RawDatasetEntry:
        seg = self.segments[idx]
        # segment 시작 frame 의 lerobot frame. *exclusive end 가 last frame+1 인
        # segment* 에 대비, 데이터셋 길이로 clamp.
        _ds_len = len(self._base._dataset)
        _fs = min(int(seg.frame_start), _ds_len - 1)
        frame = self._base._dataset[_fs]
        proprio = self._base._extract_proprio(frame)
        subgoal = self._base._extract_subgoal(frame)
        # action sequence (T_skill, dof) — bulk-read array 에서 slice.
        # empty segment 대비 — frame_start == frame_end 면 last available action 1-row.
        _fs2 = min(int(seg.frame_start), len(self._raw_actions) - 1)
        _fe2 = min(int(seg.frame_end), len(self._raw_actions))
        if _fe2 <= _fs2:
            action = self._raw_actions[_fs2: _fs2 + 1].copy()  # length-1 fallback
        else:
            action = self._raw_actions[_fs2:_fe2].copy()

        # instruction format SoT — 학습 / Phase2 inference 와 동일 분포.
        from method3.dct.instruction_format import format_skill_instruction
        _formatted_instr = format_skill_instruction(
            str(seg.skill_type), str(seg.instruction),
        )
        return RawDatasetEntry(
            episode_id=seg.episode_id,
            phase="phase1",
            skill_id=str(seg.skill_type),
            instruction=_formatted_instr,
            subgoal=subgoal,
            time_index=int(seg.skill_index),
            observation_ref={
                "dataset_path": self._dataset_path,
                "global_idx": int(seg.frame_start),
                "key": self._base._observation_key,
            },
            proprioception=proprio,
            action_chunk=action,
            raw_action_sequence=action.copy(),
            planner_type="InterpPlan",
            success_flag=True,
            validity_flag=True,
        )

    def pointer(self, idx: int) -> dict:
        seg = self.segments[idx]
        return {
            "dataset_path": self._dataset_path,
            "global_idx": int(seg.frame_start),
            "episode_id": seg.episode_id,
            "time_index": int(seg.skill_index),
            "skill_type": seg.skill_type,
            "frame_end": int(seg.frame_end),
        }

    def load_observation(self, observation_ref: dict) -> Any:
        """위임 — segment 시작 frame 의 image dict."""
        return self._base.load_observation(observation_ref)
