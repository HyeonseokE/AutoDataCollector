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
                            seed_builder 가 이걸 traj_to_dct
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
                                    ReembeddingConfig())
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
        # transit-only filter + per-episode re-index — subgoal_buffer (Phase1
        # SubgoalSelector.stage_executed) 와 같은 namespace 로 통일. gripper_*
        # / move_free 같은 non-transit segment 는 buffer 에 stage 되지 않으므로
        # DB 도 동일하게 빼야 client Phase2SubgoalReplay 의 ordinal lookup 이
        # 정합한다. 빠진 segment 의 skill_index 자리를 채우기 위해 per-episode
        # 0-based 로 재할당한다 — client `_skill_ordinal` (transit-call counter)
        # 와 같은 0..N-1 namespace.
        from dataclasses import replace as _dc_replace
        _TRANSIT = {"move", "move_initial", "move_and_open", "move_and_close"}
        _raw_segments = load_dct_targets(skill_dct_parquet)
        _filtered: list = []
        _ep_idx: dict = {}
        for _seg in _raw_segments:
            if _seg.skill_type not in _TRANSIT:
                continue
            _i = _ep_idx.get(_seg.episode_id, 0)
            _filtered.append(_dc_replace(_seg, skill_index=_i))
            _ep_idx[_seg.episode_id] = _i + 1
        self.segments = _filtered
        print(f"[reembed-adapter] transit-only filter: {len(_raw_segments)} "
              f"→ {len(_filtered)} segments ({len(_ep_idx)} episodes, "
              f"max ordinal={max(_ep_idx.values(), default=0) - 1})")
        self._dataset_path = self._base._dataset_path
        # raw action bulk pre-read — global frame index 정렬.
        self._raw_actions = self._bulk_read_actions(repo_id_or_path)
        # EE pose bulk pre-read — observation.ee_pos.robot_xyzrpy (T, 6) [xyz+rpy].
        # 데이터셋 수집 시점에 robot 의 FK 로 기록된 ground-truth pose. seed_builder
        # 가 np.diff + traj_to_dct 로 EE delta DCT 만든다 (FK 불필요, 동일 source).
        self._raw_ee = self._bulk_read_ee_poses(repo_id_or_path)

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

    @staticmethod
    def _bulk_read_ee_poses(repo_id_or_path: str | Path) -> np.ndarray:
        """모든 frame 의 observation.ee_pos.robot_xyzrpy → (N, 6) bulk read.

        column 없으면 (0, 6) 빈 배열 반환 (legacy dataset 호환 — seed_builder
        가 fallback 로 joint DCT 사용).
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        root = _resolve_dataset_root(repo_id_or_path)
        data_files = sorted((root / "data").rglob("*.parquet"))
        if not data_files:
            return np.empty((0, 6), dtype=np.float64)
        # column 존재 확인 — 첫 파일 schema 만 본다.
        col = "observation.ee_pos.robot_xyzrpy"
        first_schema = pq.read_table(data_files[0], columns=None).schema
        if col not in first_schema.names:
            print(f"[adapter] {col} column 없음 — EE delta DCT 비활성 "
                  f"(seed_builder 가 joint DCT fallback)")
            return np.empty((0, 6), dtype=np.float64)
        tbl = pa.concat_tables(
            [pq.read_table(f, columns=[col, "index"]) for f in data_files]
        )
        tbl = tbl.sort_by("index")
        return np.array(tbl[col].to_pylist(), dtype=np.float64)

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

        # arm-only 차원 정합 — so101 의 observation.state / action 은 6축
        # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
        # gripper]. Phase2 candidate 는 curobo arm-only (5축) 이므로 DB 도
        # gripper 축(마지막)을 *제외하고* build 해 차원을 통일한다.
        # traj_to_dct 는 per-axis 변환이라 '사전 제외 == 사후 slice' 지만,
        # DB 를 처음부터 arm-only 로 저장하면 mi_selector 의 slice 우회가
        # 통째로 사라진다 (저장 250/965, mismatch 위험 0).
        _ARM_DOF = 5
        proprio = np.asarray(proprio)
        if proprio.ndim == 1 and proprio.shape[0] > _ARM_DOF:
            proprio = proprio[:_ARM_DOF]
        if action.ndim == 2 and action.shape[1] > _ARM_DOF:
            action = action[:, :_ARM_DOF]

        # EE pose chunk (T_skill, 6) — segment frame 범위에서 추출. seed_builder
        # 가 np.diff + traj_to_dct 로 EE delta DCT 만든다 (translation invariant).
        # test/mock 호환을 위해 _raw_ee 부재 시 빈 배열로 안전 fallback.
        _ee_arr = getattr(self, "_raw_ee", np.empty((0, 6), dtype=np.float64))
        if _ee_arr.size > 0 and _ee_arr.ndim == 2 and _ee_arr.shape[1] == 6:
            _es = min(int(seg.frame_start), len(_ee_arr) - 1)
            _ee_end = min(int(seg.frame_end), len(_ee_arr))
            if _ee_end <= _es:
                ee_chunk = _ee_arr[_es: _es + 1].copy()  # 1-row fallback
            else:
                ee_chunk = _ee_arr[_es:_ee_end].copy()
        else:
            ee_chunk = np.empty((0, 6), dtype=np.float64)

        # VLA instruction — 학습(SkillDCTDataset)이 episode-level task 를 쓰므로
        # re-embedding 입력도 동일하게 episode task 로 맞춘다 (Option A: VLA
        # 입력은 {skill_type}: {episode_task} 고정 — per-skill nl 아님).
        from method3.dct.instruction_format import format_skill_instruction
        _ep_task = ""
        if "task" in frame:
            _t = frame["task"]
            _ep_task = str(_t.item() if hasattr(_t, "item") else _t)
        _formatted_instr = format_skill_instruction(str(seg.skill_type), _ep_task)
        return RawDatasetEntry(
            episode_id=seg.episode_id,
            phase="phase1",
            # skill_id = episode-내 ordinal key (skill_0, skill_1, ...).
            # skill.type 는 pointer() 의 ref 로 보존 (변동 정보가 아닌 안정 키).
            skill_id=f"skill_{int(seg.skill_index)}",
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
            ee_chunk=ee_chunk,
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
