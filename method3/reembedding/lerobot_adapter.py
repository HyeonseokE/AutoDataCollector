"""LeRobot dataset → §3.1 RawTrajectoryDataset-like view — 문서 §6.

Phase1 raw dataset 으로 LeRobot 데이터셋을 그대로 쓰는 adapter. seed_builder 가
요구하는 minimal interface(``__len__`` / ``get(idx)`` / ``pointer(idx)``) 만
충족하면 되므로 inherit 없이 duck-typing.

§3.1 매핑 (LeRobot column → RawDatasetEntry):

  ===================  ============================================
  RawDatasetEntry      LeRobot 출처
  ===================  ============================================
  episode_id           f"episode_{episode_index+1:02d}"
  phase                "phase1" (고정)
  skill_id             skill.type (없으면 "move")
  instruction          tasks (episode-level task string)
  subgoal              skill.goal_position.robot_xyzrpy[:3]
  time_index           frame_index_within_episode
  observation_ref      {"dataset_path", "global_idx"}  — pointer 만
  proprioception       observation.ee_pos.robot_xyzrpy or .state
  action_chunk         action[idx:idx+H]    (sliding window)
  raw_action_sequence  동일 (별도 raw 없음 → 같은 값)
  planner_type         "InterpPlan" (Phase1 기본)
  success_flag         True (TRUE 에피소드만 저장됐다고 가정)
  validity_flag        True
  ===================  ============================================

action_chunk window 가 episode 끝을 넘어가는 frame 은 entry 에서 제외한다
(``H=12`` 일 때 episode 길이 ≥ 12 이어야 1개 이상 entry 가 나온다).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from method3.storage.raw_dataset import RawDatasetEntry

# vendored lerobot 을 우선 path 에 둔다 (recorder.py 와 같은 패턴).
_LEROBOT_PATH = Path(__file__).resolve().parent.parent.parent / "lerobot" / "src"
if str(_LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_PATH))


_PROPRIO_KEYS = (
    # observation.state — VLA 가 학습한 그 분포 (so101 6-dim servo positions, norm~85).
    # DB / candidate / encoder 의 모든 state-space 가 *이 분포* 로 통일되어야 한다.
    "observation.state",                 # 우선순위 1
    "observation.ee_pos.robot_xyzrpy",   # legacy fallback — 학습 일관성 없으므로 권장 X
)
_OBSERVATION_KEYS = (
    "observation.images.top",            # 우선순위 1
    "observation.images.realsense",
    "observation.images.left_wrist",     # fallback
)
_SKILL_TYPE_KEYS = ("skill.type", "subtask.skill_type")
_SUBGOAL_KEYS = ("skill.goal_position.robot_xyzrpy", "skill.goal_position.joint")


class LeRobotPhase1RawAdapter:
    """LeRobot 데이터셋을 §3.1 RawTrajectoryDataset 형태로 노출하는 adapter.

    Phase1 raw dataset 이 별도 영속화돼 있지 않은 현재 파이프라인에서는 LeRobot
    데이터셋이 사실상 raw 역할을 한다. 이 adapter 가 그 위에 §3.1 view 를 씌워
    seed_builder 가 그대로 쓸 수 있게 한다.

    Usage::

        adapter = LeRobotPhase1RawAdapter("CoRL2026-CSI/pnp_phase1_30_table2")
        db = build_phase1_vector_db(adapter, encoder, adapter.load_observation)
    """

    def __init__(
        self,
        repo_id_or_path: str | Path,
        *,
        action_horizon: int = 50,
        proprio_key: str | None = None,
        observation_key: str | None = None,
        skill_id_default: str = "move",
        video_backend: str = "pyav",
    ) -> None:
        """
        Args:
            repo_id_or_path: LeRobot dataset repo_id ("user/name") 또는 local path.
            action_horizon: action_chunk window 길이 H (§4.2 / §7.3 의 H).
            proprio_key: proprioception 컬럼 override. None 이면 _PROPRIO_KEYS 순회.
            observation_key: VLA 입력 이미지 컬럼 override. None 이면 _OBSERVATION_KEYS 순회.
            skill_id_default: skill.type 컬럼 부재 시 사용할 기본 skill_id.
            video_backend: video decoder. 기본 ``"pyav"`` — 시스템 libavutil 부재로
                ``torchcodec`` 가 로드 실패하는 환경에서도 동작. ``"torchcodec"`` 가
                필요하면 명시. LeRobotDataset 의 그것과 같은 의미.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME

        # repo_id 로 받으면 HF_LEROBOT_HOME 아래에서 찾고, 절대경로면 그대로.
        repo_path = Path(repo_id_or_path)
        if repo_path.is_absolute() and repo_path.exists():
            self._dataset = LeRobotDataset(
                repo_id=str(repo_path), root=str(repo_path),
                video_backend=video_backend,
            )
        else:
            self._dataset = LeRobotDataset(
                repo_id=str(repo_id_or_path), video_backend=video_backend,
            )
        self._H = int(action_horizon)
        self._proprio_key = proprio_key or self._pick_first_present(_PROPRIO_KEYS)
        self._observation_key = observation_key or self._pick_first_present(_OBSERVATION_KEYS)
        self._skill_id_default = skill_id_default
        # episode_id ↔ (start_global_idx, length). _index[i] = (global_idx, ep_num, frame_in_ep).
        self._index = self._build_index()
        self._dataset_path = str(getattr(self._dataset, "root", repo_id_or_path))

    # ─────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────
    def _pick_first_present(self, keys: tuple[str, ...]) -> str | None:
        feats = set(self._dataset.features.keys())
        for k in keys:
            if k in feats:
                return k
        return None

    def _build_index(self) -> list[tuple[int, int, int]]:
        """frame 단위 entry list — action_chunk window 가 episode 안에 들어가는 것만.

        Returns: [(global_idx, episode_index, frame_in_episode), ...]
        """
        ep_meta = self._dataset.meta.episodes
        # LeRobot v3.0 에서 meta.episodes 가 pandas DataFrame → datasets.Dataset
        # 로 바뀐 영향: pandas 의 ``iterrows()`` 는 datasets.Dataset 에 없다.
        # 두 가지 다 지원하도록 row iterator 를 통일한다.
        if hasattr(ep_meta, "iterrows"):
            rows = (row for _, row in ep_meta.iterrows())
        elif hasattr(ep_meta, "to_pandas"):
            rows = (row for _, row in ep_meta.to_pandas().iterrows())
        else:
            # datasets.Dataset 또는 list[dict] — row 가 dict-like.
            rows = iter(ep_meta)

        # episodes parquet — column 명이 LeRobot 버전마다 다를 수 있어 안전 접근.
        from_col = "dataset_from_index"
        to_col = "dataset_to_index"
        idx_col = "episode_index"
        out: list[tuple[int, int, int]] = []
        for row in rows:
            ei = int(row[idx_col])
            f0 = int(row[from_col])
            f1 = int(row[to_col])
            # action_chunk window 가 episode 끝을 넘지 않도록 [f0, f1-H+1).
            last = f1 - self._H + 1
            for f in range(f0, max(f0, last)):
                out.append((f, ei, f - f0))
        return out

    # ─────────────────────────────────────────────
    # RawTrajectoryDataset-compatible interface
    # ─────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._index)

    def get(self, idx: int) -> RawDatasetEntry:
        """idx 의 entry 를 §3.1 RawDatasetEntry 로 반환."""
        global_idx, ep_idx, frame_in_ep = self._index[idx]
        global_idx = min(global_idx, len(self._dataset) - 1); frame = self._dataset[global_idx]

        proprio = self._extract_proprio(frame)
        action_chunk = self._extract_action_chunk(global_idx)
        skill_id = self._extract_skill_id(frame)
        instruction = self._extract_instruction(global_idx)
        subgoal = self._extract_subgoal(frame)

        return RawDatasetEntry(
            episode_id=f"episode_{ep_idx + 1:02d}",
            phase="phase1",
            skill_id=str(skill_id),
            instruction=str(instruction),
            subgoal=subgoal,
            time_index=int(frame_in_ep),
            observation_ref={
                "dataset_path": self._dataset_path,
                "global_idx": int(global_idx),
                "key": self._observation_key,
            },
            proprioception=proprio,
            action_chunk=action_chunk,
            raw_action_sequence=action_chunk.copy(),
            planner_type="InterpPlan",
            success_flag=True,
            validity_flag=True,
        )

    def pointer(self, idx: int) -> dict:
        """§3.1 dataset_ref pointer (vector DB ref 로 들어감)."""
        global_idx, ep_idx, frame_in_ep = self._index[idx]
        return {
            "dataset_path": self._dataset_path,
            "global_idx": int(global_idx),
            "episode_id": f"episode_{ep_idx + 1:02d}",
            "time_index": int(frame_in_ep),
        }

    # ─────────────────────────────────────────────
    # observation_loader (seed_builder 가 호출)
    # ─────────────────────────────────────────────
    def load_observation(self, observation_ref: dict) -> dict[str, np.ndarray]:
        """observation_ref → raw image dict (VLA encoder 입력 형식).

        preselective_filter VLAKeyExtractor 는 ``{camera_name: HxWxC uint8}`` 을
        받는다. LeRobot 은 CHW float [0,1] 로 주므로 변환한다.
        """
        global_idx = int(observation_ref["global_idx"])
        global_idx = min(global_idx, len(self._dataset) - 1); frame = self._dataset[global_idx]
        # 모든 image feature 를 raw 형식으로 변환해 dict 로 묶는다.
        out: dict[str, np.ndarray] = {}
        for feat in self._dataset.features:
            if not feat.startswith("observation.images."):
                continue
            arr = frame[feat]
            # torch.Tensor 또는 ndarray (CHW float [0,1]) → HWC uint8
            if hasattr(arr, "numpy"):
                arr = arr.numpy()
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            cam_name = feat[len("observation.images."):]
            out[cam_name] = arr
        return out

    # ─────────────────────────────────────────────
    # internal extractors
    # ─────────────────────────────────────────────
    def _extract_proprio(self, frame) -> np.ndarray:
        if self._proprio_key and self._proprio_key in frame:
            v = frame[self._proprio_key]
            if hasattr(v, "numpy"):
                v = v.numpy()
            return np.asarray(v, dtype=np.float64).reshape(-1)
        return np.zeros(7, dtype=np.float64)

    def _extract_action_chunk(self, global_idx: int) -> np.ndarray:
        # H-step action window. dataset[idx:idx+H] 단건 슬라이스가 안 되므로
        # frame 별로 모은다.
        chunks = []
        for i in range(self._H):
            f = self._dataset[global_idx + i]
            a = f["action"]
            if hasattr(a, "numpy"):
                a = a.numpy()
            chunks.append(np.asarray(a, dtype=np.float64).reshape(-1))
        return np.stack(chunks)  # (H, action_dim)

    def _extract_skill_id(self, frame) -> Any:
        for k in _SKILL_TYPE_KEYS:
            if k in frame:
                v = frame[k]
                if hasattr(v, "item"):
                    v = v.item()
                return v if v else self._skill_id_default
        return self._skill_id_default

    def _extract_instruction(self, global_idx: int) -> str:
        # tasks 는 episode-level. dataset[idx]["task"] 또는 meta.tasks 에서 조회.
        global_idx = min(global_idx, len(self._dataset) - 1); frame = self._dataset[global_idx]
        if "task" in frame:
            t = frame["task"]
            return str(t.item() if hasattr(t, "item") else t)
        return ""

    def _extract_subgoal(self, frame) -> np.ndarray:
        for k in _SUBGOAL_KEYS:
            if k in frame:
                v = frame[k]
                if hasattr(v, "numpy"):
                    v = v.numpy()
                v = np.asarray(v, dtype=np.float64).reshape(-1)
                return v[:3] if v.size >= 3 else np.pad(v, (0, max(0, 3 - v.size)))
        return np.zeros(3, dtype=np.float64)
