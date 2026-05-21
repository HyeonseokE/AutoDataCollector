"""skill-unit DCT 학습용 PyTorch Dataset wrapper.

사용자 명시 paradigm step [1][2]:
  VLA input  = (observation(frame_start), language, skill_type)
  VLA target = DCT_50 feature of skill trajectory

LeRobotDataset 위에 sidecar parquet (build_skill_dct.py 산물) 을 join 하여,
한 sample = 한 skill segment 가 되도록 노출한다. smolvla forward 의
``actions`` 위치에 DCT target (shape (L0, action_dim) = chunk_size 와
동일) 을 그대로 넣어 학습.

핵심:
  - ``__getitem__(idx)`` 는 segment ``frame_start`` frame 의 LeRobotDataset
    item 을 base 에서 가져와 ``action`` 자리에 DCT target 으로 교체.
  - ``task`` (language) 에 skill_type prefix 를 붙여 model input 에 노출.
  - delta_timestamps / features / video_backend 등 lerobot 의존 attribute
    는 모두 base 에 위임 (lerobot trainer 와 인터페이스 호환).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from method3.dct.skill_dataset import SkillSegment, load_dct_targets


class SkillDCTDataset:
    """LeRobotDataset 을 (episode, skill) 단위로 노출하는 thin wrapper.

    Standard ``torch.utils.data.Dataset`` 프로토콜 (``__len__`` /
    ``__getitem__``) 을 따른다. lerobot 의 ``LeRobotDataset`` 가 아니므로
    train script 측에서 dataset factory 분기 필요.

    두 sample 단위 mode:
      * ``frame_mode=False`` — 한 sample = 한 skill segment. obs 는 segment
        시작 frame 1개. 학습 sample 수 = segment 수 (~297 for 30 episode).
      * ``frame_mode=True`` (default) — 한 sample = 한 frame. obs 는 그
        frame 의 raw obs, target 은 그 frame 이 속한 skill 의 DCT_50.
        학습 sample 수 = 전체 frame 수 (~8700 for 30 episode). 같은 target
        을 progress 가 다른 obs 에서 학습 → 자연스러운 progress-conditioned
        augmentation 효과.
    """

    def __init__(
        self,
        base_dataset: Any,
        skill_dct_parquet: str | Path,
        *,
        action_key: str = "action",
        task_key: str = "task",
        actions_pad_key: str = "actions_id_pad",
        skill_type_prefix_format: str = "{skill_type}: {instruction}",
        frame_mode: bool = True,
    ) -> None:
        """
        Args:
            base_dataset: ``LeRobotDataset`` 인스턴스 (또는 동등 인터페이스 —
                ``__getitem__`` 이 frame 단위 dict 를 반환).
            skill_dct_parquet: ``build_dct_targets`` 가 만든 sidecar parquet.
            action_key: base item 의 action 컬럼 이름.
            task_key: base item 의 language/task 컬럼 이름.
            actions_pad_key: base item 의 action pad mask 컬럼 이름.
            skill_type_prefix_format: language 에 skill_type 을 noisy
                prefix 로 inject 하는 format string. ``{skill_type}`` 과
                ``{instruction}`` 를 키로 받는다.
            frame_mode: True 면 frame 단위 sample (default), False 면
                segment 단위 sample. 자세한 의미는 class docstring 참고.
        """
        self.base = base_dataset
        self.segments: list[SkillSegment] = load_dct_targets(skill_dct_parquet)
        self._action_key = action_key
        self._task_key = task_key
        self._pad_key = actions_pad_key
        self._prefix_fmt = skill_type_prefix_format
        self._frame_mode = bool(frame_mode)
        # frame_mode=True 일 때 lookup table: idx → (frame_idx, seg_idx).
        # segment.frame_start ≤ frame_idx < segment.frame_end.
        self._frame_to_seg: list[tuple[int, int]] | None
        if self._frame_mode:
            self._frame_to_seg = [
                (f, seg_idx)
                for seg_idx, seg in enumerate(self.segments)
                for f in range(seg.frame_start, seg.frame_end)
            ]
        else:
            self._frame_to_seg = None

    # ─────────────────────────────────────────────
    # base attribute 위임 — features, meta, delta_timestamps 등
    # lerobot trainer 가 dataset 에서 직접 읽는 항목들.
    # ─────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        # __init__ 후 호출되는 attribute 접근. base 로 fallback.
        return getattr(self.base, name)

    def __len__(self) -> int:
        if self._frame_to_seg is not None:
            return len(self._frame_to_seg)
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # frame_mode 분기 — frame 단위 / segment 단위.
        if self._frame_to_seg is not None:
            frame_idx, seg_idx = self._frame_to_seg[idx]
            seg = self.segments[seg_idx]
        else:
            seg = self.segments[idx]
            frame_idx = seg.frame_start

        item = self.base[frame_idx]
        # action 자리에 DCT target 으로 교체 — shape (L0, action_dim).
        target = torch.as_tensor(seg.dct_target, dtype=torch.float32)
        item = dict(item)  # shallow copy — original 보호
        item[self._action_key] = target
        # all-valid pad mask: DCT target 은 frame pad 없음.
        item[self._pad_key] = torch.zeros(target.shape[0], dtype=torch.bool)
        # language 에 skill_type prefix.
        cur_task = item.get(self._task_key, "")
        if hasattr(cur_task, "item"):
            cur_task = cur_task.item()
        item[self._task_key] = self._prefix_fmt.format(
            skill_type=seg.skill_type, instruction=str(cur_task)
        )
        # segment metadata (debug / sample logging 용).
        item["_skill_segment_meta"] = {
            "episode_id": seg.episode_id,
            "skill_index": seg.skill_index,
            "skill_type": seg.skill_type,
            "frame_start": seg.frame_start,
            "frame_end": seg.frame_end,
            "frame_idx": frame_idx,
        }
        return item
