"""skill 단위 trajectory → DCT feature 추출 + sidecar 저장.

사용자 명시 paradigm step [1]:
  Phase1 LeRobot v3 dataset 의 skill boundary 를 활용해 episode 별 skill
  segment 를 추출 → 각 segment 의 (T_skill, action_dim) action trajectory
  를 ``traj_to_dct`` 로 (L0, action_dim) DCT feature 로 변환 → sidecar
  parquet 으로 저장.

VLA 학습 sample 단위 (step [1][2]):
  (episode_id, skill_index) — 한 skill = 한 sample.
    input  = observation(frame_start), language, skill_type
    target = dct_target (L0, action_dim)

sidecar parquet schema:
  episode_id (str)
  skill_index (int)        — 0-based within episode
  skill_type (str)         — skill.type column 값
  instruction (str)        — episode-level task 또는 skill.natural_language
  frame_start (int)        — global frame index, exclusive 의 시작
  frame_end (int)          — exclusive
  dct_target (list[float]) — (L0 * action_dim,) flatten
  dct_shape (list[int])    — [L0, action_dim]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from method3.dct.transform import traj_to_dct


@dataclass(frozen=True)
class SkillSegment:
    """한 (episode, skill) 단위의 DCT target + metadata."""

    episode_id: str
    skill_index: int
    skill_type: str
    instruction: str
    frame_start: int
    frame_end: int
    dct_target: np.ndarray  # (L0, action_dim)


def _run_length_segments(skill_types: list[str]) -> list[tuple[int, int, str]]:
    """[s0, s0, s1, s1, s2] → [(0, 2, s0), (2, 4, s1), (4, 5, s2)].

    Args:
        skill_types: frame 순서대로 나열된 skill.type 값들.

    Returns:
        ``[(start_idx, end_idx_exclusive, skill_type), ...]``.
    """
    if not skill_types:
        return []
    out: list[tuple[int, int, str]] = []
    seg_start = 0
    prev = skill_types[0]
    for i in range(1, len(skill_types)):
        if skill_types[i] != prev:
            out.append((seg_start, i, prev))
            seg_start = i
            prev = skill_types[i]
    out.append((seg_start, len(skill_types), prev))
    return out


def _load_lerobot_dataset(dataset_path: str | Path):
    """vendored lerobot 경로 셋업 후 LeRobotDataset 로드."""
    _LEROBOT_PATH = Path(__file__).resolve().parent.parent.parent / "lerobot" / "src"
    if str(_LEROBOT_PATH) not in sys.path:
        sys.path.insert(0, str(_LEROBOT_PATH))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_path = Path(dataset_path)
    if repo_path.is_absolute() and repo_path.exists():
        return LeRobotDataset(
            repo_id=str(repo_path), root=str(repo_path), video_backend="pyav"
        )
    return LeRobotDataset(repo_id=str(dataset_path), video_backend="pyav")


_SKILL_TYPE_KEYS = ("skill.type", "subtask.skill_type")


def _pick_skill_type_key(features) -> str:
    for k in _SKILL_TYPE_KEYS:
        if k in features:
            return k
    raise RuntimeError(
        f"No skill.type column in dataset features; tried {_SKILL_TYPE_KEYS}"
    )


def _episode_rows(ep_meta):
    """LeRobot v3 meta.episodes 의 row iterator (pandas / datasets / list 모두 지원)."""
    if hasattr(ep_meta, "iterrows"):
        return (row for _, row in ep_meta.iterrows())
    if hasattr(ep_meta, "to_pandas"):
        return (row for _, row in ep_meta.to_pandas().iterrows())
    return iter(ep_meta)


def _scalar(v):
    if hasattr(v, "item"):
        return v.item()
    return v


def iter_skill_segments(
    dataset_path: str | Path,
    *,
    L0: int = 50,
) -> Iterator[SkillSegment]:
    """LeRobot v3 dataset → (episode, skill) 단위 iteration.

    skill boundary 는 ``skill.type`` column 의 run-length encoding 으로 결정.
    각 segment 의 action sequence → ``traj_to_dct`` → (L0, action_dim).

    Args:
        dataset_path: LeRobot dataset repo_id 또는 절대 경로.
        L0: DCT 출력 차원 (default 50).

    Yields:
        ``SkillSegment``.
    """
    ds = _load_lerobot_dataset(dataset_path)
    skill_key = _pick_skill_type_key(ds.features)

    for row in _episode_rows(ds.meta.episodes):
        ei = int(row["episode_index"])
        f0 = int(row["dataset_from_index"])
        f1 = int(row["dataset_to_index"])
        episode_id = f"episode_{ei + 1:02d}"

        # episode 내 frame 별 skill.type 수집.
        skill_types: list[str] = []
        for f in range(f0, f1):
            v = _scalar(ds[f][skill_key])
            skill_types.append(str(v) if v else "move")

        segments = _run_length_segments(skill_types)

        for skill_idx, (s, e, sk) in enumerate(segments):
            actions = _collect_actions(ds, f0 + s, f0 + e)
            instruction = _get_instruction(ds, f0 + s)
            dct_target = traj_to_dct(actions, L0=L0)
            yield SkillSegment(
                episode_id=episode_id,
                skill_index=skill_idx,
                skill_type=sk,
                instruction=instruction,
                frame_start=f0 + s,
                frame_end=f0 + e,
                dct_target=dct_target,
            )


def _collect_actions(ds, f0: int, f1: int) -> np.ndarray:
    """frame [f0, f1) 의 action 을 (T, action_dim) 으로 collect."""
    out = []
    for f in range(f0, f1):
        a = ds[f]["action"]
        if hasattr(a, "numpy"):
            a = a.numpy()
        out.append(np.asarray(a, dtype=np.float64).reshape(-1))
    return np.stack(out)


def _get_instruction(ds, global_idx: int) -> str:
    frame = ds[global_idx]
    if "task" in frame:
        return str(_scalar(frame["task"]))
    if "skill.natural_language" in frame:
        return str(_scalar(frame["skill.natural_language"]))
    return ""


def build_dct_targets(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    L0: int = 50,
) -> int:
    """모든 skill segment → DCT target sidecar parquet 저장.

    Args:
        dataset_path: LeRobot dataset repo_id 또는 절대 경로.
        output_path: 출력 parquet 파일 경로.
        L0: DCT 출력 차원.

    Returns:
        저장된 record 수.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for seg in iter_skill_segments(dataset_path, L0=L0):
        records.append({
            "episode_id": seg.episode_id,
            "skill_index": seg.skill_index,
            "skill_type": seg.skill_type,
            "instruction": seg.instruction,
            "frame_start": seg.frame_start,
            "frame_end": seg.frame_end,
            "dct_shape": list(seg.dct_target.shape),
            "dct_target": seg.dct_target.flatten().tolist(),
        })

    if not records:
        raise RuntimeError(f"No skill segments extracted from {dataset_path}")

    table = pa.Table.from_pylist(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return len(records)


def load_dct_targets(parquet_path: str | Path) -> list[SkillSegment]:
    """sidecar parquet → ``SkillSegment`` list 복원 (학습 dataset wrapper 용)."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    out: list[SkillSegment] = []
    for r in rows:
        shape = tuple(r["dct_shape"])
        arr = np.asarray(r["dct_target"], dtype=np.float64).reshape(shape)
        out.append(SkillSegment(
            episode_id=r["episode_id"],
            skill_index=int(r["skill_index"]),
            skill_type=r["skill_type"],
            instruction=r["instruction"],
            frame_start=int(r["frame_start"]),
            frame_end=int(r["frame_end"]),
            dct_target=arr,
        ))
    return out
