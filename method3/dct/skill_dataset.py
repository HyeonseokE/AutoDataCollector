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

import os
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


_SKILL_TYPE_KEYS = ("skill.type", "subtask.skill_type")


def _resolve_dataset_root(path_or_repo_id: str | Path) -> Path:
    """absolute path 또는 HF_LEROBOT_HOME / repo_id 로 dataset root 찾기.

    LeRobot v3 dataset 의 local cache root (``data/`` 와 ``meta/`` 가 있는
    디렉토리) 를 반환한다. lerobot import 없이 raw parquet 만 access 하는
    경로용.
    """
    p = Path(path_or_repo_id)
    if p.is_absolute() and (p / "meta").exists():
        return p
    home = Path(os.environ.get("HF_LEROBOT_HOME",
                                "~/.cache/huggingface/lerobot")).expanduser()
    candidate = home / str(path_or_repo_id)
    if (candidate / "meta").exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot resolve LeRobot dataset root for '{path_or_repo_id}'. "
        f"Tried '{p}' and '{candidate}'."
    )


def _pick_skill_type_column(data_columns) -> str:
    for k in _SKILL_TYPE_KEYS:
        if k in data_columns:
            return k
    raise RuntimeError(
        f"No skill.type column in data parquet; tried {_SKILL_TYPE_KEYS}"
    )


def iter_skill_segments(
    dataset_path: str | Path,
    *,
    L0: int = 50,
    episode_range: tuple[int, int] | None = None,
) -> Iterator[SkillSegment]:
    """LeRobot v3 dataset → (episode, skill) 단위 iteration (parquet 직접 access).

    raw ``data/chunk-*/file-*.parquet`` 에서 action / skill.type 만 읽어
    video decode 우회. 10분 → 수초.

    Args:
        dataset_path: LeRobot dataset repo_id (HF_LEROBOT_HOME 기준 resolve)
            또는 절대 경로 (meta/, data/ 가 있는 디렉토리).
        L0: DCT 출력 차원 (default 50).
        episode_range: 0-based ``[start, end)`` episode_index 필터.
            ``(30, 100)`` 이면 episode_index ∈ [30, 100) 만 yield (= 1-based
            31~100). None 이면 전체.

    Yields:
        ``SkillSegment``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = _resolve_dataset_root(dataset_path)

    # 1. 모든 data parquet 을 한 번에 읽어 action / skill.type / task_index /
    #    index (global frame) column 만 추출. video 컬럼은 데이터 parquet 에
    #    없으니 자동으로 skip.
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet under {root}/data/")
    schema_cols = pq.read_schema(data_files[0]).names
    skill_col = _pick_skill_type_column(schema_cols)
    needed = ["action", skill_col, "task_index", "index"]
    if "skill.natural_language" in schema_cols:
        needed.append("skill.natural_language")
    tbl = pa.concat_tables([pq.read_table(f, columns=needed) for f in data_files])
    tbl = tbl.sort_by("index")
    actions = np.array(tbl["action"].to_pylist(), dtype=np.float64)
    skill_types_all = [str(v) if v else "move" for v in tbl[skill_col].to_pylist()]
    task_indices = tbl["task_index"].to_pylist()
    skill_nl = (
        tbl["skill.natural_language"].to_pylist()
        if "skill.natural_language" in needed else None
    )

    # 2. tasks lookup (task_index → instruction string).
    tasks_table = pq.read_table(root / "meta" / "tasks.parquet")
    task_lookup: dict[int, str] = {}
    for row in tasks_table.to_pylist():
        ti = int(row.get("task_index", 0))
        # column 이름 호환: "task" 또는 "tasks" (단수/복수).
        s = row.get("task") or row.get("tasks") or ""
        if isinstance(s, list):
            s = s[0] if s else ""
        task_lookup[ti] = str(s)

    # 3. episode meta — episode_index / dataset_from_index / dataset_to_index.
    ep_meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_meta_files:
        raise FileNotFoundError(f"No episode meta parquet under {root}/meta/episodes/")
    episodes: list[dict] = []
    for f in ep_meta_files:
        episodes.extend(pq.read_table(
            f, columns=["episode_index", "dataset_from_index", "dataset_to_index"]
        ).to_pylist())
    episodes.sort(key=lambda r: int(r["episode_index"]))

    # 4. episode 별 skill boundary → DCT target.
    for ep in episodes:
        ei = int(ep["episode_index"])
        if episode_range is not None and not (episode_range[0] <= ei < episode_range[1]):
            continue
        f0 = int(ep["dataset_from_index"])
        f1 = int(ep["dataset_to_index"])
        episode_id = f"episode_{ei + 1:02d}"

        ep_skill = skill_types_all[f0:f1]
        ep_actions = actions[f0:f1]

        segments = _run_length_segments(ep_skill)
        for skill_idx, (s, e, sk) in enumerate(segments):
            seg_actions = ep_actions[s:e]
            # instruction: skill.natural_language 가 있으면 segment 시작값,
            # 없으면 task_index → task_lookup.
            if skill_nl is not None:
                v = skill_nl[f0 + s]
                instruction = str(v) if v else ""
            else:
                ti = int(task_indices[f0 + s])
                instruction = task_lookup.get(ti, "")

            dct_target = traj_to_dct(seg_actions, L0=L0)
            yield SkillSegment(
                episode_id=episode_id,
                skill_index=skill_idx,
                skill_type=sk,
                instruction=instruction,
                frame_start=f0 + s,
                frame_end=f0 + e,
                dct_target=dct_target,
            )


def build_dct_targets(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    L0: int = 50,
    episode_range: tuple[int, int] | None = None,
) -> int:
    """모든 skill segment → DCT target sidecar parquet 저장.

    Args:
        dataset_path: LeRobot dataset repo_id 또는 절대 경로.
        output_path: 출력 parquet 파일 경로.
        L0: DCT 출력 차원.
        episode_range: 0-based [start, end) ep_index 필터. None=전체.

    Returns:
        저장된 record 수.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for seg in iter_skill_segments(dataset_path, L0=L0, episode_range=episode_range):
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
