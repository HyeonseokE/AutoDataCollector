"""HF LeRobotDataset → per-episode trajectory list.

parquet 파일을 직접 읽어 episode_index 별로 grouping.
LeRobotDataset 전체 로딩을 우회해서 메타만 있으면 동작.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def resolve_dataset_root(repo_id_or_path: str) -> Path:
    """Repo id 면 HF 에서 다운로드, 절대경로면 그대로 반환."""
    p = Path(repo_id_or_path)
    if p.is_dir():
        return p.resolve()
    local = snapshot_download(repo_id=repo_id_or_path, repo_type="dataset")
    return Path(local)


def load_episode_trajectories(
    repo_id_or_path: str,
    keys: List[str] = ("observation.state.radian_urdf0", "action.radian_urdf0"),
) -> Dict[str, List[np.ndarray]]:
    """데이터셋 → {feature_key: [episode_0_array, episode_1_array, ...]}

    각 array 는 (T_i, D) shape. 에피소드 순서는 episode_index 오름차순.
    """
    root = resolve_dataset_root(repo_id_or_path)
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet under {data_dir}")

    # 모든 parquet 을 메모리에 하나로 합침 (수 MB 규모라 문제없음)
    tables = [pq.read_table(f, columns=list(keys) + ["episode_index", "frame_index"])
              for f in parquet_files]

    import pyarrow as pa
    table = pa.concat_tables(tables)
    df = table.to_pandas()

    # 정렬 (episode_index 오름차순, 내부 frame_index 오름차순)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    result: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    for ep_idx, group in df.groupby("episode_index", sort=True):
        for k in keys:
            arr = np.stack(group[k].values).astype(np.float32)
            result[k].append(arr)

    return result


def summarize(trajs: List[np.ndarray], label: str) -> Dict[str, float]:
    """에피소드 길이 통계."""
    lengths = np.array([len(t) for t in trajs])
    return {
        "label": label,
        "num_episodes": len(trajs),
        "total_frames": int(lengths.sum()),
        "len_mean": float(lengths.mean()),
        "len_std": float(lengths.std()),
        "len_min": int(lengths.min()),
        "len_max": int(lengths.max()),
    }
