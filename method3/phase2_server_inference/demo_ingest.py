"""LeRobot dataset frame iterator helpers for client → server ingest streaming.

Phase2 server (H100) 가 IngestEpisode RPC 를 통해 forward demo 의 frame 들을
받을 때, *client 측에서* 그 dataset 을 frame 단위로 읽기 위한 helper.

History — 옛 명칭/책임 (2026-05-20 정리):
  이 모듈은 원래 ``preselective_filter.integration.demo_ingest`` 였고 *client+
  server 양쪽* 의 vector DB 충전 함수 (``ingest_frame``, ``ingest_episode``,
  ``ingest_dataset``) 가 같이 있었다. server 측 buffer-write 책임은 method3
  phase2 server (grpc_server.server) 가 직접 SkillVectorDB.append 하는
  방식으로 이전됐고, 이 파일에는 *순수 LeRobot dataset 읽기 helper* 만 남는다:

    - all_episode_indices(dataset)  : 어떤 episode index 들이 있나
    - open_chunked_dataset(...)     : action delta_timestamps 가 적용된 LeRobotDataset

grpc_server.client.PreselectiveClient.ingest_episode 가 이 두 함수를 써서
frame 스트림을 만든다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def all_episode_indices(dataset) -> list[int]:
    """Episode indices of a dataset (``dataset.episodes`` is None = 'all')."""
    eps = getattr(dataset, "episodes", None)
    if eps is not None:
        return list(eps)
    return list(range(int(dataset.meta.total_episodes)))


def open_chunked_dataset(dataset_root: str | Path, chunk_size: int):
    """LeRobot dataset 을 action delta_timestamps 가 적용된 모드로 연다.

    각 item 의 ``action`` 이 training-identical (H, action_dim) chunk 가 되도록
    delta_timestamps 를 fps 기준으로 설정. server 측 IngestEpisode 가 그대로
    SkillVectorDB.append 에 사용 (DCT descriptor 까지).
    """
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    root = Path(dataset_root)
    repo_id = root.name
    fps = LeRobotDatasetMetadata(repo_id, root=root).fps
    delta = {"action": [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta)


def episode_bounds(dataset, ep_idx: int) -> tuple[int, int]:
    """``[from_index, to_index)`` of a single episode."""
    ep = dataset.meta.episodes[ep_idx]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def frame_images(item: dict) -> dict[str, np.ndarray]:
    """Pull observation.images.* frames from a dataset item as numpy arrays."""
    out: dict[str, np.ndarray] = {}
    for k, v in item.items():
        if k.startswith("observation.images"):
            out[k] = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    return out


def frame_instruction(item: dict, dataset, ep_idx: int, fallback: str = "") -> str:
    """Resolve the task instruction string for one frame."""
    task = item.get("task")
    if isinstance(task, str) and task:
        return task
    tasks = dataset.meta.episodes[ep_idx].get("tasks")
    if isinstance(tasks, (list, tuple)) and tasks:
        return str(tasks[0])
    return fallback


__all__ = [
    "all_episode_indices",
    "open_chunked_dataset",
    "episode_bounds",
    "frame_images",
    "frame_instruction",
]
