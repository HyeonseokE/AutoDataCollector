"""Method3 storage layer — 문서 final_method3_spec §3.

Method3 는 저장 구조를 두 layer 로 분리한다 (§3, §13 원칙 1):

    Raw Trajectory Dataset = source of truth        → 이 폴더 (raw_dataset.py)
    Vector DB              = searchable index B_t^m  → phase2_mi_selection/

raw trajectory dataset 은 re-embedding(§6)·action descriptor 재계산·ablation
분석을 위한 원본 정보를 담는다. vector DB 는 raw dataset 을 가리키는 pointer
(``ref``)를 들고 retrieval/scoring 용 compact feature 만 저장하며, 각 buffer
가 스스로를 영속화한다 (Phase1 ``SubgoalBuffer`` / Phase2 ``SkillVectorDB`` —
대칭). 이 폴더는 그중 raw dataset 을 담당한다.
"""
from method3.storage.raw_dataset import (
    RawDatasetEntry,
    RawTrajectoryDataset,
    resolve_pointer,
)

__all__ = [
    "RawDatasetEntry",
    "RawTrajectoryDataset",
    "resolve_pointer",
]
