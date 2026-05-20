"""Phase1 raw dataset re-embedding → Phase1 seed vector DB — 문서 §6.

§6 Step 3-4: Phase1 종료 후 Phase1 raw dataset 을 다시 읽어, frozen VLA encoder
로 re-embedding 하여 full skill-wise vector DB seed ``P_phase1^(m)`` 를 만든다.

  e_i^vla = φ_VLA^(1)(o_i, I_i)
  e_i     = [e_i^vla; p_i]                    — state retrieval key (§7.3)
  z_i^a   = ψ(A_{i:i+H-1})                    — DCT action descriptor (§4.2)
  P_phase1^(m) = {(e_i, z_i^a, ref_i, meta_i)}_{i∈D_phase1^(m)}

observation ``o_i`` 는 raw dataset 에 pointer(``observation_ref``)로만 있으므로
(§3 / §3 details §4), caller 가 그 pointer 를 실제 observation 배열로 푸는
``observation_loader`` 를 넘긴다.

§6 Step 1(VLA 학습)·Step 2(freeze)는 외부 ML job 이다 — 이 builder 는 이미
학습·freeze 된 encoder 를 받아 Step 3-4 만 수행한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from method3.phase2_mi_selection.action_descriptor import dct_action_descriptor
from method3.phase2_mi_selection.vector_db import SkillVectorDB, VectorDBEntry
from method3.reembedding.vla_encoder import VLAStateEncoder
from method3.storage.raw_dataset import RawTrajectoryDataset


@dataclass
class ReembeddingConfig:
    """re-embedding 파라미터 (문서 §6)."""

    dct_coeffs: int = 3          # §4.2 K — DCT action descriptor 저주파 성분 수
    skip_invalid: bool = False   # True → validity_flag=False entry 를 건너뜀
                                 # (기본 False — §6 은 dataset 전체 re-embed)
    show_progress: bool = True   # tqdm progress bar (False → 무음)
    frame_stride: int = 1        # 매 frame_stride 번째 entry 만 re-embed.
                                 # 1 = spec 정석 (전체). 5~10 = dense temporal
                                 # sampling 의 redundancy 를 활용한 wall-clock
                                 # speedup. retrieval-key 정확도는 거의 동일.


def _fallback_progress(iterable, total: int, label: str):
    """tqdm 미설치 시 ~5% 단위로 줄바꿈 없이 캐리지리턴 카운터만 찍는다."""
    import sys
    import time
    step = max(1, total // 20)
    start = time.monotonic()
    for i, x in enumerate(iterable):
        yield x
        done = i + 1
        if done == total or done % step == 0:
            pct = 100.0 * done / max(1, total)
            elapsed = time.monotonic() - start
            rate = done / max(1e-6, elapsed)
            eta = (total - done) / max(1e-6, rate)
            sys.stdout.write(
                f"\r[reembed][{label}] {done}/{total} ({pct:5.1f}%) "
                f"{rate:.1f} frame/s  ETA {eta:5.1f}s"
            )
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def state_retrieval_key(
    vla_embedding: np.ndarray,
    proprioception: np.ndarray,
) -> np.ndarray:
    """``e_i = [e_i^vla; p_i]`` — state retrieval key (문서 §7.3).

    proprioception 은 VLA encoder 입력이 아니라 embedding 뒤에 concat 된다.
    """
    return np.concatenate([
        np.asarray(vla_embedding, dtype=np.float64).reshape(-1),
        np.asarray(proprioception, dtype=np.float64).reshape(-1),
    ])


def build_phase1_vector_db(
    raw_dataset: RawTrajectoryDataset,
    encoder: VLAStateEncoder,
    observation_loader: Callable[[dict], np.ndarray],
    config: ReembeddingConfig | None = None,
) -> SkillVectorDB:
    """Phase1 raw dataset 을 re-embedding 하여 ``P_phase1`` vector DB 를 만든다 (§6).

    Args:
        raw_dataset: Phase1 raw trajectory dataset ``D_phase1_raw``.
        encoder: Phase1-trained·frozen VLA state encoder ``φ_VLA^(1)``.
            Phase2 candidate key 추출에도 **같은 instance** 를 써야 한다 (§6).
        observation_loader: ``observation_ref`` (dict) → observation 배열 resolver.
            raw dataset 은 §4 pointer 규약상 observation 을 ref 로만 들고 있다.
        config: re-embedding 파라미터.

    Returns:
        skill-wise 로 partition 된 Phase1 seed vector DB ``P_phase1``.
    """
    cfg = config or ReembeddingConfig()
    db = SkillVectorDB()
    stride = max(1, int(cfg.frame_stride))
    N_full = len(raw_dataset)
    indices = range(0, N_full, stride)
    N = len(indices)
    if stride > 1:
        print(f"[reembed] frame_stride={stride} → 처리할 entry {N_full} → {N} "
              f"({100.0 * N / max(1, N_full):.1f}%)")
    # progress bar — 9k+ frames × VLA forward 의 진행 가시화. tqdm 미설치 시
    # silent fallback (간단 % 로그). show_progress=False 면 무음.
    if cfg.show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(indices, desc="[reembed] skill-wise DB build",
                            unit="frame", dynamic_ncols=True, total=N)
        except ImportError:
            iterator = _fallback_progress(indices, N, "skill-wise DB")
    else:
        iterator = indices
    for idx in iterator:
        entry = raw_dataset.get(idx)
        if cfg.skip_invalid and not entry.validity_flag:
            continue
        observation = observation_loader(entry.observation_ref)
        e_vla = encoder.encode(observation, entry.instruction)            # §6
        e_i = state_retrieval_key(e_vla, entry.proprioception)            # §7.3
        z_i = dct_action_descriptor(entry.action_chunk, cfg.dct_coeffs)   # §4.2
        db.append(VectorDBEntry(
            skill_id=entry.skill_id,
            state_key=e_i,
            action_descriptor=z_i,
            ref=raw_dataset.pointer(idx),                                 # §3.1 dataset_ref
            meta={
                "phase": "phase1",
                "skill_id": entry.skill_id,
                "subgoal": np.asarray(entry.subgoal, dtype=float).tolist(),
                "planner_type": entry.planner_type,
                "instruction": entry.instruction,
                "time_index": int(entry.time_index),
                "accepted_by": "phase1_seed",
            },
        ))
    return db
