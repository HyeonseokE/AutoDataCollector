"""Phase1-trained VLA state encoder — interface + stub — 문서 §6.

§6: Phase1 종료 후 full skill-wise vector DB 를 구축할 때, Phase1 raw dataset 을
Phase1-trained·frozen VLA encoder ``φ_VLA^(1)`` 로 re-embedding 한다.

  e_i^vla = φ_VLA^(1)(o_i, I_i)        — observation·instruction → VLA embedding
  e_i     = [e_i^vla; p_i]             — proprioception 은 encoder 뒤에 concat (§7.3)

VLA encoder 학습(§6 Step 1)은 ML 학습 job 으로 이 라이브러리 밖에서 수행한다.
이 모듈은 (a) re-embedding 파이프라인이 의존하는 ``VLAStateEncoder`` Protocol 과,
(b) VLA 미가용 시 쓰는 결정적 stub ``MeanPoolStateEncoder`` 를 제공한다.

production 에서는 Phase1-trained VLA 를 이 Protocol 로 감싼다 — 기존
``preselective_filter/vectorDB/vla_embedding.py`` 의 ``VLAKeyExtractor`` /
``make_vla_key_extractor()`` 를 adapter 로 쓸 수 있다.

§6 핵심 원칙: Phase1 raw 의 re-embedding 과 Phase2 candidate 의 key 추출은
**같은 encoder instance** 를 써야 한다 (동일 metric space).
"""
from __future__ import annotations

import zlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VLAStateEncoder(Protocol):
    """VLA state encoder ``φ_VLA`` — observation·instruction → VLA embedding.

    proprioception 은 입력으로 받지 않는다 (§7.3 — embedding 뒤에 concat).
    """

    def encode(self, observation: np.ndarray, instruction: str) -> np.ndarray:
        """``e^vla = φ_VLA(o, I)`` — VLA embedding (D_vla,) 을 반환한다."""
        ...


class MeanPoolStateEncoder:
    """VLA 미가용 시 쓰는 결정적 stub encoder (§6 — geometric substitute).

    observation 을 flatten 후 ``out_dim`` 개 구간으로 average-pool 하고,
    instruction 을 CRC32 기반 결정적 offset 으로 반영한다. 학습된 VLA 가
    아니므로 retrieval 품질 baseline / 파이프라인 검증용이다 — production 은
    Phase1-trained VLA 를 ``VLAStateEncoder`` 로 감싸 대체한다.
    """

    def __init__(self, out_dim: int = 32) -> None:
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        self.out_dim = out_dim

    def encode(self, observation: np.ndarray, instruction: str) -> np.ndarray:
        """``e^vla`` — observation average-pool + instruction offset (D_vla=out_dim,)."""
        flat = np.asarray(observation, dtype=np.float64).reshape(-1)
        if flat.size < self.out_dim:
            # array_split 의 빈 구간(→ nan)을 피하려고 out_dim 까지 zero-pad.
            flat = np.pad(flat, (0, self.out_dim - flat.size))
        pooled = np.array([seg.mean() for seg in np.array_split(flat, self.out_dim)])
        # instruction 을 결정적으로(=프로세스 무관) 반영 — 파이썬 hash() 는
        # PYTHONHASHSEED 로 randomize 되므로 CRC32 를 쓴다.
        offset = (zlib.crc32(str(instruction).encode("utf-8")) % 1000) / 1000.0
        return pooled + offset
