"""Pretrained VLA encoder adapter — VLAStateEncoder Protocol 위의 임시 구현.

§6 Step 1-2 가 외부 ML job 이고 아직 Phase1-trained VLA encoder 가 준비 안 됐을
때, **pretrained VLA 체크포인트를 그대로** ``VLAStateEncoder`` Protocol 로 감싸
서 §6 Step 3-4 (re-embedding) 를 동작 확인할 수 있게 한다. Phase1-trained
encoder 가 준비되면 같은 Protocol 로 swap.

내부적으로 ``preselective_filter.vectorDB.vla_embedding.VLAKeyExtractor`` 를
재사용한다 (pi0 / pi05 / SmolVLA / Groot 등 family auto-dispatch).

⚠️ Protocol 불일치 처리:
  - ``VLAStateEncoder.encode(o, I)`` — proprio 없음 (§7.3 — embedding 뒤에
    concat).
  - ``VLAKeyExtractor.encode(o, I, state)`` — proprio 를 backbone 에 fuse.
  → adapter 는 zero proprio 를 VLAKeyExtractor 에 넘기고, 실제 proprio 는
    ``seed_builder.state_retrieval_key`` 가 별도로 concat 한다. Phase1-trained
    encoder 로 swap 시엔 이 zero-proprio 처리 없이 §6 의 정의대로 동작.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class PretrainedVLAStateEncoder:
    """VLAKeyExtractor → VLAStateEncoder adapter.

    Phase1-trained encoder 가 준비되기 전 임시로 pretrained 체크포인트(pi0 등)를
    같은 인터페이스로 쓰기 위한 어댑터. ``close()`` 로 GPU 메모리 해제.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        autocast_dtype: str = "bfloat16",
        state_dim: int = 7,
        family: str | None = None,
        debug_verbose: bool = False,
    ) -> None:
        """
        Args:
            checkpoint_path: VLA 체크포인트 경로(local) 또는 HF repo_id.
            device: torch device.
            autocast_dtype: "bfloat16" | "float16" | "float32".
            state_dim: VLAKeyExtractor 에 넘길 zero-proprio 벡터 차원 (모델
                config 의 state dim 과 일치해야 함; 보통 7).
            family: smolvla / pi0 / pi05 / groot 강제 지정 (None → 경로로 추론).
            debug_verbose: VLAKeyExtractor 디버그 로그.
        """
        from preselective_filter.vectorDB.vla_embedding import make_vla_key_extractor

        self._extractor = make_vla_key_extractor(
            checkpoint=str(checkpoint_path),
            device=device,
            autocast_dtype=autocast_dtype,
            debug_verbose=debug_verbose,
            family=family,
        )
        self._zero_state = np.zeros(int(state_dim), dtype=np.float64)
        self._closed = False

    def encode(self, observation, instruction: str) -> np.ndarray:
        """``e^vla = φ_VLA(o, I)`` — VLA embedding (D_vla,).

        VLAKeyExtractor 는 ``(observation, instruction, state)`` 시그니처라
        ``state`` 자리에 zero 벡터를 넘긴다. 실제 proprio 는 seed_builder 가
        ``state_retrieval_key`` 에서 concat (§7.3).
        """
        if self._closed:
            raise RuntimeError("encoder already closed")
        return np.asarray(
            self._extractor.encode(observation, str(instruction), self._zero_state),
            dtype=np.float64,
        ).reshape(-1)

    def encode_batch(
        self, observations: list, instructions: list,
    ) -> np.ndarray:
        """Batched encode — N contexts → (N, D_vla).

        VLAKeyExtractor 가 ``encode_batch`` 를 지원하면 GPU 에서 한 번의
        forward 로 N 개 처리. 미지원이면 single-loop fallback.
        """
        if self._closed:
            raise RuntimeError("encoder already closed")
        n = len(observations)
        assert n == len(instructions)
        if n == 0:
            return np.zeros((0, self.embedding_dim or 0), dtype=np.float64)

        states = [self._zero_state] * n
        if hasattr(self._extractor, "encode_batch"):
            return np.asarray(
                self._extractor.encode_batch(observations, instructions, states),
                dtype=np.float64,
            )
        # fallback — extractor 가 batch 미지원 family.
        return np.stack([
            self.encode(o, i) for o, i in zip(observations, instructions)
        ])

    @property
    def embedding_dim(self) -> int | None:
        """VLA backbone 출력 차원. 첫 ``encode`` 호출 전엔 None."""
        return self._extractor.embedding_dim

    def close(self) -> None:
        """GPU 메모리 해제 — encoder 사용 종료 시 호출."""
        if not self._closed:
            self._extractor.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
