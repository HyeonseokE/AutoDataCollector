"""Method3 re-embedding — Phase1 종료 후 full vector DB 구축 (문서 §6).

Phase1 raw dataset 을 Phase1-trained·frozen VLA encoder 로 re-embedding 하여
Phase2 가 쓸 skill-wise vector DB seed ``P_phase1^(m)`` 를 만든다. 파일별 역할:

    vla_encoder.py   — §6  VLAStateEncoder Protocol + MeanPoolStateEncoder stub
    seed_builder.py  — §6  build_phase1_vector_db (raw dataset → P_phase1)

§6 Step 1(VLA 학습)·Step 2(freeze)는 외부 ML job 이며, 이 패키지는 학습·freeze
된 encoder 를 받아 Step 3-4(re-embedding + seed DB 구축)를 수행한다.
"""
from method3.reembedding.seed_builder import (
    ReembeddingConfig,
    build_phase1_vector_db,
    state_retrieval_key,
)
from method3.reembedding.vla_encoder import MeanPoolStateEncoder, VLAStateEncoder

__all__ = [
    "VLAStateEncoder",
    "MeanPoolStateEncoder",
    "ReembeddingConfig",
    "state_retrieval_key",
    "build_phase1_vector_db",
]
