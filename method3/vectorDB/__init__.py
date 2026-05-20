"""method3.vectorDB — frozen VLA backbone-last-feature extractors.

pi0 / pi05 / smolvla / groot 의 frozen VLA 를 backbone-last-feature 까지 돌려
masked-mean-pool 한 retrieval key 를 만든다. method3 의 ``PretrainedVLAStateEncoder``
가 wrap 하고, server-side method3 setup 도 같은 추출기를 공유한다.

History: ``preselective_filter.vectorDB.vla_embedding`` 에서 이동 (2026-05-20).
"""
from method3.vectorDB.vla_embedding import VLAKeyExtractor, make_vla_key_extractor

__all__ = ["VLAKeyExtractor", "make_vla_key_extractor"]
