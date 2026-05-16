"""preselective_filter.vectorDB — retrieval-augmented vector-DB subsystem.

The vector DB itself: how a context becomes a key, and how keys + action
chunks are stored and retrieved. Wiring / adapters / transport live in
``preselective_filter.integration``.

Heavy-dependency layer (torch, lerobot, faiss, cv2). The parent
``preselective_filter`` package stays import-light — it does NOT import
this subpackage at module load, so ``import preselective_filter`` never
pulls torch. Import ``preselective_filter.vectorDB`` explicitly when the
VLA key extractor / FAISS buffer are actually needed.

Modules:
- vla_embedding.py : frozen VLA backbone-last-feature → FAISS key
- faiss_buffer.py  : per-skill FAISS vector DB (BufferStore impl)
"""
from .faiss_buffer import FaissBufferStore
from .vla_embedding import VLAKeyExtractor, make_vla_key_extractor

__all__ = [
    "FaissBufferStore",
    "VLAKeyExtractor",
    "make_vla_key_extractor",
]
