"""preselective_filter.integration — wiring, adapters, transport.

Composition root + adapters that connect the planner, the score metrics, and
the vector DB into a running pipeline. NOT a vector-DB concern itself.

Modules:
- pipeline_setup.py       : composition root — builds (Selector, extractor)
- demo_ingest.py          : recorded demo episodes → per-timestep DB entries
- candidate_bridge.py     : curobo TrajectoryCandidate → fixed-shape action chunk
- grpc_planner_adapter.py : remote-planner client (gRPC transport)

Heavy-dependency layer (pulls vectorDB → torch/faiss, and grpc). Import
explicitly; ``import preselective_filter`` stays light.
"""
from .candidate_bridge import trajectory_to_action_chunk
from .demo_ingest import (
    all_episode_indices,
    ingest_dataset,
    ingest_episode,
    ingest_frame,
    open_chunked_dataset,
)
from .grpc_planner_adapter import GrpcPlannerClient
from .pipeline_setup import setup_preselective_filter, teardown_preselective_filter

__all__ = [
    "GrpcPlannerClient",
    "all_episode_indices",
    "ingest_dataset",
    "ingest_episode",
    "ingest_frame",
    "open_chunked_dataset",
    "setup_preselective_filter",
    "teardown_preselective_filter",
    "trajectory_to_action_chunk",
]
