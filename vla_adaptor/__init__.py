"""vla_adaptor — Integration adapters for preselective_filter.

Production implementations of the Protocols defined in
`preselective_filter` (PolicyAdapter, BufferStore) plus the bridge from
existing perturbation TrajectoryCandidate to preselective_filter.Candidate
and the pipeline setup wiring.

This package has external dependencies (torch, lerobot, pathlib, …) by
design — `preselective_filter` stays numpy-only.
"""
from .candidate_bridge import trajectory_to_action_chunk
from .jsonl_buffer import JsonlBufferStore
from .pipeline_setup import setup_preselective_filter
from .smolvla_adapter import SmolVLAAdapter, SmolVLAAdapterConfig

__all__ = [
    "JsonlBufferStore",
    "SmolVLAAdapter",
    "SmolVLAAdapterConfig",
    "setup_preselective_filter",
    "trajectory_to_action_chunk",
]
