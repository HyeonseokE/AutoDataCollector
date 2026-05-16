"""preselective_filter.score_metric — candidate scoring metrics.

Pure-numpy scoring layer (no torch / faiss). Holds the two axes the
Selector combines into the final IG·AC score:

- information_gain   : IG — novelty of a candidate vs the skill buffer
- action_consistency : AC — closeness of a candidate to context-kNN
                       (retrieved) precedent actions

This subpackage is intentionally dependency-light so the metrics stay
unit-testable in isolation.
"""
from .action_consistency import build_mode_set, compute_ac, compute_mode_distance
from .information_gain import compute_buffer_novelty, compute_ig, minmax_norm

__all__ = [
    "build_mode_set",
    "compute_ac",
    "compute_buffer_novelty",
    "compute_ig",
    "compute_mode_distance",
    "minmax_norm",
]
