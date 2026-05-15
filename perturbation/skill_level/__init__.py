"""Skill-level perturbation: GPU-accelerated curobo planner.

Uses curobo's in-process GPU motion planner with via-point diversity.
The remote gRPC server (``preselective_rpc``) reuses the same backend.

Public API::

    from perturbation.skill_level import (
        TrajectoryCandidate, get_curobo_backend,
    )
    CuroboBackend, CuroboBackendConfig = get_curobo_backend()
"""
from perturbation.skill_level.planner import TrajectoryCandidate


def get_curobo_backend():
    """Lazy curobo import — pulls torch/curobo only when called."""
    from perturbation.skill_level.curobo_backend import (
        CuroboBackend, CuroboBackendConfig,
    )
    return CuroboBackend, CuroboBackendConfig


__all__ = [
    "TrajectoryCandidate",
    "get_curobo_backend",
]
