"""Skill-level perturbation: OMPL planner ensemble.

Two layers complement subgoal-level perturbation:
  - subgoal_level: 3D Gaussian on transit target → "where to go"
  - skill_level (this):  algorithm/seed ensemble → "how to get there"

Public API::

    from perturbation.skill_level import (
        PlannerEnsemble, PlannerEnsembleConfig, ParallelEnsemble,
        TrajectoryCandidate, DEFAULT_ALGORITHMS,
    )
"""

from perturbation.skill_level.planner import (
    DEFAULT_ALGORITHMS,
    PlannerEnsemble,
    PlannerEnsembleConfig,
    TrajectoryCandidate,
)
from perturbation.skill_level.parallel import ParallelEnsemble
from perturbation.skill_level.collision_world import (
    WorkspaceConfig,
    build_workspace_objects,
)
# Client is the public entry point used by code in lerobot_cap (numpy 2.x).
# Importing it does NOT pull mplib — the daemon runs in a separate env.
from perturbation.skill_level.client import (
    PlanServiceClient,
    PlanServiceError,
)
# Curobo backend — direct (no daemon) GPU-accelerated planner with via-point
# diversity. Import is lazy via a function so users without curobo installed
# can still use the OMPL path.
def get_curobo_backend():
    from perturbation.skill_level.curobo_backend import (
        CuroboBackend, CuroboBackendConfig,
    )
    return CuroboBackend, CuroboBackendConfig

__all__ = [
    "DEFAULT_ALGORITHMS",
    "PlannerEnsemble",
    "PlannerEnsembleConfig",
    "ParallelEnsemble",
    "TrajectoryCandidate",
    "WorkspaceConfig",
    "build_workspace_objects",
    "PlanServiceClient",
    "PlanServiceError",
    "get_curobo_backend",
]
