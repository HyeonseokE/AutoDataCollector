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
]
