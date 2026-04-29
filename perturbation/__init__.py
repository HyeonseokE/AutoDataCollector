"""Trajectory perturbation modules for AutoDataCollector.

Two-layer hierarchy:
    subgoal_level/  — 3D Gaussian offset on transit subgoal targets
    skill_level/    — planner ensemble (placeholder; future)
"""

from perturbation.subgoal_level import (
    SubgoalPerturbation,
    SubgoalPerturbationConfig,
    TRANSIT_SKILL_TYPES,
)

__all__ = [
    "SubgoalPerturbation",
    "SubgoalPerturbationConfig",
    "TRANSIT_SKILL_TYPES",
]
