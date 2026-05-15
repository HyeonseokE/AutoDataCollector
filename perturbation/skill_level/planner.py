"""Skill-level perturbation: trajectory candidate type.

`TrajectoryCandidate` is the shared dataclass used by curobo_backend.py and
by the gRPC planner adapter for the on-wire trajectory shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrajectoryCandidate:
    """One result from a planning call (curobo backend or remote server)."""
    waypoints: np.ndarray           # (N, dof) joint-space path
    algo: str                       # producer label (e.g., "curobo:via1_s7")
    seed: int                       # planner RNG seed used
    plan_time_s: float              # wall time inside the planner, seconds
    cost: Optional[float] = None    # path length / planner cost
