"""Parallel candidate generation for the OMPL planner ensemble.

Spins up a persistent ``multiprocessing.Pool`` whose workers each load mplib
+ OMPL once at startup. Subsequent plan calls reuse those instances, so the
~1.6s URDF/SRDF setup is amortized across many calls.

Usage::

    pool = ParallelEnsemble(urdf="...", config=...)
    candidates = pool.plan_batch(start_qpos, goal_qpos, n=8, rng=ep_rng)
    pool.close()  # important — joins workers

The pool is independent of the calling process's GIL, so plan calls run
concurrently across CPU cores.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .planner import (
    DEFAULT_ALGORITHMS,
    PlannerEnsemble,
    PlannerEnsembleConfig,
    TrajectoryCandidate,
)


# ── module-globals owned by each worker process ────────────────────────────
# These live in the child process namespace; the parent never touches them.
_W_PLANNER: Optional[PlannerEnsemble] = None


def _worker_init(urdf: str, cfg_dict: dict) -> None:
    """Pool initializer — runs once per worker. Loads mplib + OMPL."""
    global _W_PLANNER
    cfg = PlannerEnsembleConfig(**cfg_dict)
    _W_PLANNER = PlannerEnsemble(urdf=urdf, config=cfg)


def _worker_plan(args: tuple) -> Optional[TrajectoryCandidate]:
    """Pool task — uses the per-worker planner instance."""
    algo, seed, start, goal = args
    assert _W_PLANNER is not None, "worker not initialized"
    return _W_PLANNER._plan_one(algo, seed, start, goal)


# Prefix for dynamically-managed scene objects. Static workspace objects
# (workspace_table, etc.) never use this prefix, so we can safely diff them.
_DYNAMIC_PREFIX = "scene__"


def _worker_update_scene(payload: list) -> dict:
    """Pool task — refresh the dynamic part of THIS worker's collision world.

    Args:
        payload: list of (name, [x,y,z], [sx,sy,sz]) tuples.

    Returns:
        {"added": int, "removed": int, "kept": int}
    """
    assert _W_PLANNER is not None, "worker not initialized"
    pw = _W_PLANNER._planner_mp.planning_world

    # Collect names currently managed by us
    existing = [n for n in pw.get_object_names() if n.startswith(_DYNAMIC_PREFIX)]
    desired = {f"{_DYNAMIC_PREFIX}{name}": (pos, size) for name, pos, size in payload}

    added = 0
    removed = 0
    kept = 0

    # Remove objects no longer present
    for name in existing:
        if name not in desired:
            try:
                pw.remove_object(name)
                removed += 1
            except Exception:
                pass

    # Add / refresh desired
    from mplib.collision_detection.fcl import Box, CollisionObject, FCLObject
    from mplib.pymp import Pose

    for full_name, (pos, size) in desired.items():
        # Always remove + re-add to refresh pose (simpler than tracking changes)
        if full_name in existing:
            try:
                pw.remove_object(full_name)
            except Exception:
                pass
        try:
            box = Box(float(size[0]), float(size[1]), float(size[2]))
            co = CollisionObject(box, Pose([float(pos[0]), float(pos[1]), float(pos[2])],
                                           [1.0, 0.0, 0.0, 0.0]))
            fcl_obj = FCLObject(
                full_name,
                Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
                [co],
                [Pose([float(pos[0]), float(pos[1]), float(pos[2])],
                      [1.0, 0.0, 0.0, 0.0])],
            )
            pw.add_object(fcl_obj)
            if full_name in existing:
                kept += 1
            else:
                added += 1
        except Exception:
            pass

    return {"added": added, "removed": removed, "kept": kept}


class ParallelEnsemble:
    """Persistent worker pool that produces N trajectory candidates per call."""

    def __init__(
        self,
        urdf: str | Path,
        config: PlannerEnsembleConfig,
        n_workers: Optional[int] = None,
    ) -> None:
        self.cfg = config
        self.urdf = str(urdf)
        self.n_workers = n_workers if n_workers is not None else max(2, mp.cpu_count() // 2)
        # Start workers eagerly so the URDF/SRDF setup happens up front, not on
        # the first plan_batch call.
        self._pool = mp.Pool(
            processes=self.n_workers,
            initializer=_worker_init,
            initargs=(self.urdf, dict(
                enabled=config.enabled,
                algorithms=tuple(config.algorithms),
                planning_time=config.planning_time,
                waypoint_density=config.waypoint_density,
                fallback_to_straight_line=config.fallback_to_straight_line,
                move_group_link=config.move_group_link,
                workspace=config.workspace,
                fixed_joint_indices=tuple(config.fixed_joint_indices or ()),
            )),
        )
        # Verify workers are alive — Pool defers errors otherwise.
        self._pool.apply(_noop)

    def plan_batch(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> list[TrajectoryCandidate]:
        """Submit ``n`` plan calls in parallel and return the successes.

        Algorithm + seed for each call are RNG-driven, so the same ``rng``
        state produces the same batch. Has a hard wall-clock budget so a
        single pathological OMPL call can't hang the daemon indefinitely —
        on timeout we abandon the in-flight pool (workers continue but their
        results are discarded) and return whatever finished cleanly.
        """
        if not self.cfg.enabled or n <= 0:
            return []
        rng = rng if rng is not None else np.random.default_rng()
        algos = list(self.cfg.algorithms)
        # Round-robin algos across the batch so all modes are represented even
        # when n is small. Within each algo, seeds are RNG-drawn for variety.
        args_list = []
        for i in range(n):
            algo = algos[i % len(algos)]
            seed = int(rng.integers(0, 2**31 - 1))
            args_list.append((algo, seed, np.asarray(start_qpos, dtype=float),
                              np.asarray(goal_qpos, dtype=float)))

        # Hard wall-clock budget. Each worker's .solve() is bounded by
        # config.planning_time; we still cap total wall to (planning_time × 2 + 2s)
        # so one stuck worker can't block the next plan request.
        wall_budget = max(2.0, self.cfg.planning_time * 2.0 + 2.0)
        async_result = self._pool.map_async(_worker_plan, args_list)
        try:
            results = async_result.get(timeout=wall_budget)
        except mp.TimeoutError:
            # Pool workers may still be running; we leave them alone and
            # discard the in-flight result. The pool stays usable because the
            # remaining workers will finish eventually and only this batch is
            # abandoned. Caller falls back to cartesian for this transit.
            return []
        return [r for r in results if r is not None]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _noop() -> None:
    """No-op task used at __init__ to flush worker startup errors."""
    return None
