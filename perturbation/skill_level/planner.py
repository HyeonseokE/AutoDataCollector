"""Skill-level perturbation: OMPL planner ensemble.

Plans a joint-space trajectory between two configurations using one of N
configured OMPL algorithms (RRTConnect, PRMstar, BITstar, KPIECE1, ...).

The choice of algorithm + the planner's internal seed are RNG-driven so the
same episode RNG produces a reproducible (algo, seed) pair, while different
RNG states produce qualitatively different trajectories — RRTConnect is
greedy/short, PRMstar is roadmap-based, BITstar is anytime-optimal, KPIECE
is kinodynamic-style discretization.

Single-process use::

    ens = PlannerEnsemble(urdf="...", config=PlannerEnsembleConfig(...))
    traj = ens.plan(start_qpos, goal_qpos, rng=np.random.default_rng(seed))
    if traj is not None:
        # traj.waypoints is N × move_group_dof
        pass

For batch / parallel candidate generation, see ``parallel.ParallelEnsemble``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# OMPL planners that are reliably solvable and qualitatively distinct.
# RRTConnect = greedy bidirectional; PRMstar = roadmap; BITstar = anytime-optimal;
# KPIECE1 = grid-discretization. Adding more (RRT, RRTstar, FMT, ...) is fine
# but these four already span the "mode" axis the design doc calls for.
DEFAULT_ALGORITHMS = ("RRTConnect", "PRMstar", "BITstar", "KPIECE1")


@dataclass(frozen=True)
class TrajectoryCandidate:
    """One result from a planning call."""
    waypoints: np.ndarray           # (N, dof) joint-space path
    algo: str                       # algorithm that produced it
    seed: int                       # OMPL RNG seed used
    plan_time_s: float              # wall time inside .solve(), seconds
    cost: Optional[float] = None    # path length (sum of consecutive joint distances)


@dataclass
class PlannerEnsembleConfig:
    """Runtime configuration for the ensemble. Mirrors design-doc Phase 5 schema."""
    enabled: bool = False
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS
    planning_time: float = 0.5      # seconds per .solve() call
    waypoint_density: float = 0.01  # m of joint-distance between consecutive output waypoints
    fallback_to_straight_line: bool = True
    move_group_link: str = "gripper_frame_link"
    # Optional WorkspaceConfig for collision world. Pass via dict for picklability
    # (workers receive their own copy when ParallelEnsemble forwards it).
    workspace: Optional[dict] = None


class PlannerEnsemble:
    """Single-process ensemble. Loads mplib + ompl on construction.

    Reuse the same instance across many plan calls — the URDF / collision world
    setup is cached. For multi-process throughput, wrap one of these per worker
    via ``parallel.ParallelEnsemble``.
    """

    def __init__(self, urdf: str | Path, config: PlannerEnsembleConfig) -> None:
        import mplib
        from ompl import base as ob, util as ou
        self.cfg = config
        self.urdf = str(urdf)
        # mplib gives us URDF parsing + FCL collision world for free.
        self._mplib = mplib
        self._ob = ob
        self._ou = ou

        # Build environment objects (table / ceiling / other arms) if requested
        objects = []
        ws_obj: Optional["WorkspaceConfig"] = None
        if config.workspace is not None:
            from perturbation.skill_level.collision_world import (
                WorkspaceConfig, build_workspace_objects,
            )
            ws_obj = WorkspaceConfig(**config.workspace)
            objects = build_workspace_objects(ws_obj)

        self._planner_mp = mplib.Planner(
            urdf=self.urdf,
            move_group=config.move_group_link,
            objects=objects,
            verbose=False,
        )

        # Whitelist legitimate contacts (e.g., base_link mounted on table top)
        if ws_obj is not None and ws_obj.table_enabled and ws_obj.table_mount_links:
            acm = self._planner_mp.planning_world.get_allowed_collision_matrix()
            for link in ws_obj.table_mount_links:
                acm.set_entry(link, "workspace_table", True)
        self._move_idx = self._planner_mp.move_group_joint_indices
        self._dof = len(self._move_idx)
        self._joint_limits = self._planner_mp.joint_limits[self._move_idx]
        self._space = self._build_state_space()
        self._algo_factory = self._resolve_algo_factory(config.algorithms)

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def joint_limits(self) -> np.ndarray:
        return self._joint_limits.copy()

    def plan(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        algorithm: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Optional[TrajectoryCandidate]:
        """Plan a trajectory from ``start_qpos`` to ``goal_qpos``.

        Args:
            start_qpos / goal_qpos: shape (dof,). If shape > dof (full robot),
                the move-group slice is taken automatically.
            rng: numpy Generator. Used to pick algorithm + OMPL seed when those
                aren't provided explicitly.
            algorithm: explicit algorithm name. Overrides RNG-driven choice.
            seed: explicit OMPL seed. Overrides RNG-driven choice.

        Returns:
            TrajectoryCandidate if solved, else ``None``.
        """
        if not self.cfg.enabled:
            return None
        rng = rng if rng is not None else np.random.default_rng()
        algo = algorithm if algorithm is not None else str(rng.choice(self.cfg.algorithms))
        seed_v = int(seed) if seed is not None else int(rng.integers(0, 2**31 - 1))

        s = self._slice_to_move_group(start_qpos)
        g = self._slice_to_move_group(goal_qpos)
        return self._plan_one(algo, seed_v, s, g)

    # ── internals ───────────────────────────────────────────────────────────

    def _slice_to_move_group(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape[0] == self._dof:
            return q
        # Assume full-robot vector; pull move-group joints by index
        return q[self._move_idx]

    def _build_state_space(self):
        space = self._ob.RealVectorStateSpace(self._dof)
        bounds = self._ob.RealVectorBounds(self._dof)
        for i, (lo, hi) in enumerate(self._joint_limits):
            bounds.setLow(i, float(lo))
            bounds.setHigh(i, float(hi))
        space.setBounds(bounds)
        return space

    def _resolve_algo_factory(self, names):
        from ompl import geometric as og
        factories = {}
        for n in names:
            if not hasattr(og, n):
                raise ValueError(
                    f"Unknown OMPL algorithm: {n!r}. "
                    f"Available: {[k for k in dir(og) if not k.startswith('_')]}"
                )
            factories[n] = getattr(og, n)
        return factories

    def _make_state_validity_checker(self):
        """Closure: maps joint qpos → mplib collision check. Returns a callable
        suitable for ``SpaceInformation.setStateValidityChecker``.
        """
        planner_mp = self._planner_mp
        move_idx = self._move_idx
        dof = self._dof

        def is_valid(state) -> bool:
            q = np.array([state[i] for i in range(dof)])
            full = planner_mp.pad_move_group_qpos(q)
            planner_mp.robot.set_qpos(full, True)
            return len(planner_mp.planning_world.check_collision()) == 0

        return is_valid

    def _plan_one(
        self, algo: str, seed: int, start: np.ndarray, goal: np.ndarray
    ) -> Optional[TrajectoryCandidate]:
        import time
        ob = self._ob
        ou = self._ou
        if algo not in self._algo_factory:
            raise ValueError(f"algorithm {algo!r} not in configured pool")

        si = ob.SpaceInformation(self._space)
        si.setStateValidityChecker(self._make_state_validity_checker())
        si.setup()

        pdef = ob.ProblemDefinition(si)
        s0 = self._space.allocState()
        sg = self._space.allocState()
        for i in range(self._dof):
            s0[i] = float(start[i])
            sg[i] = float(goal[i])
        pdef.setStartAndGoalStates(s0, sg)

        ou.RNG.setSeed(seed)  # best-effort; OMPL's first-call-only restriction noted
        planner = self._algo_factory[algo](si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        t0 = time.time()
        solved = planner.solve(self.cfg.planning_time)
        plan_t = time.time() - t0
        if not bool(solved):
            return None

        path = pdef.getSolutionPath()
        n = path.getStateCount()
        if n < 2:
            return None
        wp = np.array([
            [path.getState(i)[j] for j in range(self._dof)] for i in range(n)
        ])
        wp = self._densify(wp, self.cfg.waypoint_density)
        cost = float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum())
        return TrajectoryCandidate(
            waypoints=wp, algo=algo, seed=seed, plan_time_s=plan_t, cost=cost,
        )

    @staticmethod
    def _densify(wp: np.ndarray, max_step: float) -> np.ndarray:
        """Linearly interpolate so consecutive waypoints differ by ≤ max_step
        in joint-space L2 norm. Keeps endpoints exact.
        """
        if max_step <= 0 or wp.shape[0] < 2:
            return wp
        out = [wp[0]]
        for i in range(1, wp.shape[0]):
            seg = wp[i] - wp[i - 1]
            d = float(np.linalg.norm(seg))
            n_sub = max(1, int(np.ceil(d / max_step)))
            for k in range(1, n_sub + 1):
                out.append(wp[i - 1] + seg * (k / n_sub))
        return np.array(out)
