"""Drop-in skill-planner client backed by the preselective gRPC server.

The local CuroboBackend exposes
``plan_batch(start, goal, n, seed) → list[TrajectoryCandidate]`` and lets
skills_lerobot sample ONE candidate locally. In gRPC mode the server runs
both planning AND IG·AC selection; the adapter here returns a single-element
list containing the server's chosen trajectory, so skills_lerobot's existing
selection code naturally picks it.

Per-step inputs flow:
    skills_lerobot.move_to_position
        ↓ plan_batch(start, goal, n, seed)
    GrpcPlannerClient.plan_batch
        ↓ pulls latest images + instruction from orchestrator
        ↓ PreselectiveClient.plan_and_select(server_addr)
    server.PlanAndSelect
        ↓ curobo.plan_batch + selector.select
        ↓ caches (ctx, selection) by selection_id
    GrpcPlannerClient
        ↓ wraps server response as TrajectoryCandidate (single-element list)
    skills_lerobot
        → executes trajectory
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from preselective_rpc.client import PreselectiveClient


@dataclass(frozen=True)
class _RemoteTrajectoryCandidate:
    """TrajectoryCandidate-shaped duck-type built from server response.

    Mirrors perturbation.skill_level.planner.TrajectoryCandidate fields
    that skills_lerobot.move_to_position consumes for logging + execution.
    """
    waypoints: np.ndarray  # (N, dof) joint-space
    algo: str
    seed: int
    plan_time_s: float
    cost: float | None
    # Helper: server-provided times (T,). Optional metadata for downstream use.
    times: np.ndarray | None = None


class GrpcPlannerClient:
    """Wraps PreselectiveClient to look like a skill-planner backend.

    Construct with a ContextProvider that knows how to fetch the current
    camera frames + episode instruction at plan time (typically the orchestrator
    object via duck-typed methods).
    """

    def __init__(
        self,
        client: PreselectiveClient,
        context_provider,
        skill_id: str = "move_to",
    ) -> None:
        self._client = client
        self._provider = context_provider
        self._skill_id = str(skill_id)

    # ------------------------------------------------------------------
    # plan_batch — drop-in replacement for CuroboBackend
    # ------------------------------------------------------------------
    def plan_batch(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        n: int,
        seed: int | None = None,
        rng: Any | None = None,  # accepted for signature parity; ignored
    ) -> list[_RemoteTrajectoryCandidate]:
        """Send context to server, receive ONE chosen trajectory."""
        try:
            images = self._provider._latest_observation_dict()
        except Exception:
            images = {}
        instruction = ""
        try:
            instruction = str(getattr(self._provider, "instruction", "") or "")
        except Exception:
            pass

        try:
            resp = self._client.plan_and_select(
                skill_id=self._skill_id,
                start_qpos=np.asarray(start_qpos, dtype=np.float32),
                goal_qpos=np.asarray(goal_qpos, dtype=np.float32),
                state=np.asarray(start_qpos, dtype=np.float32),
                images=images,
                instruction=instruction,
                n_candidates=int(n),
                seed=int(seed) if seed is not None else 0,
                is_transit=True,
            )
        except Exception as e:
            print(f"  [Skill Perturbation] grpc plan_and_select failed: {e}")
            return []

        if resp.get("used_fallback", False):
            return []

        traj = resp["trajectory"]
        wp = np.asarray(traj["waypoints"], dtype=np.float32)
        times = np.asarray(traj.get("times", []), dtype=np.float32)
        return [_RemoteTrajectoryCandidate(
            waypoints=wp,
            algo=str(traj.get("algo", "grpc:remote")),
            seed=int(traj.get("seed", 0)),
            plan_time_s=0.0,
            cost=float(traj.get("cost", 0.0)),
            times=times if times.size else None,
        )]

    # ------------------------------------------------------------------
    # update_scene — server-side curobo can also accept obstacle updates,
    # but the prototype skips this (TODO: extend proto with UpdateScene RPC).
    # ------------------------------------------------------------------
    def update_scene(self, obstacles) -> None:  # noqa: ARG002
        # Silent no-op for now — server's curobo uses its yaml's static workspace.
        return None

    # ------------------------------------------------------------------
    # Lifecycle helpers used by orchestrator
    # ------------------------------------------------------------------
    def commit(self, judge_true: bool, episode_id: str = "") -> dict:
        return self._client.commit(judge_true=judge_true, episode_id=episode_id)

    def reset_pending(self) -> None:
        self._client.reset_pending()

    def ready(self) -> dict:
        return self._client.ready()

    def close(self) -> None:
        self._client.close()
