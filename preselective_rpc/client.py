"""Robot-side gRPC client for the preselective acquirer service.

Drop-in replacement for the local hook in execution_forward_and_reset.py:
- `PreselectiveClient.plan_and_select(...)` → returns chosen trajectory dict
- `PreselectiveClient.commit(...)` → flushes pending selections after judge

The server owns curobo + SmolVLA + buffer; the client just ships context
(state, goal, image, instruction) and receives back one trajectory.
"""
from __future__ import annotations

from typing import Any

import grpc
import numpy as np

from preselective_rpc import preselective_pb2, preselective_pb2_grpc
from preselective_rpc._codec import (
    decode_pickle,
    encode_ndarray,
    encode_pickle,
)


class PreselectiveClient:
    def __init__(
        self,
        server_address: str = "localhost:50061",
        timeout_s: float = 30.0,
    ) -> None:
        self.server_address = server_address
        self.timeout_s = float(timeout_s)
        self.channel = grpc.insecure_channel(
            server_address,
            options=[
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ],
        )
        self.stub = preselective_pb2_grpc.PreselectiveAcquirerStub(self.channel)
        # Track outstanding selection_ids per episode for commit/discard.
        self._pending_ids: list[str] = []

    # --------------------------------------------------------------
    def ready(self) -> dict:
        info = self.stub.Ready(preselective_pb2.Empty(), timeout=self.timeout_s)
        return {
            "smolvla_checkpoint": info.smolvla_checkpoint,
            "device": info.device,
            "curobo_robot_cfg": info.curobo_robot_cfg,
            "buffer_total": info.buffer_total,
            "buffer_per_skill": dict(info.buffer_per_skill),
            "selector_summary": info.selector_summary,
        }

    # --------------------------------------------------------------
    def plan_and_select(
        self,
        skill_id: str,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        instruction: str,
        n_candidates: int,
        seed: int = 0,
        is_transit: bool = True,
    ) -> dict[str, Any]:
        """Send context + goal to server, receive chosen trajectory.

        Returns dict with keys:
          - trajectory : {waypoints, times, algo, cost, seed}
          - chosen_index : int (server's choice within the K candidates)
          - selection_id : str (track for later commit)
          - score_report_json : str
          - used_fallback : bool (true → server didn't run selection;
                                  client should fall back to its own plan)
        """
        req = preselective_pb2.PlanRequest(
            skill_id=str(skill_id),
            start_qpos=encode_ndarray(np.asarray(start_qpos, dtype=np.float32)),
            goal_qpos=encode_ndarray(np.asarray(goal_qpos, dtype=np.float32)),
            state=encode_ndarray(np.asarray(state, dtype=np.float32)),
            images_pickle=encode_pickle(images),
            instruction=str(instruction or ""),
            n_candidates=int(n_candidates),
            seed=int(seed),
            is_transit=bool(is_transit),
        )
        resp = self.stub.PlanAndSelect(req, timeout=self.timeout_s)

        if resp.used_fallback:
            return {"used_fallback": True, "selection_id": ""}

        traj = decode_pickle(resp.chosen_trajectory_pickle)
        if resp.selection_id:
            self._pending_ids.append(resp.selection_id)
        return {
            "trajectory": traj,
            "chosen_index": int(resp.chosen_index),
            "selection_id": resp.selection_id,
            "score_report_json": resp.score_report_json,
            "used_fallback": False,
        }

    # --------------------------------------------------------------
    def commit(self, judge_true: bool, episode_id: str = "") -> dict:
        """Flush pending selections to buffer (if judge_true) or drop them."""
        if not self._pending_ids:
            return {"committed": 0, "dropped": 0, "buffer_totals": {}}
        req = preselective_pb2.CommitRequest(
            selection_ids=list(self._pending_ids),
            judge_true=bool(judge_true),
            episode_id=str(episode_id),
        )
        try:
            resp = self.stub.CommitToBuffer(req, timeout=self.timeout_s)
        finally:
            self._pending_ids.clear()
        return {
            "committed": int(resp.committed),
            "dropped": int(resp.dropped),
            "buffer_totals": dict(resp.buffer_totals),
        }

    # --------------------------------------------------------------
    def reset_pending(self) -> None:
        """Drop in-flight selection_ids without server commit (e.g., new episode)."""
        self._pending_ids.clear()

    def close(self) -> None:
        self.channel.close()
