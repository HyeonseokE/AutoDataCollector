"""Robot-side gRPC client for the preselective acquirer service.

Drop-in replacement for the local hook in execution_forward_and_reset.py:
- `PreselectiveClient.plan_and_select(...)` → returns chosen trajectory dict
- `PreselectiveClient.commit(...)` → flushes pending selections after judge

The server owns curobo + the buffer-only Selector + buffer; the client just
ships joint-space context (state, goal, instruction) and receives back one
trajectory.
"""
from __future__ import annotations

from typing import Any

import grpc
import numpy as np

from grpc_server import preselective_pb2, preselective_pb2_grpc
from grpc_server._codec import (
    decode_pickle,
    encode_jpeg,
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
        current_positions: dict[str, dict] | None = None,
        target_phase1_episode_id: str = "",
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
        # current_positions: {name: {pose: [x,y,z,qx,qy,qz,qw], dims: [dx,dy,dz]}}
        # → preselective_pb2.ObstacleGeometry map field.
        _proto_obstacles = {}
        if current_positions:
            for name, geom in current_positions.items():
                pose = geom.get("pose") or [0.0]*7
                dims = geom.get("dims") or [0.0]*3
                _proto_obstacles[name] = preselective_pb2.ObstacleGeometry(
                    x=float(pose[0]), y=float(pose[1]), z=float(pose[2]),
                    qx=float(pose[3]), qy=float(pose[4]),
                    qz=float(pose[5]), qw=float(pose[6]),
                    dim_x=float(dims[0]), dim_y=float(dims[1]), dim_z=float(dims[2]),
                )
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
            current_positions=_proto_obstacles,
            target_phase1_episode_id=str(target_phase1_episode_id or ""),
        )
        resp = self.stub.PlanAndSelect(req, timeout=self.timeout_s)

        if resp.used_fallback:
            return {
                "used_fallback": True,
                "selection_id": resp.selection_id or "",
                "score_report_json": resp.score_report_json or "",
                "chosen_index": int(resp.chosen_index),
            }

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
    def ingest_episode(
        self,
        dataset_root: Any,
        chunk_size: int,
        skill_id: str = "move_to",
        episode_index: int | None = None,
        timeout_s: float = 600.0,
        max_frames: int | None = None,
    ) -> dict:
        """Stream a recorded forward demo to the server for vector-DB growth.

        Reads the episode locally (CPU only — no VLA) and streams each frame's
        (images, state, action_chunk, instruction) to the H100 server, which
        encodes key_t with its frozen VLA and appends one (key_t, value_t)
        buffer entry per frame. Returns {frames, buffer_totals, episode}.

        ``max_frames`` caps how many frames are streamed (quick checks).
        """
        from method3.phase2_server_inference.demo_ingest import (
            all_episode_indices,
            open_chunked_dataset,
        )

        ds = open_chunked_dataset(dataset_root, chunk_size)
        ep = (episode_index if episode_index is not None
              else max(all_episode_indices(ds)))
        em = ds.meta.episodes[ep]
        s, e = int(em["dataset_from_index"]), int(em["dataset_to_index"])
        if max_frames is not None:
            e = min(e, s + int(max_frames))

        def _frames():
            for idx in range(s, e):
                item = ds[idx]
                # JPEG-compress each camera frame for transport (~10x smaller).
                # Lossy but negligible for a frozen-encoder feature key.
                imgs = {
                    k: encode_jpeg(
                        v.numpy() if hasattr(v, "numpy") else np.asarray(v)
                    )
                    for k, v in item.items()
                    if k.startswith("observation.images")
                }
                instr = item.get("task")
                if not isinstance(instr, str) or not instr:
                    tasks = ds.meta.episodes[ep].get("tasks")
                    instr = str(tasks[0]) if tasks else ""
                yield preselective_pb2.FrameMessage(
                    skill_id=str(skill_id),
                    instruction=str(instr),
                    state=encode_ndarray(
                        np.asarray(item["observation.state"], dtype=np.float32)
                    ),
                    action_chunk=encode_ndarray(
                        np.asarray(item["action"], dtype=np.float32)
                    ),
                    images_pickle=encode_pickle(imgs),
                )

        resp = self.stub.IngestEpisode(_frames(), timeout=timeout_s)
        return {
            "frames": int(resp.frames_ingested),
            "buffer_totals": dict(resp.buffer_totals),
            "episode": int(ep),
        }

    # --------------------------------------------------------------
    def reset_pending(self) -> None:
        """Drop in-flight selection_ids without server commit (e.g., new episode)."""
        self._pending_ids.clear()

    def close(self) -> None:
        self.channel.close()
