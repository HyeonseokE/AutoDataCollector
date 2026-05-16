"""Preselective acquirer gRPC server (retrieval-augmented).

Boots once on the planning host (e.g., H100) holding:
- curobo MotionPlanner backend (in-process, GPU)
- frozen VLA encoder (SmolVLA, key-embedding extractor — GPU)
- IG·AC Selector + FaissBufferStore vector DB (server-local persistence)

The VLA is frozen and used ONLY to produce the FAISS key embedding
(mean-pooled embed_prefix = VL feature + proprioception).

A single client RPC `PlanAndSelect` does plan_batch → encode → IG·AC →
returns chosen trajectory + selection_id. Pending (ctx, selection) tuples
are kept in an in-memory dict keyed by selection_id; `CommitToBuffer`
flushes them to the FaissBufferStore on judge-TRUE, or drops on judge-FALSE.

Run:
    python -m preselective_rpc.server \\
        --host 0.0.0.0 --port 50061 \\
        --recording-config pipeline_config/recording_config_ws3.yaml \\
        --urdf assets/urdf/so101_robot4.urdf
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import uuid
from concurrent import futures
from pathlib import Path
from typing import Any, Optional

import grpc
import numpy as np
import yaml

# Project root on sys.path so we can import the in-tree packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preselective_filter import Candidate, Context, Selection
from preselective_rpc import preselective_pb2, preselective_pb2_grpc
from preselective_rpc._codec import (
    decode_jpeg,
    decode_ndarray,
    decode_pickle,
    encode_pickle,
)
from preselective_filter.integration import (
    setup_preselective_filter,
    trajectory_to_action_chunk,
)


# --------------------------------------------------------------------------
# Curobo backend bootstrap (one-time on server start)
# --------------------------------------------------------------------------
def _build_curobo(
    urdf_path: str,
    skill_cfg: dict,
    project_root: Path,
    transit_pitch_max_deg: Optional[float] = None,
):
    from perturbation.skill_level import get_curobo_backend

    CuroboBackend, CuroboBackendConfig = get_curobo_backend()

    cfg_path = skill_cfg.get("curobo_robot_cfg_path")
    if not Path(cfg_path).is_absolute():
        cfg_path = str((project_root / cfg_path).resolve())

    n_cand = int(skill_cfg.get("n_candidates", 4))
    cb_cfg = CuroboBackendConfig(
        enabled=True,
        robot_cfg_path=cfg_path,
        num_trajopt_seeds=int(skill_cfg.get("curobo_num_trajopt_seeds", 4)),
        num_ik_seeds=int(skill_cfg.get("curobo_num_ik_seeds", 16)),
        use_cuda_graph=bool(skill_cfg.get("curobo_use_cuda_graph", False)),
        via_offset_mag=float(skill_cfg.get("curobo_via_offset_mag", 0.10)),
        junction_smooth_k=int(skill_cfg.get("curobo_junction_smooth_k", 5)),
        fixed_joint_indices=tuple(skill_cfg.get("fixed_joint_indices") or ()),
        arm_joint_count=int(skill_cfg.get("arm_joint_count", 5)),
        max_vias_per_candidate=int(
            skill_cfg.get("curobo_max_vias_per_candidate", 1)
        ),
        # Option A wrist-cam transit bias: clamp via-point orientations to
        # pitch-down. Driven by the top-level transit_pitch_max_deg key.
        via_pitch_max_rad=(
            None if transit_pitch_max_deg is None
            else float(np.radians(float(transit_pitch_max_deg)))
        ),
        max_batch_size=n_cand,
    )
    return CuroboBackend(urdf=urdf_path, config=cb_cfg)


# --------------------------------------------------------------------------
# gRPC servicer
# --------------------------------------------------------------------------
class PreselectiveAcquirerServicer(
    preselective_pb2_grpc.PreselectiveAcquirerServicer
):
    def __init__(
        self,
        selector,                    # preselective_filter.Selector
        encoder,                     # preselective_filter.vectorDB.VLAKeyExtractor
        curobo_backend,              # CuroboBackend
        recording_fps: int = 10,
        chunk_size: int = 50,
        debug_verbose: bool = False,
    ) -> None:
        self.selector = selector
        self.encoder = encoder
        self.curobo = curobo_backend
        self.recording_fps = int(recording_fps)
        # Action-chunk resampling horizon. Buffer-only: action_dim is derived
        # per-request as arm_dof+1 (no VLA padding). Every candidate in one
        # plan_batch shares arm_dof, so all chunks are mutually comparable.
        self._chunk_size = int(chunk_size)
        self.debug_verbose = debug_verbose

        # Selection cache — selection_id → (Context, Selection)
        # Bounded by episode length × concurrent_clients; cleared on Commit.
        self._pending: dict[str, tuple[Context, Selection]] = {}

        # Serializes all VLA-encoder + buffer access. The encoder is
        # thread-unsafe; with concurrent clients PlanAndSelect and the
        # long-running IngestEpisode must not touch it at the same time.
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    def PlanAndSelect(self, request, context):
        t0 = time.perf_counter()
        try:
            start_qpos = decode_ndarray(request.start_qpos)
            goal_qpos = decode_ndarray(request.goal_qpos)
            state = decode_ndarray(request.state)
            raw_imgs = decode_pickle(request.images_pickle) if request.images_pickle else {}
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"deserialization failed: {e}")
            return preselective_pb2.PlanResponse()

        if not request.is_transit:
            # Skip selection for non-transit moves; client should fall back
            # to its own cartesian path.
            return preselective_pb2.PlanResponse(used_fallback=True)

        # 1. curobo plan_batch
        try:
            cands = self.curobo.plan_batch(
                start_qpos=np.asarray(start_qpos, dtype=float),
                goal_qpos=np.asarray(goal_qpos, dtype=float),
                n=int(request.n_candidates),
                seed=int(request.seed) if request.seed != 0 else None,
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"curobo plan_batch failed: {e}")
            return preselective_pb2.PlanResponse()

        if not cands:
            return preselective_pb2.PlanResponse(used_fallback=True)

        # 2. Wrap curobo candidates into fixed-shape action chunks.
        instruction = str(request.instruction or "")
        state_arr = np.asarray(state, dtype=float)
        skill_id = str(request.skill_id or "move_to")

        wrapped: list[Candidate] = []
        for c in cands:
            wp = c.waypoints
            arm_dof = int(wp.shape[1])
            times = np.arange(wp.shape[0], dtype=float) / float(self.recording_fps)
            chunk = trajectory_to_action_chunk(
                waypoints=wp,
                times=times,
                chunk_size=self._chunk_size,
                action_dim=arm_dof + 1,   # no VLA padding — real dims only
                fps=self.recording_fps,
                current_gripper=0.0,
                arm_dof=arm_dof,
            )
            wrapped.append(Candidate(
                skill_id=skill_id, action_chunk=chunk, payload=c,
            ))

        # 3. Encode the FAISS key + run the selector under the encoder/buffer
        #    lock (the frozen VLA is thread-unsafe; IngestEpisode shares it).
        try:
            with self._lock:
                key_emb = self.encoder.encode(raw_imgs, instruction, state_arr)
                ctx = Context(
                    observation=np.zeros(0, dtype=np.float32),
                    state=state_arr,
                    instruction=instruction,
                    skill_id=skill_id,
                    key_embedding=key_emb,
                )
                selection = self.selector.select(ctx, wrapped)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"encode/select failed: {e}")
            return preselective_pb2.PlanResponse()
        chosen = selection.chosen_candidate.payload

        # 4. selection_id is returned for protocol compatibility; the buffer is
        #    now grown by IngestEpisode (raw demo), not by committing this pick.
        sel_id = uuid.uuid4().hex

        # 5. Pack response
        traj_dict = {
            "waypoints": np.asarray(chosen.waypoints, dtype=np.float32),
            "times": (np.arange(chosen.waypoints.shape[0], dtype=np.float32)
                      / float(self.recording_fps)),
            "algo": str(getattr(chosen, "algo", "curobo")),
            "cost": float(getattr(chosen, "cost", 0.0)),
            "seed": int(getattr(chosen, "seed", 0)),
        }
        score_report_json = "[]"
        try:
            import json
            score_report_json = json.dumps([
                {
                    "i": r.candidate_index,
                    "novelty": r.novelty_raw, "ig": r.ig,
                    "consistency": r.consistency_raw, "ac": r.ac,
                    "score": r.score,
                }
                for r in selection.reports
            ])
        except Exception:
            pass

        if self.debug_verbose:
            elapsed = (time.perf_counter() - t0) * 1000.0
            print(
                f"[server] PlanAndSelect K={len(cands)} chose "
                f"idx={selection.chosen_index} ({elapsed:.0f}ms) "
                f"sel_id={sel_id[:8]}"
            )

        return preselective_pb2.PlanResponse(
            chosen_trajectory_pickle=encode_pickle(traj_dict),
            chosen_index=int(selection.chosen_index),
            score_report_json=score_report_json,
            selection_id=sel_id,
            used_fallback=False,
        )

    # ----------------------------------------------------------------
    def CommitToBuffer(self, request, context):
        committed = 0
        dropped = 0
        if request.judge_true:
            for sid in request.selection_ids:
                pending = self._pending.pop(sid, None)
                if pending is None:
                    continue
                ctx, selection = pending
                try:
                    self.selector.add_to_buffer(ctx, selection)
                    committed += 1
                except Exception as e:
                    print(f"[server] add_to_buffer failed for {sid[:8]}: {e}")
        else:
            for sid in request.selection_ids:
                if self._pending.pop(sid, None) is not None:
                    dropped += 1

        totals = {}
        try:
            totals = self.selector.buffer.summary()
        except Exception:
            pass

        if self.debug_verbose:
            print(
                f"[server] Commit: judge={request.judge_true} "
                f"committed={committed} dropped={dropped} "
                f"buffer={totals}"
            )

        return preselective_pb2.CommitResponse(
            committed=committed, dropped=dropped, buffer_totals=totals,
        )

    # ----------------------------------------------------------------
    def IngestEpisode(self, request_iterator, context):
        """Grow the vector DB from a streamed forward demo.

        The client streams the recorded episode frame-by-frame; for each frame
        the server encodes key_t with its frozen VLA and appends a
        (key_t, value_t) buffer entry. Holds the encoder/buffer lock for the
        whole episode so a concurrent PlanAndSelect can't race the encoder.
        """
        from preselective_filter.integration import ingest_frame

        t0 = time.perf_counter()
        n = 0
        try:
            with self._lock:
                for fm in request_iterator:
                    images = {}
                    if fm.images_pickle:
                        images = {
                            k: decode_jpeg(v)
                            for k, v in decode_pickle(fm.images_pickle).items()
                        }
                    ingest_frame(
                        self.encoder, self.selector.buffer,
                        images=images,
                        instruction=str(fm.instruction or ""),
                        state=decode_ndarray(fm.state),
                        action_chunk=decode_ndarray(fm.action_chunk),
                        skill_id=str(fm.skill_id or "move_to"),
                    )
                    n += 1
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"IngestEpisode failed after {n} frames: {e}")
            return preselective_pb2.IngestResponse(frames_ingested=n)

        totals = {}
        try:
            totals = self.selector.buffer.summary()
        except Exception:
            pass
        if self.debug_verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"[server] IngestEpisode: {n} demo frames in {elapsed:.1f}s, "
                f"buffer={totals}"
            )
        return preselective_pb2.IngestResponse(
            frames_ingested=n, buffer_totals=totals,
        )

    # ----------------------------------------------------------------
    def Ready(self, request, context):
        totals = {}
        try:
            totals = self.selector.buffer.summary()
        except Exception:
            pass
        sel_cfg = self.selector.config
        emb_dim = getattr(self.encoder, "embedding_dim", None)
        return preselective_pb2.ServerInfo(
            smolvla_checkpoint="<frozen VLA encoder>",
            device=str(getattr(self.encoder, "device", "cuda")),
            curobo_robot_cfg=str(getattr(self.curobo.cfg, "robot_cfg_path", "")),
            buffer_total=int(sum(totals.values())),
            buffer_per_skill=totals,
            selector_summary=(
                f"FAISS retrieval k={sel_cfg.context_k} "
                f"key_dim={emb_dim if emb_dim is not None else '?'}"
            ),
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def serve(args: argparse.Namespace) -> None:
    project_root = Path(args.recording_config).resolve().parent.parent
    cfg_path = Path(args.recording_config).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        recording_cfg = yaml.safe_load(f)

    print(f"[server] building Selector + VLA encoder from {cfg_path} ...")
    result = setup_preselective_filter(recording_cfg)
    if result is None:
        raise RuntimeError(
            "preselective_filter is disabled in the yaml — server has nothing to do."
        )
    selector, encoder = result

    print(f"[server] loading curobo backend ({args.urdf}) ...")
    skill_cfg = (recording_cfg.get("perturbation") or {}).get("skill") or {}
    curobo = _build_curobo(
        args.urdf, skill_cfg, project_root,
        transit_pitch_max_deg=recording_cfg.get("transit_pitch_max_deg"),
    )

    psf_cfg = recording_cfg.get("preselective_filter") or {}
    sel_cfg = psf_cfg.get("selector") or {}
    servicer = PreselectiveAcquirerServicer(
        selector=selector,
        encoder=encoder,
        curobo_backend=curobo,
        recording_fps=int(recording_cfg.get("recording_fps", 10)),
        chunk_size=int(sel_cfg.get("chunk_size", 50)),
        debug_verbose=bool(psf_cfg.get("debug_verbose", False)),
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(args.workers)),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    preselective_pb2_grpc.add_PreselectiveAcquirerServicer_to_server(
        servicer, server,
    )
    bind = f"{args.host}:{args.port}"
    server.add_insecure_port(bind)
    server.start()
    print(f"[server] listening on {bind} (workers={args.workers})")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[server] shutting down ...")
        server.stop(grace=2.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=50061)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--recording-config", required=True,
        help="path to recording_config_*.yaml — server reads preselective_filter "
             "+ perturbation.skill sections from this file.",
    )
    p.add_argument(
        "--urdf", required=True,
        help="path to URDF used by curobo backend.",
    )
    serve(p.parse_args())


if __name__ == "__main__":
    main()
