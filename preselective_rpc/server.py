"""Preselective acquirer gRPC server.

Boots once on the planning host (e.g., H100) holding:
- SmolVLA policy + preprocessor (loaded once, reused across episodes)
- curobo MotionPlanner backend (in-process, no daemon)
- Selector (IG·AC) + JsonlBufferStore (server-local persistence)

A single client RPC `PlanAndSelect` does plan_batch → IG·AC → returns chosen
trajectory + selection_id. Pending (ctx, selection) tuples are kept in an
in-memory dict keyed by selection_id; `CommitToBuffer` flushes them to the
JsonlBufferStore on judge-TRUE, or drops them on judge-FALSE.

Run:
    python -m preselective_rpc.server \\
        --host 0.0.0.0 --port 50061 \\
        --recording-config pipeline_config/recording_config_ws3.yaml \\
        --urdf assets/urdf/so101_robot4.urdf
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from concurrent import futures
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import yaml

# Project root on sys.path so we can import the in-tree packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preselective_filter import Candidate, Context, Selection
from preselective_rpc import preselective_pb2, preselective_pb2_grpc
from preselective_rpc._codec import (
    decode_ndarray,
    decode_pickle,
    encode_pickle,
)
from vla_adaptor import trajectory_to_action_chunk
from vla_adaptor.pipeline_setup import setup_preselective_filter


# --------------------------------------------------------------------------
# Image preprocessing helper — mirrors execution_forward_and_reset._map_images_to_policy_keys
# --------------------------------------------------------------------------
def _map_images_to_policy_keys(
    raw_imgs: dict, image_keys: list, target_shape: tuple | None,
) -> dict:
    import cv2  # lazy
    if not image_keys or not raw_imgs or target_shape is None:
        return dict(raw_imgs) if raw_imgs else {}
    C, H, W = int(target_shape[0]), int(target_shape[1]), int(target_shape[2])
    out: dict = {}
    items = list(raw_imgs.items())
    for i, key in enumerate(image_keys):
        if i >= len(items):
            break
        _, frame = items[i]
        if frame is None:
            continue
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] == 3:
            if arr.shape[:2] != (H, W):
                arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 3 and arr.shape[0] == 3:
            if arr.shape[1:] != (H, W):
                hwc = arr.transpose(1, 2, 0)
                hwc = cv2.resize(hwc, (W, H), interpolation=cv2.INTER_AREA)
                arr = hwc.transpose(2, 0, 1)
        else:
            continue
        if arr.dtype != np.float32:
            arr = (arr.astype(np.float32) / 255.0
                   if arr.dtype == np.uint8 else arr.astype(np.float32))
        out[key] = arr[None, ...]
    return out


# --------------------------------------------------------------------------
# Curobo backend bootstrap (one-time on server start)
# --------------------------------------------------------------------------
def _build_curobo(urdf_path: str, skill_cfg: dict, project_root: Path):
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
        selector,                    # preselective_filter.Selector (already loaded)
        curobo_backend,              # CuroboBackend
        recording_fps: int = 10,
        debug_verbose: bool = False,
    ) -> None:
        self.selector = selector
        self.curobo = curobo_backend
        self.recording_fps = int(recording_fps)
        self.debug_verbose = debug_verbose

        # Selection cache — selection_id → (Context, Selection)
        # Bounded by episode length × concurrent_clients; cleared on Commit.
        self._pending: dict[str, tuple[Context, Selection]] = {}

        # Pre-resolve policy config for image mapping.
        policy_cfg = selector.policy.policy.config
        self._chunk_size = int(policy_cfg.chunk_size)
        self._action_dim = int(policy_cfg.max_action_dim)
        keys = list(policy_cfg.image_features.keys())
        self._image_keys = keys
        self._image_target_shape = (
            tuple(policy_cfg.image_features[keys[0]].shape) if keys else None
        )

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

        # 2. Build IG·AC context + wrap candidates as preselective Candidates
        obs_dict = _map_images_to_policy_keys(
            raw_imgs, self._image_keys, self._image_target_shape,
        )
        ctx = Context(
            observation=obs_dict,
            state=np.asarray(state, dtype=float),
            instruction=str(request.instruction or ""),
            skill_id=str(request.skill_id or "move_to"),
        )

        wrapped: list[Candidate] = []
        for c in cands:
            wp = c.waypoints
            arm_dof = int(wp.shape[1])
            times = np.arange(wp.shape[0], dtype=float) / float(self.recording_fps)
            chunk = trajectory_to_action_chunk(
                waypoints=wp,
                times=times,
                chunk_size=self._chunk_size,
                action_dim=self._action_dim,
                fps=self.recording_fps,
                current_gripper=0.0,
                arm_dof=arm_dof,
            )
            wrapped.append(Candidate(
                skill_id=ctx.skill_id, action_chunk=chunk, payload=c,
            ))

        # 3. Run selector
        selection = self.selector.select(ctx, wrapped)
        chosen = selection.chosen_candidate.payload

        # 4. Cache pending for deferred buffer commit
        sel_id = uuid.uuid4().hex
        self._pending[sel_id] = (ctx, selection)

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
                    "u_pi0": r.u_pi0_norm, "n_buf": r.n_buffer_norm,
                    "ig": r.ig, "ac_m": r.ac_model, "ac_b": r.ac_buffer,
                    "ac": r.ac, "score": r.score,
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
    def Ready(self, request, context):
        totals = {}
        try:
            totals = self.selector.buffer.summary()
        except Exception:
            pass
        adapter_cfg = self.selector.policy.config
        sel_cfg = self.selector.config
        return preselective_pb2.ServerInfo(
            smolvla_checkpoint="<resident>",
            device=str(adapter_cfg.device),
            curobo_robot_cfg=str(getattr(self.curobo.cfg, "robot_cfg_path", "")),
            buffer_total=int(sum(totals.values())),
            buffer_per_skill=totals,
            selector_summary=(
                f"α={sel_cfg.alpha} λ={sel_cfg.lam} "
                f"M={sel_cfg.n_vla_samples} k={sel_cfg.context_k} "
                f"N_b={adapter_cfg.n_fm_mc_samples}"
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

    print(f"[server] loading SmolVLA + Selector from {cfg_path} ...")
    selector = setup_preselective_filter(recording_cfg)
    if selector is None:
        raise RuntimeError(
            "preselective_filter is disabled in the yaml — server has nothing to do."
        )

    print(f"[server] loading curobo backend ({args.urdf}) ...")
    skill_cfg = (recording_cfg.get("perturbation") or {}).get("skill") or {}
    curobo = _build_curobo(args.urdf, skill_cfg, project_root)

    servicer = PreselectiveAcquirerServicer(
        selector=selector,
        curobo_backend=curobo,
        recording_fps=int(recording_cfg.get("recording_fps", 10)),
        debug_verbose=bool((recording_cfg.get("preselective_filter") or {})
                           .get("debug_verbose", False)),
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
