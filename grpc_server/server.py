"""method3 phase2 useful-OOD gRPC server (server-side acquisition).

Boots once on the planning host (e.g., H100) holding:
- curobo MotionPlanner backend (in-process, GPU)
- frozen VLA encoder (smolvla / pi0 / pi05 / groot — GPU)
- Phase2MISelector + SkillVectorDB (server-local persistence)
- (옵션) LeRobotVLAInformativenessScorer for U_VLA (§12)

Spec reference: final_method3_spec_useful_ood_updated §11-§13.

A single client RPC ``PlanAndSelect`` does plan_batch → Phase2Candidate 변환
→ Useful-OOD selection → returns chosen trajectory + selection_id.
``IngestEpisode`` grows the SkillVectorDB from a streamed forward demo.
``CommitToBuffer`` is retained for protocol compatibility (no-op in the new
spec, since accept happens inline at PlanAndSelect time).

Run:
    python -m grpc_server.server \\
        --host 0.0.0.0 --port 50061 \\
        --recording-config pipeline_config/recording_config_ws3.yaml \\
        --urdf assets/urdf/so101_robot4.urdf
"""
from __future__ import annotations

import argparse
import os
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

from method3.phase2_mi_selection import (
    LeRobotVLAInformativenessScorer,
    Phase2MISelector,
    SkillVectorDB,
    VectorDBEntry,
)
from method3.phase2_mi_selection.action_descriptor import dct_action_descriptor
from method3.phase2_mi_selection.curobo_candidate_gen import (
    CurobogenConfig,
    candidates_from_trajectory_list,
)
from method3.reembedding.seed_builder import state_retrieval_key
from grpc_server import preselective_pb2, preselective_pb2_grpc
from grpc_server._codec import (
    decode_jpeg,
    decode_ndarray,
    decode_pickle,
    encode_pickle,
)
from grpc_server.method3_setup import setup_method3_phase2_server, Method3ServerStack


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
        # joint-space 저주파 perturbation — 경로 다양화 (A안).
        joint_perturb_k=int(skill_cfg.get("curobo_joint_perturb_k", 0)),
        joint_perturb_mag=float(skill_cfg.get("curobo_joint_perturb_mag", 0.0)),
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
    """method3 phase2 useful-OOD acquirer (server-side).

    선택 알고리즘은 옛 IG·AC 가 아니라 ``Phase2MISelector`` (§13.2). SkillVectorDB
    는 PlanAndSelect 시점에 accepted 후보를 inline 으로 append (옛 *CommitToBuffer*
    의 retro-commit 패턴 대체). IngestEpisode 는 raw demo frame 들을 직접
    SkillVectorDB 에 append.
    """

    def __init__(
        self,
        stack: Method3ServerStack,    # method3_setup 가 만든 bundle
        curobo_backend,               # CuroboBackend
        recording_fps: int = 10,
        chunk_size: int = 50,
        debug_verbose: bool = False,
        action_horizon: int = 50,
        max_T_eval: int | None = None,
        servo_calibration_file: str | None = None,
        camera_rename: dict | None = None,
        urdf_path: str | None = None,
    ) -> None:
        self.stack = stack
        self.selector = stack.selector
        self.encoder = stack.encoder
        self.vla_scorer = stack.vla_scorer
        self.db = stack.db
        self.db_path = stack.db_path
        self.curobo = curobo_backend
        # Track current attached payload so we only call attach/detach when
        # the held state actually changes (avoids redundant graph invalidation).
        self._currently_held: dict | None = None
        self.recording_fps = int(recording_fps)
        self._chunk_size = int(chunk_size)
        self.debug_verbose = debug_verbose
        # camera_rename — raw capture 키 → policy image feature 키 매핑.
        # value 가 "observation.images." prefix 없으면 자동 보강.
        self._camera_rename: dict = {}
        for _rk, _pk in (camera_rename or {}).items():
            _pk = str(_pk)
            if not _pk.startswith("observation.images."):
                _pk = f"observation.images.{_pk}"
            self._camera_rename[str(_rk)] = _pk

        # Phase2Candidate (T, H) chunking 파라미터 — curobo_candidate_gen 의 input.
        # T 는 trajectory 길이 N 에서 자동 계산 (stride=1). max_T_eval 은 cost cap.
        # EE delta DCT FK 는 curobo_backend 의 이미 부팅된 kinematics 를 재사용
        # (lerobot_cap chain → scservo_sdk import 우회). CuroboBackend._planner
        # = MotionPlanner, .kinematics property = trajopt_solver.kinematics.
        _kin_engine = None
        try:
            _kin_engine = curobo_backend._planner.kinematics
        except AttributeError as _e:
            print(f"[server] WARN: curobo_backend._planner.kinematics 접근 실패 "
                  f"({_e}) — ee_features 가 legacy fk_ee 로 fallback 시도", flush=True)

        self._candidate_cfg = CurobogenConfig(
            action_horizon=int(action_horizon),
            max_T_eval=(None if max_T_eval is None else int(max_T_eval)),
            servo_calibration_file=servo_calibration_file,
            urdf_path=urdf_path,
            kinematics_engine=_kin_engine,
        )
        if urdf_path:
            print(f"[server] EE delta DCT enabled — URDF FK ← {urdf_path}", flush=True)
        else:
            print(f"[server] EE delta DCT DISABLED (urdf_path=None) — "
                  f"MI scoring 이 joint DCT fallback (translation invariance 없음)", flush=True)
        if servo_calibration_file:
            print(f"[server] joint→servo calibration enabled ← {servo_calibration_file}", flush=True)
        else:
            print(f"[server] joint→servo calibration DISABLED (candidate is in radians, "
                  f"DB is in servo — action_descriptor unit mismatch).", flush=True)

        # Selection cache — selection_id → (Phase2Candidate, Phase2Selection)
        # CommitToBuffer 의 retro 호환 — 새 spec 에선 accept 가 inline 이라 unused.
        self._pending: dict[str, tuple[Any, Any]] = {}

        # Serializes all VLA-encoder + DB access (encoder thread-unsafe).
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    def _rename_cameras(self, raw_imgs: dict) -> dict:
        """raw capture 키 → policy image feature 키 (camera_rename map).

        학습 policy 는 observation.images.camera{N} 을 기대하는데 client 는
        raw 키 (top/left_wrist) 로 보낸다. 본 메서드가 VLA encoder·scorer 가
        policy 와 정합되도록 키를 바꾼다. map 에 없는 키는 그대로 통과.
        map 이 비어있으면 원본 그대로 반환 (legacy / 매핑 불필요 환경).
        """
        if not self._camera_rename or not isinstance(raw_imgs, dict):
            return raw_imgs
        out: dict = {}
        for k, v in raw_imgs.items():
            out[self._camera_rename.get(str(k), str(k))] = v
        # camera{N} 순서 정렬 — vla_embedding._map_raw_images 가 dict insertion
        # 순서대로 policy image_keys 에 매핑하므로, 키 이름 순(=camera1,2,3)이
        # policy feature 순서와 일치해야 한다.
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    # ----------------------------------------------------------------
    def _sync_held_object(self, request, start_qpos) -> None:
        """Reconcile self._currently_held with request.held_object.

        Calls curobo backend attach/detach only when the requested held state
        actually changes (presence, name, link, dims, or pose). The backend
        methods are no-ops if unsupported (e.g. legacy backends without
        attach_held_object) — sync silently skips in that case.

        ``start_qpos`` is the current joint configuration; passed to attach
        so the fitted spheres reflect where the gripper actually is.
        """
        held = getattr(request, "held_object", None)
        # Treat empty-name HeldObject as "no payload".
        if held is None or not getattr(held, "name", ""):
            # Detach any existing payload.
            if self._currently_held is not None:
                if hasattr(self.curobo, "detach_held_object"):
                    self.curobo.detach_held_object(
                        link_name=self._currently_held.get("link", "gripper_frame_link"),
                    )
                self._currently_held = None
            return

        # Requested attachment — compute fingerprint for change detection.
        new_state = {
            "name": str(held.name),
            "link": str(held.link_name) if held.link_name else "gripper_frame_link",
            "dims": tuple(float(x) for x in (held.dims or (0.16, 0.16, 0.04))),
            "pose": tuple(float(x) for x in (held.pose_offset or (0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0))),
        }
        if self._currently_held == new_state:
            return  # already attached with the same parameters

        if not hasattr(self.curobo, "attach_held_object"):
            return  # backend doesn't support — keep self-collision only

        # Detach previous (if any) before attaching new, so dims/link changes
        # take effect cleanly.
        if self._currently_held is not None:
            self.curobo.detach_held_object(
                link_name=self._currently_held.get("link", "gripper_frame_link"),
            )
        self.curobo.attach_held_object(
            name=new_state["name"],
            dims=new_state["dims"],
            pose_offset=new_state["pose"],
            link_name=new_state["link"],
            joint_state=start_qpos,
        )
        self._currently_held = new_state

    # ----------------------------------------------------------------
    def PlanAndSelect(self, request, context):
        t0 = time.perf_counter()
        try:
            start_qpos = decode_ndarray(request.start_qpos)
            goal_qpos = decode_ndarray(request.goal_qpos)
            state = decode_ndarray(request.state)
            raw_imgs = decode_pickle(request.images_pickle) if request.images_pickle else {}
            raw_imgs = self._rename_cameras(raw_imgs)
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"deserialization failed: {e}")
            return preselective_pb2.PlanResponse()

        if not request.is_transit:
            # Skip selection for non-transit moves; client should fall back
            # to its own cartesian path.
            return preselective_pb2.PlanResponse(used_fallback=True)

        # Reconcile held-object attachment state with the request BEFORE
        # plan_batch so the chosen trajectory routes around obstacles WITH
        # the payload volume. Skill backends without attach support no-op.
        try:
            self._sync_held_object(request, start_qpos)
        except Exception as e:
            print(f"[server] sync_held_object FAILED (continuing without attach): {e}",
                  flush=True)

        # 1. curobo plan_batch
        try:
            cands = self.curobo.plan_batch(
                start_qpos=np.asarray(start_qpos, dtype=float),
                goal_qpos=np.asarray(goal_qpos, dtype=float),
                n=int(request.n_candidates),
                seed=int(request.seed) if request.seed != 0 else None,
            )
            try:
                _img_info = {k: tuple(np.asarray(v).shape) for k, v in (raw_imgs or {}).items()}
            except Exception:
                _img_info = "<unparseable>"
            print(f"[server] plan_batch req n={request.n_candidates} seed={request.seed} "
                  f"is_transit={request.is_transit} skill_id={request.skill_id!r} "
                  f"instr={str(request.instruction)[:40]!r} "
                  f"images_pickle_bytes={len(request.images_pickle)} "
                  f"raw_imgs={_img_info} "
                  f"→ curobo returned {len(cands)} cands", flush=True)
            _t_curobo = time.perf_counter()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"curobo plan_batch failed: {e}")
            print(f"[server] plan_batch EXCEPTION: {e}", flush=True)
            return preselective_pb2.PlanResponse()

        if not cands:
            print(f"[server] plan_batch returned 0 valid candidates "
                  f"(start={np.round(start_qpos, 3).tolist()}, "
                  f"goal={np.round(goal_qpos, 3).tolist()}) — sending used_fallback=True", flush=True)
            return preselective_pb2.PlanResponse(used_fallback=True)

        # 2. TrajectoryCandidate → Phase2Candidate.
        #    encoder 와 candidate 변환은 lock 안에서 (encoder thread-unsafe).
        # instruction 은 skill-conditioned format ({skill_type}: {task}) — 학습 /
        # DB build / Phase2 inference 모두 같은 분포. client(grpc_planner_adapter)
        # 가 이미 format 해서 보낸다. skill_id 는 ordinal partition 키(skill_0..)
        # 라 VLA prefix 로 쓰면 안 되므로 server 는 재포맷하지 않는다.
        skill_id = str(request.skill_id or "skill_0")
        instruction = str(request.instruction or "")
        state_arr = np.asarray(state, dtype=float)
        # seed_subgoal anchor — client 가 보낸 goal_qpos 의 EE xyz 가 가장 자연.
        # FK 없으면 goal_qpos 첫 3 element 를 placeholder (downstream 영향 작음).
        try:
            seed_xyz = np.asarray(goal_qpos, dtype=np.float64).reshape(-1)[:3]
            if seed_xyz.shape[0] < 3:
                seed_xyz = np.pad(seed_xyz, (0, 3 - seed_xyz.shape[0]))
        except Exception:
            seed_xyz = np.zeros(3, dtype=np.float64)

        try:
            with self._lock:
                # 3. Phase2Candidate batch — encoder 가 state_keys 채움.
                p2_cands = candidates_from_trajectory_list(
                    cands,
                    skill_id=skill_id,
                    seed_subgoal=seed_xyz,
                    current_observation=raw_imgs,
                    instruction=instruction,
                    encoder=self.encoder,
                    config=self._candidate_cfg,
                    # client 가 보낸 *current robot state* (servo position 6-dim).
                    # candidate state_keys 의 proprio slot 으로 사용해 DB build pattern
                    # (proprio = observation.state) 과 unit 통일.
                    robot_state=np.asarray(state, dtype=np.float64),
                )
                _t_encode = time.perf_counter()
                if p2_cands:
                    _c0 = p2_cands[0]
                    print(f"[debug] cand#0 state_keys={_c0.state_keys.shape} "
                          f"action_chunks={_c0.action_chunks.shape} "
                          f"dct_target={_c0.dct_target.shape if _c0.dct_target is not None else None} "
                          f"proprios={_c0.proprios.shape if _c0.proprios is not None else None} "
                          f"wp[0]={getattr(cands[0],'waypoints',np.array([0])).shape} "
                          f"robot_state(arg)={np.asarray(state).shape}", flush=True)
                if not p2_cands:
                    try:
                        _wp0 = getattr(cands[0], "waypoints", None)
                        _wp0_shape = tuple(_wp0.shape) if _wp0 is not None else None
                        _wp_lens = [getattr(getattr(c, "waypoints", None), "shape", (None,))[0]
                                    for c in cands[:5]]
                    except Exception:
                        _wp0_shape = None
                        _wp_lens = []
                    print(f"[server] Phase2Candidate 변환 후 0개 — "
                          f"orig cands={len(cands)}, first wp shape={_wp0_shape}, "
                          f"wp_lens(head)={_wp_lens}, encoder={'set' if self.encoder is not None else 'None'}, "
                          f"action_horizon(H)={self._candidate_cfg.action_horizon}, "
                          f"fail_safe_min_waypoints={self._candidate_cfg.fail_safe_min_waypoints} "
                          f"→ used_fallback=True", flush=True)
                    return preselective_pb2.PlanResponse(used_fallback=True)
                # 4. Useful-OOD selection (§13.2): argmax U_VLA s.t. M̃_MI ≥ τ_MI.
                #    vla_scorer 가 None 이면 argmax M_MI fallback.
                selection = self.selector.select(p2_cands, vla_scorer=self.vla_scorer)
                _t_select = time.perf_counter()
                _reports = selection.reports or []
                _under = sum(1 for r in _reports if r.under_covered)
                _mn = [r.q2_norm for r in _reports]
                _m = [r.q2 for r in _reports]
                _dha = [r.delta_h_a for r in _reports]
                _dhas = [r.delta_h_a_given_s for r in _reports]
                _uvla = [r.u_vla for r in _reports]
                def _stats(xs):
                    if not xs: return (0.0, 0.0, 0.0)
                    n = len(xs); s = sum(xs); mn = min(xs); mx = max(xs)
                    return (mn, mx, s/n)
                print(
                    f"[server] Phase2 select: cands={len(p2_cands)} "
                    f"under_covered={_under}/{len(p2_cands)} "
                    f"eligible={len(selection.eligible_indices)} chosen=#{selection.chosen_index} "
                    f"accepted={selection.accepted} u_vla={selection.u_vla_chosen} "
                    f"mode={self.selector.cfg.selection_mode} "
                    f"tau_MI={self.selector.cfg.tau_MI} "
                    f"| M̃_MI[min,max,mean]={_stats(_mn)} "
                    f"M_MI[min,max,mean]={_stats(_m)} "
                    f"ΔH_A[min,max,mean]={_stats(_dha)} "
                    f"ΔH_A|S[min,max,mean]={_stats(_dhas)} "
                    f"U_VLA[min,max,mean]={_stats(_uvla)}",
                    flush=True,
                )
                # 단계별 소요 — curobo plan_batch / VLA embedding(encode) /
                # selection(M_MI + U_VLA score_batch) 의 wall-clock.
                print(
                    f"[timing] curobo={(_t_curobo - t0) * 1000:.0f}ms  "
                    f"encode(VLA embed)={(_t_encode - _t_curobo) * 1000:.0f}ms  "
                    f"select(M_MI+U_VLA)={(_t_select - _t_encode) * 1000:.0f}ms",
                    flush=True,
                )
                # PlanResponse.selection_id — accept_to_buffer·candidate dump·
                # client episode hook 이 공유하는 selection 식별자.
                sel_id = uuid.uuid4().hex
                # candidate dump — 128 후보 trajectory + per-candidate 점수 +
                # selection + top-view 이미지를 npz 로 보존. 파일명이 sel_id 라
                # client 가 PlanResponse.selection_id 로 이 dump 를 특정한다.
                # dump 실패는 selection 흐름과 격리.
                try:
                    from method3.phase2_mi_selection.candidate_dump import (
                        dump_phase2_candidates,
                    )
                    _dump = dump_phase2_candidates(
                        self.curobo, cands, selection,
                        skill_id=skill_id, seed_xyz=seed_xyz,
                        start_qpos=start_qpos, goal_qpos=goal_qpos,
                        tau_MI=self.selector.cfg.tau_MI,
                        selection_id=sel_id,
                        # top-view(camera2) 이미지 — client 오버레이 배경.
                        top_image=raw_imgs.get("observation.images.camera2"),
                        # g.t. descriptor(servo DCT) → radians 변환용 calib.
                        servo_calib_path=self._candidate_cfg.servo_calibration_file,
                    )
                    print(f"[server] candidate dump → {_dump}", flush=True)
                except Exception as _de:
                    print(f"[server] candidate dump skipped: {_de}", flush=True)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"phase2 candidate/select failed: {e}")
            import traceback as _tb
            print(f"[server] phase2 candidate/select EXCEPTION: {e!r}", flush=True)
            _tb.print_exc()
            return preselective_pb2.PlanResponse()

        chosen_p2 = selection.chosen_candidate
        chosen_traj = chosen_p2.payload.get("traj") if isinstance(chosen_p2.payload, dict) else None
        if chosen_traj is None:
            # candidates_from_trajectory_list 가 payload 에 algo/cost 만 넣어둠 →
            # 원본 TrajectoryCandidate 를 부속 참조로 다시 가져온다.
            chosen_traj = cands[selection.chosen_index]

        # 5. inline accept — useful-OOD 가 accepted 면 SkillVectorDB 에 즉시 append
        #    (옛 CommitToBuffer 의 retro pattern 대체. 학습 데이터의 무결성은
        #    client 의 judge-FALSE 시 IngestEpisode 미호출로 자연 보존).
        #    sel_id 는 위 candidate dump 블록에서 이미 생성됨.
        if selection.accepted:
            try:
                with self._lock:
                    self.selector.accept_to_buffer(
                        chosen_p2,
                        ref={"selection_id": sel_id, "skill_id": skill_id},
                        meta={
                            "algo": (chosen_p2.payload or {}).get("algo", ""),
                            "u_vla": selection.u_vla_chosen,
                            "instruction": instruction,
                        },
                    )
            except Exception as e:
                print(f"[server] accept_to_buffer failed: {e}")

        # 6. Pack response — TrajectoryCandidate.waypoints 그대로 (client 가 그걸 실행).
        traj_dict = {
            "waypoints": np.asarray(chosen_traj.waypoints, dtype=np.float32),
            "times": (np.arange(chosen_traj.waypoints.shape[0], dtype=np.float32)
                      / float(self.recording_fps)),
            "algo": str(getattr(chosen_traj, "algo", "curobo")),
            "cost": float(getattr(chosen_traj, "cost", 0.0)),
            "seed": int(getattr(chosen_traj, "seed", 0)),
        }
        # score_report — useful-OOD 의 per-candidate report.
        score_report_json = "[]"
        try:
            import json
            score_report_json = json.dumps([
                {
                    "i": r.candidate_index,
                    "delta_h_a": r.delta_h_a,
                    "delta_h_a_given_s": r.delta_h_a_given_s,
                    "m_mi": r.q2,
                    "m_mi_norm": r.q2_norm,
                    "u_vla": r.u_vla,
                    "under_covered": r.under_covered,
                }
                for r in selection.reports
            ])
        except Exception:
            pass

        if self.debug_verbose:
            elapsed = (time.perf_counter() - t0) * 1000.0
            print(
                f"[server] PlanAndSelect K={len(cands)} chose "
                f"idx={selection.chosen_index} accepted={selection.accepted} "
                f"u_vla={selection.u_vla_chosen} ({elapsed:.0f}ms) "
                f"sel_id={sel_id[:8]}"
            )

        return preselective_pb2.PlanResponse(
            chosen_trajectory_pickle=encode_pickle(traj_dict),
            chosen_index=int(selection.chosen_index),
            score_report_json=score_report_json,
            selection_id=sel_id,
            used_fallback=not selection.accepted,
        )

    # ----------------------------------------------------------------
    def CommitToBuffer(self, request, context):
        """Deprecated in the new spec.

        method3 phase2 useful-OOD 는 PlanAndSelect 시점에 inline accept_to_buffer.
        client 측이 judge-FALSE 면 IngestEpisode 를 *호출 안 함* 으로써 자연 보존.
        호환을 위해 RPC 는 살아남지만 no-op + DB save 만 수행.
        """
        totals = {sid: self.db.size(sid) for sid in self.db.skill_ids()}
        # PlanAndSelect 가 inline append 한 결과를 디스크로 flush — judge-TRUE 시 점만 save.
        if request.judge_true:
            try:
                self.db.save(self.db_path)
            except Exception as e:
                print(f"[server] DB save failed: {e}")
        if self.debug_verbose:
            print(
                f"[server] CommitToBuffer (compat-only): judge={request.judge_true} "
                f"buffer={totals}"
            )
        return preselective_pb2.CommitResponse(
            committed=0, dropped=0, buffer_totals=totals,
        )

    # ----------------------------------------------------------------
    def IngestEpisode(self, request_iterator, context):
        """Grow the SkillVectorDB from a streamed forward demo (raw phase1).

        client streams 각 frame; server 가 encode (image+instr → VLA key, +proprio
        concat) + action chunk DCT descriptor 로 ``VectorDBEntry`` 만들어 append.
        useful-OOD 의 reference buffer (§14 의 ``B_t^{(m)} = P_phase1 ∪ D_phase2,t``)
        가 이걸로 자란다.
        """
        t0 = time.perf_counter()
        n = 0
        # config — phase2_config.yaml.reembedding.dct_coeffs 가 있으면 그것, 없으면 3.
        dct_k = int(getattr(self.stack.config, "dct_coeffs", 3))
        try:
            with self._lock:
                for fm in request_iterator:
                    images = {}
                    if fm.images_pickle:
                        images = {
                            k: decode_jpeg(v)
                            for k, v in decode_pickle(fm.images_pickle).items()
                        }
                        images = self._rename_cameras(images)
                    proprio = decode_ndarray(fm.state)
                    action_chunk = decode_ndarray(fm.action_chunk)
                    raw_instruction = str(fm.instruction or "")
                    skill_id = str(fm.skill_id or "move_to")
                    # instruction format SoT — 학습 / Phase2 inference 와 동일 분포.
                    from method3.dct.instruction_format import format_skill_instruction
                    instruction = format_skill_instruction(skill_id, raw_instruction)
                    # VLA key (zero-state — proprio 는 별도 concat in retrieval key)
                    zero_state = np.zeros_like(np.asarray(proprio, dtype=np.float64))
                    e_vla = self.encoder.encode(images, instruction, zero_state)
                    state_key = state_retrieval_key(np.asarray(e_vla, dtype=np.float64), proprio)
                    z_a = dct_action_descriptor(action_chunk, dct_k)
                    self.db.append(VectorDBEntry(
                        skill_id=skill_id,
                        state_key=state_key,
                        action_descriptor=z_a,
                        ref={"source": "ingest_episode"},
                        meta={"phase": "phase1", "instruction": instruction,
                              "accepted_by": "phase1_seed"},
                    ))
                    n += 1
                # 에피소드 끝나면 디스크 flush
                try:
                    self.db.save(self.db_path)
                except Exception as e:
                    print(f"[server] DB save after IngestEpisode failed: {e}")
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"IngestEpisode failed after {n} frames: {e}")
            return preselective_pb2.IngestResponse(frames_ingested=n)

        totals = {sid: self.db.size(sid) for sid in self.db.skill_ids()}
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
        totals = {sid: self.db.size(sid) for sid in self.db.skill_ids()}
        emb_dim = getattr(self.encoder, "embedding_dim", None)
        return preselective_pb2.ServerInfo(
            smolvla_checkpoint="<frozen VLA encoder>",
            device=str(getattr(self.encoder, "device", "cuda")),
            curobo_robot_cfg=str(getattr(self.curobo.cfg, "robot_cfg_path", "")),
            buffer_total=int(sum(totals.values())),
            buffer_per_skill=totals,
            selector_summary=(
                f"method3 Phase2MISelector tau_MI={self.stack.config.tau_MI} "
                f"key_dim={emb_dim if emb_dim is not None else '?'} "
                f"u_vla={'on' if self.vla_scorer else 'off'}"
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

    print(f"[server] building method3 phase2 Phase2MISelector + VLA encoder from {cfg_path} ...")
    # phase2_config.yaml — project_root 기준 표준 위치 시도 후 fallback.
    phase2_yaml_path = project_root / "pipeline_config" / "phase2_config.yaml"
    stack = setup_method3_phase2_server(
        recording_cfg,
        phase2_yaml=phase2_yaml_path if phase2_yaml_path.exists() else None,
    )
    if stack is None:
        raise RuntimeError(
            "method3 phase2 server stack failed — VLA checkpoint missing or "
            "phase2_config.yaml not found."
        )

    print(f"[server] loading curobo backend ({args.urdf}) ...")
    # skill perturbation 설정 — 새 phase2_config.yaml.skill_perturbation 우선,
    # legacy recording_config.perturbation.skill 도 호환.
    skill_cfg = (recording_cfg.get("perturbation") or {}).get("skill") or {}
    if phase2_yaml_path.exists():
        try:
            with phase2_yaml_path.open("r", encoding="utf-8") as _f:
                ph2 = yaml.safe_load(_f) or {}
            skill_cfg = ph2.get("skill_perturbation") or skill_cfg
        except Exception:
            pass
    curobo = _build_curobo(
        args.urdf, skill_cfg, project_root,
        transit_pitch_max_deg=recording_cfg.get("transit_pitch_max_deg"),
    )

    psf_cfg = recording_cfg.get("preselective_filter") or {}
    sel_cfg = psf_cfg.get("selector") or {}
    # phase2_config.yaml.selector 가 *진짜* SoT (recording_config 의 옛 selector 는
    # legacy). 있으면 우선.
    if phase2_yaml_path.exists():
        try:
            with phase2_yaml_path.open("r", encoding="utf-8") as _f:
                _ph2 = yaml.safe_load(_f) or {}
            sel_cfg = _ph2.get("selector") or sel_cfg
        except Exception:
            pass
    # camera_rename — raw capture 키 (top/left_wrist) → policy image feature
    # 키 (camera1/2/...). 학습 train_DCT_smolvla.sh 의 CAMERA_RENAME_PAIRS 와
    # 동일해야 VLA encoder·scorer 가 policy 와 정합.
    _camera_rename: dict = {}
    if phase2_yaml_path.exists():
        try:
            with phase2_yaml_path.open("r", encoding="utf-8") as _f:
                _ph2 = yaml.safe_load(_f) or {}
            _camera_rename = dict(_ph2.get("camera_rename") or {})
        except Exception:
            pass

    # action_horizon: 신 키. chunk_size 는 deprecated alias (spec mismatch 였던
    # 옛 yaml 호환). 둘 다 없으면 spec §7.3 의 50 으로 default.
    # T 는 trajectory 길이의 함수로 자동 (stride=1). max_T_eval 은 *옵션* cost cap.
    _action_horizon = int(sel_cfg.get("action_horizon", sel_cfg.get("chunk_size", 50)))
    _max_T_eval = sel_cfg.get("max_T_eval")
    _max_T_eval = None if _max_T_eval is None else int(_max_T_eval)
    # A.3 — joint→servo calibration JSON path. yaml > env > default.
    _servo_calib = (
        sel_cfg.get("servo_calibration_file")
        or os.environ.get("PHASE2_SERVO_CALIB")
        or "robot_configs/motor_calibration/so101/robot4_calibration.json"
    )
    # resolve relative path to repo root (server.py 의 working dir 가 repo root).
    if _servo_calib and not Path(_servo_calib).is_absolute():
        _abs = Path(__file__).resolve().parent.parent / _servo_calib
        _servo_calib = str(_abs) if _abs.exists() else _servo_calib
    if not Path(_servo_calib).exists():
        print(f"[server] WARN: servo calibration file not found at {_servo_calib} "
              f"— joint→servo conversion disabled")
        _servo_calib = None
    # URDF path 결정 — curobo robot cfg yaml 안의 urdf_path field 추출.
    # EE delta DCT FK 에 사용 (2026-05-24 전환).
    # yaml 은 module-level import (line 37) — 함수 안에서 재 import 하면
    # Python scoping 이 yaml 을 local 로 인식해 위쪽 yaml.safe_load 가
    # UnboundLocalError 발생함 (이전 버그).
    #
    # cross-machine portability: curobo cfg yaml 의 urdf_path 가 *로컬*
    # 절대경로 (예: /home/lerobot/...) 라 원격 서버에서 그대로 못 씀.
    # basename 추출 → project_root/assets/urdf/<basename> 으로 재구성.
    _proj_root = Path(__file__).resolve().parent.parent
    _urdf_path: str | None = None
    try:
        # curobo_robot_cfg_path 는 phase2_config.yaml.skill_perturbation 의 키.
        # 이전 코드는 psf_cfg (recording_config.preselective_filter) 에서 찾았는데
        # 그 section 은 비어있어 _curobo_cfg_path="" → EE delta DCT 자동 비활성.
        # skill_cfg (line 583-588 에서 phase2_config.skill_perturbation 로 load
        # 된 것) 를 우선 시도, 그래도 없으면 legacy psf_cfg 로 fallback.
        _curobo_cfg_path = (
            skill_cfg.get("curobo_robot_cfg_path")
            or psf_cfg.get("curobo_robot_cfg_path", "")
        )
        if _curobo_cfg_path:
            _ccp = Path(_curobo_cfg_path)
            if not _ccp.is_absolute():
                _ccp = _proj_root / _ccp
            if _ccp.exists():
                with open(_ccp) as _f:
                    _yaml_data = yaml.safe_load(_f) or {}
                # curobo robot yml 구조: kinematics.urdf_path (nested 1-level).
                # 1차 top-level 시도, 2차 nested kinematics.urdf_path fallback.
                _raw_urdf = _yaml_data.get("urdf_path")
                if not _raw_urdf:
                    _raw_urdf = (_yaml_data.get("kinematics") or {}).get("urdf_path")
                if _raw_urdf:
                    # 1차 시도: yaml 의 절대경로 그대로
                    if Path(_raw_urdf).exists():
                        _urdf_path = str(_raw_urdf)
                    else:
                        # 2차: basename 만 떼서 project_root/assets/urdf/ 결합
                        _basename = Path(_raw_urdf).name  # so101_robot4.urdf
                        _fallback = _proj_root / "assets" / "urdf" / _basename
                        if _fallback.exists():
                            _urdf_path = str(_fallback)
                            print(f"[server] urdf_path basename fallback: "
                                  f"{_raw_urdf} (없음) → {_urdf_path}", flush=True)
                        else:
                            print(f"[server] urdf_path 둘 다 없음: {_raw_urdf}, "
                                  f"{_fallback} → EE delta DCT 비활성", flush=True)
    except Exception as _e:
        print(f"[server] urdf_path 추출 실패: {_e}")
    servicer = PreselectiveAcquirerServicer(
        stack=stack,
        curobo_backend=curobo,
        recording_fps=int(recording_cfg.get("recording_fps", 10)),
        chunk_size=int(sel_cfg.get("chunk_size", 50)),
        debug_verbose=bool(psf_cfg.get("debug_verbose", False)),
        action_horizon=_action_horizon,
        max_T_eval=_max_T_eval,
        servo_calibration_file=_servo_calib,
        camera_rename=_camera_rename,
        urdf_path=_urdf_path,
    )
    print(
        f"[server] selector: action_horizon (H)={_action_horizon}, "
        f"max_T_eval={_max_T_eval} (None=evaluate all T=N points)"
    )
    print(f"[server] camera_rename: {_camera_rename or '<none — raw keys passed through>'}")

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
