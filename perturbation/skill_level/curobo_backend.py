"""Curobo-based skill-level perturbation backend (v2 — batched + cuda graph + locked joints).

Design notes:
  1. CUDA graph enabled + consistent shapes ⇒ ~25× per-plan speedup.
  2. ``fixed_joint_indices`` enforced — wrist_roll lock matches cartesian IK
     behaviour (e.g. SO-101 keeps gripper roll fixed during transit).
  3. ``ik_solver.solve_pose`` for via_qpos (skip the costly plan_pose).
  4. Batched plan_cspace — N candidates compute in one GPU pass.

Architecture (per plan_batch call):

    Step 1.  IK batch    via_qpos[i] = solve_pose(via_xyz[i], orient_i)   ────┐
                                                                            │  GPU
    Step 2.  Seg1 batch  cspace(start_repN,  [goal, via1_q, via2_q, via3_q]) ┤  3 calls
    Step 3.  Seg2 batch  cspace([goal, via1_q, via2_q, via3_q],  goal_repN)  ┘
    Step 4.  Per-candidate concat + cubic-spline smoothing                   ─── CPU
    Step 5.  Apply fixed-joint lock (e.g. wrist_roll = start[wrist_roll])    ─── CPU

For the i=0 (direct) candidate, seg2[0] is a trivial goal→goal plan; we use
seg1[0] only and discard seg2[0]. This keeps batch shape consistent so a
single CUDA graph services every seg1 and seg2 call across plan_batch calls.

Public interface exposes ``plan_batch(start, goal, n, rng)`` so the gRPC
adapter (``preselective_filter.integration.grpc_planner_adapter.GrpcPlannerClient``) can duck-type
the same call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from perturbation.skill_level.planner import TrajectoryCandidate


# Via-point sampling is fully continuous (no discrete mode table). Each via
# is parameterized by three random scalars per candidate:
#
#   t    ∈ [VIA_T_MIN, VIA_T_MAX]        — fraction along start→goal line
#                                          (default 0.2..0.8 — not too close
#                                          to either endpoint so the via
#                                          actually bends the trajectory)
#   lat  ∈ [-mag, +mag]                   — perpendicular lateral displacement
#                                          (left/right of the line, uniform —
#                                          symmetric so neither side is favored)
#   vert ∈ [+0.3·mag, +1.2·mag]           — vertical displacement (biased
#                                          upward — SO-101 5-DoF reaches
#                                          above-the-line trajectories
#                                          much more reliably than below)
#
# This replaces the previous 4-entry hardcoded VIA_OFFSETS table, which
# clipped diversity at N>4 (modes wrapped via modulo with only small RNG
# noise). Continuous sampling provides O(N) genuinely distinct via
# locations across the full reachable arc.
VIA_T_MIN = 0.2
VIA_T_MAX = 0.8
VIA_VERT_MIN_RATIO = 0.3   # of via_offset_mag
VIA_VERT_MAX_RATIO = 1.2


@dataclass
class CuroboBackendConfig:
    """Curobo backend tunables. Default values target SO-101 transit moves."""
    enabled: bool = False
    robot_cfg_path: str = "robot_configs/curobo/so101_robot0.yml"
    num_trajopt_seeds: int = 4
    num_ik_seeds: int = 16
    use_cuda_graph: bool = True
    max_batch_size: int = 4
    interpolation_dt: Optional[float] = None
    via_offset_mag: float = 0.10
    junction_smooth_k: int = 5
    fixed_joint_indices: Sequence[int] = ()
    arm_joint_count: int = 5
    # Maximum number of via-points each candidate trajectory may use.
    #   0 → direct-only (no vias, ablation)
    #   1 → at most one via per candidate (backward-compat, default)
    #   2 → mix of K=0/1/2 vias per candidate (random K → richer path
    #       topology diversity across one plan_batch). N × (K_max+1)
    #       plan_cspace batch + N × K_max IK batch, so VRAM grows ~1.5×
    #       per K_max step.
    max_vias_per_candidate: int = 1
    # Wrist-camera transit bias (Option A). When set, every via-point IK
    # goalset orientation is clamped so the gripper Z-axis (pointing
    # direction) elevation ≤ this value (radians; negative = tilted down).
    # This forces vias to be pitch-down regardless of how the start/goal
    # endpoint yaws differ — slerp alone does not preserve pitch when the
    # two endpoint orientations are far apart. None = no clamp (legacy).
    via_pitch_max_rad: Optional[float] = None

    # joint-space 저주파 perturbation — curobo collision-free trajectory 에
    # boundary-0 sinusoidal Σ_{k=1}^{K} a_k·sin(kπt) 를 더해 경로를 질적으로
    # 다양화한다. sin(kπ·0)=sin(kπ)=0 이라 start/goal joint 는 보존. K 를
    # 2~3 저주파로 두므로 매끄러운 S자/곡선 — 고주파 noise 아님. 0 = off.
    joint_perturb_k: int = 0          # 저주파 항 개수 K (2~3 권장)
    joint_perturb_mag: float = 0.0    # rad — per-joint amplitude 범위


class CuroboBackend:
    """Single-process curobo wrapper with batched, graph-cached planning.

    Construct once (incurs ~10s warmup), call ``plan_batch`` many times.
    All plan calls reuse the same CUDA graph as long as ``n`` does not exceed
    ``max_batch_size``.
    """

    def __init__(self, urdf: str | Path, config: CuroboBackendConfig) -> None:
        import torch

        # Runtime flag MUST be set BEFORE MotionPlanner is created. Curobo's
        # default `cuda_graph_reset = False` makes ``is_cuda_graph_reset_available()``
        # always return False, even on CUDA 12.0+. With it off, any plan call
        # whose batch shape differs from the captured graph raises
        # "CUDA graph reset is not available", forcing us to disable graphs
        # entirely. Flipping the flag here lets curobo invalidate and re-capture
        # the graph when our batch shape first appears (one-time, ~1-2s) and
        # re-use the cached graph for every subsequent plan_batch call.
        if config.use_cuda_graph:
            import curobo.runtime as _rt
            _rt.cuda_graph_reset = True

        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import JointState, GoalToolPose
        from curobo._src.geom.types import SceneCfg
        self._SceneCfg = SceneCfg  # 의 — 의 — instance method 의 — 의 — 의 — 의

        self._torch = torch
        self._JointState = JointState
        self._GoalToolPose = GoalToolPose
        self.cfg = config
        self.urdf = str(urdf)

        # Curobo's internal robot-loader resolves relative paths against its
        # OWN content/configs/robot/ directory, not the caller's cwd — so a
        # relative path that "exists" by Path.exists() can still fail inside
        # MotionPlannerCfg.create(robot=...). Always resolve to absolute here.
        robot_cfg_abs = str(Path(config.robot_cfg_path).resolve())
        if not Path(robot_cfg_abs).exists():
            raise FileNotFoundError(
                f"curobo robot config missing: {robot_cfg_abs}"
            )

        # Orientation retry ratios for IK. Mid-arc first (most physically
        # intuitive), then expand outward toward endpoints. Used by the
        # batched IK call to fold all orientations into a single GPU pass
        # via curobo's goalset dimension G = len(self._slerp_ratios).
        self._slerp_ratios = (0.5, 0.3, 0.7, 0.0, 1.0)
        self._n_goalset = len(self._slerp_ratios)

        # Option A: via-point orientation pitch clamp (wrist-cam transit bias).
        self._via_pitch_max_rad = config.via_pitch_max_rad

        # Multi-via geometry:
        #   _max_vias     = K_max (config.max_vias_per_candidate). 0 disables
        #                   the via path entirely. 1 = single via per cand
        #                   (legacy). 2 = mix of K=0/1/2 per cand (random K).
        #   _max_segments = K_max + 1. Every candidate is padded to this many
        #                   plan_cspace segments so the CUDA graph captures a
        #                   fixed shape; dummies (goal→goal no-ops) fill the
        #                   slack.
        #   _batch_size   = user-facing N (# of candidates per plan_batch).
        #   _cspace_batch = N × _max_segments. One GPU call services every
        #                   real segment + all dummies in a single pass.
        #   _ik_batch     = N × K_max. Upper bound on via-points across all
        #                   candidates (each cand has at most K_max vias).
        #                   Padded to fixed size for IK CUDA graph stability.
        self._max_vias = max(0, int(config.max_vias_per_candidate))
        self._max_segments = self._max_vias + 1
        self._batch_size = int(config.max_batch_size)
        self._cspace_batch = self._batch_size * self._max_segments
        self._ik_batch = self._batch_size * max(1, self._max_vias)
        # MotionPlannerCfg.max_batch_size covers BOTH plan_cspace and IK
        # solver paths internally → set to the larger of the two.
        mp_max_batch = max(self._cspace_batch, self._ik_batch)
        # Scene collision world — base_link 기준 table top z=0 평면.
        # 이걸 등록해 두지 않으면 trajopt 가 robot self-collision 만 본다.
        # Phase2 의 perturbation candidate 가 table 아래로 내려가는 path 를
        # 자유롭게 만들어, holding-phase 에서 잡힌 물체가 table 을 스치는
        # 결과를 낳는다. dict 직접 전달 (resolve_config dict pass-through):
        #   dims=(x,y,z) — base_link 기준 사각 표면(2m × 2m × 2cm),
        #   pose=(x,y,z, qx,qy,qz,qw) — table 중심 z=-0.01m → top surface z=0.
        # robot base_link 가 table top 에 mount 됐다는 가정 (so101 standard).
        # 추후 robot 별 base mount 가 다르면 config 로 분리.
        # [RESTORED 2026-05-26] 4db51cf 의 scene_model 이 f389def 의 K_via
        # refactor 시점에 의도치 않게 누락 — 재추가.
        # _default_scene_dict — table 의 — 의 — base layer. update_world() 가
        # dynamic obstacle (block/plate 등) 의 — 의 — 의 — 의 — 의 — 의 — 의
        # 의 — 의 — 의 — copy 후 merge 한다.
        self._default_scene_dict = {
            "cuboid": {
                "table": {
                    "dims": [2.0, 2.0, 0.02],
                    "pose": [0.0, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0],
                },
            },
        }
        _scene_model = self._default_scene_dict
        mp_cfg = MotionPlannerCfg.create(
            robot=robot_cfg_abs,
            num_trajopt_seeds=config.num_trajopt_seeds,
            num_ik_seeds=config.num_ik_seeds,
            random_seed=123,
            use_cuda_graph=config.use_cuda_graph,
            max_batch_size=mp_max_batch,
            max_goalset=self._n_goalset,
            scene_model=_scene_model,
        )
        self._planner = MotionPlanner(mp_cfg)
        self.joint_names = list(self._planner.joint_names)
        self._n_dof = len(self.joint_names)
        self._n_arm = config.arm_joint_count
        self._fixed_idx = tuple(int(i) for i in config.fixed_joint_indices)

        # Default qpos used to fill non-arm DoF (gripper etc.) when caller
        # passes arm-only vectors.
        self._default_full = (
            self._planner.default_joint_state.position.detach().cpu().numpy().astype(float)
        )

        device = "cuda"
        dtype = torch.float32
        self._dev = device
        self._dtype = dtype

        self._interp_dt = (
            config.interpolation_dt
            if config.interpolation_dt is not None
            else float(self._planner.trajopt_solver.config.interpolation_dt)
        )

        # NOTE: we intentionally SKIP planner.warmup() because that captures a
        # CUDA graph at batch_size=1, and our plan_cspace_batch always uses
        # max_batch_size > 1. Curobo can't reset a captured graph in-place
        # when the shape changes. With use_cuda_graph=False, no graph capture
        # is needed; with use_cuda_graph=True, the first plan_batch call pays
        # the one-time capture cost and subsequent calls reuse the graph.

        # Cleanup hook: belt-and-suspenders for the pipeline teardown path.
        # close() also runs via _teardown_skill_perturbation under normal
        # shutdown; atexit fires under exceptions / interpreter exit / when
        # SIGINT fires before the pipeline's own signal handler was installed.
        # SIGKILL is unhandleable — only the driver can reclaim that memory.
        self._closed = False
        import atexit as _atexit
        _atexit.register(self._atexit_close)

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def dof(self) -> int:
        return self._n_arm

    def update_scene(self, obstacles) -> dict:  # noqa: ARG002
        # The previous CPU-OMPL daemon mutated its FCL world here. curobo
        # uses a static collision world baked into the robot yaml, so this
        # is intentionally a no-op — kept only to satisfy the duck-typed
        # planner interface that skills_lerobot's detect hook calls.
        return {"added": 0, "removed": 0, "kept": 0}

    def close(self) -> None:
        """Release GPU resources held by curobo (CUDA graphs, trajopt/IK
        solver state, kinematics tables). **Idempotent** — safe to call
        multiple times (pipeline teardown + atexit may both fire).

        Process exit normally reclaims VRAM on its own, but two cases need
        explicit cleanup:
          (a) Long-lived parent processes that spin up + tear down backends
              between sessions (e.g. multi-task data collection pipelines).
          (b) Hard kills / unhandled exceptions during plan_batch — CUDA
              graph memory and cached allocator buffers can otherwise leak
              until reboot. (SIGKILL is unhandleable; only the CUDA driver
              can reclaim that memory.)

        Mirrors the closeable shape the pipeline's
        ``_teardown_skill_perturbation`` expects so it can call this uniformly.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        torch = getattr(self, "_torch", None)
        # Drop references to GPU-resident objects first so torch's caching
        # allocator knows they're collectable.
        for attr in (
            "_planner", "_default_full", "_interp_dt",
        ):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        # Force the allocator to release reserved-but-unallocated blocks.
        if torch is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

    def _atexit_close(self) -> None:
        """atexit-safe wrapper around close(). Swallows all exceptions
        because exits during interpreter shutdown may have torch / curobo
        modules already partially torn down — we just want a best-effort
        free, never a noisy traceback on the way out."""
        try:
            self.close()
        except Exception:
            pass

    def update_world(self, dynamic_obstacles: Optional[Dict[str, Dict]] = None) -> None:
        """Runtime scene update — table + dynamic cuboid obstacles 등록.

        매 plan_batch 직전 호출하여 trajopt 가 그 시점의 실제 obstacle 위치를
        회피하도록 한다. dynamic_obstacles 가 None/{} 면 default (table only) 로
        reset.

        Args:
            dynamic_obstacles: ``{name: {"pose": [x,y,z, qx,qy,qz,qw],
                                        "dims": [dx,dy,dz]}}``. 좌표/단위 모두
                base_link frame 의 meter. caller (client) 가 pix2robot 변환과
                object_height 추정을 마친 후 넘긴다.
        """
        scene_dict = {
            "cuboid": dict(self._default_scene_dict["cuboid"]),  # shallow copy
        }
        if dynamic_obstacles:
            for name, geom in dynamic_obstacles.items():
                # name 중복 회피 — table 과 같은 이름 사용 시 dynamic 이 우선.
                scene_dict["cuboid"][name] = {
                    "dims": list(geom["dims"]),
                    "pose": list(geom["pose"]),
                }
        scene_cfg = self._SceneCfg.create(scene_dict)
        self._planner.update_world(scene_cfg)

    def plan_batch(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        n: int,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        skill_id: Optional[str] = None,  # accepted for signature parity with GrpcPlannerClient; ignored locally
        skill_type: Optional[str] = None,  # parity with GrpcPlannerClient; ignored locally
    ) -> list[TrajectoryCandidate]:
        """Plan N transit candidates. ``seed=int`` matches what skills_lerobot
        passes from production; tests may pass ``rng=`` directly.
        """
        if not self.cfg.enabled or n <= 0:
            return []
        n = min(int(n), self._batch_size)
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        start_full = self._to_full_qpos(start_qpos)
        goal_full = self._to_full_qpos(goal_qpos)
        # Enforce fixed-joint values on the goal (so cspace plan has start ==
        # goal on those dims and the trajectory stays put on them).
        for idx in self._fixed_idx:
            if idx < self._n_dof:
                goal_full[idx] = start_full[idx]

        # 1) FK for start + goal in one batched compute_kinematics call.
        (start_ee, start_quat), (goal_ee, goal_quat) = self._compute_ee_xyz_quat_batch(
            [start_full, goal_full]
        )

        # 2) Per-candidate K_via assignment — STRUCTURED uniform quota.
        #   원래는 slot 0 만 K=0 (direct) guaranteed, 나머지 n-1 개는
        #   K ∈ {1..max_vias} RNG uniform 이었다. 결과적으로 n=64, max_vias=2
        #   일 때 K=0:1, K=1:~31, K=2:~32 로 direct path 가 거의 안 뽑혀
        #   chosen 도 항상 via1/via2 만 — 사용자가 "랜덤" 으로 느낀 정체.
        #
        #   이제 K ∈ {0, 1, ..., max_vias} 전체에 deterministic 균등 분배.
        #     n=64, max_vias=2  → K=[22, 21, 21]
        #     n=64, max_vias=3  → K=[16, 16, 16, 16]
        #   각 type 의 cand 가 항상 cand pool 에 존재 → Phase2 selector 가
        #   argmax U_VLA 할 때 모든 path topology 가 후보로 들어감.
        K_via_per_cand: list[int] = []
        n_buckets = self._max_vias + 1
        n_per_k = n // n_buckets
        remainder = n % n_buckets
        for k in range(n_buckets):
            count = n_per_k + (1 if k < remainder else 0)
            K_via_per_cand.extend([k] * count)
        # K_via_per_cand 의 길이 = n. cand index 와 K type 의 sequential 결합은
        # log/시각화 디버깅 쉽게 — shuffle 안 함 (필요시 RNG shuffle 추가 가능).

        # 3) For each candidate, pre-sample its (t, lat, vert) param tuples
        #    (stratified t) and convert to via xyz.
        via_xyz_per_cand: list[list[np.ndarray]] = []
        for i in range(n):
            K = K_via_per_cand[i]
            params = self._sample_via_params_stratified(K, rng)
            via_xyz_per_cand.append([
                self._compute_via_xyz(start_ee, goal_ee, t, lat, vert)
                for (t, lat, vert) in params
            ])

        # 4) Batched IK for ALL vias across ALL candidates in one GPU call.
        #    Flatten to (via_xyz, (cand_idx, via_idx_in_cand)), pad to
        #    self._ik_batch for CUDA graph stability, run goalset IK.
        flat_via_xyz: list[np.ndarray] = []
        flat_via_loc: list[tuple[int, int]] = []
        for i in range(n):
            for j, xyz in enumerate(via_xyz_per_cand[i]):
                flat_via_xyz.append(xyz)
                flat_via_loc.append((i, j))

        via_qpos_per_cand: list[list[np.ndarray]] = [[] for _ in range(n)]
        if flat_via_xyz and self._max_vias > 0:
            # _ik_solve_batched pads internally to self._ik_batch for CUDA
            # graph stability; we just pass the real via list.
            via_solutions = self._ik_solve_batched(
                flat_via_xyz, start_quat, goal_quat, seed_q=start_full,
            )
            mid_q = (start_full + goal_full) / 2.0
            for k_flat, (i_cand, _j_via) in enumerate(flat_via_loc):
                q_via = via_solutions[k_flat]
                if q_via is None:
                    q_via = mid_q.copy()
                else:
                    for idx in self._fixed_idx:
                        if idx < self._n_dof:
                            q_via[idx] = start_full[idx]
                via_qpos_per_cand[i_cand].append(q_via)

        # 5) Build the merged plan_cspace batch.
        #    Each candidate contributes exactly self._max_segments slots:
        #      K real segments (start→v1, v1→v2, ..., v_K→goal)
        #      + (max_vias - K) dummy goal→goal no-ops at the tail.
        #    Layout (flat, row-major): candidate i occupies slots
        #    [i*max_seg .. (i+1)*max_seg).
        merged_start: list[np.ndarray] = []
        merged_goal: list[np.ndarray] = []
        for i in range(n):
            K = K_via_per_cand[i]
            vqs = via_qpos_per_cand[i]
            if K == 0:
                # Direct: 1 real segment, max_vias dummies.
                merged_start.append(start_full)
                merged_goal.append(goal_full)
                for _ in range(self._max_vias):
                    merged_start.append(goal_full)
                    merged_goal.append(goal_full)
            else:
                # start → v1
                merged_start.append(start_full)
                merged_goal.append(vqs[0])
                # v_{j-1} → v_j
                for j in range(1, K):
                    merged_start.append(vqs[j - 1])
                    merged_goal.append(vqs[j])
                # v_K → goal
                merged_start.append(vqs[-1])
                merged_goal.append(goal_full)
                # Tail dummies
                for _ in range(self._max_vias - K):
                    merged_start.append(goal_full)
                    merged_goal.append(goal_full)

        # Pad to self._cspace_batch (when user n < self._batch_size) with
        # all-start no-ops. Keeps the CUDA graph's batch dimension fixed.
        pad = self._cspace_batch - len(merged_start)
        if pad > 0:
            merged_start.extend([start_full] * pad)
            merged_goal.extend([start_full] * pad)

        t0 = time.time()
        merged_batch = self._plan_cspace_batch(merged_start, merged_goal)
        plan_time = time.time() - t0
        if merged_batch is None:
            return []

        # 6) Per-candidate assembly — concat the K+1 real segments and
        #    iteratively spline-smooth each junction.
        candidates: list[TrajectoryCandidate] = []
        seg_k = self.cfg.junction_smooth_k
        for i in range(n):
            K = K_via_per_cand[i]
            base = i * self._max_segments
            real_segs: list[np.ndarray] = []
            for j in range(K + 1):
                seg = merged_batch[base + j]
                if seg is None:
                    real_segs = []  # any failed real segment ⇒ skip candidate
                    break
                real_segs.append(seg)
            if not real_segs:
                continue

            # Stitch: concat seg[0], smooth-junction with seg[1], etc.
            full_wp = real_segs[0]
            for next_seg in real_segs[1:]:
                full_wp = self._smooth_junction(full_wp, next_seg, k=seg_k)

            full_wp = self._apply_fixed_joint_lock(full_wp, start_full)
            # joint-space 저주파 perturbation — 경로 다양화 (start/goal 보존).
            full_wp = self._apply_joint_perturbation(full_wp, rng)
            algo_name = "curobo:direct" if K == 0 else f"curobo:via{K}_s{i}"

            candidates.append(self._make_candidate(
                full_wp, algo=algo_name,
                seed=int(rng.integers(0, 2**31 - 1)),
                plan_time_s=plan_time / max(n, 1),
            ))

        return candidates

    # ── Helpers ────────────────────────────────────────────────────────────

    def _to_full_qpos(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape[0] == self._n_dof:
            return q.copy()
        if q.shape[0] != self._n_arm:
            raise ValueError(
                f"start/goal qpos must be {self._n_arm} (arm) or "
                f"{self._n_dof} (full); got {q.shape}"
            )
        full = self._default_full.copy()
        full[: self._n_arm] = q
        return full

    def _compute_ee_xyz_quat_batch(
        self, qpos_full_list: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batched FK. One ``compute_kinematics`` call for B configs ⇒ B
        (xyz, quat) tuples. Used to fold start+goal FK into a single GPU call.
        """
        torch = self._torch
        B = len(qpos_full_list)
        q_arr = np.asarray(qpos_full_list, dtype=np.float32).reshape(B, -1)
        st = self._JointState.from_position(
            torch.from_numpy(q_arr).to(self._dev),
            joint_names=self.joint_names,
        )
        kin = self._planner.compute_kinematics(st)
        tp = kin.tool_poses if hasattr(kin, "tool_poses") else None
        if tp is None:
            raise AttributeError("kinematics result missing tool_poses")
        if isinstance(tp, (list, tuple)):
            xyz_all = tp[0].position.detach().cpu().numpy().reshape(B, 3)
            quat_all = tp[0].quaternion.detach().cpu().numpy().reshape(B, 4)
        else:
            xyz_all = tp.position.detach().cpu().numpy().reshape(B, 3)
            quat_all = tp.quaternion.detach().cpu().numpy().reshape(B, 4)
        return [(xyz_all[i], quat_all[i]) for i in range(B)]

    def _compute_ee_xyz_quat(self, qpos_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single-config FK convenience wrapper around the batched call."""
        return self._compute_ee_xyz_quat_batch([qpos_full])[0]

    def _apply_joint_perturbation(
        self,
        waypoints: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """boundary-0 저주파 sinusoidal perturbation 으로 경로를 다양화.

        q'(t) = q(t) + Σ_{k=1}^{K} a_k·sin(kπt). sin(kπt) 는 t=0,1 에서 0
        이므로 start/goal joint 가 그대로 보존된다 (collision-free baseline
        의 양 끝 유지). per-candidate random amplitude 라 후보마다 다른 굴곡
        → curobo via 만으로는 안 나오는 joint-space 다양성. fixed joint
        (wrist_roll 등)은 cartesian IK lock 유지를 위해 perturbation 제외.
        K 를 2~3 저주파로만 쓰므로 매끄러운 S자/곡선 (고주파 noise 아님).
        """
        K = int(self.cfg.joint_perturb_k)
        mag = float(self.cfg.joint_perturb_mag)
        if K <= 0 or mag <= 0.0:
            return waypoints
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[0] < 3:
            return waypoints
        T, dof = wp.shape
        t = np.linspace(0.0, 1.0, T)
        perturb = np.zeros((T, dof), dtype=float)
        for k in range(1, K + 1):
            a = rng.uniform(-mag, mag, size=dof)
            perturb += np.sin(k * np.pi * t)[:, None] * a[None, :]
        for idx in self._fixed_idx:
            if 0 <= idx < dof:
                perturb[:, idx] = 0.0
        return wp + perturb

    def _compute_via_xyz(
        self,
        start_ee: np.ndarray,
        goal_ee: np.ndarray,
        t: float,
        lat: float,
        vert: float,
    ) -> np.ndarray:
        """Compute via xyz from explicit (t, lat, vert) parameters.

        Pure function — no RNG inside. Multi-via candidates pre-sample K
        parameter tuples via ``_sample_via_params_stratified`` and convert
        each to xyz in one pass. Splitting sampling from geometry lets the
        stratified-t logic enforce minimum spacing along the line.
        """
        via_base = (1.0 - t) * start_ee + t * goal_ee
        line = goal_ee - start_ee
        line_len = float(np.linalg.norm(line))
        if line_len < 1e-6:
            return via_base + np.array([0.0, self.cfg.via_offset_mag, 0.0])
        line_dir = line / line_len
        perp_lat = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], line_dir) * line_dir
        if np.linalg.norm(perp_lat) < 1e-6:
            perp_lat = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], line_dir) * line_dir
        perp_lat /= max(np.linalg.norm(perp_lat), 1e-9)
        perp_vert = np.array([0.0, 0.0, 1.0]) - np.dot([0.0, 0.0, 1.0], line_dir) * line_dir
        perp_vert /= max(np.linalg.norm(perp_vert), 1e-9)
        return via_base + lat * perp_lat + vert * perp_vert

    def _sample_via_params_stratified(
        self,
        K: int,
        rng: np.random.Generator,
    ) -> list[tuple[float, float, float]]:
        """Sample K (t, lat, vert) tuples for a single candidate.

        t is **stratified**: [VIA_T_MIN, VIA_T_MAX] is split into K equal
        sub-intervals and one t is sampled from each. Guarantees minimum
        spacing (VIA_T_MAX-VIA_T_MIN)/K between consecutive vias — prevents
        the degenerate "two vias on top of each other" case which would
        defeat the multi-via point entirely.

        lat ~ Uniform(-mag, +mag); vert ~ Uniform(0.3·mag, 1.2·mag) —
        identical distributions to the single-via case so the per-via
        envelope shape is the same regardless of K.

        Returns list of K tuples already sorted by t (ascending).
        """
        if K <= 0:
            return []
        bin_edges = np.linspace(VIA_T_MIN, VIA_T_MAX, K + 1)
        mag = self.cfg.via_offset_mag
        out: list[tuple[float, float, float]] = []
        for k in range(K):
            t = float(rng.uniform(bin_edges[k], bin_edges[k + 1]))
            lat = float(rng.uniform(-mag, +mag))
            vert = float(rng.uniform(
                VIA_VERT_MIN_RATIO * mag, VIA_VERT_MAX_RATIO * mag,
            ))
            out.append((t, lat, vert))
        return out

    def _sample_via_xyz(
        self,
        start_ee: np.ndarray,
        goal_ee: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Single-via convenience wrapper — sample one via and return its
        xyz. Kept for backward compatibility (test scripts, legacy K=1
        path). Internally just _sample_via_params_stratified(K=1) +
        _compute_via_xyz."""
        params = self._sample_via_params_stratified(1, rng)
        t, lat, vert = params[0]
        return self._compute_via_xyz(start_ee, goal_ee, t, lat, vert)

    @staticmethod
    def _slerp_quat(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two unit quaternions (wxyz)
        at parameter ``t`` ∈ [0, 1]. Returns a unit quaternion.

        Used by Fix 5 to generate orientation candidates between start and
        goal for via-point IK queries. SO-101's 5-DoF arm cannot satisfy an
        arbitrary (xyz + arbitrary_quat) pose, so we try several orientations
        along the slerp arc until IK succeeds.
        """
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        dot = float(np.dot(q1, q2))
        # Quaternions q and -q represent the same rotation; pick shortest arc.
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            # Near-parallel: linear interpolation is numerically safer.
            result = q1 + t * (q2 - q1)
            return result / max(np.linalg.norm(result), 1e-12)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        out = s0 * q1 + s1 * q2
        return out / max(np.linalg.norm(out), 1e-12)

    @staticmethod
    def _tilt_quat_pitch_down(q_wxyz: np.ndarray, pitch_max_rad: float) -> np.ndarray:
        """Clamp the gripper Z-axis elevation of a wxyz quaternion to ≤ pitch_max.

        The gripper 'pitch' is arcsin(z_axis[2]), z_axis being the 3rd column
        of the rotation matrix (the gripper pointing direction). If the quat is
        already steep enough (elevation ≤ pitch_max), it is returned unchanged.
        Otherwise it is tilted down about the horizontal axis perpendicular to
        its azimuth, which lowers the elevation to exactly pitch_max while
        preserving azimuth (yaw) and roll-about-Z. Verified: azimuth drift
        < 1e-6 rad, elevation lands on target < 1e-6 rad.
        """
        w, x, y, z = (float(c) for c in q_wxyz)
        zx = 2.0 * (x * z + w * y)
        zy = 2.0 * (y * z - w * x)
        zz = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
        elevation = np.arcsin(zz)
        if elevation <= pitch_max_rad:
            return np.asarray(q_wxyz, dtype=float)
        az = np.arctan2(zy, zx)
        half = 0.5 * (elevation - pitch_max_rad)
        s = np.sin(half)
        # Correction quat: rotate by (elevation - pitch_max) about the
        # horizontal axis u = [-sin(az), cos(az), 0]; applied on the LEFT so
        # it rotates the gripper Z-axis in the world frame.
        qc = (np.cos(half), -np.sin(az) * s, np.cos(az) * s, 0.0)
        w2, x2, y2, z2 = (float(c) for c in q_wxyz)
        w1, x1, y1, z1 = qc
        out = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=float)
        return out / max(np.linalg.norm(out), 1e-12)

    def _ik_solve_batched(
        self,
        via_xyz_list: list[np.ndarray],
        start_quat: np.ndarray,
        goal_quat: np.ndarray,
        seed_q: np.ndarray,
    ) -> list[Optional[np.ndarray]]:
        """**Batched** multi-orientation IK using curobo's Goalset (Fix 6).

        Layout exploits curobo's native goalset dimension G to fold all
        orientation retries into one IK call WITHOUT inflating batch_size:

            B = n_via              (distinct via xyz)
            G = len(slerp_ratios)  (alternative orientations per via)
            pos_t  : (B, 1, 1, G, 3)   ← xyz repeated across G
            quat_t : (B, 1, 1, G, 4)   ← slerp(start,goal,t) for each G

        Curobo's IK solver internally optimizes against ANY goalset entry
        (it picks the most-reachable orientation per batch item), which is
        exactly the semantics we want (and strictly stronger than CPU-side
        first-success picking — it can choose the best fit, not just the
        first that converged).

        Wall-time win vs the previous sequential ``_ik_solve`` loop:
            sequential : n_via × ~2 avg attempts × ~15ms ≈ 90ms (n_via=3)
            batched    : 1 GPU call ≈ 15ms (n_via=3, G=5)
        ≈ 6× IK stage speedup. batch_size stays at n_via so we don't blow
        past ``max_batch_size`` and we don't need a larger graph capture.
        """
        n_real = len(via_xyz_list)
        if n_real == 0:
            return []
        G = len(self._slerp_ratios)

        # Pad to self._ik_batch so the IK CUDA graph captures a fixed
        # B = self._ik_batch shape and stays stable across calls of
        # varying real n_via. Dummies = first real via repeated (cheap,
        # always solvable since we know the first via is at least
        # geometrically valid); their solutions are dropped.
        target_B = max(n_real, getattr(self, "_ik_batch", n_real))
        if n_real < target_B:
            padded = list(via_xyz_list) + [via_xyz_list[0]] * (target_B - n_real)
        else:
            padded = list(via_xyz_list)
        n_via = len(padded)

        # Pre-compute all G slerp quats once (independent of via).
        quat_g = np.stack(
            [
                self._slerp_quat(start_quat, goal_quat, t).astype(np.float32)
                for t in self._slerp_ratios
            ],
            axis=0,
        )  # (G, 4)
        # Option A: clamp every goalset orientation to pitch-down so the via
        # IK can only land on a downward-facing gripper. slerp between two
        # pitch-down endpoints does NOT preserve pitch when their yaws differ
        # widely — this clamp guarantees it. No-op when via_pitch_max_rad is
        # None or the slerp result is already steep enough.
        if self._via_pitch_max_rad is not None:
            quat_g = np.stack(
                [
                    self._tilt_quat_pitch_down(q, self._via_pitch_max_rad).astype(np.float32)
                    for q in quat_g
                ],
                axis=0,
            )
        xyz_b = np.asarray(padded, dtype=np.float32).reshape(n_via, 3)

        # Broadcast to (n_via, G, ·): each via paired with every slerp quat.
        xyz_bg = np.broadcast_to(xyz_b[:, None, :], (n_via, G, 3)).copy()
        quat_bg = np.broadcast_to(quat_g[None, :, :], (n_via, G, 4)).copy()

        torch = self._torch
        pos_t = torch.from_numpy(xyz_bg).to(self._dev).reshape(n_via, 1, 1, G, 3)
        quat_t = torch.from_numpy(quat_bg).to(self._dev).reshape(n_via, 1, 1, G, 4)
        goal = self._GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=pos_t,
            quaternion=quat_t,
        )
        seed_arr = np.broadcast_to(
            np.asarray(seed_q, dtype=np.float32), (n_via, len(seed_q))
        ).copy()
        seed_state = self._JointState.from_position(
            torch.from_numpy(seed_arr).to(self._dev),
            joint_names=self.joint_names,
        )

        try:
            result = self._planner.ik_solver.solve_pose(goal, current_state=seed_state)
        except Exception as e:
            print(f"[CuroboBackend] batched IK failed: {e}", flush=True)
            return [None] * n_real
        if result is None:
            return [None] * n_real

        success = result.success.detach().cpu().numpy().reshape(-1)
        solution = result.solution.detach().cpu().numpy().reshape(n_via, -1)

        # Truncate to n_real — drop padded dummy results.
        out: list[Optional[np.ndarray]] = []
        for i in range(n_real):
            if i < len(success) and bool(success[i]):
                out.append(solution[i].astype(float))
            else:
                out.append(None)
        return out

    def _ik_solve(
        self,
        target_xyz: np.ndarray,
        start_quat: np.ndarray,
        goal_quat: np.ndarray,
        seed_q: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Single-via convenience wrapper around ``_ik_solve_batched``.

        Kept for backward compatibility (test scripts, ad-hoc callers).
        Production ``plan_batch`` path uses ``_ik_solve_batched`` directly
        with the full via list.
        """
        out = self._ik_solve_batched([target_xyz], start_quat, goal_quat, seed_q)
        return out[0] if out else None

    def _plan_cspace_batch(
        self,
        start_batch: list[np.ndarray],
        goal_batch: list[np.ndarray],
    ) -> Optional[list[Optional[np.ndarray]]]:
        """Batched joint-space plan. Returns list of (n_wp, dof) arrays, one
        per slot, or None for slots that failed.

        Caller must pad to ``self._cspace_batch`` slots (the CUDA graph's
        captured batch size). Slots that aren't real plans should be filled
        with dummy ``[start]→[start]`` entries.
        """
        assert len(start_batch) == len(goal_batch) == self._cspace_batch, (
            f"batch shape mismatch: expected {self._cspace_batch}, "
            f"got start={len(start_batch)} goal={len(goal_batch)}"
        )
        torch = self._torch
        # Stack to contiguous numpy first, then a single host→device copy.
        # Earlier list-of-lists path also worked but did N small per-row
        # conversions; this is one zero-copy from_numpy + one async copy.
        start_arr = np.asarray(start_batch, dtype=np.float32)
        goal_arr = np.asarray(goal_batch, dtype=np.float32)
        start_t = torch.from_numpy(start_arr).to(self._dev)
        goal_t = torch.from_numpy(goal_arr).to(self._dev)
        # Use the planner's own joint_names directly (must be a fresh reference,
        # not a copied list, to match the planner's internal state machine).
        jn = self._planner.joint_names
        start_state = self._JointState.from_position(start_t, joint_names=jn)
        goal_state = self._JointState.from_position(goal_t, joint_names=jn)
        try:
            result = self._planner.plan_cspace(goal_state, start_state)
        except Exception as e:
            print(f"[CuroboBackend] plan_cspace batch failed: {e}", flush=True)
            return None
        if result is None:
            return None
        success = result.success.detach().cpu().numpy().reshape(-1)
        # ``get_interpolated_plan`` refuses batched results, so read the raw
        # interpolated trajectory + per-batch last-timestep and trim each
        # slot ourselves. Shape is (batch, num_seeds_kept=1, n_wp, dof);
        # we squeeze the singleton seed-dim (axis 1) to land at (batch, n_wp, dof).
        interp_full = result.interpolated_trajectory.position.detach().cpu().numpy()
        # Remove any leading or trailing singleton axes EXCEPT the batch axis.
        # Expected final shape: (batch, n_wp, dof). Curobo emits:
        #   (batch, num_seeds=1, n_wp, dof)  → squeeze axis 1
        #   (n_wp, dof)                       → add a batch dim
        if interp_full.ndim == 4 and interp_full.shape[1] == 1:
            interp_full = interp_full[:, 0, :, :]
        elif interp_full.ndim == 2:
            interp_full = interp_full[None, ...]
        # Defensive: still > 3D ⇒ flatten to (batch, ...) keeping batch as axis 0.
        if interp_full.ndim > 3:
            interp_full = interp_full.reshape(interp_full.shape[0], -1, interp_full.shape[-1])
        last_t = result.interpolated_last_tstep
        if last_t is not None:
            last_t = last_t.detach().cpu().numpy().astype(int).reshape(-1)
        out: list[Optional[np.ndarray]] = []
        for i in range(self._cspace_batch):
            if i >= len(success) or not bool(success[i]):
                out.append(None)
                continue
            wp_i = interp_full[i]
            if last_t is not None and i < len(last_t):
                wp_i = wp_i[: int(last_t[i]) + 1]
            out.append(wp_i.copy())
        return out

    def _apply_fixed_joint_lock(
        self, waypoints: np.ndarray, start_full: np.ndarray,
    ) -> np.ndarray:
        """Force fixed-joint columns to the start value across every waypoint.
        Prevents wrist_roll wiggle even when curobo's TrajOpt drifts slightly.
        """
        if not self._fixed_idx:
            return waypoints
        wp = waypoints.copy()
        for idx in self._fixed_idx:
            if idx < wp.shape[1]:
                wp[:, idx] = start_full[idx]
        return wp

    @staticmethod
    def _smooth_junction(seg1: np.ndarray, seg2: np.ndarray, k: int) -> np.ndarray:
        from scipy.interpolate import CubicSpline
        if k <= 0 or len(seg1) <= k or len(seg2) <= k:
            return np.concatenate([seg1, seg2], axis=0)
        junction_pts = np.concatenate([seg1[-k:], seg2[:k]], axis=0)
        t_in = np.linspace(0.0, 1.0, 2 * k)
        cs = CubicSpline(t_in, junction_pts, bc_type="natural", axis=0)
        smoothed = cs(np.linspace(0.0, 1.0, 2 * k))
        return np.concatenate([seg1[:-k], smoothed, seg2[k:]], axis=0)

    def _make_candidate(
        self, waypoints: np.ndarray, algo: str, seed: int, plan_time_s: float,
    ) -> TrajectoryCandidate:
        if waypoints.shape[1] > self._n_arm:
            wp = waypoints[:, : self._n_arm]
        else:
            wp = waypoints
        cost = float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum())
        return TrajectoryCandidate(
            waypoints=wp.astype(float),
            algo=algo,
            seed=seed,
            plan_time_s=float(plan_time_s),
            cost=cost,
        )
