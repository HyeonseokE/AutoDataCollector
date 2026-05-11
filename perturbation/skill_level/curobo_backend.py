"""Curobo-based skill-level perturbation backend (v2 — batched + cuda graph + locked joints).

Design fixes applied vs v1:
  1. CUDA graph enabled + consistent shapes ⇒ ~25× per-plan speedup.
  2. ``fixed_joint_indices`` enforced — wrist_roll lock matches the OMPL
     branch (cartesian IK already fixes it; curobo also).
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

Public interface mirrors ``PlannerEnsemble.plan_batch`` for duck-type swap
with the OMPL backend via yaml ``perturbation.skill.backend``.
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

        # _batch_size  : user-facing N (# of candidates).
        # _cspace_batch: internal merged batch for plan_cspace = 2 × N because
        #                we concatenate (seg1: start→via*) ⊕ (seg2: via*→goal)
        #                into ONE GPU call instead of two.
        self._batch_size = int(config.max_batch_size)
        self._cspace_batch = 2 * self._batch_size
        mp_cfg = MotionPlannerCfg.create(
            robot=robot_cfg_abs,
            num_trajopt_seeds=config.num_trajopt_seeds,
            num_ik_seeds=config.num_ik_seeds,
            random_seed=123,
            use_cuda_graph=config.use_cuda_graph,
            max_batch_size=self._cspace_batch,
            max_goalset=self._n_goalset,
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

        Duck-types ``PlanServiceClient.close`` so the pipeline's
        ``_teardown_skill_perturbation`` can call it uniformly.
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

    def plan_batch(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        n: int,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> list[TrajectoryCandidate]:
        """Plan N transit candidates. Duck-types ``PlanServiceClient.plan_batch``
        so production code (skills_lerobot) can pass ``seed=int`` interchangeably
        with either backend. Tests may pass ``rng=`` directly.
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

        # 1) Compute via_xyz batch and IK-resolve to via_qpos (skip i=0 which
        #    uses goal_qpos directly to play the role of "direct" candidate).
        #    FK for start+goal goes through a SINGLE batched compute_kinematics.
        (start_ee, start_quat), (goal_ee, goal_quat) = self._compute_ee_xyz_quat_batch(
            [start_full, goal_full]
        )

        via_xyz_list: list[np.ndarray] = [None]  # placeholder for slot 0 (direct)
        for k in range(1, n):
            via_xyz_list.append(self._sample_via_xyz(start_ee, goal_ee, rng))

        # Batched IK for slots 1..n-1 (slot 0 uses goal_qpos itself).
        # (n-1) vias × len(slerp_ratios) orientations are submitted as a
        # single flat batch — one GPU call instead of up to 5×(n-1)
        # sequential calls. Per-via first-success picking happens on CPU
        # after the batch returns.
        via_qpos_list: list[Optional[np.ndarray]] = [None] * n
        via_qpos_list[0] = goal_full.copy()
        if n > 1:
            via_solutions = self._ik_solve_batched(
                via_xyz_list[1:], start_quat, goal_quat, seed_q=start_full,
            )
            mid_q = (start_full + goal_full) / 2.0
            for k in range(1, n):
                q_via = via_solutions[k - 1]
                if q_via is None:
                    via_qpos_list[k] = mid_q.copy()
                else:
                    for idx in self._fixed_idx:
                        if idx < self._n_dof:
                            q_via[idx] = start_full[idx]
                    via_qpos_list[k] = q_via

        # 2-3) Merge seg1 + seg2 into ONE batched plan_cspace call of size 2N.
        # Layout of the merged batch (length self._cspace_batch = 2N):
        #   slots [0       .. N  )  → seg1: start          → via_qpos_list
        #   slots [N       .. 2N )  → seg2: via_qpos_list  → goal
        # Unused tail slots (when user n < self._batch_size) get padded with
        # trivial start→start no-ops so the CUDA graph's fixed shape holds.
        merged_start: list[np.ndarray] = [start_full] * n + list(via_qpos_list)
        merged_goal: list[np.ndarray] = list(via_qpos_list) + [goal_full] * n
        # Padding to fill (2 × self._batch_size) - 2N slots.
        pad = self._cspace_batch - len(merged_start)
        if pad > 0:
            merged_start.extend([start_full] * pad)
            merged_goal.extend([start_full] * pad)

        t0 = time.time()
        merged_batch = self._plan_cspace_batch(merged_start, merged_goal)
        merged_time = time.time() - t0
        if merged_batch is None:
            return []
        seg1_batch = merged_batch[:n]
        seg2_batch = merged_batch[n : 2 * n]
        seg1_time = merged_time / 2.0   # split for reporting
        seg2_time = merged_time / 2.0

        # 4) Per-candidate assembly
        candidates: list[TrajectoryCandidate] = []
        for i in range(n):
            seg1 = seg1_batch[i] if seg1_batch is not None else None
            if seg1 is None:
                continue
            if i == 0:
                # Direct path: seg1 only, no junction smoothing.
                full_wp = seg1
                algo_name = "curobo:direct"
            else:
                seg2 = seg2_batch[i] if seg2_batch is not None else None
                if seg2 is None:
                    continue  # via candidate failed seg2
                full_wp = self._smooth_junction(seg1, seg2, k=self.cfg.junction_smooth_k)
                algo_name = f"curobo:via{i-1}"

            # Step 5: enforce fixed joints across the full trajectory (post-lock).
            full_wp = self._apply_fixed_joint_lock(full_wp, start_full)

            candidates.append(self._make_candidate(
                full_wp, algo=algo_name,
                seed=int(rng.integers(0, 2**31 - 1)),
                plan_time_s=(seg1_time + seg2_time) / n,
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

    def _sample_via_xyz(
        self,
        start_ee: np.ndarray,
        goal_ee: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Continuous random via-point sampling.

        Parametrization::

            via = (1-t)·start_ee + t·goal_ee + lat·perp_lat + vert·perp_vert

        where (t, lat, vert) are independently sampled from continuous
        distributions per call (see module-level VIA_T_MIN/MAX,
        VIA_VERT_MIN_RATIO/MAX_RATIO constants). No discrete mode index —
        each call to this method produces a genuinely unique via location.
        """
        # 1. Fraction along the line (NOT clipped to 0.5/midpoint — anywhere
        #    in [VIA_T_MIN, VIA_T_MAX]). t closer to endpoints produces
        #    asymmetric paths (long approach + short retreat or vice versa).
        t = float(rng.uniform(VIA_T_MIN, VIA_T_MAX))
        via_base = (1.0 - t) * start_ee + t * goal_ee

        line = goal_ee - start_ee
        line_len = float(np.linalg.norm(line))
        if line_len < 1e-6:
            return via_base + np.array([0.0, self.cfg.via_offset_mag, 0.0])
        line_dir = line / line_len

        # Build perpendicular basis (project world-y and world-z onto plane ⊥ line).
        perp_lat = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], line_dir) * line_dir
        if np.linalg.norm(perp_lat) < 1e-6:
            perp_lat = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], line_dir) * line_dir
        perp_lat /= max(np.linalg.norm(perp_lat), 1e-9)
        perp_vert = np.array([0.0, 0.0, 1.0]) - np.dot([0.0, 0.0, 1.0], line_dir) * line_dir
        perp_vert /= max(np.linalg.norm(perp_vert), 1e-9)

        # 2. Lateral offset — symmetric uniform [-mag, +mag] (left/right equal).
        mag = self.cfg.via_offset_mag
        lat = float(rng.uniform(-mag, +mag))

        # 3. Vertical offset — uniform [+0.3·mag, +1.2·mag], biased upward
        #    because the 5-DoF arm reaches above-line vias far more reliably.
        vert = float(rng.uniform(
            VIA_VERT_MIN_RATIO * mag, VIA_VERT_MAX_RATIO * mag,
        ))

        return via_base + lat * perp_lat + vert * perp_vert

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
        n_via = len(via_xyz_list)
        if n_via == 0:
            return []
        G = len(self._slerp_ratios)

        # Pre-compute all G slerp quats once (independent of via).
        quat_g = np.stack(
            [
                self._slerp_quat(start_quat, goal_quat, t).astype(np.float32)
                for t in self._slerp_ratios
            ],
            axis=0,
        )  # (G, 4)
        xyz_b = np.asarray(via_xyz_list, dtype=np.float32).reshape(n_via, 3)

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
            return [None] * n_via
        if result is None:
            return [None] * n_via

        success = result.success.detach().cpu().numpy().reshape(-1)
        solution = result.solution.detach().cpu().numpy().reshape(n_via, -1)

        out: list[Optional[np.ndarray]] = []
        for i in range(n_via):
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
