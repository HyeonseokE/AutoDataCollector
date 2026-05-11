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


# Each preset is (line_dir_scale, perp_lateral_scale, perp_vertical_scale).
# Magnitudes are multiplied by ``CuroboBackendConfig.via_offset_mag`` (m).
# Index 0 = "near-direct" (small midline perturbation) so the direct path
# variant also goes through the via pipeline → uniform 2-segment shape.
VIA_OFFSETS = (
    (0.0,  0.00, +0.02),   # near-direct (tiny up bias)
    (0.0, -1.00, +0.30),   # left + up
    (0.0, +1.00, +0.30),   # right + up
    (0.0,  0.00, +1.20),   # high arc
)


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

        if not Path(config.robot_cfg_path).exists():
            raise FileNotFoundError(
                f"curobo robot config missing: {config.robot_cfg_path}"
            )

        # max_batch_size controls the shape of CUDA graphs; choose the largest
        # batch we'll ever submit so subsequent plan_batch calls fit.
        self._batch_size = int(config.max_batch_size)
        mp_cfg = MotionPlannerCfg.create(
            robot=str(config.robot_cfg_path),
            num_trajopt_seeds=config.num_trajopt_seeds,
            num_ik_seeds=config.num_ik_seeds,
            random_seed=123,
            use_cuda_graph=config.use_cuda_graph,
            max_batch_size=self._batch_size,
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

        # Pre-allocate reusable batch tensors on GPU. Shapes are fixed so CUDA
        # graphs capture against these and never re-compile.
        device = "cuda"
        dtype = torch.float32
        self._dev = device
        self._dtype = dtype
        self._buf_start = torch.zeros((self._batch_size, self._n_dof), device=device, dtype=dtype)
        self._buf_goal = torch.zeros((self._batch_size, self._n_dof), device=device, dtype=dtype)
        # Pose buffers for IK (xyz + quat per slot)
        # GoalToolPose tensor shape: (batch, n_tool, n_pose_per_goal, 3 or 4)
        self._buf_pose_xyz = torch.zeros(
            (self._batch_size, 1, 1, 3), device=device, dtype=dtype,
        )
        self._buf_pose_quat = torch.zeros(
            (self._batch_size, 1, 1, 4), device=device, dtype=dtype,
        )
        # Identity-ish quaternion as default (will be overwritten per call)
        self._buf_pose_quat[..., 0] = 1.0

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

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def dof(self) -> int:
        return self._n_arm

    def plan_batch(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> list[TrajectoryCandidate]:
        if not self.cfg.enabled or n <= 0:
            return []
        n = min(int(n), self._batch_size)
        rng = rng if rng is not None else np.random.default_rng()

        start_full = self._to_full_qpos(start_qpos)
        goal_full = self._to_full_qpos(goal_qpos)
        # Enforce fixed-joint values on the goal (so cspace plan has start ==
        # goal on those dims and the trajectory stays put on them).
        for idx in self._fixed_idx:
            if idx < self._n_dof:
                goal_full[idx] = start_full[idx]

        # 1) Compute via_xyz batch and IK-resolve to via_qpos (skip i=0 which
        #    uses goal_qpos directly to play the role of "direct" candidate).
        start_ee, start_quat = self._compute_ee_xyz_quat(start_full)
        goal_ee, _ = self._compute_ee_xyz_quat(goal_full)

        via_xyz_list: list[np.ndarray] = [None]  # placeholder for slot 0 (direct)
        for k in range(1, n):
            mode_idx = (k - 1) % len(VIA_OFFSETS)
            via_xyz_list.append(self._sample_via_xyz(start_ee, goal_ee, mode_idx, rng))

        # IK batch for slots 1..n-1 (slot 0 uses goal_qpos itself)
        via_qpos_list: list[Optional[np.ndarray]] = [None] * n
        via_qpos_list[0] = goal_full.copy()
        ik_success_mask = [True] * n  # slot 0 trivially succeeds (it's goal_qpos)
        for k in range(1, n):
            q_via = self._ik_solve(via_xyz_list[k], start_quat, seed_q=start_full)
            if q_via is None:
                ik_success_mask[k] = False
                # Fallback: pick the linear midpoint qpos so this candidate at
                # least produces SOMETHING (just a slightly less curved path).
                via_qpos_list[k] = (start_full + goal_full) / 2.0
            else:
                # Lock fixed joints on the via_qpos too.
                for idx in self._fixed_idx:
                    if idx < self._n_dof:
                        q_via[idx] = start_full[idx]
                via_qpos_list[k] = q_via

        # 2) Seg1 batch: start (replicated) → [goal, via1, via2, ...]
        t0 = time.time()
        seg1_batch = self._plan_cspace_batch(
            start_batch=[start_full] * n,
            goal_batch=via_qpos_list,
        )
        seg1_time = time.time() - t0

        # 3) Seg2 batch: via_qpos → goal (skip slot 0; it's a no-op)
        t0 = time.time()
        seg2_batch = self._plan_cspace_batch(
            start_batch=via_qpos_list,
            goal_batch=[goal_full] * n,
        )
        seg2_time = time.time() - t0

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
                algo_name = f"curobo:via{(i-1) % len(VIA_OFFSETS)}"

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

    def _compute_ee_xyz_quat(self, qpos_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        st = self._JointState.from_position(
            torch.tensor(qpos_full, device=self._dev, dtype=self._dtype).unsqueeze(0),
            joint_names=self.joint_names,
        )
        kin = self._planner.compute_kinematics(st)
        tp = kin.tool_poses if hasattr(kin, "tool_poses") else None
        if tp is None:
            raise AttributeError("kinematics result missing tool_poses")
        if isinstance(tp, (list, tuple)):
            xyz = tp[0].position.squeeze().detach().cpu().numpy()
            quat = tp[0].quaternion.squeeze().detach().cpu().numpy()
        else:
            xyz = tp.position.detach().cpu().numpy().reshape(-1, 3)[0]
            quat = tp.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        return np.asarray(xyz).reshape(3), np.asarray(quat).reshape(4)

    def _sample_via_xyz(
        self,
        start_ee: np.ndarray,
        goal_ee: np.ndarray,
        mode_idx: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        midpoint = (start_ee + goal_ee) / 2.0
        line = goal_ee - start_ee
        line_len = float(np.linalg.norm(line))
        if line_len < 1e-6:
            return midpoint + np.array([0.0, self.cfg.via_offset_mag, 0.0])
        line_dir = line / line_len
        # Build perpendicular basis (project world-y and world-z onto plane ⊥ line)
        perp_lat = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], line_dir) * line_dir
        if np.linalg.norm(perp_lat) < 1e-6:
            perp_lat = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], line_dir) * line_dir
        perp_lat /= max(np.linalg.norm(perp_lat), 1e-9)
        perp_vert = np.array([0.0, 0.0, 1.0]) - np.dot([0.0, 0.0, 1.0], line_dir) * line_dir
        perp_vert /= max(np.linalg.norm(perp_vert), 1e-9)
        d_line, d_lat, d_vert = VIA_OFFSETS[mode_idx]
        mag = self.cfg.via_offset_mag
        offset = (d_line * mag) * line_dir + (d_lat * mag) * perp_lat + (d_vert * mag) * perp_vert
        offset += rng.normal(0.0, 0.003, size=3)
        return midpoint + offset

    def _ik_solve(
        self, target_xyz: np.ndarray, target_quat: np.ndarray, seed_q: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Single-pose IK (batch-1) — fast (~ms) and decouples via_qpos
        computation from the heavier plan_pose call.

        GoalToolPose tensor format: 5D ``[batch, horizon, link, goalset, 3 or 4]``.
        We use (1, 1, 1, 1, ·) for a single pose query.
        """
        torch = self._torch
        pos_t = torch.tensor(target_xyz, device=self._dev, dtype=self._dtype).reshape(1, 1, 1, 1, 3)
        quat_t = torch.tensor(target_quat, device=self._dev, dtype=self._dtype).reshape(1, 1, 1, 1, 4)
        goal = self._GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=pos_t,
            quaternion=quat_t,
        )
        seed_state = self._JointState.from_position(
            torch.tensor(seed_q, device=self._dev, dtype=self._dtype).unsqueeze(0),
            joint_names=self.joint_names,
        )
        try:
            result = self._planner.ik_solver.solve_pose(goal, current_state=seed_state)
        except Exception as _e:
            return None
        if result is None or not bool(result.success.any()):
            return None
        sol = result.solution.detach().cpu().numpy()
        while sol.ndim > 1:
            sol = sol[0]
        return np.asarray(sol).astype(float)

    def _plan_cspace_batch(
        self,
        start_batch: list[np.ndarray],
        goal_batch: list[np.ndarray],
    ) -> Optional[list[Optional[np.ndarray]]]:
        """Batched joint-space plan. Returns list of (n_wp, dof) arrays, one
        per slot, or None for slots that failed."""
        assert len(start_batch) == len(goal_batch) == self._batch_size, (
            f"batch shape mismatch: expected {self._batch_size}, "
            f"got start={len(start_batch)} goal={len(goal_batch)}"
        )
        torch = self._torch
        # Convert numpy arrays → torch via stacking float lists. Direct test
        # confirms this path works; the earlier failure mode was due to a
        # pre-allocated buffer pattern that interfered with curobo's internal
        # goal_buffer manager.
        start_t = torch.tensor(
            [list(map(float, x)) for x in start_batch],
            device=self._dev, dtype=self._dtype,
        )
        goal_t = torch.tensor(
            [list(map(float, x)) for x in goal_batch],
            device=self._dev, dtype=self._dtype,
        )
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
        for i in range(self._batch_size):
            if i >= len(success) or not bool(success[i]):
                out.append(None)
                continue
            wp_i = interp_full[i]
            if last_t is not None and i < len(last_t):
                wp_i = wp_i[: int(last_t[i]) + 1]
            out.append(wp_i.copy())
        return out

    def _capture_batched_graph(self) -> None:
        """Issue one batched plan_cspace at the full batch_size so CUDA graphs
        are captured for our actual shape. Skipped silently on failure."""
        n = self._batch_size
        start = [self._default_full] * n
        goal = [self._default_full + 0.01] * n   # slightly different to not be trivial
        _ = self._plan_cspace_batch(start, goal)

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
