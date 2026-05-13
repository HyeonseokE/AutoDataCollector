"""TrajectoryCandidate → preselective_filter.Candidate.action_chunk.

Decision #5(b) — execution-aligned resampling at recording_fps with
goal-pad and gripper-hold, then zero-pad to SmolVLA's max_action_dim.
"""
from __future__ import annotations

import numpy as np


def trajectory_to_action_chunk(
    waypoints: np.ndarray,           # (N_wp, arm_dof) joint positions
    times: np.ndarray,               # (N_wp,) seconds, monotonically increasing
    chunk_size: int,                 # H, e.g., 50
    action_dim: int,                 # max_action_dim, e.g., 32
    fps: int,                        # recording_fps, e.g., 10
    current_gripper: float,          # held constant during transit
    arm_dof: int,                    # SO-101 = 6 — gripper appended → real dim arm_dof+1
) -> np.ndarray:
    """Convert a planned joint trajectory to a SmolVLA-shaped action_chunk.

    Returns shape (chunk_size, action_dim), zero-padded after the real
    (arm_dof+1) dims. Goal-padded when the trajectory is shorter than
    chunk_size/fps seconds.

    Steps:
        1. target_times = [0, 1/fps, ..., (chunk_size-1)/fps]
        2. per i: interpolate joints if target_times[i] ≤ times[-1] else hold goal
        3. append current_gripper as the last real dim
        4. zero-pad to action_dim
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    if waypoints.ndim != 2:
        raise ValueError(f"waypoints must be (N_wp, arm_dof), got {waypoints.shape}")
    if waypoints.shape[0] != times.shape[0]:
        raise ValueError(
            f"waypoints and times length mismatch: {waypoints.shape[0]} vs {times.shape[0]}"
        )
    if waypoints.shape[1] != arm_dof:
        raise ValueError(
            f"waypoints dim {waypoints.shape[1]} != arm_dof={arm_dof}"
        )
    if action_dim < arm_dof + 1:
        raise ValueError(
            f"action_dim={action_dim} too small for arm_dof+1={arm_dof + 1}"
        )

    # Step 1: target time grid
    target_times = np.arange(chunk_size, dtype=np.float64) / float(fps)

    # Step 2: per-joint interpolation with goal-hold
    t_end = float(times[-1])
    joint_chunk = np.empty((chunk_size, arm_dof), dtype=np.float64)
    goal = waypoints[-1]

    for j in range(arm_dof):
        # numpy.interp clips out-of-range to the boundary, which is exactly
        # the "goal hold" behavior we want for target_times beyond t_end.
        joint_chunk[:, j] = np.interp(target_times, times, waypoints[:, j])

    # Defensive: explicitly enforce goal-hold past t_end (np.interp already
    # does this since left=waypoints[0,j], right=waypoints[-1,j], but make
    # the intent clear)
    past_end = target_times > t_end
    if np.any(past_end):
        joint_chunk[past_end] = goal

    # Step 3: gripper held constant
    gripper_chunk = np.full((chunk_size, 1), current_gripper, dtype=np.float64)
    real = np.concatenate([joint_chunk, gripper_chunk], axis=1)  # (chunk_size, arm_dof+1)

    # Step 4: zero-pad
    action_chunk = np.zeros((chunk_size, action_dim), dtype=np.float64)
    action_chunk[:, : arm_dof + 1] = real
    return action_chunk
