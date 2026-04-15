"""Radian (5D arm) → EE (x, y, z) via KinematicsEngine (Pinocchio FK)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

# lerobot/scripts/traj_analysis/ → 프로젝트 루트는 parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from lerobot_cap.kinematics import KinematicsEngine  # noqa: E402


_ENGINE_CACHE = {}


def get_kinematics(urdf_path: str, ee_frame: str = "gripper_frame_link") -> KinematicsEngine:
    key = (str(Path(urdf_path).resolve()), ee_frame)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = KinematicsEngine(
            key[0],
            end_effector_frame=ee_frame,
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        )
    return _ENGINE_CACHE[key]


def trajs_to_ee_xyz(
    trajs_rad: List[np.ndarray],
    urdf_path: str,
    ee_frame: str = "gripper_frame_link",
) -> List[np.ndarray]:
    """각 (T_i, 6) radian → (T_i, 3) EE xyz.

    첫 5 dim (arm) 만 FK 에 사용, gripper (dim 5) 는 무시.
    """
    kin = get_kinematics(urdf_path, ee_frame)
    out = []
    for traj in trajs_rad:
        T = len(traj)
        xyz = np.empty((T, 3), dtype=np.float32)
        for i in range(T):
            pos, _R = kin.forward_kinematics(traj[i, :5].astype(np.float64))
            xyz[i] = pos
        out.append(xyz)
    return out
