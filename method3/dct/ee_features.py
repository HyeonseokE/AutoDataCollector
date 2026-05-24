"""EE-space DCT utilities — translation-invariant motion shape descriptor.

기존 joint DCT (kinematic redundancy 포함) → EE delta DCT (motion shape 직접).
SCIZOR(Open-X 7-DoF EE delta) 분야 관례 + DemInf 의 latent embedding 권장과
정렬한다. joint 공간에서 같은 EE motion 이 다른 joint 궤적(같은 EE 직선의
시작 자세별 변동)을 만드는 경우, joint DCT 는 *오히려 novel 처럼* 보이지만
EE delta DCT 는 같은 cluster 로 모인다 (translation invariance).

shape 규약:
  ee_poses        (T, 6)   = [x, y, z, roll, pitch, yaw]  per frame
  ee_deltas       (T, 6)   = np.diff(ee_poses, axis=0) + tail pad
  ee_delta_dct    (L0, 6)  = traj_to_dct(ee_deltas, L0)

so101 5-DoF arm 의 wrist_roll 이 고정되면 roll 축이 거의 상수 → DCT[0] 만
지배(나머지 AC≈0). 그래도 6축 유지하면 U_VLA pad 코드 (5→6 zero-pad) 와
dim 호환되고, candidate vs DB 양쪽 모두 (50, 6) 으로 일관된다.
"""
from __future__ import annotations

import numpy as np

from method3.dct.transform import traj_to_dct


def _rotation_matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → (roll, pitch, yaw) — XYZ Euler convention.

    scipy 의존 회피용 직접 구현. URDF Joint convention 과 일치.
    """
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def ee_delta_dct_from_poses(
    ee_poses: np.ndarray,
    L0: int = 50,
) -> np.ndarray:
    """EE pose 시계열에서 delta 잡고 DCT — pre-recorded EE column 활용 (FK 불필요).

    Args:
        ee_poses: (T, 6) [xyz + rpy].  데이터셋 ``observation.ee_pos.robot_xyzrpy``
            column 을 그대로 받을 수 있다.
        L0: DCT 길이 (skill chunk_size 와 일치 권장, default 50).

    Returns:
        (L0, 6) EE delta DCT.

    Notes:
        - ``np.diff`` 로 T → T-1 짧아진 것을 끝점 hold 로 1 row pad 해서 T 유지.
          기존 ``traj_to_dct`` 가 가변 길이 가능하지만 candidate/DB shape 일관성
          위해 pad. 1-frame degenerate 입력은 zero delta 로 안전 처리.
    """
    poses = np.asarray(ee_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(
            f"ee_poses must be (T, 6) [xyz + rpy], got {poses.shape}"
        )
    T = poses.shape[0]
    if T < 2:
        # 1-frame degenerate — zero-delta fallback (constant pose).
        return traj_to_dct(np.zeros((max(1, T), 6), dtype=np.float64), L0=L0)
    deltas = np.diff(poses, axis=0)                          # (T-1, 6)
    deltas = np.vstack([deltas, deltas[-1:]])                # (T, 6) — tail pad
    return traj_to_dct(deltas, L0=L0)                        # (L0, 6)


def ee_delta_dct_from_joints(
    joint_traj: np.ndarray,
    urdf_path: str,
    L0: int = 50,
    ee_frame: str = "gripper_frame_link",
    arm_dof: int = 5,
) -> np.ndarray:
    """joint 시계열 → FK → EE pose → delta → DCT.

    curobo candidate 는 ``waypoints`` 가 joint 5-axis 라 FK 필요. 데이터셋
    측 entry 는 ``observation.ee_pos.robot_xyzrpy`` 가 이미 있어 FK 불필요.
    두 경로 모두 결과 dim (L0, 6) 으로 통일.

    Args:
        joint_traj: (T, dof) joint trajectory (arm joints; gripper 무시).
            앞 ``arm_dof`` 열만 사용.
        urdf_path: URDF 파일 절대/repo-relative 경로.
        L0: DCT 길이.
        ee_frame: FK 대상 link name. so101 = "gripper_frame_link".
        arm_dof: FK 입력 dim (so101 = 5).

    Returns:
        (L0, 6) EE delta DCT.
    """
    import sys
    from pathlib import Path

    # FK 라이브러리 — lerobot/scripts/traj_analysis/fk_ee 재사용 (이미 viz tool 사용).
    _scripts = Path(__file__).resolve().parent.parent.parent / "lerobot" / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from traj_analysis.fk_ee import get_kinematics

    traj = np.asarray(joint_traj, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"joint_traj must be (T, dof), got {traj.shape}")
    if traj.shape[1] < arm_dof:
        raise ValueError(
            f"joint_traj.shape[1]={traj.shape[1]} < arm_dof={arm_dof}"
        )

    kin = get_kinematics(urdf_path, ee_frame)
    T = traj.shape[0]
    poses = np.empty((T, 6), dtype=np.float64)
    for i in range(T):
        pos, R = kin.forward_kinematics(traj[i, :arm_dof].astype(np.float64))
        poses[i, :3] = pos
        poses[i, 3:] = _rotation_matrix_to_rpy(R)
    return ee_delta_dct_from_poses(poses, L0=L0)
