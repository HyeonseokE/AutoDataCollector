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
    # rpy delta unwrap — atan2 가 [-π, π] 로 자른 결과 ±π 경계 넘으면
    # spurious 큰 jump (≈ ±2π) 가 생긴다. 실제 회전량은 작아도 DCT 에
    # high-frequency 인공물로 들어가 motion shape 신호 흐려짐. 양쪽
    # producer (DB recorded + candidate FK) 모두 같은 atan2 convention
    # 사용하므로 한 줄 patch 로 대칭 적용.
    deltas[:, 3:] = (deltas[:, 3:] + np.pi) % (2 * np.pi) - np.pi
    deltas = np.vstack([deltas, deltas[-1:]])                # (T, 6) — tail pad
    return traj_to_dct(deltas, L0=L0)                        # (L0, 6)


def _quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """quaternion [w, x, y, z] (scalar first — curobo Pose convention) → 3x3 R."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def ee_delta_dct_from_joints(
    joint_traj: np.ndarray,
    urdf_path: str,
    L0: int = 50,
    ee_frame: str = "gripper_frame_link",
    arm_dof: int = 5,
    kinematics_engine=None,
) -> np.ndarray:
    """joint 시계열 → FK → EE pose → delta → DCT.

    Args:
        joint_traj: (T, dof) joint trajectory.
        urdf_path: URDF path (legacy fallback path 에서만 사용).
        L0: DCT 길이.
        ee_frame: FK 대상 link.
        arm_dof: arm joint 수.
        kinematics_engine: server-side 에서 이미 부팅된 curobo Kinematics
            (= MotionPlanner.kinematics). 주입되면 curobo 의 batched FK 직접
            사용 — ``lerobot_cap`` chain (scservo_sdk 의존) 우회. 권장.
            None 이면 legacy ``traj_analysis.fk_ee`` 경로 (client/local 만).

    Returns:
        (L0, 6) EE delta DCT.
    """
    traj = np.asarray(joint_traj, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"joint_traj must be (T, dof), got {traj.shape}")
    if traj.shape[1] < arm_dof:
        raise ValueError(
            f"joint_traj.shape[1]={traj.shape[1]} < arm_dof={arm_dof}"
        )

    T = traj.shape[0]
    poses = np.empty((T, 6), dtype=np.float64)

    if kinematics_engine is not None:
        # PRIMARY — server-side 의 이미 로드된 curobo kinematics 사용.
        # attachment_manager.py:143-150 의 정확한 사용 패턴 그대로.
        import torch
        from curobo._src.state.state_joint import JointState

        joint_names = list(kinematics_engine.joint_names)
        n_dof = len(joint_names)
        q_full = np.zeros((T, n_dof), dtype=np.float32)
        q_full[:, :arm_dof] = traj[:, :arm_dof].astype(np.float32)
        device = (kinematics_engine.tensor_args.device
                  if hasattr(kinematics_engine, "tensor_args") else "cuda")
        q_t = torch.as_tensor(q_full, dtype=torch.float32, device=device)
        joint_state = JointState.from_position(q_t, joint_names=joint_names)
        fk_result = kinematics_engine.compute_kinematics(joint_state)
        if fk_result.tool_poses is None:
            raise RuntimeError(
                "kinematics_engine.compute_kinematics returned no tool_poses; "
                "check robot config / ee_frame setup"
            )
        ee_pose = fk_result.tool_poses.get_link_pose(ee_frame)
        pos_np = ee_pose.position.detach().cpu().numpy()      # (T, 3)
        quat_np = ee_pose.quaternion.detach().cpu().numpy()   # (T, 4) [w,x,y,z]
        for i in range(T):
            poses[i, :3] = pos_np[i]
            poses[i, 3:] = _rotation_matrix_to_rpy(_quat_wxyz_to_rotmat(quat_np[i]))
    else:
        # LEGACY fallback — traj_analysis.fk_ee. lerobot_cap chain 이라 server
        # 에서는 scservo_sdk 의존으로 fail. local/client 전용.
        import sys
        from pathlib import Path
        _scripts = Path(__file__).resolve().parent.parent.parent / "lerobot" / "scripts"
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        from traj_analysis.fk_ee import get_kinematics
        kin = get_kinematics(urdf_path, ee_frame)
        for i in range(T):
            pos, R = kin.forward_kinematics(traj[i, :arm_dof].astype(np.float64))
            poses[i, :3] = pos
            poses[i, 3:] = _rotation_matrix_to_rpy(R)

    return ee_delta_dct_from_poses(poses, L0=L0)
