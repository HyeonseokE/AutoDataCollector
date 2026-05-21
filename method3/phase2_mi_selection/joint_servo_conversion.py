"""Joint angles (radians) ↔ servo positions (normalized -100~+100) 변환.

A.3 paradigm 작업 — Phase2 candidate trajectory 는 curobo planning 결과로
*joint angles (radians)* 형태로 옴. 반면 학습 데이터셋의 ``observation.state`` /
``action`` 은 *normalized servo positions* (±100, lerobot feetech convention).

DB action_descriptor 는 servo space DCT 이므로 candidate.dct_target 도 같은
공간으로 변환되어야 ΔH_A 계산이 정합적이다.

본 모듈은 lerobot_cap.kinematics.calibration_limits 의 핵심 로직 (load 및
radians_to_normalized) 을 server-friendly 한 형태로 *inline* 미러링 — server
환경에 lerobot_cap module 이 없어도 동작.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


_DEFAULT_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


class JointServoConverter:
    """Per-robot calibration based joint ↔ servo conversion.

    Mirrors ``lerobot_cap.kinematics.calibration_limits`` minimal subset.

    Usage::

        conv = JointServoConverter.from_calibration("robot_configs/.../robot4_calibration.json")
        # arm joint radians (5-dim) → arm servo positions (5-dim, -100~+100)
        servo_arm = conv.radians_to_normalized(joints_rad)
    """

    def __init__(
        self,
        joint_names: List[str],
        half_range_radians: np.ndarray,
        offset_normalized: np.ndarray,
        drive_modes: np.ndarray | None = None,
    ) -> None:
        self.joint_names = list(joint_names)
        self.half_range_radians = np.asarray(half_range_radians, dtype=np.float64)
        self.offset_normalized = np.asarray(offset_normalized, dtype=np.float64)
        if drive_modes is None:
            drive_modes = np.zeros(len(joint_names), dtype=np.int64)
        self.drive_modes = np.asarray(drive_modes, dtype=np.int64)

    @classmethod
    def from_calibration(
        cls,
        calibration_file: str | Path,
        joint_names: Optional[List[str]] = None,
        use_homing_offset: bool = True,
    ) -> "JointServoConverter":
        """Build converter from a feetech calibration JSON (so101 schema)."""
        if joint_names is None:
            joint_names = list(_DEFAULT_ARM_JOINT_NAMES)
        with open(calibration_file, "r") as f:
            calib = json.load(f)

        def _find_motor(jname: str, idx: int) -> dict:
            if jname in calib:
                return calib[jname]
            mkey = f"motor_{idx + 1}"
            if mkey in calib:
                return calib[mkey]
            raise ValueError(f"motor {jname!r} (motor_{idx+1}) not in calibration")

        half_range = []
        offset_norm = []
        drive_modes = []
        HALF_TURN = 2048  # URDF 0° encoder value after homing
        for i, jname in enumerate(joint_names):
            m = _find_motor(jname, i)
            range_min = float(m["range_min"])
            range_max = float(m["range_max"])
            dm = int(m.get("drive_mode", 0))
            encoder_range = range_max - range_min
            if encoder_range <= 0:
                half_range.append(0.0)
                offset_norm.append(0.0)
                drive_modes.append(dm)
                continue
            degrees = encoder_range * 360.0 / 4096.0
            half_r = float(np.radians(degrees) / 2.0)
            half_range.append(half_r)
            if use_homing_offset:
                urdf_zero_n = ((HALF_TURN - range_min) / encoder_range) * 200.0 - 100.0
                if dm == 1:
                    urdf_zero_n = -urdf_zero_n
                offset_norm.append(float(urdf_zero_n))
            else:
                offset_norm.append(0.0)
            drive_modes.append(dm)

        return cls(
            joint_names=joint_names,
            half_range_radians=np.asarray(half_range, dtype=np.float64),
            offset_normalized=np.asarray(offset_norm, dtype=np.float64),
            drive_modes=np.asarray(drive_modes, dtype=np.int64),
        )

    def radians_to_normalized(self, radians: np.ndarray) -> np.ndarray:
        """Convert URDF radians → normalized servo positions (-100~+100).

        Vectorized over leading dimensions: input shape ``(..., n_joints)``
        returns ``(..., n_joints)``. Broadcasts ``half_range`` and ``offset``
        across leading dims.
        """
        rad = np.asarray(radians, dtype=np.float64)
        n_joints = self.half_range_radians.shape[0]
        if rad.shape[-1] != n_joints:
            raise ValueError(
                f"radians last-dim={rad.shape[-1]} ≠ n_joints={n_joints} "
                f"(joints={self.joint_names})"
            )
        # safe div — half_range==0 (uncalibrated joint) → return offset only
        hr = self.half_range_radians.copy()
        hr[hr == 0.0] = 1.0  # avoid /0
        base = (rad / hr) * 100.0
        # mask uncalibrated joints out of base contribution
        zero_mask = (self.half_range_radians == 0.0).astype(np.float64)
        base = base * (1.0 - zero_mask)
        # drive_mode inversion
        sign = np.where(self.drive_modes == 1, -1.0, 1.0)
        return base * sign + self.offset_normalized
