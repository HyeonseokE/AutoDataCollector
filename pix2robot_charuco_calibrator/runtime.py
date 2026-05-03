"""
Runtime API — Charuco 캘리브 결과를 사용해서 픽셀 ↔ 로봇 좌표 변환.

기존 Pix2RobotCalibrator(.pixel_to_robot, .robot_to_pixel) 와 호환되는 인터페이스를 제공해
skills/code_gen 에서 `from ... import Pix2RobotCharuco` 로 그대로 갈아끼울 수 있도록 함.

원리:
    픽셀 (u,v) + depth(u,v)
       │  ① cv2.undistortPoints(K, dist) — 렌즈 왜곡 제거
       ▼
    정규화 좌표 (x_n, y_n, 1)        ← 광선 방향벡터
       │  ② × depth_m
       ▼
    매칭점_camera = (X_c, Y_c, Z_c)  ← 카메라 좌표계 3D 점 (시차 자동 보정)
       │  ③ × cam_to_base (Affine 12 DoF, FK 비등방 오차 흡수)
       ▼
    매칭점_robot = (X_r, Y_r, Z_r)
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class Pix2RobotCharuco:
    """
    Charuco 캘리브 결과로 픽셀 ↔ 로봇 좌표 변환.

    기존 Pix2RobotCalibrator 와 호환되는 인터페이스:
        - pixel_to_robot(u, v, depth_m=None) -> [x, y, z]
        - robot_to_pixel(x, y, z=None) -> (u, v)

    Args:
        robot_id: 로봇 번호 (예: 0). cam_to_base 파일을 robot{N}_cam2robot.npz 로 로드.
        calibration_dir: 캘리브 폴더 (기본: robot_configs/charuco_calibration)
    """

    DEFAULT_CALIBRATION_DIR = (
        Path(__file__).parent.parent / "robot_configs" / "charuco_calibration"
    )

    def __init__(
        self,
        robot_id: int,
        calibration_dir: Optional[Path] = None,
        use_depth_for_z: bool = True,
    ):
        """
        Args:
            robot_id: 로봇 번호.
            calibration_dir: 캘리브 폴더.
            use_depth_for_z: True면 z를 depth로 직접 계산 (z = table_z + (table_depth - depth)).
                             단일 평면 캘리브에서 affine M[2,:]가 부정확한 문제 우회.
                             (x, y)는 affine으로 계산 → 시차 보정 + FK 흡수 둘 다 유지.
                             False면 기존 방식 (cam_to_base 4x4를 z에도 적용).
        """
        self.robot_id = int(robot_id)
        self.use_depth_for_z = bool(use_depth_for_z)
        cal_dir = Path(calibration_dir) if calibration_dir else self.DEFAULT_CALIBRATION_DIR
        self._calibration_dir = cal_dir

        intrinsics_path = cal_dir / "camera_intrinsics.npz"
        extrinsics_path = cal_dir / f"robot{self.robot_id}_cam2robot.npz"

        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"Camera intrinsics 파일 없음: {intrinsics_path}\n"
                f"먼저 ./pix2robot_charuco_calibrator/run_calibration.sh "
                f"--robot {self.robot_id} 로 캘리브 실행."
            )
        if not extrinsics_path.exists():
            raise FileNotFoundError(
                f"Extrinsics 파일 없음: {extrinsics_path}\n"
                f"먼저 ./pix2robot_charuco_calibrator/run_calibration.sh "
                f"--robot {self.robot_id} 로 캘리브 실행."
            )

        intr = np.load(intrinsics_path)
        ext = np.load(extrinsics_path)
        self.K = np.asarray(intr["K"], dtype=np.float64)
        self.dist = np.asarray(intr["dist"], dtype=np.float64).reshape(-1)
        self.cam_to_base = np.asarray(ext["cam_to_base"], dtype=np.float64)
        self._inv_cam_to_base: Optional[np.ndarray] = None

        # 기본 depth (테이블 평면) — 캘리브 매칭점들의 z 평균
        # depth가 명시 안 되면 이 값으로 fallback (테이블 위 객체 가정)
        self.default_depth_m: Optional[float] = None
        # 테이블 평면 reference — depth 기반 z 계산용
        # table_z_robot: 캘리브 plane이 robot 좌표계에서 어디 (z 값)
        # table_depth_cam: 그 plane을 카메라가 본 depth 값
        self.table_z_robot: Optional[float] = None
        self.table_depth_cam: Optional[float] = None

        if "camera_points" in ext.files:
            cam_pts = np.asarray(ext["camera_points"], dtype=np.float64)
            if cam_pts.size > 0:
                self.default_depth_m = float(np.median(cam_pts[:, 2]))
                self.table_depth_cam = float(np.median(cam_pts[:, 2]))
        if "robot_points" in ext.files:
            rob_pts = np.asarray(ext["robot_points"], dtype=np.float64)
            if rob_pts.size > 0:
                self.table_z_robot = float(np.median(rob_pts[:, 2]))

        # 메타정보 (디버그용)
        self.image_size: Optional[Tuple[int, int]] = None
        if "image_width" in intr.files and "image_height" in intr.files:
            self.image_size = (int(intr["image_width"][0]), int(intr["image_height"][0]))
        self.rmse_mm: Optional[float] = None
        if "rmse_m" in ext.files:
            self.rmse_mm = float(ext["rmse_m"][0]) * 1000

        self._intrinsics_path = intrinsics_path
        self._extrinsics_path = extrinsics_path

    # ── 메인 API ───────────────────────────────────────────────

    def pixel_to_robot(
        self,
        u: float,
        v: float,
        depth_m: Optional[float] = None,
    ) -> List[float]:
        """
        픽셀 (u, v) + depth → 로봇 좌표 [x, y, z] (m, base_link).

        Args:
            u, v: 픽셀 좌표
            depth_m: 해당 픽셀의 depth (m, 카메라 광축 기준 거리).
                     None이면 default_depth_m (테이블 평면 가정) 사용.
                     테이블 위 객체면 None OK, 공중에 뜬 객체는 명시 권장.

        Returns:
            [x, y, z] (m) in robot base frame.

        Raises:
            ValueError: depth_m ≤ 0이거나 default_depth_m도 없을 때.
        """
        if depth_m is None or float(depth_m) <= 0:
            if self.default_depth_m is None:
                raise ValueError(
                    "Pix2RobotCharuco.pixel_to_robot: depth_m 미지정 + "
                    "default_depth_m도 없음 (캘리브 데이터 부재)."
                )
            depth_f = self.default_depth_m
        else:
            depth_f = float(depth_m)

        # ① 픽셀 → 정규화 좌표 (광선 방향벡터)
        pixel = np.array([[[float(u), float(v)]]], dtype=np.float32)
        normalized = cv2.undistortPoints(pixel, self.K, self.dist)[0, 0]
        x_n, y_n = float(normalized[0]), float(normalized[1])

        # ② × depth → 카메라 좌표계 3D 점
        P_cam = np.array([x_n * depth_f, y_n * depth_f, depth_f, 1.0], dtype=np.float64)

        # ③ cam_to_base (Affine 12 DoF) 적용 → 로봇 좌표계
        P_base = self.cam_to_base @ P_cam
        x_out, y_out, z_affine = float(P_base[0]), float(P_base[1]), float(P_base[2])

        # ④ z를 depth로 직접 계산 (use_depth_for_z=True 인 경우)
        # 단일 평면 캘리브에서 affine M[2,:]가 약하게 학습된 문제 우회
        if (
            self.use_depth_for_z
            and self.table_z_robot is not None
            and self.table_depth_cam is not None
        ):
            z_out = self.table_z_robot + (self.table_depth_cam - depth_f)
        else:
            z_out = z_affine

        return [x_out, y_out, z_out]

    def robot_to_pixel(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
    ) -> Tuple[int, int]:
        """
        로봇 좌표 (x, y, z) → 픽셀 좌표 (u, v).

        Args:
            x, y, z: 로봇 base_link 좌표. z 미지정 시 0 (테이블 평면).

        Returns:
            (u, v) 픽셀 좌표 (정수 반올림).
        """
        if self._inv_cam_to_base is None:
            self._inv_cam_to_base = np.linalg.inv(self.cam_to_base)

        z_val = 0.0 if z is None else float(z)
        P_base = np.array([float(x), float(y), z_val, 1.0], dtype=np.float64)
        P_cam = self._inv_cam_to_base @ P_base
        Xc, Yc, Zc = P_cam[0], P_cam[1], P_cam[2]

        if Zc <= 1e-6:
            raise ValueError(
                f"로봇 좌표 ({x},{y},{z_val}) 가 카메라 뒤에 있음 (Zc={Zc})."
            )

        # 정규화 좌표 → 왜곡 적용 → 픽셀
        # cv2.projectPoints 는 rvec=0, tvec=0 으로 호출하면 K, dist 만으로 투영
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        obj_pts = np.array([[Xc, Yc, Zc]], dtype=np.float64).reshape(-1, 1, 3)
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.dist)
        u, v = img_pts[0, 0]
        return int(round(float(u))), int(round(float(v)))

    # ── 호환성 헬퍼 (기존 코드가 .load() 등을 호출할 경우 대응) ─

    def load(self, *args, **kwargs):
        """기존 Pix2RobotCalibrator.load 와 호환 — 이미 __init__ 에서 로드 완료."""
        return True

    @property
    def is_loaded(self) -> bool:
        return self.cam_to_base is not None and self.K is not None

    # ── 디버그 ───────────────────────────────────────────────

    def __repr__(self) -> str:
        rmse = f", rmse={self.rmse_mm:.2f}mm" if self.rmse_mm else ""
        size = f", image={self.image_size}" if self.image_size else ""
        depth = (
            f", default_depth={self.default_depth_m:.3f}m"
            if self.default_depth_m else ""
        )
        return (
            f"Pix2RobotCharuco(robot{self.robot_id}{size}{rmse}{depth}, "
            f"calib={self._calibration_dir.name})"
        )
