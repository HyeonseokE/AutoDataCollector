"""
Pix2Robot Calibrator
픽셀(u,v) → 로봇(x,y,z) 직접 변환을 위한 호모그래피 캘리브레이션

변환 방식:
- 호모그래피 3x3: 픽셀(u,v) → 로봇(x,y) 평면 매핑
- 테이블 z: 수집한 로봇 z값들의 평균 → 테이블 높이 상수 (base_link 기준)
- depth 센서 불필요 → 노이즈 제거, 정밀도 향상

매칭점 수집:
- Charuco 보드를 워크스페이스에 평평하게 놓고 코너를 자동 검출
- 사용자가 화면의 검출된 코너(시안색 십자)를 클릭하면 가장 가까운
  코너로 스냅 → 그 코너의 sub-pixel 위치를 매칭점 픽셀로 사용
- 임의 픽셀 클릭은 받지 않음 (코너 근처 30px 이내만 유효)
"""

import cv2
import json
import select
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot_cap.hardware.feetech import FeetechController
from lerobot_cap.hardware.calibration import MotorCalibration
from lerobot_cap.kinematics.engine import KinematicsEngine
from lerobot_cap.kinematics.calibration_limits import load_calibration_limits

from pix2robot_charuco_calibrator import CharucoBoardSpec, CharucoDetector

DEFAULT_BOARD_CONFIG = Path(__file__).parent / "board_config.yaml"
DEFAULT_SNAP_PX = 30.0


class Pix2RobotCalibrator:
    """
    픽셀 → 로봇 좌표 직접 변환 캘리브레이터.

    인터랙티브하게 픽셀(u,v) ↔ 로봇(x,y,z) 매칭쌍을 수집하여
    호모그래피(u,v→x,y) + 테이블 z상수를 계산한다.
    """

    def __init__(
        self,
        robot_id: int,
        camera_serial: str = None,
        board_config: Optional[str] = None,
    ):
        self.robot_id = robot_id
        self.camera_serial = camera_serial
        self.board_config_path = Path(board_config) if board_config else DEFAULT_BOARD_CONFIG

        # 보드 spec + 검출기
        self.board_spec, self.snap_max_px = self._load_board_spec(self.board_config_path)
        # K, dist 없이 픽셀 검출만 사용 (호모그래피는 3D 정보 불필요)
        self._detector = CharucoDetector(self.board_spec, K=None, dist=None)

        # 대응점 저장
        self.pixel_points: List[List[float]] = []     # [[u, v], ...] (sub-pixel)
        self.robot_points: List[List[float]] = []     # [[x, y, z], ...]
        self.corner_ids: List[int] = []               # 매칭점에 대응되는 charuco corner id

        # 캘리브레이션 결과
        self.homography: Optional[np.ndarray] = None  # 3x3
        self.table_z: Optional[float] = None
        self.error_stats: Optional[dict] = None

        # UI 상태
        self._current_image: Optional[np.ndarray] = None
        self._current_detections: list = []           # List[Detection]
        self._window_name = "Pix2Robot Calibration"
        self._pending_click: Optional[Tuple[int, int]] = None

        # 로봇 하드웨어
        self._controller: Optional[FeetechController] = None
        self._kinematics: Optional[KinematicsEngine] = None
        self._calibration_limits = None

    @staticmethod
    def _load_board_spec(path: Path) -> Tuple[CharucoBoardSpec, float]:
        """yaml에서 board spec과 snap 거리 로드."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"보드 설정 파일 없음: {path}")
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        if "board" not in cfg:
            raise ValueError(f"board 섹션 없음: {path}")
        spec = CharucoBoardSpec.from_dict(cfg["board"])
        snap_px = float(cfg.get("snap", {}).get("max_distance_px", DEFAULT_SNAP_PX))
        return spec, snap_px

    # ── 셋업/정리 ──────────────────────────────────────────────

    def setup_camera(self) -> np.ndarray:
        """
        RealSense 컬러 프레임 1장 캡처 + charuco 코너 검출.

        프리뷰에 실시간 검출 결과를 오버레이해서 충분한 코너가 보이는
        프레임을 사용자가 직접 골라 's' 키로 확정하게 한다.
        """
        from object_detection.camera import RealSenseD435

        with RealSenseD435(serial_number=self.camera_serial) as camera:
            print()
            print("-" * 60)
            print("[Step 1/3] 카메라 이미지 캡처")
            print("-" * 60)
            print("  Charuco 보드를 워크스페이스에 평평하게 놓으세요.")
            print("  실시간 검출(시안색 십자 + ID)이 충분히 잡히는 프레임에서")
            print("  's' 키를 눌러 캡처합니다. 'q'는 취소.")
            print()
            while True:
                color, _ = camera.get_frames()
                if color is None:
                    continue

                detections = self._detector.detect_charuco(color)
                preview = self._detector.annotate(color, detections)
                status = (
                    f"detected corners: {len(detections)}    "
                    "[s] capture  [q] quit"
                )
                cv2.putText(
                    preview, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    preview, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )

                cv2.imshow("Camera Preview", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    if len(detections) < 4:
                        print(
                            f"  검출된 코너 {len(detections)}개 — 너무 적음. "
                            "보드 자세/조명 조정 후 다시 's'."
                        )
                        continue
                    captured_color = color.copy()
                    self._current_detections = detections
                    cv2.destroyWindow("Camera Preview")
                    print(
                        f"이미지 캡처 완료: color={captured_color.shape}, "
                        f"코너 {len(detections)}개 검출"
                    )
                    return captured_color
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt("사용자 취소")

    def setup_robot(self) -> None:
        """FeetechController, KinematicsEngine 초기화."""
        config_path = (
            Path(__file__).parent.parent
            / f"robot_configs/robot/so101_robot{self.robot_id}.yaml"
        )
        if not config_path.exists():
            raise FileNotFoundError(f"로봇 설정 파일 없음: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # YAML 내 상대경로를 YAML 파일 위치 기준으로 resolve
        yaml_dir = config_path.parent
        for key in ("calibration_file", "compensation_file"):
            val = config.get(key)
            if val and not Path(val).is_absolute():
                config[key] = str((yaml_dir / val).resolve())
        kin_cfg = config.get("kinematics", {})
        if kin_cfg.get("urdf_path") and not Path(kin_cfg["urdf_path"]).is_absolute():
            kin_cfg["urdf_path"] = str((yaml_dir / kin_cfg["urdf_path"]).resolve())

        # 모터 컨트롤러
        calibration_path = Path(config['calibration_file'])
        with open(calibration_path, 'r') as f:
            calib_raw = json.load(f)

        calibration = {}
        for name, data in calib_raw.items():
            # LeRobot 포맷: "id" 필드 → MotorCalibration의 "motor_id"로 매핑
            motor_data = dict(data)
            if 'id' in motor_data and 'motor_id' not in motor_data:
                motor_data['motor_id'] = motor_data.pop('id')
            if 'model' not in motor_data:
                motor_data['model'] = 'sts3215'
            calib = MotorCalibration(**motor_data)
            calibration[calib.motor_id] = calib
        motor_ids = [m['id'] for m in config['motors'].values()]

        self._controller = FeetechController(
            port=config['port'],
            baudrate=config['baudrate'],
            motor_ids=motor_ids,
            calibration=calibration,
        )
        if not self._controller.connect():
            raise RuntimeError("모터 연결 실패")

        # 키네마틱스
        self._kinematics = KinematicsEngine(
            urdf_path=config['kinematics']['urdf_path'],
            end_effector_frame=config['kinematics']['end_effector_frame'],
        )

        # 캘리브레이션 리밋
        self._calibration_limits = load_calibration_limits(
            config['calibration_file'],
            joint_names=[
                'shoulder_pan', 'shoulder_lift', 'elbow_flex',
                'wrist_flex', 'wrist_roll',
            ],
        )
        print(f"로봇 {self.robot_id} 초기화 완료")

    def cleanup(self) -> None:
        """카메라/로봇 해제."""
        if self._controller is not None:
            try:
                self._controller.enable_torque()
            except Exception:
                pass
            self._controller.disconnect()
            self._controller = None
        cv2.destroyAllWindows()

    # ── UI ─────────────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 콜백 — 좌표만 저장, 처리는 메인 루프."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._pending_click = (x, y)

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """검출 코너 + 매칭점 + 상태 텍스트 오버레이."""
        # 1) charuco 코너 시안색 십자 + ID
        display = self._detector.annotate(frame, self._current_detections)

        # 2) 이미 사용된 코너 → 초록 원 강조 + 매칭점 라벨
        used = {
            cid: (px, rb)
            for cid, px, rb in zip(
                self.corner_ids, self.pixel_points, self.robot_points,
            )
        }
        for det in self._current_detections:
            if det.corner_id not in used:
                continue
            u, v = int(round(det.pixel_uv[0])), int(round(det.pixel_uv[1]))
            cv2.circle(display, (u, v), 10, (0, 255, 0), 2)
            _, rb = used[det.corner_id]
            label = f"({rb[0]:.3f}, {rb[1]:.3f}, {rb[2]:.3f})"
            cv2.putText(
                display, label, (u + 12, v + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                display, label, (u + 12, v + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA,
            )

        # 상태 텍스트
        n = len(self.pixel_points)
        status = (
            f"Pairs: {n} | click charuco corner | "
            "'u': undo | 'c': compute (>=4) | 's': save | 'q': quit"
        )
        cv2.putText(
            display, status, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            display, status, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        if self.homography is not None:
            cv2.putText(
                display, "[CALIBRATED]", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        return display

    # ── 포인트 수집 ────────────────────────────────────────────

    def _record_pixel_point(self, corner_id: int, u: float, v: float) -> None:
        """선택된 charuco 코너 알림."""
        n = len(self.pixel_points) + 1
        print()
        print(f"--- 포인트 {n} ---")
        print(f"[Charuco corner id={corner_id}] uv=({u:.2f}, {v:.2f}) 선택됨")

    def _record_robot_point(self) -> List[float]:
        """
        토크 OFF → FK 실시간 표시 → Enter → 기록 → 토크 ON.
        반환: [x, y, z] (meters, base_link 기준)
        """
        if self._controller is None:
            raise RuntimeError("로봇이 초기화되지 않음")

        print()
        print("  [로봇 포지셔닝]")
        print("  토크가 비활성화됩니다. 로봇 EE(그리퍼 끝)를")
        print("  방금 클릭한 이미지 위치에 해당하는 실제 지점으로 수동 이동하세요.")
        print("  아래에 실시간 TCP 좌표가 표시됩니다 (FK 자동 계산, 미터 단위).")
        print("  위치를 맞춘 뒤 Enter를 누르면 기록됩니다. (q+Enter: 취소)")
        print()
        self._controller.disable_torque()
        print("  >> 토크 OFF — 로봇을 자유롭게 움직이세요")

        try:
            while True:
                tcp_pos = self._get_tcp_position()
                sys.stdout.write(
                    f"\r  TCP (base): x={tcp_pos[0]:+.4f}, "
                    f"y={tcp_pos[1]:+.4f}, z={tcp_pos[2]:+.4f}   "
                )
                sys.stdout.flush()

                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.readline().strip()
                    if user_input.lower() == 'q':
                        print("\n취소됨")
                        self._controller.enable_torque()
                        return None
                    break

                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n중단됨")
            self._controller.enable_torque()
            return None

        # 최종 위치 기록
        tcp_pos = self._get_tcp_position()
        self._controller.enable_torque()

        print(f"\n[Robot] ({tcp_pos[0]:+.4f}, {tcp_pos[1]:+.4f}, {tcp_pos[2]:+.4f}) 기록됨")
        return tcp_pos.tolist()

    def _get_tcp_position(self) -> np.ndarray:
        """현재 관절값 → FK → TCP 위치."""
        normalized = self._controller.read_positions(normalize=True)
        joint_radians = self._calibration_limits.normalized_to_radians(
            normalized[:5]
        )
        return self._kinematics.get_ee_position(joint_radians)

    # ── 계산 ───────────────────────────────────────────────────

    def compute_homography(self) -> bool:
        """cv2.findHomography(pixels, robot_xy, RANSAC)로 호모그래피 계산."""
        n = len(self.pixel_points)
        if n < 4:
            print(f"최소 4쌍 필요 (현재 {n}쌍)")
            return False

        pixel_pts = np.array(self.pixel_points, dtype=np.float32)
        robot_xy = np.array(
            [[p[0], p[1]] for p in self.robot_points], dtype=np.float32
        )

        H, mask = cv2.findHomography(pixel_pts, robot_xy, cv2.RANSAC, 0.01)
        if H is None:
            print("호모그래피 계산 실패")
            return False

        # Outlier 감지 및 제거
        if mask is not None:
            outlier_indices = [
                i for i in range(n) if mask[i][0] == 0
            ]
            if outlier_indices:
                print(f"\n  RANSAC outlier {len(outlier_indices)}개 감지:")
                for idx in outlier_indices:
                    px = self.pixel_points[idx]
                    rb = self.robot_points[idx]
                    print(f"    P{idx+1}: pixel({px[0]}, {px[1]}) <-> "
                          f"robot({rb[0]:.4f}, {rb[1]:.4f}, {rb[2]:.4f})")

                # 역순으로 제거 (인덱스 밀림 방지)
                for idx in sorted(outlier_indices, reverse=True):
                    self.pixel_points.pop(idx)
                    self.robot_points.pop(idx)
                    if idx < len(self.corner_ids):
                        self.corner_ids.pop(idx)
                print(f"  -> {len(outlier_indices)}개 outlier 제거됨. "
                      f"남은 포인트: {len(self.pixel_points)}쌍")

                # outlier 제거 후 다시 계산
                if len(self.pixel_points) < 4:
                    print("  outlier 제거 후 포인트가 4쌍 미만입니다. 포인트를 더 수집하세요.")
                    self.homography = None
                    return False

                pixel_pts = np.array(self.pixel_points, dtype=np.float32)
                robot_xy = np.array(
                    [[p[0], p[1]] for p in self.robot_points], dtype=np.float32
                )
                H, mask = cv2.findHomography(pixel_pts, robot_xy, cv2.RANSAC, 0.01)
                if H is None:
                    print("  재계산 실패")
                    return False
                print(f"  -> outlier 제거 후 호모그래피 재계산 완료")

        self.homography = H
        self.table_z = self.compute_table_z()
        self.error_stats = self.verify()

        n_final = len(self.pixel_points)
        print()
        print("-" * 60)
        print("[Step 3/3] 호모그래피 계산 결과")
        print("-" * 60)
        print(f"  사용 포인트: {n_final}쌍" + (f" (원본 {n}쌍 중 outlier {n - n_final}개 자동 제거)" if n != n_final else ""))
        print(f"  테이블 z (base_link): {self.table_z:.4f} m")
        print(f"  평균 오차: {self.error_stats['mean_error_m']:.4f} m "
              f"({self.error_stats['mean_error_m']*100:.2f} cm)")
        print(f"  RMSE: {self.error_stats['rmse_m']:.4f} m")
        print(f"  최대 오차: {self.error_stats['max_error_m']:.4f} m")

        if mask is not None:
            inliers = int(mask.sum())
            print(f"  RANSAC inliers: {inliers}/{n_final}")

        print()
        if self.error_stats['mean_error_m'] < 0.01:
            print("  결과가 양호합니다. 's' 키로 저장하세요.")
        else:
            print("  오차가 큽니다. 포인트를 추가하거나 잘못된 포인트를 'u'로 제거 후 다시 'c'를 시도하세요.")
        print("  추가 포인트를 더 수집하려면 이미지에서 계속 클릭하세요.")

        return True

    def compute_table_z(self) -> float:
        """수집한 로봇 z값들의 평균 → 테이블 높이 상수."""
        z_values = [p[2] for p in self.robot_points]
        return float(np.mean(z_values))

    def verify(self) -> dict:
        """재투영 오차 계산 (포인트별, 평균, RMSE)."""
        if self.homography is None:
            raise RuntimeError("호모그래피가 계산되지 않음")

        errors = []
        print("\n[검증] 재투영 오차:")

        for px, rb in zip(self.pixel_points, self.robot_points):
            predicted_xy = cv2.perspectiveTransform(
                np.array([[px]], dtype=np.float32), self.homography
            )[0][0]
            err = float(np.sqrt(
                (predicted_xy[0] - rb[0])**2 + (predicted_xy[1] - rb[1])**2
            ))
            errors.append(err)
            print(
                f"  Pixel ({px[0]:.2f}, {px[1]:.2f}) → "
                f"predicted ({predicted_xy[0]:.4f}, {predicted_xy[1]:.4f}) "
                f"vs actual ({rb[0]:.4f}, {rb[1]:.4f}), "
                f"error: {err*100:.2f} cm"
            )

        errors_arr = np.array(errors)
        stats = {
            "mean_error_m": float(np.mean(errors_arr)),
            "max_error_m": float(np.max(errors_arr)),
            "rmse_m": float(np.sqrt(np.mean(errors_arr**2))),
            "per_point_errors_m": errors,
        }

        target_met = stats["mean_error_m"] < 0.01
        print(f"\n  평균 오차 < 1cm: {'OK' if target_met else 'FAIL'}")
        return stats

    # ── 변환 (캘리브레이션 후 사용) ────────────────────────────

    def pixel_to_robot(
        self, u: float, v: float, depth_m: Optional[float] = None,
    ) -> List[float]:
        """
        호모그래피(u,v→x,y) + table_z → [x, y, z].

        z는 항상 table_z (수집한 로봇 z 평균) 사용. 물체 높이 추정은
        지원하지 않음 — 테이블 평면 위 한 점만 변환한다.

        Args:
            u, v: 픽셀 좌표
            depth_m: legacy 호환용 인자. 값은 무시됨. Pix2RobotCharuco에는
                depth 기반 z 추정이 있으므로 polymorphic 호출 코드 호환을
                위해 시그니처만 유지.
        """
        if self.homography is None:
            raise RuntimeError("캘리브레이션이 완료되지 않음")

        pixel = np.array([[[u, v]]], dtype=np.float32)
        robot_xy = cv2.perspectiveTransform(pixel, self.homography)[0][0]
        return [float(robot_xy[0]), float(robot_xy[1]), float(self.table_z)]

    def robot_to_pixel(self, x: float, y: float, z: float = None) -> Tuple[int, int]:
        """
        로봇 좌표(x,y) → 픽셀 좌표(u,v) 역변환.

        호모그래피의 역행렬을 사용. legacy 캘리브레이터는 테이블 평면(z ≈ self.table_z)
        에 대해서만 호모그래피를 학습하므로, z 인자는 호환용으로만 받고 무시됩니다.
        Charuco 캘리브레이터(Pix2RobotCharuco)는 z를 실제로 사용한 3D 투영을 지원합니다.

        Args:
            x, y: 로봇 좌표 (base_link frame, meters)
            z: (legacy: 무시됨) 호환을 위해 받음. None 또는 임의 값.

        Returns:
            (u, v) 픽셀 좌표
        """
        if self.homography is None:
            raise RuntimeError("캘리브레이션이 완료되지 않음")

        H_inv = np.linalg.inv(self.homography)
        robot_pt = np.array([[[x, y]]], dtype=np.float32)
        pixel = cv2.perspectiveTransform(robot_pt, H_inv)[0][0]
        return int(round(pixel[0])), int(round(pixel[1]))

    # ── 저장/로드 ──────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        """결과를 .npz + .json으로 저장."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"robot{self.robot_id}_pix2robot_data"

        # .npz
        npz_path = output_dir / f"{prefix}.npz"
        save_dict = dict(
            homography=self.homography,
            table_z=np.array([self.table_z]),
            pixel_points=np.array(self.pixel_points),
            robot_points=np.array(self.robot_points),
            corner_ids=np.array(self.corner_ids, dtype=np.int32),
            mean_error_m=np.array([self.error_stats["mean_error_m"]]),
            max_error_m=np.array([self.error_stats["max_error_m"]]),
            rmse_m=np.array([self.error_stats["rmse_m"]]),
            per_point_errors_m=np.array(self.error_stats["per_point_errors_m"]),
        )
        np.savez(npz_path, **save_dict)

        # .json
        json_path = output_dir / f"{prefix}.json"
        json_data = {
            "robot_id": f"robot{self.robot_id}",
            "calibrated_at": datetime.now().isoformat(),
            "num_points": len(self.pixel_points),
            "homography_3x3": self.homography.tolist(),
            "table_z_robot_frame": self.table_z,
            "pixel_points": self.pixel_points,
            "robot_points": self.robot_points,
            "corner_ids": self.corner_ids,
            "board_spec": self.board_spec.to_dict(),
            "error_stats": self.error_stats,
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"\n저장 완료:")
        print(f"  {npz_path}")
        print(f"  {json_path}")

    def load(self, filepath: str) -> bool:
        """.npz 파일에서 캘리브레이션 로드."""
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"파일 없음: {filepath}")
            return False

        try:
            data = np.load(filepath, allow_pickle=True)
            self.homography = data['homography']
            self.table_z = float(data['table_z'][0])
            self.pixel_points = data['pixel_points'].tolist()
            self.robot_points = data['robot_points'].tolist()
            self.corner_ids = (
                data['corner_ids'].tolist() if 'corner_ids' in data else []
            )

            if 'mean_error_m' in data:
                self.error_stats = {
                    "mean_error_m": float(data['mean_error_m'][0]),
                    "max_error_m": float(data['max_error_m'][0]) if 'max_error_m' in data else None,
                    "rmse_m": float(data['rmse_m'][0]),
                    "per_point_errors_m": data['per_point_errors_m'].tolist() if 'per_point_errors_m' in data else [],
                }

            print(f"로드 완료: {filepath}")
            print(f"  포인트: {len(self.pixel_points)}쌍")
            print(f"  테이블 z: {self.table_z:.4f} m")
            return True
        except Exception as e:
            print(f"로드 실패: {e}")
            return False

    # ── 메인 루프 ──────────────────────────────────────────────

    def calibrate_interactive(self, resume: bool = False) -> bool:
        """
        전체 인터랙티브 캘리브레이션 워크플로우.

        Args:
            resume: True이면 기존 캘리브레이션 데이터를 로드하여 이어서 수집
        """
        print("\n" + "=" * 60)
        print(f"Pix2Robot Calibration — Robot {self.robot_id}")
        print("=" * 60)

        # 기존 데이터 로드 (resume 모드)
        if resume:
            data_path = (
                Path(__file__).parent.parent
                / "robot_configs" / "pix2robot_matrices"
                / f"robot{self.robot_id}_pix2robot_data.npz"
            )
            if data_path.exists():
                self.load(str(data_path))
                # 이어서 수집할 때는 호모그래피 초기화 (재계산 필요)
                self.homography = None
                self.error_stats = None
                print(f"\n  기존 {len(self.pixel_points)}쌍 로드됨. 이어서 포인트를 추가합니다.")
            else:
                print(f"\n  기존 데이터 없음 ({data_path}). 새로 시작합니다.")

        # 1. 카메라 캡처
        self._current_image = self.setup_camera()

        # 2. 로봇 초기화
        self.setup_robot()

        try:
            return self._run_interactive_loop()
        finally:
            self.cleanup()

    def _run_interactive_loop(self) -> bool:
        """OpenCV 창 + 로봇 포지셔닝 루프."""
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

        print()
        print("-" * 60)
        print("[Step 2/3] 포인트 수집 (Charuco 코너 스냅)")
        print("-" * 60)
        print("  화면의 시안색 십자(✚) + 숫자가 검출된 Charuco 코너입니다.")
        print(f"  코너 근처 {self.snap_max_px:.0f}px 이내를 좌클릭하면 가장 가까운")
        print("  코너로 자동 스냅되어 로봇 포지셔닝 단계로 전환됩니다.")
        print(f"  (코너에서 {self.snap_max_px:.0f}px 초과는 무시됨)")
        print()
        print("  <워크플로우>")
        print("  1) 이미지에서 코너 근처 좌클릭 (sub-pixel 좌표 기록)")
        print("  2) 로봇 토크가 풀림 → 로봇 EE를 해당 코너 위치로 수동 이동")
        print("  3) 터미널에 실시간 TCP 좌표가 표시됨 (좌표는 FK로 자동 계산)")
        print("  4) Enter → 로봇 좌표 기록, 토크 복원")
        print("  5) 이미지 창이 다시 열림 → 다음 코너 반복")
        print()
        print("  <조작키>")
        print("  좌클릭     : 가장 가까운 코너 선택 (스냅)")
        print("  'u'        : 마지막 포인트 쌍 취소")
        print("  'd'        : 특정 포인트 삭제 (번호 입력)")
        print("  'c'        : 호모그래피 계산 (최소 4쌍, 권장 8~12쌍)")
        print("  's'        : 결과 저장")
        print("  'q'        : 종료")
        print("  Enter(터미널): 로봇 위치 확인")
        print("  q+Enter    : 현재 포인트 취소")
        print()
        print("  TIP: 작업 영역의 모서리와 중앙에 고르게 코너를 골라 분포시키세요.")
        print()

        self._pending_click = None

        while True:
            display = self._draw_overlay(self._current_image)
            cv2.imshow(self._window_name, display)
            key = cv2.waitKey(50) & 0xFF

            # 클릭 처리
            if self._pending_click is not None:
                click_x, click_y = self._pending_click
                self._pending_click = None
                self._handle_pixel_click(click_x, click_y)

            if key == ord('q'):
                print("종료")
                return self.homography is not None

            elif key == ord('u'):
                self._undo_last_point()

            elif key == ord('d'):
                self._delete_points_interactive()

            elif key == ord('c'):
                self.compute_homography()

            elif key == ord('s'):
                if self.homography is not None:
                    output_dir = (
                        Path(__file__).parent.parent
                        / "robot_configs" / "pix2robot_matrices"
                    )
                    self.save(str(output_dir))
                else:
                    print("먼저 'c'로 호모그래피를 계산하세요")

    def _handle_pixel_click(self, x: int, y: int) -> None:
        """클릭 → 가장 가까운 charuco 코너로 스냅 → 로봇 포지셔닝 → 매칭쌍 저장."""
        snapped = self._snap_to_corner(x, y)
        if snapped is None:
            return  # 무시 (사유는 _snap_to_corner 가 출력)

        corner_id, u_snap, v_snap = snapped
        self._record_pixel_point(corner_id, u_snap, v_snap)

        # OpenCV 창 닫기 (터미널 포커스)
        cv2.destroyWindow(self._window_name)
        cv2.waitKey(1)

        # 로봇 위치 기록
        robot_pos = self._record_robot_point()

        if robot_pos is not None:
            self.pixel_points.append([u_snap, v_snap])
            self.robot_points.append(robot_pos)
            self.corner_ids.append(corner_id)

            n = len(self.pixel_points)
            print(f"  매칭쌍 {n} 저장: "
                  f"corner id={corner_id} uv=({u_snap:.2f}, {v_snap:.2f}) "
                  f"<-> robot({robot_pos[0]:.4f}, "
                  f"{robot_pos[1]:.4f}, {robot_pos[2]:.4f})")
            if n < 4:
                print(f"  -> 최소 {4 - n}쌍 더 필요합니다. (최소 4쌍, 권장 8~12쌍)")
            elif n < 8:
                print(f"  -> {n}쌍 수집됨. 'c'로 계산 가능. 정밀도를 위해 {8 - n}쌍 더 추가 권장.")
            else:
                print(f"  -> {n}쌍 수집됨. 'c'로 호모그래피를 계산하세요.")
        else:
            print("  포인트 취소됨")

        print()
        print("  이미지 창이 다시 열립니다. 다음 코너를 클릭하세요.")

        # OpenCV 창 다시 열기
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

    def _snap_to_corner(
        self, x: int, y: int,
    ) -> Optional[Tuple[int, float, float]]:
        """
        클릭 위치 (x,y)에서 가장 가까운 미사용 charuco 코너로 스냅.

        Returns:
            (corner_id, u, v) sub-pixel 좌표. snap 실패 시 None.
        """
        if not self._current_detections:
            print("  검출된 코너 없음 — 캡처를 다시 진행하세요")
            return None

        used = set(self.corner_ids)
        candidates = [
            d for d in self._current_detections if d.corner_id not in used
        ]
        if not candidates:
            print("  사용 가능한 코너 없음 (모두 사용됨)")
            return None

        dists = [
            (float(np.hypot(d.pixel_uv[0] - x, d.pixel_uv[1] - y)), d)
            for d in candidates
        ]
        dists.sort(key=lambda t: t[0])
        best_dist, best = dists[0]

        if best_dist > self.snap_max_px:
            print(
                f"  클릭 위치에서 가까운 코너 없음 "
                f"(최단거리 {best_dist:.1f}px > 임계값 {self.snap_max_px:.0f}px)"
            )
            return None

        return best.corner_id, float(best.pixel_uv[0]), float(best.pixel_uv[1])

    def _undo_last_point(self) -> None:
        """마지막 포인트 쌍 취소."""
        if self.pixel_points:
            px = self.pixel_points.pop()
            rb = self.robot_points.pop()
            cid = self.corner_ids.pop() if self.corner_ids else None
            cid_str = f"id={cid} " if cid is not None else ""
            print(f"취소: {cid_str}uv=({px[0]:.2f}, {px[1]:.2f}) <-> "
                  f"robot({rb[0]:.4f}, {rb[1]:.4f}, {rb[2]:.4f})")
            print(f"남은 포인트: {len(self.pixel_points)}쌍")
        else:
            print("취소할 포인트 없음")

    def _delete_points_interactive(self) -> None:
        """특정 포인트를 번호로 선택하여 삭제."""
        if not self.pixel_points:
            print("삭제할 포인트 없음")
            return

        # OpenCV 창 닫기 (터미널 포커스)
        cv2.destroyWindow(self._window_name)
        cv2.waitKey(1)

        print()
        print("-" * 40)
        print("현재 포인트 목록:")
        for i, (px, rb) in enumerate(zip(self.pixel_points, self.robot_points)):
            cid = self.corner_ids[i] if i < len(self.corner_ids) else None
            cid_str = f"id={cid:>3} " if cid is not None else "          "
            print(f"  P{i+1}: {cid_str}uv=({px[0]:.2f}, {px[1]:.2f}) <-> "
                  f"robot({rb[0]:.4f}, {rb[1]:.4f}, {rb[2]:.4f})")
        print("-" * 40)
        print("삭제할 포인트 번호를 입력하세요 (쉼표로 복수 가능, 예: 3,7,11)")
        print("취소하려면 빈 줄에서 Enter")

        user_input = input("  삭제할 번호: ").strip()

        if not user_input:
            print("  삭제 취소됨")
        else:
            try:
                indices = [int(x.strip()) - 1 for x in user_input.split(",")]
                # 유효 범위 확인
                invalid = [i + 1 for i in indices if i < 0 or i >= len(self.pixel_points)]
                if invalid:
                    print(f"  잘못된 번호: {invalid} (범위: 1~{len(self.pixel_points)})")
                else:
                    # 역순 삭제
                    for idx in sorted(set(indices), reverse=True):
                        px = self.pixel_points.pop(idx)
                        rb = self.robot_points.pop(idx)
                        cid = (
                            self.corner_ids.pop(idx)
                            if idx < len(self.corner_ids) else None
                        )
                        cid_str = f"id={cid} " if cid is not None else ""
                        print(f"  삭제: P{idx+1} {cid_str}uv=({px[0]:.2f}, {px[1]:.2f}) <-> "
                              f"robot({rb[0]:.4f}, {rb[1]:.4f}, {rb[2]:.4f})")
                    print(f"  남은 포인트: {len(self.pixel_points)}쌍")
                    # 호모그래피 무효화
                    self.homography = None
                    self.error_stats = None
                    print("  호모그래피가 초기화되었습니다. 'c'로 재계산하세요.")
            except ValueError:
                print("  잘못된 입력. 숫자를 쉼표로 구분하여 입력하세요 (예: 3,7)")

        # OpenCV 창 다시 열기
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)
