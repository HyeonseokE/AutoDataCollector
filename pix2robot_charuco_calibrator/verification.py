"""
Phase 4: Verification — 캘리브레이션 정확도 실측

워크플로:
    1. 새 보드 위치 캡처 (Phase 3에서 사용 안 한 자세)
    2. Charuco 자동 검출 → 매칭점_camera 자동 계산 (Phase 2 K, dist 사용)
    3. 변환행렬 적용 → 매칭점_robot 예측
    4. 사용자가 EE를 같은 코너에 터치 (Phase 3와 동일한 방식)
    5. 예측 vs 실측 오차 측정
    6. 통계 출력 + 저장
"""

import json
import select
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from .camera_setup import CalibrationCamera
from .charuco_detector import CharucoBoardSpec, CharucoDetector, Detection
from .geometry import apply_transform
from .robot_setup import (
    load_robot,
    get_tcp_position,
    cleanup_robot,
)
from .visualization import (
    print_phase_banner,
    print_phase_summary,
    draw_phase_overlay,
    draw_status_text,
    PHASE_HEADER_HEIGHT,
)


class VerificationResult:
    def __init__(
        self,
        corner_id: int,
        pixel_uv: Tuple[float, float],
        predicted_robot: np.ndarray,
        measured_robot: np.ndarray,
    ):
        self.corner_id = corner_id
        self.pixel_uv = pixel_uv
        self.predicted_robot = np.asarray(predicted_robot, dtype=np.float64)
        self.measured_robot = np.asarray(measured_robot, dtype=np.float64)
        self.error_vec = self.measured_robot - self.predicted_robot
        self.error_norm = float(np.linalg.norm(self.error_vec))

    def to_dict(self) -> dict:
        return {
            "corner_id": self.corner_id,
            "pixel_uv": list(self.pixel_uv),
            "predicted_robot": self.predicted_robot.tolist(),
            "measured_robot": self.measured_robot.tolist(),
            "error_vec": self.error_vec.tolist(),
            "error_mm": self.error_norm * 1000,
        }


class Verifier:
    """Phase 4: 새 보드 위치에서 holdout 검증."""

    def __init__(
        self,
        robot_id: int,
        board_spec: CharucoBoardSpec,
        K: np.ndarray,
        dist: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        num_test_corners: int = 5,
        target_avg_error_mm: float = 3.0,
        camera_serial: Optional[str] = None,
    ):
        self.robot_id = robot_id
        self.spec = board_spec
        self.K = K
        self.dist = dist
        self.R = R
        self.t = t
        self.num_test_corners = num_test_corners
        self.target_avg_error_mm = target_avg_error_mm
        self.camera_serial = camera_serial

        self.detector = CharucoDetector(board_spec, K=K, dist=dist)
        self.results: List[VerificationResult] = []

        self._current_color: Optional[np.ndarray] = None
        self._current_depth: Optional[np.ndarray] = None
        self._current_detections: List[Detection] = []
        self._window_name = "Phase 4 - Verification"
        self._pending_click: Optional[Tuple[int, int]] = None

        self._controller = None
        self._kinematics = None
        self._calibration_limits = None

    def run(self, save_path: Path) -> bool:
        print_phase_banner(
            4,
            subtitle=(
                f"새 보드 위치에서 {self.num_test_corners}개 코너로 holdout 검증\n"
                "  카메라 예측 vs EE 실측 오차 측정"
            ),
        )

        try:
            self._controller, self._kinematics, self._calibration_limits = \
                load_robot(self.robot_id)
        except Exception as e:
            print(f"  로봇 연결 실패: {e}")
            print_phase_summary(4, success=False, details=str(e))
            return False

        ok = False
        try:
            with CalibrationCamera(serial=self.camera_serial) as camera:
                if not self._capture_new_board(camera):
                    return False
                ok = self._verification_loop(camera)
        finally:
            cleanup_robot(self._controller)
            cv2.destroyAllWindows()

        if ok:
            self._save(save_path)
            return True
        return False

    def _capture_new_board(self, camera) -> bool:
        print()
        print("  Phase 3에서 사용하지 않은 새 보드 위치를 두고 's' 키로 캡처하세요.")
        win = "Phase 4 - Capture New Board"
        cv2.namedWindow(win)
        while True:
            color, depth = camera.get_frames()
            if color is None:
                continue
            detections = self.detector.detect_charuco(color)
            valid = [d for d in detections if d.has_3d]

            vis = color.copy()
            vis = self.detector.annotate(vis, detections)

            if len(valid) < self.num_test_corners:
                vis = draw_status_text(
                    vis,
                    [f"WARN: valid corners < {self.num_test_corners}"],
                    origin=(10, 30), color=(0, 165, 255),
                )

            progress = (
                f"detected: {len(detections)} (valid 3D: {len(valid)})  "
                f"target: {self.num_test_corners}"
            )
            vis = draw_phase_overlay(
                vis, phase=4,
                progress_text=progress,
                keys_text="[s] confirm capture  [q] quit",
            )

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(valid) >= self.num_test_corners:
                    self._current_color = color.copy()
                    self._current_depth = depth.copy() if depth is not None else None
                    self._current_detections = detections
                    cv2.destroyWindow(win)
                    print(f"  캡처: {len(valid)} 유효 코너")
                    return True
                else:
                    print(f"  유효 코너 부족 ({len(valid)} < {self.num_test_corners})")
            elif key == ord('q'):
                cv2.destroyWindow(win)
                return False

    def _verification_loop(self, camera) -> bool:
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_cb)

        print()
        print(f"  검증 루프 시작 ({self.num_test_corners} 코너 목표).")
        print("  키: [click] 코너 선택  [r] 보드 재캡처  "
              "[c] 결과 요약  [s] 저장+종료  [q] 종료")
        print()

        self._pending_click = None

        while True:
            vis = self._render()
            cv2.imshow(self._window_name, vis)
            key = cv2.waitKey(50) & 0xFF

            if self._pending_click is not None:
                cx, cy = self._pending_click
                self._pending_click = None
                self._handle_click(cx, cy)

            if key == ord('q'):
                if not self.results:
                    print_phase_summary(4, success=False, details="결과 없음")
                    return False
                break
            elif key == ord('r'):
                cv2.destroyWindow(self._window_name)
                if not self._capture_new_board(camera):
                    return False
                cv2.namedWindow(self._window_name)
                cv2.setMouseCallback(self._window_name, self._mouse_cb)
            elif key == ord('c'):
                self._print_summary()
            elif key == ord('s'):
                if self.results:
                    break
                else:
                    print("  검증 결과 없음")

        return self._finalize()

    def _render(self) -> np.ndarray:
        if self._current_color is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        vis = self._current_color.copy()
        vis = self.detector.annotate(vis, self._current_detections)

        # 이미 측정한 코너는 색칠
        used = {r.corner_id for r in self.results}
        for d in self._current_detections:
            if d.corner_id in used:
                u, v = int(round(d.pixel_uv[0])), int(round(d.pixel_uv[1]))
                cv2.circle(vis, (u, v), 8, (0, 255, 0), 2)

        avg = (
            np.mean([r.error_norm for r in self.results]) * 1000
            if self.results else 0
        )
        progress = (
            f"verified {len(self.results)}/{self.num_test_corners}  "
            f"avg error: {avg:.2f} mm"
        )
        vis = draw_phase_overlay(
            vis, phase=4,
            progress_text=progress,
            keys_text="[click] select corner  [c] summary  [r] recapture  [s] save+exit",
        )
        return vis

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 헤더 패딩 보정 → 원본 이미지 좌표
            self._pending_click = (x, y - PHASE_HEADER_HEIGHT)

    def _handle_click(self, cx: int, cy: int):
        used = {r.corner_id for r in self.results}
        candidates = [
            d for d in self._current_detections
            if d.has_3d and d.corner_id not in used
        ]
        if not candidates:
            print("  사용 가능한 코너 없음")
            return

        dists = [(np.hypot(d.pixel_uv[0] - cx, d.pixel_uv[1] - cy), d) for d in candidates]
        dists.sort(key=lambda x: x[0])
        if dists[0][0] > 30:
            print(f"  너무 멀음 ({dists[0][0]:.1f}px)")
            return

        det = dists[0][1]
        self._verify_corner(det)

    def _verify_corner(self, det: Detection):
        # 예측
        predicted = apply_transform(det.camera_xyz, self.R, self.t)
        print()
        print(f"  >> 코너 ID {det.corner_id}")
        print(f"     매칭점_camera: ({det.camera_xyz[0]:.4f}, "
              f"{det.camera_xyz[1]:.4f}, {det.camera_xyz[2]:.4f})")
        print(f"     예측 매칭점_robot: ({predicted[0]:.4f}, "
              f"{predicted[1]:.4f}, {predicted[2]:.4f})")
        print()
        print("  EE 팁을 이 코너에 터치 후 Enter (q+Enter: 스킵)")

        cv2.destroyWindow(self._window_name)
        cv2.waitKey(1)
        self._controller.disable_torque()
        print("  >> 토크 OFF")

        thickness = float(getattr(self.spec, "thickness_m", 0.0) or 0.0)

        try:
            cancelled = False
            while True:
                tcp = get_tcp_position(
                    self._controller, self._kinematics, self._calibration_limits,
                )
                tcp_corrected = tcp.copy()
                tcp_corrected[2] -= thickness
                err = tcp_corrected - predicted
                err_mm = np.linalg.norm(err) * 1000
                sys.stdout.write(
                    f"\r  TCP: x={tcp[0]:+.4f} y={tcp[1]:+.4f} z={tcp[2]:+.4f}  "
                    f"| 예측 대비 오차: {err_mm:.2f} mm    "
                )
                sys.stdout.flush()

                if select.select([sys.stdin], [], [], 0.1)[0]:
                    raw = sys.stdin.readline().strip()
                    if raw.lower() == 'q':
                        cancelled = True
                        break
                    break
                time.sleep(0.05)

            measured = get_tcp_position(
                self._controller, self._kinematics, self._calibration_limits,
            )
            measured = measured.copy()
            measured[2] -= thickness
        finally:
            self._controller.enable_torque()
            print("\n  >> 토크 ON")

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_cb)

        if cancelled:
            print("  스킵됨")
            return

        result = VerificationResult(
            corner_id=det.corner_id,
            pixel_uv=det.pixel_uv,
            predicted_robot=predicted,
            measured_robot=measured,
        )
        self.results.append(result)
        print(f"  검증 #{len(self.results)}: 오차 = {result.error_norm*1000:.2f} mm "
              f"(Δx={result.error_vec[0]*1000:+.2f}, "
              f"Δy={result.error_vec[1]*1000:+.2f}, "
              f"Δz={result.error_vec[2]*1000:+.2f})")

    def _print_summary(self):
        if not self.results:
            print("  결과 없음")
            return
        errs = np.array([r.error_norm for r in self.results])
        print()
        print(f"  검증 요약 ({len(self.results)} 코너):")
        print(f"    평균 오차: {errs.mean()*1000:.2f} mm")
        print(f"    Max:      {errs.max()*1000:.2f} mm")
        print(f"    Min:      {errs.min()*1000:.2f} mm")
        print(f"    Std:      {errs.std()*1000:.2f} mm")
        print()

    def _finalize(self) -> bool:
        if not self.results:
            return False
        errs = np.array([r.error_norm for r in self.results])
        avg_mm = errs.mean() * 1000
        passed = avg_mm <= self.target_avg_error_mm
        print()
        print(f"  최종 평균 오차: {avg_mm:.2f} mm "
              f"(목표 ≤ {self.target_avg_error_mm} mm) "
              f"{'PASS' if passed else 'FAIL'}")
        print_phase_summary(
            4, success=passed,
            details=f"평균 {avg_mm:.2f} mm, {len(self.results)} 코너",
        )
        return True

    def _save(self, save_path: Path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        errs = np.array([r.error_norm for r in self.results])
        with open(save_path, 'w') as f:
            json.dump({
                "robot_id": f"robot{self.robot_id}",
                "num_corners": len(self.results),
                "avg_error_mm": float(errs.mean() * 1000),
                "max_error_mm": float(errs.max() * 1000),
                "min_error_mm": float(errs.min() * 1000),
                "std_error_mm": float(errs.std() * 1000),
                "target_avg_error_mm": self.target_avg_error_mm,
                "passed": bool(errs.mean() * 1000 <= self.target_avg_error_mm),
                "results": [r.to_dict() for r in self.results],
            }, f, indent=2)
        print(f"  저장: {save_path}")
