"""
Phase 2: Camera Intrinsics Calibration

Charuco 보드를 다양한 자세로 N장 촬영 → cv2.aruco.calibrateCameraCharuco
→ K (3x3 내부 파라미터), dist (왜곡 계수) 추정.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from .camera_setup import CalibrationCamera
from .charuco_detector import CharucoBoardSpec, CharucoDetector
from .visualization import (
    print_phase_banner,
    print_phase_summary,
    draw_phase_overlay,
    draw_status_text,
)


class IntrinsicsCalibrator:
    """Phase 2: 보드를 여러 자세로 캡처하여 K, dist 추정."""

    def __init__(
        self,
        board_spec: CharucoBoardSpec,
        target_num_images: int = 25,
        min_num_images: int = 15,
        min_corners_per_image: int = 8,
        target_rms_px: float = 0.5,
        camera_serial: Optional[str] = None,
    ):
        self.spec = board_spec
        self.target_num_images = target_num_images
        self.min_num_images = min_num_images
        self.min_corners_per_image = min_corners_per_image
        self.target_rms_px = target_rms_px
        self.camera_serial = camera_serial

        self.detector = CharucoDetector(board_spec)
        self.captured: List[Tuple[np.ndarray, np.ndarray]] = []  # (corners, ids)
        self.image_size: Optional[Tuple[int, int]] = None

        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.rms_error: Optional[float] = None

    def run(self, save_path: Path) -> bool:
        print_phase_banner(
            2,
            subtitle=(
                f"보드를 다양한 자세/거리/회전으로 {self.target_num_images}장 캡처\n"
                f"  (가까이/멀리, 기울임, 화면 가장자리 모두 포함하면 좋음)"
            ),
        )

        success = self._capture_loop()
        if success:
            self._save(save_path)
        return success

    def _capture_loop(self) -> bool:
        window_name = "Phase 2 - Intrinsics"
        cv2.namedWindow(window_name)

        with CalibrationCamera(serial=self.camera_serial) as camera:
            print("  키 가이드: [s] 캡처  [c] 캘리브 계산  [u] 마지막 취소  [q] 종료")
            print()

            while True:
                color, _ = camera.get_frames()
                if color is None:
                    continue

                if self.image_size is None:
                    self.image_size = (color.shape[1], color.shape[0])

                # 실시간 검출 미리보기
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                charuco_corners, charuco_ids, _, _ = (
                    self.detector._charuco_detector.detectBoard(gray)
                )

                vis = color.copy()
                num_corners = 0
                if charuco_ids is not None and len(charuco_ids) > 0:
                    num_corners = len(charuco_ids)
                    cv2.aruco.drawDetectedCornersCharuco(
                        vis, charuco_corners, charuco_ids
                    )

                # 검출 부족 경고는 이미지 위에 (헤더 추가 전)
                if num_corners < self.min_corners_per_image:
                    vis = draw_status_text(
                        vis,
                        [f"WARN: detected corners < {self.min_corners_per_image}"],
                        origin=(10, 30),
                        color=(0, 165, 255),
                    )

                progress = (
                    f"captured: {len(self.captured)}/{self.target_num_images}  "
                    f"detected: {num_corners}"
                )
                vis = draw_phase_overlay(
                    vis, phase=2,
                    progress_text=progress,
                    keys_text="[s]ave  [c]alibrate  [u]ndo  [q]uit",
                )

                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    if num_corners >= self.min_corners_per_image:
                        self.captured.append(
                            (charuco_corners.copy(), charuco_ids.copy())
                        )
                        print(
                            f"  [Capture {len(self.captured)}] "
                            f"corners={num_corners}"
                        )
                    else:
                        print(
                            f"  검출 코너 부족 ({num_corners} < "
                            f"{self.min_corners_per_image})"
                        )

                elif key == ord('u'):
                    if self.captured:
                        self.captured.pop()
                        print(f"  마지막 캡처 취소. 남은: {len(self.captured)}")

                elif key == ord('c'):
                    if len(self.captured) >= self.min_num_images:
                        cv2.destroyWindow(window_name)
                        return self._calibrate()
                    else:
                        print(
                            f"  최소 {self.min_num_images}장 필요 "
                            f"(현재 {len(self.captured)})"
                        )

                elif key == ord('q'):
                    print("  Phase 2 사용자 중단")
                    cv2.destroyWindow(window_name)
                    print_phase_summary(2, success=False, details="사용자 중단")
                    return False

    def _calibrate(self) -> bool:
        n = len(self.captured)
        print(f"\n  {n} 장으로 캘리브레이션 계산 중...")

        all_corners = [c for c, _ in self.captured]
        all_ids = [i for _, i in self.captured]
        board = self.spec.board()

        try:
            rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_corners,
                charucoIds=all_ids,
                board=board,
                imageSize=self.image_size,
                cameraMatrix=None,
                distCoeffs=None,
            )
        except cv2.error as e:
            print(f"  캘리브 실패: {e}")
            print_phase_summary(2, success=False, details="OpenCV 오류")
            return False

        self.K = K
        self.dist = dist
        self.rms_error = float(rms)

        print()
        print(f"  RMS reprojection error: {self.rms_error:.4f} px")
        print(f"  K = ")
        for row in K:
            print(f"    [{row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f}]")
        print(f"  dist = {dist.flatten().tolist()}")
        print()

        passed = self.rms_error <= self.target_rms_px
        if passed:
            print(f"  통과 (목표 {self.target_rms_px} px 이하)")
            print_phase_summary(
                2, success=True,
                details=f"RMS={self.rms_error:.3f} px, {n} 장",
            )
        else:
            print(
                f"  주의: 목표 {self.target_rms_px} px 초과. "
                "더 다양한 자세로 재시도 권장."
            )
            print_phase_summary(
                2, success=True,  # 사용은 가능하므로 일단 PASS 처리
                details=f"RMS={self.rms_error:.3f} px (목표 초과), {n} 장",
            )
        return True

    def _save(self, save_path: Path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            save_path,
            K=self.K,
            dist=self.dist,
            rms_error=np.array([self.rms_error]),
            num_images=np.array([len(self.captured)]),
            image_width=np.array([self.image_size[0]]),
            image_height=np.array([self.image_size[1]]),
        )

        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                "K": self.K.tolist(),
                "dist": self.dist.flatten().tolist(),
                "rms_error_px": self.rms_error,
                "num_images": len(self.captured),
                "image_size": [self.image_size[0], self.image_size[1]],
                "board": self.spec.to_dict(),
            }, f, indent=2)

        print(f"  저장:")
        print(f"    {save_path}")
        print(f"    {json_path}")


def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """저장된 K, dist 로드."""
    data = np.load(path)
    return data['K'], data['dist']
