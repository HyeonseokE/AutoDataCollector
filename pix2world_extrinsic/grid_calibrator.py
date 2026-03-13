"""
Grid-based Calibration Module
1cm 그리드 판을 이용한 카메라-월드 좌표계 캘리브레이션

사용법:
1. 카메라로 그리드 판을 촬영
2. 원점(0,0)을 먼저 클릭
3. X축 양의 방향 점을 클릭 (방향 설정용)
4. 추가 그리드 교점들을 클릭하며 월드 좌표 입력
5. 캘리브레이션 완료 후 변환 행렬 저장
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import json


class GridCalibrator:
    def __init__(self, grid_size_cm: float = 1.0):
        """
        그리드 기반 캘리브레이션 초기화

        Args:
            grid_size_cm: 그리드 한 칸 크기 (cm)
        """
        self.grid_size_cm = grid_size_cm

        # 대응점 저장
        self.pixel_points: List[Tuple[int, int]] = []  # (u, v)
        self.world_points: List[Tuple[float, float, float]] = []  # (x, y, z)

        # 캘리브레이션 결과
        self.homography_matrix: Optional[np.ndarray] = None  # 2D 변환 (평면)
        self.transform_matrix: Optional[np.ndarray] = None  # 3D 변환

        # 원점 및 축 설정
        self.origin_pixel: Optional[Tuple[int, int]] = None
        self.x_axis_pixel: Optional[Tuple[int, int]] = None

        # UI 상태
        self._current_image = None
        self._click_mode = "origin"  # "origin", "x_axis", "points"
        self._window_name = "Grid Calibration"
        self._pending_click = None  # 클릭 좌표를 메인 루프로 전달

    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 콜백 — 좌표만 저장, input()은 메인 루프에서 처리"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._pending_click = (x, y)

    def _get_world_input(self, label, x, y):
        """창을 숨기고 터미널 입력을 받은 후 다시 표시"""
        print(f"\n[Calibration] {label} at pixel ({x}, {y})")
        # 창 숨기기 — input() 중 "not responding" 방지
        cv2.destroyWindow(self._window_name)
        cv2.waitKey(1)

        world_x = float(input("  Enter world X coordinate (cm): "))
        world_y = float(input("  Enter world Y coordinate (cm): "))
        world_z = float(input("  Enter world Z coordinate (cm, usually 0): "))

        # 창 다시 열기
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

        return world_x, world_y, world_z

    def _process_click(self, x, y):
        """메인 루프에서 호출: 클릭 처리 + 터미널 입력"""
        if self._click_mode == "origin":
            self.origin_pixel = (x, y)
            self.pixel_points.append((x, y))
            self.world_points.append((0.0, 0.0, 0.0))
            print(f"[Calibration] Origin set at pixel ({x}, {y}) -> World (0, 0, 0)")
            self._click_mode = "x_axis"
            self._draw_points()

        elif self._click_mode == "x_axis":
            self.x_axis_pixel = (x, y)
            world_x, world_y, world_z = self._get_world_input("X-axis direction point", x, y)

            self.pixel_points.append((x, y))
            self.world_points.append((world_x, world_y, world_z))
            print(f"  -> World ({world_x}, {world_y}, {world_z})")

            self._click_mode = "points"
            self._draw_points()

        elif self._click_mode == "points":
            world_x, world_y, world_z = self._get_world_input(f"Point #{len(self.pixel_points)+1}", x, y)

            self.pixel_points.append((x, y))
            self.world_points.append((world_x, world_y, world_z))
            print(f"  -> World ({world_x}, {world_y}, {world_z})")
            print(f"  Total points: {len(self.pixel_points)}")
            self._draw_points()

    def _draw_points(self):
        """캘리브레이션 점들을 이미지에 표시"""
        if self._current_image is None:
            return

        display = self._current_image.copy()

        # 원점 표시 (빨간색, 큰 원)
        if self.origin_pixel:
            cv2.circle(display, self.origin_pixel, 10, (0, 0, 255), -1)
            cv2.putText(display, "ORIGIN (0,0,0)",
                       (self.origin_pixel[0] + 15, self.origin_pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # X축 방향점 표시 (녹색)
        if self.x_axis_pixel:
            cv2.circle(display, self.x_axis_pixel, 8, (0, 255, 0), -1)
            cv2.arrowedLine(display, self.origin_pixel, self.x_axis_pixel,
                           (0, 255, 0), 2, tipLength=0.1)
            cv2.putText(display, "X-axis",
                       (self.x_axis_pixel[0] + 15, self.x_axis_pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 나머지 점들 표시 (파란색)
        for i, (pixel, world) in enumerate(zip(self.pixel_points[2:], self.world_points[2:]), start=2):
            cv2.circle(display, pixel, 6, (255, 0, 0), -1)
            label = f"P{i}: ({world[0]:.1f}, {world[1]:.1f})"
            cv2.putText(display, label, (pixel[0] + 10, pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # 안내 메시지
        if self._click_mode == "origin":
            msg = "Click on the ORIGIN (0,0) of the world coordinate system"
        elif self._click_mode == "x_axis":
            msg = "Click on a point along the positive X-axis"
        else:
            msg = f"Click grid points ({len(self.pixel_points)} points collected). Press 'c' to calibrate, 'q' to quit"

        cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imshow(self._window_name, display)

    def calibrate_interactive(self, image: np.ndarray) -> bool:
        """
        대화형 캘리브레이션 수행

        Args:
            image: 그리드 판이 촬영된 RGB 이미지

        Returns:
            캘리브레이션 성공 여부
        """
        self._current_image = image.copy()
        self._click_mode = "origin"

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

        print("\n" + "="*60)
        print("Grid Calibration")
        print("="*60)
        print("1. Click on the ORIGIN (0,0) point")
        print("2. Click on a point along the positive X-axis")
        print("3. Click additional grid points and enter their world coordinates")
        print("4. Press 'c' to compute calibration when done (min 4 points)")
        print("5. Press 'r' to reset, 'q' to quit")
        print("="*60 + "\n")

        self._pending_click = None
        self._draw_points()

        while True:
            key = cv2.waitKey(50) & 0xFF

            # 클릭이 있으면 메인 스레드에서 처리 (input() 안전)
            if self._pending_click is not None:
                click = self._pending_click
                self._pending_click = None
                self._process_click(click[0], click[1])

            if key == ord('q'):
                print("[Calibration] Cancelled")
                cv2.destroyWindow(self._window_name)
                return False

            elif key == ord('r'):
                # 리셋
                self.pixel_points = []
                self.world_points = []
                self.origin_pixel = None
                self.x_axis_pixel = None
                self._click_mode = "origin"
                self._pending_click = None
                print("[Calibration] Reset")
                self._draw_points()

            elif key == ord('c'):
                if len(self.pixel_points) >= 4:
                    success = self._compute_calibration()
                    if success:
                        cv2.destroyWindow(self._window_name)
                        return True
                else:
                    print(f"[Calibration] Need at least 4 points. Current: {len(self.pixel_points)}")

    def _compute_calibration(self) -> bool:
        """캘리브레이션 계산"""
        print("\n[Calibration] Computing transformation...")

        pixel_pts = np.array(self.pixel_points, dtype=np.float32)
        world_pts = np.array(self.world_points, dtype=np.float32)

        # 2D Homography (평면 가정 - Z는 별도 처리)
        pixel_2d = pixel_pts[:, :2]
        world_2d = world_pts[:, :2]

        self.homography_matrix, status = cv2.findHomography(pixel_2d, world_2d, cv2.RANSAC, 5.0)

        if self.homography_matrix is None:
            print("[Calibration] Failed to compute homography")
            return False

        # Z 오프셋 계산 (평균)
        self.z_offset = np.mean(world_pts[:, 2])

        # 검증: 각 점에 대해 변환 오차 계산
        print("\n[Calibration] Verification:")
        errors = []
        for pixel, world in zip(self.pixel_points, self.world_points):
            predicted = self.pixel_to_world(pixel[0], pixel[1])
            error = np.sqrt((predicted[0] - world[0])**2 + (predicted[1] - world[1])**2)
            errors.append(error)
            print(f"  Pixel {pixel} -> Predicted ({predicted[0]:.2f}, {predicted[1]:.2f}) vs Actual ({world[0]:.2f}, {world[1]:.2f}), Error: {error:.2f} cm")

        mean_error = np.mean(errors)
        print(f"\n[Calibration] Mean error: {mean_error:.3f} cm")

        if mean_error < 1.0:  # 1cm 이내면 성공
            print("[Calibration] SUCCESS!")
            return True
        else:
            print("[Calibration] Warning: High calibration error. Consider adding more points.")
            return True  # 경고만 하고 계속 진행

    def pixel_to_world(self, u: int, v: int, z_camera: float = None) -> Tuple[float, float, float]:
        """
        픽셀 좌표를 월드 좌표로 변환

        Args:
            u: x 픽셀 좌표
            v: y 픽셀 좌표
            z_camera: 카메라에서 측정한 Z값 (optional, 미터 단위)

        Returns:
            (x, y, z) 월드 좌표 (cm)
        """
        if self.homography_matrix is None:
            raise RuntimeError("Calibration not done. Run calibrate_interactive() first.")

        # Homography 변환
        pixel = np.array([[[u, v]]], dtype=np.float32)
        world_2d = cv2.perspectiveTransform(pixel, self.homography_matrix)

        x = float(world_2d[0, 0, 0])
        y = float(world_2d[0, 0, 1])

        # Z 좌표 처리
        if z_camera is not None:
            # 카메라 Z값을 월드 Z값으로 변환
            # 기본적으로 테이블 평면이 Z=0이라고 가정
            z = self.z_offset  # 또는 더 복잡한 변환 가능
        else:
            z = self.z_offset

        return x, y, z

    def save(self, filepath: str) -> None:
        """캘리브레이션 결과 저장"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'homography_matrix': self.homography_matrix.tolist(),
            'z_offset': float(self.z_offset),
            'pixel_points': self.pixel_points,
            'world_points': self.world_points,
            'grid_size_cm': self.grid_size_cm
        }

        # NumPy 형식으로도 저장 (빠른 로딩용)
        np.savez(filepath,
                 homography_matrix=self.homography_matrix,
                 z_offset=self.z_offset,
                 pixel_points=np.array(self.pixel_points),
                 world_points=np.array(self.world_points))

        # JSON으로도 저장 (가독성)
        json_path = filepath.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Calibration] Saved to {filepath} and {json_path}")

    def load(self, filepath: str) -> bool:
        """저장된 캘리브레이션 로드"""
        filepath = Path(filepath)

        if not filepath.exists():
            print(f"[Calibration] File not found: {filepath}")
            return False

        try:
            data = np.load(filepath)
            self.homography_matrix = data['homography_matrix']
            self.z_offset = float(data['z_offset'])
            self.pixel_points = data['pixel_points'].tolist()
            # 하위 호환성: 이전 'robot_points' 키도 지원
            if 'world_points' in data:
                self.world_points = data['world_points'].tolist()
            elif 'robot_points' in data:
                self.world_points = data['robot_points'].tolist()
            print(f"[Calibration] Loaded from {filepath}")
            return True
        except Exception as e:
            print(f"[Calibration] Failed to load: {e}")
            return False

    @property
    def is_calibrated(self) -> bool:
        """캘리브레이션 완료 여부"""
        return self.homography_matrix is not None


class DepthCalibrator(GridCalibrator):
    """깊이 정보를 활용한 3D 캘리브레이션"""

    def __init__(self, grid_size_cm: float = 1.0):
        super().__init__(grid_size_cm)
        self.depth_points: List[float] = []  # 각 점의 깊이값 (미터)

    def calibrate_with_depth(self, image: np.ndarray, depth_image: np.ndarray,
                             camera_intrinsics: dict) -> bool:
        """
        깊이 정보를 포함한 캘리브레이션

        Args:
            image: RGB 이미지
            depth_image: 깊이 이미지 (mm 단위)
            camera_intrinsics: 카메라 내부 파라미터

        Returns:
            성공 여부
        """
        self.depth_image = depth_image
        self.intrinsics = camera_intrinsics

        # 기존 interactive calibration 실행
        success = self.calibrate_interactive(image)

        if success:
            # 각 점의 깊이값 추출
            for pixel in self.pixel_points:
                depth_mm = depth_image[pixel[1], pixel[0]]
                self.depth_points.append(depth_mm * 0.001)  # mm to m

            # 3D 변환 행렬 계산 (카메라 좌표 -> 로봇 좌표)
            self._compute_3d_transform()

        return success

    def _compute_3d_transform(self):
        """카메라 3D -> 월드 3D 변환 행렬 계산"""
        # 카메라 좌표로 변환 (cm 단위로 통일)
        camera_points = []
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']

        for (u, v), depth in zip(self.pixel_points, self.depth_points):
            # depth is in meters, convert to cm for consistency with world_points
            depth_cm = depth * 100.0
            X = (u - cx) * depth_cm / fx
            Y = (v - cy) * depth_cm / fy
            Z = depth_cm
            camera_points.append([X, Y, Z])

        camera_points = np.array(camera_points)  # now in cm
        world_points = np.array(self.world_points)  # already in cm

        # Procrustes 분석으로 변환 행렬 계산
        # (rotation + translation)
        centroid_cam = np.mean(camera_points, axis=0)
        centroid_world = np.mean(world_points, axis=0)

        cam_centered = camera_points - centroid_cam
        world_centered = world_points - centroid_world

        H = cam_centered.T @ world_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # reflection 처리
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_world - R @ centroid_cam

        # 4x4 변환 행렬
        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3, :3] = R
        self.transform_matrix[:3, 3] = t

        print("[Calibration] 3D transform matrix computed")

    def camera_to_world_3d(self, cam_x: float, cam_y: float, cam_z: float) -> Tuple[float, float, float]:
        """카메라 3D 좌표를 월드 좌표로 변환"""
        if self.transform_matrix is None:
            raise RuntimeError("3D calibration not done")

        point = np.array([cam_x, cam_y, cam_z, 1.0])
        world_point = self.transform_matrix @ point

        return float(world_point[0]), float(world_point[1]), float(world_point[2])

    def pixel_depth_to_world(self, u: int, v: int, depth_m: float) -> Tuple[float, float, float]:
        """
        픽셀 좌표 + 깊이 → 월드 3D 좌표

        Args:
            u: x 픽셀 좌표
            v: y 픽셀 좌표
            depth_m: 깊이 (미터 단위)

        Returns:
            (x, y, z) 월드 좌표 (cm 단위)
        """
        if self.transform_matrix is None:
            raise RuntimeError("3D calibration not done. Run calibrate_with_depth() first.")
        if self.intrinsics is None:
            raise RuntimeError("Camera intrinsics not set")

        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']

        # Convert depth to cm for consistency with transform_matrix
        depth_cm = depth_m * 100.0
        cam_x = (u - cx) * depth_cm / fx
        cam_y = (v - cy) * depth_cm / fy
        cam_z = depth_cm

        return self.camera_to_world_3d(cam_x, cam_y, cam_z)

    def save(self, filepath: str) -> None:
        """캘리브레이션 결과 저장 (3D 변환 행렬 포함)"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'homography_matrix': self.homography_matrix if self.homography_matrix is not None else np.eye(3),
            'z_offset': self.z_offset,
            'pixel_points': np.array(self.pixel_points),
            'world_points': np.array(self.world_points),
        }

        if self.transform_matrix is not None:
            save_dict['transform_matrix_3d'] = self.transform_matrix
        if self.depth_points:
            save_dict['depth_points'] = np.array(self.depth_points)
        if self.intrinsics is not None:
            save_dict['intrinsics'] = np.array([
                self.intrinsics['fx'], self.intrinsics['fy'],
                self.intrinsics['cx'], self.intrinsics['cy']
            ])

        np.savez(path, **save_dict)

        json_data = {
            'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            'z_offset': float(self.z_offset),
            'pixel_points': self.pixel_points,
            'world_points': self.world_points,
            'grid_size_cm': self.grid_size_cm,
        }
        if self.transform_matrix is not None:
            json_data['transform_matrix_3d'] = self.transform_matrix.tolist()
        if self.depth_points:
            json_data['depth_points'] = self.depth_points
        if self.intrinsics is not None:
            json_data['intrinsics'] = self.intrinsics

        json_path = path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"[Calibration] Saved to {path} and {json_path}")
        if self.transform_matrix is not None:
            print("[Calibration] 3D transform matrix included")

    def load(self, filepath: str) -> bool:
        """저장된 캘리브레이션 로드 (3D 변환 행렬 포함)"""
        path = Path(filepath)

        if not path.exists():
            print(f"[Calibration] File not found: {path}")
            return False

        try:
            data = np.load(path, allow_pickle=True)

            self.homography_matrix = data['homography_matrix']
            self.z_offset = float(data['z_offset'])
            self.pixel_points = data['pixel_points'].tolist()

            if 'world_points' in data:
                self.world_points = data['world_points'].tolist()
            elif 'robot_points' in data:
                self.world_points = data['robot_points'].tolist()

            if 'transform_matrix_3d' in data:
                self.transform_matrix = data['transform_matrix_3d']
                print("[Calibration] 3D transform matrix loaded")

            if 'depth_points' in data:
                self.depth_points = data['depth_points'].tolist()

            if 'intrinsics' in data:
                intrinsics_arr = data['intrinsics']
                self.intrinsics = {
                    'fx': float(intrinsics_arr[0]),
                    'fy': float(intrinsics_arr[1]),
                    'cx': float(intrinsics_arr[2]),
                    'cy': float(intrinsics_arr[3])
                }

            print(f"[Calibration] Loaded from {path}")
            return True

        except Exception as e:
            print(f"[Calibration] Failed to load: {e}")
            return False

    @property
    def is_3d_calibrated(self) -> bool:
        """3D 캘리브레이션 완료 여부"""
        return self.transform_matrix is not None


# 테스트
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from camera.realsense import RealSenseD435

    print("Grid Calibration Test")
    print("Press 's' to capture image for calibration")

    with RealSenseD435() as camera:
        while True:
            color, depth = camera.get_frames()
            if color is None:
                continue

            cv2.imshow("Camera", color)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                cv2.destroyWindow("Camera")

                calibrator = GridCalibrator(grid_size_cm=1.0)
                if calibrator.calibrate_interactive(color):
                    calibrator.save("robot_configs/pix2world_matrices/pix2world_transform_data.npz")
                    print("\nCalibration complete!")

                    # 테스트
                    test_u, test_v = 320, 240
                    result = calibrator.pixel_to_world(test_u, test_v)
                    print(f"Test: Pixel ({test_u}, {test_v}) -> World {result}")
                break

            elif key == ord('q'):
                break

    cv2.destroyAllWindows()
