"""
Phase 3: Extrinsics Calibration — 카메라 → 로봇 변환행렬

워크플로:
    1. 보드를 워크스페이스에 놓음
    2. 카메라 1프레임 캡처 → Charuco 자동 검출 (모든 코너의 매칭점_camera 자동 계산)
    3. 사용자가 마우스로 코너 클릭 → 토크 OFF → EE를 그 코너에 터치 → Enter
       → 매칭점_robot (FK) 기록
    4. 보드 위치 변경 후 재캡처 → 반복 (총 10~15 매칭점)
    5. affine_align_3d → 카메라→로봇 변환행렬 (12 DoF, FK 비등방 오차 흡수) + 잔차
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
from .geometry import (
    affine_align_3d,
    compose_transform_4x4,
    apply_transform,
)
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


class MatchedPair:
    """단일 매칭점 — (매칭점_camera, 매칭점_robot, 메타)."""

    def __init__(
        self,
        camera_xyz: np.ndarray,
        robot_xyz: np.ndarray,
        pixel_uv: Tuple[float, float],
        corner_id: int,
        board_session: int,
    ):
        self.camera_xyz = np.asarray(camera_xyz, dtype=np.float64)
        self.robot_xyz = np.asarray(robot_xyz, dtype=np.float64)
        self.pixel_uv = pixel_uv
        self.corner_id = corner_id
        self.board_session = board_session

    def to_dict(self) -> dict:
        return {
            "camera_xyz": self.camera_xyz.tolist(),
            "robot_xyz": self.robot_xyz.tolist(),
            "pixel_uv": list(self.pixel_uv),
            "corner_id": self.corner_id,
            "board_session": self.board_session,
        }


class ExtrinsicsCalibrator:
    """Phase 3 메인."""

    def __init__(
        self,
        robot_id: int,
        board_spec: CharucoBoardSpec,
        K: np.ndarray,
        dist: np.ndarray,
        target_num_pairs: int = 12,
        min_num_pairs: int = 6,
        target_rmse_mm: float = 3.0,
        camera_serial: Optional[str] = None,
    ):
        self.robot_id = robot_id
        self.spec = board_spec
        self.K = K
        self.dist = dist
        self.target_num_pairs = target_num_pairs
        self.min_num_pairs = min_num_pairs
        self.target_rmse_mm = target_rmse_mm
        self.camera_serial = camera_serial

        self.detector = CharucoDetector(board_spec, K=K, dist=dist)
        self.pairs: List[MatchedPair] = []
        self.current_session: int = 0

        # 현재 보드 캡처 상태
        self._current_color: Optional[np.ndarray] = None
        self._current_depth: Optional[np.ndarray] = None
        self._current_detections: List[Detection] = []

        # UI
        self._window_name = "Phase 3 - Extrinsics"
        self._pending_click: Optional[Tuple[int, int]] = None

        # 결과
        self.R: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None
        self.cam_to_base: Optional[np.ndarray] = None
        self.stats: Optional[dict] = None

        # 로봇 핸들
        self._controller = None
        self._kinematics = None
        self._calibration_limits = None

    def run(self, save_path: Path) -> bool:
        print_phase_banner(
            3,
            subtitle=(
                "보드 위치를 2~3곳으로 옮기며 매칭점 수집\n"
                f"  목표: {self.target_num_pairs} 쌍 (최소 {self.min_num_pairs} 쌍)"
            ),
        )

        try:
            self._controller, self._kinematics, self._calibration_limits = \
                load_robot(self.robot_id)
        except Exception as e:
            print(f"  로봇 연결 실패: {e}")
            print_phase_summary(3, success=False, details=str(e))
            return False

        ok = False
        try:
            with CalibrationCamera(serial=self.camera_serial) as camera:
                if not self._capture_initial_board(camera):
                    return False

                ok = self._main_loop(camera)
        finally:
            cleanup_robot(self._controller)
            cv2.destroyAllWindows()

        if ok and self.cam_to_base is not None:
            self._save(save_path)
            return True
        return False

    # ── 보드 캡처 ───────────────────────────────────────────────

    def _capture_initial_board(self, camera) -> bool:
        """첫 보드 위치 캡처."""
        print()
        print("  ──── 보드 1 위치 캡처 ────")
        print("  보드를 워크스페이스에 평평하게 놓고, 카메라 미리보기 창에서")
        print("  검출된 코너가 충분히 보이는지 확인 후 's' 키로 캡처.")
        return self._capture_board(camera, increment_session=False)

    def _capture_board(self, camera, increment_session: bool = True) -> bool:
        """현재 카메라 프레임 → 코너 검출. 's' 누르면 확정."""
        window_preview = "Phase 3 - Capture Board"
        cv2.namedWindow(window_preview)

        while True:
            color, depth = camera.get_frames()
            if color is None:
                continue

            detections = self.detector.detect_charuco(color)
            valid = [d for d in detections if d.has_3d]

            vis = color.copy()
            vis = self.detector.annotate(vis, detections)

            if len(valid) == 0:
                vis = draw_status_text(
                    vis, ["No board detected (or depth invalid)"],
                    origin=(10, 30), color=(0, 0, 255),
                )
            else:
                vis = draw_status_text(
                    vis,
                    [
                        f"Board detected: {len(valid)} valid corners (with 3D).",
                        "Press [s] to LOCK this capture and proceed to corner selection.",
                    ],
                    origin=(10, 30),
                )

            progress = (
                f"Board pos {self.current_session + 1}  "
                f"detected: {len(detections)} (valid 3D: {len(valid)})  "
                f"pairs: {len(self.pairs)}/{self.target_num_pairs}"
            )
            vis = draw_phase_overlay(
                vis, phase=3,
                progress_text=progress,
                keys_text="[s] confirm capture  [q] quit",
            )

            cv2.imshow(window_preview, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if len(valid) >= 4:
                    self._current_color = color.copy()
                    self._current_depth = depth.copy() if depth is not None else None
                    self._current_detections = detections
                    if increment_session:
                        self.current_session += 1
                    cv2.destroyWindow(window_preview)
                    print(
                        f"  보드 위치 {self.current_session + 1} 캡처: "
                        f"{len(valid)} 코너 (3D 유효)"
                    )
                    return True
                else:
                    print(f"  유효 코너 부족 ({len(valid)} < 4). 보드 자세 조정.")
            elif key == ord('q'):
                cv2.destroyWindow(window_preview)
                return False

    # ── 메인 인터랙션 ───────────────────────────────────────────

    def _main_loop(self, camera) -> bool:
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_cb)

        print()
        print("  ──── 매칭점 수집 시작 ────")
        print("  워크플로:")
        print("    1) 화면의 시안색 십자(✚) + 숫자가 검출된 Charuco 코너입니다.")
        print("       원하는 코너를 마우스로 클릭하세요.")
        print("    2) 토크가 자동으로 OFF 됩니다.")
        print("       EE 팁을 클릭한 코너의 물리 위치에 손으로 정확히 닿게 이동.")
        print("    3) 이 터미널에서 Enter → 매칭점 기록, 토크 자동 ON.")
        print("    4) 다음 코너 클릭 → 반복.")
        print("    5) 한 보드 위치에서 5~7개 모은 후 'r' → 보드를 옮기고 다시 캡처.")
        print()
        print("  키: [click] 코너 선택  [r] 보드 새 위치 캡처  "
              "[u] 마지막 매칭점 취소  [c] 변환행렬 계산  [s] 저장+종료  [q] 종료")
        print()

        self._pending_click = None

        while True:
            vis = self._render_main_frame()
            cv2.imshow(self._window_name, vis)
            key = cv2.waitKey(50) & 0xFF

            if self._pending_click is not None:
                cx, cy = self._pending_click
                self._pending_click = None
                self._handle_corner_click(cx, cy)

            if key == ord('q'):
                print("  사용자 종료")
                print_phase_summary(3, success=False, details="사용자 중단")
                return False

            elif key == ord('u'):
                self._undo_last_pair()

            elif key == ord('r'):
                cv2.destroyWindow(self._window_name)
                if not self._capture_board(camera, increment_session=True):
                    return False
                cv2.namedWindow(self._window_name)
                cv2.setMouseCallback(self._window_name, self._mouse_cb)

            elif key == ord('c'):
                if len(self.pairs) >= self.min_num_pairs:
                    self._compute_transform()
                else:
                    print(
                        f"  최소 {self.min_num_pairs} 쌍 필요 "
                        f"(현재 {len(self.pairs)})"
                    )

            elif key == ord('s'):
                if self.cam_to_base is None:
                    if len(self.pairs) >= self.min_num_pairs:
                        self._compute_transform()
                    else:
                        print(
                            f"  변환행렬 미계산. 최소 {self.min_num_pairs} 쌍 필요."
                        )
                        continue
                if self.cam_to_base is not None:
                    return True

    def _render_main_frame(self) -> np.ndarray:
        """현재 캡처된 보드 + 검출 + 이미 수집된 매칭점 시각화."""
        if self._current_color is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        vis = self._current_color.copy()
        vis = self.detector.annotate(vis, self._current_detections)

        # 이미 수집된 매칭점은 다른 색으로 표시 (현재 보드 위치 한정)
        used_ids = {
            p.corner_id for p in self.pairs
            if p.board_session == self.current_session
        }
        for det in self._current_detections:
            if det.corner_id in used_ids:
                u, v = int(round(det.pixel_uv[0])), int(round(det.pixel_uv[1]))
                cv2.circle(vis, (u, v), 8, (0, 255, 0), 2)

        # 상태 메시지를 이미지에 먼저 그림 (헤더 추가 전)
        valid_unused = [
            d for d in self._current_detections
            if d.has_3d and d.corner_id not in used_ids
        ]
        msgs = [
            "STEP: Click any cyan cross (corner) to select it for EE touch.",
            f"Available corners (this pos): {len(valid_unused)}    "
            f"Done (green): {len(used_ids)}    "
            f"Total pairs: {len(self.pairs)}/{self.target_num_pairs}",
        ]
        vis = draw_status_text(vis, msgs, origin=(10, 30))

        rmse_str = (
            f"{self.stats['rmse_m']*1000:.2f} mm"
            if self.stats else "(not computed)"
        )
        progress = (
            f"session {self.current_session + 1} | "
            f"pairs {len(self.pairs)}/{self.target_num_pairs} | "
            f"RMSE: {rmse_str}"
        )
        vis = draw_phase_overlay(
            vis, phase=3,
            progress_text=progress,
            keys_text="[click] select corner  [r] recapture board  "
                      "[u] undo  [c] compute  [s] save+exit  [q] quit",
        )
        return vis

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 헤더 패딩 보정 → 원본 이미지 좌표
            self._pending_click = (x, y - PHASE_HEADER_HEIGHT)

    def _handle_corner_click(self, cx: int, cy: int):
        """가장 가까운 검출 코너로 스냅 → EE 터치 시퀀스."""
        if not self._current_detections:
            print("  검출된 코너 없음")
            return

        used_ids = {
            p.corner_id for p in self.pairs
            if p.board_session == self.current_session
        }

        candidates = [
            d for d in self._current_detections
            if d.has_3d and d.corner_id not in used_ids
        ]
        if not candidates:
            print("  사용 가능한 코너 없음 (모두 사용됨 또는 depth 무효)")
            return

        # 클릭 위치에 가장 가까운 코너
        dists = [
            (np.hypot(d.pixel_uv[0] - cx, d.pixel_uv[1] - cy), d)
            for d in candidates
        ]
        dists.sort(key=lambda x: x[0])
        nearest = dists[0][1]

        if dists[0][0] > 30:
            print(f"  클릭 위치에서 가까운 코너 없음 (최단거리 {dists[0][0]:.1f}px)")
            return

        self._touch_corner_sequence(nearest)

    def _touch_corner_sequence(self, det: Detection):
        """선택된 코너에 EE 터치 → 매칭점 등록."""
        print()
        print(f"  >> 코너 ID {det.corner_id} 선택")
        print(f"     픽셀: ({det.pixel_uv[0]:.1f}, {det.pixel_uv[1]:.1f})")
        print(f"     매칭점_camera: ({det.camera_xyz[0]:.4f}, "
              f"{det.camera_xyz[1]:.4f}, {det.camera_xyz[2]:.4f}) m")
        print()
        print("  [EE 포지셔닝]")
        print("  토크가 비활성화됩니다. EE 팁을 이 코너에 정확히 닿도록")
        print("  수동 이동한 후 Enter를 누르세요. (q+Enter: 취소)")

        cv2.destroyWindow(self._window_name)
        cv2.waitKey(1)

        self._controller.disable_torque()
        print("  >> 토크 OFF")

        try:
            cancelled = False
            while True:
                tcp = get_tcp_position(
                    self._controller, self._kinematics, self._calibration_limits,
                )
                sys.stdout.write(
                    f"\r  TCP: x={tcp[0]:+.4f} y={tcp[1]:+.4f} z={tcp[2]:+.4f}    "
                )
                sys.stdout.flush()

                if select.select([sys.stdin], [], [], 0.1)[0]:
                    raw = sys.stdin.readline().strip()
                    if raw.lower() == 'q':
                        print("\n  취소")
                        cancelled = True
                        break
                    break
                time.sleep(0.05)

            tcp_final = get_tcp_position(
                self._controller, self._kinematics, self._calibration_limits,
            )
        finally:
            self._controller.enable_torque()
            print("  >> 토크 ON")

        # 윈도우 복원
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_cb)

        if cancelled:
            return

        pair = MatchedPair(
            camera_xyz=det.camera_xyz,
            robot_xyz=tcp_final,
            pixel_uv=det.pixel_uv,
            corner_id=det.corner_id,
            board_session=self.current_session,
        )
        self.pairs.append(pair)
        print(f"\n  매칭점 #{len(self.pairs)} 저장:")
        print(f"    매칭점_camera: ({pair.camera_xyz[0]:.4f}, "
              f"{pair.camera_xyz[1]:.4f}, {pair.camera_xyz[2]:.4f})")
        print(f"    매칭점_robot:  ({pair.robot_xyz[0]:.4f}, "
              f"{pair.robot_xyz[1]:.4f}, {pair.robot_xyz[2]:.4f})")
        if len(self.pairs) >= self.min_num_pairs and self.cam_to_base is None:
            print(f"  -> {len(self.pairs)} 쌍 모임. 'c'로 변환행렬 계산 가능.")

    def _undo_last_pair(self):
        if not self.pairs:
            print("  취소할 매칭점 없음")
            return
        p = self.pairs.pop()
        print(
            f"  취소: 매칭점 (코너 ID {p.corner_id}, 보드세션 {p.board_session})"
        )

    # ── 변환행렬 계산 ─────────────────────────────────────────

    def _compute_transform(self):
        n = len(self.pairs)
        cam = np.array([p.camera_xyz for p in self.pairs])
        rob = np.array([p.robot_xyz for p in self.pairs])

        try:
            M, t, stats = affine_align_3d(cam, rob)
        except ValueError as e:
            print(f"  계산 실패: {e}")
            return

        self.R = M  # affine 선형부 (회전 + 비등방 스케일 + 전단)
        self.t = t
        self.cam_to_base = compose_transform_4x4(M, t)
        self.stats = stats

        print()
        print(f"  Affine 3D 정렬 결과 ({n} 쌍, 12 DoF):")
        print(f"    RMSE:    {stats['rmse_m']*1000:.3f} mm")
        print(f"    Max:     {stats['max_error_m']*1000:.3f} mm")
        print(f"    Mean:    {stats['mean_error_m']*1000:.3f} mm")
        print()
        print(f"    M (3x3 affine 선형) = ")
        for row in M:
            print(f"      [{row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f}]")
        print(f"    t = [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m")
        print()

        if stats['rmse_m'] * 1000 <= self.target_rmse_mm:
            print(f"  통과 (목표 {self.target_rmse_mm} mm 이하)")
        else:
            print(
                f"  주의: 목표 {self.target_rmse_mm} mm 초과. "
                "잔차 큰 매칭점은 'u'로 제거 후 재계산 권장."
            )

        # 매칭점별 잔차 표시 — outlier 후보 식별
        print(f"  매칭점별 잔차:")
        for i, (p, e) in enumerate(zip(self.pairs, stats['per_point_errors_m'])):
            mark = "  ⚠" if e > stats['rmse_m'] * 2 else "   "
            print(
                f"  {mark} #{i+1:2d} (id={p.corner_id:3d}, ses={p.board_session}): "
                f"{e*1000:.2f} mm"
            )

    # ── 저장 ───────────────────────────────────────────────────

    def _save(self, save_path: Path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "cam_to_base": self.cam_to_base,
            "R": self.R,
            "t": self.t,
            "K": self.K,
            "dist": self.dist,
            "rmse_m": np.array([self.stats['rmse_m']]),
            "max_error_m": np.array([self.stats['max_error_m']]),
            "mean_error_m": np.array([self.stats['mean_error_m']]),
            "per_point_errors_m": np.array(self.stats['per_point_errors_m']),
            "camera_points": np.array([p.camera_xyz for p in self.pairs]),
            "robot_points": np.array([p.robot_xyz for p in self.pairs]),
            "pixel_points": np.array([p.pixel_uv for p in self.pairs]),
            "corner_ids": np.array([p.corner_id for p in self.pairs]),
            "board_sessions": np.array([p.board_session for p in self.pairs]),
        }
        np.savez(save_path, **save_dict)

        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                "robot_id": f"robot{self.robot_id}",
                "alignment_method": "affine_3d_12dof",
                "cam_to_base_4x4": self.cam_to_base.tolist(),
                "M_3x3_affine_linear": self.R.tolist(),
                "t_3": self.t.tolist(),
                "rmse_mm": self.stats['rmse_m'] * 1000,
                "max_error_mm": self.stats['max_error_m'] * 1000,
                "mean_error_mm": self.stats['mean_error_m'] * 1000,
                "num_pairs": len(self.pairs),
                "num_board_sessions": self.current_session + 1,
                "pairs": [p.to_dict() for p in self.pairs],
                "board": self.spec.to_dict(),
            }, f, indent=2)

        print(f"  저장:")
        print(f"    {save_path}")
        print(f"    {json_path}")
        print_phase_summary(
            3, success=True,
            details=f"RMSE={self.stats['rmse_m']*1000:.2f} mm, "
                    f"{len(self.pairs)} 쌍, 보드 {self.current_session+1}곳",
        )


def load_extrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """저장된 cam_to_base, R, t, K, dist 로드."""
    data = np.load(path)
    return data['cam_to_base'], data['R'], data['t'], data['K'], data['dist']
