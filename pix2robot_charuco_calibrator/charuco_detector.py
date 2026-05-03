"""
Charuco 검출 모듈 — 검출 + 자동 사전 탐지 + 시각화

핵심 기능:
    auto_detect_dictionary: 보드 이미지에서 ArUco 사전 자동 추정
    CharucoBoardSpec:       보드 사양 데이터 클래스
    CharucoDetector:        검출 + 픽셀→3D 변환 통합
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from .geometry import pixel_to_camera_3d


# OpenCV 표준 사전 — DICT_NxN_M (NxN 비트, M개 마커)
STANDARD_DICTIONARIES: Dict[str, int] = {
    "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":  cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50":   cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":  cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":  cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50":   cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":  cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250":  cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50":   cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100":  cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250":  cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


@dataclass
class CharucoBoardSpec:
    """보드 사양."""
    dictionary: str        # 예: "DICT_5X5_100"
    squares_x: int         # 가로 사각형 개수
    squares_y: int         # 세로 사각형 개수
    square_length_m: float # 사각형 한 변 (m)
    marker_length_m: float # 사각형 안 ArUco 한 변 (m)
    thickness_m: float = 0.0  # 보드 두께 (m). EE가 보드 상면을 터치하므로 affine fit 시 robot z에서 차감.

    def aruco_dict(self) -> cv2.aruco.Dictionary:
        if self.dictionary not in STANDARD_DICTIONARIES:
            raise ValueError(
                f"알 수 없는 dictionary: {self.dictionary}\n"
                f"사용 가능: {list(STANDARD_DICTIONARIES.keys())}"
            )
        return cv2.aruco.getPredefinedDictionary(STANDARD_DICTIONARIES[self.dictionary])

    def board(self) -> cv2.aruco.CharucoBoard:
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            self.aruco_dict(),
        )

    def to_dict(self) -> dict:
        return {
            "dictionary": self.dictionary,
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "thickness_m": self.thickness_m,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CharucoBoardSpec":
        return cls(
            dictionary=d["dictionary"],
            squares_x=int(d["squares_x"]),
            squares_y=int(d["squares_y"]),
            square_length_m=float(d["square_length_m"]),
            marker_length_m=float(d["marker_length_m"]),
            thickness_m=float(d.get("thickness_m", 0.0)),
        )


def auto_detect_dictionary(
    image: np.ndarray,
    min_markers: int = 4,
) -> List[Tuple[str, int, List[int]]]:
    """
    이미지에서 어느 ArUco 사전이 사용됐는지 자동 탐지.

    모든 표준 사전을 순회하며 검출 시도, 검출되는 사전들을 반환.

    중요 — ArUco 사전 부분집합 관계:
        DICT_NxN_50 ⊂ DICT_NxN_100 ⊂ DICT_NxN_250 ⊂ DICT_NxN_1000
        (같은 비트 크기끼리 ID 0~49는 동일, 0~99도 동일 ...)
        → 마커가 모두 ID < 50 이면 50/100/250/1000이 모두 검출됨.

    따라서 같은 검출 수를 보이는 사전들 중에선 가장 작은 것이 안전한 default.
    main에서는 후보가 여러 개일 때 사용자에게 확인 요청 필요.

    Returns:
        list of (dict_name, num_detected, ids) — 검출 수 내림차순,
        같은 검출 수면 가장 작은 사전(50)이 먼저.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 사전 크기 추출용 (정렬 보조)
    def _dict_size(name: str) -> int:
        # "DICT_5X5_100" → 100, "DICT_ARUCO_ORIGINAL" → 9999
        try:
            return int(name.rsplit("_", 1)[-1])
        except ValueError:
            return 9999

    results = []
    for name, dict_id in STANDARD_DICTIONARIES.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) >= min_markers:
            results.append((name, int(len(ids)), ids.flatten().tolist()))

    # 1차: 검출 수 내림차순
    # 2차: 같은 검출 수면 사전 크기 오름차순 (작은 사전 우선)
    results.sort(key=lambda x: (-x[1], _dict_size(x[0])))
    return results


def group_compatible_dictionaries(
    detection_results: List[Tuple[str, int, List[int]]],
) -> List[List[Tuple[str, int, List[int]]]]:
    """
    부분집합 관계로 호환되는 사전들을 그룹핑.

    같은 비트 크기 (예: 5X5)이고 같은 검출 수면 같은 보드에서 호환.

    Returns:
        그룹 리스트. 각 그룹은 (사전명, 검출수, IDs) 튜플 리스트.
        그룹의 첫 원소가 가장 좁은(권장) 사전.
    """
    groups: Dict[Tuple[str, int], List[Tuple[str, int, List[int]]]] = {}
    for name, count, ids in detection_results:
        # 비트 크기 추출 ("DICT_5X5_100" → "5X5")
        parts = name.split("_")
        if len(parts) >= 3 and "X" in parts[1]:
            bits = parts[1]
            key = (bits, count)
            groups.setdefault(key, []).append((name, count, ids))
        else:
            # ARUCO_ORIGINAL 등은 단독
            groups.setdefault((name, count), []).append((name, count, ids))
    return list(groups.values())


class Detection:
    """단일 Charuco 코너 검출 결과."""

    def __init__(
        self,
        corner_id: int,
        pixel_uv: Tuple[float, float],
        depth_m: Optional[float],
        camera_xyz: Optional[np.ndarray],
    ):
        self.corner_id = corner_id
        self.pixel_uv = pixel_uv
        self.depth_m = depth_m
        self.camera_xyz = camera_xyz  # None이면 depth 무효

    @property
    def has_3d(self) -> bool:
        return self.camera_xyz is not None

    def __repr__(self) -> str:
        if self.has_3d:
            xyz = self.camera_xyz
            return (
                f"Detection(id={self.corner_id}, uv=({self.pixel_uv[0]:.1f},"
                f"{self.pixel_uv[1]:.1f}), xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}))"
            )
        return (
            f"Detection(id={self.corner_id}, uv=({self.pixel_uv[0]:.1f},"
            f"{self.pixel_uv[1]:.1f}), no depth)"
        )


class CharucoDetector:
    """Charuco 검출 + 픽셀 → 카메라 좌표계 3D 변환 통합."""

    def __init__(
        self,
        board_spec: CharucoBoardSpec,
        K: Optional[np.ndarray] = None,
        dist: Optional[np.ndarray] = None,
    ):
        self.spec = board_spec
        self.K = K
        self.dist = dist
        self._aruco_dict = board_spec.aruco_dict()
        self._board = board_spec.board()
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._charuco_params = cv2.aruco.CharucoParameters()
        self._charuco_detector = cv2.aruco.CharucoDetector(
            self._board, self._charuco_params, self._aruco_params
        )

    def update_intrinsics(self, K: np.ndarray, dist: np.ndarray):
        """Intrinsics 페이즈 후 K, dist 주입."""
        self.K = K
        self.dist = dist

    def detect_aruco(self, image: np.ndarray):
        """ArUco 마커만 검출 (intrinsics 페이즈에서 사용)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected

    def detect_charuco(self, image: np.ndarray) -> List[Detection]:
        """
        Charuco 코너 검출 + PnP로 각 코너의 3D 점 계산.

        검출된 모든 코너로 보드 자세를 PnP로 한 번에 풀고,
        각 코너의 3D는 보드 좌표계 위치를 카메라 좌표계로 변환해서 도출.
        Depth 센서 안 씀 → D435 노이즈 영향 0.

        Args:
            image: BGR 또는 grayscale

        Returns:
            Detection 리스트 (코너 ID 오름차순)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        charuco_corners, charuco_ids, _, _ = self._charuco_detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) == 0:
            return []

        detections: List[Detection] = []
        if self.K is None:
            # K 없으면 PnP 불가 — 픽셀만 채워서 반환
            for k in range(len(charuco_ids)):
                cid = int(charuco_ids[k][0])
                uv = charuco_corners[k][0]
                detections.append(
                    Detection(cid, (float(uv[0]), float(uv[1])), None, None)
                )
            detections.sort(key=lambda d: d.corner_id)
            return detections

        # 코너별 보드 좌표계 위치 (보드 격자 기하로 알려진 사실)
        all_object_points = self._board.getChessboardCorners()

        image_pts = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
        obj_pts_subset = np.asarray(
            [all_object_points[int(cid)] for cid in charuco_ids.flatten()],
            dtype=np.float32,
        )

        # solvePnP DLT requires >= 6 points; 그 미만이면 fallback 또는 빈 리스트
        if len(obj_pts_subset) < 6:
            return []

        dist = self.dist if self.dist is not None else np.zeros(5, dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts_subset, image_pts, self.K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return []  # PnP 실패 시 안전하게 빈 결과
        if not ok:
            return []

        R_bc, _ = cv2.Rodrigues(rvec)
        t_bc = tvec.reshape(3)

        for k in range(len(charuco_ids)):
            cid = int(charuco_ids[k][0])
            uv = charuco_corners[k][0]
            u, v = float(uv[0]), float(uv[1])

            board_pos = all_object_points[cid].astype(np.float64)
            cam_xyz = R_bc @ board_pos + t_bc
            depth_m = float(cam_xyz[2]) if cam_xyz[2] > 0 else None
            detections.append(Detection(cid, (u, v), depth_m, cam_xyz))

        detections.sort(key=lambda d: d.corner_id)
        return detections

    def annotate(
        self,
        image: np.ndarray,
        detections: List[Detection],
        highlight_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        검출 결과를 이미지에 오버레이.
            - has_3d (depth 유효): 시안 십자 + ID
            - depth 무효: 회색 십자 + ID
            - highlight_id: 빨간 원 강조
        """
        out = image.copy()
        for det in detections:
            u, v = int(round(det.pixel_uv[0])), int(round(det.pixel_uv[1]))
            color = (0, 255, 255) if det.has_3d else (128, 128, 128)
            if highlight_id is not None and det.corner_id == highlight_id:
                color = (0, 0, 255)
                cv2.circle(out, (u, v), 14, color, 2)
            # 십자 마커 (더 크고 두껍게)
            cv2.drawMarker(out, (u, v), color, cv2.MARKER_CROSS, 18, 2)
            # ID 라벨 — 검정 외곽 + 컬러 채움 (배경에 무관하게 잘 보이도록)
            label = str(det.corner_id)
            org = (u + 8, v - 8)
            cv2.putText(out, label, org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return out

    def annotate_aruco(
        self,
        image: np.ndarray,
        corners,
        ids,
    ) -> np.ndarray:
        """ArUco 마커를 외곽선과 ID로 표시 (intrinsics 페이즈용)."""
        out = image.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
        return out
