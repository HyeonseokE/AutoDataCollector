"""
Charuco 검출 진단 스크립트.

사용법:
    python -m pix2robot_charuco_calibrator.diagnose --camera-serial 254622079503

수행 작업:
    1. 카메라 1프레임 캡처
    2. 모든 표준 사전으로 ArUco 마커 검출 시도
    3. 검출되는 사전과 IDs 표시
    4. squares_x × squares_y 양쪽 방향 모두 Charuco 검출 테스트
    5. 결과 이미지를 disk에 저장하여 사람이 시각 확인 가능
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from .camera_setup import CalibrationCamera
from .charuco_detector import (
    STANDARD_DICTIONARIES,
    auto_detect_dictionary,
    group_compatible_dictionaries,
    CharucoBoardSpec,
)


def capture_one_frame(serial: str = None, auto: bool = False, warmup: int = 30):
    """카메라 캡처. auto=True면 warmup 후 자동 캡처."""
    if auto:
        print("  자동 캡처 모드 — 워밍업 후 한 프레임 잡기")
        with CalibrationCamera(serial=serial) as cam:
            for _ in range(warmup):
                cam.get_frames()
            color, depth = cam.get_frames()
            return (color, depth) if color is not None else None

    print("  카메라 미리보기 — 's' 키 캡처 / 'q' 종료")
    win = "Diagnose Capture"
    cv2.namedWindow(win)
    captured = None
    with CalibrationCamera(serial=serial) as cam:
        for _ in range(300):
            color, depth = cam.get_frames()
            if color is None:
                continue
            cv2.putText(
                color, "Press 's' to capture",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            cv2.imshow(win, color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                captured = (color.copy(), depth.copy() if depth is not None else None)
                break
            elif key == ord('q'):
                break
    cv2.destroyWindow(win)
    return captured


def test_aruco_all_dictionaries(image: np.ndarray) -> list:
    """모든 사전으로 ArUco 검출 시도."""
    print("\n" + "=" * 60)
    print("  ArUco 검출 — 17개 표준 사전 모두 시도")
    print("=" * 60)
    results = auto_detect_dictionary(image, min_markers=1)
    if not results:
        print("  어떤 사전으로도 ArUco 마커 검출 실패")
        print("  → 이미지 품질 / 보드 가시성 / 조명 문제일 가능성 높음")
        return []
    for name, count, ids in results[:8]:
        ids_sorted = sorted(set(ids))
        ids_str = str(ids_sorted[:10]) + ("..." if len(ids_sorted) > 10 else "")
        print(f"  {name:25s}  {count:3d} markers, IDs={ids_str}")
    if len(results) > 8:
        print(f"  ... ({len(results) - 8} 사전 더 검출됨)")
    return results


def test_charuco_both_orientations(
    image: np.ndarray,
    dictionary_name: str,
    squares_a: int,
    squares_b: int,
    square_length_m: float,
    marker_length_m: float,
):
    """squares (a,b)와 (b,a) 양쪽 모두 Charuco 검출 테스트."""
    print("\n" + "=" * 60)
    print("  Charuco 검출 — squares 두 방향 모두 시도")
    print("=" * 60)

    for sx, sy in [(squares_a, squares_b), (squares_b, squares_a)]:
        spec = CharucoBoardSpec(
            dictionary=dictionary_name,
            squares_x=sx, squares_y=sy,
            square_length_m=square_length_m,
            marker_length_m=marker_length_m,
        )
        board = spec.board()
        params = cv2.aruco.CharucoParameters()
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.CharucoDetector(board, params, aruco_params)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)

        n_corners = 0 if ids is None else len(ids)
        n_markers = 0 if marker_ids is None else len(marker_ids)
        max_corners = (sx - 1) * (sy - 1)

        print(f"\n  squares_x={sx}, squares_y={sy}:")
        print(f"    ArUco 마커:    {n_markers} 개")
        print(f"    Charuco 코너:  {n_corners} / 이론 최대 {max_corners}")
        if n_corners > 0:
            ids_list = sorted(int(x[0]) for x in ids)
            print(f"    검출 코너 IDs: {ids_list[:15]}{'...' if len(ids_list)>15 else ''}")


def annotate_and_save(
    image: np.ndarray,
    dictionary_name: str,
    save_dir: Path,
):
    """검출 결과를 이미지에 그려 저장."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # 원본 저장
    cv2.imwrite(str(save_dir / "00_raw.jpg"), image)

    # ArUco 전용 시각화 (지정 사전)
    if dictionary_name in STANDARD_DICTIONARIES:
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            STANDARD_DICTIONARIES[dictionary_name]
        )
        params = cv2.aruco.DetectorParameters()
        det = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = det.detectMarkers(image)
        vis = image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.imwrite(str(save_dir / f"01_aruco_{dictionary_name}.jpg"), vis)

    print(f"\n  저장됨: {save_dir}")
    print(f"    00_raw.jpg                  ← 원본")
    print(f"    01_aruco_<dict>.jpg         ← 마커 검출 시각화")


def main():
    parser = argparse.ArgumentParser(description="Charuco 검출 진단")
    parser.add_argument(
        "--camera-serial", default=None,
        help="카메라 일련번호 (생략 시 yaml 또는 자동)",
    )
    parser.add_argument(
        "--board-config",
        default=str(Path(__file__).parent / "board_config.yaml"),
    )
    parser.add_argument(
        "--save-dir",
        default="/tmp/charuco_diagnose",
        help="시각화 결과 저장 디렉토리",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="대화식 's' 키 없이 자동 캡처 (헤드리스 진단용)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.board_config))
    board_cfg = cfg["board"]
    serial = args.camera_serial or (cfg.get("camera") or {}).get("serial")

    print(f"보드 yaml: {args.board_config}")
    print(f"  dictionary       = {board_cfg['dictionary']}")
    print(f"  squares          = {board_cfg['squares_x']} x {board_cfg['squares_y']}")
    print(f"  square_length_m  = {board_cfg['square_length_m']}")
    print(f"  marker_length_m  = {board_cfg['marker_length_m']}")
    print(f"  camera serial    = {serial}")

    print("\n[1/3] 카메라 캡처...")
    captured = capture_one_frame(serial, auto=args.auto)
    if captured is None:
        print("  캡처 실패")
        return 1
    color, _ = captured

    print("\n[2/3] ArUco 사전 자동 탐지...")
    aruco_results = test_aruco_all_dictionaries(color)

    print("\n[3/3] Charuco 양방향 테스트...")
    test_charuco_both_orientations(
        color,
        dictionary_name=board_cfg["dictionary"],
        squares_a=board_cfg["squares_x"],
        squares_b=board_cfg["squares_y"],
        square_length_m=board_cfg["square_length_m"],
        marker_length_m=board_cfg["marker_length_m"],
    )

    annotate_and_save(color, board_cfg["dictionary"], Path(args.save_dir))

    print("\n" + "=" * 60)
    print("  진단 결론 가이드:")
    print("=" * 60)
    if not aruco_results:
        print("  → ArUco 검출 0: 이미지/조명/보드 가시성 점검")
    else:
        top_dict = aruco_results[0][0]
        if top_dict != board_cfg["dictionary"]:
            print(f"  → 권장 사전: {top_dict} (yaml의 "
                  f"{board_cfg['dictionary']} 와 다름)")
        # 양방향 테스트 결과를 보고 squares_x, squares_y 결정
        print("  → 위 'Charuco 양방향 테스트' 결과를 보고 더 많이 검출되는")
        print("    squares_x/y 조합을 yaml에 적용")
    return 0


if __name__ == "__main__":
    sys.exit(main())
