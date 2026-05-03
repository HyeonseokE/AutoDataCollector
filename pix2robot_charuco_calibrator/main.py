"""
Charuco 캘리브레이션 통합 파이프라인 — 4페이즈 순차 실행.

실행:
    ./run_calibration.sh --robot 0
    또는
    python -m pix2robot_charuco_calibrator.main --robot 0

옵션:
    --robot N                   로봇 번호
    --board-config PATH         보드 yaml 경로 (기본 board_config.yaml)
    --skip-intrinsics           Phase 2 건너뛰기 (기존 K, dist 재사용)
    --skip-verification         Phase 4 건너뛰기
    --skip-detect               Phase 1에서 자동 사전 탐지 건너뛰기
    --phases 1,2,3,4            특정 페이즈만 실행
    --output-dir PATH           저장 루트 (기본 robot_configs/charuco_calibration)
"""

import argparse
import json
import sys
import shutil
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from .camera_setup import (
    CalibrationCamera,
    list_realsense_devices,
    print_available_cameras,
    get_factory_intrinsics,
    validate_intrinsics,
)
from .charuco_detector import (
    CharucoBoardSpec,
    auto_detect_dictionary,
    group_compatible_dictionaries,
    STANDARD_DICTIONARIES,
)
from .extrinsics import ExtrinsicsCalibrator
from .intrinsics import IntrinsicsCalibrator, load_intrinsics
from .verification import Verifier
from .visualization import (
    print_phase_banner,
    print_phase_summary,
    print_session_summary,
    confirm_yn,
    prompt_float,
    prompt_int,
    ANSI,
)


DEFAULT_BOARD_CONFIG = Path(__file__).parent / "board_config.yaml"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).parent.parent / "robot_configs" / "charuco_calibration"
)


def load_board_yaml(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def phase1_board_setup(
    config_path: Path,
    skip_detect: bool = False,
    camera_serial: Optional[str] = None,
    interactive: bool = False,
) -> tuple[CharucoBoardSpec, dict]:
    """
    Phase 1: 보드 사양 확인 및 (선택적) 자동 탐지.
    Returns:
        (CharucoBoardSpec, full_yaml_dict)
    """
    print_phase_banner(1, subtitle="보드 사양 로드 및 확인")

    config = load_board_yaml(config_path)
    board_dict = config["board"]

    print(f"  Loaded: {config_path}")
    print(f"    dictionary:      {board_dict['dictionary']}")
    print(f"    squares:         {board_dict['squares_x']} × {board_dict['squares_y']}")
    print(f"    square_length:   {board_dict['square_length_m']*1000:.1f} mm")
    print(f"    marker_length:   {board_dict['marker_length_m']*1000:.1f} mm")
    print(f"    thickness:       {float(board_dict.get('thickness_m', 0.0))*1000:.1f} mm")
    print()

    if not interactive:
        print("  yaml 사양 그대로 자동 진행 (변경하려면 --interactive-board)")
    else:
        use_yaml = confirm_yn("이 사양으로 진행하시겠습니까?", default=True)
        if not use_yaml:
            if not skip_detect and confirm_yn(
                "카메라로 보드를 1프레임 촬영하여 자동 탐지를 시도할까요?",
                default=True,
            ):
                board_dict = _interactive_auto_detect(board_dict, camera_serial)
            else:
                board_dict = _interactive_manual_input(board_dict)

    spec = CharucoBoardSpec.from_dict(board_dict)
    config["board"] = spec.to_dict()
    print_phase_summary(
        1, success=True,
        details=f"{spec.dictionary}, {spec.squares_x}x{spec.squares_y}, "
                f"{spec.square_length_m*1000:.0f}mm",
    )
    return spec, config


def _interactive_auto_detect(
    current: dict, camera_serial: Optional[str] = None,
) -> dict:
    """카메라로 1프레임 촬영 → 사전/격자 자동 탐지."""
    print("\n  카메라 미리보기를 표시합니다. 보드를 잘 보이게 두고 's' 키를 눌러 캡처.")
    captured = None
    win = "Phase 1 - Board Auto-Detect"
    cv2.namedWindow(win)
    with CalibrationCamera(serial=camera_serial) as camera:
        while True:
            color, _ = camera.get_frames()
            if color is None:
                continue
            cv2.putText(
                color, "Press 's' to capture, 'q' to cancel",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            cv2.imshow(win, color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                captured = color.copy()
                break
            elif key == ord('q'):
                break
    cv2.destroyWindow(win)

    if captured is None:
        print("  캡처 취소. 수동 입력으로 전환.")
        return _interactive_manual_input(current)

    print("\n  17개 표준 사전 자동 탐지 중...")
    results = auto_detect_dictionary(captured, min_markers=4)
    if not results:
        print("  검출 실패. 수동 입력으로 전환.")
        return _interactive_manual_input(current)

    groups = group_compatible_dictionaries(results)
    print(f"  검출됨: {len(results)} 사전, {len(groups)} 호환 그룹")
    for gi, group in enumerate(groups):
        names = [name for name, _, _ in group]
        count = group[0][1]
        ids = sorted(set(group[0][2]))
        print(f"  [{gi+1}] {names} → {count} 마커, IDs={ids[:8]}{'...' if len(ids)>8 else ''}")

    if len(groups) == 1:
        chosen_group = groups[0]
    else:
        idx = prompt_int(f"\n  몇 번 그룹을 사용하시겠습니까? (1~{len(groups)})", default=1)
        chosen_group = groups[idx - 1]

    # 그룹 내 가장 작은 사전 권장
    chosen_name = chosen_group[0][0]
    print(f"\n  선택된 사전: {chosen_name}")

    # 격자 (squares_x, squares_y) 자동 추정 시도 — Charuco 검출로 확인
    out = dict(current)
    out["dictionary"] = chosen_name
    out["squares_x"] = prompt_int(
        "  squares_x (가로 사각형 개수)", default=current["squares_x"]
    )
    out["squares_y"] = prompt_int(
        "  squares_y (세로 사각형 개수)", default=current["squares_y"]
    )
    out["square_length_m"] = prompt_float(
        "  square_length_m (자로 측정한 사각형 한 변, m)",
        default=current["square_length_m"],
    )
    out["marker_length_m"] = prompt_float(
        "  marker_length_m (ArUco 마커 한 변, m)",
        default=current["marker_length_m"],
    )
    return out


def _interactive_manual_input(current: dict) -> dict:
    print("\n  보드 사양 직접 입력:")
    print(f"  사용 가능 dictionary: {list(STANDARD_DICTIONARIES.keys())}")
    while True:
        d = input(f"  dictionary [{current['dictionary']}]: ").strip()
        d = d or current['dictionary']
        if d in STANDARD_DICTIONARIES:
            break
        print("  유효하지 않은 사전. 다시 입력.")
    out = dict(current)
    out["dictionary"] = d
    out["squares_x"] = prompt_int("  squares_x", default=current["squares_x"])
    out["squares_y"] = prompt_int("  squares_y", default=current["squares_y"])
    out["square_length_m"] = prompt_float(
        "  square_length_m", default=current["square_length_m"]
    )
    out["marker_length_m"] = prompt_float(
        "  marker_length_m", default=current["marker_length_m"]
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Charuco 4페이즈 통합 캘리브레이션",
    )
    parser.add_argument("--robot", type=int, required=True, help="로봇 번호")
    parser.add_argument(
        "--board-config", type=Path, default=DEFAULT_BOARD_CONFIG,
        help="보드 yaml (기본: board_config.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="저장 디렉토리 (기본: robot_configs/charuco_calibration)",
    )
    parser.add_argument("--skip-intrinsics", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--skip-detect", action="store_true",
                        help="Phase 1에서 자동 탐지 비활성화")
    parser.add_argument(
        "--phases", type=str, default=None,
        help="실행할 페이즈 (예: '1,2', '3,4', '2,3,4'). 기본 모두.",
    )
    parser.add_argument(
        "--camera-serial", type=str, default=None,
        help=("사용할 RealSense 카메라 일련번호. 미지정 시 단일 기기 자동 선택, "
              "다중 연결이면 에러. yaml의 camera.serial 보다 우선."),
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="연결된 RealSense 기기를 출력하고 종료.",
    )
    parser.add_argument(
        "--charuco-intrinsics", action="store_true",
        help=("Phase 2에서 Charuco 보드로 K, dist 직접 캘리브 (기본은 factory K). "
              "보드 자세 다양성이 충분할 때만 권장. RMS가 낮아도 K가 잘못 나올 수 있음."),
    )
    parser.add_argument(
        "--interactive-board", action="store_true",
        help="Phase 1에서 보드 사양 확인 프롬프트 표시 (기본은 yaml 그대로 자동 진행).",
    )
    args = parser.parse_args()

    # 기본은 factory K. --charuco-intrinsics 명시한 경우만 Phase 2 실행.
    args.use_factory_intrinsics = not args.charuco_intrinsics

    # --list-cameras 처리 → 즉시 종료
    if args.list_cameras:
        print_available_cameras()
        return 0

    # 페이즈 결정
    phases_to_run: Set[int] = (
        set(int(p) for p in args.phases.split(","))
        if args.phases else {1, 2, 3, 4}
    )
    if args.skip_intrinsics or args.use_factory_intrinsics:
        phases_to_run.discard(2)
    if args.skip_verification:
        phases_to_run.discard(4)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_path = output_dir / "camera_intrinsics.npz"
    extrinsics_path = output_dir / f"robot{args.robot}_cam2robot.npz"
    verification_path = output_dir / f"robot{args.robot}_verification.json"
    saved_board_path = output_dir / "board_config.yaml"

    # 카메라 serial 결정: CLI > yaml > None(자동)
    yaml_pre = load_board_yaml(args.board_config)
    yaml_camera_serial = (yaml_pre.get("camera") or {}).get("serial")
    camera_serial = args.camera_serial or yaml_camera_serial

    # 연결된 카메라 출력 (사용자 확인용)
    print_available_cameras()
    if camera_serial:
        print(f"  사용할 카메라 serial: {camera_serial} "
              f"(출처: {'CLI' if args.camera_serial else 'yaml'})")
    else:
        print(f"  사용할 카메라 serial: (미지정 — 단일 기기 자동 선택)")

    bold = ANSI["bold"]
    reset = ANSI["reset"]
    print()
    print(f"{bold}╔═══════════════════════════════════════════════╗{reset}")
    print(f"{bold}║   Charuco Calibration Pipeline (4 Phases)     ║{reset}")
    print(f"{bold}╚═══════════════════════════════════════════════╝{reset}")
    print(f"  로봇:          robot{args.robot}")
    print(f"  보드 설정:     {args.board_config}")
    print(f"  저장 경로:     {output_dir}")
    print(f"  실행 페이즈:   {sorted(phases_to_run)}")
    print(f"  Intrinsics:    "
          f"{'Charuco (Phase 2)' if args.charuco_intrinsics else 'Factory K (자동)'}")
    print(f"  보드 확인:     "
          f"{'대화식' if args.interactive_board else '자동 (yaml 그대로)'}")
    print(f"  매칭점 계산:   PnP (보드 기하 활용, depth 우회)")

    results = {}
    spec: Optional[CharucoBoardSpec] = None
    config_dict = None

    # Phase 1
    if 1 in phases_to_run:
        try:
            spec, config_dict = phase1_board_setup(
                args.board_config,
                skip_detect=args.skip_detect,
                camera_serial=camera_serial,
                interactive=args.interactive_board,
            )
            # 결정된 카메라 serial을 yaml에도 보존
            config_dict.setdefault("camera", {})["serial"] = camera_serial
            with open(saved_board_path, 'w') as f:
                yaml.safe_dump(config_dict, f, sort_keys=False)
            results[1] = {"success": True,
                          "details": f"{spec.dictionary} {spec.squares_x}x{spec.squares_y}"}
        except Exception as e:
            print(f"  Phase 1 실패: {e}")
            results[1] = {"success": False, "details": str(e)}
            print_session_summary(results)
            return 1
    else:
        # 다른 페이즈가 spec 필요 → yaml 로드
        config_dict = yaml_pre
        spec = CharucoBoardSpec.from_dict(config_dict["board"])

    cal_opts = config_dict.get("calibration", {}) if config_dict else {}

    # Phase 2: Intrinsics
    K, dist = None, None
    if 2 in phases_to_run:
        opts = cal_opts.get("intrinsics", {})
        cal = IntrinsicsCalibrator(
            board_spec=spec,
            target_num_images=opts.get("target_num_images", 25),
            min_num_images=opts.get("min_num_images", 15),
            min_corners_per_image=opts.get("min_corners_per_image", 8),
            target_rms_px=opts.get("target_rms_px", 0.5),
            camera_serial=camera_serial,
        )
        ok = cal.run(intrinsics_path)
        results[2] = {
            "success": ok,
            "details": (
                f"RMS={cal.rms_error:.3f} px" if cal.rms_error
                else "미완료"
            ),
        }
        if not ok:
            print_session_summary(results)
            return 1
        K, dist = cal.K, cal.dist
        # Intrinsics 검증 — 비정상이면 사용자에게 경고
        warnings = validate_intrinsics(K, dist, (640, 480))
        if warnings:
            print()
            print("  ⚠ Phase 2 intrinsics 검증 경고:")
            for w in warnings:
                print(f"    - {w}")
            print()
            print("  RMS는 낮아도 보드 자세 다양성이 부족하면 잘못된 K가 나올 수 있습니다.")
            print("  권장: --use-factory-intrinsics 로 재실행하거나 더 다양한 자세로 재캡처.")
            if not confirm_yn("그래도 이 K로 계속 진행하시겠습니까?", default=False):
                print_session_summary(results)
                return 1
    elif 3 in phases_to_run or 4 in phases_to_run:
        if args.use_factory_intrinsics:
            print()
            print("  ──── D435 Factory Intrinsics 사용 ────")
            try:
                fk = get_factory_intrinsics(camera_serial)
            except Exception as e:
                print(f"  Factory intrinsics 쿼리 실패: {e}")
                return 1
            K, dist = fk["K"], fk["dist"]
            print(f"  Serial:        {fk['serial']}")
            print(f"  Image size:    {fk['image_size']}")
            print(f"  Depth scale:   {fk['depth_scale']} m/unit")
            print(f"  Distortion:    {fk['distortion_model']}")
            print(f"  K = ")
            for row in K:
                print(f"    [{row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f}]")
            print(f"  dist = {dist.tolist()}")

            warnings = validate_intrinsics(K, dist, fk["image_size"])
            if warnings:
                print("  검증 경고:")
                for w in warnings:
                    print(f"    - {w}")
            else:
                print("  검증: 모든 항목 정상 범위")

            # factory 값을 표준 캘리브 파일로 저장
            np.savez(
                intrinsics_path,
                K=K, dist=dist,
                rms_error=np.array([0.0]),
                num_images=np.array([0]),
                image_width=np.array([fk["image_size"][0]]),
                image_height=np.array([fk["image_size"][1]]),
            )
            json_path = intrinsics_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump({
                    "K": K.tolist(),
                    "dist": dist.tolist(),
                    "source": "factory",
                    "serial": fk["serial"],
                    "depth_scale": fk["depth_scale"],
                    "image_size": list(fk["image_size"]),
                    "distortion_model": fk["distortion_model"],
                }, f, indent=2)
            print(f"  저장됨: {intrinsics_path}")
            results[2] = {"success": True, "details": f"factory K (fx={K[0,0]:.1f})"}
        else:
            if not intrinsics_path.exists():
                print(f"  Intrinsics 파일 없음: {intrinsics_path}")
                print(f"  Phase 2를 먼저 실행하거나 --use-factory-intrinsics 사용.")
                return 1
            K, dist = load_intrinsics(intrinsics_path)
            print(f"\n  기존 intrinsics 로드: {intrinsics_path}")
            warnings = validate_intrinsics(K, dist, (640, 480))
            if warnings:
                print("  ⚠ Intrinsics 검증 경고 — 부정확할 가능성:")
                for w in warnings:
                    print(f"    - {w}")
                print("  → 권장: --use-factory-intrinsics 로 재실행")

    # Phase 3: Extrinsics
    R, t = None, None
    if 3 in phases_to_run:
        opts = cal_opts.get("extrinsics", {})
        cal = ExtrinsicsCalibrator(
            robot_id=args.robot,
            board_spec=spec,
            K=K, dist=dist,
            target_num_pairs=opts.get("target_num_pairs", 12),
            min_num_pairs=opts.get("min_num_pairs", 6),
            target_rmse_mm=opts.get("target_rmse_mm", 3.0),
            camera_serial=camera_serial,
        )
        ok = cal.run(extrinsics_path)
        results[3] = {
            "success": ok,
            "details": (
                f"RMSE={cal.stats['rmse_m']*1000:.2f} mm, "
                f"{len(cal.pairs)} 쌍" if cal.stats else "미완료"
            ),
        }
        if not ok:
            print_session_summary(results)
            return 1
        R, t = cal.R, cal.t
    elif 4 in phases_to_run:
        if not extrinsics_path.exists():
            print(f"  Extrinsics 파일 없음: {extrinsics_path}")
            return 1
        from .extrinsics import load_extrinsics
        cam_to_base, R, t, K2, dist2 = load_extrinsics(extrinsics_path)
        if K is None:
            K, dist = K2, dist2
        print(f"\n  기존 extrinsics 로드: {extrinsics_path}")

    # Phase 4: Verification
    if 4 in phases_to_run:
        opts = cal_opts.get("verification", {})
        ver = Verifier(
            robot_id=args.robot,
            board_spec=spec,
            K=K, dist=dist,
            R=R, t=t,
            num_test_corners=opts.get("num_test_corners", 5),
            target_avg_error_mm=opts.get("target_avg_error_mm", 3.0),
            camera_serial=camera_serial,
        )
        ok = ver.run(verification_path)
        if ver.results:
            errs = np.array([r.error_norm for r in ver.results])
            avg_mm = errs.mean() * 1000
            results[4] = {
                "success": ok,
                "details": (
                    f"평균 {avg_mm:.2f} mm, {len(ver.results)} 코너"
                ),
            }
        else:
            results[4] = {"success": False, "details": "결과 없음"}

    print_session_summary(results)

    # 캘리브 결과 파일 위치 명시 (사용자가 어디에서 로드해야 할지 확인)
    print(f"  ── 캘리브레이션 결과 파일 ──")
    if intrinsics_path.exists():
        print(f"    Intrinsics:    {intrinsics_path}")
    if extrinsics_path.exists():
        print(f"    Extrinsics:    {extrinsics_path}")
        print(f"                   (cam_to_base 4x4 = 카메라 → 로봇 변환행렬)")
    if verification_path.exists():
        print(f"    Verification:  {verification_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
