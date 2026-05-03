"""
기초 모듈 검증 테스트 — 합성 데이터로 수치 정확성 확인.

실행:
    python -m pix2robot_charuco_calibrator._test_foundations
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pix2robot_charuco_calibrator.geometry import (
    pixel_to_camera_3d,
    rigid_align_kabsch,
    apply_transform,
    reprojection_residuals,
    compose_transform_4x4,
    decompose_transform_4x4,
)
from pix2robot_charuco_calibrator.charuco_detector import (
    CharucoBoardSpec,
    CharucoDetector,
    auto_detect_dictionary,
    STANDARD_DICTIONARIES,
)
from pix2robot_charuco_calibrator.visualization import (
    print_phase_banner,
    print_phase_summary,
    draw_phase_overlay,
)


def _ok(name: str, ok: bool, detail: str = ""):
    marker = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
    print(f"  {marker} {name}{(' — ' + detail) if detail else ''}")
    return ok


def test_rigid_align_identity():
    """완벽히 동일한 점 → R=I, t=0."""
    np.random.seed(0)
    P = np.random.randn(8, 3) * 0.1
    R, t, stats = rigid_align_kabsch(P, P)
    return (
        _ok("rigid identity: R≈I", np.allclose(R, np.eye(3), atol=1e-9))
        and _ok("rigid identity: t≈0", np.allclose(t, 0, atol=1e-9))
        and _ok("rigid identity: rmse≈0", stats["rmse_m"] < 1e-9,
                f"rmse={stats['rmse_m']:.2e}")
    )


def test_rigid_align_known_transform():
    """알려진 R, t를 적용한 점 → 같은 R, t를 복원."""
    np.random.seed(42)
    # 임의의 회전 (각도-축 표현)
    axis = np.array([0.3, 0.7, 0.5])
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(35.0)
    R_true, _ = cv2.Rodrigues(axis * angle)
    t_true = np.array([0.5, -0.3, 0.2])

    # 워크스페이스 분포 점들
    P_cam = np.random.uniform(-0.2, 0.2, (15, 3))
    P_rob = (R_true @ P_cam.T).T + t_true

    R_est, t_est, stats = rigid_align_kabsch(P_cam, P_rob)

    R_diff = np.abs(R_est - R_true).max()
    t_diff = np.abs(t_est - t_true).max()

    return (
        _ok("known transform: R 복원", R_diff < 1e-9, f"max |ΔR|={R_diff:.2e}")
        and _ok("known transform: t 복원", t_diff < 1e-9, f"max |Δt|={t_diff:.2e}")
        and _ok("known transform: rmse≈0", stats["rmse_m"] < 1e-9,
                f"rmse={stats['rmse_m']:.2e}")
    )


def test_rigid_align_with_noise():
    """가우시안 노이즈 → 노이즈 수준에 비례한 잔차."""
    np.random.seed(7)
    R_true, _ = cv2.Rodrigues(np.array([0.1, 0.2, 0.05]))
    t_true = np.array([0.3, 0.1, 0.05])

    P_cam = np.random.uniform(-0.15, 0.15, (12, 3))
    P_rob_clean = (R_true @ P_cam.T).T + t_true
    noise_sigma = 0.001  # 1 mm
    P_rob = P_rob_clean + np.random.randn(*P_rob_clean.shape) * noise_sigma

    R_est, t_est, stats = rigid_align_kabsch(P_cam, P_rob)
    rmse = stats["rmse_m"]

    # 1mm 노이즈 → 잔차도 ~1mm 수준이어야 함
    return _ok(
        "noisy 1mm: rmse ~1mm",
        0.0001 < rmse < 0.003,
        f"rmse={rmse*1000:.2f} mm",
    )


def test_rigid_align_no_reflection():
    """좌표계 반사가 끼어들지 않아야 (det(R)=+1)."""
    np.random.seed(1)
    P_cam = np.random.randn(10, 3) * 0.1
    R_true, _ = cv2.Rodrigues(np.array([0.5, 0.5, 0.5]))
    t_true = np.array([0.1, 0.2, 0.3])
    P_rob = (R_true @ P_cam.T).T + t_true

    R_est, _, _ = rigid_align_kabsch(P_cam, P_rob)
    det = np.linalg.det(R_est)
    return _ok("no reflection: det(R)=+1", abs(det - 1.0) < 1e-9, f"det={det:.6f}")


def test_pixel_to_camera_3d():
    """픽셀 + depth → 3D 점 (해석적 검증).

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] 일 때
    픽셀 (cx, cy) + depth = z → (0, 0, z) 광축 위
    픽셀 (cx + fx, cy) + depth = z → (z, 0, z) (카메라 X축으로 z만큼)
    """
    K = np.array([[600.0, 0, 320.0],
                  [0, 600.0, 240.0],
                  [0, 0, 1.0]], dtype=np.float64)

    # 광축 위 점
    p1 = pixel_to_camera_3d(320, 240, depth_m=1.0, K=K)
    ok1 = _ok(
        "principal point → (0, 0, 1)",
        np.allclose(p1, [0, 0, 1], atol=1e-6),
        f"got {p1}",
    )

    # (cx + fx, cy) → x_n = 1, y_n = 0 → (1*1, 0, 1)
    p2 = pixel_to_camera_3d(320 + 600, 240, depth_m=1.0, K=K)
    ok2 = _ok(
        "+fx pixel → (1, 0, 1)",
        np.allclose(p2, [1, 0, 1], atol=1e-6),
        f"got {p2}",
    )

    # (cx, cy + fy) → (0, 1, 1)
    p3 = pixel_to_camera_3d(320, 240 + 600, depth_m=1.0, K=K)
    ok3 = _ok(
        "+fy pixel → (0, 1, 1)",
        np.allclose(p3, [0, 1, 1], atol=1e-6),
        f"got {p3}",
    )

    # depth scale
    p4 = pixel_to_camera_3d(320 + 300, 240, depth_m=2.0, K=K)
    # x_n = 0.5, y_n = 0 → depth=2 → (1.0, 0, 2.0)
    ok4 = _ok(
        "depth scaling",
        np.allclose(p4, [1.0, 0, 2.0], atol=1e-6),
        f"got {p4}",
    )
    return ok1 and ok2 and ok3 and ok4


def test_pixel_to_3d_with_distortion():
    """왜곡 계수가 있을 때 undistortPoints 통합되는지."""
    K = np.array([[600.0, 0, 320.0],
                  [0, 600.0, 240.0],
                  [0, 0, 1.0]], dtype=np.float64)
    dist = np.array([0.1, -0.05, 0, 0, 0], dtype=np.float64)

    # 광축 위는 왜곡 영향 없음 (radius=0)
    p = pixel_to_camera_3d(320, 240, depth_m=1.0, K=K, dist=dist)
    return _ok(
        "distortion at principal point",
        np.allclose(p, [0, 0, 1], atol=1e-6),
        f"got {p}",
    )


def test_apply_transform_consistency():
    """apply_transform과 직접 계산이 일치."""
    np.random.seed(11)
    R, _ = cv2.Rodrigues(np.array([0.2, 0.3, 0.1]))
    t = np.array([0.5, -0.2, 0.1])
    P = np.random.randn(5, 3)

    out_func = apply_transform(P, R, t)
    out_manual = (R @ P.T).T + t

    ok1 = _ok(
        "apply_transform batch",
        np.allclose(out_func, out_manual, atol=1e-12),
    )

    p_single = np.array([0.1, 0.2, 0.3])
    out_single = apply_transform(p_single, R, t)
    ok2 = _ok(
        "apply_transform single point",
        out_single.shape == (3,) and np.allclose(out_single, R @ p_single + t),
    )
    return ok1 and ok2


def test_compose_decompose_4x4():
    """compose ↔ decompose roundtrip."""
    R, _ = cv2.Rodrigues(np.array([0.4, 0.1, 0.7]))
    t = np.array([0.3, 0.2, 0.1])
    T = compose_transform_4x4(R, t)
    R2, t2 = decompose_transform_4x4(T)
    return (
        _ok("compose: shape (4,4)", T.shape == (4, 4))
        and _ok("compose: bottom row [0,0,0,1]",
                np.allclose(T[3], [0, 0, 0, 1]))
        and _ok("decompose roundtrip: R", np.allclose(R, R2))
        and _ok("decompose roundtrip: t", np.allclose(t, t2))
    )


def test_reprojection_residuals():
    """완벽한 변환에서 잔차=0, 노이즈 1mm에서 잔차≈1mm."""
    np.random.seed(3)
    R, _ = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))
    t = np.array([0.4, 0.1, 0.05])
    P_cam = np.random.uniform(-0.1, 0.1, (8, 3))
    P_rob_clean = (R @ P_cam.T).T + t

    s_clean = reprojection_residuals(P_cam, P_rob_clean, R, t)
    ok1 = _ok("perfect: rmse≈0", s_clean["rmse_m"] < 1e-12,
              f"rmse={s_clean['rmse_m']:.2e}")

    P_rob_noisy = P_rob_clean + np.random.randn(*P_rob_clean.shape) * 0.001
    s_noisy = reprojection_residuals(P_cam, P_rob_noisy, R, t)
    ok2 = _ok("noisy: rmse ~1mm", 0.0005 < s_noisy["rmse_m"] < 0.003,
              f"rmse={s_noisy['rmse_m']*1000:.2f} mm")
    return ok1 and ok2


def test_board_spec_roundtrip():
    spec = CharucoBoardSpec(
        dictionary="DICT_5X5_100",
        squares_x=5, squares_y=7,
        square_length_m=0.055, marker_length_m=0.041,
    )
    d = spec.to_dict()
    spec2 = CharucoBoardSpec.from_dict(d)
    return (
        _ok("board spec to_dict/from_dict",
            spec.to_dict() == spec2.to_dict())
        and _ok("aruco_dict() runs", spec.aruco_dict() is not None)
        and _ok("board() runs", spec.board() is not None)
    )


def test_auto_detect_dictionary_synthetic():
    """
    합성 Charuco 보드 이미지에서 dictionary 자동 탐지.

    참고: ArUco 사전들은 부분집합 관계라 (예: DICT_5X5_50 ⊂ DICT_5X5_100 ⊂ DICT_5X5_250)
    같은 비트 크기의 더 큰 사전도 모두 검출됨. 따라서 "패밀리 일치"만 검증.
    """
    spec = CharucoBoardSpec(
        dictionary="DICT_5X5_100",
        squares_x=5, squares_y=7,
        square_length_m=0.055, marker_length_m=0.041,
    )
    board = spec.board()
    img = board.generateImage((600, 840), marginSize=20)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    results = auto_detect_dictionary(img_bgr, min_markers=4)

    ok1 = _ok("auto-detect: 결과 ≥1개", len(results) > 0,
              f"검출 사전: {[r[0] for r in results[:3]]}")
    if not results:
        return False

    # DICT_5X5 패밀리에 포함되는지 확인
    family_5x5 = [r for r in results if r[0].startswith("DICT_5X5_")]
    ok2 = _ok(
        "auto-detect: DICT_5X5 패밀리 검출",
        len(family_5x5) > 0,
        f"5x5 패밀리: {[r[0] for r in family_5x5]}",
    )

    # 모든 5x5 사전이 같은 마커 수를 검출 → 부분집합 관계 확인
    if len(family_5x5) >= 2:
        counts = [r[1] for r in family_5x5]
        ok3 = _ok(
            "auto-detect: 5x5 패밀리 동일 검출 수 (부분집합 관계)",
            all(c == counts[0] for c in counts),
            f"검출 수: {counts}",
        )
    else:
        ok3 = True

    # 4x4, 6x6, 7x7 패밀리는 같은 보드에서 검출되면 안 됨
    other_family = [
        r for r in results
        if r[0].startswith(("DICT_4X4_", "DICT_6X6_", "DICT_7X7_"))
    ]
    ok4 = _ok(
        "auto-detect: 다른 비트 크기 사전 미검출",
        len(other_family) == 0,
        f"검출되면 안 됨: {[r[0] for r in other_family]}",
    )
    return ok1 and ok2 and ok3 and ok4


def test_charuco_detector_no_K():
    """K 없으면 픽셀만 반환 (3D 미계산)."""
    spec = CharucoBoardSpec(
        dictionary="DICT_5X5_100",
        squares_x=5, squares_y=7,
        square_length_m=0.055, marker_length_m=0.041,
    )
    detector = CharucoDetector(spec)
    img = spec.board().generateImage((600, 840), marginSize=20)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    detections = detector.detect_charuco(img_bgr)
    ok1 = _ok("K 없음: 코너 픽셀은 검출",
              len(detections) >= 10, f"{len(detections)}개")
    ok2 = _ok("K 없음: 3D 미계산 (has_3d=False)",
              all(not d.has_3d for d in detections))
    ids = [d.corner_id for d in detections]
    ok3 = _ok("코너 ID 정렬됨", ids == sorted(ids))
    return ok1 and ok2 and ok3


def test_charuco_detector_pnp():
    """PnP로 3D 좌표 도출 — depth 안 쓰고 보드 기하만 사용."""
    spec = CharucoBoardSpec(
        dictionary="DICT_5X5_100",
        squares_x=5, squares_y=7,
        square_length_m=0.055, marker_length_m=0.041,
    )
    K = np.array([[600.0, 0, 300.0], [0, 600.0, 420.0], [0, 0, 1]], dtype=np.float64)
    detector = CharucoDetector(spec, K=K, dist=np.zeros(5))

    img = spec.board().generateImage((600, 840), marginSize=20)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    detections = detector.detect_charuco(img_bgr)

    ok1 = _ok("PnP: ≥10 코너", len(detections) >= 10, f"{len(detections)}개")
    ok2 = _ok("PnP: 모든 검출에 3D 있음",
              all(d.has_3d for d in detections),
              f"3D: {sum(d.has_3d for d in detections)}/{len(detections)}")

    # 두 인접 코너 사이 거리가 square_length와 일치 (PnP의 핵심 검증)
    if len(detections) >= 2:
        d0 = detections[0]
        d1 = next((d for d in detections if d.corner_id == d0.corner_id + 1), None)
        if d1 is not None:
            distance = float(np.linalg.norm(d1.camera_xyz - d0.camera_xyz))
            ok3 = _ok(
                "PnP: 인접 코너 거리 ≈ square_length",
                abs(distance - spec.square_length_m) < 0.001,
                f"실측 {distance*1000:.2f}mm, 보드 {spec.square_length_m*1000:.2f}mm",
            )
        else:
            ok3 = True
    else:
        ok3 = False
    return ok1 and ok2 and ok3


def test_visualization():
    """시각화 배너 — 이미지 외부에 패딩 추가 동작 검증."""
    from pix2robot_charuco_calibrator.visualization import (
        PHASE_HEADER_HEIGHT, PHASE_FOOTER_HEIGHT,
    )

    print_phase_banner(2, subtitle="합성 테스트")
    print_phase_summary(2, success=True, details="합성 검증 OK")

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (60, 60, 60)

    # footer 있을 때
    out = draw_phase_overlay(img, phase=3, progress_text="5/15", keys_text="[s][q]")
    expected_h = 480 + PHASE_HEADER_HEIGHT + PHASE_FOOTER_HEIGHT
    ok1 = _ok(
        "패딩: with footer height 증가",
        out.shape == (expected_h, 640, 3),
        f"shape={out.shape}, expected ({expected_h}, 640, 3)",
    )

    # 원본 이미지가 패딩된 캔버스 안에 보존됐는지
    interior = out[PHASE_HEADER_HEIGHT:PHASE_HEADER_HEIGHT + 480]
    ok2 = _ok(
        "패딩: 원본 이미지가 가운데 영역에 보존",
        np.array_equal(interior, img),
    )

    # footer 없을 때
    out2 = draw_phase_overlay(img, phase=3, progress_text="5/15", keys_text="")
    expected_h2 = 480 + PHASE_HEADER_HEIGHT
    ok3 = _ok(
        "패딩: footer 없으면 헤더만 추가",
        out2.shape == (expected_h2, 640, 3),
        f"shape={out2.shape}, expected ({expected_h2}, 640, 3)",
    )

    return ok1 and ok2 and ok3


def test_end_to_end_synthetic_pipeline():
    """
    가상 시나리오 통합 검증 (PnP 기반):
      1. 합성 Charuco 이미지
      2. detect_charuco (PnP) → 매칭점_camera 생성 (depth 안 씀)
      3. 알려진 변환행렬로 매칭점_robot 생성
      4. rigid_align_kabsch가 변환행렬 복원하는지
    """
    np.random.seed(99)
    spec = CharucoBoardSpec(
        dictionary="DICT_5X5_100",
        squares_x=5, squares_y=7,
        square_length_m=0.055, marker_length_m=0.041,
    )
    K = np.array([[600.0, 0, 300.0], [0, 600.0, 420.0], [0, 0, 1]], dtype=np.float64)
    detector = CharucoDetector(spec, K=K, dist=np.zeros(5))

    img = spec.board().generateImage((600, 840), marginSize=20)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    detections = detector.detect_charuco(img_bgr)
    P_cam = np.array([d.camera_xyz for d in detections if d.has_3d])

    # 가상의 카메라→로봇 변환
    R_true, _ = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))
    t_true = np.array([0.4, 0.1, 0.05])
    P_rob = (R_true @ P_cam.T).T + t_true

    R_est, t_est, stats = rigid_align_kabsch(P_cam, P_rob)

    return (
        _ok("E2E: ≥10 매칭점", len(P_cam) >= 10, f"{len(P_cam)}쌍")
        and _ok("E2E: R 복원",
                np.allclose(R_est, R_true, atol=1e-9),
                f"max |ΔR|={np.abs(R_est-R_true).max():.2e}")
        and _ok("E2E: t 복원",
                np.allclose(t_est, t_true, atol=1e-9),
                f"max |Δt|={np.abs(t_est-t_true).max():.2e}")
        and _ok("E2E: rmse≈0", stats["rmse_m"] < 1e-9,
                f"rmse={stats['rmse_m']:.2e}")
    )


# ── 러너 ─────────────────────────────────────────────────────────────

TESTS = [
    ("rigid identity",                  test_rigid_align_identity),
    ("rigid known transform",           test_rigid_align_known_transform),
    ("rigid with 1mm noise",            test_rigid_align_with_noise),
    ("rigid no reflection",             test_rigid_align_no_reflection),
    ("pixel→camera 3D analytic",        test_pixel_to_camera_3d),
    ("pixel→camera 3D w/ distortion",   test_pixel_to_3d_with_distortion),
    ("apply_transform consistency",     test_apply_transform_consistency),
    ("compose/decompose 4x4",           test_compose_decompose_4x4),
    ("reprojection residuals",          test_reprojection_residuals),
    ("board spec roundtrip",            test_board_spec_roundtrip),
    ("auto_detect_dictionary",          test_auto_detect_dictionary_synthetic),
    ("CharucoDetector no K (pixels only)", test_charuco_detector_no_K),
    ("CharucoDetector PnP",             test_charuco_detector_pnp),
    ("visualization smoke",             test_visualization),
    ("end-to-end synthetic pipeline",   test_end_to_end_synthetic_pipeline),
]


def main():
    print("=" * 60)
    print("기초 모듈 수치 검증")
    print("=" * 60)
    passed = 0
    failed_names = []
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            ok = fn()
        except Exception as e:
            ok = False
            print(f"  \033[31m✗ EXCEPTION\033[0m: {type(e).__name__}: {e}")
        if ok:
            passed += 1
        else:
            failed_names.append(name)

    print("\n" + "=" * 60)
    total = len(TESTS)
    if passed == total:
        print(f"\033[32m✓ {passed}/{total} 테스트 통과\033[0m")
        return 0
    else:
        print(f"\033[31m✗ {passed}/{total} 통과 — 실패: {failed_names}\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
