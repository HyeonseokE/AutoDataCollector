"""
3D 기하 유틸리티 — Charuco 캘리브레이션 수학 코어

함수:
    pixel_to_camera_3d: 픽셀(u,v) + depth → 카메라 좌표계 3D 점
    rigid_align_kabsch: 매칭점 두 집합 → 회전 R + 평행이동 t (closed-form)
    apply_transform:    R, t로 점 집합 변환
    reprojection_residuals: 정렬 후 잔차 통계
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np


def pixel_to_camera_3d(
    u: float,
    v: float,
    depth_m: float,
    K: np.ndarray,
    dist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    픽셀 (u,v) + depth → 카메라 좌표계 3D 점.

    파이프라인:
        (u,v) ──undistort──→ (x_n, y_n)  (정규화 좌표)
        ray = (x_n, y_n, 1)
        매칭점_camera = depth · ray

    Args:
        u, v: 픽셀 좌표
        depth_m: 해당 픽셀의 depth (meters, 카메라 광축 기준)
        K: (3,3) 내부 파라미터
        dist: (5,) 왜곡 계수, None이면 왜곡 없음 가정

    Returns:
        (3,) 카메라 좌표계의 3D 점 [X, Y, Z] (meters)
    """
    if depth_m <= 0:
        raise ValueError(f"Invalid depth: {depth_m}")

    pixel = np.array([[[float(u), float(v)]]], dtype=np.float32)
    if dist is None:
        dist = np.zeros(5, dtype=np.float32)

    normalized = cv2.undistortPoints(pixel, K, dist)[0, 0]  # (x_n, y_n)
    x_n, y_n = float(normalized[0]), float(normalized[1])

    return np.array([x_n * depth_m, y_n * depth_m, depth_m], dtype=np.float64)


def affine_align_3d(
    points_camera: np.ndarray,
    points_robot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Closed-form 3D-3D affine 정렬 (12 DoF — rigid 6 DoF의 확장).

    매칭점_robot ≈ M · 매칭점_camera + t
        M: (3, 3) 일반 선형 변환 (회전 + 비등방 스케일 + 전단 모두 허용)
        → SO-101 FK의 비등방 오차를 흡수해 RMSE ↓
        → 단점: 거리 보존 강제 안 함, 12 DoF 다 학습하려면 z 분산 필요

    LSQ 풀이:
        [P_cam | 1] · [M.T; t.T] = P_rob

    Args:
        points_camera: (N, 3)
        points_robot:  (N, 3)

    Returns:
        M: (3, 3) 선형 변환
        t: (3,)   평행이동
        stats: dict
    """
    P_cam = np.asarray(points_camera, dtype=np.float64)
    P_rob = np.asarray(points_robot, dtype=np.float64)

    if P_cam.shape != P_rob.shape or P_cam.shape[1] != 3:
        raise ValueError(f"Shape 불일치: cam={P_cam.shape}, rob={P_rob.shape}")

    n = P_cam.shape[0]
    if n < 4:
        raise ValueError(
            f"Affine 3D는 최소 4쌍 필요 (12 DoF, 점당 3 식). 현재 {n}쌍."
        )

    # LSQ: A · X = B,  X = [M.T; t.T]  (4, 3)
    A = np.hstack([P_cam, np.ones((n, 1))])             # (N, 4)
    X, _, _, _ = np.linalg.lstsq(A, P_rob, rcond=None)  # (4, 3)
    M = X[:3, :].T                                       # (3, 3)
    t = X[3, :]                                          # (3,)

    # 잔차 통계
    P_pred = (M @ P_cam.T).T + t
    errors = np.linalg.norm(P_pred - P_rob, axis=1)
    stats = {
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "max_error_m": float(np.max(errors)),
        "mean_error_m": float(np.mean(errors)),
        "per_point_errors_m": errors.tolist(),
        "num_points": n,
    }
    return M, t, stats


def rigid_align_kabsch(
    points_camera: np.ndarray,
    points_robot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Closed-form 3D-3D rigid 정렬 (Kabsch / Umeyama 1991).

    매칭점_robot ≈ R · 매칭점_camera + t  (R은 회전, 거리 보존)

    Note:
        평면 데이터로도 풀림 (Affine 12-DoF와 달리 z 분산 불필요).
        직교성 제약(R^T R = I, det=+1)이 z축 방향을 자동 결정.
        단, FK 비등방 오차는 흡수 안 함 → RMSE에 그대로 드러남.

    Returns:
        R: (3,3) 회전 행렬 (det=+1 보장)
        t: (3,)  평행이동
        stats: dict
    """
    P_cam = np.asarray(points_camera, dtype=np.float64)
    P_rob = np.asarray(points_robot, dtype=np.float64)

    if P_cam.shape != P_rob.shape or P_cam.shape[1] != 3:
        raise ValueError(f"Shape 불일치: cam={P_cam.shape}, rob={P_rob.shape}")

    n = P_cam.shape[0]
    if n < 3:
        raise ValueError(f"최소 3쌍 필요 (현재 {n}쌍)")

    # 1. 중심 이동
    centroid_cam = P_cam.mean(axis=0)
    centroid_rob = P_rob.mean(axis=0)
    cam_centered = P_cam - centroid_cam
    rob_centered = P_rob - centroid_rob

    # 2. 공분산 행렬 H = sum(cam_i · rob_i^T)
    H = cam_centered.T @ rob_centered

    # 3. SVD → R
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])  # reflection 방지
    R = Vt.T @ correction @ U.T

    # 4. t
    t = centroid_rob - R @ centroid_cam

    # 5. 잔차
    P_cam_transformed = (R @ P_cam.T).T + t
    errors = np.linalg.norm(P_cam_transformed - P_rob, axis=1)
    stats = {
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "max_error_m": float(np.max(errors)),
        "mean_error_m": float(np.mean(errors)),
        "per_point_errors_m": errors.tolist(),
        "num_points": n,
    }

    return R, t, stats


def apply_transform(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    3D 점 집합에 rigid 변환 적용: out = R · p + t.

    Args:
        points: (N, 3) 또는 (3,)
        R: (3,3)
        t: (3,)

    Returns:
        같은 shape의 변환된 점들
    """
    P = np.asarray(points, dtype=np.float64)
    single = P.ndim == 1
    if single:
        P = P[None, :]
    out = (R @ P.T).T + t
    return out[0] if single else out


def reprojection_residuals(
    points_camera: np.ndarray,
    points_robot: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> dict:
    """
    R, t 적용 후 robot 측 점들과의 잔차 통계.

    검증용 — 새 매칭점에 대해 R, t의 정확도 확인 가능.
    """
    P_cam = np.asarray(points_camera, dtype=np.float64)
    P_rob = np.asarray(points_robot, dtype=np.float64)
    P_pred = (R @ P_cam.T).T + t
    errors = np.linalg.norm(P_pred - P_rob, axis=1)

    return {
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "max_error_m": float(np.max(errors)),
        "mean_error_m": float(np.mean(errors)),
        "per_point_errors_m": errors.tolist(),
        "num_points": int(P_cam.shape[0]),
    }


def compose_transform_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """R (3,3) + t (3,) → 4x4 homogeneous 변환행렬."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def decompose_transform_4x4(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 homogeneous → (R, t)."""
    return T[:3, :3].copy(), T[:3, 3].copy()
