#!/usr/bin/env python3
"""
Compute world-to-robot base frame transformation using Kabsch/Umeyama algorithm.

This script reads matching point pairs collected by find_matching_point.py
and computes the optimal rigid body transformation (rotation + translation)
from world frame to robot base frame using SVD-based point set registration.

The Kabsch algorithm finds R, t that minimizes:
    sum_i || R * world_i + t - robot_base_i ||^2

Usage:
    python world2robot_extrinsic/world_frame2robot_base_frame.py --robot 2
    python world2robot_extrinsic/world_frame2robot_base_frame.py --robot 3
    python world2robot_extrinsic/world_frame2robot_base_frame.py --robot 2 --input my_points.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def kabsch_umeyama(source_points: np.ndarray, target_points: np.ndarray) -> tuple:
    """
    Kabsch-Umeyama algorithm for rigid body point set registration.

    Finds optimal rotation R and translation t such that:
        target ≈ R @ source + t

    This transforms points from source frame (world) to target frame (robot base).

    Args:
        source_points: Nx3 array of points in source frame (world)
        target_points: Nx3 array of points in target frame (robot base)

    Returns:
        Tuple of (R, t, rmse, transform_4x4)
        - R: 3x3 rotation matrix
        - t: 3x1 translation vector
        - rmse: Root mean square error after transformation
        - transform_4x4: 4x4 homogeneous transformation matrix
    """
    assert source_points.shape == target_points.shape
    assert source_points.shape[1] == 3

    n_points = source_points.shape[0]

    # Step 1: Compute centroids
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Step 2: Center the point sets
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # Step 3: Compute cross-covariance matrix H
    H = source_centered.T @ target_centered

    # Step 4: SVD of H
    U, S, Vt = np.linalg.svd(H)

    # Step 5: Compute rotation matrix
    # R = V @ U^T
    R = Vt.T @ U.T

    # Reflection handling for coplanar calibration points (Z≈0)
    #
    # When calibration points are all at Z≈0, the SVD has a zero singular value
    # for the Z direction, making the Z-axis of R arbitrary. The standard Kabsch
    # correction (flip Vt[-1,:] when det<0) forces det=+1 but may invert Z.
    #
    # Physical constraint: World Z+ and Robot Base Z+ both point UP (away from table).
    # If det(R)=+1 but R[2,2]<0, Z is inverted — physically wrong.
    # If det(R)=-1 but R[2,2]>0, Z is preserved — physically correct.
    #
    # This happens when the World and Robot coordinate frames have opposite
    # handedness (e.g., Y-axes point in opposite directions). In that case,
    # the correct transformation IS a reflection (det=-1), not a proper rotation.
    #
    # Strategy: choose the R that preserves Z direction (R[2,2] > 0).
    det_R = np.linalg.det(R)
    print(f"  det(R) = {det_R:.4f}, R[2,2] = {R[2, 2]:.4f}")

    if R[2, 2] < 0:
        # Z is inverted — flip the smallest singular vector to fix Z direction
        print("  Z축 반전 감지됨 - SVD 최소 singular vector 반전으로 보정")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        det_R = np.linalg.det(R)
        print(f"  보정 후: det(R) = {det_R:.4f}, R[2,2] = {R[2, 2]:.4f}")

    if det_R < 0:
        print("  det(R) = -1: World/Robot 좌표계 handedness가 다름 (정상)")

    # Step 6: Compute translation
    # t = centroid_target - R @ centroid_source
    t = centroid_target - R @ centroid_source

    # Step 7: Compute RMSE
    transformed = (R @ source_points.T).T + t
    errors = np.linalg.norm(transformed - target_points, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    # Build 4x4 homogeneous transformation matrix
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = R
    transform_4x4[:3, 3] = t

    return R, t, rmse, transform_4x4


def rotation_matrix_to_rpy(R: np.ndarray) -> tuple:
    """
    Extract roll, pitch, yaw (ZYX Euler angles) from rotation matrix.

    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) in degrees
    """
    # Check for gimbal lock
    if abs(R[2, 0]) >= 1.0 - 1e-6:
        # Gimbal lock case
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def compute_error_statistics(
    source_points: np.ndarray,
    target_points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> dict:
    """
    Compute detailed error statistics for the transformation.

    Args:
        source_points: Nx3 array of source points
        target_points: Nx3 array of target points
        R: 3x3 rotation matrix
        t: 3x1 translation vector

    Returns:
        Dictionary with error statistics
    """
    transformed = (R @ source_points.T).T + t
    errors = np.linalg.norm(transformed - target_points, axis=1)
    residuals = transformed - target_points

    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "min_error": float(np.min(errors)),
        "std_error": float(np.std(errors)),
        "error_per_axis": {
            "x_rmse": float(np.sqrt(np.mean(residuals[:, 0] ** 2))),
            "y_rmse": float(np.sqrt(np.mean(residuals[:, 1] ** 2))),
            "z_rmse": float(np.sqrt(np.mean(residuals[:, 2] ** 2))),
        },
        "per_point_errors": errors.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute world-to-robot transformation from matching points"
    )
    parser.add_argument(
        "--robot", type=int, required=True, choices=[2, 3],
        help="Robot number (2 or 3)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input JSON file with matching points (default: robot{N}_matching_points.json)"
    )
    parser.add_argument(
        "--output-frames", type=str, default=None,
        help="Output frames config file (default: robot_configs/world2robot_matrices/robot{N}_matrix.json)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for calibration results (default: extrinsics)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save to frames config, just compute and display"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed per-point errors"
    )
    args = parser.parse_args()

    # Default paths
    base_dir = Path(__file__).parent.parent

    if args.input is None:
        input_path = Path(__file__).parent / "matching_points" / f"robot{args.robot}_matching_points.json"
    else:
        input_path = Path(args.input)

    if args.output_frames is None:
        output_frames_path = base_dir / f"robot_configs/world2robot_matrices/robot{args.robot}_matrix.json"
    else:
        output_frames_path = Path(args.output_frames)

    # Check input file exists
    if not input_path.exists():
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}")
        print()
        print("먼저 find_matching_point.py를 실행하여 매칭점을 수집하세요:")
        print(f"  python world2robot_extrinsic/find_matching_point.py --robot {args.robot}")
        return 1

    # Load matching points
    print("=" * 60)
    print(f"World → Robot Base 변환 행렬 계산: Robot {args.robot}")
    print("=" * 60)
    print()

    with open(input_path, 'r') as f:
        data = json.load(f)

    points = data.get("points", [])
    if len(points) < 3:
        print(f"오류: 최소 3개의 매칭점이 필요합니다 (현재: {len(points)}개)")
        return 1

    print(f"입력 파일: {input_path}")
    print(f"매칭점 수: {len(points)}")
    print()

    # Extract point arrays
    world_points = np.array([p['world'] for p in points])
    robot_base_points = np.array([p['robot_base'] for p in points])

    # Show input points
    print("입력 매칭점:")
    print("-" * 60)
    for i, (w, r) in enumerate(zip(world_points, robot_base_points)):
        print(f"  {i+1}. World({w[0]:+.4f}, {w[1]:+.4f}, {w[2]:+.4f}) -> "
              f"Base({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f})")
    print()

    # Compute transformation using Kabsch algorithm
    print("Kabsch-Umeyama 알고리즘 적용 중...")
    R, t, rmse, T = kabsch_umeyama(world_points, robot_base_points)

    # Extract RPY angles
    roll, pitch, yaw = rotation_matrix_to_rpy(R)

    # Compute detailed error statistics
    error_stats = compute_error_statistics(world_points, robot_base_points, R, t)

    print()
    print("=" * 60)
    print("결과: World → Robot Base 변환")
    print("=" * 60)
    print()

    print("회전 행렬 R:")
    for row in R:
        print(f"  [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")
    print()

    print(f"평행이동 t: [{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}] m")
    print()

    print(f"RPY 각도 (ZYX convention):")
    print(f"  Roll:  {roll:+.4f}°")
    print(f"  Pitch: {pitch:+.4f}°")
    print(f"  Yaw:   {yaw:+.4f}°")
    print()

    print("4x4 동차 변환 행렬 T:")
    for row in T:
        print(f"  [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}, {row[3]:+.6f}]")
    print()

    print("=" * 60)
    print("오차 분석")
    print("=" * 60)
    print()

    print(f"RMSE (전체):    {error_stats['rmse']*1000:.3f} mm")
    print(f"평균 오차:      {error_stats['mean_error']*1000:.3f} mm")
    print(f"최대 오차:      {error_stats['max_error']*1000:.3f} mm")
    print(f"최소 오차:      {error_stats['min_error']*1000:.3f} mm")
    print(f"표준편차:       {error_stats['std_error']*1000:.3f} mm")
    print()

    print("축별 RMSE:")
    print(f"  X: {error_stats['error_per_axis']['x_rmse']*1000:.3f} mm")
    print(f"  Y: {error_stats['error_per_axis']['y_rmse']*1000:.3f} mm")
    print(f"  Z: {error_stats['error_per_axis']['z_rmse']*1000:.3f} mm")
    print()

    if args.verbose:
        print("개별 점 오차:")
        for i, err in enumerate(error_stats['per_point_errors']):
            print(f"  Point {i+1}: {err*1000:.3f} mm")
        print()

    # Evaluate calibration quality
    print("=" * 60)
    print("캘리브레이션 품질 평가")
    print("=" * 60)
    print()

    if error_stats['rmse'] < 0.005:  # < 5mm
        print("  품질: 우수 (RMSE < 5mm)")
    elif error_stats['rmse'] < 0.01:  # < 10mm
        print("  품질: 양호 (RMSE < 10mm)")
    elif error_stats['rmse'] < 0.02:  # < 20mm
        print("  품질: 보통 (RMSE < 20mm)")
        print("  - 더 많은 점을 추가하거나 측정 정확도를 개선하세요")
    else:
        print("  품질: 불량 (RMSE >= 20mm)")
        print("  - 측정 오류가 있을 수 있습니다")
        print("  - 이상치(outlier)가 있는지 확인하세요")

    if error_stats['max_error'] > 2 * error_stats['mean_error']:
        print()
        print("  경고: 최대 오차가 평균의 2배 이상입니다")
        print("        이상치가 있을 수 있으니 데이터를 확인하세요")
    print()

    # Save to frames config
    if not args.no_save:
        print("=" * 60)
        print("프레임 설정 파일 저장")
        print("=" * 60)
        print()

        # For the frames config, we need to store the transformation in a usable format
        # The current transforms.py expects: robot position/rotation in world frame
        # But Kabsch gives us: world-to-base transformation
        #
        # T_world_to_base: transforms points from world to base
        # T_base_from_world = T_world_to_base
        #
        # To get robot position in world frame:
        # T_world_from_base = inv(T_world_to_base)
        # Robot position in world = T_world_from_base[:3, 3]
        # Robot rotation in world = T_world_from_base[:3, :3] -> RPY

        T_world_to_base = T
        T_world_from_base = np.linalg.inv(T_world_to_base)

        robot_position_in_world = T_world_from_base[:3, 3]
        robot_rotation_in_world = T_world_from_base[:3, :3]
        robot_roll, robot_pitch, robot_yaw = rotation_matrix_to_rpy(robot_rotation_in_world)

        frames_config = {
            "robot_id": f"robot{args.robot}",
            "calibrated_at": datetime.now().isoformat(),
            "calibration_source": str(input_path),
            "num_calibration_points": len(points),
            "calibration_rmse_mm": error_stats['rmse'] * 1000,
            "det_R": float(np.linalg.det(R)),
            "frames": {
                "world": {
                    "translation": robot_position_in_world.tolist(),
                    "rotation_rpy": [robot_roll, robot_pitch, robot_yaw],
                    "transform_4x4": T.tolist(),
                }
            },
            "_usage": {
                "translation": "world 좌표계 기준 로봇 위치 [x, y, z] (미터) — RPY용",
                "rotation_rpy": "world 좌표계 기준 로봇 회전 [roll, pitch, yaw] (도) — RPY용",
                "transform_4x4": "World → Base 변환 4x4 행렬 (det=-1 가능, 직접 사용 권장)",
            },
            "_raw_transform": {
                "description": "World → Base 변환 행렬 (직접 사용 시)",
                "rotation_matrix": R.tolist(),
                "translation": t.tolist(),
                "transform_4x4": T.tolist(),
            }
        }

        # Create output directory if needed
        output_frames_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_frames_path, 'w') as f:
            json.dump(frames_config, f, indent=2, ensure_ascii=False)

        print(f"저장됨: {output_frames_path}")
        print()
        print("프레임 설정 내용 (world 기준 로봇 위치):")
        print(f"  Translation: [{robot_position_in_world[0]:.6f}, {robot_position_in_world[1]:.6f}, {robot_position_in_world[2]:.6f}]")
        print(f"  Rotation RPY: [{robot_roll:.4f}, {robot_pitch:.4f}, {robot_yaw:.4f}]°")
        print()

        # Also save detailed calibration results (in output_dir)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(__file__).parent / "extrinsics"
        output_dir.mkdir(parents=True, exist_ok=True)
        detailed_output_path = output_dir / f"robot{args.robot}_calibration_result.json"

        detailed_result = {
            "robot_id": f"robot{args.robot}",
            "calibrated_at": datetime.now().isoformat(),
            "input_file": str(input_path),
            "num_points": len(points),
            "world_to_base_transform": {
                "rotation_matrix": R.tolist(),
                "translation": t.tolist(),
                "rpy_degrees": [roll, pitch, yaw],
                "transform_4x4": T.tolist(),
            },
            "base_to_world_transform": {
                "rotation_matrix": T_world_from_base[:3, :3].tolist(),
                "translation": T_world_from_base[:3, 3].tolist(),
                "rpy_degrees": [robot_roll, robot_pitch, robot_yaw],
                "transform_4x4": T_world_from_base.tolist(),
            },
            "error_statistics": error_stats,
        }

        with open(detailed_output_path, 'w') as f:
            json.dump(detailed_result, f, indent=2, ensure_ascii=False)

        print(f"상세 결과 저장됨: {detailed_output_path}")
        print()

    print("=" * 60)
    print("완료")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
