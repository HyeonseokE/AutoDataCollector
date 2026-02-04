#!/usr/bin/env python3
"""
Standalone Calibration Script
캘리브레이션만 독립적으로 실행할 때 사용

사용법:
    python calibration/run_calibration.py [--2d]

    --2d: 2D 캘리브레이션만 수행 (depth 미사용)
    기본: 3D 캘리브레이션 (depth 포함)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from camera import RealSenseD435
from calibration import GridCalibrator, DepthCalibrator


def main():
    use_depth = "--2d" not in sys.argv

    print("\n" + "="*60)
    if use_depth:
        print("3D Camera Calibration (with Depth)")
    else:
        print("2D Camera Calibration (Homography only)")
    print("="*60)
    print("\nThis will calibrate the camera to world coordinate system")
    print("using the 1cm grid on the workspace.\n")

    save_path = Path(__file__).parent / "pix2world_transform_data.npz"

    with RealSenseD435() as camera:
        intrinsics = camera.get_intrinsics()
        print(f"Camera intrinsics: {intrinsics}")
        print("\nPress 's' to capture image for calibration")
        print("Press 'q' to quit\n")

        captured_color = None
        captured_depth = None

        while True:
            color, depth = camera.get_frames()
            if color is None:
                continue

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03),
                cv2.COLORMAP_JET
            )

            display = cv2.hconcat([color, depth_colormap])
            mode_text = "3D (with depth)" if use_depth else "2D (homography)"
            cv2.putText(display, f"Mode: {mode_text} | 's': capture, 'q': quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Calibration Preview", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                captured_color = color.copy()
                captured_depth = depth.copy()
                cv2.destroyWindow("Calibration Preview")

                if use_depth:
                    calibrator = DepthCalibrator(grid_size_cm=1.0)
                    success = calibrator.calibrate_with_depth(
                        captured_color, captured_depth, intrinsics
                    )
                else:
                    calibrator = GridCalibrator(grid_size_cm=1.0)
                    success = calibrator.calibrate_interactive(captured_color)

                if success:
                    calibrator.save(str(save_path))
                    print(f"\n[Success] Calibration saved to: {save_path}")

                    if use_depth and hasattr(calibrator, 'transform_matrix'):
                        print("\n4x4 Transform Matrix (Camera -> World):")
                        print(calibrator.transform_matrix)

                    print("\n" + "-"*40)
                    print("Verification Test")
                    print("-"*40)
                    print("Click on known grid points to verify accuracy")
                    print("Press 'q' to finish\n")

                    def verify_callback(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            if use_depth and calibrator.is_3d_calibrated:
                                depth_m = captured_depth[y, x] * 0.001
                                if depth_m > 0:
                                    world = calibrator.pixel_depth_to_world(x, y, depth_m)
                                    print(f"Pixel ({x}, {y}), depth={depth_m:.3f}m -> World ({world[0]:.2f}, {world[1]:.2f}, {world[2]:.2f}) cm")
                                else:
                                    print(f"Pixel ({x}, {y}) - invalid depth")
                            else:
                                world = calibrator.pixel_to_world(x, y)
                                print(f"Pixel ({x}, {y}) -> World ({world[0]:.2f}, {world[1]:.2f}, {world[2]:.2f}) cm")

                    cv2.namedWindow("Verify")
                    cv2.setMouseCallback("Verify", verify_callback)

                    while True:
                        display_verify = captured_color.copy()
                        cv2.putText(display_verify, "Click to verify, 'q' to quit", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Verify", display_verify)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cv2.destroyAllWindows()
                break

            elif key == ord('q'):
                break

    cv2.destroyAllWindows()
    print("\nCalibration complete!")


if __name__ == "__main__":
    main()
