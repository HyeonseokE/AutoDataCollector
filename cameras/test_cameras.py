#!/usr/bin/env python3
"""
Camera Connection Test Script

연결된 카메라를 검색하고 테스트하는 스크립트
LeRobot 공식 구현과 동일한 async_read 방식 테스트 포함

Usage:
    # 모든 카메라 검색
    python cameras/test_cameras.py --find

    # 특정 카메라 테스트
    python cameras/test_cameras.py --test realsense
    python cameras/test_cameras.py --test opencv --device /dev/video6

    # 기본 설정으로 멀티 카메라 테스트
    python cameras/test_cameras.py --multi

    # async_read 성능 테스트 (LeRobot 공식 방식)
    python cameras/test_cameras.py --async

    # 라이브 뷰 (OpenCV 창에서 실시간 확인)
    python cameras/test_cameras.py --live
"""

import argparse
import time
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def find_all_cameras():
    """모든 연결된 카메라 검색"""
    print("\n" + "=" * 60)
    print("Camera Search")
    print("=" * 60)

    # RealSense 카메라 검색
    print("\n[1/2] Searching for RealSense cameras...")
    try:
        from cameras.realsense_camera import RealSenseCamera
        realsense_cams = RealSenseCamera.find_cameras()
        if realsense_cams:
            for cam in realsense_cams:
                print(f"  [OK] {cam['name']} (serial: {cam['serial']})")
        else:
            print("  No RealSense cameras found")
    except ImportError as e:
        print(f"  [SKIP] pyrealsense2 not installed: {e}")

    # OpenCV 카메라 검색
    print("\n[2/2] Searching for OpenCV cameras...")
    try:
        from cameras.opencv_camera import OpenCVCamera
        opencv_cams = OpenCVCamera.find_cameras()
        if opencv_cams:
            for cam in opencv_cams:
                print(f"  [OK] {cam['name']}: {cam['width']}x{cam['height']}@{cam['fps']:.1f}fps")
        else:
            print("  No OpenCV cameras found")
    except ImportError as e:
        print(f"  [SKIP] OpenCV not installed: {e}")

    print("\n" + "=" * 60)


def test_realsense():
    """RealSense 카메라 테스트"""
    print("\n" + "=" * 60)
    print("RealSense Camera Test")
    print("=" * 60)

    from cameras import RealSenseCamera, RealSenseCameraConfig

    config = RealSenseCameraConfig(
        name="test_realsense",
        width=640,
        height=480,
        fps=30,
    )

    camera = RealSenseCamera(config)

    try:
        print("\n[1/4] Connecting...")
        camera.connect()
        print(f"  Info: {camera.get_info()}")

        print("\n[2/4] Sync read test...")
        for i in range(3):
            frame = camera.read()
            print(f"  Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}")
            time.sleep(0.1)

        print("\n[3/4] Async read test (LeRobot style)...")
        for i in range(5):
            start = time.perf_counter()
            frame = camera.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            print(f"  Frame {i+1}: shape={frame.shape}, latency={dt_ms:.1f}ms")
            time.sleep(0.05)

        print("\n[4/4] Disconnecting...")
        camera.disconnect()

        print("\n[OK] RealSense test passed!")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        camera.disconnect()
        return False

    return True


def test_opencv(device: str = "/dev/video6"):
    """OpenCV 카메라 테스트"""
    print("\n" + "=" * 60)
    print(f"OpenCV Camera Test ({device})")
    print("=" * 60)

    from cameras import OpenCVCamera, OpenCVCameraConfig

    config = OpenCVCameraConfig(
        name="test_opencv",
        device_path=device,
        width=640,
        height=480,
        fps=30,
        fourcc="MJPG",
    )

    camera = OpenCVCamera(config)

    try:
        print("\n[1/4] Connecting...")
        camera.connect()
        print(f"  Info: {camera.get_info()}")

        print("\n[2/4] Sync read test...")
        for i in range(3):
            frame = camera.read()
            print(f"  Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}")
            time.sleep(0.1)

        print("\n[3/4] Async read test (LeRobot style)...")
        for i in range(5):
            start = time.perf_counter()
            frame = camera.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            print(f"  Frame {i+1}: shape={frame.shape}, latency={dt_ms:.1f}ms")
            time.sleep(0.05)

        print("\n[4/4] Disconnecting...")
        camera.disconnect()

        print("\n[OK] OpenCV test passed!")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        camera.disconnect()
        return False

    return True


def test_multi_camera():
    """멀티 카메라 테스트"""
    print("\n" + "=" * 60)
    print("Multi-Camera Test")
    print("=" * 60)

    from cameras import MultiCameraManager, RealSenseCameraConfig, OpenCVCameraConfig

    configs = [
        RealSenseCameraConfig(
            name="realsense",
            width=640,
            height=480,
            fps=30,
        ),
        OpenCVCameraConfig(
            name="innomaker",
            device_path="/dev/video6",
            width=640,
            height=480,
            fps=30,
        ),
    ]

    manager = MultiCameraManager(configs)

    try:
        print("\n[1/5] Connecting all cameras...")
        manager.connect_all()

        print("\n[2/5] Camera info:")
        for name, info in manager.get_info().items():
            print(f"  {name}: {info}")

        print("\n[3/5] Sync read test (read_all)...")
        for i in range(3):
            images = manager.read_all()
            shapes = {name: img.shape for name, img in images.items()}
            print(f"  Frame {i+1}: {shapes}")
            time.sleep(0.1)

        print("\n[4/5] Async read test (LeRobot style - get_observation)...")
        for i in range(5):
            start = time.perf_counter()
            obs = manager.get_observation()
            dt_ms = (time.perf_counter() - start) * 1e3
            shapes = {k: v.shape for k, v in obs.items()}
            print(f"  Frame {i+1}: latency={dt_ms:.1f}ms, {shapes}")
            time.sleep(0.05)

        print("\n[5/5] Disconnecting...")
        manager.disconnect_all()

        print("\n[OK] Multi-camera test passed!")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        manager.disconnect_all()
        return False

    return True


def test_async_performance():
    """async_read 성능 테스트 (LeRobot 공식 방식)"""
    print("\n" + "=" * 60)
    print("Async Read Performance Test (LeRobot Style)")
    print("=" * 60)

    from cameras import MultiCameraManager, RealSenseCameraConfig, OpenCVCameraConfig

    configs = [
        RealSenseCameraConfig(
            name="realsense",
            width=640,
            height=480,
            fps=30,
        ),
        OpenCVCameraConfig(
            name="innomaker",
            device_path="/dev/video6",
            width=640,
            height=480,
            fps=30,
        ),
    ]

    manager = MultiCameraManager(configs)

    try:
        manager.connect_all()

        print("\n[Test] 100 iterations of get_observation()...")
        print("  (This simulates LeRobot's observation loop)")

        latencies = []
        for i in range(100):
            start = time.perf_counter()
            obs = manager.get_observation()
            dt_ms = (time.perf_counter() - start) * 1e3
            latencies.append(dt_ms)

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/100 completed...")

            time.sleep(0.01)  # ~100Hz loop

        latencies = np.array(latencies)
        print("\n[Results]")
        print(f"  Mean latency:   {latencies.mean():.2f} ms")
        print(f"  Std latency:    {latencies.std():.2f} ms")
        print(f"  Min latency:    {latencies.min():.2f} ms")
        print(f"  Max latency:    {latencies.max():.2f} ms")
        print(f"  Median latency: {np.median(latencies):.2f} ms")

        # 실제 달성 가능한 FPS 추정
        estimated_fps = 1000.0 / latencies.mean()
        print(f"\n  Estimated achievable FPS: {estimated_fps:.1f}")

        manager.disconnect_all()
        print("\n[OK] Async performance test passed!")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        manager.disconnect_all()
        return False

    return True


def live_view():
    """실시간 카메라 뷰"""
    if not HAS_CV2:
        print("[ERROR] OpenCV not installed for display")
        return

    print("\n" + "=" * 60)
    print("Live Camera View (LeRobot Style - async_read)")
    print("Press 'q' to quit")
    print("=" * 60)

    from cameras import MultiCameraManager, RealSenseCameraConfig, OpenCVCameraConfig

    configs = [
        RealSenseCameraConfig(
            name="realsense",
            width=640,
            height=480,
            fps=30,
        ),
        OpenCVCameraConfig(
            name="innomaker",
            device_path="/dev/video6",
            width=640,
            height=480,
            fps=30,
        ),
    ]

    manager = MultiCameraManager(configs)

    try:
        manager.connect_all()

        frame_count = 0
        start_time = time.time()

        while True:
            # LeRobot 방식: async_read로 observation 가져오기
            obs = manager.get_observation()

            if not obs:
                continue

            # 이미지 합치기
            frames = []
            for key in sorted(obs.keys()):
                img = obs[key]
                # RGB -> BGR (OpenCV display)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # 카메라 이름 표시
                cam_name = key.replace("observation.images.", "")
                cv2.putText(bgr, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)

                frames.append(bgr)

            if len(frames) == 1:
                display = frames[0]
            else:
                # 가로로 연결
                display = np.hstack(frames)

            # FPS 표시
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Multi-Camera View (async_read)", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    finally:
        manager.disconnect_all()


def save_test_images():
    """테스트 이미지 저장"""
    print("\n" + "=" * 60)
    print("Saving Test Images")
    print("=" * 60)

    from cameras import MultiCameraManager, RealSenseCameraConfig, OpenCVCameraConfig
    from PIL import Image

    configs = [
        RealSenseCameraConfig(
            name="realsense",
            width=640,
            height=480,
            fps=30,
        ),
        OpenCVCameraConfig(
            name="innomaker",
            device_path="/dev/video6",
            width=640,
            height=480,
            fps=30,
        ),
    ]

    output_dir = Path("outputs/camera_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = MultiCameraManager(configs)

    try:
        manager.connect_all()

        # async_read로 안정화된 프레임 캡처
        time.sleep(0.5)
        for _ in range(10):
            manager.get_observation()  # 백그라운드 스레드 안정화
            time.sleep(0.05)

        obs = manager.get_observation()

        for key, img in obs.items():
            cam_name = key.replace("observation.images.", "")
            path = output_dir / f"{cam_name}.png"
            Image.fromarray(img).save(path)
            print(f"  Saved: {path}")

        print(f"\n[OK] Images saved to {output_dir}")

    finally:
        manager.disconnect_all()


def main():
    parser = argparse.ArgumentParser(description="Camera Connection Test")

    parser.add_argument("--find", action="store_true",
                       help="Find all connected cameras")
    parser.add_argument("--test", type=str, choices=["realsense", "opencv"],
                       help="Test specific camera type")
    parser.add_argument("--device", type=str, default="/dev/video6",
                       help="Device path for OpenCV camera")
    parser.add_argument("--multi", action="store_true",
                       help="Test multi-camera setup")
    parser.add_argument("--async", dest="async_test", action="store_true",
                       help="Test async_read performance (LeRobot style)")
    parser.add_argument("--live", action="store_true",
                       help="Live camera view")
    parser.add_argument("--save", action="store_true",
                       help="Save test images")

    args = parser.parse_args()

    if args.find:
        find_all_cameras()
    elif args.test == "realsense":
        test_realsense()
    elif args.test == "opencv":
        test_opencv(args.device)
    elif args.multi:
        test_multi_camera()
    elif args.async_test:
        test_async_performance()
    elif args.live:
        live_view()
    elif args.save:
        save_test_images()
    else:
        # 기본: 모든 카메라 검색
        find_all_cameras()


if __name__ == "__main__":
    main()
