"""
캘리브레이션용 카메라 래퍼 — serial 지정 + 기기 열거.

기존 object_detection.camera.RealSenseD435 는 serial 지정을 지원하지 않아
독립 래퍼를 작성. 인터페이스는 동일 (get_frames, context manager)하므로
phase 모듈은 이것을 import해서 사용.
"""

from typing import List, Optional, Tuple

import numpy as np


def list_realsense_devices() -> List[Tuple[str, str]]:
    """
    연결된 RealSense 기기 열거.

    Returns:
        list of (serial, name). 예: [("254622079503", "Intel RealSense D435"), ...]
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []

    ctx = rs.context()
    out: List[Tuple[str, str]] = []
    for dev in ctx.query_devices():
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            out.append((serial, name))
        except Exception:
            continue
    return out


class CalibrationCamera:
    """
    RealSense D4xx 카메라 래퍼 (color + aligned depth).

    기존 RealSenseD435 와 동일 인터페이스(get_frames, __enter__/__exit__)를 가지되,
    serial 지정과 기기 자동 선택(단일 연결 시)을 지원.

    Args:
        serial: 사용할 카메라 일련번호 (None이면 단일 기기 자동 선택)
        width, height, fps: 스트림 설정
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise ImportError(f"pyrealsense2 import 실패: {e}")
        self._rs = rs

        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.intrinsics = None
        self._is_running = False

    def start(self) -> None:
        """
        파이프라인 시작.
        실패 시 자동 처리:
            - Frame timeout → 하드웨어 리셋 후 재시도
            - Device busy (errno=16) → 다른 프로세스 점유 안내 + 하드웨어 리셋 후 재시도
        """
        rs = self._rs
        resolved_serial = self._resolve_serial()

        for attempt in range(2):
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                if resolved_serial:
                    config.enable_device(resolved_serial)
                config.enable_stream(
                    rs.stream.color, self.width, self.height,
                    rs.format.bgr8, self.fps,
                )
                config.enable_stream(
                    rs.stream.depth, self.width, self.height,
                    rs.format.z16, self.fps,
                )
                profile = pipeline.start(config)
                pipeline.wait_for_frames(timeout_ms=5000)

                self.pipeline = pipeline
                self.align = rs.align(rs.stream.color)
                color_stream = profile.get_stream(rs.stream.color)
                self.intrinsics = (
                    color_stream.as_video_stream_profile().get_intrinsics()
                )
                self._is_running = True
                tag = f"serial={resolved_serial}" if resolved_serial else "default"
                print(
                    f"[Camera] Started ({tag}): "
                    f"{self.width}x{self.height} @ {self.fps}fps"
                )
                return
            except RuntimeError as e:
                msg = str(e)
                is_busy = "errno=16" in msg or "Device or resource busy" in msg or "동작 중" in msg
                is_timeout = "Frame didn't arrive" in msg or "wait_for_frames" in msg

                if attempt == 0 and (is_busy or is_timeout):
                    if is_busy:
                        print(
                            f"[Camera] Device busy (errno=16) — 다른 프로세스가 카메라를 점유 중일 수 있습니다."
                        )
                        self._diagnose_busy(resolved_serial)
                    else:
                        print(f"[Camera] Frame timeout, hardware reset 시도...")
                    self._hardware_reset(resolved_serial)
                    continue
                if is_busy:
                    self._diagnose_busy(resolved_serial)
                raise

    @staticmethod
    def _diagnose_busy(serial: Optional[str]):
        """카메라 점유 프로세스를 찾아서 사용자에게 알림."""
        import subprocess
        try:
            result = subprocess.run(
                ["fuser", "-v", "/dev/video0", "/dev/video1", "/dev/video2",
                 "/dev/video3", "/dev/video4", "/dev/video5", "/dev/video6",
                 "/dev/video7", "/dev/video8", "/dev/video9", "/dev/video10",
                 "/dev/video11", "/dev/video12", "/dev/video13"],
                capture_output=True, text=True, timeout=3,
            )
            output = (result.stderr or "") + (result.stdout or "")
            if output.strip():
                print("  현재 /dev/video* 점유 프로세스:")
                for line in output.strip().splitlines():
                    print(f"    {line}")
                print("  점유 프로세스가 본 캘리브와 무관하면 다음으로 종료 가능:")
                print("    kill -9 <PID>")
        except Exception:
            pass

    def _resolve_serial(self) -> Optional[str]:
        """serial이 None이면 단일 기기 자동 선택. 다중 기기면 에러."""
        if self.serial:
            return self.serial
        devices = list_realsense_devices()
        if len(devices) == 0:
            raise RuntimeError("연결된 RealSense 기기 없음")
        if len(devices) == 1:
            print(f"[Camera] 단일 기기 자동 선택: {devices[0][0]} ({devices[0][1]})")
            return devices[0][0]
        # 다중 기기 — 명시적 serial 필요
        listing = "\n".join(f"  - {s} : {n}" for s, n in devices)
        raise RuntimeError(
            f"여러 RealSense 기기가 연결됨. --camera-serial로 지정 필요:\n{listing}"
        )

    def _hardware_reset(self, serial: Optional[str]):
        import time
        rs = self._rs
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        ctx = rs.context()
        for dev in ctx.query_devices():
            try:
                if serial and dev.get_info(rs.camera_info.serial_number) != serial:
                    continue
                dev.hardware_reset()
                break
            except Exception:
                continue
        time.sleep(5)

    def stop(self) -> None:
        if self._is_running and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self._is_running = False
            print("[Camera] Stopped")

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns:
            (color BGR (H,W,3), depth mm uint16 (H,W))
        """
        if not self._is_running or self.pipeline is None:
            return None, None
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        return (
            np.asanyarray(color_frame.get_data()),
            np.asanyarray(depth_frame.get_data()),
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def get_factory_intrinsics(
    serial: Optional[str] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> dict:
    """
    D435의 공장 캘리브 K, dist를 직접 쿼리.

    D435는 출고 시 캘리브 되어 있어 일반적으로 정확함.
    Charuco intrinsics가 부정확할 때 안전한 대안.

    Returns:
        dict with 'K' (3x3), 'dist' (5,), 'depth_scale', 'image_size', 'distortion_model'
    """
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise ImportError(f"pyrealsense2 import 실패: {e}")

    # serial 결정 (CalibrationCamera 와 동일 로직)
    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError("연결된 RealSense 기기 없음")
    if serial is None:
        if len(devices) > 1:
            listing = "\n".join(f"  - {s} : {n}" for s, n in devices)
            raise RuntimeError(
                f"여러 RealSense 기기 — serial 명시 필요:\n{listing}"
            )
        serial = devices[0][0]

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    try:
        sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(sensor.get_depth_scale())
        color_intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics()
        )
    finally:
        pipeline.stop()

    K = np.array([
        [color_intr.fx, 0.0,           color_intr.ppx],
        [0.0,           color_intr.fy, color_intr.ppy],
        [0.0,           0.0,           1.0],
    ], dtype=np.float64)
    dist = np.array(color_intr.coeffs, dtype=np.float64)

    return {
        "K": K,
        "dist": dist,
        "depth_scale": depth_scale,
        "image_size": (width, height),
        "distortion_model": str(color_intr.model),
        "serial": serial,
    }


def validate_intrinsics(K: np.ndarray, dist: np.ndarray, image_size) -> list:
    """
    K, dist 가 합리적인 범위인지 sanity check.

    Returns:
        warning 메시지 리스트 (비어있으면 OK).
    """
    warnings = []
    w, h = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # focal length: 640x480 이면 보통 400~800 범위
    expected_f_min = w * 0.5      # 매우 광각
    expected_f_max = w * 2.0      # 매우 망원
    if not (expected_f_min <= fx <= expected_f_max):
        warnings.append(
            f"fx={fx:.1f} 가 예상 범위 [{expected_f_min:.0f}, "
            f"{expected_f_max:.0f}] 밖. 보드 자세 다양성 부족 가능."
        )
    if not (expected_f_min <= fy <= expected_f_max):
        warnings.append(
            f"fy={fy:.1f} 가 예상 범위 [{expected_f_min:.0f}, "
            f"{expected_f_max:.0f}] 밖."
        )
    if abs(fx - fy) / max(fx, fy) > 0.1:
        warnings.append(f"fx/fy 비율 비정상: fx={fx:.1f}, fy={fy:.1f} (>10% 차이)")

    # principal point: 보통 이미지 중심 ±20%
    if abs(cx - w / 2) > w * 0.2:
        warnings.append(f"cx={cx:.1f} 가 이미지 중심에서 너무 벗어남")
    if abs(cy - h / 2) > h * 0.2:
        warnings.append(f"cy={cy:.1f} 가 이미지 중심에서 너무 벗어남")

    # 왜곡 계수: 보통 |k1|, |k2| < 1.0, |k3| < 5.0
    if len(dist) >= 5:
        k1, k2, p1, p2, k3 = dist[:5]
        if abs(k1) > 1.0:
            warnings.append(f"k1={k1:.3f} 비정상 (|k1|>1.0)")
        if abs(k2) > 5.0:
            warnings.append(f"k2={k2:.3f} 비정상 (|k2|>5.0)")
        if abs(k3) > 50.0:
            warnings.append(
                f"k3={k3:.3f} 매우 비정상 — 거의 확실히 잘못된 캘리브."
            )
    return warnings


def print_available_cameras():
    """연결된 RealSense 기기를 터미널에 출력 (사용자 확인용)."""
    devices = list_realsense_devices()
    if not devices:
        print("  연결된 RealSense 기기 없음")
        return
    print(f"  연결된 RealSense 기기 ({len(devices)}개):")
    for s, n in devices:
        print(f"    - serial={s}  name={n}")
