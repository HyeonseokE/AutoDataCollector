"""
Live streaming for the two WS2 cameras (matches recording_config_ws2.yaml).

  - wrist : OpenCV     /dev/video6   (Innomaker, MJPG 640x480@30)
  - top   : RealSense  254622079503  (color stream, YUYV 640x480@30
                                       exposed at /dev/video12)

The conda ``lerobot`` env's OpenCV is built with ``GUI: NONE``, so we
launch each stream in its own ``ffplay`` window instead of ``cv2.imshow``.
Close the windows (or hit Ctrl-C in the terminal) to stop.
"""

import shutil
import signal
import subprocess
import sys

WRIST_DEV = "/dev/video6"

REALSENSE_SERIAL = "254622079503"
REALSENSE_COLOR_DEV = "/dev/video12"

WIDTH, HEIGHT, FPS = 640, 480, 30


def _spawn(cmd, title):
    print(f"[run] {title}:  {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def main():
    if shutil.which("ffplay") is None:
        sys.exit("[ERROR] ffplay not found — run inside conda env 'lerobot'.")

    procs = []

    # --- WS2 wrist (Innomaker, MJPG) ---
    procs.append(_spawn(
        [
            "ffplay", "-hide_banner", "-loglevel", "warning",
            "-window_title", f"WS2 wrist ({WRIST_DEV})",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", str(FPS),
            WRIST_DEV,
        ],
        title=f"wrist {WRIST_DEV}",
    ))

    # --- WS2 top (RealSense color stream, YUYV via V4L2) ---
    procs.append(_spawn(
        [
            "ffplay", "-hide_banner", "-loglevel", "warning",
            "-window_title", f"WS2 top (RealSense {REALSENSE_SERIAL})",
            "-f", "v4l2",
            "-input_format", "yuyv422",
            "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", str(FPS),
            REALSENSE_COLOR_DEV,
        ],
        title=f"top RealSense {REALSENSE_SERIAL} @ {REALSENSE_COLOR_DEV}",
    ))

    def _stop(*_):
        for p in procs:
            if p.poll() is None:
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
