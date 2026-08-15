#!/usr/bin/env python3
"""Live mosaic view for currently attached workspace cameras.

OpenCV in this environment is built without GUI support, so this serves a
browser-viewable MJPEG stream.
"""

from __future__ import annotations

import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread

import cv2
import numpy as np


CAMERAS = [
    {
        "name": "RealSense 332623022975 color",
        "device": "/dev/video12",
        "fourcc": "YUYV",
    },
    {
        "name": "RealSense 422543021993 color",
        "device": "/dev/video26",
        "fourcc": "YUYV",
    },
    {
        "name": "Innomaker SN0001 wrist",
        "device": "/dev/video14",
        "fourcc": "MJPG",
    },
]


def open_camera(cam: dict) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam["device"])
    if cam.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def annotate(frame: np.ndarray, label: str, fps: float) -> np.ndarray:
    out = cv2.resize(frame, (640, 480))
    cv2.rectangle(out, (0, 0), (640, 70), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(out, f"{fps:.1f} fps", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return out


def blank_tile(label: str) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, label, (35, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, "no frame", (35, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


def main() -> int:
    states = []
    for cam in CAMERAS:
        cap = open_camera(cam)
        if cap.isOpened():
            print(f"[OK] {cam['name']} {cam['device']}", flush=True)
        else:
            print(f"[FAIL] {cam['name']} {cam['device']}", flush=True)
        states.append(
            {
                "cam": cam,
                "cap": cap,
                "timestamps": deque(maxlen=60),
                "fps": 0.0,
                "last": blank_tile(cam["name"]),
            }
        )

    latest_jpeg = {"data": None}
    latest_lock = Lock()
    stop_event = Event()

    def capture_loop() -> None:
        while not stop_event.is_set():
            tiles = []
            for state in states:
                ret, frame = state["cap"].read() if state["cap"].isOpened() else (False, None)
                if ret and frame is not None:
                    now = time.time()
                    state["timestamps"].append(now)
                    ts = state["timestamps"]
                    if len(ts) > 1 and ts[-1] > ts[0]:
                        state["fps"] = (len(ts) - 1) / (ts[-1] - ts[0])
                    state["last"] = annotate(frame, state["cam"]["name"], state["fps"])
                tiles.append(state["last"])

            spacer = np.zeros((480, 640, 3), dtype=np.uint8)
            mosaic = np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], spacer])])
            ok, encoded = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                with latest_lock:
                    latest_jpeg["data"] = encoded.tobytes()
            time.sleep(0.001)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = (
                    "<!doctype html><html><head><title>All Cameras Live</title>"
                    "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}"
                    "h1{font-size:18px;margin:12px}.wrap{display:flex;justify-content:center}"
                    "img{max-width:100vw;max-height:calc(100vh - 48px)}</style></head>"
                    "<body><h1>All Cameras Live</h1><div class='wrap'>"
                    "<img src='/stream.mjpg'></div></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while not stop_event.is_set():
                with latest_lock:
                    frame = latest_jpeg["data"]
                if frame is None:
                    time.sleep(0.05)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError):
                    break

    capture_thread = Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 8090), Handler)
        print("Serving: http://127.0.0.1:8090", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for state in states:
            state["cap"].release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
