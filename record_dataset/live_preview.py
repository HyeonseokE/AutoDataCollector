"""Live side-by-side camera preview for cycle progress monitoring.

Polls ``RecordingContext._async_capture.get_latest_images()`` on its own clock
and shows a single OpenCV window with the requested cameras concatenated
horizontally. Independent from RecordingContext / inference loop — touches no
control-path state. Frames are read via AsyncCameraCapture's thread-safe getter
(returns a copy), so concurrent reads from this preview and the dataset writer
do not race.

Non-recording mode (RECORD_DATASET=false) is also supported: the pipeline's
``_init_live_preview_camera`` spawns a long-lived AsyncCameraCapture and attaches
it to ``RecordingContext._async_capture`` so this poller works the same way.

Usage::

    pv = LivePreviewWindow(camera_names=["top", "left_wrist"], fps=10)
    pv.start()
    ...           # cycle runs; RecordingContext.setup() initializes _async_capture later
    pv.stop()     # idempotent

Env-driven toggle is done at execution_forward_and_reset.main() — this module
only provides the class.
"""

from __future__ import annotations

import time
from threading import Event, Thread
from typing import Optional

import numpy as np


class LivePreviewWindow:
    """Background thread — side-by-side cv2 window of selected cameras."""

    def __init__(
        self,
        camera_names: list[str],
        fps: float = 10.0,
        window_name: str = "Live Preview (top + wrist)",
        scale: float = 1.0,
    ):
        self.camera_names = list(camera_names)
        self.fps = max(float(fps), 1.0)
        self.window_name = window_name
        self.scale = float(scale)
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="LivePreviewWindow")
        self._thread.start()
        print(
            f"[LivePreview] start cameras={self.camera_names} @ {self.fps}fps "
            f"scale={self.scale}x",
            flush=True,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        try:
            import cv2
        except Exception as e:
            print(f"[LivePreview] cv2 unavailable ({e}); preview disabled", flush=True)
            return

        # late import to avoid cycle at module import time
        from record_dataset.context import RecordingContext

        period = 1.0 / self.fps
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[LivePreview] namedWindow failed ({e}); preview disabled", flush=True)
            return

        try:
            while not self._stop_event.is_set():
                started = time.perf_counter()
                cap = getattr(RecordingContext, "_async_capture", None)
                if cap is None:
                    # AsyncCameraCapture not initialized yet — wait and retry.
                    try:
                        cv2.waitKey(1)
                    except Exception:
                        pass
                    time.sleep(period)
                    continue
                try:
                    imgs = cap.get_latest_images()
                except Exception as e:
                    print(f"[LivePreview] get_latest_images error: {e}", flush=True)
                    time.sleep(period)
                    continue
                if not imgs:
                    cv2.waitKey(1)
                    time.sleep(period)
                    continue
                frames = []
                target_h: Optional[int] = None
                for cam in self.camera_names:
                    img = imgs.get(cam)
                    if img is None:
                        continue
                    img = np.asarray(img)
                    # Assume RGB from camera_manager → cv2 wants BGR.
                    if img.ndim == 3 and img.shape[-1] == 3:
                        img = img[..., ::-1]
                    if target_h is None:
                        target_h = int(img.shape[0])
                    elif int(img.shape[0]) != target_h:
                        h, w = img.shape[:2]
                        ratio = target_h / float(h)
                        img = cv2.resize(img, (int(w * ratio), target_h))
                    frames.append(np.ascontiguousarray(img))
                if frames:
                    concat = np.hstack(frames)
                    if self.scale != 1.0:
                        h, w = concat.shape[:2]
                        concat = cv2.resize(
                            concat,
                            (int(w * self.scale), int(h * self.scale)),
                        )
                    cv2.imshow(self.window_name, concat)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self._stop_event.set()
                    break
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
