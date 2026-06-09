#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Demo script showing how to use Real-Time Chunking (RTC) with action chunking policies on real robots.

This script demonstrates:
1. Creating a robot and policy (SmolVLA, Pi0, etc.) with RTC
2. Consuming actions from the policy while the robot executes
3. Periodically requesting new action chunks in the background using threads
4. Managing action buffers and timing for real-time operation

For simulation environments, see eval_with_simulation.py

Usage:
    # Run RTC with Real robot with RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=<USER>/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with Real robot without RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=<USER>/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=false \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with Real robot with pi0.5 policy
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=<USER>/pi05_check_rtc \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120
"""

import json
import logging
import math
import pickle
import re
import struct
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    so_follower,
    unitree_g1,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


class InterruptController:
    """Enter-key driven park/resume state shared by the worker threads.

    Two-state toggle:
    - 1st Enter: park — flush the action buffer, move the robot to free_state,
      and hold there (inference paused).
    - 2nd Enter: resume — the first chunk produced after resume is discarded so
      execution restarts cleanly, then chunks are merged normally.

    Parked time is accumulated so it can be excluded from the demo duration.
    """

    def __init__(self):
        self.parked = Event()  # set => robot parked at free_state, threads paused
        self._lock = Lock()
        self._discard_count = 0
        self._park_started: float | None = None
        self._parked_total = 0.0

    def set_parked(self) -> None:
        with self._lock:
            if not self.parked.is_set():
                self._park_started = time.time()
        self.parked.set()

    def set_running(self) -> None:
        with self._lock:
            if self.parked.is_set() and self._park_started is not None:
                self._parked_total += time.time() - self._park_started
                self._park_started = None
        self.parked.clear()

    def arm_discard(self, n: int = 1) -> None:
        """Mark the next ``n`` produced chunks to be discarded."""
        with self._lock:
            self._discard_count = n

    def consume_discard(self) -> bool:
        """Return True (and decrement) if the current chunk should be discarded."""
        with self._lock:
            if self._discard_count > 0:
                self._discard_count -= 1
                return True
            return False

    def parked_total(self) -> float:
        """Total seconds spent parked, including an in-progress park."""
        with self._lock:
            ongoing = time.time() - self._park_started if self._park_started is not None else 0.0
            return self._parked_total + ongoing

    def wait_while_parked(self, shutdown_event: Event, poll: float = 0.05) -> None:
        while self.parked.is_set() and not shutdown_event.is_set():
            time.sleep(poll)


_TK_PREVIEW_SCRIPT = r"""
import pickle
import struct
import sys
import threading
import tkinter as tk

from PIL import Image, ImageTk


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


window_name = sys.argv[1]
fps = float(sys.argv[2])
latest = {"frame": None, "closed": False, "photo": None, "shown_first_frame": False}
lock = threading.Lock()


def reader():
    try:
        stream = sys.stdin.buffer
        while True:
            header = read_exact(stream, 4)
            if header is None:
                break
            size = struct.unpack("!I", header)[0]
            if size == 0:
                break
            payload = read_exact(stream, size)
            if payload is None:
                break
            frame = pickle.loads(payload)
            with lock:
                latest["frame"] = frame
    finally:
        with lock:
            latest["closed"] = True


root = tk.Tk()
root.title(window_name)
root.geometry("640x480+60+60")
root.attributes("-topmost", True)
root.lift()
root.focus_force()
label = tk.Label(root)
label.pack()
period_ms = max(1, int(1000.0 / max(fps, 1.0)))


def close():
    with lock:
        latest["closed"] = True
    try:
        root.destroy()
    except tk.TclError:
        pass


root.protocol("WM_DELETE_WINDOW", close)


def update():
    with lock:
        frame = latest["frame"]
        latest["frame"] = None
        closed = latest["closed"]

    if frame is not None:
        image = Image.fromarray(frame if frame.ndim == 2 else frame[..., :3])
        latest["photo"] = ImageTk.PhotoImage(image=image)
        label.configure(image=latest["photo"])
        if not latest["shown_first_frame"]:
            root.geometry(f"{image.width}x{image.height}+60+60")
            root.lift()
            root.focus_force()
            latest["shown_first_frame"] = True

    if closed:
        close()
        return

    root.after(period_ms, update)


threading.Thread(target=reader, daemon=True).start()
root.after(0, update)
root.mainloop()
"""


class PauseCameraWindow:
    """Small local preview window shown while the robot is parked."""

    def __init__(self, robot: "RobotWrapper", camera: str, shutdown_event: Event, fps: float = 15.0, scale: float = 1.0):
        self.robot = robot
        self.camera = camera
        self.shutdown_event = shutdown_event
        self.fps = fps
        self.scale = float(scale)
        self.window_name = f"Pause stream: {camera}"
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="PauseCameraWindow")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("[PAUSE_STREAM] Preview thread did not stop within timeout")
            else:
                self._thread = None

    def _select_frame(self, obs: dict) -> np.ndarray | None:
        return self.select_camera_frame(obs, self.camera)

    @staticmethod
    def select_camera_frame(obs: dict, camera: str) -> np.ndarray | None:
        """Shared helper — pick a camera frame from obs by name. Used by both
        PauseCameraWindow (single-cam preview) and CameraVideoRecorder
        (multi-cam mp4 logger).
        """
        candidates = [
            camera,
            f"{OBS_IMAGES}.{camera}",
            f"observation.images.{camera}",
        ]
        for key in candidates:
            if key in obs:
                return PauseCameraWindow._to_rgb(obs[key])

        suffix = f".{camera}"
        for key, value in obs.items():
            if key.endswith(suffix) or key.endswith(f"images.{camera}"):
                return PauseCameraWindow._to_rgb(value)
        return None

    @staticmethod
    def _to_rgb(frame) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            max_v = float(np.nanmax(frame)) if frame.size else 1.0
            if max_v <= 1.5:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            return frame
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame.copy()

    def _run(self) -> None:
        # Prefer the subprocess preview: OpenCV HighGUI/Qt windows are fragile
        # when repeatedly created and destroyed from this background thread.
        try:
            self._run_tkinter()
        except Exception as exc:
            logger.warning(f"[PAUSE_STREAM] Tkinter preview unavailable ({exc}); trying OpenCV preview")
            try:
                self._run_opencv()
            except Exception as cv_exc:
                logger.warning(f"[PAUSE_STREAM] OpenCV preview stopped ({cv_exc})")

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        if self.scale == 1.0:
            return frame
        import cv2

        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * self.scale), int(h * self.scale)))

    def _run_opencv(self) -> None:
        import cv2

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        period = 1.0 / max(self.fps, 1.0)
        while not self.shutdown_event.is_set() and not self._stop_event.is_set():
            started = time.perf_counter()
            obs = self.robot.get_observation()
            frame = self._select_frame(obs)
            if frame is not None:
                frame = self._resize_for_display(frame)
                if frame.ndim == 3:
                    frame = frame[..., ::-1]
                cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._stop_event.set()
                break
            time.sleep(max(0.0, period - (time.perf_counter() - started)))
        cv2.destroyWindow(self.window_name)

    def _run_tkinter(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", _TK_PREVIEW_SCRIPT, self.window_name, str(self.fps)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"[PAUSE_STREAM] Tkinter preview subprocess started pid={process.pid}")
        period = 1.0 / max(self.fps, 1.0)
        sent_first_frame = False

        try:
            while (
                not self.shutdown_event.is_set()
                and not self._stop_event.is_set()
                and process.poll() is None
            ):
                started = time.perf_counter()
                obs = self.robot.get_observation()
                frame = self._select_frame(obs)
                if frame is not None:
                    frame = self._resize_for_display(frame)
                    try:
                        payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
                        assert process.stdin is not None
                        process.stdin.write(struct.pack("!I", len(payload)))
                        process.stdin.write(payload)
                        process.stdin.flush()
                        if not sent_first_frame:
                            logger.info(
                                f"[PAUSE_STREAM] Sent first frame to preview window "
                                f"(shape={getattr(frame, 'shape', None)})"
                            )
                            sent_first_frame = True
                    except (BrokenPipeError, OSError):
                        break
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
        finally:
            try:
                if process.stdin is not None:
                    process.stdin.write(struct.pack("!I", 0))
                    process.stdin.close()
            except Exception:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
            if process.returncode not in (None, 0, -2):
                logger.warning(f"[PAUSE_STREAM] Tkinter preview subprocess exited with code {process.returncode}")


class CameraVideoRecorder:
    """Background thread that records selected camera frames to per-cam **H.264 mp4**.

    Independent from PauseCameraWindow — runs for the *entire* inference session
    (not only Enter-park). Each camera gets its own ``<out_dir>/<cam>.mp4`` written
    via an ``ffmpeg`` subprocess pipe (raw BGR → libx264 + +faststart) so the
    resulting mp4 plays in VSCode / Chromium without post-processing.

    Why ffmpeg pipe (not cv2.VideoWriter): OpenCV's conda build only exposes the
    ``mp4v`` fourcc (= MPEG-4 part 2), which VSCode's Chromium player refuses.
    H264/avc1 fourccs fall back to ``h264_v4l2m2m`` and fail without a V4L2
    encoder. ffmpeg with libx264 sidesteps both issues.

    Failure modes are non-fatal: a missing camera key just skips that frame,
    a ffmpeg startup failure logs a warning and drops the camera.
    """

    def __init__(
        self,
        robot: "RobotWrapper",
        cameras: list[str],
        out_dir: Path,
        fps: float,
        shutdown_event: Event,
    ):
        self.robot = robot
        self.cameras = list(cameras)
        self.out_dir = Path(out_dir)
        self.fps = max(float(fps), 1.0)
        self.shutdown_event = shutdown_event
        self._stop_event = Event()
        self._thread: Thread | None = None
        # cam → subprocess.Popen of ffmpeg (stdin = raw BGR pipe)
        self.encoders: dict[str, object] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="CameraVideoRecorder")
        self._thread.start()
        logger.info(
            f"[CameraLog] recording cameras={self.cameras} @ {self.fps}fps → {self.out_dir}  "
            f"[H.264 via ffmpeg pipe]"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Close stdin → ffmpeg sees EOF and writes moov atom (+faststart=front).
        for cam, proc in list(self.encoders.items()):
            try:
                if getattr(proc, "stdin", None) and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass
        # Wait for each ffmpeg to finalize the mp4.
        finalized = 0
        for cam, proc in list(self.encoders.items()):
            try:
                proc.wait(timeout=15)
                finalized += 1
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self.encoders:
            logger.info(f"[CameraLog] saved {finalized} video(s) in {self.out_dir}")
        self.encoders = {}

    def _run(self) -> None:
        import shutil as _shutil
        import subprocess
        ffmpeg = _shutil.which("ffmpeg")
        if ffmpeg is None:
            logger.warning("[CameraLog] ffmpeg not found on PATH — recording disabled")
            return
        period = 1.0 / self.fps
        while not self.shutdown_event.is_set() and not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                obs = self.robot.get_observation()
                for cam in self.cameras:
                    frame = PauseCameraWindow.select_camera_frame(obs, cam)
                    if frame is None:
                        continue
                    # select_camera_frame returns RGB uint8 HxWx3 (or grayscale).
                    # ffmpeg pipe expects BGR (or convert pix_fmt). Force 3-ch BGR.
                    if frame.ndim != 3 or frame.shape[-1] != 3:
                        continue
                    bgr = np.ascontiguousarray(frame[..., ::-1])
                    if cam not in self.encoders:
                        h, w = bgr.shape[:2]
                        path = self.out_dir / f"{cam}.mp4"
                        cmd = [
                            ffmpeg, "-y", "-v", "error",
                            "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{w}x{h}", "-r", f"{self.fps}",
                            "-i", "pipe:0",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            "-an",
                            str(path),
                        ]
                        try:
                            proc = subprocess.Popen(
                                cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception as e:
                            logger.warning(f"[CameraLog] failed to start ffmpeg for {cam}: {e}")
                            continue
                        self.encoders[cam] = proc
                        logger.info(f"[CameraLog] {cam} ({w}x{h}) → {path}")
                    try:
                        self.encoders[cam].stdin.write(bgr.tobytes())
                    except (BrokenPipeError, ValueError, AttributeError):
                        logger.warning(f"[CameraLog] {cam} ffmpeg pipe broken — dropping")
                        try:
                            self.encoders[cam].kill()
                        except Exception:
                            pass
                        del self.encoders[cam]
            except Exception as e:
                logger.warning(f"[CameraLog] step error: {e}")
            time.sleep(max(0.0, period - (time.perf_counter() - started)))


# Module-level holders for FK-based pitch clamp resources (initialized once in main()).
_pitch_kinematics = None
_pitch_cal_lim = None


def _setup_pitch_clamp(cfg: "RTCDemoConfig"):
    """Init KinematicsEngine + CalibrationLimits for FK-based EE pitch clamp.
    Returns (engine, cal_lim) or (None, None) on failure / disabled."""
    if cfg.inference_pitch_max_deg is None:
        return None, None
    try:
        from lerobot_cap.kinematics import KinematicsEngine, load_calibration_limits
    except Exception as e:
        logger.warning(f"[PitchClamp] lerobot_cap.kinematics import failed ({e}); disabled")
        return None, None
    try:
        robot_id = getattr(cfg.robot, "id", "") or ""
        m = re.search(r"robot\d+", robot_id)
        robot_tag = m.group(0) if m else robot_id
        repo_root = Path(__file__).resolve().parents[3]
        urdf_candidates = [
            repo_root / "assets" / "urdf" / f"{robot_id}.urdf",
            repo_root / "assets" / "urdf" / f"so101_{robot_tag}.urdf",
        ]
        urdf_path = next((p for p in urdf_candidates if p.exists()), None)
        if urdf_path is None:
            logger.warning(f"[PitchClamp] URDF not found (tried {[str(p) for p in urdf_candidates]}); disabled")
            return None, None
        cal_dir = Path(cfg.robot.calibration_dir)
        cal_candidates = [
            cal_dir / f"{robot_id}.json",
            cal_dir / f"{robot_tag}_calibration.json",
        ]
        cal_path = next((p for p in cal_candidates if p.exists()), None)
        if cal_path is None:
            logger.warning(f"[PitchClamp] calibration not found (tried {[str(p) for p in cal_candidates]}); disabled")
            return None, None
        ke = KinematicsEngine(str(urdf_path))
        cal_lim = load_calibration_limits(str(cal_path))
        logger.info(
            f"[PitchClamp] enabled: max={cfg.inference_pitch_max_deg}° "
            f"(URDF={urdf_path.name}, cal={cal_path.name})"
        )
        return ke, cal_lim
    except Exception as e:
        logger.warning(f"[PitchClamp] init failed ({e}); disabled")
        return None, None


def _clamp_action_pitch(action, cfg: "RTCDemoConfig"):
    """Adjust action's wrist_flex.pos so that FK-computed EE pitch ≤ pitch_max_deg.
    No-op if engine/cal_lim missing or pitch_max_deg None. Mutates and returns action.

    Convention: pitch = arcsin(gripper_Z_world_z). 0 = horizontal, -π/2 = down.
    ``inference_pitch_max_deg = -30`` forces gripper to tilt ≥30° below horizontal.
    """
    ke = _pitch_kinematics
    cal_lim = _pitch_cal_lim
    if ke is None or cal_lim is None or cfg.inference_pitch_max_deg is None:
        return action
    target = action
    if isinstance(action, tuple) and len(action) > 0 and isinstance(action[0], dict):
        target = action[0]
    if not isinstance(target, dict):
        return action
    joint_names = cal_lim.joint_names
    try:
        normalized = np.array(
            [float(target[f"{n}.pos"]) for n in joint_names if f"{n}.pos" in target],
            dtype=float,
        )
    except Exception:
        return action
    if len(normalized) != len(joint_names):
        return action  # missing a joint key
    joints_rad = cal_lim.normalized_to_radians(normalized)
    try:
        pitch_rad = ke.get_gripper_pitch(joints_rad)
    except Exception:
        return action
    pitch_deg = float(np.degrees(pitch_rad))
    max_deg = float(cfg.inference_pitch_max_deg)
    if pitch_deg <= max_deg:
        return action
    if "wrist_flex" not in joint_names:
        return action
    wf_idx = joint_names.index("wrist_flex")
    # Sign relation (verified on so101_robot4): wrist_flex_rad ↑ → EE pitch ↓.
    # Therefore to lower pitch by Δ deg, increase wrist_flex_rad by Δ rad (1:1).
    joints_rad[wf_idx] += np.radians(pitch_deg - max_deg)
    for _ in range(4):
        try:
            pitch_rad = ke.get_gripper_pitch(joints_rad)
        except Exception:
            break
        pitch_deg = float(np.degrees(pitch_rad))
        if pitch_deg <= max_deg + 0.3:  # 0.3° tolerance
            break
        joints_rad[wf_idx] += np.radians(pitch_deg - max_deg) * 0.7
    new_normalized = cal_lim.radians_to_normalized(joints_rad)
    target["wrist_flex.pos"] = float(new_normalized[wf_idx])
    return action


def resolve_free_state_path(cfg: "RTCDemoConfig") -> Path:
    """Locate the free_state JSON for the robot being controlled.

    Picks the file matching ``robot.id`` (e.g. ``so101_robot6`` -> ``robot6``)
    under ``<robot_configs>/free_state/``, where ``<robot_configs>`` is derived
    from the calibration dir, or overridden via ``cfg.free_state_dir``.
    """
    robot_id = getattr(cfg.robot, "id", "") or ""
    m = re.search(r"robot\d+", robot_id)
    if not m:
        raise ValueError(f"Cannot extract 'robotN' from robot.id={robot_id!r}")
    robot_tag = m.group(0)

    if cfg.free_state_dir:
        free_state_dir = Path(cfg.free_state_dir)
    else:
        calib_dir = getattr(cfg.robot, "calibration_dir", None)
        if not calib_dir:
            raise ValueError("robot.calibration_dir is unset; pass --free_state_dir explicitly")
        # calibration_dir is <robot_configs>/motor_calibration/<family>
        free_state_dir = Path(calib_dir).resolve().parent.parent / "free_state"

    path = free_state_dir / f"{robot_tag}_free_state.json"
    if not path.exists():
        raise FileNotFoundError(f"free_state file not found: {path}")
    return path


def move_to_free_state(
    robot: "RobotWrapper", free_state_path: Path, fps: float, duration: float = 2.0
) -> None:
    """Smoothly drive the robot from its current pose to the free_state pose.

    Uses cosine smoothing so the servos do not jerk. Arm joints are clamped to
    [-100, 100] and the gripper to [0, 100] (free_state JSON values can be out
    of range / inconsistent across robots).
    """
    with open(free_state_path) as f:
        data = json.load(f)

    arm_target = np.asarray(data["initial_state_normalized"], dtype=np.float32)
    gripper_target = float(data["gripper_normalized"])
    target = np.concatenate([np.clip(arm_target, -100.0, 100.0), [np.clip(gripper_target, 0.0, 100.0)]])

    action_keys = list(robot.action_features())  # ordered: 5 arm joints + gripper
    if len(action_keys) != len(target):
        raise ValueError(
            f"free_state has {len(target)} joints but robot has {len(action_keys)} action features"
        )

    obs = robot.get_observation()
    current = np.asarray([float(obs[k]) for k in action_keys], dtype=np.float32)

    steps = max(1, int(duration * fps))
    period = 1.0 / fps
    logger.info(f"[INTERRUPT] Moving to free_state ({free_state_path.name}) over {duration:.1f}s")
    for i in range(1, steps + 1):
        loop_start = time.perf_counter()
        alpha = i / steps
        smooth = (1.0 - math.cos(alpha * math.pi)) / 2.0
        cmd = current + smooth * (target - current)
        robot.send_action({k: float(cmd[j]) for j, k in enumerate(action_keys)})
        time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
    logger.info("[INTERRUPT] Reached free_state.")


def keyboard_listener(
    interrupt: InterruptController,
    action_queue: "ActionQueue",
    robot: "RobotWrapper",
    free_state_path: Path | None,
    fps: float,
    move_duration: float,
    shutdown_event: Event,
    pause_stream: PauseCameraWindow | None = None,
):
    """Block on stdin; each Enter toggles park (flush + free_state) / resume."""
    logger.info("[INTERRUPT] Press Enter to park at free_state; Enter again to resume")
    while not shutdown_event.is_set():
        try:
            line = sys.stdin.readline()
        except OSError as exc:
            logger.info(f"[INTERRUPT] Keyboard listener stopped; stdin unavailable ({exc})")
            break
        if line == "":  # EOF — stdin closed
            break
        if shutdown_event.is_set():
            break

        if not interrupt.parked.is_set():
            # 1st Enter: park — pause workers, flush buffer, move to free_state.
            logger.info("[INTERRUPT] Parking: flushing action buffer + moving to free_state...")
            interrupt.set_parked()
            time.sleep(0.1)  # let the actor thread observe the parked flag
            action_queue.clear()
            if free_state_path is not None:
                try:
                    move_to_free_state(robot, free_state_path, fps, move_duration)
                except Exception as e:
                    logger.error(f"[INTERRUPT] free_state move failed ({e}); robot held in place.")
            else:
                logger.warning("[INTERRUPT] No free_state file resolved; robot held in place.")
            if pause_stream is not None:
                pause_stream.start()
            logger.info("[INTERRUPT] Parked. Press Enter to resume (first chunk will be discarded).")
        else:
            # 2nd Enter: resume — discard the first new chunk, then run normally.
            if pause_stream is not None:
                pause_stream.stop()
            interrupt.arm_discard(1)
            interrupt.set_running()
            logger.info("[INTERRUPT] Resumed.")


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies and real robots."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration
    robot: RobotConfig | None = None

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)

    # Compute device
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 50

    # Smooth non-RTC chunk boundaries for absolute action targets.
    chunk_transition_steps: int = field(
        default=4,
        metadata={"help": "Blend this many steps at the start of each appended chunk"},
    )
    action_max_delta: float | None = field(
        default=None,
        metadata={"help": "Optional per-step action delta clamp after chunk-boundary blending"},
    )
    stop_and_infer: bool = field(
        default=False,
        metadata={"help": "Run sequentially: infer one chunk only after the previous chunk finishes"},
    )

    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})

    # Enter-key park/resume: directory holding <robotN>_free_state.json files.
    # If empty, derived from robot.calibration_dir (<robot_configs>/free_state).
    free_state_dir: str = field(
        default="",
        metadata={"help": "Directory with <robotN>_free_state.json files (default: derived)"},
    )

    # Seconds for the smooth interpolated move to free_state on park.
    free_state_move_duration: float = field(
        default=2.0,
        metadata={"help": "Duration of the smooth move to free_state on Enter-park"},
    )

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

    # Action chunk saving for visualization
    save_chunks: bool = field(
        default=False,
        metadata={"help": "Save action chunks to .npz for offline visualization"},
    )
    save_chunks_dir: str = field(
        default="outputs/action_chunks",
        metadata={"help": "Directory to save action chunk files"},
    )
    save_chunks_max: int = field(
        default=15,
        metadata={"help": "Number of action chunks to save before stopping collection"},
    )

    # Small local OpenCV window shown while parked via Enter.
    pause_stream_enabled: bool = field(
        default=False,
        metadata={"help": "Show a small local camera preview window while parked"},
    )
    pause_stream_camera: str = field(
        default="top",
        metadata={"help": "Camera key to preview while parked"},
    )
    pause_stream_fps: float = field(
        default=15.0,
        metadata={"help": "Preview window refresh rate"},
    )
    pause_stream_scale: float = field(
        default=1.0,
        metadata={"help": "Preview window upscale factor (e.g. 2.0 = 2x)"},
    )

    # Per-camera mp4 logger — independent from pause_stream. Runs for the whole
    # inference session, writing one mp4 per camera (top.mp4, left_wrist.mp4, ...).
    log_cameras_enabled: bool = field(
        default=False,
        metadata={"help": "Record per-camera mp4 videos during inference"},
    )
    log_cameras_dir: str = field(
        default="",
        metadata={
            "help": (
                "Output directory for per-camera mp4s. Empty → auto "
                "(./results/inference_cam_log_<timestamp>)"
            )
        },
    )
    log_cameras_fps: float = field(
        default=30.0,
        metadata={"help": "mp4 writer fps (separate from inference control fps)"},
    )
    log_cameras_names: str = field(
        default="top,left_wrist",
        metadata={"help": "Comma-separated camera names to record (e.g. top,left_wrist)"},
    )

    # ADC ``transit_pitch_max_deg`` 동등 — FK 로 EE pitch 계산 후 한계 초과 시
    # wrist_flex 조정으로 강제 (1:1 + 4 iter refinement). 단위: degree.
    # pitch convention: 0=horizontal, -90=down (gripper Z arcsin). 따라서
    # ``-25`` = "gripper 가 horizontal 보다 최소 25° 아래" 강제.
    inference_pitch_max_deg: float | None = field(
        default=None,
        metadata={"help": "Clamp gripper pitch to ≤ this deg via FK (ADC transit_pitch_max_deg equivalent)."},
    )

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required")

        # Validate that robot configuration is provided
        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def _flush_action_chunks(saved_chunks, robot, fps, save_path: str, reason: str = "max-reached", verbose: bool = True) -> None:
    """Atomically persist collected action chunks to ``save_path``.

    Writes to ``save_path + ".tmp"`` first then renames so a kill mid-write
    never leaves a corrupt file. Called incrementally (after each chunk) and
    on graceful shutdown.
    """
    import os
    import numpy as np

    # np.savez auto-appends ".npz" if absent, so the tmp name must already end
    # in ".npz" to avoid creating "<tmp>.npz" instead of "<tmp>".
    tmp_path = save_path[:-4] + ".tmp.npz" if save_path.endswith(".npz") else save_path + ".tmp.npz"
    np.savez(
        tmp_path,
        **{f"chunk_{i}": c["actions"] for i, c in enumerate(saved_chunks)},
        timestamps=np.array([c["timestamp"] for c in saved_chunks]),
        inference_delays=np.array([c["inference_delay"] for c in saved_chunks]),
        action_features=robot.action_features(),
        fps=np.array(fps),
    )
    os.replace(tmp_path, save_path)
    if verbose:
        logger.info(
            f"[GET_ACTIONS] Saved {len(saved_chunks)} chunks to {save_path} (reason={reason})"
        )


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def get_actions(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    interrupt: InterruptController,
    cfg: RTCDemoConfig,
):
    """Thread function to request action chunks from the policy.

    Args:
        policy: The policy instance (SmolVLA, Pi0, etc.)
        robot: The robot instance for getting observations
        robot_observation_processor: Processor for raw robot observations
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[GET_ACTIONS] Starting get actions thread")

        # Action chunk saving: incremental write so we never lose collected
        # chunks even on hard kill (process.py force-exits on 2nd Ctrl+C).
        saved_chunks = []
        chunk_save_done = False
        chunk_save_path = None
        if cfg.save_chunks:
            import os as _os
            _os.makedirs(cfg.save_chunks_dir, exist_ok=True)
            chunk_save_path = _os.path.join(
                cfg.save_chunks_dir,
                f"chunks_{time.strftime('%Y%m%d_%H%M%S')}.npz",
            )
            logger.info(f"[GET_ACTIONS] Will incrementally save chunks to {chunk_save_path}")

        latency_tracker = LatencyTracker()  # Track latency of action chunks
        fps = cfg.fps
        time_per_chunk = 1.0 / fps

        dataset_features = hw_to_dataset_features(robot.observation_features(), "observation")
        policy_device = policy.config.device

        # Load preprocessor and postprocessor from pretrained files
        # The stats are embedded in the processor .safetensors files
        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,  # Will load from pretrained processor files
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )

        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully with embedded stats")

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        while not shutdown_event.is_set():
            # Hold here while parked — no new chunks requested until resumed.
            interrupt.wait_while_parked(shutdown_event)
            if shutdown_event.is_set():
                break

            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                obs = robot.get_observation()

                # Apply robot observation processor
                obs_processed = robot_observation_processor(obs)

                obs_with_policy_features = build_dataset_frame(
                    dataset_features, obs_processed, prefix="observation"
                )

                for name in obs_with_policy_features:
                    obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
                    if "image" in name:
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].type(torch.float32) / 255
                        )
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                        )
                    obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                    obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

                obs_with_policy_features["task"] = [cfg.task]  # Task should be a list, not a string!
                obs_with_policy_features["robot_type"] = (
                    robot.robot.name if hasattr(robot.robot, "name") else ""
                )

                preproceseded_obs = preprocessor(obs_with_policy_features)

                if cfg.rtc.enabled:
                    actions = policy.predict_action_chunk(
                        preproceseded_obs,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_actions,
                    )
                else:
                    actions = policy.predict_action_chunk(preproceseded_obs)

                # Drop this chunk if a park happened while inference was running
                # (the buffer was flushed — merging now would repopulate it).
                if interrupt.parked.is_set():
                    logger.info("[INTERRUPT] Dropping in-flight chunk (parked during inference)")
                    continue

                # On resume, discard the first chunk so execution restarts cleanly.
                if interrupt.consume_discard():
                    logger.info("[INTERRUPT] Discarded first chunk after resume")
                    continue

                # Store original actions (before postprocessing) for RTC
                original_actions = actions.squeeze(0).clone()

                postprocessed_actions = postprocessor(actions)

                postprocessed_actions = postprocessed_actions.squeeze(0)

                # Save chunk for offline visualization (incremental write)
                if cfg.save_chunks and not chunk_save_done:
                    saved_chunks.append({
                        "actions": postprocessed_actions.cpu().numpy().copy(),
                        "timestamp": time.time(),
                        "inference_delay": inference_delay,
                    })
                    # Overwrite snapshot after every new chunk so the file
                    # always reflects the latest state — survives any kill.
                    _flush_action_chunks(
                        saved_chunks, robot, fps, chunk_save_path,
                        reason="incremental", verbose=False,
                    )
                    if len(saved_chunks) >= cfg.save_chunks_max:
                        logger.info(
                            f"[GET_ACTIONS] Reached save_chunks_max={cfg.save_chunks_max}, "
                            f"final file at {chunk_save_path}"
                        )
                        chunk_save_done = True

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        f"[GET_ACTIONS] action_queue_size={cfg.action_queue_size_to_get_new_actions} < execution_horizon={cfg.rtc.execution_horizon} + delay={new_delay} = {cfg.rtc.execution_horizon + new_delay}. Increase --action_queue_size_to_get_new_actions."
                    )

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.1)

        # Final summary on graceful shutdown (file is already up-to-date due
        # to incremental writes — this is just for logging).
        if cfg.save_chunks and saved_chunks:
            logger.info(
                f"[GET_ACTIONS] Final: {len(saved_chunks)} chunks saved to {chunk_save_path}"
            )

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        # On crash the incremental file already has the latest snapshot —
        # log so the user knows where to find it.
        if cfg.save_chunks and saved_chunks:
            logger.error(
                f"[GET_ACTIONS] Crash mid-collection. Last snapshot ({len(saved_chunks)} chunks) at {chunk_save_path}"
            )
        logger.error(f"[GET_ACTIONS] Fatal exception in get_actions thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def _prepare_policy_observation(
    robot: RobotWrapper,
    robot_observation_processor,
    dataset_features,
    policy_device,
    cfg: RTCDemoConfig,
) -> dict[str, Tensor]:
    obs = robot.get_observation()
    obs_processed = robot_observation_processor(obs)
    obs_with_policy_features = build_dataset_frame(dataset_features, obs_processed, prefix="observation")

    for name in obs_with_policy_features:
        obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
        if "image" in name:
            obs_with_policy_features[name] = obs_with_policy_features[name].type(torch.float32) / 255
            obs_with_policy_features[name] = obs_with_policy_features[name].permute(2, 0, 1).contiguous()
        obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
        obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

    obs_with_policy_features["task"] = [cfg.task]
    obs_with_policy_features["robot_type"] = robot.robot.name if hasattr(robot.robot, "name") else ""
    return obs_with_policy_features


def stop_and_infer_control(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    robot_action_processor,
    shutdown_event: Event,
    interrupt: InterruptController,
    cfg: RTCDemoConfig,
):
    """Sequential inference loop: execute a full chunk, then infer the next one."""
    try:
        import os as _os

        logger.info("[STOP_AND_INFER] Starting sequential chunk execution")
        fps = cfg.fps
        action_interval = 1.0 / fps
        dataset_features = hw_to_dataset_features(robot.observation_features(), "observation")
        policy_device = policy.config.device

        logger.info(f"[STOP_AND_INFER] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )
        logger.info("[STOP_AND_INFER] Preprocessor/postprocessor loaded successfully with embedded stats")

        saved_chunks = []
        chunk_save_done = False
        chunk_save_path = None
        if cfg.save_chunks:
            _os.makedirs(cfg.save_chunks_dir, exist_ok=True)
            chunk_save_path = _os.path.join(
                cfg.save_chunks_dir,
                f"chunks_{time.strftime('%Y%m%d_%H%M%S')}_stop_and_infer.npz",
            )
            logger.info(f"[STOP_AND_INFER] Will incrementally save chunks to {chunk_save_path}")

        action_keys = robot.action_features()
        action_count = 0
        chunk_count = 0
        smoothing_queue = ActionQueue(cfg.rtc)
        smoothing_queue.transition_steps = cfg.chunk_transition_steps
        smoothing_queue.max_action_delta = cfg.action_max_delta

        start_time = time.time()
        while not shutdown_event.is_set() and (
            time.time() - start_time - interrupt.parked_total()
        ) < cfg.duration:
            interrupt.wait_while_parked(shutdown_event)
            if shutdown_event.is_set():
                break

            current_time = time.perf_counter()
            obs = _prepare_policy_observation(
                robot, robot_observation_processor, dataset_features, policy_device, cfg
            )
            preprocessed_obs = preprocessor(obs)

            if cfg.rtc.enabled:
                actions = policy.predict_action_chunk(
                    preprocessed_obs,
                    inference_delay=0,
                    prev_chunk_left_over=None,
                )
            else:
                actions = policy.predict_action_chunk(preprocessed_obs)

            postprocessed_actions = postprocessor(actions).squeeze(0)
            postprocessed_actions = smoothing_queue._smooth_action_boundary(postprocessed_actions.clone())

            inference_latency = time.perf_counter() - current_time
            inference_delay = math.ceil(inference_latency / action_interval)
            logger.info(
                f"[STOP_AND_INFER] Chunk {chunk_count}: inferred {len(postprocessed_actions)} actions "
                f"in {inference_latency:.3f}s (~{inference_delay} steps). Executing full chunk now."
            )

            if cfg.save_chunks and not chunk_save_done:
                saved_chunks.append({
                    "actions": postprocessed_actions.cpu().numpy().copy(),
                    "timestamp": time.time(),
                    "inference_delay": inference_delay,
                })
                _flush_action_chunks(
                    saved_chunks,
                    robot,
                    fps,
                    chunk_save_path,
                    reason="stop-and-infer-incremental",
                    verbose=False,
                )
                if len(saved_chunks) >= cfg.save_chunks_max:
                    logger.info(
                        f"[STOP_AND_INFER] Reached save_chunks_max={cfg.save_chunks_max}, "
                        f"final file at {chunk_save_path}"
                    )
                    chunk_save_done = True

            for action in postprocessed_actions:
                if shutdown_event.is_set() or interrupt.parked.is_set():
                    break
                started = time.perf_counter()
                action = action.cpu()
                smoothing_queue.last_action = action.clone()
                action_dict = {key: action[i].item() for i, key in enumerate(action_keys)}
                action_processed = robot_action_processor((action_dict, None))
                action_processed = _clamp_action_pitch(action_processed, cfg)
                robot.send_action(action_processed)
                action_count += 1
                dt_s = time.perf_counter() - started
                time.sleep(max(0, (action_interval - dt_s) - 0.001))

            chunk_count += 1

        if cfg.save_chunks and saved_chunks:
            logger.info(
                f"[STOP_AND_INFER] Final: {len(saved_chunks)} chunks saved to {chunk_save_path}"
            )

        logger.info(
            f"[STOP_AND_INFER] Sequential loop shutting down. "
            f"Chunks={chunk_count}, actions={action_count}"
        )
    except Exception as e:
        logger.error(f"[STOP_AND_INFER] Fatal exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def actor_control(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    interrupt: InterruptController,
    cfg: RTCDemoConfig,
):
    """Thread function to execute actions on the robot.

    Args:
        robot: The robot instance
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / cfg.fps

        while not shutdown_event.is_set():
            # While parked, send no policy actions — the keyboard thread owns
            # the robot (free_state move) and it holds there until resumed.
            if interrupt.parked.is_set():
                time.sleep(0.05)
                continue

            start_time = time.perf_counter()

            # Try to get an action from the queue with timeout
            action = action_queue.get()

            if action is not None:
                action = action.cpu()
                action_dict = {key: action[i].item() for i, key in enumerate(robot.action_features())}
                action_processed = robot_action_processor((action_dict, None))
                action_processed = _clamp_action_pitch(action_processed, cfg)
                robot.send_action(action_processed)

                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def _apply_torch_compile(policy, cfg: RTCDemoConfig):
    """Apply torch.compile to the policy's predict_action_chunk method.

    Args:
        policy: Policy instance to compile
        cfg: Configuration containing torch compile settings

    Returns:
        Policy with compiled predict_action_chunk method
    """

    # PI models handle their own compilation
    if policy.type == "pi05" or policy.type == "pi0":
        return policy

    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile is not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}")
        logger.info(f"  Mode: {cfg.torch_compile_mode}")
        logger.info(f"  Disable CUDA graphs: {cfg.torch_compile_disable_cudagraphs}")

        # Compile the predict_action_chunk method
        # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        # Disable CUDA graphs if requested (prevents tensor aliasing issues)
        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


@parser.wrap()
def demo_cli(cfg: RTCDemoConfig):
    """Main entry point for RTC demo with draccus configuration."""

    # Initialize logging
    init_logging()

    logger.info(f"Using device: {cfg.device}")
    logger.info(f"action_queue_size_to_get_new_actions: {cfg.action_queue_size_to_get_new_actions}")

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy = None
    robot = None
    get_actions_thread = None
    actor_thread = None
    keyboard_thread = None

    policy_class = get_policy_class(cfg.policy.type)

    # Load config and set compile_model for pi0/pi05 models
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if cfg.policy.type == "pi05" or cfg.policy.type == "pi0":
        config.compile_model = cfg.use_torch_compile

    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_pretrained_path = cfg.policy.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    # Configure RTC. Some policies, such as GR00T, do not implement the RTC
    # processor and should run normally when RTC is disabled.
    policy.config.rtc_config = cfg.rtc

    if cfg.rtc.enabled:
        assert policy.name in ["smolvla", "pi05", "pi0", "groot"], (
            "Only smolvla, pi05, pi0, and groot are supported for RTC"
        )
        # Init RTC processor, as by default if RTC disabled in the config
        # the processor won't be created.
        policy.init_rtc_processor()

    policy = policy.to(cfg.device)
    policy.eval()

    # Apply torch.compile to predict_action_chunk method if enabled
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    # Create robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_wrapper = RobotWrapper(robot)

    # Create robot observation processor
    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()

    # Create action queue for communication between threads
    action_queue = ActionQueue(cfg.rtc)
    action_queue.transition_steps = cfg.chunk_transition_steps
    action_queue.max_action_delta = cfg.action_max_delta

    # Enter-key park/resume controller, shared across all worker threads
    interrupt = InterruptController()

    # Resolve the free_state file matching this robot (Enter-park target).
    try:
        free_state_path = resolve_free_state_path(cfg)
        logger.info(f"[INTERRUPT] free_state target: {free_state_path}")
    except Exception as e:
        free_state_path = None
        logger.warning(f"[INTERRUPT] free_state unavailable ({e}); Enter-park will only freeze.")

    pause_stream = None
    if cfg.pause_stream_enabled:
        pause_stream = PauseCameraWindow(
            robot=robot_wrapper,
            camera=cfg.pause_stream_camera,
            shutdown_event=shutdown_event,
            fps=cfg.pause_stream_fps,
            scale=cfg.pause_stream_scale,
        )
        logger.info(f"[PAUSE_STREAM] Will show camera '{cfg.pause_stream_camera}' in a local window on Enter-park")

    # FK-based EE pitch clamp (ADC transit_pitch_max_deg equivalent).
    global _pitch_kinematics, _pitch_cal_lim
    _pitch_kinematics, _pitch_cal_lim = _setup_pitch_clamp(cfg)

    cam_recorder = None
    if cfg.log_cameras_enabled:
        _cam_list = [c.strip() for c in cfg.log_cameras_names.split(",") if c.strip()]
        if not _cam_list:
            logger.warning("[CameraLog] log_cameras_enabled=True but log_cameras_names empty — skipping")
        else:
            _out_dir = cfg.log_cameras_dir.strip() or f"./results/inference_cam_log_{int(time.time())}"
            cam_recorder = CameraVideoRecorder(
                robot=robot_wrapper,
                cameras=_cam_list,
                out_dir=Path(_out_dir),
                fps=cfg.log_cameras_fps,
                shutdown_event=shutdown_event,
            )
            cam_recorder.start()

    if cfg.stop_and_infer:
        logger.info("[MAIN] stop_and_infer=true: running sequential chunk mode")
        keyboard_thread = Thread(
            target=keyboard_listener,
            args=(
                interrupt,
                action_queue,
                robot_wrapper,
                free_state_path,
                cfg.fps,
                cfg.free_state_move_duration,
                shutdown_event,
                pause_stream,
            ),
            daemon=True,
            name="KeyboardListener",
        )
        keyboard_thread.start()
        logger.info("Started keyboard listener thread")

        try:
            stop_and_infer_control(
                policy,
                robot_wrapper,
                robot_observation_processor,
                robot_action_processor,
                shutdown_event,
                interrupt,
                cfg,
            )
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
        finally:
            logger.info("Demo duration reached or shutdown requested")
            shutdown_event.set()
            if pause_stream is not None:
                pause_stream.stop()
            if cam_recorder is not None:
                cam_recorder.stop()
            if keyboard_thread and keyboard_thread.is_alive():
                keyboard_thread.join(timeout=2)
            robot.disconnect()
            logger.info("Robot disconnected")
            logger.info("Cleanup completed")
        return

    # Start chunk requester thread
    get_actions_thread = Thread(
        target=get_actions,
        args=(
            policy,
            robot_wrapper,
            robot_observation_processor,
            action_queue,
            shutdown_event,
            interrupt,
            cfg,
        ),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()
    logger.info("Started get actions thread")

    # Start action executor thread
    actor_thread = Thread(
        target=actor_control,
        args=(robot_wrapper, robot_action_processor, action_queue, shutdown_event, interrupt, cfg),
        daemon=True,
        name="Actor",
    )
    actor_thread.start()
    logger.info("Started actor thread")

    # Start keyboard listener thread (Enter parks at free_state / resumes)
    keyboard_thread = Thread(
        target=keyboard_listener,
        args=(
            interrupt,
            action_queue,
            robot_wrapper,
            free_state_path,
            cfg.fps,
            cfg.free_state_move_duration,
            shutdown_event,
            pause_stream,
        ),
        daemon=True,
        name="KeyboardListener",
    )
    keyboard_thread.start()
    logger.info("Started keyboard listener thread")

    logger.info("Started stop by duration thread")

    # Main thread monitors for duration or shutdown
    logger.info(f"Running demo for {cfg.duration} seconds...")
    start_time = time.time()

    try:
        # Parked time is excluded from the duration so a park never times the run out.
        while not shutdown_event.is_set() and (
            time.time() - start_time - interrupt.parked_total()
        ) < cfg.duration:
            time.sleep(10)

            elapsed = time.time() - start_time - interrupt.parked_total()

            # Log queue status periodically
            if int(elapsed) % 5 == 0:
                logger.info(f"[MAIN] Action queue size: {action_queue.qsize()}")

            if elapsed > cfg.duration:
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    finally:
        logger.info("Demo duration reached or shutdown requested")

        # Signal shutdown
        shutdown_event.set()

        if pause_stream is not None:
            pause_stream.stop()
        if cam_recorder is not None:
            cam_recorder.stop()

        # Wait for threads to finish
        if get_actions_thread and get_actions_thread.is_alive():
            logger.info("Waiting for chunk requester thread to finish...")
            get_actions_thread.join(timeout=5)

        if actor_thread and actor_thread.is_alive():
            logger.info("Waiting for action executor thread to finish...")
            actor_thread.join(timeout=5)

        # Cleanup robot
        if robot:
            robot.disconnect()
            logger.info("Robot disconnected")

        logger.info("Cleanup completed")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")
