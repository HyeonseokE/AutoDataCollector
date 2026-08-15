"""Live 3D end-effector trajectory trace via rerun — real-time accumulation.

Polls ``ForwardAndResetPipeline._latest_robot_state()`` on its own clock
(non-intrusive — control-path state is never touched, mirroring
``record_dataset/live_preview.py``), converts the normalized servo state to URDF
radians via the robot's calibration, runs forward kinematics (Pinocchio, the same
``KinematicsEngine`` used by ``traj_analysis.fk_ee``) to obtain the EE (x, y, z),
and streams the accumulated polyline to a rerun viewer.

Only the FORWARD phase of each episode is accumulated (gated on
``pipeline.current_phase == "Forward"`` — reset motion is excluded). Each episode
is logged under its own entity path with a distinct color, so previous episodes'
lines remain on screen and stay visually separable.

On ``stop()`` the per-episode xyz arrays are dumped to an ``.npz`` so a clean mp4
can be rendered offline (``scripts/render_ee_trace_video.py``). The live view does
NOT itself encode video — keeping the control host's per-cycle load at zero.

Usage::

    tr = LiveEETraceWindow(pipeline, robot_id=3, fps=15)
    tr.start()
    ...           # episodes run; pipeline polled on its own thread
    tr.stop()     # idempotent; dumps ee_trace.npz

Env-driven toggle is done in ``execution_forward_and_reset.main()`` AFTER the
pipeline is constructed (the poller needs the pipeline object) — this module only
provides the class.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Arm joints fed to FK (5D). Gripper (6th dim) is irrelevant to EE xyz.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# AutoDataCollector project root (this file lives at <root>/record_dataset/).
# Used to resolve URDF / calibration with absolute paths regardless of CWD —
# important when launched from the vendored lerobot dir (lerobot-record).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_urdf_path(robot_id: int) -> str:
    return str(_PROJECT_ROOT / f"assets/urdf/so101_robot{int(robot_id)}.urdf")


def default_calib_path(robot_id: int) -> str:
    return str(
        _PROJECT_ROOT
        / f"robot_configs/motor_calibration/so101/robot{int(robot_id)}_calibration.json"
    )


def build_fk(urdf_path: str, calib_path: str, ee_frame: str = "gripper_frame_link"):
    """Construct (KinematicsEngine, CalibrationJointLimits) — same engine/frame as
    ``traj_analysis.fk_ee`` and ``add_radian_urdf0`` so EE xyz is consistent across
    the live trace, offline visualizers, and recorded radian features."""
    # Ensure lerobot_cap is importable regardless of the launcher's sys.path state
    # (the forward/reset pipeline only adds <root>/src lazily at runtime; some
    # conda envs don't have lerobot_cap installed). Self-sufficient → never a
    # silent disable just because of import path ordering.
    import sys as _sys

    _src = str(_PROJECT_ROOT / "src")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from lerobot_cap.kinematics import KinematicsEngine, load_calibration_limits

    if not Path(urdf_path).exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not Path(calib_path).exists():
        raise FileNotFoundError(f"calibration not found: {calib_path}")
    kin = KinematicsEngine(urdf_path, end_effector_frame=ee_frame, joint_names=ARM_JOINTS)
    calib = load_calibration_limits(calib_path, ARM_JOINTS)
    return kin, calib


def ee_xyz_from_norm(kin: Any, calib: Any, norm_state: np.ndarray) -> np.ndarray:
    """Normalized 6D (or 5D) servo state → EE (x, y, z) via calib radians + FK."""
    rad5 = calib.normalized_to_radians(np.asarray(norm_state[:5], dtype=np.float64))
    pos, _R = kin.forward_kinematics(rad5.astype(np.float64))
    return np.asarray(pos, dtype=np.float32)


def is_glitch_step(prev_xyz, prev_t, xyz, ts, max_speed_mps: float, max_jump_m: float = 0.08) -> bool:
    """True if the step prev→(xyz) is a serial-read glitch, vs the last ACCEPTED
    point. Two OR-ed tests:

      • absolute jump > ``max_jump_m`` (default 80 mm). The robot's real EE step
        is ≤~12 mm at 30 Hz, so 80 mm only fires on glitches. Crucially this also
        kills a *stuck/frozen read* (the bus froze and returned one fixed bad
        pose for hundreds of samples): every frozen sample stays >80 mm from the
        last good pose, so the whole bad run is dropped while the trajectory
        resumes cleanly when good reads return. (Real motion keeps the anchor
        advancing ≤12 mm/step, so it is never cascaded out.)
      • implied speed > ``max_speed_mps`` (default 2 m/s) — catches fast jitter.

    Set a threshold ≤0 to disable that test.
    """
    if prev_xyz is None:
        return False
    import numpy as _np

    dist = float(_np.linalg.norm(_np.asarray(xyz, dtype=_np.float64) - _np.asarray(prev_xyz, dtype=_np.float64)))
    if max_jump_m > 0 and dist > max_jump_m:
        return True
    if max_speed_mps > 0:
        dt = float(ts - prev_t) if (prev_t is not None and ts is not None) else (1.0 / 30.0)
        if dt <= 1e-6:
            dt = 1.0 / 30.0
        if (dist / dt) > max_speed_mps:
            return True
    return False


def filter_trace_outliers(
    xyz: np.ndarray,
    episode: np.ndarray,
    t: Optional[np.ndarray],
    max_speed_mps: float = 2.0,
    max_jump_m: float = 0.08,
) -> np.ndarray:
    """Boolean keep-mask dropping serial-read glitches per episode.

    Drops a point when, vs the previous *kept* point in the same episode, it
    either jumps > ``max_jump_m`` (also clears stuck/frozen-read runs) or implies
    a speed > ``max_speed_mps``. Both ≤0 → keep everything. See ``is_glitch_step``.
    """
    n = len(xyz)
    keep = np.ones(n, dtype=bool)
    if (max_speed_mps <= 0 and max_jump_m <= 0) or n == 0:
        return keep
    have_t = t is not None and len(t) == n
    for e in np.unique(episode):
        idx = np.where(episode == e)[0]
        last = None
        for i in idx:
            if last is None:
                last = i
                continue
            pt = t[last] if have_t else None
            ct = t[i] if have_t else None
            if is_glitch_step(xyz[last], pt, xyz[i], ct, max_speed_mps, max_jump_m):
                keep[i] = False  # drop; keep comparing the next point to last-good
            else:
                last = i
    return keep


def _dump_tracks_npz(
    tracks: Dict[int, List[List[float]]],
    times: Optional[Dict[int, List[float]]],
    path: Path,
    accumulate: bool = True,
) -> Optional[Path]:
    """Concatenate per-episode xyz tracks → npz.

    Saves ``xyz`` (N,3), ``episode`` (N,), and ``t`` (N,) absolute wall-clock
    timestamps (epoch seconds). ``t`` lets the offline renderer play back in REAL
    TIME (1x) with reset-gap durations preserved (front-view sync).

    With ``accumulate`` (default), episodes from a pre-existing npz at ``path``
    that are NOT in this dump are kept and merged. So phase2 collected across
    several resume runs accumulates ALL its episodes (e.g. 11–20) in one file
    instead of each run clobbering the last. Episodes present in this dump take
    precedence (a re-recorded episode is replaced). Output is sorted by timestamp.
    """
    times = times or {}
    eps = sorted(tracks.keys())
    xyz_parts: List[np.ndarray] = []
    ep_parts: List[np.ndarray] = []
    t_parts: List[np.ndarray] = []
    for ep in eps:
        pts = np.asarray(tracks[ep], dtype=np.float32)
        if pts.size == 0:
            continue
        xyz_parts.append(pts)
        ep_parts.append(np.full(len(pts), ep, dtype=np.int32))
        tt = np.asarray(times.get(ep, []), dtype=np.float64)
        if len(tt) != len(pts):  # defensive: keep arrays aligned
            tt = np.resize(tt, len(pts)) if len(tt) else np.zeros(len(pts), dtype=np.float64)
        t_parts.append(tt)
    if not xyz_parts:
        return None
    xyz = np.concatenate(xyz_parts, axis=0)
    episode = np.concatenate(ep_parts, axis=0)
    t = np.concatenate(t_parts, axis=0)

    if accumulate and Path(path).exists():
        try:
            old = np.load(path)
            old_ep = np.asarray(old["episode"])
            keep = ~np.isin(old_ep, np.asarray(eps))  # old episodes not re-recorded now
            if keep.any():
                old_t = (np.asarray(old["t"], dtype=np.float64)
                         if "t" in old.files and len(old["t"]) == len(old_ep)
                         else np.zeros(len(old_ep), dtype=np.float64))
                xyz = np.concatenate([np.asarray(old["xyz"], dtype=np.float32)[keep], xyz])
                episode = np.concatenate([old_ep[keep].astype(np.int32), episode])
                t = np.concatenate([old_t[keep], t])
        except Exception:  # noqa: BLE001 — never let merge break the dump
            pass

    # Sort by timestamp so the real-time render sees a clean non-decreasing axis
    # (older runs' episodes precede the current run's).
    if len(t) == len(xyz) and np.any(t > 0):
        order = np.argsort(t, kind="stable")
        xyz, episode, t = xyz[order], episode[order], t[order]

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, xyz=xyz, episode=episode, t=t)
    return path


def episode_color(episode: int) -> Tuple[int, int, int]:
    """Deterministic, well-separated RGB (0-255) per episode.

    Uses a golden-ratio hop around HSV hue so adjacent episode numbers get very
    different hues (not a smooth gradient that would look identical for nearby
    episodes). Shared by the live logger and the offline renderer so colors match.
    """
    import colorsys

    hue = (int(episode) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.98)
    return (int(r * 255), int(g * 255), int(b * 255))


class LiveEETraceWindow:
    """Background thread — streams accumulated EE xyz polylines to rerun."""

    # Incremental npz dump cadence (seconds) — bounds data loss on a hard kill.
    _DUMP_INTERVAL_S = 5.0

    def __init__(
        self,
        pipeline: Any,
        robot_id: int,
        fps: float = 15.0,
        ee_frame: str = "gripper_frame_link",
        urdf_path: Optional[str] = None,
        calib_path: Optional[str] = None,
        npz_path: Optional[str] = None,
        rrd_path: Optional[str] = None,
        spawn: bool = True,
        viewer: bool = True,
        max_speed_mps: float = 2.0,
        max_jump_m: float = 0.08,
    ):
        self.pipeline = pipeline
        self.robot_id = int(robot_id)
        self.fps = max(float(fps), 1.0)
        self.ee_frame = ee_frame
        self.urdf_path = urdf_path or default_urdf_path(self.robot_id)
        self.calib_path = calib_path or default_calib_path(self.robot_id)
        self.npz_path = npz_path
        self.rrd_path = rrd_path
        self.viewer = bool(viewer)
        # Viewer off ⇒ never pop a window, regardless of the legacy `spawn` arg.
        self.spawn = bool(spawn) and self.viewer
        # Skip rerun entirely when nothing consumes it: with no viewer and no
        # .rrd sink, rr.log() would buffer every message in RAM for the whole
        # session (the per-tick LineStrips3D re-logs the full track, so that
        # grows quadratically). npz accumulation is independent and still runs.
        self.use_rerun = self.viewer or bool(self.rrd_path)
        self.max_speed_mps = float(max_speed_mps)  # outlier (serial glitch) gate
        self.max_jump_m = float(max_jump_m)        # absolute-jump / stuck-read gate

        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        # episode -> list of [x, y, z] and parallel absolute timestamps (epoch s).
        # Protected by _lock for the stop()-time dump. Timestamps enable real-time
        # (1x) offline playback that stays in sync with a separate front-view video.
        self._tracks: Dict[int, List[List[float]]] = {}
        self._times: Dict[int, List[float]] = {}
        self._last_good: Dict[int, tuple] = {}  # ep -> (xyz, ts) of last accepted point
        self._lock = Lock()
        self._kin: Any = None
        self._calib: Any = None

    # ── lifecycle (mirrors LivePreviewWindow) ──────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # Build FK up-front (synchronous, ~0.5s once): a slow Pinocchio URDF load
        # must not race the first forward window in the polling thread, or the
        # opening frames of episode 1 would be silently missed.
        try:
            self._setup_fk()
        except Exception as e:  # noqa: BLE001
            print(f"[EETrace] FK setup failed ({e}); trace disabled", flush=True)
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="LiveEETraceWindow")
        self._thread.start()
        print(
            f"[EETrace] start robot{self.robot_id} @ {self.fps}fps "
            f"urdf={self.urdf_path}",
            flush=True,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._dump_npz()

    # ── setup ──────────────────────────────────────────────────────────────

    def _setup_fk(self) -> None:
        """Build FK engine + calibration (mirrors traj_analysis.fk_ee)."""
        self._kin, self._calib = build_fk(self.urdf_path, self.calib_path, self.ee_frame)

    def _ee_xyz(self, norm_state: np.ndarray) -> np.ndarray:
        """Normalized 6D servo state → EE (x, y, z) via calib radians + FK."""
        return ee_xyz_from_norm(self._kin, self._calib, norm_state)

    # ── poll loop ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        rr = None
        if self.use_rerun:
            try:
                import rerun as rr
            except Exception as e:  # noqa: BLE001
                print(f"[EETrace] rerun unavailable ({e}); npz-only mode", flush=True)
                rr = None
        if self._kin is None or self._calib is None:
            # start() builds FK before spawning the thread; guard against misuse.
            print("[EETrace] FK not initialized; trace disabled", flush=True)
            return

        if rr is not None:
            try:
                rr.init("ee_trace", spawn=self.spawn)
            except Exception as e:  # noqa: BLE001
                print(f"[EETrace] rerun init failed ({e}); npz-only mode", flush=True)
                rr = None
        else:
            print(
                "[EETrace] viewer off (TRAJ_VISUALIZER=false) — npz 누적만 수행",
                flush=True,
            )
        if rr is not None and self.rrd_path:
            try:
                rr.save(self.rrd_path)
                print(f"[EETrace] recording → {self.rrd_path}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[EETrace] rr.save failed ({e}); live-only", flush=True)

        # Robot base-frame origin (matches viz_spatial.py convention).
        if rr is not None:
            try:
                rr.log(
                    "ee_trace/origin",
                    rr.Points3D([[0.0, 0.0, 0.0]], colors=[0, 0, 0], radii=0.006),
                    static=True,
                )
            except Exception:  # noqa: BLE001
                pass

        period = 1.0 / self.fps
        step = 0
        warned_fk = False
        last_dump = time.perf_counter()
        dirty = False
        while not self._stop_event.is_set():
            started = time.perf_counter()

            # Periodic incremental dump so a hard kill (SIGKILL / double Ctrl+C)
            # still preserves work up to ~DUMP_INTERVAL ago. The atexit-driven
            # stop() does the final, complete dump on graceful exit.
            if dirty and (started - last_dump) >= self._DUMP_INTERVAL_S:
                self._dump_npz()
                last_dump = started
                dirty = False

            # Forward-only gate — reuse the pipeline's existing phase signal.
            if getattr(self.pipeline, "current_phase", None) != "Forward":
                time.sleep(period)
                continue

            norm = None
            try:
                norm = self.pipeline._latest_robot_state()
            except Exception:  # noqa: BLE001 — never disturb control path
                norm = None
            if norm is None or len(norm) < 5:
                time.sleep(period)
                continue

            try:
                xyz = self._ee_xyz(norm)
            except Exception as e:  # noqa: BLE001
                if not warned_fk:
                    print(f"[EETrace] FK error (suppressing further): {e}", flush=True)
                    warned_fk = True
                time.sleep(period)
                continue

            ep = int(getattr(self.pipeline, "current_episode", 0))
            ts = time.time()  # absolute wall-clock — used for real-time playback

            # Outlier gate: drop serial-read glitches (impossible EE jumps) so the
            # live view AND the npz stay clean. Compares to the last accepted point.
            prev = self._last_good.get(ep)
            if prev is not None and is_glitch_step(
                prev[0], prev[1], xyz, ts, self.max_speed_mps, self.max_jump_m
            ):
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
                continue
            self._last_good[ep] = (xyz, ts)

            with self._lock:
                track = self._tracks.setdefault(ep, [])
                track.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                self._times.setdefault(ep, []).append(ts)
                # Full-track copy only feeds the rerun LineStrips3D re-log; skip
                # it in npz-only mode so the poll loop stays O(1) per tick.
                track_snapshot = list(track) if rr is not None else None
            dirty = True

            if rr is None:
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
                continue

            color = episode_color(ep)
            try:
                # Real wall-clock timeline so the rerun live view / .rrd play back
                # in real time (syncable with an external front-view recording).
                rr.set_time_seconds("time", ts)
                step += 1
                ent = f"ee_trace/episode_{ep:02d}"
                if len(track_snapshot) >= 2:
                    rr.log(ent, rr.LineStrips3D([track_snapshot], colors=[color]))
                rr.log(
                    f"{ent}/head",
                    rr.Points3D([track_snapshot[-1]], colors=[color], radii=0.005),
                )
            except Exception as e:  # noqa: BLE001
                if not warned_fk:
                    print(f"[EETrace] rerun log error (suppressing): {e}", flush=True)
                    warned_fk = True

            time.sleep(max(0.0, period - (time.perf_counter() - started)))

    # ── npz dump (for offline mp4 render) ────────────────────────────────────

    def _resolve_npz_path(self) -> Path:
        if self.npz_path:
            return Path(self.npz_path)
        sd = getattr(self.pipeline, "_session_dir", None)
        if sd:
            # Phase2 dumps under <session>/phase2/ so it never clobbers the
            # phase1 trace at <session>/ee_trace.npz (needed for the combined
            # phase1+phase2 video). Mirrors _episode_dir's phase subdir.
            phase = str(getattr(self.pipeline, "method3_phase", "") or "").lower()
            sub = "phase2" if phase == "phase2" else ""
            return Path(sd) / sub / "ee_trace.npz"
        return Path("outputs/ee_trace/ee_trace.npz")

    def _dump_npz(self) -> None:
        with self._lock:
            tracks_copy = {ep: list(pts) for ep, pts in self._tracks.items()}
            times_copy = {ep: list(ts) for ep, ts in self._times.items()}
        if not tracks_copy:
            return
        try:
            out = _dump_tracks_npz(tracks_copy, times_copy, self._resolve_npz_path())
            if out is not None:
                n = sum(len(v) for v in tracks_copy.values())
                print(
                    f"[EETrace] dumped {n} pts across {len(tracks_copy)} episodes → {out}",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            print(f"[EETrace] npz dump failed ({e})", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Teleop recording (lerobot-record) — per-frame push, shared rerun session.
# ─────────────────────────────────────────────────────────────────────────────

# Servo position keys in lerobot SO101 follower observations (use_degrees=false →
# normalized -100..100 for the 5 arm motors). Order matches ARM_JOINTS / FK.
ARM_POS_KEYS = [f"{j}.pos" for j in ARM_JOINTS]


class TeleopEETraceLogger:
    """EE trajectory trace driven by the ``lerobot-record`` teleop loop.

    Unlike :class:`LiveEETraceWindow` (which polls a pipeline on its own thread),
    this is *pushed* one frame at a time from inside ``record_loop`` via
    :meth:`log_obs`. It does NOT own a rerun session — it logs into the recording
    that lerobot's ``display_data`` already started (so ``DISPLAY_DATA=true`` is
    required for the live 3D view). The per-episode xyz tracks are dumped to npz on
    :meth:`dump_npz` for offline mp4 rendering (scripts/render_ee_trace_video.py).
    """

    def __init__(
        self,
        robot_id: int,
        ee_frame: str = "gripper_frame_link",
        urdf_path: Optional[str] = None,
        calib_path: Optional[str] = None,
        npz_path: Optional[str] = None,
        max_speed_mps: float = 2.0,
        max_jump_m: float = 0.08,
    ):
        self.robot_id = int(robot_id)
        self.urdf_path = urdf_path or default_urdf_path(self.robot_id)
        self.calib_path = calib_path or default_calib_path(self.robot_id)
        self.npz_path = npz_path
        self.max_speed_mps = float(max_speed_mps)
        self.max_jump_m = float(max_jump_m)
        self._tracks: Dict[int, List[List[float]]] = {}
        self._times: Dict[int, List[float]] = {}
        self._last_good: Dict[int, tuple] = {}
        self._step = 0
        self._warned = False
        self._ok = False
        try:
            self._kin, self._calib = build_fk(self.urdf_path, self.calib_path, ee_frame)
            self._ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[EETrace] FK setup failed ({e}); teleop trace disabled", flush=True)
            self._kin = self._calib = None
        print(
            f"[EETrace] teleop logger robot{self.robot_id} "
            f"({'ready' if self._ok else 'DISABLED'}) urdf={self.urdf_path}",
            flush=True,
        )

    def log_obs(self, obs: Dict[str, Any], episode: int) -> None:
        """Extract arm servo positions from a lerobot observation dict, FK to EE
        xyz, accumulate, and stream the polyline to the shared rerun session."""
        if not self._ok:
            return
        try:
            norm5 = np.array([float(obs[k]) for k in ARM_POS_KEYS], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return  # keys absent (wrong robot type) — silently skip
        try:
            xyz = ee_xyz_from_norm(self._kin, self._calib, norm5)
        except Exception as e:  # noqa: BLE001
            if not self._warned:
                print(f"[EETrace] FK error (suppressing further): {e}", flush=True)
                self._warned = True
            return

        ep = int(episode)
        ts = time.time()  # absolute wall-clock — for real-time playback / sync
        prev = self._last_good.get(ep)
        if prev is not None and is_glitch_step(
            prev[0], prev[1], xyz, ts, self.max_speed_mps, self.max_jump_m
        ):
            return  # serial-read glitch — skip
        self._last_good[ep] = (xyz, ts)
        track = self._tracks.setdefault(ep, [])
        track.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        self._times.setdefault(ep, []).append(ts)
        color = episode_color(ep)
        try:
            import rerun as rr

            rr.set_time_seconds("time", ts)
            self._step += 1
            ent = f"ee_trace/episode_{ep:02d}"
            if len(track) >= 2:
                rr.log(ent, rr.LineStrips3D([track], colors=[color]))
            rr.log(f"{ent}/head", rr.Points3D([track[-1]], colors=[color], radii=0.005))
        except Exception as e:  # noqa: BLE001
            if not self._warned:
                print(f"[EETrace] rerun log error (suppressing): {e}", flush=True)
                self._warned = True

    def _resolve_npz_path(self) -> Path:
        if self.npz_path:
            return Path(self.npz_path)
        return Path("outputs/ee_trace/ee_trace.npz")

    def dump_npz(self) -> None:
        if not self._tracks:
            return
        try:
            out = _dump_tracks_npz(self._tracks, self._times, self._resolve_npz_path())
            if out is not None:
                n = sum(len(v) for v in self._tracks.values())
                print(
                    f"[EETrace] dumped {n} pts across {len(self._tracks)} episodes → {out}",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            print(f"[EETrace] npz dump failed ({e})", flush=True)


# Process-singleton, configured entirely from environment variables so the
# vendored lerobot-record call site stays generic (no hardcoded paths there).
_TELEOP_LOGGER: Optional[TeleopEETraceLogger] = None
_TELEOP_LOGGER_BUILT = False


def get_teleop_ee_logger() -> Optional[TeleopEETraceLogger]:
    """Return the process-wide teleop EE trace logger, or None if disabled.

    Reads ``EE_TRACE_ENABLED`` / ``EE_TRACE_ROBOT_ID`` / ``EE_TRACE_URDF`` /
    ``EE_TRACE_CALIB`` / ``EE_TRACE_NPZ`` from the environment. Built once.
    """
    global _TELEOP_LOGGER, _TELEOP_LOGGER_BUILT
    if _TELEOP_LOGGER_BUILT:
        return _TELEOP_LOGGER
    _TELEOP_LOGGER_BUILT = True
    if os.environ.get("EE_TRACE_ENABLED", "").lower() != "true":
        return None
    rid = os.environ.get("EE_TRACE_ROBOT_ID")
    if not rid:
        print("[EETrace] EE_TRACE_ENABLED but EE_TRACE_ROBOT_ID unset — disabled", flush=True)
        return None
    _TELEOP_LOGGER = TeleopEETraceLogger(
        robot_id=int(rid),
        urdf_path=os.environ.get("EE_TRACE_URDF") or None,
        calib_path=os.environ.get("EE_TRACE_CALIB") or None,
        npz_path=os.environ.get("EE_TRACE_NPZ") or None,
        max_speed_mps=float(os.environ.get("EE_TRACE_MAX_SPEED", "2.0")),
        max_jump_m=float(os.environ.get("EE_TRACE_MAX_JUMP", "0.08")),
    )
    return _TELEOP_LOGGER
