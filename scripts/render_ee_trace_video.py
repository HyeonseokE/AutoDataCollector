#!/usr/bin/env python3
"""Offline render: ee_trace .npz → accumulating 3D mp4/gif.

Reads the per-episode EE (x, y, z) trace dumped by
``record_dataset.live_ee_trace.LiveEETraceWindow`` and renders a video where the
trajectory accumulates step-by-step (matching the live view), each episode in a
distinct color, with the camera slowly rotating. Deterministic and headless-safe
(matplotlib Agg + imageio's bundled ffmpeg — no system ffmpeg required).

Usage::

    python scripts/render_ee_trace_video.py \\
        --npz outputs/ee_trace/ee_trace.npz \\
        --output outputs/ee_trace/ee_trace.mp4 \\
        --fps 30 --stride 2 --rotate-deg 0.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers 3d projection)

# Reuse the exact color mapping the live logger used.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
try:
    from record_dataset.live_ee_trace import episode_color as _episode_color
except Exception:  # noqa: BLE001 — fallback so the renderer is standalone
    import colorsys

    def _episode_color(episode: int) -> Tuple[int, int, int]:
        hue = (int(episode) * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.98)
        return (int(r * 255), int(g * 255), int(b * 255))


def _color01(episode: int) -> Tuple[float, float, float]:
    r, g, b = _episode_color(episode)
    return (r / 255.0, g / 255.0, b / 255.0)


def load_trace(npz_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (xyz (N,3), episode (N,), t (N,)) preserving dump order.

    ``npz_path`` may be a single path or a list of paths; multiple are merged in
    order (used for the combined phase1+phase2 video — episode numbers stay
    distinct, e.g. 1–10 then 11–20). ``t`` (absolute epoch seconds) is present
    for timestamped traces; empty if any input predates timestamp support.
    """
    paths = list(npz_path) if isinstance(npz_path, (list, tuple)) else [npz_path]
    xyz_parts, ep_parts, t_parts = [], [], []
    have_t = True
    for p in paths:
        data = np.load(p)
        x = np.asarray(data["xyz"], dtype=np.float32)
        e = np.asarray(data["episode"], dtype=np.int32)
        if x.ndim != 2 or x.shape[1] != 3 or len(x) != len(e):
            raise ValueError(f"bad trace arrays in {p}")
        xyz_parts.append(x)
        ep_parts.append(e)
        if "t" in data.files and len(data["t"]) == len(x):
            t_parts.append(np.asarray(data["t"], dtype=np.float64))
        else:
            have_t = False
    xyz = np.concatenate(xyz_parts) if xyz_parts else np.empty((0, 3), np.float32)
    episode = np.concatenate(ep_parts) if ep_parts else np.empty(0, np.int32)
    t = np.concatenate(t_parts) if (have_t and t_parts) else np.empty(0)
    return xyz, episode, t


def _equal_bounds(xyz: np.ndarray, pad: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Cube-ish bounds so the 3D aspect isn't distorted."""
    lo = xyz.min(axis=0) - pad
    hi = xyz.max(axis=0) + pad
    center = (lo + hi) / 2.0
    half = float((hi - lo).max()) / 2.0 + pad
    return center - half, center + half


def render(
    npz_path: Path,
    out_path: Path,
    fps: int = 30,
    stride: int = 2,
    rotate_deg: float = 0.4,
    elev: float = 22.0,
    azim0: float = -60.0,
    dpi: int = 110,
    max_frames: int = 0,
    realtime: bool = False,
    speed: float = 1.0,
    eye: Optional[Tuple[float, float, float]] = None,
    look_at: Optional[Tuple[float, float, float]] = None,
    head_size: float = 90.0,
    episode: Optional[int] = None,
    max_speed: float = 2.0,
    max_jump: float = 0.08,
    max_gap: float = 0.0,
    backdrop_npz=None,
) -> None:
    def _filter(x, e, tt):
        if max_speed <= 0 and max_jump <= 0:
            return x, e, tt, 0
        try:
            from record_dataset.live_ee_trace import filter_trace_outliers
            k = filter_trace_outliers(x, e, tt, max_speed, max_jump)
            d = int((~k).sum())
            if d:
                return x[k], e[k], (tt[k] if len(tt) == len(k) else tt), d
        except Exception as ex:  # noqa: BLE001
            print(f"[render] outlier filter skipped ({ex})", flush=True)
        return x, e, tt, 0

    xyz, episode_arr, t = load_trace(npz_path)
    if len(xyz) == 0:
        raise SystemExit("trace is empty — nothing to render")
    xyz, episode_arr, t, dropped = _filter(xyz, episode_arr, t)
    if dropped:
        print(f"[render] dropped {dropped} glitch pts "
              f"(>{max_speed} m/s or >{max_jump*1000:.0f} mm jump)", flush=True)

    # Backdrop: a separate trace (e.g. phase1) drawn in GRAY, fully + statically,
    # behind the accumulating coloured `npz_path` trace (e.g. phase2). The video
    # timeline is driven ONLY by the main trace, so the backdrop shows "what was
    # already collected" while the new phase builds up on top in colour.
    bd_episodes = {}
    if backdrop_npz is not None:
        try:
            bx, be, bt = load_trace(backdrop_npz)
            bx, be, _bt, _d = _filter(bx, be, bt)
            # The backdrop is static and redrawn EVERY frame, so downsample it
            # heavily — it's only a visual reference, and plotting tens of
            # thousands of 3D points per frame is what makes the render crawl.
            bstride = max(1, len(bx) // 3000)
            for e in np.unique(be):
                pts = bx[be == e]
                bd_episodes[int(e)] = pts[::bstride] if bstride > 1 else pts
            print(f"[render] backdrop: {len(bx)} pts → ~{sum(len(v) for v in bd_episodes.values())} "
                  f"(stride {bstride}), {len(bd_episodes)} episodes (gray)",
                  flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[render] backdrop load failed ({ex}); skipping", flush=True)

    if episode is not None:  # single-episode preview/dummy
        mask = episode_arr == int(episode)
        if not np.any(mask):
            raise SystemExit(f"episode {episode} not found (have {sorted(set(episode_arr.tolist()))})")
        xyz, episode_arr = xyz[mask], episode_arr[mask]
        t = t[mask] if len(t) == len(mask) else t
        print(f"[render] single-episode preview: episode {episode} ({len(xyz)} pts)", flush=True)
    episode = episode_arr  # downstream uses `episode` as the per-point array

    eps = sorted(np.unique(episode).tolist())
    # Per-episode contiguous spans (dump order is grouped by episode already).
    spans = {ep: np.where(episode == ep)[0] for ep in eps}
    # Bounds span main + backdrop so the gray backdrop is fully framed.
    bounds_pts = xyz if not bd_episodes else np.concatenate([xyz] + list(bd_episodes.values()))
    lo, hi = _equal_bounds(bounds_pts)
    n = len(xyz)

    # Custom viewpoint: camera positioned at `eye` (robot-origin frame) looking
    # toward `look_at` (default = first EE point = the EE initial position).
    # matplotlib's 3D camera is direction-only (elev/azim from box center), so we
    # derive elev/azim from the eye→target direction and lock rotation.
    if eye is not None:
        target = np.asarray(look_at if look_at is not None else xyz[0], dtype=np.float64)
        v = np.asarray(eye, dtype=np.float64) - target  # box-center → camera direction
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            raise SystemExit("--eye coincides with look target; pick a different eye")
        v /= norm
        elev = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
        azim0 = float(np.degrees(np.arctan2(v[1], v[0])))
        rotate_deg = 0.0  # fixed viewpoint
        print(
            f"[render] custom view: eye={tuple(round(float(x),3) for x in eye)} "
            f"look_at={tuple(round(float(x),3) for x in target)} "
            f"→ elev={elev:.1f}° azim={azim0:.1f}°",
            flush=True,
        )

    # frames: list of (reveal_count, title_suffix, azim). reveal_count = number of
    # leading points visible at that frame.
    frames: List[Tuple[int, str, float]] = []
    if realtime:
        # REAL-TIME 1x: one frame per fixed wall-clock slice; reveal points by
        # timestamp. Episode-gap (reset) durations are preserved as real elapsed
        # time → the trajectory freezes during resets while the clock keeps
        # running, so the video stays frame-for-frame aligned with a continuously
        # filmed front-view. NO temporal sub-sampling (time axis is undistorted).
        if len(t) != n or not np.any(t > 0):
            raise SystemExit(
                "npz has no usable timestamps ('t'); cannot render --realtime.\n"
                "This trace predates timestamp logging — re-record, or drop --realtime."
            )
        # t is non-decreasing (episodes happen in wall-clock order); ensure it.
        t_mono = np.maximum.accumulate(t)
        # Optionally cap idle gaps. With max_gap>0 every inter-sample gap is
        # clamped, so the long pause between two separate runs (phase1→phase2,
        # possibly hours) — and overly long resets — collapse to ``max_gap`` s
        # while motion stays 1x. Used for the combined video; keep 0 for a single
        # session that must stay frame-for-frame with its front-view.
        if max_gap and max_gap > 0:
            d = np.diff(t_mono, prepend=t_mono[0])
            d = np.minimum(d, float(max_gap))
            t_eff = np.cumsum(d)
        else:
            t_eff = t_mono - t_mono[0]
        span = max(float(t_eff[-1]), 1e-6)
        dt = float(speed) / float(fps)
        n_steps = int(np.floor(span / dt))
        for f in range(n_steps + 1):
            T = f * dt
            r = int(np.searchsorted(t_eff, T, side="right"))
            cur_ep = int(episode[min(max(r, 1), n) - 1])
            frames.append((r, f"t={T:5.1f}s  ep{cur_ep:02d}", azim0 + f * float(rotate_deg)))
        frames.append((n, f"t={span:5.1f}s  ep{int(episode[-1]):02d}", azim0 + (n_steps + 1) * float(rotate_deg)))
        print(
            f"[render] REAL-TIME {span:.1f}s @ {fps}fps (speed {speed}x) → "
            f"{len(frames)} frames, {n} pts, {len(eps)} episodes → {out_path}",
            flush=True,
        )
    else:
        stride = max(1, int(stride))
        if max_frames and (n + stride - 1) // stride > max_frames:
            stride = (n + max_frames - 1) // max_frames
        reveal_indices = list(range(1, n + 1, stride))
        if reveal_indices[-1] != n:
            reveal_indices.append(n)
        for fi, r in enumerate(reveal_indices):
            cur_ep = int(episode[min(r, n) - 1])
            frames.append((r, f"episode {cur_ep:02d}  ({r}/{n})", azim0 + fi * float(rotate_deg)))
        print(
            f"[render] {n} pts, {len(eps)} episodes, {len(frames)} frames → {out_path}",
            flush=True,
        )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_gif = out_path.suffix.lower() == ".gif"

    import imageio.v2 as imageio

    writer_kwargs = {"fps": fps} if is_gif else {"fps": fps, "macro_block_size": None}
    writer = imageio.get_writer(str(out_path), **writer_kwargs)
    last_key = None
    last_frame = None
    try:
        for r, title, azim in frames:
            # Reuse the previous frame verbatim when nothing changed (frozen
            # trajectory during a reset gap + fixed camera). Makes real-time
            # rendering of long idle/reset spans nearly free.
            key = (r, round(azim, 4))
            if key == last_key and last_frame is not None:
                writer.append_data(last_frame)
                continue

            ax.clear()
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.scatter([0], [0], [0], c="black", s=120, marker="*", zorder=10)

            # Gray backdrop (e.g. phase1), drawn fully & statically behind.
            for bpts in bd_episodes.values():
                ax.plot(bpts[:, 0], bpts[:, 1], bpts[:, 2],
                        color=(0.6, 0.6, 0.6), lw=1.0, alpha=0.45, zorder=1)

            for ep in eps:
                idx = spans[ep]
                idx = idx[idx < r]  # reveal only points up to this frame
                if len(idx) == 0:
                    continue
                pts = xyz[idx]
                c = _color01(ep)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=c, lw=1.6, alpha=0.95, zorder=5)

            # Bold marker on the CURRENT EE position (latest revealed point).
            if r >= 1:
                ci = min(r, n) - 1
                cc = _color01(int(episode[ci]))
                ax.scatter(
                    xyz[ci, 0], xyz[ci, 1], xyz[ci, 2],
                    color=cc, s=head_size, marker="o",
                    edgecolors="black", linewidths=1.2, zorder=20,
                )

            ax.set_title(f"EE trajectory — {title}")
            ax.view_init(elev=elev, azim=azim)

            fig.canvas.draw()
            frame = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            writer.append_data(frame)
            last_key = key
            last_frame = frame
    finally:
        writer.close()
        plt.close(fig)
    print(f"[render] done → {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, nargs="+",
                    help="ee_trace .npz file(s). Multiple are merged in order "
                         "(e.g. phase1.npz phase2.npz for the combined video).")
    ap.add_argument("--output", required=True, help="output .mp4 or .gif")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--stride", type=int, default=2,
        help="reveal this many points per frame (higher = shorter video)",
    )
    ap.add_argument("--rotate-deg", type=float, default=0.4,
                    help="azimuth deg per frame (0 = fixed viewpoint, no orbit)")
    ap.add_argument("--elev", type=float, default=22.0, help="camera elevation deg")
    ap.add_argument("--azim0", type=float, default=-60.0, help="starting azimuth deg")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="(index mode only) cap total frames; 0 = no cap")
    ap.add_argument("--realtime", action="store_true",
                    help="real-time 1x playback by timestamp (no time축약; syncs with "
                         "a separately-filmed real-time video). Requires 't' in npz.")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="realtime speed factor (1.0 = true real-time)")
    ap.add_argument("--eye", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                    help="camera eye position (robot-origin frame); looks toward --look-at. "
                         "Overrides elev/azim and locks rotation.")
    ap.add_argument("--look-at", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                    help="look target; default = first EE point (EE initial position)")
    ap.add_argument("--episode", type=int, default=None,
                    help="render only this episode (dummy/preview)")
    ap.add_argument("--head-size", type=float, default=90.0,
                    help="size of the bold current-EE marker")
    ap.add_argument("--max-speed", type=float, default=2.0,
                    help="drop points whose EE speed exceeds this (m/s) as glitches; "
                         "0 = disable")
    ap.add_argument("--max-jump", type=float, default=0.08,
                    help="drop points jumping more than this (m) from the last good "
                         "point (also clears stuck/frozen reads); 0 = disable")
    ap.add_argument("--max-gap", type=float, default=0.0,
                    help="(realtime) cap idle gaps to this many seconds — collapses "
                         "the long pause between merged runs. 0 = keep true gaps "
                         "(use for single-session front-view sync).")
    ap.add_argument("--backdrop", nargs="+", default=None,
                    help="npz(s) drawn in GRAY statically behind the main (coloured, "
                         "accumulating) trace. Timeline follows the main npz only. "
                         "e.g. phase1.npz as backdrop under accumulating phase2.")
    args = ap.parse_args()

    npz_arg = [Path(p) for p in args.npz]
    render(
        npz_arg if len(npz_arg) > 1 else npz_arg[0],
        Path(args.output),
        backdrop_npz=([Path(p) for p in args.backdrop] if args.backdrop else None),
        fps=args.fps,
        stride=args.stride,
        rotate_deg=args.rotate_deg,
        elev=args.elev,
        azim0=args.azim0,
        dpi=args.dpi,
        max_frames=args.max_frames,
        realtime=args.realtime,
        speed=args.speed,
        eye=tuple(args.eye) if args.eye else None,
        look_at=tuple(args.look_at) if args.look_at else None,
        head_size=args.head_size,
        episode=args.episode,
        max_speed=args.max_speed,
        max_jump=args.max_jump,
        max_gap=args.max_gap,
    )


if __name__ == "__main__":
    main()
