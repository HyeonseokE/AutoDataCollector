#!/usr/bin/env python3
"""
H3 실험: 기존 추론 chunk 에 저역통과 필터를 적용해 지그재그가
demo 수준으로 회복되는지 확인.

지표:
    - reversal_ratio : 청크 내부에서 연속 step 벡터의 cos<0 비율
    - mean_step (mm)   : per-step EE 이동 평균
    - mean_joint (deg) : per-step 조인트 변화량 평균

Usage:
    PYTHONPATH=src:scripts python -m chunk_visualizer.filter_analysis \
        --inference outputs/action_chunks/chunks_20260412_155741.npz \
        --demo outputs/action_chunks/demo_ep1.npz
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from chunk_visualizer.simple_fk import SimpleFK
from chunk_visualizer.visualize import compute_ee_trajectory, load_chunks


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA along axis 0. y[t] = alpha*x[t] + (1-alpha)*y[t-1]"""
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y


def sg(x: np.ndarray, window: int, order: int) -> np.ndarray:
    if len(x) < window:
        return x.astype(np.float64)
    return savgol_filter(x, window_length=window, polyorder=order, axis=0, mode="nearest")


def trajectory_metrics(chunk: np.ndarray, fk: SimpleFK) -> dict:
    traj = compute_ee_trajectory(chunk, fk)
    diffs = np.diff(traj, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    unit = diffs / np.maximum(norms[:, None], 1e-9)
    if len(unit) < 2:
        rev = 0.0
    else:
        cos = np.sum(unit[:-1] * unit[1:], axis=1)
        rev = float((cos < 0).sum() / len(cos))
    return {
        "reversal_ratio": rev,
        "mean_step_mm": float(norms.mean() * 1000),
        "mean_joint_deg": float(np.abs(np.diff(chunk, axis=0)).mean()),
    }


def aggregate(chunks, fk) -> dict:
    vals = [trajectory_metrics(c, fk) for c in chunks if len(c) >= 3]
    keys = vals[0].keys()
    return {k: float(np.mean([v[k] for v in vals])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--urdf", default="/home/lerobot/AutoDataCollector/assets/urdf/so101_robot2.urdf")
    ap.add_argument("--ee-frame", default="gripper_frame_link")
    ap.add_argument("--joint-names", nargs="+",
                    default=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])
    args = ap.parse_args()

    fk = SimpleFK(urdf_path=args.urdf, ee_frame=args.ee_frame, joint_names=args.joint_names)

    demo = load_chunks(args.demo)["chunks"]
    infer = load_chunks(args.inference)["chunks"]

    print(f"Loaded {len(demo)} demo chunks, {len(infer)} inference chunks")
    print()

    # baselines
    results = []
    results.append(("DEMO (baseline)", aggregate(demo, fk)))
    results.append(("INFER raw", aggregate(infer, fk)))

    # EMA on joint-space actions
    for alpha in (0.6, 0.4, 0.2):
        smoothed = [ema(c.astype(np.float64), alpha) for c in infer]
        results.append((f"INFER EMA α={alpha}", aggregate(smoothed, fk)))

    # Savitzky-Golay on joint-space actions
    for win, order in [(5, 2), (9, 2), (15, 3)]:
        smoothed = [sg(c.astype(np.float64), win, order) for c in infer]
        results.append((f"INFER SG win={win} order={order}", aggregate(smoothed, fk)))

    # pretty print
    print(f"{'config':30s} | {'reversal':>10s} | {'EE Δ mm':>8s} | {'joint Δ°':>9s}")
    print("-" * 68)
    for name, m in results:
        print(f"{name:30s} | {m['reversal_ratio']*100:>9.2f}% | {m['mean_step_mm']:>8.3f} | {m['mean_joint_deg']:>9.4f}")


if __name__ == "__main__":
    main()
