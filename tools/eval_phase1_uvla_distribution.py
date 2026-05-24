"""D_phase1 (ID) 의 U_VLA 분포 측정 — Phase2 chosen 의 OOD/ID 판정 threshold 결정용.

paper §3.4 식 (18):
    U_VLA(ξ) = Agg_τ [(1/R) Σ_r L_denoise^{(r)}(o_τ, I, p_τ, A_{τ:τ+H-1}; π_θ^{(1)})]

D_phase1 의 (skill segment, frame_start) sample 각각에 대해 위 식을 평가하여
distribution (μ, σ, p50, p90, p95, p99) 을 산출한다. 이 distribution 의 quantile
(예: p95) 을 ``tau_U_ID`` 로 채택하여 Phase2 selector 의 OOD/ID gate 로 사용.

출력은 ckpt 디렉터리의 ``../uvla_id_stats.json`` (default) — server 가 ckpt 옆
sidecar 로 자동 lookup 하여 ``Phase2MIConfig.tau_U_ID`` 로 주입한다. ``--dump_npz``
지정 시 per-sample raw u_vla 도 별도 npz 로 저장 (debug/plot).

Usage:
    python tools/eval_phase1_uvla_distribution.py \\
        --vla lerobot/outputs/train/smolvla_dct_20260524_113901/checkpoints/001400/pretrained_model \\
        --dataset CoRL2026-CSI/pnp_ours_100_table1 \\
        --n_samples 100 --R 4 --sigma 0.5 --agg mean
        # --out 미지정 → {ckpt}/../uvla_id_stats.json 자동 저장
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Camera key rename — D_phase1 (LeRobot v3) 의 좌측 wrist / top → camera1/2.
# smoke_acquisition.py 와 동일 매핑 (DCT-tuned smolvla 가 학습 때 본 키).
CAMERA_RENAME = {
    "observation.images.left_wrist": "observation.images.camera1",
    "observation.images.top": "observation.images.camera2",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet", type=Path,
        default=REPO_ROOT / "results/skill_dct/pnp_ours_100_table1.parquet",
        help="skill_dct sidecar parquet (build_skill_dct.py 산물).",
    )
    p.add_argument(
        "--dataset", type=str,
        default="CoRL2026-CSI/pnp_ours_100_table1",
        help="HF LeRobot dataset repo_id 또는 local path.",
    )
    p.add_argument(
        "--vla", type=Path,
        default=REPO_ROOT / "lerobot/outputs/train/smolvla_dct_20260524_113901/checkpoints/001400/pretrained_model",
        help="DCT-tuned smolvla ckpt 디렉터리.",
    )
    p.add_argument("--n_samples", type=int, default=100, help="parquet row sample 수.")
    p.add_argument("--R", type=int, default=4, help="noise sample 회수 (paper §3.4 eq 18).")
    p.add_argument("--sigma", type=float, default=0.5, help="flow-matching time (None=schedule random).")
    p.add_argument("--agg", choices=["mean", "max"], default="mean", help="R aggregator.")
    p.add_argument("--seed", type=int, default=20260524, help="random sample seed (재현성).")
    p.add_argument(
        "--out", type=Path, default=None,
        help="JSON sidecar 경로. 미지정 시 {ckpt}/../uvla_id_stats.json (server auto-lookup).",
    )
    p.add_argument(
        "--dump_npz", type=Path, default=None,
        help="(옵션) per-sample raw u_vla npz dump 경로. debug/plot 용.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="output JSON 이 이미 있어도 overwrite.",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="torch device (예: cuda, cuda:0, cpu).",
    )
    return p.parse_args()


def default_json_out(vla_path: Path) -> Path:
    """ckpt 디렉터리 옆 uvla_id_stats.json — server auto-lookup 위치."""
    # vla_path = .../checkpoints/001400/pretrained_model → parent = checkpoints/001400
    return Path(vla_path).parent / "uvla_id_stats.json"


def load_components(args: argparse.Namespace):
    """parquet rows, LeRobot dataset, VLA policy, preprocessor 를 한번에 로드."""
    print(f"[uvla_id] parquet  = {args.parquet}")
    table = pq.read_table(args.parquet).to_pandas()
    print(f"  rows = {len(table)}  (skill segments)")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    print(f"[uvla_id] dataset  = {args.dataset}")
    base = LeRobotDataset(args.dataset, video_backend="pyav")

    print(f"[uvla_id] vla ckpt = {args.vla}")
    policy = SmolVLAPolicy.from_pretrained(str(args.vla)).to(args.device).eval()
    preprocessor = DataProcessorPipeline.from_pretrained(
        str(args.vla), config_filename="policy_preprocessor.json"
    )
    return table, base, policy, preprocessor


def build_batch_input(frame: dict, instruction: str, skill_type: str, preprocessor, device: str):
    """LeRobotDataset frame → smolvla preprocessor batch (smoke_acquisition.py 와 동일 패턴)."""
    import torch

    item = dict(frame)
    item["task"] = f"{skill_type}: {instruction}"
    item = {CAMERA_RENAME.get(k, k): v for k, v in item.items()}
    batch_input: dict = {}
    for k, v in item.items():
        if k.startswith("_"):
            continue
        if isinstance(v, torch.Tensor):
            batch_input[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, str):
            batch_input[k] = [v]
        else:
            batch_input[k] = v
    return preprocessor(batch_input)


def build_candidate(row, base, preprocessor, device: str):
    """parquet row → Phase2Candidate (observations 채워, score_batch 가 그대로 사용)."""
    from method3.phase2_mi_selection.mi_selector import Phase2Candidate

    episode_id = str(row["episode_id"])
    skill_type = str(row["skill_type"])
    instruction = str(row["instruction"])
    frame_start = int(row["frame_start"])
    dct_target = np.asarray(row["dct_target"], dtype=np.float64).reshape(50, -1)  # (50, 6)

    frame = base[frame_start]
    batch_input = build_batch_input(frame, instruction, skill_type, preprocessor, device)

    # proprio (서비스 안 됨 — observations 안에 이미 있음). dummy 1 window.
    P = 6
    proprios = np.zeros((1, P), dtype=np.float32)
    H, A = dct_target.shape  # 50, 6
    action_chunks = np.zeros((1, H, A), dtype=np.float32)  # 안 쓰임 (dct_target 으로 swap)

    return Phase2Candidate(
        skill_id=skill_type,
        state_keys=np.zeros((1, 1), dtype=np.float32),  # MI 경로 안 씀
        action_chunks=action_chunks,
        observations=batch_input,
        instruction=instruction,
        proprios=proprios,
        dct_target=dct_target,
    ), episode_id, skill_type


def report_stats(u_vla: np.ndarray, args: argparse.Namespace) -> dict:
    """distribution 통계 print + return."""
    pcts = [50, 75, 90, 95, 99]
    stats = {
        "n": int(u_vla.size),
        "mean": float(u_vla.mean()),
        "std": float(u_vla.std(ddof=1)) if u_vla.size > 1 else 0.0,
        "min": float(u_vla.min()),
        "max": float(u_vla.max()),
    }
    for p in pcts:
        stats[f"p{p}"] = float(np.percentile(u_vla, p))

    print("\n" + "=" * 60)
    print(f"D_phase1 (ID) U_VLA distribution  [n={stats['n']}, "
          f"R={args.R}, sigma={args.sigma}, agg={args.agg}]")
    print("=" * 60)
    print(f"  mean   = {stats['mean']:.4f}")
    print(f"  std    = {stats['std']:.4f}")
    print(f"  min    = {stats['min']:.4f}")
    for p in pcts:
        print(f"  p{p:<2d}    = {stats[f'p{p}']:.4f}")
    print(f"  max    = {stats['max']:.4f}")
    print("=" * 60)
    print(f"Suggested tau_U_ID candidates:")
    print(f"  p95 = {stats['p95']:.4f}   (5% ID -> OOD, paper-friendly default)")
    print(f"  p99 = {stats['p99']:.4f}   (1% ID -> OOD, strict)")
    print(f"  mean + 2*std = {stats['mean'] + 2 * stats['std']:.4f}   (Gaussian 95%)")
    print("=" * 60)
    return stats


def main() -> int:
    args = parse_args()

    # JSON 경로 결정 + skip-if-exists.
    out_json = args.out if args.out is not None else default_json_out(args.vla)
    if out_json.exists() and not args.force:
        print(f"[uvla_id] {out_json} already exists — pass --force to overwrite.")
        return 0

    table, base, policy, preprocessor = load_components(args)

    from method3.phase2_mi_selection.vla_informativeness import (
        LeRobotVLAInformativenessScorer,
    )

    n = min(args.n_samples, len(table))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(table), size=n, replace=False)
    print(f"[uvla_id] sampling {n}/{len(table)} rows  seed={args.seed}")

    scorer = LeRobotVLAInformativenessScorer(
        policy=policy,
        batch_builder=lambda c: c.observations,  # observations 이미 batch — passthrough
        mode="dct",
        R=int(args.R),
        agg=str(args.agg),
        sigma=None if args.sigma is None else float(args.sigma),
    )

    u_vla = np.full(n, np.nan, dtype=np.float64)
    eps = np.empty(n, dtype=object)
    skills = np.empty(n, dtype=object)
    t0 = time.monotonic()
    for i, ridx in enumerate(idxs):
        row = table.iloc[int(ridx)]
        try:
            cand, ep, sk = build_candidate(row, base, preprocessor, args.device)
            scores = scorer.score_batch([cand], batch_size=1)
            u_vla[i] = float(scores[0])
            eps[i] = ep
            skills[i] = sk
        except Exception as e:
            print(f"  [warn] row idx={int(ridx)} ({row['episode_id']}, "
                  f"{row['skill_type']}) score failed: {e}", flush=True)
            eps[i] = str(row["episode_id"])
            skills[i] = str(row["skill_type"])

        if (i + 1) % 10 == 0 or i + 1 == n:
            elapsed = time.monotonic() - t0
            done = np.isfinite(u_vla[:i + 1]).sum()
            print(f"  [{i + 1:>3d}/{n}] valid={done}  elapsed={elapsed:.1f}s  "
                  f"({elapsed / (i + 1):.2f}s/cand)", flush=True)

    valid = np.isfinite(u_vla)
    if valid.sum() == 0:
        print("[uvla_id][ERROR] no valid U_VLA computed.", file=sys.stderr)
        return 2

    stats = report_stats(u_vla[valid], args)

    # JSON sidecar — server method3_setup.py 가 자동 lookup.
    payload = {
        "version": "1",
        "ckpt_path": str(args.vla),
        "dataset": args.dataset,
        "parquet": str(args.parquet),
        "params": {
            "R": int(args.R),
            "sigma": (None if args.sigma is None else float(args.sigma)),
            "agg": str(args.agg),
            "seed": int(args.seed),
            "n_samples": int(args.n_samples),
            "mean_plus_2sigma": float(stats["mean"] + 2 * stats["std"]),
        },
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "stats": {k: (int(v) if k == "n" else float(v)) for k, v in stats.items()},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[uvla_id] JSON sidecar -> {out_json}")

    # 옵션: per-sample raw dump (debug/plot).
    if args.dump_npz is not None:
        args.dump_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.dump_npz,
            u_vla=u_vla,
            episode_id=eps,
            skill_type=skills,
        )
        print(f"[uvla_id] raw npz -> {args.dump_npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
