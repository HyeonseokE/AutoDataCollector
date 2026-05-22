"""DCT-target smolvla 추론 quality 측정 — single-step inference.

각 train sample (skill segment 의 시작 frame obs + skill_type prefix) 으로:
  1. policy.predict_action_chunk → predicted DCT_50 (postprocess unnormalize 후)
  2. GT DCT_50 (sidecar parquet) 와 DCT-space MSE
  3. dct_to_traj(pred_dct, T_skill) 으로 trajectory 복원 → raw GT trajectory 와
     joint-space MSE
  4. per-skill-type aggregate

Usage:
    python -m method3.dct.validate_dct_inference \\
        --ckpt    .../checkpoints/003750/pretrained_model \\
        --dataset CoRL2026-CSI/pnp_phase1_30_table2 \\
        --parquet .../skill_dct/pnp_phase1_30_table2.parquet
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# vendored lerobot path
_LR = Path(__file__).resolve().parents[2] / "lerobot" / "src"
if str(_LR) not in sys.path:
    sys.path.insert(0, str(_LR))

from method3.dct.skill_dataset import load_dct_targets
from method3.dct.skill_dct_dataset import SkillDCTDataset
from method3.dct.transform import dct_to_traj


def _apply_rename(item: dict, rename: dict) -> dict:
    out = {}
    for k, v in item.items():
        out[rename.get(k, k)] = v
    return out


def _to_batch(item: dict, device: str) -> dict:
    """SkillDCTDataset item → policy batch (unsqueeze + device + drop meta)."""
    batch = {}
    for k, v in item.items():
        if k.startswith("_"):
            continue
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True,
                   help="preprocessor / postprocessor 가 있는 ckpt 디렉토리.")
    p.add_argument("--policy-ckpt", default=None,
                   help="설정 시 policy weights 는 여기서 load (preprocessor 는 "
                        "--ckpt 에서 유지). base smolvla baseline 평가용. "
                        "lerobot/smolvla_base 같은 HF repo id 또는 path.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_samples", type=int, default=None)
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    # train_smolvla.sh 가 적용했던 camera rename — inference 도 같은 rename 필요.
    rename = {
        "observation.images.left_wrist": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
    }

    policy_src = args.policy_ckpt or args.ckpt
    print(f"[validate] loading policy from {policy_src}"
          + (f"  (preprocessor from {args.ckpt})" if args.policy_ckpt else ""))
    policy = SmolVLAPolicy.from_pretrained(policy_src)
    policy.to(args.device).eval()

    preprocessor = DataProcessorPipeline.from_pretrained(
        args.ckpt, config_filename="policy_preprocessor.json"
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        args.ckpt, config_filename="policy_postprocessor.json"
    )

    print(f"[validate] loading dataset {args.dataset}")
    base = LeRobotDataset(args.dataset, video_backend="pyav")
    wrap = SkillDCTDataset(base, args.parquet)
    segments = load_dct_targets(args.parquet)
    assert len(wrap) == len(segments)

    n = len(wrap) if args.n_samples is None else min(args.n_samples, len(wrap))
    print(f"[validate] inferring on {n} segments")

    dct_mse_by_skill: dict[str, list[float]] = defaultdict(list)
    traj_mse_by_skill: dict[str, list[float]] = defaultdict(list)

    for idx in range(n):
        item = wrap[idx]
        seg = segments[idx]
        item = _apply_rename(item, rename)
        batch = _to_batch(item, args.device)
        batch = preprocessor(batch)

        with torch.no_grad():
            pred = policy.predict_action_chunk(batch)  # (1, 50, 6)

        # unnormalize predicted DCT 계수.
        out = postprocessor({"action": pred})
        pred_dct = out["action"].squeeze(0).cpu().numpy()  # (50, 6)
        gt_dct = seg.dct_target  # (50, 6)

        dct_mse = float(np.mean((pred_dct - gt_dct) ** 2))
        dct_mse_by_skill[seg.skill_type].append(dct_mse)

        # IDCT → trajectory (T_skill 길이). GT trajectory 는 raw dataset
        # action [frame_start:frame_end] 에서 추출.
        T = seg.frame_end - seg.frame_start
        pred_traj = dct_to_traj(pred_dct, T_target=T)
        gt_traj = np.stack([
            base[f]["action"].numpy() if hasattr(base[f]["action"], "numpy") else np.asarray(base[f]["action"])
            for f in range(seg.frame_start, seg.frame_end)
        ])
        traj_mse = float(np.mean((pred_traj - gt_traj) ** 2))
        traj_mse_by_skill[seg.skill_type].append(traj_mse)

        if (idx + 1) % 20 == 0 or idx == n - 1:
            print(f"  [{idx+1}/{n}] last skill={seg.skill_type}  dct_mse={dct_mse:.4f}  traj_mse={traj_mse:.3f}")

    # ── summary ───────────────────────────────────────────────────
    def _row(name: str, vs: list[float]) -> str:
        a = np.asarray(vs, dtype=np.float64)
        return (f"  {name:20s} n={len(a):3d}  mean={a.mean():>9.4f}  "
                f"median={np.median(a):>9.4f}  p90={np.quantile(a, 0.9):>9.4f}  "
                f"max={a.max():>9.4f}")

    print("\n=== DCT-space MSE per skill_type (normalized after unnormalize) ===")
    for k in sorted(dct_mse_by_skill):
        print(_row(k, dct_mse_by_skill[k]))
    all_dct = [v for vs in dct_mse_by_skill.values() for v in vs]
    print(_row("ALL", all_dct))

    print("\n=== Trajectory MSE per skill_type (raw joint units, IDCT 후 비교) ===")
    for k in sorted(traj_mse_by_skill):
        print(_row(k, traj_mse_by_skill[k]))
    all_traj = [v for vs in traj_mse_by_skill.values() for v in vs]
    print(_row("ALL", all_traj))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
