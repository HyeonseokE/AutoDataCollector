"""Naive baseline — per-skill-type 평균 DCT 를 prediction 으로 사용.

train sidecar parquet 에서 per-skill-type 평균 DCT 를 미리 계산한 뒤,
eval sidecar parquet 의 sample 마다 *그 skill_type 의 평균* 을 prediction
으로 두고 MSE 측정. "학습 0, 그냥 평균만 출력" 의 lower bound.

학습된 ckpt 의 validation 결과보다 이 baseline 이 잘 하면 학습이 의미 없는
것이고, baseline 보다 ckpt 가 잘 해야 paradigm 가설이 살아남는다.

Usage:
    python -m method3.dct.validate_naive_baseline \\
        --train-parquet results/skill_dct/pnp_phase1_30_table2.parquet \\
        --eval-parquet  results/skill_dct/pnp_phase1_100_table2_ep30_84.parquet
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from method3.dct.skill_dataset import load_dct_targets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--eval-parquet", required=True)
    args = p.parse_args()

    train_segs = load_dct_targets(args.train_parquet)
    eval_segs = load_dct_targets(args.eval_parquet)
    if not train_segs or not eval_segs:
        print(f"ERROR: empty parquet (train={len(train_segs)}, eval={len(eval_segs)})")
        return 1

    # train 의 per-skill-type 평균 DCT.
    L0, dof = train_segs[0].dct_target.shape
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((L0, dof)))
    counts: dict[str, int] = defaultdict(int)
    for s in train_segs:
        sums[s.skill_type] = sums[s.skill_type] + s.dct_target
        counts[s.skill_type] += 1
    means = {k: sums[k] / counts[k] for k in sums}

    print(f"[naive] train skill_types: {sorted(means.keys())}")
    print(f"[naive] eval segments:    {len(eval_segs)}")

    # eval set 의 sample 마다 train-mean prediction MSE.
    mse_by_skill: dict[str, list[float]] = defaultdict(list)
    skipped = 0
    for s in eval_segs:
        if s.skill_type not in means:
            skipped += 1
            continue
        mse = float(np.mean((means[s.skill_type] - s.dct_target) ** 2))
        mse_by_skill[s.skill_type].append(mse)

    def _row(name: str, vs: list[float]) -> str:
        a = np.asarray(vs, dtype=np.float64)
        return (f"  {name:20s} n={len(a):3d}  mean={a.mean():>9.4f}  "
                f"median={np.median(a):>9.4f}  p90={np.quantile(a, 0.9):>9.4f}  "
                f"max={a.max():>9.4f}")

    print("\n=== Naive baseline DCT MSE (train per-skill mean → eval target) ===")
    for k in sorted(mse_by_skill):
        print(_row(k, mse_by_skill[k]))
    all_v = [v for vs in mse_by_skill.values() for v in vs]
    print(_row("ALL", all_v))
    if skipped:
        print(f"\nskipped {skipped} eval segments with skill_type not in train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
