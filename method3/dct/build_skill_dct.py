"""CLI: LeRobot v3 Phase1 dataset → skill-unit DCT sidecar parquet.

사용자 명시 paradigm step [1] 의 dataset 준비 단계.

Usage:
    python -m method3.dct.build_skill_dct \\
        --dataset CoRL2026-CSI/pnp_phase1_30_table2 \\
        --output  ./results/skill_dct/pnp_phase1_30_table2.parquet \\
        --L0 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

from method3.dct.skill_dataset import build_dct_targets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="LeRobot dataset repo_id 또는 절대경로")
    p.add_argument("--output", required=True,
                   help="출력 parquet 파일 경로")
    p.add_argument("--L0", type=int, default=50,
                   help="DCT 출력 차원 (default 50)")
    p.add_argument("--episode-start", type=int, default=None,
                   help="0-based episode_index 시작 (inclusive)")
    p.add_argument("--episode-end", type=int, default=None,
                   help="0-based episode_index 끝 (exclusive)")
    args = p.parse_args()

    episode_range = None
    if args.episode_start is not None or args.episode_end is not None:
        episode_range = (
            args.episode_start if args.episode_start is not None else 0,
            args.episode_end if args.episode_end is not None else 10 ** 9,
        )

    out = Path(args.output)
    n = build_dct_targets(args.dataset, out, L0=args.L0, episode_range=episode_range)
    print(f"[build_skill_dct] saved {n} skill segments → {out}"
          + (f" (episode_range={episode_range})" if episode_range else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
