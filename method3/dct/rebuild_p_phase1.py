"""CLI: DCT-tuned VLA checkpoint 로 skill-wise vector DB 재구축.

paradigm step [5]: Phase 3 학습된 DCT-tuned smolvla 의 backbone 으로
``P_phase1`` (skill-wise vector DB) 을 재구축한다. ``build_or_load`` 가
이미 rebuild flag + path swap + use_dct_target 분기를 지원하므로 본
모듈은 thin CLI.

저장 위치:
  --subdir 가 주어지면  ``session/<subdir>/<filename>`` 에 저장.
  default subdir="dct" → ``session/dct/skill_wise_vector_db.npz``.

Usage (DCT paradigm):
    python -m method3.dct.rebuild_p_phase1 \\
        --session ./results/session_20260518_214931 \\
        --subdir  dct \\
        --vla-ckpt .../smolvla_dct_<ts>/checkpoints/.../pretrained_model \\
        --dataset  CoRL2026-CSI/pnp_phase1_30_table2 \\
        --skill-dct-parquet ./results/skill_dct/pnp_phase1_30_table2.parquet

기존 frame-level rebuild (사이드카 없이) 도 backward compat — --skill-dct-parquet
미지정 시 기존 LeRobotPhase1RawAdapter 경로 사용 (use_dct_target=False).
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from method3.reembedding.build_or_load import (
    _DEFAULT_VECTOR_DB_FILENAME,
    build_or_load_phase1_vector_db,
)
from method3.reembedding.seed_builder import ReembeddingConfig


def _archive_existing(target_path: Path) -> Path | None:
    """target_path 가 있으면 sibling ``archive_p_phase1/`` 로 timestamp 이동."""
    if not target_path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = target_path.parent / "archive_p_phase1"
    archive_dir.mkdir(parents=True, exist_ok=True)
    dst = archive_dir / f"{target_path.stem}.pre_dct.{stamp}{target_path.suffix}"
    shutil.move(str(target_path), str(dst))
    return dst


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True,
                   help="results/session_*/ — vector DB 캐시 위치")
    p.add_argument("--subdir", default="dct",
                   help='session 안 sub-namespace (default "dct"). '
                        '빈 문자열이면 session 직속.')
    p.add_argument("--vla-ckpt", required=True,
                   help="DCT-tuned VLA checkpoint 경로")
    p.add_argument("--dataset", required=True,
                   help="Phase1 dataset (LeRobot repo_id 또는 local path)")
    p.add_argument("--skill-dct-parquet", default=None,
                   help="method3.dct.build_skill_dct 산물. 명시 시 DCT paradigm "
                        "(use_dct_target=True) 으로 build. 미명시 시 기존 "
                        "frame-level path.")
    p.add_argument("--L0", type=int, default=50,
                   help="DCT paradigm 의 L0 (= action_horizon). default 50.")
    p.add_argument("--filename", default=_DEFAULT_VECTOR_DB_FILENAME,
                   help="vector DB 파일명 (default skill_wise_vector_db.npz)")
    p.add_argument("--no-archive", action="store_true",
                   help="기존 DB 가 같은 경로에 있어도 archive 안 하고 덮어쓴다")
    args = p.parse_args()

    session_dir = Path(args.session)
    # sub-namespace 조립.
    rel_filename = (
        f"{args.subdir}/{args.filename}" if args.subdir else args.filename
    )
    target_path = session_dir / rel_filename

    if not args.no_archive:
        moved = _archive_existing(target_path)
        if moved:
            print(f"[rebuild_p_phase1] archived previous DB → {moved}")

    # DCT paradigm 활성화 시 ReembeddingConfig 직접 구성 (yaml fallback 무시).
    re_cfg = None
    if args.skill_dct_parquet:
        re_cfg = ReembeddingConfig(
            action_horizon=int(args.L0),
            use_dct_target=True,
            skill_dct_parquet=str(args.skill_dct_parquet),
        )
        print(f"[rebuild_p_phase1] DCT paradigm — L0={args.L0}, "
              f"sidecar={args.skill_dct_parquet}")

    db = build_or_load_phase1_vector_db(
        session_dir,
        phase1_trained_vla_path=args.vla_ckpt,
        phase1_dataset_path=args.dataset,
        vector_db_filename=rel_filename,
        rebuild=True,
        reembedding_config=re_cfg,
    )
    print(f"[rebuild_p_phase1] done — {db.total_size()} entries, "
          f"{len(db.skill_ids())} skills → {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
