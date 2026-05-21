"""CLI: DCT-tuned VLA checkpoint 로 skill-wise vector DB 재구축.

paradigm step [5]: Phase 3 에서 학습된 DCT-tuned smolvla 의 backbone 으로
``P_phase1`` (skill-wise vector DB) 을 재구축한다. ``build_or_load`` 가
이미 rebuild flag + path swap 을 지원하므로 본 모듈은 thin CLI.

Usage:
    python -m method3.dct.rebuild_p_phase1 \\
        --session ./results/session_20260521_083952_50 \\
        --vla-ckpt ./outputs/train/smolvla_dct_YYYYMMDD_HHMMSS \\
        --dataset CoRL2026-CSI/pnp_phase1_30_table2

기존 DB 가 있으면 archive 폴더로 이동 (rebuild 직전 자동 백업).
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


def _archive_existing(session_dir: Path, filename: str) -> Path | None:
    src = session_dir / filename
    if not src.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = session_dir / "archive_p_phase1"
    archive_dir.mkdir(parents=True, exist_ok=True)
    dst = archive_dir / f"{src.stem}.pre_dct.{stamp}{src.suffix}"
    shutil.move(str(src), str(dst))
    return dst


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True,
                   help="results/session_*/ — vector DB 캐시 위치")
    p.add_argument("--vla-ckpt", required=True,
                   help="DCT-tuned VLA checkpoint 경로")
    p.add_argument("--dataset", required=True,
                   help="Phase1 dataset (LeRobot repo_id 또는 local path)")
    p.add_argument("--filename", default=_DEFAULT_VECTOR_DB_FILENAME,
                   help="vector DB 캐시 파일명")
    p.add_argument("--no-archive", action="store_true",
                   help="기존 DB 를 archive 폴더로 옮기지 않고 그대로 덮어쓴다")
    args = p.parse_args()

    session_dir = Path(args.session)
    if not args.no_archive:
        moved = _archive_existing(session_dir, args.filename)
        if moved:
            print(f"[rebuild_p_phase1] archived previous DB → {moved}")

    db = build_or_load_phase1_vector_db(
        session_dir,
        phase1_trained_vla_path=args.vla_ckpt,
        phase1_dataset_path=args.dataset,
        vector_db_filename=args.filename,
        rebuild=True,
    )
    print(f"[rebuild_p_phase1] done — {db.total_size()} entries, "
          f"{len(db.skill_ids())} skills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
