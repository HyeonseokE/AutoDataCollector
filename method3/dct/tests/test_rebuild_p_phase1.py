"""rebuild_p_phase1 — archive helper + CLI argument parsing 검증."""
from __future__ import annotations

from pathlib import Path

from method3.dct.rebuild_p_phase1 import _archive_existing


def test_archive_returns_none_when_no_existing(tmp_path):
    out = _archive_existing(tmp_path, "skill_wise_vector_db.npz")
    assert out is None


def test_archive_moves_existing_file(tmp_path):
    src = tmp_path / "skill_wise_vector_db.npz"
    src.write_bytes(b"fake-db-contents")
    moved = _archive_existing(tmp_path, "skill_wise_vector_db.npz")
    assert moved is not None
    assert moved.exists()
    assert moved.read_bytes() == b"fake-db-contents"
    assert not src.exists()  # 원본은 archive 로 이동.
    assert moved.parent.name == "archive_p_phase1"
    # 파일명에 pre_dct + timestamp prefix 가 포함돼야 한다.
    assert "pre_dct" in moved.name
