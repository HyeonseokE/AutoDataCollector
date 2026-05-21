"""rebuild_p_phase1 — archive helper sub-namespace 동작 검증."""
from __future__ import annotations

from method3.dct.rebuild_p_phase1 import _archive_existing


def test_archive_returns_none_when_no_existing(tmp_path):
    out = _archive_existing(tmp_path / "dct" / "skill_wise_vector_db.npz")
    assert out is None


def test_archive_moves_existing_file_into_sibling_archive(tmp_path):
    sub = tmp_path / "dct"
    sub.mkdir()
    src = sub / "skill_wise_vector_db.npz"
    src.write_bytes(b"fake-db-contents")

    moved = _archive_existing(src)
    assert moved is not None
    assert moved.exists()
    assert moved.read_bytes() == b"fake-db-contents"
    assert not src.exists()
    # archive 위치 = src 의 sibling (= session/dct/archive_p_phase1/)
    assert moved.parent.name == "archive_p_phase1"
    assert moved.parent.parent == sub
    assert "pre_dct" in moved.name
