"""Buffer 의 episode_id 가 phase1 폴더 이름 (SoT) 과 일치하는지 verify.

2026-05-28 RCA — `rebuild_subgoal_buffer_unified.py` 의 episode_index+1 기반 stamp
가 phase1 폴더 와 +1 shift 발생 → phase2 replay 시 episode mismatch.

이 test 는 그 invariant 를 enforce:
  1. buffer 의 모든 ep_id ⊆ phase1 폴더 이름 set
  2. phase1 폴더 의 첫 episode (episode_01) 의 staged subgoal (forward_log) 이
     buffer["episode_01"] 의 첫 transit subgoal 과 일치 (= 5mm 이내)
  3. phase1 폴더 의 *모든* episode 가 buffer 에 entry 존재
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest


_STAGED_RE = re.compile(
    r"\[Subgoal\]\s+staged\s+skill_(\d+)\s+\S+\s+\"([^\"]*)\""
    r"\s+g=\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]"
)


def _parse_forward_log_staged(log_path: Path) -> list[tuple[int, str, np.ndarray]]:
    """Return list of (skill_idx, nl, xyz) from forward_log [Subgoal] staged lines."""
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = _STAGED_RE.search(line)
        if m:
            out.append((
                int(m.group(1)),
                m.group(2),
                np.array([float(m.group(3)), float(m.group(4)), float(m.group(5))]),
            ))
    return out


def _load_buffer(buffer_path: Path):
    from method3.phase1_state_seeding.subgoal_buffer import SubgoalBuffer
    buf = SubgoalBuffer()
    buf.set_file(buffer_path)
    buf.load()
    return buf


def _session_dir_from_env(monkeypatch=None) -> Path | None:
    """Get the session dir to validate. Returns None if env unset (skip test)."""
    import os
    sd = os.environ.get("METHOD3_VALIDATION_SESSION_DIR")
    return Path(sd) if sd else None


@pytest.mark.skipif(
    _session_dir_from_env() is None,
    reason="set METHOD3_VALIDATION_SESSION_DIR to enable session-level invariant tests",
)
def test_buffer_ep_ids_subset_of_phase1_folders():
    """Buffer 의 ep_id 가 phase1 폴더 이름 set 의 subset 이어야 한다."""
    sd = _session_dir_from_env()
    phase1_dir = sd / "phase1"
    buffer_path = sd / "subgoal_buffer.npz"
    if not phase1_dir.exists() or not buffer_path.exists():
        pytest.skip(f"phase1 or buffer missing: {sd}")

    folder_eps = {p.name for p in phase1_dir.iterdir()
                  if p.is_dir() and p.name.startswith("episode_")}
    buf = _load_buffer(buffer_path)
    buffer_eps = set()
    for sk in buf.skill_ids():
        for e in buf.entries(sk):
            buffer_eps.add(str(e.episode_id))

    unexpected = buffer_eps - folder_eps
    assert not unexpected, (
        f"buffer 에 phase1 폴더 외 episode_id 가 존재 — episode shift 의심: "
        f"{sorted(unexpected)[:5]}"
    )


@pytest.mark.skipif(
    _session_dir_from_env() is None,
    reason="set METHOD3_VALIDATION_SESSION_DIR to enable session-level invariant tests",
)
def test_buffer_first_skill_matches_phase1_folder_01_staged():
    """buffer['episode_01'] 의 첫 transit subgoal 이 phase1 폴더 episode_01 의
    forward_log staged 와 일치 (= 5mm 이내).

    이게 깨지면 buffer 의 ep_id mapping 이 폴더 와 shift 됐다는 신호.
    """
    sd = _session_dir_from_env()
    phase1_dir = sd / "phase1"
    buffer_path = sd / "subgoal_buffer.npz"
    if not phase1_dir.exists() or not buffer_path.exists():
        pytest.skip(f"phase1 or buffer missing: {sd}")

    fwd_log = phase1_dir / "episode_01" / "forward" / "forward_log.txt"
    staged = _parse_forward_log_staged(fwd_log)
    if not staged:
        pytest.skip(f"no staged lines in {fwd_log}")
    first_staged = staged[0]  # (skill_idx, nl, xyz)
    expected_xyz = first_staged[2]

    buf = _load_buffer(buffer_path)
    # buffer 의 ep_01 첫 entry 찾기 — skill_id 의 가장 작은 ordinal
    candidates = []
    for sk in buf.skill_ids():
        for e in buf.entries(sk):
            if str(e.episode_id) == "episode_01":
                try:
                    kidx = int(str(sk).split("_", 1)[1])
                except (ValueError, IndexError):
                    kidx = 0
                candidates.append((kidx, np.asarray(e.subgoal, dtype=float)))
    assert candidates, "buffer['episode_01'] entry 없음 — phase1 폴더 vs buffer mapping 깨짐"
    candidates.sort(key=lambda t: t[0])
    actual_xyz = candidates[0][1]

    diff = float(np.linalg.norm(actual_xyz - expected_xyz))
    assert diff < 0.005, (
        f"buffer['episode_01'] 첫 entry {actual_xyz} ≠ phase1 폴더 episode_01 의 첫 "
        f"forward_log staged {expected_xyz} (diff={diff*1000:.1f}mm). "
        f"episode shift 의심 — buffer rebuild 시 폴더 이름 (SoT) 기반 stamp 인지 확인."
    )


@pytest.mark.skipif(
    _session_dir_from_env() is None,
    reason="set METHOD3_VALIDATION_SESSION_DIR to enable session-level invariant tests",
)
def test_buffer_covers_all_phase1_folders():
    """phase1 의 모든 episode 폴더 가 buffer 에 entry 존재해야 한다."""
    sd = _session_dir_from_env()
    phase1_dir = sd / "phase1"
    buffer_path = sd / "subgoal_buffer.npz"
    if not phase1_dir.exists() or not buffer_path.exists():
        pytest.skip(f"phase1 or buffer missing: {sd}")

    folder_eps = sorted([p.name for p in phase1_dir.iterdir()
                          if p.is_dir() and p.name.startswith("episode_")])
    buf = _load_buffer(buffer_path)
    buffer_eps = set()
    for sk in buf.skill_ids():
        for e in buf.entries(sk):
            buffer_eps.add(str(e.episode_id))

    missing = [ep for ep in folder_eps if ep not in buffer_eps]
    assert not missing, (
        f"phase1 폴더 의 {len(missing)} episode 가 buffer 에 entry 없음 "
        f"(first 5: {missing[:5]}). buffer rebuild 누락 또는 ep_id mismatch."
    )
