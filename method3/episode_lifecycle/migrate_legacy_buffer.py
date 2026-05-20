"""Legacy subgoal buffer migration — episode_id 소급 태깅 (문서 episode lifecycle).

episode_id plumbing 이전에 수집된 세션의 ``subgoal_buffer.npz`` 는 모든 entry 의
``episode_id`` 가 빈 문자열("")이다. resume reconcile 이 episode 단위로 동작하려면
각 entry 가 어느 episode 에서 왔는지 알아야 한다.

이 모듈은 세션 결과 폴더로부터 그 매핑을 **소급 복원**한다:

  1. ``episode_NN/batch_info.json`` 의 judge → buffer 에 flush 된 TRUE 에피소드
     (순서대로). FALSE/미완 에피소드는 flush 되지 않았으므로 제외한다.
  2. 각 TRUE 에피소드의 ``forward/forward_log.txt`` 에서 ``staged skill=<skill>``
     줄을 세어 그 에피소드가 skill 별로 buffer 에 몇 entry 를 flush 했는지 구한다.
  3. skill 별 staged 합계가 buffer 의 skill 별 entry 수와 정확히 일치하는지
     **검증**한다 (불일치 시 abort — 잘못된 buffer 를 쓰지 않는다).
  4. buffer 의 skill 별 append 순서 = TRUE 에피소드 flush 순서이므로, 각 entry 에
     해당 episode_id 를 stamp 한다.

buffer 는 append-only 이고 npz save/load 가 list 순서를 보존하므로 이 복원은
결정적이다. ``start_t``/``end_t`` 는 로그에서 복원되지 않으므로 -1 로 둔다
(episode reconcile 에는 episode_id 만 필요).

CLI::

    python -m method3.episode_lifecycle.migrate_legacy_buffer <session_dir>
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import sys
from pathlib import Path

from method3.episode_lifecycle.episode_ref import episode_id
from method3.phase1_state_seeding.subgoal_buffer import SubgoalBuffer

_STAGED_RE = re.compile(r"staged skill=(\S+)")


def _true_episodes(session_dir: Path) -> list[int]:
    """batch_info.json 의 judge=="TRUE" 에피소드 번호를 오름차순으로 반환한다."""
    true_eps = []
    for ep_dir in sorted(session_dir.glob("episode_*")):
        try:
            n = int(ep_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        bi = ep_dir / "batch_info.json"
        if bi.exists() and json.loads(bi.read_text()).get("judge") == "TRUE":
            true_eps.append(n)
    return sorted(true_eps)


def _staged_counts(forward_log: Path) -> dict[str, int]:
    """forward_log.txt 의 ``staged skill=<skill>`` 줄을 skill 별로 센다."""
    counts: dict[str, int] = {}
    if not forward_log.exists():
        return counts
    for line in forward_log.read_text(errors="ignore").splitlines():
        m = _STAGED_RE.search(line)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def migrate_session_buffer(session_dir: str | Path, *, backup: bool = True) -> dict:
    """세션의 subgoal_buffer.npz entry 들에 episode_id 를 소급 태깅한다.

    Returns:
        report dict — ``{"buffer_file", "true_episodes", "per_skill",
        "tagged", "skipped"}``.

    Raises:
        FileNotFoundError: subgoal_buffer.npz 가 없을 때.
        ValueError: staged 로그 합계가 buffer entry 수와 불일치할 때 (복원 불가).
    """
    session_dir = Path(session_dir)
    buf_file = session_dir / "subgoal_buffer.npz"
    if not buf_file.exists():
        raise FileNotFoundError(f"subgoal_buffer.npz not found: {buf_file}")

    buf = SubgoalBuffer()
    buf.set_file(buf_file)
    buf.load()

    # 이미 태깅돼 있으면 (전 entry episode_id 비어있지 않음) 재마이그레이션 스킵.
    all_entries = [e for sid in buf.skill_ids() for e in buf.entries(sid)]
    if all_entries and all(str(e.episode_id) for e in all_entries):
        return {
            "buffer_file": str(buf_file), "true_episodes": [],
            "per_skill": {}, "tagged": 0, "skipped": "already tagged",
        }

    true_eps = _true_episodes(session_dir)
    # 에피소드별 staged skill 카운트.
    counts = {
        n: _staged_counts(session_dir / f"episode_{n:02d}" / "forward" / "forward_log.txt")
        for n in true_eps
    }

    # 검증: skill 별 staged 합계 == buffer entry 수.
    per_skill = {}
    for sid in buf.skill_ids():
        staged_total = sum(counts[n].get(sid, 0) for n in true_eps)
        buffer_total = len(buf.entries(sid))
        per_skill[sid] = {"staged_total": staged_total, "buffer_total": buffer_total}
        if staged_total != buffer_total:
            raise ValueError(
                f"skill '{sid}': staged 합계 {staged_total} != buffer entry {buffer_total} "
                f"— 로그/버퍼 불일치로 episode_id 복원 불가 (마이그레이션 중단)"
            )

    if backup:
        shutil.copy2(buf_file, buf_file.with_suffix(".npz.pre_migration_backup"))

    # skill 별 append 순서 = TRUE 에피소드 flush 순서. entry 에 episode_id stamp.
    tagged = 0
    for sid in buf.skill_ids():
        entries = buf.entries(sid)
        rebuilt = []
        cursor = 0
        for n in true_eps:
            for _ in range(counts[n].get(sid, 0)):
                rebuilt.append(
                    dataclasses.replace(entries[cursor], episode_id=episode_id(n)))
                cursor += 1
        assert cursor == len(entries), f"{sid}: {cursor} != {len(entries)}"
        buf._skills[sid] = rebuilt
        tagged += cursor

    buf.save()
    return {
        "buffer_file": str(buf_file),
        "true_episodes": true_eps,
        "per_skill": per_skill,
        "tagged": tagged,
        "skipped": None,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m method3.episode_lifecycle.migrate_legacy_buffer "
              "<session_dir>")
        return 2
    report = migrate_session_buffer(argv[0])
    if report["skipped"]:
        print(f"[migrate] {report['buffer_file']} — {report['skipped']}, no change")
        return 0
    print(f"[migrate] {report['buffer_file']}")
    print(f"[migrate] TRUE episodes ({len(report['true_episodes'])}): "
          f"{report['true_episodes']}")
    for sid, c in report["per_skill"].items():
        print(f"[migrate]   {sid:18s} staged={c['staged_total']} "
              f"buffer={c['buffer_total']} ✓")
    print(f"[migrate] tagged {report['tagged']} entries with episode_id, saved "
          f"(backup: *.npz.pre_migration_backup)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
