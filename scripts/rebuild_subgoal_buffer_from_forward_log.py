"""phase1 forward_log.txt 의 [Subgoal] staged g=[...] 출력에서 buffer 재생성.

forward_log 의 두 단서를 join 한다:
  1. [RecordingContext] Skill: <type> - <label>   ← all-call ordinal 순서 (transit+non-transit)
  2. [Subgoal] staged skill_<X>  <type>  "<nl>"  g=[..]   ← transit-only 의 staged target

각 [Subgoal] staged 를 *그 직전의 [RecordingContext] Skill* 와 매칭해 그 *all-call ordinal*
로 buffer 에 stamp. 결과: buffer 의 entries 의 skill_id = all-call ordinal (= phase2 의
RecordingContext._skill_call_index 와 같은 namespace).

Usage:
    python -m scripts.rebuild_subgoal_buffer_from_forward_log \
        --session results/completed_logs/table3/stateseeding+random_pert \
        --out    results/completed_logs/table3/stateseeding+random_pert/subgoal_buffer.npz
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# [RecordingContext] Skill: move_and_open - Approach blue block and open gripper
_REC_RE = re.compile(
    r"\[RecordingContext\]\s+Skill:\s+(\S+)\s+-\s+(.*?)(?:\s+\(call_index=\d+\))?\s*$"
)

# [Subgoal] staged skill_1  move_and_open  "Approach blue block and open gripp"  g=[0.243, 0.219, 0.119]  (episode pending=2)
_STAGED_RE = re.compile(
    r"\[Subgoal\]\s+staged\s+skill_(\d+)\s+(\S+)\s+\"([^\"]*)\""
    r"\s+g=\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]"
)


def parse_forward_log(log_path: Path) -> list[tuple[int, str, str, np.ndarray]]:
    """Return list of (all_call_ordinal, skill_type, natural_language, xyz).

    Strategy: scan lines in order; each [RecordingContext] Skill increments the
    all-call ordinal counter. When a subsequent [Subgoal] staged appears, attach
    the current ordinal-1 (the staging is for the last-stamped skill).
    """
    out: list[tuple[int, str, str, np.ndarray]] = []
    if not log_path.exists():
        return out

    call_idx = -1  # incremented on each RecordingContext Skill line; the first
                   # becomes 0.
    pending_label: str | None = None
    pending_type: str | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        m_rec = _REC_RE.search(line)
        if m_rec:
            call_idx += 1
            pending_type = m_rec.group(1).strip()
            pending_label = m_rec.group(2).strip()
            continue
        m_st = _STAGED_RE.search(line)
        if m_st:
            sk_type = m_st.group(2).strip()
            nl = m_st.group(3).strip()
            xyz = np.array([float(m_st.group(4)), float(m_st.group(5)), float(m_st.group(6))],
                           dtype=np.float64)
            # The staged line corresponds to the most recently stamped Skill
            # (the trajectory it described has just finished).
            if call_idx >= 0:
                out.append((call_idx, sk_type or pending_type or "", nl, xyz))
    return out


def rebuild(session_dir: Path, out_path: Path) -> int:
    from method3.phase1_state_seeding.subgoal_buffer import (
        SubgoalBuffer, SubgoalBufferEntry,
    )
    from method3.phase1_state_seeding.subgoal_selector import (
        Phase1SubgoalSelector, Phase1SubgoalConfig,
    )

    phase1_dir = session_dir / "phase1"
    if not phase1_dir.exists():
        raise FileNotFoundError(f"phase1 dir not found: {phase1_dir}")

    ep_dirs = sorted([p for p in phase1_dir.iterdir()
                       if p.is_dir() and p.name.startswith("episode_")])
    if not ep_dirs:
        raise FileNotFoundError(f"no episode_* under {phase1_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = SubgoalBuffer()
    buf.set_file(out_path)
    sel = Phase1SubgoalSelector(buf, Phase1SubgoalConfig())

    total_entries = 0
    skipped: list[str] = []
    skill_id_distribution: dict[int, int] = {}

    for ep_dir in ep_dirs:
        ep_id = ep_dir.name
        fwd_log = ep_dir / "forward" / "forward_log.txt"
        stagings = parse_forward_log(fwd_log)
        if not stagings:
            skipped.append(ep_id)
            continue

        prev_ee = None
        for call_idx, sk_type, nl, xyz in stagings:
            try:
                start_ee = (prev_ee if prev_ee is not None
                            else np.array([0.226, 0.003, 0.145], dtype=np.float64))
                end_keys, h_star = sel._terminal_region(start_ee, xyz)
            except Exception:
                end_keys = np.zeros((0, 3), dtype=np.float64)
                h_star = np.zeros(3, dtype=np.float64)

            buf.append(SubgoalBufferEntry(
                skill_id=f"skill_{call_idx}",
                subgoal=xyz,
                terminal_region_key=h_star,
                end_state_keys=end_keys,
                episode_id=ep_id,
                start_t=-1,
                end_t=-1,
                success_flag=True,
                planner_type="InterpPlan",
                phase="phase1",
                natural_language=nl,
                skill_type=sk_type,
            ))
            total_entries += 1
            prev_ee = xyz
            skill_id_distribution[call_idx] = skill_id_distribution.get(call_idx, 0) + 1

    buf.save()
    sks = sorted(buf._skills.keys(),
                 key=lambda x: int(x.split("_", 1)[1]) if "_" in x else 0)
    print(f"[rebuild-from-log] saved {total_entries} entries across {len(sks)} skills")
    for sk in sks:
        ents = buf._skills[sk]
        print(f"  {sk}: n={len(ents)}  e.g. nl='{ents[0].natural_language[:38]}' "
              f"type={ents[0].skill_type}  "
              f"g=[{ents[0].subgoal[0]:.3f},{ents[0].subgoal[1]:.3f},{ents[0].subgoal[2]:.3f}]")
    if skipped:
        print(f"[rebuild-from-log] WARN: {len(skipped)} episodes had no staged lines "
              f"(skipped): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    print(f"[rebuild-from-log] file: {out_path}")
    return total_entries


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    n = rebuild(Path(args.session), Path(args.out))
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
