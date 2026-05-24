"""Phase2 — Phase1 도달 subgoal replay (방법론: 경유점 고정, 경로만 다양화).

method3 방법론에서 Phase1 은 buffer-aware subgoal 섭동으로 *상태(state) 다양성*
을 키우고, Phase2 는 Phase1 이 실제로 도달했던 경유점(subgoal)을 **그대로 통과**
하면서 *경로(trajectory)* 만 curobo ``plan_batch`` 로 다양화한다. 즉 Phase2 의
subgoal 위치는 각 episode 가 Phase1 에서 도달했던 subgoal 과 동일해야 한다.

``Phase1SubgoalSelector`` 와 동일한 ``select_subgoal`` 인터페이스를 노출하므로
``skills.set_subgoal_selector`` 훅에 그대로 꽂을 수 있다. 단 후보를 섭동·scoring
하는 대신, Phase1 세션의 ``subgoal_buffer.npz`` 에 episode 별로 기록된 도달
subgoal 을 호출 순서대로(= episode 내 frame 순서) 반환한다.

기록이 없는 episode/skill 이거나 호출 횟수가 기록 수를 초과하면 ``nominal_goal``
로 폴백한다 — 이 경우 Phase2 는 raw 검출 nominal 을 쓰게 되어 방법론에서 벗어나
므로, 폴백이 발생하면 호출부 로그에 드러난다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from method3.phase1_state_seeding.subgoal_buffer import SubgoalBuffer
from method3.phase1_state_seeding.subgoal_selector import SubgoalSelection


class Phase2SubgoalReplay:
    """Phase1 기록 subgoal 을 episode·skill 순서대로 replay 하는 selector.

    Usage::

        replay = Phase2SubgoalReplay("<session>/subgoal_buffer.npz")
        skills.set_subgoal_selector(replay)
        ...
        replay.set_episode("episode_07")   # 매 episode 시작 시 호출
    """

    # move_to_position / move-home 호출부 로그가 Phase1 섭동과 Phase2 replay 를
    # 구분하도록 하는 마커.
    is_phase2_replay = True

    def __init__(self, buffer_path: str | Path) -> None:
        buf = SubgoalBuffer()
        buf.set_file(Path(buffer_path))
        buf.load()
        # {episode_id: [(subgoal_xyz, skill_type), ...]} — episode 내 호출 순서.
        # SubgoalBuffer 는 ordinal key(skill_0, skill_1, ...) 로 저장되므로
        # entry 를 start_t(=staging 순서) 로 정렬하면 곧 호출 순서다.
        # skill_type 도 같이 저장 → select_subgoal 시 type-aware lookup 가능
        # (caller 의 skill_type 명시 받아 같은 type 의 다음 entry 반환 — 한 skill
        # 안 여러 select_subgoal 호출 시 cursor 단순 +1 의 over-advance 회피).
        _staged: dict[str, list[tuple]] = {}
        for skill_id in buf.skill_ids():
            for e in buf.entries(skill_id):
                eid = str(e.episode_id)
                if not eid:
                    # episode_id 미태깅 entry 는 replay 순서를 특정할 수 없어 제외.
                    continue
                _staged.setdefault(eid, []).append(
                    (int(e.start_t),
                     np.asarray(e.subgoal, dtype=float),
                     str(getattr(e, "skill_type", "")))
                )
        # {eid: [subgoal_xyz, ...]} — cursor fallback path 용 (legacy 호환).
        self._by_episode: dict[str, list[np.ndarray]] = {}
        # {eid: [skill_type, ...]} — type-aware lookup 용 (호출 순서 동일 인덱스).
        self._by_episode_types: dict[str, list[str]] = {}
        for eid, lst in _staged.items():
            lst.sort(key=lambda t: t[0])
            self._by_episode[eid] = [xyz for _, xyz, _ in lst]
            self._by_episode_types[eid] = [t for _, _, t in lst]

        self._episode_id: str = ""
        # cursor (legacy fallback) — skill_type 빈 caller 호환.
        self._cursor: int = 0
        # type-별 cursor — {skill_type: int}. type-aware lookup 의 진행 위치.
        # 같은 skill 안 여러 select_subgoal 호출 시 cursor 1 만 advance (over-
        # advance 방지). set_episode 에서 빈 dict 로 reset.
        self._type_cursors: dict[str, int] = {}

    def n_episodes(self) -> int:
        return len(self._by_episode)

    def set_episode(self, episode_id: str) -> None:
        """episode 전환 — replay 커서를 0 으로 리셋.

        Phase2 는 ``phase2/episode_{N}`` 으로 쌓이고 N 은 Phase1 episode 수
        다음부터 이어진다 (예: Phase1 episode_01~40 → Phase2 episode_41~).
        buffer 는 Phase1 의 episode_01~M 만 보유하므로, buffer 에 직접 매치되지
        않는 episode_id 는 번호를 buffer episode 범위로 cycle 매핑한다 — Phase2
        의 k 번째 episode 가 Phase1 의 k 번째 도달 subgoal set 을 replay 하도록
        (episode_41 → episode_01, episode_42 → episode_02, ...).
        """
        eid = str(episode_id)
        if eid not in self._by_episode and self._by_episode:
            eps = sorted(self._by_episode.keys())
            m = re.search(r"(\d+)", eid)
            if m:
                mapped = eps[(int(m.group(1)) - 1) % len(eps)]
                print(f"[Phase2SubgoalReplay] {eid} → {mapped} "
                      f"(buffer 범위로 cycle 매핑; buffer={len(eps)} episodes)")
                eid = mapped
        self._episode_id = eid
        self._cursor = 0
        self._type_cursors = {}

    def select_subgoal(
        self,
        current_ee,
        nominal_goal,
        rng=None,
        reachable_fn=None,
        feasibility_fn=None,
        skill_type: str = "",
    ) -> SubgoalSelection:
        """현재 episode 의 다음 기록 subgoal 을 반환 (Phase1 도달점 replay).

        ``skill_type`` 명시 시 buffer 의 *같은 type* 의 다음 entry 반환
        (type-aware lookup) — 한 skill 안 select_subgoal 여러 번 호출 시
        cursor 단순 +1 의 over-advance 회피. 미명시 또는 type 매칭 entry
        소진 시 cursor fallback (legacy 호환).

        기록 부재 시 ``nominal_goal`` 로 폴백. ``chosen_index=0`` (nominal),
        replay 성공 시 ``chosen_index=1``.
        """
        nominal = np.asarray(nominal_goal, dtype=float).reshape(3)
        recorded = self._by_episode.get(self._episode_id, [])
        recorded_types = self._by_episode_types.get(self._episode_id, [])

        # Path A — type-aware lookup (caller 가 skill_type 명시).
        if skill_type and recorded_types:
            type_matches = [
                xyz for xyz, t in zip(recorded, recorded_types) if t == skill_type
            ]
            if type_matches:
                tc = self._type_cursors.get(skill_type, 0)
                if tc < len(type_matches):
                    self._type_cursors[skill_type] = tc + 1
                    return SubgoalSelection(
                        chosen_goal=type_matches[tc].copy(),
                        chosen_index=1,
                        cold_start=False,
                        reports=[],
                    )
                # type-cursor 소진 → cursor fallback (아래 path)

        # Path B — legacy cursor fallback (skill_type 빈/매칭 실패).
        k = self._cursor
        if k < len(recorded):
            self._cursor = k + 1
            return SubgoalSelection(
                chosen_goal=recorded[k].copy(),
                chosen_index=1,
                cold_start=False,
                reports=[],
            )
        # 기록 소진/부재 → nominal 폴백 (방법론 이탈; 호출부 로그에 드러남).
        return SubgoalSelection(
            chosen_goal=nominal,
            chosen_index=0,
            cold_start=False,
            reports=[],
        )

    # ── Phase1SubgoalSelector 인터페이스 호환용 no-op ──
    # Phase2 는 Phase1 subgoal buffer 를 갱신하지 않는다 (Phase2 의 vector DB
    # 누적은 server 의 useful-OOD accept_to_buffer 가 담당).
    def stage_executed(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def flush_episode(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def discard_episode(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def set_current_context(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def set_trace_file(self, *args, **kwargs) -> None:  # noqa: D102
        pass
