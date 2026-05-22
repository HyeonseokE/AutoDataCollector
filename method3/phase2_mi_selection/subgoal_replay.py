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
        # {episode_id: {skill_id: [subgoal_xyz, ...]}} — episode 내 frame 순서.
        self._by_episode: dict[str, dict[str, list[np.ndarray]]] = {}
        for skill_id in buf.skill_ids():
            for e in buf.entries(skill_id):
                eid = str(e.episode_id)
                if not eid:
                    # episode_id 미태깅 entry 는 replay 순서를 특정할 수 없어 제외.
                    continue
                self._by_episode.setdefault(eid, {}).setdefault(
                    str(skill_id), []
                ).append((int(e.start_t), np.asarray(e.subgoal, dtype=float)))
        # episode 내 frame 시작 시각(start_t)으로 정렬해 호출 순서와 맞춘다.
        for skills in self._by_episode.values():
            for sk, lst in list(skills.items()):
                lst.sort(key=lambda t: t[0])
                skills[sk] = [xyz for _, xyz in lst]

        self._episode_id: str = ""
        self._cursor: dict[str, int] = {}

    def n_episodes(self) -> int:
        return len(self._by_episode)

    def skill_ids(self) -> list[str]:
        out: set[str] = set()
        for skills in self._by_episode.values():
            out.update(skills.keys())
        return sorted(out)

    def set_episode(self, episode_id: str) -> None:
        """episode 전환 — per-skill replay 커서를 0 으로 리셋."""
        self._episode_id = str(episode_id)
        self._cursor = {}

    def select_subgoal(
        self,
        current_ee,
        nominal_goal,
        skill_id,
        rng=None,
        reachable_fn=None,
        feasibility_fn=None,
    ) -> SubgoalSelection:
        """현재 episode·skill 의 다음 기록 subgoal 을 반환 (Phase1 도달점 replay).

        기록이 없거나 커서가 소진되면 ``nominal_goal`` 로 폴백한다. 폴백 시
        ``chosen_index=0`` — move-home 호출부가 이를 nominal 로 인식하도록.
        replay 성공 시 ``chosen_index=1`` 로 *비(非)-nominal* 임을 표시한다.
        """
        nominal = np.asarray(nominal_goal, dtype=float).reshape(3)
        recorded = self._by_episode.get(self._episode_id, {}).get(str(skill_id), [])
        k = self._cursor.get(str(skill_id), 0)
        if k < len(recorded):
            self._cursor[str(skill_id)] = k + 1
            return SubgoalSelection(
                chosen_goal=recorded[k].copy(),
                chosen_index=1,        # !=0 → move-home 호출부가 적용
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
