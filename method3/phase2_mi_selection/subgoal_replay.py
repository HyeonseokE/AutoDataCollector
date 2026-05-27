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

import json
import re
import time
from pathlib import Path

import numpy as np

from method3.phase1_state_seeding.subgoal_buffer import SubgoalBuffer, skill_ordinal
from method3.phase1_state_seeding.subgoal_selector import SubgoalSelection

# module-level optional import — RecordingContext 가 있는 환경에서만 paradigm
# 일관화의 _skill_call_index 직접 lookup. select_subgoal 의 hot path 라 매 호출
# import 회피.
try:
    from record_dataset.context import RecordingContext as _RecordingContext
except Exception:
    _RecordingContext = None


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

    def __init__(
        self,
        buffer_path: str | Path,
        num_seeds: int | None = None,
        schedule_mode: str = "round_robin",
        episodes_per_seed: int | None = None,
    ) -> None:
        """Phase1 도달 subgoal buffer 를 seed-aware 로 replay 한다.

        Args:
            buffer_path: Phase1 ``subgoal_buffer.npz`` 경로.
            num_seeds: 전체 seed 수. seed-aware lookup 의 backfill 용 — buffer
                entry 에 ``seed_index`` 가 stamp 되어 있지 않은 *legacy* buffer
                에 대해서만 이 값으로 episode_id → seed_index 를 복원한다
                (round_robin: ``(ep_num-1) % num_seeds``; seed_major:
                ``(ep_num-1) // episodes_per_seed``). stamp 된 buffer 는 우선.
            schedule_mode: backfill 가정의 schedule. ``round_robin`` 또는
                ``seed_major``. 기본값 round_robin (현재 실험 default).
            episodes_per_seed: seed_major backfill 에 필요. round_robin 에선 무시.
        """
        buf = SubgoalBuffer()
        buf.set_file(Path(buffer_path))
        buf.load()

        self._num_seeds = int(num_seeds) if num_seeds else None
        self._schedule_mode = str(schedule_mode)
        self._episodes_per_seed = (
            int(episodes_per_seed) if episodes_per_seed else None
        )

        def _ep_to_seed(eid: str) -> int:
            """episode_id 에서 seed_index 를 round_robin/seed_major 가정으로 추정."""
            if self._num_seeds is None or self._num_seeds <= 0:
                return -1
            m = re.search(r"(\d+)", str(eid))
            if not m:
                return -1
            n = int(m.group(1))
            if self._schedule_mode == "round_robin":
                return (n - 1) % self._num_seeds
            if self._schedule_mode == "seed_major":
                if not self._episodes_per_seed or self._episodes_per_seed <= 0:
                    return -1
                return (n - 1) // self._episodes_per_seed
            return -1

        # {episode_id: [(subgoal_xyz, skill_type), ...]} — episode 내 호출 순서.
        # SubgoalBuffer 는 ordinal key(skill_0, skill_1, ...) 로 저장되므로
        # entry 를 start_t(=staging 순서) 로 정렬하면 곧 호출 순서다.
        # skill_type 도 같이 저장 → select_subgoal 시 type-aware lookup 가능
        # (caller 의 skill_type 명시 받아 같은 type 의 다음 entry 반환 — 한 skill
        # 안 여러 select_subgoal 호출 시 cursor 단순 +1 의 over-advance 회피).
        # _staged entry 의 tuple = (start_t, skill_ordinal, subgoal, skill_type).
        # skill_ordinal = skill_id 의 숫자 부분 — Phase1SubgoalSelector 가 호출
        # 순서대로 ``skill_0, skill_1, ...`` 로 stamp 했으므로 이 값이 *episode
        # 안 호출 순서*. start_t 가 phase1 buffer 에서 -1 (미태깅) 인 경우 string
        # sort 가 호출 순서와 어긋난다 (skill_10 < skill_2 의 함정). 그래서 sort
        # key 에 skill_ordinal 을 함께 두어 string-sort 의존성 제거.
        _staged: dict[str, list[tuple]] = {}
        # {seed_index: [episode_id, ...]} — seed-aware lookup. entry stamp 우선,
        # 없으면 backfill (num_seeds + schedule_mode 가 둘 다 정해진 경우만).
        # episode_id 기준 de-dup → 다중 entry 의 같은 ep 는 1번만 등록.
        _eps_by_seed_set: dict[int, set[str]] = {}
        # {episode_id: seed_index} — 진단/로그용.
        self._seed_by_episode: dict[str, int] = {}

        # iter 순서도 numeric sort — load() 가 이미 numeric sort 로 빌드해도
        # 안전망 (legacy buffer 또는 다른 caller 가 load 수정 전 코드일 경우).
        # _staged sort 와 이중 안전.
        for skill_id in sorted(buf.skill_ids(), key=skill_ordinal):
            sk_ord = skill_ordinal(skill_id)
            for e in buf.entries(skill_id):
                eid = str(e.episode_id)
                if not eid:
                    # episode_id 미태깅 entry 는 replay 순서를 특정할 수 없어 제외.
                    continue
                _staged.setdefault(eid, []).append(
                    (int(e.start_t),
                     sk_ord,
                     np.asarray(e.subgoal, dtype=float),
                     str(getattr(e, "skill_type", "")))
                )
                # seed-aware index: entry stamp 우선, 누락 시 backfill.
                si = int(getattr(e, "seed_index", -1))
                if si < 0:
                    si = _ep_to_seed(eid)
                if si >= 0:
                    _eps_by_seed_set.setdefault(si, set()).add(eid)
                    self._seed_by_episode.setdefault(eid, si)

        # {eid: [subgoal_xyz, ...]} — cursor fallback path 용 (legacy 호환).
        self._by_episode: dict[str, list[np.ndarray]] = {}
        # {eid: [skill_type, ...]} — type-aware lookup 용 (호출 순서 동일 인덱스).
        self._by_episode_types: dict[str, list[str]] = {}
        for eid, lst in _staged.items():
            # (start_t, skill_ord) 로 정렬 — start_t 우선이지만 phase1 buffer 처럼
            # 모든 entry 가 start_t=-1 인 경우 skill_ord 가 호출 순서를 보장.
            lst.sort(key=lambda t: (t[0], t[1]))
            self._by_episode[eid] = [xyz for _, _, xyz, _ in lst]
            self._by_episode_types[eid] = [t for _, _, _, t in lst]

        # seed 별 sorted episode pool (deterministic rotation).
        self._eps_by_seed: dict[int, list[str]] = {
            si: sorted(eids) for si, eids in _eps_by_seed_set.items()
        }
        # seed 별 rotation cursor — 같은 seed pool 의 ep 들을 균등 분산 사용.
        # set_episode 호출 순서대로 +1, pool size 로 wrap-around.
        self._seed_cursor: dict[int, int] = {}

        self._episode_id: str = ""
        # cursor (legacy fallback) — skill_type 빈 caller 호환.
        self._cursor: int = 0
        # type-별 cursor — {skill_type: int}. type-aware lookup 의 진행 위치.
        # 같은 skill 안 여러 select_subgoal 호출 시 cursor 1 만 advance (over-
        # advance 방지). set_episode 에서 빈 dict 로 reset.
        self._type_cursors: dict[str, int] = {}
        # jsonl trace 경로 (opt-in) — set_trace_file 로 바인딩. None = 비활성.
        # 명시적 init 으로 deterministic attribute 보장 (linter/IDE 친화).
        self._trace_path: str | None = None

        # 진단 로그 — seed-aware coverage 보고.
        if self._eps_by_seed:
            _per_seed = sorted(
                (si, len(eids)) for si, eids in self._eps_by_seed.items()
            )
            print(
                f"[Phase2SubgoalReplay] seed-aware index built: "
                f"{len(self._eps_by_seed)} seeds, "
                f"pool sizes: {_per_seed[:10]}"
                f"{'...' if len(_per_seed) > 10 else ''}"
            )
        else:
            print(
                "[Phase2SubgoalReplay] ⚠ seed-aware index EMPTY — "
                "entry seed_index 도 num_seeds backfill 도 없음. "
                "set_episode 는 legacy cycle fallback 으로 동작 "
                "(seed-mismatch 위험)."
            )

    def n_episodes(self) -> int:
        return len(self._by_episode)

    def set_episode(self, episode_id: str, seed_index: int = -1) -> None:
        """episode 전환 — replay 커서를 0 으로 리셋.

        매핑 우선순위 (방법론 정합 + seed-mismatch 회피):

        1. **Exact match** — ``episode_id`` 가 buffer 에 있으면 그대로
           (Phase1 자기 episode replay 시).
        2. **Seed-aware** — ``seed_index ≥ 0`` 이고 그 seed 의 buffer pool 이
           비어있지 않으면 그중 하나를 rotation 으로 선택. **현재 reset 의 seed
           와 동일한 seed 의 Phase1 episode subgoal 만 replay** — approach 와
           pick descent 사이 seed-mismatch 가 원천 차단된다.
        3. **Legacy cycle fallback (위험)** — 1·2 둘 다 실패 시 옛 cycle 매핑
           (``eps[(N-1) % len(eps)]``). seed 무관 매핑이라 mismatch 위험. 경고
           로그 출력. ``__init__`` 의 num_seeds backfill 도 없이 도착한 경우에만
           이 분기에 도달한다.

        Args:
            episode_id: 현재 Phase2 episode 식별자 (e.g. ``"episode_69"``).
            seed_index: 현재 episode reset 의 0-based seed index. -1 면 모름.
        """
        eid = str(episode_id)
        # 진입 마크 — 어느 분기로 가든 호출 사실은 무조건 보이게 (RCA 진단용).
        print(
            f"[Phase2SubgoalReplay] set_episode CALLED — "
            f"episode_id={eid!r} seed_index={seed_index}"
        )

        _branch = None
        _chosen = None
        _detail = None

        # 1) exact match
        if eid in self._by_episode:
            print(
                f"[Phase2SubgoalReplay] {eid} EXACT MATCH (buffer 에 동일 ep) "
                f"— cursor 0 reset"
            )
            self._episode_id = eid
            self._cursor = 0
            self._type_cursors = {}
            _branch, _chosen, _detail = "exact_match", eid, {}
            self._trace_set_episode(eid, seed_index, _branch, _chosen, _detail)
            return

        # 2) seed-aware rotation
        if seed_index >= 0 and seed_index in self._eps_by_seed:
            pool = self._eps_by_seed[seed_index]
            sc = self._seed_cursor.get(seed_index, 0)
            chosen = pool[sc % len(pool)]
            self._seed_cursor[seed_index] = sc + 1
            print(
                f"[Phase2SubgoalReplay] {eid} (seed={seed_index}) → {chosen} "
                f"(seed-aware rotation; pool={len(pool)}, "
                f"cursor={sc % len(pool)})"
            )
            self._episode_id = chosen
            self._cursor = 0
            self._type_cursors = {}
            _branch, _chosen = "seed_aware", chosen
            _detail = {"pool": list(pool), "cursor_before": sc,
                       "cursor_used": sc % len(pool)}
            self._trace_set_episode(eid, seed_index, _branch, _chosen, _detail)
            return

        # 3) LEGACY cycle fallback — seed-mismatch 가능
        if self._by_episode:
            eps = sorted(self._by_episode.keys())
            m = re.search(r"(\d+)", eid)
            if m:
                mapped = eps[(int(m.group(1)) - 1) % len(eps)]
                _why = (
                    "seed_index 미전달" if seed_index < 0
                    else f"seed={seed_index} 가 buffer pool 에 없음"
                )
                print(
                    f"[Phase2SubgoalReplay] ⚠ {eid} → {mapped} "
                    f"(SEED-BLIND cycle fallback; {_why}; "
                    f"buffer={len(eps)} episodes) — seed mismatch 위험"
                )
                self._episode_id = mapped
                self._cursor = 0
                self._type_cursors = {}
                _branch, _chosen = "legacy_cycle", mapped
                _detail = {"why": _why, "buffer_size": len(eps)}
                self._trace_set_episode(eid, seed_index, _branch, _chosen, _detail)
                return

        # 4) buffer 비었음 — 호출부에서 nominal fallback
        print(
            f"[Phase2SubgoalReplay] ⚠ {eid} → BUFFER EMPTY "
            f"(self._by_episode 가 빔) — select_subgoal 이 nominal 폴백"
        )
        self._episode_id = eid
        self._cursor = 0
        self._type_cursors = {}
        _branch, _chosen = "buffer_empty", eid
        self._trace_set_episode(eid, seed_index, _branch, _chosen, {})

    # ── jsonl trace — TeeLogger 와 무관, set_trace_file() 로 경로 바인딩 ──
    def set_trace_file(self, path) -> None:  # type: ignore[override]
        """jsonl trace 파일 경로 바인딩. None 이면 비활성."""
        self._trace_path = str(path) if path else None

    def _trace_set_episode(self, eid, seed_index, branch, chosen, detail) -> None:
        """set_episode 호출 결과를 jsonl 한 줄로 append (forward_log 와 무관)."""
        path = self._trace_path
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": time.time(),
                "called_with": {"episode_id": eid, "seed_index": seed_index},
                "branch": branch,
                "chosen_episode": chosen,
                "detail": detail,
                "seed_cursor_snapshot": dict(self._seed_cursor),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Phase2SubgoalReplay] trace write failed: {e}")

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
        # paradigm 일관화 — cursor 자체 진행 대신 RecordingContext._skill_call_index
        # 로 직접 lookup. select_subgoal 은 plan_batch 와 같은 시점 (move_to_position
        # 안, set_skill_info *전*) 에 호출되므로 _skill_call_index 가 *현재* skill
        # 의 ordinal (곧 stamp 될 값). VDB partition key (build_skill_dct 의 episode-내
        # skill_index) 와 같은 namespace.
        if _RecordingContext is not None:
            k = max(0, int(_RecordingContext._skill_call_index))
        else:
            # RecordingContext 미가용 시 legacy cursor fallback.
            k = self._cursor
            self._cursor = k + 1
        if k < len(recorded):
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

    # NOTE: set_trace_file 는 위(__init__ 직후)에서 *진단 trace 활성화* 로 override.
    # 옛 no-op stub 제거 — Phase1SubgoalSelector 호환은 그대로 유지된다.
