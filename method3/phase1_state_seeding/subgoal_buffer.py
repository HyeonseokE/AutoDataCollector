"""Skill-wise subgoal buffer ``B_{g,t}^{(m)}`` — 문서 final_method3_spec §5.

Phase1 에서 각 skill ``m`` 이 이미 seed 한 subgoal-side terminal state region 을
추적하는 lightweight coverage memory. Phase2 의 full vector DB ``B_t^{(m)}`` 와는
목적이 다르다 (§5: ``B_{g,t}^{(m)} ≠ B_t^{(m)}``).

§5.2 — 각 entry 는 full trajectory 가 아니라 subgoal 근처 terminal region 정보를
저장한다 (``SubgoalBufferEntry``): subgoal ``g_i``, terminal region descriptor
``h_i``, 마지막 구간 descriptor set ``{h_i^τ}``, raw dataset pointer
(episode_id/start_t/end_t), success_flag, planner_type, phase.

두 거리 측정 (§5.4):
  ``knn_mean_distance(ĥ, B)`` — query 와 buffer 간 kNN 평균 거리 ``d_k^g``
  ``mean_nn_distance(B)``     — buffer 내부 평균 nearest-neighbor 거리 ``d̄_NN^g``

geometric descriptor 는 저차원(``DESCRIPTOR_DIM``)이고 skill 당 규모가 작으므로
brute-force numpy L2 로 충분하다 (FAISS 불필요). 영속화는 모든 skill·모든
entry 필드를 담는 단일 ``.npz`` 파일이며, 보통 실행 세션 디렉터리 안에 둔다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from method3.phase1_state_seeding.terminal_descriptor import DESCRIPTOR_DIM


def knn_mean_distance(query: np.ndarray, keys: np.ndarray, k: int) -> float:
    """``d_k^g(ĥ, B)`` — query 와 buffer 간 kNN 평균 L2 거리 (문서 §5.4).

    Args:
        query: descriptor (D,).
        keys: buffer terminal-region descriptor 들 (N, D), N >= 1.
        k: nearest-neighbor 개수. N 보다 크면 N 으로 clamp.

    Returns:
        가장 가까운 min(k, N) 개 이웃까지의 평균 거리.
    """
    keys = np.asarray(keys, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    if keys.ndim != 2 or keys.shape[0] == 0:
        raise ValueError("keys must be a non-empty (N, D) array")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    dists = np.linalg.norm(keys - query[None, :], axis=1)
    kk = min(k, dists.shape[0])
    nearest = np.partition(dists, kk - 1)[:kk]
    return float(nearest.mean())


# 청크 한 개의 (chunk, N, D) float64 임시 배열 메모리 상한 (bytes).
_MEAN_NN_MEM_BUDGET = 64 * 1024 * 1024


def mean_nn_distance(keys: np.ndarray) -> float:
    """``d̄_NN^g(B)`` — buffer 내부 평균 nearest-neighbor 거리 (문서 §5.4).

    각 ``h_i`` 에 대해 ``min_{l != i} d(h_i, h_l)`` 를 구해 평균낸다. 현재
    buffer 의 terminal region 들이 보통 어느 정도 떨어져 있는지를 나타내며,
    novelty 의 scale 정규화 분모로 쓰인다.

    전체 ``(N, N, D)`` 차이 배열을 한 번에 만들면 buffer 가 커질 때 OOM 이
    나므로, row 를 청크로 나눠 임시 메모리를 ``_MEAN_NN_MEM_BUDGET`` 이하로
    유지한다 (결과는 naive 계산과 부동소수 오차 한도 내 동일).
    """
    keys = np.asarray(keys, dtype=np.float64)
    if keys.ndim != 2 or keys.shape[0] < 2:
        raise ValueError(f"need a (N>=2, D) array for mean-NN distance, got {keys.shape}")

    n, d_dim = keys.shape
    chunk = max(1, min(n, _MEAN_NN_MEM_BUDGET // (n * d_dim * 8)))
    nn = np.empty(n, dtype=np.float64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        block = keys[a:b]                                       # (R, D)
        dist = np.linalg.norm(block[:, None, :] - keys[None, :, :], axis=2)  # (R, N)
        rows = np.arange(b - a)
        dist[rows, a + rows] = np.inf                           # 자기 자신 제외
        nn[a:b] = dist.min(axis=1)
    return float(nn.mean())


def _safe_skill_name(skill_id) -> str:
    """skill_id 를 안전한 문자열로 변환 (``.npz`` 아카이브 멤버 이름용)."""
    return re.sub(r"[^0-9A-Za-z._-]", "_", str(skill_id))


def _as_npz_path(path: str | Path) -> Path:
    """버퍼 파일 경로가 ``.npz`` 확장자를 갖도록 정규화한다.

    ``np.savez`` 는 확장자가 없으면 ``.npz`` 를 덧붙이므로, save 가 쓴 경로와
    load 가 확인하는 경로가 어긋나지 않도록 미리 맞춘다.
    """
    p = Path(path)
    if p.suffix != ".npz":
        p = p.parent / (p.name + ".npz")
    return p


@dataclass(frozen=True)
class SubgoalBufferEntry:
    """skill-wise subgoal buffer 의 entry ``c_i^{(m)}`` — 문서 §5.2.

    핵심 항목은 subgoal ``g_i``, terminal region descriptor ``h_i``,
    마지막 구간 descriptor set ``{h_i^τ}``, raw dataset pointer
    (episode_id/start_t/end_t) 이다.

    ``skill_id`` 는 episode 내 ordinal key (``skill_0``, ``skill_1``, ...) —
    변동 정보(natural_language)가 아니라 안정적인 호출 순번을 키로 쓴다.
    natural_language / skill_type 은 metadata 로만 보존한다.
    """

    skill_id: str
    subgoal: np.ndarray                  # g_i — 선택된 subgoal (3,) xyz
    terminal_region_key: np.ndarray      # h_i — T_end descriptor 평균 (D,)
    end_state_keys: np.ndarray           # {h_i^τ} — (T_end, D)
    episode_id: str = ""                 # raw dataset pointer — episode 식별자
    start_t: int = -1                    # raw dataset pointer — 구간 시작 time idx
    end_t: int = -1                      # raw dataset pointer — 구간 끝 time idx
    success_flag: bool = True            # §5.5 — TRUE episode 만 적재
    planner_type: str = "InterpPlan"     # §5.2 — canonical preview planner
    phase: str = "phase1"
    natural_language: str = ""           # metadata — skill.natural_language
    skill_type: str = ""                 # metadata — skill.type
    # round_robin/seed_major schedule 의 0-based seed index. -1 = unknown
    # (legacy buffer 또는 호출부가 모를 때). Phase2SubgoalReplay 가 같은 seed
    # 의 Phase1 episode 만 replay 하도록 anchor — 이 필드가 없으면 cross-seed
    # mismatch 가 발생한다 (final_method3_spec/results/RCA 참고).
    seed_index: int = -1


@dataclass
class SubgoalBuffer:
    """Append-only skill-wise subgoal buffer ``B_{g,t} = {B_{g,t}^{(m)}}``.

    Phase1 은 transit move 가 TRUE episode 로 확정될 때마다 ``SubgoalBufferEntry``
    를 append 하고, scoring 시 ``query_skill`` 로 ``B_{g,t}^{(m)}`` 의 terminal
    region descriptor 행렬을 가져온다.
    """

    buffer_file: str | Path | None = None
    # skill_id → entry 리스트.
    _skills: dict[str, list[SubgoalBufferEntry]] = field(default_factory=dict)
    _file: Path | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._file = _as_npz_path(self.buffer_file) if self.buffer_file else None

    def set_file(self, buffer_file: str | Path | None) -> None:
        """영속화 파일 경로를 (재)바인딩한다.

        세션 디렉터리는 skills 생성 시점엔 아직 없으므로, 파이프라인이
        세션을 만든 뒤 ``<session_dir>/<buffer_file>`` 같은 최종 경로를
        여기서 주입한다. None 이면 메모리 전용으로 되돌린다.
        """
        self._file = _as_npz_path(buffer_file) if buffer_file else None

    def append(self, entry: SubgoalBufferEntry) -> None:
        """skill 의 buffer 에 entry 하나를 추가한다 (문서 §5.5)."""
        h = np.asarray(entry.terminal_region_key, dtype=np.float64).reshape(-1)
        if h.shape[0] != DESCRIPTOR_DIM:
            raise ValueError(
                f"terminal_region_key dim {h.shape[0]} != DESCRIPTOR_DIM={DESCRIPTOR_DIM}"
            )
        ek = np.asarray(entry.end_state_keys, dtype=np.float64)
        if ek.ndim != 2 or ek.shape[1] != DESCRIPTOR_DIM:
            raise ValueError(
                f"end_state_keys must be (T_end, {DESCRIPTOR_DIM}), got {ek.shape}"
            )
        self._skills.setdefault(str(entry.skill_id), []).append(entry)

    def query_skill(self, skill_id) -> np.ndarray:
        """``B_{g,t}^{(m)}`` 의 terminal region descriptor 행렬 (N, D).

        novelty scoring 의 fast path. 비어 있으면 (0, DESCRIPTOR_DIM) 빈 배열.
        """
        entries = self._skills.get(str(skill_id), [])
        if not entries:
            return np.empty((0, DESCRIPTOR_DIM), dtype=np.float64)
        return np.stack([e.terminal_region_key for e in entries])

    def entries(self, skill_id) -> list[SubgoalBufferEntry]:
        """해당 skill 의 ``SubgoalBufferEntry`` 목록 (raw pointer 등 메타 포함)."""
        return list(self._skills.get(str(skill_id), []))

    def seed_subgoals(self, skill_id) -> np.ndarray:
        """``G_seed^{(m)}`` — skill m 의 seed subgoal anchor 집합 (N, 3).

        Phase1 이 확보한 seed subgoal 들. Phase2 는 새 subgoal 을 탐색하지 않고
        이 집합을 anchor 로 삼아 각 seed 주변에서만 action variation 후보를
        생성한다 (phase1_seed_anchor_logic). 비어 있으면 (0, 3) 빈 배열.
        """
        entries = self._skills.get(str(skill_id), [])
        if not entries:
            return np.empty((0, 3), dtype=np.float64)
        return np.stack([
            np.asarray(e.subgoal, dtype=np.float64).reshape(3) for e in entries])

    def size(self, skill_id) -> int:
        """해당 skill 의 buffer entry 수."""
        return len(self._skills.get(str(skill_id), []))

    def skill_ids(self) -> list[str]:
        """buffer 에 entry 가 하나라도 있는 skill_id 목록."""
        return [s for s, e in self._skills.items() if e]

    def total_size(self) -> int:
        """모든 skill 을 합친 전체 entry 수."""
        return sum(len(e) for e in self._skills.values())

    def remove_episode(self, episode_id) -> int:
        """``episode_id`` 의 모든 entry 를 제거하고 영속화한다. 제거 수 반환.

        에피소드를 삭제하고 재취득(resume)할 때 stale entry 가 buffer 에 남지
        않도록 한다 (episode lifecycle 정합성). buffer 가 파일에 바인딩돼 있으면
        제거 후 즉시 ``save()`` 로 반영한다.
        """
        eid = str(episode_id)
        removed = 0
        for skill_id, entries in self._skills.items():
            kept = [e for e in entries if str(e.episode_id) != eid]
            removed += len(entries) - len(kept)
            self._skills[skill_id] = kept
        if removed:
            self.save()
        return removed

    def retain_episodes(self, episode_ids) -> int:
        """주어진 episode_id 집합 밖의 entry 를 모두 제거하고 영속화한다. 제거 수 반환.

        resume 시 생존(judge=TRUE·폴더 존재) 에피소드 집합으로 buffer 를 맞춘다 —
        삭제된 에피소드의 stale entry 가 함께 제거된다. ``episode_id`` 가 비어 있는
        (untagged) entry 는 episode 단위로 판별할 수 없으므로 보존한다 — episode_id
        plumbing 이전에 만들어진 legacy buffer 를 resume reconcile 이 통째로
        날리지 않게 하는 안전장치.
        """
        keep = {str(e) for e in episode_ids}
        removed = 0
        for skill_id, entries in self._skills.items():
            kept = [
                e for e in entries
                if not str(e.episode_id) or str(e.episode_id) in keep
            ]
            removed += len(entries) - len(kept)
            self._skills[skill_id] = kept
        if removed:
            self.save()
        return removed

    def rewrite_episodes(self, mapping) -> dict:
        """episode_id 를 ``mapping`` ({old → new}) 에 따라 일괄 치환하고 영속화한다.

        episodes_per_seed 확장 (e.g. 30/3 → 50/5) 으로 폴더가 rename 될 때
        buffer 의 episode_id 도 같은 매핑으로 따라가야 폴더·judge_results 와의
        1:1 정합이 유지된다. 폴더 rename 만 하고 이 메서드를 호출하지 않으면
        다음 resume 의 ``retain_episodes`` reconcile 이 stale id 를 전부 drop
        하는 사고가 난다.

        Idempotent: mapping 에 없는 episode_id 는 통과 (이미 NEW layout 이거나
        다른 source 의 entry). Returns 진단 dict ``{rewritten, kept, unmapped}``.
        """
        import dataclasses as _dc
        m = {str(k): str(v) for k, v in dict(mapping).items()}
        counts = {"rewritten": 0, "kept": 0, "unmapped": 0}
        any_change = False
        for skill_id, entries in self._skills.items():
            for i, e in enumerate(entries):
                tgt = m.get(str(e.episode_id))
                if tgt is None:
                    counts["unmapped"] += 1
                    continue
                if tgt != str(e.episode_id):
                    entries[i] = _dc.replace(e, episode_id=tgt)
                    counts["rewritten"] += 1
                    any_change = True
                else:
                    counts["kept"] += 1
        if any_change:
            self.save()
        return counts

    def file_path(self) -> Path | None:
        """현재 바인딩된 영속화 파일 경로 (없으면 None)."""
        return self._file

    def save(self) -> None:
        """전체 buffer 를 단일 ``.npz`` 파일로 영속화. 파일 미바인딩 시 no-op.

        skill ``s`` 마다 §5.2 필드를 병렬 배열로 저장한다 (멤버 이름
        ``{s}::필드``). end_state_keys 는 entry 별 길이가 다를 수 있으므로
        평탄화한 ``(ΣT_end, D)`` 배열과 길이 배열 ``endlen`` 으로 저장한다.
        """
        if self._file is None:
            return
        arrays: dict[str, np.ndarray] = {}
        for skill_id, entries in self._skills.items():
            if not entries:
                continue
            s = _safe_skill_name(skill_id)
            arrays[f"{s}::subgoal"] = np.stack(
                [np.asarray(e.subgoal, dtype=np.float64).reshape(3) for e in entries])
            arrays[f"{s}::key"] = np.stack(
                [np.asarray(e.terminal_region_key, dtype=np.float64) for e in entries])
            ek_list = [np.asarray(e.end_state_keys, dtype=np.float64) for e in entries]
            arrays[f"{s}::endkeys"] = np.concatenate(ek_list, axis=0)
            arrays[f"{s}::endlen"] = np.array([ek.shape[0] for ek in ek_list], dtype=np.int64)
            arrays[f"{s}::episode"] = np.array([e.episode_id for e in entries])
            arrays[f"{s}::span"] = np.array(
                [(e.start_t, e.end_t) for e in entries], dtype=np.int64)
            arrays[f"{s}::nl"] = np.array([e.natural_language for e in entries])
            arrays[f"{s}::skilltype"] = np.array([e.skill_type for e in entries])
            arrays[f"{s}::seedidx"] = np.array(
                [e.seed_index for e in entries], dtype=np.int64)
        if not arrays:
            return
        self._file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._file, **arrays)

    def load(self) -> None:
        """``.npz`` 파일에서 buffer 를 복원한다 (preload). 파일 없으면 no-op.

        ``success_flag``/``planner_type``/``phase`` 는 Phase1 buffer 의 불변값
        (TRUE·InterpPlan·phase1)이므로 저장하지 않고 복원 시 상수로 채운다.
        """
        if self._file is None or not self._file.exists():
            return
        with np.load(self._file, allow_pickle=False) as data:
            skills = sorted({m.split("::", 1)[0] for m in data.files if "::" in m})
            for s in skills:
                if f"{s}::key" not in data.files:
                    continue
                keys = data[f"{s}::key"]
                subgoals = data[f"{s}::subgoal"]
                endkeys = data[f"{s}::endkeys"]
                endlen = data[f"{s}::endlen"]
                episodes = data[f"{s}::episode"]
                span = data[f"{s}::span"]
                # nl/skilltype/seedidx 는 구버전 npz 에 없을 수 있음 → fallback.
                nls = data[f"{s}::nl"] if f"{s}::nl" in data.files else None
                stypes = data[f"{s}::skilltype"] if f"{s}::skilltype" in data.files else None
                seedidxs = data[f"{s}::seedidx"] if f"{s}::seedidx" in data.files else None
                offsets = np.concatenate([[0], np.cumsum(endlen)])
                entries: list[SubgoalBufferEntry] = []
                for i in range(keys.shape[0]):
                    entries.append(SubgoalBufferEntry(
                        skill_id=s,
                        subgoal=np.asarray(subgoals[i], dtype=np.float64),
                        terminal_region_key=np.asarray(keys[i], dtype=np.float64),
                        end_state_keys=np.asarray(
                            endkeys[offsets[i]:offsets[i + 1]], dtype=np.float64),
                        episode_id=str(episodes[i]),
                        start_t=int(span[i, 0]),
                        end_t=int(span[i, 1]),
                        natural_language=(str(nls[i]) if nls is not None else ""),
                        skill_type=(str(stypes[i]) if stypes is not None else ""),
                        seed_index=(int(seedidxs[i]) if seedidxs is not None else -1),
                    ))
                self._skills[s] = entries
