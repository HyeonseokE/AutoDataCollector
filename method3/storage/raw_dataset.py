"""Raw trajectory dataset — 문서 final_method3_spec §3 / §3.1.

Method3 의 **source of truth**. Phase1/Phase2 가 수집한 trajectory window 를
이후 VLA re-embedding(§6), action descriptor 재계산, ablation 분석이 가능하도록
원본 정보와 함께 저장한다. vector DB ``B_t^{(m)}`` 는 이 dataset 을 가리키는
searchable index 이며, 둘은 분리된 두 layer 다 (§13 원칙 1).

  D_raw = {(o_τ, p_τ, I, g, m, A_{τ:τ+H-1}, meta_τ)}_τ

저장 레이아웃 (dataset 디렉터리, 보통 ``D_phase1_raw/`` · ``D_phase2_raw/``):

  index.jsonl       — entry 당 1줄, 메타데이터 + 배열 파일 참조 (append-only,
                       crash-safe — entry 마다 즉시 flush)
  arrays/<seq>.npz  — entry 당 numeric 배열 (subgoal, proprioception,
                       action_chunk, raw_action_sequence)

observation ``o_τ`` 는 이미지라 무겁고 LeRobot recording 에 이미 저장되므로,
§3 details §4 의 pointer 규약을 따라 복원용 포인터 ``observation_ref`` 만 둔다 —
re-embedding 시 그 포인터로 ``o_τ`` 를 가져온다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# entry 당 npz 에 담는 numeric 배열 필드.
_ARRAY_FIELDS = ("subgoal", "proprioception", "action_chunk", "raw_action_sequence")


@dataclass(frozen=True)
class RawDatasetEntry:
    """raw trajectory dataset 의 entry 하나 — 문서 §3.1.

    re-embedding 최소 정보는 ``(observation_ref, instruction, proprioception)``,
    action descriptor 재계산 정보는 ``action_chunk``, skill-wise indexing/분석
    정보는 ``(skill_id, subgoal, *_metadata)`` 이다.
    """

    episode_id: str
    phase: str                          # "phase1" | "phase2"
    skill_id: str
    instruction: str                    # I — language instruction
    subgoal: np.ndarray                 # g — (3,) xyz
    time_index: int                     # τ — window 시작 time index
    observation_ref: dict               # o_τ 복원 포인터 (§4 — 이미지는 pointer)
    proprioception: np.ndarray          # p_τ — proprioceptive state
    action_chunk: np.ndarray            # A_{τ:τ+H-1} — (H, action_dim)
    raw_action_sequence: np.ndarray     # 원본 raw action 시퀀스
    planner_type: str = "InterpPlan"    # planner_or_generator_type
    success_flag: bool = True
    validity_flag: bool = True
    environment_metadata: dict = field(default_factory=dict)
    object_metadata: dict = field(default_factory=dict)


def _scalar_record(entry: RawDatasetEntry, array_file: str) -> dict:
    """entry 의 비배열 필드를 index.jsonl 한 줄(dict)로 직렬화한다."""
    return {
        "episode_id": entry.episode_id,
        "phase": entry.phase,
        "skill_id": entry.skill_id,
        "instruction": entry.instruction,
        "time_index": int(entry.time_index),
        "observation_ref": entry.observation_ref,
        "planner_type": entry.planner_type,
        "success_flag": bool(entry.success_flag),
        "validity_flag": bool(entry.validity_flag),
        "environment_metadata": entry.environment_metadata,
        "object_metadata": entry.object_metadata,
        "_arrays": array_file,
    }


class RawTrajectoryDataset:
    """append-only raw trajectory dataset (문서 §3.1) — ``D_phase{1,2}_raw``.

    하나의 디렉터리에 entry 들을 누적한다. entry 마다 numeric 배열은 별도
    ``.npz`` 로, 메타데이터는 ``index.jsonl`` 한 줄로 즉시 영속화하므로 run 이
    중간에 끊겨도 직전 entry 까지 보존된다 (vector DB ingest 와 동일한 원칙).
    """

    def __init__(self, dataset_dir: str | Path) -> None:
        """
        Args:
            dataset_dir: dataset 디렉터리. 이미 존재하면 기존 entry 를 이어받는다.
        """
        self._dir = Path(dataset_dir)
        self._index: list[dict] = []
        index_file = self._dir / "index.jsonl"
        if index_file.exists():
            with open(index_file, encoding="utf-8") as f:
                self._index = [json.loads(line) for line in f if line.strip()]

    @property
    def dataset_dir(self) -> Path:
        """dataset 디렉터리 경로."""
        return self._dir

    def __len__(self) -> int:
        return len(self._index)

    def append(self, entry: RawDatasetEntry) -> int:
        """entry 하나를 dataset 에 추가하고 그 index 를 반환한다 (문서 §3.1).

        numeric 배열은 ``arrays/<seq>.npz`` 로, 메타데이터는 ``index.jsonl`` 에
        append 한다. 두 쓰기 모두 호출 즉시 디스크에 반영된다.
        """
        arrays_dir = self._dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        # remove_episode 후엔 len(_index) 가 살아남은 npz 와 충돌할 수 있으므로
        # 비어 있는 seq 를 찾는다 (append-after-remove 안전).
        seq = len(self._index)
        while (arrays_dir / f"{seq:06d}.npz").exists():
            seq += 1
        array_file = f"{seq:06d}.npz"
        np.savez(
            arrays_dir / array_file,
            subgoal=np.asarray(entry.subgoal, dtype=np.float64),
            proprioception=np.asarray(entry.proprioception, dtype=np.float64),
            action_chunk=np.asarray(entry.action_chunk, dtype=np.float64),
            raw_action_sequence=np.asarray(entry.raw_action_sequence, dtype=np.float64),
        )
        record = _scalar_record(entry, array_file)
        with open(self._dir / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._index.append(record)
        return seq

    def get(self, idx: int) -> RawDatasetEntry:
        """index ``idx`` 의 entry 를 (배열 포함) 복원한다."""
        rec = self._index[idx]
        with np.load(self._dir / "arrays" / rec["_arrays"]) as d:
            arrays = {k: np.asarray(d[k], dtype=np.float64) for k in _ARRAY_FIELDS}
        return RawDatasetEntry(
            episode_id=rec["episode_id"],
            phase=rec["phase"],
            skill_id=rec["skill_id"],
            instruction=rec["instruction"],
            subgoal=arrays["subgoal"],
            time_index=int(rec["time_index"]),
            observation_ref=rec["observation_ref"],
            proprioception=arrays["proprioception"],
            action_chunk=arrays["action_chunk"],
            raw_action_sequence=arrays["raw_action_sequence"],
            planner_type=rec["planner_type"],
            success_flag=bool(rec["success_flag"]),
            validity_flag=bool(rec["validity_flag"]),
            environment_metadata=rec["environment_metadata"],
            object_metadata=rec["object_metadata"],
        )

    def entries(self) -> list[RawDatasetEntry]:
        """모든 entry 를 복원해 리스트로 반환한다."""
        return [self.get(i) for i in range(len(self._index))]

    def __iter__(self):
        for i in range(len(self._index)):
            yield self.get(i)

    def skill_entries(self, skill_id) -> list[RawDatasetEntry]:
        """해당 skill 의 entry 만 (skill-wise indexing, 문서 §3.1)."""
        sid = str(skill_id)
        return [
            self.get(i) for i, rec in enumerate(self._index)
            if rec["skill_id"] == sid
        ]

    def pointer(self, idx: int) -> dict:
        """vector DB ``ref`` 로 쓸 raw dataset pointer (문서 §3.1 dataset_ref).

        이 pointer 만으로 ``resolve_pointer`` 가 entry 전체를 복원할 수 있다
        (§4 — pointer 로 observation/instruction/proprioception/action_chunk
        복원 가능 조건).
        """
        rec = self._index[idx]
        return {
            "dataset_dir": str(self._dir),
            "entry_index": idx,
            "episode_id": rec["episode_id"],
            "skill_id": rec["skill_id"],
            "time_index": int(rec["time_index"]),
        }

    def remove_episode(self, episode_id) -> int:
        """``episode_id`` 의 모든 entry 를 제거한다. 제거 수 반환 (episode lifecycle).

        index.jsonl 을 재기록하고 제거 entry 의 npz 를 삭제한다. 살아남는 entry 의
        npz 파일명은 그대로 두므로 (seq gap 허용) 재기록 비용이 작다 — ``append``
        가 빈 seq 를 찾으므로 이후 추가도 안전하다.
        """
        eid = str(episode_id)
        keep = [r for r in self._index if r["episode_id"] != eid]
        drop = [r for r in self._index if r["episode_id"] == eid]
        if drop:
            self._commit_index(keep, drop)
        return len(drop)

    def retain_episodes(self, episode_ids) -> int:
        """주어진 episode_id 집합 밖의 entry 를 모두 제거한다. 제거 수 반환.

        ``episode_id`` 가 비어 있는 untagged entry 는 보존한다 (legacy 안전장치 —
        ``SubgoalBuffer.retain_episodes`` 와 동일 규약).
        """
        ks = {str(e) for e in episode_ids}
        keep = [r for r in self._index
                if not r["episode_id"] or r["episode_id"] in ks]
        drop = [r for r in self._index
                if r["episode_id"] and r["episode_id"] not in ks]
        if drop:
            self._commit_index(keep, drop)
        return len(drop)

    def _commit_index(self, keep: list[dict], drop: list[dict]) -> None:
        """제거 entry 의 npz 를 지우고 index.jsonl 을 ``keep`` 으로 재기록한다."""
        arrays_dir = self._dir / "arrays"
        for rec in drop:
            npz = arrays_dir / rec["_arrays"]
            if npz.exists():
                npz.unlink()
        with open(self._dir / "index.jsonl", "w", encoding="utf-8") as f:
            for rec in keep:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._index = keep


def resolve_pointer(pointer: dict) -> RawDatasetEntry:
    """raw dataset pointer 를 entry 로 복원한다 (문서 §3 details §4).

    Args:
        pointer: ``RawTrajectoryDataset.pointer`` 가 만든 dict
            (``dataset_dir`` 와 ``entry_index`` 필수).

    Returns:
        해당 RawDatasetEntry.
    """
    return RawTrajectoryDataset(pointer["dataset_dir"]).get(int(pointer["entry_index"]))
