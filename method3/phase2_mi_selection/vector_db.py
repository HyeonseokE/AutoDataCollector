"""Skill-wise vector DB ``B_t^{(m)}`` — 문서 final_method3_spec §3 / §7.2.

Phase2 후보 trajectory 를 평가하기 위한 searchable index. raw trajectory
dataset 과 분리된 second layer 이며, retrieval key / action descriptor /
raw dataset pointer / metadata 를 저장한다.

  B_t^{(m)} = P_phase1^{(m)} ∪ D_phase2,t^{(m)}

skill 별로 partition 한다 — neighbor search 는 같은 skill buffer 안에서만
수행해야 서로 다른 skill 의 action mode 가 섞이지 않는다 (§8).

geometric/embedding descriptor 가 저차원이고 skill 당 규모가 작으므로
brute-force numpy L2 로 충분하다 (FAISS 는 후속 최적화).

영속화(§3): skill 별 numeric 배열은 ``.npz`` 로, ``ref``/``meta`` dict 는
JSON 문자열 배열로 같은 ``.npz`` 안에 저장한다 (Phase1 ``SubgoalBuffer`` 와
대칭 — 각 buffer 가 스스로를 영속화).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _safe_skill_name(skill_id) -> str:
    """skill_id 를 ``.npz`` 멤버 이름에 안전한 문자열로 변환한다."""
    return re.sub(r"[^0-9A-Za-z._-]", "_", str(skill_id))


def _as_npz_path(path: str | Path) -> Path:
    """경로가 ``.npz`` 확장자를 갖도록 정규화한다 (save/load 경로 일치)."""
    p = Path(path)
    return p if p.suffix == ".npz" else p.parent / (p.name + ".npz")


@dataclass(frozen=True)
class VectorDBEntry:
    """vector DB entry ``b_i^{(m)} = (e_i, z_i^a, ref_i, meta_i)`` — 문서 §3.1/§7.2."""

    skill_id: str
    state_key: np.ndarray            # e_i — [φ_VLA(o,I); p] retrieval key (D_e,)
    action_descriptor: np.ndarray    # z_i^a — DCT action descriptor (D_z,)
    ref: dict = field(default_factory=dict)   # raw dataset pointer (episode_id 등)
    meta: dict = field(default_factory=dict)  # phase, subgoal, score 등 metadata


@dataclass
class SkillVectorDB:
    """skill-wise partition 된 reference buffer ``B_t = {B_t^{(m)}}`` (문서 §7.2).

    Phase2 selector 는 같은 skill 의 entry 들로만 neighbor search 를 한다.
    """

    _skills: dict[str, list[VectorDBEntry]] = field(default_factory=dict)

    def append(self, entry: VectorDBEntry) -> None:
        """skill 의 buffer 에 entry 하나를 추가한다 (문서 §14 buffer update)."""
        self._skills.setdefault(str(entry.skill_id), []).append(entry)

    def query_skill(self, skill_id) -> list[VectorDBEntry]:
        """``B_t^{(m)}`` — 해당 skill 의 entry 목록."""
        return list(self._skills.get(str(skill_id), []))

    def state_keys(self, skill_id) -> np.ndarray:
        """해당 skill 의 state retrieval key 행렬 (N, D_e). 비면 (0, 0)."""
        entries = self._skills.get(str(skill_id), [])
        if not entries:
            return np.empty((0, 0), dtype=np.float64)
        return np.stack([np.asarray(e.state_key, dtype=np.float64) for e in entries])

    def action_descriptors(self, skill_id) -> np.ndarray:
        """``B_A^{(m)}`` — 해당 skill 의 action descriptor 행렬 (N, D_z). 비면 (0, 0)."""
        entries = self._skills.get(str(skill_id), [])
        if not entries:
            return np.empty((0, 0), dtype=np.float64)
        return np.stack(
            [np.asarray(e.action_descriptor, dtype=np.float64) for e in entries])

    def size(self, skill_id) -> int:
        """해당 skill 의 entry 수."""
        return len(self._skills.get(str(skill_id), []))

    def skill_ids(self) -> list[str]:
        """entry 가 하나라도 있는 skill_id 목록."""
        return [s for s, e in self._skills.items() if e]

    def total_size(self) -> int:
        """모든 skill 을 합친 전체 entry 수."""
        return sum(len(e) for e in self._skills.values())

    def save(self, path: str | Path) -> None:
        """vector DB 전체를 단일 ``.npz`` 로 영속화한다 (문서 §3).

        skill ``s`` 마다 ``{s}::keys`` (N, D_e), ``{s}::descriptors`` (N, D_z),
        ``{s}::ref`` / ``{s}::meta`` (N,) JSON 문자열 배열을 저장한다.
        ``ref``/``meta`` 는 JSON 직렬화 가능해야 한다 (numpy 배열 등은 불가).
        """
        path = _as_npz_path(path)
        arrays: dict[str, np.ndarray] = {}
        for skill_id, entries in self._skills.items():
            if not entries:
                continue
            s = _safe_skill_name(skill_id)
            arrays[f"{s}::keys"] = np.stack(
                [np.asarray(e.state_key, dtype=np.float64) for e in entries])
            arrays[f"{s}::descriptors"] = np.stack(
                [np.asarray(e.action_descriptor, dtype=np.float64) for e in entries])
            arrays[f"{s}::ref"] = np.array(
                [json.dumps(e.ref, ensure_ascii=False) for e in entries])
            arrays[f"{s}::meta"] = np.array(
                [json.dumps(e.meta, ensure_ascii=False) for e in entries])
        if not arrays:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    def load(self, path: str | Path) -> None:
        """``.npz`` 에서 vector DB 를 복원한다 (preload). 파일 없으면 no-op."""
        path = _as_npz_path(path)
        if not path.exists():
            return
        with np.load(path, allow_pickle=False) as data:
            skills = sorted({m.split("::", 1)[0] for m in data.files if "::" in m})
            for s in skills:
                if f"{s}::keys" not in data.files:
                    continue
                keys = data[f"{s}::keys"]
                descs = data[f"{s}::descriptors"]
                refs = data[f"{s}::ref"]
                metas = data[f"{s}::meta"]
                self._skills[s] = [
                    VectorDBEntry(
                        skill_id=s,
                        state_key=np.asarray(keys[i], dtype=np.float64),
                        action_descriptor=np.asarray(descs[i], dtype=np.float64),
                        ref=json.loads(str(refs[i])),
                        meta=json.loads(str(metas[i])),
                    )
                    for i in range(keys.shape[0])
                ]
