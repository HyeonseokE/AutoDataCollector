"""Phase2 candidate dump — plan_and_select 의 후보 trajectory + 점수 보존.

curobo ``plan_batch`` 가 만든 K(=128)개 후보는 Useful-OOD selection 후 1개만
client 로 반환되고 나머지는 버려진다 — 무엇이 후보였고 왜 그게 뽑혔는지 사후
분석할 길이 없다. 본 모듈은 selection 직후 전체 후보 상태를 npz 로 저장한다:

  * 각 후보의 EE xyz 경로 (curobo batched FK)
  * per-candidate MI/U_VLA 점수 (ΔH_A, ΔH_A|S, M_MI, M̃_MI, U_VLA, covered)
  * 선택 결과 (chosen_index, accepted, eligible_indices)

``scripts/visualize_phase2_candidates.py`` 가 이 npz 를 소비해 3D EE 경로 +
selection 산점도를 그린다.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _ee_paths_via_fk(curobo_backend, joint_wps: list[np.ndarray]):
    """candidate 별 joint waypoints (N,arm_dof) → EE xyz 경로 (N,3) 리스트.

    curobo backend 의 batched FK 를 *한 번* 호출해 모든 후보의 모든 waypoint 를
    변환한다 (GPU 호출 1회). backend 가 FK API 를 노출하지 않으면 None.
    """
    to_full = getattr(curobo_backend, "_to_full_qpos", None)
    fk_batch = getattr(curobo_backend, "_compute_ee_xyz_quat_batch", None)
    if not callable(to_full) or not callable(fk_batch):
        return None
    all_full: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for wp in joint_wps:
        offsets.append((len(all_full), len(wp)))
        for q in wp:
            all_full.append(to_full(np.asarray(q, dtype=float)))
    if not all_full:
        return None
    ee_xq = fk_batch(all_full)
    ee_all = np.asarray([np.asarray(e[0], dtype=float) for e in ee_xq])  # (ΣN, 3)
    return [ee_all[o:o + n] for o, n in offsets]


def dump_phase2_candidates(
    curobo_backend,
    cands,
    selection,
    *,
    skill_id: str,
    seed_xyz,
    start_qpos,
    goal_qpos,
    tau_MI: float,
    dump_dir: str = "results/phase2_cands",
    episode: str = "",
) -> str:
    """plan_and_select 한 회의 전체 후보 상태를 npz 로 저장. 저장 경로 반환.

    npz 키:
      ee_paths          object[(K,)] — 후보별 EE xyz 경로 (N_i, 3)
      joint_waypoints   object[(K,)] — 후보별 joint waypoints (N_i, dof)
      delta_h_a / delta_h_a_given_s / m_mi / m_mi_norm / covered_ratio /
      u_vla             float[(K,)]  — per-candidate 점수
      under_covered     bool[(K,)]
      chosen_index      int          — 선택된 후보
      accepted          bool         — Useful-OOD accept 여부
      eligible_indices  int[(*,)]    — §13.2 stage1 통과 후보
      seed_xyz / start_qpos / goal_qpos / skill_id / tau_MI / episode — meta
    """
    reports = list(selection.reports or [])
    joint_wps = [
        np.asarray(getattr(c, "waypoints", np.zeros((1, 5))), dtype=float)
        for c in cands
    ]
    K = len(joint_wps)

    try:
        ee_paths = _ee_paths_via_fk(curobo_backend, joint_wps)
    except Exception:
        ee_paths = None
    if ee_paths is None:
        # FK 불가 — EE 경로는 빈 (0,3) 으로 두고 joint waypoints 만 보존.
        ee_paths = [np.zeros((0, 3), dtype=float) for _ in joint_wps]

    def _col(attr: str) -> np.ndarray:
        if not reports:
            return np.zeros(K, dtype=float)
        return np.array([float(getattr(r, attr)) for r in reports], dtype=float)

    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    tag = f"{episode + '_' if episode else ''}{skill_id}_{stamp}"
    path = str(Path(dump_dir) / f"{tag}.npz")

    np.savez(
        path,
        ee_paths=np.array(ee_paths, dtype=object),
        joint_waypoints=np.array(joint_wps, dtype=object),
        delta_h_a=_col("delta_h_a"),
        delta_h_a_given_s=_col("delta_h_a_given_s"),
        m_mi=_col("q2"),
        m_mi_norm=_col("q2_norm"),
        covered_ratio=_col("covered_ratio"),
        u_vla=_col("u_vla"),
        under_covered=(
            np.array([bool(r.under_covered) for r in reports], dtype=bool)
            if reports else np.zeros(K, dtype=bool)
        ),
        chosen_index=int(selection.chosen_index),
        accepted=bool(selection.accepted),
        eligible_indices=np.asarray(list(selection.eligible_indices), dtype=int),
        seed_xyz=np.asarray(seed_xyz, dtype=float).reshape(-1)[:3],
        start_qpos=np.asarray(start_qpos, dtype=float).reshape(-1),
        goal_qpos=np.asarray(goal_qpos, dtype=float).reshape(-1),
        skill_id=str(skill_id),
        tau_MI=float(tau_MI),
        episode=str(episode),
    )
    return path
