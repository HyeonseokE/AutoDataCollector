"""Phase2 candidate dump — plan_and_select 의 후보 trajectory + 점수 + g.t. 보존.

curobo ``plan_batch`` 가 만든 K(=128)개 후보는 Useful-OOD selection 후 1개만
client 로 반환되고 나머지는 버려진다 — 무엇이 후보였고 왜 그게 뽑혔는지 사후
분석할 길이 없다. 본 모듈은 selection 직후 전체 후보 상태를 npz 로 저장한다:

  * 각 후보의 EE xyz 경로 (curobo batched FK)
  * per-candidate MI/U_VLA 점수 (ΔH_A, ΔH_A|S, M_MI, M̃_MI, U_VLA, covered)
  * 선택 결과 (chosen_index, accepted, eligible_indices)
  * **g.t. trajectory** — 이 subgoal 에 대응하는 Phase1 의 실제 실행 경로.
    P_phase1 DB 의 <skill>::descriptors (DCT_50) 를 dct_to_traj 로 복원 →
    curobo FK 로 EE 경로화. Phase1 dataset 손상과 무관하게 추출 가능.

``scripts/visualize_phase2_candidates.py`` 가 이 npz 를 소비해 128 후보 + chosen
+ g.t. 를 3D EE 경로로, selection 을 산점도로 그린다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# server 가 직접 P_phase1 build 할 때 쓰는 cache 경로 (method3_setup 와 동일).
_DEFAULT_DB_NPZ = "grpc_server/buffer/server_skill_wise_vector_db.npz"


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


def _goal_ee_xyz(curobo_backend, goal_qpos) -> np.ndarray | None:
    """goal_qpos (joint) → EE xyz (3,). FK 불가 시 None."""
    to_full = getattr(curobo_backend, "_to_full_qpos", None)
    fk1 = getattr(curobo_backend, "_compute_ee_xyz_quat", None)
    if not callable(to_full) or not callable(fk1):
        return None
    try:
        xyz, _quat = fk1(to_full(np.asarray(goal_qpos, dtype=float)))
        return np.asarray(xyz, dtype=float).reshape(-1)[:3]
    except Exception:
        return None


def _extract_gt(curobo_backend, db_npz_path: str, skill_id: str,
                goal_ee_xyz: np.ndarray, converter=None,
                start_ee_xyz: np.ndarray | None = None,
                target_episode_id: str = ""):
    """P_phase1 DB 에서 이 skill segment 의 g.t. trajectory 를 복원.

    DB 의 ``<skill>::descriptors`` (DCT_50, servo-space) 를 모두 ``dct_to_traj``
    로 역변환 → ``converter`` 로 radians → curobo FK 로 EE 경로화한다. 그 중
    이번 plan 의 **start_ee + goal_ee 둘 다**에 가장 가까운 entry 를 고른다.

    start 까지 보는 이유: ``move`` 처럼 episode 당 여러 번 나오는 skill 은
    도달점(goal)만으로는 instance 를 구분할 수 없다 (time_index 2 의 move 와
    4 의 move 는 goal 이 비슷해도 start 가 다름). goal 만으로 nearest 하면
    엉뚱한 instance 의 g.t. 가 잡혀 "여러 동작이 합쳐진" 경로처럼 보인다.

    ``target_episode_id`` 가 주어지면 *그 episode 의 entries 만* nearest 후보로
    제한한다 (= Phase2SubgoalReplay 의 매핑된 phase1 episode). 빈 string 이면
    legacy nearest-neighbor (= 모든 episode 의 entries 중 nearest).

    Returns:
        (gt_ee_path (T,3) | None, episode_id: str, gt_subgoal (3,) | None)
    """
    if not Path(db_npz_path).exists():
        return None, "", None
    try:
        from method3.dct.transform import dct_to_traj
    except Exception:
        return None, "", None
    d = np.load(db_npz_path, allow_pickle=True)
    dkey, mkey, rkey = (f"{skill_id}::descriptors", f"{skill_id}::meta",
                        f"{skill_id}::ref")
    if dkey not in d.files:
        return None, "", None
    descriptors = d[dkey]                                   # (N, L0*dof)
    metas = d[mkey] if mkey in d.files else None
    refs = d[rkey] if rkey in d.files else None
    if len(descriptors) == 0:
        return None, "", None

    # DB descriptor 는 EE delta DCT (seed_builder.ee_delta_dct_from_poses 결과,
    # (L0, 6) [Δxyz + Δrpy]). 옛 paradigm 의 joint DCT 가정 + curobo FK 는
    # 잘못된 visualization 산출 → 2026-05-28 RCA: descriptor 를 *EE delta DCT*
    # 그대로 역변환 + cumulative sum 으로 absolute EE xyz path 복원.
    L0 = 50
    start_xyz = (np.asarray(start_ee_xyz, dtype=float).reshape(-1)[:3]
                 if start_ee_xyz is not None else np.zeros(3))
    ee_all: list[np.ndarray] = []
    for i in range(len(descriptors)):
        desc = np.asarray(descriptors[i], dtype=float)
        # (L0, 6) Δxyz+Δrpy 로 reshape (size mismatch 시 skip).
        if desc.size != L0 * 6:
            ee_all.append(None)
            continue
        try:
            delta = np.asarray(
                dct_to_traj(desc.reshape(L0, 6), L0), dtype=float)
        except Exception:
            ee_all.append(None)
            continue
        # cumulative Δxyz + start = absolute EE xyz path (L0, 3).
        # pose[0] = start, pose[t] = start + sum(delta[:t]).
        xyz_path = np.empty((L0, 3), dtype=float)
        xyz_path[0] = start_xyz
        if L0 > 1:
            xyz_path[1:] = start_xyz + np.cumsum(delta[:L0 - 1, :3], axis=0)
        ee_all.append(xyz_path)

    # target_episode_id 가 주어지면 그 episode 의 entries 만 후보로 제한.
    # refs[i] 의 episode_id 가 매칭되는 index 만 enabled.
    _enabled_mask = None
    _target_ep = str(target_episode_id or "").strip()
    if _target_ep and refs is not None:
        _enabled_mask = np.zeros(len(ee_all), dtype=bool)
        for i in range(len(ee_all)):
            try:
                _ref_ep = str(json.loads(str(refs[i])).get("episode_id", ""))
            except Exception:
                _ref_ep = ""
            if _ref_ep == _target_ep:
                _enabled_mask[i] = True
        if not _enabled_mask.any():
            # target episode 의 entry 가 DB 에 없음 → fallback 전체 nearest.
            _enabled_mask = None

    # start+goal 동시 매칭 — entry 의 EE 경로 끝(=도달점)·시작과 비교.
    g = np.asarray(goal_ee_xyz, dtype=float).reshape(-1)[:3]
    s = (np.asarray(start_ee_xyz, dtype=float).reshape(-1)[:3]
         if start_ee_xyz is not None else None)
    best, best_score = -1, float("inf")
    for i, ee in enumerate(ee_all):
        if ee is None:
            continue
        if _enabled_mask is not None and not _enabled_mask[i]:
            continue
        ee = np.asarray(ee, dtype=float)
        if ee.ndim != 2 or len(ee) < 2:
            continue
        score = float(np.linalg.norm(ee[-1] - g))
        if s is not None:
            score += float(np.linalg.norm(ee[0] - s))
        if score < best_score:
            best_score, best = score, i
    if best < 0:
        return None, "", None

    gt_ee = np.asarray(ee_all[best], dtype=float)
    episode_id, gt_subgoal = "", None
    if refs is not None:
        try:
            episode_id = str(json.loads(str(refs[best])).get("episode_id", ""))
        except Exception:
            episode_id = ""
    if metas is not None:
        try:
            _sg = json.loads(str(metas[best])).get("subgoal")
            gt_subgoal = np.asarray(_sg, dtype=float) if _sg is not None else None
        except Exception:
            gt_subgoal = None
    return gt_ee, episode_id, gt_subgoal


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
    selection_id: str = "",
    db_npz_path: str = _DEFAULT_DB_NPZ,
    top_image=None,
    servo_calib_path: str | None = None,
    target_phase1_episode_id: str = "",
) -> str:
    """plan_and_select 한 회의 전체 후보 상태 + g.t. 를 npz 로 저장. 경로 반환.

    npz 키:
      ee_paths          object[(K,)] — 후보별 EE xyz 경로 (N_i, 3)
      joint_waypoints   object[(K,)] — 후보별 joint waypoints (N_i, dof)
      delta_h_a / delta_h_a_given_s / m_mi / m_mi_norm / covered_ratio /
      u_vla             float[(K,)]  — per-candidate 점수
      under_covered     bool[(K,)]
      chosen_index      int          — 선택된 후보
      accepted          bool         — Useful-OOD accept 여부
      eligible_indices  int[(*,)]    — §13.2 stage1 통과 후보
      gt_ee_path        float[(T,3)] — Phase1 g.t. trajectory EE 경로
      gt_episode / gt_subgoal        — 매칭된 Phase1 episode / 도달 subgoal
      goal_ee_xyz       float[(3,)]  — 이번 plan 의 목표 EE (FK)
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
        ee_paths = [np.zeros((0, 3), dtype=float) for _ in joint_wps]

    # 이번 plan 의 목표 EE (FK) + Phase1 g.t. trajectory.
    goal_ee = _goal_ee_xyz(curobo_backend, goal_qpos)
    gt_ee_path, gt_episode, gt_subgoal = None, "", None
    if goal_ee is not None:
        try:
            _conv = None
            if servo_calib_path:
                from method3.phase2_mi_selection.curobo_candidate_gen import (
                    _get_converter,
                )
                _conv = _get_converter(servo_calib_path)
            # start_ee — 이번 plan 의 출발 EE. move 처럼 episode 당 여러 번
            # 나오는 skill 의 g.t. instance 구분에 쓰인다.
            _start_ee = _goal_ee_xyz(curobo_backend, start_qpos)
            gt_ee_path, gt_episode, gt_subgoal = _extract_gt(
                curobo_backend, db_npz_path, skill_id, goal_ee, _conv,
                start_ee_xyz=_start_ee,
                target_episode_id=target_phase1_episode_id)
        except Exception:
            gt_ee_path, gt_episode, gt_subgoal = None, "", None

    def _col(attr: str) -> np.ndarray:
        if not reports:
            return np.zeros(K, dtype=float)
        return np.array([float(getattr(r, attr)) for r in reports], dtype=float)

    # selection_id 파일명 — client 가 PlanResponse.selection_id 로 이 dump 를
    # 특정해 episode 종료 시 가져간다. selection_id 가 비면 timestamp fallback.
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    _name = selection_id or (
        time.strftime("%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}_{skill_id}"
    )
    path = str(Path(dump_dir) / f"{_name}.npz")

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
        gt_ee_path=(np.asarray(gt_ee_path, dtype=float)
                    if gt_ee_path is not None else np.zeros((0, 3), dtype=float)),
        gt_episode=str(gt_episode or ""),
        gt_subgoal=(np.asarray(gt_subgoal, dtype=float).reshape(-1)[:3]
                    if gt_subgoal is not None else np.zeros(3, dtype=float)),
        goal_ee_xyz=(np.asarray(goal_ee, dtype=float)
                     if goal_ee is not None else np.zeros(3, dtype=float)),
        seed_xyz=np.asarray(seed_xyz, dtype=float).reshape(-1)[:3],
        start_qpos=np.asarray(start_qpos, dtype=float).reshape(-1),
        goal_qpos=np.asarray(goal_qpos, dtype=float).reshape(-1),
        # top-view 카메라 이미지 — client 가 robot→pixel 투영으로 trajectory
        # 오버레이를 그릴 배경. 없으면 빈 배열.
        top_image=(np.asarray(top_image, dtype=np.uint8)
                   if top_image is not None
                   else np.zeros((0, 0, 3), dtype=np.uint8)),
        skill_id=str(skill_id),
        tau_MI=float(tau_MI),
        selection_id=str(selection_id),
    )
    return path
