#!/usr/bin/env python3
"""Phase2 candidate 시각화 — top-view 오버레이 + Useful-OOD selection 산점도.

``method3/phase2_mi_selection/candidate_dump.py`` 가 저장한 npz 를 읽어 한 장의
figure 에 두 패널을 그린다:

  (a) **top-view 오버레이** — plan_and_select 때 server 로 보낸 top-view 카메라
      이미지를 배경으로, 128개 후보 traj + g.t.(Phase1 실제) + chosen 을 2D
      오버레이한다. robot xyz 경로를 charuco calibration(cam_to_base, K, dist)
      으로 pixel 에 투영. top_image 가 없는 구 dump 는 3D EE plot 으로 fallback.
  (b) U_VLA × M̃_MI 산점도 — argmax U_VLA s.t. M̃_MI≥τ_MI 를 한눈에.

사용법:
    python scripts/visualize_phase2_candidates.py <dump.npz> [...]
    python scripts/visualize_phase2_candidates.py results/phase2_cands/   # 디렉터리 재귀
    옵션: --calib <robotN_cam2robot.npz>  (기본 robot4)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

_C_CHOSEN = "#d62728"      # chosen — 빨강
_C_ELIGIBLE = "#ff7f0e"    # eligible — orange
_C_UNDER = "#bdbdbd"       # under_covered — 흐린 회색
_C_REJECT = "#9aa7b3"      # covered 지만 M̃_MI<τ — 중간 회색
_C_GT = "#9467bd"          # g.t. Phase1 — 자홍
def _autodetect_default_calib() -> str:
    """robot_configs/charuco_calibration/ 의 robotN_cam2robot.npz 중 첫 번째 매치.
    여러 robot 머신 환경에서 default 가 머신마다 다르게 풀리도록 동적 resolve."""
    import glob
    cands = sorted(glob.glob("robot_configs/charuco_calibration/robot*_cam2robot.npz"))
    return cands[0] if cands else "robot_configs/charuco_calibration/robot0_cam2robot.npz"


_DEFAULT_CALIB = _autodetect_default_calib()


def _category(i: int, chosen: int, eligible: set[int], under) -> str:
    if i == chosen:
        return "chosen"
    if bool(under[i]):
        return "under"
    if i in eligible:
        return "eligible"
    return "reject"


def _load_calib(path: str):
    """charuco calib npz → {base_to_cam, K, dist}. 없으면 None."""
    p = Path(path)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "base_to_cam": np.linalg.inv(np.asarray(d["cam_to_base"], dtype=float)),
        "K": np.asarray(d["K"], dtype=float),
        "dist": np.asarray(d["dist"], dtype=float),
    }


def _project(robot_xyz, calib) -> np.ndarray:
    """robot(base_link) xyz (N,3) → top-view pixel (N,2)."""
    import cv2

    pts = np.atleast_2d(np.asarray(robot_xyz, dtype=float))
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        return np.zeros((0, 2))
    h = np.hstack([pts[:, :3], np.ones((len(pts), 1))])
    cam = (calib["base_to_cam"] @ h.T).T[:, :3]
    px, _ = cv2.projectPoints(
        cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        calib["K"], calib["dist"],
    )
    return px.reshape(-1, 2)


def _panel_a_overlay(ax, ee, K, chosen, eligible, under, gt_ee, gt_episode,
                     goal_ee, top_image, skill, calib) -> None:
    """(a) top-view 이미지 위에 후보·g.t.·chosen trajectory 를 2D 오버레이."""
    ax.imshow(top_image)
    H, W = top_image.shape[:2]
    color = {"under": _C_UNDER, "reject": _C_REJECT,
             "eligible": _C_ELIGIBLE, "chosen": _C_CHOSEN}
    for cat in ("under", "reject", "eligible", "chosen"):
        for i in range(K):
            if _category(i, chosen, eligible, under) != cat:
                continue
            p = np.asarray(ee[i], dtype=float)
            if p.ndim != 2 or len(p) < 1:
                continue
            px = _project(p, calib)
            if len(px) < 1:
                continue
            if cat == "chosen":
                ax.plot(px[:, 0], px[:, 1], color=_C_CHOSEN, lw=2.6,
                        zorder=6, label=f"chosen #{chosen}")
            else:
                ax.plot(px[:, 0], px[:, 1], color=color[cat], lw=0.7, alpha=0.55)
    if gt_ee.ndim == 2 and len(gt_ee) > 0:
        gpx = _project(gt_ee, calib)
        if len(gpx) > 0:
            ax.plot(gpx[:, 0], gpx[:, 1], color=_C_GT, lw=2.8, zorder=7,
                    label=f"g.t. Phase1{' ' + gt_episode if gt_episode else ''}")
    if goal_ee is not None:
        spx = _project(goal_ee, calib)
        if len(spx) > 0:
            ax.scatter(spx[0, 0], spx[0, 1], color="#2ca02c", s=140,
                       marker="*", zorder=8, edgecolor="black", linewidth=0.5,
                       label="subgoal")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(f"(a) top-view overlay — {K} candidates, skill={skill}")
    ax.legend(loc="upper right", fontsize=8)


def _panel_a_3d(ax, ee, K, chosen, eligible, under, gt_ee, gt_episode,
                seed, skill) -> None:
    """(a) fallback — top_image/calib 없을 때 3D EE plot."""
    color = {"under": _C_UNDER, "reject": _C_REJECT,
             "eligible": _C_ELIGIBLE, "chosen": _C_CHOSEN}
    for cat in ("under", "reject", "eligible", "chosen"):
        for i in range(K):
            if _category(i, chosen, eligible, under) != cat:
                continue
            p = np.asarray(ee[i], dtype=float)
            if p.ndim != 2 or len(p) < 1:
                continue
            if cat == "chosen":
                ax.plot(p[:, 0], p[:, 1], p[:, 2], color=_C_CHOSEN, lw=2.6,
                        zorder=5, label=f"chosen #{chosen}")
            else:
                ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color[cat],
                        lw=0.7, alpha=0.55)
    if gt_ee.ndim == 2 and len(gt_ee) > 0:
        ax.plot(gt_ee[:, 0], gt_ee[:, 1], gt_ee[:, 2], color=_C_GT, lw=2.8,
                zorder=6, label=f"g.t. Phase1{' ' + gt_episode if gt_episode else ''}")
    if seed is not None and seed.shape[0] >= 3:
        ax.scatter(seed[0], seed[1], seed[2], color="#2ca02c", s=110,
                   marker="*", label="subgoal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"(a) {K} candidate EE paths (3D) — skill={skill}")
    ax.legend(loc="upper left", fontsize=8)


def visualize(npz_path: str, calib_path: str = _DEFAULT_CALIB,
              out_path: str | None = None) -> str:
    d = np.load(npz_path, allow_pickle=True)
    ee = d["ee_paths"]
    K = len(ee)
    chosen = int(d["chosen_index"])
    eligible = set(int(x) for x in d["eligible_indices"].tolist())
    under = d["under_covered"]
    u_vla = d["u_vla"]
    m_mi_norm = d["m_mi_norm"]
    skill = str(d["skill_id"])
    tau = float(d["tau_MI"])
    accepted = bool(d["accepted"])
    episode = str(d["episode"]) if "episode" in d.files else ""
    gt_ee = (np.asarray(d["gt_ee_path"], dtype=float)
             if "gt_ee_path" in d.files else np.zeros((0, 3)))
    gt_episode = str(d["gt_episode"]) if "gt_episode" in d.files else ""
    goal_ee = (np.asarray(d["goal_ee_xyz"], dtype=float).reshape(-1)
               if "goal_ee_xyz" in d.files else None)
    seed = (np.asarray(d["seed_xyz"], dtype=float).reshape(-1)
            if "seed_xyz" in d.files else None)
    top_image = (np.asarray(d["top_image"]) if "top_image" in d.files
                 else np.zeros((0, 0, 3), dtype=np.uint8))
    has_img = top_image.ndim == 3 and top_image.size > 0
    calib = _load_calib(calib_path) if has_img else None

    out_overlay = out_path or str(Path(npz_path).with_suffix(".png"))

    # ── fig1: top-view 오버레이 (또는 3D fallback) — 경로 시각화 ──
    fig1 = plt.figure(figsize=(8, 7))
    if has_img and calib is not None:
        ax = fig1.add_subplot(111)
        _panel_a_overlay(ax, ee, K, chosen, eligible, under, gt_ee,
                         gt_episode, goal_ee, top_image, skill, calib)
    else:
        ax = fig1.add_subplot(111, projection="3d")
        _panel_a_3d(ax, ee, K, chosen, eligible, under, gt_ee, gt_episode,
                    seed, skill)
    fig1.tight_layout()
    fig1.savefig(out_overlay, dpi=130)
    plt.close(fig1)

    # ── fig2: U_VLA × M̃_MI 산점도 — 별도 파일 (<stem>_score.png) ──
    fig2 = plt.figure(figsize=(7, 6))
    ax2 = fig2.add_subplot(111)
    _color = {"under": _C_UNDER, "eligible": _C_ELIGIBLE, "reject": _C_REJECT}
    for cat in ("under", "reject", "eligible"):
        xs = [m_mi_norm[i] for i in range(K)
              if _category(i, chosen, eligible, under) == cat]
        ys = [u_vla[i] for i in range(K)
              if _category(i, chosen, eligible, under) == cat]
        if xs:
            ax2.scatter(xs, ys, color=_color[cat], s=26, alpha=0.75,
                        label=f"{cat} ({len(xs)})")
    ax2.scatter([m_mi_norm[chosen]], [u_vla[chosen]], color=_C_CHOSEN, s=240,
                marker="*", edgecolor="black", linewidth=0.6, zorder=6,
                label=f"chosen #{chosen}")
    ax2.axvline(tau, color="#2ca02c", ls="--", lw=1.3, label=f"τ_MI = {tau:g}")
    ax2.set_xlabel("M̃_MI  (batch-normalized MI usefulness)")
    ax2.set_ylabel("U_VLA  (VLA single-step denoise loss)")
    ax2.set_title(f"Useful-OOD selection — accepted={accepted}\n"
                  f"argmax U_VLA  s.t.  M̃_MI ≥ τ_MI  (skill={skill})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    _p = Path(out_overlay)
    out_score = str(_p.with_name(_p.stem + "_score" + _p.suffix))
    fig2.savefig(out_score, dpi=130)
    plt.close(fig2)

    print(f"saved → {out_overlay} , {out_score}")
    return out_overlay


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    calib_path = _DEFAULT_CALIB
    args: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--calib" and i + 1 < len(argv):
            calib_path = argv[i + 1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    paths: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.npz")))
        elif p.suffix == ".npz":
            paths.append(p)
        else:
            print(f"skip (npz 아님): {a}")
    if not paths:
        print("처리할 npz 가 없습니다.")
        return 1
    for p in paths:
        try:
            visualize(str(p), calib_path=calib_path)
        except Exception as e:
            print(f"FAILED {p}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
