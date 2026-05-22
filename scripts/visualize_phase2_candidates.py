#!/usr/bin/env python3
"""Phase2 candidate 시각화 — plan_and_select 한 회의 후보 trajectory + selection.

``method3/phase2_mi_selection/candidate_dump.py`` 가 저장한 npz 를 읽어 한 장의
figure 에 두 패널을 그린다:

  (a) 3D EE 경로 — K개 후보 trajectory 의 end-effector xyz 경로.
      chosen 은 굵은 빨강, eligible(§13.2 stage1 통과)은 파랑, under_covered 는
      흐린 회색. subgoal(seed)·start 점 표시. → "어떤 traj 들이 생성됐나"
  (b) U_VLA × M̃_MI 산점도 — Useful-OOD rule(argmax U_VLA s.t. M̃_MI≥τ_MI)을
      한눈에. chosen 별표, τ_MI 경계선. → "누가 왜 선택됐나"

사용법:
    python scripts/visualize_phase2_candidates.py <dump.npz> [<dump.npz> ...]
    python scripts/visualize_phase2_candidates.py results/phase2_cands/   # 디렉터리 일괄
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

# 후보 분류별 색.
_C_CHOSEN = "#d62728"      # 선택된 후보 — 빨강
_C_ELIGIBLE = "#1f77b4"    # eligible (M̃_MI≥τ ∧ not under) — 파랑
_C_UNDER = "#bdbdbd"       # under_covered — 흐린 회색
_C_REJECT = "#9aa7b3"      # covered 지만 M̃_MI<τ — 중간 회색


def _category(i: int, chosen: int, eligible: set[int], under: np.ndarray) -> str:
    if i == chosen:
        return "chosen"
    if bool(under[i]):
        return "under"
    if i in eligible:
        return "eligible"
    return "reject"


def visualize(npz_path: str, out_path: str | None = None) -> str:
    d = np.load(npz_path, allow_pickle=True)
    ee = d["ee_paths"]                       # object (K,) of (N,3)
    K = len(ee)
    chosen = int(d["chosen_index"])
    eligible = set(int(x) for x in d["eligible_indices"].tolist())
    under = d["under_covered"]
    u_vla = d["u_vla"]
    m_mi_norm = d["m_mi_norm"]
    seed = np.asarray(d["seed_xyz"], dtype=float).reshape(-1)
    skill = str(d["skill_id"])
    tau = float(d["tau_MI"])
    accepted = bool(d["accepted"])
    episode = str(d["episode"]) if "episode" in d.files else ""
    gt_ee = (np.asarray(d["gt_ee_path"], dtype=float)
             if "gt_ee_path" in d.files else np.zeros((0, 3)))
    gt_episode = str(d["gt_episode"]) if "gt_episode" in d.files else ""

    _color = {"chosen": _C_CHOSEN, "eligible": _C_ELIGIBLE,
              "under": _C_UNDER, "reject": _C_REJECT}

    fig = plt.figure(figsize=(16, 7))

    # ── (a) 3D EE 경로 ────────────────────────────────────────────────
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    has_ee = any(np.asarray(ee[i]).ndim == 2 and len(ee[i]) > 0 for i in range(K))
    if has_ee:
        # chosen 을 맨 나중에 그려 위로 오게.
        for cat_draw in ("under", "reject", "eligible", "chosen"):
            for i in range(K):
                if _category(i, chosen, eligible, under) != cat_draw:
                    continue
                p = np.asarray(ee[i], dtype=float)
                if p.ndim != 2 or len(p) < 1:
                    continue
                if cat_draw == "chosen":
                    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=_C_CHOSEN,
                            lw=2.6, alpha=1.0, zorder=5,
                            label=f"chosen #{chosen}")
                else:
                    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=_color[cat_draw],
                            lw=0.7, alpha=0.55)
        # start 점 (chosen 경로의 첫 점) + subgoal.
        pc = np.asarray(ee[chosen], dtype=float)
        if len(pc) > 0:
            ax.scatter(*pc[0, :3], color="black", s=45, marker="o", label="start")
        if seed.shape[0] >= 3:
            ax.scatter(seed[0], seed[1], seed[2], color="#2ca02c", s=110,
                       marker="*", label="subgoal (seed)")
        # Phase1 ground-truth trajectory — P_phase1 DB 의 DCT_50 을 복원한,
        # 이 subgoal 에 대응하는 Phase1 episode 의 실제 실행 EE 경로.
        # 128 curobo 후보·chosen 이 g.t. 대비 어떻게 퍼졌는지 비교 기준.
        if gt_ee.ndim == 2 and len(gt_ee) > 0:
            ax.plot(gt_ee[:, 0], gt_ee[:, 1], gt_ee[:, 2], color="#9467bd",
                    lw=2.8, alpha=0.95, zorder=6, linestyle="-",
                    label=f"g.t. Phase1{' ' + gt_episode if gt_episode else ''}")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    else:
        ax.text2D(0.5, 0.5, "EE 경로 없음 (FK 미수행)",
                  ha="center", transform=ax.transAxes)
    ax.set_title(f"(a) {K} candidate EE paths — skill={skill}")
    ax.legend(loc="upper left", fontsize=8)

    # ── (b) U_VLA × M̃_MI 산점도 ─────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
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
    ax2.set_title(f"(b) Useful-OOD selection — accepted={accepted}\n"
                  f"rule: argmax U_VLA  s.t.  M̃_MI ≥ τ_MI")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle(f"Phase2 candidates — {episode + '  ' if episode else ''}"
                 f"{Path(npz_path).name}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = out_path or str(Path(npz_path).with_suffix(".png"))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved → {out}")
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    paths: list[Path] = []
    for a in argv:
        p = Path(a)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.npz")))
        elif p.suffix == ".npz":
            paths.append(p)
        else:
            print(f"skip (npz 아님): {a}")
    if not paths:
        print("처리할 npz 가 없습니다.")
        return 1
    for p in paths:
        try:
            visualize(str(p))
        except Exception as e:
            print(f"FAILED {p}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
