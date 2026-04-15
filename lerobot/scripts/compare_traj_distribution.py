#!/usr/bin/env python3
"""두 LeRobotDataset 의 궤적 분포 비교.

Spatial overlay (3D EE 경로), Step-level + Trajectory-level UMAP/t-SNE,
정량 지표 (MMD, energy distance, DTW) 를 한번에 생성.

사용:
  python scripts/compare_traj_distribution.py \\
      --human-repo skkuprism/teleop_pnp_100ep_urdf0 \\
      --auto-repo  skkuprism/cap_pnp_100ep \\
      --urdf assets/urdf/so101_robot0.urdf \\
      --output-dir outputs/traj_compare/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 스크립트 경로 등록
# lerobot/scripts/ 위치 → 프로젝트 루트는 parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traj_analysis import loader, fk_ee, resample, viz_spatial, viz_embedding, metrics  # noqa: E402


STATE_KEY = "observation.state.radian_urdf0"
ACTION_KEY = "action.radian_urdf0"


def _ep_ids(trajs, offset: int = 0) -> np.ndarray:
    """각 에피소드의 index 를 timestep 단위로 broadcast."""
    out = []
    for i, t in enumerate(trajs):
        out.append(np.full(len(t), i + offset, dtype=np.int32))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int32)


def generate_readme(output_dir: Path, summary: dict) -> None:
    """각 시각화 파일의 의미를 설명하는 README.md 자동 생성."""
    readme = output_dir / "README.md"
    readme.write_text(f"""# Trajectory Distribution Comparison

**Human**: `{summary['human_repo']}` — {summary['human']['num_episodes']} ep, {summary['human']['total_frames']} frames
**Auto**:  `{summary['auto_repo']}` — {summary['auto']['num_episodes']} ep, {summary['auto']['total_frames']} frames

두 데이터셋의 궤적 분포가 서로 다른지 (사람은 다양, 코드는 mode collapse)
를 검증하기 위한 시각화 및 정량 지표 모음.

---

## 1. Spatial Overlay (`spatial/`)

EE(gripper_frame_link) 끝단의 (x, y, z) 궤적을 3D 공간에 직접 그림.
**물리 경로의 다양성**을 시각적으로 확인.

| 파일 | 무엇을 보기 위함 | 뭘 시각화 |
|------|-----------------|--------|
| `spatial/line_overlay__ee_xyz.png` | 경로 중첩/분산 패턴 | 모든 에피소드 EE 라인을 low-alpha 로 겹쳐 그림. 같은 경로 반복 → 진한 띠, 흩어짐 → 옅은 구름 |
| `spatial/voxel_density__ee_xyz.png` | 3D 공간 커버리지 정량화 | (x,y,z) 를 50×50×50 voxel 로 이산화하고 log(방문 횟수) 를 점 투명도/크기로 표현. 좌: human, 우: auto |
| `spatial/interactive__ee_xyz.html` | 탐색/검증용 | Plotly 3D, 회전·줌 가능. 발표에는 PNG 사용 권장 |

**해석 포인트**: human 은 장애물 회피 등으로 경로가 좌/우 분기해 퍼짐 (multimodal),
auto 는 최단 경로 한 줄로 고밀도 띠 (mode collapse).

---

## 2. State-Space Embedding (`embedding/`)

고차원 radian state/action 을 UMAP·t-SNE 로 2D 축소.
**state 공간 분포** 및 **에피소드 간 다양성**을 군집으로 증명.

### 2a. Step-level — 모든 timestep = 점 1개

매 프레임을 개별 점으로 간주 (10fps 서브샘플링 적용).
**"어느 state 영역을 방문했는가" 커버리지**를 본다.

| 파일 | 무엇을 보기 위함 | 뭘 시각화 |
|------|-----------------|--------|
| `embedding/step_level__umap__state_rad.png` | 관절 configuration 공간 커버리지 | observation.state.radian_urdf0 (6D) 의 모든 timestep 을 UMAP 으로 2D 투영 |
| `embedding/step_level__umap__action_rad.png` | 명령 공간 커버리지 | action.radian_urdf0 (6D) 를 UMAP 으로 투영 |
| `embedding/step_level__tsne__state_rad.png` | UMAP 결과 교차 검증 | 같은 데이터, t-SNE perplexity=20~30 |
| `embedding/step_level__tsne__action_rad.png` | 동일 | action 신호, t-SNE |
| `embedding/step_level__*.html` | 탐색 | Plotly 인터랙티브, hover 시 episode_index 표시 |

**해석 포인트**: human 점들은 넓고 다양한 영역,
auto 점들은 좁은 고밀도 cluster → state-space coverage 차이.

### 2b. Trajectory-level — 에피소드 1개 = 점 1개

각 에피소드를 100-step 으로 선형 보간 → flatten → 600D 벡터 1개로 임베딩.
**"에피소드끼리 얼마나 다른가" 다양성**을 본다.

| 파일 | 무엇을 보기 위함 | 뭘 시각화 |
|------|-----------------|--------|
| `embedding/trajectory_level__umap__state_rad.png` | 에피소드 간 다양성 (관절) | 각 에피소드의 state_rad 궤적 전체를 한 벡터로 → UMAP 2D |
| `embedding/trajectory_level__umap__action_rad.png` | 에피소드 간 전략 다양성 | action_rad 전체 궤적 기준 |
| `embedding/trajectory_level__tsne__*.png` | UMAP 교차 검증 | t-SNE 버전 |
| `embedding/trajectory_level__*.html` | 탐색 | Plotly, hover 로 어느 에피소드인지 확인 |

**해석 포인트**: human 에피소드들은 서로 떨어진 개별 점들,
auto 에피소드들은 한두 점에 뭉침 → intra-dataset 다양성 차이.

---

## 3. Metrics (`metrics.json`)

정량 지표. 주요 항목:

- `human.episode_length` / `auto.episode_length`: 에피소드 길이 통계 (mean/std/min/max/median)
- `human.path_length_ee` / `auto.path_length_ee`: EE 누적 이동 거리 분포 — 최단 경로 여부
- `human.intra_dtw_state_rad` / `auto.intra_dtw_state_rad`: 같은 데이터셋 내 에피소드 쌍 DTW 거리 평균 (길이 정규화) — **클수록 내부 다양성 높음**
- `between.mmd_rbf_step_state_rad`: 두 분포 간 MMD (RBF kernel)
- `between.energy_distance_step_state_rad`: Székely energy distance
- `between.hull_area_ratio_state_rad_umap`: UMAP 공간에서 human hull 면적 / auto hull 면적

---

## 처리 설정

- Step-level 서브샘플링: {summary['subsample_stride']}× (원본 fps / stride)
- Trajectory-level resample 길이: {summary['target_length']} steps (선형 보간)
- FK: URDF `{summary['urdf']}`, end_effector=`gripper_frame_link`
- UMAP: n_components=2, n_neighbors=15, min_dist=0.1, random_state=42
- t-SNE: perplexity 자동 (max 30), random_state=42
- Step-level max_points: {summary['max_points']} (초과 시 random subsample)
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human-repo", required=True)
    ap.add_argument("--auto-repo", required=True)
    ap.add_argument("--urdf", default=str(PROJECT_ROOT / "assets/urdf/so101_robot0.urdf"))
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "lerobot/outputs/traj_compare"))
    ap.add_argument("--subsample-stride", type=int, default=3, help="Step-level fps subsample (30→10fps = 3)")
    ap.add_argument("--target-length", type=int, default=100, help="Trajectory-level resample length")
    ap.add_argument("--max-step-points", type=int, default=20000, help="Step-level UMAP 최대 점 수")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    (out / "spatial").mkdir(parents=True, exist_ok=True)
    (out / "embedding").mkdir(parents=True, exist_ok=True)

    # ─── 1) 궤적 로딩 ──────────────────────────────
    print("[1/5] Loading datasets ...")
    h_raw = loader.load_episode_trajectories(args.human_repo, keys=[STATE_KEY, ACTION_KEY])
    a_raw = loader.load_episode_trajectories(args.auto_repo,  keys=[STATE_KEY, ACTION_KEY])
    h_state = h_raw[STATE_KEY]; h_action = h_raw[ACTION_KEY]
    a_state = a_raw[STATE_KEY]; a_action = a_raw[ACTION_KEY]
    h_sum = loader.summarize(h_state, "human")
    a_sum = loader.summarize(a_state, "auto")
    print(f"  human: {h_sum['num_episodes']} ep, {h_sum['total_frames']} frames, "
          f"len={h_sum['len_mean']:.1f}±{h_sum['len_std']:.1f}")
    print(f"  auto:  {a_sum['num_episodes']} ep, {a_sum['total_frames']} frames, "
          f"len={a_sum['len_mean']:.1f}±{a_sum['len_std']:.1f}")

    # ─── 2) FK → EE xyz ──────────────────────────────
    print("[2/5] FK computing EE xyz ...")
    h_xyz = fk_ee.trajs_to_ee_xyz(h_state, args.urdf)
    a_xyz = fk_ee.trajs_to_ee_xyz(a_state, args.urdf)

    # ─── 3) Spatial overlay ──────────────────────────
    print("[3/5] Rendering spatial overlay ...")
    sp = out / "spatial"
    viz_spatial.plot_line_overlay(h_xyz, a_xyz, str(sp / "line_overlay__ee_xyz.png"))
    viz_spatial.plot_voxel_density(h_xyz, a_xyz, str(sp / "voxel_density__ee_xyz.png"))
    viz_spatial.plot_interactive_3d(h_xyz, a_xyz, str(sp / "interactive__ee_xyz.html"))
    print(f"  saved: {sp}")

    # ─── 4) Embeddings ──────────────────────────────
    emb_dir = out / "embedding"
    print("[4/5] Computing embeddings ...")

    # Step-level: fps subsample → concat
    h_state_sub = resample.subsample_fps(h_state, args.subsample_stride)
    a_state_sub = resample.subsample_fps(a_state, args.subsample_stride)
    h_action_sub = resample.subsample_fps(h_action, args.subsample_stride)
    a_action_sub = resample.subsample_fps(a_action, args.subsample_stride)

    h_state_step = resample.concat_all_steps(h_state_sub)
    a_state_step = resample.concat_all_steps(a_state_sub)
    h_action_step = resample.concat_all_steps(h_action_sub)
    a_action_step = resample.concat_all_steps(a_action_sub)

    h_state_step_ids = _ep_ids(h_state_sub)
    a_state_step_ids = _ep_ids(a_state_sub, offset=10_000)
    h_action_step_ids = _ep_ids(h_action_sub)
    a_action_step_ids = _ep_ids(a_action_sub, offset=10_000)

    print("  step-level UMAP/t-SNE: state_rad ...")
    viz_embedding.step_level_embed_and_plot(
        h_state_step, a_state_step, h_state_step_ids, a_state_step_ids,
        emb_dir, "state_rad", max_points=args.max_step_points,
    )
    print("  step-level UMAP/t-SNE: action_rad ...")
    viz_embedding.step_level_embed_and_plot(
        h_action_step, a_action_step, h_action_step_ids, a_action_step_ids,
        emb_dir, "action_rad", max_points=args.max_step_points,
    )

    # Trajectory-level: fixed-length resample → flatten
    h_state_fx = resample.resample_fixed_length(h_state, args.target_length)
    a_state_fx = resample.resample_fixed_length(a_state, args.target_length)
    h_action_fx = resample.resample_fixed_length(h_action, args.target_length)
    a_action_fx = resample.resample_fixed_length(a_action, args.target_length)
    h_state_flat = resample.flatten_for_embedding(h_state_fx)
    a_state_flat = resample.flatten_for_embedding(a_state_fx)
    h_action_flat = resample.flatten_for_embedding(h_action_fx)
    a_action_flat = resample.flatten_for_embedding(a_action_fx)
    h_ep_ids = np.arange(len(h_state_fx))
    a_ep_ids = np.arange(len(a_state_fx)) + 10_000

    print("  trajectory-level UMAP/t-SNE: state_rad ...")
    traj_state_res = viz_embedding.trajectory_level_embed_and_plot(
        h_state_flat, a_state_flat, h_ep_ids, a_ep_ids,
        emb_dir, "state_rad",
    )
    print("  trajectory-level UMAP/t-SNE: action_rad ...")
    viz_embedding.trajectory_level_embed_and_plot(
        h_action_flat, a_action_flat, h_ep_ids, a_ep_ids,
        emb_dir, "action_rad",
    )

    # ─── 5) Metrics ──────────────────────────────
    print("[5/5] Computing metrics ...")
    human_m = metrics.summarize("human", h_state, h_action, h_xyz)
    auto_m = metrics.summarize("auto", a_state, a_action, a_xyz)

    # 두 분포 간 지표 (step-level, state_rad 기준으로 대표)
    # 계산 비용 고려해 서브샘플
    mmd = metrics.mmd_rbf(h_state_step, a_state_step)
    ed = metrics.energy_distance(h_state_step, a_state_step)

    # UMAP hull area ratio (trajectory-level state_rad)
    coords = traj_state_res["coords_umap"]
    lab = traj_state_res["labels"]
    hull_h = metrics.hull_area_2d(coords[lab == 0])
    hull_a = metrics.hull_area_2d(coords[lab == 1])
    hull_ratio = hull_h / hull_a if hull_a > 0 else float("inf")

    all_metrics = {
        "human_repo": args.human_repo,
        "auto_repo": args.auto_repo,
        "human": human_m,
        "auto": auto_m,
        "between": {
            "mmd_rbf_step_state_rad": mmd,
            "energy_distance_step_state_rad": ed,
            "hull_area_state_rad_umap": {
                "human": hull_h,
                "auto": hull_a,
                "ratio_human_over_auto": hull_ratio,
            },
        },
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # README 생성
    summary_for_readme = {
        "human_repo": args.human_repo,
        "auto_repo": args.auto_repo,
        "human": h_sum,
        "auto": a_sum,
        "subsample_stride": args.subsample_stride,
        "target_length": args.target_length,
        "max_points": args.max_step_points,
        "urdf": args.urdf,
    }
    generate_readme(out, summary_for_readme)

    print(f"\nDone → {out}")
    print(f"  MMD(rbf) = {mmd:.4f},  Energy distance = {ed:.4f}")
    print(f"  UMAP hull area: human={hull_h:.3f}, auto={hull_a:.3f}, ratio={hull_ratio:.2f}")


if __name__ == "__main__":
    main()
