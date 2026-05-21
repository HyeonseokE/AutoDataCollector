"""Integration smoke — DCT paradigm 의 3-way wiring + mini H3 검증.

목적:
  1. DB (P_phase1 DCT) load
  2. Shape 정합 (DB action_descriptor 300-dim ↔ candidate (1, 300))
  3. MI score 계산 정상 (ΔH_A, ΔH_{A|S}, M_MI)
  4. U_VLA forward 정상 (mode='dct', single-step)
  5. Phase2MISelector.select() ξ* 선택
  6. (bonus H3) GT / small perturbation / large perturbation 3 candidate 의
     U_VLA + M_MI ranking 이 perturbation 크기와 monotone?

같은 skill segment 의 GT raw action 을 base 로 noise 추가해 3개 candidate 생성.
state_key 와 obs (skill 시작 frame) 는 3개가 공유 (skill 시작점 obs 1 종류).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_LR = Path(__file__).resolve().parents[2] / "lerobot" / "src"
if str(_LR) not in sys.path:
    sys.path.insert(0, str(_LR))

from method3.dct.skill_dataset import load_dct_targets
from method3.dct.transform import traj_to_dct
from method3.phase2_mi_selection.mi_selector import (
    Phase2Candidate,
    Phase2MIConfig,
    Phase2MISelector,
)
from method3.phase2_mi_selection.vector_db import SkillVectorDB
from method3.phase2_mi_selection.vla_informativeness import (
    LeRobotVLAInformativenessScorer,
)
from method3.reembedding.lerobot_adapter import LeRobotPhase1RawAdapter
from method3.reembedding.pretrained_encoder import PretrainedVLAStateEncoder


CKPT = "/home/lerobot/AutoDataCollector/lerobot/outputs/train/smolvla_dct_20260521_155241/checkpoints/003750/pretrained_model"
DB_PATH = "/home/lerobot/AutoDataCollector/results/session_20260518_214931/dct/skill_wise_vector_db.npz"
# smoke 목적: paradigm 의 *코드 path* 정합 확인. train sidecar 사용해
# state_key 가 DB 의 그 entry 와 self-match → covered 보장 → U_VLA forward 도
# 실제로 호출되어 동작 검증. 실제 eval ↔ train distribution gap 측정은
# 별도 H3 단계로 분리.
EVAL_SIDECAR = "/home/lerobot/AutoDataCollector/results/skill_dct/pnp_phase1_30_table2.parquet"
DATASET = "CoRL2026-CSI/pnp_phase1_30_table2"

NOISE_SCALES = [0.0, 1.0, 10.0]   # joint-unit Gaussian std
LABELS = ["GT", "perturb_small", "perturb_large"]


def main(seg_index: int = 5) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    # ── load components ───────────────────────────────────────────────
    print(f"[smoke] loading DB → {DB_PATH}")
    db = SkillVectorDB()
    db.load(DB_PATH)
    print(f"  DB: {db.total_size()} skill traj entries, skills={sorted(db.skill_ids())}")

    print(f"[smoke] loading policy + processors from {CKPT}")
    policy = SmolVLAPolicy.from_pretrained(CKPT).to("cuda").eval()
    preprocessor = DataProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_preprocessor.json"
    )

    print(f"[smoke] loading VLA state encoder (for state_key)")
    encoder = PretrainedVLAStateEncoder(CKPT)

    print(f"[smoke] loading base dataset {DATASET}")
    base = LeRobotDataset(DATASET, video_backend="pyav")
    adapter = LeRobotPhase1RawAdapter(DATASET, action_horizon=2)

    # ── pick a test skill segment from eval sidecar ───────────────────
    segments = load_dct_targets(EVAL_SIDECAR)
    seg = segments[seg_index]
    T = seg.frame_end - seg.frame_start
    print(f"[smoke] test segment idx={seg_index}:")
    print(f"  episode={seg.episode_id}, skill_idx={seg.skill_index}, "
          f"skill_type={seg.skill_type}, T_skill={T} frames")

    # ── GT raw waypoints + skill-start obs/proprio ────────────────────
    gt_waypoints = np.stack([
        base[f]["action"].numpy() if hasattr(base[f]["action"], "numpy")
        else np.asarray(base[f]["action"])
        for f in range(seg.frame_start, seg.frame_end)
    ])
    print(f"  GT waypoints: shape={gt_waypoints.shape}, "
          f"|range|=[{gt_waypoints.min():.2f}, {gt_waypoints.max():.2f}]")

    ref = {
        "dataset_path": adapter._dataset_path,
        "global_idx": int(seg.frame_start),
        "key": adapter._observation_key,
    }
    obs_raw = adapter.load_observation(ref)
    frame_start = base[seg.frame_start]
    proprio = adapter._extract_proprio(frame_start)

    # ── state_key via DCT-tuned VLA encoder ───────────────────────────
    e_vla = np.asarray(encoder.encode(obs_raw, seg.instruction),
                       dtype=np.float64).reshape(-1)
    state_key = np.concatenate([e_vla, proprio])  # (966,)
    db_state_dim = db.state_keys(seg.skill_type).shape[1]
    assert state_key.shape[0] == db_state_dim, (
        f"state_key dim mismatch: smoke={state_key.shape[0]} vs DB={db_state_dim}"
    )
    print(f"  state_key dim={state_key.shape[0]} ✓ matches DB")

    # ── pre-processed batch (shared by 3 candidates) ──────────────────
    rename = {
        "observation.images.left_wrist": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
    }
    item = dict(frame_start)
    item["task"] = f"{seg.skill_type}: {seg.instruction}"
    item = {rename.get(k, k): v for k, v in item.items()}
    batch_input: dict = {}
    for k, v in item.items():
        if k.startswith("_"):
            continue
        if isinstance(v, torch.Tensor):
            batch_input[k] = v.unsqueeze(0).to("cuda")
        elif isinstance(v, str):
            batch_input[k] = [v]
        else:
            batch_input[k] = v
    batch_input = preprocessor(batch_input)

    # ── 3 candidates: GT / small / large perturbation ─────────────────
    rng = np.random.default_rng(0)
    candidates = []
    for label, scale in zip(LABELS, NOISE_SCALES):
        noise = rng.normal(0, scale, size=gt_waypoints.shape) if scale > 0 else 0.0
        wp = gt_waypoints + noise
        dct = traj_to_dct(wp, L0=50)  # (50, 6)
        # mock action_chunks (use_dct_target=True 시 무시되지만 schema-필수)
        H = 50
        if len(wp) >= H:
            action_chunks = wp[:H][None, ...]
        else:
            pad_n = H - len(wp)
            ac = np.concatenate([wp, np.tile(wp[-1:, :], (pad_n, 1))], axis=0)
            action_chunks = ac[None, ...]
        cand = Phase2Candidate(
            skill_id=seg.skill_type,
            state_keys=state_key[None, :],         # (1, 966)
            action_chunks=action_chunks,
            observations=batch_input,
            instruction=seg.instruction,
            proprios=proprio[None, :],
            dct_target=dct,
        )
        candidates.append(cand)
        print(f"  cand[{label:14s}]: noise_std={scale:>4.1f}  "
              f"dct |max|={np.abs(dct).max():>7.2f}")

    # ── scorer + selector ────────────────────────────────────────────
    scorer = LeRobotVLAInformativenessScorer(
        policy=policy,
        batch_builder=lambda c: c.observations,
        mode="dct",
        R=1,
        sigma=None,
    )
    selector = Phase2MISelector(
        vector_db=db,
        config=Phase2MIConfig(
            use_dct_target=True,
            k_nn_a=3,
            k_min=1,
            min_covered_windows=1,
            radius_k=5,
            radius_quantile=0.9,    # smoke 에서 covered 보장
        ),
    )

    # ── run selection ────────────────────────────────────────────────
    print(f"\n[smoke] running Phase2MISelector.select() ...")
    t0 = time.monotonic()
    sel = selector.select(candidates, vla_scorer=scorer)
    elapsed = time.monotonic() - t0
    print(f"[smoke] selection done in {elapsed:.2f}s")

    # ── report ───────────────────────────────────────────────────────
    print(f"\n=== Selection result ===")
    print(f"chosen_index: {sel.chosen_index}  (label={LABELS[sel.chosen_index]})")
    print(f"accepted:     {sel.accepted}")
    print(f"eligible:     {sel.eligible_indices}  → {[LABELS[i] for i in sel.eligible_indices]}")
    print(f"u_vla_chosen: {sel.u_vla_chosen}")
    print(f"\nper-candidate reports:")
    print(f"  {'label':14s} {'ΔH_A':>10s} {'ΔH_A|S':>10s} {'M_MI':>10s} "
          f"{'M̃_MI':>8s} {'U_VLA':>10s} {'cov':>6s} {'under':>6s}")
    for r in sel.reports:
        print(f"  {LABELS[r.candidate_index]:14s} "
              f"{r.delta_h_a:>10.4f} "
              f"{r.delta_h_a_given_s:>10.4f} "
              f"{r.q2:>10.4f} "
              f"{r.q2_norm:>+8.3f} "
              f"{r.u_vla:>10.4f} "
              f"{r.covered_ratio:>6.2f} "
              f"{str(r.under_covered):>6s}")

    # ── Useful-OOD rule sanity (paradigm §13.2) ──────────────────────
    # 의도: redundant (GT 와 동일) 와 harmful OOD (큰 perturb) 모두 배제,
    # *적당한 novelty 의 후보* 만 ξ* 로 선택. monotone U_VLA 가 아님 —
    # M̃_MI ≥ τ_MI eligible filter 안에서 argmax U_VLA 라는 *조건부 ranking*.
    deltas_a = [r.delta_h_a for r in sel.reports]
    deltas_as = [r.delta_h_a_given_s for r in sel.reports]
    m_mi = [r.q2 for r in sel.reports]
    print(f"\n=== Useful-OOD rule sanity ===")
    print(f"ΔH_A    (novelty)   GT→small→large : {[f'{v:.3f}' for v in deltas_a]}")
    print(f"ΔH_A|S  (ambiguity) GT→small→large : {[f'{v:.3f}' for v in deltas_as]}")
    monotone_dh = (deltas_a[0] <= deltas_a[1] <= deltas_a[2]
                   and deltas_as[0] <= deltas_as[1] <= deltas_as[2])
    print(f"perturbation 클수록 novelty + ambiguity 둘 다 증가? {monotone_dh}")
    print(f"\nM_MI = β·ΔH_A − λ·ΔH_A|S, GT→small→large: "
          f"{[f'{v:.3f}' for v in m_mi]}")
    print(f"eligible (M̃_MI ≥ τ_MI=0): {sel.eligible_indices} → "
          f"{[LABELS[i] for i in sel.eligible_indices]}")
    print(f"chosen = argmax U_VLA in eligible → {LABELS[sel.chosen_index]}")
    print(f"\n→ paradigm 의도: redundant (GT) 와 harmful OOD (large) 배제, "
          f"useful (small) 선택  ⇒  "
          f"{'✓ 정확히 동작' if LABELS[sel.chosen_index] == 'perturb_small' else '✗ 의도 불일치'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
