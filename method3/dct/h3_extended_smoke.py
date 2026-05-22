"""확장 H3 smoke — 12 candidate (8 noise + 4 sibling) 중 useful 선택 검증.

목적: paradigm 의 Useful-OOD rule 이 *더 큰 candidate set + 다양한
diversity 축* 에서도 "GT 와 비슷하지만 약간 다른" 후보를 선택하는지 확인.

candidate 구성:
  Axis A — noise spectrum (8):
    GT, very_small(0.3), small(0.7), mid(1.5), mid_large(3.0),
    large(6.0), very_large(12.0), extreme(25.0)   # joint-unit Gaussian std

  Axis B — sibling segments (4):
    같은 skill_type 의 *다른 episode 의 segment* 의 raw waypoints.
    natural diversity (random noise 아닌 실 trajectory variation).

모든 candidate 의 state_key/obs 는 base segment 의 skill 시작 frame 공유
(paradigm 의 "한 시점 에서 K 후보 비교" 가정).
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

from method3.dct.skill_dataset import SkillSegment, load_dct_targets
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
TRAIN_SIDECAR = "/home/lerobot/AutoDataCollector/results/skill_dct/pnp_phase1_30_table2.parquet"
DATASET = "CoRL2026-CSI/pnp_phase1_30_table2"

NOISE_SPECTRUM = [0.0, 0.3, 0.7, 1.5, 3.0, 6.0, 12.0, 25.0]
NOISE_LABELS = [
    "GT", "very_small(0.3)", "small(0.7)", "mid(1.5)",
    "mid_large(3.0)", "large(6.0)", "very_large(12.0)", "extreme(25.0)",
]
N_SIBLINGS = 4


def _extract_raw(base, f0: int, f1: int) -> np.ndarray:
    return np.stack([
        base[f]["action"].numpy() if hasattr(base[f]["action"], "numpy")
        else np.asarray(base[f]["action"])
        for f in range(f0, f1)
    ])


def _make_cand(seg: SkillSegment, state_key: np.ndarray, proprio: np.ndarray,
               batch: dict, wp: np.ndarray) -> Phase2Candidate:
    dct = traj_to_dct(wp, L0=50)
    H = 50
    if len(wp) >= H:
        ac = wp[:H][None, ...]
    else:
        pad = np.tile(wp[-1:, :], (H - len(wp), 1))
        ac = np.concatenate([wp, pad], axis=0)[None, ...]
    return Phase2Candidate(
        skill_id=seg.skill_type,
        state_keys=state_key[None, :],
        action_chunks=ac,
        observations=batch,
        instruction=seg.instruction,
        proprios=proprio[None, :],
        dct_target=dct,
    )


def main(seg_index: int = 5) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    print(f"[h3] loading components...")
    db = SkillVectorDB()
    db.load(DB_PATH)
    policy = SmolVLAPolicy.from_pretrained(CKPT).to("cuda").eval()
    preprocessor = DataProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_preprocessor.json"
    )
    encoder = PretrainedVLAStateEncoder(CKPT)
    base = LeRobotDataset(DATASET, video_backend="pyav")
    adapter = LeRobotPhase1RawAdapter(DATASET, action_horizon=2)

    # base segment + siblings (same skill_type, different episode)
    segments = load_dct_targets(TRAIN_SIDECAR)
    base_seg = segments[seg_index]
    T = base_seg.frame_end - base_seg.frame_start
    siblings = [
        s for s in segments
        if s.skill_type == base_seg.skill_type
        and s.episode_id != base_seg.episode_id
    ][:N_SIBLINGS]
    print(f"[h3] base segment: ep={base_seg.episode_id}, "
          f"skill={base_seg.skill_type}, T={T}")
    print(f"[h3] siblings ({len(siblings)}): "
          f"{[(s.episode_id, s.skill_index, s.frame_end - s.frame_start) for s in siblings]}")

    gt_waypoints = _extract_raw(base, base_seg.frame_start, base_seg.frame_end)

    # shared obs / proprio / state_key (skill 시작 frame)
    ref = {
        "dataset_path": adapter._dataset_path,
        "global_idx": int(base_seg.frame_start),
        "key": adapter._observation_key,
    }
    obs_raw = adapter.load_observation(ref)
    frame_start = base[base_seg.frame_start]
    proprio = adapter._extract_proprio(frame_start)
    e_vla = np.asarray(
        encoder.encode(obs_raw, base_seg.instruction), dtype=np.float64
    ).reshape(-1)
    state_key = np.concatenate([e_vla, proprio])

    # shared pre-processed batch
    rename = {
        "observation.images.left_wrist": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
    }
    item = dict(frame_start)
    item["task"] = f"{base_seg.skill_type}: {base_seg.instruction}"
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

    # ── build candidates ─────────────────────────────────────────────
    rng = np.random.default_rng(42)
    cands: list[Phase2Candidate] = []
    labels: list[str] = []

    # Axis A — noise spectrum
    for label, scale in zip(NOISE_LABELS, NOISE_SPECTRUM):
        if scale > 0:
            wp = gt_waypoints + rng.normal(0, scale, gt_waypoints.shape)
        else:
            wp = gt_waypoints.copy()
        labels.append(label)
        cands.append(_make_cand(base_seg, state_key, proprio, batch_input, wp))

    # Axis B — sibling segments
    for sib in siblings:
        wp = _extract_raw(base, sib.frame_start, sib.frame_end)
        labels.append(f"sib[{sib.episode_id}/sk{sib.skill_index}]")
        cands.append(_make_cand(base_seg, state_key, proprio, batch_input, wp))

    print(f"[h3] {len(cands)} candidates total ({len(NOISE_LABELS)} noise + {len(siblings)} siblings)")

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
            k_nn_a=3,
            k_min=1,
            min_covered_windows=1,
            radius_k=5,
            radius_quantile=0.9,
            beta=1.0,
            lambda_=3.0,        # ambiguity penalty 강화 — 큰 OOD 자연 cut
        ),
    )

    t0 = time.monotonic()
    sel = selector.select(cands, vla_scorer=scorer)
    elapsed = time.monotonic() - t0
    print(f"[h3] selection done in {elapsed:.2f}s")

    # ── report ───────────────────────────────────────────────────────
    eligible_set = set(sel.eligible_indices)
    print(f"\n=== Selection result — base skill={base_seg.skill_type} ===")
    print(f"chosen_index: {sel.chosen_index} ({labels[sel.chosen_index]})")
    print(f"accepted:     {sel.accepted}")
    print(f"eligible:     {len(sel.eligible_indices)}/{len(cands)} → "
          f"{[labels[i] for i in sel.eligible_indices]}")
    print(f"u_vla_chosen: {sel.u_vla_chosen}")
    print()
    hdr = f"{'#':>2s} {'label':25s} {'ΔH_A':>9s} {'ΔH_A|S':>10s} {'M_MI':>9s} {'M̃_MI':>8s} {'U_VLA':>10s} {'elig':>5s} {'★':>2s}"
    print(hdr)
    print("-" * len(hdr))
    for r in sel.reports:
        star = "★" if r.candidate_index == sel.chosen_index else ""
        elig = "Y" if r.candidate_index in eligible_set else "N"
        print(f"{r.candidate_index:>2d} "
              f"{labels[r.candidate_index]:25s} "
              f"{r.delta_h_a:>9.4f} "
              f"{r.delta_h_a_given_s:>10.4f} "
              f"{r.q2:>9.4f} "
              f"{r.q2_norm:>+8.3f} "
              f"{r.u_vla:>10.4f} "
              f"{elig:>5s} "
              f"{star:>2s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
