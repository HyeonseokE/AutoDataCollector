"""Phase1 raw dataset re-embedding → Phase1 seed vector DB — 문서 §6.

§6 Step 3-4: Phase1 종료 후 Phase1 raw dataset 을 다시 읽어, frozen VLA encoder
로 re-embedding 하여 full skill-wise vector DB seed ``P_phase1^(m)`` 를 만든다.

  e_i^vla = φ_VLA^(1)(o_i, I_i)
  e_i     = [e_i^vla; p_i]                    — state retrieval key (§7.3)
  z_i^a   = ψ(A_{i:i+H-1})                    — DCT action descriptor (§4.2)
  P_phase1^(m) = {(e_i, z_i^a, ref_i, meta_i)}_{i∈D_phase1^(m)}

observation ``o_i`` 는 raw dataset 에 pointer(``observation_ref``)로만 있으므로
(§3 / §3 details §4), caller 가 그 pointer 를 실제 observation 배열로 푸는
``observation_loader`` 를 넘긴다.

§6 Step 1(VLA 학습)·Step 2(freeze)는 외부 ML job 이다 — 이 builder 는 이미
학습·freeze 된 encoder 를 받아 Step 3-4 만 수행한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from method3.phase2_mi_selection.action_descriptor import dct_action_descriptor
from method3.phase2_mi_selection.vector_db import SkillVectorDB, VectorDBEntry
from method3.reembedding.vla_encoder import VLAStateEncoder
from method3.storage.raw_dataset import RawTrajectoryDataset


@dataclass
class ReembeddingConfig:
    """re-embedding 파라미터 (문서 §6)."""

    dct_coeffs: int = 3          # §4.2 K — DCT action descriptor 저주파 성분 수
    skip_invalid: bool = False   # True → validity_flag=False entry 를 건너뜀
                                 # (기본 False — §6 은 dataset 전체 re-embed)
    show_progress: bool = True   # tqdm progress bar (False → 무음)
    frame_stride: int = 1        # 매 frame_stride 번째 entry 만 re-embed.
                                 # 1 = spec 정석 (전체, 기본). 5~10 = dense
                                 # temporal sampling redundancy 활용 wall-clock
                                 # speedup. 정석을 원하면 그대로 1 유지.
    batch_size: int = 32         # GPU batched VLA forward 배치. encoder 가
                                 # ``encode_batch`` 를 지원하면 N개 한 번에
                                 # forward → GPU 활용도 N배. 1 = single-loop
                                 # legacy 경로.
    subgoal_filter_radius_m: float | None = None
                                 # subgoal-aware filter (옵션 2). ee_pos 와
                                 # subtask.target_position 의 L2 distance 가
                                 # 이 반경 안의 frame 만 re-embed. None 이면
                                 # 비활성 (=spec 정석 전체 re-embed).
                                 # 권장 0.10 (= 10 cm), task tolerance.
    subgoal_filter_min_keep: int = 30
                                 # filter 후 frame 수가 이보다 적으면 radius
                                 # 자동 확장 (× 1.5 까지 최대 3회 retry).


def _apply_subgoal_filter(
    raw_dataset, indices: list[int], cfg: ReembeddingConfig,
    g_seed_buffer=None,
) -> list[int]:
    """ee_xyz 가 *모든 subgoal anchor 의 union ± R* 안에 있는 frame 만 keep.
    no video decode (parquet bulk + buffer cheap read).

    Subgoal pool 우선순위:
      1) ``g_seed_buffer`` (Phase1 누적 G_seed) — spec 의 진짜 Phase2 anchor
      2) dataset column ``skill.goal_position.robot_xyzrpy`` — fallback
      3) dataset column ``subtask.target_position`` — last fallback
    """
    import numpy as np
    ds = getattr(raw_dataset, "_dataset", None)
    if ds is None or not hasattr(ds, "hf_dataset"):
        print("[reembed] subgoal-filter unavailable (no hf_dataset) — full re-embed")
        return indices
    hf = ds.hf_dataset
    ee_key = getattr(raw_dataset, "_proprio_key", None) or "observation.ee_pos.robot_xyzrpy"
    if ee_key not in hf.features:
        print(f"[reembed] subgoal-filter unavailable (missing {ee_key}) — full re-embed")
        return indices
    ee_all = np.asarray(hf[ee_key], dtype=np.float64)[:, :3]   # (N_full, 3)

    # ── subgoal pool 결정 ────────────────────────────────────────────────
    sg_unique = None
    sg_source = None
    if g_seed_buffer is not None and g_seed_buffer.total_size() > 0:
        pool = [g_seed_buffer.seed_subgoals(s) for s in g_seed_buffer.skill_ids()]
        pool = [p for p in pool if p.size > 0]
        if pool:
            sg_unique = np.unique(np.concatenate(pool, axis=0), axis=0)
            sg_source = f"G_seed (skills={g_seed_buffer.skill_ids()})"
    if sg_unique is None:
        # fallback to dataset column
        for cand in ("skill.goal_position.robot_xyzrpy", "subtask.target_position"):
            if cand in hf.features:
                col = np.asarray(hf[cand], dtype=np.float64)[:, :3]
                sg_unique = np.unique(col, axis=0)
                sg_source = f"column={cand}"
                break
        if sg_unique is None:
            print("[reembed] subgoal-filter unavailable (no G_seed + no fallback column) — full re-embed")
            return indices
    # 각 indices 의 ee 가 *어느 한 subgoal* 의 R 안에 있나 — broadcasting
    gidx = np.asarray([raw_dataset._index[i][0] for i in indices], dtype=int)
    ee = ee_all[gidx]                                            # (N_sel, 3)
    # pairwise distance ee[N_sel,1,:] - sg_unique[1,N_sg,:] → norm axis=-1 → (N_sel, N_sg)
    dmat = np.linalg.norm(ee[:, None, :] - sg_unique[None, :, :], axis=-1)
    min_dist = dmat.min(axis=1)                                  # (N_sel,) — 가장 가까운 subgoal
    radius = float(cfg.subgoal_filter_radius_m)
    min_keep = int(cfg.subgoal_filter_min_keep)
    mask = min_dist < radius
    for _ in range(3):
        if int(mask.sum()) >= min_keep:
            break
        new_r = radius * 1.5
        print(f"[reembed] subgoal-filter R={radius:.3f}m kept {int(mask.sum())} (< {min_keep}); "
              f"expanding to {new_r:.3f}m")
        radius = new_r
        mask = min_dist < radius
    kept = [i for i, k in zip(indices, mask.tolist()) if k]
    print(f"[reembed] subgoal-filter (bulk-pooled): {len(indices)} → {len(kept)} frames "
          f"(R={radius:.3f}m, source={sg_source}, |sg_unique|={len(sg_unique)}, "
          f"min_dist∈[{min_dist.min():.3f}, {min_dist.max():.3f}], median={np.median(min_dist):.3f})")
    return kept


def _fallback_progress(iterable, total: int, label: str):
    """tqdm 미설치 시 ~5% 단위로 줄바꿈 없이 캐리지리턴 카운터만 찍는다."""
    import sys
    import time
    step = max(1, total // 20)
    start = time.monotonic()
    for i, x in enumerate(iterable):
        yield x
        done = i + 1
        if done == total or done % step == 0:
            pct = 100.0 * done / max(1, total)
            elapsed = time.monotonic() - start
            rate = done / max(1e-6, elapsed)
            eta = (total - done) / max(1e-6, rate)
            sys.stdout.write(
                f"\r[reembed][{label}] {done}/{total} ({pct:5.1f}%) "
                f"{rate:.1f} frame/s  ETA {eta:5.1f}s"
            )
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def state_retrieval_key(
    vla_embedding: np.ndarray,
    proprioception: np.ndarray,
) -> np.ndarray:
    """``e_i = [e_i^vla; p_i]`` — state retrieval key (문서 §7.3).

    proprioception 은 VLA encoder 입력이 아니라 embedding 뒤에 concat 된다.
    """
    return np.concatenate([
        np.asarray(vla_embedding, dtype=np.float64).reshape(-1),
        np.asarray(proprioception, dtype=np.float64).reshape(-1),
    ])


def build_phase1_vector_db(
    raw_dataset: RawTrajectoryDataset,
    encoder: VLAStateEncoder,
    observation_loader: Callable[[dict], np.ndarray],
    config: ReembeddingConfig | None = None,
    g_seed_buffer=None,
) -> SkillVectorDB:
    """Phase1 raw dataset 을 re-embedding 하여 ``P_phase1`` vector DB 를 만든다 (§6).

    Args:
        raw_dataset: Phase1 raw trajectory dataset ``D_phase1_raw``.
        encoder: Phase1-trained·frozen VLA state encoder ``φ_VLA^(1)``.
            Phase2 candidate key 추출에도 **같은 instance** 를 써야 한다 (§6).
        observation_loader: ``observation_ref`` (dict) → observation 배열 resolver.
            raw dataset 은 §4 pointer 규약상 observation 을 ref 로만 들고 있다.
        config: re-embedding 파라미터.

    Returns:
        skill-wise 로 partition 된 Phase1 seed vector DB ``P_phase1``.
    """
    cfg = config or ReembeddingConfig()
    db = SkillVectorDB()
    stride = max(1, int(cfg.frame_stride))
    N_full = len(raw_dataset)
    indices = list(range(0, N_full, stride))
    if stride > 1:
        print(f"[reembed] frame_stride={stride} → 처리할 entry {N_full} → {len(indices)} "
              f"({100.0 * len(indices) / max(1, N_full):.1f}%)")
    # subgoal-aware filter (옵션 2) — yaml.reembedding.subgoal_filter_radius_m 으로
    # 활성화. parquet bulk pre-pass + G_seed buffer (있으면 우선) 로 ee 가 어느
    # subgoal anchor R 안에 있는 frame 만 keep. *video decode 없이* 가속.
    if cfg.subgoal_filter_radius_m is not None:
        indices = _apply_subgoal_filter(raw_dataset, indices, cfg, g_seed_buffer=g_seed_buffer)
    N = len(indices)

    batch_size = max(1, int(cfg.batch_size))
    use_batch = batch_size > 1 and hasattr(encoder, "encode_batch")
    if use_batch:
        print(f"[reembed] GPU batched forward — batch_size={batch_size} "
              f"(encoder={type(encoder).__name__}.encode_batch)")
    else:
        if batch_size > 1:
            print(f"[reembed] batch_size={batch_size} 요청됐으나 encoder 가 "
                  f"encode_batch 미지원 → single-loop fallback")
        batch_size = 1

    # progress bar 는 *frame 단위* 로 카운트한다 (batch 안에서도 frame 별 append).
    if cfg.show_progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=N, desc="[reembed] skill-wise DB build",
                        unit="frame", dynamic_ncols=True)
        except ImportError:
            pbar = None
    else:
        pbar = None

    # chunked outer loop. batch_size=1 이면 기존 single-loop 와 동등.
    processed = 0
    for chunk_start in range(0, N, batch_size):
        chunk_indices = indices[chunk_start: chunk_start + batch_size]
        chunk_entries = [raw_dataset.get(i) for i in chunk_indices]

        # skip_invalid 필터 — chunk 안에서 invalid 만 제외 (단순 in-place 필터)
        if cfg.skip_invalid:
            keep = [(i, e) for i, e in zip(chunk_indices, chunk_entries) if e.validity_flag]
            if not keep:
                if pbar:
                    pbar.update(len(chunk_indices))
                processed += len(chunk_indices)
                continue
            chunk_indices = [i for i, _ in keep]
            chunk_entries = [e for _, e in keep]

        observations = [observation_loader(e.observation_ref) for e in chunk_entries]
        instructions = [e.instruction for e in chunk_entries]

        # §6 — VLA encoder. batch path 가 있으면 한 번에, 없으면 frame 별 loop.
        if use_batch and len(observations) > 1:
            e_vla_batch = encoder.encode_batch(observations, instructions)   # (B, D_vla)
        else:
            e_vla_batch = np.stack([
                encoder.encode(o, i) for o, i in zip(observations, instructions)
            ])

        # frame 단위 post-processing (CPU). DCT + state concat + db.append.
        for j, (idx, entry) in enumerate(zip(chunk_indices, chunk_entries)):
            e_vla = np.asarray(e_vla_batch[j], dtype=np.float64).reshape(-1)
            e_i = state_retrieval_key(e_vla, entry.proprioception)           # §7.3
            z_i = dct_action_descriptor(entry.action_chunk, cfg.dct_coeffs)  # §4.2
            db.append(VectorDBEntry(
                skill_id=entry.skill_id,
                state_key=e_i,
                action_descriptor=z_i,
                ref=raw_dataset.pointer(idx),                                # §3.1
                meta={
                    "phase": "phase1",
                    "skill_id": entry.skill_id,
                    "subgoal": np.asarray(entry.subgoal, dtype=float).tolist(),
                    "planner_type": entry.planner_type,
                    "instruction": entry.instruction,
                    "time_index": int(entry.time_index),
                    "accepted_by": "phase1_seed",
                },
            ))

        if pbar:
            pbar.update(len(chunk_indices))
        processed += len(chunk_indices)

    if pbar:
        pbar.close()
    return db
