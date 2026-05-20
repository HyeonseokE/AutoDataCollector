"""Server-side setup helper for method3 phase2 useful-OOD acquisition.

Replaces the old ``preselective_filter.integration.setup_preselective_filter``
(IG·AC + FaissBufferStore) with the new spec
``final_method3_spec_useful_ood_updated §11-§13`` server-side stack:

  - SkillVectorDB                        — accumulating vector DB (§14)
  - Phase2MISelector + Phase2MIConfig    — M_MI computation + Useful-OOD rule
  - LeRobotVLAInformativenessScorer (옵셔널) — U_VLA (§12), if vla_informativeness.enabled
  - VLAKeyExtractor (preselective_filter.vectorDB.vla_embedding) — shared encoder

The encoder is still the frozen VLA pulled in via ``make_vla_key_extractor``
(pi0 / pi05 / smolvla / groot auto-dispatch) — that path is intentionally
preserved because it is the *only* piece of preselective_filter that survives
the refactor (everything else is method3-native).

Persistence model — the server-local SkillVectorDB lives in a sidecar npz file
so the server can be restarted without losing the buffer (옛 FaissBufferStore
의 server-local persistence 와 동등). On first boot the file does not exist
→ a fresh empty DB.

Returns ``Method3ServerStack`` so server.py just unpacks one bundle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from method3.phase2_mi_selection import (
    LeRobotVLAInformativenessScorer,
    Phase2MIConfig,
    Phase2MISelector,
    SkillVectorDB,
)
from method3.config import load_phase2_config


@dataclass
class Method3ServerStack:
    """Bundle returned by ``setup_method3_phase2_server``."""

    selector: Phase2MISelector
    encoder: Any                     # VLAKeyExtractor — frozen VLA, share-able
    vla_scorer: Optional[LeRobotVLAInformativenessScorer]  # None if disabled
    db: SkillVectorDB
    db_path: Path                    # sidecar npz for persistence
    config: Phase2MIConfig           # for Ready() display
    encoder_lock_payload: dict       # passthrough metadata for debug


def _load_phase2_yaml(recording_cfg: dict, phase2_yaml_path: Optional[Path]) -> dict:
    """Read phase2_config.yaml (raw dict). recording_cfg 의 ``skill_planner_transport``
    / ``preselective_filter`` legacy block 은 무시 — 새 acquisition 설정은 phase2 쪽."""
    if phase2_yaml_path is None or not phase2_yaml_path.exists():
        return {}
    try:
        with open(phase2_yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[method3_setup] failed to read {phase2_yaml_path}: {e}")
        return {}


def setup_method3_phase2_server(
    recording_cfg: dict,
    *,
    phase2_yaml: Optional[str | Path] = None,
    db_save_dir: Optional[str | Path] = None,
    db_filename: str = "server_skill_wise_vector_db.npz",
) -> Optional[Method3ServerStack]:
    """Build the method3-phase2 server-side stack from yaml.

    Args:
        recording_cfg: recording_config_ws*.yaml 의 dict (encoder checkpoint 등
            기존 preselective_filter 영역에서 가져오던 키 — 호환 위해 유지).
        phase2_yaml: ``pipeline_config/phase2_config.yaml`` 경로 (mi_selection +
            vla_informativeness + skill_planner_transport 블록 source).
            None 이면 server 호스트의 기본 위치 시도.
        db_save_dir: server-local SkillVectorDB persistence 경로 root. None 이면
            ``./preselective_rpc/buffer/`` 사용.
        db_filename: 캐시 파일명.

    Returns:
        Method3ServerStack 또는 None (preselective_filter / encoder 가 yaml 에서
        비활성화돼 있으면).
    """
    # encoder yaml block — recording_config 의 preselective_filter.policy 에 있음
    # (옛 IG·AC 와 같은 위치 공유 — 호환 보존).
    psf_raw = recording_cfg.get("preselective_filter") or {}
    if not psf_raw:
        # 새 yaml convention 시도: phase2_yaml 안의 skill_planner_transport / policy
        ph2_raw = _load_phase2_yaml(recording_cfg, Path(phase2_yaml) if phase2_yaml else None)
        psf_raw = (
            ph2_raw.get("skill_planner_transport")
            or ph2_raw.get("preselective_filter")
            or {}
        )
    if not psf_raw:
        return None

    policy_cfg = psf_raw.get("policy") or {}
    ckpt = policy_cfg.get("checkpoint")
    if not ckpt:
        print("[method3_setup] no VLA checkpoint specified — disabling")
        return None

    # 1) VLA encoder (frozen) — shared dep, encode + (옵션) denoise loss
    from preselective_filter.vectorDB.vla_embedding import make_vla_key_extractor
    encoder = make_vla_key_extractor(
        checkpoint=str(ckpt),
        device=str(policy_cfg.get("device", "cuda")),
        autocast_dtype=str(policy_cfg.get("autocast_dtype", "bfloat16")),
        debug_verbose=bool(psf_raw.get("debug_verbose", False)),
        family=policy_cfg.get("family"),
    )

    # 2) Phase2MISelector + SkillVectorDB (with persistence)
    if phase2_yaml is None:
        phase2_yaml = Path(__file__).resolve().parent.parent / "pipeline_config" / "phase2_config.yaml"
    phase2_yaml = Path(phase2_yaml)
    try:
        acq = load_phase2_config(phase2_yaml, phase1_raw_dir="", phase2_raw_dir="")
        phase2_mi = acq.phase2_mi
    except Exception as e:
        print(f"[method3_setup] phase2_config.yaml load failed ({e}); using defaults")
        phase2_mi = Phase2MIConfig()

    db = SkillVectorDB()
    if db_save_dir is None:
        db_save_dir = Path(__file__).resolve().parent / "buffer"
    db_save_dir = Path(db_save_dir)
    db_save_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_save_dir / db_filename
    if db_path.exists():
        try:
            db.load(db_path)
            print(f"[method3_setup] loaded server-side DB ← {db_path} "
                  f"({db.total_size()} entries over {len(db.skill_ids())} skills)")
        except Exception as e:
            print(f"[method3_setup] DB load failed ({e}); starting fresh")
            db = SkillVectorDB()

    selector = Phase2MISelector(vector_db=db, config=phase2_mi)

    # 3) VLA-side informativeness (옵션) — yaml.vla_informativeness.enabled 일 때
    vla_scorer = None
    ph2_raw = _load_phase2_yaml(recording_cfg, phase2_yaml)
    vla_cfg = ph2_raw.get("vla_informativeness") or {}
    if vla_cfg.get("enabled", False):
        try:
            vla_scorer = LeRobotVLAInformativenessScorer(
                policy=encoder.policy,
                R=int(vla_cfg.get("R", 8)),
                agg=str(vla_cfg.get("agg", "mean")),
            )
            print(f"[method3_setup] U_VLA scorer ENABLED (R={vla_scorer.R})")
        except Exception as e:
            print(f"[method3_setup] U_VLA setup failed ({e}); selection falls back to argmax M_MI")
            vla_scorer = None

    print(
        f"[method3_setup] Phase2MISelector ready — tau_MI={phase2_mi.tau_MI}, "
        f"k_nn_a={phase2_mi.k_nn_a}, vla_scorer={'on' if vla_scorer else 'off'}"
    )
    return Method3ServerStack(
        selector=selector,
        encoder=encoder,
        vla_scorer=vla_scorer,
        db=db,
        db_path=db_path,
        config=phase2_mi,
        encoder_lock_payload={"checkpoint": str(ckpt)},
    )
