"""Method3 re-embedding — unit tests (문서 final_method3_spec §6).

VLA encoder stub 와 Phase1 seed vector DB 구축(re-embedding)을 격리 검증한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from method3.phase2_mi_selection.vector_db import SkillVectorDB
from method3.reembedding.seed_builder import (
    ReembeddingConfig,
    build_phase1_vector_db,
    state_retrieval_key,
)
from method3.reembedding.vla_encoder import MeanPoolStateEncoder, VLAStateEncoder
from method3.storage.raw_dataset import RawDatasetEntry, RawTrajectoryDataset, resolve_pointer


def _raw_entry(skill="reach", episode="ep_0", t=0, H=12, A=6, P=7, valid=True):
    """§3.1 RawDatasetEntry — re-embedding 입력용 합성 entry."""
    rng = np.random.default_rng(t + 1)
    return RawDatasetEntry(
        episode_id=episode,
        phase="phase1",
        skill_id=skill,
        instruction=f"do {skill}",
        subgoal=np.array([0.3, 0.0, 0.2]) + t * 0.01,
        time_index=t,
        observation_ref={"frame": t, "episode": episode},
        proprioception=rng.normal(0, 1, P),
        action_chunk=rng.normal(0, 1, (H, A)),
        raw_action_sequence=rng.normal(0, 1, (H, A)),
        planner_type="InterpPlan",
        success_flag=True,
        validity_flag=valid,
    )


def _obs_loader(ref: dict) -> np.ndarray:
    """observation_ref → 결정적 observation 배열 (테스트용 resolver)."""
    return np.full((3, 4), float(ref["frame"]))


# ─────────────────────────────────────────────────────────────
# §6  VLA state encoder (stub)
# ─────────────────────────────────────────────────────────────
class TestMeanPoolStateEncoder:
    def test_output_dim_is_out_dim(self):
        enc = MeanPoolStateEncoder(out_dim=16)
        e = enc.encode(np.random.default_rng(0).normal(0, 1, (8, 8)), "pick")
        assert e.shape == (16,)

    def test_encode_is_deterministic(self):
        enc = MeanPoolStateEncoder(out_dim=8)
        obs = np.arange(40, dtype=float).reshape(5, 8)
        np.testing.assert_array_equal(enc.encode(obs, "pick"), enc.encode(obs, "pick"))

    def test_different_instruction_changes_embedding(self):
        enc = MeanPoolStateEncoder(out_dim=8)
        obs = np.ones((4, 4))
        assert not np.allclose(enc.encode(obs, "pick the cube"),
                               enc.encode(obs, "place the cube"))

    def test_small_observation_padded_no_nan(self):
        enc = MeanPoolStateEncoder(out_dim=16)
        e = enc.encode(np.array([1.0, 2.0, 3.0]), "pick")    # size 3 < out_dim
        assert e.shape == (16,)
        assert np.all(np.isfinite(e))

    def test_empty_observation_handled(self):
        enc = MeanPoolStateEncoder(out_dim=8)
        e = enc.encode(np.array([]), "pick")
        assert e.shape == (8,) and np.all(np.isfinite(e))

    def test_rejects_bad_out_dim(self):
        with pytest.raises(ValueError):
            MeanPoolStateEncoder(out_dim=0)

    def test_satisfies_protocol(self):
        assert isinstance(MeanPoolStateEncoder(), VLAStateEncoder)


# ─────────────────────────────────────────────────────────────
# §7.3  state retrieval key
# ─────────────────────────────────────────────────────────────
class TestStateRetrievalKey:
    def test_concatenates_vla_and_proprioception(self):
        e_vla = np.array([1.0, 2.0, 3.0])
        p = np.array([9.0, 8.0])
        e = state_retrieval_key(e_vla, p)
        assert e.shape == (5,)
        np.testing.assert_array_equal(e, [1, 2, 3, 9, 8])


# ─────────────────────────────────────────────────────────────
# §6  build Phase1 seed vector DB
# ─────────────────────────────────────────────────────────────
class TestBuildPhase1VectorDB:
    def _dataset(self, tmp_path, specs):
        """specs: (skill, episode, t, valid) 리스트로 raw dataset 을 만든다."""
        ds = RawTrajectoryDataset(tmp_path / "D_phase1_raw")
        for skill, episode, t, valid in specs:
            ds.append(_raw_entry(skill=skill, episode=episode, t=t, valid=valid))
        return ds

    def test_builds_skill_partitioned_db(self, tmp_path):
        ds = self._dataset(tmp_path, [
            ("reach", "ep_0", 0, True), ("reach", "ep_0", 1, True),
            ("grasp", "ep_1", 0, True),
        ])
        db = build_phase1_vector_db(ds, MeanPoolStateEncoder(out_dim=16), _obs_loader)
        assert isinstance(db, SkillVectorDB)
        assert db.size("reach") == 2
        assert db.size("grasp") == 1
        assert db.total_size() == 3

    def test_state_key_dim_is_vla_plus_proprioception(self, tmp_path):
        # e_i = [e^vla(out_dim=16); p_i(P=7)] → 23.
        ds = self._dataset(tmp_path, [("reach", "ep_0", 0, True)])
        db = build_phase1_vector_db(ds, MeanPoolStateEncoder(out_dim=16), _obs_loader)
        assert db.state_keys("reach").shape == (1, 16 + 7)

    def test_action_descriptor_dim_is_k_times_action_dim(self, tmp_path):
        # z^a = DCT K=3 × action_dim=6 → 18.
        ds = self._dataset(tmp_path, [("reach", "ep_0", 0, True)])
        db = build_phase1_vector_db(ds, MeanPoolStateEncoder(),
                                    _obs_loader, ReembeddingConfig(dct_coeffs=3))
        assert db.action_descriptors("reach").shape == (1, 3 * 6)

    def test_ref_resolves_back_to_raw_entry(self, tmp_path):
        # §3/§4 — vector DB ref 로 raw dataset entry 를 복원할 수 있어야 한다.
        ds = self._dataset(tmp_path, [
            ("reach", "ep_0", 0, True), ("grasp", "ep_7", 1, True),
        ])
        db = build_phase1_vector_db(ds, MeanPoolStateEncoder(), _obs_loader)
        ref = db.query_skill("grasp")[0].ref
        resolved = resolve_pointer(ref)
        assert resolved.episode_id == "ep_7"
        assert resolved.skill_id == "grasp"

    def test_meta_records_phase1_seed_fields(self, tmp_path):
        ds = self._dataset(tmp_path, [("reach", "ep_0", 4, True)])
        db = build_phase1_vector_db(ds, MeanPoolStateEncoder(), _obs_loader)
        meta = db.query_skill("reach")[0].meta
        assert meta["phase"] == "phase1"
        assert meta["skill_id"] == "reach"
        assert meta["accepted_by"] == "phase1_seed"
        assert meta["time_index"] == 4
        assert len(meta["subgoal"]) == 3

    def test_skip_invalid_flag(self, tmp_path):
        ds = self._dataset(tmp_path, [
            ("reach", "ep_0", 0, True),
            ("reach", "ep_0", 1, False),     # validity_flag=False
            ("reach", "ep_0", 2, True),
        ])
        kept = build_phase1_vector_db(ds, MeanPoolStateEncoder(), _obs_loader,
                                      ReembeddingConfig(skip_invalid=True))
        assert kept.size("reach") == 2
        full = build_phase1_vector_db(ds, MeanPoolStateEncoder(), _obs_loader,
                                      ReembeddingConfig(skip_invalid=False))
        assert full.size("reach") == 3       # 기본값 — dataset 전체 re-embed

    def test_built_db_persists(self, tmp_path):
        # P_phase1 seed 는 vector DB 영속화로 저장할 수 있어야 한다.
        ds = self._dataset(tmp_path, [
            ("reach", "ep_0", 0, True), ("grasp", "ep_1", 0, True),
        ])
        db = build_phase1_vector_db(ds, MeanPoolStateEncoder(out_dim=12), _obs_loader)
        db.save(tmp_path / "P_phase1.npz")

        restored = SkillVectorDB()
        restored.load(tmp_path / "P_phase1.npz")
        np.testing.assert_array_equal(restored.state_keys("reach"),
                                      db.state_keys("reach"))
        assert restored.query_skill("reach")[0].meta["phase"] == "phase1"

    def test_empty_dataset_builds_empty_db(self, tmp_path):
        ds = RawTrajectoryDataset(tmp_path / "D_phase1_raw")
        db = build_phase1_vector_db(ds, MeanPoolStateEncoder(), _obs_loader)
        assert db.total_size() == 0
