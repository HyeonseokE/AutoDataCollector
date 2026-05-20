"""Method3 acquisition orchestrator — end-to-end tests (문서 final_method3_spec §2).

InMemoryAcquisitionEnvironment 로 two-phase acquisition 루프 전체를 검증한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from method3.acquisition.in_memory_env import InMemoryAcquisitionEnvironment
from method3.acquisition.orchestrator import (
    Method3Acquisition,
    Method3AcquisitionConfig,
)
from method3.phase2_mi_selection.mi_selector import Phase2MIConfig
from method3.phase_control.phase_controller import PhaseControllerConfig
from method3.reembedding.seed_builder import ReembeddingConfig


def _config(tmp_path, *, verbose=False, **controller_kw):
    base = dict(budget=12, phase1_min=3, phase1_max=6, saturation_window=3)
    base.update(controller_kw)
    return Method3AcquisitionConfig(
        phase1_raw_dir=tmp_path / "D_phase1_raw",
        phase2_raw_dir=tmp_path / "D_phase2_raw",
        phase_controller=PhaseControllerConfig(**base),
        # §6 — re-embedding 과 Phase2 는 같은 K 를 써야 한다.
        reembedding=ReembeddingConfig(dct_coeffs=3),
        # min_covered_windows=0 → covered-window fragility 없이 accept 경로 검증.
        phase2_mi=Phase2MIConfig(dct_coeffs=3, k_nn_a=3, min_covered_windows=0),
        verbose=verbose,
    )


class TestMethod3Acquisition:
    def test_runs_to_budget_when_never_ready(self, tmp_path):
        # readiness 영원히 미충족 → B_{1,max}=6 에서 강제 전환 → budget 까지.
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=None),
                                 _config(tmp_path))
        rep = acq.run()
        assert rep.total_episodes == 12
        assert rep.phase1_episodes == 6              # B_{1,max} 강제 전환
        assert rep.phase2_episodes == 6
        assert rep.final_phase == "phase2"
        assert rep.stop_reason == "budget"

    def test_transition_on_readiness(self, tmp_path):
        # episode 3 부터 covered → B_{1,min}=3 에서 readiness 전환.
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=3),
                                 _config(tmp_path))
        rep = acq.run()
        assert rep.phase1_episodes == 3
        assert rep.phase2_episodes == 9
        assert rep.last_readiness == pytest.approx(1.0)

    def test_episode_accounting_consistent(self, tmp_path):
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=4),
                                 _config(tmp_path))
        rep = acq.run()
        assert rep.phase1_episodes + rep.phase2_episodes == rep.total_episodes

    def test_phase1_raw_logged(self, tmp_path):
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=3),
                                 _config(tmp_path))
        rep = acq.run()
        assert len(acq.d_phase1_raw) == rep.phase1_episodes * 2   # window 2/episode

    def test_phase1_seed_built_at_transition(self, tmp_path):
        # §6 — 전환 시 Phase1 raw 를 re-embedding 한 seed DB 가 만들어진다.
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=3),
                                 _config(tmp_path))
        rep = acq.run()
        assert rep.phase1_seed_size == len(acq.d_phase1_raw)      # entry 당 1 seed
        assert rep.phase1_seed_size > 0

    def test_phase2_accepts_grow_vector_db(self, tmp_path):
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=3),
                                 _config(tmp_path))
        rep = acq.run()
        assert rep.phase2_accepted > 0
        # §14 — accept 시 window 별 entry 가 vector DB 에 누적된다.
        assert rep.vector_db_size > rep.phase1_seed_size

    def test_phase2_raw_logged(self, tmp_path):
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=3),
                                 _config(tmp_path))
        rep = acq.run()
        assert len(acq.d_phase2_raw) == rep.phase2_accepted      # 실행 1 entry/accept

    def test_early_stop_disabled_runs_full_budget(self, tmp_path):
        cfg = _config(tmp_path, enable_phase2_early_stop=False)
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=3), cfg)
        rep = acq.run()
        assert rep.total_episodes == 12
        assert rep.stop_reason == "budget"

    def test_config_rejects_dct_mismatch(self, tmp_path):
        # §6 — re-embedding 과 Phase2 의 K 가 다르면 metric space 불일치.
        with pytest.raises(ValueError):
            Method3AcquisitionConfig(
                phase1_raw_dir=tmp_path / "a",
                phase2_raw_dir=tmp_path / "b",
                reembedding=ReembeddingConfig(dct_coeffs=3),
                phase2_mi=Phase2MIConfig(dct_coeffs=5),
            )

    def test_transition_fires_even_when_max_equals_budget(self, tmp_path):
        # B_{1,max} == budget → t=B_{1,max} 에서 전환 후 곧바로 budget 종료.
        cfg = _config(tmp_path, budget=6, phase1_min=3, phase1_max=6)
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=None), cfg)
        rep = acq.run()
        assert rep.total_episodes == 6
        assert rep.phase1_episodes == 6             # B_{1,max} 강제 전환
        assert rep.phase2_episodes == 0             # 전환 직후 budget 소진
        assert rep.final_phase == "phase2"


class TestSeedAnchoring:
    """phase1_seed_anchor_logic — Phase2 후보는 Phase1 seed 를 anchor 로 한다."""

    def test_env_collects_seed_subgoals_from_phase1(self):
        env = InMemoryAcquisitionEnvironment(ready_after=None)
        for _ in range(4):
            env.run_phase1_episode()
        seeds = env.collect_seed_subgoals()
        assert seeds["reach"].shape == (4, 3)        # episode 당 seed 1개

    def test_phase2_candidates_anchored_to_seeds(self):
        env = InMemoryAcquisitionEnvironment(ready_after=None)
        for _ in range(3):
            env.run_phase1_episode()
        seeds = env.collect_seed_subgoals()
        cands = env.generate_phase2_candidates(env.build_phase2_encoder(None), seeds)
        assert cands and all(c.seed_subgoal is not None for c in cands)
        for c in cands:                              # anchor 는 G_seed 의 원소
            assert any(np.allclose(c.seed_subgoal, s) for s in seeds["reach"])

    def test_no_seeds_yields_no_candidates(self):
        env = InMemoryAcquisitionEnvironment(ready_after=None)
        enc = env.build_phase2_encoder(None)
        assert env.generate_phase2_candidates(enc, {}) == []

    def test_orchestrator_records_anchor_in_accepted_meta(self, tmp_path):
        # §14 — accept 된 Phase2 entry 의 meta 에 seed anchor 가 기록된다.
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=3),
                                 _config(tmp_path))
        acq.run()
        assert acq._seed_subgoals.get("reach") is not None
        anchored = [
            e for e in acq.mi_selector.db.query_skill("reach")
            if e.meta.get("seed_subgoal") is not None
        ]
        assert anchored, "accepted Phase2 entries must record their seed anchor"


class TestVerboseLogging:
    """verbose=True — run 전 과정을 [method3:acq] 로그로 narration."""

    def test_verbose_emits_full_run_trace(self, tmp_path, capsys):
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=3),
                                 _config(tmp_path, verbose=True))
        acq.run()
        out = capsys.readouterr().out
        assert "[method3:acq] run start" in out
        assert "Phase1 ep t=" in out
        assert "readiness probe" in out
        assert "Phase1→Phase2 transition" in out
        assert "re-embed" in out                       # §6 로그
        assert "Phase2 ep t=" in out
        assert "accepted →" in out                     # §14 로그
        assert "run end" in out

    def test_verbose_off_is_silent(self, tmp_path, capsys):
        acq = Method3Acquisition(InMemoryAcquisitionEnvironment(ready_after=3),
                                 _config(tmp_path, verbose=False))
        acq.run()
        assert "[method3:acq]" not in capsys.readouterr().out
