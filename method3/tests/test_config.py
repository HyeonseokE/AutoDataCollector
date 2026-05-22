"""Method3 config loaders — unit tests (phase{1,2}_config.yaml).

pipeline_config 의 실제 YAML 파일과 합성 YAML 양쪽으로 검증한다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from method3.config import load_phase1_config, load_phase2_config

_PIPELINE_CONFIG = Path(__file__).resolve().parents[2] / "pipeline_config"


# ─────────────────────────────────────────────────────────────
# phase1_config.yaml
# ─────────────────────────────────────────────────────────────
class TestLoadPhase1Config:
    def test_loads_real_phase1_config(self):
        cfg = load_phase1_config(_PIPELINE_CONFIG / "phase1_config.yaml")
        # phase1_config.yaml 의 subgoal 섹션에 mode 키는 없음 (buffer_aware 고정).
        assert cfg["n_candidates"] == 64
        assert "reachability" in cfg
        assert cfg["buffer_file"] == "subgoal_buffer.npz"

    def test_unwraps_subgoal_key(self, tmp_path):
        p = tmp_path / "p1.yaml"
        p.write_text("subgoal:\n  mode: buffer_aware\n  k_nn: 7\n", encoding="utf-8")
        cfg = load_phase1_config(p)
        assert cfg == {"mode": "buffer_aware", "k_nn": 7}

    def test_flat_yaml_also_works(self, tmp_path):
        p = tmp_path / "p1.yaml"
        p.write_text("mode: gaussian\nsigma: 0.05\n", encoding="utf-8")
        cfg = load_phase1_config(p)
        assert cfg["mode"] == "gaussian"


# ─────────────────────────────────────────────────────────────
# phase2_config.yaml
# ─────────────────────────────────────────────────────────────
class TestLoadPhase2Config:
    def test_loads_real_phase2_config(self, tmp_path):
        cfg = load_phase2_config(
            _PIPELINE_CONFIG / "phase2_config.yaml",
            phase1_raw_dir=tmp_path / "p1", phase2_raw_dir=tmp_path / "p2")
        assert cfg.phase_controller.budget == 100
        assert cfg.phase_controller.phase1_min == 30
        assert cfg.phase_controller.phase1_max == 60
        # §6 — re-embedding 과 Phase2 의 K 가 일치해야 한다.
        assert cfg.phase2_mi.dct_coeffs == cfg.reembedding.dct_coeffs

    def test_maps_sections_to_subconfigs(self, tmp_path):
        p = tmp_path / "p2.yaml"
        p.write_text(
            "budget: 60\nphase1_min: 10\nphase1_max: 30\ntau_ready: 0.8\n"
            "saturation_window: 5\ndct_coeffs: 5\nverbose: true\n"
            "readiness:\n  k_min: 4\n  radius_k: 6\n"
            "mi_selection:\n  beta: 2.0\n  lambda: 3.0\n  amb_agg: max\n"
            "reembedding:\n  skip_invalid: true\n",
            encoding="utf-8")
        cfg = load_phase2_config(p, phase1_raw_dir="a", phase2_raw_dir="b")
        assert cfg.phase_controller.budget == 60
        assert cfg.phase_controller.tau_ready == 0.8
        assert cfg.readiness.tau_ready == 0.8          # 공유 top-level 값
        assert cfg.readiness.k_min == 4
        assert cfg.phase2_mi.dct_coeffs == 5           # top-level dct_coeffs
        assert cfg.phase2_mi.beta == 2.0
        assert cfg.phase2_mi.lambda_ == 3.0            # YAML 'lambda' → lambda_
        assert cfg.phase2_mi.amb_agg == "max"
        assert cfg.reembedding.dct_coeffs == 5
        assert cfg.reembedding.skip_invalid is True
        assert cfg.verbose is True

    def test_defaults_when_sections_absent(self, tmp_path):
        p = tmp_path / "p2.yaml"
        p.write_text("budget: 40\nphase1_min: 5\nphase1_max: 20\n", encoding="utf-8")
        cfg = load_phase2_config(p, phase1_raw_dir="a", phase2_raw_dir="b")
        assert cfg.phase_controller.budget == 40
        assert cfg.phase2_mi.k_nn_a == 5               # dataclass 기본값
        assert cfg.phase2_mi.lambda_ == 1.0

    def test_result_drives_orchestrator(self, tmp_path):
        # 로드한 config 로 실제 acquisition 이 돈다 (end-to-end).
        from method3.acquisition.in_memory_env import InMemoryAcquisitionEnvironment
        from method3.acquisition.orchestrator import Method3Acquisition

        p = tmp_path / "p2.yaml"
        p.write_text(
            "budget: 12\nphase1_min: 3\nphase1_max: 6\nsaturation_window: 3\n"
            "dct_coeffs: 3\nmi_selection:\n  k_nn_a: 3\n  min_covered_windows: 0\n",
            encoding="utf-8")
        cfg = load_phase2_config(p, phase1_raw_dir=tmp_path / "d1",
                                 phase2_raw_dir=tmp_path / "d2")
        report = Method3Acquisition(
            InMemoryAcquisitionEnvironment(ready_after=3), cfg).run()
        assert report.total_episodes == 12
