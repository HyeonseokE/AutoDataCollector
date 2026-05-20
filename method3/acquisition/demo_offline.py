"""Offline demo — Method3Acquisition 을 in-memory 환경으로 end-to-end 실행한다.

하드웨어 없이 two-phase acquisition 전체 흐름(Phase1 → readiness → 전환 →
re-embed → Phase2 → 종료)을 ``verbose`` 로그로 확인하기 위한 dry-run.

    python -m method3.acquisition.demo_offline

설정은 ``pipeline_config/phase2_config.yaml`` 에서 읽는다 — 그 파일을 편집하면
이 데모 동작도 바뀐다. 실제 로봇에서 돌리려면 ``InMemoryAcquisitionEnvironment``
자리에 ``AcquisitionEnvironment`` 의 하드웨어 구현을 주입하면 된다.
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

from method3.acquisition.in_memory_env import InMemoryAcquisitionEnvironment
from method3.acquisition.orchestrator import Method3Acquisition
from method3.config import load_phase2_config

# 리포지토리 루트 기준 phase2_config.yaml 경로 (method3/acquisition/ → ../../).
_PHASE2_CONFIG = Path(__file__).resolve().parents[2] / "pipeline_config" / "phase2_config.yaml"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = load_phase2_config(
            _PHASE2_CONFIG,
            phase1_raw_dir=f"{tmp}/D_phase1_raw",
            phase2_raw_dir=f"{tmp}/D_phase2_raw",
        )
        config.verbose = True                    # 데모는 run narration 로그 ON

        # readiness 가 phase1_min 직후 충족되도록 in-memory 환경을 둔다.
        env = InMemoryAcquisitionEnvironment(
            ready_after=max(1, config.phase_controller.phase1_min // 2))
        report = Method3Acquisition(env, config).run()

        print("\n=== AcquisitionReport ===")
        for key, value in asdict(report).items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
