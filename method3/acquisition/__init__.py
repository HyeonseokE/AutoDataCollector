"""Method3 acquisition — two-phase online acquisition 오케스트레이터 (문서 §2).

phase1_state_seeding · phase2_mi_selection · storage · reembedding · phase_control
다섯 라이브러리를 하나의 acquisition 루프로 통합한다. 파일별 역할:

    ports.py          — AcquisitionEnvironment Protocol (hardware/model 경계)
    orchestrator.py   — Method3Acquisition (Phase1→전환→re-embed→Phase2→종료)
    in_memory_env.py  — InMemoryAcquisitionEnvironment (하드웨어 없는 reference 구현)
    demo_offline.py   — offline dry-run (python -m method3.acquisition.demo_offline)

오케스트레이터는 method3 로직을 직접 소유하고, 로봇 실행·VLA 모델·후보 생성은
``AcquisitionEnvironment`` 포트에 위임한다. 실제 시스템이 그 포트를 구현한다.
"""
from method3.acquisition.in_memory_env import InMemoryAcquisitionEnvironment
from method3.acquisition.orchestrator import (
    AcquisitionReport,
    Method3Acquisition,
    Method3AcquisitionConfig,
)
from method3.acquisition.ports import (
    AcquisitionEnvironment,
    Phase1EpisodeResult,
    Phase2EpisodeResult,
)

__all__ = [
    "AcquisitionEnvironment",
    "Phase1EpisodeResult",
    "Phase2EpisodeResult",
    "Method3Acquisition",
    "Method3AcquisitionConfig",
    "AcquisitionReport",
    "InMemoryAcquisitionEnvironment",
]
