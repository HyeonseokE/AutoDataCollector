"""Method3 ↔ live robot pipeline integration shims.

method3/ is a pure-logic library that depends on AcquisitionEnvironment Protocol
implementations injected from outside. This package supplies the integration
glue between the live ``execution_forward_and_reset`` pipeline and method3.

Stage 1 — Phase1 readiness measurement + auto-stop.
Stage 2 (later) — Phase2 candidate generation + execution.
Stage 3 (later) — full Method3Acquisition orchestrator wire-up.
"""

from method3_integration.phase1_readiness_hook import (
    Phase1ReadinessHook,
    Phase1ReadinessHookConfig,
    measure_phase1_readiness_loo,
)

__all__ = [
    "Phase1ReadinessHook",
    "Phase1ReadinessHookConfig",
    "measure_phase1_readiness_loo",
]
