"""Deprecated — DCTDenoiseUncertainty 는 LeRobotVLAInformativenessScorer 의
``mode='dct'`` 옵션으로 통합되었다. 본 모듈은 backward-compat alias 만 노출.

신규 코드는 다음을 사용:

    LeRobotVLAInformativenessScorer(policy=..., mode='dct', sigma=...)
"""
from __future__ import annotations

from method3.phase2_mi_selection.vla_informativeness import (
    LeRobotVLAInformativenessScorer,
)


def DCTDenoiseUncertainty(
    policy,
    batch_builder=None,
    sigma=None,
):
    """alias → LeRobotVLAInformativenessScorer(mode='dct')."""
    return LeRobotVLAInformativenessScorer(
        policy=policy,
        batch_builder=batch_builder,
        mode="dct",
        R=1,
        sigma=sigma,
    )
