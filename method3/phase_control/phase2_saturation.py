"""Phase2 saturation — MI-style gain saturation — 문서 final_method3_spec §15.2.

Phase2 의 목적은 action coverage 를 늘리되 유사 state 에서 action ambiguity 를
키우지 않는 후보를 모으는 것이다. 최근 accepted 후보들이 candidate batch 평균
보다 더 이상 좋지 않으면 useful MI-style gain 이 saturated 됐다고 본다.

  Q̄̃2^(W) = (1/W) Σ_{i=t-W+1}^t Q̃2(ξ_i*)        — 최근 W개 accepted 평균
  Phase2 종료: Q̄̃2^(W) < 0

여기서 ``Q̃2(ξ_i*)`` 는 §12 의 배치-정규화 score (``Phase2ScoreReport.q2_norm``).
accepted 후보만 기록한다 (window 은 episode 가 아니라 accepted trajectory 기준).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Phase2SaturationTracker:
    """최근 ``W`` 개 accepted 후보의 정규화 score 를 추적한다 (문서 §15.2)."""

    window: int = 10                                  # W — saturation window 크기
    _recent: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    def record(self, q2_norm: float) -> None:
        """accepted 후보 ``ξ*`` 의 정규화 score ``Q̃2(ξ*)`` 를 기록한다."""
        self._recent.append(float(q2_norm))

    def n_recorded(self) -> int:
        """기록된 accepted 후보 수."""
        return len(self._recent)

    def window_mean(self) -> float | None:
        """``Q̄̃2^(W)`` — 최근 W개 평균. accepted 가 W개 미만이면 None."""
        if len(self._recent) < self.window:
            return None
        return float(np.mean(self._recent[-self.window:]))

    def is_saturated(self) -> bool:
        """``Q̄̃2^(W) < 0`` 이면 saturation (문서 §15.2).

        accepted 가 아직 W개 미만이면 판단 불가 → False.
        """
        mean = self.window_mean()
        return mean is not None and mean < 0.0
