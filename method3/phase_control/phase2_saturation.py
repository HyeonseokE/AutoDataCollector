"""Phase2 saturation — MI-side usefulness saturation — useful_ood_updated §15.2.

Phase2 의 목적은 action coverage 를 늘리되 유사 state 에서 action ambiguity 를
키우지 않는 후보를 모으는 것이다. 최근 accepted 후보들이 candidate batch 평균
보다 더 이상 좋지 않으면 buffer-side MI usefulness 가 saturated 됐다고 본다.

  M̄̃_MI^(W) = (1/W) Σ_{i=t-W+1}^t M̃_MI(ξ_i*)     — 최근 W개 accepted 평균
  Phase2 종료: M̄̃_MI^(W) < 0

여기서 ``M̃_MI(ξ_i*)`` 는 §13.2 의 배치-정규화 MI-side usefulness score
(``Phase2ScoreReport.q2_norm`` — 필드명은 호환을 위해 보존, 의미는 M̃_MI 와 동일).

useful_ood_updated 의 명시적 주석 (§15.2):
  U_VLA 는 model-side informativeness 지만 그 자체로 dataset-level usefulness 를
  보장하지 않으므로, saturation 판정은 여전히 M̃_MI 기준이다. 어느 후보를 고를지
  는 U_VLA 가 결정해도, 언제 멈출지는 M̃_MI 가 결정.

accepted 후보만 기록한다 (window 은 episode 가 아니라 accepted trajectory 기준).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Phase2SaturationTracker:
    """최근 ``W`` 개 accepted 후보의 정규화 M̃_MI score 를 추적 (§15.2)."""

    window: int = 10                                  # W — saturation window 크기
    _recent: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    def record(self, m_mi_norm: float) -> None:
        """accepted 후보 ``ξ*`` 의 정규화 score ``M̃_MI(ξ*)`` 를 기록.

        backward-compat: 이전엔 ``q2_norm`` 으로 불렸으나 수식 동일 (필드 그대로 q2_norm).
        """
        self._recent.append(float(m_mi_norm))

    def n_recorded(self) -> int:
        """기록된 accepted 후보 수."""
        return len(self._recent)

    def window_mean(self) -> float | None:
        """``M̄̃_MI^(W)`` — 최근 W개 평균. accepted 가 W개 미만이면 None."""
        if len(self._recent) < self.window:
            return None
        return float(np.mean(self._recent[-self.window:]))

    def is_saturated(self) -> bool:
        """``M̄̃_MI^(W) < 0`` 이면 saturation (§15.2).

        accepted 가 아직 W개 미만이면 판단 불가 → False.
        """
        mean = self.window_mean()
        return mean is not None and mean < 0.0
