"""Skill-wise pre-selective acquirer (Method 3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocols import BufferStore, PolicyAdapter
from .types import Candidate, Context, Selection


@dataclass
class SelectorConfig:
    alpha: float = 0.5            # IG geometric-mean exponent
    lam: float = 0.5              # AC geometric-mean exponent
    delta_ig: float | None = None # IG gate (None → no gate)
    delta_ac: float | None = None # AC gate (None → no gate)
    context_k: int = 8            # k for context-kNN in AC_buffer (|M_B| = R ≤ k)
    eps: float = 1e-8


class Selector:
    """select() : argmax IG·AC over Ξ_m s.t. IG>δ_IG, AC>δ_AC.
    commit()  : append the executed candidate to B_t.
    """

    def __init__(
        self,
        policy: PolicyAdapter,
        buffer: BufferStore,
        config: SelectorConfig | None = None,
    ) -> None:
        self.policy = policy
        self.buffer = buffer
        self.config = config or SelectorConfig()

    def select(self, context: Context, candidates: list[Candidate]) -> Selection:
        raise NotImplementedError

    def commit(self, context: Context, chosen: Candidate, z: np.ndarray) -> None:
        raise NotImplementedError
