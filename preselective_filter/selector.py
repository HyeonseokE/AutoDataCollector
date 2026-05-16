"""Skill-wise pre-selective acquirer (Method 3): argmax IG·AC, buffer-only.

No pretrained VLA. Both axes are computed purely from the per-skill buffer:

    IG(ξ) = Ñ_B          novelty — far from everything already collected
    AC(ξ) = ÃC_buffer    consistency — close to actions chosen before in
                         similar-context (similar robot state) situations
    score = IG · AC

Cold-start: when the buffer (or the context-kNN neighbour set) is empty the
missing axis collapses to a neutral 1.0. When the whole buffer is empty there
is no signal at all, so a candidate is drawn uniformly at random from a
seeded RNG (deterministic given the seed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocols import BufferStore
from .score_metric.action_consistency import (
    build_mode_set,
    compute_ac,
    compute_mode_distance,
)
from .score_metric.information_gain import compute_buffer_novelty, compute_ig
from .types import BufferEntry, Candidate, Context, ScoreReport, Selection


@dataclass
class SelectorConfig:
    context_k: int = 8            # k — context-kNN size for AC_buffer
    eps: float = 1e-8
    debug_verbose: bool = False   # P1 logging — per-candidate ScoreReport dump


class Selector:
    """select()        : argmax IG·AC over Ξ_m (no thresholds; always picks one).
    add_to_buffer() : append the executed candidate to B_t.
    """

    def __init__(
        self,
        buffer: BufferStore,
        config: SelectorConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.buffer = buffer
        self.config = config or SelectorConfig()
        self._rng = rng if rng is not None else np.random.default_rng()

    def select(self, context: Context, candidates: list[Candidate]) -> Selection:
        if not candidates:
            raise ValueError("candidates is empty")
        K = len(candidates)
        cfg = self.config
        skill_id = context.skill_id

        # 1. IG — novelty vs the whole per-skill buffer (action-space distance)
        buffer_entries = self.buffer.query_skill(skill_id)
        if buffer_entries:
            novelty = [
                compute_buffer_novelty(c.action_chunk, buffer_entries)
                for c in candidates
            ]
            ig = compute_ig(novelty, eps=cfg.eps)
        else:
            # Cold-start: no buffer → novelty axis neutralized.
            novelty = [0.0] * K
            ig = [1.0] * K

        # 2. AC — consistency vs context-kNN neighbour actions
        neighbors = self.buffer.nearest_by_context(context, skill_id, cfg.context_k)
        if neighbors:
            mode_set = build_mode_set([e.action_chunk for e in neighbors])
            d_buffer = [
                compute_mode_distance(c.action_chunk, mode_set)
                for c in candidates
            ]
            ac = compute_ac(d_buffer, eps=cfg.eps)
        else:
            # Cold-start: no neighbours → consistency axis neutralized.
            d_buffer = [0.0] * K
            ac = [1.0] * K

        # 3. Final score + selection
        scores = [float(ig[i] * ac[i]) for i in range(K)]
        cold = not buffer_entries and not neighbors
        if cold:
            # No buffer signal whatsoever → uniform random pick (seeded).
            chosen_index = int(self._rng.integers(0, K))
        else:
            chosen_index = int(np.argmax(scores))

        # Logging — toggled entirely by debug_verbose.
        if cfg.debug_verbose:
            cold_tag = ""
            if cold:
                cold_tag = " [cold:empty-buffer → RNG pick]"
            elif not buffer_entries:
                cold_tag = " [cold:IG neutralized]"
            elif not neighbors:
                cold_tag = " [cold:AC neutralized]"
            print(
                f"[preselective_filter] skill={skill_id} | K={K} candidates "
                f"→ chose idx={chosen_index} (score={scores[chosen_index]:.4f})"
                f"{cold_tag}"
            )
            # Per-candidate breakdown.
            print(f"[preselective_filter] per-candidate IG/AC breakdown (K={K}):")
            for i in range(K):
                mark = "  <-- CHOSEN" if i == chosen_index else ""
                print(
                    f"[preselective_filter]   idx={i}: "
                    f"IG[ N_B={novelty[i]:.4f} ]={ig[i]:.3f}  |  "
                    f"AC[ D_buf={d_buffer[i]:.4f} ]={ac[i]:.3f}  |  "
                    f"score={scores[i]:.4f}{mark}"
                )

        reports = [
            ScoreReport(
                candidate_index=i,
                novelty_raw=novelty[i],
                ig=ig[i],
                consistency_raw=d_buffer[i],
                ac=ac[i],
                score=scores[i],
            )
            for i in range(K)
        ]

        return Selection(
            chosen_index=chosen_index,
            chosen_candidate=candidates[chosen_index],
            reports=reports,
        )

    def add_to_buffer(self, context: Context, selection: Selection) -> None:
        """Append ξ_m^* to B_t. Spec line: B_{t+1} = B_t ∪ {ξ_m^*}.

        Call timing: after task judge returns True, before the episode is
        persisted to the LeRobot dataset.
        """
        self.buffer.append(BufferEntry(
            context=context,
            action_chunk=selection.chosen_candidate.action_chunk,
        ))
