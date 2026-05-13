"""Skill-wise pre-selective acquirer (Method 3): argmax IG·AC, no thresholds.

Cold-start (empty buffer / empty mode set on either side) collapses the
missing axis to a neutral 1.0 contribution, so the remaining axis solely
determines ranking. See LIMITATIONS.md L1 for the OOD vs multi-modal
caveat on AC_model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .AC import build_mode_set, compute_ac, compute_mode_distance
from .IG import compute_buffer_novelty, compute_ig, minmax_norm
from .protocols import BufferStore, PolicyAdapter
from .types import (
    BufferEntry,
    Candidate,
    Context,
    ScoreReport,
    Selection,
)


@dataclass
class SelectorConfig:
    alpha: float = 0.5            # IG geometric-mean exponent
    lam: float = 0.5              # AC geometric-mean exponent
    n_vla_samples: int = 8        # M — π₀ samples for AC_model mode set
    max_modes: int | None = None  # R — mode-set clustering target; None = identity
    context_k: int = 8            # k — context-kNN size for AC_buffer
    eps: float = 1e-8


class Selector:
    """select()        : argmax IG·AC over Ξ_m (no thresholds; always picks one).
    add_to_buffer() : append the executed candidate to B_t.
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
        if not candidates:
            raise ValueError("candidates is empty")
        K = len(candidates)
        cfg = self.config
        skill_id = context.skill_id

        # 1. π₀ forward per candidate → (L_FM, z) for IG
        fm_outputs = [
            self.policy.forward_fm(context, c.action_chunk) for c in candidates
        ]
        u_values = [float(fm.l_fm) for fm in fm_outputs]

        # 2. π₀ sampling → AC_model mode set M_π₀(x_m)
        samples = self.policy.sample_actions(context, cfg.n_vla_samples)
        model_mode_set = build_mode_set(samples, cfg.max_modes)
        d_model = [
            compute_mode_distance(c.action_chunk, model_mode_set)
            for c in candidates
        ]

        # 3. IG buffer side: z-space kNN in B_t^(m)
        buffer_entries = self.buffer.query_skill(skill_id)
        if buffer_entries:
            n_values = [
                compute_buffer_novelty(fm.z, buffer_entries, eps=cfg.eps)
                for fm in fm_outputs
            ]
            u_norm, n_norm, ig = compute_ig(
                u_values, n_values, alpha=cfg.alpha, eps=cfg.eps
            )
        else:
            # Cold-start: N_B unavailable → neutralize that axis.
            u_norm = minmax_norm(u_values, cfg.eps)
            n_norm = [1.0] * K
            n_values = [0.0] * K  # placeholder for report
            ig = [float(u ** cfg.alpha) for u in u_norm]

        # 4. AC buffer side: context-kNN → expert action mode set M_B(x_m)
        #    c_m is the same for all candidates (depends only on O,S,I), so any
        #    fm_outputs[i].c_m gives the query embedding for the current x_m.
        query_c_m = fm_outputs[0].c_m
        neighbors = self.buffer.nearest_by_context(
            context, query_c_m, skill_id, cfg.context_k,
        )
        if neighbors:
            buffer_mode_set = build_mode_set(
                [e.action_chunk for e in neighbors], None
            )
            d_buffer = [
                compute_mode_distance(c.action_chunk, buffer_mode_set)
                for c in candidates
            ]
            ac_model, ac_buffer, ac = compute_ac(
                d_model, d_buffer, lam=cfg.lam, eps=cfg.eps
            )
        else:
            # Cold-start: AC_buffer unavailable → neutralize that axis.
            d_model_norm = minmax_norm(d_model, cfg.eps)
            ac_model = [1.0 - d for d in d_model_norm]
            ac_buffer = [1.0] * K
            d_buffer = [0.0] * K  # placeholder for report
            ac = [float(m ** cfg.lam) for m in ac_model]

        # 5. Final score + argmax
        scores = [float(ig[i] * ac[i]) for i in range(K)]
        chosen_index = int(np.argmax(scores))

        # 6. Per-candidate reports
        reports = [
            ScoreReport(
                candidate_index=i,
                u_pi0_raw=u_values[i],
                n_buffer_raw=n_values[i],
                d_model_ac_raw=d_model[i],
                d_buffer_ac_raw=d_buffer[i],
                u_pi0_norm=u_norm[i],
                n_buffer_norm=n_norm[i],
                ac_model=ac_model[i],
                ac_buffer=ac_buffer[i],
                ig=ig[i],
                ac=ac[i],
                score=scores[i],
            )
            for i in range(K)
        ]

        return Selection(
            chosen_index=chosen_index,
            chosen_candidate=candidates[chosen_index],
            chosen_z=fm_outputs[chosen_index].z,
            chosen_context_embedding=fm_outputs[chosen_index].c_m,
            reports=reports,
        )

    def add_to_buffer(self, context: Context, selection: Selection) -> None:
        """Append ξ_m^* to B_t. Spec line: B_{t+1} = B_t ∪ {ξ_m^*}.

        Call timing (per decision #6): after task judge returns True,
        before the episode is persisted to the LeRobot dataset.
        """
        self.buffer.append(BufferEntry(
            context=context,
            action_chunk=selection.chosen_candidate.action_chunk,
            z=selection.chosen_z,
            context_embedding=selection.chosen_context_embedding,
        ))
