"""Legacy subgoal-level perturbation — 3D Gaussian blob (ablation baseline).

Method3 Phase1 의 **이전 방식**: pure-transit move 의 target xyz 를 truncated
isotropic 3D Gaussian 에서 단일 offset 샘플로 흔든다. buffer-aware subgoal
선택(``subgoal_selector.Phase1SubgoalSelector``)의 비교 baseline 으로 남겨둔다.

Interaction skill(grasp/place/gripper/rotate)과 combined move 는 호출부의
``is_transit`` 파라미터로 보호된다 — skill_type 화이트리스트는 문서용 참고값.

    if skill_type in TRANSIT_SKILL_TYPES:
        offset = perturbation.sample(rng=ep_rng)
        if offset is not None:
            target_position += offset

``_sample_truncated_gaussian_3d`` 는 ``subgoal_candidates`` 의 gaussian 분포
샘플러로도 재사용된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Reference whitelist of skill_types that COULD be transit. Kept for
# documentation only — the actual gating is done at the call site of
# ``move_to_position`` via the explicit ``is_transit`` parameter, NOT by
# inspecting skill_type. Reason: skill_type="move" is too coarse — it covers
# both pure transit AND the descent inside execute_pick_object/execute_place_object.
# Each move_to_position caller declares its intent explicitly.
# ----------------------------------------------------------------------------
TRANSIT_SKILL_TYPES = frozenset({"move"})


@dataclass
class SubgoalPerturbationConfig:
    """Runtime-immutable perturbation parameters."""

    enabled: bool = False
    sigma: float = 0.05        # metres; isotropic std-dev (robotwin default)
    clip_factor: float = 2.0   # reject samples beyond clip_factor * sigma

    @property
    def clip_radius(self) -> float:
        return self.clip_factor * self.sigma


class SubgoalPerturbation:
    """Samples a 3D xyz offset for a single transit subgoal target.

    Usage (from inside a skill function)::

        if skill_type in TRANSIT_SKILL_TYPES:
            offset = self._perturbation.sample(rng=self._perturbation_rng)
            if offset is not None:
                target_position = target_position + offset
    """

    def __init__(self, config: SubgoalPerturbationConfig) -> None:
        self.cfg = config

    def sample(
        self,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | None:
        """Sample a 3-element xyz offset, or ``None`` when disabled.

        Caller is responsible for checking ``TRANSIT_SKILL_TYPES`` before
        invoking and for applying the returned offset to its target_position.
        """
        if not self.cfg.enabled:
            return None
        rng = rng or np.random.default_rng()
        return _sample_truncated_gaussian_3d(
            sigma=self.cfg.sigma,
            clip_radius=self.cfg.clip_radius,
            rng=rng,
        )


def _sample_truncated_gaussian_3d(
    sigma: float,
    clip_radius: float,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> np.ndarray:
    """Sample from N(0, sigma^2 I) with rejection beyond *clip_radius*."""
    for _ in range(max_attempts):
        sample = rng.normal(0.0, sigma, size=3)
        if np.linalg.norm(sample) <= clip_radius:
            return sample
    # Fallback: project onto clip sphere. P(||x|| > 2*sigma) ≈ 1.2% for 3D
    # Gaussian, so 100 attempts is more than sufficient — this is a safety net.
    sample = rng.normal(0.0, sigma, size=3)
    norm = np.linalg.norm(sample)
    if norm > clip_radius:
        sample = sample / norm * clip_radius
    return sample
