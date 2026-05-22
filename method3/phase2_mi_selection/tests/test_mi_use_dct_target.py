"""Phase2MISelector — skill-unit DCT action_descriptors 검증 (paradigm step [6])."""
from __future__ import annotations

import numpy as np
import pytest

from method3.phase2_mi_selection.mi_selector import (
    Phase2Candidate,
    Phase2MIConfig,
    Phase2MISelector,
)
from method3.phase2_mi_selection.vector_db import SkillVectorDB


def _make_selector():
    return Phase2MISelector(
        SkillVectorDB(),
        Phase2MIConfig(debug_verbose=False),
    )


def _make_candidate(dct_target=None):
    return Phase2Candidate(
        skill_id="move",
        state_keys=np.zeros((1, 8)),
        action_chunks=np.zeros((3, 50, 6)),
        dct_target=dct_target,
    )


def test_action_descriptors_returns_skill_unit_z():
    sel = _make_selector()
    rng = np.random.default_rng(0)
    z_target = rng.normal(size=(50, 6))
    cand = _make_candidate(dct_target=z_target)
    z = sel.action_descriptors(cand)
    # skill-atomic: (1, L0*dof) = (1, 300).
    assert z.shape == (1, 300)
    np.testing.assert_allclose(z[0], z_target.flatten(), atol=1e-10)


def test_action_descriptors_requires_dct_target():
    sel = _make_selector()
    cand = _make_candidate(dct_target=None)
    with pytest.raises(ValueError, match="dct_target"):
        sel.action_descriptors(cand)
