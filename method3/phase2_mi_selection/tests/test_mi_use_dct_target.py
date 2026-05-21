"""Phase2MISelector — use_dct_target=True 분기 검증 (paradigm step [6])."""
from __future__ import annotations

import numpy as np
import pytest

from method3.phase2_mi_selection.mi_selector import (
    Phase2Candidate,
    Phase2MIConfig,
    Phase2MISelector,
)
from method3.phase2_mi_selection.vector_db import SkillVectorDB


def _make_selector(use_dct: bool):
    return Phase2MISelector(
        SkillVectorDB(),
        Phase2MIConfig(use_dct_target=use_dct, debug_verbose=False),
    )


def _make_candidate(dct_target=None):
    return Phase2Candidate(
        skill_id="move",
        state_keys=np.zeros((1, 8)),
        action_chunks=np.zeros((3, 50, 6)),  # 기존 path 도 호환되게 둠
        dct_target=dct_target,
    )


def test_default_uses_chunk_level_dct():
    sel = _make_selector(use_dct=False)
    cand = _make_candidate(dct_target=None)
    z = sel.action_descriptors(cand)
    # frame-level: (T=3, K*dof) shape — Phase2MIConfig.dct_coeffs=3, dof=6.
    assert z.shape == (3, 3 * 6)


def test_use_dct_target_returns_skill_unit_z():
    sel = _make_selector(use_dct=True)
    rng = np.random.default_rng(0)
    z_target = rng.normal(size=(50, 6))
    cand = _make_candidate(dct_target=z_target)
    z = sel.action_descriptors(cand)
    # skill-atomic: (1, L0*dof) = (1, 300).
    assert z.shape == (1, 300)
    np.testing.assert_allclose(z[0], z_target.flatten(), atol=1e-10)


def test_use_dct_target_requires_dct_field():
    sel = _make_selector(use_dct=True)
    cand = _make_candidate(dct_target=None)
    with pytest.raises(ValueError, match="dct_target"):
        sel.action_descriptors(cand)
