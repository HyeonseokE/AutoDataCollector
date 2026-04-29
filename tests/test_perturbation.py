"""End-to-end tests for subgoal-level perturbation.

Run from AutoDataCollector root:
    python tests/test_perturbation.py

Coverage:
    1. perturbation/subgoal_level — sampler properties (shape, clip, stats, seed)
    2. recording_config_ws1.yaml — `perturbation.subgoal` parses correctly
    3. LeRobotSkills — set_perturbation / set_perturbation_rng hooks behave
    4. ForwardAndResetPipeline._setup_perturbation_on_skills — wires correctly
    5. ForwardAndResetPipeline._seed_episode_perturbation — pending seed mechanism
    6. move_to_position conditional — gripper_action gating actually blocks perturbation

No hardware required. (1)–(5) use real classes; (6) replicates the conditional
verbatim and asserts both branches.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import MagicMock

# Ensure ADC root is on sys.path
ADC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ADC_ROOT))

import numpy as np
import yaml

from perturbation.subgoal_level import (
    SubgoalPerturbation,
    SubgoalPerturbationConfig,
    TRANSIT_SKILL_TYPES,
)


# --------------------------------------------------------------------------
# (1) Sampler module
# --------------------------------------------------------------------------

def test_transit_whitelist():
    assert TRANSIT_SKILL_TYPES == frozenset({"move"}), \
        f"expected only 'move', got {set(TRANSIT_SKILL_TYPES)}"
    # interaction skill_types must NOT be in the whitelist
    for forbidden in ("move_and_close", "move_and_open", "move_initial",
                      "move_free", "gripper_open", "gripper_close", "rotate"):
        assert forbidden not in TRANSIT_SKILL_TYPES, \
            f"interaction skill_type {forbidden!r} leaked into whitelist"


def test_disabled_returns_none():
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=False))
    rng = np.random.default_rng(42)
    for _ in range(10):
        assert pert.sample(rng=rng) is None


def test_sample_shape_and_clip():
    cfg = SubgoalPerturbationConfig(enabled=True, sigma=0.05, clip_factor=2.0)
    pert = SubgoalPerturbation(cfg)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        s = pert.sample(rng=rng)
        assert s.shape == (3,), f"shape={s.shape}"
        assert np.linalg.norm(s) <= cfg.clip_radius + 1e-9, \
            f"norm={np.linalg.norm(s)} > clip={cfg.clip_radius}"


def test_statistics_close_to_target():
    cfg = SubgoalPerturbationConfig(enabled=True, sigma=0.05, clip_factor=2.0)
    pert = SubgoalPerturbation(cfg)
    rng = np.random.default_rng(0)
    samples = np.array([pert.sample(rng=rng) for _ in range(20000)])
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    # Mean ~ 0 (radial rejection is symmetric in each axis).
    assert np.all(np.abs(mean) < 0.005), f"mean={mean} should be near zero"
    # Per-axis std is REDUCED by radial 2-sigma truncation. Empirically the
    # surviving samples have per-axis std ~0.78 * sigma (3D radial truncation
    # discards points outside the 2-sigma sphere — most of the high-variance
    # tail). We just check it stays in a reasonable band.
    assert np.all(std < cfg.sigma), f"std={std} must be below sigma={cfg.sigma}"
    assert np.all(std > 0.5 * cfg.sigma), \
        f"std={std} must be above 0.5*sigma={0.5*cfg.sigma}"


def test_reproducibility():
    cfg = SubgoalPerturbationConfig(enabled=True, sigma=0.05, clip_factor=2.0)
    pert = SubgoalPerturbation(cfg)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    rng_c = np.random.default_rng(456)
    seq_a = [pert.sample(rng=rng_a) for _ in range(20)]
    seq_b = [pert.sample(rng=rng_b) for _ in range(20)]
    seq_c = [pert.sample(rng=rng_c) for _ in range(20)]
    for a, b in zip(seq_a, seq_b):
        assert np.allclose(a, b), "same seed must yield same samples"
    assert any(not np.allclose(a, c) for a, c in zip(seq_a, seq_c)), \
        "different seeds must yield different samples"


# --------------------------------------------------------------------------
# (2) recording_config_ws1.yaml parses
# --------------------------------------------------------------------------

def test_recording_config_ws1_yaml():
    """Verify the perturbation section is structurally well-formed.
    Whether enabled is true/false at any moment is a user toggle, not a
    correctness invariant — only the schema is checked here."""
    cfg_path = ADC_ROOT / "pipeline_config" / "recording_config_ws1.yaml"
    assert cfg_path.exists(), f"missing: {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    pert_section = (cfg.get("perturbation") or {}).get("subgoal") or {}
    assert "enabled" in pert_section, "perturbation.subgoal.enabled missing"
    assert "sigma" in pert_section
    assert "clip_factor" in pert_section
    assert isinstance(pert_section["enabled"], bool)
    assert isinstance(pert_section["sigma"], (int, float)) and pert_section["sigma"] > 0
    assert isinstance(pert_section["clip_factor"], (int, float)) \
        and pert_section["clip_factor"] > 0


# --------------------------------------------------------------------------
# (3) LeRobotSkills hooks
# --------------------------------------------------------------------------

def _make_skills_no_connect():
    """Construct LeRobotSkills without invoking connect() / hardware."""
    from skills.skills_lerobot import LeRobotSkills
    # Use any robot config path (file does not need to exist for __init__).
    return LeRobotSkills(robot_config="robot_configs/robot/so101_robot0.yaml",
                         verbose=False)


def test_lerobot_default_state():
    skills = _make_skills_no_connect()
    assert skills._perturbation is None, "default _perturbation must be None"
    assert skills._perturbation_rng is None, "default _perturbation_rng must be None"


def test_lerobot_set_perturbation():
    skills = _make_skills_no_connect()
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=True))
    skills.set_perturbation(pert)
    assert skills._perturbation is pert


def test_lerobot_set_perturbation_rng():
    skills = _make_skills_no_connect()
    skills.set_perturbation_rng(42)
    assert isinstance(skills._perturbation_rng, np.random.Generator)
    # Reproducibility through the skills object
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=True, sigma=0.05))
    skills.set_perturbation(pert)
    skills.set_perturbation_rng(42)
    a = skills._perturbation.sample(rng=skills._perturbation_rng)
    skills.set_perturbation_rng(42)
    b = skills._perturbation.sample(rng=skills._perturbation_rng)
    assert np.allclose(a, b), "same seed must yield same first sample"


# --------------------------------------------------------------------------
# (4) Pipeline _setup_perturbation_on_skills
# --------------------------------------------------------------------------

def _write_temp_recording_config(perturbation_section: dict) -> str:
    body = {"perturbation": {"subgoal": perturbation_section}}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(body, f)
    f.close()
    return f.name


def _make_pipeline_no_init():
    """Construct ForwardAndResetPipeline bypassing __init__."""
    from execution_forward_and_reset import ForwardAndResetPipeline
    p = ForwardAndResetPipeline.__new__(ForwardAndResetPipeline)
    return p


def test_setup_attaches_when_enabled():
    cfg_path = _write_temp_recording_config(
        {"enabled": True, "sigma": 0.03, "clip_factor": 1.5}
    )
    try:
        p = _make_pipeline_no_init()
        p.recording_config = cfg_path
        fake_skills = MagicMock()
        fake_skills._perturbation = None
        p._skills = fake_skills

        p._setup_perturbation_on_skills()

        assert fake_skills.set_perturbation.called, \
            "set_perturbation should be invoked when enabled=true"
        pert_arg = fake_skills.set_perturbation.call_args[0][0]
        assert isinstance(pert_arg, SubgoalPerturbation)
        assert pert_arg.cfg.enabled is True
        assert pert_arg.cfg.sigma == 0.03
        assert pert_arg.cfg.clip_factor == 1.5
    finally:
        Path(cfg_path).unlink(missing_ok=True)


def test_setup_skips_when_disabled():
    cfg_path = _write_temp_recording_config(
        {"enabled": False, "sigma": 0.05, "clip_factor": 2.0}
    )
    try:
        p = _make_pipeline_no_init()
        p.recording_config = cfg_path
        fake_skills = MagicMock()
        fake_skills._perturbation = None
        p._skills = fake_skills

        p._setup_perturbation_on_skills()

        assert not fake_skills.set_perturbation.called, \
            "set_perturbation must NOT be called when disabled"
    finally:
        Path(cfg_path).unlink(missing_ok=True)


def test_setup_skips_when_no_recording_config():
    p = _make_pipeline_no_init()
    p.recording_config = None
    fake_skills = MagicMock()
    p._skills = fake_skills
    p._setup_perturbation_on_skills()  # silent no-op
    assert not fake_skills.set_perturbation.called


def test_setup_applies_pending_seed():
    """If _seed_episode_perturbation was called before skills existed, the
    seed must be applied at setup time."""
    cfg_path = _write_temp_recording_config(
        {"enabled": True, "sigma": 0.05, "clip_factor": 2.0}
    )
    try:
        p = _make_pipeline_no_init()
        p.recording_config = cfg_path
        # Simulate pending seed (set before _skills existed)
        p._pending_perturbation_seed = 99999
        fake_skills = MagicMock()
        fake_skills._perturbation = None
        p._skills = fake_skills

        p._setup_perturbation_on_skills()

        fake_skills.set_perturbation_rng.assert_called_once_with(99999)
    finally:
        Path(cfg_path).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# (5) Pipeline _seed_episode_perturbation
# --------------------------------------------------------------------------

def test_seed_pending_when_skills_absent():
    p = _make_pipeline_no_init()
    p._seed_episode_perturbation(batch_index=2, slot_in_batch=5)
    expected = 2 * 10000 + 5  # 20005
    assert p._pending_perturbation_seed == expected


def test_seed_applied_immediately_when_skills_ready():
    p = _make_pipeline_no_init()
    fake_skills = MagicMock()
    fake_skills._perturbation = SubgoalPerturbation(
        SubgoalPerturbationConfig(enabled=True)
    )
    p._skills = fake_skills

    p._seed_episode_perturbation(batch_index=3, slot_in_batch=7)

    expected = 3 * 10000 + 7  # 30007
    fake_skills.set_perturbation_rng.assert_called_once_with(expected)
    assert p._pending_perturbation_seed == expected


def test_seed_skipped_when_perturbation_not_attached():
    p = _make_pipeline_no_init()
    fake_skills = MagicMock()
    fake_skills._perturbation = None  # no perturbation attached
    p._skills = fake_skills

    p._seed_episode_perturbation(batch_index=1, slot_in_batch=1)

    assert not fake_skills.set_perturbation_rng.called


# --------------------------------------------------------------------------
# (6) move_to_position conditional — gripper_action gating
# --------------------------------------------------------------------------

def _apply_perturbation_conditional(
    target_position: np.ndarray,
    is_transit: bool,
    perturbation,
    rng,
) -> np.ndarray:
    """Verbatim copy of the conditional from skills_lerobot.move_to_position.
    If this drifts from the source, the test must drift with it — keep them
    in sync. TWO conditions must hold to apply offset:
      (1) is_transit=True            (caller declares intent — sole gate)
      (2) perturbation + rng attached
    Note: gripper_action is INTENTIONALLY ignored. Approach + open/close are
    transit subgoals in this codebase; final-descent + gripper is handled by
    the dedicated execute_*_object functions which pass is_transit=False.
    """
    if (is_transit
            and perturbation is not None
            and rng is not None):
        offset = perturbation.sample(rng=rng)
        if offset is not None:
            return target_position + offset
    return target_position


def test_transit_pure_move_perturbs():
    """is_transit=True → offset applied."""
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=True, sigma=0.05))
    rng = np.random.default_rng(42)
    base = np.array([0.2, 0.1, 0.05])
    out = _apply_perturbation_conditional(base, True, pert, rng)
    assert not np.allclose(out, base), \
        "pure transit move must be perturbed"


def test_is_transit_false_blocks_perturbation():
    """Internal callers (execute_pick_object/execute_place_object descent)
    pass is_transit=False — must NOT be perturbed."""
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=True, sigma=0.05))
    rng = np.random.default_rng(42)
    base = np.array([0.2, 0.1, 0.05])
    out = _apply_perturbation_conditional(base, False, pert, rng)
    assert np.allclose(out, base), \
        "is_transit=False must block perturbation (interaction descent)"


def test_approach_with_gripper_open_perturbs():
    """approach + open (move_and_open) is transit and SHOULD be perturbed.
    Final-descent + open is handled by execute_place_object (is_transit=False)."""
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=True, sigma=0.05))
    rng = np.random.default_rng(42)
    base = np.array([0.2, 0.1, 0.05])
    # is_transit=True regardless of gripper_action (which the conditional
    # no longer inspects).
    out = _apply_perturbation_conditional(base, True, pert, rng)
    assert not np.allclose(out, base), \
        "approach + open should be perturbed (it's transit, not interaction)"


def test_approach_with_gripper_close_perturbs():
    """approach + close (move_and_close) is transit and SHOULD be perturbed."""
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=True, sigma=0.05))
    rng = np.random.default_rng(42)
    base = np.array([0.2, 0.1, 0.05])
    out = _apply_perturbation_conditional(base, True, pert, rng)
    assert not np.allclose(out, base), \
        "approach + close should be perturbed (it's transit, not interaction)"


def test_disabled_perturbation_no_effect():
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=False))
    rng = np.random.default_rng(42)
    base = np.array([0.2, 0.1, 0.05])
    out = _apply_perturbation_conditional(base, True, pert, rng)
    assert np.allclose(out, base), \
        "disabled perturbation must leave target unchanged even for transit"


def test_no_perturbation_attached_no_effect():
    rng = np.random.default_rng(42)
    base = np.array([0.2, 0.1, 0.05])
    out = _apply_perturbation_conditional(base, True, None, rng)
    assert np.allclose(out, base)


def test_no_rng_attached_no_effect():
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=True, sigma=0.05))
    base = np.array([0.2, 0.1, 0.05])
    out = _apply_perturbation_conditional(base, True, pert, None)
    assert np.allclose(out, base)


def test_perturbation_disabled_context_manager():
    """LeRobotSkills.perturbation_disabled() temporarily detaches perturbation
    and restores it on exit (used to skip perturbation during reset)."""
    skills = _make_skills_no_connect()
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=True, sigma=0.05))
    skills.set_perturbation(pert)
    skills.set_perturbation_rng(42)

    assert skills._perturbation is pert

    with skills.perturbation_disabled():
        assert skills._perturbation is None, \
            "perturbation must be detached inside the with-block (reset phase)"

    assert skills._perturbation is pert, \
        "perturbation must be restored after exiting the with-block"


def test_perturbation_disabled_restores_on_exception():
    """Even if the wrapped block raises, perturbation must be restored."""
    skills = _make_skills_no_connect()
    pert = SubgoalPerturbation(SubgoalPerturbationConfig(enabled=True, sigma=0.05))
    skills.set_perturbation(pert)

    try:
        with skills.perturbation_disabled():
            assert skills._perturbation is None
            raise RuntimeError("simulated reset failure")
    except RuntimeError:
        pass

    assert skills._perturbation is pert, \
        "perturbation must be restored even after an exception"


def test_internal_callers_pass_is_transit_false():
    """Source-level check: confirm the two known interaction descents in
    skills_lerobot.py pass is_transit=False. This is a brittle line-level
    check but catches accidental regressions where someone refactors the
    descent without preserving the flag."""
    src = (ADC_ROOT / "skills" / "skills_lerobot.py").read_text()
    # execute_pick_object descent
    assert "move_to_position(pick_position" in src and "is_transit=False" in src, \
        "execute_pick_object descent must pass is_transit=False"
    # execute_place_object descent
    assert "move_to_position(final_position" in src, "place descent moved?"
    # Count is_transit=False occurrences (should be at least 2: pick + place)
    n = src.count("is_transit=False")
    assert n >= 2, f"expected >=2 is_transit=False call sites, found {n}"


# --------------------------------------------------------------------------
# (8) pitch compensation — must use rotation-only drift, not full tip movement.
# --------------------------------------------------------------------------

def _pitch_drift_OLD(approach_pos, approach_rot, pick_pos, pick_rot, tip_local):
    """Buggy formula: includes FRAME displacement, breaks under perturbation."""
    approach_tip = approach_pos + approach_rot @ tip_local
    pick_tip     = pick_pos     + pick_rot     @ tip_local
    return (pick_tip - approach_tip)[:2]


def _pitch_drift_NEW(approach_rot, pick_rot, tip_local):
    """Fixed formula: rotation-induced component only."""
    return (pick_rot @ tip_local - approach_rot @ tip_local)[:2]


def _Ry(deg):
    """Rotation about Y-axis (mimics SO-101 wrist pitch)."""
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])


def test_pitch_compensation_unchanged_for_pure_vertical_descent():
    """When approach is directly above pick (no perturbation), OLD and NEW
    formulas must yield the same drift — guarantees the fix is a no-op for
    the original use case."""
    tip_local = np.array([0.0, 0.0, 0.05])
    approach_pos = np.array([0.25, 0.10, 0.20])
    pick_pos     = np.array([0.25, 0.10, 0.024])  # same xy
    approach_rot = _Ry(0)        # horizontal at approach
    pick_rot     = _Ry(-45)      # tilted at pick

    old = _pitch_drift_OLD(approach_pos, approach_rot, pick_pos, pick_rot, tip_local)
    new = _pitch_drift_NEW(approach_rot, pick_rot, tip_local)

    assert np.allclose(old, new, atol=1e-12), \
        f"pure vertical descent: OLD {old} should equal NEW {new}"


def test_pitch_compensation_ignores_xy_perturbation():
    """Under perturbation, FRAME xy differs between approach and pick. OLD
    formula picks up that displacement (huge drift); NEW formula reports only
    the rotation effect (small)."""
    tip_local = np.array([0.0, 0.0, 0.05])
    perturbed_approach = np.array([0.25 + 0.034, 0.10 + 0.054, 0.254])
    true_pick          = np.array([0.25, 0.10, 0.024])
    approach_rot = _Ry(0)
    pick_rot     = _Ry(-45)

    old = _pitch_drift_OLD(perturbed_approach, approach_rot, true_pick, pick_rot, tip_local)
    new = _pitch_drift_NEW(approach_rot, pick_rot, tip_local)

    # OLD picks up the large xy displacement (~3-5cm)
    assert np.linalg.norm(old) > 0.02, \
        f"OLD must report large drift under perturbation (got {old})"
    # NEW reports only rotation drift (~few cm at 45° tilt of a 5cm tip, but
    # importantly this matches the no-perturbation case)
    new_no_pert = _pitch_drift_NEW(approach_rot, pick_rot, tip_local)
    assert np.allclose(new, new_no_pert), \
        "NEW formula must be invariant to FRAME xy (perturbation)"


def test_skills_source_uses_rotation_only_drift():
    """Source-level guard: the new formula must be in skills_lerobot.py."""
    src = (ADC_ROOT / "skills" / "skills_lerobot.py").read_text()
    assert "approach_tip_offset = approach_rot @ tip_local" in src, \
        "rotation-only formula not present in execute_pick_object"
    assert "pick_tip_offset = pick_rot @ tip_local" in src
    assert "tip_drift = (pick_tip_offset - approach_tip_offset)[:2]" in src, \
        "tip_drift must be computed from rotation-only offsets"


def test_skills_source_has_reachability_fallback():
    """Source-level guard: perturbation must check reachability and fall back
    to the un-perturbed target when the perturbed point is unreachable.
    Without this, move_to_position returns False silently, skipping any
    gripper_action and breaking the next skill (pick/place)."""
    src = (ADC_ROOT / "skills" / "skills_lerobot.py").read_text()
    assert "is_position_reachable(_candidate)" in src, \
        "perturbed target must be reachability-checked before adoption"
    assert "[Perturbation] dropped" in src, \
        "fallback must emit a clear log line for debugging"


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # (1) sampler
        test_transit_whitelist,
        test_disabled_returns_none,
        test_sample_shape_and_clip,
        test_statistics_close_to_target,
        test_reproducibility,
        # (2) yaml
        test_recording_config_ws1_yaml,
        # (3) skills hooks
        test_lerobot_default_state,
        test_lerobot_set_perturbation,
        test_lerobot_set_perturbation_rng,
        # (4) pipeline setup
        test_setup_attaches_when_enabled,
        test_setup_skips_when_disabled,
        test_setup_skips_when_no_recording_config,
        test_setup_applies_pending_seed,
        # (5) episode seeding
        test_seed_pending_when_skills_absent,
        test_seed_applied_immediately_when_skills_ready,
        test_seed_skipped_when_perturbation_not_attached,
        # (6) conditional
        test_transit_pure_move_perturbs,
        test_is_transit_false_blocks_perturbation,
        test_approach_with_gripper_open_perturbs,
        test_approach_with_gripper_close_perturbs,
        test_disabled_perturbation_no_effect,
        test_no_perturbation_attached_no_effect,
        test_no_rng_attached_no_effect,
        # (7) reset-phase suspension
        test_perturbation_disabled_context_manager,
        test_perturbation_disabled_restores_on_exception,
        # (8) source-level guard
        test_internal_callers_pass_is_transit_false,
        # (9) pitch compensation invariance under perturbation
        test_pitch_compensation_unchanged_for_pure_vertical_descent,
        test_pitch_compensation_ignores_xy_perturbation,
        test_skills_source_uses_rotation_only_drift,
        # (10) reachability fallback
        test_skills_source_has_reachability_fallback,
    ]

    print("=" * 72)
    print(" Subgoal-Level Perturbation Test Suite ".center(72))
    print("=" * 72)

    passed = failed = 0
    failures = []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  [PASS] {t.__name__}")
        except Exception as e:
            failed += 1
            failures.append((t.__name__, e, traceback.format_exc()))
            print(f"  [FAIL] {t.__name__}: {e}")

    print("=" * 72)
    print(f"  {passed}/{passed + failed} passed, {failed} failed".center(72))
    print("=" * 72)

    if failures:
        print("\nFailure details:")
        for name, exc, tb in failures:
            print(f"\n--- {name} ---")
            print(tb)

    sys.exit(0 if failed == 0 else 1)
