"""Verify ADC's FeetechController normalization matches upstream LeRobot semantics.

Arm (motor_id 1..5): RANGE_M100_100   → range_min↦-100, range_max↦+100
Gripper (motor_id 6): RANGE_0_100     → range_min↦  0,  range_max↦100
drive_mode=1 flips (arm: -norm; gripper: 100-norm) for both directions.

Run with: pytest tests/test_feetech_normalize.py -q
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

# Add src/ to path so we can import lerobot_cap
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot_cap.hardware.feetech import FeetechController  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal calibration stub (no real hardware required).
# ---------------------------------------------------------------------------

@dataclass
class _Calib:
    range_min: int
    range_max: int
    drive_mode: int


class _Stub(FeetechController):
    """Instantiate FeetechController without opening a serial port."""

    def __init__(self, motor_ids, calib_dict):
        # Bypass full __init__; provide only attributes used by _normalize/_unnormalize.
        self.motor_ids = motor_ids
        self.calibration = calib_dict


# ---------------------------------------------------------------------------
# Upstream LeRobot reference implementation (copy-pasted from
# lerobot/src/lerobot/motors/motors_bus.py to serve as the oracle).
# ---------------------------------------------------------------------------

def upstream_normalize(raw, cal, norm_mode):
    """norm_mode ∈ {'M100_100', '0_100'}."""
    min_, max_ = cal.range_min, cal.range_max
    bounded = min(max_, max(min_, raw))
    if norm_mode == "M100_100":
        norm = (((bounded - min_) / (max_ - min_)) * 200) - 100
        return -norm if cal.drive_mode else norm
    elif norm_mode == "0_100":
        norm = ((bounded - min_) / (max_ - min_)) * 100
        return (100 - norm) if cal.drive_mode else norm
    raise ValueError(norm_mode)


def upstream_unnormalize(val, cal, norm_mode):
    min_, max_ = cal.range_min, cal.range_max
    if norm_mode == "M100_100":
        v = -val if cal.drive_mode else val
        bounded = min(100.0, max(-100.0, v))
        return int(((bounded + 100) / 200) * (max_ - min_) + min_)
    elif norm_mode == "0_100":
        v = (100 - val) if cal.drive_mode else val
        bounded = min(100.0, max(0.0, v))
        return int((bounded / 100) * (max_ - min_) + min_)
    raise ValueError(norm_mode)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOTOR_IDS = [1, 2, 3, 4, 5, 6]  # so101 convention: 1..5 arm, 6 gripper


def _make_calib(drive_modes=(0, 0, 0, 0, 0, 0), min_=(500, 500, 500, 500, 0, 2000), max_=(3500, 3500, 3500, 3500, 4095, 3500)):
    return {
        m_id: _Calib(range_min=mn, range_max=mx, drive_mode=dm)
        for m_id, dm, mn, mx in zip(MOTOR_IDS, drive_modes, min_, max_)
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_arm_boundaries_drive_mode_0(self):
        """Arm at range_min → -100, at range_max → +100, at center → 0."""
        calib = _make_calib()
        bus = _Stub(MOTOR_IDS, calib)
        raws = np.array([calib[m].range_min for m in MOTOR_IDS[:5]] + [calib[6].range_min])
        out = bus._normalize(raws)
        for i in range(5):
            assert out[i] == pytest.approx(-100.0, abs=1e-4), f"arm {i} min mismatch"

        raws = np.array([calib[m].range_max for m in MOTOR_IDS[:5]] + [calib[6].range_max])
        out = bus._normalize(raws)
        for i in range(5):
            assert out[i] == pytest.approx(100.0, abs=1e-4), f"arm {i} max mismatch"

    def test_gripper_boundaries_drive_mode_0(self):
        """Gripper at range_min → 0, at range_max → 100 (NOT -100..+100)."""
        calib = _make_calib()
        bus = _Stub(MOTOR_IDS, calib)
        raws = np.array([0, 0, 0, 0, 0, calib[6].range_min])
        out = bus._normalize(raws)
        assert out[5] == pytest.approx(0.0, abs=1e-4)

        raws = np.array([0, 0, 0, 0, 0, calib[6].range_max])
        out = bus._normalize(raws)
        assert out[5] == pytest.approx(100.0, abs=1e-4)

    def test_gripper_center_drive_mode_0(self):
        """Gripper midpoint → 50 (not 0)."""
        calib = _make_calib()
        bus = _Stub(MOTOR_IDS, calib)
        c = calib[6]
        mid_raw = (c.range_min + c.range_max) // 2
        raws = np.array([0, 0, 0, 0, 0, mid_raw])
        out = bus._normalize(raws)
        assert out[5] == pytest.approx(50.0, abs=0.1)

    def test_arm_drive_mode_1_flips_sign(self):
        calib = _make_calib(drive_modes=(1, 0, 0, 0, 0, 0))
        bus = _Stub(MOTOR_IDS, calib)
        raws = np.array([calib[1].range_min, 0, 0, 0, 0, 0])
        out = bus._normalize(raws)
        assert out[0] == pytest.approx(100.0, abs=1e-4)  # -(-100) = +100

    def test_gripper_drive_mode_1_mirrors(self):
        """drive_mode=1 on gripper: 100-norm mirror around 50."""
        calib = _make_calib(drive_modes=(0, 0, 0, 0, 0, 1))
        bus = _Stub(MOTOR_IDS, calib)
        raws = np.array([0, 0, 0, 0, 0, calib[6].range_min])
        out = bus._normalize(raws)
        assert out[5] == pytest.approx(100.0, abs=1e-4)   # 100 - 0 = 100

        raws = np.array([0, 0, 0, 0, 0, calib[6].range_max])
        out = bus._normalize(raws)
        assert out[5] == pytest.approx(0.0, abs=1e-4)     # 100 - 100 = 0

    @pytest.mark.parametrize("drive_modes", [
        (0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1),
    ])
    def test_matches_upstream(self, drive_modes):
        """Against upstream oracle for arm + gripper, all drive_mode combos."""
        calib = _make_calib(drive_modes=drive_modes)
        bus = _Stub(MOTOR_IDS, calib)

        rng = np.random.default_rng(42)
        raws = np.array([rng.integers(calib[m].range_min, calib[m].range_max + 1) for m in MOTOR_IDS])
        out = bus._normalize(raws)

        for i, m_id in enumerate(MOTOR_IDS):
            mode = "0_100" if m_id == 6 else "M100_100"
            expected = upstream_normalize(raws[i], calib[m_id], mode)
            assert out[i] == pytest.approx(expected, abs=1e-4), (
                f"motor {m_id} mismatch: ours={out[i]} upstream={expected}"
            )


class TestUnnormalize:
    def test_gripper_input_range_is_0_100(self):
        """Gripper should accept 0..100 (not -100..+100)."""
        calib = _make_calib()
        bus = _Stub(MOTOR_IDS, calib)
        # Input 0 → range_min
        norms = np.array([0, 0, 0, 0, 0, 0.0], dtype=np.float32)
        raw = bus._unnormalize(norms)
        assert raw[5] == calib[6].range_min

        # Input 100 → range_max
        norms = np.array([0, 0, 0, 0, 0, 100.0], dtype=np.float32)
        raw = bus._unnormalize(norms)
        assert raw[5] == calib[6].range_max

    def test_roundtrip_drive_mode_0(self):
        """normalize(unnormalize(x)) == x, all motors, drive_mode=0."""
        calib = _make_calib()
        bus = _Stub(MOTOR_IDS, calib)
        inputs = np.array([-73.5, 12.0, 0.0, 88.7, -100.0, 42.0], dtype=np.float32)
        raw = bus._unnormalize(inputs)
        back = bus._normalize(raw)
        # Integer rounding introduces ~0.1 error
        for i in range(6):
            assert abs(back[i] - inputs[i]) < 0.2, f"motor {i}: {back[i]} vs {inputs[i]}"

    def test_roundtrip_drive_mode_1(self):
        calib = _make_calib(drive_modes=(1, 1, 1, 1, 1, 1))
        bus = _Stub(MOTOR_IDS, calib)
        inputs = np.array([-50.0, 77.7, 0.0, -99.0, 88.8, 25.0], dtype=np.float32)
        raw = bus._unnormalize(inputs)
        back = bus._normalize(raw)
        for i in range(6):
            assert abs(back[i] - inputs[i]) < 0.2

    @pytest.mark.parametrize("drive_modes", [
        (0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1),
    ])
    def test_matches_upstream(self, drive_modes):
        calib = _make_calib(drive_modes=drive_modes)
        bus = _Stub(MOTOR_IDS, calib)
        rng = np.random.default_rng(7)
        arm_vals = rng.uniform(-100, 100, size=5).astype(np.float32)
        grip_val = rng.uniform(0, 100)
        inputs = np.array([*arm_vals, grip_val], dtype=np.float32)
        ours = bus._unnormalize(inputs)

        for i, m_id in enumerate(MOTOR_IDS):
            mode = "0_100" if m_id == 6 else "M100_100"
            expected = upstream_unnormalize(inputs[i], calib[m_id], mode)
            assert ours[i] == expected, (
                f"motor {m_id}: ours={ours[i]} upstream={expected}"
            )

    def test_clamp_out_of_range(self):
        """Input beyond valid range should clamp, not error."""
        calib = _make_calib()
        bus = _Stub(MOTOR_IDS, calib)
        # Arm input > +100 and gripper > 100
        inputs = np.array([999.0, -999.0, 0, 0, 0, 999.0], dtype=np.float32)
        raw = bus._unnormalize(inputs)
        assert raw[0] == calib[1].range_max
        assert raw[1] == calib[2].range_min
        assert raw[5] == calib[6].range_max

        # Gripper < 0 should clamp to 0 → range_min (NOT go to -50 territory)
        inputs = np.array([0, 0, 0, 0, 0, -50.0], dtype=np.float32)
        raw = bus._unnormalize(inputs)
        assert raw[5] == calib[6].range_min
