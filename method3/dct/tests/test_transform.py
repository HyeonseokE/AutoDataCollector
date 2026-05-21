"""traj_to_dct / dct_to_traj — round-trip + orthonormality + shape."""
from __future__ import annotations

import numpy as np
import pytest

from method3.dct.transform import dct_to_traj, traj_to_dct


class TestShape:
    def test_traj_to_dct_shape(self):
        traj = np.random.default_rng(0).normal(size=(40, 6))
        z = traj_to_dct(traj, L0=50)
        assert z.shape == (50, 6)

    def test_dct_to_traj_shape(self):
        z = np.random.default_rng(0).normal(size=(50, 6))
        traj = dct_to_traj(z, T_target=37)
        assert traj.shape == (37, 6)

    def test_arbitrary_dof(self):
        traj = np.random.default_rng(0).normal(size=(20, 7))
        z = traj_to_dct(traj, L0=50)
        assert z.shape == (50, 7)
        rec = dct_to_traj(z, T_target=20)
        assert rec.shape == (20, 7)


class TestRoundTrip:
    def test_round_trip_same_length(self):
        # T_skill = L0 → resample 이 identity (T==L0 인 경우 그대로).
        # DCT/IDCT 만 거치므로 거의 완벽 복원.
        rng = np.random.default_rng(42)
        traj = rng.normal(size=(50, 6))
        z = traj_to_dct(traj, L0=50)
        rec = dct_to_traj(z, T_target=50)
        np.testing.assert_allclose(rec, traj, atol=1e-10)

    def test_round_trip_short_skill(self):
        # T_skill < L0 (e.g., gripper_close length 15) → 50 으로 over-sample,
        # 다시 15 로 down-sample. cubic resample 의 distortion 만 누적.
        rng = np.random.default_rng(0)
        T = 15
        t = np.linspace(0, 1, T)
        traj = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t),
                         t, t ** 2, np.sin(np.pi * t), np.cos(np.pi * t)],
                        axis=-1)
        z = traj_to_dct(traj, L0=50)
        rec = dct_to_traj(z, T_target=T)
        # smooth traj → round-trip MSE 매우 작아야 함.
        mse = np.mean((rec - traj) ** 2)
        assert mse < 1e-6, f"round-trip MSE too large: {mse}"

    def test_round_trip_long_skill(self):
        # T_skill > L0 (e.g., move_carry length 115).
        rng = np.random.default_rng(1)
        T = 115
        t = np.linspace(0, 1, T)
        traj = np.stack([np.sin(2 * np.pi * t),
                         np.sin(3 * np.pi * t),
                         t,
                         t ** 1.5,
                         np.cos(np.pi * t),
                         np.exp(-t)],
                        axis=-1)
        z = traj_to_dct(traj, L0=50)
        rec = dct_to_traj(z, T_target=T)
        mse = np.mean((rec - traj) ** 2)
        # 길이 115 → 50 → 115 은 down-then-up 으로 high-freq 손실 가능.
        # 단, signal 이 모두 ≤ 3π 주파수라 Nyquist (L0=50 → ~25 cycle) 이하 → small.
        assert mse < 1e-4, f"round-trip MSE too large: {mse}"


class TestOrthonormality:
    def test_dct_basis_orthonormal(self):
        # DCT-II orthonormal: ||z||_2 == ||resampled||_2 (Parseval).
        rng = np.random.default_rng(2)
        traj = rng.normal(size=(50, 6))   # T = L0 라 resample identity.
        z = traj_to_dct(traj, L0=50)
        # Parseval: per-joint sum-of-squares 보존.
        for d in range(6):
            np.testing.assert_allclose(
                np.sum(z[:, d] ** 2),
                np.sum(traj[:, d] ** 2),
                rtol=1e-10,
            )


class TestEdgeCases:
    def test_single_step(self):
        traj = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])  # (1, 6)
        z = traj_to_dct(traj, L0=50)
        assert z.shape == (50, 6)
        # 단일 시점 → 50 번 반복된 constant signal → DCT[0] 만 non-zero.
        # DCT-II orthonormal 에서 constant c 의 첫 계수 = c · sqrt(L0).
        np.testing.assert_allclose(z[0], traj[0] * np.sqrt(50), atol=1e-10)
        np.testing.assert_allclose(z[1:], 0.0, atol=1e-10)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            traj_to_dct(np.zeros(10), L0=50)
        with pytest.raises(ValueError):
            dct_to_traj(np.zeros(10), T_target=20)

    def test_invalid_target_len(self):
        with pytest.raises(ValueError):
            traj_to_dct(np.zeros((10, 6)), L0=0)
