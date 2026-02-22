"""
Tests for neuroflex.kinematics – vectorised kinematic computations.
"""

import numpy as np
import pytest

from neuroflex.kinematics import (
    compute_joint_angles,
    compute_range_of_motion,
    compute_velocity,
    score_repetition,
)


class TestComputeJointAngles:
    """Tests for compute_joint_angles."""

    def test_right_angle(self) -> None:
        """Three landmarks forming a 90° angle at the middle joint."""
        landmarks = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        triplets = np.array([[0, 1, 2]], dtype=np.int64)
        angles = compute_joint_angles(landmarks, triplets)
        assert angles.shape == (1,)
        assert pytest.approx(angles[0], abs=1e-4) == 90.0

    def test_straight_line(self) -> None:
        """Co-linear landmarks should produce 180°."""
        landmarks = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        triplets = np.array([[0, 1, 2]], dtype=np.int64)
        angles = compute_joint_angles(landmarks, triplets)
        assert pytest.approx(angles[0], abs=1e-4) == 180.0

    def test_batch_joints(self) -> None:
        """Multiple joint triplets are processed in a single vectorised call."""
        # Square corners: angles at index 1 and 3 should both be 90°.
        landmarks = np.array(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [1.0, 1.0, 0.0],  # 2
                [0.0, 1.0, 0.0],  # 3
            ],
            dtype=np.float64,
        )
        triplets = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        angles = compute_joint_angles(landmarks, triplets)
        assert angles.shape == (2,)
        np.testing.assert_allclose(angles, [90.0, 90.0], atol=1e-4)

    def test_zero_length_vector_guard(self) -> None:
        """Coincident landmarks must not produce NaN/inf."""
        landmarks = np.zeros((3, 3), dtype=np.float64)
        triplets = np.array([[0, 1, 2]], dtype=np.int64)
        angles = compute_joint_angles(landmarks, triplets)
        assert np.isfinite(angles).all()


class TestComputeRangeOfMotion:
    def test_output_keys_and_shapes(self) -> None:
        rng = np.random.default_rng(0)
        seq = rng.uniform(0, 180, size=(30, 5)).astype(np.float64)
        rom = compute_range_of_motion(seq)
        for key in ("min", "max", "range", "mean", "std"):
            assert key in rom
            assert rom[key].shape == (5,)

    def test_range_is_max_minus_min(self) -> None:
        seq = np.array([[10.0, 20.0], [30.0, 40.0], [20.0, 30.0]])
        rom = compute_range_of_motion(seq)
        np.testing.assert_array_equal(rom["range"], rom["max"] - rom["min"])

    def test_single_frame(self) -> None:
        seq = np.array([[45.0, 90.0, 135.0]])
        rom = compute_range_of_motion(seq)
        np.testing.assert_array_equal(rom["range"], [0.0, 0.0, 0.0])


class TestComputeVelocity:
    def test_shape(self) -> None:
        positions = np.ones((10, 17, 3), dtype=np.float64)
        vel = compute_velocity(positions, dt=1.0 / 30.0)
        assert vel.shape == (9, 17, 3)

    def test_constant_motion(self) -> None:
        """Uniform displacement per frame → constant velocity."""
        t = np.arange(10, dtype=np.float64)
        positions = np.stack([t, t, t], axis=-1)[:, np.newaxis, :]  # (10,1,3)
        vel = compute_velocity(positions, dt=1.0)
        np.testing.assert_allclose(vel, 1.0)


class TestScoreRepetition:
    def test_identical_sequences(self) -> None:
        seq = np.array([30.0, 60.0, 90.0, 120.0, 150.0])
        assert pytest.approx(score_repetition(seq, seq), abs=1e-6) == 1.0

    def test_orthogonal_sequences(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        score = score_repetition(a, b)
        assert pytest.approx(score, abs=1e-6) == 0.0

    def test_output_in_range(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.uniform(0, 180, 10)
            b = rng.uniform(0, 180, 10)
            s = score_repetition(a, b)
            assert 0.0 <= s <= 1.0
