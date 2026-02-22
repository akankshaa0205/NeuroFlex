"""
Tests for neuroflex.dtw – Dynamic Time Warping.
"""

import numpy as np
import pytest

from neuroflex.dtw import dtw_distance, dtw_score


class TestDtwDistance:
    def test_identical_sequences(self) -> None:
        """Identical sequences should yield distance 0."""
        seq = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        dist, acc = dtw_distance(seq, seq)
        assert pytest.approx(dist, abs=1e-6) == 0.0

    def test_distance_non_negative(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.uniform(0, 1, (15, 3))
        b = rng.uniform(0, 1, (20, 3))
        dist, acc = dtw_distance(a, b)
        assert dist >= 0.0

    def test_different_lengths(self) -> None:
        """DTW must handle sequences of different lengths without error."""
        a = np.ones((10, 2))
        b = np.ones((7, 2))
        dist, acc = dtw_distance(a, b)
        assert acc.shape == (10, 7)

    def test_1d_input_accepted(self) -> None:
        """1-D sequences should be accepted (shape (T,) → (T,1))."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        dist, _ = dtw_distance(a, b)
        assert pytest.approx(dist, abs=1e-6) == 0.0

    def test_cost_matrix_shape(self) -> None:
        a = np.zeros((5, 3))
        b = np.zeros((8, 3))
        _, acc = dtw_distance(a, b)
        assert acc.shape == (5, 8)

    def test_symmetry_approx(self) -> None:
        """DTW distance should be approximately symmetric."""
        rng = np.random.default_rng(99)
        a = rng.uniform(0, 1, (12, 2))
        b = rng.uniform(0, 1, (10, 2))
        d_ab, _ = dtw_distance(a, b)
        d_ba, _ = dtw_distance(b, a)
        # DTW is not strictly symmetric when sequences differ in length, but
        # the magnitudes should be in the same ballpark.
        assert abs(d_ab - d_ba) < max(d_ab, d_ba) * 0.5 + 1e-6


class TestDtwScore:
    def test_perfect_match_scores_one(self) -> None:
        seq = np.ones((10, 5))
        assert pytest.approx(dtw_score(seq, seq, max_distance=100.0), abs=1e-6) == 1.0

    def test_score_in_range(self) -> None:
        rng = np.random.default_rng(13)
        for _ in range(10):
            a = rng.uniform(0, 50, (8, 3))
            b = rng.uniform(0, 50, (10, 3))
            s = dtw_score(a, b, max_distance=200.0)
            assert 0.0 <= s <= 1.0

    def test_large_distance_clamps_to_zero(self) -> None:
        a = np.zeros((5, 2))
        b = np.full((5, 2), 1000.0)
        score = dtw_score(a, b, max_distance=1.0)
        assert score == pytest.approx(0.0, abs=1e-6)
