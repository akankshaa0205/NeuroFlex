from __future__ import annotations

import numpy as np


def angle_between_vectors(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return the angle in degrees between two 3D vectors."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape.")
    if a.size != 3:
        raise ValueError("Each input vector must contain 3 dimensions.")

    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if np.isclose(norm_product, 0.0):
        return 0.0

    cosine = float(np.dot(a, b) / norm_product)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compute_angle_series(points: np.ndarray) -> np.ndarray:
    """Compute joint angles for a sequence of points using the vertex-angle convention.

    For a sequence of points P=[p0, p1, p2, ...], the angle at each interior point pi is
    computed from the vectors (pi - p_{i-1}) and (p_{i+1} - pi).
    """
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Expected a 2D array of shape (n_points, 3).")
    if arr.shape[0] < 3:
        return np.empty(0, dtype=float)

    vectors_a = arr[:-2] - arr[1:-1]
    vectors_b = arr[2:] - arr[1:-1]
    return np.array([
        angle_between_vectors(a, b)
        for a, b in zip(vectors_a, vectors_b)
    ], dtype=float)
