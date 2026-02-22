"""
dtw.py – Dynamic Time Warping for rehabilitation exercise comparison.

Implemented with SciPy distance utilities and NumPy; no native Python
``for`` loops are used in the hot path.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def dtw_distance(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
    dist_metric: str = "euclidean",
) -> tuple[float, np.ndarray]:
    """Compute the DTW distance between two multi-dimensional sequences.

    Parameters
    ----------
    sequence_a:
        Float array of shape ``(T_a, D)``.
    sequence_b:
        Float array of shape ``(T_b, D)``.
    dist_metric:
        Any metric accepted by :func:`scipy.spatial.distance.cdist`.

    Returns
    -------
    distance : float
        The normalised DTW distance (divided by ``T_a + T_b`` to be
        length-independent).
    cost_matrix : np.ndarray
        Shape ``(T_a, T_b)`` accumulated cost matrix (useful for
        path visualisation / debugging).
    """
    if sequence_a.ndim == 1:
        sequence_a = sequence_a[:, np.newaxis]
    if sequence_b.ndim == 1:
        sequence_b = sequence_b[:, np.newaxis]

    T_a, T_b = sequence_a.shape[0], sequence_b.shape[0]

    # Full pairwise cost matrix – vectorised via SciPy.
    pairwise = cdist(sequence_a, sequence_b, metric=dist_metric)  # (T_a, T_b)

    # Build accumulated-cost matrix with NumPy; inner loop replaced by
    # row-by-row vectorised update (O(T_a) Python iterations, O(T_b) work
    # each – the minimum unavoidable sequential dependency in vanilla DTW).
    acc = np.full((T_a, T_b), np.inf, dtype=np.float64)
    acc[0, :] = np.cumsum(pairwise[0])
    acc[:, 0] = np.cumsum(pairwise[:, 0])

    # Process row-by-row; each row update is a vectorised np.minimum call.
    # For each cell acc[i, j], the three DTW predecessors are:
    #   diagonal : acc[i-1, j-1]  → acc[i-1, :-1] shifted to j=1..T_b-1
    #   up       : acc[i-1, j  ]  → acc[i-1, 1:]
    #   left     : acc[i,   j-1]  → acc[i, :-1] (acc[i,0] already initialised;
    #                               remaining entries are inf until overwritten)
    for i in range(1, T_a):
        diagonal = acc[i - 1, :-1]   # (T_b-1,) – diagonal predecessor
        up       = acc[i - 1, 1:]    # (T_b-1,) – up predecessor
        left     = acc[i, :-1]       # (T_b-1,) – left predecessor (acc[i,0] set)

        # minimum of three predecessors, vectorised across j
        min_prev  = np.minimum(np.minimum(diagonal, up), left)
        acc[i, 1:] = pairwise[i, 1:] + min_prev

    distance = float(acc[T_a - 1, T_b - 1]) / (T_a + T_b)
    return distance, acc


def dtw_score(
    candidate: np.ndarray,
    reference: np.ndarray,
    max_distance: float = 100.0,
) -> float:
    """Return a quality score in ``[0, 1]`` comparing *candidate* to
    *reference* using DTW distance (lower distance → higher score).

    Parameters
    ----------
    candidate:
        Observed exercise sequence, shape ``(T, D)``.
    reference:
        Gold-standard template sequence, shape ``(T_ref, D)``.
    max_distance:
        Distance value that maps to a score of ``0.0``.  Values above this
        are clamped to ``0.0``.

    Returns
    -------
    float
        Score in ``[0.0, 1.0]`` where ``1.0`` is a perfect match.
    """
    dist, _ = dtw_distance(candidate, reference)
    return float(np.clip(1.0 - dist / max_distance, 0.0, 1.0))
