from __future__ import annotations

import math


def calculate_composite_score(
    rom_ratio: float,
    dtw_distance: float,
    dtw_max: float,
    theta_error: float,
    theta_threshold: float,
    penalty: float = 1.0,
) -> float:
    """Compute the PRD-defined composite score before display scaling.

    Score = (1/3) * (
        ROM_achieved/ROM_target
        + (1 - DTW(P,I)/DTW_max)
        + max(0, 1 - θ_error/θ_thresh)
    ) × Penalty
    """
    if dtw_max <= 0:
        raise ValueError("dtw_max must be greater than zero.")
    if theta_threshold <= 0:
        raise ValueError("theta_threshold must be greater than zero.")

    normalized_dtw = 1.0 - (dtw_distance / dtw_max)
    theta_component = max(0.0, 1.0 - (theta_error / theta_threshold))
    score = (1.0 / 3.0) * (
        rom_ratio + normalized_dtw + theta_component
    ) * penalty
    return float(score)


def score_to_percent(score: float) -> float:
    """Convert a normalized score in [0, 1] to a 0-100 display scale."""
    return float(max(0.0, min(100.0, score * 100.0)))


def clamp_score(score: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return float(max(minimum, min(maximum, score)))
