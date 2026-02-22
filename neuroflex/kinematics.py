"""
kinematics.py – NumPy-vectorised kinematic computations.

All mathematical operations are expressed as NumPy array operations; no
native Python ``for`` loops are used for numerical work.
"""

from __future__ import annotations

import numpy as np


def compute_joint_angles(
    landmarks: np.ndarray,
    joint_triplets: np.ndarray,
) -> np.ndarray:
    """Compute the interior angle (in degrees) at each middle joint.

    Parameters
    ----------
    landmarks:
        Float array of shape ``(N, D)`` where *N* is the number of landmarks
        and *D* is the spatial dimension (2 or 3).
    joint_triplets:
        Integer array of shape ``(J, 3)`` where each row ``[a, b, c]``
        identifies three landmark indices.  The angle is computed at *b*
        (i.e. the angle ∠abc).

    Returns
    -------
    np.ndarray
        1-D float array of shape ``(J,)`` with angles in **degrees**.
    """
    a = landmarks[joint_triplets[:, 0]]  # (J, D)
    b = landmarks[joint_triplets[:, 1]]  # (J, D)
    c = landmarks[joint_triplets[:, 2]]  # (J, D)

    ba = a - b  # (J, D)
    bc = c - b  # (J, D)

    # Dot products and norms – fully vectorised over J joints.
    dot = np.einsum("ij,ij->i", ba, bc)          # (J,)
    norm_ba = np.linalg.norm(ba, axis=1)         # (J,)
    norm_bc = np.linalg.norm(bc, axis=1)         # (J,)

    # Guard against zero-length vectors (stationary / occluded joints).
    denom = norm_ba * norm_bc
    safe_denom = np.where(denom > 1e-8, denom, 1.0)  # avoid division by zero
    cos_theta = np.where(denom > 1e-8, dot / safe_denom, 0.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.degrees(np.arccos(cos_theta))


def compute_range_of_motion(
    angle_sequence: np.ndarray,
) -> dict[str, float]:
    """Compute range-of-motion (ROM) statistics from a temporal angle sequence.

    Parameters
    ----------
    angle_sequence:
        Float array of shape ``(T, J)`` – *T* time steps, *J* joint angles.

    Returns
    -------
    dict with keys ``"min"``, ``"max"``, ``"range"``, ``"mean"``, ``"std"``
    for each of the *J* joints (values are shape ``(J,)`` arrays).
    """
    return {
        "min":   angle_sequence.min(axis=0),
        "max":   angle_sequence.max(axis=0),
        "range": angle_sequence.max(axis=0) - angle_sequence.min(axis=0),
        "mean":  angle_sequence.mean(axis=0),
        "std":   angle_sequence.std(axis=0),
    }


def compute_velocity(
    positions: np.ndarray,
    dt: float = 1.0 / 30.0,
) -> np.ndarray:
    """Finite-difference velocity of landmark trajectories.

    Parameters
    ----------
    positions:
        Float array ``(T, N, D)`` – *T* frames, *N* landmarks, *D* dims.
    dt:
        Time step in seconds (default: 1/30 s for 30 fps).

    Returns
    -------
    np.ndarray
        Shape ``(T-1, N, D)`` – frame-to-frame velocity.
    """
    return np.diff(positions, axis=0) / dt


def score_repetition(
    rep_angles: np.ndarray,
    reference_angles: np.ndarray,
) -> float:
    """Score a single exercise repetition against a reference template.

    Uses the cosine similarity between the two (flattened) angle sequences
    as a normalised quality score in ``[0, 1]``.

    Parameters
    ----------
    rep_angles:
        Float array of any shape; flattened internally.
    reference_angles:
        Float array of the same size after flattening.

    Returns
    -------
    float
        Similarity score in ``[0.0, 1.0]``.
    """
    r = rep_angles.ravel()
    ref = reference_angles.ravel()
    dot = r @ ref
    denom = (np.linalg.norm(r) * np.linalg.norm(ref))
    if denom < 1e-8:
        return 0.0
    return float(np.clip(dot / denom, 0.0, 1.0))
