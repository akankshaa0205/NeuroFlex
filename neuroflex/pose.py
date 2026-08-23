from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    joints: np.ndarray
    confidence: float = 1.0


class SimplePoseEstimator:
    """A synthetic 3D pose estimator that mirrors the PRD contract without external ML dependencies."""

    def __init__(self, joint_count: int = 33):
        self.joint_count = joint_count

    def estimate(self, frame: np.ndarray) -> Pose:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR/RGB image array with shape (H, W, 3).")

        h, w, _ = frame.shape
        y = np.linspace(0.0, 1.0, self.joint_count)
        x = np.linspace(0.0, 1.0, self.joint_count)
        joints = np.zeros((self.joint_count, 3), dtype=float)

        for i in range(self.joint_count):
            joints[i, 0] = x[i] * w
            joints[i, 1] = y[i] * h
            joints[i, 2] = np.sin(i / 3.0) * 0.5 + 0.5

        return Pose(joints=joints, confidence=0.96)
