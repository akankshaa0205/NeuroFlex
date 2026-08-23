from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover - optional runtime dependency
    mp_python = None
    mp_vision = None


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


class MediaPipePoseEstimator:
    """Real 33-landmark pose estimator backed by MediaPipe Pose."""

    def __init__(self, model_path: str | Path | None = None, min_detection_confidence: float = 0.5):
        if mp_python is None or mp_vision is None:
            raise RuntimeError("MediaPipe is not installed. Install the pose extra with: pip install -e .[pose]")
        asset_path = Path(model_path or "models/pose_landmarker_full.task")
        if not asset_path.exists():
            raise FileNotFoundError(f"Pose model asset not found: {asset_path}")
        base_options = mp_python.BaseOptions(model_asset_path=str(asset_path.resolve()))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def estimate(self, frame: np.ndarray) -> Pose:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR/RGB image array with shape (H, W, 3).")

        height, width, _ = frame.shape
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        image = mp_python.vision.core.Image(
            image_format=mp_python.vision.core.image.ImageFormat.SRGB,
            data=rgb_frame,
        )
        self._timestamp_ms += max(1, int(time.monotonic_ns() // 1_000_000) - self._timestamp_ms)
        result = self._pose.detect_for_video(image, self._timestamp_ms)
        if not result.pose_landmarks:
            return Pose(joints=np.zeros((33, 3), dtype=float), confidence=0.0)

        landmarks = result.pose_landmarks[0]
        joints = np.array(
            [[landmark.x * width, landmark.y * height, landmark.z * width] for landmark in landmarks],
            dtype=float,
        )
        confidence = float(np.mean([landmark.visibility for landmark in landmarks]))
        return Pose(joints=joints, confidence=confidence)

    def close(self) -> None:
        self._pose.close()


def create_pose_estimator(prefer_real: bool = True) -> SimplePoseEstimator | MediaPipePoseEstimator:
    """Create a real estimator when available, otherwise use the test fallback."""
    if prefer_real and mp is not None:
        return MediaPipePoseEstimator()
    return SimplePoseEstimator(joint_count=33)
