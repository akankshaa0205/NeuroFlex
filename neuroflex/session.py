from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import FrameStream
from .pose import Pose, SimplePoseEstimator
from .scoring import clamp_score
from .ui import OverlayRenderer


@dataclass
class SessionState:
    status: str
    score: float
    joint_count: int
    frame_shape: tuple[int, int, int]


class SessionRunner:
    """A minimal session loop matching the PRD's live runtime responsibilities."""

    def __init__(self, source: str = "synthetic", width: int = 640, height: int = 480):
        self.stream = FrameStream(source=source, size=(height, width))
        self.pose_estimator = SimplePoseEstimator(joint_count=33)
        self.renderer = OverlayRenderer()
        self.status = "idle"
        self.score = 0.0

    def run_once(self) -> dict[str, float | str | int | tuple[int, int, int]]:
        frame = self.stream.read()
        pose = self.pose_estimator.estimate(frame)

        if self.status == "idle":
            self.status = "running"

        self.score = clamp_score(float(np.mean(pose.joints[:, 2]) * 100.0), 0.0, 100.0)
        rendered = self.renderer.render(frame, pose, self.score, status=self.status)

        return {
            "status": self.status,
            "score": float(self.score),
            "joint_count": int(pose.joints.shape[0]),
            "frame_shape": rendered.shape,
        }

    def pause(self) -> None:
        self.status = "paused"

    def resume(self) -> None:
        self.status = "running"
