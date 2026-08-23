from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from .camera import FrameStream
from .pose import Pose, create_pose_estimator
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

    def __init__(self, source: str | int = "synthetic", width: int = 640, height: int = 480):
        self.stream = FrameStream(source=source, size=(height, width))
        self.pose_estimator = create_pose_estimator(prefer_real=source != "synthetic")
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

    def run_live(self, max_frames: int = 5, display: bool = True) -> list[dict[str, float | str | int | tuple[int, int, int]]]:
        results: list[dict[str, float | str | int | tuple[int, int, int]]] = []
        gui_available = cv2 is not None
        frame_limit = None if max_frames <= 0 else max_frames
        frame_index = 0

        if display and gui_available:
            try:
                cv2.namedWindow("NeuroFlex", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("NeuroFlex", self.stream.size[1], self.stream.size[0])
                cv2.moveWindow("NeuroFlex", 80, 80)
                if sys.platform == "win32":
                    import ctypes

                    window_handle = ctypes.windll.user32.FindWindowW(None, "NeuroFlex")
                    if window_handle:
                        ctypes.windll.user32.SetWindowPos(window_handle, -1, 80, 80, self.stream.size[1], self.stream.size[0], 0x0040)
                print("NeuroFlex webcam window opened. Press q in that window to stop.", flush=True)
            except cv2.error:
                print("GUI backend unavailable. Continuing in non-display mode.", flush=True)
                gui_available = False

        while frame_limit is None or frame_index < frame_limit:
            frame_index += 1
            frame = self.stream.read()
            pose = self.pose_estimator.estimate(frame)

            if self.status == "idle":
                self.status = "running"

            self.score = clamp_score(float(np.mean(pose.joints[:, 2]) * 100.0), 0.0, 100.0)
            rendered = self.renderer.render(frame, pose, self.score, status=self.status)
            results.append(
                {
                    "status": self.status,
                    "score": float(self.score),
                    "joint_count": int(pose.joints.shape[0]),
                    "frame_shape": rendered.shape,
                }
            )

            if display and gui_available:
                try:
                    cv2.imshow("NeuroFlex", rendered)
                    key = cv2.waitKey(30) & 0xFF
                    if key in {ord("q"), ord("Q")}:
                        break
                except cv2.error:
                    print("GUI backend unavailable. Continuing in non-display mode.", flush=True)
                    gui_available = False

        if gui_available and cv2 is not None:
            cv2.destroyAllWindows()
        self.stream.release()
        close_estimator = getattr(self.pose_estimator, "close", None)
        if close_estimator is not None:
            close_estimator()
        return results

    def pause(self) -> None:
        self.status = "paused"

    def resume(self) -> None:
        self.status = "running"
