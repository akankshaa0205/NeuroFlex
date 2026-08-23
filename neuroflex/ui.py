from __future__ import annotations

import numpy as np

from .pose import Pose


class OverlayRenderer:
    """A minimal overlay renderer that simulates the PRD feedback UI without a GUI framework."""

    def render(self, frame: np.ndarray, pose: Pose, score: float, status: str = "running") -> np.ndarray:
        overlay = frame.copy()
        h, w, _ = overlay.shape

        # Draw a simple score bar and pose marker summary in the frame.
        bar_width = int(w * 0.6)
        bar_x = int(w * 0.2)
        bar_y = 25
        bar_h = 14
        filled = int(bar_width * max(0.0, min(1.0, score / 100.0)))
        overlay[bar_y:bar_y + bar_h, bar_x:bar_x + filled] = [0, 255, 0]
        overlay[bar_y:bar_y + bar_h, bar_x + filled:bar_x + bar_width] = [80, 80, 80]

        for joint in pose.joints[:10]:
            x = int(np.clip(joint[0], 0, w - 1))
            y = int(np.clip(joint[1], 0, h - 1))
            overlay[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)] = [255, 255, 255]

        label = status.upper()
        label_width = min(w, max(120, len(label) * 10))
        cv = np.zeros((30, label_width, 3), dtype=np.uint8)
        cv[:] = (30, 30, 30)
        overlay[:30, :label_width] = cv

        return overlay.astype(np.uint8)
