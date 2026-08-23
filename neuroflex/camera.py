from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency in some environments
    cv2 = None


@dataclass
class Frame:
    array: np.ndarray


class FrameStream:
    """A lightweight frame source abstraction for the PRD's video pipeline.

    It supports both a synthetic test source and a real OpenCV webcam capture.
    """

    def __init__(self, source: str | int = "synthetic", size: tuple[int, int] = (640, 480), fps: int = 20):
        self.source = source
        self.size = size
        self.fps = fps
        self._frame_index = 0
        self._capture = None

        if source != "synthetic":
            if cv2 is None:
                raise RuntimeError("OpenCV is required for real webcam capture. Install opencv-python.")
            self._capture = cv2.VideoCapture(int(source) if isinstance(source, int) else source)
            if self.size != (0, 0):
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[1])
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[0])
            self._capture.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self) -> np.ndarray:
        if self._capture is not None:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                raise RuntimeError("Unable to read from the configured camera source.")
            return frame

        h, w = self.size
        base = np.zeros((h, w, 3), dtype=np.uint8)
        x = (self._frame_index % w)
        y = ((self._frame_index * 3) % h)
        base[y, x] = [255, 160, 0]
        base = np.clip(base + np.sin(np.linspace(0, 2 * np.pi, w)).reshape(1, -1, 1) * 20, 0, 255).astype(np.uint8)
        self._frame_index += 1
        return base

    def is_open(self) -> bool:
        return self._capture is None or self._capture.isOpened()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._frame_index = 0
