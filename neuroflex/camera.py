from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    array: np.ndarray


class FrameStream:
    """A lightweight frame source abstraction for the PRD's video pipeline.

    It supports a synthetic test source so the repository can be exercised without a
    live webcam or GPU dependencies.
    """

    def __init__(self, source: str = "synthetic", size: tuple[int, int] = (640, 480), fps: int = 20):
        self.source = source
        self.size = size
        self.fps = fps
        self._frame_index = 0

    def read(self) -> np.ndarray:
        if self.source != "synthetic":
            raise RuntimeError("Only the synthetic frame source is available in this starter runtime.")

        h, w = self.size
        base = np.zeros((h, w, 3), dtype=np.uint8)
        x = (self._frame_index % w)
        y = ((self._frame_index * 3) % h)
        base[y, x] = [255, 160, 0]
        base = np.clip(base + np.sin(np.linspace(0, 2 * np.pi, w)).reshape(1, -1, 1) * 20, 0, 255).astype(np.uint8)
        self._frame_index += 1
        return base

    def is_open(self) -> bool:
        return True

    def release(self) -> None:
        self._frame_index = 0
