"""
data_ingestion.py – Video frame capture with NVDEC hardware decoding support.

Frames are placed on a thread-safe queue; when the queue is full a frame is
dropped (with a warning) rather than blocking the capture loop, ensuring the
pipeline never stalls on a slow consumer.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum number of frames buffered between capture and consumers.
_DEFAULT_QUEUE_CAPACITY: int = 4


def _open_capture(source: int | str, use_nvdec: bool) -> cv2.VideoCapture:
    """Open a VideoCapture, requesting NVDEC (CUDA) hardware decoding when
    *use_nvdec* is True and the back-end supports it.

    Falls back silently to the default software decoder so the pipeline
    remains functional on CPU-only machines.
    """
    if use_nvdec:
        # cv2.CAP_FFMPEG with the CUDA hw-accel codec hint triggers NVDEC
        # zero-copy decoding on supported NVIDIA drivers.
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    return cap


class FrameIngester:
    """Captures video frames in a dedicated daemon thread and publishes them
    on a bounded, thread-safe :class:`queue.Queue`.

    Parameters
    ----------
    source:
        Camera index (``int``) or file/RTSP URL (``str``).
    queue_capacity:
        Maximum number of frames held in the internal queue.  When the queue
        is full the *oldest* frame is discarded and the new frame is enqueued,
        preventing indefinite pipeline back-pressure.
    use_nvdec:
        Request NVDEC hardware decoding via the OpenCV FFMPEG back-end.
    """

    def __init__(
        self,
        source: int | str = 0,
        queue_capacity: int = _DEFAULT_QUEUE_CAPACITY,
        use_nvdec: bool = True,
    ) -> None:
        self._source = source
        self._use_nvdec = use_nvdec
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_capacity)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dropped_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def frame_queue(self) -> queue.Queue[np.ndarray]:
        """The shared frame queue consumed by downstream pipeline threads."""
        return self._queue

    @property
    def dropped_frames(self) -> int:
        """Cumulative count of frames dropped due to a full queue."""
        return self._dropped_frames

    def start(self) -> None:
        """Start the background capture thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="FrameIngester",
            daemon=True,
        )
        self._thread.start()
        logger.info("FrameIngester started (source=%r, nvdec=%s)", self._source, self._use_nvdec)

    def stop(self) -> None:
        """Signal the capture thread to exit and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("FrameIngester stopped (dropped=%d frames)", self._dropped_frames)

    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """Retrieve the next frame from the queue.

        Returns ``None`` if no frame arrives within *timeout* seconds.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        cap = _open_capture(self._source, self._use_nvdec)
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("VideoCapture.read() returned False – end of stream or error.")
                    break
                self._enqueue(frame)
        finally:
            cap.release()

    def _enqueue(self, frame: np.ndarray) -> None:
        """Enqueue *frame*, dropping the oldest entry if the queue is full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()  # discard oldest
                self._dropped_frames += 1
                logger.debug("Frame dropped (total dropped=%d)", self._dropped_frames)
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Extremely rare race: another producer snuck in; just drop.
            self._dropped_frames += 1
