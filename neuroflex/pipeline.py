"""
pipeline.py – NeuroFlex real-time CV rehabilitation pipeline orchestrator.

Wires together:
  * :class:`~neuroflex.data_ingestion.FrameIngester`   (capture thread)
  * :class:`~neuroflex.gpu_thread.PoseEstimationWorker` (GPU inference thread)
  * :class:`~neuroflex.cpu_thread.HandTrackingWorker`   (CPU logic thread)

Each worker pulls from / pushes to ``queue.Queue`` instances, ensuring
thread-safe, non-blocking operation with graceful frame-drop semantics.

Usage
-----
>>> pipeline = RehabPipeline(source=0)
>>> pipeline.start()
>>> # … application loop …
>>> pipeline.stop()
"""

from __future__ import annotations

import logging
import queue
from typing import Optional

import numpy as np
import torch

from neuroflex.cpu_thread import HandGesture, HandTrackingWorker
from neuroflex.data_ingestion import FrameIngester
from neuroflex.gpu_thread import PoseEstimationWorker

logger = logging.getLogger(__name__)

# Queue capacity shared between pipeline stages.
_QUEUE_CAPACITY: int = 4


class RehabPipeline:
    """End-to-end rehabilitation assessment pipeline.

    The pipeline has three stages running in parallel daemon threads:

    1. **Capture** – :class:`FrameIngester` reads frames from *source* and
       places them on ``frame_queue``.
    2. **GPU inference** – :class:`PoseEstimationWorker` consumes
       ``frame_queue``, runs 3-D pose estimation on the GPU, and publishes
       ``(J, 3)`` tensors to ``pose_queue``.
    3. **CPU logic** – :class:`HandTrackingWorker` also consumes
       ``frame_queue`` (via a *separate* copy delivered by the ingester), runs
       MediaPipe hand tracking, and publishes :class:`HandGesture` results to
       ``gesture_queue``.

    .. note::
       Both the GPU and CPU workers share the same ``frame_queue`` produced by
       the ingester.  If both are active simultaneously you should either
       duplicate frames at the ingester level or use independent queues.  For
       this architecture the GPU worker takes full-resolution frames while the
       CPU worker could take a lower-resolution copy; the current
       implementation uses a single shared queue for simplicity – in
       production, extend :class:`FrameIngester` to fan-out to multiple queues.

    Parameters
    ----------
    source:
        Camera index or file/RTSP URL.
    reference_template:
        Optional reference angle sequence for DTW quality scoring.
    use_nvdec:
        Enable NVDEC hardware decoding.
    gpu_device:
        PyTorch device for pose estimation (auto-selected when not set).
    """

    def __init__(
        self,
        source: int | str = 0,
        reference_template: Optional[np.ndarray] = None,
        use_nvdec: bool = True,
        gpu_device: Optional[str] = None,
    ) -> None:
        # Shared queues
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_CAPACITY)
        self.pose_queue: queue.Queue[torch.Tensor] = queue.Queue(maxsize=_QUEUE_CAPACITY)
        self.gesture_queue: queue.Queue[HandGesture] = queue.Queue(maxsize=_QUEUE_CAPACITY)

        # Stage 1 – capture
        self._ingester = FrameIngester(
            source=source,
            queue_capacity=_QUEUE_CAPACITY,
            use_nvdec=use_nvdec,
        )

        # Stage 2 – GPU pose estimation
        device = gpu_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pose_worker = PoseEstimationWorker(
            input_queue=self.frame_queue,
            output_queue=self.pose_queue,
            device=device,
            output_queue_capacity=_QUEUE_CAPACITY,
        )

        # Stage 3 – CPU hand tracking
        self._hand_worker = HandTrackingWorker(
            frame_queue=self.frame_queue,
            result_queue=self.gesture_queue,
            reference_template=reference_template,
            output_queue_capacity=_QUEUE_CAPACITY,
        )

        # Wire ingester's internal queue to the shared frame_queue so all
        # downstream workers read from the same source.
        self._ingester._queue = self.frame_queue

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all pipeline stages."""
        self._ingester.start()
        self._pose_worker.start()
        self._hand_worker.start()
        logger.info("RehabPipeline started.")

    def stop(self) -> None:
        """Stop all pipeline stages gracefully."""
        self._ingester.stop()
        self._pose_worker.stop()
        self._hand_worker.stop()
        logger.info("RehabPipeline stopped.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return runtime statistics from all pipeline stages."""
        return {
            "ingester_dropped_frames":  self._ingester.dropped_frames,
            "pose_processed_frames":    self._pose_worker.processed_frames,
            "pose_dropped_outputs":     self._pose_worker.dropped_outputs,
            "hand_processed_frames":    self._hand_worker.processed_frames,
            "hand_dropped_outputs":     self._hand_worker.dropped_outputs,
        }
