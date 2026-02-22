"""
Tests for neuroflex.gpu_thread – PoseEstimationWorker.

All tests run on CPU to avoid requiring CUDA hardware in CI.
"""

from __future__ import annotations

import queue
import time
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest
import torch

from neuroflex.gpu_thread import PoseEstimationWorker, _PoseEstimatorModel, _resize_frame


class TestPoseEstimatorModel:
    """Unit tests for the pose model forward pass."""

    def test_output_shape(self) -> None:
        model = _PoseEstimatorModel(input_height=64, input_width=48)
        model.eval()
        x = torch.zeros(2, 3, 64, 48)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, _PoseEstimatorModel.NUM_JOINTS, 3)

    def test_output_is_finite(self) -> None:
        model = _PoseEstimatorModel(input_height=64, input_width=48)
        model.eval()
        rng = torch.Generator()
        rng.manual_seed(0)
        x = torch.rand(1, 3, 64, 48, generator=rng)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


class TestResizeFrame:
    def test_output_shape(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = _resize_frame(frame, height=64, width=48)
        assert resized.shape == (64, 48, 3)


class TestPoseEstimationWorker:
    """Integration tests for the worker thread (CPU-only)."""

    def _make_worker(
        self,
        input_q: Optional[queue.Queue] = None,
        output_q: Optional[queue.Queue] = None,
    ) -> PoseEstimationWorker:
        in_q = input_q or queue.Queue(maxsize=4)
        out_q = output_q or queue.Queue(maxsize=4)
        return PoseEstimationWorker(
            input_queue=in_q,
            output_queue=out_q,
            device="cpu",
            input_height=64,
            input_width=48,
            output_queue_capacity=4,
        )

    def test_start_stop(self) -> None:
        worker = self._make_worker()
        worker.start()
        time.sleep(0.05)
        worker.stop()
        assert not worker._thread.is_alive()

    def test_processes_frame(self) -> None:
        in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        out_q: queue.Queue[torch.Tensor] = queue.Queue(maxsize=4)
        worker = self._make_worker(in_q, out_q)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        worker.start()
        in_q.put(frame)
        # Wait for a result.
        result = None
        deadline = time.time() + 5.0
        while result is None and time.time() < deadline:
            try:
                result = out_q.get(timeout=0.1)
            except queue.Empty:
                pass
        worker.stop()

        assert result is not None
        assert result.shape == (_PoseEstimatorModel.NUM_JOINTS, 3)
        assert result.device.type == "cpu"

    def test_dropped_outputs_counted(self) -> None:
        """When the output queue is full, results should be dropped, not blocked."""
        in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        out_q: queue.Queue[torch.Tensor] = queue.Queue(maxsize=1)

        worker = PoseEstimationWorker(
            input_queue=in_q,
            output_queue=out_q,
            device="cpu",
            input_height=32,
            input_width=24,
            output_queue_capacity=1,
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Fill the input queue with many frames.
        for _ in range(8):
            in_q.put(frame)

        worker.start()
        time.sleep(1.0)
        worker.stop()

        assert worker.processed_frames > 0
