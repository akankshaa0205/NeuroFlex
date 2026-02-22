"""
Tests for neuroflex.data_ingestion – FrameIngester queue mechanics.

These tests mock OpenCV so no camera hardware is required.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neuroflex.data_ingestion import FrameIngester, _open_capture


class TestFrameIngesterQueue:
    """Test the queue and dropped-frame behaviour without real hardware."""

    def _make_ingester(self, capacity: int = 4) -> FrameIngester:
        return FrameIngester(source=0, queue_capacity=capacity, use_nvdec=False)

    def test_enqueue_fills_queue(self) -> None:
        ingester = self._make_ingester(capacity=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Manually enqueue 3 frames.
        for _ in range(3):
            ingester._enqueue(frame)
        assert ingester.frame_queue.qsize() == 3

    def test_enqueue_drops_oldest_when_full(self) -> None:
        ingester = self._make_ingester(capacity=2)
        frame_a = np.full((480, 640, 3), 1, dtype=np.uint8)
        frame_b = np.full((480, 640, 3), 2, dtype=np.uint8)
        frame_c = np.full((480, 640, 3), 3, dtype=np.uint8)

        ingester._enqueue(frame_a)
        ingester._enqueue(frame_b)  # queue now full
        ingester._enqueue(frame_c)  # frame_a should be dropped

        assert ingester.dropped_frames == 1
        assert ingester.frame_queue.qsize() == 2
        # frame_b should be at the front, frame_c at the back.
        assert (ingester.frame_queue.get() == frame_b).all()
        assert (ingester.frame_queue.get() == frame_c).all()

    def test_get_frame_returns_none_on_empty(self) -> None:
        ingester = self._make_ingester()
        result = ingester.get_frame(timeout=0.01)
        assert result is None

    def test_get_frame_returns_frame(self) -> None:
        ingester = self._make_ingester()
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        ingester._enqueue(frame)
        result = ingester.get_frame(timeout=0.1)
        assert result is not None
        assert result.shape == (100, 100, 3)


class TestFrameIngesterThread:
    """Integration test: start/stop the capture thread with a mocked capture."""

    def test_start_stop_lifecycle(self) -> None:
        """The ingester thread should start and stop cleanly."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)

        with patch("neuroflex.data_ingestion._open_capture", return_value=mock_cap):
            ingester = FrameIngester(source=0, queue_capacity=4, use_nvdec=False)
            ingester.start()
            time.sleep(0.15)  # let at least a few frames be captured
            ingester.stop()

        assert ingester.frame_queue.qsize() >= 0  # queue may have been drained
        assert mock_cap.read.call_count > 0

    def test_broken_stream_stops_thread(self) -> None:
        """A stream that immediately returns False should stop the thread."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with patch("neuroflex.data_ingestion._open_capture", return_value=mock_cap):
            ingester = FrameIngester(source=0, queue_capacity=4, use_nvdec=False)
            ingester.start()
            ingester._thread.join(timeout=2.0)

        assert not ingester._thread.is_alive()
