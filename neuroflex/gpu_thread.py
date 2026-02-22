"""
gpu_thread.py – GPU-side 3D Human Pose Estimation worker.

Runs in a dedicated daemon thread.  Consumes raw BGR frames from the shared
*input_queue*, runs inference on CUDA, and publishes landmark tensors to the
shared *output_queue*.  Frame drops are handled gracefully; the thread never
crashes on a temporarily empty queue.

Coding standards enforced
--------------------------
* All tensor operations stay on the GPU until the result is explicitly
  transferred to CPU for the CPU-thread consumer.
* No native Python ``for`` loops for mathematical / tensor work.
* Strict Python 3.9+ type hints throughout.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight model stub
# ---------------------------------------------------------------------------

class _PoseEstimatorModel(nn.Module):
    """Lightweight MLP that maps a flattened image patch to 3-D joint
    coordinates.

    In production this would be replaced by a full spatio-temporal model
    (e.g. MotionBERT, VideoPose3D).  The architecture here is intentionally
    minimal so that the module can be tested without a GPU or a large
    pre-trained checkpoint.

    Input:  ``(B, 3, H, W)`` float32 tensor on the target device.
    Output: ``(B, J, 3)`` float32 tensor of 3-D joint positions.
    """

    NUM_JOINTS: int = 17  # COCO-style skeleton

    def __init__(self, input_height: int = 256, input_width: int = 192) -> None:
        super().__init__()
        in_features = 3 * input_height * input_width
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.NUM_JOINTS * 3),
        )
        self._input_height = input_height
        self._input_width = input_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) → (B,J,3)
        flat = self.backbone(x)                          # (B, J*3)
        return flat.view(-1, self.NUM_JOINTS, 3)         # (B, J, 3)


# ---------------------------------------------------------------------------
# Public worker
# ---------------------------------------------------------------------------

class PoseEstimationWorker:
    """Wraps the pose model in a background daemon thread.

    Parameters
    ----------
    input_queue:
        Shared queue that supplies raw ``np.ndarray`` BGR frames
        (produced by :class:`~neuroflex.data_ingestion.FrameIngester`).
    output_queue:
        Shared queue to which ``torch.Tensor`` pose results (CPU, float32,
        shape ``(J, 3)``) are published.
    device:
        PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    input_height / input_width:
        Spatial dimensions the model expects.
    output_queue_capacity:
        Drop old results when the output queue exceeds this size.
    """

    def __init__(
        self,
        input_queue: queue.Queue[np.ndarray],
        output_queue: queue.Queue[torch.Tensor],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        input_height: int = 256,
        input_width: int = 192,
        output_queue_capacity: int = 4,
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._device = torch.device(device)
        self._input_height = input_height
        self._input_width = input_width
        self._output_queue_capacity = output_queue_capacity

        self._model: _PoseEstimatorModel = _PoseEstimatorModel(
            input_height, input_width
        ).to(self._device)
        self._model.eval()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processed_frames: int = 0
        self._dropped_outputs: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def processed_frames(self) -> int:
        return self._processed_frames

    @property
    def dropped_outputs(self) -> int:
        return self._dropped_outputs

    def start(self) -> None:
        """Start the background inference thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="PoseEstimationWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info("PoseEstimationWorker started on device=%s", self._device)

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to join."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info(
            "PoseEstimationWorker stopped (processed=%d, dropped_out=%d)",
            self._processed_frames,
            self._dropped_outputs,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """BGR uint8 ``(H, W, 3)`` → normalised float32 CUDA ``(1, 3, H, W)``."""
        # Resize without a Python loop – use OpenCV's vectorised C++ backend.
        resized = _resize_frame(frame, self._input_height, self._input_width)
        # Convert to float tensor; CHW layout; add batch dim.
        tensor = (
            torch.from_numpy(resized)          # (H, W, 3) uint8
            .to(dtype=torch.float32)
            .permute(2, 0, 1)                  # (3, H, W)
            .unsqueeze(0)                      # (1, 3, H, W)
            .div_(255.0)                       # normalise to [0, 1] in-place
            .to(self._device, non_blocking=True)
        )
        return tensor

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame: np.ndarray = self._input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            with torch.no_grad():
                tensor = self._preprocess(frame)          # (1, 3, H, W) on GPU
                poses: torch.Tensor = self._model(tensor) # (1, J, 3) on GPU
                # Transfer to CPU for the CPU-thread consumer; keep contiguous.
                result = poses.squeeze(0).cpu().contiguous()  # (J, 3) CPU

            self._processed_frames += 1
            self._enqueue_output(result)

    def _enqueue_output(self, result: torch.Tensor) -> None:
        if self._output_queue.qsize() >= self._output_queue_capacity:
            try:
                self._output_queue.get_nowait()
                self._dropped_outputs += 1
            except queue.Empty:
                pass
        try:
            self._output_queue.put_nowait(result)
        except queue.Full:
            self._dropped_outputs += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize *frame* to ``(height, width)`` using OpenCV (vectorised C++)."""
    import cv2
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
