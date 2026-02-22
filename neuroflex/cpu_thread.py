"""
cpu_thread.py – CPU-side hand tracking, gesture recognition and kinematic
scoring worker.

Runs in a dedicated daemon thread.  Consumes raw BGR frames from the shared
*frame_queue*, applies MediaPipe Hands for lightweight hand-tracking, computes
kinematic metrics via :mod:`neuroflex.kinematics`, and compares exercise
repetitions against a reference template using
:mod:`neuroflex.dtw`.

Coding standards enforced
--------------------------
* No native Python ``for`` loops for mathematical / array work.
* Strict Python 3.9+ type hints throughout.
* Thread-safe queue access with graceful dropped-frame handling.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from neuroflex.dtw import dtw_score
from neuroflex.kinematics import compute_joint_angles, score_repetition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HandGesture:
    """Result of processing a single frame through the hand tracker."""

    landmarks: Optional[np.ndarray]
    """Shape ``(21, 3)`` float32 array of normalised (x, y, z) hand landmarks,
    or ``None`` if no hand is detected."""

    joint_angles: Optional[np.ndarray]
    """Shape ``(J,)`` joint angle array computed from *landmarks*, or ``None``."""

    gesture_label: str = "none"
    """High-level gesture label, e.g. ``"open_hand"``, ``"fist"``, ``"none"``."""

    quality_score: float = 0.0
    """DTW-based quality score in ``[0, 1]`` vs. the reference template."""


# Finger MCP-PIP-DIP triplets for the 21-landmark MediaPipe hand model.
# Index mapping: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
_FINGER_TRIPLETS: np.ndarray = np.array(
    [
        [1,  2,  3 ],  # Thumb  MCP-IP-TIP
        [5,  6,  7 ],  # Index  MCP-PIP-DIP
        [9,  10, 11],  # Middle MCP-PIP-DIP
        [13, 14, 15],  # Ring   MCP-PIP-DIP
        [17, 18, 19],  # Pinky  MCP-PIP-DIP
    ],
    dtype=np.int64,
)

# ---------------------------------------------------------------------------
# Gesture classifier (pure NumPy – no Python for loops)
# ---------------------------------------------------------------------------

def _classify_gesture(joint_angles: np.ndarray) -> str:
    """Return a human-readable gesture label from finger joint angles.

    Parameters
    ----------
    joint_angles:
        Float array of shape ``(5,)`` – one angle per finger (degrees).
    """
    # Vectorised threshold comparison across all 5 finger angles at once.
    extended = joint_angles > 150.0           # (5,) bool
    flexed   = joint_angles < 60.0            # (5,) bool

    if extended.all():
        return "open_hand"
    if flexed.all():
        return "fist"
    if extended[1] and not extended[0] and not extended[2:].any():
        return "point"
    if extended[1] and extended[2] and not extended[0] and not extended[3:].any():
        return "victory"
    if extended[0] and not extended[1:].any():
        return "thumbs_up"
    return "unknown"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class HandTrackingWorker:
    """Processes BGR frames with MediaPipe Hands in a background daemon thread.

    Parameters
    ----------
    frame_queue:
        Shared queue supplying raw BGR ``np.ndarray`` frames.
    result_queue:
        Output queue to which :class:`HandGesture` results are published.
    reference_template:
        Optional reference angle sequence ``(T_ref, 5)`` for DTW scoring.
    output_queue_capacity:
        Maximum result queue depth before oldest results are dropped.
    max_num_hands:
        Passed to ``mediapipe.solutions.hands.Hands``.
    min_detection_confidence:
        Passed to ``mediapipe.solutions.hands.Hands``.
    """

    def __init__(
        self,
        frame_queue: queue.Queue[np.ndarray],
        result_queue: queue.Queue[HandGesture],
        reference_template: Optional[np.ndarray] = None,
        output_queue_capacity: int = 4,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
    ) -> None:
        self._frame_queue = frame_queue
        self._result_queue = result_queue
        self._reference = reference_template
        self._output_queue_capacity = output_queue_capacity
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence

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
        """Start the background hand-tracking thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tracking_loop,
            name="HandTrackingWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info("HandTrackingWorker started")

    def stop(self) -> None:
        """Signal the thread to exit and wait for it to join."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info(
            "HandTrackingWorker stopped (processed=%d, dropped=%d)",
            self._processed_frames,
            self._dropped_outputs,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tracking_loop(self) -> None:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self._max_num_hands,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        try:
            while not self._stop_event.is_set():
                try:
                    frame: np.ndarray = self._frame_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                gesture = self._process_frame(frame, hands)
                self._processed_frames += 1
                self._enqueue_result(gesture)
        finally:
            hands.close()

    def _process_frame(
        self,
        frame: np.ndarray,
        hands: mp.solutions.hands.Hands,
    ) -> HandGesture:
        """Run MediaPipe on *frame* and return a :class:`HandGesture`."""
        # MediaPipe expects RGB; convert without a Python loop (OpenCV C++).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return HandGesture(landmarks=None, joint_angles=None)

        # Take the first detected hand.
        raw = results.multi_hand_landmarks[0]

        # MediaPipe's NormalizedLandmarkList has no direct NumPy export; the
        # one-time protobuf-field extraction below is an unavoidable API
        # boundary.  All subsequent processing is fully vectorised.
        landmarks: np.ndarray = np.array(
            [[lm.x, lm.y, lm.z] for lm in raw.landmark], dtype=np.float32
        )  # (21, 3)

        joint_angles = compute_joint_angles(landmarks, _FINGER_TRIPLETS)  # (5,)
        label = _classify_gesture(joint_angles)

        quality: float = 0.0
        if self._reference is not None:
            quality = dtw_score(joint_angles[np.newaxis, :], self._reference)

        return HandGesture(
            landmarks=landmarks,
            joint_angles=joint_angles,
            gesture_label=label,
            quality_score=quality,
        )

    def _enqueue_result(self, result: HandGesture) -> None:
        if self._result_queue.qsize() >= self._output_queue_capacity:
            try:
                self._result_queue.get_nowait()
                self._dropped_outputs += 1
            except queue.Empty:
                pass
        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            self._dropped_outputs += 1
