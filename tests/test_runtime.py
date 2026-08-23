from __future__ import annotations

import numpy as np

from neuroflex.camera import FrameStream
from neuroflex.pose import SimplePoseEstimator
from neuroflex.session import SessionRunner
from neuroflex.ui import OverlayRenderer


def test_frame_stream_generates_test_frame():
    stream = FrameStream(source="synthetic", size=(64, 64))
    frame = stream.read()

    assert frame is not None
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8


def test_pose_estimator_returns_33_joint_pose():
    estimator = SimplePoseEstimator()
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    pose = estimator.estimate(frame)

    assert pose.joints.shape == (33, 3)
    assert np.all(np.isfinite(pose.joints))


def test_overlay_renderer_keeps_frame_shape():
    renderer = OverlayRenderer()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    pose = SimplePoseEstimator().estimate(frame)
    rendered = renderer.render(frame, pose, score=82.4, status="running")

    assert rendered.shape == frame.shape
    assert rendered.dtype == frame.dtype


def test_session_runner_reports_session_state():
    runner = SessionRunner(source="synthetic", width=96, height=96)
    state = runner.run_once()

    assert state["status"] in {"idle", "running", "paused"}
    assert 0.0 <= state["score"] <= 100.0
    assert state["joint_count"] == 33
