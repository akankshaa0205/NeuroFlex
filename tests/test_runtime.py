from __future__ import annotations

import sys

import numpy as np

from neuroflex import cli
from neuroflex.camera import FrameStream
from neuroflex.pose import SimplePoseEstimator, create_pose_estimator
from neuroflex.session import SessionRunner
from neuroflex.ui import OverlayRenderer

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


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
    assert pose.source == "synthetic"


def test_synthetic_pose_estimator_is_used_for_synthetic_stream():
    estimator = create_pose_estimator(prefer_real=False)
    assert isinstance(estimator, SimplePoseEstimator)


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


def test_session_runner_live_loop_runs_fixed_frames():
    runner = SessionRunner(source="synthetic", width=96, height=96)
    state = runner.run_live(max_frames=2, display=False)

    assert len(state) == 2
    assert all(item["joint_count"] == 33 for item in state)
    assert all(0.0 <= item["score"] <= 100.0 for item in state)


def test_session_runner_live_loop_handles_missing_gui(monkeypatch):
    runner = SessionRunner(source="synthetic", width=96, height=96)

    if cv2 is not None:
        monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: (_ for _ in ()).throw(cv2.error("gui backend missing")))

    state = runner.run_live(max_frames=2, display=True)

    assert len(state) == 2
    assert all(item["joint_count"] == 33 for item in state)


def test_cli_live_command_uses_display_loop(monkeypatch):
    seen = {}

    class DummyRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run_live(self, max_frames, display):
            seen["max_frames"] = max_frames
            seen["display"] = display
            return [{"status": "running", "score": 50.0, "joint_count": 33}]

    monkeypatch.setattr(cli, "SessionRunner", DummyRunner)
    monkeypatch.setattr(sys, "argv", ["neuroflex", "--live", "--camera-index", "0", "--frames", "3"])

    cli.main()

    assert seen == {"max_frames": 3, "display": True}


def test_cli_demo_saves_session(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "session.json"
    monkeypatch.setattr(sys, "argv", ["neuroflex", "--demo", "--save-session", str(output_path)])

    cli.main()

    assert output_path.exists()
    output = capsys.readouterr().out
    assert "Exercise library size:" in output
    assert f"Session saved to: {output_path}" in output
