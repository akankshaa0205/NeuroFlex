from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neuroflex.exercises import default_exercise_library, list_exercise_names
from neuroflex.pipeline import PipelineConfig, RehabPipeline


def test_default_exercise_library_has_required_entries():
    library = default_exercise_library()
    assert "shoulder_flexion" in library
    assert "knee_extension" in library
    assert len(library) >= 10


def test_exercise_names_are_available():
    names = list_exercise_names()
    assert "Shoulder Flexion" in names
    assert "Knee Extension" in names


def test_pipeline_assess_session_and_save(tmp_path):
    pipeline = RehabPipeline(config=PipelineConfig(dtw_max=50.0, theta_threshold=20.0))
    metrics = pipeline.assess_session(
        rom_ratio=0.8,
        dtw_distance=15.0,
        theta_error=10.0,
        exercise_name="elbow_flexion",
    )

    assert metrics.composite_score > 0
    assert metrics.schema_version == "1.0"

    out_path = tmp_path / "session.json"
    saved = pipeline.save_session(out_path, metrics, patient_id="pt-001")
    assert saved.exists()

    data = json.loads(saved.read_text())
    assert data["patient_id"] == "pt-001"
    assert "composite_score" in data


def test_pipeline_baseline_recording_is_1d_and_deterministic():
    pipeline = RehabPipeline()
    baseline = np.array([0.0, 10.0, 35.0, 60.0, 80.0], dtype=float)
    result = pipeline.record_baseline(baseline)
    assert result.shape == baseline.shape
    np.testing.assert_array_equal(result, baseline)


def test_pipeline_baseline_recording_rejects_non_1d_input():
    with np.testing.assert_raises(ValueError):
        RehabPipeline().record_baseline(np.zeros((2, 2)))
