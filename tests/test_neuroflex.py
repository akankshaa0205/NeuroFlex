from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neuroflex.kinematics import angle_between_vectors, compute_angle_series
from neuroflex.scoring import calculate_composite_score
from neuroflex.storage import atomic_write_json, export_session_csv


def test_angle_between_vectors_90_degrees():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    angle = angle_between_vectors(a, b)
    assert np.isclose(angle, 90.0, atol=1e-6)


def test_compute_angle_series_matches_expected_shape():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    angles = compute_angle_series(points)
    assert angles.shape == (points.shape[0] - 2,)
    assert np.all(angles >= 0)


def test_composite_score_formula_uses_prd_definition():
    score = calculate_composite_score(
        rom_ratio=0.8,
        dtw_distance=12.0,
        dtw_max=40.0,
        theta_error=8.0,
        theta_threshold=20.0,
        penalty=1.0,
    )

    expected = (1 / 3) * (
        0.8 + (1 - 12.0 / 40.0) + max(0.0, 1 - 8.0 / 20.0)
    )
    assert np.isclose(score, expected, atol=1e-9)


def test_atomic_write_json_and_csv_export(tmp_path):
    output_dir = tmp_path / "session"
    payload = {"schema_version": "1.0", "patient_id": "demo-patient", "score": 88.5}

    atomic_write_json(output_dir / "session.json", payload)
    export_session_csv(output_dir / "session.csv", payload)

    assert (output_dir / "session.json").exists()
    assert (output_dir / "session.csv").exists()

    loaded = json.loads((output_dir / "session.json").read_text())
    assert loaded["schema_version"] == "1.0"

    csv_text = (output_dir / "session.csv").read_text()
    assert "schema_version" in csv_text
    assert "demo-patient" in csv_text
