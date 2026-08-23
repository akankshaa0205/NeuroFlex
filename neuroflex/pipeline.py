from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .exercises import ExerciseDefinition, default_exercise_library
from .scoring import calculate_composite_score
from .storage import atomic_write_json


@dataclass
class SessionMetrics:
    rom_ratio: float
    dtw_distance: float
    theta_error: float
    composite_score: float
    schema_version: str = "1.0"


@dataclass
class PipelineConfig:
    dtw_max: float = 40.0
    theta_threshold: float = 15.0
    penalty: float = 1.0
    exercise_name: str = "shoulder_flexion"


class RehabPipeline:
    """A lightweight session pipeline aligned with the PRD architecture.

    The implementation intentionally remains deterministic and transparent; it does not
    substitute a neural model for motion grading.
    """

    def __init__(self, config: PipelineConfig | None = None, exercise_library: dict[str, ExerciseDefinition] | None = None):
        self.config = config or PipelineConfig()
        self.exercise_library = exercise_library or default_exercise_library()

    def get_exercise(self, name: str | None = None) -> ExerciseDefinition:
        lookup = name or self.config.exercise_name
        if lookup not in self.exercise_library:
            raise KeyError(f"Unknown exercise '{lookup}'.")
        return self.exercise_library[lookup]

    def assess_session(self, rom_ratio: float, dtw_distance: float, theta_error: float, exercise_name: str | None = None) -> SessionMetrics:
        exercise = self.get_exercise(exercise_name)
        theta_threshold = exercise.theta_threshold if exercise.theta_threshold else self.config.theta_threshold
        score = calculate_composite_score(
            rom_ratio=rom_ratio,
            dtw_distance=dtw_distance,
            dtw_max=self.config.dtw_max,
            theta_error=theta_error,
            theta_threshold=theta_threshold,
            penalty=self.config.penalty,
        )
        return SessionMetrics(
            rom_ratio=float(rom_ratio),
            dtw_distance=float(dtw_distance),
            theta_error=float(theta_error),
            composite_score=float(score),
            schema_version="1.0",
        )

    def save_session(self, session_path: str | Path, metrics: SessionMetrics, patient_id: str = "demo-patient") -> Path:
        payload: dict[str, Any] = {
            "schema_version": metrics.schema_version,
            "patient_id": patient_id,
            "rom_ratio": metrics.rom_ratio,
            "dtw_distance": metrics.dtw_distance,
            "theta_error": metrics.theta_error,
            "composite_score": metrics.composite_score,
        }
        return atomic_write_json(Path(session_path), payload)

    def record_baseline(self, movement_profile: np.ndarray) -> np.ndarray:
        """Accept a 1D angle trajectory and return it as the patient baseline."""
        baseline = np.asarray(movement_profile, dtype=float)
        if baseline.ndim != 1:
            raise ValueError("Baseline movement profile must be a 1D NumPy array.")
        return baseline
