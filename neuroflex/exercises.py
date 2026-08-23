from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class ExerciseDefinition:
    """A rehabilitation exercise definition used by the clinician library."""

    name: str
    category: str
    target_rom_percent: float = 100.0
    rep_count: int = 10
    session_duration_seconds: int = 60
    theta_threshold: float = 15.0
    description: str = ""


def default_exercise_library() -> dict[str, ExerciseDefinition]:
    """Return the default exercise set covering upper and lower limb rehab."""
    exercises = {
        "shoulder_flexion": ExerciseDefinition(
            name="Shoulder Flexion",
            category="Upper Limb",
            target_rom_percent=100.0,
            rep_count=10,
            session_duration_seconds=60,
            theta_threshold=15.0,
            description="Raise the arm forward in the sagittal plane.",
        ),
        "shoulder_abduction": ExerciseDefinition(
            name="Shoulder Abduction",
            category="Upper Limb",
            target_rom_percent=100.0,
            rep_count=10,
            session_duration_seconds=60,
            theta_threshold=18.0,
            description="Lift the arm sideways away from the body.",
        ),
        "elbow_flexion": ExerciseDefinition(
            name="Elbow Flexion",
            category="Upper Limb",
            target_rom_percent=100.0,
            rep_count=12,
            session_duration_seconds=60,
            theta_threshold=12.0,
            description="Bend the elbow from straight to flexed.",
        ),
        "wrist_extension": ExerciseDefinition(
            name="Wrist Extension",
            category="Upper Limb",
            target_rom_percent=100.0,
            rep_count=12,
            session_duration_seconds=45,
            theta_threshold=10.0,
            description="Extend the wrist backward with the arm supported.",
        ),
        "grip_reach": ExerciseDefinition(
            name="Grip Reach",
            category="Upper Limb",
            target_rom_percent=100.0,
            rep_count=8,
            session_duration_seconds=45,
            theta_threshold=12.0,
            description="Reach toward a target while maintaining a functional grip.",
        ),
        "knee_extension": ExerciseDefinition(
            name="Knee Extension",
            category="Lower Limb",
            target_rom_percent=100.0,
            rep_count=10,
            session_duration_seconds=60,
            theta_threshold=15.0,
            description="Straighten the knee from a bent position.",
        ),
        "hip_flexion": ExerciseDefinition(
            name="Hip Flexion",
            category="Lower Limb",
            target_rom_percent=100.0,
            rep_count=8,
            session_duration_seconds=50,
            theta_threshold=16.0,
            description="Lift the leg forward without rotating the pelvis.",
        ),
        "ankle_dorsiflexion": ExerciseDefinition(
            name="Ankle Dorsiflexion",
            category="Lower Limb",
            target_rom_percent=100.0,
            rep_count=12,
            session_duration_seconds=45,
            theta_threshold=10.0,
            description="Pull the foot upward toward the shin.",
        ),
        "sit_to_stand": ExerciseDefinition(
            name="Sit to Stand",
            category="Functional",
            target_rom_percent=100.0,
            rep_count=8,
            session_duration_seconds=90,
            theta_threshold=18.0,
            description="Stand up and sit down with control.",
        ),
        "step_up": ExerciseDefinition(
            name="Step Up",
            category="Functional",
            target_rom_percent=100.0,
            rep_count=10,
            session_duration_seconds=75,
            theta_threshold=18.0,
            description="Step up onto a stable surface with controlled alignment.",
        ),
    }
    return exercises


def list_exercise_names(exercises: Iterable[ExerciseDefinition] | None = None) -> list[str]:
    library = exercises if exercises is not None else default_exercise_library().values()
    return [exercise.name for exercise in library]
