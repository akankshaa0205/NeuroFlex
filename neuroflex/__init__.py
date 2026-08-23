"""NeuroFlex package."""

from .camera import FrameStream
from .exercises import ExerciseDefinition, default_exercise_library, list_exercise_names
from .kinematics import angle_between_vectors, compute_angle_series
from .pipeline import PipelineConfig, RehabPipeline, SessionMetrics
from .pose import Pose, SimplePoseEstimator
from .scoring import calculate_composite_score
from .session import SessionRunner
from .storage import atomic_write_json, export_session_csv
from .ui import OverlayRenderer

__all__ = [
    "ExerciseDefinition",
    "FrameStream",
    "OverlayRenderer",
    "PipelineConfig",
    "Pose",
    "RehabPipeline",
    "SessionMetrics",
    "SessionRunner",
    "SimplePoseEstimator",
    "angle_between_vectors",
    "calculate_composite_score",
    "compute_angle_series",
    "default_exercise_library",
    "list_exercise_names",
    "atomic_write_json",
    "export_session_csv",
]

__version__ = "0.1.0"
