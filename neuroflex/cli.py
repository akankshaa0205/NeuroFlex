from __future__ import annotations

import argparse
from pathlib import Path

from .exercises import default_exercise_library
from .pipeline import PipelineConfig, RehabPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroFlex rehabilitation assessment CLI")
    parser.add_argument("--demo", action="store_true", help="Print a sample score and exercise library overview.")
    parser.add_argument("--save-session", type=str, help="Path to save a sample session JSON payload.")
    parser.add_argument("--patient-id", default="demo-patient", help="Patient identifier for exported session data.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.demo or args.save_session:
        pipeline = RehabPipeline(config=PipelineConfig())
        metrics = pipeline.assess_session(
            rom_ratio=0.82,
            dtw_distance=12.0,
            theta_error=6.0,
            exercise_name="shoulder_flexion",
        )
        library = default_exercise_library()
        print(f"Exercise library size: {len(library)}")
        print(f"Composite score: {metrics.composite_score:.3f}")
        print("Available exercises:", ", ".join(sorted(library.keys())))

        if args.save_session:
            session_path = Path(args.save_session)
            pipeline.save_session(session_path, metrics, patient_id=args.patient_id)
            print(f"Session saved to: {session_path}")
        return

    print("NeuroFlex ready. Use --demo to print a sample assessment or --save-session <path> to export JSON.")


if __name__ == "__main__":
    main()
