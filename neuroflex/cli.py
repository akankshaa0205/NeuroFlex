from __future__ import annotations

import argparse
from pathlib import Path

from .exercises import default_exercise_library
from .pipeline import PipelineConfig, RehabPipeline
from .session import SessionRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroFlex rehabilitation assessment CLI")
    parser.add_argument("--demo", action="store_true", help="Print a sample score and exercise library overview.")
    parser.add_argument("--save-session", type=str, help="Path to save a sample session JSON payload.")
    parser.add_argument("--patient-id", default="demo-patient", help="Patient identifier for exported session data.")
    parser.add_argument("--live", action="store_true", help="Run a short live session from the configured camera source.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for the real webcam capture path.")
    parser.add_argument("--frames", type=int, default=0, help="Frames to process; use 0 to run until you press q.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.live:
        print(f"Starting NeuroFlex webcam on camera {args.camera_index}...", flush=True)
        runner = SessionRunner(source=args.camera_index, width=640, height=480)
        stream = getattr(runner, "stream", None)
        if stream is not None and not stream.is_open():
            raise RuntimeError(
                f"Camera {args.camera_index} could not be opened. "
                "Close other camera apps or try --camera-index 1."
            )
        states = runner.run_live(max_frames=args.frames, display=True)
        for state in states:
            print(f"status={state['status']} score={state['score']:.2f} joints={state['joint_count']}")
        return

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

    print("NeuroFlex ready. Use --demo, --save-session <path>, or --live --camera-index 0 to run a short live capture session.")


if __name__ == "__main__":
    main()
