import argparse
from pathlib import Path

from cafe.pipeline import run_cafe


def main():
    parser = argparse.ArgumentParser(description="Run the CAFE pipeline on one video.")
    parser.add_argument("--video", required=True, help="Video ID or video path")
    parser.add_argument("--config", default="config.yaml", help="Configuration YAML path")
    parser.add_argument("--out", default="results/runs", help="Output directory")
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Reserved for saving additional visual artifacts",
    )
    args = parser.parse_args()

    result = run_cafe(
        args.video,
        output_dir=Path(args.out),
        config_path=args.config,
        save_artifacts=args.save_artifacts,
    )

    print()
    print("=== CAFE RESULT ===")
    print(f"Prediction: {result.label} (score {result.score:.3f})")

    for i, candidate in enumerate(result.candidates, start=1):
        print(
            f"Candidate {i}: "
            f"{candidate['candidate']['cue']} "
            f"frames {candidate['candidate']['interval'][0]}-"
            f"{candidate['candidate']['interval'][1]} | "
            f"Delta={candidate['candidate_effect']:.4f} | "
            f"Tau={candidate['tau']:.4f} | "
            f"Supported={candidate['supported']}"
        )

    print()
    print(result.rendered_explanation)
    print()
    print(f"JSON: {result.output_path}")
    print(f"Total time: {result.timings['total_sec']:.2f}s")


if __name__ == "__main__":
    main()
