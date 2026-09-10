import random

from cafe.candidates import generate_candidates
from cafe.detector.base_detector import BaseDetector
from cafe.preprocessing import load_or_build_cache
from cafe.utils.config import load_config
from cafe.verification import (
    build_control_distribution,
    save_verification_results,
    verify_candidate,
)


def main():
    config = load_config()
    video_id = "Deepfakes_000_003"

    detector = BaseDetector()
    candidates = generate_candidates(video_id)

    frames = load_or_build_cache(video_id)["faces"]
    original_score = float(detector.score_video(frames))

    candidate_records = []

    for candidate in candidates:
        effects, control_records = build_control_distribution(
            video_id,
            candidate,
            n_controls=config["n_controls"],
            rng=random.Random(config["seed"]),
            detector=detector,
            original_score=original_score,
        )

        record = verify_candidate(
            video_id,
            candidate,
            effects,
            detector=detector,
            tau_percentile=config["tau_percentile"],
            control_records=control_records,
            output_dir="results/verification",
            original_score=original_score,
        )

        candidate_records.append(record)

    save_verification_results(
        video_id,
        candidate_records,
        output_dir="results/verification",
    )

    print(f"video_id: {video_id}")
    print(f"candidates verified: {len(candidate_records)}")

    for i, record in enumerate(candidate_records, start=1):
        print(
            f"candidate {i}: "
            f"delta={record['candidate_effect']:.6f}, "
            f"tau={record['tau']:.6f}, "
            f"p={record['p_value']:.6f}, "
            f"supported={record['supported']}"
        )


if __name__ == "__main__":
    main()
