import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cafe.candidates import generate_candidates, generate_placebo_candidates
from cafe.controls import generate_controls, apply_control
from cafe.detector.base_detector import BaseDetector
from cafe.interventions import apply_intervention
from cafe.landmarks import load_or_build_landmark_cache
from cafe.preprocessing import load_or_build_cache
from cafe.utils.config import load_config
from cafe.verification import compute_p_value, compute_threshold


def _setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _load_video_data(video_id, video_path):
    cache = load_or_build_cache(
        video_id,
        video_path=video_path,
    )
    frames = cache["faces"]
    landmarks = load_or_build_landmark_cache(video_id)
    frame_index = cache["index"]

    return frames, landmarks, frame_index

def _measure_candidate(
    frames,
    landmarks,
    detector,
    original_score,
    candidate,
    strength,
):
    modified_frames = apply_intervention(
        frames,
        landmarks,
        candidate["cue"],
        tuple(map(int, candidate["sampled_interval"])),
        strength=strength,
    )

    modified_score = float(detector.score_video(modified_frames))
    delta = float(original_score - modified_score)

    del modified_frames

    return modified_score, delta

def _measure_controls(
    frames,
    landmarks,
    detector,
    original_score,
    candidate,
    n_controls,
    rng,
):
    candidate_for_controls = dict(candidate)
    candidate_for_controls["interval"] = tuple(
        map(int, candidate["sampled_interval"])
    )

    controls = generate_controls(
        {"n_frames": len(frames)},
        candidate_for_controls,
        n_controls,
        rng,
    )

    effects = []
    records = []
    modified_batches = []

    for control in controls:
        modified_batches.append(
            apply_control(
                frames,
                landmarks,
                control,
            )
        )

    modified_scores = detector.score_frame_batches(modified_batches)

    for control, modified_score in zip(controls, modified_scores):
        modified_score = float(modified_score)
        delta = float(original_score - modified_score)

        effects.append(delta)
        records.append(
            {
                "control": control,
                "original_score": float(original_score),
                "modified_score": modified_score,
                "delta": delta,
            }
        )

    del modified_batches

    return effects, records


def _run_condition(
    video_id,
    video_path,
    label,
    method,
    condition,
    config,
    detector,
):
    start_total = time.perf_counter()

    frames, landmarks, frame_index = _load_video_data(video_id, video_path)

    if len(frames) == 0:
        return {
            "video_id": video_id,
            "label": label,
            "method": method,
            "condition": condition,
            "status": "no-candidates",
            "candidates": [],
            "timings": {
                "total_sec": time.perf_counter() - start_total,
            },
        }

    original_score = float(detector.score_video(frames))

    if condition == "real":
        candidates = generate_candidates(
            video_id,
            config=config,
            detector=detector,
        )
    elif condition == "authentic":
        rng = random.Random(
            int(config["seed"]) + sum(
                (i + 1) * ord(ch)
                for i, ch in enumerate(video_id)
            ) + 100000
        )
        candidates = generate_placebo_candidates(
            len(frames),
            int(config["n_candidates"]),
            rng,
            config=config,
            frame_index=frame_index,
        )
    elif condition == "placebo":
        rng = random.Random(
            int(config["seed"]) + sum(
                (i + 1) * ord(ch)
                for i, ch in enumerate(video_id)
            )
        )
        candidates = generate_placebo_candidates(
            len(frames),
            int(config["n_candidates"]),
            rng,
            config=config,
            frame_index=frame_index,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
    if not candidates:
        return {
            "video_id": video_id,
            "label": label,
            "method": method,
            "condition": condition,
            "status": "no-candidates",
            "original_score": original_score,
            "candidates": [],
            "timings": {
                "total_sec": time.perf_counter() - start_total,
            },
        }

    records = []
    n_controls = int(config["n_controls"])
    tau_percentile = float(config["tau_percentile"])
    seed = int(config["seed"])

    for candidate_index, candidate in enumerate(
        tqdm(
            candidates,
            desc=f"{video_id} | {condition}",
            leave=False,
        )
    ):
        rng = random.Random(
            seed + candidate_index
        )

        control_effects, control_records = _measure_controls(
            frames,
            landmarks,
            detector,
            original_score,
            candidate,
            n_controls,
            rng,
        )

        tau = compute_threshold(
            control_effects,
            percentile=tau_percentile,
        )

        modified_score, delta = _measure_candidate(
            frames,
            landmarks,
            detector,
            original_score,
            candidate,
            config["intervention"],
        )
        p_value = compute_p_value(
            delta,
            control_effects,
        )

        supported = bool(delta > tau)

        records.append(
            {
                "candidate": candidate,
                "original_score": original_score,
                "modified_score": modified_score,
                "candidate_effect": delta,
                "control_effects": [
                    float(x) for x in control_effects
                ],
                "tau_percentile": tau_percentile,
                "tau": float(tau),
                "p_value": float(p_value),
                "supported": supported,
                "control_records": control_records,
            }
        )

    return {
        "video_id": video_id,
        "label": label,
        "method": method,
        "condition": condition,
        "status": "success",
        "original_score": original_score,
        "candidates": records,
        "timings": {
            "total_sec": time.perf_counter() - start_total,
        },
    }


def _flatten_summary(record):
    rows = []

    for item in record.get("candidates", []):
        candidate = item["candidate"]
        t1, t2 = map(int, candidate["interval"])

        rows.append(
            {
                "video_id": record["video_id"],
                "label": record["label"],
                "method": record["method"],
                "condition": record["condition"],
                "cue": candidate["cue"],
                "t1": t1,
                "t2": t2,
                "delta": item.get("candidate_effect"),
                "tau": item.get("tau"),
                "p": item.get("p_value"),
                "supported": item.get("supported"),
                "status": record["status"],
            }
        )

    if not rows:
        rows.append(
            {
                "video_id": record["video_id"],
                "label": record["label"],
                "method": record["method"],
                "condition": record["condition"],
                "cue": "",
                "t1": "",
                "t2": "",
                "delta": "",
                "tau": "",
                "p": "",
                "supported": "",
                "status": record["status"],
            }
        )

    return rows


def _save_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run CAFE batch experiments."
    )
    parser.add_argument(
        "--manifest",
        default="data/manifest.csv",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
    )
    parser.add_argument(
        "--out",
        default="results/runs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N videos.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    manifest = pd.read_csv(args.manifest)

    if args.limit is not None:
        manifest = manifest.head(args.limit)

    output_dir = Path(args.out)
    log_path = Path("results/logs") / (
        f"batch_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )

    _setup_logging(log_path)

    detector = BaseDetector()

    all_rows = []
    failures = []

    conditions = ("real", "placebo", "authentic")

    total_runs = len(manifest) * len(conditions)

    with tqdm(
        total=total_runs,
        desc="Batch",
    ) as progress:

        for _, row in manifest.iterrows():
            video_id = str(row["video_id"])
            video_path = Path(str(row["path"]))
            label = str(row["label"])
            method = str(row["method"])

            for condition in conditions:
                output_path = (
                    output_dir
                    / f"{video_id}__{condition}.json"
                )

                if output_path.exists():
                    try:
                        record = json.loads(
                            output_path.read_text(
                                encoding="utf-8"
                            )
                        )
                        all_rows.extend(
                            _flatten_summary(record)
                        )
                        progress.update(1)
                        continue
                    except Exception:
                        logging.warning(
                            "Invalid existing result: %s",
                            output_path,
                        )

                try:
                    logging.info(
                        "Starting %s | %s",
                        video_id,
                        condition,
                    )

                    record = _run_condition(
                        video_id,
                        video_path,
                        label,
                        method,
                        condition,
                        config,
                        detector,
                    )

                    _save_json(output_path, record)

                    all_rows.extend(
                        _flatten_summary(record)
                    )

                    logging.info(
                        "Completed %s | %s | status=%s",
                        video_id,
                        condition,
                        record["status"],
                    )

                except Exception as exc:
                    logging.exception(
                        "Failed %s | %s",
                        video_id,
                        condition,
                    )

                    failure = {
                        "video_id": video_id,
                        "label": label,
                        "method": method,
                        "condition": condition,
                        "status": "error",
                        "error": repr(exc),
                    }

                    _save_json(output_path, failure)

                    all_rows.extend(
                        _flatten_summary(failure)
                    )

                    failures.append(failure)

                finally:
                    progress.update(1)

    summary_path = Path("results/batch_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "label",
        "method",
        "condition",
        "cue",
        "t1",
        "t2",
        "delta",
        "tau",
        "p",
        "supported",
        "status",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=== BATCH COMPLETE ===")
    print(f"Videos: {len(manifest)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Run units: {total_runs}")
    print(f"Rows: {len(all_rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Summary: {summary_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()












