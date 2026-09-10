from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random
import time
from tqdm import tqdm

from cafe.candidates import generate_candidates
from cafe.controls import apply_control, generate_controls
from cafe.decision import decide, render_explanation
from cafe.detector.base_detector import BaseDetector
from cafe.landmarks import load_or_build_landmark_cache
from cafe.preprocessing import load_or_build_cache
from cafe.utils.config import load_config
from cafe.verification import (
    compute_p_value,
    compute_threshold,
)


@dataclass
class CafeResult:
    video_id: str
    label: str
    score: float
    decision: dict
    rendered_explanation: str
    candidates: list
    timings: dict
    output_path: str | None = None

    def to_dict(self):
        return asdict(self)


def _resolve_video(video_path_or_id):
    """
    Resolve either a cached video ID or a dataset/video path.
    """
    path = Path(video_path_or_id)

    if not path.exists():
        return str(video_path_or_id), None

    manifest_path = Path("data/manifest.csv")

    if manifest_path.exists():
        import csv

        target = path.resolve()

        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                manifest_path_value = Path(row["path"])
                try:
                    manifest_target = manifest_path_value.resolve()
                except OSError:
                    manifest_target = manifest_path_value.absolute()

                if manifest_target == target:
                    return row["video_id"], path

    return path.stem, path


def _load_video_data(video_id, video_path):
    cache = load_or_build_cache(
        video_id,
        video_path=video_path,
    )

    landmarks = load_or_build_landmark_cache(video_id)

    return cache, landmarks


def _measure_modified_score(detector, frames):
    return float(detector.score_video(frames))


def run_cafe(video_path_or_id, config=None, output_dir=None, config_path=None, save_artifacts=False):
    """
    Run the complete CAFE pipeline for one video.

    Algorithm order:
        preprocess
        -> detect
        -> generate candidates
        -> generate/measure controls and compute tau
        -> measure candidate effects
        -> decide
        -> render
    """
    total_start = time.perf_counter()

    if config is None:
        config = load_config(config_path) if config_path else load_config()

    video_id, video_path = _resolve_video(video_path_or_id)

    if output_dir is None:
        output_dir = Path(config["paths"]["results"]) / "runs"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    timings = {}

    start = time.perf_counter()
    cache, landmarks = _load_video_data(video_id, video_path)
    frames = cache["faces"]
    timings["preprocess_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    detector = BaseDetector()
    per_frame_scores = detector.score_frames(frames)
    original_score = float(per_frame_scores.mean())
    label = "DEEPFAKE" if original_score >= 0.5 else "REAL"
    timings["detect_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    candidates = generate_candidates(video_id, config=config)
    timings["candidate_generation_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    rng_seed = int(config["seed"])
    n_controls = int(config["n_controls"])
    tau_percentile = float(config["tau_percentile"])
    candidate_control_data = []

    for candidate_index, candidate in enumerate(candidates):
        candidate_for_controls = dict(candidate)
        candidate_for_controls["interval"] = tuple(
            map(int, candidate["sampled_interval"])
        )

        controls = generate_controls(
            {"n_frames": len(frames)},
            candidate_for_controls,
            n_controls,
            random.Random(rng_seed + candidate_index),
        )

        control_effects = []
        control_records = []

        for control in tqdm(controls, desc=f"Controls candidate {candidate_index + 1}/{len(candidates)}", leave=True):
            modified_frames = apply_control(frames, landmarks, control)
            modified_score = _measure_modified_score(detector, modified_frames)
            delta = float(original_score - modified_score)
            control_effects.append(delta)

            control_records.append({
                "control": control,
                "original_score": original_score,
                "modified_score": modified_score,
                "delta": delta,
            })

            del modified_frames

        tau = compute_threshold(
            control_effects,
            percentile=tau_percentile,
        )

        candidate_control_data.append({
            "candidate_index": candidate_index,
            "control_effects": control_effects,
            "control_records": control_records,
            "tau": tau,
        })

    timings["controls_and_threshold_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    verification_records = []

    for candidate_index, candidate in enumerate(tqdm(candidates, desc="Candidate effects", leave=True)):
        sampled_interval = tuple(map(int, candidate["sampled_interval"]))

        from cafe.interventions import apply_intervention

        modified_frames = apply_intervention(
            frames,
            landmarks,
            candidate["cue"],
            sampled_interval,
            strength=candidate.get("strength", config["intervention"]),
        )

        modified_score = _measure_modified_score(detector, modified_frames)
        delta = float(original_score - modified_score)

        control_data = candidate_control_data[candidate_index]
        control_effects = control_data["control_effects"]
        tau = float(control_data["tau"])

        p_value = compute_p_value(delta, control_effects)
        supported = bool(delta > tau)

        verification_records.append({
            "video_id": video_id,
            "candidate": candidate,
            "original_score": original_score,
            "modified_score": modified_score,
            "candidate_effect": delta,
            "control_effects": [float(x) for x in control_effects],
            "tau_percentile": tau_percentile,
            "tau": tau,
            "p_value": p_value,
            "supported": supported,
            "control_records": control_data["control_records"],
        })

        del modified_frames

    timings["candidate_effects_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    decision = decide(verification_records)
    rendered = render_explanation(decision)
    timings["decision_render_sec"] = time.perf_counter() - start

    timings["total_sec"] = time.perf_counter() - total_start

    result = CafeResult(
        video_id=video_id,
        label=label,
        score=original_score,
        decision=decision,
        rendered_explanation=rendered,
        candidates=verification_records,
        timings=timings,
    )

    output_path = output_dir / f"{video_id}.json"
    result.output_path = str(output_path)

    record = result.to_dict()
    record["prediction"] = {
        "label": label,
        "score": original_score,
    }
    record["per_frame_scores"] = [float(x) for x in per_frame_scores]
    record["timings"] = timings

    output_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    return result








