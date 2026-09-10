import json
import random
from pathlib import Path

import numpy as np

from cafe.controls import generate_controls, apply_control
from cafe.interventions import apply_intervention
from cafe.preprocessing import load_or_build_cache
from cafe.landmarks import load_or_build_landmark_cache
from cafe.detector.base_detector import BaseDetector


DEFAULT_STRENGTH = {
    "blend_alpha": 0.8,
    "feather_px": 5,
    "blur_sigma": 8.0,
}


def _load_video(video_id):
    cache = load_or_build_cache(video_id)
    frames = cache["faces"]
    landmarks = load_or_build_landmark_cache(video_id)
    return frames, landmarks


def _get_original_score(video_id, detector):
    frames, _ = _load_video(video_id)
    return float(detector.score_video(frames))


def measure_effect(video_id, spec, detector=None, original_score=None):
    """
    Measure ? = D(V) - D(F(V,...)).
    """
    if detector is None:
        detector = BaseDetector()

    frames, landmarks = _load_video(video_id)

    if original_score is None:
        original_score = float(detector.score_video(frames))

    if spec.get("type") in {"sham", "interval_shift", "cue_swap"}:
        modified_frames = apply_control(
            frames,
            landmarks,
            spec,
        )
    else:
        modified_frames = apply_intervention(
            frames,
            landmarks,
            spec["cue"],
            spec["interval"],
            strength=spec.get("strength", DEFAULT_STRENGTH),
        )

    modified_score = float(detector.score_video(modified_frames))
    delta = float(original_score - modified_score)

    return {
        "original_score": float(original_score),
        "modified_score": modified_score,
        "delta": delta,
    }


def build_control_distribution(
    video_id,
    candidate,
    n_controls=20,
    rng=None,
    detector=None,
    original_score=None,
):
    """Generate and measure the control-effect distribution."""
    if rng is None:
        rng = random.Random(42)

    if detector is None:
        detector = BaseDetector()

    frames, _ = _load_video(video_id)

    if original_score is None:
        original_score = float(detector.score_video(frames))

    video_meta = {
        "n_frames": len(frames),
    }

    candidate_for_controls = dict(candidate)
    candidate_for_controls["interval"] = tuple(
        map(int, candidate["sampled_interval"])
    )
    candidate_for_controls.setdefault(
        "strength",
        dict(DEFAULT_STRENGTH),
    )

    controls = generate_controls(
        video_meta,
        candidate_for_controls,
        n_controls,
        rng,
    )

    effects = []
    records = []

    for spec in controls:
        result = measure_effect(
            video_id,
            spec,
            detector=detector,
            original_score=original_score,
        )

        effects.append(result["delta"])

        records.append({
            "control": spec,
            "original_score": result["original_score"],
            "modified_score": result["modified_score"],
            "delta": result["delta"],
        })

    return effects, records


def compute_threshold(control_effects, percentile=95):
    """Compute t as the configured percentile of E."""
    effects = np.asarray(control_effects, dtype=float)

    if effects.size == 0:
        raise ValueError("control_effects must not be empty")

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    return float(np.percentile(effects, percentile))


def compute_p_value(candidate_effect, control_effects):
    """
    Compute:
        p = (1 + #{?_i >= ?}) / (1 + m)
    """
    effects = np.asarray(control_effects, dtype=float)

    if effects.size == 0:
        raise ValueError("control_effects must not be empty")

    count = int(np.sum(effects >= float(candidate_effect)))
    m = int(effects.size)

    return float((1 + count) / (1 + m))


def verify_candidate(
    video_id,
    candidate,
    control_effects,
    detector=None,
    tau_percentile=95,
    control_records=None,
    output_dir="results/verification",
    original_score=None,
):
    """Verify one candidate against its control-effect distribution."""
    if detector is None:
        detector = BaseDetector()

    frames, landmarks = _load_video(video_id)

    if original_score is None:
        original_score = float(detector.score_video(frames))

    candidate_interval = candidate.get(
        "sampled_interval",
        candidate["interval"],
    )

    modified_frames = apply_intervention(
        frames,
        landmarks,
        candidate["cue"],
        candidate_interval,
        strength=candidate.get("strength", DEFAULT_STRENGTH),
    )

    modified_score = float(detector.score_video(modified_frames))
    delta = float(original_score - modified_score)

    tau = compute_threshold(
        control_effects,
        percentile=tau_percentile,
    )

    p_value = compute_p_value(
        delta,
        control_effects,
    )

    supported = bool(delta > tau)

    record = {
        "video_id": video_id,
        "candidate": candidate,
        "original_score": float(original_score),
        "modified_score": modified_score,
        "candidate_effect": delta,
        "control_effects": [float(x) for x in control_effects],
        "tau_percentile": float(tau_percentile),
        "tau": tau,
        "p_value": p_value,
        "supported": supported,
    }

    if control_records is not None:
        record["control_records"] = control_records

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"{video_id}.json"
    json_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    return record


def save_verification_results(
    video_id,
    candidate_records,
    output_dir="results/verification",
):
    """Save all verified candidates for one video in a single JSON file."""
    if not candidate_records:
        raise ValueError("candidate_records must not be empty")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    record = {
        "video_id": video_id,
        "original_score": float(candidate_records[0]["original_score"]),
        "candidates": candidate_records,
    }

    json_path = output_path / f"{video_id}.json"
    json_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    return record
