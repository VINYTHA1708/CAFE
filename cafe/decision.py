import json
from pathlib import Path

from cafe.templates import (
    ABSTENTION_TEMPLATE,
    CUE_HUMAN_NAMES,
    VERIFIED_TEMPLATE,
)


def _load_frame_index(video_id):
    index_path = Path("cache") / video_id / "index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Frame index not found: {index_path}"
        )

    return json.loads(index_path.read_text(encoding="utf-8"))


def _interval_details(video_id, sampled_interval):
    frame_index = _load_frame_index(video_id)

    t1, t2 = map(int, sampled_interval)

    if t1 < 0 or t2 >= len(frame_index) or t1 > t2:
        raise ValueError("Invalid sampled interval")

    start_entry = frame_index[t1]
    end_entry = frame_index[t2]

    return {
        "t1": int(start_entry["original_frame_index"]),
        "t2": int(end_entry["original_frame_index"]),
        "start": float(start_entry["timestamp_sec"]),
        "end": float(end_entry["timestamp_sec"]),
    }


def decide(verification_records):
    """Assemble the final decision using verified candidate records only."""
    if not verification_records:
        raise ValueError("verification_records must not be empty")

    video_id = verification_records[0]["video_id"]
    original_score = float(verification_records[0]["original_score"])

    supported = [
        record
        for record in verification_records
        if bool(record.get("supported", False))
    ]

    supported.sort(
        key=lambda record: float(record["candidate_effect"]),
        reverse=True,
    )

    explanations = []

    for record in supported:
        candidate = record["candidate"]
        interval = _interval_details(
            video_id,
            candidate["sampled_interval"],
        )

        explanations.append({
            "cue": candidate["cue"],
            "cue_human_name": CUE_HUMAN_NAMES[candidate["cue"]],
            "t1": interval["t1"],
            "t2": interval["t2"],
            "start": interval["start"],
            "end": interval["end"],
            "delta": float(record["candidate_effect"]),
            "tau": float(record["tau"]),
            "p": float(record["p_value"]),
        })

    label = "DEEPFAKE" if original_score >= 0.5 else "REAL"

    return {
        "video_id": video_id,
        "label": label,
        "score": original_score,
        "abstained": len(explanations) == 0,
        "explanations": explanations,
        "tested_candidates": verification_records,
    }


def render_explanation(result):
    """Render only fixed templates from a structured decision result."""
    label = result["label"]
    score = float(result["score"])

    if result["abstained"]:
        candidates = result["tested_candidates"]

        candidate_list = "; ".join(
            f"{CUE_HUMAN_NAMES[c['candidate']['cue']]} "
            f"(frames {c['candidate']['interval'][0]}-"
            f"{c['candidate']['interval'][1]}, "
            f"effect {float(c['candidate_effect']):.3f})"
            for c in candidates
        )

        tau_values = [
            float(c["tau"])
            for c in candidates
            if "tau" in c
        ]

        if not tau_values:
            raise ValueError("Verified records contain no threshold")

        tau = max(tau_values)

        return ABSTENTION_TEMPLATE.format(
            label=label,
            s=score,
            n=len(candidates),
            tau=tau,
            candidate_list=candidate_list,
        )

    rendered = []

    for explanation in result["explanations"]:
        rendered.append(
            VERIFIED_TEMPLATE.format(
                label=label,
                s=score,
                cue_human_name=explanation["cue_human_name"],
                t1=explanation["t1"],
                t2=explanation["t2"],
                start=explanation["start"],
                end=explanation["end"],
                delta=explanation["delta"],
                tau=explanation["tau"],
                p=explanation["p"],
            )
        )

    return "\n\n".join(rendered)
