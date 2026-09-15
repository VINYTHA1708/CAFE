from pathlib import Path
import json
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

from cafe.preprocessing import load_or_build_cache
from cafe.landmarks import load_or_build_landmark_cache
from cafe.detector.base_detector import BaseDetector
from cafe.interventions import apply_intervention


CASES = {
    "verified": {
        "video_id": "Deepfakes_000_003",
        "video_path": "dataset/fake/Deepfakes/000_003.mp4",
        "json_path": "results/runs/Deepfakes_000_003__real.json",
        "candidate_index": 2,
        "title": "Case 1: Verified explanation",
    },
    "abstained": {
        "video_id": "real_011",
        "video_path": "dataset/real/011.mp4",
        "json_path": "results/runs/real_011__real.json",
        "candidate_index": 2,
        "title": "Case 2: Abstention",
    },
    "control_rejected": {
        "video_id": "Deepfakes_001_870",
        "video_path": "dataset/fake/Deepfakes/001_870.mp4",
        "json_path": "results/runs/Deepfakes_001_870__real.json",
        "candidate_index": 1,
        "title": "Case 3: Raw effect rejected by controls",
    },
}


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
VIDEO_DIR = ROOT / "results" / "demo_videos"

FIG_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def load_case(case):
    with open(ROOT / case["json_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    candidate = data["candidates"][case["candidate_index"]]

    cache = load_or_build_cache(
        case["video_id"],
        video_path=str(ROOT / case["video_path"]),
    )
    landmarks = load_or_build_landmark_cache(case["video_id"])

    detector = BaseDetector()
    scores = detector.score_frames(cache["faces"])

    interval_original = candidate["candidate"]["interval"]; interval = tuple(candidate["candidate"]["sampled_interval"])
    start, end = int(interval[0]), int(interval[1])

    modified = apply_intervention(
        cache["faces"],
        landmarks,
        candidate["candidate"]["cue"],
        interval,
        strength=None,
    )

    modified_scores = detector.score_frames(modified)

    return data, candidate, cache, landmarks, scores, modified, modified_scores, (start, end)


def make_strip(frames, selected_interval, n=8):
    start, end = selected_interval
    positions = np.linspace(start, end, n).round().astype(int)
    positions = np.clip(positions, 0, len(frames) - 1)

    images = []
    for p in positions:
        img = frames[p]
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append((p, img))

    return images


def make_demo_video(case_name, original_frames, modified_frames, fps=4):
    path = VIDEO_DIR / f"{case_name}_sampled_face_crops.mp4"

    h, w = original_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w * 2, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {path}")

    for original, modified in zip(original_frames, modified_frames):
        left = original.copy()
        right = modified.copy()

        left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)

        combined = np.concatenate([left, right], axis=1)

        cv2.putText(
            combined,
            "Original",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined,
            "Counterfactual",
            (w + 10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(combined)

    writer.release()
    return path


def make_panel(case_name, case, data, candidate, cache, scores, modified, modified_scores, interval):
    fig = plt.figure(figsize=(16, 10))

    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 1.35],
        hspace=0.45,
    )

    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    ax3 = fig.add_subplot(grid[2])

    original_strip = make_strip(cache["faces"], interval)
    modified_strip = make_strip(modified, interval)

    ax1.axis("off")
    ax1.set_title("Original sampled face crops", loc="left", fontweight="bold")

    for i, (position, image) in enumerate(original_strip):
        ax = ax1.inset_axes([i / 8, 0, 1 / 8 - 0.006, 0.92])
        ax.imshow(image)
        ax.set_title(f"{position}", fontsize=8)
        ax.axis("off")

    ax2.axis("off")
    ax2.set_title("Counterfactual sampled face crops", loc="left", fontweight="bold")

    for i, (position, image) in enumerate(modified_strip):
        ax = ax2.inset_axes([i / 8, 0, 1 / 8 - 0.006, 0.92])
        ax.imshow(image)
        ax.set_title(f"{position}", fontsize=8)
        ax.axis("off")

    x = np.arange(len(scores))
    ax3.plot(x, scores, marker="o", markersize=3, label="Original score")
    ax3.plot(x, modified_scores, marker="o", markersize=3, label="Counterfactual score")

    start, end = interval
    ax3.axvspan(start, end, alpha=0.18, label="Intervention interval")

    ax3.set_xlabel("Sampled frame position")
    ax3.set_ylabel("Detector score")
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")

    original_score = float(data["original_score"])
    delta = float(candidate["candidate_effect"])
    tau = float(candidate["tau"])
    p_value = float(candidate["p_value"])
    supported = bool(candidate["supported"])

    controls = np.asarray(candidate["control_effects"], dtype=float)

    control_ax = ax3.inset_axes([0.68, 0.08, 0.29, 0.62])
    control_ax.hist(controls, bins=10, alpha=0.65)
    control_ax.axvline(delta, linewidth=2, label="Candidate delta")
    control_ax.axvline(tau, linestyle="--", linewidth=2, label="Tau")
    control_ax.set_xlabel("Effect")
    control_ax.set_ylabel("Controls")
    control_ax.legend(fontsize=7)
    control_ax.grid(alpha=0.2)

    status = "SUPPORTED" if supported else "ABSTAINED / REJECTED"

    fig.suptitle(
        f"{case['title']} — {case['video_id']}",
        fontsize=16,
        fontweight="bold",
    )

    fig.text(
        0.02,
        0.015,
        (
            f"Prediction score: {original_score:.4f}    "
            f"Cue: {candidate["candidate"]["cue"]}    "
            f"Original interval: {candidate["candidate"]["interval"][0]}–{candidate["candidate"]["interval"][1]}    "
            f"Delta: {delta:.5f}    Tau: {tau:.5f}    "
            f"p: {p_value:.4f}    Result: {status}"
        ),
        fontsize=9,
    )

    output = FIG_DIR / f"case_{case_name}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output


def main():
    print("Generating qualitative cases...")

    for case_name, case in CASES.items():
        print(f"\n[{case_name}] {case['video_id']}")

        (
            data,
            candidate,
            cache,
            landmarks,
            scores,
            modified,
            modified_scores,
            interval,
        ) = load_case(case)

        original_rgb = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in cache["faces"]
        ]
        modified_rgb = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in modified
        ]

        video_path = make_demo_video(
            case_name,
            original_rgb,
            modified_rgb,
        )

        figure_path = make_panel(
            case_name,
            case,
            data,
            candidate,
            cache,
            scores,
            modified,
            modified_scores,
            interval,
        )

        print(f"  cue       : {candidate["candidate"]["cue"]}")
        print(f"  interval  : {candidate["candidate"]["interval"]}")
        print(f"  delta     : {candidate["candidate_effect"]:.6f}")
        print(f"  tau       : {candidate['tau']:.6f}")
        print(f"  p-value   : {candidate['p_value']:.6f}")
        print(f"  supported : {candidate['supported']}")
        print(f"  figure    : {figure_path}")
        print(f"  video     : {video_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
