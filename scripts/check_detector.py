from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import subprocess
import tempfile
import cv2
import numpy as np

from cafe.detector.base_detector import BaseDetector
from cafe.preprocessing import load_or_build_cache
from cafe.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "sanity"
RESULTS_CSV = RESULTS_DIR / "reencode_sensitivity.csv"

FFMPEG = Path(
    r"C:\Users\vinyt\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.1-full_build-shared\bin\ffmpeg.exe"
)


def load_selected_videos(max_videos=20):
    """Select a balanced subset: up to 10 fake and 10 real videos."""
    rows = []

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        fake = []
        real = []

        for row in reader:
            if row["label"] == "fake":
                fake.append(row)
            else:
                real.append(row)

        rows.extend(fake[: max_videos // 2])
        rows.extend(real[: max_videos // 2])

    return rows


def write_png_roundtrip(frames, output_dir):
    """Write every frame as PNG and read it back."""
    output_dir.mkdir(parents=True, exist_ok=True)

    restored = []

    for i, frame in enumerate(frames):
        path = output_dir / f"{i:04d}.png"

        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Failed to write PNG: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Failed to read PNG: {path}")

        restored.append(image)

    return np.asarray(restored, dtype=np.uint8)


def write_h264_video(frames, output_path, crf):
    """Encode sampled frames into H.264 and decode them back."""
    height, width = frames[0].shape[:2]

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "frames"
        input_dir.mkdir()

        for i, frame in enumerate(frames):
            frame_path = input_dir / f"{i:04d}.png"

            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to write temporary frame: {frame_path}")

        command = [
            str(FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "25",
            "-i",
            str(input_dir / "%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        subprocess.run(command, check=True)

    cap = cv2.VideoCapture(str(output_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open encoded video: {output_path}")

    restored = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        restored.append(frame)

    cap.release()

    if len(restored) != len(frames):
        raise RuntimeError(
            f"Frame count changed after H.264 round-trip: "
            f"{len(frames)} -> {len(restored)}"
        )

    restored = np.asarray(restored, dtype=np.uint8)

    if restored.shape[1] != height or restored.shape[2] != width:
        raise RuntimeError("Frame dimensions changed after H.264 round-trip")

    return restored


def null_blend_face_region(frames, frame_index):
    """
    Null intervention:
    alpha-blend the detected face region with an identical copy of itself.

    Because both inputs are identical, the intended visual transformation
    should be a no-op apart from floating-point rounding.
    """
    output = frames.copy()

    frame = output[frame_index].astype(np.float32)

    # Identical copy of the face region.
    identical = frame.copy()

    alpha = 0.5
    blended = alpha * frame + (1.0 - alpha) * identical

    output[frame_index] = np.clip(blended, 0, 255).astype(np.uint8)

    return output


def score(detector, faces):
    return detector.score_video(faces)


def run_video(detector, row):
    video_id = row["video_id"]
    video_path = PROJECT_ROOT / row["path"]

    cache = load_or_build_cache(
        video_id=video_id,
        video_path=video_path,
    )

    frames = cache["frames"]
    faces = cache["faces"]
    index = cache["index"]

    # Baseline detector score.
    baseline = score(detector, faces)

    results = []

    results.append(
        {
            "video_id": video_id,
            "label": row["label"],
            "transformation": "original",
            "original_score": baseline,
            "transformed_score": baseline,
            "absolute_delta": 0.0,
        }
    )

    # ---------------------------------------------------------
    # Helper: crop transformed frames using ORIGINAL face boxes.
    # This keeps face localization fixed across transformations.
    # ---------------------------------------------------------
    def crop_using_original_boxes(transformed_frames):
        transformed_faces = []

        for frame, item, original_face in zip(
            transformed_frames,
            index,
            faces,
        ):
            bbox = item["bbox"]

            if bbox is None:
                transformed_faces.append(original_face)
                continue

            x1, y1, x2, y2 = bbox

            h, w = frame.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                transformed_faces.append(original_face)
                continue

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                transformed_faces.append(original_face)
                continue

            crop = cv2.resize(
                crop,
                (224, 224),
                interpolation=cv2.INTER_LINEAR,
            )

            transformed_faces.append(
                crop.astype(np.uint8)
            )

        return np.asarray(transformed_faces, dtype=np.uint8)

    # ---------------------------------------------------------
    # 2. PNG lossless round-trip
    # ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as temp_dir:
        png_frames = write_png_roundtrip(
            frames,
            Path(temp_dir) / "png",
        )

    png_faces = crop_using_original_boxes(png_frames)
    png_score = score(detector, png_faces)

    results.append(
        {
            "video_id": video_id,
            "label": row["label"],
            "transformation": "png_lossless",
            "original_score": baseline,
            "transformed_score": png_score,
            "absolute_delta": abs(png_score - baseline),
        }
    )

    # ---------------------------------------------------------
    # 3. High-quality H.264
    # ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as temp_dir:
        output_video = Path(temp_dir) / "high_quality.mp4"

        hq_frames = write_h264_video(
            frames,
            output_video,
            crf=18,
        )

    hq_faces = crop_using_original_boxes(hq_frames)
    hq_score = score(detector, hq_faces)

    results.append(
        {
            "video_id": video_id,
            "label": row["label"],
            "transformation": "h264_high_quality",
            "original_score": baseline,
            "transformed_score": hq_score,
            "absolute_delta": abs(hq_score - baseline),
        }
    )

    # ---------------------------------------------------------
    # 4. Moderate H.264
    # ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as temp_dir:
        output_video = Path(temp_dir) / "moderate.mp4"

        moderate_frames = write_h264_video(
            frames,
            output_video,
            crf=28,
        )

    moderate_faces = crop_using_original_boxes(moderate_frames)
    moderate_score = score(detector, moderate_faces)

    results.append(
        {
            "video_id": video_id,
            "label": row["label"],
            "transformation": "h264_moderate",
            "original_score": baseline,
            "transformed_score": moderate_score,
            "absolute_delta": abs(moderate_score - baseline),
        }
    )

    # ---------------------------------------------------------
    # 5. Null blend
    #
    # Apply an identity blend to the ORIGINAL face crop.
    # Since the two inputs are identical, this should be a
    # minimal/no-op control.
    # ---------------------------------------------------------
    blended_faces = faces.copy()

    middle = len(blended_faces) // 2

    face = blended_faces[middle].astype(np.float32)

    identical_copy = face.copy()

    alpha = 0.5

    blended = (
        alpha * face
        + (1.0 - alpha) * identical_copy
    )

    blended_faces[middle] = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)

    blended_score = score(
        detector,
        blended_faces,
    )

    results.append(
        {
            "video_id": video_id,
            "label": row["label"],
            "transformation": "null_blend",
            "original_score": baseline,
            "transformed_score": blended_score,
            "absolute_delta": abs(blended_score - baseline),
        }
    )

    return results
def main():
    if not FFMPEG.exists():
        raise FileNotFoundError(
            f"FFmpeg not found at expected location:\n{FFMPEG}"
        )

    config = load_config()
    print("Phase 8: Detector Re-Encode Sensitivity Study")
    print(f"Frames per video: {config['frames_per_video']}")
    print(f"FFmpeg: {FFMPEG}")

    rows = load_selected_videos(max_videos=20)

    print(f"Videos selected: {len(rows)}")
    print(
        f"Fake: {sum(row['label'] == 'fake' for row in rows)}, "
        f"Real: {sum(row['label'] == 'real' for row in rows)}"
    )

    detector = BaseDetector()

    all_results = []

    for number, row in enumerate(rows, start=1):
        print(
            f"[{number}/{len(rows)}] "
            f"{row['video_id']} ({row['label']})"
        )

        results = run_video(detector, row)
        all_results.extend(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "label",
        "transformation",
        "original_score",
        "transformed_score",
        "absolute_delta",
    ]

    with RESULTS_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print()
    print(f"Saved: {RESULTS_CSV}")
    print(f"Rows: {len(all_results)}")


if __name__ == "__main__":
    main()
