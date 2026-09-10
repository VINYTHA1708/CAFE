from pathlib import Path
import csv
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"

MIN_FRAMES = 150


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or n_frames < MIN_FRAMES:
        return None

    duration = n_frames / fps
    return n_frames, fps, duration, width, height


def build_manifest():
    rows = []
    skipped = 0

    fake_root = DATASET_ROOT / "fake"
    real_root = DATASET_ROOT / "real"

    for path in sorted(fake_root.rglob("*.mp4")):
        info = inspect_video(path)

        if info is None:
            print(f"WARNING: skipped {path}")
            skipped += 1
            continue

        n_frames, fps, duration, width, height = info
        method = path.parent.name
        video_id = f"{method}_{path.stem}"
        source_id = path.stem.split("_")[0]

        rows.append({
            "video_id": video_id,
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "label": "fake",
            "method": method,
            "source_id": source_id,
            "n_frames": n_frames,
            "fps": fps,
            "duration_sec": duration,
            "width": width,
            "height": height,
        })

    for path in sorted(real_root.glob("*.mp4")):
        info = inspect_video(path)

        if info is None:
            print(f"WARNING: skipped {path}")
            skipped += 1
            continue

        n_frames, fps, duration, width, height = info
        video_id = f"real_{path.stem}"

        rows.append({
            "video_id": video_id,
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "label": "real",
            "method": "original",
            "source_id": path.stem,
            "n_frames": n_frames,
            "fps": fps,
            "duration_sec": duration,
            "width": width,
            "height": height,
        })

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "path",
        "label",
        "method",
        "source_id",
        "n_frames",
        "fps",
        "duration_sec",
        "width",
        "height",
    ]

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fake_rows = [r for r in rows if r["label"] == "fake"]
    real_rows = [r for r in rows if r["label"] == "real"]

    print("\nManifest created:", MANIFEST_PATH)
    print("Total videos:", len(rows))
    print("Fake videos:", len(fake_rows))
    print("Real videos:", len(real_rows))
    print("Skipped:", skipped)

    print("\nFake counts by method:")
    for method in sorted(set(r["method"] for r in fake_rows)):
        count = sum(r["method"] == method for r in fake_rows)
        print(f"  {method}: {count}")

    if rows:
        mean_duration = sum(r["duration_sec"] for r in rows) / len(rows)
        print(f"\nMean duration: {mean_duration:.2f} seconds")


if __name__ == "__main__":
    build_manifest()
