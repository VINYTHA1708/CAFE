import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cafe.landmarks import load_or_build_landmark_cache


manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifest.csv")

total = len(manifest)
processed = 0
failed_videos = 0
total_success = 0
total_failed = 0

for _, row in manifest.iterrows():
    video_id = row["video_id"]

    try:
        landmarks = load_or_build_landmark_cache(video_id)

        valid = np.isfinite(landmarks).all(axis=(1, 2))
        success = int(valid.sum())
        failures = int((~valid).sum())

        total_success += success
        total_failed += failures
        processed += 1

        print(
            f"[{processed}/{total}] {video_id}: "
            f"{success} success, {failures} failed"
        )

    except Exception as e:
        failed_videos += 1
        print(f"[ERROR] {video_id}: {e}")

print("\n=== Landmark Precomputation Summary ===")
print("Videos processed:", processed)
print("Videos failed:", failed_videos)
print("Successful frames:", total_success)
print("Failed frames:", total_failed)
