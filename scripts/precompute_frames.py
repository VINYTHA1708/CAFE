from pathlib import Path
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cafe.preprocessing import load_or_build_cache
from cafe.utils.logging_utils import setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"


def main():
    logger = setup_logger("precompute")

    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    processed = 0
    failed = 0

    for row in rows:
        video_path = PROJECT_ROOT / row["path"]

        try:
            cache = load_or_build_cache(row["video_id"], video_path)
            face_count = sum(
                item["face_found"] for item in cache["index"]
            )
            logger.info(
                f"{row["video_id"]}: {len(cache["index"])} frames, "
                f"{face_count} faces detected"
            )
            processed += 1
        except Exception as exc:
            logger.error(f"{row["video_id"]}: {exc}")
            failed += 1

    print("\nPreprocessing summary")
    print("Total:", total)
    print("Processed:", processed)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
