from pathlib import Path
import json

import numpy as np

from cafe.preprocessing import load_or_build_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_precomputed_cache():
    cache_dir = PROJECT_ROOT / "cache" / "Deepfakes_000_003"

    frames = np.load(cache_dir / "frames.npy")
    faces = np.load(cache_dir / "faces.npy")

    with open(cache_dir / "index.json", "r", encoding="utf-8") as f:
        index = json.load(f)

    assert len(frames) == 32
    assert len(faces) == 32
    assert len(index) == 32

    timestamps = [item["timestamp_sec"] for item in index]
    assert all(
        timestamps[i] <= timestamps[i + 1]
        for i in range(len(timestamps) - 1)
    )

    for item in index:
        assert item["face_found"] is True

        x1, y1, x2, y2 = item["bbox"]

        height, width = frames[item["sampled_position"]].shape[:2]

        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height


def test_cache_reuse():
    video_path = PROJECT_ROOT / "dataset" / "fake" / "Deepfakes" / "000_003.mp4"

    cache = load_or_build_cache(
        "Deepfakes_000_003",
        video_path,
    )

    assert cache["frames"].shape == (32, 480, 640, 3)
    assert cache["faces"].shape == (32, 224, 224, 3)
    assert len(cache["index"]) == 32