from pathlib import Path
import json

import numpy as np
import pytest

from cafe.preprocessing import load_or_build_cache
from cafe.utils.paths import get_cache_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_precomputed_cache():
    cache_dir = get_cache_path() / "Deepfakes_000_003"

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


def test_face_detector_receives_rgb_and_preserves_source_frame(monkeypatch):
    from cafe import preprocessing

    source_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    source_bgr[:] = (7, 11, 239)
    detector_input = []

    class FakeExtractor:
        def process_image(self, img):
            detector_input.append(img.copy())
            return {
                "faces": [img.copy()],
                "detections": [[0.25, 0.25, 0.75, 0.75]],
            }

    monkeypatch.setattr(
        preprocessing,
        "_create_face_extractor",
        lambda: FakeExtractor(),
    )

    faces, metadata = preprocessing.detect_and_crop_faces([source_bgr])

    expected_rgb = np.array([239, 11, 7], dtype=np.uint8)
    assert np.array_equal(detector_input[0][0, 0], expected_rgb)
    assert np.array_equal(faces[0][0, 0], expected_rgb)
    assert np.array_equal(source_bgr[0, 0], [7, 11, 239])
    assert metadata[0]["face_found"] is True
    assert metadata[0]["face_color_space"] == "RGB"


def test_cache_loader_rejects_unmarked_legacy_cache(tmp_path):
    cache_dir = tmp_path / "Deepfakes_000_003"
    cache_dir.mkdir()
    np.save(cache_dir / "frames.npy", np.zeros((1, 8, 8, 3), dtype=np.uint8))
    np.save(cache_dir / "faces.npy", np.zeros((1, 224, 224, 3), dtype=np.uint8))
    (cache_dir / "index.json").write_text(
        json.dumps([{"face_found": True}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not marked as RGB"):
        load_or_build_cache(
            "Deepfakes_000_003",
            cache_root=tmp_path,
        )