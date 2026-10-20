import json

import numpy as np
import pytest

from cafe.landmarks import load_or_build_landmark_cache
from cafe.landmarks import region_mask
from cafe.utils.paths import get_cache_path


def test_region_masks():
    landmarks = np.zeros((478, 3), dtype=np.float32)

    # Create a valid synthetic landmark layout.
    for i in range(478):
        landmarks[i, 0] = 0.2 + 0.6 * ((i * 37) % 100) / 100
        landmarks[i, 1] = 0.2 + 0.6 * ((i * 53) % 100) / 100

    for region in ["eyes", "mouth", "face"]:
        mask = region_mask(landmarks, region, feather_px=5)

        assert mask.shape == (224, 224)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert np.count_nonzero(mask) > 0


def test_landmark_cache_shape():
    landmarks = np.load(get_cache_path() / "Deepfakes_000_003" / "landmarks.npy")

    assert landmarks.shape == (32, 478, 3)

    valid = np.isfinite(landmarks).all(axis=(1, 2))
    assert valid.sum() > 0


def test_corrected_cache_root_rejects_unmarked_legacy_cache(tmp_path):
    cache_dir = tmp_path / "Deepfakes_000_003"
    cache_dir.mkdir()
    np.save(cache_dir / "faces.npy", np.zeros((1, 224, 224, 3), dtype=np.uint8))
    (cache_dir / "index.json").write_text(
        json.dumps([{"face_found": True}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not marked as RGB"):
        load_or_build_landmark_cache(
            "Deepfakes_000_003",
            cache_root=tmp_path,
        )
