import numpy as np
from cafe.landmarks import region_mask


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
    landmarks = np.load("cache/Deepfakes_000_003/landmarks.npy")

    assert landmarks.shape == (32, 478, 3)

    valid = np.isfinite(landmarks).all(axis=(1, 2))
    assert valid.sum() > 0
