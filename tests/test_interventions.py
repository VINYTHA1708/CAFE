import numpy as np

from cafe.landmarks import load_or_build_landmark_cache
from cafe.interventions import apply_intervention


VIDEO_ID = "Deepfakes_000_003"


def _load_data():
    frames = np.load(f"cache/{VIDEO_ID}/faces.npy")
    landmarks = load_or_build_landmark_cache(VIDEO_ID)
    return frames, landmarks


def test_outside_interval_is_bit_identical():
    frames, landmarks = _load_data()

    output = apply_intervention(
        frames,
        landmarks,
        "face_texture",
        (8, 23)
    )

    assert np.array_equal(output[:8], frames[:8])
    assert np.array_equal(output[24:], frames[24:])


def test_output_shape_and_dtype_preserved():
    frames, landmarks = _load_data()

    output = apply_intervention(
        frames,
        landmarks,
        "face_texture",
        (8, 23)
    )

    assert output.shape == frames.shape
    assert output.dtype == frames.dtype


def test_interval_is_modified():
    frames, landmarks = _load_data()

    output = apply_intervention(
        frames,
        landmarks,
        "face_texture",
        (8, 23)
    )

    assert np.any(output[8:24] != frames[8:24])


def test_start_boundary():
    frames, landmarks = _load_data()

    output = apply_intervention(
        frames,
        landmarks,
        "face_texture",
        (0, 7)
    )

    assert np.array_equal(output[8:], frames[8:])
    assert np.any(output[:8] != frames[:8])


def test_end_boundary():
    frames, landmarks = _load_data()

    output = apply_intervention(
        frames,
        landmarks,
        "face_texture",
        (24, 31)
    )

    assert np.array_equal(output[:24], frames[:24])
    assert np.any(output[24:] != frames[24:])


def test_landmark_failures_do_not_crash():
    frames, landmarks = _load_data()

    assert not np.isfinite(landmarks).all()

    output = apply_intervention(
        frames,
        landmarks,
        "face_texture",
        (0, 31)
    )

    assert output.shape == frames.shape
    assert output.dtype == frames.dtype
