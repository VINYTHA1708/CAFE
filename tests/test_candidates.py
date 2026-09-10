import numpy as np
import pytest

from cafe.candidates import assign_cue
from cafe.detector.base_detector import BaseDetector


def test_assign_cue_returns_highest_overlap():
    faces = np.load("cache/Deepfakes_000_003/faces.npy")
    landmarks = np.load("cache/Deepfakes_000_003/landmarks.npy")

    detector = BaseDetector()

    cue, overlaps = assign_cue(
        faces,
        landmarks,
        (13, 20),
        detector,
    )

    assert cue in {
        "eye_motion",
        "mouth_motion",
        "face_texture",
    }

    assert set(overlaps) == {
        "eye_motion",
        "mouth_motion",
        "face_texture",
    }

    assert all(np.isfinite(v) for v in overlaps.values())
    assert overlaps[cue] == max(overlaps.values())


def test_assign_cue_rejects_invalid_interval():
    faces = np.load("cache/Deepfakes_000_003/faces.npy")
    landmarks = np.load("cache/Deepfakes_000_003/landmarks.npy")

    detector = BaseDetector()

    with pytest.raises(ValueError):
        assign_cue(
            faces,
            landmarks,
            (-1, 5),
            detector,
        )


def test_assign_cue_rejects_interval_outside_frames():
    faces = np.load("cache/Deepfakes_000_003/faces.npy")
    landmarks = np.load("cache/Deepfakes_000_003/landmarks.npy")

    detector = BaseDetector()

    with pytest.raises(ValueError):
        assign_cue(
            faces,
            landmarks,
            (28, 35),
            detector,
        )
