import numpy as np

from cafe.detector.base_detector import BaseDetector


def test_detector_determinism():
    faces = np.load("cache/Deepfakes_000_003/faces.npy")

    detector = BaseDetector()

    scores_1 = detector.score_frames(faces)
    scores_2 = detector.score_frames(faces)

    assert np.array_equal(scores_1, scores_2)


def test_detector_score_range():
    faces = np.load("cache/Deepfakes_000_003/faces.npy")

    detector = BaseDetector()
    scores = detector.score_frames(faces)

    assert len(scores) == 32
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_video_score_is_mean():
    faces = np.load("cache/Deepfakes_000_003/faces.npy")

    detector = BaseDetector()

    frame_scores = detector.score_frames(faces)
    video_score = detector.score_video(faces)

    assert video_score == float(frame_scores.mean())