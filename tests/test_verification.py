import numpy as np

from cafe.verification import (
    compute_p_value,
    compute_threshold,
    verify_candidate,
)


def test_compute_threshold():
    effects = [0.01, 0.02, 0.03, 0.04, 0.05]
    tau = compute_threshold(effects, percentile=95)

    assert np.isclose(tau, np.percentile(effects, 95))


def test_compute_p_value_far_above_controls():
    effects = [0.01, 0.02, 0.03, 0.04, 0.05]
    p = compute_p_value(0.20, effects)

    assert np.isclose(p, 1 / 6)


def test_compute_p_value_near_zero_effect():
    effects = [0.01, 0.02, 0.03, 0.04, 0.05]
    p = compute_p_value(0.0, effects)

    assert np.isclose(p, 1.0)


def test_verify_candidate_supported(monkeypatch, tmp_path):
    class FakeDetector:
        def score_video(self, frames):
            return 0.50 if frames == "modified" else 0.90

    monkeypatch.setattr(
        "cafe.verification._load_video",
        lambda video_id: ("original", "landmarks"),
    )

    monkeypatch.setattr(
        "cafe.verification.apply_intervention",
        lambda frames, landmarks, cue, interval, strength=None: "modified",
    )

    candidate = {
        "cue": "face_texture",
        "interval": (10, 17),
        "sampled_interval": (3, 10),
    }

    record = verify_candidate(
        "test_video",
        candidate,
        [0.01, 0.02, 0.03, 0.04, 0.05],
        detector=FakeDetector(),
        output_dir=tmp_path,
    )

    assert np.isclose(record["candidate_effect"], 0.40)
    assert record["supported"] is True


def test_verify_candidate_not_supported(monkeypatch, tmp_path):
    class FakeDetector:
        def score_video(self, frames):
            return 0.899 if frames == "modified" else 0.90

    monkeypatch.setattr(
        "cafe.verification._load_video",
        lambda video_id: ("original", "landmarks"),
    )

    monkeypatch.setattr(
        "cafe.verification.apply_intervention",
        lambda frames, landmarks, cue, interval, strength=None: "modified",
    )

    candidate = {
        "cue": "face_texture",
        "interval": (10, 17),
        "sampled_interval": (3, 10),
    }

    record = verify_candidate(
        "test_video",
        candidate,
        [0.01, 0.02, 0.03, 0.04, 0.05],
        detector=FakeDetector(),
        output_dir=tmp_path,
    )

    assert np.isclose(record["candidate_effect"], 0.001)
    assert record["supported"] is False
