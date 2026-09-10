import random

from cafe.controls import generate_controls


VIDEO = {"n_frames": 32}

CANDIDATE = {
    "cue": "eye_motion",
    "interval": (8, 15),
    "strength": {
        "blend_alpha": 0.8,
        "feather_px": 5,
        "blur_sigma": 8.0,
    },
}


def test_control_count_matches_request():
    controls = generate_controls(
        VIDEO,
        CANDIDATE,
        20,
        random.Random(42),
    )
    assert len(controls) == 20


def test_interval_shift_controls_do_not_overlap():
    controls = generate_controls(
        VIDEO,
        CANDIDATE,
        20,
        random.Random(42),
    )

    c1, c2 = CANDIDATE["interval"]

    for control in controls:
        if control["type"] != "interval_shift":
            continue

        t1, t2 = control["interval"]

        assert t2 < c1 or t1 > c2


def test_interval_shift_length_matches_candidate():
    controls = generate_controls(
        VIDEO,
        CANDIDATE,
        20,
        random.Random(42),
    )

    c1, c2 = CANDIDATE["interval"]
    candidate_length = c2 - c1 + 1

    for control in controls:
        if control["type"] == "interval_shift":
            t1, t2 = control["interval"]
            assert t2 - t1 + 1 == candidate_length


def test_cue_swap_is_different_from_candidate():
    controls = generate_controls(
        VIDEO,
        CANDIDATE,
        20,
        random.Random(42),
    )

    for control in controls:
        if control["type"] == "cue_swap":
            assert control["cue"] != CANDIDATE["cue"]


def test_all_controls_match_candidate_length_and_strength():
    controls = generate_controls(
        VIDEO,
        CANDIDATE,
        20,
        random.Random(42),
    )

    candidate_length = (
        CANDIDATE["interval"][1]
        - CANDIDATE["interval"][0]
        + 1
    )

    for control in controls:
        t1, t2 = control["interval"]

        assert t2 - t1 + 1 == candidate_length
        assert control["strength"] == CANDIDATE["strength"]


def test_generation_is_reproducible():
    controls_a = generate_controls(
        VIDEO,
        CANDIDATE,
        20,
        random.Random(42),
    )

    controls_b = generate_controls(
        VIDEO,
        CANDIDATE,
        20,
        random.Random(42),
    )

    assert controls_a == controls_b
