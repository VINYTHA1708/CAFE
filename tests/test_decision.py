from cafe.decision import decide, render_explanation


def _record(
    video_id,
    cue,
    sampled_interval,
    effect,
    tau,
    p,
    supported,
    original_score=0.9,
):
    return {
        "video_id": video_id,
        "candidate": {
            "cue": cue,
            "interval": (100, 200),
            "sampled_interval": sampled_interval,
        },
        "original_score": original_score,
        "modified_score": original_score - effect,
        "candidate_effect": effect,
        "control_effects": [0.01, 0.02, 0.03],
        "tau": tau,
        "p_value": p,
        "supported": supported,
    }


def test_abstention_when_nothing_is_supported():
    records = [
        _record(
            "Deepfakes_000_003",
            "face_texture",
            (5, 12),
            0.010,
            0.020,
            0.50,
            False,
        ),
        _record(
            "Deepfakes_000_003",
            "eye_motion",
            (15, 22),
            0.005,
            0.020,
            1.00,
            False,
        ),
    ]

    result = decide(records)

    assert result["abstained"] is True
    assert result["explanations"] == []

    text = render_explanation(result)

    assert "No explanation could be verified." in text
    assert "face texture" in text
    assert "eye motion" in text
    assert "Verified explanation:" not in text


def test_multiple_supported_candidates_are_ordered_by_effect():
    records = [
        _record(
            "Deepfakes_000_003",
            "eye_motion",
            (5, 12),
            0.040,
            0.020,
            0.10,
            True,
        ),
        _record(
            "Deepfakes_000_003",
            "face_texture",
            (15, 22),
            0.080,
            0.020,
            0.05,
            True,
        ),
        _record(
            "Deepfakes_000_003",
            "mouth_motion",
            (23, 30),
            0.010,
            0.020,
            0.50,
            False,
        ),
    ]

    result = decide(records)

    assert result["abstained"] is False
    assert len(result["explanations"]) == 2
    assert result["explanations"][0]["cue"] == "face_texture"
    assert result["explanations"][1]["cue"] == "eye_motion"

    text = render_explanation(result)

    assert text.count("Verified explanation:") == 2
    assert "face texture" in text
    assert "eye motion" in text
    assert "mouth motion" not in text


def test_unsupported_candidate_never_reaches_verified_output():
    records = [
        _record(
            "Deepfakes_000_003",
            "face_texture",
            (5, 12),
            0.050,
            0.020,
            0.10,
            True,
        ),
        _record(
            "Deepfakes_000_003",
            "mouth_motion",
            (15, 22),
            0.090,
            0.100,
            0.50,
            False,
        ),
    ]

    result = decide(records)
    text = render_explanation(result)

    assert "face texture" in text
    assert "mouth motion" not in text
