from pathlib import Path

from cafe.pipeline import run_cafe


def test_pipeline_end_to_end(tmp_path):
    result = run_cafe(
        "Deepfakes_000_003",
        output_dir=tmp_path,
    )

    assert result.video_id == "Deepfakes_000_003"
    assert result.label in {"REAL", "DEEPFAKE"}
    assert 0.0 <= result.score <= 1.0

    assert result.candidates
    for record in result.candidates:
        assert "candidate_effect" in record
        assert "control_effects" in record
        assert len(record["control_effects"]) == 20
        assert "tau" in record
        assert "p_value" in record
        assert "supported" in record

    assert result.decision["video_id"] == result.video_id
    assert result.decision["abstained"] == (
        len(result.decision["explanations"]) == 0
    )

    assert result.rendered_explanation
    assert result.output_path is not None

    output_path = Path(result.output_path)
    assert output_path.exists()

    assert "per_frame_scores" in output_path.read_text(encoding="utf-8")
