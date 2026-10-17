from swm.microduck.labeling import build_outcome_string, label_command, score_from_outcome
from swm.microduck.prompts import build_prompt, format_action


def test_format_action():
    assert "vx=0.40" in format_action([0.4, 0.0, 0.0])


def test_label_and_score_forward():
    labels = label_command([0.5, 0.0, 0.0], upright=True)
    outcome = build_outcome_string(labels)
    assert "progressed=yes" in outcome
    assert "fell=no" in outcome
    assert score_from_outcome(outcome) > 0


def test_fall_penalty():
    labels = label_command([1.5, 0.0, 0.0], upright=True)
    outcome = build_outcome_string(labels)
    assert "fell=yes" in outcome
    assert score_from_outcome(outcome) < 0


def test_prompt_contains_velocity():
    text = build_prompt("walk forward", [0.2, 0.0, -0.3])
    assert "velocity=" in text
    assert "Answer:" in text
