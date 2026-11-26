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


def test_hold_never_freezes_when_ball_outside_zone():
    task = "keep the orange ball inside the green circle"
    hold_outside = (
        "moved=no; lifted=no; gripper=hold; progress=flat; "
        "success_near=no; ball_in_zone=no; facing_goal=no; skill=hold"
    )
    recovery_turn = (
        "moved=yes; lifted=no; gripper=hold; progress=down; "
        "success_near=no; ball_in_zone=no; facing_goal=no; skill=walk"
    )
    hold_inside = (
        "moved=no; lifted=no; gripper=hold; progress=up; "
        "success_near=yes; ball_in_zone=yes; facing_goal=yes; skill=hold"
    )
    assert score_from_outcome(recovery_turn, task=task) > score_from_outcome(hold_outside, task=task)
    assert score_from_outcome(hold_inside, task=task) > score_from_outcome(recovery_turn, task=task)

