from swm.data.labeling import LabelConfig, build_outcome_string, label_transition, score_from_outcome


def test_label_transition_lift():
    cfg = LabelConfig()
    labels = label_transition(
        action=[0.0, 0.0, 0.02, 0, 0, 0, 1.0],
        state_t=[0, 0, 0.1, 0, 0, 0, 0, 1.0],
        state_tp1=[0, 0, 0.13, 0, 0, 0, 0, 1.0],
        frame_idx=10,
        episode_len=40,
        episode_success=True,
        cfg=cfg,
    )
    assert labels["lifted"] == "yes"
    assert labels["gripper"] == "close"
    s = build_outcome_string(labels)
    assert "lifted=yes" in s
    assert score_from_outcome(s) > 0
