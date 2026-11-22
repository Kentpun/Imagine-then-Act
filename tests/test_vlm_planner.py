from swm.planning.vlm_planner import (
    pad_proposals_with_grid,
    parse_libero_proposals,
    parse_microduck_proposals,
    parse_preference_ranking,
    ranking_to_scores,
)


def test_parse_libero_proposals() -> None:
    text = (
        "dx=0.05; dy=-0.02; dz=0.00; gripper=close\n"
        "dx=-0.03; dy=0.01; dz=0.02; gripper=open\n"
        "noise line\n"
    )
    acts = parse_libero_proposals(text, max_n=4)
    assert len(acts) == 2
    assert acts[0][6] == 1.0
    assert acts[1][6] == 0.0


def test_parse_microduck_proposals_kick() -> None:
    text = "vx=0.20; yaw=0.10; kick=no\nvx=0.00; yaw=0.00; kick=yes\n"
    acts = parse_microduck_proposals(text, max_n=4, allow_kick=True)
    assert len(acts) == 2
    assert acts[1][2] == 9.0


def test_parse_microduck_without_kick_field() -> None:
    text = "1) vx=0.25 yaw=0.0\n2) vx=0.15; yaw=-0.5\n"
    acts = parse_microduck_proposals(text, max_n=4, allow_kick=False)
    assert len(acts) == 2


def test_parse_microduck_turn_in_place_fix() -> None:
    text = "vx=0.00; yaw=0.50; kick=no\n"
    acts = parse_microduck_proposals(text, max_n=2, allow_kick=False)
    assert len(acts) == 1
    assert acts[0][0] > 0.0


def test_pad_proposals_with_grid() -> None:
    proposed = [[0.25, 0.0, 0.0]]
    grid = [[0.15, 0.0, 0.5], [0.15, 0.0, -0.5], [0.25, 0.0, 0.0]]
    merged = pad_proposals_with_grid(proposed, grid, max_n=4)
    assert len(merged) >= 3
    assert merged[0] == [0.25, 0.0, 0.0]


def test_preference_ranking_parse() -> None:
    order = parse_preference_ranking("best is 3 then 1>2", n=3)
    assert order[0] == 3
    assert set(order) == {1, 2, 3}
    scores = ranking_to_scores(order, 3)
    assert scores[2] == 3.0
    assert scores[0] == 2.0
    assert scores[1] == 1.0
