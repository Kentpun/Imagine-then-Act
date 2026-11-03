from swm.eval.planner_eval import (
    _inject_logged,
    heuristic_score,
    oracle_outcome_from_action,
    spearman,
)
from swm.data.labeling import LabelConfig


def test_spearman_perfect() -> None:
    assert abs(spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) - 1.0) < 1e-6


def test_spearman_inverse() -> None:
    assert abs(spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) + 1.0) < 1e-6


def test_inject_logged() -> None:
    cands = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    logged = [0.05, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = _inject_logged(cands, logged)
    assert len(out) == 2


def test_heuristic_outcome() -> None:
    cfg = LabelConfig()
    action = [0.05, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]
    text = oracle_outcome_from_action(action, cfg)
    assert "moved=yes" in text.replace(" ", "")
    assert heuristic_score(action, cfg) > 0.0
