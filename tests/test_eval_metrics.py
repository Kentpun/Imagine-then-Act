from swm.eval.metrics import exact_match, macro_f1_from_pairs, parse_fields, summarize_predictions


GOLD = "moved=yes; lifted=no; gripper=close; progress=up; success_near=no"


def test_parse_and_exact_match() -> None:
    assert parse_fields(GOLD)["progress"] == "up"
    assert exact_match(GOLD, GOLD.replace(" ", ""))


def test_macro_f1_perfect() -> None:
    f1 = macro_f1_from_pairs([(GOLD, GOLD), (GOLD, GOLD)])
    assert f1["macro_f1"] == 1.0
    assert f1["progress"] == 1.0


def test_summarize_predictions() -> None:
    bad = "moved=no; lifted=no; gripper=open; progress=flat; success_near=no"
    metrics = summarize_predictions([GOLD, bad], [GOLD, GOLD])
    assert metrics["n"] == 2
    assert 0.0 < float(metrics["exact_match"]) < 1.0
    assert "macro_f1" in metrics
    assert "f1_progress" in metrics
