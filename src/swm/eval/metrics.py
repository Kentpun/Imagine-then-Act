"""Shared Layer A metrics for semantic outcome evaluation."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

FIELD_NAMES = ("moved", "lifted", "gripper", "progress", "success_near")

FIELD_RE = re.compile(
    r"moved=(yes|no);\s*lifted=(yes|no);\s*gripper=(open|close|hold);\s*"
    r"progress=(up|flat|down);\s*success_near=(yes|no)",
    re.IGNORECASE,
)

_FIELD_VALUES = {
    "moved": {"yes", "no"},
    "lifted": {"yes", "no"},
    "gripper": {"open", "close", "hold"},
    "progress": {"up", "flat", "down"},
    "success_near": {"yes", "no"},
}


def parse_fields(text: str) -> dict[str, str] | None:
    m = FIELD_RE.search(text.replace(" ", ""))
    if m:
        return {
            "moved": m.group(1).lower(),
            "lifted": m.group(2).lower(),
            "gripper": m.group(3).lower(),
            "progress": m.group(4).lower(),
            "success_near": m.group(5).lower(),
        }
    out: dict[str, str] = {}
    for key in FIELD_NAMES:
        mm = re.search(rf"{key}=([a-z]+)", text.lower())
        if not mm:
            return None
        out[key] = mm.group(1)
    return out


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower().replace(" ", "") == gold.strip().lower().replace(
        " ", ""
    )


def field_accuracy(pred: str, gold: str) -> dict[str, float]:
    p = parse_fields(pred)
    g = parse_fields(gold)
    if p is None or g is None:
        return {k: 0.0 for k in FIELD_NAMES}
    return {k: float(p.get(k) == g.get(k)) for k in FIELD_NAMES}


def _f1(tp: float, fp: float, fn: float) -> float | None:
    """Return F1, or None if the class never appears (undefined)."""
    if tp <= 0 and fp <= 0 and fn <= 0:
        return None
    prec = tp / max(1e-9, tp + fp)
    rec = tp / max(1e-9, tp + fn)
    if prec + rec <= 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def macro_f1_from_pairs(
    pairs: Iterable[tuple[str | None, str | None]],
) -> dict[str, float]:
    """Compute per-field macro-F1 and the average across fields.

    For each field, F1 is macro-averaged over that field's label values.
    Unparseable predictions count as wrong for every class (fn on gold).
    """
    # counts[field][label] -> [tp, fp, fn]
    counts: dict[str, dict[str, list[float]]] = {
        f: {v: [0.0, 0.0, 0.0] for v in vals} for f, vals in _FIELD_VALUES.items()
    }
    for pred_text, gold_text in pairs:
        p = parse_fields(pred_text or "")
        g = parse_fields(gold_text or "")
        for field, values in _FIELD_VALUES.items():
            gold_v = g.get(field) if g else None
            pred_v = p.get(field) if p else None
            for label in values:
                tp, fp, fn = counts[field][label]
                if gold_v == label and pred_v == label:
                    tp += 1.0
                elif pred_v == label and gold_v != label:
                    fp += 1.0
                elif gold_v == label and pred_v != label:
                    fn += 1.0
                counts[field][label] = [tp, fp, fn]

    per_field: dict[str, float] = {}
    for field, values in _FIELD_VALUES.items():
        f1s = [v for label in values if (v := _f1(*counts[field][label])) is not None]
        per_field[field] = sum(f1s) / max(1, len(f1s))
    per_field["macro_f1"] = sum(per_field[f] for f in FIELD_NAMES) / len(FIELD_NAMES)
    return per_field


def bootstrap_ci(
    preds: list[str],
    golds: list[str],
    *,
    key: str = "exact_match",
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap percentile CI for exact_match or macro_f1."""
    import random

    n = len(preds)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": 0}
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        p = [preds[i] for i in idx]
        g = [golds[i] for i in idx]
        m = summarize_predictions(p, g)
        stats.append(float(m[key]))  # type: ignore[arg-type]
    stats.sort()
    lo_i = int(alpha / 2 * (n_boot - 1))
    hi_i = int((1 - alpha / 2) * (n_boot - 1))
    return {
        "mean": sum(stats) / n_boot,
        "lo": stats[lo_i],
        "hi": stats[hi_i],
        "n_boot": float(n_boot),
    }


def summarize_predictions(
    preds: list[str], golds: list[str], *, bootstrap: bool = False, seed: int = 42
) -> dict[str, float | int | bool | dict]:
    assert len(preds) == len(golds)
    n = len(preds)
    exact = sum(float(exact_match(p, g)) for p, g in zip(preds, golds))
    parsed = sum(
        float(parse_fields(p) is not None and parse_fields(g) is not None)
        for p, g in zip(preds, golds)
    )
    field_sums: dict[str, float] = defaultdict(float)
    for p, g in zip(preds, golds):
        for k, v in field_accuracy(p, g).items():
            field_sums[k] += v
    f1 = macro_f1_from_pairs(zip(preds, golds))
    out: dict[str, float | int | bool | dict] = {
        "exact_match": exact / max(1, n),
        "parse_rate": parsed / max(1, n),
        "macro_f1": float(f1["macro_f1"]),
        "n": n,
    }
    for k in FIELD_NAMES:
        out[f"acc_{k}"] = field_sums[k] / max(1, n)
        out[f"f1_{k}"] = float(f1[k])
    if bootstrap and n > 0:
        out["bootstrap_exact_match"] = bootstrap_ci(
            preds, golds, key="exact_match", seed=seed
        )
        out["bootstrap_macro_f1"] = bootstrap_ci(
            preds, golds, key="macro_f1", seed=seed + 1
        )
    return out


def summarize_by_task(
    preds: list[str],
    golds: list[str],
    tasks: list[str],
    *,
    bootstrap: bool = False,
    seed: int = 42,
) -> dict[str, dict]:
    """Per-task Layer A metrics (truncated task string as key)."""
    buckets: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for p, g, t in zip(preds, golds, tasks):
        key = (t or "unknown").strip()[:120] or "unknown"
        buckets[key][0].append(p)
        buckets[key][1].append(g)
    return {
        task: summarize_predictions(ps, gs, bootstrap=bootstrap, seed=seed)
        for task, (ps, gs) in sorted(buckets.items(), key=lambda kv: (-len(kv[1][0]), kv[0]))
    }
