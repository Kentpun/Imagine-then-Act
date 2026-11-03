"""Layer B evaluation: imagination planner vs random / heuristic / zero-shot / LoRA.

Without a full LIBERO rollout simulator for every candidate, we:
  1. Build K candidate actions (planner grid) and inject the logged action.
  2. Score candidates with each baseline.
  3. Use action-only semantic labels as a reproducible pseudo-oracle ranking
     (same rules as dataset labeling without proprioception).
  4. Also report whether the logged demonstrator action is retrieved in top-k.

Horizon ablation (A5): horizon=1 is standard; horizon=2 greedily picks a second
action on the *same* observation (limited — no true next-frame), documented in
the metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image
from tqdm import tqdm

from swm.data.dataset import SemanticTransitionDataset
from swm.data.labeling import (
    LabelConfig,
    build_outcome_string,
    label_transition,
    score_from_outcome,
)
from swm.models.semantic_vlm import (
    build_model_and_processor,
    generate_outcome,
    load_finetuned,
)
from swm.planning.imagination_planner import candidate_actions
from swm.utils.config import load_config
from swm.utils.io import ensure_dir
from swm.utils.seed import set_seed


@dataclass
class ScoredAction:
    action: list[float]
    score: float
    outcome: str = ""


def _as7(action: Sequence[float]) -> list[float]:
    a = [float(x) for x in action] + [0.0] * 7
    return a[:7]


def _action_key(action: Sequence[float], ndigits: int = 3) -> tuple[float, ...]:
    return tuple(round(float(x), ndigits) for x in _as7(action))


def _inject_logged(
    candidates: list[list[float]], logged: list[float]
) -> list[list[float]]:
    keys = {_action_key(a) for a in candidates}
    out = [list(a) for a in candidates]
    if _action_key(logged) not in keys:
        out.append(_as7(logged))
    return out


def oracle_outcome_from_action(action: Sequence[float], cfg: LabelConfig) -> str:
    labels = label_transition(
        _as7(action),
        None,
        None,
        frame_idx=0,
        episode_len=100,
        episode_success=False,
        cfg=cfg,
    )
    return build_outcome_string(labels)


def heuristic_score(action: Sequence[float], cfg: LabelConfig) -> float:
    return score_from_outcome(oracle_outcome_from_action(action, cfg))


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    n = len(xs)
    if n < 2:
        return 0.0

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx) ** 0.5
    deny = sum((b - my) ** 2 for b in ry) ** 0.5
    if denx <= 1e-12 or deny <= 1e-12:
        return 0.0
    return float(num / (denx * deny))


def pick_top(scored: list[ScoredAction], rng: random.Random | None = None) -> ScoredAction:
    if not scored:
        raise ValueError("empty scored list")
    best = max(s.score for s in scored)
    tied = [s for s in scored if abs(s.score - best) < 1e-9]
    if rng is None:
        return tied[0]
    return rng.choice(tied)


def score_random(
    actions: list[list[float]], rng: random.Random
) -> list[ScoredAction]:
    return [ScoredAction(action=a, score=rng.random()) for a in actions]


def score_heuristic(
    actions: list[list[float]], label_cfg: LabelConfig
) -> list[ScoredAction]:
    out = []
    for a in actions:
        outcome = oracle_outcome_from_action(a, label_cfg)
        out.append(ScoredAction(action=a, score=score_from_outcome(outcome), outcome=outcome))
    return out


def score_vlm(
    actions: list[list[float]],
    *,
    model,
    processor,
    image: Image.Image,
    task: str,
) -> list[ScoredAction]:
    out = []
    for a in actions:
        outcome = generate_outcome(model, processor, image, task, a)
        out.append(
            ScoredAction(action=a, score=score_from_outcome(outcome), outcome=outcome)
        )
    return out


def evaluate_split(
    ds: SemanticTransitionDataset,
    cfg: dict[str, Any],
    *,
    scorers: dict[str, Callable[[list[list[float]], Image.Image, str], list[ScoredAction]]],
    max_samples: int,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    label_cfg = LabelConfig(
        move_threshold=float(cfg["data"].get("move_threshold", 0.01)),
        lift_threshold=float(cfg["data"].get("lift_threshold", 0.01)),
        success_near_frames=int(cfg["data"].get("success_near_frames", 15)),
    )
    base_candidates = candidate_actions(cfg)
    n = min(len(ds), max_samples)

    stats = {
        name: {
            "top1_vs_oracle": 0.0,
            "regret": 0.0,
            "spearman": 0.0,
            "logged_top1": 0.0,
            "logged_top3": 0.0,
            "mean_plan_score": 0.0,
            "n": 0,
        }
        for name in scorers
    }
    spearman_sums = {name: 0.0 for name in scorers}

    for i in tqdm(range(n), desc=f"planner_h{horizon}"):
        sample = ds[i]
        logged = _as7(sample.get("action") or [0.0] * 7)
        actions = _inject_logged(base_candidates, logged)
        oracle_scored = score_heuristic(actions, label_cfg)
        oracle_best = pick_top(oracle_scored)
        oracle_by_key = {_action_key(s.action): s for s in oracle_scored}
        logged_key = _action_key(logged)
        step_rng = random.Random(seed + i)

        for name, scorer in scorers.items():
            if name == "B0_random":
                scored = score_random(actions, step_rng)
            else:
                scored = scorer(actions, sample["image"], sample["task"])
            chosen_first = pick_top(scored, step_rng if name == "B0_random" else None)
            plan_score = chosen_first.score
            if horizon >= 2:
                if name == "B0_random":
                    second_scored = score_random(actions, step_rng)
                else:
                    second_scored = scorer(
                        actions, sample["image"], sample["task"]
                    )
                second = pick_top(second_scored, step_rng if name == "B0_random" else None)
                plan_score = chosen_first.score + second.score

            chosen_key = _action_key(chosen_first.action)
            oracle_chosen = oracle_by_key[chosen_key]
            top1 = float(chosen_key == _action_key(oracle_best.action))
            regret = float(oracle_best.score - oracle_chosen.score)

            ranked = sorted(scored, key=lambda s: s.score, reverse=True)
            keys = [_action_key(s.action) for s in ranked]
            logged_top1 = float(keys[0] == logged_key) if keys else 0.0
            logged_top3 = float(logged_key in keys[:3]) if keys else 0.0

            model_scores = [s.score for s in scored]
            oracle_scores = [
                oracle_by_key[_action_key(s.action)].score for s in scored
            ]
            spearman_sums[name] += spearman(model_scores, oracle_scores)

            st = stats[name]
            st["top1_vs_oracle"] += top1
            st["regret"] += regret
            st["logged_top1"] += logged_top1
            st["logged_top3"] += logged_top3
            st["mean_plan_score"] += plan_score
            st["n"] += 1

    for name, st in stats.items():
        m = max(1, int(st["n"]))
        st["top1_vs_oracle"] /= m
        st["regret"] /= m
        st["logged_top1"] /= m
        st["logged_top3"] /= m
        st["mean_plan_score"] /= m
        st["spearman"] = spearman_sums[name] / m
    return {"horizon": horizon, "baselines": stats, "n_frames": n}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument(
        "--horizons",
        type=str,
        default="1,2",
        help="Comma-separated horizons for ablation A5 (e.g. 1 or 1,2).",
    )
    p.add_argument(
        "--skip_zero_shot",
        action="store_true",
        help="Skip B2 (faster; still runs B0/B1/B3).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    seed = int(cfg.get("seed", 42))

    processed = Path(cfg["paths"]["processed_dir"])
    frames = Path(cfg["paths"]["frames_dir"])
    split_path = processed / f"{args.split}.jsonl"
    if not split_path.is_file():
        if args.split == "test" and (processed / "val.jsonl").is_file():
            print("test.jsonl missing; falling back to val.jsonl")
            split_path = processed / "val.jsonl"
        else:
            raise SystemExit(f"Split not found: {split_path}")

    ds = SemanticTransitionDataset(split_path, frames)
    max_samples = int(
        args.max_samples
        or cfg["eval"].get("planner_max_samples")
        or min(64, int(cfg["eval"].get("max_samples", 512)))
    )
    # Planner queries K models per frame — keep default smaller than Layer A.
    max_samples = max(1, max_samples)

    label_cfg = LabelConfig(
        move_threshold=float(cfg["data"].get("move_threshold", 0.01)),
        lift_threshold=float(cfg["data"].get("lift_threshold", 0.01)),
        success_near_frames=int(cfg["data"].get("success_near_frames", 15)),
    )
    scorers: dict[str, Callable[[list[list[float]], Image.Image, str], list[ScoredAction]]] = {
        # B0 is sampled inside evaluate_split with a per-frame RNG.
        "B0_random": lambda acts, _img, _task: score_random(acts, random.Random(0)),
        "B1_heuristic": lambda acts, _img, _task: score_heuristic(acts, label_cfg),
    }

    zs_model = zs_proc = None
    if not args.skip_zero_shot:
        print("Loading zero-shot VLM for B2...")
        zs_model, zs_proc = build_model_and_processor(cfg, for_training=False)

        def b2(acts, img, task, _m=zs_model, _p=zs_proc):
            return score_vlm(acts, model=_m, processor=_p, image=img, task=task)

        scorers["B2_zero_shot"] = b2

    ckpt = args.checkpoint or cfg["eval"]["checkpoint"]
    if not Path(ckpt).exists():
        raise SystemExit(f"Fine-tuned checkpoint required for B3: {ckpt}")
    print(f"Loading fine-tuned VLM for B3: {ckpt}")
    ft_model, ft_proc = load_finetuned(cfg, ckpt)

    def b3(acts, img, task, _m=ft_model, _p=ft_proc):
        return score_vlm(acts, model=_m, processor=_p, image=img, task=task)

    scorers["B3_lora"] = b3

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    results: dict[str, Any] = {
        "split": args.split,
        "seed": seed,
        "max_samples": max_samples,
        "k_candidates": len(candidate_actions(cfg)),
        "note": (
            "Oracle ranking uses action-only semantic labels (no proprioception). "
            "Horizon>1 reuses the same observation (no true next-frame)."
        ),
        "horizons": {},
    }
    for h in horizons:
        print(f"=== Horizon {h} ===")
        results["horizons"][str(h)] = evaluate_split(
            ds,
            cfg,
            scorers=scorers,
            max_samples=max_samples,
            horizon=h,
            seed=seed,
        )

    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    out_path = Path(out_dir) / f"planner_metrics_{args.split}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
