#!/usr/bin/env python3
"""Compare zero-shot vs action-command LoRA on the prepared val set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from swm.data.action_commands import parse_action_command
from swm.data.dataset import SemanticTransitionDataset
from swm.models.semantic_vlm import (
    build_model_and_processor,
    generate_outcome,
    load_finetuned,
)
from swm.utils.config import load_config
from swm.utils.io import ensure_dir
from swm.utils.seed import set_seed


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower().replace(" ", "") == gold.strip().lower().replace(" ", "")


def field_scores(pred: str, gold: str) -> dict[str, float]:
    p = parse_action_command(pred)
    g = parse_action_command(gold)
    keys = ("target_visible", "action", "magnitude", "gripper")
    if p is None or g is None:
        return {k: 0.0 for k in keys}
    return {k: float(p.get(k) == g.get(k)) for k in keys}


def run_eval(model, processor, ds: SemanticTransitionDataset, max_samples: int) -> dict:
    exact = parse_ok = 0.0
    field_sums: dict[str, float] = defaultdict(float)
    n = 0
    for i in tqdm(range(min(len(ds), max_samples)), desc="eval"):
        sample = ds[i]
        pred = generate_outcome(
            model,
            processor,
            sample["image"],
            sample["task"],
            sample.get("action", []),
            prompt=sample["prompt"],
            max_new_tokens=48,
        )
        gold = sample["answer"]
        exact += float(exact_match(pred, gold))
        parse_ok += float(parse_action_command(pred) is not None)
        for key, value in field_scores(pred, gold).items():
            field_sums[key] += value
        n += 1
    metrics = {
        "exact_match": exact / max(1, n),
        "parse_rate": parse_ok / max(1, n),
        "n": n,
    }
    metrics.update({f"acc_{k}": field_sums[k] / max(1, n) for k in field_sums})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/superpod_action.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/superpod_action/checkpoint-final",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    processed = Path(cfg["paths"]["processed_dir"])
    frames = Path(cfg["paths"]["frames_dir"])
    ds = SemanticTransitionDataset(processed / "val.jsonl", frames)
    max_samples = int(args.max_samples or cfg["eval"].get("max_samples", 512))
    out_dir = ensure_dir(cfg["paths"]["output_dir"])

    results: dict[str, dict] = {}

    print("Evaluating zero-shot frozen Qwen3-VL-2B ...")
    base, processor = build_model_and_processor(cfg, for_training=False)
    results["zero_shot"] = run_eval(base, processor, ds, max_samples)
    print(json.dumps(results["zero_shot"], indent=2))
    del base

    print(f"Evaluating fine-tuned LoRA at {args.checkpoint} ...")
    model, processor = load_finetuned(cfg, args.checkpoint)
    results["finetuned"] = run_eval(model, processor, ds, max_samples)
    print(json.dumps(results["finetuned"], indent=2))

    out_path = out_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
