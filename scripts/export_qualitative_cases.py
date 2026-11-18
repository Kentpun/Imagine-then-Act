#!/usr/bin/env python3
"""Export 5 success + 5 failure qualitative LIBERO cases for the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swm.data.dataset import SemanticTransitionDataset
from swm.data.prompts import build_prompt
from swm.eval.metrics import exact_match
from swm.models.semantic_vlm import generate_outcome, load_finetuned
from swm.utils.config import load_config
from swm.utils.io import ensure_dir
from swm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/superpod.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--max_scan", type=int, default=256)
    p.add_argument("--n_each", type=int, default=5)
    p.add_argument("--out_dir", default="outputs/superpod/qualitative")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    processed = Path(cfg["paths"]["processed_dir"])
    frames = Path(cfg["paths"]["frames_dir"])
    ds = SemanticTransitionDataset(processed / f"{args.split}.jsonl", frames)
    ckpt = args.checkpoint or cfg["eval"]["checkpoint"]
    model, processor = load_finetuned(cfg, ckpt)

    successes: list[dict] = []
    failures: list[dict] = []
    n = min(len(ds), args.max_scan)
    out_dir = ensure_dir(args.out_dir)
    img_dir = ensure_dir(out_dir / "frames")

    for i in range(n):
        if len(successes) >= args.n_each and len(failures) >= args.n_each:
            break
        sample = ds[i]
        action = list(sample.get("action") or [0.0] * 7)
        pred = generate_outcome(
            model,
            processor,
            sample["image"],
            sample["task"],
            action,
            prompt=build_prompt(sample["task"], action),
        )
        gold = sample["answer"]
        ok = exact_match(pred, gold)
        bucket = successes if ok else failures
        if len(bucket) >= args.n_each:
            continue
        rel = f"{'ok' if ok else 'fail'}_{len(bucket):02d}.jpg"
        dest = img_dir / rel
        sample["image"].save(dest, quality=90)
        bucket.append(
            {
                "index": i,
                "task": sample["task"],
                "action": action,
                "gold": gold,
                "pred": pred,
                "exact_match": ok,
                "image": str(dest),
            }
        )

    report = {
        "checkpoint": str(ckpt),
        "n_success": len(successes),
        "n_failure": len(failures),
        "successes": successes,
        "failures": failures,
    }
    path = Path(out_dir) / "qualitative_cases.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = Path(out_dir) / "qualitative_cases.md"
    lines = ["# Qualitative cases (LIBERO val)\n"]
    for title, rows in (("Successes", successes), ("Failures", failures)):
        lines.append(f"## {title}\n")
        for r in rows:
            lines.append(f"### idx={r['index']}")
            lines.append(f"- task: `{r['task']}`")
            lines.append(f"- gold: `{r['gold']}`")
            lines.append(f"- pred: `{r['pred']}`")
            lines.append(f"- image: `{r['image']}`\n")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path} and {md}")


if __name__ == "__main__":
    main()
