from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from swm.data.dataset import SemanticTransitionDataset
from swm.data.prompts import build_prompt
from swm.models.semantic_vlm import build_model_and_processor, generate_outcome, load_finetuned
from swm.utils.config import load_config
from swm.utils.io import ensure_dir
from swm.utils.seed import set_seed


FIELD_RE = re.compile(
    r"moved=(yes|no);\s*lifted=(yes|no);\s*gripper=(open|close|hold);\s*"
    r"progress=(up|flat|down);\s*success_near=(yes|no)",
    re.IGNORECASE,
)


def parse_fields(text: str) -> dict[str, str] | None:
    m = FIELD_RE.search(text.replace(" ", ""))
    if not m:
        # try loose parse
        out = {}
        for key in ("moved", "lifted", "gripper", "progress", "success_near"):
            mm = re.search(rf"{key}=([a-z]+)", text.lower())
            if not mm:
                return None
            out[key] = mm.group(1)
        return out
    return {
        "moved": m.group(1).lower(),
        "lifted": m.group(2).lower(),
        "gripper": m.group(3).lower(),
        "progress": m.group(4).lower(),
        "success_near": m.group(5).lower(),
    }


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower().replace(" ", "") == gold.strip().lower().replace(" ", "")


def field_accuracy(pred: str, gold: str) -> dict[str, float]:
    p = parse_fields(pred)
    g = parse_fields(gold)
    if p is None or g is None:
        return {k: 0.0 for k in ("moved", "lifted", "gripper", "progress", "success_near")}
    return {k: float(p.get(k) == g.get(k)) for k in g}


def run_model(
    model,
    processor,
    ds: SemanticTransitionDataset,
    max_samples: int,
    *,
    format_hint: bool = False,
):
    exact, field_sums, n, n_parsed = 0, defaultdict(float), 0, 0
    for i in tqdm(range(min(len(ds), max_samples)), desc="eval"):
        sample = ds[i]
        prompt = build_prompt(sample["task"], sample["action"], format_hint=format_hint)
        pred = generate_outcome(
            model,
            processor,
            sample["image"],
            sample["task"],
            sample["action"],
            prompt=prompt,
        )
        gold = sample["answer"]
        parsed = parse_fields(pred) is not None and parse_fields(gold) is not None
        n_parsed += float(parsed)
        exact += float(exact_match(pred, gold))
        for k, v in field_accuracy(pred, gold).items():
            field_sums[k] += v
        n += 1
    metrics = {
        "exact_match": exact / max(1, n),
        "parse_rate": n_parsed / max(1, n),
        "n": n,
        "format_hint": format_hint,
    }
    metrics.update({f"acc_{k}": field_sums[k] / max(1, n) for k in field_sums})
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--zero_shot_only",
        action="store_true",
        help="Skip the fine-tuned checkpoint (re-score the frozen baseline).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    processed = Path(cfg["paths"]["processed_dir"])
    frames = Path(cfg["paths"]["frames_dir"])
    ds = SemanticTransitionDataset(processed / "val.jsonl", frames)
    max_samples = int(cfg["eval"].get("max_samples", 512))
    out_dir = ensure_dir(cfg["paths"]["output_dir"])

    results = {}
    # Format hint is eval-only for the frozen model. The LoRA checkpoint was
    # trained without it, so fine-tuned eval keeps the original prompt.
    if cfg["eval"].get("zero_shot_baseline", True):
        print("Evaluating frozen zero-shot baseline (with output format hint)...")
        zs_model, zs_proc = build_model_and_processor(cfg, for_training=False)
        results["zero_shot"] = run_model(
            zs_model, zs_proc, ds, max_samples, format_hint=True
        )
        del zs_model

    ckpt = args.checkpoint or cfg["eval"]["checkpoint"]
    if args.zero_shot_only:
        print("Skipping fine-tuned eval (--zero_shot_only).")
    elif Path(ckpt).exists():
        print(f"Evaluating fine-tuned checkpoint: {ckpt}")
        ft_model, ft_proc = load_finetuned(cfg, ckpt)
        results["finetuned"] = run_model(ft_model, ft_proc, ds, max_samples)
    else:
        print(f"Checkpoint not found ({ckpt}); skipped fine-tuned eval.")

    out_path = Path(out_dir) / "eval_metrics.json"
    if args.zero_shot_only and out_path.is_file():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        previous["zero_shot"] = results.get("zero_shot", previous.get("zero_shot"))
        results = previous
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
