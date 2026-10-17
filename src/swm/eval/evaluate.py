from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from swm.data.dataset import SemanticTransitionDataset
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


def run_model(model, processor, ds: SemanticTransitionDataset, max_samples: int):
    exact, field_sums, n = 0, defaultdict(float), 0
    for i in tqdm(range(min(len(ds), max_samples)), desc="eval"):
        sample = ds[i]
        pred = generate_outcome(
            model, processor, sample["image"], sample["task"], sample["action"]
        )
        gold = sample["answer"]
        exact += float(exact_match(pred, gold))
        for k, v in field_accuracy(pred, gold).items():
            field_sums[k] += v
        n += 1
    metrics = {"exact_match": exact / max(1, n), "n": n}
    metrics.update({f"acc_{k}": field_sums[k] / max(1, n) for k in field_sums})
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
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
    if cfg["eval"].get("zero_shot_baseline", True):
        print("Evaluating frozen zero-shot baseline...")
        zs_model, zs_proc = build_model_and_processor(cfg, for_training=False)
        results["zero_shot"] = run_model(zs_model, zs_proc, ds, max_samples)
        del zs_model

    ckpt = args.checkpoint or cfg["eval"]["checkpoint"]
    if Path(ckpt).exists():
        print(f"Evaluating fine-tuned checkpoint: {ckpt}")
        ft_model, ft_proc = load_finetuned(cfg, ckpt)
        results["finetuned"] = run_model(ft_model, ft_proc, ds, max_samples)
    else:
        print(f"Checkpoint not found ({ckpt}); skipped fine-tuned eval.")

    out_path = Path(out_dir) / "eval_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
