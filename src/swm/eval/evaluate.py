"""Layer A evaluation: semantic outcome prediction vs zero-shot + ablations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from swm.data.dataset import SemanticTransitionDataset
from swm.data.prompts import build_prompt
from swm.eval.metrics import summarize_predictions
from swm.models.semantic_vlm import (
    build_model_and_processor,
    generate_outcome,
    load_finetuned,
)
from swm.utils.config import load_config
from swm.utils.io import ensure_dir
from swm.utils.seed import set_seed


def _blank_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
    return Image.new("RGB", size, color=(128, 128, 128))


def _shuffle_action(action: list[float], rng: random.Random) -> list[float]:
    """Permute action channels so the vector is valid-looking but wrong."""
    a = list(action) + [0.0] * 7
    a = a[:7]
    order = list(range(7))
    rng.shuffle(order)
    return [float(a[i]) for i in order]


def collect_predictions(
    model,
    processor,
    ds: SemanticTransitionDataset,
    max_samples: int,
    *,
    format_hint: bool = False,
    blank_image: bool = False,
    shuffle_action: bool = False,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    preds: list[str] = []
    golds: list[str] = []
    n = min(len(ds), max_samples)
    for i in tqdm(range(n), desc="eval"):
        sample = ds[i]
        image = _blank_image(sample["image"].size) if blank_image else sample["image"]
        action = list(sample["action"]) if sample.get("action") else [0.0] * 7
        if shuffle_action:
            action = _shuffle_action(action, rng)
        prompt = build_prompt(sample["task"], action, format_hint=format_hint)
        pred = generate_outcome(
            model,
            processor,
            image,
            sample["task"],
            action,
            prompt=prompt,
        )
        preds.append(pred)
        golds.append(sample["answer"])
    return preds, golds


def run_model(
    model,
    processor,
    ds: SemanticTransitionDataset,
    max_samples: int,
    *,
    format_hint: bool = False,
    blank_image: bool = False,
    shuffle_action: bool = False,
    seed: int = 42,
) -> dict:
    preds, golds = collect_predictions(
        model,
        processor,
        ds,
        max_samples,
        format_hint=format_hint,
        blank_image=blank_image,
        shuffle_action=shuffle_action,
        seed=seed,
    )
    metrics = summarize_predictions(preds, golds)
    metrics["format_hint"] = format_hint
    metrics["blank_image"] = blank_image
    metrics["shuffle_action"] = shuffle_action
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test"),
        help="Which processed JSONL split to evaluate.",
    )
    p.add_argument(
        "--zero_shot_only",
        action="store_true",
        help="Skip the fine-tuned checkpoint (re-score the frozen baseline).",
    )
    p.add_argument(
        "--finetuned_only",
        action="store_true",
        help="Skip zero-shot baseline.",
    )
    p.add_argument(
        "--blank_image",
        action="store_true",
        help="Ablation A3: replace the frame with a gray image (text-only).",
    )
    p.add_argument(
        "--shuffle_action",
        action="store_true",
        help="Ablation A2: shuffle action channels at eval time.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override config eval.max_samples.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for the metrics filename, e.g. blank_image.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    processed = Path(cfg["paths"]["processed_dir"])
    frames = Path(cfg["paths"]["frames_dir"])
    split_path = processed / f"{args.split}.jsonl"
    if not split_path.is_file():
        # Fall back to val if test was never written.
        if args.split == "test" and (processed / "val.jsonl").is_file():
            print("test.jsonl missing; falling back to val.jsonl")
            split_path = processed / "val.jsonl"
        else:
            raise SystemExit(f"Split not found: {split_path}")

    ds = SemanticTransitionDataset(split_path, frames)
    max_samples = int(args.max_samples or cfg["eval"].get("max_samples", 512))
    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    seed = int(cfg.get("seed", 42))

    ablation = {
        "blank_image": bool(args.blank_image),
        "shuffle_action": bool(args.shuffle_action),
        "split": args.split,
    }
    results: dict = {"ablation": ablation, "seed": seed}

    run_zs = cfg["eval"].get("zero_shot_baseline", True) and not args.finetuned_only
    if run_zs:
        print("Evaluating frozen zero-shot baseline (with output format hint)...")
        zs_model, zs_proc = build_model_and_processor(cfg, for_training=False)
        results["zero_shot"] = run_model(
            zs_model,
            zs_proc,
            ds,
            max_samples,
            format_hint=True,
            blank_image=args.blank_image,
            shuffle_action=args.shuffle_action,
            seed=seed,
        )
        del zs_model

    ckpt = args.checkpoint or cfg["eval"]["checkpoint"]
    if args.zero_shot_only:
        print("Skipping fine-tuned eval (--zero_shot_only).")
    elif Path(ckpt).exists():
        print(f"Evaluating fine-tuned checkpoint: {ckpt}")
        ft_model, ft_proc = load_finetuned(cfg, ckpt)
        results["finetuned"] = run_model(
            ft_model,
            ft_proc,
            ds,
            max_samples,
            blank_image=args.blank_image,
            shuffle_action=args.shuffle_action,
            seed=seed,
        )
    else:
        print(f"Checkpoint not found ({ckpt}); skipped fine-tuned eval.")

    tag = args.tag.strip()
    if not tag:
        bits = [args.split]
        if args.blank_image:
            bits.append("blank_image")
        if args.shuffle_action:
            bits.append("shuffle_action")
        tag = "_".join(bits) if bits != ["val"] else ""
    name = "eval_metrics.json" if not tag else f"eval_metrics_{tag}.json"
    out_path = Path(out_dir) / name

    if args.zero_shot_only and out_path.is_file() and not args.blank_image and not args.shuffle_action:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        previous["zero_shot"] = results.get("zero_shot", previous.get("zero_shot"))
        previous["ablation"] = ablation
        previous["seed"] = seed
        results = previous

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
