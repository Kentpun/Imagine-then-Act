#!/usr/bin/env python3
"""Convert semantic world-model splits into matched direct-policy JSONL splits.

Guarantees an apples-to-apples equal-compute baseline:
  - identical train/val frames and episodes as the semantic world model
  - identical Qwen3-VL-2B + LoRA architecture & hyperparameters via swm.training.train
  - supervision changes from (o_t, g, a_t) -> outcome  to  (o_t, g) -> a_t
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swm.models.direct_policy import (
    build_libero_policy_prompt,
    build_microduck_policy_prompt,
    format_libero_policy_target,
    format_microduck_policy_target,
)
from swm.utils.io import ensure_dir, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src_dir",
        required=True,
        help="Processed semantic dataset dir containing train.jsonl & val.jsonl",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Destination dir for direct-policy train.jsonl & val.jsonl",
    )
    p.add_argument(
        "--domain",
        choices=("libero", "microduck"),
        default="libero",
        help="Action formatting schema (libero 7D delta vs microduck vx/yaw/kick).",
    )
    return p.parse_args()


def convert_split(rows: list[dict], domain: str) -> list[dict]:
    converted: list[dict] = []
    for row in rows:
        task = str(row["task"])
        action = row.get("action") or [0.0] * 7
        if domain == "microduck":
            prompt = build_microduck_policy_prompt(task)
            answer = format_microduck_policy_target(action)
        else:
            prompt = build_libero_policy_prompt(task)
            answer = format_libero_policy_target(action)
        new_row = dict(row)
        new_row["prompt"] = prompt
        new_row["answer"] = answer
        converted.append(new_row)
    return converted


def main() -> None:
    args = parse_args()
    src = Path(args.src_dir)
    out = ensure_dir(args.out_dir)

    for split in ("train", "val", "test"):
        in_path = src / f"{split}.jsonl"
        if not in_path.is_file():
            continue
        rows = load_jsonl(in_path)
        converted = convert_split(rows, args.domain)
        n = write_jsonl(Path(out) / f"{split}.jsonl", converted)
        print(f"Wrote {n} {args.domain} direct-policy rows -> {Path(out) / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
