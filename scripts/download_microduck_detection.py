#!/usr/bin/env python3
"""Download pngwn/microduck-detection-dataset (YOLO) for CCTV duck perception."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_id", default="pngwn/microduck-detection-dataset")
    p.add_argument("--out_dir", default="data/raw/microduck_detection")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} → {out}")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out),
    )
    print(f"Done: {path}")
    for split in ("train", "val"):
        img = out / "images" / split
        lab = out / "labels" / split
        print(f"  images/{split}: {img.is_dir()}  labels/{split}: {lab.is_dir()}")


if __name__ == "__main__":
    main()
