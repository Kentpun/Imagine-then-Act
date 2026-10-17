#!/usr/bin/env python3
"""Download public LIBERO dataset from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", type=str, default="lerobot/libero")
    p.add_argument("--out_dir", type=str, default="data/raw/libero")
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Skip download; prepare_dataset.py can synthesize a tiny set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        (out / "SMOKE_TEST").write_text("ok\n", encoding="utf-8")
        print(f"Smoke-test marker written to {out / 'SMOKE_TEST'}")
        return

    print(f"Downloading {args.repo_id} -> {out} ...")
    print("(Includes data/, meta/, and videos/ MP4s — needed for images.)")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    videos = Path(path) / "videos"
    if not videos.is_dir():
        raise RuntimeError(
            f"Download finished but {videos} is missing. "
            "Camera frames will not load; delete the folder and re-download."
        )
    print(f"Done: {path}")
    print(f"Videos present: {videos}")


if __name__ == "__main__":
    main()
