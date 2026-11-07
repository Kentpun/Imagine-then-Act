#!/usr/bin/env python3
"""Train a tiny YOLO detector on pngwn/microduck-detection-dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data/raw/microduck_detection")
    p.add_argument("--out_dir", default="outputs/microduck_detector")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics base checkpoint")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    return p.parse_args()


def _ensure_data_yaml(data_dir: Path) -> Path:
    yaml_path = data_dir / "data.yaml"
    if yaml_path.is_file():
        return yaml_path
    # Minimal YOLO data.yaml if missing
    text = (
        f"path: {data_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: microduck\n"
    )
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not (data_dir / "images" / "train").is_dir():
        raise SystemExit(
            f"Missing {data_dir}/images/train — run scripts/download_microduck_detection.py first"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required: pip install ultralytics"
        ) from exc

    data_yaml = _ensure_data_yaml(data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(out),
        name="yolo_microduck",
        device=args.device,
        exist_ok=True,
    )
    best = out / "yolo_microduck" / "weights" / "best.pt"
    print(f"Best weights: {best}")
    if best.is_file():
        link = out / "best.pt"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(best.resolve())
        print(f"Symlinked {link}")


if __name__ == "__main__":
    main()
