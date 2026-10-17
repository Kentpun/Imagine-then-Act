#!/usr/bin/env python3
"""Create Part B smoke fixtures (placeholder RGB + labeled commands)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from swm.microduck.labeling import build_outcome_string, label_command
from swm.utils.io import write_jsonl


SAMPLES = [
    {
        "task": "walk forward without falling",
        "action": [0.40, 0.0, 0.0],
        "goal_near": False,
    },
    {
        "task": "turn left in place",
        "action": [0.0, 0.0, 0.50],
        "goal_near": False,
    },
    {
        "task": "approach the marker ahead",
        "action": [0.55, 0.0, 0.0],
        "goal_near": True,
    },
    {
        "task": "stop and stay upright",
        "action": [0.0, 0.0, 0.0],
        "goal_near": False,
    },
]


def make_placeholder(path: Path, title: str) -> None:
    img = Image.new("RGB", (640, 480), color=(36, 48, 62))
    draw = ImageDraw.Draw(img)
    # Simple “duck-ish” body for a recognizable smoke frame
    draw.ellipse([220, 160, 420, 360], fill=(240, 190, 50))
    draw.ellipse([360, 120, 460, 220], fill=(240, 190, 50))
    draw.polygon([(450, 160), (520, 175), (450, 190)], fill=(230, 120, 40))
    draw.rectangle([280, 340, 310, 430], fill=(40, 40, 40))
    draw.rectangle([330, 340, 360, 430], fill=(40, 40, 40))
    draw.text((24, 24), f"MicroDuck smoke · {title}", fill=(230, 230, 230))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=str, default="data/microduck_smoke")
    args = p.parse_args()

    out = Path(args.out_dir)
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, sample in enumerate(SAMPLES):
        image_path = frames / f"sample_{i:03d}.jpg"
        make_placeholder(image_path, sample["task"][:40])
        labels = label_command(
            sample["action"], upright=True, goal_near=bool(sample["goal_near"])
        )
        rows.append(
            {
                "episode_id": i,
                "image_path": str(image_path),
                "task": sample["task"],
                "action": sample["action"],
                "outcome": build_outcome_string(labels),
                "source": "microduck_smoke",
            }
        )

    write_jsonl(out / "samples.jsonl", rows)
    print(f"Wrote {len(rows)} smoke samples → {out}")


if __name__ == "__main__":
    main()
