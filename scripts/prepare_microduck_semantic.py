#!/usr/bin/env python3
"""Convert MicroDuck raw logs → semantic JSONL for LoRA training.

Outcome schema (duck-adapted, LIBERO-compatible training format):
  moved=yes|no; progress=up|flat|down; ball_in_zone=yes|no;
  facing_goal=yes|no; success_near=yes|no; skill=walk|kick|hold
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from swm.utils.config import load_config
from swm.utils.io import ensure_dir, write_jsonl
from swm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/microduck_train.yaml")
    p.add_argument("--raw_dir", default=None)
    return p.parse_args()


def build_duck_outcome(row: dict, *, next_row: dict | None, episode_len: int) -> str:
    vx = float(row.get("vx", 0.0))
    moved = "yes" if abs(vx) > 0.05 else "no"
    # progress toward success
    if next_row is not None:
        if next_row.get("success") and not row.get("success"):
            progress = "up"
        elif row.get("success") and not next_row.get("success"):
            progress = "down"
        elif abs(vx) > 0.1:
            progress = "up"
        else:
            progress = "flat"
    else:
        progress = "up" if abs(vx) > 0.1 else "flat"
    ball_in = "yes" if row.get("ball_in_zone") else "no"
    facing = "yes" if row.get("facing_goal") else "no"
    remaining = episode_len - int(row.get("frame_idx", 0)) - 1
    success_near = (
        "yes"
        if row.get("success") or (remaining <= 5 and next_row and next_row.get("success"))
        else "no"
    )
    skill = str(row.get("skill") or "hold")
    if skill not in {"walk", "kick", "hold"}:
        skill = "walk" if abs(vx) > 0.05 else "hold"
    # Pack into a single line; extra fields after the LIBERO five for duck.
    return (
        f"moved={moved}; lifted=no; gripper=hold; progress={progress}; "
        f"success_near={success_near}; ball_in_zone={ball_in}; "
        f"facing_goal={facing}; skill={skill}"
    )


def action_vec(row: dict) -> list[float]:
    """Map (vx,yaw,skill) into a 7D vector for the existing prompt template."""
    vx = float(row.get("vx", 0.0))
    yaw = float(row.get("yaw", 0.0))
    skill = str(row.get("skill") or "hold")
    grip = 1.0 if skill == "kick" else 0.0
    return [vx, yaw, 0.0, 0.0, 0.0, 0.0, grip]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    raw = Path(args.raw_dir or cfg["paths"].get("raw_dir", "data/microduck_raw"))
    processed = ensure_dir(cfg["paths"]["processed_dir"])
    frames_out = ensure_dir(cfg["paths"]["frames_dir"])
    index_path = raw / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"Missing {index_path}; run scripts/record_microduck_logs.py first")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    rng = random.Random(int(cfg.get("seed", 42)))
    ep_ids = [e["episode_id"] for e in index]
    rng.shuffle(ep_ids)
    n_val = max(1, int(len(ep_ids) * float(cfg["data"].get("val_ratio", 0.15))))
    val_eps = set(ep_ids[:n_val])

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for meta in index:
        ep = int(meta["episode_id"])
        steps = []
        with Path(meta["path"]).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line))
        for i, row in enumerate(steps):
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            # copy / ensure frame under processed frames
            src = Path(row["image_path"])
            rel = src.name
            dest = frames_out / rel
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())
            out_row = {
                "episode_id": ep,
                "frame_idx": int(row["frame_idx"]),
                "task": row["task"],
                "task_kind": row.get("task_kind"),
                "action": action_vec(row),
                "outcome": build_duck_outcome(row, next_row=nxt, episode_len=len(steps)),
                "image_path": str(dest if dest.is_file() else src),
                "skill": row.get("skill"),
            }
            (val_rows if ep in val_eps else train_rows).append(out_row)

    n_train = write_jsonl(Path(processed) / "train.jsonl", train_rows)
    n_val = write_jsonl(Path(processed) / "val.jsonl", val_rows)
    print(f"Wrote {n_train} train / {n_val} val → {processed}")


if __name__ == "__main__":
    main()
