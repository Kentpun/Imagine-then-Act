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

from swm.microduck.semantic_tasks import (
    SceneState,
    TaskKind,
    label_microduck_transition,
    step_scene,
)
from swm.utils.config import load_config
from swm.utils.io import ensure_dir, write_jsonl
from swm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/microduck_train.yaml")
    p.add_argument("--raw_dir", default=None)
    return p.parse_args()


def _scene_from_row(row: dict, *, use_next: bool = False) -> SceneState:
    prefix = "next_" if use_next else ""
    return SceneState(
        duck_x=float(row.get(f"{prefix}duck_x", row["duck_x"])),
        duck_y=float(row.get(f"{prefix}duck_y", row["duck_y"])),
        duck_yaw=float(row.get(f"{prefix}duck_yaw", row["duck_yaw"])),
        ball_x=float(row.get(f"{prefix}ball_x", row["ball_x"])),
        ball_y=float(row.get(f"{prefix}ball_y", row["ball_y"])),
        zone_cx=float(row.get("zone_cx", 0.0)),
        zone_cy=float(row.get("zone_cy", 0.0)),
        zone_r=float(row.get("zone_r", 0.35)),
        goal_x=float(row.get("goal_x", 1.2)),
        goal_y=float(row.get("goal_y", 0.0)),
        zone_b_x=float(row.get("zone_b_x", 1.0)),
        zone_b_y=float(row.get("zone_b_y", 0.5)),
    )


def build_duck_outcome(row: dict, *, next_row: dict | None, episode_len: int) -> str:
    del episode_len
    vx = float(row.get("vx", 0.0))
    yaw = float(row.get("yaw", 0.0))
    skill = str(row.get("skill") or ("walk" if abs(vx) > 0.05 else "hold"))
    task_kind: TaskKind = row.get("task_kind", "walk_ab")

    s0 = _scene_from_row(row, use_next=False)
    if "next_duck_x" in row:
        s1 = _scene_from_row(row, use_next=True)
    elif next_row is not None:
        s1 = _scene_from_row(next_row, use_next=False)
    else:
        s1 = step_scene(
            s0,
            0.0 if skill == "kick" else max(0.0, vx),
            0.0 if skill == "kick" else yaw,
            skill=skill,
        )
    return label_microduck_transition(task_kind, s0, s1, vx, yaw, skill=skill)


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
            scene_state = {
                k: float(row[k])
                for k in (
                    "duck_x",
                    "duck_y",
                    "duck_yaw",
                    "ball_x",
                    "ball_y",
                    "zone_cx",
                    "zone_cy",
                    "zone_r",
                    "goal_x",
                    "goal_y",
                    "zone_b_x",
                    "zone_b_y",
                )
                if k in row and row[k] is not None
            }
            out_row = {
                "episode_id": ep,
                "frame_idx": int(row["frame_idx"]),
                "episode_len": len(steps),
                "task": row["task"],
                "task_kind": row.get("task_kind"),
                "action": action_vec(row),
                "scene_state": scene_state or None,
                "outcome": build_duck_outcome(row, next_row=nxt, episode_len=len(steps)),
                "image_path": str(dest if dest.is_file() else src),
                "skill": row.get("skill"),
            }
            (val_rows if ep in val_eps else train_rows).append(out_row)

    rng.shuffle(train_rows)
    n_train = write_jsonl(Path(processed) / "train.jsonl", train_rows)
    n_val = write_jsonl(Path(processed) / "val.jsonl", val_rows)
    print(f"Wrote {n_train} train / {n_val} val → {processed}")


if __name__ == "__main__":
    main()
