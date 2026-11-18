#!/usr/bin/env python3
"""System-level MicroDuck success table (kinematic privileged vs policies).

Baselines: random | geometric (privileged) | microduck_lora (optional checkpoint)
Uses synthetic scenes (no MuJoCo) so it runs anywhere; MQTT demos are separate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from swm.microduck.semantic_tasks import (
    TASK_PROMPTS,
    TaskKind,
    ball_in_zone,
    privileged_command,
    random_scene,
    render_topdown,
    step_scene,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/microduck_semantic/system_eval.json")
    p.add_argument("--checkpoint", default=None, help="Optional MicroDuck LoRA")
    p.add_argument("--config", default="configs/microduck_train.yaml")
    return p.parse_args()


def success(task: TaskKind, s) -> bool:
    if task == "walk_ab":
        return (s.duck_x - s.zone_b_x) ** 2 + (s.duck_y - s.zone_b_y) ** 2 < 0.09
    if task == "kick_goal":
        return (s.ball_x - s.goal_x) ** 2 + (s.ball_y - s.goal_y) ** 2 < 0.12
    return ball_in_zone(s)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tasks: list[TaskKind] = ["walk_ab", "kick_goal", "keep_zone", "conditional"]

    model = processor = planner = None
    ckpt = args.checkpoint
    if ckpt is None:
        cand = Path("outputs/microduck_semantic/checkpoint-final")
        if cand.exists():
            ckpt = str(cand)
    if ckpt and Path(ckpt).exists():
        from swm.microduck.planner import MicroDuckImaginationPlanner
        from swm.models.semantic_vlm import load_finetuned
        from swm.utils.config import load_config

        cfg = load_config(args.config)
        model, processor = load_finetuned(cfg, ckpt)
        planner = MicroDuckImaginationPlanner(
            cfg, model=model, processor=processor, mock=False
        )
        print(f"Loaded LoRA for system eval: {ckpt}")
    else:
        print("No MicroDuck LoRA — geometric/random only (LoRA column = 0)")

    results: dict = {"n": args.n, "checkpoint": ckpt, "tasks": {}}
    for task in tasks:
        counts = {"random": 0, "geometric": 0, "microduck_lora": 0}
        for _ in range(args.n):
            s = random_scene(task, rng)
            for _t in range(80):
                vx, yaw, skill = privileged_command(task, s)
                s = step_scene(s, vx, yaw, skill=skill)
                if success(task, s):
                    counts["geometric"] += 1
                    break

            s = random_scene(task, rng)
            for _t in range(80):
                vx = rng.choice([0.0, 0.25])
                yaw = rng.choice([-0.5, 0.0, 0.5]) if vx > 0 else 0.0
                s = step_scene(s, vx, yaw)
                if success(task, s):
                    counts["random"] += 1
                    break

            if planner is not None:
                s = random_scene(task, rng)
                for _t in range(80):
                    img = render_topdown(s)
                    best = planner.plan(img, TASK_PROMPTS[task])
                    yaw = float(best.action[2])
                    if abs(yaw - 9.0) < 1e-6:
                        s = step_scene(s, 0.0, 0.0, skill="kick")
                    else:
                        s = step_scene(s, float(best.action[0]), yaw)
                    if success(task, s):
                        counts["microduck_lora"] += 1
                        break

        results["tasks"][task] = {k: v / args.n for k, v in counts.items()}
        print(task, results["tasks"][task])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
