#!/usr/bin/env python3
"""Generate synthetic MicroDuck CCTV logs with privileged policies.

Produces episode JSONL + frames under data/microduck_raw/ for semantic prep.
Works offline without MuJoCo (kinematic top-down renderer). Optional --mujoco
hook is reserved for future live sim attachment.
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
    facing_goal,
    privileged_command,
    random_scene,
    render_topdown,
    step_scene,
)
from swm.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default="data/microduck_raw")
    p.add_argument("--episodes_per_task", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tasks",
        nargs="*",
        default=["walk_ab", "kick_goal", "keep_zone", "conditional"],
    )
    return p.parse_args()


def episode_success(task: TaskKind, s) -> bool:
    if task == "walk_ab":
        return math_hypot(s.duck_x - s.zone_b_x, s.duck_y - s.zone_b_y) < 0.3
    if task == "kick_goal":
        return math_hypot(s.ball_x - s.goal_x, s.ball_y - s.goal_y) < 0.35
    return ball_in_zone(s)


def math_hypot(a: float, b: float) -> float:
    return (a * a + b * b) ** 0.5


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out = ensure_dir(args.out_dir)
    frames = ensure_dir(out / "frames")
    ep_dir = ensure_dir(out / "episodes")
    index: list[dict] = []

    ep_id = 0
    for task_name in args.tasks:
        task = task_name  # type: ignore[assignment]
        prompt = TASK_PROMPTS[task]  # type: ignore[index]
        for _ in range(args.episodes_per_task):
            s = random_scene(task, rng)  # type: ignore[arg-type]
            steps = []
            for t in range(args.max_steps):
                vx, yaw, skill = privileged_command(task, s)  # type: ignore[arg-type]
                img = render_topdown(s)
                rel = f"ep{ep_id:05d}_t{t:04d}.jpg"
                img_path = frames / rel
                img.save(img_path, quality=90)
                s_next = step_scene(s, vx, yaw, skill=skill)
                success = episode_success(task, s_next)  # type: ignore[arg-type]
                row = {
                    "episode_id": ep_id,
                    "frame_idx": t,
                    "task_kind": task,
                    "task": prompt,
                    "vx": vx,
                    "yaw": yaw,
                    "skill": skill,
                    "duck_x": s.duck_x,
                    "duck_y": s.duck_y,
                    "duck_yaw": s.duck_yaw,
                    "ball_x": s.ball_x,
                    "ball_y": s.ball_y,
                    "zone_cx": s.zone_cx,
                    "zone_cy": s.zone_cy,
                    "zone_r": s.zone_r,
                    "goal_x": s.goal_x,
                    "goal_y": s.goal_y,
                    "zone_b_x": s.zone_b_x,
                    "zone_b_y": s.zone_b_y,
                    "ball_in_zone": ball_in_zone(s),
                    "facing_goal": facing_goal(
                        s,
                        s.zone_b_x if task == "walk_ab" else s.goal_x,
                        s.zone_b_y if task == "walk_ab" else s.goal_y,
                    ),
                    "image_path": str(img_path),
                    "success": success,
                }
                steps.append(row)
                s = s_next
                if success and t > 5:
                    break
            ep_path = ep_dir / f"ep{ep_id:05d}.jsonl"
            with ep_path.open("w", encoding="utf-8") as f:
                for row in steps:
                    f.write(json.dumps(row) + "\n")
            index.append(
                {
                    "episode_id": ep_id,
                    "task_kind": task,
                    "task": prompt,
                    "n_steps": len(steps),
                    "success": bool(steps and steps[-1]["success"]),
                    "path": str(ep_path),
                }
            )
            ep_id += 1

    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {ep_id} episodes under {out}")


if __name__ == "__main__":
    main()
