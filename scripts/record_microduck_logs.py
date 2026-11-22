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
    p.add_argument("--episodes_per_task", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument(
        "--behavior_mode",
        choices=("mixed_play", "epsilon_expert", "expert"),
        default="mixed_play",
        help=(
            "mixed_play: realistic suboptimal teleop + exploratory play "
            "(20%% expert, 45%% myopic forward walk, 35%% uniform 8-action play); "
            "epsilon_expert: (1 - explore_prob) expert + explore_prob random; "
            "expert: 100%% privileged_command."
        ),
    )
    p.add_argument(
        "--explore_prob",
        type=float,
        default=0.35,
        help="Exploration probability when --behavior_mode=epsilon_expert.",
    )
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

    explore_actions: list[tuple[float, float, str]] = [
        (0.0, 0.0, "hold"),
        (0.15, 0.0, "walk"),
        (0.15, 0.5, "walk"),
        (0.15, -0.5, "walk"),
        (0.25, 0.0, "walk"),
        (0.25, 0.5, "walk"),
        (0.25, -0.5, "walk"),
        (0.0, 0.0, "kick"),
    ]

    ep_id = 0
    for task_name in args.tasks:
        task = task_name  # type: ignore[assignment]
        prompt = TASK_PROMPTS[task]  # type: ignore[index]
        for _ in range(args.episodes_per_task):
            s = random_scene(task, rng)  # type: ignore[arg-type]
            steps = []
            for t in range(args.max_steps):
                u = rng.random()
                if args.behavior_mode == "expert":
                    vx, yaw, skill = privileged_command(task, s)  # type: ignore[arg-type]
                elif args.behavior_mode == "epsilon_expert":
                    if u < float(args.explore_prob):
                        vx, yaw, skill = rng.choice(explore_actions)
                    else:
                        vx, yaw, skill = privileged_command(task, s)  # type: ignore[arg-type]
                else:
                    # mixed_play: 20% expert, 45% myopic cautious walker, 35% uniform play
                    if u < 0.20:
                        vx, yaw, skill = privileged_command(task, s)  # type: ignore[arg-type]
                    elif u < 0.65:
                        vx, yaw, skill = 0.15, 0.0, "walk"
                    else:
                        vx, yaw, skill = rng.choice(explore_actions)
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
                    "next_duck_x": s_next.duck_x,
                    "next_duck_y": s_next.duck_y,
                    "next_duck_yaw": s_next.duck_yaw,
                    "next_ball_x": s_next.ball_x,
                    "next_ball_y": s_next.ball_y,
                    "zone_cx": s.zone_cx,
                    "zone_cy": s.zone_cy,
                    "zone_r": s.zone_r,
                    "goal_x": s.goal_x,
                    "goal_y": s.goal_y,
                    "zone_b_x": s.zone_b_x,
                    "zone_b_y": s.zone_b_y,
                    "ball_in_zone": ball_in_zone(s_next),
                    "facing_goal": facing_goal(
                        s_next,
                        s_next.zone_b_x if task == "walk_ab" else s_next.goal_x,
                        s_next.zone_b_y if task == "walk_ab" else s_next.goal_y,
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
