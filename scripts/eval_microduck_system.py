#!/usr/bin/env python3
"""Closed-loop MicroDuck system evaluation across matched initial scenes.

Compares 7 controllers on identical seeded episodes per task:
  1. random: uniform legal (vx, yaw[/kick]) command
  2. blind_heuristic: image-blind constant forward command (B1 analog)
  3. zero_shot_policy: base frozen Qwen3-VL-2B direct action policy
  4. react_vlm: base Qwen3-VL-2B ReAct (verbal Thought -> Action)
  5. direct_policy: equal-compute fine-tuned VLA LoRA (--policy_checkpoint)
  6. imagine_lora: Imagine-then-Act semantic world model LoRA (--checkpoint)
  7. privileged_oracle: ground-truth scene geometry upper bound

Evaluates one VLM checkpoint in VRAM at a time over pre-sampled initial scenes
so memory stays low (~6 GB) and incremental JSON progress is saved continuously.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path
from typing import Any

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
    p.add_argument("--n", type=int, default=15, help="Episodes per task")
    p.add_argument("--max_steps", type=int, default=60, help="Max sim steps per episode")
    p.add_argument(
        "--action_repeat",
        type=int,
        default=2,
        help="Steps to hold each VLM decision (1 = replan every 0.2s step)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/microduck_semantic/system_eval.json")
    p.add_argument("--checkpoint", default=None, help="MicroDuck semantic world-model LoRA")
    p.add_argument(
        "--policy_checkpoint",
        default=None,
        help="Equal-compute MicroDuck direct-policy VLA LoRA (B4_direct_policy)",
    )
    p.add_argument(
        "--skip_base_vlm",
        action="store_true",
        help="Skip zero_shot_policy and react_vlm (faster smoke test)",
    )
    p.add_argument("--config", default="configs/microduck_train.yaml")
    p.add_argument("--planner_mode", choices=("grid", "propose"), default="grid")
    p.add_argument("--score_mode", choices=("heuristic", "vlm_preference"), default="heuristic")
    return p.parse_args()


def success(task: TaskKind, s) -> bool:
    if task == "walk_ab":
        return (s.duck_x - s.zone_b_x) ** 2 + (s.duck_y - s.zone_b_y) ** 2 < 0.09
    if task == "kick_goal":
        return (s.ball_x - s.goal_x) ** 2 + (s.ball_y - s.goal_y) ** 2 < 0.1225
    return ball_in_zone(s)


def goal_distance(task: TaskKind, s) -> float:
    if task == "walk_ab":
        return math.hypot(s.duck_x - s.zone_b_x, s.duck_y - s.zone_b_y)
    if task == "kick_goal":
        return math.hypot(s.ball_x - s.goal_x, s.ball_y - s.goal_y)
    return max(0.0, math.hypot(s.ball_x - s.zone_cx, s.ball_y - s.zone_cy) - s.zone_r)


def _apply_act3(s, act3: list[float]):
    yaw = float(act3[2]) if len(act3) > 2 else 0.0
    if abs(yaw - 9.0) < 1e-6:
        return step_scene(s, 0.0, 0.0, skill="kick")
    return step_scene(s, float(act3[0]), yaw)


def _run_episode(
    task: TaskKind,
    init_state,
    *,
    max_steps: int,
    action_repeat: int,
    choose_fn,
    hold_steps_on_conditional: int = 5,
) -> dict[str, float]:
    s = init_state
    consec_inside = 0
    repeat = max(1, int(action_repeat))
    cached_act: list[float] | None = None

    for t in range(1, max_steps + 1):
        if cached_act is None or (t - 1) % repeat == 0:
            cached_act = choose_fn(s, t - 1)
        s = _apply_act3(s, cached_act)

        if success(task, s):
            if task == "conditional":
                consec_inside += 1
                if consec_inside >= hold_steps_on_conditional:
                    return {
                        "success": 1.0,
                        "steps": float(t),
                        "final_dist": goal_distance(task, s),
                    }
            else:
                return {
                    "success": 1.0,
                    "steps": float(t),
                    "final_dist": goal_distance(task, s),
                }
        else:
            consec_inside = 0

    return {
        "success": 0.0,
        "steps": float(max_steps),
        "final_dist": goal_distance(task, s),
    }


def _free_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    tasks: list[TaskKind] = ["walk_ab", "kick_goal", "keep_zone", "conditional"]

    ckpt = args.checkpoint
    if ckpt is None:
        cand = Path("outputs/microduck_semantic/checkpoint-final")
        if cand.exists():
            ckpt = str(cand)

    pol_ckpt = args.policy_checkpoint
    if pol_ckpt is None:
        cand_pol = Path("outputs/microduck_direct_policy/checkpoint-final")
        if cand_pol.exists():
            pol_ckpt = str(cand_pol)

    # Pre-sample identical initial scenes across every controller
    episodes: dict[TaskKind, list[Any]] = {
        task: [
            random_scene(task, random.Random(args.seed + t_idx * 10000 + ep))
            for ep in range(args.n)
        ]
        for t_idx, task in enumerate(tasks)
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "n_episodes_per_task": args.n,
        "max_steps": args.max_steps,
        "action_repeat": args.action_repeat,
        "seed": args.seed,
        "semantic_checkpoint": ckpt,
        "policy_checkpoint": pol_ckpt,
        "planner_mode": args.planner_mode,
        "score_mode": args.score_mode,
        "tasks": {t: {} for t in tasks},
        "macro_average": {},
    }

    def _evaluate_controller(name: str, make_choose_fn, repeat: int) -> None:
        print(f"\n=== Evaluating controller: {name} ===", flush=True)
        for t_idx, task in enumerate(tasks):
            agg = {"success_rate": 0.0, "mean_steps": 0.0, "mean_final_dist": 0.0}
            for ep, init_scene in enumerate(episodes[task]):
                ep_rng = random.Random(args.seed + t_idx * 10000 + ep + 999)
                m = _run_episode(
                    task,
                    init_scene,
                    max_steps=args.max_steps,
                    action_repeat=repeat,
                    choose_fn=make_choose_fn(task, ep_rng),
                )
                agg["success_rate"] += m["success"] / args.n
                agg["mean_steps"] += m["steps"] / args.n
                agg["mean_final_dist"] += m["final_dist"] / args.n
            results["tasks"][task][name] = agg
            print(f"  {task}: {json.dumps(agg)}", flush=True)
        results["macro_average"][name] = {
            metric: sum(results["tasks"][t][name][metric] for t in tasks) / len(tasks)
            for metric in ("success_rate", "mean_steps", "mean_final_dist")
        }
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # 1. Non-VLM baselines
    _evaluate_controller(
        "random",
        lambda _task, rng: lambda _s, _step: list(
            rng.choice(
                [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.0, 0.0],
                    [0.25, 0.0, 0.5],
                    [0.25, 0.0, -0.5],
                    [0.0, 0.0, 9.0],
                ]
            )
        ),
        repeat=1,
    )
    _evaluate_controller(
        "blind_heuristic",
        lambda _task, _rng: lambda _s, _step: [0.25, 0.0, 0.0],
        repeat=1,
    )
    _evaluate_controller(
        "privileged_oracle",
        lambda task, _rng: lambda s, _step: (
            [0.0, 0.0, 9.0]
            if privileged_command(task, s)[2] == "kick"
            else [privileged_command(task, s)[0], 0.0, privileged_command(task, s)[1]]
        ),
        repeat=1,
    )

    need_vlm = (
        (not args.skip_base_vlm)
        or (pol_ckpt and Path(pol_ckpt).exists())
        or (ckpt and Path(ckpt).exists())
    )
    if need_vlm:
        from swm.models.direct_policy import (
            predict_microduck_policy_action,
            predict_microduck_react_action,
        )
        from swm.models.semantic_vlm import build_model_and_processor, load_finetuned
        from swm.utils.config import load_config

        cfg = load_config(args.config)
        cfg.setdefault("planner", {})
        cfg["planner"]["merge_lora"] = True
        cfg["planner"]["mode"] = args.planner_mode
        cfg["planner"]["score_mode"] = args.score_mode
        cfg["planner"]["max_new_tokens"] = 48
        cfg["planner"]["vx_values"] = [0.0, 0.15, 0.25]
        cfg["planner"]["yaw_values"] = [-0.5, 0.0, 0.5]
        cfg["planner"]["include_kick"] = True
        cfg["planner"]["max_candidates"] = 8

        # 2. Base Zero-Shot + ReAct on frozen Qwen3-VL-2B
        if not args.skip_base_vlm:
            base_model, base_proc = build_model_and_processor(cfg, for_training=False)
            _evaluate_controller(
                "zero_shot_policy",
                lambda task, _rng: lambda s, _step: predict_microduck_policy_action(
                    base_model,
                    base_proc,
                    render_topdown(s),
                    TASK_PROMPTS[task],
                    allow_kick=True,
                    format_example=True,
                )[0],
                repeat=args.action_repeat,
            )
            _evaluate_controller(
                "react_vlm",
                lambda task, _rng: lambda s, _step: predict_microduck_react_action(
                    base_model,
                    base_proc,
                    render_topdown(s),
                    TASK_PROMPTS[task],
                    allow_kick=True,
                )[0],
                repeat=args.action_repeat,
            )
            del base_model, base_proc
            _free_vram()

        # 3. Equal-compute fine-tuned VLA (direct_policy)
        if pol_ckpt and Path(pol_ckpt).exists():
            pol_model, pol_proc = load_finetuned(cfg, pol_ckpt)
            _evaluate_controller(
                "direct_policy_lora",
                lambda task, _rng: lambda s, _step: predict_microduck_policy_action(
                    pol_model,
                    pol_proc,
                    render_topdown(s),
                    TASK_PROMPTS[task],
                    allow_kick=True,
                )[0],
                repeat=args.action_repeat,
            )
            del pol_model, pol_proc
            _free_vram()

        # 4. Imagine-then-Act semantic world model LoRA
        if ckpt and Path(ckpt).exists():
            from swm.microduck.planner import MicroDuckImaginationPlanner

            wm_model, wm_proc = load_finetuned(cfg, ckpt)
            planner = MicroDuckImaginationPlanner(
                cfg, model=wm_model, processor=wm_proc, mock=False
            )
            _evaluate_controller(
                "imagine_then_act_lora",
                lambda task, _rng: lambda s, _step: list(
                    planner.plan(render_topdown(s), TASK_PROMPTS[task]).action
                ),
                repeat=args.action_repeat,
            )
            del wm_model, wm_proc, planner
            _free_vram()

    print("\n=== Macro Average Across All 4 Tasks ===")
    print(json.dumps(results["macro_average"], indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
