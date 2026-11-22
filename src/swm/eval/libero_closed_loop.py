"""Closed-loop LIBERO evaluation: B0 random vs B1 heuristic vs B3 Imagine-then-Act vs B4 Direct Policy.

Supports two backends:
  1. ``libero`` (default when ``libero`` + ``robosuite`` are installed): rolls
     out ``OffScreenRenderEnv`` across tasks in ``libero_spatial`` /
     ``libero_object`` / ``libero_goal`` / ``libero_10`` from official initial
     states, measuring closed-loop task success rate and episode length.
  2. ``synthetic_tabletop`` (``--backend synthetic`` or automatic fallback when
     ``--allow_synthetic_fallback`` is passed): lightweight 3D tabletop
     reach-grasp-lift kinematics + rendered RGB frames for pipeline validation
     and CPU/Mac smoke testing without MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from swm.data.action_commands import build_action_prompt, parse_action_command
from swm.data.labeling import LabelConfig
from swm.eval.planner_eval import _command_to_vec7, pick_top, score_heuristic
from swm.models.direct_policy import (
    predict_libero_policy_action,
    predict_libero_react_action,
)
from swm.models.semantic_vlm import (
    build_model_and_processor,
    generate_text,
    load_finetuned,
)
from swm.planning.imagination_planner import ImaginationPlanner, candidate_actions
from swm.utils.config import load_config
from swm.utils.io import ensure_dir
from swm.utils.seed import set_seed


def _to_robosuite_osc(action7: list[float], osc_scale: float = 10.0) -> np.ndarray:
    """Convert dataset-space 7D action (deltas ~0.05, grip in {0,1}) to robosuite OSC_POSE."""
    a = list(action7) + [0.0] * 7
    dx = max(-1.0, min(1.0, float(a[0]) * osc_scale))
    dy = max(-1.0, min(1.0, float(a[1]) * osc_scale))
    dz = max(-1.0, min(1.0, float(a[2]) * osc_scale))
    droll = max(-1.0, min(1.0, float(a[3]) * osc_scale))
    dpitch = max(-1.0, min(1.0, float(a[4]) * osc_scale))
    dyaw = max(-1.0, min(1.0, float(a[5]) * osc_scale))
    grip = 1.0 if float(a[6]) >= 0.5 else -1.0
    return np.array([dx, dy, dz, droll, dpitch, dyaw, grip], dtype=np.float32)


def _obs_to_pil(obs: dict[str, Any]) -> Image.Image:
    for key in ("agentview_image", "robot0_eye_in_hand_image", "image"):
        if key in obs and obs[key] is not None:
            arr = np.asarray(obs[key])
            # robosuite / MuJoCo offscreen buffers are upside-down by convention
            arr = np.flipud(arr)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
    raise KeyError(f"No RGB key in LIBERO obs: {list(obs.keys())}")


@dataclass
class SyntheticTabletopEnv:
    """Lightweight 3D reach-grasp-lift environment for MuJoCo-free testing."""

    task: str = "pick up the red block"
    eef_x: float = 0.0
    eef_y: float = 0.0
    eef_z: float = 0.12
    grip_closed: bool = False
    obj_x: float = 0.06
    obj_y: float = -0.04
    obj_z: float = 0.02
    grasped: bool = False

    @classmethod
    def random_init(cls, rng: random.Random, task_idx: int = 0) -> "SyntheticTabletopEnv":
        tasks = [
            "pick up the red block",
            "lift the blue mug from the table",
            "grasp the yellow bowl and lift it up",
        ]
        return cls(
            task=tasks[task_idx % len(tasks)],
            eef_x=rng.uniform(-0.06, 0.06),
            eef_y=rng.uniform(-0.06, 0.06),
            eef_z=rng.uniform(0.06, 0.12),
            grip_closed=False,
            obj_x=rng.uniform(-0.08, 0.08),
            obj_y=rng.uniform(-0.08, 0.08),
            obj_z=0.02,
            grasped=False,
        )

    def render(self) -> Image.Image:
        img = Image.new("RGB", (224, 224), (225, 220, 210))
        drw = ImageDraw.Draw(img)
        # Table surface
        drw.rectangle([20, 40, 204, 204], fill=(190, 165, 130), outline=(120, 95, 70))

        def to_px(x: float, y: float, z: float) -> tuple[int, int]:
            px = int(112 + x * 600)
            py = int(130 - y * 500 - z * 350)
            return max(8, min(216, px)), max(8, min(216, py))

        ox, oy = to_px(self.obj_x, self.obj_y, self.obj_z)
        drw.rectangle([ox - 10, oy - 10, ox + 10, oy + 10], fill=(210, 45, 45), outline=(90, 15, 15))
        ex, ey = to_px(self.eef_x, self.eef_y, self.eef_z)
        jaw = 4 if self.grip_closed else 11
        drw.line([ex, 10, ex, ey], fill=(60, 65, 75), width=5)
        drw.line([ex - jaw, ey, ex - jaw, ey + 10], fill=(40, 40, 50), width=3)
        drw.line([ex + jaw, ey, ex + jaw, ey + 10], fill=(40, 40, 50), width=3)
        return img

    def step(self, action7: list[float]) -> tuple[Image.Image, bool]:
        a = list(action7) + [0.0] * 7
        self.eef_x = max(-0.15, min(0.15, self.eef_x + float(a[0])))
        self.eef_y = max(-0.15, min(0.15, self.eef_y + float(a[1])))
        self.eef_z = max(0.01, min(0.25, self.eef_z + float(a[2])))
        want_close = float(a[6]) >= 0.5

        dist = math.sqrt(
            (self.eef_x - self.obj_x) ** 2
            + (self.eef_y - self.obj_y) ** 2
            + (self.eef_z - self.obj_z) ** 2
        )
        if want_close and dist < 0.045:
            self.grasped = True
        elif not want_close:
            self.grasped = False
            self.obj_z = 0.02

        self.grip_closed = want_close
        if self.grasped:
            self.obj_x, self.obj_y, self.obj_z = self.eef_x, self.eef_y, self.eef_z

        done = self.grasped and self.obj_z >= 0.08
        return self.render(), done


def _build_controllers(
    cfg: dict[str, Any],
    *,
    wm_ckpt: str | None,
    policy_ckpt: str | None,
    action_cmd_ckpt: str | None = None,
    include_base_vlm: bool = True,
) -> dict[str, Any]:
    base_cands = candidate_actions(cfg)
    label_cfg = LabelConfig(
        move_threshold=float(cfg.get("data", {}).get("move_threshold", 0.01)),
        lift_threshold=float(cfg.get("data", {}).get("lift_threshold", 0.01)),
        success_near_frames=int(cfg.get("data", {}).get("success_near_frames", 15)),
    )

    controllers: dict[str, Any] = {}

    def ctrl_random(_img: Image.Image, _task: str, rng: random.Random) -> list[float]:
        return list(rng.choice(base_cands))

    def ctrl_heuristic(_img: Image.Image, _task: str, _rng: random.Random) -> list[float]:
        scored = score_heuristic(base_cands, label_cfg)
        return list(pick_top(scored).action)

    controllers["B0_random"] = ctrl_random
    controllers["B1_heuristic"] = ctrl_heuristic

    cfg_infer = dict(cfg)
    cfg_infer["planner"] = dict(cfg.get("planner", {}))
    cfg_infer["planner"]["merge_lora"] = True

    if include_base_vlm:
        base_model, base_proc = build_model_and_processor(cfg_infer, for_training=False)

        def ctrl_zs_pol(img: Image.Image, task: str, _rng: random.Random) -> list[float]:
            act7, _ = predict_libero_policy_action(
                base_model, base_proc, img, task, format_example=True
            )
            return act7

        def ctrl_react(img: Image.Image, task: str, _rng: random.Random) -> list[float]:
            act7, _ = predict_libero_react_action(base_model, base_proc, img, task)
            return act7

        controllers["B2_zero_shot_policy"] = ctrl_zs_pol
        controllers["B5_react"] = ctrl_react
        print("Loaded frozen base Qwen3-VL-2B for B2_zero_shot_policy & B5_react", flush=True)

    if wm_ckpt and Path(wm_ckpt).exists():
        wm_model, wm_proc = load_finetuned(cfg_infer, wm_ckpt)
        planner = ImaginationPlanner(wm_model, wm_proc, cfg_infer)

        def ctrl_imagine(img: Image.Image, task: str, _rng: random.Random) -> list[float]:
            return list(planner.plan(img, task).action)

        controllers["B3_imagine_lora"] = ctrl_imagine
        print(f"Loaded B3_imagine_lora from {wm_ckpt}", flush=True)

    if policy_ckpt and Path(policy_ckpt).exists():
        pol_model, pol_proc = load_finetuned(cfg_infer, policy_ckpt)

        def ctrl_policy(img: Image.Image, task: str, _rng: random.Random) -> list[float]:
            act7, _ = predict_libero_policy_action(pol_model, pol_proc, img, task)
            return act7

        controllers["B4_direct_policy"] = ctrl_policy
        print(f"Loaded B4_direct_policy from {policy_ckpt}", flush=True)

    if action_cmd_ckpt and Path(action_cmd_ckpt).exists():
        ac_model, ac_proc = load_finetuned(cfg_infer, action_cmd_ckpt)

        def ctrl_ac(img: Image.Image, task: str, _rng: random.Random) -> list[float]:
            raw = generate_text(
                ac_model,
                ac_proc,
                img,
                build_action_prompt(task),
                max_new_tokens=48,
                do_sample=False,
            )
            return _command_to_vec7(parse_action_command(raw))

        controllers["B4b_action_cmd_vla"] = ctrl_ac
        print(f"Loaded B4b_action_cmd_vla from {action_cmd_ckpt}", flush=True)

    return controllers


def run_libero_sim(
    cfg: dict[str, Any],
    controllers: dict[str, Any],
    *,
    suite_name: str,
    max_tasks: int,
    episodes_per_task: int,
    max_steps: int,
    action_repeat: int,
    osc_scale: float,
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench_dict = benchmark.get_benchmark_dict()
    if suite_name not in bench_dict:
        raise ValueError(f"Unknown LIBERO suite {suite_name}; choices={list(bench_dict)}")
    suite = bench_dict[suite_name]()
    n_tasks = min(int(max_tasks), suite.n_tasks)

    results: dict[str, Any] = {
        "backend": "libero_mujoco",
        "suite": suite_name,
        "n_tasks": n_tasks,
        "episodes_per_task": episodes_per_task,
        "max_steps": max_steps,
        "action_repeat": action_repeat,
        "osc_scale": osc_scale,
        "tasks": {},
        "macro_average": {},
    }

    for task_id in range(n_tasks):
        task_obj = suite.get_task(task_id)
        task_desc = task_obj.language
        bddl_file = Path(get_libero_path("bddl_files")) / task_obj.problem_folder / task_obj.bddl_file
        init_states = suite.get_task_init_states(task_id)

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=224,
            camera_widths=224,
        )
        env.seed(seed + task_id)

        task_stats = {
            name: {"success_rate": 0.0, "mean_steps": 0.0}
            for name in controllers
        }
        try:
            for ep in range(episodes_per_task):
                init_state = init_states[ep % len(init_states)]
                for name, ctrl in controllers.items():
                    env.reset()
                    obs = env.set_init_state(init_state)
                    # Settle physics for 5 neutral steps (standard LIBERO eval protocol)
                    for _ in range(5):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

                    ep_rng = random.Random(seed + task_id * 1000 + ep)
                    cached_act: list[float] | None = None
                    succeeded = False
                    steps_taken = max_steps

                    for t in range(max_steps):
                        if cached_act is None or t % max(1, action_repeat) == 0:
                            img = _obs_to_pil(obs)
                            cached_act = ctrl(img, task_desc, ep_rng)
                        osc = _to_robosuite_osc(cached_act, osc_scale=osc_scale)
                        obs, _rew, done, _info = env.step(osc.tolist())
                        if done or env.check_success():
                            succeeded = True
                            steps_taken = t + 1
                            break

                    task_stats[name]["success_rate"] += float(succeeded) / episodes_per_task
                    task_stats[name]["mean_steps"] += float(steps_taken) / episodes_per_task
        finally:
            env.close()

        results["tasks"][f"{task_id:02d}_{task_obj.name}"] = {
            "instruction": task_desc,
            "controllers": task_stats,
        }
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[Task {task_id}] {task_desc}: {json.dumps(task_stats)}", flush=True)

    for name in controllers:
        results["macro_average"][name] = {
            "success_rate": sum(
                t["controllers"][name]["success_rate"] for t in results["tasks"].values()
            )
            / max(1, len(results["tasks"])),
            "mean_steps": sum(
                t["controllers"][name]["mean_steps"] for t in results["tasks"].values()
            )
            / max(1, len(results["tasks"])),
        }
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_synthetic_tabletop(
    controllers: dict[str, Any],
    *,
    n_tasks: int,
    episodes_per_task: int,
    max_steps: int,
    action_repeat: int,
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "backend": "synthetic_tabletop",
        "note": (
            "MuJoCo/LIBERO simulator not installed; ran synthetic 3D tabletop "
            "reach-grasp-lift closed-loop benchmark. Install libero on SuperPOD "
            "for official OffScreenRenderEnv rollouts."
        ),
        "n_tasks": n_tasks,
        "episodes_per_task": episodes_per_task,
        "max_steps": max_steps,
        "tasks": {},
        "macro_average": {},
    }
    for task_id in range(n_tasks):
        task_stats = {
            name: {"success_rate": 0.0, "mean_steps": 0.0}
            for name in controllers
        }
        task_desc = ""
        for ep in range(episodes_per_task):
            init_rng = random.Random(seed + task_id * 1000 + ep)
            proto = SyntheticTabletopEnv.random_init(init_rng, task_id)
            task_desc = proto.task

            for name, ctrl in controllers.items():
                env = SyntheticTabletopEnv(
                    task=proto.task,
                    eef_x=proto.eef_x,
                    eef_y=proto.eef_y,
                    eef_z=proto.eef_z,
                    obj_x=proto.obj_x,
                    obj_y=proto.obj_y,
                    obj_z=proto.obj_z,
                )
                ep_rng = random.Random(seed + task_id * 1000 + ep + 77)
                img = env.render()
                cached_act: list[float] | None = None
                succeeded = False
                steps_taken = max_steps

                for t in range(max_steps):
                    if cached_act is None or t % max(1, action_repeat) == 0:
                        cached_act = ctrl(img, env.task, ep_rng)
                    img, done = env.step(cached_act)
                    if done:
                        succeeded = True
                        steps_taken = t + 1
                        break

                task_stats[name]["success_rate"] += float(succeeded) / episodes_per_task
                task_stats[name]["mean_steps"] += float(steps_taken) / episodes_per_task

        results["tasks"][f"synthetic_{task_id:02d}"] = {
            "instruction": task_desc,
            "controllers": task_stats,
        }
    for name in controllers:
        results["macro_average"][name] = {
            "success_rate": sum(
                t["controllers"][name]["success_rate"] for t in results["tasks"].values()
            )
            / max(1, len(results["tasks"])),
            "mean_steps": sum(
                t["controllers"][name]["mean_steps"] for t in results["tasks"].values()
            )
            / max(1, len(results["tasks"])),
        }
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/superpod.yaml")
    p.add_argument("--checkpoint", default=None, help="Semantic world-model LoRA (B3)")
    p.add_argument("--policy_checkpoint", default=None, help="Equal-compute direct policy LoRA (B4)")
    p.add_argument("--action_cmd_checkpoint", default=None, help="Discrete VLA LoRA (B4b)")
    p.add_argument("--skip_base_vlm", action="store_true", help="Skip B2 zero-shot & B5 ReAct")
    p.add_argument(
        "--backend",
        choices=("auto", "libero", "synthetic"),
        default="auto",
    )
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--max_tasks", type=int, default=5)
    p.add_argument("--episodes_per_task", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=120)
    p.add_argument("--action_repeat", type=int, default=4)
    p.add_argument("--osc_scale", type=float, default=10.0)
    p.add_argument("--planner_mode", choices=("grid", "propose"), default="grid")
    p.add_argument("--score_mode", choices=("heuristic", "vlm_preference"), default="heuristic")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    cfg.setdefault("planner", {})
    cfg["planner"]["mode"] = args.planner_mode
    cfg["planner"]["score_mode"] = args.score_mode

    wm_ckpt = args.checkpoint or cfg.get("eval", {}).get("checkpoint")
    pol_ckpt = args.policy_checkpoint or cfg.get("eval", {}).get("policy_checkpoint")
    if pol_ckpt is None:
        cand_pol = Path("outputs/superpod_direct_policy/checkpoint-final")
        if cand_pol.exists():
            pol_ckpt = str(cand_pol)

    act_ckpt = args.action_cmd_checkpoint or cfg.get("eval", {}).get("action_cmd_checkpoint")
    if act_ckpt is None:
        cand_ac = Path("outputs/superpod_action_v3/checkpoint-final")
        if cand_ac.exists():
            act_ckpt = str(cand_ac)

    controllers = _build_controllers(
        cfg,
        wm_ckpt=wm_ckpt,
        policy_ckpt=pol_ckpt,
        action_cmd_ckpt=act_ckpt,
        include_base_vlm=not args.skip_base_vlm,
    )

    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    out_path = Path(args.out) if args.out else Path(out_dir) / f"closed_loop_{args.suite}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_libero = False
    if args.backend in ("auto", "libero"):
        try:
            import libero.libero  # noqa: F401

            use_libero = True
        except Exception as exc:
            if args.backend == "libero":
                raise SystemExit(
                    f"LIBERO simulator package not installed ({exc}). "
                    "Install with: pip install libero robosuite"
                ) from exc
            print(
                f"LIBERO MuJoCo not installed ({exc}); falling back to synthetic_tabletop.",
                flush=True,
            )

    if use_libero:
        res = run_libero_sim(
            cfg,
            controllers,
            suite_name=args.suite,
            max_tasks=args.max_tasks,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            action_repeat=args.action_repeat,
            osc_scale=args.osc_scale,
            seed=seed,
            out_path=out_path,
        )
    else:
        res = run_synthetic_tabletop(
            controllers,
            n_tasks=min(3, args.max_tasks),
            episodes_per_task=args.episodes_per_task,
            max_steps=min(40, args.max_steps),
            action_repeat=max(1, min(2, args.action_repeat)),
            seed=seed,
            out_path=out_path,
        )

    print(json.dumps(res["macro_average"], indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
