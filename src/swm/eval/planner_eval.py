"""Layer B evaluation: imagination planner vs random / heuristic / zero-shot / LoRA / direct policy.

Upgraded evaluation addressing three methodological gaps:
  1. Scene/state-aware counterfactual oracle (``state_counterfactual`` on LIBERO,
     ``microduck_rollout`` on MicroDuck) replaces the tautological action-only
     pseudo-oracle so ``B1_heuristic`` is no longer a 1.0 tautology.
  2. Equal-compute direct-policy baseline (``B4_direct_policy``) on the exact
     same Qwen3-VL-2B + LoRA backbone and training transitions.
  3. Optional existing action-command LoRA adapter (``B4b_action_cmd``) for
     immediate cross-paradigm comparison.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image
from tqdm import tqdm

from swm.data.action_commands import build_action_prompt, parse_action_command
from swm.data.dataset import SemanticTransitionDataset
from swm.data.labeling import (
    LabelConfig,
    build_outcome_string,
    label_transition,
    score_from_outcome,
)
from swm.eval.oracles import (
    action_only_legacy_score,
    microduck_rollout_score,
    state_counterfactual_score,
)
from swm.microduck.labeling import score_from_outcome as microduck_score_from_outcome
from swm.microduck.planner import candidate_actions as microduck_candidate_actions
from swm.models.direct_policy import (
    action_distance,
    predict_libero_policy_action,
    predict_libero_react_action,
    predict_microduck_policy_action,
    predict_microduck_react_action,
)
from swm.models.semantic_vlm import (
    build_model_and_processor,
    generate_outcome,
    generate_outcomes_batch,
    generate_text,
    load_finetuned,
)
from swm.planning.imagination_planner import candidate_actions
from swm.utils.config import load_config
from swm.utils.io import ensure_dir
from swm.utils.seed import set_seed


@dataclass
class ScoredAction:
    action: list[float]
    score: float
    outcome: str = ""


def _as7(action: Sequence[float]) -> list[float]:
    a = [float(x) for x in action] + [0.0] * 7
    return a[:7]


def domain_candidate_actions(cfg: dict[str, Any], domain: str = "libero") -> list[list[float]]:
    """Build 7D candidate action vectors for either LIBERO or MicroDuck configs."""
    planner = cfg.get("planner", {})
    if domain == "microduck" or "grid_xy" not in planner:
        raw3 = microduck_candidate_actions(cfg)
        out7: list[list[float]] = []
        for a3 in raw3:
            vx = float(a3[0]) if a3 else 0.0
            yaw = float(a3[2]) if len(a3) > 2 else 0.0
            is_kick = abs(yaw - 9.0) < 1e-6
            out7.append(
                [
                    0.0 if is_kick else vx,
                    0.0 if is_kick else yaw,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 if is_kick else 0.0,
                ]
            )
        return out7
    return candidate_actions(cfg)


def _action_key(action: Sequence[float], ndigits: int = 3) -> tuple[float, ...]:
    return tuple(round(float(x), ndigits) for x in _as7(action))


def _inject_logged(
    candidates: list[list[float]], logged: list[float]
) -> list[list[float]]:
    keys = {_action_key(a) for a in candidates}
    out = [list(a) for a in candidates]
    if _action_key(logged) not in keys:
        out.append(_as7(logged))
    return out


def oracle_outcome_from_action(action: Sequence[float], cfg: LabelConfig) -> str:
    labels = label_transition(
        _as7(action),
        None,
        None,
        frame_idx=0,
        episode_len=100,
        episode_success=False,
        cfg=cfg,
    )
    return build_outcome_string(labels)


def heuristic_score(action: Sequence[float], cfg: LabelConfig) -> float:
    return score_from_outcome(oracle_outcome_from_action(action, cfg))


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    n = len(xs)
    if n < 2:
        return 0.0

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx) ** 0.5
    deny = sum((b - my) ** 2 for b in ry) ** 0.5
    if denx <= 1e-12 or deny <= 1e-12:
        return 0.0
    return float(num / (denx * deny))


def pick_top(scored: list[ScoredAction], rng: random.Random | None = None) -> ScoredAction:
    if not scored:
        raise ValueError("empty scored list")
    best = max(s.score for s in scored)
    tied = [s for s in scored if abs(s.score - best) < 1e-9]
    if rng is None:
        return tied[0]
    return rng.choice(tied)


def score_random(
    actions: list[list[float]], rng: random.Random
) -> list[ScoredAction]:
    return [ScoredAction(action=a, score=rng.random()) for a in actions]


def score_heuristic(
    actions: list[list[float]],
    label_cfg: LabelConfig,
    *,
    disabled_fields: Sequence[str] | None = None,
) -> list[ScoredAction]:
    out = []
    for a in actions:
        outcome = oracle_outcome_from_action(a, label_cfg)
        out.append(
            ScoredAction(
                action=a,
                score=score_from_outcome(outcome, disabled_fields=disabled_fields),
                outcome=outcome,
            )
        )
    return out


def score_oracle_candidates(
    actions: list[list[float]],
    sample: dict[str, Any],
    label_cfg: LabelConfig,
    *,
    oracle_mode: str = "auto",
    disabled_fields: Sequence[str] | None = None,
) -> tuple[list[ScoredAction], str]:
    mode = oracle_mode.lower()
    if mode == "auto":
        mode = "microduck_rollout" if sample.get("scene_state") else "state_counterfactual"

    out: list[ScoredAction] = []
    for a in actions:
        if mode == "microduck_rollout":
            sc, outcome = microduck_rollout_score(a, sample)
        elif mode == "action_only_legacy":
            sc, outcome = action_only_legacy_score(
                a, label_cfg, disabled_fields=disabled_fields
            )
        else:
            sc, outcome = state_counterfactual_score(
                a, sample, label_cfg, disabled_fields=disabled_fields
            )
        out.append(ScoredAction(action=a, score=float(sc), outcome=outcome))
    return out, mode


def score_vlm(
    actions: list[list[float]],
    *,
    model,
    processor,
    image: Image.Image,
    task: str,
    domain: str = "libero",
    disabled_fields: Sequence[str] | None = None,
) -> list[ScoredAction]:
    max_tok = 48 if domain == "microduck" else 64
    outcomes = generate_outcomes_batch(
        model, processor, image, task, actions, max_new_tokens=max_tok
    )
    if domain == "microduck":
        return [
            ScoredAction(
                action=a,
                score=microduck_score_from_outcome(outcome, task=task),
                outcome=outcome,
            )
            for a, outcome in zip(actions, outcomes)
        ]
    return [
        ScoredAction(
            action=a,
            score=score_from_outcome(outcome, disabled_fields=disabled_fields),
            outcome=outcome,
        )
        for a, outcome in zip(actions, outcomes)
    ]


def score_react_policy(
    actions: list[list[float]],
    *,
    model,
    processor,
    image: Image.Image,
    task: str,
    domain: str = "libero",
) -> list[ScoredAction]:
    """ReAct baseline: generate Thought + Action, rank candidates by proximity."""
    if domain == "microduck":
        pred3, raw = predict_microduck_react_action(model, processor, image, task)
        is_kick = abs(pred3[2] - 9.0) < 1e-6
        pred7 = [
            0.0 if is_kick else pred3[0],
            0.0 if is_kick else pred3[2],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if is_kick else 0.0,
        ]
    else:
        pred7, raw = predict_libero_react_action(model, processor, image, task)

    return [
        ScoredAction(
            action=a,
            score=-action_distance(_as7(a), pred7),
            outcome=f"react:{raw}",
        )
        for a in actions
    ]


def score_direct_policy(
    actions: list[list[float]],
    *,
    model,
    processor,
    image: Image.Image,
    task: str,
    domain: str = "libero",
    format_example: bool = False,
) -> list[ScoredAction]:
    """Single forward decode of direct-policy VLM; rank candidates by proximity."""
    if domain == "microduck":
        pred3, raw = predict_microduck_policy_action(
            model, processor, image, task, format_example=format_example
        )
        is_kick = abs(pred3[2] - 9.0) < 1e-6
        pred7 = [
            0.0 if is_kick else pred3[0],
            0.0 if is_kick else pred3[2],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if is_kick else 0.0,
        ]
    else:
        pred7, raw = predict_libero_policy_action(
            model, processor, image, task, format_example=format_example
        )

    return [
        ScoredAction(
            action=a,
            score=-action_distance(_as7(a), pred7),
            outcome=f"direct_policy:{raw}",
        )
        for a in actions
    ]


def _command_to_vec7(cmd: dict[str, Any] | None) -> list[float]:
    if cmd is None:
        return [0.0] * 7
    mag_map = {"none": 0.0, "small": 0.015, "medium": 0.04, "large": 0.06}
    step = mag_map.get(str(cmd.get("magnitude", "medium")).lower(), 0.04)
    act = str(cmd.get("action", "HOLD")).upper()
    dx = dy = dz = 0.0
    if act == "MOVE_X_POSITIVE":
        dx = step
    elif act == "MOVE_X_NEGATIVE":
        dx = -step
    elif act == "MOVE_Y_POSITIVE":
        dy = step
    elif act == "MOVE_Y_NEGATIVE":
        dy = -step
    elif act == "MOVE_Z_POSITIVE":
        dz = max(0.02, step)
    elif act == "MOVE_Z_NEGATIVE":
        dz = -max(0.02, step)
    grip_str = str(cmd.get("gripper", "hold")).lower()
    if act == "GRASP" or grip_str == "close":
        grip = 1.0
    elif act == "RELEASE" or grip_str == "open":
        grip = 0.0
    else:
        grip = 0.5
    return [dx, dy, dz, 0.0, 0.0, 0.0, grip]


def score_action_command_lora(
    actions: list[list[float]],
    *,
    model,
    processor,
    image: Image.Image,
    task: str,
) -> list[ScoredAction]:
    """Rank candidates by proximity to decoded discrete action-command LoRA."""
    raw = generate_text(
        model,
        processor,
        image,
        build_action_prompt(task),
        max_new_tokens=48,
        do_sample=False,
    )
    pred7 = _command_to_vec7(parse_action_command(raw))
    return [
        ScoredAction(
            action=a,
            score=-action_distance(_as7(a), pred7),
            outcome=f"action_cmd:{raw}",
        )
        for a in actions
    ]


def evaluate_split(
    ds: SemanticTransitionDataset,
    cfg: dict[str, Any],
    *,
    scorers: dict[str, Callable[[list[list[float]], Image.Image, str], list[ScoredAction]]],
    max_samples: int,
    horizon: int,
    seed: int,
    domain: str = "libero",
    oracle_mode: str = "auto",
    disabled_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    label_cfg = LabelConfig(
        move_threshold=float(cfg["data"].get("move_threshold", 0.01)),
        lift_threshold=float(cfg["data"].get("lift_threshold", 0.01)),
        success_near_frames=int(cfg["data"].get("success_near_frames", 15)),
    )
    base_candidates = domain_candidate_actions(cfg, domain)
    n = min(len(ds), max_samples)

    stats = {
        name: {
            "top1_vs_oracle": 0.0,
            "regret": 0.0,
            "spearman": 0.0,
            "logged_top1": 0.0,
            "logged_top3": 0.0,
            "mean_plan_score": 0.0,
            "mean_l2_to_logged": 0.0,
            "n": 0,
        }
        for name in scorers
    }
    spearman_sums = {name: 0.0 for name in scorers}
    resolved_oracle_modes: set[str] = set()

    for i in tqdm(range(n), desc=f"planner_h{horizon}"):
        sample = ds[i]
        logged = _as7(sample.get("action") or [0.0] * 7)
        cand_pool = base_candidates
        if domain == "microduck":
            t_low = str(sample.get("task", "")).lower()
            if "kick" not in t_low and "goal" not in t_low:
                cand_pool = [a for a in base_candidates if float(a[6]) < 0.5]
        actions = _inject_logged(cand_pool, logged)
        oracle_scored, used_mode = score_oracle_candidates(
            actions,
            sample,
            label_cfg,
            oracle_mode=oracle_mode,
            disabled_fields=disabled_fields,
        )
        resolved_oracle_modes.add(used_mode)
        oracle_best = pick_top(oracle_scored)
        oracle_by_key = {_action_key(s.action): s for s in oracle_scored}
        logged_key = _action_key(logged)
        step_rng = random.Random(seed + i)

        for name, scorer in scorers.items():
            if name == "B0_random":
                scored = score_random(actions, step_rng)
            else:
                scored = scorer(actions, sample["image"], sample["task"])
            chosen_first = pick_top(scored, step_rng if name == "B0_random" else None)
            plan_score = chosen_first.score
            if horizon >= 2:
                if name == "B0_random":
                    second_scored = score_random(actions, step_rng)
                else:
                    second_scored = scorer(
                        actions, sample["image"], sample["task"]
                    )
                second = pick_top(second_scored, step_rng if name == "B0_random" else None)
                plan_score = chosen_first.score + second.score

            chosen_key = _action_key(chosen_first.action)
            oracle_chosen = oracle_by_key[chosen_key]
            top1 = float(chosen_key == _action_key(oracle_best.action))
            regret = float(oracle_best.score - oracle_chosen.score)

            ranked = sorted(scored, key=lambda s: s.score, reverse=True)
            keys = [_action_key(s.action) for s in ranked]
            logged_top1 = float(keys[0] == logged_key) if keys else 0.0
            logged_top3 = float(logged_key in keys[:3]) if keys else 0.0
            l2_err = action_distance(_as7(chosen_first.action), logged)

            model_scores = [s.score for s in scored]
            oracle_scores = [
                oracle_by_key[_action_key(s.action)].score for s in scored
            ]
            spearman_sums[name] += spearman(model_scores, oracle_scores)

            st = stats[name]
            st["top1_vs_oracle"] += top1
            st["regret"] += regret
            st["logged_top1"] += logged_top1
            st["logged_top3"] += logged_top3
            st["mean_plan_score"] += plan_score
            st["mean_l2_to_logged"] += l2_err
            st["n"] += 1

    for name, st in stats.items():
        m = max(1, int(st["n"]))
        st["top1_vs_oracle"] /= m
        st["regret"] /= m
        st["logged_top1"] /= m
        st["logged_top3"] /= m
        st["mean_plan_score"] /= m
        st["mean_l2_to_logged"] /= m
        st["spearman"] = spearman_sums[name] / m
    return {
        "horizon": horizon,
        "oracle_modes_used": sorted(resolved_oracle_modes),
        "baselines": stats,
        "n_frames": n,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--policy_checkpoint",
        type=str,
        default=None,
        help="Equal-compute direct-policy LoRA checkpoint for B4_direct_policy.",
    )
    p.add_argument(
        "--action_cmd_checkpoint",
        type=str,
        default=None,
        help="Optional discrete action-command LoRA checkpoint for B4b_action_cmd.",
    )
    p.add_argument(
        "--domain",
        type=str,
        choices=("libero", "microduck"),
        default="libero",
    )
    p.add_argument(
        "--oracle_mode",
        type=str,
        choices=("auto", "state_counterfactual", "microduck_rollout", "action_only_legacy"),
        default="auto",
        help="Scene/state-grounded oracle (default: auto) vs legacy action-only.",
    )
    p.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument(
        "--horizons",
        type=str,
        default="1,2",
        help="Comma-separated horizons for ablation A5 (e.g. 1 or 1,2).",
    )
    p.add_argument(
        "--skip_zero_shot",
        action="store_true",
        help="Skip B2 (faster; still runs B0/B1/B3[/B4]).",
    )
    p.add_argument(
        "--disable_score_fields",
        type=str,
        default="",
        help=(
            "Planner score ablation: comma-separated fields to drop from "
            "score_from_outcome (e.g. progress or success_near)."
        ),
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for planner_metrics filename.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    seed = int(cfg.get("seed", 42))
    disabled_fields = [
        x.strip() for x in args.disable_score_fields.split(",") if x.strip()
    ]

    processed = Path(cfg["paths"]["processed_dir"])
    frames = Path(cfg["paths"]["frames_dir"])
    split_path = processed / f"{args.split}.jsonl"
    if not split_path.is_file():
        if args.split == "test" and (processed / "val.jsonl").is_file():
            print("test.jsonl missing; falling back to val.jsonl")
            split_path = processed / "val.jsonl"
        else:
            raise SystemExit(f"Split not found: {split_path}")

    ds = SemanticTransitionDataset(split_path, frames)
    max_samples = int(
        args.max_samples
        or cfg["eval"].get("planner_max_samples")
        or min(64, int(cfg["eval"].get("max_samples", 512)))
    )
    max_samples = max(1, max_samples)

    label_cfg = LabelConfig(
        move_threshold=float(cfg["data"].get("move_threshold", 0.01)),
        lift_threshold=float(cfg["data"].get("lift_threshold", 0.01)),
        success_near_frames=int(cfg["data"].get("success_near_frames", 15)),
    )
    scorers: dict[str, Callable[[list[list[float]], Image.Image, str], list[ScoredAction]]] = {
        "B0_random": lambda acts, _img, _task: score_random(acts, random.Random(0)),
        "B1_heuristic": lambda acts, _img, _task: score_heuristic(
            acts, label_cfg, disabled_fields=disabled_fields
        ),
    }

    if not args.skip_zero_shot:
        print("Loading base frozen Qwen3-VL-2B for B2_zero_shot & B5_react...")
        zs_model, zs_proc = build_model_and_processor(cfg, for_training=False)

        def b2_wm(acts, img, task, _m=zs_model, _p=zs_proc, _dom=args.domain):
            return score_vlm(
                acts,
                model=_m,
                processor=_p,
                image=img,
                task=task,
                domain=_dom,
                disabled_fields=disabled_fields,
            )

        def b2_pol(acts, img, task, _m=zs_model, _p=zs_proc, _dom=args.domain):
            return score_direct_policy(
                acts,
                model=_m,
                processor=_p,
                image=img,
                task=task,
                domain=_dom,
                format_example=True,
            )

        def b5_react(acts, img, task, _m=zs_model, _p=zs_proc, _dom=args.domain):
            return score_react_policy(
                acts,
                model=_m,
                processor=_p,
                image=img,
                task=task,
                domain=_dom,
            )

        scorers["B2_zero_shot_imagine"] = b2_wm
        scorers["B2b_zero_shot_policy"] = b2_pol
        scorers["B5_react"] = b5_react

    ckpt = args.checkpoint or cfg["eval"]["checkpoint"]
    if not Path(ckpt).exists():
        raise SystemExit(f"Fine-tuned checkpoint required for B3: {ckpt}")
    cfg.setdefault("planner", {})
    cfg["planner"]["merge_lora"] = True
    print(f"Loading fine-tuned world model for B3: {ckpt}")
    ft_model, ft_proc = load_finetuned(cfg, ckpt)

    def b3(acts, img, task, _m=ft_model, _p=ft_proc, _dom=args.domain):
        return score_vlm(
            acts,
            model=_m,
            processor=_p,
            image=img,
            task=task,
            domain=_dom,
            disabled_fields=disabled_fields,
        )

    scorers["B3_lora"] = b3

    policy_ckpt = args.policy_checkpoint or cfg.get("eval", {}).get("policy_checkpoint")
    if policy_ckpt and Path(policy_ckpt).exists():
        print(f"Loading equal-compute direct-policy LoRA for B4: {policy_ckpt}")
        pol_model, pol_proc = load_finetuned(cfg, policy_ckpt)

        def b4(acts, img, task, _m=pol_model, _p=pol_proc, _dom=args.domain):
            return score_direct_policy(
                acts,
                model=_m,
                processor=_p,
                image=img,
                task=task,
                domain=_dom,
            )

        scorers["B4_direct_policy"] = b4
    elif policy_ckpt:
        print(f"Note: policy_checkpoint {policy_ckpt} not found yet; skipping B4_direct_policy.")

    act_cmd_ckpt = args.action_cmd_checkpoint or cfg.get("eval", {}).get(
        "action_cmd_checkpoint"
    )
    if act_cmd_ckpt and Path(act_cmd_ckpt).exists():
        print(f"Loading action-command LoRA for B4b: {act_cmd_ckpt}")
        ac_model, ac_proc = load_finetuned(cfg, act_cmd_ckpt)

        def b4b(acts, img, task, _m=ac_model, _p=ac_proc):
            return score_action_command_lora(
                acts,
                model=_m,
                processor=_p,
                image=img,
                task=task,
            )

        scorers["B4b_action_cmd"] = b4b

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    results: dict[str, Any] = {
        "split": args.split,
        "seed": seed,
        "oracle_mode": args.oracle_mode,
        "domain": args.domain,
        "max_samples": max_samples,
        "k_candidates": len(domain_candidate_actions(cfg, args.domain)),
        "disabled_score_fields": disabled_fields,
        "note": (
            "Oracle uses scene/state-grounded counterfactual scoring by default "
            "(eliminating B1 action-only tautology). B4_direct_policy compares "
            "an equal-compute direct VLM policy against B3_lora imagination."
        ),
        "horizons": {},
    }
    for h in horizons:
        print(f"=== Horizon {h} (oracle={args.oracle_mode}) ===")
        results["horizons"][str(h)] = evaluate_split(
            ds,
            cfg,
            scorers=scorers,
            max_samples=max_samples,
            horizon=h,
            seed=seed,
            domain=args.domain,
            oracle_mode=args.oracle_mode,
            disabled_fields=disabled_fields,
        )

    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    tag = args.tag.strip()
    if not tag and disabled_fields:
        tag = "score_" + "_".join(disabled_fields)
    suffix = f"_{tag}" if tag else ""
    out_path = Path(out_dir) / f"planner_metrics_{args.split}{suffix}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
