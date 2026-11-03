"""Imagine-then-Act: counterfactual outcome imagination, then execute the best action.

**Imagine** (this module's core): for the *same* observation and goal, ask the
semantic world model what would happen under many candidate actions that were
*not* necessarily executed in the dataset. That is one-step (or short-horizon)
model-based planning — not chain-of-thought "reasoning" like ReAct.

**Act**: return / publish the highest-scoring candidate.

Pipeline:
    observe (o, g)  →  imagine ŷ(a) for each a in A  →  act = argmax score(ŷ)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from swm.data.dataset import SemanticTransitionDataset
from swm.models.semantic_vlm import load_finetuned
from swm.planning.imagination_planner import ImaginationPlanner, RankedAction
from swm.utils.config import load_config
from swm.utils.seed import set_seed


@dataclass
class ImaginationStep:
    """One counterfactual query: action → imagined semantic outcome."""

    action: list[float]
    imagined_outcome: str
    score: float


@dataclass
class ImagineThenActResult:
    task: str
    n_imagined: int
    imaginations: list[ImaginationStep]
    chosen: ImaginationStep
    horizon: int


class ImagineThenAct:
    """Full Imagine-then-Act controller around the semantic LoRA world model."""

    def __init__(self, model, processor, cfg: dict[str, Any]):
        self.cfg = cfg
        self.planner = ImaginationPlanner(model, processor, cfg)
        self.max_horizon = int(cfg.get("planner", {}).get("max_horizon", 1))

    def imagine(
        self,
        image: Image.Image,
        task: str,
        *,
        actions: list[list[float]] | None = None,
    ) -> list[ImaginationStep]:
        """Counterfactual rollouts: predict an outcome for every candidate action."""
        ranked = self.planner.rank(image, task, actions=actions)
        return [
            ImaginationStep(
                action=list(r.action),
                imagined_outcome=r.outcome,
                score=float(r.score),
            )
            for r in ranked
        ]

    def act(self, imaginations: list[ImaginationStep]) -> ImaginationStep:
        """Select the action with the best imagined outcome score."""
        if not imaginations:
            raise ValueError("No imaginations to act on")
        return max(imaginations, key=lambda s: s.score)

    def imagine_then_act(
        self,
        image: Image.Image,
        task: str,
        *,
        horizon: int | None = None,
        actions: list[list[float]] | None = None,
    ) -> ImagineThenActResult:
        """Run the full loop. Horizon>1 greedily re-imagines on the same frame.

        True multi-step would require the next observation after acting; for the
        course demo we document same-frame greedy imagination as a limited A5.
        """
        h = int(horizon if horizon is not None else self.max_horizon)
        h = max(1, h)

        imaginations = self.imagine(image, task, actions=actions)
        chosen = self.act(imaginations)
        # Optional short greedy horizon: keep re-selecting under the same o_t.
        for _ in range(h - 1):
            again = self.imagine(image, task, actions=actions)
            nxt = self.act(again)
            chosen = ImaginationStep(
                action=chosen.action,
                imagined_outcome=(
                    f"{chosen.imagined_outcome} || next:{nxt.imagined_outcome}"
                ),
                score=chosen.score + nxt.score,
            )

        return ImagineThenActResult(
            task=task,
            n_imagined=len(imaginations),
            imaginations=imaginations,
            chosen=chosen,
            horizon=h,
        )


def _result_to_dict(result: ImagineThenActResult) -> dict[str, Any]:
    return {
        "task": result.task,
        "n_imagined": result.n_imagined,
        "horizon": result.horizon,
        "chosen": asdict(result.chosen),
        "imaginations": [asdict(s) for s in result.imaginations],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Imagine-then-Act CLI: for one observation, imagine outcomes under "
            "K candidate actions, then print the selected action."
        )
    )
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--index", type=int, default=0, help="Sample index in the split")
    p.add_argument("--image", type=str, default=None, help="Optional image path override")
    p.add_argument("--task", type=str, default=None, help="Optional task override")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--top", type=int, default=5, help="How many imaginations to print")
    p.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    ckpt = args.checkpoint or cfg["eval"]["checkpoint"]
    if not Path(ckpt).is_file() and not Path(ckpt).is_dir():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    print(f"Loading semantic world model from {ckpt}")
    model, processor = load_finetuned(cfg, ckpt)
    controller = ImagineThenAct(model, processor, cfg)

    if args.image:
        image = Image.open(args.image).convert("RGB")
        task = args.task or "complete the manipulation task"
    else:
        processed = Path(cfg["paths"]["processed_dir"])
        frames = Path(cfg["paths"]["frames_dir"])
        split = processed / f"{args.split}.jsonl"
        if not split.is_file():
            raise SystemExit(f"Split not found: {split}")
        ds = SemanticTransitionDataset(split, frames)
        if not (0 <= args.index < len(ds)):
            raise SystemExit(f"index out of range 0..{len(ds)-1}")
        sample = ds[args.index]
        image = sample["image"]
        task = args.task or sample["task"]
        print(f"Sample #{args.index}  episode={sample.get('episode_id')}")

    print("\n=== OBSERVE ===")
    print(f"Task: {task}")

    print("\n=== IMAGINE ===")
    print(f"Querying world model for {len(controller.planner.candidates)} candidate actions...")
    result = controller.imagine_then_act(image, task, horizon=args.horizon)

    for i, step in enumerate(result.imaginations[: max(1, args.top)]):
        a = step.action
        print(
            f"  [{i+1}] score={step.score:.2f}  "
            f"dx={a[0]:+.3f} dy={a[1]:+.3f} dz={a[2]:+.3f} "
            f"grip={'close' if a[6] >= 0.5 else 'open'}"
        )
        print(f"       imagined: {step.imagined_outcome}")

    print("\n=== ACT ===")
    c = result.chosen
    print(
        f"Selected action: dx={c.action[0]:+.3f}, dy={c.action[1]:+.3f}, "
        f"dz={c.action[2]:+.3f}, gripper={'close' if c.action[6] >= 0.5 else 'open'}"
    )
    print(f"Imagined outcome: {c.imagined_outcome}")
    print(f"Score: {c.score:.2f}  (horizon={result.horizon})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_result_to_dict(result), indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
