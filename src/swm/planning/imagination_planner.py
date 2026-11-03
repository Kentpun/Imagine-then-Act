"""Counterfactual action scoring with a semantic world model.

For a fixed observation and goal, the planner asks: for each candidate action
a^(k), what semantic outcome ŷ would occur? That query *is* the imagination
step. Ranking by score(ŷ) yields the action to execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from swm.data.labeling import score_from_outcome
from swm.models.semantic_vlm import generate_outcome


def candidate_actions(cfg: dict[str, Any]) -> list[list[float]]:
    planner = cfg["planner"]
    actions: list[list[float]] = []
    for dx in planner["grid_xy"]:
        for dy in planner["grid_xy"]:
            for dz in planner["dz_values"]:
                for g in planner["gripper_values"]:
                    # 7D: dx dy dz droll dpitch dyaw gripper
                    actions.append(
                        [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0, float(g)]
                    )
    return actions[: int(planner.get("max_candidates", 16))]


@dataclass
class RankedAction:
    action: list[float]
    outcome: str
    score: float


class ImaginationPlanner:
    """Imagine outcomes under many actions, then prefer the best score."""

    def __init__(self, model, processor, cfg: dict[str, Any]):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.candidates = candidate_actions(cfg)

    def imagine_one(
        self, image: Image.Image, task: str, action: list[float]
    ) -> RankedAction:
        """Imagine the semantic outcome of a single counterfactual action."""
        outcome = generate_outcome(self.model, self.processor, image, task, action)
        return RankedAction(
            action=list(action),
            outcome=outcome,
            score=score_from_outcome(outcome),
        )

    def rank(
        self, image: Image.Image, task: str, actions: list[list[float]] | None = None
    ) -> list[RankedAction]:
        """Imagine outcomes for every candidate; return sorted by score (desc)."""
        acts = actions or self.candidates
        ranked = [self.imagine_one(image, task, a) for a in acts]
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    def plan(self, image: Image.Image, task: str) -> RankedAction:
        """Imagine-then-act for one step: return the best imagined action."""
        return self.rank(image, task)[0]

    def heatmap_xy(
        self, image: Image.Image, task: str, gripper: float = 1.0, dz: float = 0.0
    ) -> tuple[list[float], list[float], list[list[float]]]:
        """Return xs, ys, score_grid of imagined outcomes over the xy grid."""
        xs = list(self.cfg["planner"]["grid_xy"])
        ys = list(self.cfg["planner"]["grid_xy"])
        grid: list[list[float]] = []
        for dy in ys:
            row: list[float] = []
            for dx in xs:
                action = [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0, float(gripper)]
                row.append(self.imagine_one(image, task, action).score)
            grid.append(row)
        return xs, ys, grid
