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
    # Cap for interactive demos
    return actions[: int(planner.get("max_candidates", 16))]


@dataclass
class RankedAction:
    action: list[float]
    outcome: str
    score: float


class ImaginationPlanner:
    def __init__(self, model, processor, cfg: dict[str, Any]):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.candidates = candidate_actions(cfg)

    def rank(
        self, image: Image.Image, task: str, actions: list[list[float]] | None = None
    ) -> list[RankedAction]:
        acts = actions or self.candidates
        ranked: list[RankedAction] = []
        for a in acts:
            outcome = generate_outcome(self.model, self.processor, image, task, a)
            ranked.append(
                RankedAction(action=a, outcome=outcome, score=score_from_outcome(outcome))
            )
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    def plan(self, image: Image.Image, task: str) -> RankedAction:
        return self.rank(image, task)[0]

    def heatmap_xy(
        self, image: Image.Image, task: str, gripper: float = 1.0, dz: float = 0.0
    ) -> tuple[list[float], list[float], list[list[float]]]:
        """Return xs, ys, score_grid for Streamlit/Plotly visualization."""
        xs = list(self.cfg["planner"]["grid_xy"])
        ys = list(self.cfg["planner"]["grid_xy"])
        grid: list[list[float]] = []
        for dy in ys:
            row: list[float] = []
            for dx in xs:
                action = [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0, float(gripper)]
                outcome = generate_outcome(
                    self.model, self.processor, image, task, action
                )
                row.append(score_from_outcome(outcome))
            grid.append(row)
        return xs, ys, grid
