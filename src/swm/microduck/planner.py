from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from swm.microduck.labeling import mock_outcome, score_from_outcome


def candidate_actions(cfg: dict[str, Any]) -> list[list[float]]:
    """Build velocity candidates legal for the EMQX MicroDuck walking demo.

    Skips pairs where yaw != 0 and vx <= 0 (policy cannot turn in place).
    Forces vy = 0.
    """
    planner = cfg["planner"]
    actions: list[list[float]] = []
    for vx in planner["vx_values"]:
        vx_f = float(vx)
        if vx_f < 0.0:
            continue  # demo walking policy: no reverse
        for yaw in planner["yaw_values"]:
            yaw_f = float(yaw)
            if abs(yaw_f) > 1e-6 and vx_f <= 1e-6:
                continue
            actions.append([vx_f, 0.0, yaw_f])
    return actions[: int(planner.get("max_candidates", 16))]


@dataclass
class RankedAction:
    action: list[float]
    outcome: str
    score: float


class MicroDuckImaginationPlanner:
    """Rank locomotion commands by imagined semantic outcomes."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        model=None,
        processor=None,
        mock: bool = False,
    ):
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.mock = mock or model is None
        self.candidates = candidate_actions(cfg)

    def _predict(self, image: Image.Image, task: str, action: list[float]) -> str:
        if self.mock:
            return mock_outcome(action, task)
        from swm.data.prompts import build_prompt as libero_prompt
        from swm.models.semantic_vlm import generate_outcome

        # The SuperPOD LoRA was trained on delta_xyz prompts. Map vx→dx, yaw→dy.
        vx = float(action[0]) if action else 0.0
        yaw = float(action[2]) if len(action) > 2 else 0.0
        libero_action = [vx, yaw, 0.0, 0.0, 0.0, 0.0, 0.0]
        prompt = libero_prompt(
            f"{task.strip()} (locomotion vx={vx:.2f}, yaw={yaw:.2f})",
            libero_action,
        )
        return generate_outcome(
            self.model,
            self.processor,
            image,
            task,
            libero_action,
            prompt=prompt,
        )

    def rank(
        self, image: Image.Image, task: str, actions: list[list[float]] | None = None
    ) -> list[RankedAction]:
        acts = actions or self.candidates
        ranked: list[RankedAction] = []
        for a in acts:
            outcome = self._predict(image, task, a)
            ranked.append(
                RankedAction(
                    action=a, outcome=outcome, score=score_from_outcome(outcome)
                )
            )
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    def plan(self, image: Image.Image, task: str) -> RankedAction:
        return self.rank(image, task)[0]

    def heatmap_vx_yaw(
        self, image: Image.Image, task: str, vy: float = 0.0
    ) -> tuple[list[float], list[float], list[list[float]]]:
        xs = [float(v) for v in self.cfg["planner"]["vx_values"] if float(v) >= 0.0]
        ys = [float(v) for v in self.cfg["planner"]["yaw_values"]]
        grid: list[list[float]] = []
        for yaw in ys:
            row: list[float] = []
            for vx in xs:
                if abs(yaw) > 1e-6 and vx <= 1e-6:
                    row.append(float("nan"))
                    continue
                action = [float(vx), 0.0, float(yaw)]
                outcome = self._predict(image, task, action)
                row.append(score_from_outcome(outcome))
            grid.append(row)
        return xs, ys, grid
