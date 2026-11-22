"""Counterfactual action scoring with a semantic world model.

For a fixed observation and goal, the planner asks: for each candidate action
a^(k), what semantic outcome ŷ would occur? That query *is* the imagination
step. Ranking by score(ŷ) yields the action to execute.

Candidate generation / scoring modes (config ``planner.mode`` / ``score_mode``):

- ``mode=grid``: fixed Cartesian/gripper grid (legacy ablation baseline)
- ``mode=propose``: VLM proposes task-conditioned actions (fallback to grid)
- ``score_mode=heuristic``: hand-mapped outcome fields → scalar
- ``score_mode=vlm_preference``: VLM ranks imagined outcomes for the goal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from swm.data.labeling import score_from_outcome
from swm.models.semantic_vlm import generate_outcome, generate_outcomes_batch
from swm.planning.vlm_planner import (
    format_libero_action,
    pad_proposals_with_grid,
    preference_scores,
    propose_libero_actions,
)


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
        planner = cfg.get("planner", {})
        self.mode = str(planner.get("mode", "grid")).lower()
        self.score_mode = str(planner.get("score_mode", "heuristic")).lower()
        self.n_proposals = int(planner.get("n_proposals", planner.get("max_candidates", 8)))
        self._grid_candidates = candidate_actions(cfg)
        # For propose mode this is filled on each rank(); demos may read it.
        self.candidates = list(self._grid_candidates)

    def _resolve_actions(
        self,
        image: Image.Image,
        task: str,
        actions: list[list[float]] | None,
    ) -> list[list[float]]:
        if actions is not None:
            self.candidates = list(actions)
            return self.candidates

        if self.mode == "propose":
            max_n = int(self.cfg.get("planner", {}).get("max_candidates", 16))
            proposed = propose_libero_actions(
                self.model,
                self.processor,
                image,
                task,
                n=self.n_proposals,
                max_new_tokens=int(
                    self.cfg.get("planner", {}).get("propose_max_new_tokens", 160)
                ),
                verbose=bool(self.cfg.get("planner", {}).get("verbose_propose", False)),
            )
            merged = pad_proposals_with_grid(
                proposed, self._grid_candidates, max_n=max_n
            )
            self.candidates = merged if merged else list(self._grid_candidates)
            return self.candidates
        self.candidates = list(self._grid_candidates)
        return self.candidates

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

    def _score_batch(
        self,
        image: Image.Image,
        task: str,
        actions: list[list[float]],
        outcomes: list[str],
    ) -> list[float]:
        if self.score_mode == "vlm_preference":
            return preference_scores(
                self.model,
                self.processor,
                image,
                task,
                actions,
                outcomes,
                action_formatter=format_libero_action,
                max_new_tokens=int(
                    self.cfg.get("planner", {}).get("prefer_max_new_tokens", 64)
                ),
            )
        return [score_from_outcome(o) for o in outcomes]

    def rank(
        self, image: Image.Image, task: str, actions: list[list[float]] | None = None
    ) -> list[RankedAction]:
        """Imagine outcomes for every candidate; return sorted by score (desc)."""
        acts = self._resolve_actions(image, task, actions)
        bs = int(self.cfg.get("planner", {}).get("batch_size", 8))
        outcomes = generate_outcomes_batch(
            self.model, self.processor, image, task, acts, batch_size=bs
        )
        scores = self._score_batch(image, task, acts, outcomes)
        ranked = [
            RankedAction(action=list(a), outcome=o, score=float(s))
            for a, o, s in zip(acts, outcomes, scores)
        ]
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
