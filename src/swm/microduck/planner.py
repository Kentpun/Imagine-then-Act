from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from swm.microduck.labeling import mock_outcome, score_from_outcome
from swm.planning.vlm_planner import (
    format_microduck_action,
    pad_proposals_with_grid,
    preference_scores,
    propose_microduck_actions,
)


def candidate_actions(cfg: dict[str, Any]) -> list[list[float]]:
    """Build velocity candidates legal for the EMQX MicroDuck walking demo.

    Skips pairs where yaw != 0 and vx <= 0 (policy cannot turn in place).
    Forces vy = 0.
    Optional kick sentinel: [0,0,0] with gripper=1 encoded as action[6]=1 via
    libero mapping when include_kick is true (planner stores marker yaw=9.0).
    """
    planner = cfg["planner"]
    actions: list[list[float]] = []
    vx_sorted = sorted(
        (float(v) for v in planner["vx_values"] if float(v) >= 0.0),
        reverse=True,
    )
    yaw_sorted = sorted(
        (float(y) for y in planner["yaw_values"]),
        key=lambda y: (abs(y), -y),
    )
    for vx_f in vx_sorted:
        for yaw_f in yaw_sorted:
            if abs(yaw_f) > 1e-6 and vx_f <= 1e-6:
                continue
            actions.append([vx_f, 0.0, yaw_f])
    if planner.get("include_kick"):
        # Sentinel: yaw=9.0 means "kick" skill for the multitask demo router.
        actions.append([0.0, 0.0, 9.0])
    return actions[: int(planner.get("max_candidates", 24))]


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
        planner = cfg.get("planner", {})
        self.mode = str(planner.get("mode", "grid")).lower()
        self.score_mode = str(planner.get("score_mode", "heuristic")).lower()
        self.n_proposals = int(planner.get("n_proposals", planner.get("max_candidates", 4)))
        self._grid_candidates = candidate_actions(cfg)
        self.candidates = list(self._grid_candidates)

    def _to_libero_action_and_prompt(
        self, task: str, action: list[float]
    ) -> tuple[list[float], str]:
        from swm.data.prompts import build_prompt as libero_prompt

        vx = float(action[0]) if action else 0.0
        yaw = float(action[2]) if len(action) > 2 else 0.0
        is_kick = abs(yaw - 9.0) < 1e-6
        libero_action = [
            0.0 if is_kick else vx,
            0.0 if is_kick else yaw,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if is_kick else 0.0,
        ]
        # Match prepare_microduck_semantic.py prompt distribution by default;
        # optional note is available when transferring a LIBERO-only checkpoint.
        if self.cfg.get("planner", {}).get("append_locomotion_note", False):
            note = "skill=kick" if is_kick else f"locomotion vx={vx:.2f}, yaw={yaw:.2f}"
            prompt = libero_prompt(f"{task.strip()} ({note})", libero_action)
        else:
            prompt = libero_prompt(task.strip(), libero_action)
        return libero_action, prompt

    def _predict(self, image: Image.Image, task: str, action: list[float]) -> str:
        if self.mock:
            return mock_outcome(action, task)
        from swm.models.semantic_vlm import generate_outcome

        libero_action, prompt = self._to_libero_action_and_prompt(task, action)
        max_new = int(self.cfg.get("planner", {}).get("max_new_tokens", 48))
        return generate_outcome(
            self.model,
            self.processor,
            image,
            task,
            libero_action,
            prompt=prompt,
            max_new_tokens=max_new,
        )

    def _predict_batch(
        self, image: Image.Image, task: str, actions: list[list[float]]
    ) -> tuple[list[str], list[float]]:
        if self.mock:
            return [mock_outcome(a, task) for a in actions], [0.0] * len(actions)
        from swm.models.semantic_vlm import (
            generate_microduck_outcomes_fast,
            generate_outcomes_batch,
        )

        pairs = [self._to_libero_action_and_prompt(task, a) for a in actions]
        libero_actions = [p[0] for p in pairs]
        prompts = [p[1] for p in pairs]
        plan_cfg = self.cfg.get("planner", {})
        max_new = int(plan_cfg.get("max_new_tokens", 40))
        bs = int(plan_cfg.get("batch_size", 8))
        if bool(plan_cfg.get("fast_structured_decode", True)):
            return generate_microduck_outcomes_fast(
                self.model,
                self.processor,
                image,
                task,
                libero_actions,
                prompts=prompts,
                max_new_tokens=max_new,
                batch_size=bs,
            )
        res = generate_outcomes_batch(
            self.model,
            self.processor,
            image,
            task,
            libero_actions,
            prompts=prompts,
            max_new_tokens=max_new,
            batch_size=bs,
            return_margins=True,
        )
        assert isinstance(res, tuple)
        return res

    def _resolve_actions(
        self,
        image: Image.Image,
        task: str,
        actions: list[list[float]] | None,
    ) -> list[list[float]]:
        if actions is not None:
            self.candidates = list(actions)
            return self.candidates

        t_low = (task or "").lower()
        task_allows_kick = (not t_low) or ("kick" in t_low) or ("goal" in t_low)
        base_grid = [
            a
            for a in self._grid_candidates
            if task_allows_kick or abs(float(a[2]) - 9.0) > 1e-6
        ]

        if self.mode == "propose" and not self.mock:
            allow_kick = bool(self.cfg.get("planner", {}).get("include_kick", False)) and task_allows_kick
            max_n = int(self.cfg.get("planner", {}).get("max_candidates", 24))
            proposed = propose_microduck_actions(
                self.model,
                self.processor,
                image,
                task,
                n=self.n_proposals,
                allow_kick=allow_kick,
                max_new_tokens=int(
                    self.cfg.get("planner", {}).get("propose_max_new_tokens", 160)
                ),
                verbose=bool(self.cfg.get("planner", {}).get("verbose_propose", True)),
            )
            # Always pad with the legal grid so ranking never collapses to 1 action.
            merged = pad_proposals_with_grid(
                proposed, base_grid, max_n=max_n
            )
            if merged:
                self.candidates = merged
                return self.candidates

        self.candidates = list(base_grid)
        return self.candidates

    def _score_batch(
        self,
        image: Image.Image,
        task: str,
        actions: list[list[float]],
        outcomes: list[str],
    ) -> list[float]:
        if self.score_mode == "vlm_preference" and not self.mock:
            return preference_scores(
                self.model,
                self.processor,
                image,
                task,
                actions,
                outcomes,
                action_formatter=format_microduck_action,
                max_new_tokens=int(
                    self.cfg.get("planner", {}).get("prefer_max_new_tokens", 64)
                ),
            )
        return [score_from_outcome(o, task=task) for o in outcomes]

    def rank(
        self,
        image: Image.Image,
        task: str,
        actions: list[list[float]] | None = None,
        *,
        progress: bool = False,
    ) -> list[RankedAction]:
        acts = self._resolve_actions(image, task, actions)
        if progress:
            print(
                f"  planner mode={self.mode} score={self.score_mode} "
                f"n={len(acts)} (batched)",
                flush=True,
            )
        outcomes, margins = self._predict_batch(image, task, acts)
        scores = self._score_batch(image, task, acts, outcomes)
        ranked = []
        for a, o, s, m in zip(acts, outcomes, scores, margins):
            yaw_v = float(a[2]) if len(a) > 2 else 0.0
            yaw_pen = 0.0 if abs(yaw_v - 9.0) < 1e-6 else 0.15 * abs(yaw_v)
            ranked.append(
                RankedAction(
                    action=a,
                    outcome=o,
                    score=float(s) - yaw_pen + 0.08 * float(m),
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
                row.append(score_from_outcome(outcome, task=task))
            grid.append(row)
        return xs, ys, grid
