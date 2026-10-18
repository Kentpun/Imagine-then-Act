from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MicroDuckLabelConfig:
    progress_vx: float = 0.15
    turn_yaw: float = 0.25
    fall_speed: float = 1.2


def label_command(
    action: Sequence[float],
    *,
    upright: bool = True,
    goal_near: bool = False,
    cfg: MicroDuckLabelConfig | None = None,
) -> dict[str, str]:
    """Heuristic labels for smoke / mock demos (no MuJoCo required)."""
    cfg = cfg or MicroDuckLabelConfig()
    a = np.asarray(list(action) + [0.0] * 3, dtype=np.float32)
    vx, vy, yaw = float(a[0]), float(a[1]), float(a[2])
    speed = float(np.hypot(vx, vy))

    progressed = "yes" if abs(vx) >= cfg.progress_vx or abs(vy) >= cfg.progress_vx else "no"
    if abs(yaw) < cfg.turn_yaw:
        turned = "none"
    else:
        turned = "left" if yaw > 0 else "right"

    fell = "yes" if (not upright) or speed >= cfg.fall_speed else "no"
    upright_s = "no" if fell == "yes" else ("yes" if upright else "no")

    return {
        "upright": upright_s,
        "progressed": progressed,
        "turned": turned,
        "fell": fell,
        "goal_near": "yes" if goal_near else "no",
    }


def build_outcome_string(labels: dict[str, str]) -> str:
    return (
        f"upright={labels['upright']}; "
        f"progressed={labels['progressed']}; "
        f"turned={labels['turned']}; "
        f"fell={labels['fell']}; "
        f"goal_near={labels['goal_near']}"
    )


def score_from_outcome(outcome: str) -> float:
    """Map a predicted outcome string to a scalar for planning.

    Accepts MicroDuck labels and the LIBERO LoRA schema
    (moved / progress / success_near), so a Part A checkpoint can rank velocities.
    """
    text = outcome.lower().replace(" ", "")
    score = 0.0
    if "progressed=yes" in text:
        score += 2.0
    if "upright=yes" in text:
        score += 1.5
    if "fell=yes" in text:
        score -= 3.0
    if "turned=left" in text or "turned=right" in text:
        score += 0.5
    if "goal_near=yes" in text:
        score += 3.0
    # LIBERO-trained checkpoint emits this schema.
    if "moved=yes" in text:
        score += 1.5
    if "progress=up" in text:
        score += 2.0
    if "progress=down" in text:
        score -= 1.0
    if "success_near=yes" in text:
        score += 2.0
    if "lifted=yes" in text:
        score += 0.25
    return score


def mock_outcome(action: Sequence[float], task: str = "") -> str:
    """Deterministic mock world model for UI demos without a VLM."""
    labels = label_command(action)
    # Tiny task bias: prefer forward motion when the task mentions walk/forward
    t = task.lower()
    if any(k in t for k in ("walk", "forward", "go to", "approach")):
        vx = float(list(action)[0]) if action else 0.0
        if vx > 0.1:
            labels["progressed"] = "yes"
            labels["goal_near"] = "yes" if vx >= 0.4 else labels["goal_near"]
        if vx < -0.05:
            labels["progressed"] = "no"
    if "turn left" in t or "left" in t:
        yaw = float(list(action)[2]) if len(action) > 2 else 0.0
        if yaw > 0.2:
            labels["turned"] = "left"
            labels["progressed"] = "yes"
    if "turn right" in t or "right" in t:
        yaw = float(list(action)[2]) if len(action) > 2 else 0.0
        if yaw < -0.2:
            labels["turned"] = "right"
            labels["progressed"] = "yes"
    return build_outcome_string(labels)
