from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class LabelConfig:
    move_threshold: float = 0.01
    lift_threshold: float = 0.01
    success_near_frames: int = 15


def _as_np(x: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    return np.asarray(x, dtype=np.float32)


def label_transition(
    action: Sequence[float],
    state_t: Sequence[float] | None,
    state_tp1: Sequence[float] | None,
    *,
    frame_idx: int,
    episode_len: int,
    episode_success: bool,
    cfg: LabelConfig,
) -> dict[str, str]:
    """Derive semantic labels from action + proprioceptive state.

    LIBERO-style states are typically 8D (eef pose-ish + gripper). We only
    rely on relative deltas so this stays robust across minor schema changes.
    """
    a = _as_np(action)
    assert a is not None
    s0 = _as_np(state_t)
    s1 = _as_np(state_tp1)

    # Movement from action command (always available)
    planar = float(np.linalg.norm(a[:2]))
    moved = "yes" if planar > cfg.move_threshold else "no"

    grip_cmd = float(a[6] if a.shape[0] > 6 else a[-1])
    if grip_cmd >= 0.65:
        gripper = "close"
    elif grip_cmd <= 0.35:
        gripper = "open"
    else:
        gripper = "hold"

    lifted = "no"
    progress = "flat"
    if s0 is not None and s1 is not None and s0.shape[0] >= 3 and s1.shape[0] >= 3:
        dz = float(s1[2] - s0[2])
        if dz > cfg.lift_threshold and gripper in {"close", "hold"}:
            lifted = "yes"
        # Crude progress: moving toward higher |z| while closing often means pick progress
        delta = float(np.linalg.norm(s1[:3] - s0[:3]))
        if delta > cfg.move_threshold:
            progress = "up" if dz >= 0 else "down"
        else:
            progress = "flat"
    else:
        # Fallback using action only
        if float(a[2]) > cfg.lift_threshold and gripper == "close":
            lifted = "yes"
            progress = "up"
        elif float(a[2]) < -cfg.lift_threshold:
            progress = "down"

    remaining = episode_len - frame_idx - 1
    success_near = (
        "yes"
        if episode_success and remaining <= cfg.success_near_frames
        else "no"
    )

    return {
        "moved": moved,
        "lifted": lifted,
        "gripper": gripper,
        "progress": progress,
        "success_near": success_near,
    }


def build_outcome_string(labels: dict[str, str]) -> str:
    return (
        f"moved={labels['moved']}; "
        f"lifted={labels['lifted']}; "
        f"gripper={labels['gripper']}; "
        f"progress={labels['progress']}; "
        f"success_near={labels['success_near']}"
    )


def score_from_outcome(outcome: str) -> float:
    """Map a predicted outcome string to a scalar for planning."""
    text = outcome.lower().replace(" ", "")
    score = 0.0
    if "progress=up" in text:
        score += 2.0
    if "progress=down" in text:
        score -= 1.0
    if "lifted=yes" in text:
        score += 1.5
    if "success_near=yes" in text:
        score += 3.0
    if "moved=yes" in text:
        score += 0.25
    if "gripper=close" in text:
        score += 0.1
    return score
