"""Robot-independent text action commands for VLA LoRA training.

Public robot datasets store actions in incompatible robot coordinates. This
module converts a short continuous action into one stable text command. The
command describes motion in Cartesian task space; it contains no MicroDuck
velocity, joint, MQTT, or task-specific values.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

ACTION_FORMAT = (
    "Reply with exactly one line and no other text.\n"
    "Format: target_visible=<yes|no>; action=<SEARCH|HOLD|MOVE_X_POSITIVE|"
    "MOVE_X_NEGATIVE|MOVE_Y_POSITIVE|MOVE_Y_NEGATIVE|MOVE_Z_POSITIVE|"
    "MOVE_Z_NEGATIVE|GRASP|RELEASE>; magnitude=<none|small|medium|large>; "
    "gripper=<open|close|hold>\n"
    "First identify the object or region requested by the instruction. "
    "target_visible says whether that requested target is visible. "
    "Then choose the next short action toward completing the instruction."
)


_FIELD_RE = re.compile(
    r"(target_visible|action|magnitude|gripper)\s*=\s*([A-Za-z_]+)"
)

ACTION_NAMES = {
    "SEARCH",
    "HOLD",
    "MOVE_X_POSITIVE",
    "MOVE_X_NEGATIVE",
    "MOVE_Y_POSITIVE",
    "MOVE_Y_NEGATIVE",
    "MOVE_Z_POSITIVE",
    "MOVE_Z_NEGATIVE",
    "GRASP",
    "RELEASE",
}


def parse_action_command(text: str) -> dict[str, Any] | None:
    """Parse one trained action-command line; return None for other text."""
    fields = {
        match.group(1).lower(): match.group(2).upper()
        for match in _FIELD_RE.finditer(text or "")
    }
    action = fields.get("action")
    if action not in ACTION_NAMES:
        return None
    visible = fields.get("target_visible")
    if visible not in ("YES", "NO"):
        return None
    magnitude = fields.get("magnitude", "NONE")
    if magnitude not in ("NONE", "SMALL", "MEDIUM", "LARGE"):
        magnitude = "NONE"
    gripper = fields.get("gripper", "HOLD")
    if gripper not in ("OPEN", "CLOSE", "HOLD"):
        gripper = "HOLD"
    return {
        "target_visible": visible == "YES",
        "action": action,
        "magnitude": magnitude.lower(),
        "gripper": gripper.lower(),
    }


def format_action_command(command: dict[str, Any]) -> str:
    """Return the canonical one-line form produced during LoRA training."""
    visible = "yes" if command.get("target_visible") else "no"
    return (
        f"target_visible={visible}; action={str(command['action']).upper()}; "
        f"magnitude={str(command.get('magnitude', 'none')).lower()}; "
        f"gripper={str(command.get('gripper', 'hold')).lower()}"
    )


def build_action_prompt(instruction: str) -> str:
    """Prompt used both for training rows and later VLA inference."""
    return (
        "You are the visual action planner for a robot.\n"
        f"Instruction: {instruction.strip()}\n"
        "Look at the image, identify the requested target, and choose the "
        "next short robot-independent action.\n"
        f"{ACTION_FORMAT}\n"
        "Answer:"
    )


def _finite(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        number = float(value)
        out.append(number if math.isfinite(number) else 0.0)
    return out


def canonical_action_command(
    action: Sequence[float],
    *,
    translation_threshold: float = 0.003,
    gripper_index: int = -1,
    collapse_magnitude: bool = False,
) -> str:
    """Convert one robot action vector to the shared text command format.

    The first three action values are treated as Cartesian translation. A
    final gripper value, when present, uses the common convention that a
    larger value means close. Datasets with only planar actions therefore
    produce HOLD for gripper instead of inventing a grasp state.

    When ``collapse_magnitude`` is true, every MOVE uses magnitude=medium.
    That removes noisy small/medium/large labels so training can focus on
    the action direction that MicroDuck actually uses.
    """
    values = _finite(action)
    if not values:
        return "target_visible=yes; action=HOLD; magnitude=none; gripper=hold"

    translation = (values + [0.0, 0.0, 0.0])[:3]
    norms = [abs(v) for v in translation]
    distance = math.sqrt(sum(v * v for v in translation))

    gripper = "hold"
    gripper_value: float | None = None
    if len(values) >= 7:
        gripper_value = values[gripper_index]
        gripper = "close" if gripper_value >= 0.5 else "open"

    if distance < float(translation_threshold):
        if gripper_value is not None and gripper == "close":
            command = "GRASP"
        elif gripper_value is not None and gripper == "open":
            command = "RELEASE"
        else:
            command = "HOLD"
        return (
            "target_visible=yes; "
            f"action={command}; magnitude=none; gripper={gripper}"
        )

    axis = max(range(3), key=lambda i: norms[i])
    # Prefer a clear dominant axis. Near-ties are labeled HOLD so the model
    # is not punished for arbitrary axis choices on diagonal noise.
    ordered = sorted(range(3), key=lambda i: norms[i], reverse=True)
    if norms[ordered[0]] < max(1e-6, 1.25 * norms[ordered[1]]):
        return (
            "target_visible=yes; action=HOLD; magnitude=none; "
            f"gripper={gripper}"
        )

    axis_name = ("X", "Y", "Z")[axis]
    sign = "POSITIVE" if translation[axis] >= 0.0 else "NEGATIVE"
    if collapse_magnitude:
        magnitude = "medium"
    elif distance < 0.02:
        magnitude = "small"
    elif distance < 0.08:
        magnitude = "medium"
    else:
        magnitude = "large"

    return (
        "target_visible=yes; "
        f"action=MOVE_{axis_name}_{sign}; "
        f"magnitude={magnitude}; gripper={gripper}"
    )
