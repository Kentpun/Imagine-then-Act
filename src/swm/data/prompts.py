from __future__ import annotations

from typing import Sequence

ACTION_TEMPLATE = (
    "Action: delta_xyz=({dx:.3f},{dy:.3f},{dz:.3f}), gripper={gripper}"
)


def format_action(action: Sequence[float]) -> str:
    """Format a 7D (or longer) action vector into a stable prompt string."""
    a = list(action) + [0.0] * 7
    dx, dy, dz = float(a[0]), float(a[1]), float(a[2])
    grip = float(a[6] if len(action) > 6 else a[-1])
    gripper = "close" if grip >= 0.5 else "open"
    return ACTION_TEMPLATE.format(dx=dx, dy=dy, dz=dz, gripper=gripper)


FORMAT_HINT = (
    "Reply with exactly one line and no other text.\n"
    "Format: moved=<yes|no>; lifted=<yes|no>; gripper=<open|close|hold>; "
    "progress=<up|flat|down>; success_near=<yes|no>\n"
    "Syntax example only (do not copy these values): "
    "moved=yes; lifted=no; gripper=close; progress=flat; success_near=no\n"
)


def build_prompt(
    task: str,
    action: Sequence[float],
    *,
    format_hint: bool = False,
    include_action: bool = True,
) -> str:
    """Build the Layer-A prompt.

    ``include_action=False`` is the strong action ablation: train/eval without
    any action vector in the prompt (stronger than shuffle-at-eval).
    """
    lines = [f"Task: {task.strip()}"]
    if include_action:
        lines.append(format_action(action))
        lines.append("Predict the semantic outcome after this action.")
    else:
        lines.append(
            "Predict the semantic outcome for the next transition "
            "(no action is provided)."
        )
    body = "\n".join(lines) + "\n"
    if format_hint:
        body += FORMAT_HINT
    return body + "Answer:"
