from __future__ import annotations

from typing import Sequence

ACTION_TEMPLATE = (
    "Action: velocity=(vx={vx:.2f}, vy={vy:.2f}, yaw_rate={yaw:.2f})"
)


def format_action(action: Sequence[float]) -> str:
    """Format a locomotion command: [vx, vy, yaw_rate, ...]."""
    a = list(action) + [0.0] * 3
    return ACTION_TEMPLATE.format(vx=float(a[0]), vy=float(a[1]), yaw=float(a[2]))


def build_prompt(task: str, action: Sequence[float]) -> str:
    return (
        f"Task: {task.strip()}\n"
        f"{format_action(action)}\n"
        "Predict the semantic outcome after this action.\n"
        "Answer:"
    )
