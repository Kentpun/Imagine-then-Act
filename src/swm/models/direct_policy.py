"""Equal-compute direct-policy VLM baseline: (o_t, g) -> a_t.

Uses the exact same Qwen3-VL-2B + LoRA trainer as the semantic world model,
on the exact same training/validation frames, so Layer B and closed-loop
comparisons isolate *imagination planning* vs *direct policy decoding* at
matched model capacity and data budget.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from PIL import Image

_DX_RE = re.compile(r"dx\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_DY_RE = re.compile(r"dy\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_DZ_RE = re.compile(r"dz\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_GRIP_RE = re.compile(r"gripper\s*[:=]\s*(open|close|hold)", re.IGNORECASE)

_VX_RE = re.compile(r"vx\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_YAW_RE = re.compile(r"yaw\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_KICK_RE = re.compile(r"kick\s*[:=]\s*(yes|no)", re.IGNORECASE)


def build_libero_policy_prompt(task: str, *, format_example: bool = False) -> str:
    ex = (
        "Syntax example only (do not copy values): "
        "dx=+0.040; dy=-0.020; dz=+0.000; gripper=open\n"
        if format_example
        else ""
    )
    return (
        f"Task: {task.strip()}\n"
        "Predict the next robot action for this observation.\n"
        "Reply with exactly one line: "
        "dx=<float>; dy=<float>; dz=<float>; gripper=<open|close>\n"
        f"{ex}"
        "Answer:"
    )


def build_libero_react_prompt(task: str) -> str:
    """ReAct baseline: generate explicit verbal Thought before Action."""
    return (
        f"Task: {task.strip()}\n"
        "Use ReAct (Reason + Act): first write a short Thought analyzing the "
        "current visual scene and what step is needed next, then output Action.\n"
        "Use exact two-line format:\n"
        "Thought: <one short sentence about object, gripper, and subgoal>\n"
        "Action: dx=<float>; dy=<float>; dz=<float>; gripper=<open|close>\n"
        "Keep dx, dy, dz in [-0.06, 0.06].\n"
        "Answer:\n"
    )


def format_libero_policy_target(action: Sequence[float]) -> str:
    a = [float(x) for x in action] + [0.0] * 7
    dx, dy, dz = a[0], a[1], a[2]
    grip = "close" if a[6] >= 0.5 else "open"
    return f"dx={dx:+.3f}; dy={dy:+.3f}; dz={dz:+.3f}; gripper={grip}"


def parse_libero_policy_action(text: str) -> list[float] | None:
    dx_m = _DX_RE.search(text or "")
    dy_m = _DY_RE.search(text or "")
    dz_m = _DZ_RE.search(text or "")
    if not (dx_m and dy_m and dz_m):
        return None
    grip_m = _GRIP_RE.search(text or "")
    grip = 1.0 if (grip_m and grip_m.group(1).lower() == "close") else 0.0
    dx = max(-1.0, min(1.0, float(dx_m.group(1))))
    dy = max(-1.0, min(1.0, float(dy_m.group(1))))
    dz = max(-1.0, min(1.0, float(dz_m.group(1))))
    return [dx, dy, dz, 0.0, 0.0, 0.0, grip]


def build_microduck_policy_prompt(task: str, *, format_example: bool = False) -> str:
    ex = (
        "Syntax example only (do not copy values): "
        "vx=0.25; yaw=+0.00; kick=no\n"
        if format_example
        else ""
    )
    return (
        f"Task: {task.strip()}\n"
        "Predict the next biped command for this overhead scene.\n"
        "Reply with exactly one line: "
        "vx=<float>; yaw=<float>; kick=<yes|no>\n"
        f"{ex}"
        "Answer:"
    )


def build_microduck_react_prompt(task: str) -> str:
    """ReAct baseline on MicroDuck overhead scenes: verbal Thought then Action."""
    return (
        f"Task: {task.strip()}\n"
        "Use ReAct (Reason + Act): first write a short Thought about where the "
        "duck, orange ball, and target/zone are, then choose the command.\n"
        "Use exact two-line format:\n"
        "Thought: <one short sentence reasoning about positions and next move>\n"
        "Action: vx=<float>; yaw=<float>; kick=<yes|no>\n"
        "vx in [0.0, 0.25], yaw in [-0.5, 0.5] (positive=left, negative=right).\n"
        "Answer:\n"
    )


def format_microduck_policy_target(action: Sequence[float]) -> str:
    """Convert stored 7D MicroDuck vector [vx, yaw, 0,0,0,0, kick_flag] to text."""
    a = [float(x) for x in action] + [0.0] * 7
    vx, yaw, grip = a[0], a[1], a[6]
    if grip >= 0.5 or abs(yaw - 9.0) < 1e-6:
        return "vx=0.00; yaw=+0.00; kick=yes"
    return f"vx={max(0.0, vx):.2f}; yaw={yaw:+.2f}; kick=no"


def parse_microduck_policy_action(
    text: str, *, allow_kick: bool = True
) -> list[float] | None:
    """Return [vx, 0.0, yaw] or kick sentinel [0.0, 0.0, 9.0]."""
    vx_m = _VX_RE.search(text or "")
    yaw_m = _YAW_RE.search(text or "")
    kick_m = _KICK_RE.search(text or "")
    if kick_m and kick_m.group(1).lower() == "yes" and allow_kick:
        return [0.0, 0.0, 9.0]
    if not (vx_m and yaw_m):
        return None
    vx = max(0.0, min(0.35, float(vx_m.group(1))))
    yaw = max(-1.0, min(1.0, float(yaw_m.group(1))))
    if abs(yaw) > 1e-6 and vx <= 1e-6:
        vx = 0.15
    return [vx, 0.0, yaw]


def action_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Weighted Euclidean distance in (dx, dy, dz, gripper) space."""
    aa = [float(x) for x in a] + [0.0] * 7
    bb = [float(x) for x in b] + [0.0] * 7
    d_xyz = math.sqrt(sum((aa[i] - bb[i]) ** 2 for i in range(3)))
    d_grip = 0.05 * abs((1.0 if aa[6] >= 0.5 else 0.0) - (1.0 if bb[6] >= 0.5 else 0.0))
    return d_xyz + d_grip


def predict_libero_policy_action(
    model,
    processor,
    image: Image.Image,
    task: str,
    *,
    format_example: bool = False,
    max_new_tokens: int = 36,
) -> tuple[list[float], str]:
    """Single forward decode of the direct-policy VLM on LIBERO."""
    from swm.models.semantic_vlm import generate_text

    raw = generate_text(
        model,
        processor,
        image,
        build_libero_policy_prompt(task, format_example=format_example),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    parsed = parse_libero_policy_action(raw)
    if parsed is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], raw
    return parsed, raw


def predict_libero_react_action(
    model,
    processor,
    image: Image.Image,
    task: str,
    *,
    max_new_tokens: int = 96,
) -> tuple[list[float], str]:
    """ReAct controller: decode Thought + Action, then parse Action line."""
    from swm.models.semantic_vlm import generate_text

    raw = generate_text(
        model,
        processor,
        image,
        build_libero_react_prompt(task),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    action_tail = raw.split("Action:", 1)[-1] if "Action:" in raw else raw
    parsed = parse_libero_policy_action(action_tail)
    if parsed is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], raw
    return parsed, raw


def predict_microduck_policy_action(
    model,
    processor,
    image: Image.Image,
    task: str,
    *,
    allow_kick: bool = True,
    format_example: bool = False,
    max_new_tokens: int = 32,
) -> tuple[list[float], str]:
    """Single forward decode of the direct-policy VLM on MicroDuck."""
    from swm.models.semantic_vlm import generate_text

    raw = generate_text(
        model,
        processor,
        image,
        build_microduck_policy_prompt(task, format_example=format_example),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    parsed = parse_microduck_policy_action(raw, allow_kick=allow_kick)
    if parsed is None:
        return [0.0, 0.0, 0.0], raw
    return parsed, raw


def predict_microduck_react_action(
    model,
    processor,
    image: Image.Image,
    task: str,
    *,
    allow_kick: bool = True,
    max_new_tokens: int = 96,
) -> tuple[list[float], str]:
    """ReAct controller on MicroDuck: decode Thought + Action, parse Action."""
    from swm.models.semantic_vlm import generate_text

    raw = generate_text(
        model,
        processor,
        image,
        build_microduck_react_prompt(task),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    action_tail = raw.split("Action:", 1)[-1] if "Action:" in raw else raw
    parsed = parse_microduck_policy_action(action_tail, allow_kick=allow_kick)
    if parsed is None:
        return [0.0, 0.0, 0.0], raw
    return parsed, raw
