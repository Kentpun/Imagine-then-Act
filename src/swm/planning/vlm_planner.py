"""VLM-driven action proposal and preference ranking (non-hardcoded planner).

The LoRA world model still answers: given (o, g, a) → imagined outcome.
This module replaces the fixed action grid / hand score with:

1. **Propose**: ask the VLM for task-conditioned candidate actions
2. **Prefer**: ask the VLM which imagined outcomes best advance the goal

``grid`` + ``heuristic`` remain available as ablation / fallback modes.
"""

from __future__ import annotations

import re
from typing import Sequence

from PIL import Image

from swm.models.semantic_vlm import generate_text

_VX_RE = re.compile(r"vx\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_YAW_RE = re.compile(r"yaw\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_KICK_RE = re.compile(r"kick\s*[:=]\s*(yes|no)", re.IGNORECASE)
_DX_RE = re.compile(r"dx\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_DY_RE = re.compile(r"dy\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_DZ_RE = re.compile(r"dz\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_GRIP_RE = re.compile(r"gripper\s*[:=]\s*(open|close)", re.IGNORECASE)

_RANK_TOKEN_RE = re.compile(r"\d+")


def build_libero_propose_prompt(task: str, n: int) -> str:
    n = int(n)
    examples = "\n".join(
        [
            "dx=0.05; dy=0.00; dz=0.00; gripper=open",
            "dx=0.00; dy=0.05; dz=0.00; gripper=close",
            "dx=-0.03; dy=-0.02; dz=0.02; gripper=open",
            "dx=0.02; dy=-0.04; dz=0.00; gripper=close",
        ][:n]
    )
    return (
        f"Task: {task.strip()}\n"
        "Propose several DIFFERENT short robot actions that might help.\n"
        f"Output exactly {n} lines. No numbering, no explanation.\n"
        "Each line must look like:\n"
        "dx=<float>; dy=<float>; dz=<float>; gripper=<open|close>\n"
        f"Example of {n} diverse lines:\n{examples}\n"
        "Now propose your own diverse lines for this image/task.\n"
        "Answer:"
    )


def build_microduck_propose_prompt(task: str, n: int, *, allow_kick: bool) -> str:
    n = int(n)
    kick_rule = (
        "At most one line may use kick=yes.\n"
        if allow_kick
        else "Every line must use kick=no.\n"
    )
    examples = "\n".join(
        [
            "vx=0.25; yaw=0.00; kick=no",
            "vx=0.15; yaw=0.50; kick=no",
            "vx=0.15; yaw=-0.50; kick=no",
            "vx=0.00; yaw=0.00; kick=yes"
            if allow_kick
            else "vx=0.20; yaw=0.20; kick=no",
        ][:n]
    )
    return (
        f"Task: {task.strip()}\n"
        "Propose several DIFFERENT biped walking commands.\n"
        f"Output exactly {n} lines. No numbering, no explanation.\n"
        "Each line must look like:\n"
        "vx=<float>; yaw=<float>; kick=<yes|no>\n"
        "vx in [0, 0.35], yaw in [-1.0, 1.0].\n"
        f"{kick_rule}"
        f"Example of {n} diverse lines:\n{examples}\n"
        "Now propose your own diverse lines for this image/task.\n"
        "Answer:"
    )

def build_preference_prompt(
    task: str,
    actions: Sequence[Sequence[float]],
    outcomes: Sequence[str],
    *,
    action_formatter,
) -> str:
    lines = [
        f"Task: {task.strip()}",
        "Below are candidate actions and their imagined semantic outcomes.",
        "Rank ALL candidates from best to worst for completing the task.",
        "Reply with only a chain of 1-based indices like: 2>1>3",
        "",
    ]
    for i, (action, outcome) in enumerate(zip(actions, outcomes), start=1):
        lines.append(f"Candidate {i}: {action_formatter(action)}")
        lines.append(f"  imagined: {outcome}")
    lines.append("Answer:")
    return "\n".join(lines)


def parse_libero_proposals(text: str, *, max_n: int) -> list[list[float]]:
    actions: list[list[float]] = []
    for line in (text or "").splitlines():
        dx_m, dy_m, dz_m = _DX_RE.search(line), _DY_RE.search(line), _DZ_RE.search(line)
        if not (dx_m and dy_m and dz_m):
            continue
        grip_m = _GRIP_RE.search(line)
        grip = 1.0 if grip_m and grip_m.group(1).lower() == "close" else 0.0
        dx = max(-0.1, min(0.1, float(dx_m.group(1))))
        dy = max(-0.1, min(0.1, float(dy_m.group(1))))
        dz = max(-0.05, min(0.05, float(dz_m.group(1))))
        actions.append([dx, dy, dz, 0.0, 0.0, 0.0, grip])
        if len(actions) >= max_n:
            break
    return _dedupe_actions(actions)


def parse_microduck_proposals(
    text: str, *, max_n: int, allow_kick: bool
) -> list[list[float]]:
    actions: list[list[float]] = []
    # Prefer line-wise parse; also accept a dense blob with repeated vx/yaw pairs.
    chunks = (text or "").splitlines() or [text or ""]
    dense = text or ""
    if _VX_RE.search(dense) and len(_VX_RE.findall(dense)) > 1:
        # Split densely emitted proposals on each new vx=
        parts = re.split(r"(?=vx\s*[:=])", dense, flags=re.IGNORECASE)
        chunks = [p for p in parts if p.strip()] + chunks

    for line in chunks:
        vx_m, yaw_m = _VX_RE.search(line), _YAW_RE.search(line)
        if not (vx_m and yaw_m):
            continue
        kick_m = _KICK_RE.search(line)
        kick = bool(kick_m and kick_m.group(1).lower() == "yes" and allow_kick)
        if kick:
            actions.append([0.0, 0.0, 9.0])
        else:
            vx = max(0.0, min(0.35, float(vx_m.group(1))))
            yaw = max(-1.0, min(1.0, float(yaw_m.group(1))))
            if abs(yaw) > 1e-6 and vx <= 1e-6:
                vx = 0.1  # walking policy cannot turn in place
            actions.append([vx, 0.0, yaw])
        if len(actions) >= max_n:
            break
    return _dedupe_actions(actions)


def pad_proposals_with_grid(
    proposed: list[list[float]],
    grid: list[list[float]],
    *,
    max_n: int,
) -> list[list[float]]:
    """Keep VLM proposals first, then fill from the grid so ranking has options."""
    out = _dedupe_actions(list(proposed))
    for action in grid:
        if len(out) >= max_n:
            break
        out = _dedupe_actions(out + [list(action)])
    return out[:max_n]

def parse_preference_ranking(text: str, n: int) -> list[int]:
    """Parse '2>1>3' / '2,1,3' / 'best is 2' into a permutation of 1..n."""
    tokens = [int(t) for t in _RANK_TOKEN_RE.findall(text or "")]
    order: list[int] = []
    for idx in tokens:
        if 1 <= idx <= n and idx not in order:
            order.append(idx)
    for idx in range(1, n + 1):
        if idx not in order:
            order.append(idx)
    return order


def ranking_to_scores(order: Sequence[int], n: int) -> list[float]:
    """Higher score = better. order is 1-based best→worst."""
    scores = [0.0] * n
    for rank, idx in enumerate(order):
        if 1 <= idx <= n:
            scores[idx - 1] = float(n - rank)
    return scores


def format_libero_action(action: Sequence[float]) -> str:
    a = list(action) + [0.0] * 7
    grip = "close" if float(a[6]) >= 0.5 else "open"
    return f"dx={float(a[0]):+.3f}; dy={float(a[1]):+.3f}; dz={float(a[2]):+.3f}; gripper={grip}"


def format_microduck_action(action: Sequence[float]) -> str:
    vx = float(action[0]) if action else 0.0
    yaw = float(action[2]) if len(action) > 2 else 0.0
    if abs(yaw - 9.0) < 1e-6:
        return "vx=0.00; yaw=0.00; kick=yes"
    return f"vx={vx:.2f}; yaw={yaw:+.2f}; kick=no"


def _dedupe_actions(actions: list[list[float]], tol: float = 1e-3) -> list[list[float]]:
    unique: list[list[float]] = []
    for action in actions:
        if any(
            all(abs(float(a) - float(b)) <= tol for a, b in zip(action, prev))
            for prev in unique
        ):
            continue
        unique.append(action)
    return unique


def propose_libero_actions(
    model,
    processor,
    image: Image.Image,
    task: str,
    *,
    n: int,
    max_new_tokens: int = 160,
    verbose: bool = False,
) -> list[list[float]]:
    raw = generate_text(
        model,
        processor,
        image,
        build_libero_propose_prompt(task, n),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
    )
    if verbose:
        print(f"  [propose raw]\n{raw}", flush=True)
    return parse_libero_proposals(raw, max_n=n)


def propose_microduck_actions(
    model,
    processor,
    image: Image.Image,
    task: str,
    *,
    n: int,
    allow_kick: bool,
    max_new_tokens: int = 120,
    verbose: bool = False,
) -> list[list[float]]:
    raw = generate_text(
        model,
        processor,
        image,
        build_microduck_propose_prompt(task, n, allow_kick=allow_kick),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
    )
    if verbose:
        print(f"  [propose raw]\n{raw}", flush=True)
    return parse_microduck_proposals(raw, max_n=n, allow_kick=allow_kick)

def preference_scores(
    model,
    processor,
    image: Image.Image,
    task: str,
    actions: Sequence[Sequence[float]],
    outcomes: Sequence[str],
    *,
    action_formatter,
    max_new_tokens: int = 64,
) -> list[float]:
    if not actions:
        return []
    if len(actions) == 1:
        return [1.0]
    prompt = build_preference_prompt(
        task, actions, outcomes, action_formatter=action_formatter
    )
    raw = generate_text(
        model,
        processor,
        image,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    order = parse_preference_ranking(raw, len(actions))
    return ranking_to_scores(order, len(actions))
