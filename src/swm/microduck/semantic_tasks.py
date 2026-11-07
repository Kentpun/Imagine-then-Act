"""Multi-task MicroDuck goals + privileged geometric policies for synthetic logs."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

TaskKind = Literal["walk_ab", "kick_goal", "keep_zone", "conditional"]

TASK_PROMPTS: dict[TaskKind, str] = {
    "walk_ab": "walk from zone A to zone B",
    "kick_goal": "kick the orange ball into the goal",
    "keep_zone": "keep the orange ball inside the small green zone",
    "conditional": "if the orange ball is outside the zone then push it in else hold",
}


@dataclass
class SceneState:
    duck_x: float
    duck_y: float
    duck_yaw: float  # radians
    ball_x: float
    ball_y: float
    zone_cx: float = 0.0
    zone_cy: float = 0.0
    zone_r: float = 0.35
    goal_x: float = 1.2
    goal_y: float = 0.0
    zone_b_x: float = 1.0
    zone_b_y: float = 0.5


def ball_in_zone(s: SceneState) -> bool:
    return math.hypot(s.ball_x - s.zone_cx, s.ball_y - s.zone_cy) <= s.zone_r


def facing_goal(s: SceneState, tx: float, ty: float, tol: float = 0.6) -> bool:
    desired = math.atan2(ty - s.duck_y, tx - s.duck_x)
    err = (desired - s.duck_yaw + math.pi) % (2 * math.pi) - math.pi
    return abs(err) < tol


def random_scene(task: TaskKind, rng: random.Random) -> SceneState:
    if task == "walk_ab":
        return SceneState(
            duck_x=rng.uniform(-1.0, -0.3),
            duck_y=rng.uniform(-0.5, 0.5),
            duck_yaw=rng.uniform(-0.3, 0.3),
            ball_x=rng.uniform(0.2, 0.8),
            ball_y=rng.uniform(-0.4, 0.4),
            zone_b_x=rng.uniform(0.8, 1.3),
            zone_b_y=rng.uniform(-0.4, 0.4),
        )
    if task == "kick_goal":
        return SceneState(
            duck_x=rng.uniform(-0.8, -0.2),
            duck_y=rng.uniform(-0.4, 0.4),
            duck_yaw=0.0,
            ball_x=rng.uniform(0.0, 0.5),
            ball_y=rng.uniform(-0.3, 0.3),
            goal_x=1.4,
            goal_y=0.0,
        )
    # keep_zone / conditional — ball starts outside
    ang = rng.uniform(0, 2 * math.pi)
    r = rng.uniform(0.45, 0.9)
    return SceneState(
        duck_x=rng.uniform(-0.6, 0.6),
        duck_y=rng.uniform(-0.6, 0.6),
        duck_yaw=rng.uniform(-math.pi, math.pi),
        ball_x=r * math.cos(ang),
        ball_y=r * math.sin(ang),
        zone_r=0.35,
    )


def privileged_command(task: TaskKind, s: SceneState) -> tuple[float, float, str]:
    """Return (vx, yaw, skill) from privileged geometry (not the LoRA)."""
    if task == "walk_ab":
        if math.hypot(s.duck_x - s.zone_b_x, s.duck_y - s.zone_b_y) < 0.25:
            return 0.0, 0.0, "hold"
        if not facing_goal(s, s.zone_b_x, s.zone_b_y):
            desired = math.atan2(s.zone_b_y - s.duck_y, s.zone_b_x - s.duck_x)
            err = (desired - s.duck_yaw + math.pi) % (2 * math.pi) - math.pi
            return 0.25, 0.5 if err > 0 else -0.5, "walk"
        return 0.25, 0.0, "walk"

    if task == "kick_goal":
        d_ball = math.hypot(s.duck_x - s.ball_x, s.duck_y - s.ball_y)
        # Align duck → ball → goal, then kick
        if d_ball < 0.35 and facing_goal(s, s.goal_x, s.goal_y, tol=1.0):
            return 0.0, 0.0, "kick"
        tx = s.ball_x - 0.2 * math.cos(math.atan2(s.goal_y - s.ball_y, s.goal_x - s.ball_x))
        ty = s.ball_y - 0.2 * math.sin(math.atan2(s.goal_y - s.ball_y, s.goal_x - s.ball_x))
        if not facing_goal(s, tx, ty, tol=0.5):
            desired = math.atan2(ty - s.duck_y, tx - s.duck_x)
            err = (desired - s.duck_yaw + math.pi) % (2 * math.pi) - math.pi
            return 0.25, 0.5 if err > 0 else -0.5, "walk"
        return 0.25, 0.0, "walk"

    # keep_zone / conditional
    inside = ball_in_zone(s)
    if task == "conditional" and inside:
        return 0.0, 0.0, "hold"
    if inside:
        return 0.0, 0.0, "hold"
    # Stand on the far side of the ball relative to zone, then push inward
    ang = math.atan2(s.ball_y - s.zone_cy, s.ball_x - s.zone_cx)
    approach_x = s.ball_x + 0.25 * math.cos(ang)
    approach_y = s.ball_y + 0.25 * math.sin(ang)
    if math.hypot(s.duck_x - approach_x, s.duck_y - approach_y) > 0.2:
        if not facing_goal(s, approach_x, approach_y, tol=0.5):
            desired = math.atan2(approach_y - s.duck_y, approach_x - s.duck_x)
            err = (desired - s.duck_yaw + math.pi) % (2 * math.pi) - math.pi
            return 0.25, 0.5 if err > 0 else -0.5, "walk"
        return 0.25, 0.0, "walk"
    if not facing_goal(s, s.zone_cx, s.zone_cy, tol=0.5):
        desired = math.atan2(s.zone_cy - s.duck_y, s.zone_cx - s.duck_x)
        err = (desired - s.duck_yaw + math.pi) % (2 * math.pi) - math.pi
        return 0.25, 0.5 if err > 0 else -0.5, "walk"
    return 0.25, 0.0, "walk"


def step_scene(
    s: SceneState,
    vx: float,
    yaw_cmd: float,
    dt: float = 0.2,
    *,
    skill: str | None = None,
) -> SceneState:
    """Simple kinematic update for offline synthetic logs (no MuJoCo)."""
    yaw = s.duck_yaw + yaw_cmd * dt * 1.2
    nx = s.duck_x + vx * math.cos(yaw) * dt
    ny = s.duck_y + vx * math.sin(yaw) * dt
    bx, by = s.ball_x, s.ball_y
    if skill == "kick":
        # Kick sends the ball toward the goal marker
        bx += 0.45 * (s.goal_x - bx)
        by += 0.45 * (s.goal_y - by)
    elif math.hypot(nx - bx, ny - by) < 0.28 and vx > 0.05:
        # Walk contact: push ball along duck facing, with a bias to zone center
        push = 0.18
        bx += push * math.cos(yaw) + 0.06 * (s.zone_cx - bx)
        by += push * math.sin(yaw) + 0.06 * (s.zone_cy - by)
    return SceneState(
        duck_x=nx,
        duck_y=ny,
        duck_yaw=yaw,
        ball_x=bx,
        ball_y=by,
        zone_cx=s.zone_cx,
        zone_cy=s.zone_cy,
        zone_r=s.zone_r,
        goal_x=s.goal_x,
        goal_y=s.goal_y,
        zone_b_x=s.zone_b_x,
        zone_b_y=s.zone_b_y,
    )


def render_topdown(s: SceneState, size: int = 256) -> "Image.Image":
    """Synthetic CCTV-like top-down RGB for training without MuJoCo."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (34, 90, 45))
    draw = ImageDraw.Draw(img)

    def xy(x: float, y: float) -> tuple[int, int]:
        # map ~[-1.5,1.5] → pixels
        u = int((x + 1.5) / 3.0 * (size - 1))
        v = int((1.5 - y) / 3.0 * (size - 1))
        return max(0, min(size - 1, u)), max(0, min(size - 1, v))

    def box(x0: float, y0: float, x1: float, y1: float) -> list[tuple[int, int]]:
        a = xy(x0, y0)
        b = xy(x1, y1)
        return [(min(a[0], b[0]), min(a[1], b[1])), (max(a[0], b[0]), max(a[1], b[1]))]

    # zone
    draw.ellipse(
        box(s.zone_cx - s.zone_r, s.zone_cy - s.zone_r, s.zone_cx + s.zone_r, s.zone_cy + s.zone_r),
        outline=(40, 200, 80),
        width=3,
    )
    # goal marker
    g = xy(s.goal_x, s.goal_y)
    draw.rectangle([g[0] - 8, g[1] - 14, g[0] + 8, g[1] + 14], outline=(220, 220, 40), width=2)
    # zone B
    b = xy(s.zone_b_x, s.zone_b_y)
    draw.ellipse([b[0] - 10, b[1] - 10, b[0] + 10, b[1] + 10], outline=(80, 160, 255), width=2)
    # ball
    p = xy(s.ball_x, s.ball_y)
    draw.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=(240, 100, 30))
    # duck
    d = xy(s.duck_x, s.duck_y)
    draw.ellipse([d[0] - 10, d[1] - 10, d[0] + 10, d[1] + 10], fill=(240, 230, 200))
    nose = xy(s.duck_x + 0.15 * math.cos(s.duck_yaw), s.duck_y + 0.15 * math.sin(s.duck_yaw))
    draw.line([d, nose], fill=(200, 80, 40), width=3)
    return img
