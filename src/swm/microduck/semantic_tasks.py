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
    "keep_zone": "keep the orange ball inside the green circle",
    "conditional": "if the orange ball is outside the circle then push it in else hold",
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
    return abs(heading_error(s, tx, ty)) < tol


def heading_error(s: SceneState, tx: float, ty: float) -> float:
    desired = math.atan2(ty - s.duck_y, tx - s.duck_x)
    return (desired - s.duck_yaw + math.pi) % (2 * math.pi) - math.pi


def navigation_target(task: TaskKind, s: SceneState) -> tuple[float, float]:
    """Subgoal (tx, ty) that the duck needs to approach/face right now."""
    if task == "walk_ab":
        return s.zone_b_x, s.zone_b_y
    if task == "kick_goal":
        d_ball = math.hypot(s.duck_x - s.ball_x, s.duck_y - s.ball_y)
        if d_ball < 0.35:
            return s.goal_x, s.goal_y
        ang_bg = math.atan2(s.goal_y - s.ball_y, s.goal_x - s.ball_x)
        tx = s.ball_x - 0.2 * math.cos(ang_bg)
        ty = s.ball_y - 0.2 * math.sin(ang_bg)
        if math.hypot(s.duck_x - tx, s.duck_y - ty) < 0.22:
            return s.goal_x, s.goal_y
        return tx, ty
    # keep_zone / conditional
    if ball_in_zone(s):
        return s.duck_x, s.duck_y
    ang = math.atan2(s.ball_y - s.zone_cy, s.ball_x - s.zone_cx)
    approach_x = s.ball_x + 0.32 * math.cos(ang)
    approach_y = s.ball_y + 0.32 * math.sin(ang)
    if math.hypot(s.duck_x - approach_x, s.duck_y - approach_y) > 0.24:
        return approach_x, approach_y
    return s.zone_cx, s.zone_cy


def random_scene(task: TaskKind, rng: random.Random) -> SceneState:
    if task == "walk_ab":
        return SceneState(
            duck_x=rng.uniform(-1.0, -0.3),
            duck_y=rng.uniform(-0.5, 0.5),
            duck_yaw=rng.uniform(-0.8, 0.8),
            ball_x=rng.uniform(0.2, 0.8),
            ball_y=rng.uniform(-0.4, 0.4),
            zone_b_x=rng.uniform(0.8, 1.3),
            zone_b_y=rng.uniform(-0.5, 0.5),
        )
    if task == "kick_goal":
        return SceneState(
            duck_x=rng.uniform(-0.8, -0.2),
            duck_y=rng.uniform(-0.4, 0.4),
            duck_yaw=rng.uniform(-0.7, 0.7),
            ball_x=rng.uniform(0.0, 0.5),
            ball_y=rng.uniform(-0.3, 0.3),
            goal_x=1.4,
            goal_y=0.0,
        )
    ang = rng.uniform(-math.pi, math.pi)
    if task == "conditional" and rng.random() < 0.5:
        # "else hold" branch: ball already near inner edge, duck right behind it facing across
        r_in = rng.uniform(0.20, 0.31)
        bx = r_in * math.cos(ang)
        by = r_in * math.sin(ang)
        dx = bx - 0.19 * math.cos(ang)
        dy = by - 0.19 * math.sin(ang)
        return SceneState(
            duck_x=dx,
            duck_y=dy,
            duck_yaw=ang + rng.uniform(-0.2, 0.2),
            ball_x=bx,
            ball_y=by,
            zone_r=0.35,
        )
    # keep_zone / "push it in" branch — ball outside, duck on outer side with heading offset
    r = rng.uniform(0.42, 0.68)
    bx = r * math.cos(ang)
    by = r * math.sin(ang)
    duck_ang = ang + rng.uniform(-0.55, 0.55)
    duck_r = r + rng.uniform(0.30, 0.48)
    inward_yaw = ((ang + math.pi) + math.pi) % (2 * math.pi) - math.pi
    return SceneState(
        duck_x=duck_r * math.cos(duck_ang),
        duck_y=duck_r * math.sin(duck_ang),
        duck_yaw=inward_yaw + rng.uniform(-0.85, 0.85),
        ball_x=bx,
        ball_y=by,
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
    approach_x = s.ball_x + 0.32 * math.cos(ang)
    approach_y = s.ball_y + 0.32 * math.sin(ang)
    if math.hypot(s.duck_x - approach_x, s.duck_y - approach_y) > 0.24:
        if not facing_goal(s, approach_x, approach_y, tol=0.45):
            desired = math.atan2(approach_y - s.duck_y, approach_x - s.duck_x)
            err = (desired - s.duck_yaw + math.pi) % (2 * math.pi) - math.pi
            return 0.25, 0.5 if err > 0 else -0.5, "walk"
        return 0.25, 0.0, "walk"
    if not facing_goal(s, s.zone_cx, s.zone_cy, tol=0.45):
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
        # Kick only contacts the ball when duck is within striking range
        if math.hypot(s.duck_x - bx, s.duck_y - by) < 0.38:
            bx += 0.45 * (s.goal_x - bx)
            by += 0.45 * (s.goal_y - by)
    elif math.hypot(nx - bx, ny - by) < 0.24 and vx > 0.05:
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


def task_solved(task: TaskKind, s: SceneState) -> bool:
    if task == "walk_ab":
        return math.hypot(s.duck_x - s.zone_b_x, s.duck_y - s.zone_b_y) < 0.30
    if task == "kick_goal":
        return math.hypot(s.ball_x - s.goal_x, s.ball_y - s.goal_y) < 0.35
    return ball_in_zone(s)


def label_microduck_transition(
    task: TaskKind,
    s0: SceneState,
    s1: SceneState,
    vx: float,
    yaw: float,
    skill: str | None = None,
) -> str:
    """Truthful next-state (s0, a) -> s1 semantic outcome for world-model supervision."""
    eff_skill = str(skill or ("walk" if abs(vx) > 0.05 else "hold"))
    if eff_skill not in ("walk", "kick", "hold"):
        eff_skill = "walk" if abs(vx) > 0.05 else "hold"

    moved = "yes" if (eff_skill != "kick" and abs(vx) > 0.05) else "no"
    inside1 = ball_in_zone(s1)
    ball_in = "yes" if inside1 else "no"

    # Case 1: keep_zone / conditional when ball is already inside green circle
    if task in ("keep_zone", "conditional") and ball_in_zone(s0):
        if eff_skill == "hold" and inside1:
            return (
                f"moved=no; lifted=no; gripper=hold; progress=up; "
                f"success_near=yes; ball_in_zone=yes; facing_goal=yes; skill=hold"
            )
        if not inside1:
            return (
                f"moved={moved}; lifted=no; gripper=hold; progress=down; "
                f"success_near=no; ball_in_zone=no; facing_goal=no; skill={eff_skill}"
            )
        return (
            f"moved={moved}; lifted=no; gripper=hold; progress=flat; "
            f"success_near=yes; ball_in_zone=yes; facing_goal=yes; skill={eff_skill}"
        )

    # Case 2: walk_ab already inside target Zone B
    if task == "walk_ab" and task_solved("walk_ab", s0):
        if eff_skill == "hold":
            return (
                f"moved=no; lifted=no; gripper=hold; progress=up; "
                f"success_near=yes; ball_in_zone={ball_in}; facing_goal=yes; skill=hold"
            )

    # Case 3: kick skill — only positive if ball actually moved toward goal on kick_goal
    if eff_skill == "kick":
        d_goal0 = math.hypot(s0.ball_x - s0.goal_x, s0.ball_y - s0.goal_y)
        d_goal1 = math.hypot(s1.ball_x - s1.goal_x, s1.ball_y - s1.goal_y)
        if task == "kick_goal" and d_goal1 < d_goal0 - 0.05:
            near = "yes" if d_goal1 < 0.65 else "no"
            return (
                f"moved=no; lifted=no; gripper=close; progress=up; "
                f"success_near={near}; ball_in_zone={ball_in}; facing_goal=yes; skill=kick"
            )
        return (
            f"moved=no; lifted=no; gripper=close; progress=down; "
            f"success_near=no; ball_in_zone={ball_in}; facing_goal=no; skill=kick"
        )

    # Case 4: general locomotion / ball-pushing toward current subgoal
    tx, ty = navigation_target(task, s0)
    err0 = abs(heading_error(s0, tx, ty))
    err1 = abs(heading_error(s1, tx, ty))
    d0 = math.hypot(s0.duck_x - tx, s0.duck_y - ty)
    d1 = math.hypot(s1.duck_x - tx, s1.duck_y - ty)

    facing = "yes" if err1 < 0.45 else "no"
    succ_near = "yes" if task_solved(task, s1) else "no"
    if task == "walk_ab" and math.hypot(s1.duck_x - s1.zone_b_x, s1.duck_y - s1.zone_b_y) < 0.42:
        succ_near = "yes"

    # Check if ball itself moved meaningfully closer to target zone/goal
    ball_prog = False
    if task in ("keep_zone", "conditional"):
        bz0 = math.hypot(s0.ball_x - s0.zone_cx, s0.ball_y - s0.zone_cy)
        bz1 = math.hypot(s1.ball_x - s1.zone_cx, s1.ball_y - s1.zone_cy)
        if bz1 < bz0 - 0.03:
            ball_prog = True
    elif task == "kick_goal":
        bg0 = math.hypot(s0.ball_x - s0.goal_x, s0.ball_y - s0.goal_y)
        bg1 = math.hypot(s1.ball_x - s1.goal_x, s1.ball_y - s1.goal_y)
        if bg1 < bg0 - 0.03:
            ball_prog = True

    if abs(vx) <= 0.05:
        progress = "flat"
    elif ball_prog or (not task_solved(task, s0) and task_solved(task, s1)):
        progress = "up"
    elif err0 >= 0.45:
        # Misaligned: turning toward subgoal is progress=up; turning away is progress=down
        if err1 < err0 - 0.04:
            progress = "up"
        elif err1 > err0 + 0.04:
            progress = "down"
        else:
            progress = "flat"
    else:
        # Aligned: walking straight closer is progress=up; veering out of alignment is down
        if err1 < 0.48 and d1 < d0 - 0.01:
            progress = "up"
        elif err1 >= 0.48 or d1 > d0 + 0.01:
            progress = "down"
        else:
            progress = "flat"

    return (
        f"moved={moved}; lifted=no; gripper=hold; progress={progress}; "
        f"success_near={succ_near}; ball_in_zone={ball_in}; "
        f"facing_goal={facing}; skill={eff_skill}"
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
    # duck + clear heading beak/tip so VLMs can read yaw accurately
    d = xy(s.duck_x, s.duck_y)
    draw.ellipse([d[0] - 10, d[1] - 10, d[0] + 10, d[1] + 10], fill=(240, 230, 200))
    nose = xy(s.duck_x + 0.26 * math.cos(s.duck_yaw), s.duck_y + 0.26 * math.sin(s.duck_yaw))
    draw.line([d, nose], fill=(220, 60, 30), width=4)
    draw.ellipse([nose[0] - 3, nose[1] - 3, nose[0] + 3, nose[1] + 3], fill=(255, 220, 40))
    return img
