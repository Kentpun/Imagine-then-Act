"""Map a head-camera frame to one legal MicroDuck walk command."""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

# vx must be 0.25 before this walking policy leaves the stand. 0.18 looks
# like a walk command in the log and still holds trunk_z at about 115 mm.
WALK_VX = 0.25

# Names the duck can execute. vy stays 0; yaw is only used with vx > 0.
WALK_ACTIONS: dict[str, tuple[float, float]] = {
    "forward": (WALK_VX, 0.0),
    "forward-left": (WALK_VX, 0.50),
    "forward-right": (WALK_VX, -0.50),
    "search": (WALK_VX, 0.45),
    "stop": (0.0, 0.0),
}

_ACTION_RE = re.compile(r"forward-left|forward-right|forward|search|stop")

VISION_PROMPT = """This image is the forward camera on a small walking robot.
An orange ball and a blue box are in the room. The task is: walk to the orange ball. Ignore the blue box.
Reply with one word only, nothing else:
forward
forward-left
forward-right
search
stop
Choose forward when the orange ball is near the center.
Choose forward-left when the orange ball is left of center.
Choose forward-right when the orange ball is right of center.
Choose search when the orange ball is not visible.
Choose stop when the orange ball is very close and fills the center of the view.
"""


DUCK_SYSTEM_INSTRUCTION = (
    "The duck's turn radius is large. It cannot pivot, sidestep, or follow "
    "a tight curve. It only walks forward, and a turn is a wide arc of about "
    "two floor tiles (0.2 m). To push the ball back into the circle, do not aim at the ball from "
    "the side or from inside the circle. First walk a wide arc until the duck "
    "is outside the ball and already facing the center, then walk straight so "
    "the ball rolls inward. If the ball is already inside the circle, hold."
)

CIRCLE_PROMPT = """This image is a top-down view of a green circle on the floor and an orange ball.
The task is to keep the orange ball inside the green circle.
Reply with one word only, nothing else:
hold
push_in
Choose hold when the orange ball is inside the green circle.
Choose push_in when the orange ball is outside the green circle.
Ignore the robot. Do not say anything else.
"""


def parse_circle_instruction(text: str) -> str:
    """Return hold or push_in from a model reply."""
    lowered = text.lower()
    if "push_in" in lowered or "push" in lowered:
        return "push_in"
    return "hold"


def parse_vision_action(text: str) -> str:
    """Return the last legal action name mentioned in a model reply."""
    matches = list(_ACTION_RE.finditer(text.lower()))
    if not matches:
        return "search"
    return matches[-1].group(0)


def velocity_for_action(name: str) -> tuple[float, float]:
    vx, yaw = WALK_ACTIONS.get(name, WALK_ACTIONS["search"])
    return float(vx), float(yaw)


def _orange_green(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"))
    orange = (
        (rgb[:, :, 0] > 160)
        & (rgb[:, :, 1] > 50)
        & (rgb[:, :, 1] < 200)
        & (rgb[:, :, 2] < 100)
    )
    green = (
        (rgb[:, :, 1] > 120)
        & (rgb[:, :, 1] > rgb[:, :, 0] + 30)
        & (rgb[:, :, 1] > rgb[:, :, 2] + 30)
        & (rgb[:, :, 0] < 140)
    )
    return orange, green


def keep_ball_in_circle(image: Image.Image) -> tuple[str, float, float, str | None]:
    """Steer from the head camera and kick when the ball is at the feet.

    Orange pixels are the ball. Green pixels are the circle on the floor.
    A kick is sent only when the ball is large, low in the frame, and near
    the center, so the foot meets the ball inside the marked area.
    """
    orange, green = _orange_green(image)
    frac = float(orange.mean())
    if frac < 0.0015:
        return "search", WALK_VX, 0.45, None

    height, width = orange.shape
    ys, xs = np.nonzero(orange)
    cx = float(xs.mean()) / float(width) - 0.5
    cy = float(ys.mean()) / float(height)
    yaw = float(np.clip(-1.6 * cx, -0.7, 0.7))
    if green.any():
        gys, gxs = np.nonzero(green)
        gx = float(gxs.mean()) / float(width) - 0.5
        # Turn so a forward kick points toward the green circle.
        yaw = float(np.clip(-1.4 * (gx - cx), -0.7, 0.7))

    close = frac > 0.035 and cy > 0.45
    aligned = abs(cx) < 0.16
    if cx < -0.12:
        name = "forward-left"
    elif cx > 0.12:
        name = "forward-right"
    else:
        name = "forward"
    # Stay at walking speed when the ball is close. Stopping to "kick"
    # drops the gait back to a stand, so the duck never reaches the ball.
    return name, WALK_VX, 0.0 if close and aligned else yaw, None


def track_orange_ball(image: Image.Image) -> tuple[str, float, float]:
    """Real-time steering from the head camera.

    Orange pixels are the ball. Blue pixels are ignored. The result is a
    legal (vx, yaw) updated on every frame, with no VLM call.
    """
    rgb = np.asarray(image.convert("RGB"))
    orange = (
        (rgb[:, :, 0] > 160)
        & (rgb[:, :, 1] > 50)
        & (rgb[:, :, 1] < 200)
        & (rgb[:, :, 2] < 100)
    )
    frac = float(orange.mean())
    if frac < 0.002:
        return "search", WALK_VX, 0.45
    ys, xs = np.nonzero(orange)
    width = orange.shape[1]
    center = float(xs.mean()) / float(width) - 0.5
    if frac > 0.12 and abs(center) < 0.15:
        return "stop", 0.0, 0.0
    # Image right is positive center. Positive yaw turns the duck left.
    yaw = float(np.clip(-1.6 * center, -0.7, 0.7))
    vx = WALK_VX
    if center < -0.12:
        name = "forward-left"
    elif center > 0.12:
        name = "forward-right"
    else:
        name = "forward"
    return name, vx, yaw


def mock_action_from_image(image: Image.Image) -> str:
    """Color heuristic used when the VLM is not loaded.

    Orange pixels stand in for the ball. This is only a stand-in so the
    camera loop can be checked without Qwen.
    """
    rgb = np.asarray(image.convert("RGB"))
    orange = (
        (rgb[:, :, 0] > 160)
        & (rgb[:, :, 1] > 50)
        & (rgb[:, :, 1] < 200)
        & (rgb[:, :, 2] < 100)
    )
    frac = float(orange.mean())
    if frac < 0.002:
        return "search"
    height, width = orange.shape
    thirds = width // 3
    left = float(orange[:, :thirds].mean())
    mid = float(orange[:, thirds : 2 * thirds].mean())
    right = float(orange[:, 2 * thirds :].mean())
    if frac > 0.08 and mid >= left and mid >= right:
        return "stop"
    scores = {"forward-left": left, "forward": mid, "forward-right": right}
    return max(scores, key=scores.get)
