from pathlib import Path

import numpy as np
from PIL import Image

from swm.microduck.vision_actions import (
    keep_ball_in_circle,
    mock_action_from_image,
    parse_vision_action,
    track_orange_ball,
    velocity_for_action,
)


def test_parse_takes_the_last_legal_word():
    text = "The options are forward and stop.\nforward-left"
    assert parse_vision_action(text) == "forward-left"


def test_parse_does_not_split_forward_left():
    assert parse_vision_action("forward-left") == "forward-left"
    assert parse_vision_action("I cannot see it, search") == "search"


def test_parse_unknown_searches():
    assert parse_vision_action("moved=yes; progress=up") == "search"


def test_stop_has_zero_speed():
    assert velocity_for_action("stop") == (0.0, 0.0)
    vx, yaw = velocity_for_action("forward-right")
    assert vx > 0 and yaw < 0


def test_mock_turns_toward_orange_on_the_left():
    image = Image.new("RGB", (90, 30), (20, 20, 20))
    for x in range(0, 20):
        for y in range(5, 25):
            image.putpixel((x, y), (220, 120, 20))
    assert mock_action_from_image(image) == "forward-left"
    name, vx, yaw = track_orange_ball(image)
    assert name == "forward-left"
    assert vx > 0 and yaw > 0


def test_close_centered_ball_keeps_walking():
    image = Image.new("RGB", (90, 60), (20, 20, 20))
    for x in range(38, 52):
        for y in range(40, 58):
            image.putpixel((x, y), (220, 120, 20))
    name, vx, yaw, kick = keep_ball_in_circle(image)
    assert kick is None
    assert name == "forward"
    assert vx == 0.25 and yaw == 0.0


def test_circle_instruction_holds_inside_and_pushes_outside():
    from swm.microduck.vision_actions import parse_circle_instruction

    assert parse_circle_instruction("push_in") == "push_in"
    assert parse_circle_instruction("The ball should hold still") == "hold"
    assert parse_circle_instruction("moved=yes; lifted=no") is None

    sys_path = Path(__file__).resolve().parents[2] / "microduck_demo_part1" / "src"
    import sys

    sys.path.insert(0, str(sys_path))
    from microduck_demo.circle_instruction import instruction_from_rgb

    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cy, cx = 60, 60
    yy, xx = np.ogrid[:120, :120]
    ring = (np.hypot(xx - cx, yy - cy) > 46) & (np.hypot(xx - cx, yy - cy) < 50)
    image[ring] = (30, 200, 40)
    inside = image.copy()
    inside[54:66, 54:66] = (220, 120, 20)
    assert instruction_from_rgb(inside) == "hold"
    outside = image.copy()
    outside[8:20, 100:112] = (220, 120, 20)
    assert instruction_from_rgb(outside) == "push_in"


def test_missing_ball_searches_without_a_kick():
    image = Image.new("RGB", (90, 60), (20, 20, 20))
    name, vx, yaw, kick = keep_ball_in_circle(image)
    assert name == "search"
    assert kick is None
    assert vx > 0
