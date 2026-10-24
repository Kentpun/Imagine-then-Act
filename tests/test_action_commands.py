from pathlib import Path

from PIL import Image

from swm.data.action_commands import (
    build_action_prompt,
    canonical_action_command,
    format_action_command,
    parse_action_command,
)
from swm.data.dataset import SemanticTransitionDataset
from swm.utils.io import write_jsonl


def test_canonical_action_uses_task_space_not_microduck_velocity():
    command = canonical_action_command([0.05, 0.0, 0.0, 0, 0, 0, 0])
    assert "action=MOVE_X_POSITIVE" in command
    assert "vx" not in command
    assert "push_in" not in command


def test_canonical_action_hold_and_grasp():
    assert "action=HOLD" in canonical_action_command([0.0, 0.0])
    assert "action=GRASP" in canonical_action_command([0, 0, 0, 0, 0, 0, 1])


def test_collapse_magnitude_and_diagonal_hold():
    collapsed = canonical_action_command(
        [0.05, 0.0, 0.0], collapse_magnitude=True
    )
    assert "action=MOVE_X_POSITIVE" in collapsed
    assert "magnitude=medium" in collapsed
    # Near-equal X/Y should not invent a dominant MOVE axis.
    assert "action=HOLD" in canonical_action_command([0.04, 0.04, 0.0])


def test_action_prompt_asks_for_identification_and_command():
    prompt = build_action_prompt("move the red block to the blue block")
    assert "identify the requested target" in prompt
    assert prompt.endswith("Answer:")


def test_parse_and_format_action_command():
    command = canonical_action_command([0.05, 0.0, 0.0, 0, 0, 0, 0])
    parsed = parse_action_command(command)
    assert parsed == {
        "target_visible": True,
        "action": "MOVE_X_POSITIVE",
        "magnitude": "medium",
        "gripper": "open",
    }
    assert format_action_command(parsed) == command
    assert parse_action_command("push_in") is None


def test_dataset_prefers_action_prompt_and_answer(tmp_path: Path):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 16), (255, 128, 0)).save(image_path)
    row = {
        "episode_id": 3,
        "task": "move the block left",
        "action": [-0.04, 0.0],
        "image_path": str(image_path),
        "prompt": build_action_prompt("move the block left"),
        "answer": canonical_action_command([-0.04, 0.0]),
    }
    jsonl_path = tmp_path / "train.jsonl"
    write_jsonl(jsonl_path, [row])

    sample = SemanticTransitionDataset(jsonl_path, tmp_path)[0]
    assert sample["answer"].startswith("target_visible=yes; action=MOVE_X_NEGATIVE")
    assert "move the block left" in sample["prompt"]
