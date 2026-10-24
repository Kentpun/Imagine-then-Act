"""Ask the general action LoRA what to do in the MicroDuck simulator.

The simulator writes `.demo/vision/zone.ppm`. For each new frame, the
Qwen3-VL-2B action LoRA receives the user's instruction and replies with a
robot-independent action command. This loop republishes that command at 10 Hz
together with the MicroDuck (vx, yaw) mapping so the duck keeps walking
between slow VLM frames.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from PIL import Image

from swm.data.action_commands import (
    build_action_prompt,
    format_action_command,
    parse_action_command,
)
from swm.microduck.mqtt_control import MqttVelocityClient, resolve_session

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRAME = (
    REPO_ROOT.parent / "microduck_demo_part1" / ".demo" / "vision" / "zone.ppm"
)
DEFAULT_SESSION = REPO_ROOT / ".demo" / "session.env"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "superpod_action" / "checkpoint-final"

# Keep in sync with microduck_demo_part1 action_adapter (send.sh presets).
_VX = 0.25
_YAW = 0.50


def _velocity_for_action(action_command: str) -> tuple[float, float]:
    """Map LoRA action text to MicroDuck walking-policy (vx, yaw)."""
    fields = parse_action_command(action_command)
    if fields is None or not fields.get("target_visible"):
        return 0.0, 0.0
    action = str(fields.get("action", "HOLD")).upper()
    if action == "MOVE_X_POSITIVE":
        return _VX, 0.0
    if action == "MOVE_Y_POSITIVE":
        return _VX, _YAW
    if action == "MOVE_Y_NEGATIVE":
        return _VX, -_YAW
    return 0.0, 0.0


def decide_action_command(
    image: Image.Image, *, model, processor, instruction: str
) -> tuple[str | None, str]:
    """Ask the action LoRA with exactly the prompt used in its training."""
    from swm.models.semantic_vlm import generate_outcome

    raw = generate_outcome(
        model,
        processor,
        image,
        instruction,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        prompt=build_action_prompt(instruction),
        max_new_tokens=32,
    )
    parsed = parse_action_command(raw)
    if parsed is None:
        return None, raw
    return format_action_command(parsed), raw


def _hold_publisher(
    client: MqttVelocityClient,
    state: dict,
    lock: threading.Lock,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        with lock:
            action_command = state["action_command"]
            task = state["task"]
        if not action_command:
            stop.wait(0.1)
            continue
        vx, yaw = _velocity_for_action(action_command)
        client.publish_velocity(
            vx,
            0.0,
            yaw,
            action_command=action_command,
            task=task,
        )
        stop.wait(0.1)


def run_loop(
    *,
    frame_path: Path,
    session_env: Path,
    checkpoint: Path,
    task_note: str,
) -> None:
    if not checkpoint.exists():
        raise SystemExit(f"LoRA checkpoint not found: {checkpoint}")
    from swm.models.semantic_vlm import load_finetuned
    from swm.utils.config import load_config

    cfg = load_config(str(REPO_ROOT / "configs" / "superpod_action.yaml"))
    print(f"Loading Qwen3-VL-2B general action LoRA from {checkpoint}")
    model, processor = load_finetuned(cfg, str(checkpoint))

    session = resolve_session(session_env=session_env)
    print("Mode: general action LoRA -> MicroDuck action adapter")
    print(f"Task: {task_note}")
    print(f"Publishing to {session.topic} on {session.host}:{session.port}")
    print(f"Waiting for the overhead circle view at {frame_path}")

    state = {"action_command": None, "task": task_note}
    lock = threading.Lock()
    stop = threading.Event()
    seen_mtime: float | None = None

    with MqttVelocityClient(session, dry_run=False) as client:
        thread = threading.Thread(
            target=_hold_publisher, args=(client, state, lock, stop), daemon=True
        )
        thread.start()
        try:
            while True:
                if not frame_path.is_file():
                    time.sleep(0.2)
                    continue
                mtime = frame_path.stat().st_mtime
                if seen_mtime is not None and mtime <= seen_mtime:
                    time.sleep(0.05)
                    continue
                try:
                    with Image.open(frame_path) as img:
                        image = img.convert("RGB").copy()
                except OSError:
                    time.sleep(0.05)
                    continue
                seen_mtime = mtime
                action_command, raw = decide_action_command(
                    image,
                    model=model,
                    processor=processor,
                    instruction=task_note,
                )
                print(
                    f"General action LoRA reply: {raw!r} -> {action_command}",
                    flush=True,
                )
                if action_command is None:
                    continue
                with lock:
                    state["action_command"] = action_command
        finally:
            with lock:
                # Release the duck when this process exits.
                state["action_command"] = (
                    "target_visible=no; action=HOLD; magnitude=none; gripper=hold"
                )
            time.sleep(0.3)
            stop.set()
            thread.join(timeout=2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--session-env", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--task",
        default="keep the orange ball inside the green circle",
        help="Instruction given to the LoRA and carried to the MicroDuck adapter.",
    )
    args = parser.parse_args()
    run_loop(
        frame_path=args.frame,
        session_env=args.session_env,
        checkpoint=args.checkpoint,
        task_note=args.task,
    )


if __name__ == "__main__":
    main()
