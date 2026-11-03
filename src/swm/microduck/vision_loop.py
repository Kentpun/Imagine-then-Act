"""MicroDuck camera loop: Imagine-then-Act (default) or action-LoRA mode.

**imagine** (default, matches the project title):
  For each new overhead frame, score many (vx, yaw) candidates by asking the
  *semantic* LIBERO LoRA what would happen under each mapped action. Publish
  the best velocity (Imagine → Act).

**action** (legacy / exploratory):
  Single forward decode from an action-command LoRA (MOVE_*), then map to
  velocity. No counterfactual imagination.
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
DEFAULT_ACTION_CKPT = REPO_ROOT / "outputs" / "superpod_action_v3" / "checkpoint-final"
DEFAULT_SEMANTIC_CKPT = REPO_ROOT / "outputs" / "superpod" / "checkpoint-final"

_VX = 0.25
_YAW = 0.50


def _velocity_for_action(action_command: str) -> tuple[float, float]:
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


def _velocity_publisher(
    client: MqttVelocityClient,
    state: dict,
    lock: threading.Lock,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        with lock:
            vx = float(state.get("vx", 0.0))
            yaw = float(state.get("yaw", 0.0))
            task = state.get("task")
            action_command = state.get("action_command")
        client.publish_velocity(
            vx,
            0.0,
            yaw,
            action_command=action_command,
            task=task,
        )
        stop.wait(0.1)


def _watch_frames(frame_path: Path, seen_mtime: float | None) -> tuple[Image.Image | None, float | None]:
    if not frame_path.is_file():
        return None, seen_mtime
    mtime = frame_path.stat().st_mtime
    if seen_mtime is not None and mtime <= seen_mtime:
        return None, seen_mtime
    try:
        with Image.open(frame_path) as img:
            return img.convert("RGB").copy(), mtime
    except OSError:
        return None, seen_mtime


def run_imagine_loop(
    *,
    frame_path: Path,
    session_env: Path,
    checkpoint: Path,
    config_path: Path,
    task_note: str,
) -> None:
    """Imagine-then-Act: rank velocity candidates with the semantic world model."""
    from swm.microduck.planner import MicroDuckImaginationPlanner
    from swm.models.semantic_vlm import load_finetuned
    from swm.utils.config import load_config

    if not checkpoint.exists():
        raise SystemExit(f"Semantic checkpoint not found: {checkpoint}")
    cfg = load_config(str(config_path))
    print(f"Loading semantic world model (Imagine-then-Act) from {checkpoint}")
    model, processor = load_finetuned(cfg, str(checkpoint))
    planner = MicroDuckImaginationPlanner(cfg, model=model, processor=processor, mock=False)

    session = resolve_session(session_env=session_env)
    print("Mode: IMAGINE candidate (vx,yaw) outcomes → ACT best velocity")
    print(f"Task: {task_note}")
    print(f"Candidates: {len(planner.candidates)}")
    print(f"Publishing to {session.topic} on {session.host}:{session.port}")
    print(f"Waiting for frames at {frame_path}")

    state: dict = {"vx": 0.0, "yaw": 0.0, "task": task_note, "action_command": None}
    lock = threading.Lock()
    stop = threading.Event()
    seen_mtime: float | None = None

    with MqttVelocityClient(session, dry_run=False) as client:
        thread = threading.Thread(
            target=_velocity_publisher, args=(client, state, lock, stop), daemon=True
        )
        thread.start()
        try:
            while True:
                image, seen_mtime = _watch_frames(frame_path, seen_mtime)
                if image is None:
                    time.sleep(0.05)
                    continue
                print("=== IMAGINE ===", flush=True)
                ranked = planner.rank(image, task_note)
                for i, item in enumerate(ranked[:5]):
                    a = item.action
                    print(
                        f"  [{i+1}] score={item.score:.2f}  "
                        f"vx={a[0]:.2f} yaw={a[2]:+.2f}  "
                        f"imagined={item.outcome!r}",
                        flush=True,
                    )
                best = ranked[0]
                vx, yaw = float(best.action[0]), float(best.action[2])
                print(
                    f"=== ACT === vx={vx:.2f} yaw={yaw:+.2f} score={best.score:.2f}",
                    flush=True,
                )
                with lock:
                    state["vx"] = vx
                    state["yaw"] = yaw
                    state["action_command"] = (
                        f"imagined_best; vx={vx:.2f}; yaw={yaw:.2f}; "
                        f"outcome={best.outcome[:120]}"
                    )
        finally:
            with lock:
                state["vx"] = 0.0
                state["yaw"] = 0.0
            time.sleep(0.3)
            stop.set()
            thread.join(timeout=2.0)


def run_action_loop(
    *,
    frame_path: Path,
    session_env: Path,
    checkpoint: Path,
    task_note: str,
) -> None:
    """Legacy single-decode action-command LoRA (no imagination)."""
    if not checkpoint.exists():
        raise SystemExit(f"Action LoRA checkpoint not found: {checkpoint}")
    from swm.models.semantic_vlm import load_finetuned
    from swm.utils.config import load_config

    cfg = load_config(str(REPO_ROOT / "configs" / "superpod_action.yaml"))
    print(f"Loading action-command LoRA from {checkpoint}")
    model, processor = load_finetuned(cfg, str(checkpoint))

    session = resolve_session(session_env=session_env)
    print("Mode: action LoRA (single decode, no counterfactual imagination)")
    print(f"Task: {task_note}")
    print(f"Publishing to {session.topic} on {session.host}:{session.port}")

    state: dict = {
        "vx": 0.0,
        "yaw": 0.0,
        "task": task_note,
        "action_command": None,
    }
    lock = threading.Lock()
    stop = threading.Event()
    seen_mtime: float | None = None

    with MqttVelocityClient(session, dry_run=False) as client:
        thread = threading.Thread(
            target=_velocity_publisher, args=(client, state, lock, stop), daemon=True
        )
        thread.start()
        try:
            while True:
                image, seen_mtime = _watch_frames(frame_path, seen_mtime)
                if image is None:
                    time.sleep(0.05)
                    continue
                action_command, raw = decide_action_command(
                    image,
                    model=model,
                    processor=processor,
                    instruction=task_note,
                )
                print(f"Action LoRA: {raw!r} -> {action_command}", flush=True)
                if action_command is None:
                    continue
                vx, yaw = _velocity_for_action(action_command)
                with lock:
                    state["vx"] = vx
                    state["yaw"] = yaw
                    state["action_command"] = action_command
        finally:
            with lock:
                state["vx"] = 0.0
                state["yaw"] = 0.0
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
    parser.add_argument(
        "--mode",
        choices=("imagine", "action"),
        default="imagine",
        help="imagine = counterfactual velocity ranking (default); action = MOVE_* LoRA",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults: semantic superpod for imagine; action_v3 for action mode",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "microduck_demo.yaml",
        help="Config for imagine mode (candidate velocities).",
    )
    parser.add_argument(
        "--task",
        default="keep the orange ball inside the green circle",
    )
    args = parser.parse_args()

    if args.mode == "imagine":
        ckpt = args.checkpoint or DEFAULT_SEMANTIC_CKPT
        run_imagine_loop(
            frame_path=args.frame,
            session_env=args.session_env,
            checkpoint=ckpt,
            config_path=args.config,
            task_note=args.task,
        )
    else:
        ckpt = args.checkpoint or DEFAULT_ACTION_CKPT
        run_action_loop(
            frame_path=args.frame,
            session_env=args.session_env,
            checkpoint=ckpt,
            task_note=args.task,
        )


if __name__ == "__main__":
    main()
