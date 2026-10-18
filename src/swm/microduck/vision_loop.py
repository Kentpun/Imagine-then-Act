"""Read the overhead circle camera and tell the duck to hold or push the ball in.

The simulator writes `.demo/vision/zone.ppm`. This process looks at each new
frame and publishes an instruction. The walking policy only supplies the gait.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from swm.microduck.mqtt_control import MqttVelocityClient, resolve_session
from swm.microduck.vision_actions import (
    CIRCLE_PROMPT,
    DUCK_SYSTEM_INSTRUCTION,
    parse_circle_instruction,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRAME = (
    REPO_ROOT.parent / "microduck_demo_part1" / ".demo" / "vision" / "zone.ppm"
)
SIM_SRC = REPO_ROOT.parent / "microduck_demo_part1" / "src"
DEFAULT_SESSION = REPO_ROOT / ".demo" / "session.env"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "superpod" / "checkpoint-final"


def decide_instruction(
    image: Image.Image, *, model, processor, use_vlm: bool
) -> tuple[str, str]:
    """Return hold or push_in, plus the text the decision came from."""
    if use_vlm and model is not None:
        from swm.models.semantic_vlm import generate_outcome

        raw = generate_outcome(
            model,
            processor,
            image,
            "keep the orange ball inside the green circle",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            prompt=CIRCLE_PROMPT,
            system=DUCK_SYSTEM_INSTRUCTION,
            max_new_tokens=16,
        )
        return parse_circle_instruction(raw), raw
    if str(SIM_SRC) not in sys.path:
        sys.path.insert(0, str(SIM_SRC))
    from microduck_demo.circle_instruction import instruction_from_rgb

    word = instruction_from_rgb(np.asarray(image.convert("RGB")))
    if word not in ("hold", "push_in"):
        return "hold", "zone camera could not see the circle"
    return word, "zone camera"


def _hold_publisher(
    client: MqttVelocityClient,
    state: dict,
    lock: threading.Lock,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        with lock:
            instruction = state["instruction"]
        client.publish_velocity(0.0, 0.0, 0.0, instruction=instruction)
        stop.wait(0.1)


def run_loop(
    *,
    frame_path: Path,
    session_env: Path,
    checkpoint: Path | None,
    use_vlm: bool,
    task_note: str,
) -> None:
    model = processor = None
    if use_vlm:
        if checkpoint is None or not checkpoint.exists():
            raise SystemExit(
                f"Checkpoint not found: {checkpoint}. Omit --vlm for the real-time camera tracker."
            )
        from swm.models.semantic_vlm import load_finetuned
        from swm.utils.config import load_config

        cfg = load_config(str(REPO_ROOT / "configs" / "microduck_demo.yaml"))
        print(f"Loading LoRA from {checkpoint}")
        model, processor = load_finetuned(cfg, str(checkpoint))

    session = resolve_session(session_env=session_env)
    mode = "slow VLM" if use_vlm else "real-time camera: keep the ball inside the circle"
    print(f"Mode: {mode}")
    print(f"Task: {task_note}")
    print(f"Publishing to {session.topic} on {session.host}:{session.port}")
    print(f"Waiting for the overhead circle view at {frame_path}")

    state = {"instruction": "hold"}
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
                instruction, raw = decide_instruction(
                    image, model=model, processor=processor, use_vlm=use_vlm
                )
                with lock:
                    state["instruction"] = instruction
                print(f"VLA instruction: {instruction}  ({raw})", flush=True)
        finally:
            with lock:
                state["instruction"] = "hold"
            time.sleep(0.3)
            stop.set()
            thread.join(timeout=2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--session-env", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Use the LoRA VLM. This is too slow for a live walk on this Mac.",
    )
    args = parser.parse_args()
    run_loop(
        frame_path=args.frame,
        session_env=args.session_env,
        checkpoint=args.checkpoint if args.vlm else None,
        use_vlm=args.vlm,
        task_note="keep the orange ball inside the green circle",
    )


if __name__ == "__main__":
    main()
