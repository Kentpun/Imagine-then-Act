"""MicroDuck camera loop: Imagine-then-Act (default) or action-LoRA mode.

**imagine** (default, matches the project title):
  For each new overhead frame, score candidate (vx, yaw[/kick]) commands by asking
  the semantic LoRA world model what outcome occurs next, then publish the winning
  command over MQTT.
  - With ``--raw-control`` (or ``planner.raw_control: true`` in config), ``action_command``
    is omitted so MuJoCo ``CircleKeeper`` is completely bypassed and raw ``(vx, yaw)``
    drives the ONNX walking policy directly without velocity snapping.
  - With ``--step-and-think`` (or ``planner.step_and_think: true``), the duck pauses
    (``vx=0, yaw=0``) before each planning step so it never walks blindly while the
    VLM evaluates candidates.

**action** (legacy / exploratory):
  Single forward decode from an action-command LoRA (MOVE_*), then map to
  velocity. No counterfactual imagination.
"""

from __future__ import annotations

import argparse
import io
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
_PART1_SESSION = REPO_ROOT.parent / "microduck_demo_part1" / ".demo" / "session.env"
DEFAULT_SESSION = (
    _PART1_SESSION if _PART1_SESSION.is_file() else REPO_ROOT / ".demo" / "session.env"
)
DEFAULT_ACTION_CKPT = REPO_ROOT / "outputs" / "superpod_action_v3" / "checkpoint-final"
DEFAULT_SEMANTIC_CKPT = REPO_ROOT / "outputs" / "superpod" / "checkpoint-final"

_VX = 0.25
_YAW = 0.50


def _action_command_for_velocity(vx: float, yaw: float) -> str:
    """Legacy CircleKeeper VLA one-liner (used only when raw_control=False)."""
    if abs(vx) <= 1e-6:
        return "target_visible=yes; action=HOLD; magnitude=none; gripper=hold"
    if yaw > 0.2:
        return "target_visible=yes; action=MOVE_Y_POSITIVE; magnitude=medium; gripper=hold"
    if yaw < -0.2:
        return "target_visible=yes; action=MOVE_Y_NEGATIVE; magnitude=medium; gripper=hold"
    return "target_visible=yes; action=MOVE_X_POSITIVE; magnitude=medium; gripper=hold"


def _demo_velocity(vx: float, yaw: float) -> tuple[float, float]:
    """Legacy speed snap (used only when raw_control=False)."""
    if abs(vx) <= 1e-6:
        return 0.0, 0.0
    if abs(yaw) > 0.2:
        return _VX, _YAW if yaw > 0 else -_YAW
    return _VX, 0.0


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
            kick = state.pop("pending_kick", None)
        client.publish_velocity(
            vx,
            0.0,
            yaw,
            kick=kick,
            action_command=action_command,
            task=task,
        )
        stop.wait(0.1)


def _watch_frames(frame_path: Path, seen_mtime: float | None) -> tuple[Image.Image | None, float | None]:
    if not frame_path.is_file():
        return None, seen_mtime
    try:
        mtime = frame_path.stat().st_mtime
        if seen_mtime is not None and mtime <= seen_mtime:
            return None, seen_mtime
        # Read into memory immediately so Windows NTFS never holds a file lock on zone.ppm
        raw_bytes = frame_path.read_bytes()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            return img.convert("RGB").copy(), mtime
    except OSError:
        return None, seen_mtime


def _check_checkpoint_vintage(checkpoint: Path) -> None:
    """Warn if local adapter_model.safetensors predates the Regime-2 SuperPOD run."""
    weights = checkpoint / "adapter_model.safetensors"
    hist = checkpoint.parent / "train_history.json"
    if weights.is_file() and hist.is_file():
        if weights.stat().st_mtime + 3600 < hist.stat().st_mtime:
            print(
                "\n[NOTE] Local adapter_model.safetensors is older than train_history.json "
                "(likely Regime-1 pure-expert weights, which tie on progress=up).\n"
                "       Connect to HKUST VPN and run `bash scripts/sync_regime2_checkpoint.sh` "
                "to pull the Regime-2 mixed-play weights from SuperPOD.\n",
                flush=True,
            )


def run_imagine_loop(
    *,
    frame_path: Path,
    session_env: Path,
    checkpoint: Path,
    config_path: Path,
    task_note: str,
    detector_weights: Path | None = None,
    hud_path: Path | None = None,
    replan_every_s: float | None = None,
    raw_control: bool | None = None,
    step_and_think: bool | None = None,
    act_duration_s: float | None = None,
    four_bit: bool = False,
    allow_hold: bool = True,
) -> None:
    """Imagine-then-Act: rank velocity candidates with the semantic world model."""
    from swm.microduck.detector import DuckDetector, draw_detection
    from swm.microduck.planner import MicroDuckImaginationPlanner
    from swm.models.semantic_vlm import load_finetuned
    from swm.utils.config import load_config

    if not checkpoint.exists():
        raise SystemExit(f"Semantic checkpoint not found: {checkpoint}")
    _check_checkpoint_vintage(checkpoint)

    cfg = load_config(str(config_path))
    if four_bit:
        cfg.setdefault("model", {})["load_in_4bit"] = True
        cfg.setdefault("planner", {})["merge_lora"] = False
    plan_cfg = cfg.setdefault("planner", {})
    if not (("kick" in task_note.lower()) or ("goal" in task_note.lower())):
        plan_cfg["include_kick"] = False
    if not allow_hold:
        non_zero_vx = [float(v) for v in plan_cfg.get("vx_values", [0.25]) if float(v) > 1e-6]
        plan_cfg["vx_values"] = non_zero_vx or [0.25]
    if replan_every_s is None:
        replan_every_s = float(plan_cfg.get("replan_every_s", 4.0))
    if raw_control is None:
        raw_control = bool(plan_cfg.get("raw_control", False))
    if step_and_think is None:
        step_and_think = bool(plan_cfg.get("step_and_think", raw_control))
    if act_duration_s is None:
        act_duration_s = float(plan_cfg.get("act_duration_s", 1.6))
    pause_before_plan_s = float(plan_cfg.get("pause_before_plan_s", 0.40))

    print(f"Loading semantic world model (Imagine-then-Act) from {checkpoint}")
    model, processor = load_finetuned(cfg, str(checkpoint))
    planner = MicroDuckImaginationPlanner(cfg, model=model, processor=processor, mock=False)
    detector = DuckDetector(detector_weights)

    session = resolve_session(session_env=session_env)
    print("Mode: IMAGINE candidate (vx,yaw[/kick]) outcomes -> ACT best command")
    print(f"Task: {task_note}")
    print(
        f"Planner mode={planner.mode} score={planner.score_mode}; "
        f"raw_control={raw_control} (CircleKeeper={'OFF' if raw_control else 'ON'}); "
        f"step_and_think={step_and_think}"
    )
    if step_and_think:
        print(
            f"Cadence: PAUSE ({pause_before_plan_s:.2f}s) -> IMAGINE -> "
            f"ACT ({act_duration_s:.2f}s burst)"
        )
    else:
        print(f"Replan every: {replan_every_s:.1f}s (keeps last ACT between ranks)")
    print(f"Publishing to {session.topic} on {session.host}:{session.port}")
    print(f"Waiting for frames at {frame_path}")

    state: dict = {
        "vx": 0.0,
        "yaw": 0.0,
        "task": None if raw_control else task_note,
        "action_command": None,
        "pending_kick": None,
    }
    lock = threading.Lock()
    stop = threading.Event()
    seen_mtime: float | None = None
    last_plan_t = 0.0

    with MqttVelocityClient(session, dry_run=False) as client:
        thread = threading.Thread(
            target=_velocity_publisher, args=(client, state, lock, stop), daemon=True
        )
        thread.start()
        try:
            while True:
                if step_and_think:
                    # 1. Pause duck completely while LoRA evaluates so it never drifts blindly
                    with lock:
                        was_moving = abs(float(state["vx"])) > 1e-6 or abs(float(state["yaw"])) > 1e-6
                        state["vx"] = 0.0
                        state["yaw"] = 0.0
                        state["action_command"] = None
                    if was_moving:
                        print("=== PAUSE === stopping duck (vx=0.00, yaw=0.00) before observing frame", flush=True)
                        time.sleep(pause_before_plan_s)

                    # Grab newest camera frame rendered after pause
                    image = None
                    wait_deadline = time.monotonic() + 2.0
                    while time.monotonic() < wait_deadline:
                        cand, seen_mtime = _watch_frames(frame_path, seen_mtime)
                        if cand is not None:
                            image = cand
                            break
                        time.sleep(0.05)
                    if image is None and frame_path.is_file():
                        try:
                            with Image.open(frame_path) as img:
                                image = img.convert("RGB").copy()
                        except OSError:
                            image = None
                    if image is None:
                        time.sleep(0.1)
                        continue
                else:
                    image, seen_mtime = _watch_frames(frame_path, seen_mtime)
                    if image is None:
                        time.sleep(0.05)
                        continue
                    now = time.monotonic()
                    if last_plan_t > 0.0 and (now - last_plan_t) < replan_every_s:
                        time.sleep(0.05)
                        continue

                det = detector.best(image)
                if hud_path is not None:
                    draw_detection(image, det).save(hud_path)
                print("=== CCTV / DETECT ===", flush=True)
                if det:
                    print(
                        f"  duck box cx={det.cx:.2f} cy={det.cy:.2f} "
                        f"score={det.score:.2f} label={det.label}",
                        flush=True,
                    )
                else:
                    print("  duck not detected (YOLO/heuristic HUD miss; VLM still sees full frame)", flush=True)

                t_rank0 = time.perf_counter()
                ranked = planner.rank(image, task_note, progress=True)
                dt_rank = time.perf_counter() - t_rank0
                print(
                    f"=== IMAGINE ({dt_rank:.2f}s) === ranked {len(ranked)} candidates ===",
                    flush=True,
                )
                for i, item in enumerate(ranked):
                    a = item.action
                    label = "KICK" if abs(float(a[2]) - 9.0) < 1e-6 else f"vx={a[0]:.2f} yaw={a[2]:+.2f}"
                    print(
                        f"  [{i+1}] score={item.score:+.2f}  {label}  "
                        f"imagined={item.outcome!r}",
                        flush=True,
                    )
                best = ranked[0]
                is_kick = abs(float(best.action[2]) - 9.0) < 1e-6
                if is_kick:
                    vx, yaw = 0.0, 0.0
                    action_command = None if raw_control else "skill=kick"
                    pending_kick = "left"
                    print(f"=== ACT (RAW) === KICK score={best.score:+.2f}", flush=True)
                elif raw_control:
                    vx = max(0.0, float(best.action[0]))
                    yaw = float(best.action[2]) if vx > 1e-6 else 0.0
                    action_command = None
                    pending_kick = None
                    print(
                        f"=== ACT (RAW MQTT, no CircleKeeper) === "
                        f"vx={vx:.2f} yaw={yaw:+.2f} score={best.score:+.2f}",
                        flush=True,
                    )
                else:
                    raw_vx = float(best.action[0])
                    raw_yaw = float(best.action[2])
                    vx, yaw = _demo_velocity(raw_vx, raw_yaw)
                    action_command = _action_command_for_velocity(vx, yaw)
                    pending_kick = None
                    print(
                        f"=== ACT === vx={vx:.2f} yaw={yaw:+.2f} score={best.score:.2f} "
                        f"cmd={action_command}",
                        flush=True,
                    )

                with lock:
                    state["vx"] = vx
                    state["yaw"] = yaw
                    state["action_command"] = action_command
                    if pending_kick:
                        state["pending_kick"] = pending_kick

                last_plan_t = time.monotonic()
                if step_and_think:
                    time.sleep(0.8 if is_kick else act_duration_s)
        finally:
            with lock:
                state["vx"] = 0.0
                state["yaw"] = 0.0
                state["action_command"] = None
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
        "pending_kick": None,
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
        help="Defaults: semantic MicroDuck LoRA for imagine; action_v3 for action mode",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "microduck_demo.yaml",
        help="Config for imagine mode (use configs/microduck_demo_raw.yaml for no-CircleKeeper demo).",
    )
    parser.add_argument(
        "--raw-control",
        action="store_true",
        default=None,
        help="Bypass CircleKeeper and velocity snapping; publish exact (vx,yaw[/kick]).",
    )
    parser.add_argument(
        "--step-and-think",
        action="store_true",
        default=None,
        help="Pause the duck (vx=0, yaw=0) during each VLM evaluation step.",
    )
    parser.add_argument(
        "--act-duration",
        type=float,
        default=None,
        help="Seconds to execute each chosen velocity burst in step-and-think mode.",
    )
    parser.add_argument(
        "--replan-every",
        type=float,
        default=None,
        help="Seconds between full Imagine ranks when not in step-and-think mode.",
    )
    parser.add_argument(
        "--task",
        default="walk from zone A to zone B",
    )
    parser.add_argument(
        "--task-preset",
        choices=("push_ab", "round_trip", "walk_ab", "kick_goal", "keep_zone", "conditional", "shuttle_ab", "custom"),
        default="custom",
        help="Named Part-2 multitask prompts (push_ab: A->B+hold; round_trip: A->B->turnaround->B->A+hold).",
    )
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=REPO_ROOT / "outputs" / "microduck_detector" / "best.pt",
    )
    parser.add_argument(
        "--hud",
        type=Path,
        default=REPO_ROOT.parent
        / "microduck_demo_part1"
        / ".demo"
        / "vision"
        / "hud.jpg",
        help="Optional CCTV overlay image path.",
    )
    parser.add_argument(
        "--4bit",
        dest="four_bit",
        action="store_true",
        help="Load Qwen3-VL in 4-bit NF4 (~2.2 GB VRAM) for tight GPU budgets.",
    )
    args = parser.parse_args()

    from swm.microduck.semantic_tasks import TASK_PROMPTS

    if args.task_preset == "custom":
        task_note = args.task
    elif args.task_preset in ("push_ab", "round_trip", "shuttle_ab"):
        # Active target Goal lights up bright green in MuJoCo (push_ab & round_trip hold once final zone is reached!).
        task_note = TASK_PROMPTS["keep_zone"]
    else:
        task_note = TASK_PROMPTS[args.task_preset]

    if args.mode == "imagine":
        ckpt = args.checkpoint or DEFAULT_SEMANTIC_CKPT
        duck_ckpt = REPO_ROOT / "outputs" / "microduck_semantic" / "checkpoint-final"
        if args.checkpoint is None and duck_ckpt.exists():
            ckpt = duck_ckpt
        run_imagine_loop(
            frame_path=args.frame,
            session_env=args.session_env,
            checkpoint=ckpt,
            config_path=args.config,
            task_note=task_note,
            detector_weights=args.detector_weights if args.detector_weights.is_file() else None,
            hud_path=args.hud,
            replan_every_s=args.replan_every,
            raw_control=args.raw_control,
            step_and_think=args.step_and_think,
            act_duration_s=args.act_duration,
            four_bit=args.four_bit,
            allow_hold=(args.task_preset not in ("shuttle_ab", "round_trip")),
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
