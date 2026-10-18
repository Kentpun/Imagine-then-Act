# How the LoRA controls MicroDuck

The trained checkpoint does not move the duck’s joints. It chooses a walking speed. A separate simulator turns that speed into footsteps.

```text
camera frame + task
        │
        ▼
Qwen3-VL + your LoRA          outputs/superpod/checkpoint-final
  imagines each candidate     (trained on LIBERO tabletop labels)
        │
        ▼
highest score wins            {vx, vy, yaw}
        │
        ▼
MQTT publish                  topic microduck/demo/<session>/cmd/velocity
        │
        ▼
microduck_demo_part1          MuJoCo body + BEST_alpha_walking.onnx
        │
        ▼
the duck walks
```

Three programs have to agree on the same session id. If they do not, messages go to a topic nobody is listening on.

| Program | Job | Where |
| --- | --- | --- |
| This repo’s Streamlit UI | Rank velocities, publish the winner | `semantic-world-model` |
| Public MQTT broker | Carry the JSON | `broker.emqx.io:1883` |
| MicroDuck simulator | Walk from that velocity | `../microduck_demo_part1` |

The browser page at https://huggingface.co/spaces/pollen-robotics/microduck-simulator is a different viewer. It does not subscribe to your MQTT topic, so the LoRA cannot drive it.

---

## What the LoRA is doing

The SuperPOD adapter was trained to finish one line of tabletop labels:

```text
moved=yes; lifted=no; gripper=close; progress=up; success_near=no
```

For each legal speed, the planner asks that same question. It maps `vx` onto the arm’s `dx` and `yaw` onto `dy`, then scores the answer:

| Predicted field | Score |
| --- | --- |
| `progress=up` | +2 |
| `moved=yes` | +1.5 |
| `success_near=yes` | +2 |
| `progress=down` | −1 |
| `lifted=yes` | +0.25 |

The highest score is the command that gets published. This is a transfer of the LIBERO world model onto velocity choices. It is not a walking policy trained on MicroDuck.

The mock toggle skips the LoRA and uses a fixed heuristic instead. Use that only to check the UI. Live control should have **Mock world model** off.

## What the simulator is doing

`microduck_demo_part1` loads `BEST_alpha_walking.onnx` and a MuJoCo duck. It subscribes to:

```text
microduck/demo/<MICRODUCK_SESSION_ID>/cmd/velocity
```

A legal payload is:

```json
{"vx": 0.2, "vy": 0.0, "yaw": 0.5}
```

Limits from the walking policy:

- `vx` is between 0 and 0.25 (no reverse)
- `vy` is always 0 (no sideways step)
- `yaw` is between −0.8 and 0.8, and only while `vx` is greater than 0 (no turn in place)

The UI publishes that JSON about 10 times a second for a few seconds, then sends a stop. If messages stop, the simulator zeros the velocity after about one second.

MQTT never contains joint targets. The ONNX policy is what moves the legs.

---

## Camera loop: walk to the orange ball

This mode puts two objects in the room and lets the model choose a walk from the duck’s own camera.

```text
MuJoCo scene
  orange ball (ahead, to the left) and a blue box (ahead, to the right)
        │
        ▼
head_camera rendered to microduck_demo_part1/.demo/vision/frame.ppm
        │
        ▼
Real-time tracker reads the head camera (orange = ball, green = circle)
        │
        ▼
walk until the ball is at the feet, then kick, then find it again
        │
        ▼
MQTT {vx, yaw}  →  ONNX gait walks
```

The privileged `--auto-ball` controller is not used here. It reads the ball’s pose from the simulator. This loop does not.

Stop any simulator that is already running, then start this scene:

```bash
cd "../microduck_demo_part1"
./scripts/run_simulator.sh --vision-ball
```

The terminal should print `VISION: head_camera frames -> .../.demo/vision/frame.ppm`. The window shows the duck, an orange ball, and a blue box.

In a second terminal, start the real-time tracker. It does not load Qwen. The head camera steers the duck onto the orange ball inside the green circle. When the ball is large and centered, it sends one kick, waits, then looks for the ball again.

```bash
cd semantic-world-model
bash scripts/run_microduck_vision.sh
```

`--vlm` runs the LoRA on each frame instead. On this Mac that decision takes minutes, so the duck cannot walk in real time that way. The measured LoRA result remains the LIBERO eval.

---

## Setup on this machine

The simulator already lives next to this repo:

```text
_02. Project/microduck_demo_part1
_02. Project/semantic-world-model
```

From `semantic-world-model`, finish the install (Python 3.12 via `uv`, walking-policy download, shared session file):

```bash
bash scripts/setup_microduck_sim.sh
```

That script installs `uv` if it is missing, runs the simulator’s `bootstrap.sh`, and links:

```text
semantic-world-model/.demo/session.env
    → microduck_demo_part1/.demo/session.env
```

Both sides then use the same `MICRODUCK_SESSION_ID`.

---

## Run it

**Terminal 1 — the duck.** A MuJoCo window opens. Leave this running.

```bash
cd "../microduck_demo_part1"
./scripts/run_simulator.sh
```

**Terminal 2 — the LoRA.** First launch downloads `Qwen/Qwen3-VL-2B-Instruct` (a few GB). The adapter itself is already in `outputs/superpod/checkpoint-final`.

```bash
cd semantic-world-model
bash scripts/run_microduck_demo.sh --checkpoint outputs/superpod/checkpoint-final
```

Open the URL Streamlit prints (port 8502). Then:

1. Leave **Mock world model** off.
2. Turn **Enable MQTT publish** on.
3. Turn **Dry-run** off.
4. Confirm **session.env path** is `.demo/session.env`. Leave **Session ID** blank so the file’s id is used.
5. Click **Run imagination planner**, then **Execute best on MicroDuck**.

The duck in Terminal 1 should walk for the stream duration (default 3 seconds) and then stop.

A preset without the planner (same session file):

```bash
python scripts/microduck_send.py forward --duration 3
```

Dry-run prints the payload and does not touch the broker:

```bash
python scripts/microduck_send.py forward --duration 3 --dry-run
```

---

## If the duck does not move

- Terminal 1 must already be on the MuJoCo window before you press Execute.
- Dry-run must be off, and MQTT must be on.
- Session ID in the sidebar must be blank, or it must match `MICRODUCK_SESSION_ID` in `.demo/session.env`. A typed id that differs from the simulator sends commands into the void.
- Broker host `broker.emqx.io` and port `1883` must match the session file. That broker is public and unencrypted; it is only for this demo.
- The Mac must be able to open a GUI window. `run_simulator.sh` uses `mjpython` on macOS.

---

## What to claim in the report

The LoRA selects a velocity intent. MicroDuck walks because the ONNX gait executes that intent. Part A (LIBERO exact-match, 38.9% zero-shot vs 91.8% fine-tuned on 512 frames) is still the measured result. Part B is the interactive control demo.
