# Part B — MicroDuck Imagination + MQTT Control (optional)

**Status:** Stretch / demo track — does **not** replace Part A (LIBERO).  
**Control:** MQTT velocity intents → [microduck_demo_part1](https://github.com/hjianbo/microduck_demo_part1) MuJoCo + ONNX gait.  
**Hardware:** Not required. Real MicroDuck / Device Agent remain out of scope.

---

## 1. Why a separate part?

Part A learns semantic dynamics on **tabletop arm** data (LIBERO).  
[MicroDuck](https://microduck.homes/docs) is a **~25 cm biped**. Control uses the same *Imagine-then-Act* idea, then publishes `{vx, vy, yaw}` over MQTT. The **walking policy is still ONNX RL** — we do not replace the gait with Qwen.

| | Part A (core) | Part B (this doc) |
| --- | --- | --- |
| Body | Tabletop manipulator | MicroDuck biped |
| Data | `lerobot/libero` | Smoke fixtures (+ optional sim captures) |
| Actions | `delta_xyz` + gripper | `vx, vy, yaw` (EMQX-legal) |
| Actuation | Offline ranking only | MQTT → MuJoCo sim |
| Success bar | +10 pts EM vs zero-shot | Demo: duck walks on best intent |

---

## 2. Architecture

```text
Streamlit / CLI  →  Imagination planner  →  MQTT JSON {vx,vy,yaw}
                                              ↓
                                    EMQX broker (e.g. broker.emqx.io)
                                              ↓
                         microduck_demo_part1 adapter + ONNX + MuJoCo
```

Protocol (EMQX Part 1):

| Field | Topic / value |
| --- | --- |
| Broker | `broker.emqx.io:1883` (demo; public) |
| Topic | `microduck/demo/{sessionId}/cmd/velocity` |
| Payload | `{"vx": 0–0.25, "vy": 0.0, "yaw": -0.8–0.8}` |
| Rules | `vy=0`; non-zero `yaw` requires `vx > 0` |

---

## 3. Semantic label schema (biped)

```text
upright=yes; progressed=yes; turned=left; fell=no; goal_near=no
```

---

## 4. Demo modes

### B0 — Browser simulator (zero install)

https://huggingface.co/spaces/pollen-robotics/microduck-simulator

### B1 — Streamlit imagination UI

```bash
python scripts/prepare_microduck_smoke.py
bash scripts/run_microduck_demo.sh --mock
```

### B1c — MQTT control (Imagine → Act)

**Terminal 1** (desktop that can open MuJoCo; Python 3.12 + uv):

```bash
git clone --recurse-submodules https://github.com/hjianbo/microduck_demo_part1.git
cd microduck_demo_part1
./scripts/bootstrap.sh
./scripts/run_simulator.sh
# note SESSION_ID in .demo/session.env
```

**Terminal 2** (this repo):

```bash
# copy or symlink their session file
cp /path/to/microduck_demo_part1/.demo/session.env .demo/session.env

# dry-run (no broker)
python scripts/microduck_send.py forward --duration 3 --dry-run

# live teleop
python scripts/microduck_send.py forward --duration 3
python scripts/microduck_send.py forward-left --duration 3

# Streamlit: enable MQTT, unset dry-run, Rank → Execute best
bash scripts/run_microduck_demo.sh --mock
```

### B2 — Local captures + optional LoRA (stretch)

Capture RGB + commands from sim → Part B JSONL → dedicated LoRA. Not required for P0.

### B3 — Real hardware / Device Agent

**Out of scope** for this course pass.

---

## 5. Planner scoring

```text
score = 2·progressed + 1.5·upright − 3·fell + 0.5·|turned≠none| + 3·goal_near
```

Candidates are filtered to EMQX-legal velocities in `swm.microduck.planner.candidate_actions`.

---

## 6. Repo map (Part B)

```text
docs/MICRODUCK_DEMO.md
configs/microduck_demo.yaml
scripts/prepare_microduck_smoke.py
scripts/run_microduck_demo.sh
scripts/microduck_send.py
src/swm/microduck/
  mqtt_control.py      ← clamp, session.env, publish / stream
  prompts.py
  labeling.py
  planner.py
  demo_app.py
```

---

## 7. Honest claims for the report

- We publish **movement intents**; MicroDuck **walks via the ONNX policy**.  
- Part B is **not** a full VLA replacement for gait learning.  
- LIBERO metrics remain the primary scientific result; MQTT control is the interactive demo bridge.
