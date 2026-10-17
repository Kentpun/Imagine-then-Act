# Technical Specification  
## Imagine-then-Act: Semantic World Model for Tabletop Manipulation

**Version:** 1.0  
**Hardware target:** Single NVIDIA GPU ≥16 GB VRAM (24 GB preferred)  
**Primary data:** LIBERO via Hugging Face (`lerobot/libero`)

---

## 1. System overview

```
LIBERO episodes
      │
      ▼
prepare_dataset.py ──► JSONL transitions
      │                   {image_path, task, action, labels}
      ▼
train.py (LoRA VLM) ──► checkpoint/
      │
      ├──► evaluate.py  (accuracy / F1 vs zero-shot)
      ├──► imagination_planner.py  (score K actions)
      └──► streamlit demo  (heatmap + ranked actions)
```

**Model:** \(f_\theta(o_t, g, a_t) \rightarrow \hat{y}\)  
where \(\hat{y}\) is a structured semantic outcome string, not pixels.

---

## 2. Semantic label schema

Each training example predicts a **single answer string** built from binary / categorical fields derived from proprioception and episode metadata (no manual annotation):

| Field | Values | Derivation (LIBERO) |
| --- | --- | --- |
| `moved` | yes / no | \(\| \Delta ee_{xy} \| > \tau_{move}\) |
| `lifted` | yes / no | \(\Delta z > \tau_{lift}\) (gripper closed) |
| `gripper` | open / close / hold | gripper action channel |
| `progress` | up / flat / down | distance-to-goal proxy from state delta vs episode end |
| `success_near` | yes / no | within last \(N\) frames of a successful episode |

**Canonical target format:**
```text
moved=yes; lifted=no; gripper=close; progress=up; success_near=no
```

This keeps supervision fully automatic while remaining VQA-compatible for any chat-style VLM.

---

## 3. Model architecture

| Stage | Choice | Notes |
| --- | --- | --- |
| Backbone | `Qwen/Qwen3-VL-2B-Instruct` (default) | Open weights; requires `transformers>=4.57` |
| Adaptation | LoRA on attention / MLP projections | `r=16`, `alpha=32`, dropout `0.05` |
| Precision | bf16 (Ampere+) or fp16 (T4) | Optional 4-bit QLoRA for ≤16 GB |
| Max image tokens | controlled via `max_pixels` | Primary OOM lever |
| Loss | Causal LM cross-entropy on answer tokens only | Prompt tokens masked with `-100` |

**Prompt template:**
```text
<image>
Task: {task_instruction}
Action: delta_xyz=({dx:.3f},{dy:.3f},{dz:.3f}), gripper={open|close}
Predict the semantic outcome after this action.
Answer:
```

---

## 4. Data pipeline

1. Download LIBERO: `python scripts/download_data.py`
2. Build transitions: `python scripts/prepare_dataset.py`
   - Sample frames every `stride` steps (default 5)
   - Write `data/processed/{train,val}.jsonl` + cached JPEG frames
3. Train / val split: by **episode id** (not random frames) to avoid leakage

**Expected scale (default settings):** ~20k–60k transitions from LIBERO — enough for LoRA in a few hours on 24 GB.

---

## 5. Training protocol

| Hyperparameter | Default |
| --- | --- |
| Epochs | 1–3 |
| Batch size | 1–2 |
| Grad accumulation | 8–16 |
| LR | 2e-4 cosine, warmup 3% |
| Max seq length | 512 |
| Gradient checkpointing | on |
| Freeze vision tower | optional (`true` saves VRAM) |

**Success criteria**
- Val token / exact-match accuracy **≥ +10 pts** over frozen zero-shot baseline
- Planner success rate **> random** on ≥2 held-out tasks

---

## 6. Imagination planner

Given \((o_t, g)\):

1. Build candidate set \(A = \{a^{(k)}\}_{k=1}^{K}\) (planar grid + gripper open/close), \(K \in [8,16]\)
2. Batch-score each candidate with \(f_\theta\)
3. Parse `progress` / `success_near` / `lifted` into a scalar score
4. Return \(\arg\max_k \mathrm{score}(a^{(k)})\)
5. Optional: 2–4 step greedy rollout (re-query with previous best)

---

## 7. Directory map

```text
semantic-world-model/
├── TECH_SPEC.md
├── docs/MICRODUCK_DEMO.md   # Part B optional biped demo
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   └── microduck_demo.yaml
├── scripts/
│   ├── download_data.py
│   ├── prepare_dataset.py
│   ├── prepare_microduck_smoke.py
│   ├── microduck_send.py      # MQTT velocity CLI
│   ├── train.sh
│   ├── eval.sh
│   ├── run_demo.sh
│   └── run_microduck_demo.sh
├── src/swm/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── planning/
│   ├── eval/
│   ├── demo/                # Part A LIBERO Streamlit
│   ├── microduck/           # Part B planner + MQTT + Streamlit
│   └── utils/
├── data/                 # downloaded + processed (gitignored)
└── outputs/              # checkpoints + logs
```

---

## 8. Single-GPU runbook

```bash
# 0) env
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt

# 1) data
python scripts/download_data.py --repo_id lerobot/libero
python scripts/prepare_dataset.py --config configs/default.yaml

# 2) train (RTX 3090/4090)
bash scripts/train.sh

# 3) eval + demo
bash scripts/eval.sh
bash scripts/run_demo.sh
```

**VRAM tips if OOM**
1. Use `configs/8gb.yaml` on 8 GB GPUs (`low_vram.yaml` targets ~16 GB)
2. Set `model.load_in_4bit: true`
3. Set `model.freeze_vision: true`
4. Lower `model.max_pixels` (e.g. `50176` or `100352`)
5. `train.batch_size: 1`, raise `grad_accum`; optionally set `train.max_train_samples`

---

## 9. Out of scope (do not implement in v1)

- Pixel video generation
- Full VLA-JEPA multi-GPU pretraining
- Real robot deployment (Franka / MicroDuck hardware)
- Training on full DROID / Open X-Embodiment

---

## 10. Part B — MicroDuck demo + MQTT control (optional)

Separate stretch track. Does not replace LIBERO metrics. Walking remains the **ONNX policy**; we publish velocity **intents**.

| Mode | What | Command / link |
| --- | --- | --- |
| B0 | Browser simulator | https://huggingface.co/spaces/pollen-robotics/microduck-simulator |
| B1 | Streamlit planner (mock or checkpoint) | `bash scripts/run_microduck_demo.sh --mock` |
| B1c | MQTT → MuJoCo (`microduck_demo_part1`) | Rank → Execute best / `scripts/microduck_send.py` |
| B2 | Local sim captures + optional LoRA | See `docs/MICRODUCK_DEMO.md` |

Actions: EMQX-legal `vx∈[0,0.25]`, `vy=0`, `yaw` only with `vx>0`.  
Labels: `upright; progressed; turned; fell; goal_near`.  
Module: `src/swm/microduck/mqtt_control.py`.
