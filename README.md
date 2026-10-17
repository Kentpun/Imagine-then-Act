# Imagine-then-Act — Semantic World Model

Lightweight **semantic world model** for tabletop manipulation: a LoRA-tuned VLM predicts action-conditioned outcomes and ranks candidate actions with an imagination planner.

Designed for a **single GPU** and the public **LIBERO** dataset.

## Quick start

```bash
cd semantic-world-model
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt

# Optional smoke test (no LIBERO download; validates data pipeline)
python scripts/prepare_dataset.py --config configs/smoke.yaml --smoke_test

# Full pipeline: download LIBERO (~1.9 GB) and build labeled transitions
python scripts/download_data.py
python scripts/prepare_dataset.py --config configs/default.yaml

# Train LoRA (single GPU)
bash scripts/train.sh                 # or: bash scripts/train.sh configs/low_vram.yaml

# Evaluate vs zero-shot baseline
bash scripts/eval.sh

# Interactive demo (Part A — LIBERO)
bash scripts/run_demo.sh

# Optional Part B — MicroDuck imagination + MQTT control
python scripts/prepare_microduck_smoke.py
bash scripts/run_microduck_demo.sh --mock
# Teleop dry-run (no broker):
python scripts/microduck_send.py forward --duration 3 --dry-run
# Live control: run hjianbo/microduck_demo_part1 simulator, share session.env, then:
#   python scripts/microduck_send.py forward --duration 3
# Browser twin: https://huggingface.co/spaces/pollen-robotics/microduck-simulator
```

> Install PyTorch for **your** CUDA version from https://pytorch.org before `requirements.txt` if needed.

## Windows (Part A training)

Open **cmd.exe** in `semantic-world-model` (needs NVIDIA GPU + CUDA PyTorch):

```bat
scripts\setup_windows.bat
:: when prompted, install requirements; also install CUDA torch from pytorch.org first

scripts\train_pipeline_windows.bat
:: low VRAM (16 GB):
scripts\train_pipeline_windows.bat configs\low_vram.yaml

:: or step-by-step:
scripts\download_data.bat
scripts\prepare_dataset.bat
scripts\train.bat
scripts\train.bat configs\low_vram.yaml
scripts\eval.bat
```

## What you get

| Module | Path | Role |
| --- | --- | --- |
| Tech spec | `TECH_SPEC.md` | Architecture, labels, training protocol |
| Data prep | `scripts/prepare_dataset.py` | LIBERO → JSONL + frames |
| Trainer | `src/swm/training/train.py` | Qwen3-VL + LoRA |
| Planner | `src/swm/planning/imagination_planner.py` | Score / pick actions |
| Demo (Part A) | `src/swm/demo/app.py` | Streamlit heatmap UI (LIBERO) |
| Demo (Part B) | `src/swm/microduck/` + `docs/MICRODUCK_DEMO.md` | Imagine velocities + MQTT → MuJoCo sim |

## Hardware

| GPU | Mode |
| --- | --- |
| 24 GB (3090/4090/L4) | Default bf16 LoRA |
| 16 GB | Enable `load_in_4bit` + freeze vision |
| Colab T4 (16 GB) | QLoRA, batch 1, lower `max_pixels` |

## Project docs

- Proposal: `../_00. Ideate/Project_Proposal_Imagine_then_Act.md`
- Spec: `TECH_SPEC.md`
- Part B (MicroDuck): `docs/MICRODUCK_DEMO.md`
- Config: `configs/default.yaml` · `configs/microduck_demo.yaml`
