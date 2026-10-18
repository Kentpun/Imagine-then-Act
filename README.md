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

# Optional Part B — LoRA picks a velocity; the simulator walks the duck.
# How it works: docs/MICRODUCK_DEMO.md
bash scripts/setup_microduck_sim.sh
# Terminal 1: cd ../microduck_demo_part1 && ./scripts/run_simulator.sh
# Terminal 2:
bash scripts/run_microduck_demo.sh --checkpoint outputs/superpod/checkpoint-final
# Camera loop (orange ball + blue box, head camera -> walk command):
#   Terminal 1: cd ../microduck_demo_part1 && ./scripts/run_simulator.sh --vision-ball
#   Terminal 2: bash scripts/run_microduck_vision.sh
```

> Install PyTorch for **your** CUDA version from https://pytorch.org before `requirements.txt` if needed.

## Windows (Part A training)

Open **cmd.exe**, `cd` into the `semantic-world-model` folder (needs NVIDIA GPU + CUDA PyTorch).  
Do **not** paste markdown comments into cmd — only run the `.bat` lines below.

```bat
scripts\setup_windows.bat INSTALL_REQ
```

Install CUDA PyTorch from https://pytorch.org into the same venv if the default CPU torch is not enough, then:

```bat
scripts\train_pipeline_windows.bat
```

Smoke-train (sanity check):

```bat
scripts\train.bat configs\windows_quick.yaml
```

**8 GB VRAM** (use this — `low_vram.yaml` needs ~16 GB):

```bat
scripts\train.bat configs\8gb.yaml
```

16 GB VRAM:

```bat
scripts\train.bat configs\low_vram.yaml
```

## HKUST SuperPOD (H800 80 GB)

Login nodes have no GPU. Edit `--account` in `scripts/superpod_train.sbatch`, then on a login node:

```bash
sbatch scripts/superpod_train.sbatch
squeue -u "$USER"
```

Partition `normal`, 1× H800 ([ITSO SuperPOD](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod)). Config: `configs/superpod.yaml`.

Step-by-step:

```bat
scripts\download_data.bat
scripts\prepare_dataset.bat
scripts\train.bat
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
