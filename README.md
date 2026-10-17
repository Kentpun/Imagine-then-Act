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

# Interactive demo
bash scripts/run_demo.sh
```

> Install PyTorch for **your** CUDA version from https://pytorch.org before `requirements.txt` if needed.

## What you get

| Module | Path | Role |
| --- | --- | --- |
| Tech spec | `TECH_SPEC.md` | Architecture, labels, training protocol |
| Data prep | `scripts/prepare_dataset.py` | LIBERO → JSONL + frames |
| Trainer | `src/swm/training/train.py` | Qwen3-VL + LoRA |
| Planner | `src/swm/planning/imagination_planner.py` | Score / pick actions |
| Demo | `src/swm/demo/app.py` | Streamlit heatmap UI |

## Hardware

| GPU | Mode |
| --- | --- |
| 24 GB (3090/4090/L4) | Default bf16 LoRA |
| 16 GB | Enable `load_in_4bit` + freeze vision |
| Colab T4 (16 GB) | QLoRA, batch 1, lower `max_pixels` |

## Project docs

- Proposal: `../Project_Proposal_Semantic_World_Model.md`
- Spec: `TECH_SPEC.md`
- Config: `configs/default.yaml`
