# Part 2 — MicroDuck synthetic VLA + CCTV demo

## Data

1. **Detection (perception only):** [pngwn/microduck-detection-dataset](https://huggingface.co/datasets/pngwn/microduck-detection-dataset)

```bash
python scripts/download_microduck_detection.py
pip install ultralytics   # once
python scripts/train_duck_detector.py --epochs 30
# → outputs/microduck_detector/best.pt
```

2. **Semantic / action logs (required for VLA):**

```bash
python scripts/record_microduck_logs.py --episodes_per_task 60
python scripts/prepare_microduck_semantic.py --config configs/microduck_train.yaml
```

## Train (SuperPOD)

```bash
sbatch scripts/superpod_microduck_train.sbatch
```

## Multitask demo

```bash
# Terminal 1
cd ../microduck_demo_part1 && MICRODUCK_SKIP_VLM=1 ./scripts/run_simulator.sh --vision-ball

# Terminal 2 — presets: walk_ab | kick_goal | keep_zone | conditional
bash scripts/run_microduck_multitask_demo.sh keep_zone
```

## System eval table (kinematic)

```bash
python scripts/eval_microduck_system.py --n 10 \
  --checkpoint outputs/microduck_semantic/checkpoint-final
```
