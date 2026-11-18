#!/usr/bin/env bash
# Part 2 multitask CCTV demo: Imagine-then-Act + duck detector + MQTT.
# Terminal 1: cd ../microduck_demo_part1 && MICRODUCK_SKIP_VLM=1 ./scripts/run_simulator.sh --vision-ball
# Terminal 2: this script
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

PRESET="${1:-keep_zone}"
CKPT="${CHECKPOINT:-outputs/microduck_semantic/checkpoint-final}"
if [[ ! -d "$CKPT" ]]; then
  CKPT="outputs/superpod/checkpoint-final"
  echo "MicroDuck LoRA missing; falling back to LIBERO LoRA at $CKPT"
fi
CONFIG="${CONFIG:-configs/microduck_train.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  CONFIG=configs/microduck_demo.yaml
fi

DET_ARGS=()
if [[ -f outputs/microduck_detector/best.pt ]]; then
  DET_ARGS=(--detector-weights outputs/microduck_detector/best.pt)
fi

python -m swm.microduck.vision_loop \
  --mode imagine \
  --task-preset "$PRESET" \
  --checkpoint "$CKPT" \
  --config "$CONFIG" \
  "${DET_ARGS[@]}"
