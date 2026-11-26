#!/usr/bin/env bash
# Genuine MicroDuck Imagine-then-Act demo (NO CircleKeeper, paper-faithful grid + algebraic scoring,
# pauses duck while LoRA thinks).
#
# Terminal 1 (MuJoCo body, randomized spawn, CircleKeeper OFF):
#   cd ../microduck_demo_part1 && ./scripts/run_raw_simulator.sh walk_ab
#
# Terminal 2 (LoRA world-model brain):
#   cd ../semantic-world-model && bash scripts/run_microduck_raw_demo.sh walk_ab
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

PY_BIN="python3"
if [[ -x ".venv_mac/bin/python" ]]; then
  PY_BIN=".venv_mac/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi

PRESET="${1:-push_ab}"
shift || true

CKPT="${CHECKPOINT:-outputs/microduck_semantic/checkpoint-final}"
if [[ ! -d "$CKPT" ]]; then
  CKPT="outputs/superpod/checkpoint-final"
  echo "MicroDuck LoRA missing; falling back to LIBERO LoRA at $CKPT"
fi

CONFIG="${CONFIG:-configs/microduck_demo_raw.yaml}"

DET_ARGS=()
if [[ -f outputs/microduck_detector/best.pt ]]; then
  DET_ARGS=(--detector-weights outputs/microduck_detector/best.pt)
fi

exec "$PY_BIN" -m swm.microduck.vision_loop \
  --mode imagine \
  --raw-control \
  --step-and-think \
  --task-preset "$PRESET" \
  --checkpoint "$CKPT" \
  --config "$CONFIG" \
  "${DET_ARGS[@]}" \
  "$@"
