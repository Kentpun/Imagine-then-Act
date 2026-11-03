#!/usr/bin/env bash
# Run Layer A ablations A1–A3 (eval-time) and Layer B planner (+ A5 horizons).
# A4 (fewer label fields) needs a separate retrain — skipped here.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

CONFIG="${1:-configs/default.yaml}"
CKPT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  CKPT_ARGS=(--checkpoint "$CHECKPOINT")
fi

echo "=== A1: baseline Layer A (zero-shot vs LoRA) ==="
python -m swm.eval.evaluate --config "$CONFIG" "${CKPT_ARGS[@]}" --split val

echo "=== A2: shuffle action channels ==="
python -m swm.eval.evaluate --config "$CONFIG" "${CKPT_ARGS[@]}" --split val \
  --shuffle_action --finetuned_only

echo "=== A3: blank image (text-only) ==="
python -m swm.eval.evaluate --config "$CONFIG" "${CKPT_ARGS[@]}" --split val \
  --blank_image --finetuned_only

echo "=== A5 + Layer B: planner horizons 1 and 2 ==="
python -m swm.eval.planner_eval --config "$CONFIG" "${CKPT_ARGS[@]}" --split val \
  --horizons 1,2

echo "Done. Metrics are under paths.output_dir from $CONFIG"
