#!/usr/bin/env bash
# Layer A ablations A1–A3 + A6, and Layer B planner (+ A5 horizons).
# A4 (fewer label fields) needs prepare + retrain — see superpod_a4.sbatch.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

CONFIG="${1:-configs/default.yaml}"
CKPT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  CKPT_ARGS=(--checkpoint "$CHECKPOINT")
fi

MAX_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_ARGS=(--max_samples "$MAX_SAMPLES")
fi

EXTRA=()
if [[ "${BOOTSTRAP:-0}" == "1" ]]; then
  EXTRA+=(--bootstrap)
fi
if [[ "${PER_TASK:-0}" == "1" ]]; then
  EXTRA+=(--per_task)
fi

echo "=== A1: baseline Layer A (zero-shot vs LoRA) ==="
python -m swm.eval.evaluate --config "$CONFIG" "${CKPT_ARGS[@]}" --split val \
  "${MAX_ARGS[@]}" "${EXTRA[@]}"

echo "=== A2: shuffle action channels ==="
python -m swm.eval.evaluate --config "$CONFIG" "${CKPT_ARGS[@]}" --split val \
  --shuffle_action --finetuned_only "${MAX_ARGS[@]}"

echo "=== A3: blank image (text-only) ==="
python -m swm.eval.evaluate --config "$CONFIG" "${CKPT_ARGS[@]}" --split val \
  --blank_image --finetuned_only "${MAX_ARGS[@]}"

echo "=== A6: zero-shot without format hint ==="
python -m swm.eval.evaluate --config "$CONFIG" "${CKPT_ARGS[@]}" --split val \
  --zero_shot_only --no_format_hint "${MAX_ARGS[@]}"

echo "=== A5 + Layer B: planner horizons 1 and 2 ==="
python -m swm.eval.planner_eval --config "$CONFIG" "${CKPT_ARGS[@]}" --split val \
  --horizons 1,2

echo "Done. Metrics are under paths.output_dir from $CONFIG"
