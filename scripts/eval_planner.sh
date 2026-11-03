#!/usr/bin/env bash
# Layer B — imagination planner baselines B0–B3 (+ horizon ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
CONFIG="${1:-configs/default.yaml}"
shift || true
python -m swm.eval.planner_eval --config "$CONFIG" "$@"
