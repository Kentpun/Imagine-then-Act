#!/usr/bin/env bash
# Layer A — semantic outcome metrics (EM, macro-F1, field acc).
# Optional flags after config: --blank_image --shuffle_action --split test
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
CONFIG="${1:-configs/default.yaml}"
shift || true
python -m swm.eval.evaluate --config "$CONFIG" "$@"
