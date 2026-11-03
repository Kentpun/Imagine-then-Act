#!/usr/bin/env bash
# Observe → Imagine K counterfactual outcomes → Act (print best action).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
CONFIG="${1:-configs/superpod.yaml}"
shift || true
python -m swm.planning.imagine_then_act --config "$CONFIG" "$@"
