#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
CONFIG="${1:-configs/default.yaml}"
python -m swm.training.train --config "$CONFIG"
