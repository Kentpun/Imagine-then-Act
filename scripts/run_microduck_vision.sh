#!/usr/bin/env bash
# Watch the simulator's head-camera frames and publish a walk command.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import peft" >/dev/null 2>&1; then
  if [[ -x /opt/homebrew/anaconda3/bin/python3.12 ]]; then
    PYTHON=/opt/homebrew/anaconda3/bin/python3.12
  fi
fi

exec "$PYTHON" -m swm.microduck.vision_loop "$@"
