#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

MOCK=1
CHECKPOINT=""
PORT=8502

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock) MOCK=1; shift ;;
    --checkpoint) CHECKPOINT="$2"; MOCK=0; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ ! -f data/microduck_smoke/samples.jsonl ]]; then
  python scripts/prepare_microduck_smoke.py
fi

# Streamlit reads sidebar defaults; mock is the default toggle in the app.
# Pass checkpoint via env for convenience when not using mock.
export SWM_MICRODUCK_CHECKPOINT="${CHECKPOINT}"
export SWM_MICRODUCK_MOCK="${MOCK}"

streamlit run src/swm/microduck/demo_app.py \
  --server.address 127.0.0.1 \
  --server.port "${PORT}"
