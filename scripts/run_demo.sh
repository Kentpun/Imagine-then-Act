#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
streamlit run src/swm/demo/app.py --server.address 127.0.0.1 --server.port 8501
