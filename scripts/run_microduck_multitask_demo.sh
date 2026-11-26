#!/usr/bin/env bash
# Part 2 multitask CCTV demo: genuine Imagine-then-Act + duck detector + raw MQTT (no CircleKeeper).
# Terminal 1: cd ../microduck_demo_part1 && ./scripts/run_raw_simulator.sh walk_ab
# Terminal 2: bash scripts/run_microduck_multitask_demo.sh walk_ab
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_microduck_raw_demo.sh "$@"
