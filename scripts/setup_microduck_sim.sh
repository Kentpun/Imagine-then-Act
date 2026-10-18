#!/usr/bin/env bash
# Install the MQTT MuJoCo simulator next to this repo and share its session.env.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
sim_root="$(cd "$repo_root/.." && pwd)/microduck_demo_part1"

if [[ ! -d "$sim_root/.git" ]]; then
  echo "Cloning microduck_demo_part1 into $sim_root"
  git clone --recurse-submodules https://github.com/hjianbo/microduck_demo_part1.git "$sim_root"
fi

if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "Installing uv (Python package runner required by the simulator)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

command -v uv >/dev/null 2>&1 || {
  echo "uv is still not on PATH. Open a new terminal and rerun this script." >&2
  exit 1
}

echo "Bootstrapping simulator (downloads the walking policy and Python 3.12 deps)"
(
  cd "$sim_root"
  ./scripts/bootstrap.sh
)

mkdir -p "$repo_root/.demo"
ln -sfn "$sim_root/.demo/session.env" "$repo_root/.demo/session.env"
echo
echo "Shared session file:"
echo "  $repo_root/.demo/session.env -> $sim_root/.demo/session.env"
echo
echo "Next, in two terminals:"
echo "  1) cd \"$sim_root\" && ./scripts/run_simulator.sh"
echo "  2) cd \"$repo_root\" && bash scripts/run_microduck_demo.sh --checkpoint outputs/superpod/checkpoint-final"
echo "     Then in the browser: Enable MQTT, turn Dry-run off, Run imagination planner, Execute best."
