#!/usr/bin/env bash
# From your laptop (HKUST VPN / campus net required):
#   bash scripts/submit_part1_superpod.sh
#
# Syncs code (not outputs/.venv), submits Part-1 eval, prints job id.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${SUPERPOD_HOST:-superpod}"
REMOTE="${SUPERPOD_REMOTE:-~/semantic-world-model}"

echo "Checking SSH to ${HOST}..."
ssh -o ConnectTimeout=20 -o ConnectionAttempts=1 "${HOST}" 'echo ok; hostname; whoami'

echo "Rsyncing code -> ${HOST}:${REMOTE}"
# Never --delete data/outputs/.venv — an earlier wipe removed LIBERO frames.
rsync -az \
  --exclude '.venv/' --exclude '.tmp_venv/' --exclude '.hf_cache/' \
  --exclude '__pycache__/' --exclude '.git/' --exclude 'outputs/' \
  --exclude 'data/' --exclude '*.pyc' --exclude '.DS_Store' \
  "${ROOT}/" "${HOST}:${REMOTE}/"

echo "Submitting Part-1 eval..."
ssh "${HOST}" "bash -lc '
  set -euo pipefail
  module load slurm 2>/dev/null || true
  cd ${REMOTE/#\~/\$HOME}
  mkdir -p outputs/slurm
  test -d outputs/superpod/checkpoint-final || {
    echo \"Missing outputs/superpod/checkpoint-final on SuperPOD\"
    exit 1
  }
  test -f .venv/bin/activate || {
    echo \"Missing .venv on SuperPOD\"
    exit 1
  }
  JOB=\$(sbatch --parsable scripts/superpod_part1_eval.sbatch)
  echo \"JOB=\$JOB\"
  squeue -u \"\$USER\" | head -20
'"
