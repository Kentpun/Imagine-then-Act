#!/usr/bin/env bash
# Sync code (never touching data/, outputs/, or .venv/) and submit
# scripts/superpod_rigorous_eval.sbatch to HKUST SuperPOD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${SUPERPOD_HOST:-superpod}"
REMOTE="${SUPERPOD_REMOTE:-~/semantic-world-model}"

echo "Checking SSH to ${HOST}..."
ssh -o ConnectTimeout=20 -o ConnectionAttempts=1 "${HOST}" 'echo ok; hostname; whoami'

echo "Rsyncing code -> ${HOST}:${REMOTE}"
rsync -az \
  --exclude '.venv/' --exclude '.venv_mac/' --exclude '.venv_local/' --exclude '.tmp_venv/' --exclude '.hf_cache/' \
  --exclude '__pycache__/' --exclude '.git/' --exclude 'outputs/' \
  --exclude 'data/' --exclude '*.pyc' --exclude '.DS_Store' \
  "${ROOT}/" "${HOST}:${REMOTE}/"

SBATCH_FILE="${1:-scripts/superpod_rigorous_eval.sbatch}"

echo "Submitting ${SBATCH_FILE}..."
ssh "${HOST}" "bash -lc '
  set -euo pipefail
  module load slurm 2>/dev/null || true
  cd ${REMOTE/#\~/\$HOME}
  mkdir -p outputs/slurm
  test -f .venv/bin/activate || {
    echo \"Missing .venv on SuperPOD\"
    exit 1
  }
  JOB=\$(sbatch --parsable ${SBATCH_FILE})
  echo \"SUBMITTED_JOB=\$JOB\"
  squeue -u \"\$USER\"
'"
