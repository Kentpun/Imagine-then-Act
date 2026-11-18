#!/usr/bin/env bash
# Submit A4 + MicroDuck train when SuperPOD user job slots free.
# Usage (VPN on): bash scripts/submit_remaining_part2.sh
set -euo pipefail
ssh -o BatchMode=yes -o ServerAliveInterval=15 superpod 'bash -lc "
  module load slurm 2>/dev/null || true
  cd \$HOME/semantic-world-model
  squeue -u \$USER
  J4=\$(sbatch --parsable scripts/superpod_a4.sbatch)
  echo A4=\$J4
  J5=\$(sbatch --parsable scripts/superpod_microduck_train.sbatch)
  echo MICRODUCK=\$J5
  squeue -u \$USER
"'
