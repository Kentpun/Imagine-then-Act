#!/bin/bash
# HKUST SuperPOD — Imagine-then-Act LoRA training
# Hardware: NVIDIA DGX H800 (80 GB). One GPU => ~14 CPU cores, 224 GB RAM.
# Docs: https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod
# First job: https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/submit-first-job
# Partitions: https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/partition
#
# Edit ACCOUNT below, then from the repo root on a login node:
#   sbatch scripts/superpod_train.sbatch
#   squeue -u "$USER"
# Logs: outputs/slurm/train-<jobid>.out

#SBATCH --job-name=swm-lora
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/train-%j.out
#SBATCH --error=outputs/slurm/train-%j.err

set -euo pipefail

# Slurm starts in the directory you submitted from if you cd there first.
# This also works if the script path is absolute.
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm

module purge
# Adjust if `module avail` shows a different Anaconda/CUDA name.
module load Anaconda3/2023.09-0 || true
module load CUDA/12.4.0 || module load cuda || true

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No .venv found. Create one on the login node first:"
  echo "  python -m venv .venv && source .venv/bin/activate"
  echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
  echo "  pip install -e . && pip install -r requirements.txt"
  exit 1
fi

echo "Host $(hostname)  job ${SLURM_JOB_ID:-local}"
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

CONFIG="${CONFIG:-configs/superpod.yaml}"

if [[ ! -f data/processed/train.jsonl ]]; then
  echo "Preparing LIBERO transitions (first run downloads/decodes videos)..."
  if [[ ! -d data/raw/libero/videos ]]; then
    python scripts/download_data.py --repo_id lerobot/libero
  fi
  python scripts/prepare_dataset.py --config "${CONFIG}"
fi

python -m swm.training.train --config "${CONFIG}"
python -m swm.eval.evaluate --config "${CONFIG}"

echo "Done. Metrics: outputs/superpod/eval_metrics.json"
