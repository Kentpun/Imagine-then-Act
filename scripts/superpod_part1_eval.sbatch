#!/bin/bash
# Part 1 expanded LIBERO eval on SuperPOD (existing semantic LoRA).
# A1–A3, A6, Layer B/A5, bootstrap + per-task, qualitative cases.
#
#   cd ~/semantic-world-model && sbatch scripts/superpod_part1_eval.sbatch

#SBATCH --job-name=swm-part1-eval
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/part1-eval-%j.out
#SBATCH --error=outputs/slurm/part1-eval-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm outputs/superpod

module purge
module load Anaconda3/2023.09-0 || true
module load CUDA/12.4.0 || module load cuda || true

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

# shellcheck disable=SC1091
source .venv/bin/activate

CONFIG="${CONFIG:-configs/superpod_eval_expanded.yaml}"
export CHECKPOINT="${CHECKPOINT:-outputs/superpod/checkpoint-final}"
export BOOTSTRAP=1
export PER_TASK=1

echo "Host $(hostname) job ${SLURM_JOB_ID:-local}"
nvidia-smi || true

if [[ ! -f data/processed/val.jsonl ]]; then
  echo "Preparing LIBERO transitions..."
  [[ -d data/raw/libero/videos ]] || python scripts/download_data.py --repo_id lerobot/libero
  python scripts/prepare_dataset.py --config configs/superpod.yaml
fi

bash scripts/run_ablations.sh "${CONFIG}"

echo "=== Qualitative 5+5 cases ==="
python scripts/export_qualitative_cases.py --config "${CONFIG}" --checkpoint "${CHECKPOINT}"

echo "Done."
ls -la outputs/superpod/eval_metrics*.json outputs/superpod/planner_metrics*.json 2>/dev/null || true
