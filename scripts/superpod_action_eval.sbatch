#!/bin/bash
#SBATCH --job-name=swm-action-eval
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/action-eval-%j.out
#SBATCH --error=outputs/slurm/action-eval-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm

module purge
module load Anaconda3/2023.09-0 || true

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate
if ! python -c "from transformers import get_cosine_schedule_with_warmup" >/dev/null 2>&1; then
  pip install -U "huggingface_hub>=0.36"
fi

python scripts/eval_action_commands.py \
  --config configs/superpod_action.yaml \
  --checkpoint outputs/superpod_action/checkpoint-final \
  --max-samples 512
