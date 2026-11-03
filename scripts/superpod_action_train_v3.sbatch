#!/bin/bash
# Continue action LoRA from v2 for 3 more epochs at lower LR.
#
#   sbatch scripts/superpod_action_train_v3.sbatch
#
# Final adapter: outputs/superpod_action_v3/checkpoint-final

#SBATCH --job-name=swm-action-v3
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/action-train-v3-%j.out
#SBATCH --error=outputs/slurm/action-train-v3-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm

module purge
module load Anaconda3/2023.09-0 || true

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
mkdir -p "${HF_HOME}"

source .venv/bin/activate
if ! python -c "from transformers import get_cosine_schedule_with_warmup" >/dev/null 2>&1; then
  pip install -U "huggingface_hub>=0.36"
fi

echo "Host $(hostname)  job ${SLURM_JOB_ID:-local}"
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

CONFIG="${CONFIG:-configs/superpod_action_v3.yaml}"

if [[ ! -f outputs/superpod_action_v2/checkpoint-final/adapter_model.safetensors ]]; then
  echo "Missing v2 adapter at outputs/superpod_action_v2/checkpoint-final" >&2
  exit 1
fi
if [[ ! -s data/processed_action_v2/train.jsonl ]]; then
  echo "Missing cleaned dataset data/processed_action_v2/train.jsonl" >&2
  exit 1
fi

python -m swm.training.train --config "${CONFIG}"

echo "Done. v3 adapter: outputs/superpod_action_v3/checkpoint-final"
python scripts/eval_action_commands.py \
  --config "${CONFIG}" \
  --checkpoint outputs/superpod_action_v3/checkpoint-final \
  --max-samples 512
