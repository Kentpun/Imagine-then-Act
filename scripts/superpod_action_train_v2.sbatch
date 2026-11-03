#!/bin/bash
# Improved action-command LoRA: resume v1 + cleaner labels + more data.
#
#   sbatch scripts/superpod_action_train_v2.sbatch
#
# Final adapter: outputs/superpod_action_v2/checkpoint-final

#SBATCH --job-name=swm-action-v2
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/action-train-v2-%j.out
#SBATCH --error=outputs/slurm/action-train-v2-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm data/processed_action_v2

module purge
module load Anaconda3/2023.09-0 || true
module load CUDA/12.4.0 || module load cuda || true

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
mkdir -p "${HF_HOME}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No .venv found." >&2
  exit 1
fi

if ! python -c "import lerobot" >/dev/null 2>&1; then
  pip install "lerobot>=0.3"
fi
if ! python -c "from transformers import get_cosine_schedule_with_warmup" >/dev/null 2>&1; then
  pip install -U "huggingface_hub>=0.36"
fi

echo "Host $(hostname)  job ${SLURM_JOB_ID:-local}"
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

CONFIG="${CONFIG:-configs/superpod_action_v2.yaml}"

if [[ ! -f outputs/superpod_action/checkpoint-final/adapter_model.safetensors ]]; then
  echo "Missing v1 adapter at outputs/superpod_action/checkpoint-final" >&2
  exit 1
fi

if [[ ! -s data/processed_action_v2/train.jsonl ]]; then
  echo "Preparing improved action-command dataset (collapse_magnitude=true)..."
  python scripts/prepare_action_dataset.py --config "${CONFIG}"
fi

python -m swm.training.train --config "${CONFIG}"

echo "Done. Improved adapter: outputs/superpod_action_v2/checkpoint-final"
python scripts/eval_action_commands.py \
  --config "${CONFIG}" \
  --checkpoint outputs/superpod_action_v2/checkpoint-final \
  --max-samples 512
