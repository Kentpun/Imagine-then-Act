#!/bin/bash
# v4 action LoRA: Language Table + BridgeData V2, resume from v3.
#
# Bridge raw data is stored under project storage because home is ~97% full:
#   /project/mscbdtsuperpod/kkpun/swm_data/
#
#   sbatch scripts/superpod_action_train_v4.sbatch

#SBATCH --job-name=swm-action-v4
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/action-train-v4-%j.out
#SBATCH --error=outputs/slurm/action-train-v4-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm

PROJECT_DATA="/project/mscbdtsuperpod/kkpun/swm_data"
mkdir -p \
  "${PROJECT_DATA}/raw/bridge" \
  "${PROJECT_DATA}/processed_action_v4/frames" \
  "${PROJECT_DATA}/hf_cache"

module purge
module load Anaconda3/2023.09-0 || true

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
# Keep HF downloads off the tiny home quota.
export HF_HOME="${PROJECT_DATA}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
mkdir -p "${HF_HOME}"

source .venv/bin/activate
if ! python -c "import lerobot" >/dev/null 2>&1; then
  pip install "lerobot>=0.3"
fi
if ! python -c "from transformers import get_cosine_schedule_with_warmup" >/dev/null 2>&1; then
  pip install -U "huggingface_hub>=0.36"
fi

echo "Host $(hostname)  job ${SLURM_JOB_ID:-local}"
echo "Project data root: ${PROJECT_DATA}"
df -h /project /home | sed -n '1,5p'
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

CONFIG="${CONFIG:-configs/superpod_action_v4.yaml}"

if [[ ! -f outputs/superpod_action_v3/checkpoint-final/adapter_model.safetensors ]]; then
  echo "Missing v3 adapter at outputs/superpod_action_v3/checkpoint-final" >&2
  exit 1
fi

TRAIN_JSONL="${PROJECT_DATA}/processed_action_v4/train.jsonl"
if [[ ! -s "${TRAIN_JSONL}" ]]; then
  echo "Preparing v4 dataset (Language Table + Bridge subset)..."
  echo "Bridge videos download into ${PROJECT_DATA}/raw/bridge (episode subset only)."
  python scripts/prepare_action_dataset.py --config "${CONFIG}"
else
  echo "Reusing existing ${TRAIN_JSONL}"
fi

wc -l "${PROJECT_DATA}/processed_action_v4/"*.jsonl || true
du -sh "${PROJECT_DATA}/raw/bridge" "${PROJECT_DATA}/processed_action_v4" || true

python -m swm.training.train --config "${CONFIG}"

echo "Done. v4 adapter: outputs/superpod_action_v4/checkpoint-final"
python scripts/eval_action_commands.py \
  --config "${CONFIG}" \
  --checkpoint outputs/superpod_action_v4/checkpoint-final \
  --max-samples 512
