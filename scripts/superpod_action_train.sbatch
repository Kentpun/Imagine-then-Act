#!/bin/bash
# HKUST SuperPOD — general action-command LoRA training
#
# This trains Qwen/Qwen3-VL-2B-Instruct to produce a robot-independent text
# action command from an image and an instruction. It is separate from the
# LIBERO outcome LoRA in outputs/superpod and from all MicroDuck code.
#
# From the semantic-world-model repo root on a SuperPOD login node:
#   sbatch scripts/superpod_action_train.sbatch
#
# Logs: outputs/slurm/action-train-<jobid>.out
# Final adapter: outputs/superpod_action/checkpoint-final

#SBATCH --job-name=swm-action-lora
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/action-train-%j.out
#SBATCH --error=outputs/slurm/action-train-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm data/processed_action

module purge
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
  echo "No .venv found. Create one on the login node first:" >&2
  echo "  python -m venv .venv && source .venv/bin/activate" >&2
  echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124" >&2
  echo "  pip install -e . && pip install -r requirements.txt" >&2
  echo "  pip install 'lerobot>=0.3'" >&2
  exit 1
fi

# The action dataset loader uses LeRobot's public dataset packages. Install it
# only when the existing project venv does not already provide it.
if ! python -c "import lerobot" >/dev/null 2>&1; then
  echo "Installing LeRobot dataset loader..."
  pip install "lerobot>=0.3"
fi

# lerobot 0.4.x pins huggingface_hub<0.36, which breaks transformers 5.x
# (ImportError: ResolvedRevision). Restore a hub version that imports cleanly.
if ! python -c "from transformers import get_cosine_schedule_with_warmup" >/dev/null 2>&1; then
  echo "Repairing huggingface_hub for transformers compatibility..."
  pip install -U "huggingface_hub>=0.36"
fi

# Avoid torchcodec on SuperPOD: it needs system FFmpeg shared libs that are
# usually missing. prepare_action_dataset.py also forces video_backend=pyav.
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"

echo "Host $(hostname)  job ${SLURM_JOB_ID:-local}"
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

CONFIG="${CONFIG:-configs/superpod_action.yaml}"

if [[ ! -s data/processed_action/train.jsonl ]]; then
  echo "Preparing general action-command dataset..."
  echo "Default source: Language Table. BridgeData V2 is disabled in the config because it is about 90 GB."
  python scripts/prepare_action_dataset.py --config "${CONFIG}"
fi

python -m swm.training.train --config "${CONFIG}"

echo "Done. General action-command adapter: outputs/superpod_action/checkpoint-final"
echo "This checkpoint is not evaluated by the LIBERO outcome evaluator."
