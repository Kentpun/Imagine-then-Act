#!/bin/bash
# Multi-task RoboCasa OOD eval (frozen LIBERO LoRA). No RoboCasa sim install.
#SBATCH --job-name=swm-robocasa-ood
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/robocasa-ood-%j.out
#SBATCH --error=outputs/slurm/robocasa-ood-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm
module purge
module load Anaconda3/2023.09-0 || true
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
# shellcheck disable=SC1091
source .venv/bin/activate

CONFIG=configs/robocasa_ood.yaml
CKPT="${CHECKPOINT:-outputs/superpod/checkpoint-final}"

# Prefer PyAV over broken torchcodec on SuperPOD login/compute images.
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export HF_HUB_ENABLE_HF_TRANSFER=0

# Re-prepare if previous run failed mid-way (empty / missing val).
if [[ ! -s data/processed_robocasa_ood/val.jsonl ]]; then
  python scripts/prepare_robocasa_eval.py --config "${CONFIG}"
fi

python -m swm.eval.evaluate --config "${CONFIG}" --checkpoint "${CKPT}" \
  --split val --bootstrap --per_task --tag robocasa_ood

echo "Wrote outputs/robocasa_ood/eval_metrics_robocasa_ood.json"
