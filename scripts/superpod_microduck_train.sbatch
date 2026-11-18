#!/bin/bash
# Part 2: synthesize MicroDuck semantic data + train LoRA on SuperPOD.
#SBATCH --job-name=swm-microduck
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/microduck-%j.out
#SBATCH --error=outputs/slurm/microduck-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm
module purge
module load Anaconda3/2023.09-0 || true
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
# shellcheck disable=SC1091
source .venv/bin/activate

CONFIG=configs/microduck_train.yaml

# Prefer existing semantic JSONL (laptop/headless synth). Only record if missing.
if [[ ! -s data/processed_microduck/train.jsonl ]]; then
  if [[ ! -f data/microduck_raw/index.json ]]; then
    python scripts/record_microduck_logs.py --episodes_per_task 60
  fi
  python scripts/prepare_microduck_semantic.py --config "${CONFIG}"
fi

python -m swm.training.train --config "${CONFIG}"
python -m swm.eval.evaluate --config "${CONFIG}" --bootstrap --per_task \
  --tag microduck_val

# Optional CCTV detector (pngwn). Skip if already trained or ultralytics missing.
if [[ ! -f outputs/microduck_detector/best.pt ]]; then
  if python -c "import ultralytics" 2>/dev/null; then
    if [[ ! -d data/raw/microduck_detection/images/train ]]; then
      python scripts/download_microduck_detection.py || true
    fi
    if [[ -d data/raw/microduck_detection/images/train ]]; then
      python scripts/train_duck_detector.py --epochs 30 --device 0 || true
    fi
  else
    echo "ultralytics not installed — detector skipped (pip install ultralytics)"
  fi
fi

echo "MicroDuck LoRA → outputs/microduck_semantic/"
