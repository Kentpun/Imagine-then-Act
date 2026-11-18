#!/bin/bash
# Run stronger LIBERO ablations sequentially on one GPU allocation.
# Use this when QOSMaxSubmitJobPerUserLimit blocks --array=0-3.
#
#   sbatch scripts/superpod_strong_ablations_sequential.sbatch
#
# Optional: TASKS="0 3" sbatch ...  (subset; default all 0 1 2 3)
#
#SBATCH --job-name=swm-ablate-all
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/ablate-all-%j.out
#SBATCH --error=outputs/slurm/ablate-all-%j.err

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

echo "Host $(hostname) job ${SLURM_JOB_ID:-local}"
nvidia-smi || true

if [[ ! -f data/processed/val.jsonl ]]; then
  echo "Preparing LIBERO transitions..."
  [[ -d data/raw/libero/videos ]] || python scripts/download_data.py --repo_id lerobot/libero
  python scripts/prepare_dataset.py --config configs/superpod.yaml
fi

prepare_a4_labels() {
  if [[ -d data/processed/frames && ! -d data/processed_a4/frames ]]; then
    mkdir -p data/processed_a4
    ln -sfn "$(pwd)/data/processed/frames" data/processed_a4/frames
  fi
  if [[ ! -s data/processed_a4/train.jsonl ]]; then
    python - <<'PY'
import json, re
from pathlib import Path
from swm.data.labeling import build_outcome_string
src = Path("data/processed")
dst = Path("data/processed_a4")
dst.mkdir(parents=True, exist_ok=True)
active = ["moved", "progress", "success_near"]
defaults = {"moved":"no","lifted":"no","gripper":"hold","progress":"flat","success_near":"no"}
for split in ("train", "val"):
    rows = []
    for line in (src / f"{split}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = r.get("outcome", "")
        labels = dict(defaults)
        for k in defaults:
            m = re.search(rf"{k}=([a-z]+)", text.lower())
            if m:
                labels[k] = m.group(1)
        r["outcome"] = build_outcome_string(labels, active_fields=active)
        rows.append(r)
    out = dst / f"{split}.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", out, len(rows))
PY
  fi
}

run_task() {
  local TASK_ID="$1"
  echo ""
  echo "######################################################################"
  echo "### Ablation task ${TASK_ID}  $(date -Is)"
  echo "######################################################################"
  case "${TASK_ID}" in
    0)
      CONFIG=configs/superpod_ablate_no_action.yaml
      echo "=== Train/eval: no-action prompt ablation ==="
      python -m swm.training.train --config "${CONFIG}"
      python -m swm.eval.evaluate --config "${CONFIG}" --finetuned_only \
        --bootstrap --per_task --max_samples 2048 --tag no_action
      ;;
    1)
      CONFIG=configs/superpod_ablate_freeze_vision.yaml
      echo "=== Train/eval: freeze-vision ablation ==="
      python -m swm.training.train --config "${CONFIG}"
      python -m swm.eval.evaluate --config "${CONFIG}" --finetuned_only \
        --bootstrap --per_task --max_samples 2048 --tag freeze_vision
      ;;
    2)
      CONFIG=configs/superpod_a4.yaml
      echo "=== Train/eval: fewer-fields (matched n=2048) ==="
      prepare_a4_labels
      python -m swm.training.train --config "${CONFIG}"
      python -m swm.eval.evaluate --config "${CONFIG}" --finetuned_only \
        --bootstrap --per_task --max_samples 2048 --tag a4_fewer_fields
      ;;
    3)
      CONFIG=configs/superpod.yaml
      CKPT="${CHECKPOINT:-outputs/superpod/checkpoint-final}"
      echo "=== Eval-only: planner score-field ablations on ${CKPT} ==="
      if [[ ! -d "${CKPT}" ]]; then
        echo "Missing main checkpoint: ${CKPT}"
        exit 1
      fi
      python -m swm.eval.planner_eval --config "${CONFIG}" --checkpoint "${CKPT}" \
        --horizons 1 --skip_zero_shot --max_samples 64 --tag score_full
      python -m swm.eval.planner_eval --config "${CONFIG}" --checkpoint "${CKPT}" \
        --horizons 1 --skip_zero_shot --max_samples 64 \
        --disable_score_fields progress --tag score_no_progress
      python -m swm.eval.planner_eval --config "${CONFIG}" --checkpoint "${CKPT}" \
        --horizons 1 --skip_zero_shot --max_samples 64 \
        --disable_score_fields success_near --tag score_no_success_near
      ;;
    *)
      echo "Unknown task id ${TASK_ID}"
      exit 1
      ;;
  esac
  echo "### Finished task ${TASK_ID}  $(date -Is)"
}

TASKS="${TASKS:-0 1 2 3}"
for t in ${TASKS}; do
  run_task "${t}"
done

echo "All ablation tasks done: ${TASKS}"
ls -la outputs/superpod_ablate_no_action/eval_metrics*.json \
       outputs/superpod_ablate_freeze_vision/eval_metrics*.json \
       outputs/superpod_a4/eval_metrics*.json \
       outputs/superpod/planner_metrics_val_score*.json 2>/dev/null || true
