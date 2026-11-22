#!/bin/bash
# Seamless continuation / evaluation guard for Job 793831:
# Runs automatically after Job 793831 ends; skips any already-completed stage
# and finishes Layer A + Layer B + Closed-Loop evaluation if 793831 hit walltime.

#SBATCH --job-name=swm-md-cf-eval
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/md-cf-eval-%j.out
#SBATCH --error=outputs/slurm/md-cf-eval-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm

module purge
module load Anaconda3/2023.09-0 || true

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Continuation Host $(hostname) | Job ${SLURM_JOB_ID:-local} ==="

# If B4_direct_policy was interrupted near the end of epoch 3, use latest saved checkpoint or finish
if [[ ! -d outputs/microduck_direct_policy/checkpoint-final ]]; then
  LATEST_CKPT=$(ls -dt outputs/microduck_direct_policy/checkpoint-epoch-* outputs/microduck_direct_policy/checkpoint-* 2>/dev/null | head -n 1 || true)
  if [[ -n "${LATEST_CKPT}" ]]; then
    echo "Using latest saved B4 checkpoint ${LATEST_CKPT} as checkpoint-final..."
    cp -r "${LATEST_CKPT}" outputs/microduck_direct_policy/checkpoint-final
  else
    echo "Training B4_direct_policy from scratch..."
    python -m swm.training.train --config configs/microduck_direct_policy.yaml
  fi
fi

if [[ ! -f outputs/microduck_semantic/eval_metrics_val_counterfactual_mixed_v2.json ]]; then
  echo "=== [5a/5] Evaluating Layer A Semantic Accuracy on Next-State Outcomes ==="
  python -m swm.eval.evaluate \
    --config configs/microduck_train.yaml \
    --checkpoint outputs/microduck_semantic/checkpoint-final \
    --finetuned_only \
    --max_samples 256 \
    --tag counterfactual_mixed_v2
fi

if [[ ! -f outputs/microduck_semantic/planner_metrics_val_counterfactual_mixed_v2.json ]]; then
  echo "=== [5b/5] Evaluating Offline Layer B (Physics Rollout Oracle, All Baselines) ==="
  python -m swm.eval.planner_eval \
    --config configs/microduck_train.yaml \
    --checkpoint outputs/microduck_semantic/checkpoint-final \
    --policy_checkpoint outputs/microduck_direct_policy/checkpoint-final \
    --domain microduck \
    --oracle_mode microduck_rollout \
    --horizons 1 \
    --tag counterfactual_mixed_v2
fi

if ! grep -q '"imagine_then_act_lora"' outputs/microduck_semantic/system_eval_mixed_v2.json 2>/dev/null; then
  echo "=== [5c/5] Evaluating Closed-Loop Multi-Task Control (N=20 per task, 7 controllers) ==="
  python scripts/eval_microduck_system.py \
    --n 20 \
    --max_steps 60 \
    --action_repeat 2 \
    --checkpoint outputs/microduck_semantic/checkpoint-final \
    --policy_checkpoint outputs/microduck_direct_policy/checkpoint-final \
    --out outputs/microduck_semantic/system_eval_mixed_v2.json
fi

echo "=== All Counterfactual Mixed-Play Evaluations Complete ==="
ls -lh outputs/microduck_semantic/eval_metrics_val_counterfactual_mixed_v2.json \
       outputs/microduck_semantic/planner_metrics_val_counterfactual_mixed_v2.json \
       outputs/microduck_semantic/system_eval_mixed_v2.json
