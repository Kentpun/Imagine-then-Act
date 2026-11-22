#!/bin/bash
# Equal-compute MicroDuck counterfactual world-model vs Direct VLA vs ReAct experiment:
#   1. Archives expert-only v1 results (from Job 793144) for Regime-1 paper comparison
#   2. Generates realistic mixed-play / suboptimal demonstrations (400 episodes, ~11k frames)
#      with truthful next-state (s_{t+1}) semantic transition labels
#   3. Trains BOTH Semantic World Model (B3_lora) and Direct VLA (B4_direct_policy)
#      at strictly matched compute (Qwen3-VL-2B, LoRA r=16/a=32, 3 epochs, identical frames)
#   4. Evaluates Offline Layer B (physics rollout oracle) and Closed-Loop Multi-Task Control

#SBATCH --job-name=swm-md-cf
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/md-cf-%j.out
#SBATCH --error=outputs/slurm/md-cf-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm outputs/microduck_semantic/archive_expert_only_v1

module purge
module load Anaconda3/2023.09-0 || true

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Host $(hostname) | Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi || true

# 1. Archive Regime-1 (100% expert) JSON artifacts so both regimes are preserved for the paper
for f in outputs/microduck_semantic/system_eval.json \
         outputs/microduck_semantic/planner_metrics_val_rollout_oracle_all_baselines.json \
         outputs/microduck_semantic/eval_metrics.json; do
  if [[ -f "${f}" ]] && [[ ! -f "outputs/microduck_semantic/archive_expert_only_v1/$(basename "${f}")" ]]; then
    cp "${f}" "outputs/microduck_semantic/archive_expert_only_v1/$(basename "${f}")"
  fi
done

# 2. Generate mixed-play MicroDuck dataset (20% expert, 45% myopic teleop, 35% exploratory play)
echo "=== [1/5] Recording mixed-play MicroDuck dataset (400 episodes) ==="
rm -rf data/microduck_raw data/processed_microduck data/processed_microduck_policy
python scripts/record_microduck_logs.py \
  --out_dir data/microduck_raw \
  --episodes_per_task 100 \
  --max_steps 40 \
  --behavior_mode mixed_play \
  --seed 42

echo "=== [2/5] Preparing next-state semantic & equal-compute direct-policy splits ==="
python scripts/prepare_microduck_semantic.py --config configs/microduck_train.yaml
python scripts/prepare_direct_policy_dataset.py \
  --src_dir data/processed_microduck \
  --out_dir data/processed_microduck_policy \
  --domain microduck

# 3. Train both models from scratch on the exact same frames & hyperparameters
echo "=== [3/5] Training Semantic World Model LoRA (B3_lora, 3 epochs) ==="
rm -rf outputs/microduck_semantic/checkpoint-*
python -m swm.training.train --config configs/microduck_train.yaml

echo "=== [4/5] Training Equal-Compute Direct Policy VLA LoRA (B4_direct_policy, 3 epochs) ==="
rm -rf outputs/microduck_direct_policy/checkpoint-*
python -m swm.training.train --config configs/microduck_direct_policy.yaml

# 4. Run Layer A, Offline Layer B (physics rollout oracle), and Closed-Loop System Evaluation
echo "=== [5a/5] Evaluating Layer A Semantic Accuracy on Next-State Outcomes ==="
python -m swm.eval.evaluate \
  --config configs/microduck_train.yaml \
  --checkpoint outputs/microduck_semantic/checkpoint-final \
  --finetuned_only \
  --max_samples 256 \
  --tag counterfactual_mixed_v2

echo "=== [5b/5] Evaluating Offline Layer B (Physics Rollout Oracle, All Baselines) ==="
python -m swm.eval.planner_eval \
  --config configs/microduck_train.yaml \
  --checkpoint outputs/microduck_semantic/checkpoint-final \
  --policy_checkpoint outputs/microduck_direct_policy/checkpoint-final \
  --domain microduck \
  --oracle_mode microduck_rollout \
  --horizons 1 \
  --tag counterfactual_mixed_v2

echo "=== [5c/5] Evaluating Closed-Loop Multi-Task Control (N=20 per task, 7 controllers) ==="
python scripts/eval_microduck_system.py \
  --n 20 \
  --max_steps 60 \
  --action_repeat 2 \
  --checkpoint outputs/microduck_semantic/checkpoint-final \
  --policy_checkpoint outputs/microduck_direct_policy/checkpoint-final \
  --out outputs/microduck_semantic/system_eval_mixed_v2.json

echo "=== Completed MicroDuck Counterfactual Experiment ==="
ls -lh outputs/microduck_semantic/eval_metrics* \
       outputs/microduck_semantic/planner_metrics_val_counterfactual_mixed_v2.json \
       outputs/microduck_semantic/system_eval_mixed_v2.json
