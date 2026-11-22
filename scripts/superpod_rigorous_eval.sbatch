#!/bin/bash
# End-to-end rigorous evaluation on HKUST SuperPOD (fixes 1, 2, and 3):
#   (3) State/scene-aware non-tautological Layer B oracles (LIBERO + MicroDuck)
#   (2) Equal-compute direct-policy LoRA baselines (B4_direct_policy) on identical frames
#   (1) Closed-loop evaluation: MicroDuck multi-task rollouts + LIBERO closed-loop rollouts
#
# Submit:
#   cd ~/semantic-world-model && sbatch scripts/superpod_rigorous_eval.sbatch

#SBATCH --job-name=swm-rigorous-123
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/rigorous-123-%j.out
#SBATCH --error=outputs/slurm/rigorous-123-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm outputs/superpod outputs/microduck_semantic

module purge
module load Anaconda3/2023.09-0 || true
module load CUDA/12.4.0 || module load cuda || true

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
mkdir -p "${HF_HOME}"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Host $(hostname) | Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi || true

# ---------------------------------------------------------------------------
# A. MicroDuck: refresh scene_state metadata, train B4_direct_policy, run #1-#3
# ---------------------------------------------------------------------------
if [[ ! -f data/microduck_raw/episodes/ep00000.jsonl ]] || [[ ! -d data/microduck_raw/frames ]]; then
  echo "Generating synthetic MicroDuck raw logs (episodes + frames)..."
  rm -rf data/microduck_raw
  python scripts/record_microduck_logs.py --out_dir data/microduck_raw
fi
if ! grep -q '"scene_state"' data/processed_microduck/val.jsonl 2>/dev/null; then
  echo "Preparing MicroDuck semantic dataset (with scene_state for rollout oracle)..."
  python scripts/prepare_microduck_semantic.py --config configs/microduck_train.yaml
fi

if [[ ! -f data/processed_microduck_policy/train.jsonl ]]; then
  echo "Preparing equal-compute MicroDuck direct-policy splits..."
  python scripts/prepare_direct_policy_dataset.py \
    --src_dir data/processed_microduck \
    --out_dir data/processed_microduck_policy \
    --domain microduck
fi

if [[ ! -d outputs/microduck_direct_policy/checkpoint-final ]]; then
  echo "Training equal-compute MicroDuck B4_direct_policy LoRA..."
  python -m swm.training.train --config configs/microduck_direct_policy.yaml
fi

echo "Running MicroDuck Layer B with true rollout oracle + Zero-Shot + ReAct + B4_direct_policy..."
python -m swm.eval.planner_eval \
  --config configs/microduck_train.yaml \
  --checkpoint outputs/microduck_semantic/checkpoint-final \
  --policy_checkpoint outputs/microduck_direct_policy/checkpoint-final \
  --domain microduck \
  --oracle_mode microduck_rollout \
  --horizons 1 \
  --tag rollout_oracle_all_baselines

echo "Running MicroDuck closed-loop system evaluation (N=20 per task)..."
python scripts/eval_microduck_system.py \
  --n 20 \
  --max_steps 60 \
  --action_repeat 2 \
  --checkpoint outputs/microduck_semantic/checkpoint-final \
  --policy_checkpoint outputs/microduck_direct_policy/checkpoint-final \
  --out outputs/microduck_semantic/system_eval.json

# ---------------------------------------------------------------------------
# B. LIBERO: train equal-compute B4_direct_policy on canonical splits, run #1-#3
# ---------------------------------------------------------------------------
ACTION_CMD_CKPT="outputs/superpod_action_v4/checkpoint-final"
if [[ ! -d "${ACTION_CMD_CKPT}" ]]; then
  ACTION_CMD_CKPT="outputs/superpod_action_v3/checkpoint-final"
fi

if [[ -f data/processed/train.jsonl ]]; then
  echo "Preparing equal-compute LIBERO direct-policy splits..."
  python scripts/prepare_direct_policy_dataset.py \
    --src_dir data/processed \
    --out_dir data/processed_direct_policy \
    --domain libero

  if [[ ! -d outputs/superpod_direct_policy/checkpoint-final ]]; then
    echo "Training equal-compute LIBERO B4_direct_policy LoRA..."
    python -m swm.training.train --config configs/superpod_direct_policy.yaml
  fi

  echo "Running LIBERO Layer B with state_counterfactual oracle + Zero-Shot + ReAct + B4_direct_policy..."
  python -m swm.eval.planner_eval \
    --config configs/superpod.yaml \
    --checkpoint outputs/superpod/checkpoint-final \
    --policy_checkpoint outputs/superpod_direct_policy/checkpoint-final \
    --action_cmd_checkpoint "${ACTION_CMD_CKPT}" \
    --domain libero \
    --oracle_mode state_counterfactual \
    --horizons 1,2 \
    --tag state_oracle_b4

  echo "Running legacy action-only oracle comparison for paper table..."
  python -m swm.eval.planner_eval \
    --config configs/superpod.yaml \
    --checkpoint outputs/superpod/checkpoint-final \
    --policy_checkpoint outputs/superpod_direct_policy/checkpoint-final \
    --action_cmd_checkpoint "${ACTION_CMD_CKPT}" \
    --domain libero \
    --oracle_mode action_only_legacy \
    --horizons 1 \
    --skip_zero_shot \
    --tag legacy_oracle_b4
fi

echo "Running closed-loop LIBERO evaluation (uses MuJoCo OffScreenRenderEnv if installed)..."
python -m swm.eval.libero_closed_loop \
  --config configs/superpod.yaml \
  --checkpoint outputs/superpod/checkpoint-final \
  --policy_checkpoint outputs/superpod_direct_policy/checkpoint-final \
  --action_cmd_checkpoint "${ACTION_CMD_CKPT}" \
  --backend auto \
  --suite libero_spatial \
  --max_tasks 5 \
  --episodes_per_task 10 \
  --out outputs/superpod/closed_loop_libero_spatial.json

echo "=== All rigorous evaluations finished ==="
ls -lh outputs/microduck_semantic/system_eval.json \
       outputs/microduck_semantic/planner_metrics_*.json \
       outputs/superpod/planner_metrics_*.json \
       outputs/superpod/closed_loop_*.json 2>/dev/null || true
