# Evaluation & ablations (Imagine-then-Act)

Aligned with the project proposal §5–§6.

## What “Imagine” means

For one fixed image and goal, the world model is queried on **many candidate
actions that were not necessarily taken**. Each query predicts a semantic
outcome (“what would happen if…”). Ranking those imagined outcomes and
executing the best action is **Imagine-then-Act**.

That is **model-based planning**, not ReAct (ReAct = language thoughts → tools).
Here the “thought” is a predicted world outcome, not a chain-of-thought sentence.

```bash
# See imaginations + selected action on one LIBERO frame
bash scripts/imagine_then_act.sh configs/superpod.yaml --index 0
```

## Layer A — semantic prediction

```bash
bash scripts/eval.sh configs/superpod.yaml
# writes outputs/.../eval_metrics.json  (or eval_metrics_val.json)
```

Metrics: `exact_match`, `macro_f1`, `parse_rate`, per-field `acc_*` / `f1_*`.

### Ablations (eval-time, no retrain)

| ID | Command |
|----|---------|
| A1 Adaptation | default eval (zero-shot vs LoRA) |
| A2 Action conditioning | `bash scripts/eval.sh configs/superpod.yaml --shuffle_action --finetuned_only` |
| A3 Vision | `bash scripts/eval.sh configs/superpod.yaml --blank_image --finetuned_only` |
| A4 Label fields | requires a separate short retrain with fewer target fields (optional) |
| A5 Horizon | see Layer B `--horizons 1,2` |

## Layer B — imagination planner

```bash
bash scripts/eval_planner.sh configs/superpod.yaml
# or: CHECKPOINT=outputs/superpod/checkpoint-final bash scripts/run_ablations.sh configs/superpod.yaml
```

Baselines: **B0** random, **B1** action-heuristic, **B2** zero-shot VLM, **B3** LoRA VLM.

Reports: `top1_vs_oracle`, `regret`, `spearman`, `logged_top1` / `logged_top3`.

Oracle ranking uses **action-only** semantic labels (same rules as dataset labeling without proprioception). Horizon `>1` reuses the same frame (documented limitation).

## One-shot ablation suite

```bash
CHECKPOINT=outputs/superpod/checkpoint-final bash scripts/run_ablations.sh configs/superpod.yaml
```

## SuperPOD (Part 1 science)

No retrain required for main LoRA — uses existing `outputs/superpod/checkpoint-final`.

```bash
# Expanded LIBERO eval (n=2048, bootstrap, per-task, A1–A3, A6, Layer B, qualitative)
cd ~/semantic-world-model && sbatch scripts/superpod_part1_eval.sbatch

# Optional A4 fewer-field retrain
sbatch scripts/superpod_a4.sbatch

# Multi-task RoboCasa OOD (atomic_4 + extras, frozen LIBERO LoRA)
sbatch scripts/superpod_robocasa_ood.sbatch
```

See also `docs/PART2_MICRODUCK.md` for MicroDuck synthetic VLA.
