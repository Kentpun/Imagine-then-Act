# Stronger LIBERO ablations (SuperPOD)

These replace the weaker “call everything an ablation” story with single-factor controls.

## Naming in the report

| Scientific wording | What it is |
|---|---|
| Main comparison: zero-shot vs LoRA | **Not** an ablation |
| Shuffle action / blank image | Eval-time input ablations (fixed checkpoint) |
| Fewer supervised fields | Retrain ablation (matched $n{=}2048$) |
| No action in prompt | **Strong** action ablation (retrain) |
| Freeze vision encoder | **Strong** vision ablation (retrain) |
| Drop score terms (`progress` / `success_near`) | Planner score ablation (eval only) |
| Two-step same-frame plan | Horizon **sensitivity**, not an ablation |

## Submit on SuperPOD

From the repo root on a login node (after `data/processed/` exists or will be prepared by the job):

```bash
sbatch --array=0-3 scripts/superpod_strong_ablations.sbatch
```

Array map:

| ID | Job |
|----|-----|
| 0 | Train/eval **no-action** prompt |
| 1 | Train/eval **freeze-vision** |
| 2 | Retrain/eval **fewer fields** at $n{=}2048$ |
| 3 | Planner **score-field** ablations on the main checkpoint |

One task only:

```bash
sbatch --array=0 scripts/superpod_strong_ablations.sbatch
```

## Outputs

- `outputs/superpod_ablate_no_action/`
- `outputs/superpod_ablate_freeze_vision/`
- `outputs/superpod_a4/`
- `outputs/superpod/planner_metrics_val_score_*.json`

After jobs finish, fold the new EM / Spearman numbers into the report tables (replace the interim shuffle/blank story where the stronger control supersedes it).
