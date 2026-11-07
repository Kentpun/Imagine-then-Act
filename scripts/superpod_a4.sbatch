#!/bin/bash
# A4 ablation: prepare fewer-field labels, short LoRA train, eval.
#SBATCH --job-name=swm-a4
#SBATCH --partition=normal
#SBATCH --account=mscbdtsuperpod
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/a4-%j.out
#SBATCH --error=outputs/slurm/a4-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs/slurm
module purge
module load Anaconda3/2023.09-0 || true
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
# shellcheck disable=SC1091
source .venv/bin/activate

CONFIG=configs/superpod_a4.yaml
# Reuse LIBERO frames if present: symlink processed frames into a4 tree
if [[ -d data/processed/frames && ! -d data/processed_a4/frames ]]; then
  mkdir -p data/processed_a4
  ln -sfn "$(pwd)/data/processed/frames" data/processed_a4/frames
fi
# Prefer rewriting outcomes from existing full JSONL (seconds). Only call
# prepare_dataset.py if no prior LIBERO processed labels exist.
if [[ ! -s data/processed_a4/train.jsonl ]]; then
  if [[ -s data/processed/train.jsonl ]]; then
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
  else
    python scripts/prepare_dataset.py --config "${CONFIG}"
  fi
fi

python -m swm.training.train --config "${CONFIG}"
python -m swm.eval.evaluate --config "${CONFIG}" --finetuned_only --max_samples 512
echo "A4 done → outputs/superpod_a4/"
