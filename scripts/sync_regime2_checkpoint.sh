#!/usr/bin/env bash
# Pull the Nov 22 Regime-2 (mixed-play counterfactual) MicroDuck LoRA weights from HKUST SuperPOD.
# Run once while connected to HKUST campus Wi-Fi or HKUST VPN.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${SUPERPOD_HOST:-superpod}"
REMOTE="${SUPERPOD_REMOTE:-~/semantic-world-model}"

echo "Checking SSH connection to ${HOST}..."
ssh -o ConnectTimeout=20 "${HOST}" "ls -lh ${REMOTE}/outputs/microduck_semantic/checkpoint-final/adapter_model.safetensors"

mkdir -p "${ROOT}/outputs/microduck_semantic/archive_expert_only_v1/checkpoint-final"
if [[ -f "${ROOT}/outputs/microduck_semantic/checkpoint-final/adapter_model.safetensors" ]] && \
   [[ ! -f "${ROOT}/outputs/microduck_semantic/archive_expert_only_v1/checkpoint-final/adapter_model.safetensors" ]]; then
  echo "Archiving local Regime-1 checkpoint to outputs/microduck_semantic/archive_expert_only_v1/checkpoint-final..."
  cp -a "${ROOT}/outputs/microduck_semantic/checkpoint-final/." \
        "${ROOT}/outputs/microduck_semantic/archive_expert_only_v1/checkpoint-final/"
fi

echo "Downloading Regime-2 counterfactual MicroDuck LoRA checkpoint (~70 MB)..."
rsync -avz --progress \
  "${HOST}:${REMOTE}/outputs/microduck_semantic/checkpoint-final/" \
  "${ROOT}/outputs/microduck_semantic/checkpoint-final/"

echo "Done! Updated local checkpoint:"
ls -lh "${ROOT}/outputs/microduck_semantic/checkpoint-final/adapter_model.safetensors"
