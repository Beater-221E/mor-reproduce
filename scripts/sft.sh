#!/usr/bin/env bash
# Usage: bash scripts/sft.sh <dataset> <0.5B|1.5B|3B>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE=1
# Avoid NCCL P2P hangs on some multi-V100 topologies (busy-wait 100% util, ~50W).
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
# Keep HF/transformers quiet (config dumps are extremely noisy under torchrun).
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

DS="${1:?dataset}"
SIZE="${2:?size}"
# optional 3rd arg or RESUME=auto|path
RESUME="${3:-${RESUME:-}}"
NPROC="${NPROC:-4}"

echo "[sft.sh] ds=${DS} size=${SIZE} nproc=${NPROC} resume=${RESUME:-none} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"

EXTRA=()
if [[ -n "${RESUME}" ]]; then
  EXTRA+=(--resume "${RESUME}")
fi

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29501}" \
  --log_dir "${ROOT}/checkpoints/sft_${DS}_${SIZE}/torchrun_logs" \
  -m minionerec.sft \
  --config "${ROOT}/configs/sft.yaml" \
  --model_size "${SIZE}" \
  --train_file "${ROOT}/data/processed/${DS}/tasks/train.jsonl" \
  --eval_file "${ROOT}/data/processed/${DS}/tasks/valid.jsonl" \
  --output_dir "${ROOT}/checkpoints/sft_${DS}_${SIZE}" \
  --deepspeed "${ROOT}/configs/ds_sft.json" \
  "${EXTRA[@]}"
