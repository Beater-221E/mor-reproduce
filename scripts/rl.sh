#!/usr/bin/env bash
# Usage: bash scripts/rl.sh <dataset> <0.5B|1.5B|3B> [G]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE=1

DS="${1:?dataset}"
SIZE="${2:?size}"
G="${3:-16}"
NPROC="${NPROC:-4}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29502}" \
  -m minionerec.rl \
  --config "${ROOT}/configs/rl.yaml" \
  --model_path "${ROOT}/checkpoints/sft_${DS}_${SIZE}" \
  --train_file "${ROOT}/data/processed/${DS}/tasks/rl_train.jsonl" \
  --sid_map "${ROOT}/data/processed/${DS}/sid/sid_map.json" \
  --output_dir "${ROOT}/checkpoints/rl_${DS}_${SIZE}" \
  --num_generations "${G}"
