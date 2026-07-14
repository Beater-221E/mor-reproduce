#!/usr/bin/env bash
# Usage: bash scripts/eval.sh <dataset> <0.5B|1.5B|3B> [sft|rl]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

DS="${1:?dataset}"
SIZE="${2:?size}"
STAGE="${3:-rl}"

python -m minionerec.eval \
  --model_path "${ROOT}/checkpoints/${STAGE}_${DS}_${SIZE}/final" \
  --test_file "${ROOT}/data/processed/${DS}/tasks/test.jsonl" \
  --sid_map "${ROOT}/data/processed/${DS}/sid/sid_map.json" \
  --output "${ROOT}/experiments/${STAGE}_${DS}_${SIZE}.json" \
  --num_beams 50 --batch_size 2 --device "${DEVICE:-cuda:0}"
