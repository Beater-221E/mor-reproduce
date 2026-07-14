#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for ds in Industrial_and_Scientific Office_Products; do
  python -m minionerec.prep --dataset "${ds}" \
    --raw_dir "${ROOT}/data/raw" --out_dir "${ROOT}/data/processed"
done
