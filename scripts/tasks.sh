#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for ds in Industrial_and_Scientific Office_Products; do
  python -m minionerec.tasks \
    --data_dir "${ROOT}/data/processed/${ds}" \
    --sid_map "${ROOT}/data/processed/${ds}/sid/sid_map.json" \
    --out_dir "${ROOT}/data/processed/${ds}/tasks"
done
