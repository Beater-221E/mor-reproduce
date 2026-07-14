#!/usr/bin/env bash
# Full pipeline for one dataset + size. Usage: bash scripts/all.sh Industrial_and_Scientific 1.5B
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DS="${1:-Industrial_and_Scientific}"
SIZE="${2:-1.5B}"
cd "${ROOT}"
bash scripts/prep.sh
bash scripts/emb.sh
bash scripts/sid.sh
bash scripts/tasks.sh
bash scripts/sft.sh "${DS}" "${SIZE}"
bash scripts/rl.sh "${DS}" "${SIZE}"
bash scripts/eval.sh "${DS}" "${SIZE}" rl
