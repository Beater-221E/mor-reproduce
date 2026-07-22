#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/env_mor.sh"

CFG="${1:-configs/rl/official.yaml}"
shift || true

# shellcheck disable=SC2086
$TORCHRUN --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29512}" \
  -m minionerec.train_rl --config "${CFG}" "$@"
