#!/usr/bin/env bash
# Multi/single GPU SFT (official_source). Uses mor conda + torchrun.
#   NPROC=1 bash scripts/sft_0.5b_official.sh          # single GPU
#   NPROC=4 bash scripts/sft_0.5b_official.sh          # 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/env_mor.sh"

CFG="${1:-configs/sft/base.yaml}"
shift || true

# shellcheck disable=SC2086
$TORCHRUN --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  -m minionerec.train_sft --config "${CFG}" "$@"
