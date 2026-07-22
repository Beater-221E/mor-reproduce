#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/env_mor.sh"

# Eval defaults to single process; override with CUDA_VISIBLE_DEVICES if needed.
CFG="${1:-configs/eval/base.yaml}"
shift || true
"$PYTHON" -m minionerec.evaluate --config "${CFG}" "$@"
