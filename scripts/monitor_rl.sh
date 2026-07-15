#!/usr/bin/env bash
# Usage: bash scripts/monitor_rl.sh [run_dir] [interval_sec]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
RUN_DIR="${1:-checkpoints/rl_Industrial_and_Scientific_1.5B}"
INTERVAL="${2:-30}"
python scripts/monitor_rl.py --run-dir "${RUN_DIR}" --interval "${INTERVAL}"
