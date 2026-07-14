#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Paper: Qwen3-Embedding-4B (frozen). Do not fall back to generative Qwen2.5.
# Override with EMBED_MODEL=/path/to/weights if needed.
MODEL="${EMBED_MODEL:-${ROOT}/data/models/Qwen3-Embedding-4B}"
if [[ ! -f "${MODEL}/config.json" || ! -f "${MODEL}/model.safetensors.index.json" ]]; then
  echo "ERROR: Qwen3-Embedding-4B not found at ${MODEL}. Place weights under data/models/ or set EMBED_MODEL." >&2
  exit 1
fi

# Fail fast if safetensors still uploading / truncated
python - <<PY
from pathlib import Path
from safetensors import safe_open
p = Path("${MODEL}")
for f in sorted(p.glob("model-*.safetensors")):
    with safe_open(f, framework="pt") as s:
        _ = list(s.keys())[:1]
print("weights ok:", p)
PY

echo "emb model: ${MODEL}"
DEVICE="${DEVICE:-all}"
# 4B fp16 ~8GB; leave headroom on 16GB V100
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

for ds in Industrial_and_Scientific Office_Products; do
  python -m minionerec.emb \
    --item_meta "${ROOT}/data/processed/${ds}/item_meta.json" \
    --output "${ROOT}/data/processed/${ds}/item_emb.npy" \
    --model_name "${MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --device "${DEVICE}"
done
