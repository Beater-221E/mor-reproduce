#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Paper: neural RQ-VAE on Qwen3-Embedding vectors (L=3, K=256).
# Fallback: METHOD=rq_kmeans if needed.
METHOD="${METHOD:-rqvae}"
DEVICE="${DEVICE:-cuda:0}"
LR="${LR:-3e-4}"

for ds in Industrial_and_Scientific Office_Products; do
  python -m minionerec.rqvae \
    --emb_path "${ROOT}/data/processed/${ds}/item_emb.npy" \
    --ids_path "${ROOT}/data/processed/${ds}/item_emb.ids.json" \
    --out_dir "${ROOT}/data/processed/${ds}/sid" \
    --method "${METHOD}" \
    --pca_dim "${PCA_DIM:-256}" \
    --epochs "${EPOCHS:-10000}" \
    --batch_size "${BATCH_SIZE:-2048}" \
    --lr "${LR}" \
    --device "${DEVICE}"
done
