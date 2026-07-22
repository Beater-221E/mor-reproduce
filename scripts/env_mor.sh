#!/usr/bin/env bash
# Shared env for MiniOneRec alignment scripts.
# Always use the mor conda environment.

# Resolve mor python (prefer explicit conda env path).
if [[ -x "${HOME}/.conda/envs/mor/bin/python" ]]; then
  MOR_PYTHON="${HOME}/.conda/envs/mor/bin/python"
elif [[ -x "/home/sheng/.conda/envs/mor/bin/python" ]]; then
  MOR_PYTHON="/home/sheng/.conda/envs/mor/bin/python"
else
  echo "[env_mor] ERROR: mor conda env not found at ~/.conda/envs/mor" >&2
  exit 1
fi

# Allow override, but default strictly to mor.
export PYTHON="${PYTHON:-$MOR_PYTHON}"
export PATH="$(dirname "$PYTHON"):${PATH}"
# Prefer mor's torchrun (same env as PYTHON); fall back to python -m.
if [[ -x "$(dirname "$PYTHON")/torchrun" ]]; then
  export TORCHRUN="$(dirname "$PYTHON")/torchrun"
else
  export TORCHRUN="$PYTHON -m torch.distributed.run"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NCCL knobs for multi-V100 (same as legacy scripts/sft.sh)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp193s0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp193s0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Do NOT default CUDA_VISIBLE_DEVICES to 0 — that forced single-GPU previously.
# User may still set CUDA_VISIBLE_DEVICES explicitly to subset GPUs.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _VISIBLE_COUNT="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
else
  _VISIBLE_COUNT="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
fi
export NPROC="${NPROC:-${_VISIBLE_COUNT}}"
export MASTER_PORT="${MASTER_PORT:-29511}"

echo "[env_mor] PYTHON=$PYTHON"
echo "[env_mor] TORCHRUN=$TORCHRUN"
echo "[env_mor] NPROC=$NPROC CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>} MASTER_PORT=$MASTER_PORT"
