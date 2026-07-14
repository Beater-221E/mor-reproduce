# MiniOneRec

全参 SFT + 全参 RL。

## 结构

```
minionerec/         
  util.py prep.py emb.py rqvae.py tasks.py
  dataset.py decode.py reward.py model.py
  sft.py rl.py eval.py
scripts/             # download prep emb sid tasks sft rl eval all
configs/             # sft.yaml rl.yaml ds_sft.json ds_rl.json
data/raw|processed|models
checkpoints/ experiments/
```

## 用法

```bash
conda activate mor
cd /path/to/mor-reproduce   # project root
pip install -e .
export PYTHONPATH=$PWD:$PYTHONPATH
export NCCL_IB_DISABLE=1

# 数据已在 data/raw/ 时可跳过 download
bash scripts/prep.sh
# SID: Qwen3-Embedding-4B (title+description) -> RQ-VAE (L=3,K=256)
bash scripts/emb.sh    # needs data/models/Qwen3-Embedding-4B
bash scripts/sid.sh    # METHOD=rqvae (paper); fallback METHOD=rq_kmeans
bash scripts/tasks.sh
bash scripts/sft.sh Industrial_and_Scientific 1.5B
bash scripts/rl.sh  Industrial_and_Scientific 1.5B
bash scripts/eval.sh Industrial_and_Scientific 1.5B rl
```

Embedding：`data/models/Qwen3-Embedding-4B`（论文同款，last-token pool + L2 norm）。  
Backbone：`data/models/Qwen2.5-{0.5B,1.5B,3B}`（见 `configs/sft.yaml`；可把权重放到该目录或做软链）。
