# MiniOneRec Reproduction on Amazon23

## 1. Overview

This project reproduces the MiniOneRec training pipeline on **Amazon Reviews 2023**.

Supports:

- Amazon23 preprocessing
- Item embedding + SID generation
- SFT training
- GRPO-based RL (`official` / `paper` configuration variants)
- Constrained SID generation
- HR@K / NDCG@K evaluation

Default smoke model: **Qwen2.5-0.5B**. Default category: **Industrial_and_Scientific**.  
Source layout: [docs/architecture.md](docs/architecture.md).

---

## 2. Environment Setup

```bash
git clone <REPO_URL> mor-reproduce
cd mor-reproduce

conda create -n mor python=3.11 -y
conda activate mor

# PyTorch for your CUDA (example: CUDA 11.8)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -e .
```

| Item | Value |
| --- | --- |
| Python | ≥ 3.11 |
| PyTorch | 2.x + CUDA |
| GPU | NVIDIA (`torchrun`; 4× V100 validated) |

```bash
source scripts/env.sh   # required for multi-GPU: NCCL + mor python + NPROC
```

Multi-GPU without this often hangs at torchrun startup (GPUs stuck ~100% util / ~500MiB, no `[sft]` logs). Prefer `bash scripts/sft_0.5b_official.sh`, or:

```bash
source scripts/env.sh
torchrun --standalone --nproc_per_node "$NPROC" --master_port "$MASTER_PORT" \
  -m minionerec.cli train-sft --config configs/sft/base.yaml
```

Place weights under `data/models/` (e.g. `Qwen2.5-0.5B`, `Qwen3-Embedding-4B`).

---

## 3. Project Workflow

```text
Amazon23 Raw Data
        |
        v
Data Preprocessing
        |
        v
Item Embeddings → SID Generation
        |
        v
SFT Training → RL Training → Evaluation
```

---

## 4. Data Preparation

```bash
bash scripts/download.sh
```

Input: `data/raw/`

```bash
python -m minionerec.cli prepare-data \
  --config configs/data/amazon23.yaml
```

Output: `data/processed/<dataset>/`

---

## 5. SID Construction

```bash
python -m minionerec.cli build-embeddings \
  --config configs/sid/default.yaml

python -m minionerec.cli build-sid \
  --config configs/sid/default.yaml
```

Default method is `residual_kmeans` (PCA → 3-layer KMeans). With `enforce_unique: true`, the last code is reassigned inside each `(c0,c1)` prefix so raw SID collision is 0 while keeping prefix semantics. Output: `data/processed/<dataset>/sid/` (`sid_map.json`, `best_metrics.json`).

Full data+SID chain:

```bash
python -m minionerec.cli run-pipeline \
  --data-config configs/data/amazon23.yaml \
  --sid-config configs/sid/default.yaml
```

---

## 6. SFT Training

```bash
python -m minionerec.cli validate \
  --config configs/data/amazon23.yaml
```

Checks SID coverage, collisions, and split statistics (writes `artifacts/*_validation.json`).

Smoke:

```bash
torchrun --standalone --nproc_per_node 1 \
  -m minionerec.cli train-sft \
  --config configs/sft/smoke.yaml
```

Full:

```bash
torchrun --standalone --nproc_per_node 4 \
  -m minionerec.cli train-sft \
  --config configs/sft/base.yaml
```

Output: `checkpoints/sft_*` (`best_checkpoint/`, `final_checkpoint/`). V100 configs use FP32 fallback for SID embeddings.

---

## 7. RL Training

Released-implementation objective:

```bash
torchrun --standalone --nproc_per_node 4 \
  -m minionerec.cli train-rl \
  --config configs/rl/official.yaml
```

Paper-aligned objective:

```bash
torchrun --standalone --nproc_per_node 4 \
  -m minionerec.cli train-rl \
  --config configs/rl/paper.yaml
```

Smoke:

```bash
torchrun --standalone --nproc_per_node 2 \
  -m minionerec.cli train-rl \
  --config configs/rl/smoke.yaml
```

`official` follows the released implementation; `paper` follows the algorithm described in the paper.  
Point the RL config model path at your SFT checkpoint. Output: `checkpoints/rl_*`.

---

## 8. Evaluation

```bash
python -m minionerec.cli evaluate \
  --config configs/eval/default.yaml
```

Set `model_name_or_path` to the checkpoint. Typical output:

```text
experiments/eval/
├── eval_metrics.json
└── generations.json
```

---

## 9. Experiment Configuration

```text
configs/
├── data/     # preprocess / validate paths
├── model/    # base LLM defaults
├── sid/      # embedding + SID hyperparameters
├── sft/      # SFT smoke / full
├── rl/       # RL base + objective variants
└── eval/     # evaluation
```

YAML may use `extends:`. Edit YAML for hyperparameters.

---

## 10. Output Structure

```text
data/{raw,processed,official_format,models}/
checkpoints/          # SFT / RL checkpoints
experiments/          # eval metrics & generations
artifacts/            # validation reports
```

---

## 11. Troubleshooting

| Issue | Action |
| --- | --- |
| Single GPU only | `unset CUDA_VISIBLE_DEVICES`; set `--nproc_per_node` |
| NaN on V100 | Keep FP32 fallback in configs |
| Embedding build fails | Check `embed_model` in `configs/sid/default.yaml` |
| Invalid RL SIDs | Verify SID map + constrained decode; discard run if invalid rate &gt; 0 |

```bash
pytest -q
```
