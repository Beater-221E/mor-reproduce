"""Encode item title+description with frozen Qwen3-Embedding (paper §3.2).

Matches MiniOneRec: concatenate title+description, run Qwen3-Embedding-4B,
L2-normalize. Uses official last-token pooling (not mean-pool).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEFAULT_EMBED_MODEL = "/home/sheng/proj/minionerec/data/models/Qwen3-Embedding-4B"


class ItemTextDataset(Dataset):
    def __init__(self, item_ids: list[str], texts: list[str]):
        self.item_ids = item_ids
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.item_ids[idx], self.texts[idx]


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Qwen3-Embedding official pooling (supports left or right padding)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def parse_devices(device: str) -> list[str]:
    """Parse --device into a list. Accepts cuda:0 / cuda:0,1,2,3 / all."""
    device = device.strip()
    if device in {"all", "cuda"}:
        n = torch.cuda.device_count()
        if n == 0:
            return ["cpu"]
        return [f"cuda:{i}" for i in range(n)]
    parts = [p.strip() for p in device.split(",") if p.strip()]
    out: list[str] = []
    for p in parts:
        if p.startswith("cuda:"):
            rest = p[len("cuda:") :]
            if rest.isdigit() or rest == "":
                out.append(p if rest != "" else "cuda:0")
            else:
                out.append(p)
        elif p.isdigit():
            out.append(f"cuda:{p}")
        else:
            out.append(p)
    return out or ["cuda:0"]


def resolve_embed_model(model_name: str) -> str:
    """Prefer local Qwen3-Embedding-4B; never silently fall back to a generative LM."""
    candidates = [
        model_name,
        DEFAULT_EMBED_MODEL,
        "/home/sheng/data/models/Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-4B",
    ]
    for c in candidates:
        if not c:
            continue
        if c.startswith("Qwen/") or Path(c).exists():
            # local path must look like an embedding checkpoint
            if not c.startswith("Qwen/"):
                cfg = Path(c) / "config.json"
                if not cfg.exists():
                    continue
            return c
    raise FileNotFoundError(
        "Qwen3-Embedding-4B not found. Set --model_name or place weights at "
        f"{DEFAULT_EMBED_MODEL}"
    )


@torch.no_grad()
def _encode_shard(
    item_ids: list[str],
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    desc: str = "text2emb",
) -> tuple[np.ndarray, list[str]]:
    # Qwen3-Embedding docs: left padding + last-token pool
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    # V100 has no native bf16 matmul; use fp16 for inference
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    ds = ItemTextDataset(item_ids, texts)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    embs = []
    ordered_ids: list[str] = []
    pos = 0
    if "cuda:" in device:
        try:
            pos = int(device.split(":")[-1])
        except ValueError:
            pos = 0
    for batch_ids, batch_texts in tqdm(loader, desc=desc, position=pos):
        # document side: raw title+description, no instruct prefix (paper §3.2)
        enc = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        pooled = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=-1)
        embs.append(pooled.float().cpu().numpy())
        ordered_ids.extend(list(batch_ids))

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not embs:
        return np.zeros((0, 1), dtype=np.float32), ordered_ids
    return np.concatenate(embs, axis=0), ordered_ids


def _worker(
    rank: int,
    devices: list[str],
    item_ids: list[str],
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    tmp_dir: str,
) -> None:
    world = len(devices)
    n = len(texts)
    start = rank * n // world
    end = (rank + 1) * n // world
    device = devices[rank]
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[-1]))

    emb, ids = _encode_shard(
        item_ids[start:end],
        texts[start:end],
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        desc=f"text2emb[{device}]",
    )
    np.save(os.path.join(tmp_dir, f"emb.rank{rank}.npy"), emb)
    with open(os.path.join(tmp_dir, f"ids.rank{rank}.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)


def encode_items(
    item_meta_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 4,
    max_length: int = 1024,
    device: str = "all",
) -> None:
    model_name = resolve_embed_model(model_name)
    with open(item_meta_path, encoding="utf-8") as f:
        item_meta = json.load(f)
    item_ids = sorted(item_meta.keys())
    texts = []
    for iid in item_ids:
        title = item_meta[iid].get("title", "")
        desc = item_meta[iid].get("description", "")
        # paper: concatenate title and textual description into one sentence
        texts.append(f"{title} {desc}".strip())

    devices = parse_devices(device)
    print(f"model={model_name}")
    print(f"encoding {len(texts)} items on {devices} (batch_size={batch_size}, max_length={max_length})")

    if len(devices) == 1:
        emb, ordered_ids = _encode_shard(
            item_ids,
            texts,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            device=devices[0],
        )
    else:
        tmp_dir = output_path.parent / f".emb_tmp_{os.getpid()}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            mp.spawn(
                _worker,
                args=(devices, item_ids, texts, model_name, batch_size, max_length, str(tmp_dir)),
                nprocs=len(devices),
                join=True,
            )
            parts = []
            ordered_ids = []
            for rank in range(len(devices)):
                parts.append(np.load(tmp_dir / f"emb.rank{rank}.npy"))
                with open(tmp_dir / f"ids.rank{rank}.json", encoding="utf-8") as f:
                    ordered_ids.extend(json.load(f))
            emb = np.concatenate(parts, axis=0)
        finally:
            for p in tmp_dir.glob("*"):
                p.unlink()
            tmp_dir.rmdir()

    assert ordered_ids == item_ids, "item id order mismatch after multi-GPU merge"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb)
    with open(output_path.with_suffix(".ids.json"), "w", encoding="utf-8") as f:
        json.dump(ordered_ids, f)
    print(f"Saved {emb.shape} -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--batch_size", type=int, default=4)
    # product title+desc; 1024 covers p99 on Amazon subsets (override if needed)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--device",
        type=str,
        default="all",
        help="cuda:0 | cuda:0,1,2,3 | all (data-parallel shard)",
    )
    args = parser.parse_args()
    encode_items(
        args.item_meta,
        args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
