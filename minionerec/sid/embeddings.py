"""Encode item title+description with a frozen text encoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class ItemTextDataset(Dataset):
    def __init__(self, item_ids: list[str], texts: list[str]):
        self.item_ids = item_ids
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.item_ids[idx], self.texts[idx]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


@torch.no_grad()
def encode_items(
    item_meta_path: Path,
    output_path: Path,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    batch_size: int = 4,
    max_length: int = 256,
    device: str = "cuda:0",
) -> None:
    with open(item_meta_path, encoding="utf-8") as f:
        item_meta = json.load(f)
    item_ids = sorted(item_meta.keys())
    texts = []
    for iid in item_ids:
        title = item_meta[iid].get("title", "")
        desc = item_meta[iid].get("description", "")
        texts.append(f"{title} {desc}".strip())

    print(
        f"[text2emb] model={model_name} items={len(item_ids)} "
        f"batch_size={batch_size} max_length={max_length} device={device}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Prefer `dtype=` (transformers deprecated `torch_dtype`).
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float16)
    model.to(device)
    model.eval()
    print(f"[text2emb] model loaded on {device}", flush=True)

    ds = ItemTextDataset(item_ids, texts)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_batches = len(loader)
    # Plain newline logs survive narrow terminals / Cursor snapshots (tqdm \\r bars get truncated).
    log_every = max(1, min(50, total_batches // 20 or 1))

    embs = []
    ordered_ids = []
    t0 = time.perf_counter()
    for step, (batch_ids, batch_texts) in enumerate(loader, start=1):
        enc = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        embs.append(pooled.float().cpu().numpy())
        ordered_ids.extend(list(batch_ids))

        if step == 1 or step % log_every == 0 or step == total_batches:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed if elapsed > 0 else 0.0
            remain = (total_batches - step) / rate if rate > 0 else float("inf")
            pct = 100.0 * step / total_batches if total_batches else 100.0
            print(
                f"[text2emb] {step}/{total_batches} ({pct:.1f}%) "
                f"items={len(ordered_ids)}/{len(item_ids)} "
                f"elapsed={elapsed:.0f}s rate={rate:.2f}it/s eta={remain:.0f}s",
                flush=True,
            )

    emb = np.concatenate(embs, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb)
    with open(output_path.with_suffix(".ids.json"), "w", encoding="utf-8") as f:
        json.dump(ordered_ids, f)
    print(f"[text2emb] Saved {emb.shape} -> {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
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
