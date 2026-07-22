"""Multi-task SFT dataset with on-disk token cache for throughput."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# process-pool globals (set in worker initializer)
_POOL_TOK = None
_POOL_CUTOFF = None
_POOL_USE_CHAT = None


def _pool_init(tokenizer_dir: str, cutoff_len: int):
    global _POOL_TOK, _POOL_CUTOFF, _POOL_USE_CHAT
    import logging
    import os

    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)

    _POOL_TOK = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    _POOL_CUTOFF = cutoff_len
    _POOL_USE_CHAT = hasattr(_POOL_TOK, "apply_chat_template") and getattr(
        _POOL_TOK, "chat_template", None
    )


def _as_id_list(x) -> list[int]:
    """Normalize tokenizer outputs to a plain list[int] (HF may return BatchEncoding)."""
    if x is None:
        return []
    if isinstance(x, dict) or hasattr(x, "input_ids"):
        ids = x["input_ids"] if "input_ids" in x else getattr(x, "input_ids", x)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        # BatchEncoding / nested batch
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, list) and x and isinstance(x[0], list):
        x = x[0]
    return list(x)


def tokenize_messages(tokenizer, messages: list[dict], cutoff_len: int, use_chat: bool) -> dict:
    if use_chat:
        full_ids = _as_id_list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=cutoff_len,
            )
        )
        prompt_ids = _as_id_list(
            tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=cutoff_len,
            )
        )
    else:
        prompt_text = f"User: {messages[0]['content']}\nAssistant:"
        full_text = prompt_text + f" {messages[1]['content']}" + (tokenizer.eos_token or "")
        full_ids = _as_id_list(
            tokenizer(
                full_text,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                add_special_tokens=True,
            )["input_ids"]
        )
        prompt_ids = _as_id_list(
            tokenizer(
                prompt_text,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                add_special_tokens=True,
            )["input_ids"]
        )

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _pool_tokenize_row(row: dict) -> dict:
    return tokenize_messages(_POOL_TOK, row["messages"], _POOL_CUTOFF, _POOL_USE_CHAT)


def _cache_key(path: Path, cutoff_len: int, vocab_size: int) -> str:
    st = path.stat()
    # v2: store plain list[int] ids (not BatchEncoding)
    raw = f"v2|{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{cutoff_len}|{vocab_size}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def build_token_cache(
    jsonl_path: Path,
    tokenizer,
    cutoff_len: int,
    cache_path: Path,
    tokenizer_dir: Path,
    num_workers: int | None = None,
) -> list[dict]:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    workers = num_workers or min(32, max(1, (os.cpu_count() or 8) - 2))
    use_chat = hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)
    print(f"[sft-cache] tokenizing {len(rows)} rows -> {cache_path} (workers={workers})", flush=True)
    t0 = time.time()

    cached: list[dict]
    if workers <= 1:
        cached = [
            tokenize_messages(tokenizer, r["messages"], cutoff_len, use_chat)
            for r in tqdm(rows, desc="tokenize")
        ]
    else:
        cached = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_pool_init,
            initargs=(str(tokenizer_dir), cutoff_len),
        ) as ex:
            for item in tqdm(ex.map(_pool_tokenize_row, rows, chunksize=64), total=len(rows), desc="tokenize"):
                cached.append(item)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save({"cutoff_len": cutoff_len, "n": len(cached), "data": cached}, tmp)
    tmp.replace(cache_path)
    print(f"[sft-cache] wrote {len(cached)} examples in {time.time() - t0:.1f}s", flush=True)
    return cached


def pack_examples(examples: list[dict], cutoff_len: int) -> list[dict]:
    """Greedy pack short examples into cutoff_len sequences to cut padding waste."""
    packed: list[dict] = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []
    cur_mask: list[int] = []

    def _flush():
        nonlocal cur_ids, cur_labels, cur_mask
        if not cur_ids:
            return
        packed.append(
            {
                "input_ids": cur_ids,
                "attention_mask": cur_mask,
                "labels": cur_labels,
            }
        )
        cur_ids, cur_labels, cur_mask = [], [], []

    for ex in examples:
        ids = list(ex["input_ids"])
        labs = list(ex["labels"])
        if len(ids) > cutoff_len:
            ids = ids[:cutoff_len]
            labs = labs[:cutoff_len]
        if cur_ids and len(cur_ids) + len(ids) > cutoff_len:
            _flush()
        cur_ids.extend(ids)
        cur_labels.extend(labs)
        cur_mask.extend([1] * len(ids))
    _flush()
    return packed


class SFTData(Dataset):
    """SFT dataset. Prefer pretokenized cache to keep GPUs fed."""

    def __init__(
        self,
        path: Path,
        tokenizer,
        cutoff_len: int = 512,
        cache_dir: Path | None = None,
        tokenizer_dir: Path | None = None,
        use_cache: bool = True,
        num_proc: int | None = None,
        pack: bool = False,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.use_chat = hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        )
        self._cached: list[dict] | None = None
        self.rows: list[dict] | None = None

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if use_cache:
            cdir = Path(cache_dir) if cache_dir else self.path.parent / ".tok_cache"
            key = _cache_key(self.path, cutoff_len, len(tokenizer))
            cache_path = cdir / f"{self.path.stem}.{key}.pt"
            tok_dir = Path(tokenizer_dir) if tokenizer_dir else None

            if local_rank == 0 and not cache_path.exists():
                if tok_dir is None:
                    raise ValueError("tokenizer_dir required to build token cache")
                build_token_cache(self.path, tokenizer, cutoff_len, cache_path, tok_dir, num_workers=num_proc)

            if world_size > 1:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                else:
                    while not cache_path.exists():
                        time.sleep(1)

            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            data = blob["data"]
            n_raw = len(data)
            if pack:
                data = pack_examples(data, cutoff_len)
                if local_rank == 0:
                    print(
                        f"[sft-cache] packed {n_raw} -> {len(data)} "
                        f"(~{n_raw / max(1, len(data)):.2f}x denser)",
                        flush=True,
                    )
            self._cached = data
            if local_rank == 0:
                print(f"[sft-cache] loaded {len(self._cached)} from {cache_path.name}", flush=True)
        else:
            self.rows = []
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.rows.append(json.loads(line))
            if pack:
                # pack after on-the-fly is awkward; materialize once
                use_chat = self.use_chat
                raw = [
                    tokenize_messages(tokenizer, r["messages"], cutoff_len, use_chat)
                    for r in self.rows
                ]
                self._cached = pack_examples(raw, cutoff_len)
                self.rows = None

    def __len__(self):
        return len(self._cached) if self._cached is not None else len(self.rows or [])

    def __getitem__(self, idx):
        if self._cached is not None:
            return self._cached[idx]
        assert self.rows is not None
        return tokenize_messages(
            self.tokenizer, self.rows[idx]["messages"], self.cutoff_len, self.use_chat
        )


def collate(features, pad_token_id: int):
    def _ids(x):
        if hasattr(x, "tolist"):
            x = x.tolist()
        if isinstance(x, dict) or hasattr(x, "input_ids"):
            x = x["input_ids"] if "input_ids" in x else x.input_ids
            if hasattr(x, "tolist"):
                x = x.tolist()
        return list(x)

    max_len = max(len(_ids(f["input_ids"])) for f in features)
    max_len = (max_len + 7) // 8 * 8
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        ids = _ids(f["input_ids"])
        labs = _ids(f["labels"])
        mask = _ids(f["attention_mask"]) if "attention_mask" in f else [1] * len(ids)
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append(mask + [0] * pad)
        labels.append(labs + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class RLData(Dataset):
    """Dataset yielding prompts + gold answers for GRPO."""

    def __init__(self, path: Path, tokenizer, cutoff_len: int = 512):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        messages = row["messages"]
        user_msgs = messages[:-1]
        use_chat = hasattr(self.tokenizer, "apply_chat_template") and getattr(
            self.tokenizer, "chat_template", None
        )
        if use_chat:
            prompt = self.tokenizer.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"User: {user_msgs[0]['content']}\nAssistant:"
        return {
            "prompt": prompt,
            "answer": row.get("answer", messages[-1]["content"]),
            "task": row.get("task", "generative_retrieval"),
        }
