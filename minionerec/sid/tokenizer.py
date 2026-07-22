"""Tokenizer / embedding setup aligned with official sft.py TokenExtender."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minionerec.sid.map_io import used_sid_tokens


def load_base_tokenizer(model_name_or_path: str):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    # Official: pad = eos, left padding
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def extend_tokenizer_with_sid(
    tokenizer,
    sid_map: dict | None = None,
    index_json: Path | str | None = None,
    *,
    add_all_codebook: bool = False,
) -> tuple[Any, list[str], int]:
    """
    Official TokenExtender: add only tokens appearing in the index.
    Optionally add full codebook (legacy mor-reproduce behavior) via flag.
    Uses tokenizer.add_tokens (official), not add_special_tokens.
    """
    original = len(tokenizer)
    if add_all_codebook:
        from minionerec.sid.codec import all_sid_tokens

        new_tokens = all_sid_tokens()
    elif sid_map is not None:
        new_tokens = used_sid_tokens(sid_map)
    elif index_json is not None:
        with open(index_json, encoding="utf-8") as f:
            index = json.load(f)
        toks: set[str] = set()
        for sids in index.values():
            for t in sids:
                toks.add(t)
        new_tokens = sorted(toks)
    else:
        raise ValueError("Need sid_map or index_json")
    n_added = tokenizer.add_tokens(new_tokens)
    return tokenizer, new_tokens, n_added


def load_model_for_sft(
    model_name_or_path: str,
    tokenizer,
    *,
    torch_dtype=None,
    device_map=None,
    freeze_llm: bool = False,
    v100_fp16: bool = True,
):
    """
    Load causal LM and resize embeddings.

    Important (V100 / HF Trainer fp16): do NOT load weights as float16 when
    TrainingArguments.fp16=True — GradScaler cannot unscale FP16 parameter grads.
    Keep parameters in float32 and let autocast handle compute dtype.
    """
    # Prefer float32 params; Trainer fp16/bf16 autocast handles compute.
    if torch_dtype is None:
        torch_dtype = torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    original_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    emb_shape = tuple(model.get_input_embeddings().weight.shape)
    lm_head = model.get_output_embeddings()
    lm_shape = tuple(lm_head.weight.shape) if lm_head is not None else None

    if freeze_llm:
        for p in model.parameters():
            p.requires_grad = False
        emb = model.get_input_embeddings()
        emb.weight.requires_grad = True

        def mask_grad(grad):
            grad[:original_vocab].zero_()
            return grad

        emb.weight.register_hook(mask_grad)

    info = {
        "original_vocab_size": original_vocab,
        "new_vocab_size": len(tokenizer),
        "number_of_sid_tokens": len(tokenizer) - original_vocab,
        "new_embedding_shape": list(emb_shape),
        "lm_head_shape": list(lm_shape) if lm_shape else None,
        "freeze_llm": freeze_llm,
        "torch_dtype": str(torch_dtype),
        "v100_fp16_compute": v100_fp16,
    }
    return model, info


def tokenizer_roundtrip_test(tokenizer, sid_strings: list[str], max_n: int = 200) -> dict:
    fails = []
    for sid in sid_strings[:max_n]:
        ids = tokenizer.encode(sid, add_special_tokens=False)
        import re

        toks = [m.group(0) for m in re.finditer(r"<(a|b|c)_(\d+)>", sid)]
        if len(ids) != len(toks):
            # try encode each token
            ids = [tokenizer.convert_tokens_to_ids(t) for t in toks]
        recovered = [tokenizer.convert_ids_to_tokens(i) for i in ids]
        if recovered != toks or any(i is None or i < 0 for i in ids):
            fails.append({"sid": sid, "ids": ids, "recovered": recovered, "expected": toks})
    return {"n_tested": min(max_n, len(sid_strings)), "n_failed": len(fails), "failures": fails[:20]}
