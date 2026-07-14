"""Tokenizer / model utilities: add SID tokens and resize embeddings."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from minionerec.util import all_sid_tokens


def load_tokenizer(model_name: str, sid_tokens: list[str] | None = None):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tokens = sid_tokens if sid_tokens is not None else all_sid_tokens()
    n_added = tok.add_special_tokens({"additional_special_tokens": tokens})
    return tok, n_added


def load_tokenizer_from_dir(path: Path):
    """Load an already-saved tokenizer (preferred when resuming)."""
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str, tokenizer, torch_dtype=None, device_map=None):
    kwargs = {"trust_remote_code": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    # Match HF/DeepSpeed checkpoints that pad vocab to multiple of 8.
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    if hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id
    return model


def save_tok(tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(path)
