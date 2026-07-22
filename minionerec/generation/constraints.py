"""
Official constrained decoding.

Source: MiniOneRec-official @ 0c64b955
  - LogitProcessor.py: ConstrainedLogitsProcessor
  - evaluate.py / minionerec_trainer.py: hash_dict over '### Response:\\n' + SID (+ EOS)

prefix_index = 3 for non-GPT2 models (tokens of '### Response:\\n' prefix used as root key).
"""

from __future__ import annotations

import warnings
from typing import Callable, List

import torch
from transformers.generation import LogitsProcessor


def get_hash(x) -> str:
    return "-".join(str(_) for _ in x)


class ConstrainedLogitsProcessor(LogitsProcessor):
    """Port of official LogitProcessor.ConstrainedLogitsProcessor."""

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, list], List[int]],
        num_beams: int,
        base_model: str = "",
        eos_token_id: int | None = None,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count = 0
        self.base_model = base_model or ""
        self.eos_token_id = eos_token_id
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index :]
                else:
                    hash_key = sent[-self.count :]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)
                if len(prefix_allowed_tokens) == 0:
                    warnings.warn(
                        f"No valid tokens for hash_key={hash_key} at step={self.count}."
                    )
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0
        self.count += 1
        return scores + mask

    def reset(self) -> None:
        self.count = 0


def build_sid_hash_dict(tokenizer, semantic_ids: list[str], base_model: str = "") -> dict[str, list[int]]:
    """
    Build prefix->allowed next token map from SID strings.
    semantic_ids should already include trailing newline as in official info parsing,
    or plain SID; we format as '### Response:\\n{sid}'.
    """
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3

    info_semantic = [f"### Response:\n{_}" if not _.startswith("###") else _ for _ in semantic_ids]
    if "llama" in base_model.lower():
        prefix_id = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefix_id = [tokenizer(_).input_ids for _ in info_semantic]

    hash_dict: dict[str, set[int]] = {}
    for ids in prefix_id:
        ids = list(ids) + [tokenizer.eos_token_id]
        for i in range(prefix_index, len(ids)):
            if i == prefix_index:
                h = get_hash(ids[:i])
            else:
                h = get_hash(ids[prefix_index:i])
            hash_dict.setdefault(h, set()).add(ids[i])
    return {k: list(v) for k, v in hash_dict.items()}


def make_prefix_fn(hash_dict: dict[str, list[int]]):
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        h = get_hash(input_ids)
        return hash_dict.get(h, [])

    return prefix_allowed_tokens_fn


def load_semantic_ids_from_info(info_file: str) -> list[str]:
    ids = []
    with open(info_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid = line.split("\t")[0].strip() + "\n"
            ids.append(sid)
    return ids


class SidConstraint:
    """
    Unified constrained-decoding facade used by RL and evaluation.

    Internally uses the official hash_dict + ConstrainedLogitsProcessor path
    (behavior must match MiniOneRec-official @ 0c64b955).
    """

    def __init__(self, hash_dict: dict, base_model: str, eos_token_id: int | None):
        self.hash_dict = hash_dict
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        self._prefix_fn = make_prefix_fn(hash_dict)

    @classmethod
    def from_semantic_ids(cls, tokenizer, semantic_ids: list[str], base_model: str = "") -> "SidConstraint":
        hd = build_sid_hash_dict(tokenizer, semantic_ids, base_model=base_model)
        return cls(hd, base_model=base_model, eos_token_id=tokenizer.eos_token_id)

    @classmethod
    def from_info_file(cls, tokenizer, info_file: str, base_model: str = "") -> "SidConstraint":
        return cls.from_semantic_ids(tokenizer, load_semantic_ids_from_info(info_file), base_model=base_model)

    def allowed_token_ids(self, batch_id: int, input_ids) -> list[int]:
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        return self._prefix_fn(batch_id, input_ids)

    def logits_processor(self, num_beams: int = 1) -> ConstrainedLogitsProcessor:
        return ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=self._prefix_fn,
            num_beams=num_beams,
            base_model=self.base_model,
            eos_token_id=self.eos_token_id,
        )
