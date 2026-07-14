"""SID trie constrained decoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from transformers import LogitsProcessor

from minionerec.util import NUM_CODEBOOK_LAYERS, SID_LAYER_PREFIXES, parse_sid, sid_token


class SIDTrie:
    """Prefix trie over valid SID token-id sequences."""

    def __init__(self):
        self.root: dict = {}
        self.sequences: list[list[int]] = []

    def insert(self, token_ids: list[int]) -> None:
        node = self.root
        for tid in token_ids:
            if tid not in node:
                node[tid] = {}
            node = node[tid]
        node["__end__"] = True
        self.sequences.append(token_ids)

    def allowed(self, prefix: list[int]) -> set[int]:
        node = self.root
        for tid in prefix:
            if tid not in node:
                return set()
            node = node[tid]
        return {k for k in node.keys() if k != "__end__"}


def make_trie(tokenizer, sid_map: dict) -> SIDTrie:
    trie = SIDTrie()
    for info in sid_map.values():
        sid_text = info["sid"]
        # encode without special tokens; SID tokens are atomic after add_tokens
        ids = tokenizer.encode(sid_text, add_special_tokens=False)
        if len(ids) != NUM_CODEBOOK_LAYERS:
            # fallback: encode each token separately
            codes = info["codes"]
            ids = []
            for layer, code in enumerate(codes):
                tok = sid_token(layer, code)
                tid = tokenizer.convert_tokens_to_ids(tok)
                ids.append(tid)
        trie.insert(ids)
    return trie


def load_sids(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class SIDConstraint(LogitsProcessor):
    """
    Mask logits so that generation can only follow valid SID paths in the trie.
    Assumes the model generates SID tokens immediately as the answer.
    """

    def __init__(
        self,
        trie: SIDTrie,
        tokenizer,
        prompt_lengths: list[int] | None = None,
        eos_token_id: int | None = None,
    ):
        self.trie = trie
        self.tokenizer = tokenizer
        self.prompt_lengths = prompt_lengths
        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    def set_prompt_lengths(self, prompt_lengths: list[int]) -> None:
        self.prompt_lengths = prompt_lengths

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        for b in range(batch_size):
            if self.prompt_lengths is not None:
                plen = self.prompt_lengths[b] if b < len(self.prompt_lengths) else self.prompt_lengths[0]
            else:
                plen = 0
            gen = input_ids[b, plen:].tolist()
            # only constrain first NUM_CODEBOOK_LAYERS tokens of the answer
            if len(gen) >= NUM_CODEBOOK_LAYERS:
                # force EOS afterwards
                mask = torch.full_like(scores[b], float("-inf"))
                mask[self.eos_token_id] = 0
                scores[b] = mask
                continue
            allowed = self.trie.allowed(gen)
            if not allowed:
                # no legal continuation: force EOS
                mask = torch.full_like(scores[b], float("-inf"))
                mask[self.eos_token_id] = 0
                scores[b] = mask
            else:
                mask = torch.full_like(scores[b], float("-inf"))
                idx = torch.tensor(list(allowed), device=scores.device, dtype=torch.long)
                mask[idx] = 0
                scores[b] = scores[b] + mask
        return scores


def sid_to_item(sid_text: str, sid_map: dict) -> str | None:
    codes = parse_sid(sid_text)
    if codes is None:
        return None
    key = tuple(codes)
    for iid, info in sid_map.items():
        if tuple(info["codes"]) == key:
            return iid
    # also match by sid string
    for iid, info in sid_map.items():
        if info["sid"] in sid_text:
            return iid
    return None
