"""Ensure title2sid prompts do not contain the SID answer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(
    not (ROOT / "data/official_format/Industrial_and_Scientific/train.csv").exists(),
    reason="official format not exported yet",
)
def test_title2sid_no_sid_in_prompt():
    from transformers import AutoTokenizer

    from minionerec.official_data import SidItemFeatDataset
    from minionerec.tokenizer_official import extend_tokenizer_with_sid, load_base_tokenizer
    from minionerec.sid.map_io import load_sid_map

    base = ROOT / "data/models/Qwen2.5-0.5B"
    tok = load_base_tokenizer(str(base))
    sid_map = load_sid_map(ROOT / "data/processed/Industrial_and_Scientific/sid/sid_map.json")
    tok, _, _ = extend_tokenizer_with_sid(tok, sid_map=sid_map)
    paths = ROOT / "data/official_format/Industrial_and_Scientific"
    ds = SidItemFeatDataset(
        str(paths / "Industrial_and_Scientific.item.json"),
        str(paths / "Industrial_and_Scientific.index.json"),
        tokenizer=tok,
        max_len=512,
        sample=64,
        seed=0,
    )
    for i in range(min(32, len(ds))):
        item = ds[i]
        if item.get("task") != "title2sid":
            continue
        ids, labels = item["input_ids"], item["labels"]
        prompt_ids = [t for t, y in zip(ids, labels) if y == -100]
        resp_ids = [y for y in labels if y != -100]
        prompt = tok.decode(prompt_ids)
        resp = tok.decode(resp_ids).strip()
        # response SID must not appear in prompt
        assert resp.replace("\n", "") not in prompt
