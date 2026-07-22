from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(
    not (ROOT / "data/official_format/Industrial_and_Scientific").exists(),
    reason="need exported official format",
)
def test_constrained_prefix_nonempty_for_response_root():
    from transformers import AutoTokenizer

    from minionerec.constrained_decode_official import (
        build_sid_hash_dict,
        get_hash,
        load_semantic_ids_from_info,
        make_prefix_fn,
    )
    from minionerec.tokenizer_official import extend_tokenizer_with_sid, load_base_tokenizer
    from minionerec.sid.map_io import load_sid_map

    tok = load_base_tokenizer(str(ROOT / "data/models/Qwen2.5-0.5B"))
    sid_map = load_sid_map(ROOT / "data/processed/Industrial_and_Scientific/sid/sid_map.json")
    tok, _, _ = extend_tokenizer_with_sid(tok, sid_map=sid_map)
    info = ROOT / "data/official_format/Industrial_and_Scientific/Industrial_and_Scientific.info.txt"
    sids = load_semantic_ids_from_info(str(info))[:100]
    hd = build_sid_hash_dict(tok, sids, base_model="qwen")
    fn = make_prefix_fn(hd)
    # first step uses last 3 tokens of '### Response:\n' encoding
    prefix = tok.encode("### Response:\n", add_special_tokens=False)
    allowed = fn(0, prefix[-3:])
    assert len(allowed) > 0
