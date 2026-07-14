"""Shared constants and helpers."""

from __future__ import annotations

DATASETS = {
    "Industrial_and_Scientific": {
        # Amazon Reviews'23 filenames (https://amazon-reviews-2023.github.io/)
        "review_file": "Industrial_and_Scientific.jsonl.gz",
        "meta_file": "meta_Industrial_and_Scientific.jsonl.gz",
        # default window for amazon23 MiniOneRec script
        "st_year": 2018,
        "st_month": 10,
        "ed_year": 2023,
        "ed_month": 9,
    },
    "Office_Products": {
        "review_file": "Office_Products.jsonl.gz",
        "meta_file": "meta_Office_Products.jsonl.gz",
        "st_year": 2018,
        "st_month": 10,
        "ed_year": 2023,
        "ed_month": 9,
    },
}

SID_LAYER_PREFIXES = ("a", "b", "c")
CODEBOOK_SIZE = 256
NUM_CODEBOOK_LAYERS = 3


def sid_token(layer: int, code: int) -> str:
    return f"<{SID_LAYER_PREFIXES[layer]}_{code}>"


def all_sid_tokens() -> list[str]:
    tokens = []
    for layer in range(NUM_CODEBOOK_LAYERS):
        for code in range(CODEBOOK_SIZE):
            tokens.append(sid_token(layer, code))
    return tokens


def format_sid(codes: list[int] | tuple[int, ...]) -> str:
    assert len(codes) == NUM_CODEBOOK_LAYERS
    return "".join(sid_token(i, int(c)) for i, c in enumerate(codes))


def parse_sid(text: str) -> list[int] | None:
    """Parse contiguous SID tokens from generated text."""
    import re

    pattern = re.compile(r"<(a|b|c)_(\d+)>")
    matches = pattern.findall(text)
    if len(matches) < NUM_CODEBOOK_LAYERS:
        return None
    # take first a/b/c triple in order
    codes: list[int] = []
    expected = list(SID_LAYER_PREFIXES)
    idx = 0
    for layer, code in matches:
        if layer != expected[idx]:
            continue
        codes.append(int(code))
        idx += 1
        if idx == NUM_CODEBOOK_LAYERS:
            return codes
    return None
