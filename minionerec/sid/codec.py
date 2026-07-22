"""SID string codec: format / parse / token inventory. No Transformers."""

from __future__ import annotations

import re

from minionerec.constants import CODEBOOK_SIZE, NUM_CODEBOOK_LAYERS, SID_LAYER_PREFIXES


def sid_token(layer: int, code: int) -> str:
    return f"<{SID_LAYER_PREFIXES[layer]}_{code}>"


def all_sid_tokens() -> list[str]:
    tokens: list[str] = []
    for layer in range(NUM_CODEBOOK_LAYERS):
        for code in range(CODEBOOK_SIZE):
            tokens.append(sid_token(layer, code))
    return tokens


def format_sid(codes: list[int] | tuple[int, ...]) -> str:
    assert len(codes) == NUM_CODEBOOK_LAYERS
    return "".join(sid_token(i, int(c)) for i, c in enumerate(codes))


def parse_sid(text: str) -> list[int] | None:
    """Parse contiguous SID tokens from generated text."""
    pattern = re.compile(r"<(a|b|c)_(\d+)>")
    matches = pattern.findall(text)
    if len(matches) < NUM_CODEBOOK_LAYERS:
        return None
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
