"""Shared constants and helpers."""

from __future__ import annotations

import shutil
from pathlib import Path


def prepare_save_dir(path: Path) -> Path:
    """Make ``path`` a fresh directory, removing any prior file/symlink/dir."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

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


def project_root(start: Path | None = None) -> Path:
    """Resolve repo root from a path under the project (or cwd)."""
    p = (start or Path.cwd()).resolve()
    if p.is_file():
        p = p.parent
    for cand in (p, *p.parents):
        if (cand / "minionerec").is_dir() and (cand / "configs").is_dir():
            return cand
    return Path.cwd().resolve()


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve relative paths against project root; leave absolute paths unchanged."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base or project_root()) / p


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
