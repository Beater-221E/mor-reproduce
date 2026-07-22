"""Project-wide constants (datasets, SID layers). No heavy imports."""

from __future__ import annotations

DATASETS = {
    "Industrial_and_Scientific": {
        "review_file": "Industrial_and_Scientific.jsonl.gz",
        "meta_file": "meta_Industrial_and_Scientific.jsonl.gz",
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

DEFAULT_SEED = 42
DEFAULT_KS = (3, 5, 10)
