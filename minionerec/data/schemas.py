"""Canonical paths and schema for Amazon23 + current SID products."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResolvedDataPaths:
    """Paths to immutable Amazon23 / SID artifacts (must not be regenerated)."""

    dataset: str
    processed_dir: Path
    train_jsonl: Path
    valid_jsonl: Path
    test_jsonl: Path
    item_meta: Path
    stats: Path
    sid_dir: Path
    sid_map: Path
    codebooks: Path | None = None
    pca: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


@dataclass
class OfficialPaths:
    """Official MiniOneRec-compatible layout emitted by the adapter."""

    root: Path
    train_csv: Path
    valid_csv: Path
    test_csv: Path
    item_json: Path
    index_json: Path
    info_txt: Path
    category: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


CATEGORY_PROMPT_NAME = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}


def resolve_amazon23_paths(processed_root: Path | str, dataset: str) -> ResolvedDataPaths:
    root = Path(processed_root) / dataset
    sid_dir = root / "sid"
    return ResolvedDataPaths(
        dataset=dataset,
        processed_dir=root,
        train_jsonl=root / "train.jsonl",
        valid_jsonl=root / "valid.jsonl",
        test_jsonl=root / "test.jsonl",
        item_meta=root / "item_meta.json",
        stats=root / "stats.json",
        sid_dir=sid_dir,
        sid_map=sid_dir / "sid_map.json",
        codebooks=sid_dir / "codebooks.npz" if (sid_dir / "codebooks.npz").exists() else None,
        pca=sid_dir / "pca.npz" if (sid_dir / "pca.npz").exists() else None,
    )


def official_layout(out_root: Path | str, dataset: str) -> OfficialPaths:
    root = Path(out_root) / dataset
    return OfficialPaths(
        root=root,
        train_csv=root / "train.csv",
        valid_csv=root / "valid.csv",
        test_csv=root / "test.csv",
        item_json=root / f"{dataset}.item.json",
        index_json=root / f"{dataset}.index.json",
        info_txt=root / f"{dataset}.info.txt",
        category=dataset,
    )
