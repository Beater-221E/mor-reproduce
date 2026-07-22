"""Amazon Reviews'23 preprocessing for MiniOneRec-style splits.

Raw files from https://amazon-reviews-2023.github.io/ :
  - {Category}.jsonl.gz          (reviews)
  - meta_{Category}.jsonl.gz     (item metadata)

Schema notes (Amazon'23):
  - user_id, parent_asin (item id), timestamp in milliseconds
  - meta keyed by parent_asin; description is a list
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from minionerec.constants import DATASETS


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path) as f:
        for line in tqdm(f, desc=f"load {path.name}"):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_raw_files(dataset: str, raw_dir: Path) -> tuple[Path, Path]:
    """Find review/meta files for a category under raw_dir."""
    review_candidates = [
        raw_dir / f"{dataset}.jsonl.gz",
        raw_dir / f"{dataset}.jsonl",
        raw_dir / f"{dataset}_5.json.gz",
        raw_dir / f"{dataset}_5.json",
        raw_dir / DATASETS[dataset]["review_file"],
    ]
    meta_candidates = [
        raw_dir / f"meta_{dataset}.jsonl.gz",
        raw_dir / f"meta_{dataset}.jsonl",
        raw_dir / f"meta_{dataset}.json.gz",
        raw_dir / f"meta_{dataset}.json",
        raw_dir / DATASETS[dataset]["meta_file"],
    ]
    review = next((p for p in review_candidates if p.exists() and p.stat().st_size > 0), None)
    meta = next((p for p in meta_candidates if p.exists() and p.stat().st_size > 0), None)
    if review is None:
        raise FileNotFoundError(
            f"Missing review file for {dataset} under {raw_dir}. Tried: {[str(p) for p in review_candidates]}"
        )
    if meta is None:
        raise FileNotFoundError(
            f"Missing meta file for {dataset} under {raw_dir}. Tried: {[str(p) for p in meta_candidates]}"
        )
    return review, meta


def to_unix_seconds(ts: Any) -> int:
    """Amazon'23 timestamps are usually ms; Amazon'18 are seconds."""
    try:
        t = int(ts)
    except (TypeError, ValueError):
        return 0
    if t > 10_000_000_000:  # ms
        return t // 1000
    return t


def in_window(unix_time: int, st_year: int, st_month: int, ed_year: int, ed_month: int) -> bool:
    dt = datetime.utcfromtimestamp(int(unix_time))
    start = datetime(st_year, st_month, 1)
    if ed_month == 12:
        end = datetime(ed_year + 1, 1, 1)
    else:
        end = datetime(ed_year, ed_month + 1, 1)
    return start <= dt < end


def filter_k_core(interactions: pd.DataFrame, user_k: int, item_k: int) -> pd.DataFrame:
    df = interactions
    changed = True
    while changed:
        before = len(df)
        user_cnt = df.groupby("user_id").size()
        item_cnt = df.groupby("item_id").size()
        keep_users = user_cnt[user_cnt >= user_k].index
        keep_items = item_cnt[item_cnt >= item_k].index
        df = df[df["user_id"].isin(keep_users) & df["item_id"].isin(keep_items)]
        changed = len(df) != before
    return df.reset_index(drop=True)


def chronological_split(user_seq: list[dict], ratios=(0.8, 0.1, 0.1)):
    n = len(user_seq)
    if n < 3:
        return None
    n_train = max(1, int(n * ratios[0]))
    n_valid = max(1, int(n * ratios[1]))
    if n_train + n_valid >= n:
        n_train = n - 2
        n_valid = 1
    n_test = n - n_train - n_valid
    if n_test < 1:
        return None
    return (
        user_seq[:n_train],
        user_seq[n_train : n_train + n_valid],
        user_seq[n_train + n_valid :],
    )


def build_leave_one_style_rows(sequences: dict[str, list[dict]], split: str, max_hist: int = 10):
    rows = []
    for user_id, seq in sequences.items():
        for i, event in enumerate(seq):
            if event["split"] != split:
                continue
            hist = [e["item_id"] for e in seq[:i]][-max_hist:]
            if not hist:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "history": hist,
                    "target": event["item_id"],
                    "time": event["time"],
                }
            )
    return rows


def load_reviews_amazon23(path: Path, st_year, st_month, ed_year, ed_month) -> list[dict]:
    records = []
    for r in iter_jsonl(path):
        user = r.get("user_id") or r.get("reviewerID")
        item = r.get("parent_asin") or r.get("asin")
        ts = to_unix_seconds(r.get("timestamp") or r.get("unixReviewTime") or r.get("sort_timestamp") or 0)
        if not user or not item or ts <= 0:
            continue
        if not in_window(ts, st_year, st_month, ed_year, ed_month):
            continue
        records.append(
            {
                "user_id": str(user),
                "item_id": str(item),
                "time": ts,
                "review": r.get("text") or r.get("reviewText") or "",
            }
        )
    return records


def load_meta_amazon23(path: Path) -> dict[str, dict]:
    meta_map: dict[str, dict] = {}
    for m in iter_jsonl(path):
        asin = m.get("parent_asin") or m.get("asin")
        if not asin:
            continue
        title = m.get("title") or ""
        desc = m.get("description") or ""
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc)
        features = m.get("features") or []
        if isinstance(features, list) and features:
            desc = (desc + " " + " ".join(str(x) for x in features)).strip()
        meta_map[str(asin)] = {"title": str(title).strip(), "description": str(desc).strip()}
    return meta_map


def preprocess_dataset(
    dataset: str,
    raw_dir: Path,
    out_dir: Path,
    user_k: int = 5,
    item_k: int = 5,
    max_hist: int = 10,
    seed: int = 42,
    st_year: int | None = None,
    st_month: int | None = None,
    ed_year: int | None = None,
    ed_month: int | None = None,
) -> None:
    random.seed(seed)
    cfg = DATASETS[dataset]
    # Amazon'23 default window aligned with MiniOneRec amazon23_data_process.sh
    st_year = st_year if st_year is not None else 2018
    st_month = st_month if st_month is not None else 10
    ed_year = ed_year if ed_year is not None else 2023
    ed_month = ed_month if ed_month is not None else 9

    review_path, meta_path = resolve_raw_files(dataset, raw_dir)
    print(f"reviews={review_path}")
    print(f"meta={meta_path}")
    print(f"window={st_year}-{st_month:02d} .. {ed_year}-{ed_month:02d}")

    records = load_reviews_amazon23(review_path, st_year, st_month, ed_year, ed_month)
    df = pd.DataFrame(records).drop_duplicates(subset=["user_id", "item_id", "time"])
    df = df.sort_values(["user_id", "time"]).reset_index(drop=True)
    print(f"interactions in window: {len(df)}")
    df = filter_k_core(df, user_k=user_k, item_k=item_k)
    print(f"after {user_k}-core: {len(df)} interactions, "
          f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    meta_map = load_meta_amazon23(meta_path)
    items = sorted(df["item_id"].unique().tolist())
    item_meta = {}
    for asin in items:
        info = meta_map.get(asin, {"title": asin, "description": ""})
        if not info["title"]:
            info["title"] = asin
        item_meta[asin] = info

    sequences: dict[str, list[dict]] = {}
    for user_id, g in df.groupby("user_id"):
        seq = [{"item_id": row.item_id, "time": int(row.time), "split": "train"} for row in g.itertuples()]
        parts = chronological_split(seq)
        if parts is None:
            continue
        train, valid, test = parts
        for e in train:
            e["split"] = "train"
        for e in valid:
            e["split"] = "valid"
        for e in test:
            e["split"] = "test"
        sequences[user_id] = train + valid + test

    out = out_dir / dataset
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        rows = build_leave_one_style_rows(sequences, split, max_hist=max_hist)
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{dataset} {split}: {len(rows)} -> {path}")

    with open(out / "item_meta.json", "w", encoding="utf-8") as f:
        json.dump(item_meta, f, ensure_ascii=False, indent=2)

    stats = {
        "dataset": dataset,
        "source": "amazon_reviews_2023",
        "window": f"{st_year}-{st_month:02d}..{ed_year}-{ed_month:02d}",
        "num_items": len(item_meta),
        "num_users": len(sequences),
        "train": sum(1 for _ in open(out / "train.jsonl")),
        "valid": sum(1 for _ in open(out / "valid.jsonl")),
        "test": sum(1 for _ in open(out / "test.jsonl")),
    }
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--user_k", type=int, default=5)
    parser.add_argument("--item_k", type=int, default=5)
    parser.add_argument("--max_hist", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--st_year", type=int, default=2018)
    parser.add_argument("--st_month", type=int, default=10)
    parser.add_argument("--ed_year", type=int, default=2023)
    parser.add_argument("--ed_month", type=int, default=9)
    args = parser.parse_args()
    preprocess_dataset(
        args.dataset,
        args.raw_dir,
        args.out_dir,
        user_k=args.user_k,
        item_k=args.item_k,
        max_hist=args.max_hist,
        seed=args.seed,
        st_year=args.st_year,
        st_month=args.st_month,
        ed_year=args.ed_year,
        ed_month=args.ed_month,
    )


if __name__ == "__main__":
    main()
