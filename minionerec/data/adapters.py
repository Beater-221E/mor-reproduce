"""Convert Amazon Reviews 2023 JSONL + current SID map into official MiniOneRec files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from minionerec.util import NUM_CODEBOOK_LAYERS, SID_LAYER_PREFIXES, parse_sid
from minionerec.data.schemas import (
    CATEGORY_PROMPT_NAME,
    OfficialPaths,
    ResolvedDataPaths,
    official_layout,
    resolve_amazon23_paths,
)


_SID_TOKEN_RE = re.compile(r"<(a|b|c)_(\d+)>")


def sid_string_to_tokens(sid: str) -> list[str]:
    """Split combined SID '<a_x><b_y><c_z>' into official index token list."""
    toks = [m.group(0) for m in _SID_TOKEN_RE.finditer(sid)]
    if len(toks) < NUM_CODEBOOK_LAYERS:
        raise ValueError(f"SID has fewer than {NUM_CODEBOOK_LAYERS} tokens: {sid!r}")
    return toks[:NUM_CODEBOOK_LAYERS]


def codes_to_tokens(codes: list[int]) -> list[str]:
    return [f"<{SID_LAYER_PREFIXES[i]}_{int(c)}>" for i, c in enumerate(codes)]


def combined_sid_from_tokens(tokens: list[str]) -> str:
    return "".join(tokens)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_index_and_item(
    sid_map: dict[str, Any],
    item_meta: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]], list[str]]:
    """
    Official formats:
      index.json: {item_id: [tok_a, tok_b, tok_c]}
      item.json:  {item_id: {title, description}}
      info.txt:   semantic_id \\t title \\t item_id
    """
    index: dict[str, list[str]] = {}
    item: dict[str, dict[str, str]] = {}
    info_lines: list[str] = []
    for iid, info in sid_map.items():
        tokens = codes_to_tokens(info["codes"]) if "codes" in info else sid_string_to_tokens(info["sid"])
        combined = info.get("sid") or combined_sid_from_tokens(tokens)
        index[iid] = tokens
        meta = item_meta.get(iid, {})
        title = (meta.get("title") or iid).strip() or iid
        desc = meta.get("description") or title
        if isinstance(desc, list):
            desc = max((d for d in desc if d and str(d).strip()), key=len, default=title)
        desc = str(desc).strip() or title
        item[iid] = {"title": title, "description": desc}
        info_lines.append(f"{combined}\t{title}\t{iid}")
    return index, item, info_lines


def split_to_official_rows(
    samples: list[dict[str, Any]],
    sid_map: dict[str, Any],
    item_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in samples:
        hist = s["history"]
        target = s["target"]
        if target not in sid_map or any(h not in sid_map for h in hist):
            continue
        hist_sids = [sid_map[h]["sid"] for h in hist]
        hist_titles = [(item_meta.get(h, {}).get("title") or h) for h in hist]
        target_sid = sid_map[target]["sid"]
        target_title = item_meta.get(target, {}).get("title") or target
        rows.append(
            {
                "user_id": s.get("user_id", ""),
                "history_item_id": str(hist),
                "history_item_title": str(hist_titles),
                "history_item_sid": str(hist_sids),
                "item_id": target,
                "item_title": target_title,
                "item_sid": target_sid,
                "time": s.get("time", 0),
            }
        )
    return rows


def export_official_format(
    processed_root: Path | str,
    dataset: str,
    out_root: Path | str,
    *,
    force: bool = False,
) -> OfficialPaths:
    """
    Write official-compatible CSV / index / item / info under out_root/dataset.
    Does not touch Amazon23 raw or SID generation artifacts.
    """
    src = resolve_amazon23_paths(processed_root, dataset)
    out = official_layout(out_root, dataset)
    if out.train_csv.exists() and not force:
        return out

    out.root.mkdir(parents=True, exist_ok=True)
    with open(src.sid_map, encoding="utf-8") as f:
        sid_map = json.load(f)
    with open(src.item_meta, encoding="utf-8") as f:
        item_meta = json.load(f)

    index, item, info_lines = build_index_and_item(sid_map, item_meta)
    with open(out.index_json, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    with open(out.item_json, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False)
    with open(out.info_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines) + "\n")

    for split, path in (
        ("train", out.train_csv),
        ("valid", out.valid_csv),
        ("test", out.test_csv),
    ):
        samples = load_jsonl(getattr(src, f"{split}_jsonl"))
        rows = split_to_official_rows(samples, sid_map, item_meta)
        pd.DataFrame(rows).to_csv(path, index=False)

    meta = {
        "source": "amazon_reviews_2023",
        "dataset": dataset,
        "processed_dir": str(src.processed_dir),
        "sid_map": str(src.sid_map),
        "category_prompt": CATEGORY_PROMPT_NAME.get(dataset, dataset),
        "num_items": len(index),
        "adapter": "minionerec_compat.amazon23_adapter",
    }
    with open(out.root / "adapter_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out


def ensure_official_paths(cfg: dict[str, Any]) -> OfficialPaths:
    """Resolve or create official-format paths from YAML config."""
    processed = Path(cfg["processed_data_root"])
    dataset = cfg["dataset"]
    out_root = Path(cfg.get("official_format_root", "data/official_format"))
    force = bool(cfg.get("force_export_official_format", False))
    return export_official_format(processed, dataset, out_root, force=force)
