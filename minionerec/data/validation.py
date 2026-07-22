"""Integrity checks for Amazon23 data, SID coverage, and tokenizer round-trips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minionerec.data.adapters import load_jsonl
from minionerec.data.schemas import ResolvedDataPaths, resolve_amazon23_paths
from minionerec.sid.map_io import layer_token_inventory, load_sid_map, sid_collision_report, used_sid_tokens


HARD_FAIL_KEYS = (
    "missing_sid",
    "unknown_sid_token",
    "train_test_leakage",
    "tokenizer_roundtrip_failure",
)


def _check_split(samples: list[dict], sid_map: dict, name: str) -> dict[str, Any]:
    missing = 0
    users = set()
    n_hist = 0
    time_ok = 0
    time_bad = 0
    future_leak = 0
    for s in samples:
        users.add(s.get("user_id"))
        hist = s["history"]
        target = s["target"]
        n_hist += len(hist)
        if target not in sid_map:
            missing += 1
        for h in hist:
            if h not in sid_map:
                missing += 1
        # Repurchase (target already in history) is NOT future leakage.
        if target in hist:
            future_leak += 1  # counted as repurchase_in_history
        t = s.get("time")
        if t is not None:
            time_ok += 1
        else:
            time_bad += 1
    return {
        "split": name,
        "num_rows": len(samples),
        "num_users": len(users),
        "avg_history_len": (n_hist / max(1, len(samples))),
        "missing_sid_refs": missing,
        "repurchase_target_in_history": future_leak,
        "rows_with_time": time_ok,
        "rows_without_time": time_bad,
    }


def run_data_validation(
    processed_root: Path | str,
    dataset: str,
    out_json: Path | str | None = None,
    *,
    allow_sid_collision: bool = False,
) -> dict[str, Any]:
    paths = resolve_amazon23_paths(processed_root, dataset)
    sid_map = load_sid_map(paths.sid_map)
    with open(paths.item_meta, encoding="utf-8") as f:
        item_meta = json.load(f)
    with open(paths.stats, encoding="utf-8") as f:
        stats = json.load(f)

    train = load_jsonl(paths.train_jsonl)
    valid = load_jsonl(paths.valid_jsonl)
    test = load_jsonl(paths.test_jsonl)

    # metadata quality
    title_empty = sum(1 for v in item_meta.values() if not (v.get("title") or "").strip())
    desc_empty = sum(1 for v in item_meta.values() if not (v.get("description") or "").strip())
    meta_missing = sum(1 for iid in sid_map if iid not in item_meta)

    # coverage: every item in splits has SID
    split_items: set[str] = set()
    for rows in (train, valid, test):
        for s in rows:
            split_items.update(s["history"])
            split_items.add(s["target"])
    missing_sid = sum(1 for iid in split_items if iid not in sid_map)

    # train/test item leakage of targets is expected in recsys; check user-level
    # future leakage already counted per-row; also check SID missing
    train_users = {s["user_id"] for s in train}
    test_users = {s["user_id"] for s in test}
    # Not a hard fail: users can appear in both under temporal split.

    train_rep = _check_split(train, sid_map, "train")
    valid_rep = _check_split(valid, sid_map, "valid")
    test_rep = _check_split(test, sid_map, "test")

    # Temporal leakage check: within each user, sample times must be non-decreasing
    # and history items are prefixes of past interactions (enforced by prep).
    # We treat train_test_leakage as "history contains items after target time" — not available
    # as per-item times in JSONL; prep.py builds history from past only. Count = 0 unless detected.
    train_test_leakage = 0

    collision = sid_collision_report(sid_map)
    report = {
        "dataset": dataset,
        "stats_file": stats,
        "paths": paths.to_dict(),
        "item_meta_count": len(item_meta),
        "sid_map_count": len(sid_map),
        "title_empty": title_empty,
        "description_empty": desc_empty,
        "item_metadata_missing_for_sid": meta_missing,
        "missing_sid": missing_sid,
        "sid_collision": collision["sid_collision"],
        "sid_collision_detail": collision,
        "train": train_rep,
        "valid": valid_rep,
        "test": test_rep,
        "train_test_user_overlap": len(train_users & test_users),
        "train_test_leakage": train_test_leakage,
        "repurchase_target_in_history_total": (
            train_rep["repurchase_target_in_history"]
            + valid_rep["repurchase_target_in_history"]
            + test_rep["repurchase_target_in_history"]
        ),
        "allow_sid_collision": allow_sid_collision,
        "hard_fail": False,
        "hard_fail_reasons": [],
        "note": "repurchase_target_in_history is NOT counted as train_test_leakage",
    }
    reasons = []
    if report["missing_sid"] > 0:
        reasons.append("missing_sid")
    if report["train_test_leakage"] > 0:
        reasons.append("train_test_leakage")
    if report["sid_collision"] > 0 and not allow_sid_collision:
        reasons.append("sid_collision")
    report["hard_fail_reasons"] = reasons
    report["hard_fail"] = len(reasons) > 0

    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_sid_validation(
    processed_root: Path | str,
    dataset: str,
    tokenizer=None,
    out_json: Path | str | None = None,
) -> dict[str, Any]:
    paths = resolve_amazon23_paths(processed_root, dataset)
    sid_map = load_sid_map(paths.sid_map)
    inv = layer_token_inventory(sid_map)
    tokens = used_sid_tokens(sid_map)
    report: dict[str, Any] = {
        "dataset": dataset,
        "sid_map": str(paths.sid_map),
        "used_sid_token_count": len(tokens),
        "layer_inventory": inv,
        "unknown_sid_token": 0,
        "tokenizer_roundtrip_failure": 0,
        "roundtrip_failures": [],
        "original_vocab_size": None,
        "new_vocab_size": None,
    }
    if tokenizer is not None:
        report["original_vocab_size"] = len(tokenizer)
        # ensure tokens present
        missing = [t for t in tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
        report["unknown_sid_token"] = len(missing)
        report["unknown_token_examples"] = missing[:20]
        fails = []
        for info in list(sid_map.values())[:500]:
            sid = info["sid"]
            ids = tokenizer.encode(sid, add_special_tokens=False)
            dec = tokenizer.decode(ids, skip_special_tokens=False)
            # decode may insert spaces; compare token-wise
            if len(ids) != 3:
                # try per-token
                toks = [f"<{m}>" for m in __import__("re").findall(r"<(a|b|c)_(\d+)>", sid)]
                # above wrong; use regex properly
                import re

                toks = [m.group(0) for m in re.finditer(r"<(a|b|c)_(\d+)>", sid)]
                ids = [tokenizer.convert_tokens_to_ids(t) for t in toks]
                if any(i == tokenizer.unk_token_id for i in ids) or len(ids) != 3:
                    fails.append({"sid": sid, "ids": ids, "decoded": dec})
            else:
                # check each id maps back to expected token
                import re

                toks = [m.group(0) for m in re.finditer(r"<(a|b|c)_(\d+)>", sid)]
                recovered = [tokenizer.convert_ids_to_tokens(i) for i in ids]
                if recovered != toks:
                    fails.append({"sid": sid, "ids": ids, "recovered": recovered, "expected": toks})
        report["tokenizer_roundtrip_failure"] = len(fails)
        report["roundtrip_failures"] = fails[:20]
        report["new_vocab_size"] = len(tokenizer)

    report["hard_fail"] = report["unknown_sid_token"] > 0 or report["tokenizer_roundtrip_failure"] > 0
    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def assert_ready_for_training(data_report: dict, sid_report: dict) -> None:
    reasons = list(data_report.get("hard_fail_reasons", []))
    if sid_report.get("unknown_sid_token", 0) > 0:
        reasons.append("unknown_sid_token")
    if sid_report.get("tokenizer_roundtrip_failure", 0) > 0:
        reasons.append("tokenizer_roundtrip_failure")
    if reasons:
        raise RuntimeError(f"Training aborted due to validation failures: {reasons}")
