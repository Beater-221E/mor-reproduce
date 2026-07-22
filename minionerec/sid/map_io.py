"""SID helpers shared by adapters and validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from minionerec.constants import NUM_CODEBOOK_LAYERS
from minionerec.sid.codec import all_sid_tokens, parse_sid
from minionerec.data.adapters import codes_to_tokens, sid_string_to_tokens


def load_sid_map(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sid_collision_report(sid_map: dict[str, Any]) -> dict[str, Any]:
    """Map combined SID -> list of item ids; collisions if >1 item share SID."""
    inv: dict[str, list[str]] = defaultdict(list)
    for iid, info in sid_map.items():
        inv[info["sid"]].append(iid)
    collisions = {sid: ids for sid, ids in inv.items() if len(ids) > 1}
    return {
        "num_items": len(sid_map),
        "num_unique_sids": len(inv),
        "sid_collision": len(collisions),
        "collision_examples": {k: v for i, (k, v) in enumerate(collisions.items()) if i < 20},
    }


def layer_token_inventory(sid_map: dict[str, Any]) -> dict[str, Any]:
    layer_sets: list[set[str]] = [set() for _ in range(NUM_CODEBOOK_LAYERS)]
    lengths: Counter[int] = Counter()
    for info in sid_map.values():
        toks = codes_to_tokens(info["codes"]) if "codes" in info else sid_string_to_tokens(info["sid"])
        lengths[len(toks)] += 1
        for i, t in enumerate(toks[:NUM_CODEBOOK_LAYERS]):
            layer_sets[i].add(t)
    return {
        "tokens_per_layer": [len(s) for s in layer_sets],
        "sid_length_distribution": dict(lengths),
        "expected_layers": NUM_CODEBOOK_LAYERS,
        "all_possible_sid_tokens": len(all_sid_tokens()),
    }


def used_sid_tokens(sid_map: dict[str, Any]) -> list[str]:
    toks: set[str] = set()
    for info in sid_map.values():
        for t in codes_to_tokens(info["codes"]) if "codes" in info else sid_string_to_tokens(info["sid"]):
            toks.add(t)
    return sorted(toks)
