"""Multi-task SFT / RL prompts and JSONL builders."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _hist_sid(history_sids: list[str]) -> str:
    return " ".join(history_sids)


def _hist_titles(history_titles: list[str]) -> str:
    return " | ".join(history_titles)


def next_item(history_sids: list[str]) -> tuple[str, str]:
    prompt = (
        "You are a generative recommender. Given the user's chronologically ordered "
        f"item Semantic IDs:\n{_hist_sid(history_sids)}\n"
        "Recommend the next item as Semantic ID tokens only."
    )
    return prompt, "Recommend the next item."


def text_hist_to_sid(history_titles: list[str]) -> str:
    return (
        "Given the user's recent items (titles):\n"
        f"{_hist_titles(history_titles)}\n"
        "Predict the Semantic ID of the next item."
    )


def sid_hist_to_title(history_sids: list[str]) -> str:
    return (
        "Given the user's recent item Semantic IDs:\n"
        f"{_hist_sid(history_sids)}\n"
        "Predict the title of the next item."
    )


def sid_to_title(sid: str) -> str:
    return f"What is the product title corresponding to Semantic ID {sid}?"


def title_to_sid(title: str) -> str:
    return f"What is the Semantic ID of the product titled: {title}?"


def sid_to_description(sid: str) -> str:
    return f"Write a short product description for Semantic ID {sid}."


def description_to_sid(description: str) -> str:
    desc = description[:500]
    return f"Infer the Semantic ID for this product description:\n{desc}"


def example(task: str, prompt: str, answer: str) -> dict[str, Any]:
    return {
        "task": task,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "answer": answer,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_tasks(
    data_dir: Path,
    sid_map_path: Path,
    out_dir: Path,
    seed: int = 42,
    include_description: bool = True,
) -> None:
    random.seed(seed)
    with open(sid_map_path, encoding="utf-8") as f:
        sid_map = json.load(f)
    with open(data_dir / "item_meta.json", encoding="utf-8") as f:
        item_meta = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    align_rows = []
    for iid, info in item_meta.items():
        if iid not in sid_map:
            continue
        sid = sid_map[iid]["sid"]
        title = info.get("title") or iid
        desc = info.get("description") or title
        align_rows.append(example("sid_to_title", sid_to_title(sid), title))
        align_rows.append(example("title_to_sid", title_to_sid(title), sid))
        if include_description:
            align_rows.append(
                example("sid_to_description", sid_to_description(sid), desc[:256])
            )
            align_rows.append(
                example("description_to_sid", description_to_sid(desc), sid)
            )

    for split in ("train", "valid", "test"):
        samples = load_jsonl(data_dir / f"{split}.jsonl")
        rec_rows = []
        for s in tqdm(samples, desc=f"build {split}"):
            hist = s["history"]
            target = s["target"]
            if target not in sid_map or any(h not in sid_map for h in hist):
                continue
            hist_sids = [sid_map[h]["sid"] for h in hist]
            hist_titles = [item_meta[h].get("title") or h for h in hist]
            target_sid = sid_map[target]["sid"]
            target_title = item_meta[target].get("title") or target

            prompt, _ = next_item(hist_sids)
            rec_rows.append(example("generative_retrieval", prompt, target_sid))
            rec_rows.append(
                example(
                    "asymmetric_text_to_sid",
                    text_hist_to_sid(hist_titles),
                    target_sid,
                )
            )
            rec_rows.append(
                example(
                    "asymmetric_sid_to_title",
                    sid_hist_to_title(hist_sids),
                    target_title,
                )
            )

        if split == "train":
            mixed = rec_rows + align_rows
            random.shuffle(mixed)
        elif split == "valid":
            mixed = [r for r in rec_rows if r["task"] == "generative_retrieval"]
            mixed += align_rows[:: max(1, len(align_rows) // 2000)]
        else:
            mixed = [r for r in rec_rows if r["task"] == "generative_retrieval"]

        path = out_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in mixed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(mixed)} -> {path}")

        if split in ("train", "valid"):
            rl_rows = [r for r in rec_rows if r["task"] in ("generative_retrieval", "title_to_sid")]
            if split == "train":
                rl_rows += [r for r in align_rows if r["task"] == "title_to_sid"]
            rl_path = out_dir / f"rl_{split}.jsonl"
            with open(rl_path, "w", encoding="utf-8") as f:
                for row in rl_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"rl_{split}: {len(rl_rows)} -> {rl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--sid_map", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_description", action="store_true")
    args = parser.parse_args()
    make_tasks(
        args.data_dir,
        args.sid_map,
        args.out_dir,
        seed=args.seed,
        include_description=not args.no_description,
    )


if __name__ == "__main__":
    main()
