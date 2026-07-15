#!/usr/bin/env python3
"""Live RL training monitor: tail metrics.jsonl and refresh plots."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_KEYS = ("loss", "reward_mean", "kl_mean", "lp_mean", "grad_norm", "learning_rate")
LOG_RE = re.compile(
    r"\[rl\] metrics step=(?P<step>\d+) epoch=(?P<epoch>[0-9.]+) "
    r"loss=(?P<loss>[-+0-9.eE]+) reward_mean=(?P<reward_mean>[-+0-9.eE]+) "
    r"kl_mean=(?P<kl_mean>[-+0-9.eE]+) lp_mean=(?P<lp_mean>[-+0-9.eE]+) "
    r"grad_norm=(?P<grad_norm>[-+0-9.eE]+) learning_rate=(?P<learning_rate>[-+0-9.eE]+)"
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            row = {k: float(m.group(k)) if k != "step" else int(m.group(k)) for k in m.groupdict()}
            rows.append(row)
    return rows


def infer_accum(rows: list[dict]) -> int | None:
    if len(rows) < 2:
        return None
    diffs = []
    for i in range(1, len(rows)):
        if "micro_step" in rows[i] and "micro_step" in rows[i - 1]:
            d = int(rows[i]["micro_step"]) - int(rows[i - 1]["micro_step"])
            if d > 0:
                diffs.append(d)
    if not diffs:
        return None
    return max(set(diffs), key=diffs.count)


def normalize_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    accum = rows[-1].get("accum") or infer_accum(rows)
    out = []
    for row in rows:
        r = dict(row)
        if accum and abs(float(r.get("reward_mean", 0.0))) > 5.0:
            # Backward compat: old logs summed micro-batch rewards.
            r["reward_mean"] = float(r["reward_mean"]) / float(accum)
        out.append(r)
    return out


def merge_rows(primary: list[dict], fallback: list[dict]) -> list[dict]:
    if primary:
        return normalize_rows(primary)
    return normalize_rows(fallback)


def plot_metrics(rows: list[dict], out_png: Path, title: str) -> None:
    if not rows:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Waiting for metrics...", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    steps = [int(r["step"]) for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()
    labels = {
        "loss": "Loss",
        "reward_mean": "Reward (mean)",
        "kl_mean": "KL mean",
        "lp_mean": "Logprob mean",
        "grad_norm": "Grad norm",
        "learning_rate": "Learning rate",
    }
    for ax, key in zip(axes, METRIC_KEYS):
        ys = [float(r.get(key, float("nan"))) for r in rows]
        ax.plot(steps, ys, linewidth=1.5)
        ax.set_title(labels[key])
        ax.grid(True, alpha=0.3)
        if key in ("kl_mean", "reward_mean"):
            ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    axes[-2].set_xlabel("Optimizer step")
    axes[-1].set_xlabel("Optimizer step")
    last = rows[-1]
    fig.suptitle(
        f"{title} | step={last.get('step')} epoch={float(last.get('epoch', 0)):.3f} "
        f"eta_step≈{float(last.get('sec_per_step', 0)):.1f}s",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: list[dict], out_txt: Path) -> None:
    if not rows:
        return
    last = rows[-1]
    lines = [
        f"points={len(rows)}",
        f"last_step={last.get('step')}",
        f"last_epoch={last.get('epoch')}",
        f"loss={last.get('loss')}",
        f"reward_mean={last.get('reward_mean')}",
        f"kl_mean={last.get('kl_mean')}",
        f"grad_norm={last.get('grad_norm')}",
        f"learning_rate={last.get('learning_rate')}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor RL metrics and plot curves.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("checkpoints/rl_Industrial_and_Scientific_1.5B"),
        help="RL output dir containing metrics.jsonl",
    )
    parser.add_argument("--interval", type=float, default=30.0, help="Refresh interval seconds")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Plot PNG path (default: <run-dir>/metrics.png)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.jsonl"
    log_path = run_dir / "tmux_rl.log"
    out_png = args.out or (run_dir / "metrics.png")
    out_txt = run_dir / "metrics_latest.txt"
    title = run_dir.name

    print(f"[monitor] run_dir={run_dir}", flush=True)
    print(f"[monitor] metrics={metrics_path}", flush=True)
    print(f"[monitor] plot={out_png}", flush=True)

    while True:
        rows = merge_rows(load_jsonl(metrics_path), load_log(log_path))
        plot_metrics(rows, out_png, title)
        write_summary(rows, out_txt)
        print(
            f"[monitor] updated points={len(rows)} step={rows[-1]['step'] if rows else 'n/a'} -> {out_png}",
            flush=True,
        )
        if args.once:
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
