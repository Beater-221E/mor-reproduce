#!/usr/bin/env python3
"""Live SFT training monitor: parse tmux_sft.log / trainer_state and plot curves."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRICS_RE = re.compile(
    r"\[sft\] metrics step=(?P<step>\d+) epoch=(?P<epoch>[0-9.]+) "
    r"loss=(?P<loss>[-+0-9.eE]+) grad_norm=(?P<grad_norm>[-+0-9.eE]+) "
    r"learning_rate=(?P<learning_rate>[-+0-9.eE]+)"
)
EVAL_RE = re.compile(
    r"\[sft\] eval step=(?P<step>\d+) eval_loss=(?P<eval_loss>[-+0-9.eE]+)"
    r"(?: .*?epoch=(?P<epoch>[0-9.]+))?"
)
STEP_RE = re.compile(
    r"\[sft\] step=(?P<step>\d+)/(?P<max_steps>\d+) epoch=(?P<epoch>[0-9.]+) "
    r"(?P<sec_per_step>[0-9.]+)s/step eta≈(?P<eta_h>[0-9.]+)h"
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


def load_trainer_state(run_dir: Path) -> list[dict]:
    ckpts = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        return []
    state_path = ckpts[-1] / "trainer_state.json"
    if not state_path.exists():
        return []
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    rows = []
    for item in state.get("log_history", []):
        if "step" not in item:
            continue
        row = {
            "step": int(item["step"]),
            "epoch": float(item.get("epoch", 0.0)),
            "source": "trainer_state",
        }
        for k in ("loss", "grad_norm", "learning_rate", "eval_loss"):
            if k in item:
                row[k] = float(item[k])
        rows.append(row)
    return rows


def load_log(path: Path) -> tuple[list[dict], list[dict]]:
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    if not path.exists():
        return train_rows, eval_rows
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = METRICS_RE.search(line)
            if m:
                train_rows.append(
                    {
                        "step": int(m.group("step")),
                        "epoch": float(m.group("epoch")),
                        "loss": float(m.group("loss")),
                        "grad_norm": float(m.group("grad_norm")),
                        "learning_rate": float(m.group("learning_rate")),
                        "source": "log",
                    }
                )
                continue
            m = EVAL_RE.search(line)
            if m:
                row = {
                    "step": int(m.group("step")),
                    "eval_loss": float(m.group("eval_loss")),
                    "source": "log",
                }
                if m.group("epoch"):
                    row["epoch"] = float(m.group("epoch"))
                eval_rows.append(row)
                continue
            m = STEP_RE.search(line)
            if m and train_rows and train_rows[-1]["step"] == int(m.group("step")):
                train_rows[-1]["sec_per_step"] = float(m.group("sec_per_step"))
                train_rows[-1]["eta_h"] = float(m.group("eta_h"))
                train_rows[-1]["max_steps"] = int(m.group("max_steps"))
    return train_rows, eval_rows


def dedupe_train(rows: list[dict]) -> list[dict]:
    by_step: dict[int, dict] = {}
    for row in rows:
        step = int(row["step"])
        prev = by_step.get(step)
        if prev is None or prev.get("source") != "log":
            by_step[step] = row
    return [by_step[k] for k in sorted(by_step)]


def dedupe_eval(rows: list[dict]) -> list[dict]:
    by_step: dict[int, dict] = {}
    for row in rows:
        by_step[int(row["step"])] = row
    return [by_step[k] for k in sorted(by_step)]


def merge_train(*sources: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for src in sources:
        merged.extend(src)
    rows = dedupe_train(merged)
    return rows


def merge_eval(*sources: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for src in sources:
        for row in src:
            if "eval_loss" in row:
                merged.append(row)
    return dedupe_eval(merged)


def export_jsonl(train_rows: list[dict], eval_rows: list[dict], path: Path) -> None:
    eval_by_step = {int(r["step"]): float(r["eval_loss"]) for r in eval_rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in train_rows:
            out = dict(row)
            out.pop("source", None)
            if int(row["step"]) in eval_by_step:
                out["eval_loss"] = eval_by_step[int(row["step"])]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def plot_sft(train_rows: list[dict], eval_rows: list[dict], out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not train_rows and not eval_rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Waiting for SFT metrics...", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_loss, ax_eval, ax_gn, ax_lr = axes.ravel()

    if train_rows:
        steps = [int(r["step"]) for r in train_rows]
        ax_loss.plot(steps, [float(r["loss"]) for r in train_rows], linewidth=1.5, label="train loss")
        ax_gn.plot(steps, [float(r.get("grad_norm", float("nan"))) for r in train_rows], linewidth=1.5)
        ax_lr.plot(steps, [float(r.get("learning_rate", float("nan"))) for r in train_rows], linewidth=1.5)

    if eval_rows:
        esteps = [int(r["step"]) for r in eval_rows]
        ax_eval.plot(
            esteps,
            [float(r["eval_loss"]) for r in eval_rows],
            "o-",
            linewidth=1.5,
            markersize=4,
            color="tab:orange",
            label="eval loss",
        )
        if train_rows:
            ax_loss.plot(
                esteps,
                [float(r["eval_loss"]) for r in eval_rows],
                "o",
                markersize=4,
                color="tab:orange",
                alpha=0.7,
                label="eval loss",
            )

    ax_loss.set_title("Train / Eval Loss")
    ax_eval.set_title("Eval Loss")
    ax_gn.set_title("Grad Norm")
    ax_lr.set_title("Learning Rate")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    ax_loss.legend(loc="best", fontsize=8)
    ax_lr.set_xlabel("Step")
    ax_gn.set_xlabel("Step")

    last = train_rows[-1] if train_rows else eval_rows[-1]
    extra = ""
    if "sec_per_step" in last:
        extra = f" {float(last['sec_per_step']):.1f}s/step"
    if "eta_h" in last:
        extra += f" eta≈{float(last['eta_h']):.1f}h"
    fig.suptitle(
        f"{title} | step={last.get('step')} epoch={float(last.get('epoch', 0)):.3f}{extra}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_summary(train_rows: list[dict], eval_rows: list[dict], out_txt: Path) -> None:
    lines = [f"train_points={len(train_rows)}", f"eval_points={len(eval_rows)}"]
    if train_rows:
        last = train_rows[-1]
        lines += [
            f"last_step={last.get('step')}",
            f"last_epoch={last.get('epoch')}",
            f"loss={last.get('loss')}",
            f"grad_norm={last.get('grad_norm')}",
            f"learning_rate={last.get('learning_rate')}",
        ]
    if eval_rows:
        ev = eval_rows[-1]
        lines += [f"last_eval_step={ev.get('step')}", f"eval_loss={ev.get('eval_loss')}"]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor SFT metrics and plot curves.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("checkpoints/sft_Industrial_and_Scientific_1.5B"),
    )
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_path = run_dir / "tmux_sft.log"
    metrics_path = run_dir / "metrics.jsonl"
    out_png = args.out or (run_dir / "metrics.png")
    out_txt = run_dir / "metrics_latest.txt"
    title = run_dir.name

    print(f"[monitor-sft] run_dir={run_dir}", flush=True)
    print(f"[monitor-sft] log={log_path}", flush=True)
    print(f"[monitor-sft] plot={out_png}", flush=True)

    while True:
        log_train, log_eval = load_log(log_path)
        state_rows = load_trainer_state(run_dir)
        state_train = [r for r in state_rows if "loss" in r]
        state_eval = [r for r in state_rows if "eval_loss" in r]

        train_rows = merge_train(load_jsonl(metrics_path), log_train, state_train)
        eval_rows = merge_eval(log_eval, state_eval)
        export_jsonl(train_rows, eval_rows, metrics_path)
        plot_sft(train_rows, eval_rows, out_png, title)
        write_summary(train_rows, eval_rows, out_txt)
        print(
            f"[monitor-sft] updated train={len(train_rows)} eval={len(eval_rows)} "
            f"step={train_rows[-1]['step'] if train_rows else 'n/a'} -> {out_png}",
            flush=True,
        )
        if args.once:
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
