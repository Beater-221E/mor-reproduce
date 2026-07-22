"""Unified CLI: ``mor <command>`` or ``python -m minionerec.cli``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_raw(path: str) -> dict:
    from minionerec.config import load_yaml_with_extends

    return load_yaml_with_extends(path)


def cmd_train_sft(args: argparse.Namespace) -> int:
    from minionerec.config import parse_sft_config
    from minionerec.training.sft import train_sft

    cfg = parse_sft_config(_load_raw(args.config))
    result = train_sft(cfg)
    print(f"[mor] SFT done best={result.best_checkpoint}")
    return 0


def cmd_train_rl(args: argparse.Namespace) -> int:
    from minionerec.config import parse_rl_config
    from minionerec.training.rl import train_rl

    cfg = parse_rl_config(_load_raw(args.config))
    result = train_rl(cfg)
    print(f"[mor] RL done best={result.best_checkpoint}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from minionerec.config import parse_eval_config
    from minionerec.evaluation.evaluator import evaluate_checkpoint

    cfg = parse_eval_config(_load_raw(args.config))
    metrics = evaluate_checkpoint(cfg.to_legacy_dict())
    print(f"[mor] eval done: {metrics}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from minionerec.config import migrate_legacy_config
    from minionerec.data.adapters import ensure_official_paths
    from minionerec.data.validation import assert_ready_for_training, run_data_validation, run_sid_validation
    from minionerec.runtime.paths import project_root, resolve_path

    raw = migrate_legacy_config(_load_raw(args.config))
    root = project_root()
    data = raw.get("data", raw)
    dataset = data.get("dataset", raw.get("dataset", "Industrial_and_Scientific"))
    processed = resolve_path(data.get("processed_root", raw.get("processed_data_root", "data/processed")), root)
    adapted = resolve_path(data.get("adapted_root", raw.get("official_format_root", "data/official_format")), root)
    paths = ensure_official_paths(
        {
            "dataset": dataset,
            "processed_data_root": str(processed),
            "official_format_root": str(adapted),
        }
    )
    data_report = run_data_validation(processed, dataset, out_json=root / "artifacts" / "data_validation.json")
    sid_report = run_sid_validation(processed, dataset, out_json=root / "artifacts" / "sid_validation.json")
    assert_ready_for_training(data_report, sid_report)
    print("[mor] validate ok:", paths)
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from minionerec.data.amazon23 import preprocess_dataset
    from minionerec.runtime.paths import project_root, resolve_path

    raw = _load_raw(args.config) if args.config else {}
    root = project_root()
    dataset = raw.get("dataset", "Industrial_and_Scientific")
    preprocess_dataset(
        dataset,
        resolve_path(raw.get("raw_dir", "data/raw"), root),
        resolve_path(raw.get("processed_root", raw.get("out_dir", "data/processed")), root),
        user_k=int(raw.get("user_k", 5)),
        item_k=int(raw.get("item_k", 5)),
        max_hist=int(raw.get("max_hist", 10)),
        seed=int(raw.get("seed", 42)),
        st_year=int(raw.get("st_year", 2018)),
        st_month=int(raw.get("st_month", 10)),
        ed_year=int(raw.get("ed_year", 2023)),
        ed_month=int(raw.get("ed_month", 9)),
    )
    return 0


def cmd_build_embeddings(args: argparse.Namespace) -> int:
    from minionerec.runtime.paths import project_root, resolve_path
    from minionerec.sid import embeddings as emb_mod

    raw = _load_raw(args.config)
    root = project_root()
    dataset = raw.get("dataset", "Industrial_and_Scientific")
    processed = resolve_path(raw.get("processed_root", "data/processed"), root)
    meta = processed / dataset / "item_meta.json"
    out = processed / dataset / "item_emb.npy"
    model = str(resolve_path(raw.get("embed_model", "data/models/Qwen3-Embedding-4B"), root))
    sys.argv = [
        "build-embeddings",
        "--item_meta",
        str(meta),
        "--output",
        str(out),
        "--model_name",
        model,
        "--batch_size",
        str(raw.get("embed_batch_size", 4)),
        "--max_length",
        str(raw.get("embed_max_length", 1024)),
        "--device",
        str(raw.get("device", "cuda:0")),
    ]
    emb_mod.main()
    return 0


def cmd_build_sid(args: argparse.Namespace) -> int:
    from minionerec.runtime.paths import project_root, resolve_path
    from minionerec.sid import rqvae as rqvae_mod

    raw = _load_raw(args.config)
    root = project_root()
    dataset = raw.get("dataset", "Industrial_and_Scientific")
    processed = resolve_path(raw.get("processed_root", "data/processed"), root)
    emb = processed / dataset / "item_emb.npy"
    ids = processed / dataset / "item_emb.ids.json"
    out = processed / dataset / "sid"
    sys.argv = [
        "build-sid",
        "--emb_path",
        str(emb),
        "--ids_path",
        str(ids),
        "--out_dir",
        str(out),
        "--method",
        str(raw.get("method", "residual_kmeans")),
        "--epochs",
        str(raw.get("epochs", 4000)),
        "--batch_size",
        str(raw.get("batch_size", 2048)),
        "--lr",
        str(raw.get("lr", 3e-4)),
        "--latent_dim",
        str(raw.get("latent_dim", 64)),
        "--hidden_dim",
        str(raw.get("hidden_dim", 256)),
        "--beta",
        str(raw.get("beta", 0.25)),
        "--device",
        str(raw.get("device", "cuda:0")),
        "--log_every",
        str(raw.get("log_every", 50)),
        "--early_collision_patience",
        str(raw.get("early_collision_patience", 40)),
        "--pca_dim",
        str(raw.get("pca_dim", 256)),
        "--dead_code_every",
        str(raw.get("dead_code_every", 100)),
        "--warm_start_size",
        str(raw.get("warm_start_size", 8192)),
        "--enforce_unique",
        str(int(bool(raw.get("enforce_unique", True)))),
    ]
    rqvae_mod.main()
    # Keep official_format index/CSV in sync with the new SID map.
    from minionerec.data.adapters import export_official_format

    adapted = resolve_path(raw.get("adapted_root", raw.get("official_format_root", "data/official_format")), root)
    export_official_format(processed, dataset, adapted, force=True)
    print(f"[mor] synced official format under {adapted / dataset}")
    return 0


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    """Sequential: prepare-data → embeddings → SID → validate."""
    data_cfg = args.data_config or "configs/data/amazon23.yaml"
    sid_cfg = args.sid_config or "configs/sid/default.yaml"
    ns = argparse.Namespace
    cmd_prepare_data(ns(config=data_cfg))
    cmd_build_embeddings(ns(config=sid_cfg))
    cmd_build_sid(ns(config=sid_cfg))
    cmd_validate(ns(config=data_cfg))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mor", description="MiniOneRec Amazon23 reproduction CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_config(sp, required: bool = True):
        sp.add_argument("--config", required=required, help="YAML config path (supports extends:)")

    for name, fn, help_ in (
        ("train-sft", cmd_train_sft, "Run SFT"),
        ("train-rl", cmd_train_rl, "Run RL (objective variant via config)"),
        ("evaluate", cmd_evaluate, "Evaluate checkpoint"),
        ("validate", cmd_validate, "Validate data / SID / adapted paths"),
        ("prepare-data", cmd_prepare_data, "Prepare Amazon23 processed splits"),
        ("build-embeddings", cmd_build_embeddings, "Build item text embeddings"),
        ("build-sid", cmd_build_sid, "Build SID mapping (RQ-VAE)"),
    ):
        sp = sub.add_parser(name, help=help_)
        add_config(sp)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("run-pipeline", help="prepare-data → embeddings → SID → validate")
    sp.add_argument("--data-config", default="configs/data/amazon23.yaml")
    sp.add_argument("--sid-config", default="configs/sid/default.yaml")
    sp.set_defaults(func=cmd_run_pipeline)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
