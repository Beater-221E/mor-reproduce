"""Typed configuration loading with YAML ``extends`` and legacy field migration."""

from __future__ import annotations

import copy
import warnings
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from minionerec.runtime.paths import project_root


class RLVariant(StrEnum):
    OFFICIAL = "official"
    PAPER = "paper"
    LEGACY = "legacy"


_LEGACY_VARIANT = {
    "official_source": RLVariant.OFFICIAL,
    "paper_aligned": RLVariant.PAPER,
    "legacy_grpo_like": RLVariant.LEGACY,
    "official": RLVariant.OFFICIAL,
    "paper": RLVariant.PAPER,
    "legacy": RLVariant.LEGACY,
}


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "Industrial_and_Scientific"
    processed_root: str = "data/processed"
    adapted_root: str = "data/official_format"
    category_prompt: str = "industrial and scientific items"
    max_train_samples: int = -1
    max_eval_samples: int = -1
    # legacy key aliases filled by migrate
    processed_data_root: str | None = None
    official_format_root: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    path: str = "data/models/Qwen2.5-0.5B"
    diagnostic_path: str | None = None
    variant: str | None = None
    precision: str = "fp32"  # fp32 | fp16 | bf16
    gradient_checkpointing: bool = True
    full_finetuning: bool = True
    freeze_llm: bool = False
    use_lora: bool = False
    add_all_codebook_tokens: bool = False


@dataclass(frozen=True)
class AlgorithmConfig:
    variant: RLVariant = RLVariant.OFFICIAL
    reward: str = "ranking"
    group_size: int = 16
    beta: float = 1e-3
    clip_epsilon: float | None = 0.2
    dapo: bool = False
    gspo: bool = False
    early_stopping_patience: int | None = None


@dataclass(frozen=True)
class GenerationConfig:
    constrained: bool = True
    beam_search: bool = True
    max_new_tokens: int = 128
    temperature: float = 1.0
    max_prompt_length: int = 512
    num_beams: int | None = None  # defaults to group_size when beam


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 1e-5
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    global_batch_size: int | None = None
    warmup_steps: int = 0
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    logging_steps: int = 1
    lr_scheduler_type: str = "cosine"
    cutoff_len: int = 512
    load_best_model_at_end: bool = True
    early_stopping: bool = False
    updates_per_rollout: int = 1
    rl_seq_title_sample: int | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    data_seed: int = 42
    distributed: bool = True
    fail_on_invalid: bool = True
    strict_invalid_abort: bool = False
    output_dir: str = "checkpoints/run"
    artifacts_dir: str = "artifacts"
    smoke_g4: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "run"
    seed: int = 42
    output_dir: str = "checkpoints/run"


@dataclass(frozen=True)
class SFTConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=lambda: AlgorithmConfig(early_stopping_patience=3))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Bridge to existing trainers during transition (flat keys)."""
        d: dict[str, Any] = {}
        d["implementation_target"] = {
            RLVariant.OFFICIAL: "official_source",
            RLVariant.PAPER: "paper_aligned",
            RLVariant.LEGACY: "legacy_grpo_like",
        }[self.algorithm.variant]
        d["dataset"] = self.data.dataset
        d["processed_data_root"] = self.data.processed_root
        d["official_format_root"] = self.data.adapted_root
        d["category_prompt"] = self.data.category_prompt
        d["max_train_samples"] = self.data.max_train_samples
        d["max_eval_samples"] = self.data.max_eval_samples
        d["model_name_or_path"] = self.model.path
        if self.model.diagnostic_path:
            d["diagnostic_model_name_or_path"] = self.model.diagnostic_path
        if self.model.variant:
            d["model_variant"] = self.model.variant
        d["full_finetuning"] = self.model.full_finetuning
        d["freeze_llm"] = self.model.freeze_llm
        d["use_lora"] = self.model.use_lora
        d["add_all_codebook_tokens"] = self.model.add_all_codebook_tokens
        d["cutoff_len"] = self.training.cutoff_len
        d["learning_rate"] = self.training.learning_rate
        d["lr_scheduler_type"] = self.training.lr_scheduler_type
        d["warmup_steps"] = self.training.warmup_steps
        d["weight_decay"] = self.training.weight_decay
        d["batch_size"] = self.training.global_batch_size or 1024
        d["micro_batch_size"] = self.training.micro_batch_size
        d["num_train_epochs"] = self.training.epochs
        d["max_steps"] = self.training.max_steps
        d["load_best_model_at_end"] = self.training.load_best_model_at_end
        d["early_stopping"] = self.training.early_stopping
        d["early_stopping_patience"] = {
            "official_source": self.algorithm.early_stopping_patience or 3,
            "paper_aligned": 1,
        }
        d["seed"] = self.runtime.seed
        d["data_seed"] = self.runtime.data_seed
        d["logging_steps"] = self.training.logging_steps
        d["gradient_checkpointing"] = self.model.gradient_checkpointing
        d["output_dir"] = self.runtime.output_dir or self.experiment.output_dir
        d["artifacts_dir"] = self.runtime.artifacts_dir
        d["precision"] = {
            "preferred": "bf16" if self.model.precision == "bf16" else self.model.precision,
            "fallback_for_v100": "fp32",
        }
        return d


@dataclass(frozen=True)
class RLConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_legacy_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        d["implementation_target"] = {
            RLVariant.OFFICIAL: "official_source",
            RLVariant.PAPER: "paper_aligned",
            RLVariant.LEGACY: "legacy_grpo_like",
        }[self.algorithm.variant]
        d["smoke_g4"] = self.runtime.smoke_g4
        d["dataset"] = self.data.dataset
        d["processed_data_root"] = self.data.processed_root
        d["official_format_root"] = self.data.adapted_root
        d["category_prompt"] = self.data.category_prompt
        d["max_train_samples"] = self.data.max_train_samples
        d["max_eval_samples"] = self.data.max_eval_samples
        d["model_name_or_path"] = self.model.path
        d["num_generations"] = self.algorithm.group_size
        d["beam_search"] = self.generation.beam_search
        d["reward_type"] = self.algorithm.reward
        d["constrained_decoding"] = self.generation.constrained
        d["learning_rate"] = self.training.learning_rate
        d["beta"] = self.algorithm.beta
        d["clip_eps"] = self.algorithm.clip_epsilon if self.algorithm.clip_epsilon is not None else 0.2
        d["dapo"] = self.algorithm.dapo
        d["gspo"] = self.algorithm.gspo
        d["num_train_epochs"] = self.training.epochs
        d["max_steps"] = self.training.max_steps
        d["temperature"] = self.generation.temperature
        d["train_batch_size"] = self.training.micro_batch_size
        d["gradient_accumulation_steps"] = self.training.gradient_accumulation_steps
        d["rl_seq_title_sample"] = self.training.rl_seq_title_sample
        d["max_completion_length"] = self.generation.max_new_tokens
        d["max_prompt_length"] = self.generation.max_prompt_length
        d["max_grad_norm"] = self.training.max_grad_norm
        d["updates_per_rollout"] = self.training.updates_per_rollout
        d["seed"] = self.runtime.seed
        d["logging_steps"] = self.training.logging_steps
        d["fail_on_invalid"] = self.runtime.fail_on_invalid
        d["strict_invalid_abort"] = self.runtime.strict_invalid_abort
        d["output_dir"] = self.runtime.output_dir or self.experiment.output_dir
        return d


@dataclass(frozen=True)
class EvaluationConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    ks: tuple[int, ...] = (3, 5, 10)
    max_samples: int = -1
    split: str = "test"

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.data.dataset,
            "processed_data_root": self.data.processed_root,
            "official_format_root": self.data.adapted_root,
            "model_name_or_path": self.model.path,
            "output_dir": self.runtime.output_dir or self.experiment.output_dir,
            "num_beams": self.generation.num_beams or 10,
            "max_new_tokens": self.generation.max_new_tokens,
            "max_samples": self.max_samples if self.max_samples > 0 else self.data.max_eval_samples,
            "seed": self.runtime.seed,
            "ks": list(self.ks),
            "split": self.split,
        }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_with_extends(path: Path | str, *, _stack: tuple[str, ...] = ()) -> dict[str, Any]:
    path = Path(path).resolve()
    key = str(path)
    if key in _stack:
        raise ValueError(f"Circular extends: {_stack + (key,)}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be mapping: {path}")
    extends = raw.get("extends")
    if extends:
        parent = Path(extends)
        if not parent.is_absolute():
            # relative to config file, then project root
            cand = (path.parent / parent).resolve()
            if not cand.exists():
                cand = (project_root() / parent).resolve()
            parent = cand
        base = load_yaml_with_extends(parent, _stack=_stack + (key,))
        return deep_merge(base, raw)
    return dict(raw)


def migrate_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize flat legacy YAML into nested schema. Emits one-time warnings."""
    cfg = copy.deepcopy(raw)
    migrated: list[str] = []

    # Flat -> nested
    if "implementation_target" in cfg or "algorithm" not in cfg:
        algo = dict(cfg.get("algorithm") or {})
        if "implementation_target" in cfg:
            old = cfg.pop("implementation_target")
            algo["variant"] = str(_LEGACY_VARIANT.get(old, old))
            migrated.append(f"implementation_target={old} -> algorithm.variant={algo['variant']}")
        if "num_generations" in cfg:
            algo.setdefault("group_size", cfg.pop("num_generations"))
            migrated.append("num_generations -> algorithm.group_size")
        if "reward_type" in cfg:
            algo.setdefault("reward", cfg.pop("reward_type"))
        if "beta" in cfg:
            algo.setdefault("beta", cfg.pop("beta"))
        if "clip_eps" in cfg:
            algo.setdefault("clip_epsilon", cfg.pop("clip_eps"))
        if "dapo" in cfg:
            algo.setdefault("dapo", cfg.pop("dapo"))
        if "gspo" in cfg:
            algo.setdefault("gspo", cfg.pop("gspo"))
        # early stopping patience map
        esp = cfg.pop("early_stopping_patience", None)
        if isinstance(esp, dict):
            variant = algo.get("variant", "official")
            key = "official_source" if variant in ("official", "official_source") else "paper_aligned"
            if variant in ("paper", "paper_aligned"):
                key = "paper_aligned"
            else:
                key = "official_source"
            algo.setdefault("early_stopping_patience", esp.get(key, esp.get("official_source", 3)))
        cfg["algorithm"] = algo

    if "model_name_or_path" in cfg or "model" not in cfg:
        model = dict(cfg.get("model") or {})
        if "model_name_or_path" in cfg:
            model.setdefault("path", cfg.pop("model_name_or_path"))
            migrated.append("model_name_or_path -> model.path")
        if "diagnostic_model_name_or_path" in cfg:
            model.setdefault("diagnostic_path", cfg.pop("diagnostic_model_name_or_path"))
        if "model_variant" in cfg:
            model.setdefault("variant", cfg.pop("model_variant"))
        for k in ("full_finetuning", "freeze_llm", "use_lora", "add_all_codebook_tokens", "gradient_checkpointing"):
            if k in cfg:
                model.setdefault(k, cfg.pop(k))
        prec = cfg.pop("precision", None)
        if isinstance(prec, dict):
            # V100 fallback already applied in trainers; store preferred
            model.setdefault("precision", prec.get("fallback_for_v100") or prec.get("preferred") or "fp32")
        elif isinstance(prec, str):
            model.setdefault("precision", prec)
        cfg["model"] = model

    if "processed_data_root" in cfg or "data" not in cfg:
        data = dict(cfg.get("data") or {})
        if "dataset" in cfg:
            data.setdefault("dataset", cfg.pop("dataset"))
        if "processed_data_root" in cfg:
            data.setdefault("processed_root", cfg.pop("processed_data_root"))
            migrated.append("processed_data_root -> data.processed_root")
        if "official_format_root" in cfg:
            data.setdefault("adapted_root", cfg.pop("official_format_root"))
        for k in ("category_prompt", "max_train_samples", "max_eval_samples"):
            if k in cfg:
                data.setdefault(k, cfg.pop(k))
        cfg["data"] = data

    if (
        any(k in cfg for k in ("beam_search", "temperature", "max_completion_length", "constrained_decoding"))
        or "generation" not in cfg
    ):
        gen = dict(cfg.get("generation") or {})
        if "beam_search" in cfg:
            gen.setdefault("beam_search", cfg.pop("beam_search"))
        if "temperature" in cfg:
            gen.setdefault("temperature", cfg.pop("temperature"))
        if "max_completion_length" in cfg:
            gen.setdefault("max_new_tokens", cfg.pop("max_completion_length"))
        if "max_prompt_length" in cfg:
            gen.setdefault("max_prompt_length", cfg.pop("max_prompt_length"))
        if "constrained_decoding" in cfg:
            gen.setdefault("constrained", cfg.pop("constrained_decoding"))
        if "num_beams" in cfg:
            gen.setdefault("num_beams", cfg.pop("num_beams"))
        cfg["generation"] = gen

    if (
        any(
            k in cfg
            for k in (
                "learning_rate",
                "num_train_epochs",
                "max_steps",
                "train_batch_size",
                "micro_batch_size",
                "gradient_accumulation_steps",
                "batch_size",
            )
        )
        or "training" not in cfg
    ):
        tr = dict(cfg.get("training") or {})
        if "learning_rate" in cfg:
            tr.setdefault("learning_rate", cfg.pop("learning_rate"))
        if "num_train_epochs" in cfg:
            tr.setdefault("epochs", cfg.pop("num_train_epochs"))
        if "max_steps" in cfg:
            tr.setdefault("max_steps", cfg.pop("max_steps"))
        if "train_batch_size" in cfg:
            tr.setdefault("micro_batch_size", cfg.pop("train_batch_size"))
        if "micro_batch_size" in cfg:
            tr.setdefault("micro_batch_size", cfg.pop("micro_batch_size"))
        if "gradient_accumulation_steps" in cfg:
            tr.setdefault("gradient_accumulation_steps", cfg.pop("gradient_accumulation_steps"))
        if "batch_size" in cfg:
            tr.setdefault("global_batch_size", cfg.pop("batch_size"))
        for k in (
            "warmup_steps",
            "weight_decay",
            "max_grad_norm",
            "logging_steps",
            "lr_scheduler_type",
            "cutoff_len",
            "load_best_model_at_end",
            "early_stopping",
            "updates_per_rollout",
            "rl_seq_title_sample",
        ):
            if k in cfg:
                tr.setdefault(k, cfg.pop(k))
        cfg["training"] = tr

    rt = dict(cfg.get("runtime") or {})
    if "seed" in cfg:
        rt.setdefault("seed", cfg.pop("seed"))
    if "data_seed" in cfg:
        rt.setdefault("data_seed", cfg.pop("data_seed"))
    if "output_dir" in cfg:
        rt.setdefault("output_dir", cfg.pop("output_dir"))
    if "artifacts_dir" in cfg:
        rt.setdefault("artifacts_dir", cfg.pop("artifacts_dir"))
    if "fail_on_invalid" in cfg:
        rt.setdefault("fail_on_invalid", cfg.pop("fail_on_invalid"))
    if "strict_invalid_abort" in cfg:
        rt.setdefault("strict_invalid_abort", cfg.pop("strict_invalid_abort"))
    if "smoke_g4" in cfg:
        rt.setdefault("smoke_g4", cfg.pop("smoke_g4"))
    cfg["runtime"] = rt

    exp = dict(cfg.get("experiment") or {})
    exp.setdefault("seed", rt.get("seed", 42))
    exp.setdefault("output_dir", rt.get("output_dir", "checkpoints/run"))
    exp.setdefault("name", Path(str(exp.get("output_dir", "run"))).name)
    cfg["experiment"] = exp

    # Normalize variant enum string
    if "algorithm" in cfg and "variant" in cfg["algorithm"]:
        v = cfg["algorithm"]["variant"]
        cfg["algorithm"]["variant"] = str(_LEGACY_VARIANT.get(v, v))

    if migrated:
        warnings.warn(
            "Migrated legacy config fields: " + "; ".join(migrated),
            DeprecationWarning,
            stacklevel=2,
        )
    return cfg


def _filter_kwargs(cls, data: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def _build_nested(cls, data: dict[str, Any]):
    if cls is AlgorithmConfig and "variant" in data:
        data = dict(data)
        data["variant"] = RLVariant(str(_LEGACY_VARIANT.get(data["variant"], data["variant"])))
    return cls(**_filter_kwargs(cls, data))


def parse_sft_config(raw: dict[str, Any]) -> SFTConfig:
    m = migrate_legacy_config(raw)
    return SFTConfig(
        experiment=_build_nested(ExperimentConfig, m.get("experiment", {})),
        data=_build_nested(DataConfig, m.get("data", {})),
        model=_build_nested(ModelConfig, m.get("model", {})),
        algorithm=_build_nested(AlgorithmConfig, m.get("algorithm", {})),
        training=_build_nested(TrainingConfig, m.get("training", {})),
        runtime=_build_nested(RuntimeConfig, m.get("runtime", {})),
    )


def parse_rl_config(raw: dict[str, Any]) -> RLConfig:
    m = migrate_legacy_config(raw)
    return RLConfig(
        experiment=_build_nested(ExperimentConfig, m.get("experiment", {})),
        data=_build_nested(DataConfig, m.get("data", {})),
        model=_build_nested(ModelConfig, m.get("model", {})),
        algorithm=_build_nested(AlgorithmConfig, m.get("algorithm", {})),
        generation=_build_nested(GenerationConfig, m.get("generation", {})),
        training=_build_nested(TrainingConfig, m.get("training", {})),
        runtime=_build_nested(RuntimeConfig, m.get("runtime", {})),
    )


def parse_eval_config(raw: dict[str, Any]) -> EvaluationConfig:
    m = migrate_legacy_config(raw)
    return EvaluationConfig(
        experiment=_build_nested(ExperimentConfig, m.get("experiment", {})),
        data=_build_nested(DataConfig, m.get("data", {})),
        model=_build_nested(ModelConfig, m.get("model", {})),
        generation=_build_nested(GenerationConfig, m.get("generation", {})),
        runtime=_build_nested(RuntimeConfig, m.get("runtime", {})),
        ks=tuple(m.get("ks", (3, 5, 10))),
        max_samples=int(m.get("max_samples", -1)),
        split=str(m.get("split", "test")),
    )


def load_config(path: Path | str, kind: str) -> SFTConfig | RLConfig | EvaluationConfig | dict[str, Any]:
    raw = load_yaml_with_extends(path)
    if kind == "sft":
        return parse_sft_config(raw)
    if kind == "rl":
        return parse_rl_config(raw)
    if kind == "eval":
        return parse_eval_config(raw)
    return migrate_legacy_config(raw)


def dump_resolved(config: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(config, "__dataclass_fields__"):
        obj = asdict(config)

        # enums -> values
        def conv(o):
            if isinstance(o, StrEnum):
                return str(o.value)
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, list):
                return [conv(v) for v in o]
            return o

        obj = conv(obj)
    else:
        obj = config
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")
