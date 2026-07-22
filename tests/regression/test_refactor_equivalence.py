"""Regression: refactor must match artifacts/refactor/baseline.json fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "artifacts/refactor/baseline.json"


@pytest.fixture(scope="module")
def baseline():
    assert BASELINE.exists(), "missing baseline; run capture script first"
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_metrics_match_baseline(baseline):
    from minionerec.evaluation.metrics import compute_metrics, hr_at_k, ndcg_at_k

    fx = baseline["metrics_fixture"]
    assert hr_at_k(2, 3) == fx["hr3"]
    assert hr_at_k(4, 5) == fx["hr5"]
    assert hr_at_k(None, 10) == fx["hr10"]
    assert ndcg_at_k(0, 3) == pytest.approx(fx["ndcg3"])
    assert ndcg_at_k(4, 5) == pytest.approx(fx["ndcg5"])
    assert ndcg_at_k(None, 10) == fx["ndcg10"]
    agg = compute_metrics(
        [["sid_a", "sid_b", "sid_c"], ["x", "y", "gold"], ["a", "b", "c", "d", "e"]],
        ["sid_a", "gold", "z"],
        ks=(3, 5, 10),
    )
    for k, v in baseline["metrics_agg"].items():
        if k == "num_examples":
            assert float(agg[k]) == float(v)
        else:
            assert agg[k] == pytest.approx(v, rel=1e-5, abs=1e-7)


def test_reward_and_advantage_match_baseline(baseline):
    from minionerec.rewards.ranking import (
        group_advantages,
        hybrid_ranking_reward,
        make_ndcg_rank_table,
        ndcg_rule_reward,
        rule_reward,
    )

    fx = baseline["reward_fixture"]
    G = fx["G"]
    rule = rule_reward(fx["prompts"], fx["completions"], fx["targets"])
    rank = ndcg_rule_reward(fx["prompts"], fx["completions"], fx["targets"], G)
    total, _, _ = hybrid_ranking_reward(fx["prompts"], fx["completions"], fx["targets"], G)
    adv = group_advantages(total, G)
    assert rule == fx["rule"]
    assert rank == pytest.approx(fx["rank"], rel=1e-5, abs=1e-7)
    assert total == pytest.approx(fx["total"], rel=1e-5, abs=1e-7)
    assert adv == pytest.approx(fx["advantages"], rel=1e-5, abs=1e-7)
    assert make_ndcg_rank_table(G) == pytest.approx(fx["ndcg_table"], rel=1e-5, abs=1e-7)


def test_objectives_match_baseline(baseline):
    from minionerec.training.objectives import official_source_policy_loss, paper_aligned_policy_loss

    fx = baseline["objective_fixture"]
    # recreate tensors from stored sha payload is heavy; use same seed as capture
    torch.manual_seed(0)
    B, T = 4, 6
    logps = torch.randn(B, T)
    ref = torch.randn(B, T)
    old = logps.detach() + 0.01 * torch.randn(B, T)
    mask = torch.ones(B, T)
    advantages = torch.tensor([0.5, -0.2, 0.1, -0.4])
    off_loss, off_m = official_source_policy_loss(logps.clone(), ref, advantages, mask, beta=1e-3)
    pap_loss, pap_m = paper_aligned_policy_loss(
        logps.clone(), old, ref, advantages, mask, beta=1e-3, clip_eps=0.2
    )
    torch.testing.assert_close(
        off_loss.detach(),
        torch.tensor(fx["official_loss"]),
        rtol=1e-5,
        atol=1e-7,
    )
    torch.testing.assert_close(
        pap_loss.detach(),
        torch.tensor(fx["paper_loss"]),
        rtol=1e-5,
        atol=1e-7,
    )
    assert off_m["kl_mean"] == pytest.approx(fx["official_metrics"]["kl_mean"], rel=1e-5, abs=1e-7)
    assert pap_m["kl_mean"] == pytest.approx(fx["paper_metrics"]["kl_mean"], rel=1e-5, abs=1e-7)


def test_sid_codec_roundtrip_strings():
    from minionerec.sid.codec import format_sid, parse_sid

    codes = [1, 2, 3]
    s = format_sid(codes)
    assert parse_sid(s) == codes


def test_build_objective_variants():
    from minionerec.training.objectives import LegacyObjective, OfficialObjective, PaperGrpoObjective, build_objective

    assert isinstance(build_objective("official"), OfficialObjective)
    assert isinstance(build_objective("official_source"), OfficialObjective)
    assert isinstance(build_objective("paper"), PaperGrpoObjective)
    assert isinstance(build_objective("paper_aligned"), PaperGrpoObjective)
    assert isinstance(build_objective("legacy"), LegacyObjective)


def test_config_migration_legacy_keys():
    from minionerec.config import RLVariant, parse_rl_config

    cfg = parse_rl_config(
        {
            "implementation_target": "paper_aligned",
            "num_generations": 4,
            "model_name_or_path": "m",
            "processed_data_root": "data/processed",
            "official_format_root": "data/official_format",
            "dataset": "Industrial_and_Scientific",
            "beta": 1e-3,
            "clip_eps": 0.2,
            "seed": 42,
            "output_dir": "checkpoints/x",
            "train_batch_size": 1,
            "gradient_accumulation_steps": 2,
        }
    )
    assert cfg.algorithm.variant is RLVariant.PAPER
    assert cfg.algorithm.group_size == 4
    assert cfg.model.path == "m"
    assert cfg.data.processed_root == "data/processed"
