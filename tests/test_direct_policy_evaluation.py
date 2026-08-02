from __future__ import annotations

from pathlib import Path

import pytest
import torch

from smart_reward.direct_policy_evaluation import (
    _cross_u_regret,
    load_direct_policy_config,
    policy_name,
)


def test_formal_direct_policy_config_is_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_direct_policy_config(root / "configs" / "real_policy_dpo_aux_m6.yaml")
    assert config["experiment"]["seeds"] == [20261001, 20261002, 20261003]
    assert config["experiment"]["beta"] == 0.2
    assert config["rollout"]["responses_per_prompt"] == 6
    assert config["reward_evaluation"]["folds"] == 2
    assert policy_name("dpo") == "dpo__beta_0p2"
    assert policy_name("auxdpo") == "auxdpo__beta_0p2"


def test_two_fold_cross_u_uses_cross_products_without_clipping() -> None:
    train_scores = torch.tensor([[[-1.0], [1.0]], [[-1.0], [1.0]]], dtype=torch.float64)
    test_scores = train_scores.clone()
    reward_error = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float64)
    result = _cross_u_regret(
        train_scores,
        test_scores,
        reward_error,
        ("p0", "p1"),
        relative_damping=0.1,
        geometry={
            "cg_max_iterations": 20,
            "cg_tolerance": 1.0e-12,
            "residual_recompute_interval": 2,
        },
    )
    assert result["cross_moment_inverse_fisher_quadratic"] == pytest.approx(1.0 / 1.1)
    assert result["approximate_regret"] == pytest.approx((1.0 / 1.1) / 0.4)
    assert result["folds"] == 2
