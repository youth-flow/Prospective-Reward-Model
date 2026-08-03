from __future__ import annotations

import random
import statistics

import pytest
import torch

from smart_reward.exact import ExactSplitData, pairwise_differences, policy_reward_moment
from smart_reward.h_ablation import (
    _edge_coordinates,
    _fit_mse,
    h_policy_moment,
    load_h_ablation_config,
    randomized_logit_estimate,
    sample_h_annotation,
    sample_positive_geometric,
)


def make_split(seed: int = 19, prompts: int = 24) -> ExactSplitData:
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(prompts, 6, 5, generator=generator, dtype=torch.float64)
    features = torch.randn(prompts, 6, 3, generator=generator, dtype=torch.float64)
    weight = torch.tensor([0.8, -0.35, 0.2], dtype=torch.float64)
    rewards = features @ weight + 0.1 * torch.randn(
        prompts, 6, generator=generator, dtype=torch.float64
    )
    return ExactSplitData(
        prompt_ids=tuple(f"h-prompt-{index}" for index in range(prompts)),
        policy_scores=scores,
        reward_features=features,
        true_rewards=rewards,
    )


def test_formal_h_config_is_frozen() -> None:
    value = load_h_ablation_config("configs/h_mse_ablation.yaml")
    assert value["annotation"]["gamma"] == 0.9
    assert value["policy_update"]["beta"] == 0.2
    assert value["experiment"]["methods"] == [
        "oracle_mse",
        "h_mle",
        "h_mse",
        "h_pro",
    ]


def test_positive_geometric_has_declared_mean() -> None:
    rng = random.Random(1729)
    draws = [sample_positive_geometric(rng, 0.9) for _ in range(100_000)]
    assert statistics.fmean(draws) == pytest.approx(10.0, abs=0.08)
    assert min(draws) == 1


def test_randomized_logit_orientation_is_exactly_antisymmetric() -> None:
    for trials in range(1, 50):
        for successes in range(trials + 1):
            forward = randomized_logit_estimate(successes, trials, 0.9)
            reverse = randomized_logit_estimate(trials - successes, trials, 0.9)
            assert forward == pytest.approx(-reverse, abs=1.0e-14)


@pytest.mark.parametrize("delta", [-1.0, 0.0, 1.0])
def test_randomized_logit_estimator_is_monte_carlo_unbiased(delta: float) -> None:
    rng = random.Random(713 + int(100 * delta))
    estimates = [sample_h_annotation(delta, gamma=0.9, rng=rng)[2] for _ in range(50_000)]
    standard_error = statistics.stdev(estimates) / len(estimates) ** 0.5
    assert statistics.fmean(estimates) == pytest.approx(delta, abs=5.0 * standard_error)


def test_h_edge_moment_matches_node_reward_moment_for_integrable_margins() -> None:
    split = make_split()
    margins = pairwise_differences(split.true_rewards).reshape(-1)
    edge = h_policy_moment(split, margins)
    node = policy_reward_moment(split.policy_scores, split.true_rewards)
    assert torch.allclose(edge, node, atol=1.0e-12, rtol=1.0e-12)


def test_oracle_mse_closed_form_passes_normal_equation_gate() -> None:
    split = make_split(prompts=48)
    design, margins, coordinates = _edge_coordinates(split)
    fit = _fit_mse(
        "oracle_mse",
        design,
        coordinates,
        margins,
        tolerance=1.0e-8,
    )
    assert fit.converged
    assert fit.relative_residual <= 1.0e-8
    assert fit.weight.shape == (split.reward_dimension,)


def test_mse_and_moment_targets_are_prompt_shift_invariant() -> None:
    split = make_split()
    shifts = torch.randn(split.num_prompts, 1, dtype=torch.float64)
    shifted = ExactSplitData(
        prompt_ids=split.prompt_ids,
        policy_scores=split.policy_scores,
        reward_features=split.reward_features,
        true_rewards=split.true_rewards + shifts,
    )
    assert torch.allclose(
        pairwise_differences(split.true_rewards),
        pairwise_differences(shifted.true_rewards),
        atol=1.0e-12,
    )
    assert torch.allclose(
        policy_reward_moment(split.policy_scores, split.true_rewards),
        policy_reward_moment(shifted.policy_scores, shifted.true_rewards),
        atol=1.0e-12,
    )
