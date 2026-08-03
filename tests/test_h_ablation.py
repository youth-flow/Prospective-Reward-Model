from __future__ import annotations

import math
import random
import statistics

import pytest
import torch

from smart_reward.exact import (
    ExactSplitData,
    MLETrainingConfig,
    ProTrainingConfig,
    pairwise_differences,
    policy_reward_moment,
)
from smart_reward.h_ablation import (
    _edge_coordinates,
    _edge_seed,
    _fit_h_mle,
    _fit_h_pro,
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


def test_h_annotation_rng_is_stateless_and_edge_order_invariant() -> None:
    edges = [("prompt-a", 0, 1), ("prompt-a", 0, 5), ("prompt-b", 2, 4)]

    def draw(edge: tuple[str, int, int]) -> tuple[int, int, float]:
        prompt_id, left, right = edge
        rng = random.Random(_edge_seed(20261001, prompt_id, left, right))
        return sample_h_annotation(0.35, gamma=0.9, rng=rng)

    forward = {edge: draw(edge) for edge in edges}
    reverse = {edge: draw(edge) for edge in reversed(edges)}
    assert forward == reverse
    assert len({_edge_seed(20261001, *edge) for edge in edges}) == len(edges)


def test_randomized_logit_orientation_is_exactly_antisymmetric() -> None:
    for trials in range(1, 50):
        for successes in range(trials + 1):
            forward = randomized_logit_estimate(successes, trials, 0.9)
            reverse = randomized_logit_estimate(trials - successes, trials, 0.9)
            assert forward == pytest.approx(-reverse, abs=1.0e-14)


@pytest.mark.parametrize("trials", [1, 10, 100, 1000])
def test_randomized_logit_tail_cases_remain_finite_and_unclipped(trials: int) -> None:
    successes = sorted({0, 1, trials // 2, max(0, trials - 1), trials})
    values = [randomized_logit_estimate(value, trials, 0.9) for value in successes]
    assert all(math.isfinite(value) for value in values)
    assert values[0] == pytest.approx(-values[-1], rel=1.0e-12, abs=1.0e-12)
    if trials >= 100:
        assert abs(values[-1]) > 10.0


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


def test_h_fits_are_oracle_independent_after_annotations_are_fixed() -> None:
    split = make_split(prompts=48)
    oracle_replaced = ExactSplitData(
        prompt_ids=split.prompt_ids,
        policy_scores=split.policy_scores,
        reward_features=split.reward_features,
        true_rewards=1.0e6 * torch.randn_like(split.true_rewards),
    )
    design, _, coordinates = _edge_coordinates(split)
    replaced_design, _, replaced_coordinates = _edge_coordinates(oracle_replaced)
    assert torch.equal(design, replaced_design)
    assert all(
        torch.equal(left, right)
        for left, right in zip(coordinates, replaced_coordinates, strict=True)
    )

    generating_weight = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float64)
    h = design @ generating_weight
    frequencies = torch.sigmoid(h)
    mle_config = MLETrainingConfig(
        max_iterations=100,
        history_size=10,
        gradient_tolerance=1.0e-7,
        change_tolerance=1.0e-12,
        microbatch_size=128,
    )
    pro_config = ProTrainingConfig(
        relative_damping=1.0e-2,
        fisher_estimator="raw_second_moment",
        inner_max_iterations=200,
        inner_tolerance=1.0e-10,
        outer_max_iterations=200,
        outer_tolerance=1.0e-9,
        residual_recompute_interval=10,
    )

    def fit_all(train: ExactSplitData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_mle = _fit_h_mle(design, coordinates, frequencies, mle_config)
        h_mse = _fit_mse("h_mse", design, coordinates, h, tolerance=1.0e-8)
        h_pro = _fit_h_pro(train, h_policy_moment(train, h), pro_config)
        assert h_mle.converged and h_mse.converged and h_pro.converged
        return h_mle.weight, h_mse.weight, h_pro.weight

    original = fit_all(split)
    replaced = fit_all(oracle_replaced)
    for original_weight, replaced_weight in zip(original, replaced, strict=True):
        assert torch.equal(original_weight, replaced_weight)
