import math

import pytest
import torch

from smart_reward.annotations import (
    geometric_annotation_counts,
    randomized_truncation_u_statistic_from_counts,
    repeated_labels_to_h,
    sample_geometric_repeated_labels,
    sample_replicated_geometric_repeated_labels,
)


@pytest.mark.parametrize("probability", [0.25, 0.4, 0.5, 0.6, 0.75])
def test_randomized_u_statistic_is_monte_carlo_unbiased(probability: float) -> None:
    trials = 40_000
    gamma = 0.9
    generator = torch.Generator().manual_seed(7_000 + round(100 * probability))
    totals = geometric_annotation_counts(trials, gamma, generator=generator)
    probabilities = torch.full((trials,), probability, dtype=torch.float64)
    wins = torch.binomial(totals.to(torch.float64), probabilities, generator=generator).to(
        torch.int64
    )

    estimates = randomized_truncation_u_statistic_from_counts(wins, totals, gamma)
    truth = math.log(probability / (1.0 - probability))

    assert abs(float(estimates.mean()) - truth) < 0.02


def test_stable_recurrence_and_fixed_order_special_case() -> None:
    expected_two_wins = 1.0 + 1.0 / (2.0 * 0.9)
    assert repeated_labels_to_h([1, 1], gamma=0.9).item() == pytest.approx(expected_two_wins)
    assert repeated_labels_to_h([1, 0], gamma=0.9).item() == pytest.approx(0.0)
    assert repeated_labels_to_h([1, 1, 1], gamma=1.0).item() == pytest.approx(1.0 + 0.5 + 1.0 / 3.0)


def test_geometric_repeated_label_batch_is_consistent() -> None:
    generator = torch.Generator().manual_seed(123)
    probabilities = torch.tensor([[0.0, 1.0], [0.25, 0.75]], dtype=torch.float64)
    batch = sample_geometric_repeated_labels(
        probabilities,
        gamma=0.8,
        generator=generator,
        max_total_annotations=10_000,
    )

    assert batch.counts.shape == probabilities.shape
    assert batch.labels.numel() == int(batch.counts.sum())
    assert batch.wins[0, 0].item() == 0
    assert batch.wins[0, 1].item() == batch.counts[0, 1].item()
    assert torch.isfinite(batch.logit_estimates(gamma=0.8)).all()


def test_one_replicate_exactly_matches_the_existing_sampler() -> None:
    probabilities = torch.tensor([[0.25, 0.5], [0.75, 0.4]], dtype=torch.float64)
    direct_generator = torch.Generator().manual_seed(31_415)
    replicated_generator = torch.Generator().manual_seed(31_415)

    direct = sample_geometric_repeated_labels(
        probabilities,
        gamma=0.9,
        generator=direct_generator,
    )
    replicated = sample_replicated_geometric_repeated_labels(
        probabilities,
        num_replicates=1,
        gamma=0.9,
        generator=replicated_generator,
    )

    only = replicated.replicates[0]
    assert torch.equal(only.counts, direct.counts)
    assert torch.equal(only.pair_indices, direct.pair_indices)
    assert torch.equal(only.labels, direct.labels)
    assert torch.equal(replicated.counts[0], direct.counts)
    assert torch.equal(replicated.wins[0], direct.wins)
    assert torch.equal(replicated.replicate_h[0], direct.logit_estimates(gamma=0.9))
    assert torch.equal(replicated.mean_h, replicated.replicate_h[0])
    assert torch.equal(replicated.pooled_wins, direct.wins)
    assert torch.equal(replicated.pooled_totals, direct.counts)
    assert replicated.total_annotations == direct.labels.numel()


def test_replicates_are_deterministic_and_keep_boundaries_separate() -> None:
    probabilities = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)

    def draw() -> object:
        return sample_replicated_geometric_repeated_labels(
            probabilities,
            num_replicates=3,
            gamma=0.9,
            generator=torch.Generator().manual_seed(27_182),
        )

    first = draw()
    second = draw()

    assert first.num_replicates == 3
    assert first.counts.shape == (3, 3)
    assert first.wins.shape == (3, 3)
    assert first.replicate_h.shape == (3, 3)
    assert torch.equal(first.counts, second.counts)
    assert torch.equal(first.wins, second.wins)
    assert torch.equal(first.replicate_h, second.replicate_h)
    assert torch.equal(first.mean_h, first.replicate_h.mean(dim=0))
    assert torch.equal(first.pooled_wins, first.wins.sum(dim=0))
    assert torch.equal(first.pooled_totals, first.counts.sum(dim=0))
    assert first.total_annotations == int(first.pooled_totals.sum().item())

    pooled_h = randomized_truncation_u_statistic_from_counts(
        first.pooled_wins,
        first.pooled_totals,
        gamma=0.9,
    )
    assert not torch.allclose(first.mean_h, pooled_h)


def test_replicate_mean_is_unbiased_and_reduces_variance_by_one_over_r() -> None:
    trials = 30_000
    num_replicates = 4
    gamma = 0.9
    probability = 0.5
    probabilities = torch.full((trials,), probability, dtype=torch.float64)
    batch = sample_replicated_geometric_repeated_labels(
        probabilities,
        num_replicates=num_replicates,
        gamma=gamma,
        generator=torch.Generator().manual_seed(16_180),
    )

    truth = math.log(probability / (1.0 - probability))
    marginal_variance = batch.replicate_h.var(unbiased=True)
    mean_variance = batch.mean_h.var(unbiased=True)
    variance_ratio = float(mean_variance / marginal_variance)

    assert abs(float(batch.mean_h.mean()) - truth) < 0.02
    assert variance_ratio == pytest.approx(1.0 / num_replicates, abs=0.02)
    expected_annotations_per_edge = num_replicates / (1.0 - gamma)
    realized_annotations_per_edge = float(batch.pooled_totals.to(torch.float64).mean())
    assert realized_annotations_per_edge == pytest.approx(
        expected_annotations_per_edge,
        rel=0.015,
    )


@pytest.mark.parametrize("num_replicates", [True, 0, -1, 1.5])
def test_replicated_sampler_validates_replicate_count(num_replicates: object) -> None:
    with pytest.raises(ValueError, match="num_replicates"):
        sample_replicated_geometric_repeated_labels(
            torch.tensor([0.5], dtype=torch.float64),
            num_replicates=num_replicates,  # type: ignore[arg-type]
        )


def test_replicated_sampler_annotation_guard_applies_across_replicates() -> None:
    with pytest.raises(RuntimeError, match="max_total_annotations"):
        sample_replicated_geometric_repeated_labels(
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            num_replicates=2,
            gamma=0.9,
            generator=torch.Generator().manual_seed(9),
            max_total_annotations=3,
        )


@pytest.mark.parametrize(
    ("labels", "gamma"),
    [([], 0.9), ([0, 2], 0.9), ([0, 1], 0.0), ([0, 1], 1.1)],
)
def test_repeated_label_input_validation(labels: list[int], gamma: float) -> None:
    with pytest.raises(ValueError):
        repeated_labels_to_h(labels, gamma=gamma)
