from pathlib import Path

import pytest
import torch

from smart_reward.direct_preference import (
    _initialize_plateau_baseline,
    auxdpo_loss,
    candidate_policy_metrics,
    centered,
    extension_hash,
    load_direct_preference_config,
    pair_indices,
    resolve_source_config,
    soft_preference_loss,
)

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "configs" / "dpo_auxdpo_main.yaml"
SMOKE_EXTENSION = ROOT / "configs" / "dpo_auxdpo_smoke.yaml"
CONVERGED_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged.yaml"
CONVERGED_V2_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged_v2.yaml"
CONVERGED_V3_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged_v3.yaml"
CONVERGED_SMOKE_EXTENSION = ROOT / "configs" / "dpo_auxdpo_converged_smoke.yaml"


def test_formal_direct_preference_config_is_bound_to_source() -> None:
    extension = load_direct_preference_config(EXTENSION)
    source, source_config = resolve_source_config(EXTENSION, extension)
    assert source.name == "fisher_trpo_main.yaml"
    assert source_config["run"]["seeds"] == [20261001, 20261002, 20261003]
    assert len(extension_hash(extension)) == 64


def test_smoke_config_is_a_bounded_subset_of_the_formal_source() -> None:
    extension = load_direct_preference_config(SMOKE_EXTENSION)
    _, source_config = resolve_source_config(SMOKE_EXTENSION, extension)
    assert extension["training"]["limit_prompts_per_split"] == 4
    assert set(extension["experiment"]["seeds"]).issubset(source_config["run"]["seeds"])


def test_converged_config_uses_validation_only_adaptive_stopping() -> None:
    extension = load_direct_preference_config(CONVERGED_EXTENSION)
    _, source_config = resolve_source_config(CONVERGED_EXTENSION, extension)
    training = extension["training"]
    assert extension["experiment"]["betas"] == [0.2]
    assert extension["experiment"]["seeds"] == source_config["run"]["seeds"]
    assert training["validation_selection_metric"] == "policy_implied_soft_btl_nll"
    assert training["test_usage"] == "final_evaluation_only"
    assert training["gradient_accumulation_steps"] == 1
    assert training["min_epochs"] < training["max_epochs"]
    assert training["minimum_lr_reductions"] == 2
    assert training["restore_best_validation_checkpoint"] is True


def test_converged_smoke_changes_only_budget_and_prompt_limit() -> None:
    formal = load_direct_preference_config(CONVERGED_EXTENSION)
    smoke = load_direct_preference_config(CONVERGED_SMOKE_EXTENSION)
    assert smoke["experiment"]["seeds"] == [formal["experiment"]["seeds"][0]]
    assert smoke["experiment"]["betas"] == formal["experiment"]["betas"]
    assert smoke["training"]["prompt_batch_size"] == formal["training"]["prompt_batch_size"]
    assert smoke["training"]["policy_learning_rate"] == formal["training"][
        "policy_learning_rate"
    ]
    assert smoke["training"]["limit_prompts_per_split"] == 8


def test_memory_safe_converged_config_preserves_science_and_halves_physical_batch() -> None:
    first = load_direct_preference_config(CONVERGED_EXTENSION)
    second = load_direct_preference_config(CONVERGED_V2_EXTENSION)
    assert second["experiment"]["seeds"] == first["experiment"]["seeds"]
    assert second["experiment"]["betas"] == first["experiment"]["betas"]
    assert second["training"]["prompt_batch_size"] == 2
    for key in (
        "policy_learning_rate",
        "max_epochs",
        "min_epochs",
        "validation_min_delta",
        "early_stopping_patience",
        "validation_selection_metric",
        "test_usage",
    ):
        assert second["training"][key] == first["training"][key]


def test_scaled_lr_config_changes_only_batch_dependent_optimizer_rates() -> None:
    second = load_direct_preference_config(CONVERGED_V2_EXTENSION)
    third = load_direct_preference_config(CONVERGED_V3_EXTENSION)
    assert third["experiment"]["seeds"] == second["experiment"]["seeds"]
    assert third["experiment"]["betas"] == second["experiment"]["betas"]
    assert third["training"]["prompt_batch_size"] == second["training"]["prompt_batch_size"]
    assert third["training"]["policy_learning_rate"] == pytest.approx(
        second["training"]["policy_learning_rate"] / 2.0
    )
    assert third["auxdpo"]["auxiliary_learning_rate"] == pytest.approx(
        second["auxdpo"]["auxiliary_learning_rate"] / 2.0
    )
    for key in (
        "max_epochs",
        "min_epochs",
        "validation_min_delta",
        "minimum_validation_improvement",
        "early_stopping_patience",
        "minimum_lr_reductions",
        "validation_selection_metric",
        "test_usage",
    ):
        assert third["training"][key] == second["training"][key]


def test_soft_preference_loss_uses_every_unordered_edge() -> None:
    oracle = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float64)
    pairs = pair_indices(3)
    margins = oracle[:, pairs[0]] - oracle[:, pairs[1]]
    targets = torch.sigmoid(margins)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(margins, targets)
    assert torch.equal(soft_preference_loss(oracle, oracle), expected)
    assert pairs.tolist() == [[0, 0, 1], [1, 2, 2]]


def test_auxiliary_delta_enters_reward_but_zero_moment_is_policy_invisible() -> None:
    implicit = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    oracle = torch.tensor([[1.0, -0.5, -0.5]], dtype=torch.float64)
    # Candidate score rows sum to zero.  This delta is orthogonal to the score
    # coordinate, so it changes the preference fit but has zero policy moment.
    scores = torch.tensor([[[-1.0], [0.0], [1.0]]], dtype=torch.float64)
    delta_raw = torch.tensor([[0.4, -0.8, 0.4]], dtype=torch.float64, requires_grad=True)
    loss, diagnostics = auxdpo_loss(
        implicit,
        oracle,
        delta_raw,
        scores,
        nullspace_weight=1.0,
        amplitude_weight=0.01,
        delta_cap=1.0,
    )
    assert diagnostics["delta"].abs().max() > 0
    assert diagnostics["nullspace_moment"].abs().item() == pytest.approx(0.0, abs=1e-15)
    loss.backward()
    assert implicit.grad is not None
    assert delta_raw.grad is not None


def test_candidate_policy_metrics_satisfy_gibbs_identities() -> None:
    rewards = torch.tensor([[1.2, 0.1, -0.7], [0.3, 0.2, -0.2]], dtype=torch.float64)
    beta = 0.2
    # The tabular policy has log ratios equal to reward/beta up to a prompt constant.
    metrics = candidate_policy_metrics(rewards / beta, rewards, beta=beta)
    assert metrics["delta_J"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["beta_KL"] == pytest.approx(0.0, abs=1e-12)
    assert max(metrics["identity_residuals"].values()) < 1e-12


def test_centering_removes_only_prompt_constants() -> None:
    values = torch.tensor([[3.0, 4.0, 5.0], [-2.0, 0.0, 2.0]])
    shifted = values + torch.tensor([[91.0], [-17.0]])
    assert torch.allclose(centered(values), centered(shifted))
    assert torch.allclose(centered(values).mean(dim=1), torch.zeros(2))


def test_plateau_scheduler_is_initialized_against_epoch_zero_policy() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=0, threshold=1.0e-5, threshold_mode="abs"
    )
    _initialize_plateau_baseline(scheduler, 0.693147)
    scheduler.step(0.694)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
