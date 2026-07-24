import pytest
import torch

import smart_reward.metrics as metrics_module
from smart_reward.metrics import (
    FixedBetaLocalResult,
    FixedKNormalizedLocalResult,
    fixed_beta_deployed_direction_regret,
    fixed_k_normalized_local_regret,
    gauge_center,
    local_regret,
    natural_direction,
    natural_direction_metrics,
    policy_reward_moment,
)


def test_default_score_fisher_solve_is_unpreconditioned(monkeypatch) -> None:
    original_pcg = metrics_module.pcg
    observed_preconditioners: list[torch.Tensor | None] = []

    def recording_pcg(*args, **kwargs):
        observed_preconditioners.append(kwargs.get("inverse_diagonal"))
        return original_pcg(*args, **kwargs)

    monkeypatch.setattr(metrics_module, "pcg", recording_pcg)
    scores = torch.tensor([[[1.0, 0.0], [-1.0, 1.0], [0.5, -0.5]]], dtype=torch.float64)
    rewards = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
    natural_direction(scores, rewards, damping=0.2)

    assert observed_preconditioners == [None]


def test_default_fp32_metric_geometry_is_solved_in_fp64(
    monkeypatch,
) -> None:
    original_pcg = metrics_module.pcg
    observed_rhs_dtypes: list[torch.dtype] = []

    def recording_pcg(*args, **kwargs):
        observed_rhs_dtypes.append(args[1].dtype)
        return original_pcg(*args, **kwargs)

    monkeypatch.setattr(metrics_module, "pcg", recording_pcg)
    scores = torch.tensor([[[1.0, 0.0], [-1.0, 1.0], [0.5, -0.5]]])
    rewards = torch.tensor([[1.0, -0.5, 0.25]])
    direction = natural_direction(scores, rewards, damping=0.2)

    assert observed_rhs_dtypes == [torch.float64]
    assert direction.dtype == torch.float64


def test_external_fisher_operator_must_honor_solver_dtype() -> None:
    scores = torch.tensor([[[1.0, 0.0], [-1.0, 1.0], [0.5, -0.5]]])
    rewards = torch.tensor([[1.0, -0.5, 0.25]])

    with pytest.raises(ValueError, match="already expressed in pcg_dtype"):
        natural_direction(
            scores,
            rewards,
            damping=0.2,
            fisher_operator=lambda vector: vector,
        )


def test_gauge_centering_removes_per_prompt_constants() -> None:
    rewards = torch.tensor([[1.0, -1.0, 2.0], [3.0, 4.0, -2.0]], dtype=torch.float64)
    offsets = torch.tensor([[8.0], [-13.0]], dtype=torch.float64)

    assert torch.allclose(gauge_center(rewards + offsets), gauge_center(rewards))
    assert torch.allclose(gauge_center(rewards).mean(dim=-1), torch.zeros(2, dtype=torch.float64))


def test_local_metrics_are_gauge_invariant() -> None:
    generator = torch.Generator().manual_seed(77)
    scores = torch.randn(5, 4, 3, generator=generator, dtype=torch.float64)
    # Exact candidate score centering is the finite-policy score identity.
    scores = scores - scores.mean(dim=1, keepdim=True)
    predicted = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    target = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    predicted_shift = torch.randn(5, 1, generator=generator, dtype=torch.float64)
    target_shift = torch.randn(5, 1, generator=generator, dtype=torch.float64)

    base_regret = local_regret(scores, predicted, target, damping=0.2)
    shifted_regret = local_regret(
        scores,
        predicted + predicted_shift,
        target + target_shift,
        damping=0.2,
    )
    assert torch.allclose(base_regret, shifted_regret, rtol=1.0e-12, atol=1.0e-13)

    base_metrics = natural_direction_metrics(scores, predicted, target, damping=0.2)
    shifted_metrics = natural_direction_metrics(
        scores,
        predicted + predicted_shift,
        target + target_shift,
        damping=0.2,
    )
    assert torch.allclose(
        base_metrics.predicted_direction,
        shifted_metrics.predicted_direction,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    assert torch.allclose(
        base_metrics.target_direction,
        shifted_metrics.target_direction,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    assert torch.allclose(
        base_metrics.squared_fisher_error,
        shifted_metrics.squared_fisher_error,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    assert torch.allclose(
        base_metrics.fisher_cosine,
        shifted_metrics.fisher_cosine,
        rtol=1.0e-12,
        atol=1.0e-13,
    )


def test_moment_uses_m_minus_one_covariance_but_fisher_uses_node_mean() -> None:
    scores = torch.tensor(
        [
            [[1.0, 0.0], [2.0, -1.0], [-1.0, 2.0]],
            [[0.0, 2.0], [3.0, 1.0], [1.0, -2.0]],
        ],
        dtype=torch.float64,
    )
    rewards = torch.tensor([[2.0, -1.0, 4.0], [0.5, 3.0, -2.0]], dtype=torch.float64)
    num_prompts, num_candidates = rewards.shape
    centered_scores = scores - scores.mean(dim=1, keepdim=True)
    centered_rewards = rewards - rewards.mean(dim=1, keepdim=True)
    expected_moment = (centered_scores.reshape(-1, 2).mT @ centered_rewards.reshape(-1)) / (
        num_prompts * (num_candidates - 1)
    )

    actual_moment = policy_reward_moment(scores, rewards)
    assert torch.allclose(actual_moment, expected_moment, rtol=1.0e-13, atol=1.0e-13)

    damping = 0.3
    flat_scores = scores.reshape(-1, 2)
    node_fisher = flat_scores.mT @ flat_scores / flat_scores.shape[0]
    expected_direction = torch.linalg.solve(
        node_fisher + damping * torch.eye(2, dtype=torch.float64),
        expected_moment,
    )
    actual_direction = natural_direction(scores, rewards, damping=damping)
    assert torch.allclose(actual_direction, expected_direction, rtol=1.0e-12, atol=1.0e-13)


def test_metrics_accept_external_fisher_matrix() -> None:
    scores = torch.tensor([[[1.0, 0.0], [-1.0, 2.0], [0.5, -0.5]]], dtype=torch.float64)
    rewards = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    external_fisher = torch.tensor([[2.0, 0.3], [0.3, 1.4]], dtype=torch.float64)
    damping = 0.2
    moment = policy_reward_moment(scores, rewards)
    expected = torch.linalg.solve(
        external_fisher + damping * torch.eye(2, dtype=torch.float64),
        moment,
    )

    actual = natural_direction(
        scores,
        rewards,
        damping=damping,
        fisher_matrix=external_fisher,
    )
    assert torch.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-13)


def test_perfect_rewards_have_zero_regret_and_direction_error() -> None:
    generator = torch.Generator().manual_seed(88)
    scores = torch.randn(3, 3, 2, generator=generator, dtype=torch.float64)
    rewards = torch.randn(3, 3, generator=generator, dtype=torch.float64)

    regret = local_regret(scores, rewards, rewards, damping=0.1)
    metrics = natural_direction_metrics(scores, rewards, rewards, damping=0.1)

    assert regret.item() == 0.0
    assert metrics.squared_fisher_error.item() == 0.0
    assert torch.allclose(metrics.fisher_cosine, torch.ones((), dtype=torch.float64), atol=1.0e-12)


def test_fixed_beta_evaluates_the_deployed_displacement_with_singular_fisher() -> None:
    fisher = torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64))
    target_moment = torch.tensor([4.0, 0.0], dtype=torch.float64)
    # The second coordinate is Fisher-null and must not change utility or regret.
    deployed = torch.tensor([0.5, 17.0], dtype=torch.float64)

    result = fixed_beta_deployed_direction_regret(
        fisher,
        target_moment,
        deployed,
        beta=2.0,
    )

    assert isinstance(result, FixedBetaLocalResult)
    assert torch.equal(
        result.effective_deployed_direction,
        torch.tensor([0.5, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        result.target_optimal_direction,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert result.deployed_target_utility.item() == pytest.approx(1.5)
    assert result.optimal_target_utility.item() == pytest.approx(2.0)
    assert result.regret.item() == pytest.approx(0.5)
    assert result.deployed_quadratic_kl.item() == pytest.approx(0.25)
    assert result.optimal_quadratic_kl.item() == pytest.approx(1.0)


def test_fixed_beta_and_fixed_k_are_distinct_estimands() -> None:
    fisher = torch.eye(2, dtype=torch.float64)
    target_moment = torch.tensor([1.0, 0.0], dtype=torch.float64)
    unit_ray = torch.tensor([1.0, 0.0], dtype=torch.float64)
    scaled_ray = 2.0 * unit_ray

    unit_fixed_beta = fixed_beta_deployed_direction_regret(
        fisher,
        target_moment,
        unit_ray,
        beta=1.0,
    )
    scaled_fixed_beta = fixed_beta_deployed_direction_regret(
        fisher,
        target_moment,
        scaled_ray,
        beta=1.0,
    )
    unit_fixed_k = fixed_k_normalized_local_regret(
        fisher,
        target_moment,
        unit_ray,
        kappa=0.5,
    )
    scaled_fixed_k = fixed_k_normalized_local_regret(
        fisher,
        target_moment,
        scaled_ray,
        kappa=0.5,
    )

    assert unit_fixed_beta.regret.item() == 0.0
    assert scaled_fixed_beta.regret.item() == pytest.approx(0.5)
    assert unit_fixed_k.regret.item() == 0.0
    assert scaled_fixed_k.regret.item() == 0.0
    assert torch.allclose(
        unit_fixed_k.normalized_deployed_direction,
        scaled_fixed_k.normalized_deployed_direction,
    )


def test_fixed_k_projects_fisher_null_components_and_spends_the_requested_budget() -> None:
    fisher = torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64))
    target_moment = torch.tensor([4.0, 0.0], dtype=torch.float64)
    deployed_ray = torch.tensor([0.5, 9.0], dtype=torch.float64)

    result = fixed_k_normalized_local_regret(
        fisher,
        target_moment,
        deployed_ray,
        kappa=0.25,
    )

    assert isinstance(result, FixedKNormalizedLocalResult)
    assert torch.allclose(
        result.normalized_deployed_direction,
        torch.tensor([0.5, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        result.target_optimal_direction,
        torch.tensor([0.5, 0.0], dtype=torch.float64),
    )
    assert result.deployed_quadratic_kl.item() == pytest.approx(0.25)
    assert result.optimal_quadratic_kl.item() == pytest.approx(0.25)
    assert result.deployed_target_improvement.item() == pytest.approx(2.0)
    assert result.optimal_target_improvement.item() == pytest.approx(2.0)
    assert result.regret.item() == 0.0
    assert result.fisher_cosine.item() == pytest.approx(1.0)
    assert result.deployed_direction_has_zero_fisher_norm is False
    assert result.target_moment_is_zero is False


def test_fixed_k_handles_zero_deployed_and_zero_target_directions_explicitly() -> None:
    fisher = torch.eye(2, dtype=torch.float64)
    target_moment = torch.tensor([1.0, 0.0], dtype=torch.float64)
    zero = torch.zeros(2, dtype=torch.float64)

    zero_deployed = fixed_k_normalized_local_regret(
        fisher,
        target_moment,
        zero,
        kappa=0.5,
    )
    assert zero_deployed.deployed_direction_has_zero_fisher_norm is True
    assert torch.equal(zero_deployed.normalized_deployed_direction, zero)
    assert zero_deployed.optimal_target_improvement.item() == pytest.approx(1.0)
    assert zero_deployed.deployed_target_improvement.item() == 0.0
    assert zero_deployed.regret.item() == pytest.approx(1.0)
    assert torch.isnan(zero_deployed.fisher_cosine)

    zero_target = fixed_k_normalized_local_regret(
        fisher,
        zero,
        torch.tensor([3.0, -1.0], dtype=torch.float64),
        kappa=0.5,
    )
    assert zero_target.target_moment_is_zero is True
    assert torch.equal(zero_target.target_optimal_direction, zero)
    assert zero_target.optimal_target_improvement.item() == 0.0
    assert zero_target.deployed_target_improvement.item() == 0.0
    assert zero_target.regret.item() == 0.0
    assert torch.isnan(zero_target.fisher_cosine)


def test_fixed_k_zero_budget_produces_zero_updates() -> None:
    result = fixed_k_normalized_local_regret(
        torch.eye(2, dtype=torch.float64),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([-3.0, 4.0], dtype=torch.float64),
        kappa=0.0,
    )

    assert torch.count_nonzero(result.normalized_deployed_direction).item() == 0
    assert torch.count_nonzero(result.target_optimal_direction).item() == 0
    assert result.deployed_quadratic_kl.item() == 0.0
    assert result.optimal_quadratic_kl.item() == 0.0
    assert result.regret.item() == 0.0


@pytest.mark.parametrize(
    "function, keyword",
    [
        (fixed_beta_deployed_direction_regret, {"beta": 1.0}),
        (fixed_k_normalized_local_regret, {"kappa": 0.5}),
    ],
)
def test_local_estimand_primitives_reject_target_moment_outside_fisher_range(
    function,
    keyword,
) -> None:
    fisher = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    target_moment = torch.tensor([0.0, 1.0], dtype=torch.float64)
    deployed = torch.tensor([1.0, 0.0], dtype=torch.float64)

    with pytest.raises(ValueError, match=r"Range\(fisher_matrix\)"):
        function(fisher, target_moment, deployed, **keyword)


@pytest.mark.parametrize(
    ("fisher", "target", "deployed", "error", "match"),
    [
        (
            torch.eye(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            ValueError,
            "same shape",
        ),
        (
            torch.eye(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            ValueError,
            "share dtype",
        ),
        (
            torch.tensor([[1.0, float("nan")], [0.0, 1.0]], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            ValueError,
            "finite",
        ),
        (
            torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            ValueError,
            "symmetric",
        ),
        (
            torch.diag(torch.tensor([1.0, -0.5], dtype=torch.float64)),
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            ValueError,
            "positive semidefinite",
        ),
    ],
)
def test_local_estimand_primitives_validate_shape_dtype_and_finiteness(
    fisher,
    target,
    deployed,
    error,
    match,
) -> None:
    with pytest.raises(error, match=match):
        fixed_beta_deployed_direction_regret(
            fisher,
            target,
            deployed,
            beta=1.0,
        )


def test_local_estimand_primitives_reject_unsupported_dtype_and_invalid_scalars() -> None:
    fisher = torch.eye(2, dtype=torch.float16)
    target = torch.ones(2, dtype=torch.float16)
    deployed = torch.ones(2, dtype=torch.float16)
    with pytest.raises(TypeError, match="torch.float32 or torch.float64"):
        fixed_k_normalized_local_regret(
            fisher,
            target,
            deployed,
            kappa=0.5,
        )

    fisher64 = torch.eye(2, dtype=torch.float64)
    target64 = torch.ones(2, dtype=torch.float64)
    deployed64 = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="beta"):
        fixed_beta_deployed_direction_regret(
            fisher64,
            target64,
            deployed64,
            beta=0.0,
        )
    with pytest.raises(ValueError, match="kappa"):
        fixed_k_normalized_local_regret(
            fisher64,
            target64,
            deployed64,
            kappa=-1.0,
        )
    with pytest.raises(ValueError, match="rcond"):
        fixed_k_normalized_local_regret(
            fisher64,
            target64,
            deployed64,
            kappa=0.5,
            rcond=1.0,
        )
