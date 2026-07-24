import math

import pytest
import torch

from smart_reward.common_beta import (
    assess_measured_kl_safety,
    calibrate_common_beta,
    deploy_with_common_beta,
    summarize_downstream_utility,
)


def _dense_operator(matrix: torch.Tensor):
    return lambda vector: matrix @ vector


def test_train_oracle_calibration_and_common_scaling_identity() -> None:
    fisher = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    oracle = torch.tensor([1.0, -0.5], dtype=torch.float64)
    target_kl = 0.003

    calibration = calibrate_common_beta(
        oracle,
        _dense_operator(fisher),
        target_oracle_quadratic_kl=target_kl,
    )
    expected_curvature = float(oracle @ fisher @ oracle)
    expected_beta = math.sqrt(expected_curvature / (2.0 * target_kl))
    assert calibration.beta_common == pytest.approx(expected_beta)
    assert calibration.predicted_oracle_quadratic_kl == pytest.approx(target_kl)
    torch.testing.assert_close(calibration.oracle_displacement, oracle / expected_beta)

    directions = {
        "BT-MLE": torch.tensor([2.0, 0.0], dtype=torch.float64),
        "ProRM+": torch.tensor([0.0, 3.0], dtype=torch.float64),
        "oracle": oracle,
    }
    deployed = deploy_with_common_beta(
        directions,
        _dense_operator(fisher),
        calibration=calibration,
    )
    assert {result.beta_common for result in deployed.values()} == {calibration.beta_common}
    for name, direction in directions.items():
        torch.testing.assert_close(
            deployed[name].displacement,
            direction / calibration.beta_common,
        )
        expected = 0.5 * float(direction @ fisher @ direction) / expected_beta**2
        assert deployed[name].predicted_quadratic_kl == pytest.approx(expected)
    assert deployed["oracle"].predicted_quadratic_kl == pytest.approx(target_kl)


def test_calibration_rejects_fisher_null_or_invalid_oracle_direction() -> None:
    fisher = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_common_beta(
            torch.tensor([0.0, 2.0], dtype=torch.float64),
            _dense_operator(fisher),
            target_oracle_quadratic_kl=0.01,
        )
    with pytest.raises(ValueError, match="target_oracle_quadratic_kl"):
        calibrate_common_beta(
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            _dense_operator(fisher),
            target_oracle_quadratic_kl=0.0,
        )
    with pytest.raises(ValueError, match="detached"):
        calibrate_common_beta(
            torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True),
            _dense_operator(fisher),
            target_oracle_quadratic_kl=0.01,
        )


def test_measured_kl_safety_is_a_fail_closed_outcome_not_a_rescaling_rule() -> None:
    passed = assess_measured_kl_safety(
        {"ProRM+": 0.019, "BT-MLE": 0.02},
        cap=0.02,
    )
    assert passed.passed is True
    assert passed.violations == ()

    failed = assess_measured_kl_safety(
        {"ProRM+": 0.0201, "BT-MLE": 0.004},
        cap=0.02,
    )
    assert failed.passed is False
    assert failed.violations == ("ProRM+",)
    assert failed.to_dict()["beta_retuned"] is False
    assert failed.measured_by_policy == (("BT-MLE", 0.004), ("ProRM+", 0.0201))

    with pytest.raises(ValueError, match="non-negative"):
        assess_measured_kl_safety({"BT-MLE": -1.0e-6}, cap=0.02)


def test_downstream_utility_uses_on_policy_kl_and_prompt_clustered_pairs() -> None:
    dtype = torch.float64
    rewards = torch.tensor([[2.0, 4.0], [1.0, 3.0]], dtype=dtype)
    kl = torch.tensor([[0.5, 0.5], [0.0, 1.0]], dtype=dtype)
    reference_rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=dtype)
    oracle_rewards = torch.tensor([[4.0, 4.0], [4.0, 4.0]], dtype=dtype)
    oracle_kl = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype)

    summary = summarize_downstream_utility(
        rewards,
        kl,
        reference_rewards,
        beta_common=2.0,
        oracle_step_transformed_target_rewards=oracle_rewards,
        oracle_step_on_policy_updated_to_reference_kl=oracle_kl,
    )

    # Method utilities by candidate are [[1,3],[1,1]], hence prompt means [2,1].
    assert summary.mean_target_reward == pytest.approx(2.5)
    assert summary.mean_on_policy_kl == pytest.approx(0.5)
    assert summary.mean_target_utility == pytest.approx(1.5)
    assert summary.target_utility_sample_standard_error == pytest.approx(0.5)
    # Reference prompt utilities are [1,2], so paired improvements are [1,-1].
    assert summary.improvement_over_zero_b.mean == pytest.approx(0.0)
    assert summary.improvement_over_zero_b.sample_standard_error == pytest.approx(1.0)
    # Oracle-step utility is 3 on both prompts, so gaps are [1,2].
    assert summary.oracle_step_reference_gap is not None
    assert summary.oracle_step_reference_gap.mean == pytest.approx(1.5)
    assert summary.oracle_step_reference_gap.sample_standard_error == pytest.approx(0.5)
    assert summary.to_dict()["oracle_step_is_global_optimum"] is False


def test_downstream_utility_can_be_negative_and_oracle_reference_is_optional() -> None:
    rewards = torch.zeros((2, 1), dtype=torch.float32)
    kl = torch.ones((2, 1), dtype=torch.float32)
    reference = torch.zeros((2, 1), dtype=torch.float32)
    summary = summarize_downstream_utility(
        rewards,
        kl,
        reference,
        beta_common=3.0,
    )
    assert summary.mean_target_utility == pytest.approx(-3.0)
    assert summary.improvement_over_zero_b.mean == pytest.approx(-3.0)
    assert summary.oracle_step_reference_gap is None


@pytest.mark.parametrize(
    ("rewards", "kl", "reference", "match"),
    [
        (
            torch.zeros((1, 2), dtype=torch.float64),
            torch.zeros((1, 2), dtype=torch.float64),
            torch.zeros((1, 2), dtype=torch.float64),
            "at least two prompts",
        ),
        (
            torch.zeros((2, 2), dtype=torch.float64),
            torch.zeros((2, 1), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            "shape",
        ),
        (
            torch.zeros((2, 2), dtype=torch.float64),
            torch.tensor([[0.0, -0.1], [0.0, 0.0]], dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            "non-negative",
        ),
    ],
)
def test_downstream_utility_validates_experimental_units(
    rewards: torch.Tensor,
    kl: torch.Tensor,
    reference: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        summarize_downstream_utility(
            rewards,
            kl,
            reference,
            beta_common=1.0,
        )


def test_downstream_utility_requires_complete_oracle_step_pair() -> None:
    values = torch.zeros((2, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="supplied together"):
        summarize_downstream_utility(
            values,
            values,
            values,
            beta_common=1.0,
            oracle_step_transformed_target_rewards=values,
        )
