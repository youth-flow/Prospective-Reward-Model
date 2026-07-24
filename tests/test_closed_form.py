import math

import pytest

from smart_reward.closed_form import (
    A0,
    BETA,
    F0,
    PI0,
    THREE_EDGE_DISTRIBUTION,
    TRUE_REWARD,
    apply_a0,
    auxiliary_direction,
    bt_rm_optimal_w,
    categorical_kl,
    closed_form_results,
    exact_true_regret,
    exact_true_regret_derivative,
    expected_reward,
    fisher_from_reference_policy,
    fixed_kl_learned_theta,
    fixed_kl_local_regret,
    fixed_kl_target_theta,
    learned_policy,
    learned_theta,
    local_fisher_kl,
    local_prorm_regret,
    ordered_iid_pair_distribution,
    pair_fisher,
    pair_reward_moment,
    policy_from_theta,
    population_nll,
    reward_vector,
    score_mean,
    sigmoid,
    subtract,
    target_policy_objective,
    true_optimal_policy,
    true_optimal_theta,
)


def _assert_vector_close(
    actual: tuple[float, ...], expected: tuple[float, ...], *, absolute: float = 1.0e-12
) -> None:
    assert actual == pytest.approx(expected, abs=absolute, rel=1.0e-12)


def _assert_matrix_close(
    actual: tuple[tuple[float, ...], ...],
    expected: tuple[tuple[float, ...], ...],
) -> None:
    for actual_row, expected_row in zip(actual, expected, strict=True):
        _assert_vector_close(actual_row, expected_row)


def test_reference_geometry_and_auxiliary_null_direction() -> None:
    _assert_vector_close(PI0, (0.25, 0.25, 0.25, 0.25))
    _assert_vector_close(score_mean(), (0.0, 0.0))
    _assert_matrix_close(fisher_from_reference_policy(), F0)

    basis = tuple(tuple(1.0 if row == column else 0.0 for row in range(4)) for column in range(4))
    for column, vector in enumerate(basis):
        _assert_vector_close(apply_a0(vector), (A0[0][column], A0[1][column]))

    _assert_vector_close(apply_a0(TRUE_REWARD), (1.5, 0.0))
    _assert_vector_close(apply_a0(reward_vector(3.0)), (0.75, 0.75))
    _assert_vector_close(apply_a0(auxiliary_direction(7.0)), (0.0, 0.0))

    # The fixed group masses make delta_eta exactly invisible, not merely
    # first-order invisible.  Check several nonlocal policy parameters.
    for theta_one, theta_two in ((0.0, 0.0), (1.7, -0.9), (-3.0, 2.5)):
        first = sigmoid(2.0 * theta_one)
        second = sigmoid(2.0 * theta_two)
        policy = (0.5 * first, 0.5 * (1.0 - first), 0.5 * second, 0.5 * (1.0 - second))
        assert expected_reward(policy, auxiliary_direction(7.0)) == pytest.approx(0.0, abs=1.0e-15)


def test_four_population_solutions_and_audited_metric_table() -> None:
    rows = {row.method: row for row in closed_form_results()}
    assert tuple(rows) == ("BT-RM", "Aux-BT-RM", "ProRM", "Aux-ProRM")

    bt_w = bt_rm_optimal_w()
    assert bt_w == pytest.approx(1.0920294543521607, rel=1.0e-14)
    assert (rows["BT-RM"].w, rows["BT-RM"].eta) == pytest.approx((bt_w, 0.0))
    assert (rows["Aux-BT-RM"].w, rows["Aux-BT-RM"].eta) == pytest.approx((bt_w, 6.0))
    assert (rows["ProRM"].w, rows["ProRM"].eta) == pytest.approx((3.0, 0.0))
    assert (rows["Aux-ProRM"].w, rows["Aux-ProRM"].eta) == pytest.approx((3.0, 6.0))
    _assert_vector_close(
        rows["BT-RM"].reward,
        (0.5460147271760803, -0.5460147271760803, 0.5460147271760803, -0.5460147271760803),
    )
    _assert_vector_close(
        rows["Aux-BT-RM"].reward,
        (3.5460147271760803, 2.4539852728239197, -2.4539852728239197, -3.5460147271760803),
    )
    _assert_vector_close(rows["ProRM"].reward, (1.5, -1.5, 1.5, -1.5))
    _assert_vector_close(rows["Aux-ProRM"].reward, (4.5, 1.5, -1.5, -4.5))

    expected_metrics = {
        "BT-RM": (0.6068419270316471, 0.09875274689890402, 0.09795085527621777),
        "Aux-BT-RM": (0.3815633415375112, 0.09875274689890402, 0.09795085527621777),
        "ProRM": (0.7659132510591111, 0.0703125, 0.06959892470876661),
        "Aux-ProRM": (0.5406346655649753, 0.0703125, 0.06959892470876661),
    }
    for method, expected in expected_metrics.items():
        row = rows[method]
        _assert_vector_close((row.nll, row.local_regret, row.exact_regret), expected)

    # These are closed-form global optima.  Local perturbations guard their
    # implementation without introducing a numerical optimizer into the test.
    epsilon = 1.0e-3
    assert population_nll(bt_w, 0.0) < population_nll(bt_w - epsilon, 0.0)
    assert population_nll(bt_w, 0.0) < population_nll(bt_w + epsilon, 0.0)
    assert population_nll(bt_w, 6.0) < population_nll(bt_w, 6.0 - epsilon)
    assert population_nll(bt_w, 6.0) < population_nll(bt_w, 6.0 + epsilon)
    assert local_prorm_regret(3.0) < local_prorm_regret(3.0 - epsilon)
    assert local_prorm_regret(3.0) < local_prorm_regret(3.0 + epsilon)


def test_exact_policy_regret_and_nll_regret_ranking_reversal() -> None:
    rows = {row.method: row for row in closed_form_results()}

    _assert_vector_close(true_optimal_theta(), (0.1875, 0.0))
    _assert_vector_close(
        true_optimal_policy(),
        (0.29633329997703484, 0.20366670002296516, 0.25, 0.25),
    )
    _assert_vector_close(
        learned_theta(rows["BT-RM"].w),
        (0.03412592044850502, 0.03412592044850502),
    )
    _assert_vector_close(
        learned_policy(rows["BT-RM"].w),
        (0.25852816979488236, 0.24147183020511764, 0.25852816979488236, 0.24147183020511764),
    )
    _assert_vector_close(learned_theta(3.0), (0.09375, 0.09375))
    _assert_vector_close(
        learned_policy(3.0),
        (0.2733690759923069, 0.22663092400769308, 0.2733690759923069, 0.22663092400769308),
    )

    assert exact_true_regret_derivative(2.0) < 0.0
    assert exact_true_regret_derivative(3.0) == pytest.approx(0.0, abs=1.0e-16)
    assert exact_true_regret_derivative(4.0) > 0.0
    assert exact_true_regret(3.0) < exact_true_regret(2.999)
    assert exact_true_regret(3.0) < exact_true_regret(3.001)

    nll_order = sorted(rows, key=lambda method: rows[method].nll)
    assert nll_order == ["Aux-BT-RM", "Aux-ProRM", "BT-RM", "ProRM"]
    assert rows["ProRM"].exact_regret == pytest.approx(rows["Aux-ProRM"].exact_regret)
    assert rows["BT-RM"].exact_regret == pytest.approx(rows["Aux-BT-RM"].exact_regret)
    assert rows["ProRM"].exact_regret < rows["BT-RM"].exact_regret

    true_regret_reduction = 1.0 - rows["ProRM"].exact_regret / rows["BT-RM"].exact_regret
    bt_aux_nll_reduction = 1.0 - rows["Aux-BT-RM"].nll / rows["BT-RM"].nll
    prorm_aux_nll_reduction = 1.0 - rows["Aux-ProRM"].nll / rows["ProRM"].nll
    _assert_vector_close(
        (true_regret_reduction, bt_aux_nll_reduction, prorm_aux_nll_reduction),
        (0.2894505666897933, 0.37123108252602244, 0.2941306801816247),
    )


def test_fixed_kl_normalization_erases_the_closed_form_scale_advantage() -> None:
    """The original example separates methods only under a common beta."""

    kl_budget = 0.01
    bt_w = bt_rm_optimal_w()
    bt_theta = fixed_kl_learned_theta(bt_w, kl_budget=kl_budget)
    prorm_theta = fixed_kl_learned_theta(3.0, kl_budget=kl_budget)

    _assert_vector_close(bt_theta, prorm_theta)
    _assert_vector_close(
        bt_theta,
        (math.sqrt(2.0 * kl_budget), math.sqrt(2.0 * kl_budget)),
    )
    _assert_vector_close(
        fixed_kl_target_theta(kl_budget=kl_budget),
        (2.0 * math.sqrt(kl_budget), 0.0),
    )
    _assert_vector_close(policy_from_theta(bt_theta), policy_from_theta(prorm_theta))
    assert local_fisher_kl(bt_theta) == pytest.approx(kl_budget)
    assert local_fisher_kl(prorm_theta) == pytest.approx(kl_budget)
    assert local_fisher_kl(fixed_kl_target_theta(kl_budget=kl_budget)) == pytest.approx(kl_budget)
    assert fixed_kl_local_regret(bt_w, kl_budget=kl_budget) == pytest.approx(
        fixed_kl_local_regret(3.0, kl_budget=kl_budget)
    )

    # In contrast, the theorem's common-beta objective retains reward scale.
    assert local_prorm_regret(3.0) < local_prorm_regret(bt_w)


@pytest.mark.parametrize("w", [0.0, float("nan"), float("inf")])
def test_fixed_kl_rejects_invalid_reward_scale(w: float) -> None:
    with pytest.raises(ValueError):
        fixed_kl_learned_theta(w, kl_budget=0.01)


@pytest.mark.parametrize("kl_budget", [0.0, -1.0, float("nan"), float("inf")])
def test_fixed_kl_rejects_invalid_budget(kl_budget: float) -> None:
    with pytest.raises(ValueError):
        fixed_kl_target_theta(kl_budget=kl_budget)


def test_exact_regret_matches_original_target_objective_gap() -> None:
    optimum = true_optimal_policy()
    optimum_value = target_policy_objective(optimum)
    assert categorical_kl(PI0) == pytest.approx(0.0, abs=1.0e-16)

    for row in closed_form_results():
        direct_gap = optimum_value - target_policy_objective(row.policy)
        assert direct_gap == pytest.approx(row.exact_regret, abs=1.0e-14, rel=1.0e-12)


def test_ordered_iid_natural_q0_identifies_reward_moment_and_fisher() -> None:
    natural_q0 = ordered_iid_pair_distribution()
    assert len(natural_q0) == 16
    assert sum(probability for _, _, probability in natural_q0) == pytest.approx(1.0)

    rewards = (
        TRUE_REWARD,
        reward_vector(bt_rm_optimal_w(), 0.0),
        reward_vector(3.0, 6.0),
        (1.25, -0.75, 2.5, -4.0),
    )
    for reward in rewards:
        _assert_vector_close(pair_reward_moment(reward, natural_q0), apply_a0(reward))
    _assert_matrix_close(pair_fisher(natural_q0), F0)


def test_three_equal_tree_edges_are_not_a_natural_q0_stream() -> None:
    tree_reward_moment = pair_reward_moment(TRUE_REWARD, THREE_EDGE_DISTRIBUTION)
    _assert_vector_close(tree_reward_moment, (3.0, -1.0))
    _assert_vector_close(apply_a0(TRUE_REWARD), (1.5, 0.0))
    assert tree_reward_moment != pytest.approx(apply_a0(TRUE_REWARD))

    tree_fisher = pair_fisher(THREE_EDGE_DISTRIBUTION)
    _assert_matrix_close(tree_fisher, ((5.0 / 6.0, -1.0 / 6.0), (-1.0 / 6.0, 5.0 / 6.0)))
    assert tree_fisher != F0

    # The strongest guard against the invalid connection: the raw tree moment
    # can be zero at (w, eta)=(3, 12) although downstream error is nonzero.
    residual_reward = subtract(reward_vector(3.0, 12.0), TRUE_REWARD)
    _assert_vector_close(pair_reward_moment(residual_reward, THREE_EDGE_DISTRIBUTION), (0.0, 0.0))
    _assert_vector_close(apply_a0(residual_reward), (-0.75, 0.75))


@pytest.mark.parametrize("w", [bt_rm_optimal_w(), 3.0], ids=["BT-RM", "ProRM"])
def test_local_approximation_error_strictly_decreases_on_beta_grid(w: float) -> None:
    relative_errors = []
    for beta in (4.0, 8.0, BETA, 32.0, 64.0):
        local = local_prorm_regret(w, beta=beta)
        exact = exact_true_regret(w, beta=beta)
        relative_errors.append(abs(local - exact) / exact)

    assert all(
        relative_errors[index] > relative_errors[index + 1]
        for index in range(len(relative_errors) - 1)
    )
