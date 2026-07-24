"""Deterministic population calculations for the four-response ProRM example.

The example isolates two different choices made by reward learning:

* a quotient coordinate visible to the downstream policy; and
* a representative inside a policy-equivalence class.

It is deliberately a population ProRM counterexample, not a ProRM+ training
example.  In particular, the three comparison edges used for its likelihood
table are not an iid pair stream from the reference policy.  The helpers below
keep those distributions distinct so the invalid moment identification cannot
be introduced accidentally.

All calculations use Python ``float`` values (IEEE-754 binary64 in CPython),
the standard library, and deterministic finite sums.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import TypeAlias

Vector2: TypeAlias = tuple[float, float]
Vector4: TypeAlias = tuple[float, float, float, float]
Matrix2: TypeAlias = tuple[Vector2, Vector2]
Matrix2x4: TypeAlias = tuple[Vector4, Vector4]
WeightedEdge: TypeAlias = tuple[int, int, float]

BETA = 16.0
PI0: Vector4 = (0.25, 0.25, 0.25, 0.25)
SCORES: tuple[Vector2, ...] = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
TRUE_REWARD: Vector4 = (6.0, 0.0, 0.0, 0.0)

# A_0 r = E_{y~pi_0}[s_0(y) r(y)].
A0: Matrix2x4 = (
    (0.25, -0.25, 0.0, 0.0),
    (0.0, 0.0, 0.25, -0.25),
)
F0: Matrix2 = ((0.5, 0.0), (0.0, 0.5))

# The likelihood memo uses e_12, e_34, and e_13 with equal mass.  This is a
# connected comparison tree, but it is not pi_0 x pi_0 and therefore is not a
# valid raw natural-pair stream for identifying A_0 r.
THREE_EDGE_DISTRIBUTION: tuple[WeightedEdge, ...] = (
    (0, 1, 1.0 / 3.0),
    (2, 3, 1.0 / 3.0),
    (0, 2, 1.0 / 3.0),
)


@dataclass(frozen=True)
class ClosedFormResult:
    """One row of the audited four-method population comparison."""

    method: str
    w: float
    eta: float
    nll: float
    local_regret: float
    exact_regret: float
    theta: Vector2
    policy: Vector4
    reward: Vector4


def sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid."""

    value = float(value)
    if value >= 0.0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def logit(probability: float) -> float:
    """Return the finite log odds of a probability in ``(0, 1)``."""

    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    return math.log(probability) - math.log1p(-probability)


def softplus(value: float) -> float:
    """Numerically stable ``log(1 + exp(value))``."""

    value = float(value)
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def btl_cross_entropy(true_margin: float, predicted_margin: float) -> float:
    """Population BTL cross-entropy for one oriented comparison edge."""

    predicted_margin = float(predicted_margin)
    return softplus(predicted_margin) - sigmoid(true_margin) * predicted_margin


def reward_vector(w: float, eta: float = 0.0) -> Vector4:
    """Return ``r_w + delta_eta`` in response order ``(y1, ..., y4)``."""

    half_w = 0.5 * float(w)
    half_eta = 0.5 * float(eta)
    return (
        half_w + half_eta,
        -half_w + half_eta,
        half_w - half_eta,
        -half_w - half_eta,
    )


def auxiliary_direction(eta: float = 1.0) -> Vector4:
    """Return the policy-null auxiliary direction ``delta_eta``."""

    half_eta = 0.5 * float(eta)
    return (half_eta, half_eta, -half_eta, -half_eta)


def subtract(left: Vector4, right: Vector4) -> Vector4:
    """Subtract two four-response reward vectors."""

    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def apply_a0(reward: Vector4) -> Vector2:
    """Apply the reference-policy reward-to-gradient operator ``A_0``."""

    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, reward, strict=True))
        for row in A0
    )  # type: ignore[return-value]


def score_mean() -> Vector2:
    """Return ``E_pi0[s_0]`` by finite enumeration."""

    return tuple(
        sum(probability * score[coordinate] for probability, score in zip(PI0, SCORES, strict=True))
        for coordinate in range(2)
    )  # type: ignore[return-value]


def fisher_from_reference_policy() -> Matrix2:
    """Return ``E_pi0[s_0 s_0^T]`` by finite enumeration."""

    return tuple(
        tuple(
            sum(
                probability * score[row] * score[column]
                for probability, score in zip(PI0, SCORES, strict=True)
            )
            for column in range(2)
        )
        for row in range(2)
    )  # type: ignore[return-value]


def ordered_iid_pair_distribution() -> tuple[WeightedEdge, ...]:
    """Enumerate all 16 ordered pairs under ``pi_0 x pi_0``."""

    return tuple(
        (left, right, PI0[left] * PI0[right]) for left, right in product(range(4), repeat=2)
    )


def pair_reward_moment(reward: Vector4, distribution: tuple[WeightedEdge, ...]) -> Vector2:
    """Compute ``(1/2) E[z_0(e) Delta r(e)]`` for an explicit edge law."""

    total = [0.0, 0.0]
    for left, right, probability in distribution:
        margin = reward[left] - reward[right]
        for coordinate in range(2):
            score_difference = SCORES[left][coordinate] - SCORES[right][coordinate]
            total[coordinate] += 0.5 * probability * score_difference * margin
    return (total[0], total[1])


def pair_fisher(distribution: tuple[WeightedEdge, ...]) -> Matrix2:
    """Compute ``(1/2) E[z_0(e) z_0(e)^T]`` for an explicit edge law."""

    total = [[0.0, 0.0], [0.0, 0.0]]
    for left, right, probability in distribution:
        difference = tuple(
            SCORES[left][coordinate] - SCORES[right][coordinate] for coordinate in range(2)
        )
        for row in range(2):
            for column in range(2):
                total[row][column] += 0.5 * probability * difference[row] * difference[column]
    return ((total[0][0], total[0][1]), (total[1][0], total[1][1]))


def population_nll(w: float, eta: float = 0.0) -> float:
    """Mean population BTL NLL on the memo's three equally weighted edges."""

    predicted_reward = reward_vector(w, eta)
    weighted_losses = []
    total_mass = math.fsum(weight for _, _, weight in THREE_EDGE_DISTRIBUTION)
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError("three-edge distribution must have positive finite mass")
    for left, right, weight in THREE_EDGE_DISTRIBUTION:
        true_margin = TRUE_REWARD[left] - TRUE_REWARD[right]
        predicted_margin = predicted_reward[left] - predicted_reward[right]
        weighted_losses.append(weight * btl_cross_entropy(true_margin, predicted_margin))
    return math.fsum(weighted_losses) / total_mass


def bt_rm_optimal_w() -> float:
    """Closed-form BT-RM optimum for the shared quotient coordinate ``w``."""

    optimum_probability = 0.5 * (sigmoid(6.0) + 0.5)
    return logit(optimum_probability)


def local_prorm_regret(w: float, *, beta: float = BETA) -> float:
    """Population ProRM loss against ``r_star`` for the restricted class."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    error_moment = apply_a0(subtract(reward_vector(w), TRUE_REWARD))
    # F_0^{-1} = 2 I in this example.
    fisher_inverse_norm_squared = 2.0 * math.fsum(value * value for value in error_moment)
    return fisher_inverse_norm_squared / (2.0 * beta)


def learned_theta(w: float, *, beta: float = BETA) -> Vector2:
    """Exact KL-regularized policy optimizer induced by ``r_w + delta_eta``."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    coordinate = float(w) / (2.0 * beta)
    return (coordinate, coordinate)


def policy_from_theta(theta: Vector2) -> Vector4:
    """Map a two-dimensional policy parameter to the four-action policy."""

    theta_one, theta_two = (float(value) for value in theta)
    if not math.isfinite(theta_one) or not math.isfinite(theta_two):
        raise ValueError("theta must contain only finite values")
    first_group = sigmoid(2.0 * theta_one)
    second_group = sigmoid(2.0 * theta_two)
    return (
        0.5 * first_group,
        0.5 * (1.0 - first_group),
        0.5 * second_group,
        0.5 * (1.0 - second_group),
    )


def learned_policy(w: float, *, beta: float = BETA) -> Vector4:
    """Exact policy induced by any reward ``r_w + delta_eta``."""

    return policy_from_theta(learned_theta(w, beta=beta))


def local_fisher_kl(theta: Vector2) -> float:
    """Second-order ``KL(pi_theta || pi_0)`` in the reference Fisher metric."""

    theta_one, theta_two = (float(value) for value in theta)
    if not math.isfinite(theta_one) or not math.isfinite(theta_two):
        raise ValueError("theta must contain only finite values")
    # 1/2 theta^T F_0 theta with F_0 = (1/2) I.
    return 0.25 * (theta_one * theta_one + theta_two * theta_two)


def _fixed_kl_theta(moment: Vector2, kl_budget: float) -> Vector2:
    """Maximize a linear moment over the local Fisher-KL ellipsoid."""

    kl_budget = float(kl_budget)
    if not math.isfinite(kl_budget) or kl_budget <= 0.0:
        raise ValueError("kl_budget must be finite and strictly positive")
    first, second = (float(value) for value in moment)
    squared_norm = first * first + second * second
    if not math.isfinite(squared_norm) or squared_norm <= 0.0:
        raise ValueError("moment must be finite and nonzero")
    scale = 2.0 * math.sqrt(kl_budget / squared_norm)
    return (scale * first, scale * second)


def fixed_kl_learned_theta(w: float, *, kl_budget: float) -> Vector2:
    """Local policy step from ``r_w`` after normalizing to a fixed KL budget."""

    w = float(w)
    if not math.isfinite(w) or w == 0.0:
        raise ValueError("w must be finite and nonzero")
    return _fixed_kl_theta(apply_a0(reward_vector(w)), kl_budget)


def fixed_kl_target_theta(*, kl_budget: float) -> Vector2:
    """Oracle local true-reward step at the same fixed Fisher-KL budget."""

    return _fixed_kl_theta(apply_a0(TRUE_REWARD), kl_budget)


def fixed_kl_local_regret(w: float, *, kl_budget: float) -> float:
    """True local utility regret when learner and oracle use equal KL budgets."""

    learned = fixed_kl_learned_theta(w, kl_budget=kl_budget)
    target = fixed_kl_target_theta(kl_budget=kl_budget)
    target_moment = apply_a0(TRUE_REWARD)
    return math.fsum(
        moment * (target_value - learned_value)
        for moment, target_value, learned_value in zip(target_moment, target, learned, strict=True)
    )


def true_optimal_theta(*, beta: float = BETA) -> Vector2:
    """Exact optimizer of the true reward in the constrained policy family."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    return (3.0 / beta, 0.0)


def true_optimal_policy(*, beta: float = BETA) -> Vector4:
    """Exact KL-regularized policy optimizer for ``r_star``."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    first_group = sigmoid(6.0 / beta)
    return (0.5 * first_group, 0.5 * (1.0 - first_group), 0.25, 0.25)


def bernoulli_kl(probability: float, target_probability: float) -> float:
    """Forward KL between two Bernoulli laws with interior probabilities."""

    probability = float(probability)
    target_probability = float(target_probability)
    if not 0.0 < probability < 1.0 or not 0.0 < target_probability < 1.0:
        raise ValueError("both probabilities must lie strictly between zero and one")
    return probability * math.log(probability / target_probability) + (
        1.0 - probability
    ) * math.log((1.0 - probability) / (1.0 - target_probability))


def exact_true_regret(w: float, *, beta: float = BETA) -> float:
    """Exact downstream regret under the true KL-regularized objective."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    learned_group_probability = sigmoid(float(w) / beta)
    true_first_group_probability = sigmoid(6.0 / beta)
    return (
        0.5
        * beta
        * (
            bernoulli_kl(learned_group_probability, true_first_group_probability)
            + bernoulli_kl(learned_group_probability, 0.5)
        )
    )


def exact_true_regret_derivative(w: float, *, beta: float = BETA) -> float:
    """Analytic derivative whose sign is the sign of ``2w - 6``."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    probability = sigmoid(float(w) / beta)
    return probability * (1.0 - probability) * (2.0 * float(w) - 6.0) / (2.0 * beta)


def expected_reward(policy: Vector4, reward: Vector4) -> float:
    """Return the finite expectation of a reward under a four-response policy."""

    return math.fsum(probability * value for probability, value in zip(policy, reward, strict=True))


def categorical_kl(policy: Vector4, reference: Vector4 = PI0) -> float:
    """Return ``KL(policy || reference)`` for two strictly positive distributions."""

    if not math.isclose(math.fsum(policy), 1.0, abs_tol=1.0e-12):
        raise ValueError("policy must sum to one")
    if not math.isclose(math.fsum(reference), 1.0, abs_tol=1.0e-12):
        raise ValueError("reference must sum to one")
    if any(probability <= 0.0 for probability in (*policy, *reference)):
        raise ValueError("policy and reference probabilities must be strictly positive")
    return math.fsum(
        probability * math.log(probability / reference_probability)
        for probability, reference_probability in zip(policy, reference, strict=True)
    )


def target_policy_objective(policy: Vector4, *, beta: float = BETA) -> float:
    """Evaluate the original target objective ``J(policy; r_star)`` exactly."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    return expected_reward(policy, TRUE_REWARD) - beta * categorical_kl(policy)


def closed_form_results() -> tuple[ClosedFormResult, ...]:
    """Return the four audited method rows in their canonical order."""

    bt_w = bt_rm_optimal_w()
    specifications = (
        ("BT-RM", bt_w, 0.0),
        ("Aux-BT-RM", bt_w, 6.0),
        ("ProRM", 3.0, 0.0),
        ("Aux-ProRM", 3.0, 6.0),
    )
    return tuple(
        ClosedFormResult(
            method=method,
            w=w,
            eta=eta,
            nll=population_nll(w, eta),
            local_regret=local_prorm_regret(w),
            exact_regret=exact_true_regret(w),
            theta=learned_theta(w),
            policy=learned_policy(w),
            reward=reward_vector(w, eta),
        )
        for method, w, eta in specifications
    )


__all__ = [
    "A0",
    "BETA",
    "F0",
    "PI0",
    "SCORES",
    "THREE_EDGE_DISTRIBUTION",
    "TRUE_REWARD",
    "ClosedFormResult",
    "apply_a0",
    "auxiliary_direction",
    "bernoulli_kl",
    "btl_cross_entropy",
    "bt_rm_optimal_w",
    "categorical_kl",
    "closed_form_results",
    "exact_true_regret",
    "exact_true_regret_derivative",
    "expected_reward",
    "fisher_from_reference_policy",
    "fixed_kl_learned_theta",
    "fixed_kl_local_regret",
    "fixed_kl_target_theta",
    "learned_policy",
    "learned_theta",
    "local_fisher_kl",
    "local_prorm_regret",
    "logit",
    "ordered_iid_pair_distribution",
    "pair_fisher",
    "pair_reward_moment",
    "population_nll",
    "policy_from_theta",
    "reward_vector",
    "score_mean",
    "sigmoid",
    "softplus",
    "subtract",
    "target_policy_objective",
    "true_optimal_policy",
    "true_optimal_theta",
]
