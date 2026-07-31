"""Reward-fit, local-policy, tabular, and rollout evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .exact import (
    ExactSplitData,
    FisherEstimator,
    empirical_fisher_score_rows,
    policy_reward_moment,
)
from .linear import DampedEmpiricalFisher
from .pcg import pcg
from .policy_update import scale_direction_to_quadratic_kl


@dataclass(frozen=True, slots=True)
class GeometrySettings:
    fisher_estimator: FisherEstimator
    relative_damping: float
    cg_tolerance: float
    cg_max_iterations: int
    residual_recompute_interval: int


@dataclass(frozen=True, slots=True)
class LocalPolicyMetrics:
    local_regret: float
    fisher_cosine: float
    local_target_utility: float
    tabular_optimal_utility: float
    tabular_regret: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrustRegionLocalMetrics:
    fisher_cosine: float
    local_reward_improvement: float
    quadratic_forward_kl: float
    finite_pool_reward_improvement: float
    finite_pool_forward_kl: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _geometry(
    split: ExactSplitData,
    settings: GeometrySettings,
) -> tuple[DampedEmpiricalFisher, DampedEmpiricalFisher, float]:
    scores = split.policy_scores.to(dtype=torch.float64)
    rows = empirical_fisher_score_rows(scores, settings.fisher_estimator)
    fisher = DampedEmpiricalFisher(rows, damping=0.0)
    mean_diagonal = float(fisher.diagonal().mean().item())
    damping = settings.relative_damping * mean_diagonal
    return fisher, DampedEmpiricalFisher(rows, damping=damping), damping


def solve_natural_direction(
    split: ExactSplitData,
    rewards: torch.Tensor,
    settings: GeometrySettings,
) -> torch.Tensor:
    """Solve the beta-free direction ``(F + lambda I)^-1 g(reward)``."""

    if rewards.shape != split.true_rewards.shape:
        raise ValueError("rewards must have shape (P, M)")
    _, damped, _ = _geometry(split, settings)
    moment = policy_reward_moment(
        split.policy_scores.to(dtype=torch.float64),
        rewards.to(device=split.policy_scores.device, dtype=torch.float64),
    )
    result = pcg(
        damped.matvec,
        moment,
        inverse_diagonal=damped.pcg_inverse_diagonal(),
        max_iterations=settings.cg_max_iterations,
        tolerance=settings.cg_tolerance,
        residual_recompute_interval=settings.residual_recompute_interval,
    )
    if not result.converged:
        raise RuntimeError(
            "natural-direction CG did not converge: "
            f"iterations={result.iterations}, residual={result.relative_residual:.3e}"
        )
    return result.solution


@torch.no_grad()
def validate_natural_direction(
    split: ExactSplitData,
    rewards: torch.Tensor,
    direction: torch.Tensor,
    settings: GeometrySettings,
) -> float:
    """Validate a saved natural direction without repeating its PCG solve."""

    if rewards.shape != split.true_rewards.shape:
        raise ValueError("rewards must have shape (P, M)")
    if (
        not isinstance(direction, torch.Tensor)
        or direction.shape != (split.policy_dimension,)
        or not bool(torch.isfinite(direction).all())
    ):
        raise ValueError("direction must be a finite policy-tangent vector")
    _, damped, _ = _geometry(split, settings)
    moment = policy_reward_moment(
        split.policy_scores.to(dtype=torch.float64),
        rewards.to(device=split.policy_scores.device, dtype=torch.float64),
    )
    moment_norm = torch.linalg.vector_norm(moment)
    if float(moment_norm.item()) == 0.0:
        raise ValueError("natural-direction equation has a zero right-hand side")
    candidate = direction.to(
        device=split.policy_scores.device,
        dtype=torch.float64,
    )
    residual = damped.matvec(candidate) - moment
    relative_residual = float((torch.linalg.vector_norm(residual) / moment_norm).item())
    if not math.isfinite(relative_residual) or relative_residual > settings.cg_tolerance:
        raise RuntimeError(
            "reused natural direction did not pass recomputed residual gate: "
            f"relative_residual={relative_residual:.3e}"
        )
    return relative_residual


def _quadratic(vector: torch.Tensor, operator: DampedEmpiricalFisher) -> torch.Tensor:
    return torch.dot(vector, operator.matvec(vector))


def _fisher_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    fisher: DampedEmpiricalFisher,
) -> float:
    numerator = torch.dot(left, fisher.matvec(right))
    denominator = torch.sqrt(_quadratic(left, fisher) * _quadratic(right, fisher))
    if float(denominator.item()) <= 0.0:
        return 0.0
    return float((numerator / denominator).clamp(-1.0, 1.0).item())


def _tabular_utility(
    split: ExactSplitData,
    direction: torch.Tensor,
    beta: float,
) -> tuple[float, float]:
    """Return candidate-conditional utility and its exact oracle optimum.

    The finite candidate pool treats the empirical reference distribution as
    uniform.  This is exact for that discrete pool, not for the LLM population.
    """

    beta_value = float(beta)
    scores = split.policy_scores.to(dtype=torch.float64)
    rewards = split.true_rewards.to(dtype=torch.float64)
    logits = torch.einsum("pmd,d->pm", scores, direction / beta_value)
    policy = torch.softmax(logits, dim=1)
    log_reference = -math.log(split.num_candidates)
    kl = (policy * (torch.log(policy) - log_reference)).sum(dim=1)
    utility = (policy * rewards).sum(dim=1) - beta_value * kl
    optimum = beta_value * (
        torch.logsumexp(rewards / beta_value, dim=1) - math.log(split.num_candidates)
    )
    return float(utility.mean().item()), float(optimum.mean().item())


@torch.no_grad()
def evaluate_local_policy(
    split: ExactSplitData,
    direction: torch.Tensor,
    *,
    beta: float,
    settings: GeometrySettings,
) -> LocalPolicyMetrics:
    """Evaluate one train-fitted direction on held-out reference candidates."""

    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value <= 0.0:
        raise ValueError("beta must be finite and positive")
    if (
        not isinstance(direction, torch.Tensor)
        or direction.shape != (split.policy_dimension,)
        or not bool(torch.isfinite(direction).all())
    ):
        raise ValueError("direction must be a finite policy-tangent vector")
    fisher, damped, _ = _geometry(split, settings)
    predicted = direction.to(device=split.policy_scores.device, dtype=torch.float64)
    oracle = solve_natural_direction(split, split.true_rewards, settings)
    difference = predicted - oracle
    local_regret = float((_quadratic(difference, damped) / (2.0 * beta_value)).item())
    oracle_moment = policy_reward_moment(
        split.policy_scores.to(dtype=torch.float64),
        split.true_rewards.to(dtype=torch.float64),
    )
    deployed = predicted / beta_value
    local_utility = torch.dot(oracle_moment, deployed) - (
        beta_value * _quadratic(deployed, damped) / 2.0
    )
    tabular_utility, tabular_optimum = _tabular_utility(split, predicted, beta_value)
    return LocalPolicyMetrics(
        local_regret=local_regret,
        fisher_cosine=_fisher_cosine(predicted, oracle, fisher),
        local_target_utility=float(local_utility.item()),
        tabular_optimal_utility=tabular_optimum,
        tabular_regret=tabular_optimum - tabular_utility,
    )


def evaluate_reference_policy(
    split: ExactSplitData,
    *,
    beta: float,
    settings: GeometrySettings,
) -> LocalPolicyMetrics:
    zeros = torch.zeros(
        split.policy_dimension,
        device=split.policy_scores.device,
        dtype=torch.float64,
    )
    return evaluate_local_policy(split, zeros, beta=beta, settings=settings)


@torch.no_grad()
def evaluate_trpo_local_policy(
    split: ExactSplitData,
    update: torch.Tensor,
    *,
    kl_target: float,
    settings: GeometrySettings,
) -> TrustRegionLocalMetrics:
    """Evaluate a fixed train-scaled update on a held-out candidate pool."""

    target = float(kl_target)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("kl_target must be finite and positive")
    if (
        not isinstance(update, torch.Tensor)
        or update.shape != (split.policy_dimension,)
        or not bool(torch.isfinite(update).all())
    ):
        raise ValueError("update must be a finite policy-tangent vector")
    fisher, _, _ = _geometry(split, settings)
    predicted = update.to(device=split.policy_scores.device, dtype=torch.float64)
    oracle_direction = solve_natural_direction(split, split.true_rewards, settings)
    oracle_update, _, _ = scale_direction_to_quadratic_kl(
        oracle_direction,
        fisher.matvec,
        kl_target=target,
    )
    rewards = split.true_rewards.to(dtype=torch.float64)
    scores = split.policy_scores.to(dtype=torch.float64)
    moment = policy_reward_moment(scores, rewards)
    logits = torch.einsum("pmd,d->pm", scores, predicted)
    probabilities = torch.softmax(logits, dim=1)
    reference_reward = rewards.mean(dim=1)
    updated_reward = (probabilities * rewards).sum(dim=1)
    log_reference = -math.log(split.num_candidates)
    finite_kl = (probabilities * (torch.log(probabilities) - log_reference)).sum(dim=1)
    return TrustRegionLocalMetrics(
        fisher_cosine=_fisher_cosine(predicted, oracle_update, fisher),
        local_reward_improvement=float(torch.dot(moment, predicted).item()),
        quadratic_forward_kl=0.5 * float(_quadratic(predicted, fisher).item()),
        finite_pool_reward_improvement=float((updated_reward - reference_reward).mean().item()),
        finite_pool_forward_kl=float(finite_kl.mean().item()),
    )


@torch.no_grad()
def evaluate_trpo_reference_policy(
    split: ExactSplitData,
    *,
    kl_target: float,
    settings: GeometrySettings,
) -> TrustRegionLocalMetrics:
    zeros = torch.zeros(
        split.policy_dimension,
        device=split.policy_scores.device,
        dtype=torch.float64,
    )
    return evaluate_trpo_local_policy(
        split,
        zeros,
        kl_target=kl_target,
        settings=settings,
    )


def summarize_rollouts(
    oracle_rewards: torch.Tensor,
    forward_log_ratios: torch.Tensor,
    *,
    beta: float,
    reference_oracle_rewards: torch.Tensor,
    oracle_ngd_oracle_rewards: torch.Tensor,
    oracle_ngd_forward_log_ratios: torch.Tensor,
) -> dict[str, Any]:
    """Aggregate fresh-policy Monte Carlo samples by prompt, then prompts."""

    tensors = (
        oracle_rewards,
        forward_log_ratios,
        reference_oracle_rewards,
        oracle_ngd_oracle_rewards,
        oracle_ngd_forward_log_ratios,
    )
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in tensors):
        raise TypeError("rollout inputs must have shape (prompts, responses)")
    if any(value.shape != oracle_rewards.shape for value in tensors[1:]):
        raise ValueError("all rollout inputs must have identical shapes")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("rollout inputs must be finite")
    beta_value = float(beta)
    reward_by_prompt = oracle_rewards.mean(dim=1)
    kl_by_prompt = forward_log_ratios.mean(dim=1)
    reference_by_prompt = reference_oracle_rewards.mean(dim=1)
    oracle_reward_by_prompt = oracle_ngd_oracle_rewards.mean(dim=1)
    oracle_kl_by_prompt = oracle_ngd_forward_log_ratios.mean(dim=1)
    utility_by_prompt = reward_by_prompt - beta_value * kl_by_prompt
    reference_utility = reference_by_prompt
    oracle_utility = oracle_reward_by_prompt - beta_value * oracle_kl_by_prompt
    return {
        "oracle_reward": float(reward_by_prompt.mean().item()),
        "reward_improvement": float((reward_by_prompt - reference_by_prompt).mean().item()),
        "forward_kl": float(kl_by_prompt.mean().item()),
        "regularized_utility": float(utility_by_prompt.mean().item()),
        "utility_improvement": float((utility_by_prompt - reference_utility).mean().item()),
        "oracle_ngd_regret": float((oracle_utility - utility_by_prompt).mean().item()),
        "sampling_unit": "prompt",
        "num_prompts": oracle_rewards.shape[0],
        "responses_per_prompt": oracle_rewards.shape[1],
    }


def summarize_trpo_rollouts(
    oracle_rewards: torch.Tensor,
    forward_log_ratios: torch.Tensor,
    *,
    kl_target: float,
    reference_oracle_rewards: torch.Tensor,
) -> dict[str, Any]:
    """Aggregate matched-KL rollout metrics by prompt."""

    tensors = (oracle_rewards, forward_log_ratios, reference_oracle_rewards)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in tensors):
        raise TypeError("rollout inputs must have shape (prompts, responses)")
    if any(value.shape != oracle_rewards.shape for value in tensors[1:]):
        raise ValueError("all rollout inputs must have identical shapes")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("rollout inputs must be finite")
    target = float(kl_target)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("kl_target must be finite and positive")
    reward_by_prompt = oracle_rewards.mean(dim=1)
    kl_by_prompt = forward_log_ratios.mean(dim=1)
    reference_by_prompt = reference_oracle_rewards.mean(dim=1)
    return {
        "oracle_reward": float(reward_by_prompt.mean().item()),
        "reward_improvement": float((reward_by_prompt - reference_by_prompt).mean().item()),
        "forward_kl": float(kl_by_prompt.mean().item()),
        "kl_target_error": float(kl_by_prompt.mean().item() - target),
        "sampling_unit": "prompt",
        "num_prompts": oracle_rewards.shape[0],
        "responses_per_prompt": oracle_rewards.shape[1],
    }


__all__ = [
    "GeometrySettings",
    "LocalPolicyMetrics",
    "TrustRegionLocalMetrics",
    "evaluate_local_policy",
    "evaluate_reference_policy",
    "evaluate_trpo_local_policy",
    "evaluate_trpo_reference_policy",
    "solve_natural_direction",
    "summarize_rollouts",
    "summarize_trpo_rollouts",
    "validate_natural_direction",
]
