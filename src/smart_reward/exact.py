"""Exact-delta reward learning for the frozen main experiment.

The module contains no model loading or annotation simulation.  Every split is
an immutable node table with exact standardized oracle rewards.  Pairwise
edges are the complete ``j < k`` graph reconstructed from those nodes.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .linear import DampedEmpiricalFisher
from .pcg import PCGResult, pcg

FisherEstimator = Literal[
    "raw_second_moment",
    "prompt_centered_sample_covariance",
]


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _head_sha256(weight: torch.Tensor) -> str:
    value = weight.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(bytes(value.view(torch.uint8).tolist()))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExactSplitData:
    """One prompt split with node geometry and exact standardized ``r*``."""

    prompt_ids: tuple[str, ...]
    policy_scores: torch.Tensor
    reward_features: torch.Tensor
    true_rewards: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_ids, tuple) or not self.prompt_ids:
            raise TypeError("prompt_ids must be a non-empty tuple")
        if any(not isinstance(value, str) or not value for value in self.prompt_ids):
            raise ValueError("prompt_ids must contain non-empty strings")
        if len(set(self.prompt_ids)) != len(self.prompt_ids):
            raise ValueError("prompt_ids must be unique within a split")
        for name, tensor in (
            ("policy_scores", self.policy_scores),
            ("reward_features", self.reward_features),
            ("true_rewards", self.true_rewards),
        ):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise TypeError(f"{name} must be a floating-point torch.Tensor")
            if tensor.requires_grad:
                raise ValueError(f"{name} must be detached")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite")
        if self.policy_scores.ndim != 3:
            raise ValueError("policy_scores must have shape (P, M, D)")
        prompts, candidates, policy_dimension = self.policy_scores.shape
        if prompts != len(self.prompt_ids) or candidates < 2 or policy_dimension < 1:
            raise ValueError("policy_scores dimensions do not match the prompt split")
        if self.reward_features.ndim != 3 or self.reward_features.shape[:2] != (
            prompts,
            candidates,
        ):
            raise ValueError("reward_features must have shape (P, M, H)")
        if self.reward_features.shape[2] < 1:
            raise ValueError("reward feature dimension must be positive")
        if self.true_rewards.shape != (prompts, candidates):
            raise ValueError("true_rewards must have shape (P, M)")
        for tensor in (self.reward_features, self.true_rewards):
            if (
                tensor.dtype != self.policy_scores.dtype
                or tensor.device != self.policy_scores.device
            ):
                raise ValueError("all split tensors must share dtype and device")

    @property
    def num_prompts(self) -> int:
        return self.policy_scores.shape[0]

    @property
    def num_candidates(self) -> int:
        return self.policy_scores.shape[1]

    @property
    def policy_dimension(self) -> int:
        return self.policy_scores.shape[2]

    @property
    def reward_dimension(self) -> int:
        return self.reward_features.shape[2]

    @property
    def num_edges(self) -> int:
        return self.num_prompts * self.num_candidates * (self.num_candidates - 1) // 2


@dataclass(frozen=True, slots=True)
class ExactDeltaExperiment:
    """Disjoint train/validation/test node tables for exact-delta learning."""

    train: ExactSplitData
    validation: ExactSplitData
    test: ExactSplitData

    def __post_init__(self) -> None:
        splits = (self.train, self.validation, self.test)
        if not all(isinstance(split, ExactSplitData) for split in splits):
            raise TypeError("all experiment splits must be ExactSplitData")
        reference = self.train
        for split in splits[1:]:
            if (
                split.num_candidates != reference.num_candidates
                or split.policy_dimension != reference.policy_dimension
                or split.reward_dimension != reference.reward_dimension
                or split.policy_scores.dtype != reference.policy_scores.dtype
                or split.policy_scores.device != reference.policy_scores.device
            ):
                raise ValueError("all experiment splits must share tensor geometry")
        prompt_sets = [set(split.prompt_ids) for split in splits]
        if any(
            prompt_sets[i].intersection(prompt_sets[j]) for i in range(3) for j in range(i + 1, 3)
        ):
            raise ValueError("train, validation, and test prompt IDs must be disjoint")


def pair_indices(num_candidates: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return every unordered candidate pair with deterministic ``j < k`` orientation."""

    count = _positive_integer("num_candidates", num_candidates)
    if count < 2:
        raise ValueError("num_candidates must be at least two")
    return torch.combinations(torch.arange(count, device=device), r=2)


def pairwise_differences(values: torch.Tensor) -> torch.Tensor:
    """Return all ``values[:, j] - values[:, k]`` for ``j < k``."""

    if not isinstance(values, torch.Tensor) or values.ndim < 2:
        raise TypeError("values must be a torch.Tensor with shape (P, M, ...)")
    pairs = pair_indices(values.shape[1], device=values.device)
    return values.index_select(1, pairs[:, 0]) - values.index_select(1, pairs[:, 1])


def empirical_fisher_score_rows(
    scores: torch.Tensor,
    estimator: FisherEstimator,
) -> torch.Tensor:
    """Return rows whose normalized Gram matrix is the configured Fisher estimate."""

    if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
        raise TypeError("scores must be a tensor with shape (P, M, D)")
    if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite and floating point")
    prompts, candidates, dimension = scores.shape
    if prompts < 1 or candidates < 2 or dimension < 1:
        raise ValueError("scores must have positive P and D and at least two candidates")
    if estimator == "raw_second_moment":
        return scores.reshape(-1, dimension)
    if estimator == "prompt_centered_sample_covariance":
        centered = scores - scores.mean(dim=1, keepdim=True)
        bessel_scale = math.sqrt(candidates / (candidates - 1.0))
        return (centered * bessel_scale).reshape(-1, dimension)
    raise ValueError(f"unsupported Fisher estimator: {estimator!r}")


@dataclass(frozen=True, slots=True)
class MLETrainingConfig:
    max_iterations: int = 200
    history_size: int = 20
    gradient_tolerance: float = 1.0e-7
    change_tolerance: float = 1.0e-12
    microbatch_size: int = 256

    def __post_init__(self) -> None:
        _positive_integer("max_iterations", self.max_iterations)
        _positive_integer("history_size", self.history_size)
        _positive_float("gradient_tolerance", self.gradient_tolerance)
        _positive_float("change_tolerance", self.change_tolerance)
        _positive_integer("microbatch_size", self.microbatch_size)


@dataclass(frozen=True, slots=True)
class ProTrainingConfig:
    relative_damping: float = 1.0e-3
    fisher_estimator: FisherEstimator = "prompt_centered_sample_covariance"
    inner_max_iterations: int = 200
    inner_tolerance: float = 1.0e-5
    outer_max_iterations: int = 256
    outer_tolerance: float = 1.0e-6
    residual_recompute_interval: int = 20

    def __post_init__(self) -> None:
        _positive_float("relative_damping", self.relative_damping)
        if self.fisher_estimator not in {
            "raw_second_moment",
            "prompt_centered_sample_covariance",
        }:
            raise ValueError("fisher_estimator is not implemented")
        _positive_integer("inner_max_iterations", self.inner_max_iterations)
        _positive_float("inner_tolerance", self.inner_tolerance)
        _positive_integer("outer_max_iterations", self.outer_max_iterations)
        _positive_float("outer_tolerance", self.outer_tolerance)
        _positive_integer("residual_recompute_interval", self.residual_recompute_interval)


@dataclass(frozen=True, slots=True)
class RewardFitResult:
    method: str
    weight: torch.Tensor
    objective: float
    gradient_norm: float
    converged: bool
    iterations: int
    head_sha256: str
    inner_pcg_calls: int = 0
    relative_residual: float | None = None
    effective_inner_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"MLE-RM", "Pro-RM"}:
            raise ValueError("method must be 'MLE-RM' or 'Pro-RM'")
        if not isinstance(self.weight, torch.Tensor) or self.weight.ndim != 1:
            raise TypeError("weight must be a one-dimensional torch.Tensor")
        if self.weight.requires_grad or not bool(torch.isfinite(self.weight).all()):
            raise ValueError("weight must be detached and finite")
        for name in ("objective", "gradient_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool")
        _positive_integer("iterations", self.iterations)
        if self.inner_pcg_calls < 0:
            raise ValueError("inner_pcg_calls must be non-negative")
        for name in ("relative_residual", "effective_inner_tolerance"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "objective": self.objective,
            "gradient_norm": self.gradient_norm,
            "converged": self.converged,
            "iterations": self.iterations,
            "head_sha256": self.head_sha256,
            "inner_pcg_calls": self.inner_pcg_calls,
            "relative_residual": self.relative_residual,
            "effective_inner_tolerance": self.effective_inner_tolerance,
            "head_weight": self.weight.detach().cpu().tolist(),
        }


def _edge_design(split: ExactSplitData) -> tuple[torch.Tensor, torch.Tensor]:
    feature_differences = pairwise_differences(split.reward_features).reshape(
        -1, split.reward_dimension
    )
    target_margins = pairwise_differences(split.true_rewards).reshape(-1)
    return feature_differences, target_margins


def policy_reward_moment(scores: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """Estimate ``E_x Cov_{y|x}(score, reward)`` from a candidate pool."""

    if scores.ndim != 3 or rewards.shape != scores.shape[:2]:
        raise ValueError("scores and rewards must have shapes (P, M, D) and (P, M)")
    centered_scores = scores - scores.mean(dim=1, keepdim=True)
    centered_rewards = rewards - rewards.mean(dim=1, keepdim=True)
    return torch.einsum("pmd,pm->d", centered_scores, centered_rewards) / (
        scores.shape[0] * (scores.shape[1] - 1)
    )


def _mle_parameterization(
    design: torch.Tensor,
    *,
    maximum_identifiable_rank: int,
    source_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return stable MLE coordinates and a map back to head space.

    Complete pairwise edges from one prompt span at most ``M - 1`` directions.
    When that structural rank bound is below the head dimension, optimizing the
    1536-dimensional head directly is needlessly singular.  A compact SVD gives
    scaled orthogonal coordinates for exactly the same feasible logits.  The
    scale makes the configured coordinate-gradient tolerance upper-bound the
    original head-gradient norm.  Mapping through the pseudoinverse returns the
    minimum-norm head.

    Once the design can be full column rank, a reduced QR supplies the same
    scaled orthogonal coordinates without changing the represented logits.  A
    numerically rank-deficient QR falls back to the rank-revealing SVD path.
    """

    if maximum_identifiable_rank >= design.shape[1]:
        orthogonal, upper = torch.linalg.qr(design, mode="reduced")
        diagonal = torch.diagonal(upper).abs()
        tolerance = max(design.shape) * source_epsilon * diagonal.max()
        if bool((diagonal > tolerance).all()):
            coordinate_scale = torch.linalg.vector_norm(upper)
            identity = torch.eye(
                upper.shape[0],
                dtype=upper.dtype,
                device=upper.device,
            )
            head_map = torch.linalg.solve_triangular(
                upper,
                identity * coordinate_scale,
                upper=True,
            )
            return (
                (orthogonal * coordinate_scale).contiguous(),
                head_map.contiguous(),
            )
    left, singular_values, right_transpose = torch.linalg.svd(
        design,
        full_matrices=False,
    )
    if singular_values.numel() == 0:
        raise RuntimeError("MLE edge design has no singular values")
    tolerance = max(design.shape) * source_epsilon * singular_values[0]
    rank = int(torch.count_nonzero(singular_values > tolerance).item())
    if rank < 1:
        raise RuntimeError("MLE edge design has zero numerical rank")
    retained_singular_values = singular_values[:rank]
    coordinate_scale = torch.linalg.vector_norm(retained_singular_values)
    coordinate_design = (left[:, :rank] * coordinate_scale).contiguous()
    head_map = (
        right_transpose[:rank].mT * (coordinate_scale / retained_singular_values).unsqueeze(0)
    ).contiguous()
    return coordinate_design, head_map


def _mle_objective_and_gradient(
    design: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = design @ weight
    objective = F.binary_cross_entropy_with_logits(logits, targets)
    gradient = design.mT @ (torch.sigmoid(logits) - targets) / design.shape[0]
    return objective, gradient


def fit_mle_reward(
    train: ExactSplitData,
    config: MLETrainingConfig | None = None,
) -> RewardFitResult:
    """Fit the convex exact-soft-label Bradley--Terry linear head."""

    effective = MLETrainingConfig() if config is None else config
    if not isinstance(effective, MLETrainingConfig):
        raise TypeError("config must be MLETrainingConfig")
    design, target_margins = _edge_design(train)
    source_epsilon = torch.finfo(design.dtype).eps
    design = design.to(dtype=torch.float64)
    target_margins = target_margins.to(dtype=torch.float64)
    targets = torch.sigmoid(target_margins).detach()
    coordinate_design, head_map = _mle_parameterization(
        design,
        maximum_identifiable_rank=train.num_prompts * (train.num_candidates - 1),
        source_epsilon=source_epsilon,
    )
    parameter = torch.zeros(
        coordinate_design.shape[1],
        dtype=torch.float64,
        device=train.reward_features.device,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [parameter],
        lr=1.0,
        max_iter=effective.max_iterations,
        history_size=effective.history_size,
        tolerance_grad=effective.gradient_tolerance,
        # In identifiable logit coordinates, tiny loss changes can coexist with
        # a head-space gradient above the scientific convergence gate.  Keep
        # LBFGS gradient-controlled for this structurally underdetermined case.
        tolerance_change=(
            effective.change_tolerance
            if head_map is None
            else min(effective.change_tolerance, torch.finfo(design.dtype).eps)
        ),
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), dtype=parameter.dtype, device=parameter.device)
        total = coordinate_design.shape[0]
        for start in range(0, total, effective.microbatch_size):
            stop = min(start + effective.microbatch_size, total)
            chunk_loss = F.binary_cross_entropy_with_logits(
                coordinate_design[start:stop] @ parameter,
                targets[start:stop],
            )
            scale = (stop - start) / total
            (chunk_loss * scale).backward()
            objective = objective + chunk_loss.detach() * scale
        return objective

    optimizer.step(closure)
    detached_parameter = parameter.detach()
    detached = detached_parameter.clone() if head_map is None else head_map @ detached_parameter
    objective, full_gradient = _mle_objective_and_gradient(design, targets, detached)
    gradient_norm = float(torch.linalg.vector_norm(full_gradient).item())
    return RewardFitResult(
        method="MLE-RM",
        weight=detached,
        objective=float(objective.item()),
        gradient_norm=gradient_norm,
        converged=gradient_norm <= effective.gradient_tolerance,
        iterations=max(1, closure_calls),
        head_sha256=_head_sha256(detached),
    )


def _require_converged(name: str, result: PCGResult) -> torch.Tensor:
    if not result.converged:
        raise RuntimeError(
            f"{name} did not converge: iterations={result.iterations}, "
            f"relative_residual={result.relative_residual:.3e}"
        )
    return result.solution


@torch.no_grad()
def fit_pro_reward(
    train: ExactSplitData,
    config: ProTrainingConfig | None = None,
) -> RewardFitResult:
    """Solve the exact linear-head ProRM normal equation with nested CG.

    For ``G = A X`` and ``g* = A r*``, the normal equation is

    ``G.T (F + lambda I)^-1 G w = G.T (F + lambda I)^-1 g*``.

    The outer solve is in reward-head coordinates. Each Hessian-vector product
    applies the inverse damped Fisher with an inner PCG solve.
    """

    effective = ProTrainingConfig() if config is None else config
    if not isinstance(effective, ProTrainingConfig):
        raise TypeError("config must be ProTrainingConfig")
    solve_dtype = torch.float64
    scores = train.policy_scores.to(dtype=solve_dtype)
    features = train.reward_features.to(dtype=solve_dtype)
    rewards = train.true_rewards.to(dtype=solve_dtype)
    centered_scores = scores - scores.mean(dim=1, keepdim=True)
    centered_features = features - features.mean(dim=1, keepdim=True)
    denominator = train.num_prompts * (train.num_candidates - 1)
    score_rows = centered_scores.reshape(-1, train.policy_dimension)
    feature_rows = centered_features.reshape(-1, train.reward_dimension)
    feature_moment = score_rows.mT @ feature_rows / denominator
    target_moment = policy_reward_moment(scores, rewards)

    fisher_rows = empirical_fisher_score_rows(scores, effective.fisher_estimator)
    mean_fisher_diagonal = float(fisher_rows.square().mean(dim=0).mean().item())
    if not math.isfinite(mean_fisher_diagonal) or mean_fisher_diagonal <= 0.0:
        raise ValueError("train policy Fisher has non-positive mean diagonal")
    damping = effective.relative_damping * mean_fisher_diagonal
    fisher = DampedEmpiricalFisher(fisher_rows, damping=damping)
    inner_calls = 0
    # A nested solve cannot reliably target an outer residual below the error
    # of its Fisher inverse. Treat the configured inner tolerance as an upper
    # bound and tighten it relative to the requested outer accuracy.
    effective_inner_tolerance = min(
        effective.inner_tolerance,
        effective.outer_tolerance * 0.1,
    )

    def inverse_fisher(vector: torch.Tensor) -> torch.Tensor:
        nonlocal inner_calls
        inner_calls += 1
        result = pcg(
            fisher.matvec,
            vector,
            inverse_diagonal=fisher.pcg_inverse_diagonal(),
            max_iterations=effective.inner_max_iterations,
            tolerance=effective_inner_tolerance,
            residual_recompute_interval=effective.residual_recompute_interval,
        )
        return _require_converged("inner Fisher PCG", result)

    target_natural = inverse_fisher(target_moment)
    rhs = feature_moment.mT @ target_natural

    def normal_matvec(vector: torch.Tensor) -> torch.Tensor:
        return feature_moment.mT @ inverse_fisher(feature_moment @ vector)

    normal_diagonal_proxy = feature_moment.square().sum(dim=0)
    diagonal_scale = float(normal_diagonal_proxy.mean().item())
    if not math.isfinite(diagonal_scale) or diagonal_scale <= 0.0:
        raise ValueError("Pro-RM feature moment has non-positive diagonal scale")
    diagonal_floor = torch.finfo(normal_diagonal_proxy.dtype).eps * diagonal_scale
    inverse_normal_diagonal = normal_diagonal_proxy.clamp_min(diagonal_floor).reciprocal()
    result = pcg(
        normal_matvec,
        rhs,
        inverse_diagonal=inverse_normal_diagonal,
        max_iterations=effective.outer_max_iterations,
        tolerance=effective.outer_tolerance,
        residual_recompute_interval=effective.residual_recompute_interval,
    )
    solution = _require_converged("Pro-RM normal-equation CG", result)
    error_moment = feature_moment @ solution - target_moment
    weighted_error = inverse_fisher(error_moment)
    objective = float((torch.dot(error_moment, weighted_error) / 2.0).item())
    gradient = feature_moment.mT @ weighted_error
    weight = solution.detach().clone()
    return RewardFitResult(
        method="Pro-RM",
        weight=weight,
        objective=objective,
        gradient_norm=float(torch.linalg.vector_norm(gradient).item()),
        converged=result.converged,
        iterations=max(1, result.iterations),
        head_sha256=_head_sha256(weight),
        inner_pcg_calls=inner_calls,
        relative_residual=result.relative_residual,
        effective_inner_tolerance=effective_inner_tolerance,
    )


@dataclass(frozen=True, slots=True)
class RewardEvaluation:
    pair_kl: float
    soft_btl_nll: float
    probability_mse: float
    pairwise_accuracy: float
    centered_reward_nmse: float
    centered_reward_nmse_defined_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {
            "pair_kl": self.pair_kl,
            "soft_btl_nll": self.soft_btl_nll,
            "probability_mse": self.probability_mse,
            "pairwise_accuracy": self.pairwise_accuracy,
            "centered_reward_nmse": self.centered_reward_nmse,
            "centered_reward_nmse_defined_fraction": (self.centered_reward_nmse_defined_fraction),
        }


@torch.no_grad()
def evaluate_reward_head(split: ExactSplitData, weight: torch.Tensor) -> RewardEvaluation:
    """Evaluate reward fit on all pairs while keeping prompts as the sampling unit."""

    if not isinstance(weight, torch.Tensor) or weight.shape != (split.reward_dimension,):
        raise ValueError("weight must match the reward feature dimension")
    features = split.reward_features.to(dtype=torch.float64)
    target_rewards = split.true_rewards.to(dtype=torch.float64)
    predicted = features @ weight.to(
        device=split.reward_features.device,
        dtype=torch.float64,
    )
    predicted_margins = pairwise_differences(predicted)
    target_margins = pairwise_differences(target_rewards)
    target_probabilities = torch.sigmoid(target_margins)
    predicted_probabilities = torch.sigmoid(predicted_margins)
    prompt_nll = F.binary_cross_entropy_with_logits(
        predicted_margins,
        target_probabilities,
        reduction="none",
    ).mean(dim=1)
    prompt_probability_mse = (predicted_probabilities - target_probabilities).square().mean(dim=1)
    oracle_entropy = F.binary_cross_entropy_with_logits(
        target_margins,
        target_probabilities,
        reduction="none",
    ).mean(dim=1)
    nonzero = target_margins != 0.0
    correct = torch.where(
        nonzero,
        torch.sign(predicted_margins) == torch.sign(target_margins),
        torch.ones_like(nonzero),
    ).to(dtype=predicted.dtype)
    centered_predicted = predicted - predicted.mean(dim=1, keepdim=True)
    centered_target = target_rewards - target_rewards.mean(dim=1, keepdim=True)
    centered_squared_error = (centered_predicted - centered_target).square().mean(dim=1)
    centered_target_energy = centered_target.square().mean(dim=1)
    positive_energy = centered_target_energy > 0.0
    if not bool(positive_energy.any()):
        raise ValueError("centered oracle reward must have positive energy for some prompt")
    return RewardEvaluation(
        pair_kl=float((prompt_nll - oracle_entropy).mean().item()),
        soft_btl_nll=float(prompt_nll.mean().item()),
        probability_mse=float(prompt_probability_mse.mean().item()),
        pairwise_accuracy=float(correct.mean(dim=1).mean().item()),
        centered_reward_nmse=float(
            (centered_squared_error[positive_energy] / centered_target_energy[positive_energy])
            .mean()
            .item()
        ),
        centered_reward_nmse_defined_fraction=float(
            positive_energy.to(dtype=predicted.dtype).mean().item()
        ),
    )


__all__ = [
    "ExactDeltaExperiment",
    "ExactSplitData",
    "FisherEstimator",
    "MLETrainingConfig",
    "ProTrainingConfig",
    "RewardEvaluation",
    "RewardFitResult",
    "evaluate_reward_head",
    "empirical_fisher_score_rows",
    "fit_mle_reward",
    "fit_pro_reward",
    "pair_indices",
    "pairwise_differences",
    "policy_reward_moment",
]
