"""Policy-relevant local-regret and natural-direction metrics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .linear import (
    DampedEmpiricalFisher,
    FisherSolveDType,
    resolve_fisher_solve_dtype,
)
from .pcg import pcg


def gauge_center(rewards: torch.Tensor, *, candidate_dim: int = -1) -> torch.Tensor:
    """Remove the unidentifiable per-prompt additive reward gauge.

    ``candidate_dim`` identifies the responses belonging to the same prompt.
    For the standard layout ``(num_prompts, num_candidates)``, use the default.
    """

    if not isinstance(rewards, torch.Tensor):
        raise TypeError("rewards must be a torch.Tensor")
    if not rewards.is_floating_point():
        raise TypeError("rewards must have a floating-point dtype")
    if rewards.ndim < 1 or rewards.numel() < 1:
        raise ValueError("rewards must be non-empty")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite")
    normalized_dim = candidate_dim if candidate_dim >= 0 else rewards.ndim + candidate_dim
    if normalized_dim < 0 or normalized_dim >= rewards.ndim:
        raise ValueError("candidate_dim is out of range")
    return rewards - rewards.mean(dim=normalized_dim, keepdim=True)


def _validate_metric_inputs(
    score_matrix: torch.Tensor,
    rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(score_matrix, torch.Tensor) or not isinstance(rewards, torch.Tensor):
        raise TypeError("score_matrix and rewards must be torch.Tensor objects")
    if rewards.ndim < 1 or rewards.numel() < 1:
        raise ValueError("rewards must be non-empty")
    if score_matrix.ndim != rewards.ndim + 1:
        raise ValueError("score_matrix must have shape rewards.shape + (dimension,)")
    if score_matrix.shape[:-1] != rewards.shape or score_matrix.shape[-1] < 1:
        raise ValueError("score_matrix must have shape rewards.shape + (dimension,)")
    if not score_matrix.is_floating_point() or not rewards.is_floating_point():
        raise TypeError("score_matrix and rewards must have floating-point dtypes")
    if score_matrix.dtype != rewards.dtype or score_matrix.device != rewards.device:
        raise ValueError("score_matrix and rewards must share dtype and device")
    if not bool(torch.isfinite(score_matrix).all()) or not bool(torch.isfinite(rewards).all()):
        raise ValueError("score_matrix and rewards must be finite")
    return score_matrix.reshape(-1, score_matrix.shape[-1]), rewards.reshape(-1)


def _prepare_covariance_moment_samples(
    score_matrix: torch.Tensor,
    rewards: torch.Tensor,
    *,
    center_candidates: bool,
    candidate_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened samples normalized for the reward covariance moment.

    With prompt/candidate data, both score and reward are centered within each
    prompt and multiplied by ``sqrt(M / (M - 1))``.  Taking an ordinary mean
    over the resulting ``P*M`` rows is therefore exactly the average unbiased
    sample cross-covariance, whose denominator is ``P*(M-1)``.  This scaling
    applies to ``A_hat r`` only; the default Fisher remains the node estimator
    ``sum(s s.T)/(P*M)`` used during training.
    """

    flat_scores, flat_rewards = _validate_metric_inputs(score_matrix, rewards)
    if not center_candidates:
        return flat_scores, flat_rewards
    if rewards.ndim < 2:
        raise ValueError(
            "center_candidates=True requires a prompt/candidate reward layout; "
            "set it to False for flat population samples"
        )
    normalized_dim = candidate_dim if candidate_dim >= 0 else rewards.ndim + candidate_dim
    if normalized_dim < 0 or normalized_dim >= rewards.ndim:
        raise ValueError("candidate_dim is out of range")
    num_candidates = rewards.shape[normalized_dim]
    if num_candidates < 2:
        raise ValueError("per-prompt covariance requires at least two candidates")

    centered_scores = score_matrix - score_matrix.mean(
        dim=normalized_dim,
        keepdim=True,
    )
    centered_rewards = rewards - rewards.mean(dim=normalized_dim, keepdim=True)
    bessel_scale = math.sqrt(num_candidates / (num_candidates - 1.0))
    return (
        (centered_scores * bessel_scale).reshape(-1, score_matrix.shape[-1]),
        (centered_rewards * bessel_scale).reshape(-1),
    )


def empirical_fisher_matrix(score_matrix: torch.Tensor) -> torch.Tensor:
    """Form ``S.T S / n`` for metric evaluation on controlled test pools."""

    if not isinstance(score_matrix, torch.Tensor):
        raise TypeError("score_matrix must be a torch.Tensor")
    if score_matrix.ndim < 2 or score_matrix.shape[-1] < 1:
        raise ValueError("score_matrix must have at least one sample and one feature dimension")
    if score_matrix.numel() == 0:
        raise ValueError("score_matrix must be non-empty")
    if not score_matrix.is_floating_point():
        raise TypeError("score_matrix must have a floating-point dtype")
    if not bool(torch.isfinite(score_matrix).all()):
        raise ValueError("score_matrix must be finite")
    flat_scores = score_matrix.reshape(-1, score_matrix.shape[-1])
    return flat_scores.mT @ flat_scores / flat_scores.shape[0]


def policy_reward_moment(
    score_matrix: torch.Tensor,
    rewards: torch.Tensor,
    *,
    center_candidates: bool = True,
    candidate_dim: int = -1,
) -> torch.Tensor:
    """Estimate ``A r`` from node scores and rewards.

    By default this is the average per-prompt sample cross-covariance

    ``sum((s_ij-sbar_i) * (r_ij-rbar_i)) / (P * (M-1))``.

    This is gauge invariant and uses the finite-sample covariance correction.
    With ``center_candidates=False``, it instead returns the raw node moment
    ``S.T r / n``.
    """

    flat_scores, flat_rewards = _prepare_covariance_moment_samples(
        score_matrix,
        rewards,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
    )
    return flat_scores.mT @ flat_rewards / flat_scores.shape[0]


def _validate_damping_beta(damping: float, beta: float = 1.0) -> tuple[float, float]:
    damping_value = float(damping)
    beta_value = float(beta)
    if not math.isfinite(damping_value) or damping_value < 0.0:
        raise ValueError("damping must be finite and non-negative")
    if not math.isfinite(beta_value) or beta_value <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    return damping_value, beta_value


def _promote_policy_geometry(
    score_matrix: torch.Tensor,
    rewards: tuple[torch.Tensor, ...],
    *,
    pcg_dtype: FisherSolveDType,
    fisher_matrix: torch.Tensor | None,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor] | None,
    fisher_diagonal: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, ...],
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Promote built-in Fisher computations to an FP64 numerical workspace.

    A caller-supplied operator owns its dtype contract and is therefore left
    untouched.  The default score Fisher and explicit Fisher matrices are
    fixed tensors, so promoting them removes the FP32 true-residual floor
    observed for the formal high-dimensional geometry.
    """

    solve_dtype = resolve_fisher_solve_dtype(pcg_dtype)
    if fisher_operator is not None:
        if score_matrix.dtype != solve_dtype or any(
            value.dtype != solve_dtype for value in rewards
        ):
            raise ValueError(
                "a caller-supplied fisher_operator requires score/reward inputs "
                "already expressed in pcg_dtype"
            )
        return score_matrix, rewards, fisher_matrix, fisher_diagonal
    return (
        score_matrix.to(dtype=solve_dtype),
        tuple(value.to(dtype=solve_dtype) for value in rewards),
        None if fisher_matrix is None else fisher_matrix.to(dtype=solve_dtype),
        None if fisher_diagonal is None else fisher_diagonal.to(dtype=solve_dtype),
    )


def _resolve_fisher_geometry(
    flat_scores: torch.Tensor,
    *,
    fisher_matrix: torch.Tensor | None,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor] | None,
    fisher_diagonal: torch.Tensor | None,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor | None]:
    """Resolve the default node Fisher or a caller-supplied test geometry."""

    dimension = flat_scores.shape[1]
    if fisher_matrix is not None and fisher_operator is not None:
        raise ValueError("provide at most one of fisher_matrix and fisher_operator")
    if fisher_matrix is not None:
        if not isinstance(fisher_matrix, torch.Tensor):
            raise TypeError("fisher_matrix must be a torch.Tensor")
        if fisher_matrix.shape != (dimension, dimension):
            raise ValueError(f"fisher_matrix must have shape ({dimension}, {dimension})")
        if fisher_matrix.dtype != flat_scores.dtype or fisher_matrix.device != flat_scores.device:
            raise ValueError("fisher_matrix must share dtype and device with score_matrix")
        if not bool(torch.isfinite(fisher_matrix).all()):
            raise ValueError("fisher_matrix must be finite")
        if not torch.allclose(fisher_matrix, fisher_matrix.mT):
            raise ValueError("fisher_matrix must be symmetric")
        if fisher_diagonal is not None:
            raise ValueError("fisher_diagonal is redundant when fisher_matrix is provided")
        return lambda vector: fisher_matrix @ vector, torch.diagonal(fisher_matrix)

    if fisher_operator is None:
        operator = DampedEmpiricalFisher(flat_scores, damping=0.0)
        return (
            operator.matvec,
            # The default empirical Fisher is low rank plus isotropic damping.
            # Retaining that spectrum is substantially better conditioned for
            # CG than applying a coordinate-wise Jacobi rescaling.
            None,
        )
    if not callable(fisher_operator):
        raise TypeError("fisher_operator must be callable")

    def checked_operator(vector: torch.Tensor) -> torch.Tensor:
        product = fisher_operator(vector)
        if not isinstance(product, torch.Tensor):
            raise TypeError("fisher_operator must return a torch.Tensor")
        if product.shape != (dimension,):
            raise ValueError(f"fisher_operator output must have shape ({dimension},)")
        if product.dtype != flat_scores.dtype or product.device != flat_scores.device:
            raise ValueError("fisher_operator output must share dtype and device with score_matrix")
        if not bool(torch.isfinite(product).all()):
            raise ValueError("fisher_operator output must be finite")
        return product

    if fisher_diagonal is not None:
        if not isinstance(fisher_diagonal, torch.Tensor):
            raise TypeError("fisher_diagonal must be a torch.Tensor")
        if fisher_diagonal.shape != (dimension,):
            raise ValueError(f"fisher_diagonal must have shape ({dimension},)")
        if (
            fisher_diagonal.dtype != flat_scores.dtype
            or fisher_diagonal.device != flat_scores.device
        ):
            raise ValueError("fisher_diagonal must share dtype and device with score_matrix")
        if not bool(torch.isfinite(fisher_diagonal).all()):
            raise ValueError("fisher_diagonal must be finite")
        if bool((fisher_diagonal < 0.0).any()):
            raise ValueError("fisher_diagonal must be non-negative")
    return checked_operator, fisher_diagonal


def _matrix_free_damped_solve(
    fisher_operator: Callable[[torch.Tensor], torch.Tensor],
    fisher_diagonal: torch.Tensor | None,
    moment: torch.Tensor,
    damping: float,
    *,
    pcg_tolerance: float,
    pcg_max_iterations: int,
) -> torch.Tensor:
    def damped_operator(vector: torch.Tensor) -> torch.Tensor:
        return fisher_operator(vector) + damping * vector

    damped_diagonal = None if fisher_diagonal is None else fisher_diagonal + damping
    inverse_diagonal = (
        damped_diagonal.reciprocal()
        if damped_diagonal is not None and bool((damped_diagonal > 0.0).all())
        else None
    )
    result = pcg(
        damped_operator,
        moment,
        inverse_diagonal=inverse_diagonal,
        tolerance=pcg_tolerance,
        max_iterations=pcg_max_iterations,
    )
    if not result.converged:
        raise RuntimeError(
            "natural-direction PCG did not converge: "
            f"relative residual={result.relative_residual:.3e} after "
            f"{result.iterations} iterations"
        )
    return result.solution


def local_regret(
    score_matrix: torch.Tensor,
    predicted_rewards: torch.Tensor,
    target_rewards: torch.Tensor,
    *,
    damping: float,
    beta: float = 1.0,
    center_candidates: bool = True,
    candidate_dim: int = -1,
    pcg_tolerance: float = 1.0e-5,
    pcg_max_iterations: int = 100,
    pcg_dtype: FisherSolveDType = "float64",
    fisher_matrix: torch.Tensor | None = None,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fisher_diagonal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute held-out damped local regret.

    The returned value is

    ``m_error.T @ (F + damping I)^-1 @ m_error / (2 beta)``,

    By default ``F=sum(s s.T)/(P*M)`` is the same node Fisher estimator used
    in training, while ``m_error`` is the per-prompt sample cross-covariance
    with denominator ``P*(M-1)``.  ``fisher_matrix`` or ``fisher_operator``
    can replace the default test Fisher without changing the reward moment.
    """

    damping_value, beta_value = _validate_damping_beta(damping, beta)
    flat_scores, _ = _validate_metric_inputs(score_matrix, predicted_rewards)
    _validate_metric_inputs(score_matrix, target_rewards)
    if predicted_rewards.shape != target_rewards.shape:
        raise ValueError("predicted_rewards and target_rewards must have identical shapes")
    if (
        predicted_rewards.dtype != target_rewards.dtype
        or predicted_rewards.device != target_rewards.device
    ):
        raise ValueError("predicted_rewards and target_rewards must share dtype and device")

    (
        score_matrix,
        promoted_rewards,
        fisher_matrix,
        fisher_diagonal,
    ) = _promote_policy_geometry(
        score_matrix,
        (predicted_rewards, target_rewards),
        pcg_dtype=pcg_dtype,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    predicted_rewards, target_rewards = promoted_rewards
    flat_scores, _ = _validate_metric_inputs(score_matrix, predicted_rewards)
    error = predicted_rewards - target_rewards
    moment = policy_reward_moment(
        score_matrix,
        error,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
    )
    resolved_operator, resolved_diagonal = _resolve_fisher_geometry(
        flat_scores,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    direction = _matrix_free_damped_solve(
        resolved_operator,
        resolved_diagonal,
        moment,
        damping_value,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=pcg_max_iterations,
    )
    value = 0.5 * torch.dot(moment, direction) / beta_value
    # A negative result can only be numerical noise or an invalid solve.
    tolerance = 32.0 * torch.finfo(value.dtype).eps * max(1.0, abs(float(value.item())))
    if float(value.item()) < -tolerance:
        raise FloatingPointError(
            "local regret is negative; the Fisher solve is numerically invalid"
        )
    return value.clamp_min(0.0)


def natural_direction(
    score_matrix: torch.Tensor,
    rewards: torch.Tensor,
    *,
    damping: float,
    center_candidates: bool = True,
    candidate_dim: int = -1,
    pcg_tolerance: float = 1.0e-5,
    pcg_max_iterations: int = 100,
    pcg_dtype: FisherSolveDType = "float64",
    fisher_matrix: torch.Tensor | None = None,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fisher_diagonal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(F + damping I)^-1 A_hat r``."""

    damping_value, _ = _validate_damping_beta(damping)
    score_matrix, promoted_rewards, fisher_matrix, fisher_diagonal = _promote_policy_geometry(
        score_matrix,
        (rewards,),
        pcg_dtype=pcg_dtype,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    (rewards,) = promoted_rewards
    flat_scores, _ = _validate_metric_inputs(score_matrix, rewards)
    moment = policy_reward_moment(
        score_matrix,
        rewards,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
    )
    resolved_operator, resolved_diagonal = _resolve_fisher_geometry(
        flat_scores,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    return _matrix_free_damped_solve(
        resolved_operator,
        resolved_diagonal,
        moment,
        damping_value,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=pcg_max_iterations,
    )


@dataclass(frozen=True)
class NaturalDirectionMetrics:
    """Comparison of predicted and target local policy update directions."""

    predicted_direction: torch.Tensor
    target_direction: torch.Tensor
    squared_fisher_error: torch.Tensor
    fisher_cosine: torch.Tensor
    predicted_fisher_norm: torch.Tensor
    target_fisher_norm: torch.Tensor


@dataclass(frozen=True)
class FixedBetaLocalResult:
    """Target utility of one deployed displacement under a fixed KL penalty.

    All directions are represented in the identifiable Fisher range.  Components
    of the supplied displacement in ``Null(F)`` are discarded because they have
    zero reward moment and zero quadratic KL under the required range condition.
    """

    beta: float
    effective_deployed_direction: torch.Tensor
    target_optimal_direction: torch.Tensor
    deployed_target_utility: torch.Tensor
    optimal_target_utility: torch.Tensor
    regret: torch.Tensor
    deployed_quadratic_kl: torch.Tensor
    optimal_quadratic_kl: torch.Tensor


@dataclass(frozen=True)
class FixedKNormalizedLocalResult:
    """Target regret after normalizing a deployed ray to a Fisher-KL budget."""

    kappa: float
    normalized_deployed_direction: torch.Tensor
    target_optimal_direction: torch.Tensor
    deployed_target_improvement: torch.Tensor
    optimal_target_improvement: torch.Tensor
    regret: torch.Tensor
    deployed_quadratic_kl: torch.Tensor
    optimal_quadratic_kl: torch.Tensor
    fisher_cosine: torch.Tensor
    deployed_direction_has_zero_fisher_norm: bool
    target_moment_is_zero: bool


@dataclass(frozen=True)
class _DenseFisherGeometry:
    fisher_matrix: torch.Tensor
    range_basis: torch.Tensor
    positive_eigenvalues: torch.Tensor
    rcond: float

    def project_to_range(self, vector: torch.Tensor) -> torch.Tensor:
        if self.range_basis.shape[1] == 0:
            return torch.zeros_like(vector)
        return self.range_basis @ (self.range_basis.mT @ vector)

    def pseudoinverse_matvec(self, vector: torch.Tensor) -> torch.Tensor:
        if self.range_basis.shape[1] == 0:
            return torch.zeros_like(vector)
        coefficients = self.range_basis.mT @ vector
        return self.range_basis @ (coefficients / self.positive_eigenvalues)


def _validate_dense_local_geometry(
    fisher_matrix: torch.Tensor,
    target_moment: torch.Tensor,
    deployed_direction: torch.Tensor,
    *,
    rcond: float | None,
) -> _DenseFisherGeometry:
    tensors = {
        "fisher_matrix": fisher_matrix,
        "target_moment": target_moment,
        "deployed_direction": deployed_direction,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must use torch.float32 or torch.float64")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")

    if target_moment.ndim != 1 or target_moment.numel() < 1:
        raise ValueError("target_moment must be a non-empty one-dimensional tensor")
    if deployed_direction.shape != target_moment.shape:
        raise ValueError("deployed_direction must have the same shape as target_moment")
    dimension = target_moment.numel()
    if fisher_matrix.shape != (dimension, dimension):
        raise ValueError(f"fisher_matrix must have shape ({dimension}, {dimension})")
    if (
        fisher_matrix.dtype != target_moment.dtype
        or deployed_direction.dtype != target_moment.dtype
    ):
        raise ValueError("all local-geometry tensors must share dtype")
    if (
        fisher_matrix.device != target_moment.device
        or deployed_direction.device != target_moment.device
    ):
        raise ValueError("all local-geometry tensors must share device")

    epsilon = torch.finfo(fisher_matrix.dtype).eps
    rcond_value = dimension * epsilon if rcond is None else float(rcond)
    if not math.isfinite(rcond_value) or rcond_value <= 0.0 or rcond_value >= 1.0:
        raise ValueError("rcond must be finite and lie strictly between zero and one")

    matrix_scale = max(1.0, float(torch.linalg.matrix_norm(fisher_matrix, ord=2).item()))
    symmetry_tolerance = 32.0 * rcond_value * matrix_scale
    if not torch.allclose(
        fisher_matrix,
        fisher_matrix.mT,
        rtol=32.0 * rcond_value,
        atol=symmetry_tolerance,
    ):
        raise ValueError("fisher_matrix must be symmetric")
    symmetric_fisher = 0.5 * (fisher_matrix + fisher_matrix.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_fisher)
    maximum_eigenvalue = max(0.0, float(eigenvalues[-1].item()))
    psd_tolerance = 32.0 * rcond_value * max(1.0, maximum_eigenvalue)
    if float(eigenvalues[0].item()) < -psd_tolerance:
        raise ValueError("fisher_matrix must be positive semidefinite")

    cutoff = rcond_value * maximum_eigenvalue
    positive = eigenvalues > cutoff
    range_basis = eigenvectors[:, positive]
    positive_eigenvalues = eigenvalues[positive]
    geometry = _DenseFisherGeometry(
        fisher_matrix=symmetric_fisher,
        range_basis=range_basis,
        positive_eigenvalues=positive_eigenvalues,
        rcond=rcond_value,
    )

    projected_target = geometry.project_to_range(target_moment)
    range_residual = torch.linalg.vector_norm(target_moment - projected_target)
    target_scale = max(1.0, float(torch.linalg.vector_norm(target_moment).item()))
    range_tolerance = 32.0 * rcond_value * target_scale
    if float(range_residual.item()) > range_tolerance:
        raise ValueError(
            "target_moment must lie in Range(fisher_matrix); otherwise the "
            "local policy objective is unbounded along a Fisher-null direction"
        )
    return geometry


def _validate_positive_scalar(name: str, value: float, *, allow_zero: bool) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (not allow_zero and result == 0.0):
        qualifier = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _checked_nonnegative(
    value: torch.Tensor,
    *,
    name: str,
    scale: float,
) -> torch.Tensor:
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise FloatingPointError(f"{name} must be a finite scalar")
    tolerance = 64.0 * torch.finfo(value.dtype).eps * max(1.0, scale)
    if float(value.item()) < -tolerance:
        raise FloatingPointError(f"{name} is negative beyond numerical tolerance")
    return value.clamp_min(0.0)


def fixed_beta_deployed_direction_regret(
    fisher_matrix: torch.Tensor,
    target_moment: torch.Tensor,
    deployed_direction: torch.Tensor,
    *,
    beta: float,
    rcond: float | None = None,
) -> FixedBetaLocalResult:
    """Evaluate a deployed displacement for the fixed-``beta`` local objective.

    This function evaluates the actual displacement supplied by the caller:

    ``G_*(delta) = g_*.T delta - beta * delta.T F delta / 2``.

    It is deliberately distinct from fixed-K normalization.  The minimum-norm
    target optimum is ``F^dagger g_* / beta``.  A singular Fisher is supported
    when ``g_*`` lies in its range; otherwise the quadratic objective is
    unbounded and the function rejects the input.
    """

    beta_value = _validate_positive_scalar("beta", beta, allow_zero=False)
    geometry = _validate_dense_local_geometry(
        fisher_matrix,
        target_moment,
        deployed_direction,
        rcond=rcond,
    )
    effective_deployed = geometry.project_to_range(deployed_direction)
    target_natural_direction = geometry.pseudoinverse_matvec(target_moment)
    target_optimal_direction = target_natural_direction / beta_value

    fisher_deployed = geometry.fisher_matrix @ effective_deployed
    fisher_optimal = geometry.fisher_matrix @ target_optimal_direction
    deployed_quadratic = _checked_nonnegative(
        0.5 * torch.dot(effective_deployed, fisher_deployed),
        name="deployed quadratic KL",
        scale=float(torch.linalg.vector_norm(effective_deployed).item()) ** 2,
    )
    optimal_quadratic = _checked_nonnegative(
        0.5 * torch.dot(target_optimal_direction, fisher_optimal),
        name="optimal quadratic KL",
        scale=float(torch.linalg.vector_norm(target_optimal_direction).item()) ** 2,
    )
    deployed_utility = (
        torch.dot(target_moment, effective_deployed) - beta_value * deployed_quadratic
    )
    optimal_utility = 0.5 * torch.dot(target_moment, target_natural_direction) / beta_value
    if not bool(torch.isfinite(deployed_utility)) or not bool(torch.isfinite(optimal_utility)):
        raise FloatingPointError("fixed-beta local utilities must be finite")
    regret = _checked_nonnegative(
        optimal_utility - deployed_utility,
        name="fixed-beta local regret",
        scale=max(abs(float(optimal_utility.item())), abs(float(deployed_utility.item()))),
    )
    return FixedBetaLocalResult(
        beta=beta_value,
        effective_deployed_direction=effective_deployed,
        target_optimal_direction=target_optimal_direction,
        deployed_target_utility=deployed_utility,
        optimal_target_utility=optimal_utility,
        regret=regret,
        deployed_quadratic_kl=deployed_quadratic,
        optimal_quadratic_kl=optimal_quadratic,
    )


def fixed_k_normalized_local_regret(
    fisher_matrix: torch.Tensor,
    target_moment: torch.Tensor,
    deployed_direction: torch.Tensor,
    *,
    kappa: float,
    rcond: float | None = None,
) -> FixedKNormalizedLocalResult:
    """Evaluate a deployed ray after Fisher normalization to budget ``kappa``.

    For a nonzero deployed ray ``d``, the effective displacement is

    ``sqrt(2 kappa / (d.T F d)) * d``.

    The target comparator independently solves the same constrained local
    problem.  Consequently this regret is invariant to positive rescaling of
    the deployed ray and is generally *not* the fixed-beta quadratic regret.
    A zero-Fisher-norm deployed ray deterministically means no effective update;
    a zero target moment has zero optimal improvement and zero regret.
    """

    kappa_value = _validate_positive_scalar("kappa", kappa, allow_zero=True)
    geometry = _validate_dense_local_geometry(
        fisher_matrix,
        target_moment,
        deployed_direction,
        rcond=rcond,
    )
    effective_ray = geometry.project_to_range(deployed_direction)
    target_natural_direction = geometry.pseudoinverse_matvec(target_moment)
    fisher_ray = geometry.fisher_matrix @ effective_ray
    fisher_target = geometry.fisher_matrix @ target_natural_direction
    ray_squared_norm = _checked_nonnegative(
        torch.dot(effective_ray, fisher_ray),
        name="deployed Fisher squared norm",
        scale=float(torch.linalg.vector_norm(effective_ray).item()) ** 2,
    )
    target_squared_norm = _checked_nonnegative(
        torch.dot(target_natural_direction, fisher_target),
        name="target Fisher squared norm",
        scale=float(torch.linalg.vector_norm(target_natural_direction).item()) ** 2,
    )

    raw_ray_norm = float(torch.linalg.vector_norm(deployed_direction).item())
    effective_ray_norm = float(torch.linalg.vector_norm(effective_ray).item())
    deployed_is_zero = raw_ray_norm == 0.0 or (
        effective_ray_norm <= 32.0 * geometry.rcond * raw_ray_norm
    )
    # Scale invariance requires every nonzero target moment in the identifiable
    # range to remain nonzero, however small its cardinal reward scale is.
    target_is_zero = not bool(torch.count_nonzero(target_moment))

    if kappa_value == 0.0 or deployed_is_zero:
        normalized_deployed = torch.zeros_like(effective_ray)
    else:
        deployed_scale = math.sqrt(2.0 * kappa_value / float(ray_squared_norm.item()))
        normalized_deployed = effective_ray * deployed_scale

    if kappa_value == 0.0 or target_is_zero:
        target_optimal_direction = torch.zeros_like(target_natural_direction)
    else:
        target_scale = math.sqrt(2.0 * kappa_value / float(target_squared_norm.item()))
        target_optimal_direction = target_natural_direction * target_scale

    deployed_quadratic = _checked_nonnegative(
        0.5
        * torch.dot(
            normalized_deployed,
            geometry.fisher_matrix @ normalized_deployed,
        ),
        name="deployed quadratic KL",
        scale=max(1.0, 2.0 * kappa_value),
    )
    optimal_quadratic = _checked_nonnegative(
        0.5
        * torch.dot(
            target_optimal_direction,
            geometry.fisher_matrix @ target_optimal_direction,
        ),
        name="optimal quadratic KL",
        scale=max(1.0, 2.0 * kappa_value),
    )
    deployed_improvement = torch.dot(target_moment, normalized_deployed)
    optimal_improvement = torch.dot(target_moment, target_optimal_direction)
    if not bool(torch.isfinite(deployed_improvement)) or not bool(
        torch.isfinite(optimal_improvement)
    ):
        raise FloatingPointError("fixed-K local improvements must be finite")
    regret = _checked_nonnegative(
        optimal_improvement - deployed_improvement,
        name="fixed-K normalized local regret",
        scale=max(
            abs(float(optimal_improvement.item())),
            abs(float(deployed_improvement.item())),
        ),
    )

    if deployed_is_zero or target_is_zero:
        cosine = torch.full(
            (),
            float("nan"),
            dtype=target_moment.dtype,
            device=target_moment.device,
        )
    else:
        cosine = (
            torch.dot(effective_ray, fisher_target)
            / torch.sqrt(ray_squared_norm * target_squared_norm)
        ).clamp(min=-1.0, max=1.0)
    return FixedKNormalizedLocalResult(
        kappa=kappa_value,
        normalized_deployed_direction=normalized_deployed,
        target_optimal_direction=target_optimal_direction,
        deployed_target_improvement=deployed_improvement,
        optimal_target_improvement=optimal_improvement,
        regret=regret,
        deployed_quadratic_kl=deployed_quadratic,
        optimal_quadratic_kl=optimal_quadratic,
        fisher_cosine=cosine,
        deployed_direction_has_zero_fisher_norm=deployed_is_zero,
        target_moment_is_zero=target_is_zero,
    )


def natural_direction_metrics(
    score_matrix: torch.Tensor,
    predicted_rewards: torch.Tensor,
    target_rewards: torch.Tensor,
    *,
    damping: float,
    center_candidates: bool = True,
    candidate_dim: int = -1,
    pcg_tolerance: float = 1.0e-5,
    pcg_max_iterations: int = 100,
    pcg_dtype: FisherSolveDType = "float64",
    fisher_matrix: torch.Tensor | None = None,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fisher_diagonal: torch.Tensor | None = None,
) -> NaturalDirectionMetrics:
    """Report Fisher error and Fisher cosine between natural directions.

    Directions use the damped inverse, while norms and the error metric use
    the undamped empirical Fisher, matching the local policy geometry.
    ``fisher_cosine`` is ``NaN`` when either direction has zero Fisher norm.
    """

    _validate_metric_inputs(score_matrix, predicted_rewards)
    _validate_metric_inputs(score_matrix, target_rewards)
    if predicted_rewards.shape != target_rewards.shape:
        raise ValueError("predicted_rewards and target_rewards must have identical shapes")
    if (
        predicted_rewards.dtype != target_rewards.dtype
        or predicted_rewards.device != target_rewards.device
    ):
        raise ValueError("predicted_rewards and target_rewards must share dtype and device")

    (
        score_matrix,
        promoted_rewards,
        fisher_matrix,
        fisher_diagonal,
    ) = _promote_policy_geometry(
        score_matrix,
        (predicted_rewards, target_rewards),
        pcg_dtype=pcg_dtype,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    predicted_rewards, target_rewards = promoted_rewards
    predicted_direction = natural_direction(
        score_matrix,
        predicted_rewards,
        damping=damping,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=pcg_max_iterations,
        pcg_dtype=pcg_dtype,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    target_direction = natural_direction(
        score_matrix,
        target_rewards,
        damping=damping,
        center_candidates=center_candidates,
        candidate_dim=candidate_dim,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=pcg_max_iterations,
        pcg_dtype=pcg_dtype,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    flat_scores, _ = _validate_metric_inputs(score_matrix, predicted_rewards)
    resolved_operator, _ = _resolve_fisher_geometry(
        flat_scores,
        fisher_matrix=fisher_matrix,
        fisher_operator=fisher_operator,
        fisher_diagonal=fisher_diagonal,
    )
    difference = predicted_direction - target_direction
    fisher_difference = resolved_operator(difference)
    squared_error = torch.dot(difference, fisher_difference).clamp_min(0.0)
    fisher_predicted = resolved_operator(predicted_direction)
    fisher_target = resolved_operator(target_direction)
    predicted_squared_norm = torch.dot(predicted_direction, fisher_predicted).clamp_min(0.0)
    target_squared_norm = torch.dot(target_direction, fisher_target).clamp_min(0.0)
    predicted_norm = torch.sqrt(predicted_squared_norm)
    target_norm = torch.sqrt(target_squared_norm)
    denominator = predicted_norm * target_norm
    if float(denominator.item()) == 0.0:
        cosine = torch.full((), float("nan"), dtype=flat_scores.dtype, device=flat_scores.device)
    else:
        cosine = (torch.dot(predicted_direction, fisher_target) / denominator).clamp(
            min=-1.0, max=1.0
        )
    return NaturalDirectionMetrics(
        predicted_direction=predicted_direction,
        target_direction=target_direction,
        squared_fisher_error=squared_error,
        fisher_cosine=cosine,
        predicted_fisher_norm=predicted_norm,
        target_fisher_norm=target_norm,
    )


__all__ = [
    "FixedBetaLocalResult",
    "FixedKNormalizedLocalResult",
    "NaturalDirectionMetrics",
    "empirical_fisher_matrix",
    "fixed_beta_deployed_direction_regret",
    "fixed_k_normalized_local_regret",
    "gauge_center",
    "local_regret",
    "natural_direction",
    "natural_direction_metrics",
    "policy_reward_moment",
]
