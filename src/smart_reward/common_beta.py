"""Pure primitives for a pilot-calibrated, globally frozen-beta experiment.

The primary ProRM estimand fixes one downstream KL penalty for every learner.
This module deliberately separates pilot scale selection, confirmatory binding,
and learner-specific trust-region normalization:

* pilot train-oracle directions produce beta candidates for later design selection;
* confirmatory runs bind the single beta already frozen in the design identity;
* every saved natural direction is divided by exactly that same scalar;
* post-deployment measured KL is a safety outcome and never rescales a learner;
* downstream utility uses on-policy ``KL(pi_updated || pi_0)`` per sequence.

No function in this module loads a model, reads a split, or mutates parameters.
Callers remain responsible for enforcing the train/test information boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch

from .policy_update import fisher_quadratic


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _finite_direction(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional torch.Tensor")
    if value.numel() < 1 or not value.is_floating_point():
        raise ValueError(f"{name} must be non-empty and floating point")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class CommonBetaCalibration:
    """Train-only calibration record for the common downstream KL penalty."""

    beta_common: float
    target_oracle_quadratic_kl: float
    oracle_natural_curvature: float
    oracle_displacement: torch.Tensor
    predicted_oracle_quadratic_kl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "common-beta-calibration/v1",
            "beta_common": self.beta_common,
            "target_oracle_quadratic_kl": self.target_oracle_quadratic_kl,
            "oracle_natural_curvature": self.oracle_natural_curvature,
            "oracle_displacement": self.oracle_displacement.detach().cpu().tolist(),
            "predicted_oracle_quadratic_kl": self.predicted_oracle_quadratic_kl,
            "calibration_split": "train_only",
            "learner_specific_rescaling": False,
        }


@dataclass(frozen=True, slots=True)
class CommonBetaDirection:
    """One natural direction deployed under the already frozen common beta."""

    name: str
    beta_common: float
    natural_direction: torch.Tensor
    displacement: torch.Tensor
    natural_fisher_curvature: float
    predicted_quadratic_kl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "common-beta-direction/v1",
            "name": self.name,
            "beta_common": self.beta_common,
            "natural_direction": self.natural_direction.detach().cpu().tolist(),
            "displacement": self.displacement.detach().cpu().tolist(),
            "natural_fisher_curvature": self.natural_fisher_curvature,
            "predicted_quadratic_kl": self.predicted_quadratic_kl,
            "learner_specific_rescaling": False,
        }


@torch.no_grad()
def calibrate_common_beta(
    oracle_natural_direction: torch.Tensor,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor],
    *,
    target_oracle_quadratic_kl: float,
) -> CommonBetaCalibration:
    """Calibrate one beta so the train-oracle local step spends target KL.

    If ``u_* = F_train^{-1} g_*`` is the undivided train-oracle natural
    direction, this returns

    ``beta_common = sqrt(u_*^T F_train u_* / (2 K_cal))``.

    The calibration target is a scale-setting device, not a fixed-K
    normalization of each learner.  Only the oracle direction enters it.
    """

    direction = _finite_direction("oracle_natural_direction", oracle_natural_direction)
    target = _positive_float(
        "target_oracle_quadratic_kl",
        target_oracle_quadratic_kl,
    )
    curvature = float(fisher_quadratic(direction, fisher_operator).item())
    if curvature <= 0.0:
        raise ValueError(
            "oracle_natural_direction must have strictly positive train Fisher curvature"
        )
    beta = math.sqrt(curvature / (2.0 * target))
    displacement = (direction / beta).detach().clone()
    predicted = 0.5 * curvature / (beta * beta)
    if not math.isclose(predicted, target, rel_tol=2.0e-12, abs_tol=1.0e-15):
        raise FloatingPointError("common-beta calibration failed its quadratic-KL identity")
    return CommonBetaCalibration(
        beta_common=beta,
        target_oracle_quadratic_kl=target,
        oracle_natural_curvature=curvature,
        oracle_displacement=displacement,
        predicted_oracle_quadratic_kl=predicted,
    )


@torch.no_grad()
def bind_frozen_common_beta(
    oracle_natural_direction: torch.Tensor,
    fisher_operator: Callable[[torch.Tensor], torch.Tensor],
    *,
    frozen_global_beta: float,
    reference_target_oracle_quadratic_kl: float,
) -> CommonBetaCalibration:
    """Bind current-seed diagnostics to a beta frozen before this seed ran.

    Unlike :func:`calibrate_common_beta`, current-seed Fisher curvature has no
    authority over the returned beta.  It is used only to report the local KL
    predicted for the already frozen scalar.
    """

    direction = _finite_direction("oracle_natural_direction", oracle_natural_direction)
    beta = _positive_float("frozen_global_beta", frozen_global_beta)
    target = _positive_float(
        "reference_target_oracle_quadratic_kl",
        reference_target_oracle_quadratic_kl,
    )
    curvature = float(fisher_quadratic(direction, fisher_operator).item())
    if curvature <= 0.0:
        raise ValueError(
            "oracle_natural_direction must have strictly positive train Fisher curvature"
        )
    displacement = (direction / beta).detach().clone()
    predicted = 0.5 * curvature / (beta * beta)
    return CommonBetaCalibration(
        beta_common=beta,
        target_oracle_quadratic_kl=target,
        oracle_natural_curvature=curvature,
        oracle_displacement=displacement,
        predicted_oracle_quadratic_kl=predicted,
    )


@torch.no_grad()
def deploy_with_common_beta(
    natural_directions: Mapping[str, torch.Tensor],
    fisher_operator: Callable[[torch.Tensor], torch.Tensor],
    *,
    calibration: CommonBetaCalibration,
) -> dict[str, CommonBetaDirection]:
    """Divide every natural direction by the same frozen ``beta_common``."""

    if not isinstance(calibration, CommonBetaCalibration):
        raise TypeError("calibration must be CommonBetaCalibration")
    if not isinstance(natural_directions, Mapping) or not natural_directions:
        raise ValueError("natural_directions must be a non-empty mapping")
    beta = _positive_float("calibration.beta_common", calibration.beta_common)
    deployed: dict[str, CommonBetaDirection] = {}
    for name, raw_direction in natural_directions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("natural direction names must be non-empty strings")
        direction = _finite_direction(f"natural_directions[{name!r}]", raw_direction)
        curvature = float(fisher_quadratic(direction, fisher_operator).item())
        displacement = (direction / beta).detach().clone()
        deployed[name] = CommonBetaDirection(
            name=name,
            beta_common=beta,
            natural_direction=direction.detach().clone(),
            displacement=displacement,
            natural_fisher_curvature=curvature,
            predicted_quadratic_kl=0.5 * curvature / (beta * beta),
        )
    return deployed


@dataclass(frozen=True, slots=True)
class MeasuredKLSafety:
    """Post-deployment KL safety decision with no retuning authority."""

    cap: float
    passed: bool
    measured_by_policy: tuple[tuple[str, float], ...]
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "measured-kl-safety/v1",
            "cap": self.cap,
            "passed": self.passed,
            "measured_by_policy": dict(self.measured_by_policy),
            "violations": list(self.violations),
            "beta_retuned": False,
        }


def assess_measured_kl_safety(
    measured_by_policy: Mapping[str, float],
    *,
    cap: float,
) -> MeasuredKLSafety:
    """Fail policies above a prespecified cap without changing common beta."""

    cap_value = _positive_float("cap", cap)
    if not isinstance(measured_by_policy, Mapping) or not measured_by_policy:
        raise ValueError("measured_by_policy must be a non-empty mapping")
    checked: list[tuple[str, float]] = []
    for name, raw_value in measured_by_policy.items():
        if not isinstance(name, str) or not name:
            raise ValueError("policy names must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise TypeError(f"measured KL for {name!r} must be a real scalar")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"measured KL for {name!r} must be finite and non-negative")
        checked.append((name, value))
    checked.sort(key=lambda item: item[0])
    violations = tuple(name for name, value in checked if value > cap_value)
    return MeasuredKLSafety(
        cap=cap_value,
        passed=not violations,
        measured_by_policy=tuple(checked),
        violations=violations,
    )


@dataclass(frozen=True, slots=True)
class PairedPromptEstimate:
    """Mean and prompt-clustered sample standard error of paired values."""

    mean: float
    sample_standard_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "sample_standard_error": self.sample_standard_error,
        }


@dataclass(frozen=True, slots=True)
class DownstreamUtilitySummary:
    """Prompt-level summary of ``reward - beta * KL(updated || reference)``."""

    beta_common: float
    num_prompts: int
    candidates_per_prompt: int
    mean_target_reward: float
    mean_on_policy_kl: float
    mean_target_utility: float
    target_utility_sample_standard_error: float
    improvement_over_zero_b: PairedPromptEstimate
    oracle_step_reference_gap: PairedPromptEstimate | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "downstream-policy-utility/v1",
            "beta_common": self.beta_common,
            "num_prompts": self.num_prompts,
            "candidates_per_prompt": self.candidates_per_prompt,
            "mean_target_reward": self.mean_target_reward,
            "mean_on_policy_kl_pi_updated_to_pi0": self.mean_on_policy_kl,
            "mean_target_utility": self.mean_target_utility,
            "target_utility_sample_standard_error": (self.target_utility_sample_standard_error),
            "improvement_over_zero_b": self.improvement_over_zero_b.to_dict(),
            "oracle_step_reference_gap": (
                None
                if self.oracle_step_reference_gap is None
                else self.oracle_step_reference_gap.to_dict()
            ),
            "oracle_step_is_global_optimum": False,
        }


def _paired_prompt_estimate(values: torch.Tensor) -> PairedPromptEstimate:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("paired prompt values must contain at least two prompts")
    values64 = values.detach().to(device="cpu", dtype=torch.float64)
    return PairedPromptEstimate(
        mean=float(values64.mean().item()),
        sample_standard_error=float(
            (values64.std(unbiased=True) / math.sqrt(values64.numel())).item()
        ),
    )


def _validate_rollout_matrix(
    name: str,
    value: object,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TypeError(f"{name} must be a two-dimensional torch.Tensor")
    if min(value.shape) < 1 or not value.is_floating_point():
        raise ValueError(f"{name} must be non-empty and floating point")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    if expected_shape is not None and tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    return value


def summarize_downstream_utility(
    transformed_target_rewards: torch.Tensor,
    on_policy_updated_to_reference_kl: torch.Tensor,
    zero_b_transformed_target_rewards: torch.Tensor,
    *,
    beta_common: float,
    oracle_step_transformed_target_rewards: torch.Tensor | None = None,
    oracle_step_on_policy_updated_to_reference_kl: torch.Tensor | None = None,
) -> DownstreamUtilitySummary:
    """Aggregate finite-rollout target utility at the prompt level.

    Matrices have shape ``(num_prompts, candidates_per_prompt)``.  Candidate
    values are averaged within a prompt before standard errors are computed,
    so candidates sharing one prompt are not treated as independent units.

    ``oracle_step_reference_gap`` means oracle-step utility minus this method's
    utility.  It is a positive-control reference gap, not regret to the global
    optimum of the nonlinear policy problem.
    """

    beta = _positive_float("beta_common", beta_common)
    rewards = _validate_rollout_matrix(
        "transformed_target_rewards",
        transformed_target_rewards,
    )
    shape = (int(rewards.shape[0]), int(rewards.shape[1]))
    if shape[0] < 2:
        raise ValueError("at least two prompts are required for a prompt-level sample SE")
    kl = _validate_rollout_matrix(
        "on_policy_updated_to_reference_kl",
        on_policy_updated_to_reference_kl,
        expected_shape=shape,
    )
    reference = _validate_rollout_matrix(
        "zero_b_transformed_target_rewards",
        zero_b_transformed_target_rewards,
        expected_shape=shape,
    )
    for name, tensor in (
        ("on_policy_updated_to_reference_kl", kl),
        ("zero_b_transformed_target_rewards", reference),
    ):
        if tensor.dtype != rewards.dtype or tensor.device != rewards.device:
            raise ValueError(f"{name} must share dtype and device with target rewards")
    if bool((kl < 0.0).any()):
        raise ValueError("on-policy updated-to-reference KL must be non-negative")

    utility = rewards - beta * kl
    prompt_utility = utility.mean(dim=1)
    prompt_reference_utility = reference.mean(dim=1)
    improvement = _paired_prompt_estimate(prompt_utility - prompt_reference_utility)

    oracle_gap: PairedPromptEstimate | None = None
    if (oracle_step_transformed_target_rewards is None) != (
        oracle_step_on_policy_updated_to_reference_kl is None
    ):
        raise ValueError("oracle-step rewards and KL must be supplied together")
    if oracle_step_transformed_target_rewards is not None:
        oracle_rewards = _validate_rollout_matrix(
            "oracle_step_transformed_target_rewards",
            oracle_step_transformed_target_rewards,
            expected_shape=shape,
        )
        oracle_kl = _validate_rollout_matrix(
            "oracle_step_on_policy_updated_to_reference_kl",
            oracle_step_on_policy_updated_to_reference_kl,
            expected_shape=shape,
        )
        for name, tensor in (
            ("oracle_step_transformed_target_rewards", oracle_rewards),
            ("oracle_step_on_policy_updated_to_reference_kl", oracle_kl),
        ):
            if tensor.dtype != rewards.dtype or tensor.device != rewards.device:
                raise ValueError(f"{name} must share dtype and device with target rewards")
        if bool((oracle_kl < 0.0).any()):
            raise ValueError("oracle-step updated-to-reference KL must be non-negative")
        prompt_oracle_utility = (oracle_rewards - beta * oracle_kl).mean(dim=1)
        oracle_gap = _paired_prompt_estimate(prompt_oracle_utility - prompt_utility)

    prompt_utility64 = prompt_utility.detach().to(device="cpu", dtype=torch.float64)
    return DownstreamUtilitySummary(
        beta_common=beta,
        num_prompts=shape[0],
        candidates_per_prompt=shape[1],
        mean_target_reward=float(rewards.detach().to(torch.float64).mean().item()),
        mean_on_policy_kl=float(kl.detach().to(torch.float64).mean().item()),
        mean_target_utility=float(prompt_utility64.mean().item()),
        target_utility_sample_standard_error=float(
            (prompt_utility64.std(unbiased=True) / math.sqrt(prompt_utility64.numel())).item()
        ),
        improvement_over_zero_b=improvement,
        oracle_step_reference_gap=oracle_gap,
    )


__all__ = [
    "CommonBetaCalibration",
    "CommonBetaDirection",
    "DownstreamUtilitySummary",
    "MeasuredKLSafety",
    "PairedPromptEstimate",
    "assess_measured_kl_safety",
    "bind_frozen_common_beta",
    "calibrate_common_beta",
    "deploy_with_common_beta",
    "summarize_downstream_utility",
]
