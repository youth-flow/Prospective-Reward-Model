"""Phase-1 reward-head directions and fail-closed measured-KL policy updates.

This module is deliberately model agnostic.  It consumes the immutable tensors
produced by Phase 1 and the strict fixed-A LoRA/token contracts from
``smart_reward.hf``; it never downloads a model or regenerates a candidate.

The policy update has two distinct geometries:

* the train-node Fisher supplies a natural-gradient direction and, optionally,
  a quadratic *initial guess* for the step size; and
* response-token vocabulary forward KL, evaluated on the saved reference
  candidates, is the only quantity allowed to accept the final step.

Every line-search trial overwrites LoRA-B from the same zero-B origin.  A
failed search or any exception restores that origin.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import torch

from . import hf as _hf
from .experiment import TrainingTensorData
from .hf import ExactTokenCandidates, FixedALoRASetup
from .linear import (
    DampedEmpiricalFisher,
    FisherSolveDType,
    resolve_fisher_solve_dtype,
)
from .metrics import policy_reward_moment
from .pcg import pcg
from .policy_update import (
    fisher_quadratic,
    line_search_measured_kl,
    select_causal_response_logits,
    selected_causal_forward_kl,
    set_tangent_update_,
    step_size_for_kl_budget,
)


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _coerce_head_weight(
    head_weight: torch.Tensor | Sequence[float],
    train: TrainingTensorData,
) -> torch.Tensor:
    if isinstance(head_weight, torch.Tensor):
        if not head_weight.is_floating_point():
            raise TypeError("head_weight must be floating point")
        value = head_weight.detach()
    elif isinstance(head_weight, Sequence) and not isinstance(head_weight, (str, bytes, bytearray)):
        try:
            value = torch.as_tensor(
                tuple(head_weight),
                dtype=train.reward_features.dtype,
                device=train.reward_features.device,
            )
        except (TypeError, ValueError) as error:
            raise TypeError("head_weight must contain real scalars") from error
    else:
        raise TypeError("head_weight must be a tensor or a sequence of real scalars")
    if value.shape == (1, train.reward_dimension):
        value = value.reshape(train.reward_dimension)
    if value.shape != (train.reward_dimension,):
        raise ValueError(f"head_weight must have shape ({train.reward_dimension},)")
    value = value.to(
        dtype=train.reward_features.dtype,
        device=train.reward_features.device,
    )
    if not bool(torch.isfinite(value).all()):
        raise ValueError("head_weight must be finite")
    return value


def _validate_node_rewards(
    node_rewards: torch.Tensor,
    train: TrainingTensorData,
) -> torch.Tensor:
    if not isinstance(node_rewards, torch.Tensor):
        raise TypeError("node_rewards must be a torch.Tensor")
    expected_shape = (train.num_prompts, train.num_candidates)
    if node_rewards.shape != expected_shape:
        raise ValueError(f"node_rewards must have shape {expected_shape}")
    if not node_rewards.is_floating_point():
        raise TypeError("node_rewards must have a floating-point dtype")
    if (
        node_rewards.dtype != train.policy_scores.dtype
        or node_rewards.device != train.policy_scores.device
    ):
        raise ValueError("node_rewards must share dtype and device with policy_scores")
    if node_rewards.requires_grad or node_rewards.grad_fn is not None:
        raise ValueError("node_rewards must be frozen and detached")
    if not bool(torch.isfinite(node_rewards).all()):
        raise ValueError("node_rewards must be finite")
    return node_rewards


@dataclass(frozen=True, slots=True)
class PolicyDirectionResult:
    """Natural direction plus auditable damping, solve, and curvature evidence."""

    direction: torch.Tensor
    beta: float
    relative_damping: float
    absolute_damping: float
    mean_fisher_diagonal: float
    moment_norm: float
    direction_norm: float
    fisher_curvature: float
    damped_curvature: float
    moment_alignment: float
    pcg_iterations: int
    pcg_residual_norm: float
    pcg_relative_residual: float
    pcg_converged: bool
    pcg_reason: Literal["converged", "zero_rhs", "max_iterations"]

    def __post_init__(self) -> None:
        if not isinstance(self.direction, torch.Tensor) or self.direction.ndim != 1:
            raise TypeError("direction must be a one-dimensional tensor")
        if not self.direction.is_floating_point() or not bool(torch.isfinite(self.direction).all()):
            raise ValueError("direction must be finite and floating point")
        for name in (
            "beta",
            "relative_damping",
            "absolute_damping",
            "mean_fisher_diagonal",
        ):
            _finite_positive(name, getattr(self, name))
        for name in (
            "moment_norm",
            "direction_norm",
            "fisher_curvature",
            "damped_curvature",
            "pcg_residual_norm",
            "pcg_relative_residual",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if not math.isfinite(float(self.moment_alignment)):
            raise ValueError("moment_alignment must be finite")
        if (
            isinstance(self.pcg_iterations, bool)
            or not isinstance(self.pcg_iterations, int)
            or self.pcg_iterations < 0
        ):
            raise ValueError("pcg_iterations must be a non-negative integer")
        if not isinstance(self.pcg_converged, bool):
            raise TypeError("pcg_converged must be bool")
        if self.pcg_reason not in {"converged", "zero_rhs", "max_iterations"}:
            raise ValueError("invalid pcg_reason")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible record, including the direction."""

        return {
            "schema_version": "policy-direction/v1",
            "direction": self.direction.detach().cpu().tolist(),
            "beta": self.beta,
            "relative_damping": self.relative_damping,
            "absolute_damping": self.absolute_damping,
            "mean_fisher_diagonal": self.mean_fisher_diagonal,
            "moment_norm": self.moment_norm,
            "direction_norm": self.direction_norm,
            "fisher_curvature": self.fisher_curvature,
            "damped_curvature": self.damped_curvature,
            "moment_alignment": self.moment_alignment,
            "pcg": {
                "iterations": self.pcg_iterations,
                "residual_norm": self.pcg_residual_norm,
                "relative_residual": self.pcg_relative_residual,
                "converged": self.pcg_converged,
                "reason": self.pcg_reason,
            },
        }


@torch.no_grad()
def _solve_policy_direction_from_node_rewards(
    train: TrainingTensorData,
    node_rewards: torch.Tensor,
    *,
    relative_damping: float,
    beta: float = 1.0,
    pcg_dtype: FisherSolveDType = "float64",
    pcg_max_iterations: int = 200,
    pcg_tolerance: float = 1.0e-6,
    pcg_absolute_tolerance: float = 0.0,
    pcg_residual_recompute_interval: int = 20,
    require_pcg_convergence: bool = True,
) -> PolicyDirectionResult:
    """Shared validated natural-direction solve for frozen node rewards."""

    if not isinstance(train, TrainingTensorData):
        raise TypeError("train must be TrainingTensorData")
    relative = _finite_positive("relative_damping", relative_damping)
    beta_value = _finite_positive("beta", beta)
    solve_dtype = resolve_fisher_solve_dtype(pcg_dtype)
    _positive_integer("pcg_max_iterations", pcg_max_iterations)
    _positive_integer("pcg_residual_recompute_interval", pcg_residual_recompute_interval)
    tolerance = _finite_nonnegative("pcg_tolerance", pcg_tolerance)
    absolute_tolerance = _finite_nonnegative("pcg_absolute_tolerance", pcg_absolute_tolerance)
    if not isinstance(require_pcg_convergence, bool):
        raise TypeError("require_pcg_convergence must be bool")

    rewards = _validate_node_rewards(node_rewards, train)
    policy_scores = train.policy_scores.to(dtype=solve_dtype)
    geometry_rewards = rewards.to(dtype=solve_dtype)
    moment = policy_reward_moment(
        policy_scores,
        geometry_rewards,
        center_candidates=True,
        candidate_dim=1,
    )
    flat_scores = policy_scores.reshape(-1, train.policy_dimension)
    undamped_fisher = DampedEmpiricalFisher(flat_scores, damping=0.0)
    mean_diagonal = float(undamped_fisher.diagonal().mean().item())
    if not math.isfinite(mean_diagonal) or mean_diagonal <= 0.0:
        raise ValueError(
            "mean(diag(F)) must be finite and positive; the policy tangent is degenerate"
        )
    damping = relative * mean_diagonal
    damped_fisher = DampedEmpiricalFisher(flat_scores, damping=damping)
    right_hand_side = moment / beta_value
    solve = pcg(
        damped_fisher.matvec,
        right_hand_side,
        # Preserve the low-rank-plus-isotropic-damping spectrum instead of
        # breaking it with a coordinate-wise Jacobi rescaling.
        inverse_diagonal=None,
        max_iterations=pcg_max_iterations,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        residual_recompute_interval=pcg_residual_recompute_interval,
    )
    if require_pcg_convergence and not solve.converged:
        raise RuntimeError(
            "policy-direction PCG did not converge: "
            f"relative residual={solve.relative_residual:.3e} after "
            f"{solve.iterations} iterations"
        )

    direction = solve.solution.detach().clone()
    fisher_curvature = float(
        torch.dot(direction, undamped_fisher.matvec(direction)).clamp_min(0.0).item()
    )
    damped_curvature = float(
        torch.dot(direction, damped_fisher.matvec(direction)).clamp_min(0.0).item()
    )
    return PolicyDirectionResult(
        direction=direction,
        beta=beta_value,
        relative_damping=relative,
        absolute_damping=damping,
        mean_fisher_diagonal=mean_diagonal,
        moment_norm=float(torch.linalg.vector_norm(moment).item()),
        direction_norm=float(torch.linalg.vector_norm(direction).item()),
        fisher_curvature=fisher_curvature,
        damped_curvature=damped_curvature,
        moment_alignment=float(torch.dot(moment, direction).item()),
        pcg_iterations=solve.iterations,
        pcg_residual_norm=solve.residual_norm,
        pcg_relative_residual=solve.relative_residual,
        pcg_converged=solve.converged,
        pcg_reason=solve.reason,
    )


@torch.no_grad()
def policy_direction_from_head(
    train: TrainingTensorData,
    head_weight: torch.Tensor | Sequence[float],
    *,
    relative_damping: float,
    beta: float = 1.0,
    pcg_dtype: FisherSolveDType = "float64",
    pcg_max_iterations: int = 200,
    pcg_tolerance: float = 1.0e-6,
    pcg_absolute_tolerance: float = 0.0,
    pcg_residual_recompute_interval: int = 20,
    require_pcg_convergence: bool = True,
) -> PolicyDirectionResult:
    """Build the train-only learned-head natural policy direction.

    If ``r_ij = feature_ij @ head_weight``, this computes

    ``direction = (F + lambda I)^-1 A_hat r / beta``

    from all train candidates.  ``F`` is the raw node mean ``S.T S/(P*M)``;
    ``A_hat r`` is the average per-prompt sample covariance with denominator
    ``P*(M-1)``; and ``lambda = relative_damping * mean(diag(F))``.
    """

    if not isinstance(train, TrainingTensorData):
        raise TypeError("train must be TrainingTensorData")
    weight = _coerce_head_weight(head_weight, train)
    predicted_rewards = train.reward_features @ weight
    return _solve_policy_direction_from_node_rewards(
        train,
        predicted_rewards,
        relative_damping=relative_damping,
        beta=beta,
        pcg_dtype=pcg_dtype,
        pcg_max_iterations=pcg_max_iterations,
        pcg_tolerance=pcg_tolerance,
        pcg_absolute_tolerance=pcg_absolute_tolerance,
        pcg_residual_recompute_interval=pcg_residual_recompute_interval,
        require_pcg_convergence=require_pcg_convergence,
    )


@torch.no_grad()
def policy_direction_from_node_rewards(
    train: TrainingTensorData,
    node_rewards: torch.Tensor,
    *,
    relative_damping: float,
    beta: float = 1.0,
    pcg_dtype: FisherSolveDType = "float64",
    pcg_max_iterations: int = 200,
    pcg_tolerance: float = 1.0e-6,
    pcg_absolute_tolerance: float = 0.0,
    pcg_residual_recompute_interval: int = 20,
    require_pcg_convergence: bool = True,
) -> PolicyDirectionResult:
    """Build a damped natural direction from arbitrary frozen node rewards.

    ``node_rewards`` must have exact shape ``(P, M)`` matching the training
    candidates.  The solve, precision policy, Fisher definition, damping,
    convergence requirement, and evidence record are identical to
    :func:`policy_direction_from_head`.  Passing ``beta=1`` returns the native
    common-beta calibration direction.
    """

    return _solve_policy_direction_from_node_rewards(
        train,
        node_rewards,
        relative_damping=relative_damping,
        beta=beta,
        pcg_dtype=pcg_dtype,
        pcg_max_iterations=pcg_max_iterations,
        pcg_tolerance=pcg_tolerance,
        pcg_absolute_tolerance=pcg_absolute_tolerance,
        pcg_residual_recompute_interval=pcg_residual_recompute_interval,
        require_pcg_convergence=require_pcg_convergence,
    )


@dataclass(frozen=True, slots=True)
class MeasuredKLUpdateResult:
    """JSON-compatible outcome of one reference-anchored measured-KL update."""

    target_kl: float
    initialization: Literal["train_fisher_quadratic", "explicit_step"]
    initial_step_size: float
    fisher_curvature: float | None
    best_step_size: float
    best_measured_kl: float
    applied_step_size: float
    applied_measured_kl: float
    line_search_evaluations: int
    converged: bool
    applied: bool
    reference_forward_evaluations: int
    tangent_dimension: int
    a_state_sha256: str

    def __post_init__(self) -> None:
        _finite_positive("target_kl", self.target_kl)
        if self.initialization not in {"train_fisher_quadratic", "explicit_step"}:
            raise ValueError("invalid initialization")
        _finite_positive("initial_step_size", self.initial_step_size)
        if self.fisher_curvature is not None:
            _finite_positive("fisher_curvature", self.fisher_curvature)
        for name in (
            "best_step_size",
            "best_measured_kl",
            "applied_step_size",
            "applied_measured_kl",
        ):
            _finite_nonnegative(name, getattr(self, name))
        _positive_integer("line_search_evaluations", self.line_search_evaluations)
        if self.reference_forward_evaluations != 1:
            raise ValueError("reference logits must be evaluated exactly once")
        _positive_integer("tangent_dimension", self.tangent_dimension)
        if not isinstance(self.converged, bool) or not isinstance(self.applied, bool):
            raise TypeError("converged and applied must be bool")
        if self.converged != self.applied:
            raise ValueError("a measured-KL step is applied if and only if it converged")
        if not self.applied and (self.applied_step_size != 0.0 or self.applied_measured_kl != 0.0):
            raise ValueError("a failed update must report the restored zero-B state")
        if (
            not isinstance(self.a_state_sha256, str)
            or len(self.a_state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.a_state_sha256)
        ):
            raise ValueError("a_state_sha256 must be a lowercase SHA256 digest")

    def to_dict(self) -> dict[str, object]:
        """Return a structure accepted by strict JSON encoders."""

        return {
            "schema_version": "measured-kl-update/v1",
            "target_kl": self.target_kl,
            "initialization": self.initialization,
            "initial_step_size": self.initial_step_size,
            "fisher_curvature": self.fisher_curvature,
            "best_step_size": self.best_step_size,
            "best_measured_kl": self.best_measured_kl,
            "applied_step_size": self.applied_step_size,
            "applied_measured_kl": self.applied_measured_kl,
            "line_search_evaluations": self.line_search_evaluations,
            "converged": self.converged,
            "applied": self.applied,
            "reference_forward_evaluations": self.reference_forward_evaluations,
            "tangent_dimension": self.tangent_dimension,
            "a_state_sha256": self.a_state_sha256,
        }


def _named_a_parameters(model: torch.nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    named_a = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if _hf._lora_kind(name) == "A"
    )
    if not named_a:
        raise RuntimeError("the configured model no longer exposes LoRA-A parameters")
    return named_a


def _current_a_sha256(model: torch.nn.Module) -> str:
    named_a = _named_a_parameters(model)
    return _hf._fingerprint_named_tensors(named_a)


def _zero_tangent_(named_tangent: Sequence[tuple[str, torch.Tensor]]) -> None:
    with torch.no_grad():
        for _, parameter in named_tangent:
            parameter.zero_()


def _validate_fixed_a_reference(
    model: torch.nn.Module,
    setup: FixedALoRASetup,
    candidates: ExactTokenCandidates,
) -> tuple[tuple[str, torch.Tensor], ...]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(setup, FixedALoRASetup):
        raise TypeError("setup must be FixedALoRASetup")
    if setup.model is not model:
        raise ValueError("model must be the exact instance stored in FixedALoRASetup")
    if not isinstance(candidates, ExactTokenCandidates):
        raise TypeError("reference_candidates must be ExactTokenCandidates")
    if any(module.training for module in model.modules()):
        raise ValueError("model must be in eval mode for measured-KL matching")

    named_tangent = setup.named_tangent_parameters()
    trainable_names = tuple(
        sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    )
    if trainable_names != setup.trainable_names:
        raise RuntimeError("the model trainable-parameter set changed after LoRA setup")
    if any(bool(torch.count_nonzero(parameter.detach())) for _, parameter in named_tangent):
        raise ValueError("measured-KL matching must start from the exact zero-B reference")
    if _current_a_sha256(model) != setup.a_state_sha256:
        raise RuntimeError("fixed LoRA-A state changed after setup")

    if candidates.source_model_id != id(model):
        raise ValueError(
            "reference candidates must carry provenance from this exact model instance"
        )
    current_trainable_sha256 = _hf._fingerprint_named_tensors(named_tangent)
    if candidates.source_trainable_sha256 != current_trainable_sha256:
        raise ValueError("reference candidates were not generated at the current zero-B state")
    return named_tangent


def _flatten_train_node_scores(
    train_node_scores: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(train_node_scores, torch.Tensor):
        raise TypeError("train_node_scores must be a torch.Tensor")
    if train_node_scores.ndim < 2 or train_node_scores.shape[-1] != direction.numel():
        raise ValueError(
            "train_node_scores must have a non-empty sample axis and final tangent dimension"
        )
    if train_node_scores.numel() == 0:
        raise ValueError("train_node_scores must be non-empty")
    if not train_node_scores.is_floating_point() or not bool(
        torch.isfinite(train_node_scores).all()
    ):
        raise ValueError("train_node_scores must be finite and floating point")
    if train_node_scores.dtype != direction.dtype or train_node_scores.device != direction.device:
        raise ValueError("train_node_scores and direction must share dtype and device")
    return train_node_scores.reshape(-1, direction.numel())


def match_fixed_a_measured_kl(
    model: torch.nn.Module,
    setup: FixedALoRASetup,
    reference_candidates: ExactTokenCandidates,
    direction: torch.Tensor,
    *,
    target_kl: float = 0.01,
    train_node_scores: torch.Tensor | None = None,
    initial_step: float | None = None,
    relative_tolerance: float = 0.05,
    max_iterations: int = 30,
) -> MeasuredKLUpdateResult:
    """Match and apply one fixed-A LoRA-B update using measured forward KL.

    Exactly one of ``train_node_scores`` and ``initial_step`` must be supplied.
    In the normal experiment, train scores produce the quadratic Fisher step
    ``sqrt(2*target_kl/(direction.T F direction))``.  The explicit route is a
    deterministic testing/debugging escape hatch.  In both cases the value is
    only a line-search initializer; acceptance always uses measured response-
    token vocabulary forward KL on ``reference_candidates``.
    """

    named_tangent = _validate_fixed_a_reference(model, setup, reference_candidates)
    if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
        raise TypeError("direction must be a one-dimensional torch.Tensor")
    if direction.numel() != setup.layout.dimension:
        raise ValueError(f"direction must have length {setup.layout.dimension}")
    if not direction.is_floating_point() or not bool(torch.isfinite(direction).all()):
        raise ValueError("direction must be finite and floating point")
    if any(parameter.device != direction.device for _, parameter in named_tangent):
        raise ValueError("direction and LoRA-B parameters must be on the same device")
    target = _finite_positive("target_kl", target_kl)
    tolerance = _finite_positive("relative_tolerance", relative_tolerance)
    if tolerance >= 1.0:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    _positive_integer("max_iterations", max_iterations)
    if (train_node_scores is None) == (initial_step is None):
        raise ValueError("provide exactly one of train_node_scores and initial_step")

    fisher_curvature_value: float | None
    initialization: Literal["train_fisher_quadratic", "explicit_step"]
    if train_node_scores is not None:
        flat_scores = _flatten_train_node_scores(train_node_scores, direction)
        fisher = DampedEmpiricalFisher(flat_scores, damping=0.0)
        fisher_curvature_value = float(fisher_quadratic(direction, fisher.matvec).item())
        if fisher_curvature_value <= 0.0:
            raise ValueError("a Fisher-null direction cannot spend a positive KL budget")
        initial_step_value = step_size_for_kl_budget(
            direction,
            fisher.matvec,
            kl_budget=target,
        )
        initialization = "train_fisher_quadratic"
    else:
        fisher_curvature_value = None
        initial_step_value = _finite_positive("initial_step", initial_step)
        initialization = "explicit_step"

    named_a = _named_a_parameters(model)
    original_a = tuple(parameter.detach().clone() for _, parameter in named_a)
    reference_selected_logits = None
    completed = False
    try:
        with torch.no_grad():
            output = model(
                input_ids=reference_candidates.input_ids,
                attention_mask=reference_candidates.attention_mask,
                use_cache=False,
            )
            full_reference_logits = _hf._extract_model_logits(output)
            reference_selected_logits = select_causal_response_logits(
                full_reference_logits,
                reference_candidates.response_mask,
            )
            # Advanced indexing owns the selected response rows.  Release the
            # B x L x V model output before any KL softmax is materialized.
            del full_reference_logits, output
        if any(module.training for module in model.modules()):
            raise RuntimeError("model left eval mode during reference forward")
        if _current_a_sha256(model) != setup.a_state_sha256:
            raise RuntimeError("fixed LoRA-A state changed during reference forward")

        def measure_kl(step_size: float) -> float:
            # ``line_search_measured_kl`` probes zero first.  The saved
            # reference distribution makes that value exactly zero, so this
            # branch avoids a second zero-B model forward.
            if step_size == 0.0:
                set_tangent_update_(
                    named_tangent,
                    setup.layout,
                    direction,
                    step_size=0.0,
                )
                return 0.0
            set_tangent_update_(
                named_tangent,
                setup.layout,
                direction,
                step_size=step_size,
            )
            if any(module.training for module in model.modules()):
                raise RuntimeError("model must remain in eval mode during KL trials")
            if _current_a_sha256(model) != setup.a_state_sha256:
                raise RuntimeError("fixed LoRA-A state changed before a KL trial")
            with torch.no_grad():
                output = model(
                    input_ids=reference_candidates.input_ids,
                    attention_mask=reference_candidates.attention_mask,
                    use_cache=False,
                )
                full_updated_logits = _hf._extract_model_logits(output)
                updated_selected_logits = select_causal_response_logits(
                    full_updated_logits,
                    reference_candidates.response_mask,
                )
                del full_updated_logits, output
                measured = selected_causal_forward_kl(
                    reference_selected_logits,
                    updated_selected_logits,
                )
            if any(module.training for module in model.modules()):
                raise RuntimeError("model left eval mode during a KL trial")
            if _current_a_sha256(model) != setup.a_state_sha256:
                raise RuntimeError("fixed LoRA-A state changed during a KL trial")
            setup.named_tangent_parameters()
            return float(measured.item())

        search = line_search_measured_kl(
            measure_kl,
            target_kl=target,
            initial_step=initial_step_value,
            relative_tolerance=tolerance,
            max_iterations=max_iterations,
        )
        if not search.converged:
            _zero_tangent_(named_tangent)
            return MeasuredKLUpdateResult(
                target_kl=target,
                initialization=initialization,
                initial_step_size=initial_step_value,
                fisher_curvature=fisher_curvature_value,
                best_step_size=search.step_size,
                best_measured_kl=search.measured_kl,
                applied_step_size=0.0,
                applied_measured_kl=0.0,
                line_search_evaluations=search.iterations,
                converged=False,
                applied=False,
                reference_forward_evaluations=1,
                tangent_dimension=setup.layout.dimension,
                a_state_sha256=setup.a_state_sha256,
            )

        # The best trial need not be the final callback evaluation.  Overwrite
        # from the zero-B coordinate origin once more to make the committed
        # policy state exactly match the reported step.
        set_tangent_update_(
            named_tangent,
            setup.layout,
            direction,
            step_size=search.step_size,
        )
        if any(module.training for module in model.modules()):
            raise RuntimeError("model must remain in eval mode after KL matching")
        setup.named_tangent_parameters()
        if _current_a_sha256(model) != setup.a_state_sha256:
            raise RuntimeError("fixed LoRA-A state changed during KL matching")
        result = MeasuredKLUpdateResult(
            target_kl=target,
            initialization=initialization,
            initial_step_size=initial_step_value,
            fisher_curvature=fisher_curvature_value,
            best_step_size=search.step_size,
            best_measured_kl=search.measured_kl,
            applied_step_size=search.step_size,
            applied_measured_kl=search.measured_kl,
            line_search_evaluations=search.iterations,
            converged=True,
            applied=True,
            reference_forward_evaluations=1,
            tangent_dimension=setup.layout.dimension,
            a_state_sha256=setup.a_state_sha256,
        )
        completed = True
        return result
    finally:
        # This covers model-forward errors, invalid KL values, line-search
        # exceptions, and every non-converged return.  A successful return is
        # the sole path that may leave LoRA-B nonzero.
        if not completed:
            _zero_tangent_(named_tangent)
            with torch.no_grad():
                for (_, parameter), original in zip(named_a, original_a, strict=True):
                    parameter.copy_(original)


@dataclass(frozen=True, slots=True)
class OracleRolloutImprovement:
    """Descriptive paired rollout change; it deliberately carries no test."""

    num_pairs: int
    mean_difference: float
    sample_standard_error: float
    significance_claimed: bool = False

    def __post_init__(self) -> None:
        if self.num_pairs < 2:
            raise ValueError("num_pairs must be at least two for a sample standard error")
        if not math.isfinite(self.mean_difference):
            raise ValueError("mean_difference must be finite")
        _finite_nonnegative("sample_standard_error", self.sample_standard_error)
        if self.significance_claimed:
            raise ValueError("this descriptive statistic must not claim significance")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "oracle-rollout-improvement/v1",
            "num_pairs": self.num_pairs,
            "mean_difference": self.mean_difference,
            "sample_standard_error": self.sample_standard_error,
            "significance_claimed": False,
        }


def oracle_rollout_improvement(
    reference_rewards: torch.Tensor,
    updated_rewards: torch.Tensor,
) -> OracleRolloutImprovement:
    """Return paired ``updated-reference`` mean and sample standard error.

    Inputs must be the same number of finite transformed-oracle rewards.  The
    output is descriptive uncertainty only: this function performs no test,
    emits no p-value, and makes no significance claim.
    """

    if not isinstance(reference_rewards, torch.Tensor) or not isinstance(
        updated_rewards, torch.Tensor
    ):
        raise TypeError("reference_rewards and updated_rewards must be tensors")
    if reference_rewards.ndim != 1 or updated_rewards.shape != reference_rewards.shape:
        raise ValueError("rewards must be equal-length one-dimensional tensors")
    if reference_rewards.numel() < 2:
        raise ValueError("at least two paired rewards are required for a sample SE")
    if not reference_rewards.is_floating_point() or not updated_rewards.is_floating_point():
        raise TypeError("rewards must be floating point")
    if (
        reference_rewards.dtype != updated_rewards.dtype
        or reference_rewards.device != updated_rewards.device
    ):
        raise ValueError("reference and updated rewards must share dtype and device")
    if not bool(torch.isfinite(reference_rewards).all()) or not bool(
        torch.isfinite(updated_rewards).all()
    ):
        raise ValueError("all transformed rewards must be finite")

    differences = (updated_rewards - reference_rewards).detach().to(torch.float64)
    mean_difference = float(differences.mean().item())
    sample_standard_error = float(
        (differences.std(unbiased=True) / math.sqrt(differences.numel())).item()
    )
    return OracleRolloutImprovement(
        num_pairs=differences.numel(),
        mean_difference=mean_difference,
        sample_standard_error=sample_standard_error,
    )


__all__ = [
    "MeasuredKLUpdateResult",
    "OracleRolloutImprovement",
    "PolicyDirectionResult",
    "match_fixed_a_measured_kl",
    "oracle_rollout_improvement",
    "policy_direction_from_head",
    "policy_direction_from_node_rewards",
]
