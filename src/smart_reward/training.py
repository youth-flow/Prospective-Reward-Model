"""Deterministic frozen-feature training loops for BT-MLE and ProRM+.

The ProRM+ loop in this module deliberately implements one, and only one,
reward-model optimizer update per freshly solved dual problem::

    full margins -> moment -> warm-started PCG -> detach direction
                 -> one optimizer step -> repeat

``microbatch_size`` only changes how that *full-batch* gradient is accumulated.
It never turns the ProRM+ moment into a stochastic minibatch estimate and never
reuses a stale dual direction after a parameter update.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn

from .baseline import repeated_btl_nll
from .contracts import (
    BT_MLE,
    CHECKPOINT_FORMAT_V1,
    CHECKPOINT_FORMAT_V2,
    LEGACY_SRM_PLUS,
    PRORM_PLUS,
)
from .linear import (
    DampedEmpiricalFisher,
    FisherSolveDType,
    resolve_fisher_solve_dtype,
)
from .objective import (
    dual_loss,
    dual_saddle_value,
    empirical_moment,
    envelope_surrogate,
    envelope_weights,
)
from .pcg import PCGResult, pcg

_ORIENTATION = "left_minus_right"
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    result = float(value)
    valid = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not valid:
        qualifier = "strictly positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True)
class FeatureTrainingBatch:
    """Validated frozen tensors for one canonical set of comparison edges.

    Every edge is oriented as ``left - right`` simultaneously in the reward
    features, policy-score difference ``edge_scores`` (the matrix ``Z``),
    target ``h``, and left-win counts.  Since feature and policy-score spaces
    need not have the same dimension, this convention cannot be inferred from
    values; the explicit ``orientation`` token makes it part of the data
    contract.  Use :meth:`swapped` rather than manually changing endpoints.

    ``node_scores`` is the independently sampled node score matrix ``S`` used
    for the empirical Fisher.  It is intentionally not derived from endpoint
    frequency in ``edge_scores``.
    """

    left_features: torch.Tensor
    right_features: torch.Tensor
    edge_scores: torch.Tensor
    node_scores: torch.Tensor
    h: torch.Tensor
    left_wins: torch.Tensor
    num_annotations: torch.Tensor
    orientation: Literal["left_minus_right"] = _ORIENTATION

    def __post_init__(self) -> None:
        tensor_fields = {
            "left_features": self.left_features,
            "right_features": self.right_features,
            "edge_scores": self.edge_scores,
            "node_scores": self.node_scores,
            "h": self.h,
            "left_wins": self.left_wins,
            "num_annotations": self.num_annotations,
        }
        for name, value in tensor_fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        if self.orientation != _ORIENTATION:
            raise ValueError(
                "orientation must be 'left_minus_right'; call swapped() to reverse edges"
            )
        if self.left_features.ndim != 2:
            raise ValueError("left_features must have shape (num_edges, reward_dimension)")
        if self.right_features.shape != self.left_features.shape:
            raise ValueError("right_features must have the same shape as left_features")
        num_edges, reward_dimension = self.left_features.shape
        if num_edges < 1 or reward_dimension < 1:
            raise ValueError("reward feature dimensions must both be positive")
        if self.edge_scores.ndim != 2 or self.edge_scores.shape[0] != num_edges:
            raise ValueError("edge_scores must have shape (num_edges, policy_dimension)")
        if self.node_scores.ndim != 2:
            raise ValueError("node_scores must have shape (num_nodes, policy_dimension)")
        if self.edge_scores.shape[1] < 1:
            raise ValueError("the policy dimension must be positive")
        if self.node_scores.shape[0] < 1:
            raise ValueError("node_scores must contain at least one node")
        if self.node_scores.shape[1] != self.edge_scores.shape[1]:
            raise ValueError("edge_scores and node_scores must share the policy dimension")
        for name, value in (
            ("h", self.h),
            ("left_wins", self.left_wins),
            ("num_annotations", self.num_annotations),
        ):
            if value.shape != (num_edges,):
                raise ValueError(f"{name} must have shape ({num_edges},)")

        floating = {
            "left_features": self.left_features,
            "right_features": self.right_features,
            "edge_scores": self.edge_scores,
            "node_scores": self.node_scores,
            "h": self.h,
        }
        for name, value in floating.items():
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if value.requires_grad:
                raise ValueError(f"{name} must be frozen (requires_grad=False)")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        reference = self.left_features
        for name, value in floating.items():
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(
                    f"{name} must have the same floating dtype and device as left_features"
                )

        if (
            self.left_wins.dtype not in _INTEGER_DTYPES
            or self.num_annotations.dtype not in _INTEGER_DTYPES
        ):
            raise TypeError("left_wins and num_annotations must have integer dtypes")
        if self.left_wins.dtype != self.num_annotations.dtype:
            raise ValueError("left_wins and num_annotations must have the same dtype")
        if (
            self.left_wins.device != reference.device
            or self.num_annotations.device != reference.device
        ):
            raise ValueError("all tensors must be on the same device")
        if bool((self.num_annotations < 1).any()):
            raise ValueError("every edge must have at least one annotation")
        if bool(((self.left_wins < 0) | (self.left_wins > self.num_annotations)).any()):
            raise ValueError("left_wins must satisfy 0 <= left_wins <= num_annotations")

    @property
    def num_edges(self) -> int:
        """Number of comparison edges."""

        return self.left_features.shape[0]

    @property
    def reward_dimension(self) -> int:
        """Dimension of the frozen reward features."""

        return self.left_features.shape[1]

    @property
    def policy_dimension(self) -> int:
        """Dimension of the fixed policy tangent coordinates."""

        return self.edge_scores.shape[1]

    @property
    def Z(self) -> torch.Tensor:
        """Mathematical alias for :attr:`edge_scores`."""

        return self.edge_scores

    @property
    def S(self) -> torch.Tensor:
        """Mathematical alias for :attr:`node_scores`."""

        return self.node_scores

    @property
    def N(self) -> torch.Tensor:
        """Mathematical alias for :attr:`num_annotations`."""

        return self.num_annotations

    @property
    def feature_differences(self) -> torch.Tensor:
        """Return frozen reward features oriented as ``left - right``."""

        return self.left_features - self.right_features

    def swapped(self) -> FeatureTrainingBatch:
        """Return the globally reversed but mathematically equivalent batch."""

        return FeatureTrainingBatch(
            left_features=self.right_features,
            right_features=self.left_features,
            edge_scores=-self.edge_scores,
            node_scores=self.node_scores,
            h=-self.h,
            left_wins=self.num_annotations - self.left_wins,
            num_annotations=self.num_annotations,
            orientation=_ORIENTATION,
        )


class FrozenFeatureLinearReward(nn.Module):
    """Bias-free scalar reward head over immutable precomputed features."""

    weight: nn.Parameter

    def __init__(
        self,
        feature_dimension: int,
        initial_weight: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        dimension = _positive_integer("feature_dimension", feature_dimension)
        if initial_weight is None:
            effective_dtype = torch.get_default_dtype() if dtype is None else dtype
            if not isinstance(effective_dtype, torch.dtype):
                raise TypeError("dtype must be a torch dtype")
            if not effective_dtype.is_floating_point:
                raise TypeError("dtype must be a floating-point torch dtype")
            value = torch.zeros(dimension, dtype=effective_dtype, device=device)
        else:
            if not isinstance(initial_weight, torch.Tensor):
                raise TypeError("initial_weight must be a torch.Tensor")
            if initial_weight.shape not in {(dimension,), (1, dimension)}:
                raise ValueError(f"initial_weight must have shape ({dimension},)")
            if not initial_weight.is_floating_point():
                raise TypeError("initial_weight must have a floating-point dtype")
            if not bool(torch.isfinite(initial_weight).all()):
                raise ValueError("initial_weight must be finite")
            if dtype is not None:
                if not isinstance(dtype, torch.dtype):
                    raise TypeError("dtype must be a torch dtype")
                if not dtype.is_floating_point:
                    raise TypeError("dtype must be a floating-point torch dtype")
            value = (
                initial_weight.detach()
                .reshape(dimension)
                .clone()
                .to(
                    device=device if device is not None else initial_weight.device,
                    dtype=dtype if dtype is not None else initial_weight.dtype,
                )
            )
        self.weight = nn.Parameter(value)

    @property
    def feature_dimension(self) -> int:
        """Input feature dimension."""

        return self.weight.numel()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the scalar head to a tensor whose last axis is the feature axis."""

        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim < 1 or features.shape[-1] != self.feature_dimension:
            raise ValueError(f"features must have final dimension {self.feature_dimension}")
        if not features.is_floating_point():
            raise TypeError("features must have a floating-point dtype")
        if features.dtype != self.weight.dtype or features.device != self.weight.device:
            raise ValueError("features and reward head must have the same dtype and device")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must be finite")
        return features @ self.weight

    def margins(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return canonical ``reward(left) - reward(right)`` margins."""

        if not isinstance(left_features, torch.Tensor) or not isinstance(
            right_features, torch.Tensor
        ):
            raise TypeError("left_features and right_features must be torch.Tensor objects")
        if right_features.shape != left_features.shape:
            raise ValueError("left_features and right_features must have identical shapes")
        return self(left_features - right_features)


OptimizerName = Literal["adamw", "sgd"]


@dataclass(frozen=True)
class BTMLETrainingConfig:
    """Optimizer and deterministic accumulation settings for repeated-label BT-MLE."""

    learning_rate: float = 1.0e-2
    optimizer: OptimizerName = "adamw"
    weight_decay: float = 0.0
    microbatch_size: int | None = None
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        _finite_scalar("learning_rate", self.learning_rate, positive=True)
        _finite_scalar("weight_decay", self.weight_decay)
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        if self.microbatch_size is not None:
            _positive_integer("microbatch_size", self.microbatch_size)
        if self.max_grad_norm is not None:
            _finite_scalar("max_grad_norm", self.max_grad_norm, positive=True)


@dataclass(frozen=True)
class ProRMPlusTrainingConfig:
    """Fixed ProRM+ objective, optimizer, and PCG settings."""

    learning_rate: float = 1.0e-2
    optimizer: OptimizerName = "adamw"
    weight_decay: float = 0.0
    microbatch_size: int | None = None
    max_grad_norm: float | None = None
    beta: float = 1.0
    damping: float = 1.0e-3
    pcg_dtype: FisherSolveDType = "float64"
    pcg_max_iterations: int = 200
    pcg_tolerance: float = 1.0e-5
    pcg_absolute_tolerance: float = 0.0
    pcg_residual_recompute_interval: int = 20
    require_pcg_convergence: bool = True

    def __post_init__(self) -> None:
        _finite_scalar("learning_rate", self.learning_rate, positive=True)
        _finite_scalar("weight_decay", self.weight_decay)
        _finite_scalar("beta", self.beta, positive=True)
        _finite_scalar("damping", self.damping)
        resolve_fisher_solve_dtype(self.pcg_dtype)
        _finite_scalar("pcg_tolerance", self.pcg_tolerance)
        _finite_scalar("pcg_absolute_tolerance", self.pcg_absolute_tolerance)
        _nonnegative_integer("pcg_max_iterations", self.pcg_max_iterations)
        _positive_integer(
            "pcg_residual_recompute_interval",
            self.pcg_residual_recompute_interval,
        )
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        if self.microbatch_size is not None:
            _positive_integer("microbatch_size", self.microbatch_size)
        if self.max_grad_norm is not None:
            _finite_scalar("max_grad_norm", self.max_grad_norm, positive=True)
        if not isinstance(self.require_pcg_convergence, bool):
            raise TypeError("require_pcg_convergence must be bool")


@dataclass(frozen=True)
class TrainingStepDiagnostics:
    """Scalar evidence recorded for one parameter update.

    ``objective`` is evaluated at the parameters *before* the update whose
    gradient norm is reported.  For ProRM+ it equals ``dual_loss``.
    """

    step: int
    objective: float
    gradient_norm: float
    dual_loss: float | None = None
    dual_saddle_value: float | None = None
    dual_refresh: int | None = None
    pcg_iterations: int | None = None
    pcg_residual_norm: float | None = None
    pcg_relative_residual: float | None = None
    pcg_converged: bool | None = None

    def __post_init__(self) -> None:
        _positive_integer("step", self.step)
        for name in ("objective", "gradient_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.gradient_norm < 0.0:
            raise ValueError("gradient_norm must be non-negative")
        optional_floats = (
            "dual_loss",
            "dual_saddle_value",
            "pcg_residual_norm",
            "pcg_relative_residual",
        )
        for name in optional_floats:
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when present")
        for name in ("dual_refresh", "pcg_iterations"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative_integer(name, value)
        if self.pcg_converged is not None and not isinstance(self.pcg_converged, bool):
            raise TypeError("pcg_converged must be bool when present")


@dataclass(frozen=True)
class ProRMPlusEvaluation:
    """Full-data ProRM+ value and numerical solve evidence."""

    moment: torch.Tensor
    direction: torch.Tensor
    dual_loss: float
    dual_saddle_value: float
    pcg_iterations: int
    pcg_residual_norm: float
    pcg_relative_residual: float
    pcg_converged: bool


@dataclass(frozen=True)
class _ProRMPolicyGeometry:
    """Cached FP64 tensors for the Fisher/GMM inner problem.

    Reward features and the trainable head remain in their configured dtype.
    Only the fixed policy tangent geometry is promoted, which prevents an
    FP32 residual floor without turning the outer optimizer into FP64.
    """

    edge_scores: torch.Tensor
    node_scores: torch.Tensor
    h: torch.Tensor
    operator: DampedEmpiricalFisher

    @classmethod
    def from_batch(
        cls,
        batch: FeatureTrainingBatch,
        damping: float,
        pcg_dtype: FisherSolveDType,
    ) -> _ProRMPolicyGeometry:
        solve_dtype = resolve_fisher_solve_dtype(pcg_dtype)
        edge_scores = batch.edge_scores.to(dtype=solve_dtype)
        node_scores = batch.node_scores.to(dtype=solve_dtype)
        h = batch.h.to(dtype=solve_dtype)
        return cls(
            edge_scores=edge_scores,
            node_scores=node_scores,
            h=h,
            operator=DampedEmpiricalFisher(node_scores, damping),
        )


def _validate_model_batch(
    model: FrozenFeatureLinearReward,
    batch: FeatureTrainingBatch,
) -> None:
    if not isinstance(model, FrozenFeatureLinearReward):
        raise TypeError("model must be a FrozenFeatureLinearReward")
    if not isinstance(batch, FeatureTrainingBatch):
        raise TypeError("batch must be a FeatureTrainingBatch")
    if model.feature_dimension != batch.reward_dimension:
        raise ValueError("model and batch reward feature dimensions do not match")
    if model.weight.dtype != batch.left_features.dtype:
        raise ValueError("model and batch must have the same floating dtype")
    if model.weight.device != batch.left_features.device:
        raise ValueError("model and batch must be on the same device")
    if not bool(torch.isfinite(model.weight).all()):
        raise ValueError("model weight must be finite")


def _slices(num_items: int, microbatch_size: int | None) -> tuple[slice, ...]:
    size = num_items if microbatch_size is None else min(microbatch_size, num_items)
    return tuple(slice(start, min(start + size, num_items)) for start in range(0, num_items, size))


@torch.no_grad()
def _full_margins(
    model: FrozenFeatureLinearReward,
    batch: FeatureTrainingBatch,
    microbatch_size: int | None,
) -> torch.Tensor:
    pieces = [
        model.margins(batch.left_features[index], batch.right_features[index])
        for index in _slices(batch.num_edges, microbatch_size)
    ]
    return torch.cat(pieces)


def _make_optimizer(
    model: FrozenFeatureLinearReward,
    *,
    name: OptimizerName,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(
            [model.weight],
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    return torch.optim.SGD(
        [model.weight],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def _validate_optimizer(
    optimizer: torch.optim.Optimizer,
    model: FrozenFeatureLinearReward,
) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(parameters) != 1 or parameters[0] is not model.weight:
        raise ValueError("optimizer must contain exactly the reward head weight")
    for group in optimizer.param_groups:
        weight_decay = float(group.get("weight_decay", 0.0))
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("optimizer weight_decay must be finite and non-negative")


def _gradient_norm(model: FrozenFeatureLinearReward) -> float:
    gradient = model.weight.grad
    if gradient is None:
        raise RuntimeError("reward head did not receive a gradient")
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("reward head gradient is non-finite")
    return float(torch.linalg.vector_norm(gradient.detach()).item())


def _clip_gradient(
    model: FrozenFeatureLinearReward,
    max_grad_norm: float | None,
) -> None:
    if max_grad_norm is None:
        return
    torch.nn.utils.clip_grad_norm_(
        [model.weight],
        max_norm=float(max_grad_norm),
        error_if_nonfinite=True,
    )


def _check_updated_model(model: FrozenFeatureLinearReward) -> None:
    if not bool(torch.isfinite(model.weight).all()):
        raise FloatingPointError("optimizer produced a non-finite reward head")


def _solve_prorm_dual(
    batch: FeatureTrainingBatch,
    margins: torch.Tensor,
    config: ProRMPlusTrainingConfig,
    warm_start: torch.Tensor | None,
    geometry: _ProRMPolicyGeometry | None = None,
) -> tuple[torch.Tensor, DampedEmpiricalFisher, PCGResult]:
    workspace = (
        _ProRMPolicyGeometry.from_batch(batch, config.damping, config.pcg_dtype)
        if geometry is None
        else geometry
    )
    moment = empirical_moment(
        workspace.edge_scores,
        margins.to(dtype=workspace.edge_scores.dtype),
        workspace.h,
    )
    operator = workspace.operator
    promoted_warm_start = (
        None if warm_start is None else warm_start.to(device=moment.device, dtype=moment.dtype)
    )
    result = pcg(
        operator.matvec,
        moment,
        # The empirical Fisher is low rank plus isotropic damping.  Jacobi
        # scaling destroys the large repeated damping eigenvalue and is
        # markedly worse for this geometry; unpreconditioned CG preserves the
        # rank(S)+1 Krylov structure.
        inverse_diagonal=None,
        x0=promoted_warm_start,
        max_iterations=config.pcg_max_iterations,
        tolerance=config.pcg_tolerance,
        absolute_tolerance=config.pcg_absolute_tolerance,
        residual_recompute_interval=config.pcg_residual_recompute_interval,
    )
    if config.require_pcg_convergence and not result.converged:
        raise RuntimeError(
            "PCG did not converge: "
            f"iterations={result.iterations}, relative_residual={result.relative_residual:.3e}"
        )
    return moment, operator, result


@torch.no_grad()
def evaluate_bt_mle(
    model: FrozenFeatureLinearReward,
    batch: FeatureTrainingBatch,
    *,
    microbatch_size: int | None = None,
) -> float:
    """Evaluate the exact label-level repeated-label BT negative log-likelihood."""

    _validate_model_batch(model, batch)
    if microbatch_size is not None:
        _positive_integer("microbatch_size", microbatch_size)
    margins = _full_margins(model, batch, microbatch_size)
    return float(repeated_btl_nll(margins, batch.left_wins, batch.num_annotations).item())


@torch.no_grad()
def evaluate_prorm_plus(
    model: FrozenFeatureLinearReward,
    batch: FeatureTrainingBatch,
    config: ProRMPlusTrainingConfig | None = None,
    *,
    warm_start: torch.Tensor | None = None,
) -> ProRMPlusEvaluation:
    """Evaluate ProRM+ on all edges and return PCG residual diagnostics."""

    effective_config = ProRMPlusTrainingConfig() if config is None else config
    if not isinstance(effective_config, ProRMPlusTrainingConfig):
        raise TypeError("config must be a ProRMPlusTrainingConfig")
    _validate_model_batch(model, batch)
    if warm_start is not None:
        if not isinstance(warm_start, torch.Tensor):
            raise TypeError("warm_start must be a torch.Tensor")
        if warm_start.shape != (batch.policy_dimension,):
            raise ValueError("warm_start has the wrong policy dimension")
        if not warm_start.is_floating_point():
            raise TypeError("warm_start must have a floating-point dtype")
        if warm_start.device != batch.edge_scores.device:
            raise ValueError("warm_start must match the batch device")
        if not bool(torch.isfinite(warm_start).all()):
            raise ValueError("warm_start must be finite")
    margins = _full_margins(model, batch, effective_config.microbatch_size)
    moment, operator, result = _solve_prorm_dual(
        batch,
        margins,
        effective_config,
        warm_start,
    )
    direction = result.solution.detach()
    loss = dual_loss(moment, direction, beta=effective_config.beta)
    saddle = dual_saddle_value(
        moment,
        direction,
        operator.matvec(direction),
        beta=effective_config.beta,
    )
    return ProRMPlusEvaluation(
        moment=moment.detach().clone(),
        direction=direction.clone(),
        dual_loss=float(loss.item()),
        dual_saddle_value=float(saddle.item()),
        pcg_iterations=result.iterations,
        pcg_residual_norm=result.residual_norm,
        pcg_relative_residual=result.relative_residual,
        pcg_converged=result.converged,
    )


class BTMLETrainer:
    """Stateful deterministic trainer for the repeated-label BT baseline."""

    def __init__(
        self,
        model: FrozenFeatureLinearReward,
        batch: FeatureTrainingBatch,
        config: BTMLETrainingConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        effective_config = BTMLETrainingConfig() if config is None else config
        if not isinstance(effective_config, BTMLETrainingConfig):
            raise TypeError("config must be a BTMLETrainingConfig")
        _validate_model_batch(model, batch)
        self.model = model
        self.batch = batch
        self.config = effective_config
        self.optimizer = optimizer or _make_optimizer(
            model,
            name=effective_config.optimizer,
            learning_rate=effective_config.learning_rate,
            weight_decay=effective_config.weight_decay,
        )
        _validate_optimizer(self.optimizer, model)
        self.completed_steps = 0
        self.history: list[TrainingStepDiagnostics] = []

    def step(self) -> TrainingStepDiagnostics:
        """Accumulate the exact full-data BT gradient and update once."""

        self.optimizer.zero_grad(set_to_none=True)
        total_annotations = self.batch.num_annotations.sum().to(
            dtype=self.batch.left_features.dtype
        )
        objective = torch.zeros((), dtype=self.model.weight.dtype, device=self.model.weight.device)
        for index in _slices(self.batch.num_edges, self.config.microbatch_size):
            margins = self.model.margins(
                self.batch.left_features[index],
                self.batch.right_features[index],
            )
            chunk_loss = repeated_btl_nll(
                margins,
                self.batch.left_wins[index],
                self.batch.num_annotations[index],
            )
            chunk_annotations = self.batch.num_annotations[index].sum().to(dtype=margins.dtype)
            scaled_loss = chunk_loss * (chunk_annotations / total_annotations)
            scaled_loss.backward()
            objective = objective + scaled_loss.detach()
        gradient_norm = _gradient_norm(self.model)
        _clip_gradient(self.model, self.config.max_grad_norm)
        self.optimizer.step()
        _check_updated_model(self.model)
        self.completed_steps += 1
        diagnostic = TrainingStepDiagnostics(
            step=self.completed_steps,
            objective=float(objective.item()),
            gradient_norm=gradient_norm,
        )
        self.history.append(diagnostic)
        return diagnostic

    def fit(self, num_steps: int) -> tuple[TrainingStepDiagnostics, ...]:
        """Run ``num_steps`` updates and return only the newly recorded diagnostics."""

        count = _nonnegative_integer("num_steps", num_steps)
        return tuple(self.step() for _ in range(count))

    def evaluate(self) -> float:
        """Evaluate the current full repeated-label BT objective."""

        return evaluate_bt_mle(
            self.model,
            self.batch,
            microbatch_size=self.config.microbatch_size,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a detached in-memory checkpoint; this method performs no I/O."""

        return {
            "format_version": CHECKPOINT_FORMAT_V2,
            "trainer": BT_MLE,
            "config": asdict(self.config),
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "completed_steps": self.completed_steps,
            "history": [asdict(item) for item in self.history],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpoint produced by :meth:`state_dict`."""

        common = _validate_checkpoint(state, BT_MLE, asdict(self.config))
        self.model.load_state_dict(common["model"], strict=True)
        self.optimizer.load_state_dict(common["optimizer"])
        _validate_optimizer(self.optimizer, self.model)
        self.completed_steps = common["completed_steps"]
        self.history = common["history"]
        _validate_model_batch(self.model, self.batch)


class ProRMPlusTrainer:
    """Stateful one-dual-solve/one-update trainer for ProRM+."""

    def __init__(
        self,
        model: FrozenFeatureLinearReward,
        batch: FeatureTrainingBatch,
        config: ProRMPlusTrainingConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        effective_config = ProRMPlusTrainingConfig() if config is None else config
        if not isinstance(effective_config, ProRMPlusTrainingConfig):
            raise TypeError("config must be a ProRMPlusTrainingConfig")
        _validate_model_batch(model, batch)
        self.model = model
        self.batch = batch
        self.config = effective_config
        self._policy_geometry = _ProRMPolicyGeometry.from_batch(
            batch,
            effective_config.damping,
            effective_config.pcg_dtype,
        )
        self.optimizer = optimizer or _make_optimizer(
            model,
            name=effective_config.optimizer,
            learning_rate=effective_config.learning_rate,
            weight_decay=effective_config.weight_decay,
        )
        _validate_optimizer(self.optimizer, model)
        self.completed_steps = 0
        self.dual_refreshes = 0
        self.dual_direction: torch.Tensor | None = None
        self._last_pcg_reason: Literal["converged", "zero_rhs", "max_iterations"] | None = None
        self.history: list[TrainingStepDiagnostics] = []

    @property
    def last_pcg_reason(
        self,
    ) -> Literal["converged", "zero_rhs", "max_iterations"] | None:
        """Return the raw PCG stop reason for the latest successful update.

        The value is transient diagnostic evidence rather than future-update
        state, so it is intentionally absent from :meth:`state_dict`.  Loading
        a checkpoint clears it until the resumed trainer completes an update.
        """

        return self._last_pcg_reason

    def step(self) -> TrainingStepDiagnostics:
        """Refresh the full dual direction, detach it, and update exactly once."""

        self._last_pcg_reason = None
        # This no-grad pass is intentionally complete before any outer graph is built.
        margins = _full_margins(self.model, self.batch, self.config.microbatch_size)
        moment, operator, result = _solve_prorm_dual(
            self.batch,
            margins,
            self.config,
            self.dual_direction,
            self._policy_geometry,
        )
        direction = result.solution.detach().clone()
        self.dual_direction = direction
        self.dual_refreshes += 1
        loss = dual_loss(moment, direction, beta=self.config.beta)
        saddle = dual_saddle_value(
            moment,
            direction,
            operator.matvec(direction),
            beta=self.config.beta,
        )

        self.optimizer.zero_grad(set_to_none=True)
        for index in _slices(self.batch.num_edges, self.config.microbatch_size):
            chunk_margins = self.model.margins(
                self.batch.left_features[index],
                self.batch.right_features[index],
            )
            policy_weights = envelope_weights(
                self._policy_geometry.edge_scores[index],
                direction,
                beta=self.config.beta,
                detach_direction=True,
            )
            # This is the only precision boundary in the ProRM envelope
            # gradient: the accurately solved scalar edge weights are cast to
            # the reward-head dtype used by autograd and AdamW.
            weights = policy_weights.to(dtype=chunk_margins.dtype)
            chunk_surrogate = envelope_surrogate(
                chunk_margins,
                self.batch.h[index],
                weights,
            )
            # Preserve the global 1 / num_edges factor exactly.  No weight
            # standardization or other dynamic normalization is permitted.
            scaled_surrogate = chunk_surrogate * ((index.stop - index.start) / self.batch.num_edges)
            scaled_surrogate.backward()
        gradient_norm = _gradient_norm(self.model)
        _clip_gradient(self.model, self.config.max_grad_norm)
        self.optimizer.step()
        _check_updated_model(self.model)

        self._last_pcg_reason = result.reason
        self.completed_steps += 1
        diagnostic = TrainingStepDiagnostics(
            step=self.completed_steps,
            objective=float(loss.item()),
            gradient_norm=gradient_norm,
            dual_loss=float(loss.item()),
            dual_saddle_value=float(saddle.item()),
            dual_refresh=self.dual_refreshes,
            pcg_iterations=result.iterations,
            pcg_residual_norm=result.residual_norm,
            pcg_relative_residual=result.relative_residual,
            pcg_converged=result.converged,
        )
        self.history.append(diagnostic)
        return diagnostic

    def fit(self, num_steps: int) -> tuple[TrainingStepDiagnostics, ...]:
        """Run ``num_steps`` fresh-dual updates."""

        count = _nonnegative_integer("num_steps", num_steps)
        return tuple(self.step() for _ in range(count))

    def evaluate(self, *, use_warm_start: bool = True) -> ProRMPlusEvaluation:
        """Evaluate current full-data ProRM+ loss without changing trainer state."""

        warm_start = self.dual_direction if use_warm_start else None
        margins = _full_margins(
            self.model,
            self.batch,
            self.config.microbatch_size,
        )
        moment, operator, result = _solve_prorm_dual(
            self.batch,
            margins,
            self.config,
            warm_start,
            self._policy_geometry,
        )
        direction = result.solution.detach()
        loss = dual_loss(moment, direction, beta=self.config.beta)
        saddle = dual_saddle_value(
            moment,
            direction,
            operator.matvec(direction),
            beta=self.config.beta,
        )
        return ProRMPlusEvaluation(
            moment=moment.detach().clone(),
            direction=direction.clone(),
            dual_loss=float(loss.item()),
            dual_saddle_value=float(saddle.item()),
            pcg_iterations=result.iterations,
            pcg_residual_norm=result.residual_norm,
            pcg_relative_residual=result.relative_residual,
            pcg_converged=result.converged,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a deterministic detached checkpoint dictionary without I/O."""

        return {
            "format_version": CHECKPOINT_FORMAT_V2,
            "trainer": PRORM_PLUS,
            "config": asdict(self.config),
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "completed_steps": self.completed_steps,
            "history": [asdict(item) for item in self.history],
            "dual_refreshes": self.dual_refreshes,
            "dual_direction": (
                None if self.dual_direction is None else self.dual_direction.detach().clone()
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpoint produced by :meth:`state_dict`."""

        common = _validate_checkpoint(
            state,
            PRORM_PLUS,
            asdict(self.config),
            legacy_trainer=LEGACY_SRM_PLUS,
        )
        required = {"dual_refreshes", "dual_direction"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
        refreshes = _nonnegative_integer("dual_refreshes", state["dual_refreshes"])
        if refreshes != common["completed_steps"]:
            raise ValueError("dual_refreshes must equal completed_steps")
        direction = state["dual_direction"]
        if direction is None:
            if refreshes != 0:
                raise ValueError("a trained ProRM+ checkpoint must contain dual_direction")
            restored_direction = None
        else:
            if not isinstance(direction, torch.Tensor):
                raise TypeError("dual_direction must be a torch.Tensor or None")
            if direction.shape != (self.batch.policy_dimension,):
                raise ValueError("dual_direction has the wrong shape")
            if (
                direction.dtype != self._policy_geometry.edge_scores.dtype
                or direction.device != self._policy_geometry.edge_scores.device
            ):
                raise ValueError("dual_direction has the wrong solver dtype or device")
            if not bool(torch.isfinite(direction).all()):
                raise ValueError("dual_direction must be finite")
            restored_direction = direction.detach().clone()

        self.model.load_state_dict(common["model"], strict=True)
        self.optimizer.load_state_dict(common["optimizer"])
        _validate_optimizer(self.optimizer, self.model)
        self.completed_steps = common["completed_steps"]
        self.history = common["history"]
        self.dual_refreshes = refreshes
        self.dual_direction = restored_direction
        self._last_pcg_reason = None
        _validate_model_batch(self.model, self.batch)


def _validate_checkpoint(
    state: Mapping[str, Any],
    expected_trainer: str,
    expected_config: dict[str, Any],
    *,
    legacy_trainer: str | None = None,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    required = {
        "format_version",
        "trainer",
        "config",
        "model",
        "optimizer",
        "completed_steps",
        "history",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    format_version = state["format_version"]
    if format_version not in {CHECKPOINT_FORMAT_V1, CHECKPOINT_FORMAT_V2}:
        raise ValueError("unsupported checkpoint format_version")
    serialized_trainer = (
        legacy_trainer
        if format_version == CHECKPOINT_FORMAT_V1 and legacy_trainer is not None
        else expected_trainer
    )
    if state["trainer"] != serialized_trainer:
        raise ValueError(f"checkpoint trainer must be {serialized_trainer!r}")
    if state["config"] != expected_config:
        raise ValueError("checkpoint config does not match the trainer config")
    completed_steps = _nonnegative_integer("completed_steps", state["completed_steps"])
    raw_history = state["history"]
    if not isinstance(raw_history, list):
        raise TypeError("checkpoint history must be a list")
    history: list[TrainingStepDiagnostics] = []
    for raw in raw_history:
        if not isinstance(raw, Mapping):
            raise TypeError("every checkpoint history entry must be a mapping")
        history.append(TrainingStepDiagnostics(**dict(raw)))
    if len(history) != completed_steps:
        raise ValueError("history length must equal completed_steps")
    if any(item.step != index for index, item in enumerate(history, start=1)):
        raise ValueError("checkpoint history steps must be consecutive and one-indexed")
    if not isinstance(state["model"], Mapping) or not isinstance(state["optimizer"], Mapping):
        raise TypeError("model and optimizer checkpoint entries must be mappings")
    return {
        "model": copy.deepcopy(state["model"]),
        "optimizer": copy.deepcopy(state["optimizer"]),
        "completed_steps": completed_steps,
        "history": history,
    }


def train_bt_mle(
    model: FrozenFeatureLinearReward,
    batch: FeatureTrainingBatch,
    num_steps: int,
    config: BTMLETrainingConfig | None = None,
) -> BTMLETrainer:
    """Construct, run, and return a repeated-label BT trainer."""

    trainer = BTMLETrainer(model, batch, config)
    trainer.fit(num_steps)
    return trainer


def train_prorm_plus(
    model: FrozenFeatureLinearReward,
    batch: FeatureTrainingBatch,
    num_steps: int,
    config: ProRMPlusTrainingConfig | None = None,
) -> ProRMPlusTrainer:
    """Construct, run, and return a ProRM+ trainer."""

    trainer = ProRMPlusTrainer(model, batch, config)
    trainer.fit(num_steps)
    return trainer


# Public compatibility aliases.  Canonical names above are used by all new
# code and serialized output; these assignments keep pre-ProRM imports valid.
SRMPlusTrainingConfig = ProRMPlusTrainingConfig
SRMEvaluation = ProRMPlusEvaluation
SRMPlusTrainer = ProRMPlusTrainer
evaluate_srm_plus = evaluate_prorm_plus
train_srm_plus = train_prorm_plus


__all__ = [
    "BTMLETrainer",
    "BTMLETrainingConfig",
    "FeatureTrainingBatch",
    "FrozenFeatureLinearReward",
    "ProRMPlusEvaluation",
    "ProRMPlusTrainer",
    "ProRMPlusTrainingConfig",
    "SRMEvaluation",
    "SRMPlusTrainer",
    "SRMPlusTrainingConfig",
    "TrainingStepDiagnostics",
    "evaluate_bt_mle",
    "evaluate_prorm_plus",
    "evaluate_srm_plus",
    "train_bt_mle",
    "train_prorm_plus",
    "train_srm_plus",
]
