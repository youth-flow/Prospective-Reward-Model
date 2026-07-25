"""Fresh train-only reward-head fitting for the common-beta campaign.

This module is the only Phase-2 bridge from transient transformed train-oracle
rewards to reward-head parameters.  It deliberately has no held-out input and
no argument through which an old Phase-1 comparison head can enter.

The primary arm:

1. converts canonical candidate-0-minus-candidate-1 oracle margins to BTL
   probabilities;
2. derives one named, explicit Torch generator from the configured base seed;
3. samples four independent ``gamma=0.9`` randomized-truncation streams;
4. trains fresh zero-initialized BT-MLE on all pooled raw Bernoulli counts and
   fresh zero-initialized ProRM+ on the arithmetic mean of the four ``h``
   estimators; and
5. evaluates both saved heads with a cold-start, full-data first-order audit.

The exact-margin trained-head control and the direct-oracle algebraic identity
are reported separately.  The former can retain reward-class and finite-
optimizer error; the latter bypasses both and tests only moment algebra and the
policy-geometry solve.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from numbers import Real
from typing import Any, Literal

import torch

from .contracts import BT_MLE, CANONICAL_LEARNERS, PRORM_PLUS
from .experiment import TrainingTensorData
from .linear import FisherSolveDType, resolve_fisher_solve_dtype
from .metrics import policy_reward_moment
from .objective import (
    dual_loss,
    dual_saddle_value,
    empirical_moment,
    envelope_surrogate,
    envelope_weights,
)
from .optimization_audit import evaluate_saved_head_optimization
from .phase2_config import (
    PHASE1_MAIN_SEEDS,
    PHASE2_BUDGETED_END_TO_END_SEEDS,
    PHASE2_BUDGETED_END_TO_END_STAGE,
    PHASE2_CONFIRMATORY_EXCLUDED_SEEDS,
    PHASE2_CONFIRMATORY_SEEDS,
    PHASE2_PILOT_SEEDS,
    PHASE2_POST_RECOVERY_CALIBRATION_SCHEMA_VERSION,
    PHASE2_RECOVERY_SCHEMA_VERSION,
    Phase2ConfigBundle,
    phase2_design_identity,
    validate_phase2_config,
)
from .phase2_controls import (
    PRIMARY_GAMMA,
    PRIMARY_NUM_REPLICATES,
    SeededOrthonormalTangentControl,
    TangentCoordinateLayout,
    build_direct_oracle_geometry_control,
    build_exact_margin_canonical_arm,
    sample_canonical_r4_noisy_arm,
    select_seeded_orthonormal_tangent,
)
from .repeated_label_diagnostics import (
    build_repeated_label_tail_diagnostics,
    validate_repeated_label_tail_diagnostics,
)
from .rollout import policy_direction_from_head
from .training import (
    BTMLETrainer,
    BTMLETrainingConfig,
    FeatureTrainingBatch,
    FrozenFeatureLinearReward,
    ProRMPlusTrainer,
    ProRMPlusTrainingConfig,
    TrainingStepDiagnostics,
)

PHASE2_TRAINING_SCHEMA = "phase2-fresh-head-training/v2"
PHASE2_RECOVERY_TRAINING_SCHEMA = "phase2-fresh-head-training/v3"
PRIMARY_TRAINING_ARM = "r4_independent_gamma_0.9"
EXACT_SOFT_BT_ARM = "exact_soft_label_bt_secondary_diagnostic"
EXACT_SOFT_BT_ROLE = "noise_free_positive_control_and_secondary_misspecification_diagnostic"
EXACT_SOFT_BT_INPUT = "sigmoid_of_train_transformed_oracle_margin"
LABEL_RNG_NAMESPACE = "prorm-common-beta-r4-labels-v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_RECOVERY_TIE_BREAK = "exact_zero_initialized_deterministic_adamw_lr_decay_path"
_LEGACY_TIE_BREAK = "zero_initialized_adamw_implicit_bias"


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _validate_seed(value: object, *, name: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be an integer in [0, 2**63 - 1]")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strictly_greater: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = result <= minimum if strictly_greater else result < minimum
        if invalid:
            operator = ">" if strictly_greater else ">="
            raise ValueError(f"{name} must be {operator} {minimum}")
    return result


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    tensor = value.detach().to(device="cpu").contiguous().clone()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def _strict_json_copy(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain strict JSON data") from error
    if not isinstance(decoded, dict):
        raise RuntimeError("internal JSON mapping round-trip failed")
    return decoded


def _validate_frozen_oracle_rewards(
    training: TrainingTensorData,
    rewards: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(rewards, torch.Tensor):
        raise TypeError("train_oracle_rewards must be a torch.Tensor")
    expected_shape = (training.num_prompts, training.num_candidates)
    if rewards.shape != expected_shape:
        raise ValueError(f"train_oracle_rewards must have shape {expected_shape}")
    if not rewards.is_floating_point():
        raise TypeError("train_oracle_rewards must have a floating-point dtype")
    if (
        rewards.dtype != training.policy_scores.dtype
        or rewards.device != training.policy_scores.device
    ):
        raise ValueError(
            "train_oracle_rewards must share dtype and device with training.policy_scores"
        )
    if rewards.requires_grad or rewards.grad_fn is not None:
        raise ValueError("train_oracle_rewards must be frozen and detached")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("train_oracle_rewards must be finite")
    return rewards


@dataclass(frozen=True, slots=True)
class LearningRateStage:
    """One inclusive, one-indexed deterministic optimizer-update interval."""

    first_update: int
    last_update: int
    learning_rate: float

    def __post_init__(self) -> None:
        _positive_integer(self.first_update, name="first_update")
        _positive_integer(self.last_update, name="last_update")
        if self.last_update < self.first_update:
            raise ValueError("last_update must not precede first_update")
        _finite_float(
            self.learning_rate,
            name="learning_rate",
            minimum=0.0,
            strictly_greater=True,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "first_update": self.first_update,
            "last_update": self.last_update,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True, slots=True)
class AdamWRecoveryProtocol:
    """Hash-bound deterministic AdamW decay path and its adoption mode.

    The default ``recovery`` mode preserves the original one-shot recovery
    serialization.  ``adopted`` uses the identical numerical path but records
    that a fresh Phase-2 design inherited only the schedule through an
    immutable recovery-success authorization.
    """

    stages: tuple[LearningRateStage, ...]
    schedule_sha256: str
    mode: Literal["recovery", "adopted"] = "recovery"
    source_recovery_authorization_sha256: str | None = None
    legacy_boundary_snapshot_steps: int = 5760
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    amsgrad: bool = False
    maximize: bool = False
    foreach: bool = False
    fused: bool = False
    capturable: bool = False
    differentiable: bool = False
    reward_head_dtype: str = "float32"
    first_order_audit_dtype: str = "float64"
    microbatch_order: str = "canonical_edge_order_contiguous_ascending_no_shuffle"
    optimizer_state_reset_at_lr_milestone: bool = False
    one_optimizer_update_per_step: bool = True
    tie_break: str = _RECOVERY_TIE_BREAK

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ValueError("recovery learning-rate stages must be a non-empty tuple")
        if not all(isinstance(stage, LearningRateStage) for stage in self.stages):
            raise TypeError("recovery learning-rate stages must contain LearningRateStage values")
        expected_first = 1
        for stage in self.stages:
            if stage.first_update != expected_first:
                raise ValueError("recovery learning-rate stages must be contiguous from update 1")
            expected_first = stage.last_update + 1
        _validate_digest(self.schedule_sha256, name="schedule_sha256")
        schedule_payload = {
            "update_indexing": "one_indexed_inclusive",
            "application": "set_learning_rate_immediately_before_optimizer_update",
            "stages": [stage.to_dict() for stage in self.stages],
        }
        if self.schedule_sha256 != _canonical_sha256(schedule_payload):
            raise ValueError("schedule_sha256 does not bind the declared recovery schedule")
        if self.mode not in {"recovery", "adopted"}:
            raise ValueError("optimizer decay protocol mode must be recovery or adopted")
        if self.mode == "recovery":
            if self.source_recovery_authorization_sha256 is not None:
                raise ValueError(
                    "recovery optimizer protocol must not bind a success authorization"
                )
        else:
            if self.source_recovery_authorization_sha256 is None:
                raise ValueError(
                    "adopted optimizer protocol requires a recovery-success authorization"
                )
            _validate_digest(
                self.source_recovery_authorization_sha256,
                name="source_recovery_authorization_sha256",
            )
        if self.legacy_boundary_snapshot_steps != 5760:
            raise ValueError("legacy_boundary_snapshot_steps must equal 5760")
        if self.stages[0].last_update != self.legacy_boundary_snapshot_steps:
            raise ValueError(
                "the first recovery schedule stage must end at the legacy 5760-step boundary"
            )
        if self.betas != (0.9, 0.999):
            raise ValueError("recovery AdamW betas must equal (0.9, 0.999)")
        if self.eps != 1.0e-8:
            raise ValueError("recovery AdamW eps must equal 1e-8")
        for name in (
            "amsgrad",
            "maximize",
            "foreach",
            "fused",
            "capturable",
            "differentiable",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"recovery AdamW {name} must be false")
        if self.reward_head_dtype != "float32":
            raise ValueError("recovery reward-head dtype must equal float32")
        if self.first_order_audit_dtype != "float64":
            raise ValueError("recovery first-order audit dtype must equal float64")
        if self.microbatch_order != ("canonical_edge_order_contiguous_ascending_no_shuffle"):
            raise ValueError("recovery microbatch order does not match the locked protocol")
        if self.optimizer_state_reset_at_lr_milestone is not False:
            raise ValueError("recovery optimizer state must not reset at learning-rate milestones")
        if self.one_optimizer_update_per_step is not True:
            raise ValueError("recovery protocol requires one optimizer update per step")
        if self.tie_break != _RECOVERY_TIE_BREAK:
            raise ValueError("recovery tie-break does not match the locked protocol")

    @property
    def maximum_update(self) -> int:
        return self.stages[-1].last_update

    def learning_rate_for_update(self, update: int) -> float:
        _positive_integer(update, name="update")
        for stage in self.stages:
            if stage.first_update <= update <= stage.last_update:
                return stage.learning_rate
        raise ValueError(f"update {update} is outside the locked recovery schedule")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "scope": "every_phase2_first_order_convergence_trainer",
            "initialization": "exact_zero_head_and_fresh_optimizer_state",
            "learning_rate_schedule": {
                "update_indexing": "one_indexed_inclusive",
                "application": ("set_learning_rate_immediately_before_optimizer_update"),
                "stages": [stage.to_dict() for stage in self.stages],
                "schedule_sha256": self.schedule_sha256,
            },
            "legacy_constant_lr_boundary_snapshot_steps": (self.legacy_boundary_snapshot_steps),
            "state_transition": ("preserve_all_adamw_moments_across_learning_rate_boundaries"),
            "adamw": {
                "betas": list(self.betas),
                "eps": self.eps,
                "amsgrad": self.amsgrad,
                "maximize": self.maximize,
                "foreach": self.foreach,
                "fused": self.fused,
                "capturable": self.capturable,
                "differentiable": self.differentiable,
            },
            "reward_head_dtype": self.reward_head_dtype,
            "first_order_audit_dtype": self.first_order_audit_dtype,
            "microbatch_order": self.microbatch_order,
            "optimizer_state_reset_at_lr_milestone": (self.optimizer_state_reset_at_lr_milestone),
            "one_optimizer_update_per_step": self.one_optimizer_update_per_step,
            "tie_break": self.tie_break,
            "validation_or_test_selection": False,
        }
        if self.mode == "recovery":
            return {
                "schema_version": "deterministic-adamw-lr-decay-recovery/v1",
                "one_time_recovery": True,
                **result,
            }
        return {
            "schema_version": "deterministic-adamw-lr-decay/v1",
            "role": "frozen_post_recovery_phase2_optimizer",
            "source_recovery_authorization_sha256": (self.source_recovery_authorization_sha256),
            **result,
        }


@dataclass(frozen=True, slots=True)
class FirstOrderConvergenceSpec:
    """Objective-specific, train-only first-order stopping rule.

    The primary head is the first iterate satisfying the full-data,
    post-update, unclipped gradient-ratio gate for ``consecutive_checks``
    consecutive scheduled checks.  The fixed 720-step iterate is retained
    independently as a compute-matched diagnostic and is never a selection
    rule.
    """

    gradient_ratio_tolerance: float = 1.0e-3
    min_steps: int = 100
    max_steps: int = 5760
    check_interval: int = 20
    consecutive_checks: int = 3
    gradient_norm_denominator_floor: float = 1.0e-30
    fail_closed: bool = True
    optimizer_protocol: AdamWRecoveryProtocol | None = None

    def __post_init__(self) -> None:
        _finite_float(
            self.gradient_ratio_tolerance,
            name="gradient_ratio_tolerance",
            minimum=0.0,
            strictly_greater=True,
        )
        _positive_integer(self.min_steps, name="min_steps")
        _positive_integer(self.max_steps, name="max_steps")
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps must not exceed max_steps")
        _positive_integer(self.check_interval, name="check_interval")
        _positive_integer(self.consecutive_checks, name="consecutive_checks")
        if self.min_steps % self.check_interval != 0:
            raise ValueError("min_steps must be an exact scheduled check")
        if self.max_steps % self.check_interval != 0:
            raise ValueError("max_steps must be an exact scheduled check")
        _finite_float(
            self.gradient_norm_denominator_floor,
            name="gradient_norm_denominator_floor",
            minimum=0.0,
            strictly_greater=True,
        )
        if self.fail_closed is not True:
            raise ValueError("first-order convergence must fail closed")
        if self.optimizer_protocol is not None:
            if not isinstance(self.optimizer_protocol, AdamWRecoveryProtocol):
                raise TypeError("optimizer_protocol must be an AdamWRecoveryProtocol")
            if self.max_steps != self.optimizer_protocol.maximum_update:
                raise ValueError(
                    "recovery convergence max_steps must equal the schedule maximum update"
                )
            if self.optimizer_protocol.legacy_boundary_snapshot_steps % self.check_interval != 0:
                raise ValueError("legacy boundary snapshot must be a scheduled convergence check")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": (
                "objective-first-order-convergence-spec/v2"
                if self.optimizer_protocol is not None
                else "objective-first-order-convergence-spec/v1"
            ),
            "gradient_ratio_tolerance": self.gradient_ratio_tolerance,
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "check_interval": self.check_interval,
            "consecutive_checks": self.consecutive_checks,
            "gradient_norm_denominator_floor": (self.gradient_norm_denominator_floor),
            "fail_closed": self.fail_closed,
            "gradient": "full_data_post_update_unclipped",
            "denominator": "exact_zero_initialization_gradient_l2_norm",
            "validation_or_test_selection": False,
        }
        if self.optimizer_protocol is not None:
            result["optimizer_protocol"] = self.optimizer_protocol.to_dict()
        return result


class OptimizationConvergenceError(RuntimeError):
    """Fail-closed error carrying strict-JSON first-order evidence."""

    def __init__(self, message: str, *, evidence: Mapping[str, object]) -> None:
        super().__init__(message)
        self.evidence = _strict_json_copy(evidence, name="convergence_failure_evidence")


@dataclass(frozen=True, slots=True)
class Phase2TrainingSettings:
    """Fully compiled, identity-bound settings for fresh Phase-2 heads."""

    phase2_config_hash: str
    source_config_hash: str
    stage: Literal["pilot", "budgeted_end_to_end", "confirmatory"]
    formal_eligibility: bool
    seeds: tuple[int, ...]
    outer_steps: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    microbatch_size: int
    max_grad_norm: float
    training_beta: float
    relative_damping: float
    pcg_dtype: FisherSolveDType
    pcg_max_iterations: int
    pcg_tolerance: float
    pcg_absolute_tolerance: float = 0.0
    pcg_residual_recompute_interval: int = 20
    require_pcg_convergence: bool = True
    num_label_replicates: int = PRIMARY_NUM_REPLICATES
    annotation_gamma: float = PRIMARY_GAMMA
    probability_floor: float = 0.25
    label_rng_namespace: str = LABEL_RNG_NAMESPACE
    max_total_annotations: int | None = None
    low_dimensional_enabled: bool = True
    low_dimensional_selected_dimension: int = 256
    low_dimensional_namespace: str = "prorm-common-beta-low-dimensional-tangent-v1"
    low_dimensional_regularization: str = "moore_penrose_pseudoinverse"
    low_dimensional_relative_eigenvalue_tolerance: float = 1.0e-10
    low_dimensional_source_layout_id: str = "training-policy-score-flatten-order/v1"
    exact_soft_label_bt_enabled: bool = True
    exact_soft_label_bt_role: str = EXACT_SOFT_BT_ROLE
    exact_soft_label_bt_noise_free: bool = True
    exact_soft_label_bt_input: str = EXACT_SOFT_BT_INPUT
    exact_soft_label_bt_eligible_for_primary_claim: bool = False
    convergence: FirstOrderConvergenceSpec = field(default_factory=FirstOrderConvergenceSpec)
    identifiability_relative_rank_tolerance: float = 1.0e-10
    identifiability_role: str = "pilot_measure_only"
    identifiability_require_full_column_rank: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.phase2_config_hash, name="phase2_config_hash")
        _validate_digest(self.source_config_hash, name="source_config_hash")
        if self.stage not in {
            "pilot",
            PHASE2_BUDGETED_END_TO_END_STAGE,
            "confirmatory",
        }:
            raise ValueError("stage must be 'pilot', 'budgeted_end_to_end', or 'confirmatory'")
        if not isinstance(self.formal_eligibility, bool):
            raise TypeError("formal_eligibility must be bool")
        if not isinstance(self.seeds, tuple) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a tuple of unique Phase-2 seeds")
        for seed in self.seeds:
            _validate_seed(seed, name="seeds item")
        if self.stage == "pilot":
            if set(self.seeds) != set(PHASE2_PILOT_SEEDS):
                raise ValueError(
                    "pilot seeds must equal the permanently excluded Phase-2 pilot set"
                )
            if self.formal_eligibility:
                raise ValueError("pilot training settings cannot be formally eligible")
        elif self.stage == PHASE2_BUDGETED_END_TO_END_STAGE:
            if self.seeds != PHASE2_BUDGETED_END_TO_END_SEEDS:
                raise ValueError(
                    "budgeted_end_to_end training settings require the exact ordered "
                    f"seed list {list(PHASE2_BUDGETED_END_TO_END_SEEDS)!r}"
                )
            disallowed = (
                PHASE1_MAIN_SEEDS | PHASE2_PILOT_SEEDS | frozenset(PHASE2_CONFIRMATORY_SEEDS)
            )
            overlap = set(self.seeds).intersection(disallowed)
            if overlap:
                raise ValueError(
                    "budgeted_end_to_end seeds overlap Phase-1, pilot, or "
                    f"confirmatory seeds: {sorted(overlap)!r}"
                )
            if self.formal_eligibility:
                raise ValueError(
                    "budgeted_end_to_end training settings cannot be formally eligible"
                )
        elif self.stage == "confirmatory":
            if self.seeds != PHASE2_CONFIRMATORY_SEEDS:
                raise ValueError(
                    "confirmatory training settings require the exact preregistered "
                    f"ordered seed list {list(PHASE2_CONFIRMATORY_SEEDS)!r}"
                )
            overlap = set(self.seeds).intersection(PHASE2_CONFIRMATORY_EXCLUDED_SEEDS)
            if overlap:
                raise ValueError(
                    f"confirmatory seeds overlap Phase-1 or pilot seeds: {sorted(overlap)!r}"
                )
            if not self.formal_eligibility:
                raise ValueError("confirmatory training settings must be formally eligible")
        if self.outer_steps != 720:
            raise ValueError(
                "Phase-2 compute-matched snapshot is locked to exactly 720 optimizer steps"
            )
        _finite_float(self.learning_rate, name="learning_rate", minimum=0.0, strictly_greater=True)
        if self.optimizer != "adamw":
            raise ValueError("Phase-2 head training requires optimizer='adamw'")
        if _finite_float(self.weight_decay, name="weight_decay", minimum=0.0) != 0.0:
            raise ValueError("Phase-2 head training requires zero weight decay")
        _positive_integer(self.microbatch_size, name="microbatch_size")
        _finite_float(self.max_grad_norm, name="max_grad_norm", minimum=0.0, strictly_greater=True)
        if (
            _finite_float(
                self.training_beta,
                name="training_beta",
                minimum=0.0,
                strictly_greater=True,
            )
            != 1.0
        ):
            raise ValueError("the empirical ProRM+ training loss is locked to beta=1")
        if (
            _finite_float(
                self.relative_damping,
                name="relative_damping",
                minimum=0.0,
                strictly_greater=True,
            )
            != 0.001
        ):
            raise ValueError("the primary full-tangent ridge coefficient must equal 0.001")
        resolve_fisher_solve_dtype(self.pcg_dtype)
        _positive_integer(self.pcg_max_iterations, name="pcg_max_iterations")
        _finite_float(
            self.pcg_tolerance,
            name="pcg_tolerance",
            minimum=0.0,
            strictly_greater=True,
        )
        _finite_float(
            self.pcg_absolute_tolerance,
            name="pcg_absolute_tolerance",
            minimum=0.0,
        )
        _positive_integer(
            self.pcg_residual_recompute_interval,
            name="pcg_residual_recompute_interval",
        )
        if self.require_pcg_convergence is not True:
            raise ValueError("Phase-2 ProRM+ and oracle solves must fail on PCG non-convergence")
        if self.num_label_replicates != PRIMARY_NUM_REPLICATES:
            raise ValueError("the primary noisy arm requires exactly four label replicates")
        if self.annotation_gamma != PRIMARY_GAMMA:
            raise ValueError("the primary noisy arm requires gamma=0.9")
        if self.probability_floor != 0.25:
            raise ValueError("the transformed-oracle probability floor must equal 0.25")
        if self.label_rng_namespace != LABEL_RNG_NAMESPACE:
            raise ValueError(f"label_rng_namespace must equal {LABEL_RNG_NAMESPACE!r}")
        if self.max_total_annotations is not None:
            _positive_integer(self.max_total_annotations, name="max_total_annotations")
        if self.low_dimensional_enabled is not True:
            raise ValueError("the configured low-dimensional positive control must be enabled")
        _positive_integer(
            self.low_dimensional_selected_dimension,
            name="low_dimensional_selected_dimension",
        )
        if self.low_dimensional_namespace != ("prorm-common-beta-low-dimensional-tangent-v1"):
            raise ValueError("low_dimensional_namespace does not match the Phase-2 config")
        if self.low_dimensional_regularization != "moore_penrose_pseudoinverse":
            raise ValueError(
                "low-dimensional control requires Moore-Penrose pseudoinverse regularization"
            )
        if (
            _finite_float(
                self.low_dimensional_relative_eigenvalue_tolerance,
                name="low_dimensional_relative_eigenvalue_tolerance",
                minimum=0.0,
                strictly_greater=True,
            )
            != 1.0e-10
        ):
            raise ValueError("low-dimensional relative eigenvalue tolerance must equal 1e-10")
        if (
            not isinstance(self.low_dimensional_source_layout_id, str)
            or not self.low_dimensional_source_layout_id
        ):
            raise ValueError("low_dimensional_source_layout_id must be non-empty")
        if self.exact_soft_label_bt_enabled is not True:
            raise ValueError("the exact soft-label BT diagnostic must be enabled")
        if self.exact_soft_label_bt_role != EXACT_SOFT_BT_ROLE:
            raise ValueError("exact_soft_label_bt_role does not match the Phase-2 config")
        if self.exact_soft_label_bt_noise_free is not True:
            raise ValueError("the exact soft-label BT diagnostic must be noise-free")
        if self.exact_soft_label_bt_input != EXACT_SOFT_BT_INPUT:
            raise ValueError("exact_soft_label_bt_input does not match the Phase-2 config")
        if self.exact_soft_label_bt_eligible_for_primary_claim is not False:
            raise ValueError(
                "the exact soft-label BT diagnostic cannot be eligible for the primary claim"
            )
        if not isinstance(self.convergence, FirstOrderConvergenceSpec):
            raise TypeError("convergence must be FirstOrderConvergenceSpec")
        if self.convergence.max_steps < self.outer_steps:
            raise ValueError(
                "convergence.max_steps must reach the fixed 720-step compute-matched snapshot"
            )
        _finite_float(
            self.identifiability_relative_rank_tolerance,
            name="identifiability_relative_rank_tolerance",
            minimum=0.0,
            strictly_greater=True,
        )
        if not isinstance(self.identifiability_role, str) or not self.identifiability_role:
            raise ValueError("identifiability_role must be a non-empty string")
        if not isinstance(self.identifiability_require_full_column_rank, bool):
            raise TypeError("identifiability_require_full_column_rank must be bool")
        if self.stage == "pilot":
            if self.identifiability_role != "pilot_measure_only":
                raise ValueError("pilot rank evidence must be measure-only")
            if self.identifiability_require_full_column_rank:
                raise ValueError("pilot rank evidence cannot be a full-rank acceptance gate")
        elif self.stage == PHASE2_BUDGETED_END_TO_END_STAGE:
            if (
                self.identifiability_role
                != "budgeted_end_to_end_exploratory_frozen_identifiability_audit"
            ):
                raise ValueError(
                    "budgeted_end_to_end rank evidence must use its independent "
                    "exploratory measure-only role"
                )
            if self.identifiability_require_full_column_rank:
                raise ValueError(
                    "budgeted_end_to_end rank evidence cannot be a formal full-rank acceptance gate"
                )
        elif self.stage == "confirmatory":
            if self.identifiability_role != ("confirmatory_frozen_identifiability_contract"):
                raise ValueError(
                    "confirmatory rank evidence must use the frozen identifiability contract"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": (
                "phase2-training-settings/v3"
                if self.convergence.optimizer_protocol is not None
                else "phase2-training-settings/v2"
            ),
            "phase2_config_hash": self.phase2_config_hash,
            "source_config_hash": self.source_config_hash,
            "stage": self.stage,
            "formal_eligibility": self.formal_eligibility,
            "seeds": list(self.seeds),
            "outer_steps": self.outer_steps,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "microbatch_size": self.microbatch_size,
            "max_grad_norm": self.max_grad_norm,
            "training_beta": self.training_beta,
            "relative_damping": self.relative_damping,
            "pcg_dtype": self.pcg_dtype,
            "pcg_max_iterations": self.pcg_max_iterations,
            "pcg_tolerance": self.pcg_tolerance,
            "pcg_absolute_tolerance": self.pcg_absolute_tolerance,
            "pcg_residual_recompute_interval": self.pcg_residual_recompute_interval,
            "require_pcg_convergence": self.require_pcg_convergence,
            "num_label_replicates": self.num_label_replicates,
            "annotation_gamma": self.annotation_gamma,
            "probability_floor": self.probability_floor,
            "label_rng_namespace": self.label_rng_namespace,
            "max_total_annotations": self.max_total_annotations,
            "low_dimensional_enabled": self.low_dimensional_enabled,
            "low_dimensional_selected_dimension": (self.low_dimensional_selected_dimension),
            "low_dimensional_namespace": self.low_dimensional_namespace,
            "low_dimensional_regularization": self.low_dimensional_regularization,
            "low_dimensional_relative_eigenvalue_tolerance": (
                self.low_dimensional_relative_eigenvalue_tolerance
            ),
            "low_dimensional_source_layout_id": self.low_dimensional_source_layout_id,
            "exact_soft_label_bt_enabled": self.exact_soft_label_bt_enabled,
            "exact_soft_label_bt_role": self.exact_soft_label_bt_role,
            "exact_soft_label_bt_noise_free": self.exact_soft_label_bt_noise_free,
            "exact_soft_label_bt_input": self.exact_soft_label_bt_input,
            "exact_soft_label_bt_eligible_for_primary_claim": (
                self.exact_soft_label_bt_eligible_for_primary_claim
            ),
            "convergence": self.convergence.to_dict(),
            "identifiability_relative_rank_tolerance": (
                self.identifiability_relative_rank_tolerance
            ),
            "identifiability_role": self.identifiability_role,
            "identifiability_require_full_column_rank": (
                self.identifiability_require_full_column_rank
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def _settings_from_overlay(
    overlay: Mapping[str, object],
    *,
    base_config: Mapping[str, object] | None = None,
) -> Phase2TrainingSettings:
    normalized = validate_phase2_config(overlay, base_config=base_config)
    reward = normalized["reward_model"]
    objective = normalized["objective"]
    ridge = objective["full_tangent"]["ridge"]
    annotations = normalized["annotations"]
    oracle = normalized["oracle"]
    low_dimensional = normalized["positive_controls"]["low_dimensional_tangent"]
    exact_soft_bt = normalized["positive_controls"]["exact_soft_label_bt"]
    convergence = reward["adaptive_convergence"]
    identifiability = reward["identifiability"]
    optimizer_protocol: AdamWRecoveryProtocol | None = None
    if normalized["schema_version"] in {
        PHASE2_RECOVERY_SCHEMA_VERSION,
        PHASE2_POST_RECOVERY_CALIBRATION_SCHEMA_VERSION,
    }:
        declared_protocol = reward["optimizer_protocol"]
        declared_schedule = declared_protocol["learning_rate_schedule"]
        optimizer_protocol = AdamWRecoveryProtocol(
            stages=tuple(
                LearningRateStage(
                    first_update=int(stage["first_update"]),
                    last_update=int(stage["last_update"]),
                    learning_rate=float(stage["learning_rate"]),
                )
                for stage in declared_schedule["stages"]
            ),
            schedule_sha256=str(declared_schedule["schedule_sha256"]),
            mode=(
                "recovery"
                if normalized["schema_version"] == PHASE2_RECOVERY_SCHEMA_VERSION
                else "adopted"
            ),
            source_recovery_authorization_sha256=(
                None
                if normalized["schema_version"] == PHASE2_RECOVERY_SCHEMA_VERSION
                else str(normalized["recovery_success_reference"]["artifact_sha256"])
            ),
            legacy_boundary_snapshot_steps=int(
                declared_protocol["legacy_constant_lr_boundary_snapshot_steps"]
            ),
            betas=tuple(float(value) for value in declared_protocol["adamw"]["betas"]),
            eps=float(declared_protocol["adamw"]["eps"]),
            amsgrad=bool(declared_protocol["adamw"]["amsgrad"]),
            maximize=bool(declared_protocol["adamw"]["maximize"]),
            foreach=bool(declared_protocol["adamw"]["foreach"]),
            fused=bool(declared_protocol["adamw"]["fused"]),
            capturable=bool(declared_protocol["adamw"]["capturable"]),
            differentiable=bool(declared_protocol["adamw"]["differentiable"]),
            reward_head_dtype=str(declared_protocol["reward_head_dtype"]),
            first_order_audit_dtype=str(declared_protocol["first_order_audit_dtype"]),
            microbatch_order=str(declared_protocol["microbatch_order"]),
            optimizer_state_reset_at_lr_milestone=bool(
                declared_protocol["optimizer_state_reset_at_lr_milestone"]
            ),
            one_optimizer_update_per_step=bool(declared_protocol["one_optimizer_update_per_step"]),
            tie_break=str(declared_protocol["tie_break"]),
        )
    return Phase2TrainingSettings(
        phase2_config_hash=phase2_design_identity(normalized),
        source_config_hash=normalized["design"]["source_config_hash"],
        stage=str(normalized["design"]["stage"]),
        formal_eligibility=bool(normalized["design"]["formal_eligibility"]),
        seeds=tuple(int(seed) for seed in normalized["run"]["seeds"]),
        outer_steps=int(reward["outer_steps"]),
        learning_rate=float(reward["learning_rate"]),
        optimizer=str(reward["optimizer"]),
        weight_decay=float(reward["weight_decay"]),
        microbatch_size=int(reward["microbatch_size"]),
        max_grad_norm=float(reward["max_grad_norm"]),
        training_beta=1.0,
        relative_damping=float(ridge["relative_coefficient"]),
        pcg_dtype=str(ridge["solver_dtype"]),
        pcg_max_iterations=int(ridge["pcg_max_iterations"]),
        pcg_tolerance=float(ridge["pcg_tolerance"]),
        num_label_replicates=int(annotations["independent_replicates_per_edge"]),
        annotation_gamma=float(annotations["gamma"]),
        probability_floor=float(oracle["probability_floor"]),
        low_dimensional_enabled=bool(low_dimensional["enabled"]),
        low_dimensional_selected_dimension=int(low_dimensional["dimension"]),
        low_dimensional_namespace=str(low_dimensional["seed_namespace"]),
        low_dimensional_regularization=str(low_dimensional["regularization"]),
        low_dimensional_relative_eigenvalue_tolerance=float(
            low_dimensional["relative_eigenvalue_tolerance"]
        ),
        exact_soft_label_bt_enabled=bool(exact_soft_bt["enabled"]),
        exact_soft_label_bt_role=str(exact_soft_bt["role"]),
        exact_soft_label_bt_noise_free=bool(exact_soft_bt["noise_free"]),
        exact_soft_label_bt_input=str(exact_soft_bt["input"]),
        exact_soft_label_bt_eligible_for_primary_claim=bool(
            exact_soft_bt["eligible_for_primary_claim"]
        ),
        convergence=FirstOrderConvergenceSpec(
            gradient_ratio_tolerance=float(convergence["relative_gradient_ratio_tolerance"]),
            min_steps=int(convergence["minimum_steps"]),
            max_steps=int(convergence["maximum_steps"]),
            check_interval=int(convergence["check_interval_steps"]),
            consecutive_checks=int(convergence["consecutive_passing_checks"]),
            gradient_norm_denominator_floor=float(convergence["denominator_floor"]),
            fail_closed=bool(convergence["fail_closed"]),
            optimizer_protocol=optimizer_protocol,
        ),
        identifiability_relative_rank_tolerance=float(identifiability["relative_rank_tolerance"]),
        identifiability_role=str(identifiability["role"]),
        identifiability_require_full_column_rank=bool(identifiability["require_full_column_rank"]),
    )


def compile_phase2_training_settings(
    settings: Phase2TrainingSettings | Phase2ConfigBundle | Mapping[str, object],
) -> Phase2TrainingSettings:
    """Compile a dataclass, overlay mapping, or explicit overlay/base bundle."""

    if isinstance(settings, Phase2TrainingSettings):
        return settings
    if isinstance(settings, Phase2ConfigBundle):
        return _settings_from_overlay(settings.config, base_config=settings.base_config)
    if not isinstance(settings, Mapping):
        raise TypeError(
            "settings must be Phase2TrainingSettings, Phase2ConfigBundle, "
            "or a Phase-2 configuration mapping"
        )
    if "config" in settings or "base_config" in settings:
        if set(settings) != {"config", "base_config"}:
            raise ValueError("settings bundle mapping must contain exactly config and base_config")
        overlay = settings["config"]
        base = settings["base_config"]
        if not isinstance(overlay, Mapping) or not isinstance(base, Mapping):
            raise TypeError("settings bundle config and base_config must be mappings")
        return _settings_from_overlay(overlay, base_config=base)
    return _settings_from_overlay(settings)


@dataclass(frozen=True, slots=True)
class LabelStreamEvidence:
    """Non-invertible identity, RNG, routing, and cost evidence for R=4 labels."""

    namespace: str
    base_seed: int
    derived_seed: int
    derivation_sha256: str
    generator_device: str
    initial_state_sha256: str
    final_state_sha256: str
    oracle_reward_sha256: str
    canonical_probability_sha256: str
    replicate_count_sha256: str
    replicate_win_sha256: str
    replicate_h_sha256: str
    mean_h_sha256: str
    label_stream_sha256: str
    repeated_label_tail_diagnostics: Mapping[str, object]
    realized_total_annotations: int
    realized_annotations_per_edge: float
    expected_annotations_per_edge: float
    num_edges: int
    num_replicates: Literal[4] = 4
    gamma: Literal[0.9] = 0.9
    bt_target: Literal["pooled_raw_wins_and_totals"] = "pooled_raw_wins_and_totals"
    prorm_target: Literal["mean_of_per_replicate_h"] = "mean_of_per_replicate_h"
    raw_labels_retained: Literal[False] = False
    raw_node_rewards_retained: Literal[False] = False

    def __post_init__(self) -> None:
        if self.namespace != LABEL_RNG_NAMESPACE:
            raise ValueError("label stream namespace is not the locked named RNG stream")
        _validate_seed(self.base_seed, name="base_seed")
        _validate_seed(self.derived_seed, name="derived_seed")
        for name in (
            "derivation_sha256",
            "initial_state_sha256",
            "final_state_sha256",
            "oracle_reward_sha256",
            "canonical_probability_sha256",
            "replicate_count_sha256",
            "replicate_win_sha256",
            "replicate_h_sha256",
            "mean_h_sha256",
            "label_stream_sha256",
        ):
            _validate_digest(getattr(self, name), name=name)
        if not isinstance(self.generator_device, str) or not self.generator_device:
            raise ValueError("generator_device must be a non-empty string")
        _positive_integer(self.realized_total_annotations, name="realized_total_annotations")
        _positive_integer(self.num_edges, name="num_edges")
        for name in ("realized_annotations_per_edge", "expected_annotations_per_edge"):
            _finite_float(
                getattr(self, name),
                name=name,
                minimum=0.0,
                strictly_greater=True,
            )
        if self.num_replicates != 4 or self.gamma != 0.9:
            raise ValueError("label evidence must describe the locked R=4, gamma=0.9 arm")
        normalized_tail_diagnostics = validate_repeated_label_tail_diagnostics(
            self.repeated_label_tail_diagnostics,
            expected_num_edges=self.num_edges,
            replicate_count_sha256=self.replicate_count_sha256,
            replicate_h_sha256=self.replicate_h_sha256,
            mean_h_sha256=self.mean_h_sha256,
            name="repeated_label_tail_diagnostics",
        )
        object.__setattr__(
            self,
            "repeated_label_tail_diagnostics",
            normalized_tail_diagnostics,
        )
        if (
            self.bt_target != "pooled_raw_wins_and_totals"
            or self.prorm_target != "mean_of_per_replicate_h"
            or self.raw_labels_retained is not False
            or self.raw_node_rewards_retained is not False
        ):
            raise ValueError("invalid label routing or retention evidence")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainedHeadEvidence:
    """Immutable primary head plus convergence and compute-matched evidence."""

    arm: str
    method: str
    head_weight: tuple[float, ...]
    head_dtype: str
    initial_head_sha256: str
    head_sha256: str
    initial_objective: float
    final_objective: float
    history_summary: Mapping[str, object]
    final_pcg: Mapping[str, object] | None
    first_order_convergence: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.arm, str) or not self.arm:
            raise ValueError("arm must be a non-empty string")
        if self.method not in CANONICAL_LEARNERS:
            raise ValueError(f"method must be one of {CANONICAL_LEARNERS!r}")
        if (
            not isinstance(self.head_weight, tuple)
            or not self.head_weight
            or not all(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in self.head_weight
            )
        ):
            raise ValueError("head_weight must be a finite non-empty tuple")
        if not isinstance(self.head_dtype, str) or not self.head_dtype:
            raise ValueError("head_dtype must be a non-empty string")
        _validate_digest(self.initial_head_sha256, name="initial_head_sha256")
        _validate_digest(self.head_sha256, name="head_sha256")
        for name in ("initial_objective", "final_objective"):
            value = _finite_float(getattr(self, name), name=name)
            if value < -1.0e-10:
                raise ValueError(f"{name} cannot be materially negative")
        _strict_json_copy(self.history_summary, name="history_summary")
        if self.final_pcg is not None:
            _strict_json_copy(self.final_pcg, name="final_pcg")
        convergence = _strict_json_copy(
            self.first_order_convergence,
            name="first_order_convergence",
        )
        if convergence.get("converged") is not True:
            raise ValueError("serialized primary head must pass first-order convergence")

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "method": self.method,
            "head_weight": list(self.head_weight),
            "head_dtype": self.head_dtype,
            "initial_head_sha256": self.initial_head_sha256,
            "head_sha256": self.head_sha256,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "history_summary": _strict_json_copy(
                self.history_summary,
                name="history_summary",
            ),
            "final_pcg": (
                None
                if self.final_pcg is None
                else _strict_json_copy(self.final_pcg, name="final_pcg")
            ),
            "first_order_convergence": _strict_json_copy(
                self.first_order_convergence,
                name="first_order_convergence",
            ),
        }


@dataclass(frozen=True, slots=True)
class ExactMarginTrainingControl:
    """Trained exact-margin head, distinct from the algebraic oracle control."""

    head: TrainedHeadEvidence
    target_audit: Mapping[str, object]
    optimization_audit: Mapping[str, object]
    reward_class_and_optimizer_gap: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.head.arm != "exact_margin_positive_control" or self.head.method != PRORM_PLUS:
            raise ValueError("exact-margin control must contain a ProRM+ control head")
        for name in (
            "target_audit",
            "optimization_audit",
            "reward_class_and_optimizer_gap",
        ):
            _strict_json_copy(getattr(self, name), name=name)

    def to_dict(self) -> dict[str, object]:
        return {
            "head": self.head.to_dict(),
            "target_audit": _strict_json_copy(self.target_audit, name="target_audit"),
            "optimization_audit": _strict_json_copy(
                self.optimization_audit,
                name="optimization_audit",
            ),
            "reward_class_and_optimizer_gap": _strict_json_copy(
                self.reward_class_and_optimizer_gap,
                name="reward_class_and_optimizer_gap",
            ),
        }


@dataclass(frozen=True, slots=True)
class ExactSoftLabelBTControl:
    """Noise-free expected-BT diagnostic, never a deployable learner arm."""

    head: TrainedHeadEvidence
    target_audit: Mapping[str, object]
    optimization_audit: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.head.arm != EXACT_SOFT_BT_ARM or self.head.method != BT_MLE:
            raise ValueError("exact soft-label BT control must contain its dedicated BT head")
        if self.head.final_pcg is not None:
            raise ValueError("exact soft-label BT must not contain a PCG audit")
        target = _strict_json_copy(self.target_audit, name="target_audit")
        if target.get("noise_free") is not True:
            raise ValueError("exact soft-label BT target audit must be noise-free")
        if target.get("sampled_label_stream_accessed") is not False:
            raise ValueError("exact soft-label BT cannot access sampled labels")
        if target.get("test_or_validation_data_accessed") is not False:
            raise ValueError("exact soft-label BT cannot access held-out data")
        if target.get("eligible_for_primary_claim") is not False:
            raise ValueError("exact soft-label BT cannot be eligible for the primary claim")
        _strict_json_copy(self.optimization_audit, name="optimization_audit")

    def to_dict(self) -> dict[str, object]:
        return {
            "head": self.head.to_dict(),
            "target_audit": _strict_json_copy(self.target_audit, name="target_audit"),
            "optimization_audit": _strict_json_copy(
                self.optimization_audit,
                name="optimization_audit",
            ),
        }


@dataclass(frozen=True, slots=True)
class Phase2TrainingResult:
    """Fresh primary heads and train-only controls; contains no oracle values."""

    settings: Phase2TrainingSettings
    training_design_sha256: str
    training_settings_sha256: str
    training_instance_sha256: str
    input_training_sha256: str
    absolute_damping: float
    label_stream: LabelStreamEvidence
    bt_mle: TrainedHeadEvidence
    prorm_plus: TrainedHeadEvidence
    low_dimensional_control: Mapping[str, object]
    exact_margin_control: ExactMarginTrainingControl
    exact_soft_label_bt_control: ExactSoftLabelBTControl
    direct_oracle_identity: Mapping[str, object]
    primary_optimization_audit: Mapping[str, object]
    training_arm: Literal["r4_independent_gamma_0.9"] = PRIMARY_TRAINING_ARM
    test_data_accessed: Literal[False] = False
    old_phase1_comparison_heads_used: Literal[False] = False
    raw_node_rewards_retained: Literal[False] = False

    def __post_init__(self) -> None:
        if not isinstance(self.settings, Phase2TrainingSettings):
            raise TypeError("settings must be Phase2TrainingSettings")
        if self.training_design_sha256 != self.settings.phase2_config_hash:
            raise ValueError("training_design_sha256 must equal the full Phase-2 overlay identity")
        _validate_digest(self.training_design_sha256, name="training_design_sha256")
        if self.training_settings_sha256 != self.settings.sha256:
            raise ValueError("training_settings_sha256 does not match settings")
        for name in (
            "training_settings_sha256",
            "training_instance_sha256",
            "input_training_sha256",
        ):
            _validate_digest(getattr(self, name), name=name)
        _finite_float(
            self.absolute_damping,
            name="absolute_damping",
            minimum=0.0,
            strictly_greater=True,
        )
        if not isinstance(self.label_stream, LabelStreamEvidence):
            raise TypeError("label_stream must be LabelStreamEvidence")
        if self.bt_mle.arm != PRIMARY_TRAINING_ARM or self.bt_mle.method != BT_MLE:
            raise ValueError("bt_mle must be the fresh R=4 primary BT head")
        if self.prorm_plus.arm != PRIMARY_TRAINING_ARM or self.prorm_plus.method != PRORM_PLUS:
            raise ValueError("prorm_plus must be the fresh R=4 primary ProRM+ head")
        if self.bt_mle.initial_head_sha256 != self.prorm_plus.initial_head_sha256:
            raise ValueError("primary learners must share the exact zero initialization")
        if not isinstance(self.exact_margin_control, ExactMarginTrainingControl):
            raise TypeError("exact_margin_control must be ExactMarginTrainingControl")
        if not isinstance(self.exact_soft_label_bt_control, ExactSoftLabelBTControl):
            raise TypeError("exact_soft_label_bt_control must be ExactSoftLabelBTControl")
        _strict_json_copy(
            self.low_dimensional_control,
            name="low_dimensional_control",
        )
        _strict_json_copy(self.direct_oracle_identity, name="direct_oracle_identity")
        _strict_json_copy(
            self.primary_optimization_audit,
            name="primary_optimization_audit",
        )
        if (
            self.training_arm != PRIMARY_TRAINING_ARM
            or self.test_data_accessed is not False
            or self.old_phase1_comparison_heads_used is not False
            or self.raw_node_rewards_retained is not False
        ):
            raise ValueError("invalid Phase-2 training isolation contract")

    @property
    def bt_head(self) -> tuple[float, ...]:
        return self.bt_mle.head_weight

    @property
    def prorm_plus_head(self) -> tuple[float, ...]:
        return self.prorm_plus.head_weight

    @property
    def heads(self) -> dict[str, tuple[float, ...]]:
        return {
            BT_MLE: self.bt_head,
            PRORM_PLUS: self.prorm_plus_head,
        }

    @property
    def schema_version(self) -> str:
        return (
            PHASE2_RECOVERY_TRAINING_SCHEMA
            if self.settings.convergence.optimizer_protocol is not None
            else PHASE2_TRAINING_SCHEMA
        )

    @property
    def audit(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "training_design_sha256": self.training_design_sha256,
            "training_settings_sha256": self.training_settings_sha256,
            "training_instance_sha256": self.training_instance_sha256,
            "input_training_sha256": self.input_training_sha256,
            "training_arm": self.training_arm,
            "absolute_damping": self.absolute_damping,
            "label_stream": self.label_stream.to_dict(),
            "primary_heads": {
                BT_MLE: self.bt_mle.to_dict(),
                PRORM_PLUS: self.prorm_plus.to_dict(),
            },
            "primary_optimization_audit": _strict_json_copy(
                self.primary_optimization_audit,
                name="primary_optimization_audit",
            ),
            "low_dimensional_control": _strict_json_copy(
                self.low_dimensional_control,
                name="low_dimensional_control",
            ),
            "exact_margin_control": self.exact_margin_control.to_dict(),
            "exact_soft_label_bt_control": self.exact_soft_label_bt_control.to_dict(),
            "direct_oracle_identity": _strict_json_copy(
                self.direct_oracle_identity,
                name="direct_oracle_identity",
            ),
            "isolation": {
                "test_data_accessed": self.test_data_accessed,
                "old_phase1_comparison_heads_used": (self.old_phase1_comparison_heads_used),
                "raw_node_rewards_retained": self.raw_node_rewards_retained,
                "raw_labels_retained": False,
                "primary_heads_are_fresh_zero_initialized": True,
            },
        }

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "heads": {learner: list(self.heads[learner]) for learner in CANONICAL_LEARNERS},
            "audit": self.audit,
            "test_data_accessed": self.test_data_accessed,
        }
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        return payload

    def to_runner_head_result(self) -> object:
        """Adapt lazily to the runner without making this module depend on it."""

        from .phase2_rollout import Phase2HeadTrainingResult

        return Phase2HeadTrainingResult(
            heads=self.heads,
            training_design_sha256=self.training_design_sha256,
            training_arm=self.training_arm,
            audit=self.audit,
            test_data_accessed=self.test_data_accessed,
        )


def _input_training_sha256(training: TrainingTensorData) -> str:
    return _canonical_sha256(
        {
            "schema_version": "phase2-training-input/v1",
            "prompt_ids": list(training.prompt_ids),
            "policy_scores_sha256": _tensor_sha256(training.policy_scores),
            "reward_features_sha256": _tensor_sha256(training.reward_features),
            "input_h_sha256": _tensor_sha256(training.h),
            "input_left_wins_sha256": _tensor_sha256(training.left_wins),
            "input_num_annotations_sha256": _tensor_sha256(training.num_annotations),
        }
    )


@torch.no_grad()
def _reward_head_identifiability(
    training: TrainingTensorData,
    settings: Phase2TrainingSettings,
) -> dict[str, object]:
    """Measure reward-feature rank and a recovery-only BT existence diagnostic.

    Full column rank of the complete edge design makes the finite-parameter BT
    Hessian positive definite, but does not exclude complete or quasi
    separation and therefore does not prove existence of a finite minimizer.
    In the one-shot recovery schema we additionally audit the submatrix of
    edges having both outcomes.  Full column rank of that submatrix is a clean
    sufficient condition for coercivity, finite existence, and uniqueness.
    """

    recovery = settings.convergence.optimizer_protocol is not None
    tie_break = (
        settings.convergence.optimizer_protocol.tie_break
        if settings.convergence.optimizer_protocol is not None
        else _LEGACY_TIE_BREAK
    )
    differences = (training.reward_features[:, 0] - training.reward_features[:, 1]).detach()
    matrix = differences.to(dtype=torch.float64)
    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.ndim != 1 or singular_values.numel() < 1:
        raise RuntimeError("reward-head rank audit returned no singular values")
    if not bool(torch.isfinite(singular_values).all()):
        raise FloatingPointError("reward-head rank audit produced non-finite singular values")
    largest = float(singular_values[0].item())
    if not math.isfinite(largest) or largest <= 0.0:
        raise ValueError("reward feature-difference design matrix is identically zero")
    threshold = settings.identifiability_relative_rank_tolerance * largest
    retained = singular_values > threshold
    numerical_rank = int(torch.count_nonzero(retained).item())
    column_dimension = int(matrix.shape[1])
    full_column_rank = numerical_rank == column_dimension
    smallest = float(singular_values[-1].item())
    smallest_retained = float(singular_values[retained][-1].item()) if numerical_rank > 0 else None
    condition_number = (
        float(largest / smallest_retained)
        if smallest_retained is not None and smallest_retained > 0.0
        else None
    )
    gate_passed = full_column_rank if settings.identifiability_require_full_column_rank else True
    evidence: dict[str, object] = {
        "schema_version": (
            "reward-head-identifiability/v2" if recovery else "reward-head-identifiability/v1"
        ),
        "design_matrix": "canonical_edge_reward_feature_differences",
        "split": "train",
        "shape": [int(matrix.shape[0]), column_dimension],
        "source_dtype": str(differences.dtype),
        "audit_dtype": str(matrix.dtype),
        "design_matrix_sha256": _tensor_sha256(differences),
        "relative_rank_tolerance": (settings.identifiability_relative_rank_tolerance),
        "absolute_singular_value_threshold": threshold,
        "numerical_rank": numerical_rank,
        "column_dimension": column_dimension,
        "full_column_rank": full_column_rank,
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "smallest_retained_singular_value": smallest_retained,
        "retained_condition_number": condition_number,
        "role": settings.identifiability_role,
        "require_full_column_rank": (settings.identifiability_require_full_column_rank),
        "acceptance_gate_passed": gate_passed,
        "prorm_moment_map_full_rank_proved": False,
        "algorithmic_tie_break": tie_break,
        "minimum_norm_solution_claimed": False,
        "test_or_validation_data_accessed": False,
    }
    if recovery:
        mixed_mask = (training.left_wins > 0) & (training.left_wins < training.num_annotations)
        mixed_source = differences[mixed_mask].detach()
        mixed_matrix = mixed_source.to(dtype=torch.float64)
        mixed_count = int(mixed_matrix.shape[0])
        if mixed_count:
            mixed_singular_values = torch.linalg.svdvals(mixed_matrix)
            if mixed_singular_values.ndim != 1 or not bool(
                torch.isfinite(mixed_singular_values).all()
            ):
                raise FloatingPointError(
                    "mixed-outcome reward-head rank audit produced invalid singular values"
                )
            mixed_values, mixed_spectrum_sha256 = _singular_value_payload(mixed_singular_values)
            mixed_largest = mixed_values[0]
            mixed_smallest = mixed_values[-1]
            mixed_threshold = settings.identifiability_relative_rank_tolerance * mixed_largest
            mixed_retained = [value for value in mixed_values if value > mixed_threshold]
        else:
            mixed_values = []
            mixed_spectrum_sha256 = None
            mixed_largest = None
            mixed_smallest = None
            mixed_threshold = 0.0
            mixed_retained = []
        mixed_rank = len(mixed_retained)
        mixed_smallest_retained = mixed_retained[-1] if mixed_retained else None
        mixed_full_column_rank = mixed_rank == column_dimension
        mixed_condition_number = (
            mixed_largest / mixed_smallest_retained
            if mixed_largest is not None
            and mixed_smallest_retained is not None
            and mixed_smallest_retained > 0.0
            else None
        )
        evidence.update(
            {
                "full_design_rank_implication": (
                    "strict_convexity_at_finite_parameters_and_at_most_one_finite_minimizer;"
                    "does_not_exclude_complete_or_quasi_separation"
                ),
                "full_design_rank_proves_finite_bt_minimizer_exists": False,
                "mixed_outcome_edge_coercivity_diagnostic": {
                    "schema_version": "bt-mixed-outcome-coercivity-diagnostic/v1",
                    "definition": "0 < left_wins < num_annotations",
                    "split": "train",
                    "orientation": "candidate_0_minus_candidate_1",
                    "num_total_edges": int(matrix.shape[0]),
                    "num_mixed_outcome_edges": mixed_count,
                    "mixed_outcome_mask_sha256": _tensor_sha256(mixed_mask),
                    "left_wins_sha256": _tensor_sha256(training.left_wins),
                    "num_annotations_sha256": _tensor_sha256(training.num_annotations),
                    "design_matrix": (
                        "reward_feature_differences_restricted_to_mixed_outcome_edges"
                    ),
                    "shape": [mixed_count, column_dimension],
                    "source_dtype": str(mixed_source.dtype),
                    "audit_dtype": str(mixed_matrix.dtype),
                    "design_matrix_sha256": _tensor_sha256(mixed_source),
                    "relative_rank_tolerance": (settings.identifiability_relative_rank_tolerance),
                    "absolute_singular_value_threshold": mixed_threshold,
                    "singular_values_descending": mixed_values,
                    "singular_values_sha256": mixed_spectrum_sha256,
                    "largest_singular_value": mixed_largest,
                    "smallest_singular_value": mixed_smallest,
                    "smallest_retained_singular_value": mixed_smallest_retained,
                    "numerical_rank": mixed_rank,
                    "column_dimension": column_dimension,
                    "full_column_rank": mixed_full_column_rank,
                    "retained_condition_number": mixed_condition_number,
                    "sufficient_condition": (
                        "full_column_rank_implies_coercive_strictly_convex_binomial_bt;"
                        "therefore_a_unique_finite_minimizer_exists"
                    ),
                    "sufficient_condition_observed": mixed_full_column_rank,
                    "acceptance_gate_applied": False,
                    "role": "train_only_measure_and_interpret",
                    "raw_outcomes_serialized": False,
                    "test_or_validation_data_accessed": False,
                },
                "bt_unique_finite_minimizer_sufficient_condition": (
                    "mixed_outcome_edge_difference_matrix_full_column_rank"
                ),
                "bt_unique_finite_minimizer_sufficient_condition_observed": (
                    mixed_full_column_rank
                ),
            }
        )
    else:
        # Frozen legacy-v1 serialization is retained only for the superseded
        # constant-LR protocol.  Recovery-v2 above carries the corrected
        # separation-aware scientific statement.
        evidence["bt_unique_finite_optimum_sufficient_condition_only"] = True
    if not gate_passed:
        raise OptimizationConvergenceError(
            "reward-head identifiability gate failed",
            evidence=evidence,
        )
    return evidence


def _singular_value_payload(singular_values: torch.Tensor) -> tuple[list[float], str]:
    """Serialize one FP64 spectrum with a canonical, JSON-recomputable digest."""

    if (
        not isinstance(singular_values, torch.Tensor)
        or singular_values.ndim != 1
        or singular_values.dtype != torch.float64
    ):
        raise TypeError("singular_values must be a rank-one FP64 tensor")
    values = [float(value) for value in singular_values.detach().to(device="cpu").tolist()]
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise FloatingPointError("singular-value spectrum must be finite and nonnegative")
    payload: dict[str, object] = {
        "dtype": "torch.float64",
        "shape": [len(values)],
        "values_descending": values,
    }
    return values, _canonical_sha256(payload)


@torch.no_grad()
def _prorm_moment_map_identifiability(
    training: TrainingTensorData,
    settings: Phase2TrainingSettings,
    *,
    projected_geometry: _DensePseudoinverseGeometry | None = None,
    projection_sha256: str | None = None,
) -> dict[str, object]:
    """Audit the empirical ProRM head Jacobian without random approximation.

    On the canonical train edges, let ``Z = s0(y0) - s0(y1)`` and
    ``D = f(y0) - f(y1)``.  For the linear reward head,

        J_m = Z.T @ D / (2 * n_edges).

    The potentially large ``policy_dimension x reward_dimension`` matrix is
    streamed in deterministic FP64 row blocks.  A deterministic TSQR reduction
    preserves its singular spectrum without materializing the full matrix or
    using a randomized rank estimator.
    """

    projected = projected_geometry is not None
    if projected != (projection_sha256 is not None):
        raise ValueError(
            "projected_geometry and projection_sha256 must either both be supplied or both omitted"
        )
    if projection_sha256 is not None:
        _validate_digest(projection_sha256, name="projection_sha256")
    recovery = settings.convergence.optimizer_protocol is not None
    tie_break = (
        settings.convergence.optimizer_protocol.tie_break
        if settings.convergence.optimizer_protocol is not None
        else _LEGACY_TIE_BREAK
    )

    edge_policy_scores = (training.policy_scores[:, 0] - training.policy_scores[:, 1]).detach()
    edge_reward_features = (
        training.reward_features[:, 0] - training.reward_features[:, 1]
    ).detach()
    num_edges = int(edge_policy_scores.shape[0])
    policy_dimension = int(edge_policy_scores.shape[1])
    reward_dimension = int(edge_reward_features.shape[1])
    if num_edges < 1 or policy_dimension < 1 or reward_dimension < 1:
        raise ValueError("ProRM moment-map dimensions must be positive")

    # Keep the largest transient J_m block near 16 MiB.  At least one reward
    # dimension's worth of rows is useful for TSQR, while the hard cap prevents
    # an unexpectedly small reward dimension from creating a very large block.
    target_block_bytes = 16 * 1024 * 1024
    bytes_per_row = torch.empty((), dtype=torch.float64).element_size() * reward_dimension
    block_rows = min(
        policy_dimension,
        max(
            1,
            min(4096, target_block_bytes // max(1, bytes_per_row)),
        ),
    )
    reward_features_fp64 = edge_reward_features.to(device="cpu", dtype=torch.float64)
    scale = 1.0 / (2.0 * float(num_edges))
    digest = hashlib.sha256()
    digest.update(str(torch.float64).encode("ascii"))
    digest.update(repr((policy_dimension, reward_dimension)).encode("ascii"))
    reduced_r: torch.Tensor | None = None
    num_blocks = 0

    for start in range(0, policy_dimension, block_rows):
        stop = min(policy_dimension, start + block_rows)
        score_block = edge_policy_scores[:, start:stop].to(
            device="cpu",
            dtype=torch.float64,
        )
        moment_block = (score_block.mT @ reward_features_fp64).mul_(scale).contiguous()
        if not bool(torch.isfinite(moment_block).all()):
            raise FloatingPointError("ProRM moment-map block contains NaN or infinity")
        digest.update(bytes(moment_block.untyped_storage()))
        _, block_r = torch.linalg.qr(moment_block, mode="r")
        if reduced_r is None:
            reduced_r = block_r
        else:
            _, reduced_r = torch.linalg.qr(
                torch.cat((reduced_r, block_r), dim=0),
                mode="r",
            )
        num_blocks += 1

    if reduced_r is None or num_blocks < 1:
        raise RuntimeError("ProRM moment-map TSQR produced no reduction")
    singular_values = torch.linalg.svdvals(reduced_r)
    if singular_values.ndim != 1 or singular_values.numel() < 1:
        raise RuntimeError("ProRM moment-map rank audit returned no singular values")
    singular_values = singular_values.to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(singular_values).all()):
        raise FloatingPointError("ProRM moment-map rank audit produced non-finite values")
    # LAPACK returns descending values, but sort explicitly so the serialized
    # contract is independent of backend ordering conventions.
    singular_values = torch.sort(singular_values, descending=True).values
    values, spectrum_sha256 = _singular_value_payload(singular_values)
    largest = values[0]
    smallest = values[-1]
    threshold = settings.identifiability_relative_rank_tolerance * largest
    retained_values = [value for value in values if value > threshold]
    numerical_rank = len(retained_values)
    full_column_rank = numerical_rank == reward_dimension
    full_row_rank = numerical_rank == policy_dimension
    smallest_retained = retained_values[-1] if retained_values else None
    condition_number = (
        largest / smallest_retained
        if smallest_retained is not None and smallest_retained > 0.0
        else None
    )
    require_full_column_rank = (
        False if projected else settings.identifiability_require_full_column_rank
    )
    gate_passed = (
        full_row_rank if projected else (full_column_rank if require_full_column_rank else True)
    )
    if projected:
        if projected_geometry is None or projection_sha256 is None:
            raise RuntimeError("projected ProRM moment-map geometry was not resolved")
        if projected_geometry.rank != policy_dimension:
            raise OptimizationConvergenceError(
                "projected ProRM moment-map requires a full-rank low-dimensional Fisher",
                evidence={
                    "selected_dimension": policy_dimension,
                    "fisher_numerical_rank": projected_geometry.rank,
                    "projection_sha256": projection_sha256,
                },
            )
        geometry_key = "projected_geometry"
        geometry_evidence: dict[str, object] = {
            "matrix": "H_low = F_hat_low",
            "positive_definite": True,
            "reason": "projected_fisher_numerical_rank_equals_selected_dimension",
            "regularization": "moore_penrose_pseudoinverse_on_full_rank_H_low",
            "fisher_sha256": projected_geometry.fisher_sha256,
            "pseudoinverse_sha256": projected_geometry.pseudoinverse_sha256,
            "relative_eigenvalue_tolerance": (
                settings.low_dimensional_relative_eigenvalue_tolerance
            ),
            "head_hessian": "J_m^T H_low^{-1} J_m / beta",
            "rank_identity": "rank(J_m^T H_low^{-1} J_m) = rank(J_m)",
        }
    else:
        geometry_key = "ridge_geometry"
        geometry_evidence = {
            "matrix": "H = F_hat + lambda I",
            "positive_definite": True,
            "reason": "configured_relative_damping_is_strictly_positive",
            "head_hessian": "J_m^T H^{-1} J_m / beta",
            "rank_identity": "rank(J_m^T H^{-1} J_m) = rank(J_m)",
        }
    uniqueness_key = (
        "unique_projected_prorm_quadratic_head_iff_full_column_rank"
        if projected
        else "unique_ridge_prorm_quadratic_head_iff_full_column_rank"
    )
    observed_uniqueness_key = (
        "observed_unique_projected_prorm_quadratic_head"
        if projected
        else "observed_unique_ridge_prorm_quadratic_head"
    )
    evidence: dict[str, object] = {
        "schema_version": (
            (
                "projected-prorm-moment-map-identifiability/v2"
                if projected
                else "prorm-moment-map-identifiability/v2"
            )
            if recovery
            else (
                "projected-prorm-moment-map-identifiability/v1"
                if projected
                else "prorm-moment-map-identifiability/v1"
            )
        ),
        "design_matrix": (
            "canonical_train_edge_projected_moment_jacobian"
            if projected
            else "canonical_train_edge_moment_jacobian"
        ),
        "formula": "J_m = Z^T D / (2 n_edges)",
        "split": "train",
        "orientation": "candidate_0_minus_candidate_1",
        "shape": [policy_dimension, reward_dimension],
        "num_edges": num_edges,
        "policy_dimension": policy_dimension,
        "column_dimension": reward_dimension,
        "source_policy_score_dtype": str(edge_policy_scores.dtype),
        "source_reward_feature_dtype": str(edge_reward_features.dtype),
        "audit_dtype": str(torch.float64),
        "edge_policy_score_difference_sha256": _tensor_sha256(edge_policy_scores),
        "edge_reward_feature_difference_sha256": _tensor_sha256(edge_reward_features),
        "moment_map_sha256": digest.hexdigest(),
        "computation": {
            "algorithm": "deterministic_blocked_fp64_tsqr",
            "row_block_size": block_rows,
            "num_row_blocks": num_blocks,
            "full_moment_map_materialized": False,
            "randomized_rank_approximation_used": False,
        },
        "relative_rank_tolerance": settings.identifiability_relative_rank_tolerance,
        "absolute_singular_value_threshold": threshold,
        "singular_values_descending": values,
        "singular_values_sha256": spectrum_sha256,
        "singular_spectrum_summary": {
            "count": len(values),
            "largest": largest,
            "smallest": smallest,
            "smallest_retained": smallest_retained,
        },
        "numerical_rank": numerical_rank,
        "full_column_rank": full_column_rank,
        "retained_condition_number": condition_number,
        geometry_key: geometry_evidence,
        uniqueness_key: True,
        observed_uniqueness_key: full_column_rank,
        "population_identifiability_theorem_claimed": False,
        "role": settings.identifiability_role,
        "require_full_column_rank": require_full_column_rank,
        "acceptance_gate_passed": gate_passed,
        "algorithmic_tie_break": tie_break,
        "minimum_norm_solution_claimed": False,
        "test_or_validation_data_accessed": False,
    }
    if projection_sha256 is not None:
        evidence["projection_sha256"] = projection_sha256
        evidence["row_dimension"] = policy_dimension
        evidence["full_row_rank"] = full_row_rank
        evidence["require_full_row_rank"] = True
        evidence["acceptance_gate_definition"] = (
            "full_row_rank_for_projected_policy_moment_coverage"
        )
    if not gate_passed:
        raise OptimizationConvergenceError(
            (
                "projected ProRM moment-map policy-coverage gate failed"
                if projected
                else "ProRM moment-map identifiability gate failed"
            ),
            evidence=evidence,
        )
    return evidence


def _absolute_damping(training: TrainingTensorData, settings: Phase2TrainingSettings) -> float:
    solve_dtype = resolve_fisher_solve_dtype(settings.pcg_dtype)
    flat_scores = training.policy_scores.reshape(-1, training.policy_dimension).to(
        dtype=solve_dtype
    )
    mean_diagonal = float(flat_scores.square().mean(dim=0).mean().item())
    if not math.isfinite(mean_diagonal) or mean_diagonal <= 0.0:
        raise ValueError(
            "mean(diag(F)) must be finite and positive; the policy tangent is degenerate"
        )
    damping = settings.relative_damping * mean_diagonal
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("absolute damping must be finite and positive")
    return damping


def _derived_label_seed(base_seed: int, namespace: str) -> tuple[int, str]:
    derivation = {
        "schema_version": "named-label-seed/v1",
        "namespace": namespace,
        "base_seed": base_seed,
    }
    digest = _canonical_sha256(derivation)
    derived = int.from_bytes(bytes.fromhex(digest)[:8], byteorder="big") & (2**63 - 1)
    return derived, digest


def _generator_for_training(
    training: TrainingTensorData,
    *,
    base_seed: int,
    namespace: str,
) -> tuple[torch.Generator, int, str]:
    derived_seed, derivation_sha256 = _derived_label_seed(base_seed, namespace)
    generator = torch.Generator(device=training.policy_scores.device)
    generator.manual_seed(derived_seed)
    return generator, derived_seed, derivation_sha256


def _history_summary(
    history: Sequence[TrainingStepDiagnostics],
) -> dict[str, object]:
    if not history:
        raise ValueError("training history must not be empty")
    steps = len(history)
    checkpoint_steps = sorted(
        {
            1,
            max(1, steps // 4),
            max(1, steps // 2),
            max(1, (3 * steps) // 4),
            steps,
        }
    )
    by_step = {item.step: item for item in history}
    checkpoints = [asdict(by_step[step]) for step in checkpoint_steps]
    objectives = [float(item.objective) for item in history]
    gradients = [float(item.gradient_norm) for item in history]
    pcg_records = [item for item in history if item.pcg_converged is not None]
    summary: dict[str, object] = {
        "num_steps": steps,
        "history_objective_timing": "pre_update",
        "stored_checkpoint_steps": checkpoint_steps,
        "checkpoints": checkpoints,
        "objective": {
            "first": objectives[0],
            "last_pre_update": objectives[-1],
            "minimum": min(objectives),
            "maximum": max(objectives),
        },
        "gradient_l2_norm": {
            "first": gradients[0],
            "last_pre_update": gradients[-1],
            "minimum": min(gradients),
            "maximum": max(gradients),
        },
    }
    if pcg_records:
        if any(item.pcg_converged is not True for item in pcg_records):
            raise RuntimeError("a recorded ProRM+ training PCG solve did not converge")
        relative = [
            float(item.pcg_relative_residual)
            for item in pcg_records
            if item.pcg_relative_residual is not None
        ]
        iterations = [
            int(item.pcg_iterations) for item in pcg_records if item.pcg_iterations is not None
        ]
        summary["pcg"] = {
            "num_fresh_solves": len(pcg_records),
            "all_converged": True,
            "maximum_relative_residual": max(relative),
            "maximum_iterations": max(iterations),
        }
    else:
        summary["pcg"] = None
    return _strict_json_copy(summary, name="history_summary")


@dataclass(frozen=True, slots=True)
class _FirstOrderMeasurement:
    """One immutable full-data, unclipped objective/gradient measurement."""

    objective: float
    gradient_l2_norm: float
    inner_solver: Mapping[str, object] | None = None
    audit_dtype: str | None = None

    def __post_init__(self) -> None:
        _finite_float(self.objective, name="objective")
        _finite_float(
            self.gradient_l2_norm,
            name="gradient_l2_norm",
            minimum=0.0,
        )
        if self.inner_solver is not None:
            _strict_json_copy(self.inner_solver, name="inner_solver")
        if self.audit_dtype is not None and self.audit_dtype != "float64":
            raise ValueError("first-order audit_dtype must equal float64 when declared")

    def to_dict(self, *, include_audit_dtype: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "objective": self.objective,
            "gradient_l2_norm": self.gradient_l2_norm,
            "inner_solver": (
                None
                if self.inner_solver is None
                else _strict_json_copy(self.inner_solver, name="inner_solver")
            ),
        }
        if include_audit_dtype:
            if self.audit_dtype != "float64":
                raise ValueError(
                    "recovery first-order measurements must declare audit_dtype='float64'"
                )
            result["audit_dtype"] = self.audit_dtype
        return result


@dataclass(frozen=True, slots=True)
class _ConvergedTrainingRun:
    """Selected primary state and its independently retained 720-step state."""

    history: tuple[TrainingStepDiagnostics, ...]
    initial: _FirstOrderMeasurement
    final: _FirstOrderMeasurement
    evidence: Mapping[str, object]


def _gradient_ratio(
    gradient_l2_norm: float,
    initial_gradient_l2_norm: float,
    spec: FirstOrderConvergenceSpec,
) -> float:
    denominator = max(
        initial_gradient_l2_norm,
        spec.gradient_norm_denominator_floor,
    )
    ratio = gradient_l2_norm / denominator
    if not math.isfinite(ratio) or ratio < 0.0:
        raise FloatingPointError("first-order gradient ratio is non-finite")
    return ratio


def _recovery_adamw_group_payload(
    optimizer: torch.optim.Optimizer,
    parameter: torch.Tensor,
    protocol: AdamWRecoveryProtocol,
    *,
    expected_learning_rate: float,
) -> dict[str, object]:
    """Validate and serialize every protocol-relevant AdamW group field."""

    if len(optimizer.param_groups) != 1:
        raise RuntimeError("recovery AdamW must retain exactly one parameter group")
    group = optimizer.param_groups[0]
    parameters = group.get("params")
    if not isinstance(parameters, list) or len(parameters) != 1 or parameters[0] is not parameter:
        raise RuntimeError("recovery AdamW parameter group no longer contains only the head")
    expected: dict[str, object] = {
        "lr": expected_learning_rate,
        "betas": protocol.betas,
        "eps": protocol.eps,
        "weight_decay": 0.0,
        "amsgrad": protocol.amsgrad,
        "maximize": protocol.maximize,
        "foreach": protocol.foreach,
        "capturable": protocol.capturable,
        "differentiable": protocol.differentiable,
        "fused": protocol.fused,
    }
    for name, expected_value in expected.items():
        if group.get(name) != expected_value:
            raise RuntimeError(
                f"recovery AdamW parameter-group field {name!r} changed from its locked value"
            )
    return {
        "learning_rate": float(group["lr"]),
        "betas": [float(value) for value in group["betas"]],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]),
        "maximize": bool(group["maximize"]),
        "foreach": bool(group["foreach"]),
        "capturable": bool(group["capturable"]),
        "differentiable": bool(group["differentiable"]),
        "fused": bool(group["fused"]),
        "parameter_count": 1,
        "parameter_is_reward_head": True,
    }


def _recovery_adamw_step_value(value: object, *, expected: int) -> dict[str, object]:
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or value.numel() != 1:
        raise RuntimeError("recovery AdamW state['step'] must be a scalar tensor")
    if value.dtype not in {torch.float32, torch.float64, torch.int32, torch.int64}:
        raise RuntimeError("recovery AdamW scalar step has an unsupported dtype")
    if value.device.type != "cpu":
        raise RuntimeError(
            "non-capturable, non-fused recovery AdamW scalar step must remain on CPU"
        )
    scalar = value.item()
    if (
        isinstance(scalar, bool)
        or not isinstance(scalar, Real)
        or not math.isfinite(float(scalar))
        or float(scalar) != float(expected)
    ):
        raise RuntimeError(f"recovery AdamW scalar step must equal {expected}, observed {scalar!r}")
    return {
        "value": expected,
        "shape": [],
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _recovery_adamw_moment_descriptor(
    value: object,
    parameter: torch.Tensor,
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"recovery AdamW state[{name!r}] must be a tensor")
    if value.shape != parameter.shape:
        raise RuntimeError(f"recovery AdamW state[{name!r}] has the wrong shape")
    if value.dtype != parameter.dtype:
        raise RuntimeError(f"recovery AdamW state[{name!r}] has the wrong dtype")
    if value.device != parameter.device:
        raise RuntimeError(f"recovery AdamW state[{name!r}] has the wrong device")
    if value.layout != torch.strided or value.is_sparse:
        raise RuntimeError(f"recovery AdamW state[{name!r}] must be a dense strided tensor")
    if value.requires_grad or value.grad_fn is not None:
        raise RuntimeError(f"recovery AdamW state[{name!r}] must be detached")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": str(value.layout),
        "detached": True,
    }


def _validate_recovery_adamw_state(
    optimizer: torch.optim.Optimizer,
    parameter: torch.Tensor,
    protocol: AdamWRecoveryProtocol,
    *,
    expected_completed_updates: int,
    expected_learning_rate: float,
) -> dict[str, object]:
    """Fail closed unless one AdamW update maps to one exact state step."""

    if isinstance(expected_completed_updates, bool) or expected_completed_updates < 0:
        raise ValueError("expected_completed_updates must be non-negative")
    group = _recovery_adamw_group_payload(
        optimizer,
        parameter,
        protocol,
        expected_learning_rate=expected_learning_rate,
    )
    if expected_completed_updates == 0:
        if len(optimizer.state) != 0:
            raise RuntimeError("recovery AdamW state must be empty before its first update")
        return {
            "expected_completed_updates": 0,
            "optimizer_state_empty": True,
            "scalar_step": None,
            "exp_avg": None,
            "exp_avg_sq": None,
            "parameter_group": group,
        }

    if len(optimizer.state) != 1 or parameter not in optimizer.state:
        raise RuntimeError(
            "recovery AdamW must retain exactly one non-empty state entry for the head"
        )
    state = optimizer.state[parameter]
    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise RuntimeError(
            "recovery AdamW head state must contain exactly step, exp_avg, and exp_avg_sq"
        )
    return {
        "expected_completed_updates": expected_completed_updates,
        "optimizer_state_empty": False,
        "scalar_step": _recovery_adamw_step_value(
            state["step"],
            expected=expected_completed_updates,
        ),
        "exp_avg": _recovery_adamw_moment_descriptor(
            state["exp_avg"],
            parameter,
            name="exp_avg",
        ),
        "exp_avg_sq": _recovery_adamw_moment_descriptor(
            state["exp_avg_sq"],
            parameter,
            name="exp_avg_sq",
        ),
        "parameter_group": group,
    }


def _optimizer_moment_state_sha256(
    optimizer: torch.optim.Optimizer,
    parameter: torch.Tensor,
) -> str:
    """Hash AdamW state without serializing it or changing device placement."""

    state = optimizer.state.get(parameter, {})
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode("utf-8"))
        value = state[key]
        if isinstance(value, torch.Tensor):
            digest.update(_tensor_sha256(value).encode("ascii"))
        elif isinstance(value, (int, float, bool)):
            digest.update(repr(value).encode("ascii"))
        else:
            raise TypeError(f"unsupported AdamW state value for {key!r}")
    return digest.hexdigest()


def _recovery_optimizer_state_sha256(
    optimizer: torch.optim.Optimizer,
    parameter: torch.Tensor,
    protocol: AdamWRecoveryProtocol,
    *,
    expected_completed_updates: int,
    expected_learning_rate: float,
) -> str:
    """Hash validated moments plus every protocol-relevant group field."""

    validated = _validate_recovery_adamw_state(
        optimizer,
        parameter,
        protocol,
        expected_completed_updates=expected_completed_updates,
        expected_learning_rate=expected_learning_rate,
    )
    state = optimizer.state.get(parameter, {})
    tensor_state: dict[str, object] = {}
    for name in ("step", "exp_avg", "exp_avg_sq"):
        value = state.get(name)
        tensor_state[name] = (
            None
            if value is None
            else {
                "sha256": _tensor_sha256(value),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        )
    return _canonical_sha256(
        {
            "schema_version": "validated-recovery-adamw-state/v1",
            "optimizer_class": "torch.optim.AdamW",
            "completed_updates": expected_completed_updates,
            "state": tensor_state,
            "parameter_group": validated["parameter_group"],
            "protocol": {
                "learning_rate_schedule_sha256": protocol.schedule_sha256,
                "betas": list(protocol.betas),
                "eps": protocol.eps,
                "weight_decay": 0.0,
                "amsgrad": protocol.amsgrad,
                "maximize": protocol.maximize,
                "foreach": protocol.foreach,
                "capturable": protocol.capturable,
                "differentiable": protocol.differentiable,
                "fused": protocol.fused,
                "one_optimizer_update_per_step": protocol.one_optimizer_update_per_step,
                "optimizer_state_reset_at_lr_milestone": (
                    protocol.optimizer_state_reset_at_lr_milestone
                ),
            },
        }
    )


def _checkpoint_value_fingerprint(value: object) -> object:
    """Convert an in-memory Torch checkpoint to a canonical JSON fingerprint."""

    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "sha256": _tensor_sha256(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        entries: list[dict[str, object]] = []
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise TypeError("checkpoint mappings may use only string or integer keys")
            entries.append(
                {
                    "key_type": type(key).__name__,
                    "key": key,
                    "value": _checkpoint_value_fingerprint(item),
                }
            )
        entries.sort(
            key=lambda item: json.dumps(
                [item["key_type"], item["key"]],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return {"kind": "mapping", "entries": entries}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_checkpoint_value_fingerprint(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_checkpoint_value_fingerprint(item) for item in value],
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("checkpoint scalar values must be finite")
        return {"kind": type(value).__name__, "value": value}
    raise TypeError(f"unsupported recovery checkpoint value {type(value).__name__}")


def _checkpoint_value_sha256(value: object) -> str:
    return _canonical_sha256(
        {
            "schema_version": "in-memory-torch-checkpoint-fingerprint/v1",
            "value": _checkpoint_value_fingerprint(value),
        }
    )


def _checkpoint_optimizer_state_dict(
    trainer_state: Mapping[str, object],
) -> Mapping[str, object]:
    optimizer_state = trainer_state.get("optimizer")
    if not isinstance(optimizer_state, Mapping):
        raise RuntimeError("recovery selected checkpoint must contain optimizer state")
    return optimizer_state


def _record_recovery_state_check(
    summary: dict[str, object],
    digest: Any,
    *,
    phase: Literal["before_update", "after_update"],
    update: int,
    observation: Mapping[str, object],
) -> None:
    counter = f"{phase}_checks"
    current = summary.get(counter)
    if isinstance(current, bool) or not isinstance(current, int):
        raise RuntimeError("recovery per-update state-check counter is invalid")
    summary[counter] = current + 1
    if phase == "before_update" and update == 1:
        if observation.get("optimizer_state_empty") is not True:
            raise RuntimeError("the first recovery pre-update state check was not empty")
        summary["first_pre_update_state_empty"] = True
    digest.update(
        json.dumps(
            {
                "phase": phase,
                "update": update,
                "observation": dict(observation),
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _finalize_recovery_state_checks(
    summary: dict[str, object],
    digest: Any,
    *,
    completed_updates: int,
) -> None:
    before = summary.get("before_update_checks")
    after = summary.get("after_update_checks")
    if before != completed_updates or after != completed_updates:
        raise RuntimeError("recovery AdamW did not receive one before/after state check per update")
    if summary.get("first_pre_update_state_empty") is not True:
        raise RuntimeError("recovery first pre-update empty-state proof is missing")
    summary["completed_updates_covered"] = completed_updates
    summary["check_sequence_sha256"] = digest.hexdigest()
    summary["all_updates_checked_before_and_after"] = True
    summary["all_subsequent_pre_update_scalar_steps_exact"] = True
    summary["all_post_update_scalar_steps_exact"] = True
    summary["exp_avg_and_exp_avg_sq_shape_dtype_device_valid"] = True


def _install_recovery_optimizer(
    trainer: Any,
    protocol: AdamWRecoveryProtocol,
) -> tuple[torch.optim.AdamW, dict[str, object]]:
    """Install one explicit fresh AdamW instance before the first update."""

    model = getattr(trainer, "model", None)
    weight = getattr(model, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("recovery trainer must expose model.weight")
    if weight.dtype != torch.float32:
        raise ValueError("recovery reward head must have torch.float32 dtype")
    if bool(torch.count_nonzero(weight.detach())):
        raise ValueError("recovery optimizer requires an exact zero-initialized head")
    previous = getattr(trainer, "optimizer", None)
    if not isinstance(previous, torch.optim.AdamW):
        raise TypeError("recovery protocol requires the trainer's declared optimizer to be AdamW")
    if previous.state:
        raise ValueError("recovery protocol requires fresh empty optimizer state")
    previous_parameters = [
        parameter for group in previous.param_groups for parameter in group["params"]
    ]
    if len(previous_parameters) != 1 or previous_parameters[0] is not weight:
        raise ValueError("recovery optimizer must contain exactly the reward head")
    previous_group = previous.param_groups[0]
    if float(previous_group["lr"]) != protocol.stages[0].learning_rate:
        raise ValueError("trainer learning rate does not match recovery schedule update 1")
    if float(previous_group["weight_decay"]) != 0.0:
        raise ValueError("recovery AdamW requires zero weight decay")

    optimizer = torch.optim.AdamW(
        [weight],
        lr=protocol.stages[0].learning_rate,
        betas=protocol.betas,
        eps=protocol.eps,
        weight_decay=0.0,
        amsgrad=protocol.amsgrad,
        maximize=protocol.maximize,
        foreach=protocol.foreach,
        capturable=protocol.capturable,
        differentiable=protocol.differentiable,
        fused=protocol.fused,
    )
    trainer.optimizer = optimizer
    group = optimizer.param_groups[0]
    expected = {
        "lr": protocol.stages[0].learning_rate,
        "betas": protocol.betas,
        "eps": protocol.eps,
        "weight_decay": 0.0,
        "amsgrad": protocol.amsgrad,
        "maximize": protocol.maximize,
        "foreach": protocol.foreach,
        "capturable": protocol.capturable,
        "differentiable": protocol.differentiable,
        "fused": protocol.fused,
    }
    for parameter_name, expected_value in expected.items():
        if group.get(parameter_name) != expected_value:
            raise RuntimeError(f"constructed recovery AdamW field {parameter_name!r} is not locked")
    evidence = {
        "schema_version": "deterministic-adamw-lr-decay-execution/v2",
        "protocol": protocol.to_dict(),
        "optimizer_class": "torch.optim.AdamW",
        "parameter_count": 1,
        "fresh_optimizer_state_before_first_update": len(optimizer.state) == 0,
        "reward_head_dtype_observed": str(weight.dtype),
        "first_order_audit_dtype_required": protocol.first_order_audit_dtype,
        "microbatch_order": protocol.microbatch_order,
        "one_optimizer_update_per_step": protocol.one_optimizer_update_per_step,
        "learning_rate_set_immediately_before_every_update": True,
        "single_optimizer_instance_for_all_updates": True,
        "optimizer_state_reset_at_lr_milestone": False,
        "adamw_moments_preserved_at_learning_rate_boundaries": True,
        "boundary_transitions": [],
        "completed_updates_observed": None,
        "per_update_state_checks": {
            "schema_version": "recovery-adamw-per-update-state-checks/v1",
            "before_update_checks": 0,
            "after_update_checks": 0,
            "first_pre_update_state_empty": False,
            "completed_updates_covered": None,
            "check_sequence_sha256": None,
            "all_updates_checked_before_and_after": False,
            "all_subsequent_pre_update_scalar_steps_exact": False,
            "all_post_update_scalar_steps_exact": False,
            "exp_avg_and_exp_avg_sq_shape_dtype_device_valid": False,
        },
        "selected_primary_optimizer_state_restored_and_verified": False,
        "selected_optimizer_object_identity_preserved": False,
        "selected_optimizer_moments_restored_and_verified": False,
        "selected_head_sha256": None,
        "restored_head_sha256": None,
        "selected_optimizer_state_sha256": None,
        "restored_optimizer_state_sha256": None,
        "selected_checkpoint_optimizer_state_dict_sha256": None,
        "restored_optimizer_state_dict_sha256": None,
        "selected_checkpoint_sha256": None,
        "test_or_validation_data_accessed": False,
    }
    return optimizer, evidence


def _solution_identification_evidence(
    rank_diagnostic: Mapping[str, object] | None = None,
    *,
    tie_break: str = _LEGACY_TIE_BREAK,
) -> dict[str, object]:
    return {
        "initialization": "exact_zero_head",
        "tie_break": tie_break,
        "primary_iterate_selection": (
            "first_scheduled_iterate_completing_the_sustained_first_order_gate"
        ),
        "validation_or_test_checkpoint_selection": False,
        "objective_value_checkpoint_selection": False,
        "minimum_norm_projection_applied": False,
        "minimum_norm_solution_claimed": False,
        "unique_reward_head_solution_claimed": False,
        "optional_objective_rank_diagnostic": (
            {"evaluated": False}
            if rank_diagnostic is None
            else {
                "evaluated": True,
                "evidence": _strict_json_copy(
                    rank_diagnostic,
                    name="rank_diagnostic",
                ),
            }
        ),
        "minimum_norm_note": (
            "exact_zero_initialization_and_the_AdamW_path_are_reported; "
            "zero initialization alone does not prove an Euclidean "
            "minimum-norm solution under adaptive preconditioning"
        ),
    }


def _convergence_failure_evidence(
    *,
    objective_name: str,
    spec: FirstOrderConvergenceSpec,
    fixed_snapshot_steps: int,
    initial: _FirstOrderMeasurement,
    checks: Sequence[Mapping[str, object]],
    fixed_snapshot: Mapping[str, object] | None,
    legacy_boundary_snapshot: Mapping[str, object] | None = None,
    optimizer_protocol_execution: Mapping[str, object] | None = None,
    rank_diagnostic: Mapping[str, object] | None = None,
) -> dict[str, object]:
    recovery = spec.optimizer_protocol is not None
    evidence: dict[str, object] = {
        "schema_version": (
            "objective-first-order-convergence/v2"
            if recovery
            else "objective-first-order-convergence/v1"
        ),
        "objective": objective_name,
        "converged": False,
        "fail_closed": True,
        "spec": spec.to_dict(),
        "gradient_ratio_formula": (
            "||full_data_unclipped_gradient(w_t)||_2 / "
            "max(||full_data_unclipped_gradient(w_zero)||_2, denominator_floor)"
        ),
        "initial_zero_head_measurement": initial.to_dict(
            include_audit_dtype=recovery,
        ),
        "checks": [dict(check) for check in checks],
        "selected_primary_step": None,
        "final_gate": None,
        "fixed_step_compute_matched_snapshot": (
            None if fixed_snapshot is None else dict(fixed_snapshot)
        ),
        "fixed_step_snapshot_steps": fixed_snapshot_steps,
        "solution_identification": _solution_identification_evidence(
            rank_diagnostic,
            tie_break=(
                spec.optimizer_protocol.tie_break
                if spec.optimizer_protocol is not None
                else _LEGACY_TIE_BREAK
            ),
        ),
        "test_or_validation_data_accessed": False,
    }
    if recovery:
        evidence["legacy_constant_lr_boundary_snapshot"] = (
            None if legacy_boundary_snapshot is None else dict(legacy_boundary_snapshot)
        )
        evidence["optimizer_protocol_execution"] = (
            None if optimizer_protocol_execution is None else dict(optimizer_protocol_execution)
        )
    return evidence


def _run_trainer_to_first_order_convergence(
    trainer: Any,
    *,
    audit: Callable[[], _FirstOrderMeasurement],
    spec: FirstOrderConvergenceSpec,
    fixed_snapshot_steps: int,
    objective_name: str,
    rank_diagnostic: Mapping[str, object] | None = None,
) -> _ConvergedTrainingRun:
    """Train one objective to its own gate and retain a diagnostic fixed path.

    If the primary gate is reached before a required diagnostic snapshot,
    training continues only long enough to materialize every such snapshot
    and the exact primary trainer state is then restored.  Hence neither the
    720-step state nor the recovery-only 5760-step legacy boundary can select
    or replace an earlier converged primary head.
    """

    if not isinstance(spec, FirstOrderConvergenceSpec):
        raise TypeError("spec must be FirstOrderConvergenceSpec")
    fixed_steps = _positive_integer(
        fixed_snapshot_steps,
        name="fixed_snapshot_steps",
    )
    if fixed_steps > spec.max_steps:
        raise ValueError("fixed_snapshot_steps must not exceed convergence max_steps")
    if not isinstance(objective_name, str) or not objective_name:
        raise ValueError("objective_name must be non-empty")
    if getattr(trainer, "completed_steps", None) != 0:
        raise ValueError("convergence training requires a fresh zero-step trainer")
    model = getattr(trainer, "model", None)
    weight = getattr(model, "weight", None)
    if not isinstance(weight, torch.Tensor) or bool(torch.count_nonzero(weight.detach())):
        raise ValueError("convergence training requires an exact zero-initialized head")

    protocol = spec.optimizer_protocol
    recovery_optimizer: torch.optim.AdamW | None = None
    optimizer_protocol_execution: dict[str, object] | None = None
    recovery_state_check_digest: Any | None = None
    recovery_state_check_summary: dict[str, object] | None = None
    if protocol is not None:
        recovery_optimizer, optimizer_protocol_execution = _install_recovery_optimizer(
            trainer,
            protocol,
        )
        raw_summary = optimizer_protocol_execution.get("per_update_state_checks")
        if not isinstance(raw_summary, dict):
            raise RuntimeError("recovery per-update state-check summary was not initialized")
        recovery_state_check_summary = raw_summary
        recovery_state_check_digest = hashlib.sha256()

    initial = audit()
    if protocol is not None and initial.audit_dtype != protocol.first_order_audit_dtype:
        raise RuntimeError("recovery initial first-order audit did not execute in float64")
    checks: list[dict[str, object]] = []
    consecutive_passes = 0
    selected_state: Mapping[str, object] | None = None
    selected_measurement: _FirstOrderMeasurement | None = None
    selected_step: int | None = None
    selected_head_sha256: str | None = None
    selected_optimizer_state_sha256: str | None = None
    selected_checkpoint_optimizer_state_dict_sha256: str | None = None
    selected_checkpoint_sha256: str | None = None
    fixed_snapshot: dict[str, object] | None = None
    legacy_boundary_snapshot: dict[str, object] | None = None

    while trainer.completed_steps < spec.max_steps:
        next_update = int(trainer.completed_steps) + 1
        learning_rate_used: float | None = None
        if protocol is not None:
            if (
                recovery_optimizer is None
                or optimizer_protocol_execution is None
                or recovery_state_check_digest is None
                or recovery_state_check_summary is None
            ):
                raise RuntimeError("recovery optimizer protocol was not initialized")
            if trainer.optimizer is not recovery_optimizer:
                raise RuntimeError("trainer replaced the locked recovery AdamW instance")
            learning_rate_used = protocol.learning_rate_for_update(next_update)
            previous_update = next_update - 1
            previous_learning_rate_expected = protocol.learning_rate_for_update(
                max(1, previous_update)
            )
            before_update_state = _validate_recovery_adamw_state(
                recovery_optimizer,
                weight,
                protocol,
                expected_completed_updates=previous_update,
                expected_learning_rate=previous_learning_rate_expected,
            )
            _record_recovery_state_check(
                recovery_state_check_summary,
                recovery_state_check_digest,
                phase="before_update",
                update=next_update,
                observation=before_update_state,
            )
            groups = recovery_optimizer.param_groups
            if len(groups) != 1:
                raise RuntimeError("recovery AdamW must retain exactly one parameter group")
            group = groups[0]
            previous_learning_rate = float(group["lr"])
            transition = next(
                (stage for stage in protocol.stages[1:] if stage.first_update == next_update),
                None,
            )
            moment_sha256_before = (
                _optimizer_moment_state_sha256(recovery_optimizer, weight)
                if transition is not None
                else None
            )
            # This assignment happens for every update, including within a
            # stage.  The optimizer instance and its state are never replaced.
            group["lr"] = learning_rate_used
            if float(group["lr"]) != learning_rate_used:
                raise RuntimeError("recovery learning rate was not set before the update")
            _recovery_adamw_group_payload(
                recovery_optimizer,
                weight,
                protocol,
                expected_learning_rate=learning_rate_used,
            )
            if transition is not None:
                moment_sha256_after = _optimizer_moment_state_sha256(
                    recovery_optimizer,
                    weight,
                )
                if moment_sha256_before != moment_sha256_after:
                    raise RuntimeError(
                        "setting the recovery learning rate changed AdamW moment state"
                    )
                boundary_transitions = optimizer_protocol_execution["boundary_transitions"]
                if not isinstance(boundary_transitions, list):
                    raise RuntimeError("recovery transition evidence is not mutable")
                boundary_transitions.append(
                    {
                        "next_update": next_update,
                        "previous_learning_rate": previous_learning_rate,
                        "new_learning_rate": learning_rate_used,
                        "moment_state_sha256_before_lr_assignment": (moment_sha256_before),
                        "moment_state_sha256_after_lr_assignment": (moment_sha256_after),
                        "same_optimizer_instance": True,
                        "moments_preserved": True,
                    }
                )
        diagnostic = trainer.step()
        if not isinstance(diagnostic, TrainingStepDiagnostics):
            raise TypeError("trainer.step() must return TrainingStepDiagnostics")
        if protocol is not None and trainer.optimizer is not recovery_optimizer:
            raise RuntimeError("trainer replaced recovery AdamW during an update")
        step = int(trainer.completed_steps)
        if protocol is not None and step != next_update:
            raise RuntimeError(
                "recovery trainer did not perform exactly one optimizer update per step"
            )
        if protocol is not None:
            if (
                recovery_optimizer is None
                or recovery_state_check_digest is None
                or recovery_state_check_summary is None
                or learning_rate_used is None
            ):
                raise RuntimeError("recovery optimizer state checks were not initialized")
            after_update_state = _validate_recovery_adamw_state(
                recovery_optimizer,
                weight,
                protocol,
                expected_completed_updates=next_update,
                expected_learning_rate=learning_rate_used,
            )
            _record_recovery_state_check(
                recovery_state_check_summary,
                recovery_state_check_digest,
                phase="after_update",
                update=next_update,
                observation=after_update_state,
            )
        if diagnostic.step != step:
            raise RuntimeError("trainer diagnostic step does not match trainer state")

        scheduled = step % spec.check_interval == 0
        needs_snapshot = step == fixed_steps
        needs_legacy_boundary_snapshot = (
            protocol is not None and step == protocol.legacy_boundary_snapshot_steps
        )
        measurement: _FirstOrderMeasurement | None = None
        if (
            (scheduled and selected_state is None)
            or needs_snapshot
            or needs_legacy_boundary_snapshot
        ):
            measurement = audit()
            if protocol is not None and measurement.audit_dtype != protocol.first_order_audit_dtype:
                raise RuntimeError(
                    "recovery scheduled first-order audit did not execute in float64"
                )

        if scheduled and selected_state is None:
            if measurement is None:
                raise RuntimeError("scheduled convergence audit was not evaluated")
            ratio = _gradient_ratio(
                measurement.gradient_l2_norm,
                initial.gradient_l2_norm,
                spec,
            )
            eligible = step >= spec.min_steps
            passed = eligible and ratio <= spec.gradient_ratio_tolerance
            consecutive_passes = consecutive_passes + 1 if passed else 0
            check = {
                "step": step,
                "post_update": True,
                "full_data": True,
                "gradient_clipping_applied": False,
                "measurement": measurement.to_dict(
                    include_audit_dtype=protocol is not None,
                ),
                "gradient_ratio_to_zero_initialization": ratio,
                "eligible_after_min_steps": eligible,
                "threshold_passed": passed,
                "consecutive_threshold_passes": consecutive_passes,
            }
            if learning_rate_used is not None:
                check["learning_rate_used_for_update"] = learning_rate_used
                check["learning_rate_schedule_sha256"] = protocol.schedule_sha256
            checks.append(check)
            if consecutive_passes >= spec.consecutive_checks:
                candidate_state = trainer.state_dict()
                if not isinstance(candidate_state, Mapping):
                    raise TypeError("recovery trainer state_dict() must return a mapping")
                if protocol is not None:
                    if recovery_optimizer is None or learning_rate_used is None:
                        raise RuntimeError("recovery optimizer was not available at selection")
                    selected_head_sha256 = _tensor_sha256(weight)
                    selected_optimizer_state_sha256 = _recovery_optimizer_state_sha256(
                        recovery_optimizer,
                        weight,
                        protocol,
                        expected_completed_updates=step,
                        expected_learning_rate=learning_rate_used,
                    )
                    checkpoint_optimizer = _checkpoint_optimizer_state_dict(candidate_state)
                    selected_checkpoint_optimizer_state_dict_sha256 = _checkpoint_value_sha256(
                        checkpoint_optimizer
                    )
                    live_optimizer_state_dict_sha256 = _checkpoint_value_sha256(
                        recovery_optimizer.state_dict()
                    )
                    if (
                        selected_checkpoint_optimizer_state_dict_sha256
                        != live_optimizer_state_dict_sha256
                    ):
                        raise RuntimeError(
                            "selected recovery checkpoint optimizer state differs "
                            "from the live optimizer"
                        )
                    selected_checkpoint_sha256 = _canonical_sha256(
                        {
                            "schema_version": "selected-recovery-state-binding/v1",
                            "completed_updates": step,
                            "head_sha256": selected_head_sha256,
                            "optimizer_state_sha256": selected_optimizer_state_sha256,
                            "optimizer_state_dict_sha256": (
                                selected_checkpoint_optimizer_state_dict_sha256
                            ),
                        }
                    )
                selected_state = candidate_state
                selected_measurement = measurement
                selected_step = step

        if needs_snapshot:
            if measurement is None:
                raise RuntimeError("fixed-step snapshot audit was not evaluated")
            fixed_snapshot = {
                "schema_version": "fixed-step-compute-matched-snapshot/v1",
                "step": step,
                "head_sha256": _tensor_sha256(trainer.model.weight),
                "measurement": measurement.to_dict(
                    include_audit_dtype=protocol is not None,
                ),
                "gradient_ratio_to_zero_initialization": _gradient_ratio(
                    measurement.gradient_l2_norm,
                    initial.gradient_l2_norm,
                    spec,
                ),
                "history_summary": _history_summary(tuple(trainer.history)),
                "role": "compute_matched_and_pilot_diagnostic_only",
                "used_as_primary_selection_rule": False,
                "coincides_with_selected_primary_iterate": (selected_step == fixed_steps),
            }

        if needs_legacy_boundary_snapshot:
            if measurement is None or protocol is None:
                raise RuntimeError("legacy-boundary snapshot audit was not evaluated")
            legacy_boundary_snapshot = {
                "schema_version": "legacy-constant-lr-boundary-snapshot/v1",
                "step": step,
                "head_sha256": _tensor_sha256(trainer.model.weight),
                "measurement": measurement.to_dict(include_audit_dtype=True),
                "gradient_ratio_to_zero_initialization": _gradient_ratio(
                    measurement.gradient_l2_norm,
                    initial.gradient_l2_norm,
                    spec,
                ),
                "history_summary": _history_summary(tuple(trainer.history)),
                "learning_rate_used_for_update": learning_rate_used,
                "learning_rate_schedule_sha256": protocol.schedule_sha256,
                "role": "immutable_legacy_constant_lr_failure_boundary_diagnostic",
                "used_as_primary_selection_rule": False,
                "coincides_with_selected_primary_iterate": (
                    selected_step == protocol.legacy_boundary_snapshot_steps
                ),
                "test_or_validation_data_accessed": False,
            }

        all_required_snapshots_observed = fixed_snapshot is not None and (
            protocol is None or legacy_boundary_snapshot is not None
        )
        if selected_state is not None and all_required_snapshots_observed:
            break

    if fixed_snapshot is None:
        raise RuntimeError("fixed-step compute-matched snapshot was not reached")
    if protocol is not None and legacy_boundary_snapshot is None:
        raise RuntimeError("recovery legacy-boundary snapshot was not reached")
    controller_updates_executed = int(trainer.completed_steps)
    if optimizer_protocol_execution is not None:
        optimizer_protocol_execution["completed_updates_observed"] = controller_updates_executed
        if recovery_state_check_summary is None or recovery_state_check_digest is None:
            raise RuntimeError("recovery optimizer state-check evidence is missing")
        _finalize_recovery_state_checks(
            recovery_state_check_summary,
            recovery_state_check_digest,
            completed_updates=controller_updates_executed,
        )
    if selected_state is None or selected_measurement is None or selected_step is None:
        evidence = _convergence_failure_evidence(
            objective_name=objective_name,
            spec=spec,
            fixed_snapshot_steps=fixed_steps,
            initial=initial,
            checks=checks,
            fixed_snapshot=fixed_snapshot,
            legacy_boundary_snapshot=legacy_boundary_snapshot,
            optimizer_protocol_execution=optimizer_protocol_execution,
            rank_diagnostic=rank_diagnostic,
        )
        raise OptimizationConvergenceError(
            f"{objective_name} did not satisfy the sustained first-order "
            f"gradient-ratio gate by {spec.max_steps} steps",
            evidence=evidence,
        )

    if protocol is not None:
        if (
            recovery_optimizer is None
            or optimizer_protocol_execution is None
            or selected_head_sha256 is None
            or selected_optimizer_state_sha256 is None
            or selected_checkpoint_optimizer_state_dict_sha256 is None
            or selected_checkpoint_sha256 is None
        ):
            raise RuntimeError("selected recovery optimizer checkpoint evidence is incomplete")
        if (
            _checkpoint_value_sha256(_checkpoint_optimizer_state_dict(selected_state))
            != selected_checkpoint_optimizer_state_dict_sha256
        ):
            raise RuntimeError("selected recovery optimizer checkpoint mutated before restoration")
    trainer.load_state_dict(selected_state)
    if trainer.completed_steps != selected_step:
        raise RuntimeError("restored primary trainer step does not match selected step")
    if protocol is not None:
        if recovery_optimizer is None or optimizer_protocol_execution is None:
            raise RuntimeError("recovery optimizer restoration evidence is missing")
        if trainer.optimizer is not recovery_optimizer:
            raise RuntimeError(
                "recovery trainer replaced the optimizer object while restoring the selected state"
            )
        restored_head_sha256 = _tensor_sha256(weight)
        if restored_head_sha256 != selected_head_sha256:
            raise RuntimeError("restored recovery head differs from the selected head")
        selected_learning_rate = protocol.learning_rate_for_update(selected_step)
        restored_optimizer_state_sha256 = _recovery_optimizer_state_sha256(
            recovery_optimizer,
            weight,
            protocol,
            expected_completed_updates=selected_step,
            expected_learning_rate=selected_learning_rate,
        )
        if restored_optimizer_state_sha256 != selected_optimizer_state_sha256:
            raise RuntimeError(
                "restored recovery AdamW moments or parameter-group state differ "
                "from the selected optimizer state"
            )
        restored_optimizer_state_dict_sha256 = _checkpoint_value_sha256(
            recovery_optimizer.state_dict()
        )
        if restored_optimizer_state_dict_sha256 != selected_checkpoint_optimizer_state_dict_sha256:
            raise RuntimeError(
                "restored recovery optimizer state_dict differs from the selected checkpoint"
            )
        optimizer_protocol_execution.update(
            {
                "selected_primary_optimizer_state_restored_without_reconstruction": True,
                "selected_primary_optimizer_state_restored_and_verified": True,
                "selected_optimizer_object_identity_preserved": True,
                "selected_optimizer_moments_restored_and_verified": True,
                "selected_head_sha256": selected_head_sha256,
                "restored_head_sha256": restored_head_sha256,
                "selected_optimizer_state_sha256": selected_optimizer_state_sha256,
                "restored_optimizer_state_sha256": restored_optimizer_state_sha256,
                "selected_checkpoint_optimizer_state_dict_sha256": (
                    selected_checkpoint_optimizer_state_dict_sha256
                ),
                "restored_optimizer_state_dict_sha256": (restored_optimizer_state_dict_sha256),
                "selected_checkpoint_sha256": selected_checkpoint_sha256,
            }
        )
    final = audit()
    if protocol is not None and final.audit_dtype != protocol.first_order_audit_dtype:
        raise RuntimeError("recovery final first-order audit did not execute in float64")
    final_ratio = _gradient_ratio(
        final.gradient_l2_norm,
        initial.gradient_l2_norm,
        spec,
    )
    if final_ratio > spec.gradient_ratio_tolerance:
        raise OptimizationConvergenceError(
            f"{objective_name} failed its restored final first-order gate",
            evidence=_convergence_failure_evidence(
                objective_name=objective_name,
                spec=spec,
                fixed_snapshot_steps=fixed_steps,
                initial=initial,
                checks=checks,
                fixed_snapshot=fixed_snapshot,
                legacy_boundary_snapshot=legacy_boundary_snapshot,
                optimizer_protocol_execution=optimizer_protocol_execution,
                rank_diagnostic=rank_diagnostic,
            ),
        )
    final_gate = {
        "step": selected_step,
        "measurement": final.to_dict(
            include_audit_dtype=protocol is not None,
        ),
        "gradient_ratio_to_zero_initialization": final_ratio,
        "threshold_passed": True,
        "fresh_post_restore_audit": True,
    }
    if protocol is not None:
        final_gate["learning_rate_at_selected_iterate"] = protocol.learning_rate_for_update(
            selected_step
        )
        if optimizer_protocol_execution is None:
            raise RuntimeError("recovery optimizer execution evidence is missing")
        if (
            optimizer_protocol_execution.get(
                "selected_primary_optimizer_state_restored_and_verified"
            )
            is not True
        ):
            raise RuntimeError("recovery selected optimizer state was not verified")
    evidence = {
        "schema_version": (
            "objective-first-order-convergence/v2"
            if protocol is not None
            else "objective-first-order-convergence/v1"
        ),
        "objective": objective_name,
        "converged": True,
        "fail_closed": True,
        "spec": spec.to_dict(),
        "gradient_ratio_formula": (
            "||full_data_unclipped_gradient(w_t)||_2 / "
            "max(||full_data_unclipped_gradient(w_zero)||_2, denominator_floor)"
        ),
        "initial_zero_head_measurement": initial.to_dict(
            include_audit_dtype=protocol is not None,
        ),
        "checks": checks,
        "selected_primary_step": selected_step,
        "selected_primary_head_sha256": _tensor_sha256(trainer.model.weight),
        "consecutive_threshold_passes_at_selection": spec.consecutive_checks,
        "final_gate": final_gate,
        "fixed_step_compute_matched_snapshot": fixed_snapshot,
        "fixed_step_snapshot_steps": fixed_steps,
        "fixed_step_snapshot_is_not_primary_selection": True,
        "solution_identification": _solution_identification_evidence(
            rank_diagnostic,
            tie_break=(protocol.tie_break if protocol is not None else _LEGACY_TIE_BREAK),
        ),
        "test_or_validation_data_accessed": False,
    }
    if protocol is not None:
        evidence["legacy_constant_lr_boundary_snapshot"] = legacy_boundary_snapshot
        evidence["optimizer_protocol_execution"] = optimizer_protocol_execution
    history = tuple(trainer.history)
    if len(history) != selected_step:
        raise RuntimeError("restored primary history does not match selected step")
    return _ConvergedTrainingRun(
        history=history,
        initial=initial,
        final=final,
        evidence=_strict_json_copy(evidence, name="first_order_convergence"),
    )


def _bt_first_order_measurement(trainer: BTMLETrainer) -> _FirstOrderMeasurement:
    """Evaluate exact BT objective/gradient in FP64 without clipping or updates."""

    batch = trainer.batch
    solve_dtype = torch.float64
    features = batch.feature_differences.to(dtype=solve_dtype)
    head = trainer.model.weight.detach().to(dtype=solve_dtype)
    margins = features @ head
    counts = batch.num_annotations.to(dtype=solve_dtype)
    wins = batch.left_wins.to(dtype=solve_dtype)
    objective = (
        counts * torch.nn.functional.softplus(margins) - wins * margins
    ).sum() / counts.sum()
    residual = counts * torch.sigmoid(margins) - wins
    gradient = features.mT @ residual / counts.sum()
    if not bool(torch.isfinite(objective)) or not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("BT first-order audit produced non-finite values")
    return _FirstOrderMeasurement(
        objective=float(objective.item()),
        gradient_l2_norm=float(torch.linalg.vector_norm(gradient).item()),
        inner_solver=None,
        audit_dtype="float64",
    )


def _exact_soft_bt_first_order_measurement(
    trainer: _ExactSoftLabelBTTrainer,
) -> _FirstOrderMeasurement:
    """Audit exact expected-BT CE in FP64 with no clipping, update, or sampling."""

    if not isinstance(trainer, _ExactSoftLabelBTTrainer):
        raise TypeError("trainer must be _ExactSoftLabelBTTrainer")
    features = trainer.batch.feature_differences.to(dtype=torch.float64)
    targets = trainer.batch.target_probabilities.to(dtype=torch.float64)
    head = trainer.model.weight.detach().to(dtype=torch.float64)
    margins = features @ head
    objective = (torch.nn.functional.softplus(margins) - targets * margins).mean()
    gradient = features.mT @ (torch.sigmoid(margins) - targets) / trainer.batch.num_edges
    if not bool(torch.isfinite(objective)) or not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("exact soft-label BT audit produced non-finite values")
    return _FirstOrderMeasurement(
        objective=float(objective.item()),
        gradient_l2_norm=float(torch.linalg.vector_norm(gradient).item()),
        inner_solver=None,
        audit_dtype="float64",
    )


def _prorm_first_order_measurement(
    trainer: ProRMPlusTrainer,
) -> _FirstOrderMeasurement:
    """Evaluate ProRM+ with a fresh cold-start FP64 PCG and exact envelope gradient."""

    evaluation = trainer.evaluate(use_warm_start=False)
    if not evaluation.pcg_converged:
        raise RuntimeError("fresh cold-start FP64 ProRM+ PCG audit did not converge")
    direction = evaluation.direction
    if direction.dtype != torch.float64:
        raise RuntimeError("ProRM+ first-order gate requires an FP64 dual direction")
    batch = trainer.batch
    features = batch.feature_differences.to(dtype=direction.dtype)
    edge_scores = batch.edge_scores.to(dtype=direction.dtype)
    weights = envelope_weights(
        edge_scores,
        direction,
        beta=trainer.config.beta,
        detach_direction=True,
    )
    gradient = features.mT @ weights / batch.num_edges
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("ProRM+ first-order audit produced a non-finite gradient")
    return _FirstOrderMeasurement(
        objective=float(evaluation.dual_loss),
        gradient_l2_norm=float(torch.linalg.vector_norm(gradient).item()),
        inner_solver={
            "method": "pcg",
            "dtype": "float64",
            "cold_start": True,
            "warm_start_used": False,
            "iterations": int(evaluation.pcg_iterations),
            "residual_norm": float(evaluation.pcg_residual_norm),
            "relative_residual": float(evaluation.pcg_relative_residual),
            "converged": True,
        },
        audit_dtype="float64",
    )


def _zero_model(training: TrainingTensorData) -> FrozenFeatureLinearReward:
    zero = torch.zeros(
        training.reward_dimension,
        dtype=training.reward_features.dtype,
        device=training.reward_features.device,
    )
    model = FrozenFeatureLinearReward(training.reward_dimension, zero)
    if bool(torch.count_nonzero(model.weight.detach())):
        raise RuntimeError("fresh Phase-2 reward head is not exactly zero initialized")
    return model


def _bt_config(settings: Phase2TrainingSettings) -> BTMLETrainingConfig:
    return BTMLETrainingConfig(
        learning_rate=settings.learning_rate,
        optimizer="adamw",
        weight_decay=settings.weight_decay,
        microbatch_size=settings.microbatch_size,
        max_grad_norm=settings.max_grad_norm,
    )


@dataclass(frozen=True, slots=True)
class _ExactSoftLabelBTBatch:
    """Transient train-only expected-label BT data with no sampled-label channel."""

    feature_differences: torch.Tensor
    target_probabilities: torch.Tensor

    def __post_init__(self) -> None:
        features = self.feature_differences
        probabilities = self.target_probabilities
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise TypeError("feature_differences must be a rank-two torch.Tensor")
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("target_probabilities must be a torch.Tensor")
        if probabilities.shape != (features.shape[0],):
            raise ValueError("target_probabilities must align with canonical train edges")
        if features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError("exact soft-label BT dimensions must be positive")
        for name, value in (
            ("feature_differences", features),
            ("target_probabilities", probabilities),
        ):
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError(f"{name} must be frozen and detached")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if probabilities.dtype != features.dtype or probabilities.device != features.device:
            raise ValueError(
                "target_probabilities must share dtype and device with feature_differences"
            )
        if bool(((probabilities <= 0.0) | (probabilities >= 1.0)).any()):
            raise ValueError(
                "exact soft-label BT probabilities must be strictly between zero and one"
            )

    @classmethod
    def from_exact_margin_training(
        cls,
        training: TrainingTensorData,
    ) -> _ExactSoftLabelBTBatch:
        """Apply sigmoid exactly once to canonical transformed-oracle margins."""

        if not isinstance(training, TrainingTensorData):
            raise TypeError("training must be TrainingTensorData")
        features = (training.reward_features[:, 0] - training.reward_features[:, 1]).detach()
        probabilities = torch.sigmoid(training.h).detach()
        return cls(
            feature_differences=features.clone(),
            target_probabilities=probabilities.clone(),
        )

    @property
    def num_edges(self) -> int:
        return int(self.feature_differences.shape[0])

    @property
    def reward_dimension(self) -> int:
        return int(self.feature_differences.shape[1])


class _ExactSoftLabelBTTrainer:
    """Deterministic AdamW trainer for exact expected Bernoulli cross-entropy."""

    def __init__(
        self,
        model: FrozenFeatureLinearReward,
        batch: _ExactSoftLabelBTBatch,
        config: BTMLETrainingConfig,
    ) -> None:
        if not isinstance(model, FrozenFeatureLinearReward):
            raise TypeError("model must be FrozenFeatureLinearReward")
        if not isinstance(batch, _ExactSoftLabelBTBatch):
            raise TypeError("batch must be _ExactSoftLabelBTBatch")
        if not isinstance(config, BTMLETrainingConfig):
            raise TypeError("config must be BTMLETrainingConfig")
        if model.feature_dimension != batch.reward_dimension:
            raise ValueError("model and exact soft-label batch dimensions do not match")
        if (
            model.weight.dtype != batch.feature_differences.dtype
            or model.weight.device != batch.feature_differences.device
        ):
            raise ValueError("model and exact soft-label batch must share dtype and device")
        self.model = model
        self.batch = batch
        self.config = config
        if config.optimizer == "adamw":
            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                [model.weight],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                [model.weight],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.completed_steps = 0
        self.history: list[TrainingStepDiagnostics] = []

    def step(self) -> TrainingStepDiagnostics:
        """Accumulate the exact full-data expected-BT gradient and update once."""

        self.optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros(
            (),
            dtype=self.model.weight.dtype,
            device=self.model.weight.device,
        )
        microbatch_size = (
            self.batch.num_edges
            if self.config.microbatch_size is None
            else self.config.microbatch_size
        )
        for index in _training_slices(self.batch.num_edges, microbatch_size):
            margins = self.batch.feature_differences[index] @ self.model.weight
            target = self.batch.target_probabilities[index]
            loss = (
                torch.nn.functional.softplus(margins) - target * margins
            ).sum() / self.batch.num_edges
            loss.backward()
            objective = objective + loss.detach()
        gradient = self.model.weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("exact soft-label BT produced a non-finite gradient")
        gradient_norm = float(torch.linalg.vector_norm(gradient.detach()).item())
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [self.model.weight],
                max_norm=float(self.config.max_grad_norm),
                error_if_nonfinite=True,
            )
        self.optimizer.step()
        if not bool(torch.isfinite(self.model.weight).all()):
            raise FloatingPointError("exact soft-label BT produced a non-finite head")
        self.completed_steps += 1
        diagnostic = TrainingStepDiagnostics(
            step=self.completed_steps,
            objective=float(objective.item()),
            gradient_norm=gradient_norm,
        )
        self.history.append(diagnostic)
        return diagnostic

    def state_dict(self) -> dict[str, object]:
        """Return an in-memory checkpoint used only by the convergence controller."""

        return {
            "schema_version": "exact-soft-label-bt-checkpoint/v1",
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "completed_steps": self.completed_steps,
            "history": tuple(self.history),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {
            "schema_version",
            "model",
            "optimizer",
            "completed_steps",
            "history",
        }:
            raise ValueError("invalid exact soft-label BT checkpoint keys")
        if state["schema_version"] != "exact-soft-label-bt-checkpoint/v1":
            raise ValueError("invalid exact soft-label BT checkpoint schema")
        model_state = state["model"]
        optimizer_state = state["optimizer"]
        if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
            raise TypeError("exact soft-label BT checkpoint states must be mappings")
        completed_steps = state["completed_steps"]
        if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
            raise TypeError("exact soft-label BT checkpoint step must be an integer")
        history = state["history"]
        if not isinstance(history, tuple) or not all(
            isinstance(item, TrainingStepDiagnostics) for item in history
        ):
            raise TypeError("exact soft-label BT checkpoint history is invalid")
        if len(history) != completed_steps:
            raise ValueError("exact soft-label BT checkpoint history does not match its step")
        self.model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(dict(optimizer_state))
        self.completed_steps = completed_steps
        self.history = list(history)
        if not bool(torch.isfinite(self.model.weight).all()):
            raise FloatingPointError("restored exact soft-label BT head is non-finite")


def _prorm_config(
    settings: Phase2TrainingSettings,
    *,
    absolute_damping: float,
) -> ProRMPlusTrainingConfig:
    return ProRMPlusTrainingConfig(
        learning_rate=settings.learning_rate,
        optimizer="adamw",
        weight_decay=settings.weight_decay,
        microbatch_size=settings.microbatch_size,
        max_grad_norm=settings.max_grad_norm,
        beta=settings.training_beta,
        damping=absolute_damping,
        pcg_dtype=settings.pcg_dtype,
        pcg_max_iterations=settings.pcg_max_iterations,
        pcg_tolerance=settings.pcg_tolerance,
        pcg_absolute_tolerance=settings.pcg_absolute_tolerance,
        pcg_residual_recompute_interval=settings.pcg_residual_recompute_interval,
        require_pcg_convergence=True,
    )


def _head_tuple(model: FrozenFeatureLinearReward) -> tuple[float, ...]:
    weight = model.weight.detach()
    if weight.requires_grad or not bool(torch.isfinite(weight).all()):
        raise RuntimeError("trained head is not finite and detached")
    return tuple(float(value) for value in weight.to(device="cpu"))


def _pcg_evidence(evaluation: object) -> dict[str, object]:
    if isinstance(evaluation, Mapping):
        iterations = evaluation["iterations"]
        residual_norm = evaluation["residual_norm"]
        relative_residual = evaluation["relative_residual"]
        converged = evaluation["converged"]
    else:
        iterations = evaluation.pcg_iterations
        residual_norm = evaluation.pcg_residual_norm
        relative_residual = evaluation.pcg_relative_residual
        converged = evaluation.pcg_converged
    return {
        "iterations": int(iterations),
        "residual_norm": float(residual_norm),
        "relative_residual": float(relative_residual),
        "converged": bool(converged),
        "cold_start": True,
    }


def _make_head_evidence(
    *,
    arm: str,
    method: str,
    model: FrozenFeatureLinearReward,
    initial_head_sha256: str,
    initial_objective: float,
    final_objective: float,
    history: Sequence[TrainingStepDiagnostics],
    final_pcg: Mapping[str, object] | None,
    first_order_convergence: Mapping[str, object],
) -> TrainedHeadEvidence:
    return TrainedHeadEvidence(
        arm=arm,
        method=method,
        head_weight=_head_tuple(model),
        head_dtype=str(model.weight.dtype),
        initial_head_sha256=initial_head_sha256,
        head_sha256=_tensor_sha256(model.weight),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        history_summary=_history_summary(history),
        final_pcg=final_pcg,
        first_order_convergence=first_order_convergence,
    )


def _train_exact_soft_label_bt_control(
    exact_margin_training: TrainingTensorData,
    *,
    source_node_rewards_sha256: str,
    exact_margin_sha256: str,
    settings: Phase2TrainingSettings,
    rank_diagnostic: Mapping[str, object],
) -> ExactSoftLabelBTControl:
    """Fit expected-label BT from train oracle margins without an RNG input."""

    _validate_digest(
        source_node_rewards_sha256,
        name="source_node_rewards_sha256",
    )
    _validate_digest(exact_margin_sha256, name="exact_margin_sha256")
    if _tensor_sha256(exact_margin_training.h) != exact_margin_sha256:
        raise ValueError("exact soft-label BT margin digest does not match its train input")
    if settings.exact_soft_label_bt_enabled is not True:
        raise ValueError("exact soft-label BT diagnostic is disabled")

    batch = _ExactSoftLabelBTBatch.from_exact_margin_training(exact_margin_training)
    model = _zero_model(exact_margin_training)
    initial_head_sha = _tensor_sha256(model.weight)
    trainer = _ExactSoftLabelBTTrainer(model, batch, _bt_config(settings))
    convergence = _run_trainer_to_first_order_convergence(
        trainer,
        audit=partial(_exact_soft_bt_first_order_measurement, trainer),
        spec=settings.convergence,
        fixed_snapshot_steps=settings.outer_steps,
        objective_name="exact_soft_label_bt_cross_entropy",
        rank_diagnostic=rank_diagnostic,
    )
    head = _make_head_evidence(
        arm=EXACT_SOFT_BT_ARM,
        method=BT_MLE,
        model=model,
        initial_head_sha256=initial_head_sha,
        initial_objective=convergence.initial.objective,
        final_objective=convergence.final.objective,
        history=convergence.history,
        final_pcg=None,
        first_order_convergence=convergence.evidence,
    )
    fresh_final = _exact_soft_bt_first_order_measurement(trainer)
    if not math.isclose(
        fresh_final.objective,
        head.final_objective,
        rel_tol=2.0e-7,
        abs_tol=2.0e-9,
    ):
        raise RuntimeError("exact soft-label BT final objective is not bound to the saved head")
    if _tensor_sha256(model.weight) != head.head_sha256:
        raise RuntimeError("exact soft-label BT saved head digest changed during its audit")

    feature_sha = _tensor_sha256(batch.feature_differences)
    target_sha = _tensor_sha256(batch.target_probabilities)
    target_audit = {
        "schema_version": "exact-soft-label-bt-target/v1",
        "split": "train",
        "orientation": "candidate_0_minus_candidate_1",
        "input": settings.exact_soft_label_bt_input,
        "target_construction": "p_star = sigmoid(delta_r_star)",
        "source_node_rewards_sha256": source_node_rewards_sha256,
        "canonical_margin_sha256": exact_margin_sha256,
        "target_probability_sha256": target_sha,
        "reward_feature_difference_sha256": feature_sha,
        "num_canonical_edges": batch.num_edges,
        "reward_dimension": batch.reward_dimension,
        "same_reward_features_and_canonical_edges_as": "exact_margin_prorm_plus",
        "noise_free": settings.exact_soft_label_bt_noise_free,
        "bernoulli_sampling_used": False,
        "sampled_label_stream_accessed": False,
        "raw_target_probabilities_retained": False,
        "raw_oracle_margins_retained": False,
        "raw_node_rewards_retained": False,
        "test_or_validation_data_accessed": False,
        "role": settings.exact_soft_label_bt_role,
        "eligible_for_primary_claim": (settings.exact_soft_label_bt_eligible_for_primary_claim),
    }
    final_ratio = head.first_order_convergence["final_gate"][
        "gradient_ratio_to_zero_initialization"
    ]
    optimization_audit = {
        "schema_version": "exact-soft-label-bt-optimization/v1",
        "objective": "mean(softplus(delta_r_phi) - p_star * delta_r_phi)",
        "objective_name": "exact_soft_label_bt_cross_entropy",
        "optimizer": settings.optimizer,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "microbatch_size": settings.microbatch_size,
        "max_grad_norm": settings.max_grad_norm,
        "fresh_zero_initialized_bias_free_linear_head": True,
        "head_sha256": head.head_sha256,
        "target_probability_sha256": target_sha,
        "reward_feature_difference_sha256": feature_sha,
        "initial_objective": head.initial_objective,
        "final_objective": head.final_objective,
        "objective_change_final_minus_initial": (head.final_objective - head.initial_objective),
        "final_full_data_unclipped_gradient_l2_norm": (fresh_final.gradient_l2_norm),
        "final_gradient_ratio_to_zero_initialization": final_ratio,
        "first_order_convergence_passed": True,
        "fixed_720_step_checkpoint_role": ("compute_matched_and_pilot_diagnostic_only"),
        "fixed_720_step_checkpoint_used_for_head_selection": False,
        "favorable_ordering_gate_applied": False,
        "pilot_measure_only": settings.stage == "pilot",
        "eligible_for_primary_claim": False,
        "sampled_label_stream_accessed": False,
        "test_or_validation_data_accessed": False,
        "saved_head_mutated_by_audit": False,
    }
    control = ExactSoftLabelBTControl(
        head=head,
        target_audit=target_audit,
        optimization_audit=optimization_audit,
    )
    control.to_dict()
    return control


@dataclass(frozen=True, slots=True)
class _DensePseudoinverseGeometry:
    """Fixed ridge-free policy geometry for the low-dimensional control."""

    edge_scores: torch.Tensor
    node_scores: torch.Tensor
    h: torch.Tensor
    fisher: torch.Tensor
    pseudoinverse: torch.Tensor
    rank: int
    eigenvalue_threshold: float
    smallest_retained_eigenvalue: float
    largest_retained_eigenvalue: float
    smallest_eigenvalue: float
    fisher_sha256: str
    pseudoinverse_sha256: str


@dataclass(frozen=True, slots=True)
class _DenseProRMEvaluation:
    """Full-data evaluation under a fixed Moore-Penrose geometry."""

    moment: torch.Tensor
    direction: torch.Tensor
    objective: float
    saddle_value: float
    gradient_norm: float
    solve_residual_norm: float
    solve_relative_residual: float


def _training_slices(num_edges: int, microbatch_size: int) -> tuple[slice, ...]:
    size = min(num_edges, microbatch_size)
    return tuple(slice(start, min(start + size, num_edges)) for start in range(0, num_edges, size))


@torch.no_grad()
def _full_reward_margins(
    model: FrozenFeatureLinearReward,
    batch: FeatureTrainingBatch,
    microbatch_size: int,
) -> torch.Tensor:
    return torch.cat(
        [
            model.margins(
                batch.left_features[index],
                batch.right_features[index],
            )
            for index in _training_slices(batch.num_edges, microbatch_size)
        ]
    )


def _build_dense_pseudoinverse_geometry(
    training: TrainingTensorData,
    settings: Phase2TrainingSettings,
) -> tuple[FeatureTrainingBatch, _DensePseudoinverseGeometry]:
    """Build the configured truncated Moore-Penrose inverse in FP64."""

    batch = training.to_training_batch()
    solve_dtype = resolve_fisher_solve_dtype(settings.pcg_dtype)
    edge_scores = batch.edge_scores.to(dtype=solve_dtype)
    node_scores = batch.node_scores.to(dtype=solve_dtype)
    h = batch.h.to(dtype=solve_dtype)
    fisher = node_scores.mT @ node_scores / node_scores.shape[0]
    fisher = 0.5 * (fisher + fisher.mT)
    if not bool(torch.isfinite(fisher).all()):
        raise FloatingPointError("low-dimensional Fisher contains non-finite values")
    eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
    if not bool(torch.isfinite(eigenvalues).all()) or not bool(torch.isfinite(eigenvectors).all()):
        raise FloatingPointError(
            "low-dimensional Fisher eigendecomposition produced non-finite values"
        )
    largest = float(eigenvalues[-1].item())
    if not math.isfinite(largest) or largest <= 0.0:
        raise ValueError("low-dimensional Fisher has no positive eigenvalue")
    threshold = settings.low_dimensional_relative_eigenvalue_tolerance * largest
    smallest = float(eigenvalues[0].item())
    if smallest < -threshold:
        raise FloatingPointError(
            "low-dimensional empirical Fisher is materially non-positive-semidefinite"
        )
    retained = eigenvalues > threshold
    rank = int(torch.count_nonzero(retained).item())
    if rank < 1:
        raise ValueError("low-dimensional Moore-Penrose solve retained no Fisher eigenvalue")
    retained_values = eigenvalues[retained]
    retained_vectors = eigenvectors[:, retained]
    pseudoinverse = (retained_vectors / retained_values.unsqueeze(0)) @ retained_vectors.mT
    pseudoinverse = 0.5 * (pseudoinverse + pseudoinverse.mT)
    if not bool(torch.isfinite(pseudoinverse).all()):
        raise FloatingPointError("low-dimensional Moore-Penrose inverse contains non-finite values")
    return batch, _DensePseudoinverseGeometry(
        edge_scores=edge_scores.detach(),
        node_scores=node_scores.detach(),
        h=h.detach(),
        fisher=fisher.detach(),
        pseudoinverse=pseudoinverse.detach(),
        rank=rank,
        eigenvalue_threshold=threshold,
        smallest_retained_eigenvalue=float(retained_values[0].item()),
        largest_retained_eigenvalue=float(retained_values[-1].item()),
        smallest_eigenvalue=smallest,
        fisher_sha256=_tensor_sha256(fisher),
        pseudoinverse_sha256=_tensor_sha256(pseudoinverse),
    )


@torch.no_grad()
def _evaluate_dense_prorm(
    model: FrozenFeatureLinearReward,
    batch: FeatureTrainingBatch,
    geometry: _DensePseudoinverseGeometry,
    settings: Phase2TrainingSettings,
) -> _DenseProRMEvaluation:
    margins = _full_reward_margins(
        model,
        batch,
        settings.microbatch_size,
    ).to(dtype=geometry.edge_scores.dtype)
    moment = empirical_moment(
        geometry.edge_scores,
        margins,
        geometry.h,
    )
    direction = geometry.pseudoinverse @ moment
    operator_direction = geometry.fisher @ direction
    objective = dual_loss(
        moment,
        direction,
        beta=settings.training_beta,
    )
    saddle = dual_saddle_value(
        moment,
        direction,
        operator_direction,
        beta=settings.training_beta,
    )
    weights = envelope_weights(
        geometry.edge_scores,
        direction,
        beta=settings.training_beta,
        detach_direction=True,
    )
    feature_differences = batch.feature_differences.to(dtype=geometry.edge_scores.dtype)
    gradient = feature_differences.mT @ weights / batch.num_edges
    residual = moment - operator_direction
    moment_norm = float(torch.linalg.vector_norm(moment).item())
    residual_norm = float(torch.linalg.vector_norm(residual).item())
    values = (
        moment,
        direction,
        operator_direction,
        objective,
        saddle,
        weights,
        gradient,
        residual,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("low-dimensional ProRM+ evaluation produced a non-finite value")
    objective_value = float(objective.item())
    saddle_value = float(saddle.item())
    if objective_value < -1.0e-10:
        raise FloatingPointError("low-dimensional Moore-Penrose objective is materially negative")
    return _DenseProRMEvaluation(
        moment=moment.detach().clone(),
        direction=direction.detach().clone(),
        objective=objective_value,
        saddle_value=saddle_value,
        gradient_norm=float(torch.linalg.vector_norm(gradient).item()),
        solve_residual_norm=residual_norm,
        solve_relative_residual=(0.0 if moment_norm == 0.0 else residual_norm / moment_norm),
    )


class _DensePseudoinverseProRMTrainer:
    """Minimal stateful trainer for the ridge-free low-dimensional control."""

    def __init__(
        self,
        model: FrozenFeatureLinearReward,
        batch: FeatureTrainingBatch,
        geometry: _DensePseudoinverseGeometry,
        settings: Phase2TrainingSettings,
    ) -> None:
        self.model = model
        self.batch = batch
        self.geometry = geometry
        self.settings = settings
        self.optimizer = torch.optim.AdamW(
            [model.weight],
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        self.completed_steps = 0
        self.history: list[TrainingStepDiagnostics] = []

    def audit(self) -> _FirstOrderMeasurement:
        evaluation = _evaluate_dense_prorm(
            self.model,
            self.batch,
            self.geometry,
            self.settings,
        )
        return _FirstOrderMeasurement(
            objective=evaluation.objective,
            gradient_l2_norm=evaluation.gradient_norm,
            inner_solver={
                "method": "truncated_moore_penrose_pseudoinverse",
                "dtype": self.settings.pcg_dtype,
                "cold_start": True,
                "warm_start_used": False,
                "numerical_rank": self.geometry.rank,
                "relative_eigenvalue_tolerance": (
                    self.settings.low_dimensional_relative_eigenvalue_tolerance
                ),
                "solve_residual_norm": evaluation.solve_residual_norm,
                "solve_relative_residual": evaluation.solve_relative_residual,
                "converged": True,
            },
            audit_dtype="float64",
        )

    def step(self) -> TrainingStepDiagnostics:
        evaluation = _evaluate_dense_prorm(
            self.model,
            self.batch,
            self.geometry,
            self.settings,
        )
        self.optimizer.zero_grad(set_to_none=True)
        for index in _training_slices(
            self.batch.num_edges,
            self.settings.microbatch_size,
        ):
            chunk_margins = self.model.margins(
                self.batch.left_features[index],
                self.batch.right_features[index],
            )
            policy_weights = envelope_weights(
                self.geometry.edge_scores[index],
                evaluation.direction,
                beta=self.settings.training_beta,
                detach_direction=True,
            ).to(dtype=chunk_margins.dtype)
            chunk_surrogate = envelope_surrogate(
                chunk_margins,
                self.batch.h[index],
                policy_weights,
            )
            (chunk_surrogate * ((index.stop - index.start) / self.batch.num_edges)).backward()
        gradient = self.model.weight.grad
        if gradient is None:
            raise RuntimeError("low-dimensional ProRM+ head did not receive a gradient")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("low-dimensional ProRM+ head gradient is non-finite")
        gradient_norm = float(torch.linalg.vector_norm(gradient).item())
        torch.nn.utils.clip_grad_norm_(
            [self.model.weight],
            max_norm=self.settings.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        if not bool(torch.isfinite(self.model.weight).all()):
            raise FloatingPointError("low-dimensional ProRM+ optimizer produced a non-finite head")
        self.completed_steps += 1
        diagnostic = TrainingStepDiagnostics(
            step=self.completed_steps,
            objective=evaluation.objective,
            gradient_norm=gradient_norm,
            dual_loss=evaluation.objective,
            dual_saddle_value=evaluation.saddle_value,
            dual_refresh=self.completed_steps,
        )
        self.history.append(diagnostic)
        return diagnostic

    def state_dict(self) -> dict[str, object]:
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "completed_steps": self.completed_steps,
            "history": copy.deepcopy(tuple(self.history)),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"model", "optimizer", "completed_steps", "history"}:
            raise ValueError("invalid low-dimensional trainer checkpoint")
        model_state = state["model"]
        optimizer_state = state["optimizer"]
        history = state["history"]
        completed_steps = state["completed_steps"]
        if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
            raise TypeError("invalid low-dimensional model/optimizer checkpoint")
        if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
            raise TypeError("invalid low-dimensional completed_steps")
        if not isinstance(history, tuple) or len(history) != completed_steps:
            raise ValueError("invalid low-dimensional history checkpoint")
        if not all(isinstance(item, TrainingStepDiagnostics) for item in history):
            raise TypeError("invalid low-dimensional diagnostics checkpoint")
        self.model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(dict(optimizer_state))
        self.completed_steps = completed_steps
        self.history = list(copy.deepcopy(history))


def _train_low_dimensional_control(
    control: SeededOrthonormalTangentControl,
    *,
    full_training: TrainingTensorData,
    settings: Phase2TrainingSettings,
    label_stream_sha256: str,
    primary_bt_head_sha256: str,
    expected_zero_head_sha256: str,
) -> tuple[dict[str, object], str]:
    """Train and audit a fresh ProRM+ head under projected ridge-free geometry."""

    batch, geometry = _build_dense_pseudoinverse_geometry(
        control.training,
        settings,
    )
    projected_moment_map_identifiability = _prorm_moment_map_identifiability(
        control.training,
        settings,
        projected_geometry=geometry,
        projection_sha256=control.projection_sha256,
    )
    model = _zero_model(control.training)
    initial_head_sha256 = _tensor_sha256(model.weight)
    if initial_head_sha256 != expected_zero_head_sha256:
        raise RuntimeError("low-dimensional ProRM+ does not share the primary zero initialization")
    trainer = _DensePseudoinverseProRMTrainer(
        model,
        batch,
        geometry,
        settings,
    )
    convergence = _run_trainer_to_first_order_convergence(
        trainer,
        audit=trainer.audit,
        spec=settings.convergence,
        fixed_snapshot_steps=settings.outer_steps,
        objective_name="low_dimensional_prorm_plus",
        rank_diagnostic=projected_moment_map_identifiability,
    )
    final = _evaluate_dense_prorm(model, batch, geometry, settings)
    head = _make_head_evidence(
        arm="low_dimensional_tangent_positive_control",
        method=PRORM_PLUS,
        model=model,
        initial_head_sha256=initial_head_sha256,
        initial_objective=convergence.initial.objective,
        final_objective=final.objective,
        history=convergence.history,
        final_pcg=None,
        first_order_convergence=convergence.evidence,
    )

    low_scores = control.training.policy_scores.reshape(
        -1,
        control.selected_dimension,
    ).to(dtype=final.direction.dtype)
    scattered_direction = control.scatter_direction_to_full(final.direction)
    full_scores = full_training.policy_scores.reshape(
        -1,
        full_training.policy_dimension,
    ).to(dtype=final.direction.dtype)
    low_projected_scores = low_scores @ final.direction
    full_projected_scores = full_scores @ scattered_direction
    score_error = low_projected_scores - full_projected_scores
    max_absolute_error = float(torch.max(torch.abs(score_error)).item())
    l2_error = float(torch.linalg.vector_norm(score_error).item())
    scale = max(
        float(torch.max(torch.abs(low_projected_scores)).item()),
        float(torch.max(torch.abs(full_projected_scores)).item()),
    )
    tolerance = 5.0e-5 * (1.0 + scale)
    score_identity_passed = max_absolute_error <= tolerance
    if not score_identity_passed:
        raise RuntimeError(
            "low-dimensional projection/scatter score identity failed: "
            f"max_error={max_absolute_error:.3e}, tolerance={tolerance:.3e}"
        )
    control_evidence = {
        "schema_version": "low-dimensional-tangent-training-control/v1",
        "interpretation": (
            "positive_control_only;fresh_prorm_plus_head_under_projected_geometry;"
            "ineligible_for_primary_claim"
        ),
        "enabled": True,
        "eligible_for_primary_claim": False,
        "training_arm": PRIMARY_TRAINING_ARM,
        "label_stream_sha256": label_stream_sha256,
        "target": "same_r4_mean_h_as_primary_prorm_plus",
        "bt_head": {
            "head_sha256": primary_bt_head_sha256,
            "retrained": False,
            "reason": "bt_objective_is_independent_of_policy_tangent_geometry",
        },
        "projection": control.to_dict(),
        "projected_prorm_moment_map_identifiability": (projected_moment_map_identifiability),
        "geometry": {
            "regularization": settings.low_dimensional_regularization,
            "ridge_enabled": False,
            "ridge_coefficient": 0.0,
            "solver": "torch.linalg.eigh_truncated_moore_penrose",
            "solver_dtype": settings.pcg_dtype,
            "selected_dimension": control.selected_dimension,
            "numerical_rank": geometry.rank,
            "relative_eigenvalue_tolerance": (
                settings.low_dimensional_relative_eigenvalue_tolerance
            ),
            "absolute_eigenvalue_threshold": geometry.eigenvalue_threshold,
            "smallest_eigenvalue": geometry.smallest_eigenvalue,
            "smallest_retained_eigenvalue": (geometry.smallest_retained_eigenvalue),
            "largest_retained_eigenvalue": (geometry.largest_retained_eigenvalue),
            "fisher_sha256": geometry.fisher_sha256,
            "pseudoinverse_sha256": geometry.pseudoinverse_sha256,
            "pcg_used": False,
        },
        "head": head.to_dict(),
        "final_full_data_audit": {
            "optimizer_constructed": False,
            "optimizer_step_called": False,
            "saved_head_mutated": False,
            "objective": final.objective,
            "dual_saddle_value": final.saddle_value,
            "gradient": "full_data_unclipped",
            "gradient_l2_norm": final.gradient_norm,
            "moment_sha256": _tensor_sha256(final.moment),
            "selected_direction_sha256": _tensor_sha256(final.direction),
            "pseudoinverse_solve_residual_norm": (final.solve_residual_norm),
            "pseudoinverse_solve_relative_residual": (final.solve_relative_residual),
        },
        "deployment_score_identity": {
            "formula": ("(S_full @ P) @ u_low == S_full @ (P @ u_low)"),
            "selected_direction_sha256": _tensor_sha256(final.direction),
            "scattered_full_direction_sha256": _tensor_sha256(scattered_direction),
            "low_projected_score_sha256": _tensor_sha256(low_projected_scores),
            "full_projected_score_sha256": _tensor_sha256(full_projected_scores),
            "max_absolute_error": max_absolute_error,
            "l2_error": l2_error,
            "absolute_tolerance": tolerance,
            "passed": score_identity_passed,
        },
        "fresh_zero_initialized": True,
        "raw_labels_retained": False,
        "raw_node_rewards_retained": False,
    }
    return (
        _strict_json_copy(
            control_evidence,
            name="low_dimensional_control",
        ),
        head.head_sha256,
    )


def _verify_audit_objective(
    audit: Mapping[str, object],
    *,
    learner: str,
    expected: float,
) -> None:
    learners = audit.get("learners")
    if not isinstance(learners, Mapping):
        raise RuntimeError("optimization audit is missing learners")
    learner_audit = learners.get(learner)
    if not isinstance(learner_audit, Mapping):
        raise RuntimeError(f"optimization audit is missing {learner}")
    observed = learner_audit.get("objective")
    if isinstance(observed, bool) or not isinstance(observed, Real):
        raise RuntimeError("optimization audit objective is malformed")
    if not math.isclose(float(observed), expected, rel_tol=2.0e-5, abs_tol=2.0e-7):
        raise RuntimeError(f"saved-head optimization audit does not bind {learner} final objective")


def _compact_direct_oracle_identity(control: object) -> dict[str, object]:
    reference_norm = float(torch.linalg.vector_norm(control.all_node_covariance_moment).item())
    relative_error = (
        None if reference_norm == 0.0 else control.identity_absolute_error / reference_norm
    )
    direction = control.native_oracle_direction
    return {
        "schema_version": "direct-oracle-exact-moment-identity/v1",
        "interpretation": "algebraic_identity_bypasses_reward_class_and_optimizer",
        "source_node_rewards_sha256": control.source_node_rewards_sha256,
        "num_prompts": control.num_prompts,
        "num_candidates": control.num_candidates,
        "policy_dimension": control.all_node_covariance_moment.numel(),
        "canonical_margin_sha256": _tensor_sha256(control.canonical_margins),
        "canonical_pair_moment_sha256": _tensor_sha256(control.canonical_pair_moment),
        "complete_pair_u_stat_moment_sha256": _tensor_sha256(control.complete_pair_u_stat_moment),
        "all_node_covariance_moment_sha256": _tensor_sha256(control.all_node_covariance_moment),
        "complete_pair_identity_absolute_error": control.identity_absolute_error,
        "complete_pair_identity_relative_error": relative_error,
        "complete_pair_identity_is_algebraic": True,
        "reward_head_bypassed": True,
        "optimizer_bypassed": True,
        "trained_exact_margin_head_required_to_match": False,
        "raw_node_rewards_retained": False,
        "native_oracle_direction": {
            "direction_sha256": _tensor_sha256(direction.direction),
            "absolute_damping": direction.absolute_damping,
            "mean_fisher_diagonal": direction.mean_fisher_diagonal,
            "moment_norm": direction.moment_norm,
            "direction_norm": direction.direction_norm,
            "fisher_curvature": direction.fisher_curvature,
            "damped_curvature": direction.damped_curvature,
            "moment_alignment": direction.moment_alignment,
            "pcg": {
                "iterations": direction.pcg_iterations,
                "residual_norm": direction.pcg_residual_norm,
                "relative_residual": direction.pcg_relative_residual,
                "converged": direction.pcg_converged,
                "reason": direction.pcg_reason,
            },
        },
    }


def _exact_head_gap(
    training: TrainingTensorData,
    exact_head: tuple[float, ...],
    direct_control: object,
    settings: Phase2TrainingSettings,
    *,
    optimizer_steps: int,
) -> dict[str, object]:
    head_tensor = torch.tensor(
        exact_head,
        dtype=training.reward_features.dtype,
        device=training.reward_features.device,
    )
    predicted_rewards = torch.einsum("pmh,h->pm", training.reward_features, head_tensor)
    solve_dtype = resolve_fisher_solve_dtype(settings.pcg_dtype)
    predicted_moment = policy_reward_moment(
        training.policy_scores.to(dtype=solve_dtype),
        predicted_rewards.to(dtype=solve_dtype),
        center_candidates=True,
        candidate_dim=1,
    )
    oracle_moment = direct_control.all_node_covariance_moment
    moment_error = predicted_moment - oracle_moment
    oracle_moment_norm = float(torch.linalg.vector_norm(oracle_moment).item())
    moment_error_norm = float(torch.linalg.vector_norm(moment_error).item())
    head_direction = policy_direction_from_head(
        training,
        head_tensor,
        relative_damping=settings.relative_damping,
        beta=1.0,
        pcg_dtype=settings.pcg_dtype,
        pcg_max_iterations=settings.pcg_max_iterations,
        pcg_tolerance=settings.pcg_tolerance,
        pcg_absolute_tolerance=settings.pcg_absolute_tolerance,
        pcg_residual_recompute_interval=settings.pcg_residual_recompute_interval,
        require_pcg_convergence=True,
    )
    oracle_direction = direct_control.native_oracle_direction.direction
    direction_error = head_direction.direction - oracle_direction
    flat_scores = training.policy_scores.reshape(-1, training.policy_dimension).to(
        dtype=solve_dtype
    )
    head_projected = flat_scores @ head_direction.direction
    oracle_projected = flat_scores @ oracle_direction
    fisher_error = float(torch.mean((head_projected - oracle_projected).square()).item())
    head_fisher_norm = float(torch.mean(head_projected.square()).item())
    oracle_fisher_norm = float(torch.mean(oracle_projected.square()).item())
    denominator = math.sqrt(head_fisher_norm * oracle_fisher_norm)
    fisher_cosine = (
        None
        if denominator == 0.0
        else float(torch.mean(head_projected * oracle_projected).item()) / denominator
    )
    return {
        "schema_version": "trained-exact-margin-gap/v1",
        "interpretation": (
            "combined_restricted_reward_class_and_finite_optimizer_gap;"
            "not_the_direct_algebraic_identity"
        ),
        "restricted_reward_class": "frozen_backbone_linear_head",
        "finite_optimizer_steps": _positive_integer(
            optimizer_steps,
            name="optimizer_steps",
        ),
        "predicted_all_node_moment_sha256": _tensor_sha256(predicted_moment),
        "oracle_all_node_moment_sha256": _tensor_sha256(oracle_moment),
        "all_node_moment_error_l2": moment_error_norm,
        "all_node_moment_error_relative": (
            None if oracle_moment_norm == 0.0 else moment_error_norm / oracle_moment_norm
        ),
        "trained_direction_sha256": _tensor_sha256(head_direction.direction),
        "oracle_direction_sha256": _tensor_sha256(oracle_direction),
        "direction_error_l2": float(torch.linalg.vector_norm(direction_error).item()),
        "squared_fisher_direction_error": fisher_error,
        "trained_direction_fisher_norm": head_fisher_norm,
        "oracle_direction_fisher_norm": oracle_fisher_norm,
        "fisher_cosine": fisher_cosine,
        "trained_direction_pcg": {
            "iterations": head_direction.pcg_iterations,
            "relative_residual": head_direction.pcg_relative_residual,
            "converged": head_direction.pcg_converged,
            "reason": head_direction.pcg_reason,
        },
        "algebraic_identity_claimed": False,
        "raw_node_rewards_retained": False,
    }


def train_phase2_heads(
    training: TrainingTensorData,
    train_oracle_rewards: torch.Tensor,
    *,
    seed: int,
    settings: Phase2TrainingSettings | Phase2ConfigBundle | Mapping[str, object],
) -> Phase2TrainingResult:
    """Train fresh R=4 primary heads plus exact/direct train-only controls."""

    if not isinstance(training, TrainingTensorData):
        raise TypeError("training must be TrainingTensorData")
    compiled = compile_phase2_training_settings(settings)
    base_seed = _validate_seed(seed)
    if base_seed not in compiled.seeds:
        raise ValueError("seed is not one of the configured Phase-2 design seeds")
    rewards = _validate_frozen_oracle_rewards(training, train_oracle_rewards)
    input_training_sha = _input_training_sha256(training)
    oracle_reward_sha = _tensor_sha256(rewards)
    train_fisher_nodes = training.num_prompts * training.num_candidates
    selected_dimension = compiled.low_dimensional_selected_dimension
    if selected_dimension >= training.policy_dimension:
        raise ValueError(
            "configured low-dimensional control cannot execute: "
            "selected dimension must be smaller than the full policy dimension"
        )
    if selected_dimension >= train_fisher_nodes:
        raise ValueError(
            "configured low-dimensional control cannot execute: "
            "selected dimension must satisfy d < n_F"
        )
    if compiled.pcg_max_iterations < train_fisher_nodes + 1:
        raise ValueError(
            "pcg_max_iterations must cover the train Fisher rank bound plus one "
            f"({train_fisher_nodes + 1})"
        )
    absolute_damping = _absolute_damping(training, compiled)

    canonical_margins = rewards[:, 0] - rewards[:, 1]
    probabilities = torch.sigmoid(canonical_margins)
    floor = compiled.probability_floor
    tolerance = 2.0e-6 if probabilities.dtype == torch.float32 else 2.0e-12
    if bool(
        ((probabilities < floor - tolerance) | (probabilities > 1.0 - floor + tolerance)).any()
    ):
        raise ValueError(
            "train_oracle_rewards do not satisfy the locked transformed-oracle "
            "BTL probability range [0.25, 0.75]"
        )
    generator, derived_seed, derivation_sha = _generator_for_training(
        training,
        base_seed=base_seed,
        namespace=compiled.label_rng_namespace,
    )
    initial_generator_state_sha = _tensor_sha256(generator.get_state())
    noisy_arm = sample_canonical_r4_noisy_arm(
        training,
        probabilities,
        generator=generator,
        max_total_annotations=compiled.max_total_annotations,
    )
    final_generator_state_sha = _tensor_sha256(generator.get_state())
    labels = noisy_arm.repeated_labels
    replicate_count_sha = _tensor_sha256(labels.counts)
    replicate_win_sha = _tensor_sha256(labels.wins)
    replicate_h_sha = _tensor_sha256(labels.replicate_h)
    mean_h_sha = _tensor_sha256(noisy_arm.training.h)
    repeated_label_tail_diagnostics = build_repeated_label_tail_diagnostics(
        replicate_counts=labels.counts,
        replicate_h=labels.replicate_h,
        mean_h=noisy_arm.training.h,
        replicate_count_sha256=replicate_count_sha,
        replicate_h_sha256=replicate_h_sha,
        mean_h_sha256=mean_h_sha,
    )
    label_payload = {
        "namespace": compiled.label_rng_namespace,
        "base_seed": base_seed,
        "derived_seed": derived_seed,
        "derivation_sha256": derivation_sha,
        "initial_state_sha256": initial_generator_state_sha,
        "final_state_sha256": final_generator_state_sha,
        "probability_sha256": noisy_arm.audit.probability_sha256,
        "replicate_count_sha256": replicate_count_sha,
        "replicate_win_sha256": replicate_win_sha,
        "replicate_h_sha256": replicate_h_sha,
        "mean_h_sha256": mean_h_sha,
        "repeated_label_tail_diagnostics_sha256": repeated_label_tail_diagnostics[
            "diagnostics_sha256"
        ],
        "realized_total_annotations": labels.total_annotations,
    }
    label_stream_sha = _canonical_sha256(label_payload)
    label_evidence = LabelStreamEvidence(
        namespace=compiled.label_rng_namespace,
        base_seed=base_seed,
        derived_seed=derived_seed,
        derivation_sha256=derivation_sha,
        generator_device=str(generator.device),
        initial_state_sha256=initial_generator_state_sha,
        final_state_sha256=final_generator_state_sha,
        oracle_reward_sha256=oracle_reward_sha,
        canonical_probability_sha256=noisy_arm.audit.probability_sha256,
        replicate_count_sha256=label_payload["replicate_count_sha256"],
        replicate_win_sha256=label_payload["replicate_win_sha256"],
        replicate_h_sha256=label_payload["replicate_h_sha256"],
        mean_h_sha256=label_payload["mean_h_sha256"],
        label_stream_sha256=label_stream_sha,
        repeated_label_tail_diagnostics=repeated_label_tail_diagnostics,
        realized_total_annotations=labels.total_annotations,
        realized_annotations_per_edge=noisy_arm.audit.realized_annotations_per_edge,
        expected_annotations_per_edge=noisy_arm.audit.expected_annotations_per_edge,
        num_edges=training.num_prompts,
    )

    primary_batch = noisy_arm.training.to_training_batch()
    reward_head_identifiability = _reward_head_identifiability(
        noisy_arm.training,
        compiled,
    )
    prorm_moment_map_identifiability = _prorm_moment_map_identifiability(
        noisy_arm.training,
        compiled,
    )
    bt_model = _zero_model(noisy_arm.training)
    prorm_model = _zero_model(noisy_arm.training)
    bt_initial_hash = _tensor_sha256(bt_model.weight)
    prorm_initial_hash = _tensor_sha256(prorm_model.weight)
    if bt_initial_hash != prorm_initial_hash:
        raise RuntimeError("BT-MLE and ProRM+ do not share the same zero initialization")
    bt_trainer = BTMLETrainer(bt_model, primary_batch, _bt_config(compiled))
    prorm_trainer = ProRMPlusTrainer(
        prorm_model,
        primary_batch,
        _prorm_config(compiled, absolute_damping=absolute_damping),
    )
    bt_convergence = _run_trainer_to_first_order_convergence(
        bt_trainer,
        audit=partial(_bt_first_order_measurement, bt_trainer),
        spec=compiled.convergence,
        fixed_snapshot_steps=compiled.outer_steps,
        objective_name=BT_MLE,
        rank_diagnostic=reward_head_identifiability,
    )
    prorm_convergence = _run_trainer_to_first_order_convergence(
        prorm_trainer,
        audit=partial(_prorm_first_order_measurement, prorm_trainer),
        spec=compiled.convergence,
        fixed_snapshot_steps=compiled.outer_steps,
        objective_name=PRORM_PLUS,
        rank_diagnostic=prorm_moment_map_identifiability,
    )
    prorm_final_solver = prorm_convergence.final.inner_solver
    if prorm_final_solver is None or prorm_final_solver.get("converged") is not True:
        raise RuntimeError("final ProRM+ cold-start FP64 PCG audit did not converge")
    prorm_final_pcg = _pcg_evidence(prorm_final_solver)
    bt_evidence = _make_head_evidence(
        arm=PRIMARY_TRAINING_ARM,
        method=BT_MLE,
        model=bt_model,
        initial_head_sha256=bt_initial_hash,
        initial_objective=bt_convergence.initial.objective,
        final_objective=bt_convergence.final.objective,
        history=bt_convergence.history,
        final_pcg=None,
        first_order_convergence=bt_convergence.evidence,
    )
    prorm_evidence = _make_head_evidence(
        arm=PRIMARY_TRAINING_ARM,
        method=PRORM_PLUS,
        model=prorm_model,
        initial_head_sha256=prorm_initial_hash,
        initial_objective=prorm_convergence.initial.objective,
        final_objective=prorm_convergence.final.objective,
        history=prorm_convergence.history,
        final_pcg=prorm_final_pcg,
        first_order_convergence=prorm_convergence.evidence,
    )
    primary_audit = evaluate_saved_head_optimization(
        noisy_arm.training,
        {
            BT_MLE: bt_evidence.head_weight,
            PRORM_PLUS: prorm_evidence.head_weight,
        },
        beta=compiled.training_beta,
        absolute_damping=absolute_damping,
        pcg_max_iterations=compiled.pcg_max_iterations,
        pcg_tolerance=compiled.pcg_tolerance,
        pcg_absolute_tolerance=compiled.pcg_absolute_tolerance,
        pcg_residual_recompute_interval=compiled.pcg_residual_recompute_interval,
    )
    _verify_audit_objective(
        primary_audit,
        learner=BT_MLE,
        expected=bt_evidence.final_objective,
    )
    _verify_audit_objective(
        primary_audit,
        learner=PRORM_PLUS,
        expected=prorm_evidence.final_objective,
    )
    primary_audit = {
        **primary_audit,
        "reward_head_identifiability": reward_head_identifiability,
        "prorm_moment_map_identifiability": prorm_moment_map_identifiability,
    }
    # The full-tangent FP64 geometry is the dominant resident allocation.
    # Release the primary trainers before constructing the independently
    # projected, ridge-free positive control.
    del (
        bt_model,
        bt_trainer,
        bt_convergence,
        primary_batch,
        prorm_convergence,
        prorm_model,
        prorm_trainer,
    )

    coordinate_layout = TangentCoordinateLayout(
        layout_id=compiled.low_dimensional_source_layout_id,
        coordinate_ids=tuple(range(training.policy_dimension)),
    )
    low_dimensional_projection = select_seeded_orthonormal_tangent(
        noisy_arm.training,
        selected_dimension=selected_dimension,
        coordinate_layout=coordinate_layout,
        seed=base_seed,
        namespace=compiled.low_dimensional_namespace,
    )
    low_dimensional_evidence, low_dimensional_head_sha = _train_low_dimensional_control(
        low_dimensional_projection,
        full_training=noisy_arm.training,
        settings=compiled,
        label_stream_sha256=label_stream_sha,
        primary_bt_head_sha256=bt_evidence.head_sha256,
        expected_zero_head_sha256=bt_initial_hash,
    )
    low_dimensional_projection_sha = low_dimensional_projection.projection_sha256
    low_dimensional_moment_map_sha = _canonical_sha256(
        _strict_json_copy(
            low_dimensional_evidence["projected_prorm_moment_map_identifiability"],
            name="projected_prorm_moment_map_identifiability",
        )
    )
    del labels, low_dimensional_projection, noisy_arm

    exact_arm = build_exact_margin_canonical_arm(training, rewards)
    exact_model = _zero_model(exact_arm.training)
    exact_initial_hash = _tensor_sha256(exact_model.weight)
    exact_trainer = ProRMPlusTrainer(
        exact_model,
        exact_arm.training.to_training_batch(),
        _prorm_config(compiled, absolute_damping=absolute_damping),
    )
    exact_convergence = _run_trainer_to_first_order_convergence(
        exact_trainer,
        audit=partial(_prorm_first_order_measurement, exact_trainer),
        spec=compiled.convergence,
        fixed_snapshot_steps=compiled.outer_steps,
        objective_name="exact_margin_prorm_plus",
        rank_diagnostic=prorm_moment_map_identifiability,
    )
    exact_final_solver = exact_convergence.final.inner_solver
    if exact_final_solver is None or exact_final_solver.get("converged") is not True:
        raise RuntimeError("final exact-margin cold-start FP64 PCG audit did not converge")
    exact_final_pcg = _pcg_evidence(exact_final_solver)
    exact_head = _make_head_evidence(
        arm="exact_margin_positive_control",
        method=PRORM_PLUS,
        model=exact_model,
        initial_head_sha256=exact_initial_hash,
        initial_objective=exact_convergence.initial.objective,
        final_objective=exact_convergence.final.objective,
        history=exact_convergence.history,
        final_pcg=exact_final_pcg,
        first_order_convergence=exact_convergence.evidence,
    )
    exact_full_audit = evaluate_saved_head_optimization(
        exact_arm.training,
        {
            BT_MLE: tuple(0.0 for _ in range(training.reward_dimension)),
            PRORM_PLUS: exact_head.head_weight,
        },
        beta=compiled.training_beta,
        absolute_damping=absolute_damping,
        pcg_max_iterations=compiled.pcg_max_iterations,
        pcg_tolerance=compiled.pcg_tolerance,
        pcg_absolute_tolerance=compiled.pcg_absolute_tolerance,
        pcg_residual_recompute_interval=compiled.pcg_residual_recompute_interval,
    )
    _verify_audit_objective(
        exact_full_audit,
        learner=PRORM_PLUS,
        expected=exact_head.final_objective,
    )
    exact_optimization_audit = {
        "geometry": exact_full_audit["geometry"],
        "learner": exact_full_audit["learners"][PRORM_PLUS],
        "bt_audit_discarded": True,
        "optimizer_constructed": False,
        "optimizer_step_called": False,
    }
    exact_target_audit = exact_arm.audit.to_dict()
    exact_soft_bt_control = _train_exact_soft_label_bt_control(
        exact_arm.training,
        source_node_rewards_sha256=exact_arm.audit.source_node_rewards_sha256,
        exact_margin_sha256=exact_arm.audit.exact_margin_sha256,
        settings=compiled,
        rank_diagnostic=reward_head_identifiability,
    )
    if exact_soft_bt_control.head.initial_head_sha256 != exact_initial_hash:
        raise RuntimeError(
            "exact soft-label BT and exact-margin ProRM+ do not share zero initialization"
        )
    del exact_arm, exact_convergence, exact_full_audit, exact_model, exact_trainer

    direct_control = build_direct_oracle_geometry_control(
        training,
        rewards,
        relative_damping=compiled.relative_damping,
        pcg_dtype=compiled.pcg_dtype,
        pcg_max_iterations=compiled.pcg_max_iterations,
        pcg_tolerance=compiled.pcg_tolerance,
        pcg_absolute_tolerance=compiled.pcg_absolute_tolerance,
        pcg_residual_recompute_interval=compiled.pcg_residual_recompute_interval,
        require_pcg_convergence=True,
    )
    if not direct_control.native_oracle_direction.pcg_converged:
        raise RuntimeError("direct-oracle policy-direction PCG did not converge")
    if direct_control.source_node_rewards_sha256 != oracle_reward_sha:
        raise RuntimeError("direct-oracle control does not bind the supplied train rewards")
    direct_identity = _compact_direct_oracle_identity(direct_control)
    exact_gap = _exact_head_gap(
        training,
        exact_head.head_weight,
        direct_control,
        compiled,
        optimizer_steps=int(exact_head.history_summary["num_steps"]),
    )
    exact_control = ExactMarginTrainingControl(
        head=exact_head,
        target_audit=exact_target_audit,
        optimization_audit=exact_optimization_audit,
        reward_class_and_optimizer_gap=exact_gap,
    )

    training_instance_sha = _canonical_sha256(
        {
            "schema_version": "phase2-training-instance/v1",
            "phase2_config_hash": compiled.phase2_config_hash,
            "settings_sha256": compiled.sha256,
            "input_training_sha256": input_training_sha,
            "oracle_reward_sha256": oracle_reward_sha,
            "seed": base_seed,
            "label_stream_sha256": label_stream_sha,
            "reward_head_identifiability_sha256": _canonical_sha256(reward_head_identifiability),
            "prorm_moment_map_identifiability_sha256": (
                _canonical_sha256(prorm_moment_map_identifiability)
            ),
            "bt_head_sha256": bt_evidence.head_sha256,
            "prorm_plus_head_sha256": prorm_evidence.head_sha256,
            "low_dimensional_head_sha256": low_dimensional_head_sha,
            "low_dimensional_projection_sha256": (low_dimensional_projection_sha),
            "low_dimensional_moment_map_identifiability_sha256": (low_dimensional_moment_map_sha),
            "exact_margin_head_sha256": exact_head.head_sha256,
            "exact_soft_label_bt_head_sha256": (exact_soft_bt_control.head.head_sha256),
            "direct_oracle_direction_sha256": (
                direct_identity["native_oracle_direction"]["direction_sha256"]
            ),
        }
    )
    result = Phase2TrainingResult(
        settings=compiled,
        training_design_sha256=compiled.phase2_config_hash,
        training_settings_sha256=compiled.sha256,
        training_instance_sha256=training_instance_sha,
        input_training_sha256=input_training_sha,
        absolute_damping=absolute_damping,
        label_stream=label_evidence,
        bt_mle=bt_evidence,
        prorm_plus=prorm_evidence,
        low_dimensional_control=low_dimensional_evidence,
        exact_margin_control=exact_control,
        exact_soft_label_bt_control=exact_soft_bt_control,
        direct_oracle_identity=direct_identity,
        primary_optimization_audit=primary_audit,
    )
    result.to_dict()
    return result


class FreshPhase2HeadTrainer:
    """Runner adapter that always performs fresh train-only R=4 fitting."""

    def __init__(
        self,
        settings: Phase2TrainingSettings | Phase2ConfigBundle | Mapping[str, object],
    ) -> None:
        self.settings = compile_phase2_training_settings(settings)
        self.last_result: Phase2TrainingResult | None = None

    def train_heads(
        self,
        train: TrainingTensorData,
        train_oracle_rewards: torch.Tensor,
        *,
        seed: int,
    ) -> object:
        result = train_phase2_heads(
            train,
            train_oracle_rewards,
            seed=seed,
            settings=self.settings,
        )
        self.last_result = result
        return result.to_runner_head_result()


__all__ = [
    "EXACT_SOFT_BT_ARM",
    "EXACT_SOFT_BT_INPUT",
    "EXACT_SOFT_BT_ROLE",
    "LABEL_RNG_NAMESPACE",
    "PHASE2_RECOVERY_TRAINING_SCHEMA",
    "PHASE2_TRAINING_SCHEMA",
    "PRIMARY_TRAINING_ARM",
    "AdamWRecoveryProtocol",
    "ExactMarginTrainingControl",
    "ExactSoftLabelBTControl",
    "FirstOrderConvergenceSpec",
    "FreshPhase2HeadTrainer",
    "LabelStreamEvidence",
    "LearningRateStage",
    "OptimizationConvergenceError",
    "Phase2TrainingResult",
    "Phase2TrainingSettings",
    "TrainedHeadEvidence",
    "compile_phase2_training_settings",
    "train_phase2_heads",
]
