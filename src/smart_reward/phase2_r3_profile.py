"""Formal Gate-P CUDA profiling and fail-closed resource projection.

The claim-free profiling core remains useful for local CPU instrumentation,
but it cannot authorize Gate P.  This module is the only promotion boundary:
it accepts an exact :class:`ValidatedGatePRun`, verifies live CUDA, executes
the fixed core itself, binds external scheduler/resource evidence, and emits
only operational measurements.  There is deliberately no caller-controlled
``formal`` switch and no API that promotes a caller-supplied core payload.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from numbers import Real
from pathlib import Path
from typing import Final, Literal

import torch

from . import phase2_training as _training
from .phase2_checkpoint import DurableCheckpointStore
from .phase2_primary import build_primary_core_trainer
from .phase2_profile import (
    PHASE2_PROFILE_AUDIT_UPDATES,
    PHASE2_PROFILE_LEARNER_ORDER,
    PHASE2_PROFILE_STOP_REASON,
    PHASE2_PROFILE_UPDATES,
    profile_core_binding,
    run_gate_p_profile_core,
    validate_gate_p_profile_core_result,
)
from .phase2_r3_identity import (
    FORMAL_CUDA_PROFILE_RESULT_ROLE,
    FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
    RESOURCE_PLAN_ROLE,
    RESOURCE_PLAN_SCHEMA,
    ArtifactRef,
    ValidatedGatePRun,
)

PROFILE_SAFETY_MARGIN_POLICY_SCHEMA: Final = "phase2-recovery-r3-profile-safety-margin-policy/v1"
SCHEDULER_RESOURCE_ENVELOPE_SCHEMA: Final = "phase2-recovery-r3-scheduler-resource-envelope/v1"
PROFILE_PREPARATION_TIMINGS_SCHEMA: Final = "phase2-recovery-r3-profile-preparation-timings/v1"
PRODUCTION_CHECKPOINT_IO_PROFILE_SCHEMA: Final = (
    "phase2-recovery-r3-production-checkpoint-io-profile/v1"
)
PRODUCTION_OUTER_CHECKPOINT_PAYLOAD_SCHEMA: Final = (
    "phase2-recovery-r3-primary-checkpoint-payload/v1"
)
PRODUCTION_PROFILE_BENCHMARK_PAYLOAD_SCHEMA: Final = (
    "phase2-recovery-r3-profile-checkpoint-io-benchmark-payload/v1"
)
PRODUCTION_DURABLE_CHECKPOINT_ENVELOPE_SCHEMA: Final = "prorm-phase2-durable-training-checkpoint/v1"
PRODUCTION_CONTROLLER_CHECKPOINT_SCHEMA: Final = "phase2-first-order-controller-checkpoint/v2"
RESOURCE_PROJECTION_SCHEMA: Final = "phase2-recovery-r3-resource-projection/v1"
FORMAL_PROFILE_INFORMATION_BOUNDARY: Final = {
    "train_only": True,
    "validation_or_test_data_accessed": False,
    "policy_session_opened": False,
    "policy_rollout_performed": False,
    "controls_executed": False,
    "serialized_training_state_retained": False,
    "profile_consumable_as_primary_evidence": False,
}
R3_MAXIMUM_UPDATES_PER_HEAD: Final = 12_760
R3_AUDIT_CADENCE_UPDATES: Final = 20
R3_MANDATORY_CHECKPOINT_UPDATES: Final = (
    5_760,
    6_760,
    8_760,
    10_760,
    12_760,
)
R3_MANDATORY_CHECKPOINT_ROLES: Final = (
    "learning_rate_boundaries",
    "post_selection",
    "pre_head_transition",
    "signal_safe_boundary",
    "scheduler_segment_terminal",
    "pre_resume",
)
R3_TARGET_AUDIT_COUNT_PER_HEAD: Final = R3_MAXIMUM_UPDATES_PER_HEAD // R3_AUDIT_CADENCE_UPDATES + 1
R3_TOTAL_SAFE_UPDATE_BLOCKS: Final = (
    len(PHASE2_PROFILE_LEARNER_ORDER) * R3_MAXIMUM_UPDATES_PER_HEAD // R3_AUDIT_CADENCE_UPDATES
)
HPC4_SLURM_ACCOUNT: Final = "sigroup"

_PROFILE_MAX_FINITE_FLOAT: Final = 1.7976931348623157e308
_PROFILE_MAX_SIGNED_INT: Final = 2_147_483_647
_PROFILE_DIGEST: Final = "f" * 64
_PRIMARY_OUTER_PAYLOAD_FIELDS: Final = frozenset(
    {
        "schema_version",
        "design_sha256",
        "admission_sha256",
        "logical_run_id",
        "head_run_id",
        "scheduler_segment_id",
        "segment_index",
        "task_id",
        "seed",
        "objective",
        "runtime_sha256",
        "head_execution_slice_sha256",
        "controller_checkpoint_sha256",
        "controller_checkpoint",
        "information_boundary",
    }
)
_CONTROLLER_CHECKPOINT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "identity",
        "identity_sha256",
        "trainer_state",
        "controller_state",
        "checkpoint_sha256",
    }
)
_CONTROLLER_STATE_FIELDS: Final = frozenset(
    {
        "completed_steps",
        "initial",
        "checkpoint_boundary_step",
        "checkpoint_boundary_measurement",
        "checkpoint_boundary_trainer_state_sha256",
        "checkpoint_boundary_head_sha256",
        "checkpoint_boundary_optimizer_state_dict_sha256",
        "checks",
        "consecutive_passes",
        "selected_state",
        "selected_measurement",
        "selected_step",
        "selected_head_sha256",
        "selected_optimizer_state_sha256",
        "selected_checkpoint_optimizer_state_dict_sha256",
        "selected_checkpoint_sha256",
        "fixed_snapshot",
        "legacy_boundary_snapshot",
        "optimizer_protocol_execution",
        "recovery_state_check_transcript",
    }
)
_RECOVERY_CONVERGENCE_CHECK_FIELDS: Final = frozenset(
    {
        "step",
        "post_update",
        "full_data",
        "gradient_clipping_applied",
        "measurement",
        "gradient_ratio_to_zero_initialization",
        "eligible_after_min_steps",
        "threshold_passed",
        "consecutive_threshold_passes",
        "learning_rate_used_for_update",
        "learning_rate_schedule_sha256",
    }
)
_FIXED_SNAPSHOT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "step",
        "head_sha256",
        "measurement",
        "gradient_ratio_to_zero_initialization",
        "history_summary",
        "role",
        "used_as_primary_selection_rule",
        "coincides_with_selected_primary_iterate",
    }
)
_LEGACY_SNAPSHOT_FIELDS: Final = frozenset(
    {
        *_FIXED_SNAPSHOT_FIELDS,
        "learning_rate_used_for_update",
        "learning_rate_schedule_sha256",
        "test_or_validation_data_accessed",
    }
)
_OPTIMIZER_PROTOCOL_EXECUTION_FIELDS: Final = frozenset(
    {
        "schema_version",
        "protocol",
        "optimizer_class",
        "parameter_count",
        "fresh_optimizer_state_before_first_update",
        "reward_head_dtype_observed",
        "first_order_audit_dtype_required",
        "microbatch_order",
        "one_optimizer_update_per_step",
        "learning_rate_set_immediately_before_every_update",
        "single_optimizer_instance_for_all_updates",
        "optimizer_state_reset_at_lr_milestone",
        "adamw_moments_preserved_at_learning_rate_boundaries",
        "boundary_transitions",
        "completed_updates_observed",
        "per_update_state_checks",
        "selected_primary_optimizer_state_restored_and_verified",
        "selected_optimizer_object_identity_preserved",
        "selected_optimizer_moments_restored_and_verified",
        "selected_head_sha256",
        "restored_head_sha256",
        "selected_optimizer_state_sha256",
        "restored_optimizer_state_sha256",
        "selected_checkpoint_optimizer_state_dict_sha256",
        "restored_optimizer_state_dict_sha256",
        "selected_checkpoint_sha256",
        "test_or_validation_data_accessed",
    }
)

_FACTORY_TOKEN = object()
_SHA256_HEX = frozenset("0123456789abcdef")
_PCG_REASONS = frozenset({"converged", "zero_rhs", "max_iterations"})
_CORE_BOUNDARY = {
    "train_only": True,
    "validation_or_test_data_accessed": False,
    "policy_session_opened": False,
    "policy_rollout_performed": False,
    "controls_executed": False,
    "serialized_training_state_retained": False,
    "profile_consumable_as_primary_evidence": False,
}
_CUDA_IDENTITY_FIELDS = {
    "logical_device_index",
    "name",
    "total_memory_bytes",
    "compute_capability_major",
    "compute_capability_minor",
    "torch_cuda_version",
    "cuda_visible_devices",
}
_GPU_SAMPLE_FIELDS = {
    "sample_index",
    "wall_time_ns",
    "monotonic_time_ns",
    "uuid",
    "name",
    "total_memory_bytes",
    "gpu_utilization_percent",
    "memory_utilization_percent",
}
_CPU_MEMORY_FIELDS = {"current_rss_bytes", "peak_rss_bytes", "measurement"}
_SENSITIVE_STATE_KEYS = {
    "head_weight",
    "trained_head",
    "optimizer",
    "optimizer_state",
    "rng",
    "rng_state",
    "raw_reward",
    "raw_rewards",
    "raw_oracle",
    "raw_oracle_rewards",
    "raw_label",
    "raw_labels",
    "labels",
    "beta",
    "outcome",
    "gradient_direction",
    "checkpoint_state",
}


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("identity payload must contain strict JSON data") from error
    return hashlib.sha256(encoded).hexdigest()


def _json_copy(value: object, *, name: str) -> object:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain strict JSON data") from error


def _digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _nonnegative_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _exact_mapping(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{name} has non-exact fields: "
            f"missing={sorted(expected.difference(value))}, "
            f"extra={sorted(set(value).difference(expected))}"
        )
    return value


def _require_factory(value: object, *, name: str) -> None:
    if value is not _FACTORY_TOKEN:
        raise TypeError(f"{name} must be produced by its validating factory")


def _validated_run(value: object) -> ValidatedGatePRun:
    if type(value) is not ValidatedGatePRun:
        raise TypeError("profile_run must be exactly ValidatedGatePRun")
    value.validate_integrity()
    return value


def _assert_no_sensitive_state(value: object, *, path: str = "formal_profile") -> None:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{path} must not contain tensors")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            if key.casefold() in _SENSITIVE_STATE_KEYS:
                raise ValueError(f"{path}.{key} contains forbidden training state")
            _assert_no_sensitive_state(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_sensitive_state(item, path=f"{path}[{index}]")
        return
    if value is not None and not isinstance(value, (str, bool, int, float)):
        raise TypeError(f"{path} contains a non-JSON value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite float")


def _safety_payload(
    *,
    profile_run_sha256: str,
    walltime_margin_fraction: float,
    fixed_walltime_margin_seconds: float,
    memory_margin_fraction: float,
    signal_margin_seconds: float,
    durable_checkpoint_cadence_updates: int,
    checkpoint_on_selection: bool,
    checkpoint_before_head_transition: bool,
    checkpoint_on_signal_safe_boundary: bool,
    checkpoint_at_segment_terminal: bool,
    checkpoint_before_resume: bool,
) -> dict[str, object]:
    return {
        "schema_version": PROFILE_SAFETY_MARGIN_POLICY_SCHEMA,
        "profile_run_sha256": profile_run_sha256,
        "declared_before_profile": True,
        "walltime_margin_fraction": walltime_margin_fraction,
        "fixed_walltime_margin_seconds": fixed_walltime_margin_seconds,
        "memory_margin_fraction": memory_margin_fraction,
        "signal_margin_seconds": signal_margin_seconds,
        "durable_checkpoint_cadence_updates": (durable_checkpoint_cadence_updates),
        "mandatory_checkpoint_updates": list(R3_MANDATORY_CHECKPOINT_UPDATES),
        "mandatory_checkpoint_roles": list(R3_MANDATORY_CHECKPOINT_ROLES),
        "checkpoint_on_selection": checkpoint_on_selection,
        "checkpoint_before_head_transition": checkpoint_before_head_transition,
        "checkpoint_on_signal_safe_boundary": (checkpoint_on_signal_safe_boundary),
        "checkpoint_at_segment_terminal": checkpoint_at_segment_terminal,
        "checkpoint_before_resume": checkpoint_before_resume,
    }


@dataclass(frozen=True, slots=True)
class ProfileSafetyMarginPolicy:
    """Positive margins frozen before any formal profile measurement."""

    profile_run: ValidatedGatePRun = field(repr=False, compare=False)
    schema_version: str
    profile_run_sha256: str
    declared_before_profile: Literal[True]
    walltime_margin_fraction: float
    fixed_walltime_margin_seconds: float
    memory_margin_fraction: float
    signal_margin_seconds: float
    durable_checkpoint_cadence_updates: int
    mandatory_checkpoint_updates: tuple[int, ...]
    mandatory_checkpoint_roles: tuple[str, ...]
    checkpoint_on_selection: Literal[True]
    checkpoint_before_head_transition: Literal[True]
    checkpoint_on_signal_safe_boundary: Literal[True]
    checkpoint_at_segment_terminal: Literal[True]
    checkpoint_before_resume: Literal[True]
    policy_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        run = _validated_run(self.profile_run)
        if self.schema_version != PROFILE_SAFETY_MARGIN_POLICY_SCHEMA:
            raise ValueError("profile safety policy schema is not frozen")
        if self.profile_run_sha256 != run.profile_run_sha256:
            raise ValueError("profile safety policy is bound to another Gate-P run")
        if self.declared_before_profile is not True:
            raise ValueError("profile safety policy was not predeclared")
        wall_fraction = _positive_real(
            self.walltime_margin_fraction,
            name="walltime_margin_fraction",
        )
        fixed_wall = _positive_real(
            self.fixed_walltime_margin_seconds,
            name="fixed_walltime_margin_seconds",
        )
        memory_fraction = _positive_real(
            self.memory_margin_fraction,
            name="memory_margin_fraction",
        )
        signal_margin = _positive_real(
            self.signal_margin_seconds,
            name="signal_margin_seconds",
        )
        checkpoint_cadence = _exact_int(
            self.durable_checkpoint_cadence_updates,
            name="durable_checkpoint_cadence_updates",
            minimum=R3_AUDIT_CADENCE_UPDATES,
        )
        if (
            checkpoint_cadence > R3_MAXIMUM_UPDATES_PER_HEAD
            or checkpoint_cadence % R3_AUDIT_CADENCE_UPDATES != 0
        ):
            raise ValueError(
                "durable checkpoint cadence must be an audit-aligned positive "
                "interval no larger than 12,760"
            )
        if (
            type(self.mandatory_checkpoint_updates) is not tuple
            or self.mandatory_checkpoint_updates != R3_MANDATORY_CHECKPOINT_UPDATES
        ):
            raise ValueError("mandatory learning-rate checkpoint updates changed")
        if (
            type(self.mandatory_checkpoint_roles) is not tuple
            or self.mandatory_checkpoint_roles != R3_MANDATORY_CHECKPOINT_ROLES
        ):
            raise ValueError("mandatory checkpoint roles changed")
        if self.checkpoint_on_selection is not True:
            raise ValueError("post-selection checkpoint must remain mandatory")
        if self.checkpoint_before_head_transition is not True:
            raise ValueError("pre-head-transition checkpoint must remain mandatory")
        if self.checkpoint_on_signal_safe_boundary is not True:
            raise ValueError("signal-safe-boundary checkpoint must remain mandatory")
        if self.checkpoint_at_segment_terminal is not True:
            raise ValueError("segment-terminal checkpoint must remain mandatory")
        if self.checkpoint_before_resume is not True:
            raise ValueError("pre-resume checkpoint must remain mandatory")
        payload = _safety_payload(
            profile_run_sha256=run.profile_run_sha256,
            walltime_margin_fraction=wall_fraction,
            fixed_walltime_margin_seconds=fixed_wall,
            memory_margin_fraction=memory_fraction,
            signal_margin_seconds=signal_margin,
            durable_checkpoint_cadence_updates=checkpoint_cadence,
            checkpoint_on_selection=self.checkpoint_on_selection,
            checkpoint_before_head_transition=(self.checkpoint_before_head_transition),
            checkpoint_on_signal_safe_boundary=(self.checkpoint_on_signal_safe_boundary),
            checkpoint_at_segment_terminal=self.checkpoint_at_segment_terminal,
            checkpoint_before_resume=self.checkpoint_before_resume,
        )
        _digest(self.policy_sha256, name="policy_sha256")
        if _canonical_sha256(payload) != self.policy_sha256:
            raise ValueError("profile safety policy SHA256 does not match its contents")

    def validate_integrity(self) -> None:
        _require_factory(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _safety_payload(
            profile_run_sha256=self.profile_run_sha256,
            walltime_margin_fraction=self.walltime_margin_fraction,
            fixed_walltime_margin_seconds=self.fixed_walltime_margin_seconds,
            memory_margin_fraction=self.memory_margin_fraction,
            signal_margin_seconds=self.signal_margin_seconds,
            durable_checkpoint_cadence_updates=(self.durable_checkpoint_cadence_updates),
            checkpoint_on_selection=self.checkpoint_on_selection,
            checkpoint_before_head_transition=(self.checkpoint_before_head_transition),
            checkpoint_on_signal_safe_boundary=(self.checkpoint_on_signal_safe_boundary),
            checkpoint_at_segment_terminal=self.checkpoint_at_segment_terminal,
            checkpoint_before_resume=self.checkpoint_before_resume,
        )
        return {**payload, "policy_sha256": self.policy_sha256}


def freeze_profile_safety_margin_policy(
    profile_run: ValidatedGatePRun,
    *,
    walltime_margin_fraction: float,
    fixed_walltime_margin_seconds: float,
    memory_margin_fraction: float,
    signal_margin_seconds: float,
    durable_checkpoint_cadence_updates: int,
) -> ProfileSafetyMarginPolicy:
    """Freeze strictly positive margins before executing the profile."""

    run = _validated_run(profile_run)
    payload = _safety_payload(
        profile_run_sha256=run.profile_run_sha256,
        walltime_margin_fraction=_positive_real(
            walltime_margin_fraction,
            name="walltime_margin_fraction",
        ),
        fixed_walltime_margin_seconds=_positive_real(
            fixed_walltime_margin_seconds,
            name="fixed_walltime_margin_seconds",
        ),
        memory_margin_fraction=_positive_real(
            memory_margin_fraction,
            name="memory_margin_fraction",
        ),
        signal_margin_seconds=_positive_real(
            signal_margin_seconds,
            name="signal_margin_seconds",
        ),
        durable_checkpoint_cadence_updates=_exact_int(
            durable_checkpoint_cadence_updates,
            name="durable_checkpoint_cadence_updates",
            minimum=R3_AUDIT_CADENCE_UPDATES,
        ),
        checkpoint_on_selection=True,
        checkpoint_before_head_transition=True,
        checkpoint_on_signal_safe_boundary=True,
        checkpoint_at_segment_terminal=True,
        checkpoint_before_resume=True,
    )
    constructor_payload = {
        **payload,
        "mandatory_checkpoint_updates": R3_MANDATORY_CHECKPOINT_UPDATES,
        "mandatory_checkpoint_roles": R3_MANDATORY_CHECKPOINT_ROLES,
    }
    result = ProfileSafetyMarginPolicy(
        profile_run=run,
        **constructor_payload,
        policy_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _envelope_payload(
    *,
    profile_run_sha256: str,
    scheduler_raw_evidence_sha256: str,
    resource_raw_evidence_sha256: str,
    partition: str,
    gpu_name: str,
    gpu_total_memory_bytes: int,
    max_allocation_wall_seconds: int,
    max_array_concurrency: int,
    max_scheduler_segments: int,
    max_gpus_per_task: int,
    max_cpus_per_task: int,
    max_memory_bytes: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEDULER_RESOURCE_ENVELOPE_SCHEMA,
        "profile_run_sha256": profile_run_sha256,
        "scheduler_raw_evidence_sha256": scheduler_raw_evidence_sha256,
        "resource_raw_evidence_sha256": resource_raw_evidence_sha256,
        "slurm_account": HPC4_SLURM_ACCOUNT,
        "partition": partition,
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "max_allocation_wall_seconds": max_allocation_wall_seconds,
        "max_array_concurrency": max_array_concurrency,
        "max_scheduler_segments": max_scheduler_segments,
        "max_gpus_per_task": max_gpus_per_task,
        "max_cpus_per_task": max_cpus_per_task,
        "max_memory_bytes": max_memory_bytes,
    }


@dataclass(frozen=True, slots=True)
class SchedulerResourceEnvelope:
    """Decoded HPC4 limits bound to immutable raw scheduler/resource evidence."""

    profile_run: ValidatedGatePRun = field(repr=False, compare=False)
    schema_version: str
    profile_run_sha256: str
    scheduler_raw_evidence_sha256: str
    resource_raw_evidence_sha256: str
    slurm_account: str
    partition: str
    gpu_name: str
    gpu_total_memory_bytes: int
    max_allocation_wall_seconds: int
    max_array_concurrency: int
    max_scheduler_segments: int
    max_gpus_per_task: int
    max_cpus_per_task: int
    max_memory_bytes: int
    envelope_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        run = _validated_run(self.profile_run)
        if self.schema_version != SCHEDULER_RESOURCE_ENVELOPE_SCHEMA:
            raise ValueError("scheduler/resource envelope schema is not frozen")
        if self.profile_run_sha256 != run.profile_run_sha256:
            raise ValueError("scheduler/resource envelope is bound to another Gate-P run")
        _digest(
            self.scheduler_raw_evidence_sha256,
            name="scheduler_raw_evidence_sha256",
        )
        _digest(
            self.resource_raw_evidence_sha256,
            name="resource_raw_evidence_sha256",
        )
        if self.slurm_account != HPC4_SLURM_ACCOUNT:
            raise ValueError("Gate-P must use the granted HPC4 Slurm account sigroup")
        partition = _text(self.partition, name="partition")
        if not partition.startswith("gpu-"):
            raise ValueError("Gate-P partition must be an explicit GPU partition")
        gpu_name = _text(self.gpu_name, name="gpu_name")
        positive_fields = (
            "gpu_total_memory_bytes",
            "max_allocation_wall_seconds",
            "max_array_concurrency",
            "max_scheduler_segments",
            "max_gpus_per_task",
            "max_cpus_per_task",
            "max_memory_bytes",
        )
        for name in positive_fields:
            _exact_int(getattr(self, name), name=name, minimum=1)
        payload = _envelope_payload(
            profile_run_sha256=run.profile_run_sha256,
            scheduler_raw_evidence_sha256=self.scheduler_raw_evidence_sha256,
            resource_raw_evidence_sha256=self.resource_raw_evidence_sha256,
            partition=partition,
            gpu_name=gpu_name,
            gpu_total_memory_bytes=self.gpu_total_memory_bytes,
            max_allocation_wall_seconds=self.max_allocation_wall_seconds,
            max_array_concurrency=self.max_array_concurrency,
            max_scheduler_segments=self.max_scheduler_segments,
            max_gpus_per_task=self.max_gpus_per_task,
            max_cpus_per_task=self.max_cpus_per_task,
            max_memory_bytes=self.max_memory_bytes,
        )
        _digest(self.envelope_sha256, name="envelope_sha256")
        if _canonical_sha256(payload) != self.envelope_sha256:
            raise ValueError("scheduler/resource envelope SHA256 does not match")

    def validate_integrity(self) -> None:
        _require_factory(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _envelope_payload(
            profile_run_sha256=self.profile_run_sha256,
            scheduler_raw_evidence_sha256=self.scheduler_raw_evidence_sha256,
            resource_raw_evidence_sha256=self.resource_raw_evidence_sha256,
            partition=self.partition,
            gpu_name=self.gpu_name,
            gpu_total_memory_bytes=self.gpu_total_memory_bytes,
            max_allocation_wall_seconds=self.max_allocation_wall_seconds,
            max_array_concurrency=self.max_array_concurrency,
            max_scheduler_segments=self.max_scheduler_segments,
            max_gpus_per_task=self.max_gpus_per_task,
            max_cpus_per_task=self.max_cpus_per_task,
            max_memory_bytes=self.max_memory_bytes,
        )
        return {**payload, "envelope_sha256": self.envelope_sha256}


def validate_scheduler_resource_envelope(
    profile_run: ValidatedGatePRun,
    *,
    scheduler_raw_evidence_sha256: str,
    resource_raw_evidence_sha256: str,
    partition: str,
    gpu_name: str,
    gpu_total_memory_bytes: int,
    max_allocation_wall_seconds: int,
    max_array_concurrency: int,
    max_scheduler_segments: int,
    max_gpus_per_task: int,
    max_cpus_per_task: int,
    max_memory_bytes: int,
) -> SchedulerResourceEnvelope:
    """Bind decoded limits to the exact raw evidence captured on HPC4."""

    run = _validated_run(profile_run)
    payload = _envelope_payload(
        profile_run_sha256=run.profile_run_sha256,
        scheduler_raw_evidence_sha256=_digest(
            scheduler_raw_evidence_sha256,
            name="scheduler_raw_evidence_sha256",
        ),
        resource_raw_evidence_sha256=_digest(
            resource_raw_evidence_sha256,
            name="resource_raw_evidence_sha256",
        ),
        partition=_text(partition, name="partition"),
        gpu_name=_text(gpu_name, name="gpu_name"),
        gpu_total_memory_bytes=_exact_int(
            gpu_total_memory_bytes,
            name="gpu_total_memory_bytes",
            minimum=1,
        ),
        max_allocation_wall_seconds=_exact_int(
            max_allocation_wall_seconds,
            name="max_allocation_wall_seconds",
            minimum=1,
        ),
        max_array_concurrency=_exact_int(
            max_array_concurrency,
            name="max_array_concurrency",
            minimum=1,
        ),
        max_scheduler_segments=_exact_int(
            max_scheduler_segments,
            name="max_scheduler_segments",
            minimum=1,
        ),
        max_gpus_per_task=_exact_int(
            max_gpus_per_task,
            name="max_gpus_per_task",
            minimum=1,
        ),
        max_cpus_per_task=_exact_int(
            max_cpus_per_task,
            name="max_cpus_per_task",
            minimum=1,
        ),
        max_memory_bytes=_exact_int(
            max_memory_bytes,
            name="max_memory_bytes",
            minimum=1,
        ),
    )
    result = SchedulerResourceEnvelope(
        profile_run=run,
        **payload,
        envelope_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _preparation_payload(
    *,
    profile_run_sha256: str,
    artifact_verification_wall_seconds: float,
    oracle_rescore_wall_seconds: float,
    label_reconstruction_wall_seconds: float,
) -> dict[str, object]:
    return {
        "schema_version": PROFILE_PREPARATION_TIMINGS_SCHEMA,
        "profile_run_sha256": profile_run_sha256,
        "artifact_verification_wall_seconds": artifact_verification_wall_seconds,
        "oracle_rescore_wall_seconds": oracle_rescore_wall_seconds,
        "label_reconstruction_wall_seconds": label_reconstruction_wall_seconds,
        "source_artifacts_reverified": True,
        "labels_reconstructed_from_attested_train_only_source": True,
        "heldout_bytes_decoded": False,
    }


@dataclass(frozen=True, slots=True)
class ProfilePreparationTimings:
    """Upstream train-only preparation timings captured before trainer entry."""

    profile_run: ValidatedGatePRun = field(repr=False, compare=False)
    schema_version: str
    profile_run_sha256: str
    artifact_verification_wall_seconds: float
    oracle_rescore_wall_seconds: float
    label_reconstruction_wall_seconds: float
    source_artifacts_reverified: Literal[True]
    labels_reconstructed_from_attested_train_only_source: Literal[True]
    heldout_bytes_decoded: Literal[False]
    preparation_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        run = _validated_run(self.profile_run)
        if self.schema_version != PROFILE_PREPARATION_TIMINGS_SCHEMA:
            raise ValueError("profile preparation timing schema is not frozen")
        if self.profile_run_sha256 != run.profile_run_sha256:
            raise ValueError("profile preparation timings bind another Gate-P run")
        artifact_seconds = _positive_real(
            self.artifact_verification_wall_seconds,
            name="artifact_verification_wall_seconds",
        )
        oracle_seconds = _positive_real(
            self.oracle_rescore_wall_seconds,
            name="oracle_rescore_wall_seconds",
        )
        label_seconds = _positive_real(
            self.label_reconstruction_wall_seconds,
            name="label_reconstruction_wall_seconds",
        )
        if (
            self.source_artifacts_reverified is not True
            or self.labels_reconstructed_from_attested_train_only_source is not True
            or self.heldout_bytes_decoded is not False
        ):
            raise ValueError("profile preparation crossed its train-only boundary")
        payload = _preparation_payload(
            profile_run_sha256=run.profile_run_sha256,
            artifact_verification_wall_seconds=artifact_seconds,
            oracle_rescore_wall_seconds=oracle_seconds,
            label_reconstruction_wall_seconds=label_seconds,
        )
        _digest(self.preparation_sha256, name="preparation_sha256")
        if _canonical_sha256(payload) != self.preparation_sha256:
            raise ValueError("profile preparation SHA256 does not match")

    def validate_integrity(self) -> None:
        _require_factory(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _preparation_payload(
            profile_run_sha256=self.profile_run_sha256,
            artifact_verification_wall_seconds=self.artifact_verification_wall_seconds,
            oracle_rescore_wall_seconds=self.oracle_rescore_wall_seconds,
            label_reconstruction_wall_seconds=self.label_reconstruction_wall_seconds,
        )
        return {**payload, "preparation_sha256": self.preparation_sha256}


def record_profile_preparation_timings(
    profile_run: ValidatedGatePRun,
    *,
    artifact_verification_wall_seconds: float,
    oracle_rescore_wall_seconds: float,
    label_reconstruction_wall_seconds: float,
) -> ProfilePreparationTimings:
    """Record required upstream timings without retaining labels or rewards."""

    run = _validated_run(profile_run)
    payload = _preparation_payload(
        profile_run_sha256=run.profile_run_sha256,
        artifact_verification_wall_seconds=_positive_real(
            artifact_verification_wall_seconds,
            name="artifact_verification_wall_seconds",
        ),
        oracle_rescore_wall_seconds=_positive_real(
            oracle_rescore_wall_seconds,
            name="oracle_rescore_wall_seconds",
        ),
        label_reconstruction_wall_seconds=_positive_real(
            label_reconstruction_wall_seconds,
            name="label_reconstruction_wall_seconds",
        ),
    )
    result = ProfilePreparationTimings(
        profile_run=run,
        **payload,
        preparation_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def record_profile_preparation_from_train_input(
    profile_run: ValidatedGatePRun,
    input_timings: object,
) -> ProfilePreparationTimings:
    """Bind all three measured input phases into the Gate-P projection once."""

    from .phase2_r3_inputs import R3InputPreparationTimings

    run = _validated_run(profile_run)
    if type(input_timings) is not R3InputPreparationTimings:
        raise TypeError("input_timings must be exactly R3InputPreparationTimings")
    input_timings.validate_integrity()
    if input_timings.materialization_attestation_sha256 != run.materialization.attestation_sha256:
        raise ValueError("input timings belong to another train materialization")
    return record_profile_preparation_timings(
        run,
        artifact_verification_wall_seconds=(input_timings.artifact_verification_wall_seconds),
        oracle_rescore_wall_seconds=input_timings.oracle_rescore_wall_seconds,
        label_reconstruction_wall_seconds=(input_timings.label_reconstruction_wall_seconds),
    )


def _require_live_cuda(profile_run: ValidatedGatePRun) -> dict[str, object]:
    context = profile_run.materialization.context
    device = context.training.reward_features.device
    if device.type != "cuda":
        raise RuntimeError("formal Gate-P profiling requires CUDA-resident train tensors")
    if not torch.cuda.is_available():
        raise RuntimeError("formal Gate-P profiling requires an available CUDA runtime")
    logical_index = torch.cuda.current_device() if device.index is None else device.index
    if logical_index < 0 or logical_index >= torch.cuda.device_count():
        raise RuntimeError("formal Gate-P CUDA device index is unavailable")
    properties = torch.cuda.get_device_properties(logical_index)
    cuda_version = torch.version.cuda
    if not isinstance(cuda_version, str) or not cuda_version:
        raise RuntimeError("formal Gate-P profiling requires a reported CUDA version")
    torch.cuda.synchronize(device)
    return {
        "logical_device_index": int(logical_index),
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability_major": int(properties.major),
        "compute_capability_minor": int(properties.minor),
        "torch_cuda_version": cuda_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _nvidia_smi_selector(logical_index: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",")]
        if logical_index >= len(devices) or not devices[logical_index]:
            raise RuntimeError("CUDA_VISIBLE_DEVICES does not cover the live logical device")
        return devices[logical_index]
    return str(logical_index)


def _sample_gpu_utilization(
    cuda_identity: Mapping[str, object],
    *,
    sample_index: int,
) -> dict[str, object]:
    logical_index = _exact_int(
        cuda_identity["logical_device_index"],
        name="logical_device_index",
    )
    selector = _nvidia_smi_selector(logical_index)
    command = [
        "nvidia-smi",
        f"--id={selector}",
        "--query-gpu=uuid,name,memory.total,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi utilization sample failed")
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("nvidia-smi did not return exactly one GPU row")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError("nvidia-smi utilization row has an invalid schema")
    uuid, name, total_mib_raw, gpu_util_raw, memory_util_raw = fields
    try:
        total_memory_bytes = int(float(total_mib_raw) * 1024 * 1024)
        gpu_utilization = float(gpu_util_raw)
        memory_utilization = float(memory_util_raw)
    except ValueError as error:
        raise RuntimeError("nvidia-smi utilization sample is non-numeric") from error
    for metric_name, metric in (
        ("gpu_utilization_percent", gpu_utilization),
        ("memory_utilization_percent", memory_utilization),
    ):
        if not math.isfinite(metric) or not 0.0 <= metric <= 100.0:
            raise RuntimeError(f"nvidia-smi {metric_name} is outside [0, 100]")
    return {
        "sample_index": sample_index,
        "wall_time_ns": time.time_ns(),
        "monotonic_time_ns": time.perf_counter_ns(),
        "uuid": _text(uuid, name="gpu uuid"),
        "name": _text(name, name="gpu name"),
        "total_memory_bytes": total_memory_bytes,
        "gpu_utilization_percent": gpu_utilization,
        "memory_utilization_percent": memory_utilization,
    }


def _read_process_memory() -> dict[str, object]:
    status = Path("/proc/self/status")
    try:
        lines = status.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError("formal Gate-P requires Linux /proc RSS evidence") from error
    parsed: dict[str, int] = {}
    for line in lines:
        if line.startswith(("VmRSS:", "VmHWM:")):
            parts = line.split()
            if len(parts) != 3 or parts[2] != "kB":
                raise RuntimeError("/proc RSS evidence has an invalid unit/schema")
            try:
                parsed[parts[0].rstrip(":")] = int(parts[1]) * 1024
            except ValueError as error:
                raise RuntimeError("/proc RSS evidence is non-numeric") from error
    if set(parsed) != {"VmRSS", "VmHWM"}:
        raise RuntimeError("/proc RSS evidence omitted current or peak memory")
    current = _exact_int(parsed["VmRSS"], name="current_rss_bytes", minimum=1)
    peak = _exact_int(parsed["VmHWM"], name="peak_rss_bytes", minimum=current)
    return {
        "current_rss_bytes": current,
        "peak_rss_bytes": peak,
        "measurement": "linux_proc_status_vmrss_vmhwm",
    }


def _validate_cuda_identity(
    value: object,
    *,
    envelope: SchedulerResourceEnvelope,
) -> dict[str, object]:
    identity = _exact_mapping(value, _CUDA_IDENTITY_FIELDS, name="cuda_identity")
    index = _exact_int(identity["logical_device_index"], name="logical_device_index")
    name = _text(identity["name"], name="cuda_identity.name")
    memory = _exact_int(
        identity["total_memory_bytes"],
        name="cuda_identity.total_memory_bytes",
        minimum=1,
    )
    _exact_int(
        identity["compute_capability_major"],
        name="compute_capability_major",
        minimum=1,
    )
    _exact_int(
        identity["compute_capability_minor"],
        name="compute_capability_minor",
    )
    _text(identity["torch_cuda_version"], name="torch_cuda_version")
    visible = identity["cuda_visible_devices"]
    if visible is not None and type(visible) is not str:
        raise TypeError("cuda_visible_devices must be string or null")
    if name != envelope.gpu_name or memory != envelope.gpu_total_memory_bytes:
        raise ValueError("live CUDA identity differs from the frozen resource envelope")
    normalized = dict(identity)
    normalized["logical_device_index"] = index
    return normalized


def _validate_gpu_samples(
    value: object,
    *,
    cuda_identity: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("gpu_utilization_samples must be a sequence")
    if len(value) != 2:
        raise ValueError("formal Gate-P requires exactly before/after GPU samples")
    normalized: list[dict[str, object]] = []
    first_uuid: str | None = None
    previous_monotonic: int | None = None
    previous_wall: int | None = None
    live_memory = _exact_int(
        cuda_identity["total_memory_bytes"],
        name="cuda_identity.total_memory_bytes",
        minimum=1,
    )
    for expected_index, raw in enumerate(value):
        sample = _exact_mapping(
            raw,
            _GPU_SAMPLE_FIELDS,
            name=f"gpu_utilization_samples[{expected_index}]",
        )
        if sample["sample_index"] != expected_index:
            raise ValueError("GPU utilization samples are not in exact before/after order")
        wall_ns = _exact_int(sample["wall_time_ns"], name="wall_time_ns", minimum=1)
        if previous_wall is not None and wall_ns <= previous_wall:
            raise ValueError("GPU utilization wall-clock sample time did not advance")
        previous_wall = wall_ns
        monotonic_ns = _exact_int(
            sample["monotonic_time_ns"],
            name="monotonic_time_ns",
            minimum=1,
        )
        if previous_monotonic is not None and monotonic_ns <= previous_monotonic:
            raise ValueError("GPU utilization monotonic sample time did not advance")
        previous_monotonic = monotonic_ns
        uuid = _text(sample["uuid"], name="gpu uuid")
        if first_uuid is None:
            first_uuid = uuid
        elif uuid != first_uuid:
            raise ValueError("GPU UUID changed during the formal profile")
        if sample["name"] != cuda_identity["name"]:
            raise ValueError("nvidia-smi GPU name differs from the live Torch device")
        sampled_memory = _exact_int(
            sample["total_memory_bytes"],
            name="sample.total_memory_bytes",
            minimum=1,
        )
        if abs(sampled_memory - live_memory) > 2 * 1024 * 1024:
            raise ValueError("nvidia-smi and Torch total GPU memory disagree")
        for name in ("gpu_utilization_percent", "memory_utilization_percent"):
            metric = _nonnegative_real(sample[name], name=name)
            if metric > 100.0:
                raise ValueError(f"{name} must not exceed 100")
        normalized.append({**dict(sample), "wall_time_ns": wall_ns})
    return normalized[0], normalized[1]


def _validate_cpu_memory(value: object) -> dict[str, object]:
    memory = _exact_mapping(value, _CPU_MEMORY_FIELDS, name="cpu_memory")
    current = _exact_int(
        memory["current_rss_bytes"],
        name="current_rss_bytes",
        minimum=1,
    )
    _exact_int(memory["peak_rss_bytes"], name="peak_rss_bytes", minimum=current)
    if memory["measurement"] != "linux_proc_status_vmrss_vmhwm":
        raise ValueError("CPU memory measurement is not the formal Linux RSS source")
    return dict(memory)


def _strict_core_profile(
    value: object,
    *,
    profile_run: ValidatedGatePRun,
) -> dict[str, object]:
    validate_gate_p_profile_core_result(value)
    copied = _json_copy(value, name="core_profile")
    if not isinstance(copied, dict):
        raise TypeError("core_profile must be a JSON object")
    expected = {
        "seed": profile_run.seed,
        "context_sha256": profile_run.materialization.context.context_sha256,
        "settings_sha256": profile_run.science.settings.sha256,
        "input_training_sha256": (profile_run.materialization.context.input_training_sha256),
        "learner_order": list(profile_run.head_order),
        "update_cap_per_learner": profile_run.completed_updates_per_head,
        "audit_update_indices": list(profile_run.audit_updates),
        "stop_reason": profile_run.stop_reason,
        "binding_sha256": _canonical_sha256(
            profile_core_binding(profile_run.materialization.context)
        ),
        "device_type": "cuda",
        "formal_cuda_profile": True,
        "profile_nonreusable": True,
    }
    for name, expected_value in expected.items():
        if copied[name] != expected_value:
            raise ValueError(f"core profile {name} is not bound to the formal Gate-P run")
    if copied["information_boundary"] != _CORE_BOUNDARY:
        raise ValueError("core profile crossed the formal train-only boundary")
    _positive_real(copied["setup"]["wall_seconds"], name="setup.wall_seconds")  # type: ignore[index]
    setup_memory = copied["setup"]["cuda_memory"]  # type: ignore[index]
    if setup_memory["measurement"] != "cuda_allocator":  # type: ignore[index]
        raise ValueError("formal core setup lacks CUDA allocator evidence")
    _exact_int(setup_memory["peak_bytes"], name="setup.peak_bytes", minimum=1)  # type: ignore[index]
    learners = copied["learners"]
    if not isinstance(learners, list):
        raise TypeError("core learners must be a list")
    for expected_learner, learner in zip(
        PHASE2_PROFILE_LEARNER_ORDER,
        learners,
        strict=True,
    ):
        if learner["learner"] != expected_learner:
            raise ValueError("core learner order changed")
        _positive_real(learner["build_wall_seconds"], name="build_wall_seconds")
        _positive_real(learner["phase_wall_seconds"], name="phase_wall_seconds")
        if (
            learner["updates_executed"] != PHASE2_PROFILE_UPDATES
            or learner["stop_reason"] != PHASE2_PROFILE_STOP_REASON
        ):
            raise ValueError("core learner did not stop at exactly 100 updates")
        for step in learner["steps"]:
            _positive_real(step["wall_seconds"], name="step.wall_seconds")
            _exact_int(
                step["cuda_memory"]["peak_bytes"],
                name="step.cuda_memory.peak_bytes",
                minimum=1,
            )
            if expected_learner == "prorm_plus":
                pcg = step["pcg"]
                if pcg["reason"] not in _PCG_REASONS:
                    raise ValueError("formal ProRM+ profile omitted raw PCG reason")
        for audit in learner["audits"]:
            _positive_real(audit["wall_seconds"], name="audit.wall_seconds")
            if audit["trainer_state_unchanged"] is not True:
                raise ValueError("formal profile audit mutated trainer state")
        for probe in learner["ephemeral_checkpoint_io"]:
            _exact_int(
                probe["serialized_bytes"],
                name="probe.serialized_bytes",
                minimum=1,
            )
            for field_name in (
                "serialize_wall_seconds",
                "fsync_wall_seconds",
                "reload_wall_seconds",
            ):
                _positive_real(probe[field_name], name=f"probe.{field_name}")
            if (
                probe["roundtrip_verified"] is not True
                or probe["artifact_retained"] is not False
                or probe["reusable"] is not False
                or probe["filesystem_scope"] != "declared_profile_directory"
            ):
                raise ValueError("formal checkpoint I/O probe is incomplete or reusable")
    _assert_no_sensitive_state(copied, path="core_profile")
    return copied


_PRODUCTION_IO_SAMPLE_FIELDS = {
    "update",
    "serialized_bytes",
    "serialize_wall_seconds",
    "fsync_wall_seconds",
    "reload_and_verify_wall_seconds",
    "progress_receipt_publish_fsync_verify_wall_seconds",
    "signal_or_planned_boundary_receipt_publish_fsync_verify_wall_seconds",
    "final_restore_load_terminal_payload_wall_seconds",
    "production_outer_payload_isomorphic_profile_schema",
    "durable_outer_envelope_exact_schema",
    "profile_role",
    "reusable_as_primary_state",
    "same_pass_live_trainer_state",
    "full_live_history_records",
    "worst_case_selected_state_copy_included",
    "selected_state_history_records",
    "convergence_check_records",
    "recovery_state_check_records",
    "fixed_tensor_components_included",
    "fixed_snapshot_included",
    "legacy_boundary_snapshot_included",
    "optimizer_protocol_stage_records",
    "optimizer_protocol_transition_records",
    "primary_outer_exact_key_contract_verified",
    "production_controller_builder_used",
    "convergence_check_exact_key_contract_verified",
    "snapshot_exact_key_contract_verified",
    "optimizer_protocol_execution_exact_key_contract_verified",
    "selected_terminal_production_builder_used",
    "fresh_trainer_restore_load_measured",
    "atomic_publication_fsync_included",
    "committed_byte_verification_included",
}
_PRODUCTION_IO_LEARNER_FIELDS = {
    "learner",
    "samples",
    "fixed_serialized_bytes_upper_bound",
    "per_update_serialized_byte_slope_upper_bound",
    "target_serialized_bytes_upper_bound",
    "minimum_serialize_throughput_bytes_per_second",
    "minimum_fsync_throughput_bytes_per_second",
    "minimum_reload_verify_throughput_bytes_per_second",
    "maximum_progress_receipt_wall_seconds",
    "maximum_boundary_receipt_wall_seconds",
    "maximum_finalization_noncheckpoint_wall_seconds",
}
_PRODUCTION_IO_GROWTH_FIELDS = {
    "formula",
    "live_trainer_history_records_per_update",
    "selected_state_history_records_per_update",
    "convergence_check_records_per_audit_block",
    "recovery_state_check_records_per_update",
    "only_declared_linear_fields_extrapolated",
    "fixed_tensor_components_measured_at_every_sample",
    "worst_case_selected_state_copy_measured_at_every_sample",
    "nonlinear_or_unbounded_fields_absent",
    "simple_fixed_io_times_event_count_forbidden",
    "target_12760_envelope_actually_serialized",
    "projection_uses_measured_target_not_linear_extrapolation",
}
_PRODUCTION_IO_FIELDS = {
    "schema_version",
    "capture_mode",
    "core_profile_sha256",
    "production_outer_payload_schema",
    "benchmark_payload_schema",
    "durable_checkpoint_envelope_schema",
    "controller_checkpoint_schema",
    "sample_updates",
    "schema_growth_model",
    "learners",
    "information_boundary",
    "evidence_sha256",
}


def _validate_production_checkpoint_io_evidence(
    value: object,
    *,
    core_profile: Mapping[str, object],
) -> dict[str, object]:
    """Validate same-pass, production-schema checkpoint I/O evidence.

    The current claim-free core exposes only ``trainer.state_dict()`` probes.
    Formal promotion additionally requires a producer that has access to each
    live trainer at the same six boundaries and serializes the complete R3
    outer checkpoint envelope.  No synthetic copy multiplier is accepted here.
    """

    copied = _json_copy(value, name="production_checkpoint_io_evidence")
    evidence = _exact_mapping(
        copied,
        _PRODUCTION_IO_FIELDS,
        name="production_checkpoint_io_evidence",
    )
    if evidence["schema_version"] != PRODUCTION_CHECKPOINT_IO_PROFILE_SCHEMA:
        raise ValueError("production checkpoint I/O evidence schema changed")
    if evidence["capture_mode"] != (
        "same_pass_live_production_outer_envelope_worst_case_selected_copy"
    ):
        raise ValueError("production checkpoint evidence was not captured in the live pass")
    if evidence["core_profile_sha256"] != core_profile["profile_sha256"]:
        raise ValueError("production checkpoint evidence belongs to another core profile")
    exact_schemas = {
        "production_outer_payload_schema": (PRODUCTION_OUTER_CHECKPOINT_PAYLOAD_SCHEMA),
        "benchmark_payload_schema": PRODUCTION_PROFILE_BENCHMARK_PAYLOAD_SCHEMA,
        "durable_checkpoint_envelope_schema": (PRODUCTION_DURABLE_CHECKPOINT_ENVELOPE_SCHEMA),
        "controller_checkpoint_schema": PRODUCTION_CONTROLLER_CHECKPOINT_SCHEMA,
    }
    for name, expected in exact_schemas.items():
        if evidence[name] != expected:
            raise ValueError(f"production checkpoint evidence {name} changed")
    diagnostic_updates = list(PHASE2_PROFILE_AUDIT_UPDATES)
    expected_updates = [*diagnostic_updates, R3_MAXIMUM_UPDATES_PER_HEAD]
    if evidence["sample_updates"] != expected_updates:
        raise ValueError("production checkpoint evidence must cover 0/20/.../100")
    growth = _exact_mapping(
        evidence["schema_growth_model"],
        _PRODUCTION_IO_GROWTH_FIELDS,
        name="schema_growth_model",
    )
    expected_growth = {
        "formula": (
            "diagnostic_bytes(step<=100)=fixed+slope*step;"
            "projection_bound(step>100)=measured_target_12760_envelope"
        ),
        "live_trainer_history_records_per_update": 1,
        "selected_state_history_records_per_update": 1,
        "convergence_check_records_per_audit_block": 1,
        "recovery_state_check_records_per_update": 2,
        "only_declared_linear_fields_extrapolated": True,
        "fixed_tensor_components_measured_at_every_sample": True,
        "worst_case_selected_state_copy_measured_at_every_sample": True,
        "nonlinear_or_unbounded_fields_absent": True,
        "simple_fixed_io_times_event_count_forbidden": True,
        "target_12760_envelope_actually_serialized": True,
        "projection_uses_measured_target_not_linear_extrapolation": True,
    }
    if dict(growth) != expected_growth:
        raise ValueError("production checkpoint schema-growth certificate is incomplete")
    if evidence["information_boundary"] != FORMAL_PROFILE_INFORMATION_BOUNDARY:
        raise ValueError("production checkpoint I/O evidence crossed train-only boundary")

    core_learners = core_profile["learners"]
    raw_learners = evidence["learners"]
    if (
        not isinstance(core_learners, list)
        or not isinstance(raw_learners, list)
        or len(raw_learners) != len(PHASE2_PROFILE_LEARNER_ORDER)
    ):
        raise ValueError("production checkpoint evidence must cover exactly two heads")
    for expected_learner, raw_learner, core_learner in zip(
        PHASE2_PROFILE_LEARNER_ORDER,
        raw_learners,
        core_learners,
        strict=True,
    ):
        learner = _exact_mapping(
            raw_learner,
            _PRODUCTION_IO_LEARNER_FIELDS,
            name=f"production_io.{expected_learner}",
        )
        if learner["learner"] != expected_learner or core_learner["learner"] != expected_learner:
            raise ValueError("production checkpoint learner order changed")
        raw_samples = learner["samples"]
        core_samples = core_learner["ephemeral_checkpoint_io"]
        if (
            not isinstance(raw_samples, list)
            or not isinstance(core_samples, list)
            or len(raw_samples) != len(expected_updates)
            or len(core_samples) != len(diagnostic_updates)
        ):
            raise ValueError("production checkpoint sample count is incomplete")
        sizes: list[int] = []
        serialize_throughputs: list[float] = []
        fsync_throughputs: list[float] = []
        reload_throughputs: list[float] = []
        progress_receipt_times: list[float] = []
        boundary_receipt_times: list[float] = []
        finalization_times: list[float] = []
        for sample_index, (update, raw_sample) in enumerate(
            zip(expected_updates, raw_samples, strict=True)
        ):
            sample = _exact_mapping(
                raw_sample,
                _PRODUCTION_IO_SAMPLE_FIELDS,
                name=f"production_io.{expected_learner}.sample[{update}]",
            )
            size = _exact_int(
                sample["serialized_bytes"],
                name="production serialized_bytes",
                minimum=1,
            )
            if sample["update"] != update:
                raise ValueError("production checkpoint sample update differs")
            if sample_index < len(core_samples):
                core_sample = core_samples[sample_index]
                core_size = _exact_int(
                    core_sample["serialized_bytes"],
                    name="core serialized_bytes",
                    minimum=1,
                )
                if core_sample["update"] != update:
                    raise ValueError("production and core checkpoint sample updates differ")
                if size < core_size:
                    raise ValueError("production outer checkpoint is smaller than trainer state")
            flags = (
                "production_outer_payload_isomorphic_profile_schema",
                "durable_outer_envelope_exact_schema",
                "profile_role",
                "same_pass_live_trainer_state",
                "worst_case_selected_state_copy_included",
                "fixed_tensor_components_included",
                "fixed_snapshot_included",
                "legacy_boundary_snapshot_included",
                "primary_outer_exact_key_contract_verified",
                "production_controller_builder_used",
                "convergence_check_exact_key_contract_verified",
                "snapshot_exact_key_contract_verified",
                "optimizer_protocol_execution_exact_key_contract_verified",
                "fresh_trainer_restore_load_measured",
                "atomic_publication_fsync_included",
                "committed_byte_verification_included",
            )
            if any(sample[name] is not True for name in flags):
                raise ValueError("production checkpoint sample omitted required outer state")
            if sample["reusable_as_primary_state"] is not False:
                raise ValueError("profile checkpoint benchmark cannot be primary state")
            if (
                sample["full_live_history_records"] != update
                or sample["selected_state_history_records"] != update
                or sample["convergence_check_records"] != update // R3_AUDIT_CADENCE_UPDATES
                or sample["recovery_state_check_records"] != 2 * update
                or sample["optimizer_protocol_stage_records"] != 5
                or sample["optimizer_protocol_transition_records"] != 4
            ):
                raise ValueError("production checkpoint sample growth counts are wrong")
            if sample["selected_terminal_production_builder_used"] is not (update > 0):
                raise ValueError("selected-terminal production builder coverage is inconsistent")
            serialize = _positive_real(
                sample["serialize_wall_seconds"],
                name="production serialize_wall_seconds",
            )
            fsync = _positive_real(
                sample["fsync_wall_seconds"],
                name="production fsync_wall_seconds",
            )
            reload_verify = _positive_real(
                sample["reload_and_verify_wall_seconds"],
                name="production reload_and_verify_wall_seconds",
            )
            progress_receipt = _positive_real(
                sample["progress_receipt_publish_fsync_verify_wall_seconds"],
                name="progress receipt wall seconds",
            )
            boundary_receipt = _positive_real(
                sample["signal_or_planned_boundary_receipt_publish_fsync_verify_wall_seconds"],
                name="signal or planned-boundary receipt wall seconds",
            )
            finalization = _positive_real(
                sample["final_restore_load_terminal_payload_wall_seconds"],
                name="final restore/load/terminal payload wall seconds",
            )
            sizes.append(size)
            serialize_throughputs.append(size / serialize)
            fsync_throughputs.append(size / fsync)
            reload_throughputs.append(size / reload_verify)
            progress_receipt_times.append(progress_receipt)
            boundary_receipt_times.append(boundary_receipt)
            finalization_times.append(finalization)
        if any(current < previous for previous, current in zip(sizes, sizes[1:], strict=False)):
            raise ValueError("production checkpoint bytes decreased with growing state")
        diagnostic_sizes = sizes[:-1]
        observed_slope = max(
            math.ceil(
                (diagnostic_sizes[right] - diagnostic_sizes[left])
                / (diagnostic_updates[right] - diagnostic_updates[left])
            )
            for left in range(len(diagnostic_sizes))
            for right in range(left + 1, len(diagnostic_sizes))
        )
        slope = _exact_int(
            learner["per_update_serialized_byte_slope_upper_bound"],
            name="per_update_serialized_byte_slope_upper_bound",
            minimum=1,
        )
        if slope != max(1, observed_slope):
            raise ValueError("production checkpoint byte slope is not sample-derived")
        fixed = _exact_int(
            learner["fixed_serialized_bytes_upper_bound"],
            name="fixed_serialized_bytes_upper_bound",
            minimum=1,
        )
        expected_fixed = max(
            size - slope * update
            for update, size in zip(
                diagnostic_updates,
                diagnostic_sizes,
                strict=True,
            )
        )
        if fixed != expected_fixed:
            raise ValueError("production checkpoint fixed-byte bound is not derived")
        if any(
            fixed + slope * update < size
            for update, size in zip(
                diagnostic_updates,
                diagnostic_sizes,
                strict=True,
            )
        ):
            raise ValueError("production checkpoint byte model misses a measured sample")
        target = sizes[-1]
        if (
            learner["target_serialized_bytes_upper_bound"] != target
            or target < fixed + slope * R3_MAXIMUM_UPDATES_PER_HEAD
        ):
            raise ValueError("production checkpoint target byte bound is inconsistent")
        for name, observed in (
            (
                "minimum_serialize_throughput_bytes_per_second",
                min(serialize_throughputs),
            ),
            (
                "minimum_fsync_throughput_bytes_per_second",
                min(fsync_throughputs),
            ),
            (
                "minimum_reload_verify_throughput_bytes_per_second",
                min(reload_throughputs),
            ),
        ):
            declared = _positive_real(learner[name], name=name)
            if not math.isclose(declared, observed, rel_tol=1.0e-12, abs_tol=0.0):
                raise ValueError(f"{name} is not the measured conservative minimum")
        for name, observed in (
            (
                "maximum_progress_receipt_wall_seconds",
                max(progress_receipt_times),
            ),
            (
                "maximum_boundary_receipt_wall_seconds",
                max(boundary_receipt_times),
            ),
            (
                "maximum_finalization_noncheckpoint_wall_seconds",
                max(finalization_times),
            ),
        ):
            declared = _positive_real(learner[name], name=name)
            if not math.isclose(declared, observed, rel_tol=1.0e-12, abs_tol=0.0):
                raise ValueError(f"{name} is not the measured conservative maximum")

    evidence_sha = _digest(evidence["evidence_sha256"], name="evidence_sha256")
    unhashed = dict(evidence)
    del unhashed["evidence_sha256"]
    if _canonical_sha256(unhashed) != evidence_sha:
        raise ValueError("production checkpoint I/O evidence SHA256 is invalid")
    _assert_no_sensitive_state(evidence, path="production_checkpoint_io_evidence")
    return dict(evidence)


def _expanded_profile_trainer_state(
    trainer_state: Mapping[str, object],
    *,
    target_update: int,
) -> dict[str, object]:
    copied = copy.deepcopy(dict(trainer_state))
    history = copied.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("target checkpoint benchmark requires live trainer history")
    templates = [copy.deepcopy(record) for record in history]
    expanded: list[object] = []
    for step in range(1, target_update + 1):
        record = copy.deepcopy(templates[(step - 1) % len(templates)])
        if not isinstance(record, dict):
            raise TypeError("trainer history record must be a mapping")
        record["step"] = step
        expanded.append(record)
    copied["history"] = expanded
    copied["completed_steps"] = target_update
    if "dual_refreshes" in copied:
        copied["dual_refreshes"] = target_update
    return copied


def _profile_measurement_payload() -> dict[str, object]:
    return {
        "objective": _PROFILE_MAX_FINITE_FLOAT,
        "gradient_l2_norm": _PROFILE_MAX_FINITE_FLOAT,
        "inner_solver": {
            "method": "pcg",
            "dtype": "float64",
            "cold_start": True,
            "warm_start_used": False,
            "iterations": _PROFILE_MAX_SIGNED_INT,
            "residual_norm": _PROFILE_MAX_FINITE_FLOAT,
            "relative_residual": _PROFILE_MAX_FINITE_FLOAT,
            "converged": True,
        },
        "audit_dtype": "float64",
    }


def _profile_measurement() -> _training._FirstOrderMeasurement:
    payload = _profile_measurement_payload()
    return _training._FirstOrderMeasurement(
        objective=float(payload["objective"]),
        gradient_l2_norm=float(payload["gradient_l2_norm"]),
        inner_solver=payload["inner_solver"],  # type: ignore[arg-type]
        audit_dtype=str(payload["audit_dtype"]),
    )


def _profile_history_summary() -> dict[str, object]:
    diagnostic = {
        "step": _PROFILE_MAX_SIGNED_INT,
        "objective": _PROFILE_MAX_FINITE_FLOAT,
        "gradient_norm": _PROFILE_MAX_FINITE_FLOAT,
        "dual_loss": _PROFILE_MAX_FINITE_FLOAT,
        "dual_saddle_value": _PROFILE_MAX_FINITE_FLOAT,
        "dual_refresh": True,
        "pcg_iterations": _PROFILE_MAX_SIGNED_INT,
        "pcg_residual_norm": _PROFILE_MAX_FINITE_FLOAT,
        "pcg_relative_residual": _PROFILE_MAX_FINITE_FLOAT,
        "pcg_converged": True,
    }
    return {
        "num_steps": _PROFILE_MAX_SIGNED_INT,
        "history_objective_timing": "pre_update",
        "stored_checkpoint_steps": [
            1,
            536_870_911,
            1_073_741_823,
            1_610_612_735,
            _PROFILE_MAX_SIGNED_INT,
        ],
        "checkpoints": [copy.deepcopy(diagnostic) for _ in range(5)],
        "objective": {
            "first": _PROFILE_MAX_FINITE_FLOAT,
            "last_pre_update": _PROFILE_MAX_FINITE_FLOAT,
            "minimum": _PROFILE_MAX_FINITE_FLOAT,
            "maximum": _PROFILE_MAX_FINITE_FLOAT,
        },
        "gradient_l2_norm": {
            "first": _PROFILE_MAX_FINITE_FLOAT,
            "last_pre_update": _PROFILE_MAX_FINITE_FLOAT,
            "minimum": _PROFILE_MAX_FINITE_FLOAT,
            "maximum": _PROFILE_MAX_FINITE_FLOAT,
        },
        "pcg": {
            "num_fresh_solves": _PROFILE_MAX_SIGNED_INT,
            "all_converged": True,
            "maximum_relative_residual": _PROFILE_MAX_FINITE_FLOAT,
            "maximum_iterations": _PROFILE_MAX_SIGNED_INT,
        },
    }


def _profile_snapshots(
    *,
    protocol: _training.AdamWRecoveryProtocol,
) -> tuple[dict[str, object], dict[str, object]]:
    common = {
        "step": _PROFILE_MAX_SIGNED_INT,
        "head_sha256": _PROFILE_DIGEST,
        "measurement": _profile_measurement_payload(),
        "gradient_ratio_to_zero_initialization": _PROFILE_MAX_FINITE_FLOAT,
        "history_summary": _profile_history_summary(),
        "used_as_primary_selection_rule": False,
        "coincides_with_selected_primary_iterate": True,
    }
    fixed = {
        "schema_version": "fixed-step-compute-matched-snapshot/v1",
        **copy.deepcopy(common),
        "role": "compute_matched_and_pilot_diagnostic_only",
    }
    legacy = {
        "schema_version": "legacy-constant-lr-boundary-snapshot/v1",
        **copy.deepcopy(common),
        "learning_rate_used_for_update": _PROFILE_MAX_FINITE_FLOAT,
        "learning_rate_schedule_sha256": protocol.schedule_sha256,
        "role": "immutable_legacy_constant_lr_failure_boundary_diagnostic",
        "test_or_validation_data_accessed": False,
    }
    if set(fixed) != _FIXED_SNAPSHOT_FIELDS:
        raise RuntimeError("profile fixed snapshot drifted from production keys")
    if set(legacy) != _LEGACY_SNAPSHOT_FIELDS:
        raise RuntimeError("profile legacy snapshot drifted from production keys")
    return fixed, legacy


def _profile_recovery_transcript(
    *,
    update: int,
    head_shape: list[int],
    head_dtype: str,
    head_device: str,
) -> list[dict[str, object]]:
    moment = {
        "shape": head_shape,
        "dtype": head_dtype,
        "device": head_device,
        "layout": "torch.strided",
        "detached": True,
    }
    parameter_group = {
        "learning_rate": _PROFILE_MAX_FINITE_FLOAT,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "parameter_count": 1,
        "parameter_is_reward_head": True,
    }
    transcript: list[dict[str, object]] = []
    for step in range(1, update + 1):
        for phase, completed in (
            ("before_update", step - 1),
            ("after_update", step),
        ):
            transcript.append(
                {
                    "phase": phase,
                    "update": step,
                    "observation": {
                        "expected_completed_updates": completed,
                        "optimizer_state_empty": False,
                        "scalar_step": {
                            "value": completed,
                            "shape": [],
                            "dtype": "torch.float64",
                            "device": "cpu",
                        },
                        "exp_avg": copy.deepcopy(moment),
                        "exp_avg_sq": copy.deepcopy(moment),
                        "parameter_group": copy.deepcopy(parameter_group),
                    },
                }
            )
    return transcript


def _profile_optimizer_protocol_execution(
    *,
    update: int,
    protocol: _training.AdamWRecoveryProtocol,
    head_dtype: str,
) -> dict[str, object]:
    if len(protocol.stages) != 5 or protocol.stages[-1].last_update != R3_MAXIMUM_UPDATES_PER_HEAD:
        raise ValueError("R3 profile checkpoint envelope requires the locked five-stage schedule")
    transitions = [
        {
            "next_update": stage.first_update,
            "previous_learning_rate": _PROFILE_MAX_FINITE_FLOAT,
            "new_learning_rate": _PROFILE_MAX_FINITE_FLOAT,
            "moment_state_sha256_before_lr_assignment": _PROFILE_DIGEST,
            "moment_state_sha256_after_lr_assignment": _PROFILE_DIGEST,
            "same_optimizer_instance": True,
            "moments_preserved": True,
        }
        for stage in protocol.stages[1:]
    ]
    execution = {
        "schema_version": "deterministic-adamw-lr-decay-execution/v2",
        "protocol": protocol.to_dict(),
        "optimizer_class": "torch.optim.AdamW",
        "parameter_count": 1,
        "fresh_optimizer_state_before_first_update": True,
        "reward_head_dtype_observed": head_dtype,
        "first_order_audit_dtype_required": protocol.first_order_audit_dtype,
        "microbatch_order": protocol.microbatch_order,
        "one_optimizer_update_per_step": True,
        "learning_rate_set_immediately_before_every_update": True,
        "single_optimizer_instance_for_all_updates": True,
        "optimizer_state_reset_at_lr_milestone": False,
        "adamw_moments_preserved_at_learning_rate_boundaries": True,
        "boundary_transitions": transitions,
        "completed_updates_observed": update,
        "per_update_state_checks": {
            "schema_version": "recovery-adamw-per-update-state-checks/v1",
            "before_update_checks": update,
            "after_update_checks": update,
            "first_pre_update_state_empty": update > 0,
            "completed_updates_covered": update,
            "check_sequence_sha256": _PROFILE_DIGEST,
            "all_updates_checked_before_and_after": True,
            "all_subsequent_pre_update_scalar_steps_exact": True,
            "all_post_update_scalar_steps_exact": True,
            "exp_avg_and_exp_avg_sq_shape_dtype_device_valid": True,
        },
        "selected_primary_optimizer_state_restored_and_verified": True,
        "selected_optimizer_object_identity_preserved": True,
        "selected_optimizer_moments_restored_and_verified": True,
        "selected_head_sha256": _PROFILE_DIGEST,
        "restored_head_sha256": _PROFILE_DIGEST,
        "selected_optimizer_state_sha256": _PROFILE_DIGEST,
        "restored_optimizer_state_sha256": _PROFILE_DIGEST,
        "selected_checkpoint_optimizer_state_dict_sha256": _PROFILE_DIGEST,
        "restored_optimizer_state_dict_sha256": _PROFILE_DIGEST,
        "selected_checkpoint_sha256": _PROFILE_DIGEST,
        "test_or_validation_data_accessed": False,
    }
    if set(execution) != _OPTIMIZER_PROTOCOL_EXECUTION_FIELDS:
        raise RuntimeError("profile optimizer execution drifted from production checkpoint keys")
    if len(transitions) != 4:
        raise RuntimeError("profile optimizer execution omitted LR transitions")
    return execution


class _ProfileStateTrainer:
    def __init__(self, state: Mapping[str, object], *, completed_steps: int) -> None:
        self._state = state
        self.completed_steps = completed_steps

    def state_dict(self) -> Mapping[str, object]:
        return self._state


def _profile_benchmark_outer_payload(
    *,
    learner: str,
    update: int,
    trainer_state: Mapping[str, object],
    identity: Mapping[str, object],
    spec: _training.FirstOrderConvergenceSpec,
) -> dict[str, object]:
    model = trainer_state.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("profile trainer state has no model mapping")
    head = next(
        (value for value in model.values() if isinstance(value, torch.Tensor)),
        None,
    )
    if head is None:
        raise TypeError("profile trainer state has no tensor-bearing head")
    protocol = spec.optimizer_protocol
    if not isinstance(protocol, _training.AdamWRecoveryProtocol):
        raise TypeError("R3 profile checkpoint envelope requires recovery AdamW")
    identity_copy = _json_copy(identity, name="profile controller identity")
    measurement = _profile_measurement_payload()
    checks = [
        {
            "step": step,
            "post_update": True,
            "full_data": True,
            "gradient_clipping_applied": False,
            "measurement": copy.deepcopy(measurement),
            "gradient_ratio_to_zero_initialization": _PROFILE_MAX_FINITE_FLOAT,
            "eligible_after_min_steps": True,
            "threshold_passed": False,
            "consecutive_threshold_passes": _PROFILE_MAX_SIGNED_INT,
            "learning_rate_used_for_update": _PROFILE_MAX_FINITE_FLOAT,
            "learning_rate_schedule_sha256": protocol.schedule_sha256,
        }
        for step in range(
            R3_AUDIT_CADENCE_UPDATES,
            update + 1,
            R3_AUDIT_CADENCE_UPDATES,
        )
    ]
    if checks and any(set(check) != _RECOVERY_CONVERGENCE_CHECK_FIELDS for check in checks):
        raise RuntimeError("profile convergence checks drifted from production checkpoint keys")
    fixed_snapshot, legacy_snapshot = _profile_snapshots(protocol=protocol)
    optimizer_execution = _profile_optimizer_protocol_execution(
        update=update,
        protocol=protocol,
        head_dtype=str(head.dtype),
    )
    selected_state = copy.deepcopy(dict(trainer_state))
    controller = _training._build_first_order_controller_checkpoint(
        _ProfileStateTrainer(trainer_state, completed_steps=update),
        identity=identity_copy,
        initial=_profile_measurement(),
        checkpoint_boundary_measurement=_profile_measurement(),
        checks=checks,
        consecutive_passes=_PROFILE_MAX_SIGNED_INT,
        selected_state=selected_state,
        selected_measurement=_profile_measurement(),
        selected_step=update,
        selected_head_sha256=_PROFILE_DIGEST,
        selected_optimizer_state_sha256=_PROFILE_DIGEST,
        selected_checkpoint_optimizer_state_dict_sha256=_PROFILE_DIGEST,
        selected_checkpoint_sha256=_PROFILE_DIGEST,
        fixed_snapshot=fixed_snapshot,
        legacy_boundary_snapshot=legacy_snapshot,
        optimizer_protocol_execution=optimizer_execution,
        recovery_state_check_transcript=_profile_recovery_transcript(
            update=update,
            head_shape=list(head.shape),
            head_dtype=str(head.dtype),
            head_device=str(head.device),
        ),
    )
    if (
        set(controller) != _CONTROLLER_CHECKPOINT_FIELDS
        or controller["schema_version"] != PRODUCTION_CONTROLLER_CHECKPOINT_SCHEMA
    ):
        raise RuntimeError("production controller builder returned an unexpected schema")
    controller_state = controller["controller_state"]
    if (
        not isinstance(controller_state, Mapping)
        or set(controller_state) != _CONTROLLER_STATE_FIELDS
    ):
        raise RuntimeError("profile controller state drifted from production keys")
    payload = {
        "schema_version": PRODUCTION_PROFILE_BENCHMARK_PAYLOAD_SCHEMA,
        "design_sha256": _PROFILE_DIGEST,
        "admission_sha256": _PROFILE_DIGEST,
        "logical_run_id": _PROFILE_DIGEST,
        "head_run_id": _PROFILE_DIGEST,
        "scheduler_segment_id": _PROFILE_DIGEST,
        "segment_index": _PROFILE_MAX_SIGNED_INT,
        "task_id": _PROFILE_MAX_SIGNED_INT,
        "seed": _PROFILE_MAX_SIGNED_INT,
        "objective": learner,
        "runtime_sha256": _PROFILE_DIGEST,
        "head_execution_slice_sha256": _PROFILE_DIGEST,
        "controller_checkpoint_sha256": controller["checkpoint_sha256"],
        "controller_checkpoint": controller,
        "information_boundary": "train_only_profile_nonreusable",
    }
    if set(payload) != _PRIMARY_OUTER_PAYLOAD_FIELDS:
        raise RuntimeError("profile outer payload drifted from primary checkpoint keys")
    return payload


def _profile_terminal_convergence_evidence(
    *,
    learner: str,
    spec: _training.FirstOrderConvergenceSpec,
    identity: Mapping[str, object],
    controller_state: Mapping[str, object],
    selected_head_sha256: str,
    update: int,
) -> dict[str, object]:
    protocol = spec.optimizer_protocol
    if not isinstance(protocol, _training.AdamWRecoveryProtocol):
        raise TypeError("R3 terminal benchmark requires recovery AdamW")
    optimizer_execution = copy.deepcopy(controller_state["optimizer_protocol_execution"])
    if not isinstance(optimizer_execution, dict):
        raise TypeError("profile optimizer execution must be a mutable mapping")
    optimizer_execution["selected_primary_optimizer_state_restored_without_reconstruction"] = True
    measurement = _profile_measurement_payload()
    rank_diagnostic = identity.get("rank_diagnostic")
    if rank_diagnostic is not None and not isinstance(rank_diagnostic, Mapping):
        raise TypeError("profile controller rank diagnostic must be a mapping")
    return {
        "schema_version": "objective-first-order-convergence/v2",
        "objective": learner,
        "converged": True,
        "fail_closed": True,
        "spec": spec.to_dict(),
        "gradient_ratio_formula": (
            "||full_data_unclipped_gradient(w_t)||_2 / "
            "max(||full_data_unclipped_gradient(w_zero)||_2, denominator_floor)"
        ),
        "initial_zero_head_measurement": copy.deepcopy(measurement),
        "checks": copy.deepcopy(controller_state["checks"]),
        "selected_primary_step": update,
        "selected_primary_head_sha256": selected_head_sha256,
        "consecutive_threshold_passes_at_selection": spec.consecutive_checks,
        "final_gate": {
            "step": update,
            "measurement": copy.deepcopy(measurement),
            "gradient_ratio_to_zero_initialization": _PROFILE_MAX_FINITE_FLOAT,
            "threshold_passed": True,
            "fresh_post_restore_audit": True,
            "learning_rate_at_selected_iterate": _PROFILE_MAX_FINITE_FLOAT,
        },
        "fixed_step_compute_matched_snapshot": copy.deepcopy(controller_state["fixed_snapshot"]),
        "fixed_step_snapshot_steps": _PROFILE_MAX_SIGNED_INT,
        "fixed_step_snapshot_is_not_primary_selection": True,
        "solution_identification": _training._solution_identification_evidence(
            rank_diagnostic,
            tie_break=protocol.tie_break,
        ),
        "test_or_validation_data_accessed": False,
        "legacy_constant_lr_boundary_snapshot": copy.deepcopy(
            controller_state["legacy_boundary_snapshot"]
        ),
        "optimizer_protocol_execution": optimizer_execution,
    }


def _profile_selected_terminal_outer_payload(
    *,
    learner: str,
    selected_terminal_checkpoint: Mapping[str, object],
) -> dict[str, object]:
    if (
        selected_terminal_checkpoint.get("schema_version")
        != _training._SELECTED_PRIMARY_TERMINAL_CHECKPOINT_SCHEMA
    ):
        raise RuntimeError(
            "profile finalization benchmark did not use the production selected-terminal schema"
        )
    terminal_sha = selected_terminal_checkpoint.get("terminal_checkpoint_sha256")
    if not isinstance(terminal_sha, str):
        raise TypeError("profile selected terminal checkpoint lacks its hash")
    return {
        "schema_version": ("phase2-recovery-r3-profile-selected-terminal-benchmark/v1"),
        "design_sha256": _PROFILE_DIGEST,
        "admission_sha256": _PROFILE_DIGEST,
        "logical_run_id": _PROFILE_DIGEST,
        "head_run_id": _PROFILE_DIGEST,
        "scheduler_segment_id": _PROFILE_DIGEST,
        "segment_index": _PROFILE_MAX_SIGNED_INT,
        "task_id": _PROFILE_MAX_SIGNED_INT,
        "seed": _PROFILE_MAX_SIGNED_INT,
        "objective": learner,
        "runtime_sha256": _PROFILE_DIGEST,
        "head_execution_slice_sha256": _PROFILE_DIGEST,
        "selected_terminal_checkpoint_sha256": terminal_sha,
        "selected_terminal_checkpoint": dict(selected_terminal_checkpoint),
        "information_boundary": "train_only_profile_nonreusable",
    }


def _benchmark_production_checkpoint_envelope(
    *,
    learner: str,
    update: int,
    trainer_state: Mapping[str, object],
    identity: Mapping[str, object],
    spec: _training.FirstOrderConvergenceSpec,
    fresh_trainer_factory: Callable[[], object],
    directory: Path,
) -> dict[str, object]:
    build_start = time.perf_counter_ns()
    outer_payload = _profile_benchmark_outer_payload(
        learner=learner,
        update=update,
        trainer_state=trainer_state,
        identity=identity,
        spec=spec,
    )
    payload_digest = _training._checkpoint_value_sha256(outer_payload)
    build_seconds = (time.perf_counter_ns() - build_start) / 1.0e9
    objective = f"profile_io_{learner}_{update}"
    binding = {
        "schema_version": "phase2-profile-checkpoint-io-binding/v1",
        "role": "profile_nonreusable",
        "primary_resume_allowed": False,
        "payload_sha256": payload_digest,
    }
    with tempfile.TemporaryDirectory(
        prefix=f".profile-io-{learner}-{update}-",
        dir=directory,
    ) as temporary:
        store = DurableCheckpointStore(
            temporary,
            objective=objective,
            binding=binding,
        )
        save_start = time.perf_counter_ns()
        manifest = store.save(
            outer_payload,
            completed_steps=update,
            reason="manual",
        )
        save_seconds = (time.perf_counter_ns() - save_start) / 1.0e9
        generation = int(manifest["generation"])
        state_path = store.generations_path / f"generation-{generation:08d}" / "state.pt"
        serialized_bytes = state_path.stat().st_size

        reload_start = time.perf_counter_ns()
        loaded = store.load()
        if loaded is None:
            raise RuntimeError("production benchmark checkpoint did not reload")
        if _training._checkpoint_value_sha256(loaded) != payload_digest:
            raise RuntimeError("production benchmark checkpoint failed exact reload")
        audited = store.audit_generations(verify_all_checkpoint_bytes=True)
        if not audited:
            raise RuntimeError("production benchmark checkpoint audit is empty")
        reload_seconds = (time.perf_counter_ns() - reload_start) / 1.0e9

        progress_start = time.perf_counter_ns()
        store.record_progress(
            status="checkpointed",
            completed_steps=update,
            details={
                "profile_role": "production_io_benchmark_nonreusable",
                "primary_resume_allowed": False,
            },
        )
        progress_sha = store.latest_progress_sha256()
        progress_seconds = (time.perf_counter_ns() - progress_start) / 1.0e9

        boundary_start = time.perf_counter_ns()
        store.record_planned_boundary_receipt(
            head_name=objective,
            completed_steps=update,
            checkpoint_metadata_sha256=str(manifest["metadata_sha256"]),
            checkpoint_verified=True,
            last_progress_sha256=progress_sha,
            scheduler_identity={"profile_role": "production_io_benchmark_nonreusable"},
            execution_slice_sha256="e" * 64,
            update_blocks_consumed=math.ceil(update / R3_AUDIT_CADENCE_UPDATES),
            update_blocks_remaining=0,
            planned_action="continue_same_logical_run",
        )
        boundary_seconds = (time.perf_counter_ns() - boundary_start) / 1.0e9

        source_state_sha = _training._checkpoint_value_sha256(trainer_state)
        head_device = next(
            value.device
            for value in trainer_state["model"].values()  # type: ignore[union-attr]
            if isinstance(value, torch.Tensor)
        )
        if head_device.type == "cuda":
            torch.cuda.synchronize(head_device)
        restore_start = time.perf_counter_ns()
        restored_trainer = fresh_trainer_factory()
        load_state_dict = getattr(restored_trainer, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError("fresh profile trainer has no load_state_dict")
        load_state_dict(trainer_state)
        if head_device.type == "cuda":
            torch.cuda.synchronize(head_device)
        restore_seconds = (time.perf_counter_ns() - restore_start) / 1.0e9
        if _training._checkpoint_value_sha256(trainer_state) != source_state_sha:
            raise RuntimeError("fresh restore benchmark mutated its selected state")

        terminal_start = time.perf_counter_ns()
        terminal_builder_used = update > 0
        if terminal_builder_used:
            controller = outer_payload["controller_checkpoint"]
            if not isinstance(controller, Mapping):
                raise TypeError("profile outer payload lacks controller checkpoint")
            controller_state = controller["controller_state"]
            if not isinstance(controller_state, Mapping):
                raise TypeError("profile controller state must be a mapping")
            restored_state = restored_trainer.state_dict()
            restored_head = next(
                value
                for value in restored_state["model"].values()
                if isinstance(value, torch.Tensor)
            )
            selected_head_sha = _training._tensor_sha256(restored_head)
            convergence_evidence = _profile_terminal_convergence_evidence(
                learner=learner,
                spec=spec,
                identity=identity,
                controller_state=controller_state,
                selected_head_sha256=selected_head_sha,
                update=update,
            )
            selected_terminal = _training._build_selected_primary_terminal_checkpoint(
                restored_trainer,
                identity=identity,
                selected_primary_step=update,
                controller_updates_executed=update,
                initial=_profile_measurement(),
                final=_profile_measurement(),
                evidence=convergence_evidence,
            )
            terminal = _profile_selected_terminal_outer_payload(
                learner=learner,
                selected_terminal_checkpoint=selected_terminal,
            )
        else:
            terminal = {
                "schema_version": ("phase2-recovery-r3-profile-selected-terminal-benchmark/v1"),
                "profile_role": "nonreusable_zero_update_diagnostic",
                "payload_sha256": payload_digest,
            }
        _training._checkpoint_value_sha256(terminal)
        if head_device.type == "cuda":
            torch.cuda.synchronize(head_device)
        terminal_seconds = (time.perf_counter_ns() - terminal_start) / 1.0e9
        finalization_seconds = restore_seconds + terminal_seconds
        del restored_trainer
    return {
        "update": update,
        "serialized_bytes": serialized_bytes,
        # Each component is bounded by the measured complete atomic save path.
        "serialize_wall_seconds": save_seconds,
        "fsync_wall_seconds": save_seconds,
        "reload_and_verify_wall_seconds": reload_seconds,
        "progress_receipt_publish_fsync_verify_wall_seconds": (progress_seconds),
        "signal_or_planned_boundary_receipt_publish_fsync_verify_wall_seconds": (boundary_seconds),
        "final_restore_load_terminal_payload_wall_seconds": max(
            build_seconds,
            finalization_seconds,
        ),
        "production_outer_payload_isomorphic_profile_schema": True,
        "durable_outer_envelope_exact_schema": True,
        "profile_role": True,
        "reusable_as_primary_state": False,
        "same_pass_live_trainer_state": True,
        "full_live_history_records": update,
        "worst_case_selected_state_copy_included": True,
        "selected_state_history_records": update,
        "convergence_check_records": update // R3_AUDIT_CADENCE_UPDATES,
        "recovery_state_check_records": 2 * update,
        "fixed_tensor_components_included": True,
        "fixed_snapshot_included": True,
        "legacy_boundary_snapshot_included": True,
        "optimizer_protocol_stage_records": len(spec.optimizer_protocol.stages),
        "optimizer_protocol_transition_records": (len(spec.optimizer_protocol.stages) - 1),
        "primary_outer_exact_key_contract_verified": (
            set(outer_payload) == _PRIMARY_OUTER_PAYLOAD_FIELDS
        ),
        "production_controller_builder_used": True,
        "convergence_check_exact_key_contract_verified": True,
        "snapshot_exact_key_contract_verified": True,
        "optimizer_protocol_execution_exact_key_contract_verified": True,
        "selected_terminal_production_builder_used": terminal_builder_used,
        "fresh_trainer_restore_load_measured": True,
        "atomic_publication_fsync_included": True,
        "committed_byte_verification_included": True,
    }


def _run_core_with_production_checkpoint_io(
    profile_run: ValidatedGatePRun,
    *,
    io_probe_directory: Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Capture live profile boundaries through the real durable-store path."""

    samples_by_learner: dict[str, list[dict[str, object]]] = {
        learner: [] for learner in PHASE2_PROFILE_LEARNER_ORDER
    }
    context = profile_run.materialization.context
    spec = context.settings.convergence

    def live_probe(
        learner: str,
        update: int,
        trainer_state: Mapping[str, object],
        directory: Path,
    ) -> None:
        rank_diagnostic = (
            context.reward_head_identifiability
            if learner == "bt_mle"
            else context.prorm_moment_map_identifiability
        )
        identity = _training._first_order_controller_identity(
            objective_name=learner,
            execution_role="phase2_recovery_r3_primary",
            spec=spec,
            fixed_snapshot_steps=context.settings.outer_steps,
            rank_diagnostic=rank_diagnostic,
        )

        def fresh_trainer_factory() -> object:
            return build_primary_core_trainer(context, learner)

        samples_by_learner[learner].append(
            _benchmark_production_checkpoint_envelope(
                learner=learner,
                update=update,
                trainer_state=trainer_state,
                identity=identity,
                spec=spec,
                fresh_trainer_factory=fresh_trainer_factory,
                directory=directory,
            )
        )
        if update == PHASE2_PROFILE_UPDATES:
            samples_by_learner[learner].append(
                _benchmark_production_checkpoint_envelope(
                    learner=learner,
                    update=R3_MAXIMUM_UPDATES_PER_HEAD,
                    trainer_state=_expanded_profile_trainer_state(
                        trainer_state,
                        target_update=R3_MAXIMUM_UPDATES_PER_HEAD,
                    ),
                    identity=identity,
                    spec=spec,
                    fresh_trainer_factory=fresh_trainer_factory,
                    directory=directory,
                )
            )

    core = run_gate_p_profile_core(
        profile_run.materialization.context,
        io_probe_directory=io_probe_directory,
        live_boundary_probe=live_probe,
    )
    core_sha = core["profile_sha256"]
    learners: list[dict[str, object]] = []
    diagnostic_updates = list(PHASE2_PROFILE_AUDIT_UPDATES)
    for learner_name in PHASE2_PROFILE_LEARNER_ORDER:
        samples = samples_by_learner[learner_name]
        expected_sample_updates = [
            *diagnostic_updates,
            R3_MAXIMUM_UPDATES_PER_HEAD,
        ]
        if [sample["update"] for sample in samples] != expected_sample_updates:
            raise RuntimeError("same-pass production checkpoint samples are incomplete")
        diagnostic_sizes = [int(sample["serialized_bytes"]) for sample in samples[:-1]]
        slope = max(
            1,
            max(
                math.ceil(
                    (diagnostic_sizes[right] - diagnostic_sizes[left])
                    / (diagnostic_updates[right] - diagnostic_updates[left])
                )
                for left in range(len(diagnostic_sizes))
                for right in range(left + 1, len(diagnostic_sizes))
            ),
        )
        fixed = max(
            size - slope * update
            for update, size in zip(
                diagnostic_updates,
                diagnostic_sizes,
                strict=True,
            )
        )
        learners.append(
            {
                "learner": learner_name,
                "samples": samples,
                "fixed_serialized_bytes_upper_bound": fixed,
                "per_update_serialized_byte_slope_upper_bound": slope,
                "target_serialized_bytes_upper_bound": int(samples[-1]["serialized_bytes"]),
                "minimum_serialize_throughput_bytes_per_second": min(
                    int(sample["serialized_bytes"]) / float(sample["serialize_wall_seconds"])
                    for sample in samples
                ),
                "minimum_fsync_throughput_bytes_per_second": min(
                    int(sample["serialized_bytes"]) / float(sample["fsync_wall_seconds"])
                    for sample in samples
                ),
                "minimum_reload_verify_throughput_bytes_per_second": min(
                    int(sample["serialized_bytes"])
                    / float(sample["reload_and_verify_wall_seconds"])
                    for sample in samples
                ),
                "maximum_progress_receipt_wall_seconds": max(
                    float(sample["progress_receipt_publish_fsync_verify_wall_seconds"])
                    for sample in samples
                ),
                "maximum_boundary_receipt_wall_seconds": max(
                    float(
                        sample[
                            "signal_or_planned_boundary_receipt_publish_fsync_verify_wall_seconds"
                        ]
                    )
                    for sample in samples
                ),
                "maximum_finalization_noncheckpoint_wall_seconds": max(
                    float(sample["final_restore_load_terminal_payload_wall_seconds"])
                    for sample in samples
                ),
            }
        )
    payload: dict[str, object] = {
        "schema_version": PRODUCTION_CHECKPOINT_IO_PROFILE_SCHEMA,
        "capture_mode": ("same_pass_live_production_outer_envelope_worst_case_selected_copy"),
        "core_profile_sha256": core_sha,
        "production_outer_payload_schema": (PRODUCTION_OUTER_CHECKPOINT_PAYLOAD_SCHEMA),
        "benchmark_payload_schema": PRODUCTION_PROFILE_BENCHMARK_PAYLOAD_SCHEMA,
        "durable_checkpoint_envelope_schema": (PRODUCTION_DURABLE_CHECKPOINT_ENVELOPE_SCHEMA),
        "controller_checkpoint_schema": PRODUCTION_CONTROLLER_CHECKPOINT_SCHEMA,
        "sample_updates": [
            *diagnostic_updates,
            R3_MAXIMUM_UPDATES_PER_HEAD,
        ],
        "schema_growth_model": {
            "formula": (
                "diagnostic_bytes(step<=100)=fixed+slope*step;"
                "projection_bound(step>100)=measured_target_12760_envelope"
            ),
            "live_trainer_history_records_per_update": 1,
            "selected_state_history_records_per_update": 1,
            "convergence_check_records_per_audit_block": 1,
            "recovery_state_check_records_per_update": 2,
            "only_declared_linear_fields_extrapolated": True,
            "fixed_tensor_components_measured_at_every_sample": True,
            "worst_case_selected_state_copy_measured_at_every_sample": True,
            "nonlinear_or_unbounded_fields_absent": True,
            "simple_fixed_io_times_event_count_forbidden": True,
            "target_12760_envelope_actually_serialized": True,
            "projection_uses_measured_target_not_linear_extrapolation": True,
        },
        "learners": learners,
        "information_boundary": dict(FORMAL_PROFILE_INFORMATION_BOUNDARY),
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)
    return core, payload


def _identity_bindings(profile_run: ValidatedGatePRun) -> dict[str, object]:
    materialization = profile_run.materialization
    provenance = materialization.provenance
    return {
        "profile_run_sha256": profile_run.profile_run_sha256,
        "gate_p_admission_sha256": profile_run.admission.admission_sha256,
        "gate0_artifact_sha256": profile_run.admission.gate0.artifact_sha256,
        "gate1_artifact_sha256": profile_run.admission.gate1.artifact_sha256,
        "source_artifact_sha256": profile_run.admission.source.artifact_sha256,
        "container_artifact_sha256": profile_run.admission.container.artifact_sha256,
        "config_artifact_sha256": profile_run.admission.config.artifact_sha256,
        "materialization_attestation_sha256": materialization.attestation_sha256,
        "materialization_provenance_sha256": provenance.provenance_sha256,
        "context_sha256": materialization.context.context_sha256,
        "settings_sha256": materialization.settings_sha256,
        "input_training_sha256": materialization.input_training_sha256,
        "prepared_training_sha256": materialization.prepared_training_sha256,
        "oracle_reward_sha256": materialization.oracle_reward_sha256,
        "label_stream_sha256": materialization.label_stream_sha256,
    }


def _formal_profile_payload(
    *,
    profile_run: ValidatedGatePRun,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    preparation: ProfilePreparationTimings,
    core_profile: Mapping[str, object],
    production_checkpoint_io_evidence: Mapping[str, object],
    materialization_revalidation_wall_seconds: float,
    wrapper_wall_seconds: float,
    cuda_identity: Mapping[str, object],
    gpu_utilization_samples: Sequence[Mapping[str, object]],
    cpu_memory: Mapping[str, object],
) -> dict[str, object]:
    learners = core_profile["learners"]
    trainer_enter = {
        learner["learner"]: learner["build_wall_seconds"]  # type: ignore[index]
        for learner in learners  # type: ignore[union-attr]
    }
    return {
        "schema_version": FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
        "role": FORMAL_CUDA_PROFILE_RESULT_ROLE,
        "identity_bindings": _identity_bindings(profile_run),
        "safety_margin_policy_sha256": safety_policy.policy_sha256,
        "scheduler_resource_envelope": envelope.to_dict(),
        "scheduler_raw_evidence_sha256": envelope.scheduler_raw_evidence_sha256,
        "resource_raw_evidence_sha256": envelope.resource_raw_evidence_sha256,
        "preparation": preparation.to_dict(),
        "materialization_revalidation_wall_seconds": (materialization_revalidation_wall_seconds),
        "trainer_enter_wall_seconds": trainer_enter,
        "wrapper_wall_seconds": wrapper_wall_seconds,
        "cuda_identity": dict(cuda_identity),
        "gpu_utilization_samples": [dict(sample) for sample in gpu_utilization_samples],
        "cpu_memory": dict(cpu_memory),
        "core_profile": dict(core_profile),
        "core_profile_sha256": core_profile["profile_sha256"],
        "production_checkpoint_io_evidence": dict(production_checkpoint_io_evidence),
        "stop_reason": PHASE2_PROFILE_STOP_REASON,
        "information_boundary": dict(FORMAL_PROFILE_INFORMATION_BOUNDARY),
    }


@dataclass(frozen=True, slots=True)
class FormalCudaProfileResult:
    """Self-hashed formal CUDA measurements with no consumable train state."""

    profile_run: ValidatedGatePRun = field(repr=False, compare=False)
    safety_policy: ProfileSafetyMarginPolicy = field(repr=False, compare=False)
    envelope: SchedulerResourceEnvelope = field(repr=False, compare=False)
    preparation: ProfilePreparationTimings = field(repr=False, compare=False)
    schema_version: str
    role: str
    identity_bindings: Mapping[str, object]
    safety_margin_policy_sha256: str
    scheduler_resource_envelope: Mapping[str, object]
    scheduler_raw_evidence_sha256: str
    resource_raw_evidence_sha256: str
    preparation_evidence: Mapping[str, object]
    materialization_revalidation_wall_seconds: float
    trainer_enter_wall_seconds: Mapping[str, object]
    wrapper_wall_seconds: float
    cuda_identity: Mapping[str, object]
    gpu_utilization_samples: tuple[Mapping[str, object], Mapping[str, object]]
    cpu_memory: Mapping[str, object]
    core_profile: Mapping[str, object]
    core_profile_sha256: str
    production_checkpoint_io_evidence: Mapping[str, object]
    stop_reason: str
    information_boundary: Mapping[str, object]
    formal_profile_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        run = _validated_run(self.profile_run)
        if type(self.safety_policy) is not ProfileSafetyMarginPolicy:
            raise TypeError("safety_policy must be ProfileSafetyMarginPolicy")
        if type(self.envelope) is not SchedulerResourceEnvelope:
            raise TypeError("envelope must be SchedulerResourceEnvelope")
        if type(self.preparation) is not ProfilePreparationTimings:
            raise TypeError("preparation must be ProfilePreparationTimings")
        self.safety_policy.validate_integrity()
        self.envelope.validate_integrity()
        self.preparation.validate_integrity()
        for bound in (self.safety_policy, self.envelope, self.preparation):
            if bound.profile_run_sha256 != run.profile_run_sha256:
                raise ValueError("formal profile dependency belongs to another Gate-P run")
        core = _strict_core_profile(self.core_profile, profile_run=run)
        production_io = _validate_production_checkpoint_io_evidence(
            self.production_checkpoint_io_evidence,
            core_profile=core,
        )
        cuda_identity = _validate_cuda_identity(
            self.cuda_identity,
            envelope=self.envelope,
        )
        samples = _validate_gpu_samples(
            self.gpu_utilization_samples,
            cuda_identity=cuda_identity,
        )
        cpu_memory = _validate_cpu_memory(self.cpu_memory)
        revalidation_seconds = _positive_real(
            self.materialization_revalidation_wall_seconds,
            name="materialization_revalidation_wall_seconds",
        )
        wrapper_seconds = _positive_real(
            self.wrapper_wall_seconds,
            name="wrapper_wall_seconds",
        )
        minimum_wrapper_seconds = math.fsum(
            (
                revalidation_seconds,
                float(core["setup"]["wall_seconds"]),  # type: ignore[index]
                *(
                    float(learner["phase_wall_seconds"])
                    for learner in core["learners"]  # type: ignore[union-attr]
                ),
            )
        )
        if wrapper_seconds < minimum_wrapper_seconds:
            raise ValueError("wrapper time is shorter than its sequential measured components")
        expected = _formal_profile_payload(
            profile_run=run,
            safety_policy=self.safety_policy,
            envelope=self.envelope,
            preparation=self.preparation,
            core_profile=core,
            production_checkpoint_io_evidence=production_io,
            materialization_revalidation_wall_seconds=revalidation_seconds,
            wrapper_wall_seconds=wrapper_seconds,
            cuda_identity=cuda_identity,
            gpu_utilization_samples=samples,
            cpu_memory=cpu_memory,
        )
        observed = {
            "schema_version": self.schema_version,
            "role": self.role,
            "identity_bindings": self.identity_bindings,
            "safety_margin_policy_sha256": self.safety_margin_policy_sha256,
            "scheduler_resource_envelope": self.scheduler_resource_envelope,
            "scheduler_raw_evidence_sha256": self.scheduler_raw_evidence_sha256,
            "resource_raw_evidence_sha256": self.resource_raw_evidence_sha256,
            "preparation": self.preparation_evidence,
            "materialization_revalidation_wall_seconds": (
                self.materialization_revalidation_wall_seconds
            ),
            "trainer_enter_wall_seconds": self.trainer_enter_wall_seconds,
            "wrapper_wall_seconds": self.wrapper_wall_seconds,
            "cuda_identity": self.cuda_identity,
            "gpu_utilization_samples": list(self.gpu_utilization_samples),
            "cpu_memory": self.cpu_memory,
            "core_profile": self.core_profile,
            "core_profile_sha256": self.core_profile_sha256,
            "production_checkpoint_io_evidence": (self.production_checkpoint_io_evidence),
            "stop_reason": self.stop_reason,
            "information_boundary": self.information_boundary,
        }
        if _json_copy(observed, name="formal profile fields") != expected:
            raise ValueError("formal CUDA profile fields differ from their live bindings")
        _assert_no_sensitive_state(expected)
        _digest(self.formal_profile_sha256, name="formal_profile_sha256")
        if _canonical_sha256(expected) != self.formal_profile_sha256:
            raise ValueError("formal CUDA profile SHA256 does not match its contents")

    def validate_integrity(self) -> None:
        _require_factory(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _formal_profile_payload(
            profile_run=self.profile_run,
            safety_policy=self.safety_policy,
            envelope=self.envelope,
            preparation=self.preparation,
            core_profile=self.core_profile,
            production_checkpoint_io_evidence=(self.production_checkpoint_io_evidence),
            materialization_revalidation_wall_seconds=(
                self.materialization_revalidation_wall_seconds
            ),
            wrapper_wall_seconds=self.wrapper_wall_seconds,
            cuda_identity=self.cuda_identity,
            gpu_utilization_samples=self.gpu_utilization_samples,
            cpu_memory=self.cpu_memory,
        )
        return {**payload, "formal_profile_sha256": self.formal_profile_sha256}


def validate_formal_cuda_profile_result(value: object) -> FormalCudaProfileResult:
    if type(value) is not FormalCudaProfileResult:
        raise TypeError("formal profile must be exactly FormalCudaProfileResult")
    value.validate_integrity()
    return value


def _validate_formal_dependencies(
    profile_run: ValidatedGatePRun,
    *,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    preparation: ProfilePreparationTimings,
) -> tuple[
    ValidatedGatePRun,
    ProfileSafetyMarginPolicy,
    SchedulerResourceEnvelope,
    ProfilePreparationTimings,
]:
    run = _validated_run(profile_run)
    if type(safety_policy) is not ProfileSafetyMarginPolicy:
        raise TypeError("safety_policy must be ProfileSafetyMarginPolicy")
    if type(envelope) is not SchedulerResourceEnvelope:
        raise TypeError("envelope must be SchedulerResourceEnvelope")
    if type(preparation) is not ProfilePreparationTimings:
        raise TypeError("preparation must be ProfilePreparationTimings")
    safety_policy.validate_integrity()
    envelope.validate_integrity()
    preparation.validate_integrity()
    for bound in (safety_policy, envelope, preparation):
        if bound.profile_run_sha256 != run.profile_run_sha256:
            raise ValueError("formal profile dependency belongs to another Gate-P run")
    return run, safety_policy, envelope, preparation


def run_formal_gate_p_cuda_profile(
    profile_run: ValidatedGatePRun,
    *,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    preparation: ProfilePreparationTimings,
    io_probe_directory: str | os.PathLike[str],
) -> FormalCudaProfileResult:
    """Execute and promote the fixed profile only on a live CUDA allocation."""

    run, safety, resources, prep = _validate_formal_dependencies(
        profile_run,
        safety_policy=safety_policy,
        envelope=envelope,
        preparation=preparation,
    )
    directory = Path(os.fspath(io_probe_directory)).resolve()
    if not directory.is_dir():
        raise ValueError("formal profile I/O probe directory must exist")
    wrapper_start_ns = time.perf_counter_ns()
    revalidation_start_ns = time.perf_counter_ns()
    run.materialization.validate_integrity()
    revalidation_end_ns = time.perf_counter_ns()
    revalidation_seconds = (revalidation_end_ns - revalidation_start_ns) / 1e9
    _positive_real(
        revalidation_seconds,
        name="materialization_revalidation_wall_seconds",
    )

    cuda_identity = _require_live_cuda(run)
    _validate_cuda_identity(cuda_identity, envelope=resources)
    memory_before = _read_process_memory()
    sample_before = _sample_gpu_utilization(cuda_identity, sample_index=0)
    core, production_io = _run_core_with_production_checkpoint_io(
        run,
        io_probe_directory=directory,
    )
    sample_after = _sample_gpu_utilization(cuda_identity, sample_index=1)
    memory_after = _read_process_memory()
    core = _strict_core_profile(core, profile_run=run)
    production_io = _validate_production_checkpoint_io_evidence(
        production_io,
        core_profile=core,
    )
    cpu_memory = {
        "current_rss_bytes": memory_after["current_rss_bytes"],
        "peak_rss_bytes": max(
            int(memory_before["peak_rss_bytes"]),
            int(memory_after["peak_rss_bytes"]),
        ),
        "measurement": "linux_proc_status_vmrss_vmhwm",
    }
    wrapper_end_ns = time.perf_counter_ns()
    wrapper_seconds = (wrapper_end_ns - wrapper_start_ns) / 1e9
    payload = _formal_profile_payload(
        profile_run=run,
        safety_policy=safety,
        envelope=resources,
        preparation=prep,
        core_profile=core,
        production_checkpoint_io_evidence=production_io,
        materialization_revalidation_wall_seconds=revalidation_seconds,
        wrapper_wall_seconds=wrapper_seconds,
        cuda_identity=cuda_identity,
        gpu_utilization_samples=(sample_before, sample_after),
        cpu_memory=cpu_memory,
    )
    result = FormalCudaProfileResult(
        profile_run=run,
        safety_policy=safety,
        envelope=resources,
        preparation=prep,
        schema_version=str(payload["schema_version"]),
        role=str(payload["role"]),
        identity_bindings=payload["identity_bindings"],  # type: ignore[arg-type]
        safety_margin_policy_sha256=str(payload["safety_margin_policy_sha256"]),
        scheduler_resource_envelope=payload[  # type: ignore[arg-type]
            "scheduler_resource_envelope"
        ],
        scheduler_raw_evidence_sha256=str(payload["scheduler_raw_evidence_sha256"]),
        resource_raw_evidence_sha256=str(payload["resource_raw_evidence_sha256"]),
        preparation_evidence=payload["preparation"],  # type: ignore[arg-type]
        materialization_revalidation_wall_seconds=revalidation_seconds,
        trainer_enter_wall_seconds=payload[  # type: ignore[arg-type]
            "trainer_enter_wall_seconds"
        ],
        wrapper_wall_seconds=wrapper_seconds,
        cuda_identity=payload["cuda_identity"],  # type: ignore[arg-type]
        gpu_utilization_samples=(sample_before, sample_after),
        cpu_memory=cpu_memory,
        core_profile=core,
        core_profile_sha256=str(payload["core_profile_sha256"]),
        production_checkpoint_io_evidence=production_io,
        stop_reason=str(payload["stop_reason"]),
        information_boundary=payload["information_boundary"],  # type: ignore[arg-type]
        formal_profile_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def formal_cuda_profile_artifact_ref(
    result: FormalCudaProfileResult,
) -> ArtifactRef:
    validated = validate_formal_cuda_profile_result(result)
    return ArtifactRef(
        schema_version=FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
        artifact_sha256=validated.formal_profile_sha256,
        role=FORMAL_CUDA_PROFILE_RESULT_ROLE,
    )


def _base_checkpoint_updates(
    policy: ProfileSafetyMarginPolicy,
) -> tuple[int, ...]:
    scheduled = set(
        range(
            policy.durable_checkpoint_cadence_updates,
            R3_MAXIMUM_UPDATES_PER_HEAD + 1,
            policy.durable_checkpoint_cadence_updates,
        )
    )
    scheduled.update(R3_MANDATORY_CHECKPOINT_UPDATES)
    return tuple(sorted(scheduled))


def _checkpoint_bytes_at_step(io_model: Mapping[str, object], step: int) -> int:
    _exact_int(step, name="checkpoint step", minimum=0)
    if step > R3_MAXIMUM_UPDATES_PER_HEAD:
        raise ValueError("checkpoint step exceeds the fail-closed maximum")
    samples = io_model.get("samples", io_model.get("production_checkpoint_io_samples"))
    if not isinstance(samples, list):
        raise TypeError("checkpoint I/O model has no measured samples")
    for sample in samples:
        if int(sample["update"]) >= step:
            return int(sample["serialized_bytes"])
    raise ValueError("checkpoint I/O model lacks the measured target envelope")


def _checkpoint_wall_seconds_at_step(
    io_model: Mapping[str, object],
    step: int,
) -> float:
    samples = io_model.get("samples", io_model.get("production_checkpoint_io_samples"))
    if not isinstance(samples, list):
        raise TypeError("checkpoint I/O model has no measured samples")
    for sample in samples:
        if int(sample["update"]) >= step:
            return math.fsum(
                float(sample[name])
                for name in (
                    "serialize_wall_seconds",
                    "fsync_wall_seconds",
                    "reload_and_verify_wall_seconds",
                )
            )
    raise ValueError("checkpoint I/O model lacks the measured target envelope")


def _checkpoint_and_progress_wall_seconds_at_step(
    io_model: Mapping[str, object],
    step: int,
) -> float:
    return math.fsum(
        (
            _checkpoint_wall_seconds_at_step(io_model, step),
            float(io_model["maximum_progress_receipt_wall_seconds"]),
        )
    )


def _safe_boundary_evidence_chain_wall_seconds_at_step(
    io_model: Mapping[str, object],
    step: int,
) -> float:
    return math.fsum(
        (
            _checkpoint_and_progress_wall_seconds_at_step(io_model, step),
            float(io_model["maximum_boundary_receipt_wall_seconds"]),
        )
    )


def _learner_projection(
    learner: Mapping[str, object],
    *,
    production_io: Mapping[str, object],
    policy: ProfileSafetyMarginPolicy,
    learner_index: int,
    learner_count: int,
) -> dict[str, object]:
    steps = learner["steps"]
    audits = learner["audits"]
    probes = learner["ephemeral_checkpoint_io"]
    max_update = max(float(step["wall_seconds"]) for step in steps)  # type: ignore[index]
    max_audit = max(float(audit["wall_seconds"]) for audit in audits)  # type: ignore[index]
    samples = production_io["samples"]
    maximum_measured_production_io = max(
        math.fsum(
            float(sample[field_name])
            for field_name in (
                "serialize_wall_seconds",
                "fsync_wall_seconds",
                "reload_and_verify_wall_seconds",
            )
        )
        for sample in samples  # type: ignore[union-attr]
    )
    build = float(learner["build_wall_seconds"])
    projected_updates = max_update * R3_MAXIMUM_UPDATES_PER_HEAD
    projected_audits = max_audit * R3_TARGET_AUDIT_COUNT_PER_HEAD
    checkpoint_updates = _base_checkpoint_updates(policy)
    selection_checkpoint_events = int(policy.checkpoint_on_selection)
    head_transition_checkpoint_events = int(
        policy.checkpoint_before_head_transition and learner_index < learner_count - 1
    )
    dynamic_checkpoint_events = selection_checkpoint_events + head_transition_checkpoint_events
    checkpoint_events = [
        {
            "update": update,
            "reason": "cadence_or_learning_rate_boundary",
            "serialized_bytes_upper_bound": _checkpoint_bytes_at_step(
                production_io,
                update,
            ),
            "projected_roundtrip_wall_seconds": (
                _checkpoint_and_progress_wall_seconds_at_step(
                    production_io,
                    update,
                )
            ),
        }
        for update in checkpoint_updates
    ]
    if selection_checkpoint_events:
        checkpoint_events.append(
            {
                "update": R3_MAXIMUM_UPDATES_PER_HEAD,
                "reason": "worst_case_post_selection",
                "serialized_bytes_upper_bound": _checkpoint_bytes_at_step(
                    production_io,
                    R3_MAXIMUM_UPDATES_PER_HEAD,
                ),
                "projected_roundtrip_wall_seconds": (
                    _checkpoint_and_progress_wall_seconds_at_step(
                        production_io,
                        R3_MAXIMUM_UPDATES_PER_HEAD,
                    )
                ),
            }
        )
    if head_transition_checkpoint_events:
        checkpoint_events.append(
            {
                "update": R3_MAXIMUM_UPDATES_PER_HEAD,
                "reason": "pre_head_transition",
                "serialized_bytes_upper_bound": _checkpoint_bytes_at_step(
                    production_io,
                    R3_MAXIMUM_UPDATES_PER_HEAD,
                ),
                "projected_roundtrip_wall_seconds": (
                    _checkpoint_and_progress_wall_seconds_at_step(
                        production_io,
                        R3_MAXIMUM_UPDATES_PER_HEAD,
                    )
                ),
            }
        )
    target_checkpoint_events = len(checkpoint_events)
    projected_io = math.fsum(
        float(event["projected_roundtrip_wall_seconds"]) for event in checkpoint_events
    )
    projected_total = math.fsum((build, projected_updates, projected_audits, projected_io))
    return {
        "learner": learner["learner"],
        "measured_updates": PHASE2_PROFILE_UPDATES,
        "target_updates": R3_MAXIMUM_UPDATES_PER_HEAD,
        "maximum_measured_update_wall_seconds": max_update,
        "projected_update_wall_seconds": projected_updates,
        "measured_audits": len(audits),  # type: ignore[arg-type]
        "target_audits": R3_TARGET_AUDIT_COUNT_PER_HEAD,
        "maximum_measured_audit_wall_seconds": max_audit,
        "projected_audit_wall_seconds": projected_audits,
        "measured_trainer_state_checkpoint_probes": len(probes),  # type: ignore[arg-type]
        "measured_production_outer_checkpoint_probes": len(samples),  # type: ignore[arg-type]
        "production_checkpoint_io_samples": list(samples),  # type: ignore[arg-type]
        "durable_checkpoint_cadence_updates": (policy.durable_checkpoint_cadence_updates),
        "mandatory_checkpoint_updates": list(R3_MANDATORY_CHECKPOINT_UPDATES),
        "mandatory_checkpoint_roles": list(R3_MANDATORY_CHECKPOINT_ROLES),
        "base_checkpoint_updates": list(checkpoint_updates),
        "base_target_checkpoint_events": len(checkpoint_updates),
        "selection_checkpoint_events": selection_checkpoint_events,
        "head_transition_checkpoint_events": head_transition_checkpoint_events,
        "dynamic_pre_segmentation_checkpoint_events": (dynamic_checkpoint_events),
        "target_checkpoint_events_before_segmentation": (target_checkpoint_events),
        "fixed_serialized_bytes_upper_bound": int(
            production_io["fixed_serialized_bytes_upper_bound"]
        ),
        "per_update_serialized_byte_slope_upper_bound": int(
            production_io["per_update_serialized_byte_slope_upper_bound"]
        ),
        "target_serialized_bytes_upper_bound": int(
            production_io["target_serialized_bytes_upper_bound"]
        ),
        "minimum_serialize_throughput_bytes_per_second": float(
            production_io["minimum_serialize_throughput_bytes_per_second"]
        ),
        "minimum_fsync_throughput_bytes_per_second": float(
            production_io["minimum_fsync_throughput_bytes_per_second"]
        ),
        "minimum_reload_verify_throughput_bytes_per_second": float(
            production_io["minimum_reload_verify_throughput_bytes_per_second"]
        ),
        "maximum_progress_receipt_wall_seconds": float(
            production_io["maximum_progress_receipt_wall_seconds"]
        ),
        "maximum_boundary_receipt_wall_seconds": float(
            production_io["maximum_boundary_receipt_wall_seconds"]
        ),
        "maximum_finalization_noncheckpoint_wall_seconds": float(
            production_io["maximum_finalization_noncheckpoint_wall_seconds"]
        ),
        "maximum_measured_production_checkpoint_roundtrip_wall_seconds": (
            maximum_measured_production_io
        ),
        "maximum_projected_checkpoint_roundtrip_wall_seconds": (
            _checkpoint_wall_seconds_at_step(
                production_io,
                R3_MAXIMUM_UPDATES_PER_HEAD,
            )
        ),
        "maximum_projected_checkpoint_and_progress_wall_seconds": (
            _checkpoint_and_progress_wall_seconds_at_step(
                production_io,
                R3_MAXIMUM_UPDATES_PER_HEAD,
            )
        ),
        "maximum_projected_safe_boundary_evidence_chain_wall_seconds": (
            _safe_boundary_evidence_chain_wall_seconds_at_step(
                production_io,
                R3_MAXIMUM_UPDATES_PER_HEAD,
            )
        ),
        "checkpoint_events_before_segmentation": checkpoint_events,
        "projected_checkpoint_wall_seconds": projected_io,
        "trainer_enter_wall_seconds": build,
        "projected_learner_wall_seconds": projected_total,
        "projection_rule": (
            "sum_each_event_using_next_measured_schema_isomorphic_envelope;"
            "all_steps_above_100_use_actual_measured_12760_target_envelope"
        ),
    }


def _projection(
    result: FormalCudaProfileResult,
    policy: ProfileSafetyMarginPolicy,
) -> dict[str, object]:
    core = result.core_profile
    learners = core["learners"]
    production_io_learners = result.production_checkpoint_io_evidence["learners"]
    if not isinstance(production_io_learners, list):
        raise TypeError("production checkpoint I/O learners must be a list")
    learner_count = len(learners)  # type: ignore[arg-type]
    projected_learners = [
        _learner_projection(
            learner,
            production_io=production_io,
            policy=policy,
            learner_index=learner_index,
            learner_count=learner_count,
        )
        for learner_index, (learner, production_io) in enumerate(
            zip(
                learners,  # type: ignore[arg-type]
                production_io_learners,
                strict=True,
            )
        )
    ]
    setup = math.fsum(
        (
            result.preparation.artifact_verification_wall_seconds,
            result.preparation.oracle_rescore_wall_seconds,
            result.preparation.label_reconstruction_wall_seconds,
            result.materialization_revalidation_wall_seconds,
            float(core["setup"]["wall_seconds"]),  # type: ignore[index]
        )
    )
    base = math.fsum(
        [
            setup,
            *[float(learner["projected_learner_wall_seconds"]) for learner in projected_learners],
        ]
    )
    margin = base * policy.walltime_margin_fraction + policy.fixed_walltime_margin_seconds
    required = base + margin
    maximum_update = max(
        float(learner["maximum_measured_update_wall_seconds"]) for learner in projected_learners
    )
    maximum_checkpoint_chain = max(
        float(learner["maximum_projected_safe_boundary_evidence_chain_wall_seconds"])
        for learner in projected_learners
    )
    maximum_audit = max(
        float(learner["maximum_measured_audit_wall_seconds"]) for learner in projected_learners
    )
    maximum_finalization_noncheckpoint = max(
        float(learner["maximum_finalization_noncheckpoint_wall_seconds"])
        for learner in projected_learners
    )
    maximum_inflight_noncheckpoint = math.fsum(
        (
            maximum_update,
            maximum_audit,
            maximum_finalization_noncheckpoint,
        )
    )
    signal_lead = math.ceil(
        maximum_inflight_noncheckpoint + maximum_checkpoint_chain + policy.signal_margin_seconds
    )
    required_memory = math.ceil(
        int(result.cpu_memory["peak_rss_bytes"]) * (1.0 + policy.memory_margin_fraction)
    )
    additional_segment_startup = math.fsum((setup, maximum_audit)) * (
        1.0 + policy.walltime_margin_fraction
    )
    segment_terminal_checkpoint = maximum_checkpoint_chain * (1.0 + policy.walltime_margin_fraction)
    signal_safe_boundary_checkpoint = maximum_checkpoint_chain * (
        1.0 + policy.walltime_margin_fraction
    )
    before_resume_checkpoint = maximum_checkpoint_chain * (1.0 + policy.walltime_margin_fraction)
    base_checkpoint_events = sum(
        int(learner["base_target_checkpoint_events"]) for learner in projected_learners
    )
    selection_checkpoint_events = sum(
        int(learner["selection_checkpoint_events"]) for learner in projected_learners
    )
    head_transition_checkpoint_events = sum(
        int(learner["head_transition_checkpoint_events"]) for learner in projected_learners
    )
    return {
        "schema_version": RESOURCE_PROJECTION_SCHEMA,
        "formula": (
            "setup + sum_head(build + 12760*max_update + "
            "639*max_audit + sum_event("
            "(fixed_bytes+slope*event_step)/min_component_throughput)); "
            "selection/head-transition use step 12760; "
            "plus one signal-safe and one terminal checkpoint per segment "
            "and one pre-resume checkpoint per continuation; "
            "required = base*(1+predeclared_margin_fraction) + fixed_margin"
        ),
        "maximum_updates_per_head": R3_MAXIMUM_UPDATES_PER_HEAD,
        "audit_cadence_updates": R3_AUDIT_CADENCE_UPDATES,
        "target_audit_events_per_head": R3_TARGET_AUDIT_COUNT_PER_HEAD,
        "durable_checkpoint_cadence_updates": (policy.durable_checkpoint_cadence_updates),
        "mandatory_checkpoint_updates": list(R3_MANDATORY_CHECKPOINT_UPDATES),
        "mandatory_checkpoint_roles": list(R3_MANDATORY_CHECKPOINT_ROLES),
        "checkpoint_on_selection": policy.checkpoint_on_selection,
        "checkpoint_before_head_transition": (policy.checkpoint_before_head_transition),
        "checkpoint_on_signal_safe_boundary": (policy.checkpoint_on_signal_safe_boundary),
        "checkpoint_at_segment_terminal": (policy.checkpoint_at_segment_terminal),
        "checkpoint_before_resume": policy.checkpoint_before_resume,
        "projected_base_checkpoint_events_before_segmentation": (base_checkpoint_events),
        "projected_selection_checkpoint_events": selection_checkpoint_events,
        "projected_head_transition_checkpoint_events": (head_transition_checkpoint_events),
        "checkpoint_event_coalescing_credit_assumed": False,
        "production_checkpoint_io_evidence_sha256": (
            result.production_checkpoint_io_evidence["evidence_sha256"]
        ),
        "production_checkpoint_byte_growth_formula": (
            result.production_checkpoint_io_evidence["schema_growth_model"]["formula"]
        ),
        "setup_wall_seconds": setup,
        "learners": projected_learners,
        "projected_base_wall_seconds": base,
        "walltime_margin_fraction": policy.walltime_margin_fraction,
        "fixed_walltime_margin_seconds": policy.fixed_walltime_margin_seconds,
        "projected_safety_margin_seconds": margin,
        "projected_required_wall_seconds": required,
        "maximum_measured_update_wall_seconds": maximum_update,
        "maximum_measured_audit_wall_seconds": maximum_audit,
        "maximum_projected_safe_boundary_evidence_chain_wall_seconds": (maximum_checkpoint_chain),
        "maximum_inflight_noncheckpoint_wall_seconds": (maximum_inflight_noncheckpoint),
        "maximum_finalization_noncheckpoint_wall_seconds": (maximum_finalization_noncheckpoint),
        "advance_signal_lead_formula": (
            "ceil(max_update+max_audit+max_finalization_noncheckpoint+"
            "max_safe_boundary_evidence_chain+signal_margin)"
        ),
        "advance_signal_lead_seconds": signal_lead,
        "observed_peak_cpu_memory_bytes": int(result.cpu_memory["peak_rss_bytes"]),
        "memory_margin_fraction": policy.memory_margin_fraction,
        "required_cpu_memory_bytes": required_memory,
        "projected_additional_segment_startup_seconds": (additional_segment_startup),
        "projected_segment_terminal_checkpoint_seconds": (segment_terminal_checkpoint),
        "projected_signal_safe_boundary_checkpoint_seconds": (signal_safe_boundary_checkpoint),
        "projected_before_resume_checkpoint_seconds": (before_resume_checkpoint),
    }


def _boundary(global_block: int) -> dict[str, object]:
    blocks_per_head = R3_MAXIMUM_UPDATES_PER_HEAD // R3_AUDIT_CADENCE_UPDATES
    if global_block < 0 or global_block > 2 * blocks_per_head:
        raise ValueError("global safe block is outside the two-head workload")
    if global_block <= blocks_per_head:
        bt_updates = global_block * R3_AUDIT_CADENCE_UPDATES
        prorm_updates = 0
    else:
        bt_updates = R3_MAXIMUM_UPDATES_PER_HEAD
        prorm_updates = (global_block - blocks_per_head) * R3_AUDIT_CADENCE_UPDATES
    if global_block < blocks_per_head:
        next_head: str | None = "bt_mle"
    elif global_block < 2 * blocks_per_head:
        next_head = "prorm_plus"
    else:
        next_head = None
    return {
        "global_safe_block": global_block,
        "bt_mle_completed_updates": bt_updates,
        "prorm_plus_completed_updates": prorm_updates,
        "next_head": next_head,
    }


def _segment_boundaries(
    projection: Mapping[str, object],
    *,
    effective_capacity_seconds: float,
) -> tuple[list[dict[str, object]], float]:
    learners = projection["learners"]
    if not isinstance(learners, list) or len(learners) != 2:
        raise ValueError("projection must contain exactly two learners")
    margin_multiplier = 1.0 + float(projection["walltime_margin_fraction"])
    setup = float(projection["setup_wall_seconds"])
    fixed_margin = float(projection["fixed_walltime_margin_seconds"])
    blocks_per_head = R3_MAXIMUM_UPDATES_PER_HEAD // R3_AUDIT_CADENCE_UPDATES
    head_unit_costs: list[list[float]] = []
    head_enter_costs: list[float] = []
    for learner_index, learner in enumerate(learners):
        base_checkpoint_updates = set(learner["base_checkpoint_updates"])
        dynamic_checkpoint_events = int(learner["dynamic_pre_segmentation_checkpoint_events"])
        initial_cost = math.fsum(
            (
                float(learner["trainer_enter_wall_seconds"]),
                float(learner["maximum_measured_audit_wall_seconds"]),
            )
        )
        head_enter_costs.append(initial_cost * margin_multiplier)
        learner_unit_costs: list[float] = []
        for block_index in range(blocks_per_head):
            completed_update = (block_index + 1) * R3_AUDIT_CADENCE_UPDATES
            checkpoint_events_at_boundary = int(completed_update in base_checkpoint_updates)
            if block_index == blocks_per_head - 1:
                checkpoint_events_at_boundary += dynamic_checkpoint_events
            block_cost = math.fsum(
                (
                    R3_AUDIT_CADENCE_UPDATES
                    * float(learner["maximum_measured_update_wall_seconds"]),
                    float(learner["maximum_measured_audit_wall_seconds"]),
                    _checkpoint_and_progress_wall_seconds_at_step(
                        learner,
                        completed_update,
                    )
                    * checkpoint_events_at_boundary,
                )
            )
            cost = block_cost * margin_multiplier
            if block_index == 0:
                cost += initial_cost * margin_multiplier
                if learner_index == 0:
                    cost += setup * margin_multiplier + fixed_margin
            learner_unit_costs.append(cost)
        head_unit_costs.append(learner_unit_costs)
    required = float(projection["projected_required_wall_seconds"])
    exact_unit_costs = [cost for learner_costs in head_unit_costs for cost in learner_costs]
    if not math.isclose(
        math.fsum(exact_unit_costs),
        required,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("safe-block projection does not reconstruct required walltime")
    # Nominal head boundaries cannot price an actual journal once BT converges
    # early: the next executable block may be ProRM+ in the same segment.  Bind
    # every transferable local block to the more expensive of the two heads,
    # and repeat that head-agnostic schedule for the full two-head worst case.
    pairwise_worst_costs = [
        max(bt_cost, prorm_cost)
        for bt_cost, prorm_cost in zip(
            head_unit_costs[0],
            head_unit_costs[1],
            strict=True,
        )
    ]
    unit_costs = [*pairwise_worst_costs, *pairwise_worst_costs]
    head_agnostic_block_pricing_reserve = math.fsum(unit_costs) - required
    if head_agnostic_block_pricing_reserve < -1.0e-9:
        raise RuntimeError("head-agnostic safe-block prices are not conservative")
    head_agnostic_block_pricing_reserve = max(
        0.0,
        head_agnostic_block_pricing_reserve,
    )
    maximum_dynamic_events = max(
        int(learner["dynamic_pre_segmentation_checkpoint_events"]) for learner in learners
    )
    maximum_target_boundary_chain = max(
        _safe_boundary_evidence_chain_wall_seconds_at_step(
            learner,
            R3_MAXIMUM_UPDATES_PER_HEAD,
        )
        for learner in learners
    )
    early_transition_reserve = math.fsum(
        (
            max(head_enter_costs),
            maximum_dynamic_events * maximum_target_boundary_chain * margin_multiplier,
        )
    )

    segments: list[dict[str, object]] = []
    continuation_startup = float(projection["projected_additional_segment_startup_seconds"])

    def checkpoint_at_boundary(
        global_block: int,
    ) -> tuple[str, int, int, float, float]:
        if global_block <= 0 or global_block > 2 * blocks_per_head:
            raise ValueError("checkpoint boundary must follow a completed safe block")
        if global_block <= blocks_per_head:
            learner = learners[0]
            step = global_block * R3_AUDIT_CADENCE_UPDATES
        else:
            learner = learners[1]
            step = (global_block - blocks_per_head) * R3_AUDIT_CADENCE_UPDATES
        worst_bytes = max(
            _checkpoint_bytes_at_step(
                candidate,
                R3_MAXIMUM_UPDATES_PER_HEAD,
            )
            for candidate in learners
        )
        worst_chain = max(
            _safe_boundary_evidence_chain_wall_seconds_at_step(
                candidate,
                R3_MAXIMUM_UPDATES_PER_HEAD,
            )
            for candidate in learners
        )
        worst_finalization = max(
            float(candidate["maximum_finalization_noncheckpoint_wall_seconds"])
            for candidate in learners
        )
        return (
            str(learner["learner"]),
            step,
            worst_bytes,
            worst_chain * margin_multiplier,
            worst_finalization * margin_multiplier,
        )

    start_block = 0
    total_blocks = len(unit_costs)
    while start_block < total_blocks:
        startup = 0.0 if not segments else continuation_startup
        transition_reserve = early_transition_reserve if start_block < blocks_per_head else 0.0
        remaining_work = math.fsum(unit_costs[start_block:])
        final_checkpoint = checkpoint_at_boundary(total_blocks)
        final_boundary_overhead = 2 * final_checkpoint[3] + final_checkpoint[4]
        if (
            startup + transition_reserve + remaining_work + final_boundary_overhead
            <= effective_capacity_seconds
        ):
            end_block = total_blocks
            work = startup + transition_reserve + remaining_work
            continuation_required = False
            boundary_overhead = final_boundary_overhead
        else:
            end_block = start_block
            work = startup + transition_reserve
            while end_block < total_blocks:
                candidate = work + unit_costs[end_block]
                candidate_end = end_block + 1
                candidate_checkpoint = checkpoint_at_boundary(candidate_end)
                event_multiplier = 2 if candidate_end == total_blocks else 3
                candidate_boundary_overhead = (
                    event_multiplier * candidate_checkpoint[3] + candidate_checkpoint[4]
                )
                if candidate + candidate_boundary_overhead > effective_capacity_seconds:
                    break
                work = candidate
                end_block = candidate_end
            if end_block == start_block:
                raise ValueError(
                    "continuation startup plus one 20-update safe block and "
                    "all mandatory boundary checkpoints cannot fit in one "
                    "allocation"
                )
            continuation_required = True
            checkpoint = checkpoint_at_boundary(end_block)
            boundary_overhead = 3 * checkpoint[3] + checkpoint[4]
        if not continuation_required:
            checkpoint = final_checkpoint
        segments.append(
            {
                "segment_index": len(segments) + 1,
                "start_boundary": _boundary(start_block),
                "end_boundary": _boundary(end_block),
                "projected_work_seconds": work + boundary_overhead,
                "effective_capacity_seconds": effective_capacity_seconds,
                "continuation_required": continuation_required,
                "checkpoint_learner": checkpoint[0],
                "checkpoint_update": checkpoint[1],
                "checkpoint_serialized_bytes_upper_bound": checkpoint[2],
                "projected_checkpoint_roundtrip_wall_seconds_with_margin": (checkpoint[3]),
                "projected_finalization_noncheckpoint_wall_seconds_with_margin": (checkpoint[4]),
                "max_safe_update_blocks_to_execute": (end_block - start_block),
                "safe_block_pricing_rule": ("pairwise_max_bt_mle_prorm_plus_by_local_block_index"),
                "head_agnostic_block_prices": True,
                "projected_early_head_transition_reserve_seconds": (transition_reserve),
                "fixed_ordered_head_transition_allowed": [
                    "bt_mle",
                    "prorm_plus",
                ],
                "journal_actual_cursor_required": True,
                "nominal_boundaries_are_worst_case_projection_only": True,
                "actual_cursor_must_reach_nominal_end": False,
                "projected_signal_safe_boundary_checkpoint_events": 1,
                "projected_segment_terminal_checkpoint_events": 1,
                "projected_before_resume_checkpoint_events": int(continuation_required),
                "checkpoint_event_coalescing_credit_assumed": False,
            }
        )
        start_block = end_block
    return segments, head_agnostic_block_pricing_reserve


def _resource_plan_payload(
    *,
    result: FormalCudaProfileResult,
    policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    requested_walltime_seconds: int,
    array_concurrency: int,
    cpus_per_task: int,
    memory_bytes: int,
) -> dict[str, object]:
    projection = _projection(result, policy)
    signal_lead = int(projection["advance_signal_lead_seconds"])
    if requested_walltime_seconds > envelope.max_allocation_wall_seconds:
        raise ValueError("requested walltime exceeds the scheduler evidence")
    effective_capacity = requested_walltime_seconds - signal_lead
    if effective_capacity <= 0:
        raise ValueError("requested walltime cannot cover the frozen signal lead")
    if array_concurrency > min(3, envelope.max_array_concurrency):
        raise ValueError("array concurrency exceeds the R3 wave or scheduler evidence")
    if cpus_per_task > envelope.max_cpus_per_task:
        raise ValueError("CPU request exceeds the resource evidence")
    required_memory = int(projection["required_cpu_memory_bytes"])
    if memory_bytes < required_memory:
        raise ValueError("memory request does not cover observed peak plus frozen margin")
    if memory_bytes > envelope.max_memory_bytes:
        raise ValueError("memory request exceeds the resource evidence")
    if envelope.max_gpus_per_task < 1:
        raise ValueError("resource evidence cannot provide the required primary GPU")
    segments, head_agnostic_block_pricing_reserve = _segment_boundaries(
        projection,
        effective_capacity_seconds=float(effective_capacity),
    )
    if len(segments) > envelope.max_scheduler_segments:
        raise ValueError("projected workload exceeds the frozen scheduler segment limit")
    total_safe_block_budget = sum(
        int(segment["max_safe_update_blocks_to_execute"]) for segment in segments
    )
    if total_safe_block_budget != R3_TOTAL_SAFE_UPDATE_BLOCKS:
        raise RuntimeError("segment block budgets do not cover the nominal worst case")
    segment_execution_contract = {
        "safe_update_block_size": R3_AUDIT_CADENCE_UPDATES,
        "fixed_ordered_heads": list(PHASE2_PROFILE_LEARNER_ORDER),
        "early_convergence_transition_within_segment_allowed": True,
        "journal_actual_cursor_required": True,
        "per_segment_block_budget_must_not_be_exceeded": True,
        "nominal_start_end_are_worst_case_projection_only": True,
        "actual_cursor_must_reach_nominal_end": False,
        "reverse_or_repeated_head_transition_forbidden": True,
        "safe_block_pricing_rule": ("pairwise_max_bt_mle_prorm_plus_by_local_block_index"),
        "every_transferable_block_is_head_agnostic": True,
        "early_transition_enter_and_dynamic_checkpoint_reserve_required": True,
        "total_nominal_safe_update_blocks": R3_TOTAL_SAFE_UPDATE_BLOCKS,
        "total_max_safe_update_block_budget": total_safe_block_budget,
    }
    early_head_transition_reserve = math.fsum(
        float(segment["projected_early_head_transition_reserve_seconds"]) for segment in segments
    )
    continuation_startup_overhead = (len(segments) - 1) * float(
        projection["projected_additional_segment_startup_seconds"]
    )
    signal_safe_boundary_checkpoint_events = len(segments)
    segment_terminal_checkpoint_events = len(segments)
    before_resume_checkpoint_events = len(segments) - 1
    signal_safe_boundary_checkpoint_overhead = math.fsum(
        float(segment["projected_checkpoint_roundtrip_wall_seconds_with_margin"])
        for segment in segments
    )
    segment_terminal_checkpoint_overhead = math.fsum(
        float(segment["projected_checkpoint_roundtrip_wall_seconds_with_margin"])
        for segment in segments
    )
    before_resume_checkpoint_overhead = math.fsum(
        float(segment["projected_checkpoint_roundtrip_wall_seconds_with_margin"])
        for segment in segments
        if segment["continuation_required"] is True
    )
    finalization_noncheckpoint_overhead = math.fsum(
        float(segment["projected_finalization_noncheckpoint_wall_seconds_with_margin"])
        for segment in segments
    )
    continuation_overhead = (
        continuation_startup_overhead
        + signal_safe_boundary_checkpoint_overhead
        + segment_terminal_checkpoint_overhead
        + before_resume_checkpoint_overhead
        + finalization_noncheckpoint_overhead
    )
    total_projected = (
        float(projection["projected_required_wall_seconds"])
        + head_agnostic_block_pricing_reserve
        + early_head_transition_reserve
        + continuation_overhead
    )
    total_effective_capacity = len(segments) * effective_capacity
    if total_projected > total_effective_capacity:
        raise RuntimeError("segmented resource plan does not cover its projection")
    base_checkpoint_events = int(projection["projected_base_checkpoint_events_before_segmentation"])
    selection_checkpoint_events = int(projection["projected_selection_checkpoint_events"])
    head_transition_checkpoint_events = int(
        projection["projected_head_transition_checkpoint_events"]
    )
    checkpoint_event_projection = {
        "cadence_and_learning_rate_events": base_checkpoint_events,
        "selection_events": selection_checkpoint_events,
        "head_transition_events": head_transition_checkpoint_events,
        "signal_safe_boundary_events": signal_safe_boundary_checkpoint_events,
        "segment_terminal_events": segment_terminal_checkpoint_events,
        "before_resume_events": before_resume_checkpoint_events,
        "total_events": (
            base_checkpoint_events
            + selection_checkpoint_events
            + head_transition_checkpoint_events
            + signal_safe_boundary_checkpoint_events
            + segment_terminal_checkpoint_events
            + before_resume_checkpoint_events
        ),
        "coalescing_credit_assumed": False,
    }
    return {
        "schema_version": RESOURCE_PLAN_SCHEMA,
        "role": RESOURCE_PLAN_ROLE,
        "formal_profile_sha256": result.formal_profile_sha256,
        "profile_run_sha256": result.profile_run.profile_run_sha256,
        "safety_margin_policy_sha256": policy.policy_sha256,
        "scheduler_resource_envelope_sha256": envelope.envelope_sha256,
        "scheduler_raw_evidence_sha256": envelope.scheduler_raw_evidence_sha256,
        "resource_raw_evidence_sha256": envelope.resource_raw_evidence_sha256,
        "projection": projection,
        "slurm_account": HPC4_SLURM_ACCOUNT,
        "partition": envelope.partition,
        "gpu_name": envelope.gpu_name,
        "gpus_per_task": 1,
        "cpus_per_task": cpus_per_task,
        "memory_bytes": memory_bytes,
        "requested_walltime_seconds_per_segment": requested_walltime_seconds,
        "scheduler_max_walltime_seconds": envelope.max_allocation_wall_seconds,
        "advance_signal_lead_seconds": signal_lead,
        "audit_cadence_updates": R3_AUDIT_CADENCE_UPDATES,
        "durable_checkpoint_cadence_updates": (policy.durable_checkpoint_cadence_updates),
        "mandatory_checkpoint_updates": list(R3_MANDATORY_CHECKPOINT_UPDATES),
        "mandatory_checkpoint_roles": list(R3_MANDATORY_CHECKPOINT_ROLES),
        "checkpoint_on_selection": policy.checkpoint_on_selection,
        "checkpoint_before_head_transition": (policy.checkpoint_before_head_transition),
        "checkpoint_on_signal_safe_boundary": (policy.checkpoint_on_signal_safe_boundary),
        "checkpoint_at_segment_terminal": (policy.checkpoint_at_segment_terminal),
        "checkpoint_before_resume": policy.checkpoint_before_resume,
        "checkpoint_event_projection": checkpoint_event_projection,
        "array_concurrency": array_concurrency,
        "scheduler_max_array_concurrency": envelope.max_array_concurrency,
        "max_scheduler_segments": len(segments),
        "scheduler_evidence_max_segments": envelope.max_scheduler_segments,
        "segment_boundaries": segments,
        "segment_execution_contract": segment_execution_contract,
        "total_effective_capacity_seconds": total_effective_capacity,
        "projected_continuation_startup_overhead_seconds": (continuation_startup_overhead),
        "projected_head_agnostic_block_pricing_reserve_seconds": (
            head_agnostic_block_pricing_reserve
        ),
        "projected_early_head_transition_reserve_seconds": (early_head_transition_reserve),
        "projected_signal_safe_boundary_checkpoint_overhead_seconds": (
            signal_safe_boundary_checkpoint_overhead
        ),
        "projected_segment_terminal_checkpoint_overhead_seconds": (
            segment_terminal_checkpoint_overhead
        ),
        "projected_before_resume_checkpoint_overhead_seconds": (before_resume_checkpoint_overhead),
        "projected_finalization_noncheckpoint_overhead_seconds": (
            finalization_noncheckpoint_overhead
        ),
        "projected_continuation_overhead_seconds": continuation_overhead,
        "projected_required_wall_seconds": total_projected,
        "coverage_proved": True,
        "information_boundary": "runtime_memory_io_only_no_scientific_adaptation",
    }


@dataclass(frozen=True, slots=True)
class GatePResourcePlan:
    """Self-hashed Slurm plan derived only from formal operational evidence."""

    formal_result: FormalCudaProfileResult = field(repr=False, compare=False)
    safety_policy: ProfileSafetyMarginPolicy = field(repr=False, compare=False)
    envelope: SchedulerResourceEnvelope = field(repr=False, compare=False)
    schema_version: str
    role: str
    formal_profile_sha256: str
    profile_run_sha256: str
    safety_margin_policy_sha256: str
    scheduler_resource_envelope_sha256: str
    scheduler_raw_evidence_sha256: str
    resource_raw_evidence_sha256: str
    projection: Mapping[str, object]
    slurm_account: str
    partition: str
    gpu_name: str
    gpus_per_task: int
    cpus_per_task: int
    memory_bytes: int
    requested_walltime_seconds_per_segment: int
    scheduler_max_walltime_seconds: int
    advance_signal_lead_seconds: int
    audit_cadence_updates: int
    durable_checkpoint_cadence_updates: int
    mandatory_checkpoint_updates: tuple[int, ...]
    mandatory_checkpoint_roles: tuple[str, ...]
    checkpoint_on_selection: Literal[True]
    checkpoint_before_head_transition: Literal[True]
    checkpoint_on_signal_safe_boundary: Literal[True]
    checkpoint_at_segment_terminal: Literal[True]
    checkpoint_before_resume: Literal[True]
    checkpoint_event_projection: Mapping[str, object]
    array_concurrency: int
    scheduler_max_array_concurrency: int
    max_scheduler_segments: int
    scheduler_evidence_max_segments: int
    segment_boundaries: tuple[Mapping[str, object], ...]
    segment_execution_contract: Mapping[str, object]
    total_effective_capacity_seconds: int
    projected_continuation_startup_overhead_seconds: float
    projected_head_agnostic_block_pricing_reserve_seconds: float
    projected_early_head_transition_reserve_seconds: float
    projected_signal_safe_boundary_checkpoint_overhead_seconds: float
    projected_segment_terminal_checkpoint_overhead_seconds: float
    projected_before_resume_checkpoint_overhead_seconds: float
    projected_finalization_noncheckpoint_overhead_seconds: float
    projected_continuation_overhead_seconds: float
    projected_required_wall_seconds: float
    coverage_proved: Literal[True]
    information_boundary: str
    resource_plan_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        result = validate_formal_cuda_profile_result(self.formal_result)
        if type(self.safety_policy) is not ProfileSafetyMarginPolicy:
            raise TypeError("safety_policy must be ProfileSafetyMarginPolicy")
        if type(self.envelope) is not SchedulerResourceEnvelope:
            raise TypeError("envelope must be SchedulerResourceEnvelope")
        self.safety_policy.validate_integrity()
        self.envelope.validate_integrity()
        if (
            result.safety_margin_policy_sha256 != self.safety_policy.policy_sha256
            or result.envelope.envelope_sha256 != self.envelope.envelope_sha256
        ):
            raise ValueError("resource plan dependencies differ from the formal profile")
        expected = _resource_plan_payload(
            result=result,
            policy=self.safety_policy,
            envelope=self.envelope,
            requested_walltime_seconds=self.requested_walltime_seconds_per_segment,
            array_concurrency=self.array_concurrency,
            cpus_per_task=self.cpus_per_task,
            memory_bytes=self.memory_bytes,
        )
        observed = {
            "schema_version": self.schema_version,
            "role": self.role,
            "formal_profile_sha256": self.formal_profile_sha256,
            "profile_run_sha256": self.profile_run_sha256,
            "safety_margin_policy_sha256": self.safety_margin_policy_sha256,
            "scheduler_resource_envelope_sha256": (self.scheduler_resource_envelope_sha256),
            "scheduler_raw_evidence_sha256": self.scheduler_raw_evidence_sha256,
            "resource_raw_evidence_sha256": self.resource_raw_evidence_sha256,
            "projection": self.projection,
            "slurm_account": self.slurm_account,
            "partition": self.partition,
            "gpu_name": self.gpu_name,
            "gpus_per_task": self.gpus_per_task,
            "cpus_per_task": self.cpus_per_task,
            "memory_bytes": self.memory_bytes,
            "requested_walltime_seconds_per_segment": (self.requested_walltime_seconds_per_segment),
            "scheduler_max_walltime_seconds": self.scheduler_max_walltime_seconds,
            "advance_signal_lead_seconds": self.advance_signal_lead_seconds,
            "audit_cadence_updates": self.audit_cadence_updates,
            "durable_checkpoint_cadence_updates": (self.durable_checkpoint_cadence_updates),
            "mandatory_checkpoint_updates": list(self.mandatory_checkpoint_updates),
            "mandatory_checkpoint_roles": list(self.mandatory_checkpoint_roles),
            "checkpoint_on_selection": self.checkpoint_on_selection,
            "checkpoint_before_head_transition": (self.checkpoint_before_head_transition),
            "checkpoint_on_signal_safe_boundary": (self.checkpoint_on_signal_safe_boundary),
            "checkpoint_at_segment_terminal": (self.checkpoint_at_segment_terminal),
            "checkpoint_before_resume": self.checkpoint_before_resume,
            "checkpoint_event_projection": self.checkpoint_event_projection,
            "array_concurrency": self.array_concurrency,
            "scheduler_max_array_concurrency": (self.scheduler_max_array_concurrency),
            "max_scheduler_segments": self.max_scheduler_segments,
            "scheduler_evidence_max_segments": self.scheduler_evidence_max_segments,
            "segment_boundaries": list(self.segment_boundaries),
            "segment_execution_contract": self.segment_execution_contract,
            "total_effective_capacity_seconds": (self.total_effective_capacity_seconds),
            "projected_continuation_startup_overhead_seconds": (
                self.projected_continuation_startup_overhead_seconds
            ),
            "projected_head_agnostic_block_pricing_reserve_seconds": (
                self.projected_head_agnostic_block_pricing_reserve_seconds
            ),
            "projected_early_head_transition_reserve_seconds": (
                self.projected_early_head_transition_reserve_seconds
            ),
            "projected_signal_safe_boundary_checkpoint_overhead_seconds": (
                self.projected_signal_safe_boundary_checkpoint_overhead_seconds
            ),
            "projected_segment_terminal_checkpoint_overhead_seconds": (
                self.projected_segment_terminal_checkpoint_overhead_seconds
            ),
            "projected_before_resume_checkpoint_overhead_seconds": (
                self.projected_before_resume_checkpoint_overhead_seconds
            ),
            "projected_finalization_noncheckpoint_overhead_seconds": (
                self.projected_finalization_noncheckpoint_overhead_seconds
            ),
            "projected_continuation_overhead_seconds": (
                self.projected_continuation_overhead_seconds
            ),
            "projected_required_wall_seconds": (self.projected_required_wall_seconds),
            "coverage_proved": self.coverage_proved,
            "information_boundary": self.information_boundary,
        }
        if _json_copy(observed, name="resource plan fields") != expected:
            raise ValueError("resource plan fields differ from the derived projection")
        if self.coverage_proved is not True:
            raise ValueError("resource plan does not prove walltime coverage")
        _assert_no_sensitive_state(expected, path="resource_plan")
        _digest(self.resource_plan_sha256, name="resource_plan_sha256")
        if _canonical_sha256(expected) != self.resource_plan_sha256:
            raise ValueError("resource plan SHA256 does not match its contents")

    def validate_integrity(self) -> None:
        _require_factory(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _resource_plan_payload(
            result=self.formal_result,
            policy=self.safety_policy,
            envelope=self.envelope,
            requested_walltime_seconds=self.requested_walltime_seconds_per_segment,
            array_concurrency=self.array_concurrency,
            cpus_per_task=self.cpus_per_task,
            memory_bytes=self.memory_bytes,
        )
        return {**payload, "resource_plan_sha256": self.resource_plan_sha256}


def build_gate_p_resource_plan(
    formal_result: FormalCudaProfileResult,
    *,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    requested_walltime_seconds_per_segment: int,
    array_concurrency: int,
    cpus_per_task: int,
    memory_bytes: int,
) -> GatePResourcePlan:
    """Project both heads to update 12,760 and freeze the exact Slurm plan."""

    result = validate_formal_cuda_profile_result(formal_result)
    if type(safety_policy) is not ProfileSafetyMarginPolicy:
        raise TypeError("safety_policy must be ProfileSafetyMarginPolicy")
    if type(envelope) is not SchedulerResourceEnvelope:
        raise TypeError("envelope must be SchedulerResourceEnvelope")
    safety_policy.validate_integrity()
    envelope.validate_integrity()
    walltime = _exact_int(
        requested_walltime_seconds_per_segment,
        name="requested_walltime_seconds_per_segment",
        minimum=1,
    )
    concurrency = _exact_int(
        array_concurrency,
        name="array_concurrency",
        minimum=1,
    )
    cpus = _exact_int(cpus_per_task, name="cpus_per_task", minimum=1)
    memory = _exact_int(memory_bytes, name="memory_bytes", minimum=1)
    payload = _resource_plan_payload(
        result=result,
        policy=safety_policy,
        envelope=envelope,
        requested_walltime_seconds=walltime,
        array_concurrency=concurrency,
        cpus_per_task=cpus,
        memory_bytes=memory,
    )
    segments = tuple(payload["segment_boundaries"])  # type: ignore[arg-type]
    plan = GatePResourcePlan(
        formal_result=result,
        safety_policy=safety_policy,
        envelope=envelope,
        schema_version=str(payload["schema_version"]),
        role=str(payload["role"]),
        formal_profile_sha256=str(payload["formal_profile_sha256"]),
        profile_run_sha256=str(payload["profile_run_sha256"]),
        safety_margin_policy_sha256=str(payload["safety_margin_policy_sha256"]),
        scheduler_resource_envelope_sha256=str(payload["scheduler_resource_envelope_sha256"]),
        scheduler_raw_evidence_sha256=str(payload["scheduler_raw_evidence_sha256"]),
        resource_raw_evidence_sha256=str(payload["resource_raw_evidence_sha256"]),
        projection=payload["projection"],  # type: ignore[arg-type]
        slurm_account=str(payload["slurm_account"]),
        partition=str(payload["partition"]),
        gpu_name=str(payload["gpu_name"]),
        gpus_per_task=int(payload["gpus_per_task"]),
        cpus_per_task=int(payload["cpus_per_task"]),
        memory_bytes=int(payload["memory_bytes"]),
        requested_walltime_seconds_per_segment=int(
            payload["requested_walltime_seconds_per_segment"]
        ),
        scheduler_max_walltime_seconds=int(payload["scheduler_max_walltime_seconds"]),
        advance_signal_lead_seconds=int(payload["advance_signal_lead_seconds"]),
        audit_cadence_updates=int(payload["audit_cadence_updates"]),
        durable_checkpoint_cadence_updates=int(payload["durable_checkpoint_cadence_updates"]),
        mandatory_checkpoint_updates=tuple(
            payload["mandatory_checkpoint_updates"]  # type: ignore[arg-type]
        ),
        mandatory_checkpoint_roles=tuple(
            payload["mandatory_checkpoint_roles"]  # type: ignore[arg-type]
        ),
        checkpoint_on_selection=True,
        checkpoint_before_head_transition=True,
        checkpoint_on_signal_safe_boundary=True,
        checkpoint_at_segment_terminal=True,
        checkpoint_before_resume=True,
        checkpoint_event_projection=payload[  # type: ignore[arg-type]
            "checkpoint_event_projection"
        ],
        array_concurrency=int(payload["array_concurrency"]),
        scheduler_max_array_concurrency=int(payload["scheduler_max_array_concurrency"]),
        max_scheduler_segments=int(payload["max_scheduler_segments"]),
        scheduler_evidence_max_segments=int(payload["scheduler_evidence_max_segments"]),
        segment_boundaries=segments,
        segment_execution_contract=payload[  # type: ignore[arg-type]
            "segment_execution_contract"
        ],
        total_effective_capacity_seconds=int(payload["total_effective_capacity_seconds"]),
        projected_continuation_startup_overhead_seconds=float(
            payload["projected_continuation_startup_overhead_seconds"]
        ),
        projected_head_agnostic_block_pricing_reserve_seconds=float(
            payload["projected_head_agnostic_block_pricing_reserve_seconds"]
        ),
        projected_early_head_transition_reserve_seconds=float(
            payload["projected_early_head_transition_reserve_seconds"]
        ),
        projected_signal_safe_boundary_checkpoint_overhead_seconds=float(
            payload["projected_signal_safe_boundary_checkpoint_overhead_seconds"]
        ),
        projected_segment_terminal_checkpoint_overhead_seconds=float(
            payload["projected_segment_terminal_checkpoint_overhead_seconds"]
        ),
        projected_before_resume_checkpoint_overhead_seconds=float(
            payload["projected_before_resume_checkpoint_overhead_seconds"]
        ),
        projected_finalization_noncheckpoint_overhead_seconds=float(
            payload["projected_finalization_noncheckpoint_overhead_seconds"]
        ),
        projected_continuation_overhead_seconds=float(
            payload["projected_continuation_overhead_seconds"]
        ),
        projected_required_wall_seconds=float(payload["projected_required_wall_seconds"]),
        coverage_proved=True,
        information_boundary=str(payload["information_boundary"]),
        resource_plan_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    plan.validate_integrity()
    return plan


def validate_gate_p_resource_plan(value: object) -> GatePResourcePlan:
    if type(value) is not GatePResourcePlan:
        raise TypeError("resource plan must be exactly GatePResourcePlan")
    value.validate_integrity()
    return value


def resource_plan_artifact_ref(plan: GatePResourcePlan) -> ArtifactRef:
    validated = validate_gate_p_resource_plan(plan)
    return ArtifactRef(
        schema_version=RESOURCE_PLAN_SCHEMA,
        artifact_sha256=validated.resource_plan_sha256,
        role=RESOURCE_PLAN_ROLE,
    )


__all__ = [
    "FORMAL_PROFILE_INFORMATION_BOUNDARY",
    "HPC4_SLURM_ACCOUNT",
    "PROFILE_PREPARATION_TIMINGS_SCHEMA",
    "PROFILE_SAFETY_MARGIN_POLICY_SCHEMA",
    "PRODUCTION_CHECKPOINT_IO_PROFILE_SCHEMA",
    "PRODUCTION_CONTROLLER_CHECKPOINT_SCHEMA",
    "PRODUCTION_DURABLE_CHECKPOINT_ENVELOPE_SCHEMA",
    "PRODUCTION_OUTER_CHECKPOINT_PAYLOAD_SCHEMA",
    "PRODUCTION_PROFILE_BENCHMARK_PAYLOAD_SCHEMA",
    "RESOURCE_PROJECTION_SCHEMA",
    "R3_AUDIT_CADENCE_UPDATES",
    "R3_MANDATORY_CHECKPOINT_UPDATES",
    "R3_MANDATORY_CHECKPOINT_ROLES",
    "R3_MAXIMUM_UPDATES_PER_HEAD",
    "SCHEDULER_RESOURCE_ENVELOPE_SCHEMA",
    "FormalCudaProfileResult",
    "GatePResourcePlan",
    "ProfilePreparationTimings",
    "ProfileSafetyMarginPolicy",
    "SchedulerResourceEnvelope",
    "build_gate_p_resource_plan",
    "formal_cuda_profile_artifact_ref",
    "freeze_profile_safety_margin_policy",
    "record_profile_preparation_timings",
    "record_profile_preparation_from_train_input",
    "resource_plan_artifact_ref",
    "run_formal_gate_p_cuda_profile",
    "validate_formal_cuda_profile_result",
    "validate_gate_p_resource_plan",
    "validate_scheduler_resource_envelope",
]
