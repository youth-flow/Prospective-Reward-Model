"""Formal train-only primary execution for Phase-2 recovery revision 3.

This module is the promotion boundary between the claim-free mathematical
training core and an admitted R3 scheduler segment.  It deliberately accepts
only typed R3 capabilities, requires a durable checkpoint store, binds every
checkpoint to the stable logical/head run, and records the scheduler segment
inside each checkpoint payload.  A result from :mod:`phase2_primary` is never
formal evidence by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias

import torch

from . import phase2_training as _training
from .contracts import BT_MLE, PRORM_PLUS
from .phase2_checkpoint import (
    CHECKPOINT_SCHEMA,
    PLANNED_BOUNDARY_RECEIPT_SCHEMA,
    SIGNAL_RECEIPT_SCHEMA,
    TRAINING_PROGRESS_DETAILS_SCHEMA,
    CheckpointInterruption,
    CheckpointSignal,
    DurableCheckpointStore,
    PlannedSegmentBoundary,
)
from .phase2_primary import (
    NeutralPhase2TrainingContext,
    build_primary_core_trainer,
)
from .phase2_r3_identity import (
    R3_PRIMARY_HEADS,
    VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
    ArtifactRef,
    PrimarySegmentAdmission,
)
from .phase2_r3_profile_artifacts import (
    VerifiedGatePOperationalBundle,
    formal_profile_artifact_ref,
    resource_plan_artifact_ref,
)

FORMAL_PRIMARY_CHECKPOINT_BINDING_SCHEMA: Final = "phase2-recovery-r3-primary-checkpoint-binding/v1"
FORMAL_PRIMARY_CHECKPOINT_PAYLOAD_SCHEMA: Final = "phase2-recovery-r3-primary-checkpoint-payload/v1"
FORMAL_PRIMARY_HEAD_RESULT_SCHEMA: Final = "phase2-recovery-r3-primary-head-segment-output/v1"
FORMAL_PRIMARY_SELECTED_TERMINAL_PAYLOAD_SCHEMA: Final = (
    "phase2-recovery-r3-selected-primary-terminal-payload/v1"
)
FORMAL_PRIMARY_TERMINAL_RESULT_CORE_SCHEMA: Final = (
    "phase2-recovery-r3-primary-terminal-result-core/v1"
)
FORMAL_PRIMARY_SELECTED_TERMINAL_ARTIFACT_SCHEMA: Final = (
    "phase2-recovery-r3-selected-primary-terminal-artifact/v1"
)
FORMAL_PRIMARY_SELECTED_TERMINAL_ARTIFACT_ROLE: Final = "selected_primary_terminal_checkpoint"
HEAD_EXECUTION_SLICE_SCHEMA: Final = "phase2-recovery-r3-primary-head-execution-slice/v1"
FORMAL_PRIMARY_EXECUTION_ROLE: Final = "phase2_recovery_r3_primary"
FORMAL_PRIMARY_INFORMATION_BOUNDARY: Final = "train_only"

PrimaryLearner: TypeAlias = Literal["bt_mle", "prorm_plus"]
_RUNTIME_FACTORY_TOKEN = object()


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
        raise ValueError("R3 primary identity must contain strict JSON data") from error
    return hashlib.sha256(encoded).hexdigest()


def _orchestrator_evidence_sha256(value: object) -> str:
    """Hash pure-data orchestrator evidence using its newline convention."""

    try:
        encoded = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("head execution slice must contain strict JSON data") from error
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _learner(value: object) -> PrimaryLearner:
    if value not in R3_PRIMARY_HEADS:
        raise ValueError(f"learner must be one of {R3_PRIMARY_HEADS!r}")
    return value  # type: ignore[return-value]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_seconds(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _require_admission(value: object) -> PrimarySegmentAdmission:
    if type(value) is not PrimarySegmentAdmission:
        raise TypeError("admission must be an exact PrimarySegmentAdmission")
    value.validate_integrity()
    return value


@dataclass(frozen=True, slots=True)
class SlurmSegmentRuntime:
    """Runtime identity captured from the actual Slurm process environment."""

    schema_version: str
    design_sha256: str
    admission_sha256: str
    scheduler_segment_id: str
    segment_index: int
    task_id: int
    seed: int
    cluster: str
    job_id: str
    array_job_id: str
    array_task_id: int
    account: str
    partition: str
    requested_walltime_seconds: int
    captured_monotonic_ns: int
    runtime_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _RUNTIME_FACTORY_TOKEN:
            raise TypeError("Slurm runtime must be captured from the process environment")
        object.__setattr__(self, "_seal", _RUNTIME_FACTORY_TOKEN)
        self._validate_structure()

    def _validate_structure(self) -> None:
        if self.schema_version != "phase2-recovery-r3-slurm-segment-runtime/v1":
            raise ValueError("Slurm runtime schema is not frozen")
        for name in (
            "design_sha256",
            "admission_sha256",
            "scheduler_segment_id",
            "runtime_sha256",
        ):
            _training._validate_digest(getattr(self, name), name=name)
        for name in ("cluster", "job_id", "array_job_id", "account", "partition"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        _positive_integer(self.segment_index, name="segment_index")
        _positive_integer(self.seed, name="seed")
        if isinstance(self.task_id, bool) or not isinstance(self.task_id, int):
            raise TypeError("task_id must be an integer")
        if isinstance(self.array_task_id, bool) or not isinstance(
            self.array_task_id,
            int,
        ):
            raise TypeError("array_task_id must be an integer")
        if self.account != "sigroup":
            raise ValueError("formal R3 execution must use the frozen sigroup account")
        _positive_integer(
            self.requested_walltime_seconds,
            name="requested_walltime_seconds",
        )
        _positive_integer(self.captured_monotonic_ns, name="captured_monotonic_ns")
        expected = _canonical_sha256(self._identity_payload())
        if self.runtime_sha256 != expected:
            raise ValueError("Slurm runtime SHA does not match its environment payload")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "design_sha256": self.design_sha256,
            "admission_sha256": self.admission_sha256,
            "scheduler_segment_id": self.scheduler_segment_id,
            "segment_index": self.segment_index,
            "task_id": self.task_id,
            "seed": self.seed,
            "cluster": self.cluster,
            "job_id": self.job_id,
            "array_job_id": self.array_job_id,
            "array_task_id": self.array_task_id,
            "account": self.account,
            "partition": self.partition,
            "requested_walltime_seconds": self.requested_walltime_seconds,
            "captured_monotonic_ns": self.captured_monotonic_ns,
        }

    def validate_integrity(self) -> None:
        if getattr(self, "_seal", None) is not _RUNTIME_FACTORY_TOKEN:
            raise TypeError("Slurm runtime is not sealed by process capture")
        self._validate_structure()

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "runtime_sha256": self.runtime_sha256}


def capture_slurm_segment_runtime(
    admission: PrimarySegmentAdmission,
    *,
    requested_walltime_seconds: int,
) -> SlurmSegmentRuntime:
    """Capture the exact Slurm identity; no mapping or ``formal`` flag is accepted."""

    admitted = _require_admission(admission)
    walltime = _positive_integer(
        requested_walltime_seconds,
        name="requested_walltime_seconds",
    )
    required_environment = {
        "cluster": "SLURM_CLUSTER_NAME",
        "job_id": "SLURM_JOB_ID",
        "array_job_id": "SLURM_ARRAY_JOB_ID",
        "array_task_id": "SLURM_ARRAY_TASK_ID",
        "account": "SLURM_JOB_ACCOUNT",
        "partition": "SLURM_JOB_PARTITION",
    }
    observed: dict[str, str] = {}
    for name, variable in required_environment.items():
        value = os.environ.get(variable)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"formal R3 execution requires {variable}")
        observed[name] = value
    try:
        array_task_id = int(observed["array_task_id"])
    except ValueError as error:
        raise ValueError("SLURM_ARRAY_TASK_ID must be an integer") from error
    if array_task_id != admitted.task_id:
        raise ValueError("Slurm array task differs from the admitted R3 task")
    monotonic_ns = time.monotonic_ns()
    payload = {
        "schema_version": "phase2-recovery-r3-slurm-segment-runtime/v1",
        "design_sha256": admitted.design.design_sha256,
        "admission_sha256": admitted.admission_sha256,
        "scheduler_segment_id": admitted.scheduler_segment_id,
        "segment_index": admitted.segment_index,
        "task_id": admitted.task_id,
        "seed": admitted.seed,
        "cluster": observed["cluster"],
        "job_id": observed["job_id"],
        "array_job_id": observed["array_job_id"],
        "array_task_id": array_task_id,
        "account": observed["account"],
        "partition": observed["partition"],
        "requested_walltime_seconds": walltime,
        "captured_monotonic_ns": monotonic_ns,
    }
    result = SlurmSegmentRuntime(
        **payload,
        runtime_sha256=_canonical_sha256(payload),
        _factory_token=_RUNTIME_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _runtime_from_terminal_identity(value: object) -> SlurmSegmentRuntime:
    required = {
        "schema_version",
        "design_sha256",
        "admission_sha256",
        "scheduler_segment_id",
        "segment_index",
        "task_id",
        "seed",
        "cluster",
        "job_id",
        "array_job_id",
        "array_task_id",
        "account",
        "partition",
        "requested_walltime_seconds",
        "captured_monotonic_ns",
        "runtime_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("selected terminal runtime identity fields are invalid")
    arguments = dict(value)
    runtime = SlurmSegmentRuntime(
        **arguments,  # type: ignore[arg-type]
        _factory_token=_RUNTIME_FACTORY_TOKEN,
    )
    runtime.validate_integrity()
    return runtime


def formal_primary_checkpoint_binding(
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> dict[str, object]:
    """Return the stable logical-head binding shared by all admitted segments."""

    admitted = _require_admission(admission)
    method = _learner(learner)
    head_index = R3_PRIMARY_HEADS.index(method)
    return {
        "schema_version": FORMAL_PRIMARY_CHECKPOINT_BINDING_SCHEMA,
        "campaign_kind": admitted.design.campaign_kind,
        "execution_revision": admitted.design.execution_revision,
        "campaign_role": admitted.design.campaign_role,
        "execution_role": FORMAL_PRIMARY_EXECUTION_ROLE,
        "design_sha256": admitted.design.design_sha256,
        "science_semantic_sha256": admitted.design.science.semantic_sha256,
        "science_file_sha256": admitted.design.science.file_sha256,
        "materialization_attestation_sha256": (admitted.materialization.attestation_sha256),
        "logical_run_id": admitted.logical_run_id,
        "head_run_id": admitted.head_run_ids[head_index],
        "task_id": admitted.task_id,
        "seed": admitted.seed,
        "objective": method,
        "checkpoint_policy_sha256": admitted.design.checkpoint_policy_sha256,
        "progress_policy_sha256": admitted.design.progress_policy_sha256,
        "signal_policy_sha256": admitted.design.signal_policy_sha256,
        "continuation_policy_sha256": admitted.design.continuation_policy_sha256,
        "max_scheduler_segments": admitted.design.max_scheduler_segments,
        "active_named_rng_states": [],
        "information_boundary": FORMAL_PRIMARY_INFORMATION_BOUNDARY,
        "formal_r3_evidence": True,
    }


def r3_operational_policy_hashes(
    operational_bundle: VerifiedGatePOperationalBundle,
) -> dict[str, str]:
    """Derive the four execution-policy hashes consumed by R3PrimaryDesign."""

    if type(operational_bundle) is not VerifiedGatePOperationalBundle:
        raise TypeError("operational_bundle must be an exact VerifiedGatePOperationalBundle")
    operational_bundle.validate_integrity()
    plan = operational_bundle
    shared = {
        "resource_plan_sha256": plan.resource_plan_sha256,
        "information_boundary": FORMAL_PRIMARY_INFORMATION_BOUNDARY,
    }
    return {
        "checkpoint_policy_sha256": _canonical_sha256(
            {
                "schema_version": "phase2-recovery-r3-checkpoint-policy/v1",
                **shared,
                "state_schema": CHECKPOINT_SCHEMA,
                "durable_checkpoint_cadence_updates": (plan.durable_checkpoint_cadence_updates),
                "mandatory_checkpoint_updates": list(plan.mandatory_checkpoint_updates),
                "mandatory_checkpoint_roles": list(plan.mandatory_checkpoint_roles),
                "atomic_no_overwrite_fsync": True,
            }
        ),
        "progress_policy_sha256": _canonical_sha256(
            {
                "schema_version": "phase2-recovery-r3-progress-policy/v1",
                **shared,
                "details_schema": TRAINING_PROGRESS_DETAILS_SCHEMA,
                "audit_cadence_updates": plan.audit_cadence_updates,
                "publish_after_every_audit_or_checkpoint": True,
                "hash_chain_no_overwrite": True,
            }
        ),
        "signal_policy_sha256": _canonical_sha256(
            {
                "schema_version": "phase2-recovery-r3-signal-policy/v1",
                **shared,
                "receipt_schema": SIGNAL_RECEIPT_SCHEMA,
                "advance_signal_lead_seconds": (plan.advance_signal_lead_seconds),
                "handled_signals": ["USR1", "TERM", "INT"],
                "safe_boundary_checkpoint_required": True,
                "terminal_success_claimed": False,
            }
        ),
        "continuation_policy_sha256": _canonical_sha256(
            {
                "schema_version": "phase2-recovery-r3-continuation-policy/v1",
                **shared,
                "max_scheduler_segments": plan.max_scheduler_segments,
                "segment_boundaries": plan.to_dict()["resource_plan"]["segment_boundaries"],
                "same_logical_run_only": True,
                "fresh_restart_forbidden": True,
                "discrete_replay_exact": True,
                "numeric_replay_relative_tolerance": 1.0e-10,
                "numeric_replay_absolute_tolerance": 1.0e-14,
            }
        ),
    }


def _validate_primary_resource_plan(
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    resource_plan: object,
) -> VerifiedGatePOperationalBundle:
    if type(resource_plan) is not VerifiedGatePOperationalBundle:
        raise TypeError("resource_plan must be an exact VerifiedGatePOperationalBundle")
    resource_plan.validate_integrity()
    plan = resource_plan
    if resource_plan_artifact_ref(plan) != admission.design.profile_authorization.resource_plan:
        raise ValueError("resource plan is not the Gate-P authorized artifact")
    if (
        formal_profile_artifact_ref(plan)
        != admission.design.profile_authorization.formal_cuda_profile_result
    ):
        raise ValueError("resource plan formal profile differs from Gate-P authorization")
    expected_policy_hashes = r3_operational_policy_hashes(plan)
    for name, expected in expected_policy_hashes.items():
        if getattr(admission.design, name) != expected:
            raise ValueError(f"R3 design {name} differs from the Gate-P plan")
    if (
        admission.design.max_scheduler_segments != plan.max_scheduler_segments
        or admission.segment_index > plan.max_scheduler_segments
    ):
        raise ValueError("R3 segment count differs from the Gate-P resource plan")
    if (
        runtime.account != plan.slurm_account
        or runtime.partition != plan.partition
        or runtime.requested_walltime_seconds != plan.requested_walltime_seconds_per_segment
    ):
        raise ValueError("live Slurm runtime differs from the Gate-P resource plan")
    if plan.gpus_per_task != 1:
        raise ValueError("formal R3 primary requires exactly one GPU per task")
    return plan


def _validate_head_execution_slice(
    value: object,
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    plan: VerifiedGatePOperationalBundle,
    learner: PrimaryLearner,
) -> dict[str, object]:
    """Validate the orchestrator's closed, self-hashed per-head budget slice."""

    required = {
        "schema_version",
        "resource_plan_sha256",
        "formal_profile_sha256",
        "profile_run_sha256",
        "design_sha256",
        "admission_sha256",
        "logical_run_id",
        "head_run_id",
        "scheduler_segment_id",
        "runtime_sha256",
        "segment_index",
        "task_id",
        "seed",
        "head",
        "fresh_or_resume",
        "science_audit_cadence_updates",
        "maximum_updates_per_head",
        "max_safe_update_blocks_to_execute",
        "safe_update_blocks_consumed_before_head",
        "safe_update_blocks_available_to_head",
        "start_completed_updates",
        "end_completed_updates_inclusive",
        "nominal_segment_start_cursor",
        "nominal_segment_start_cursor_sha256",
        "nominal_segment_end_cursor",
        "nominal_segment_end_cursor_sha256",
        "actual_cursor_before_head",
        "actual_cursor_before_head_sha256",
        "predecessor_checkpoint",
        "information_boundary",
        "slice_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("head execution slice fields are invalid")
    result = json.loads(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    slice_sha = result.pop("slice_sha256")
    _training._validate_digest(slice_sha, name="head execution slice SHA")
    if slice_sha != _orchestrator_evidence_sha256(result):
        raise ValueError("head execution slice self-hash is invalid")
    result["slice_sha256"] = slice_sha
    head_run_id = admission.head_run_ids[R3_PRIMARY_HEADS.index(learner)]
    exact = {
        "schema_version": HEAD_EXECUTION_SLICE_SCHEMA,
        "resource_plan_sha256": plan.resource_plan_sha256,
        "formal_profile_sha256": plan.formal_profile_sha256,
        "profile_run_sha256": plan.profile_run_sha256,
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "logical_run_id": admission.logical_run_id,
        "head_run_id": head_run_id,
        "scheduler_segment_id": admission.scheduler_segment_id,
        "runtime_sha256": runtime.runtime_sha256,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "head": learner,
        "science_audit_cadence_updates": (
            admission.design.science.settings.convergence.check_interval
        ),
        "maximum_updates_per_head": (admission.design.science.settings.convergence.max_steps),
        "information_boundary": ("operational_cursor_only_no_scientific_adaptation"),
    }
    for name, expected in exact.items():
        if result.get(name) != expected:
            raise ValueError(f"head execution slice {name} is invalid")

    segment = plan.segment_boundaries[admission.segment_index - 1]
    if not isinstance(segment, Mapping):
        raise TypeError("Gate-P segment boundary must be a mapping")
    cursor_keys = {
        "global_safe_block",
        "bt_mle_completed_updates",
        "prorm_plus_completed_updates",
        "next_head",
    }
    for position, plan_name in (
        ("start", "start_boundary"),
        ("end", "end_boundary"),
    ):
        cursor = result[f"nominal_segment_{position}_cursor"]
        if (
            not isinstance(cursor, Mapping)
            or set(cursor) != cursor_keys
            or dict(cursor) != dict(segment[plan_name])  # type: ignore[arg-type]
        ):
            raise ValueError(f"head execution slice nominal {position} cursor is invalid")
        if result[f"nominal_segment_{position}_cursor_sha256"] != _orchestrator_evidence_sha256(
            cursor
        ):
            raise ValueError(f"head execution slice nominal {position} cursor hash is invalid")

    cadence = int(exact["science_audit_cadence_updates"])
    maximum = int(exact["maximum_updates_per_head"])
    total_blocks = _positive_integer(
        result["max_safe_update_blocks_to_execute"],
        name="slice max_safe_update_blocks_to_execute",
    )
    if total_blocks != segment.get("max_safe_update_blocks_to_execute"):
        raise ValueError("head execution slice segment block budget is invalid")
    consumed = _nonnegative_integer(
        result["safe_update_blocks_consumed_before_head"],
        name="slice safe_update_blocks_consumed_before_head",
    )
    available = _positive_integer(
        result["safe_update_blocks_available_to_head"],
        name="slice safe_update_blocks_available_to_head",
    )
    if consumed >= total_blocks:
        raise ValueError("head execution slice begins after its segment budget")
    start = _nonnegative_integer(
        result["start_completed_updates"],
        name="slice start_completed_updates",
    )
    end = _positive_integer(
        result["end_completed_updates_inclusive"],
        name="slice end_completed_updates_inclusive",
    )
    if start > maximum or start % cadence:
        raise ValueError("head execution slice start is outside the science grid")
    expected_available = min(
        total_blocks - consumed,
        (maximum - start) // cadence,
    )
    if available != expected_available or end != start + available * cadence or end > maximum:
        raise ValueError("head execution slice available update budget is invalid")

    actual_keys = {
        "bt_mle_completed_updates",
        "bt_mle_complete",
        "prorm_plus_completed_updates",
        "prorm_plus_complete",
        "next_head",
    }
    actual = result["actual_cursor_before_head"]
    if not isinstance(actual, Mapping) or set(actual) != actual_keys:
        raise ValueError("head execution slice actual cursor fields are invalid")
    if result["actual_cursor_before_head_sha256"] != _orchestrator_evidence_sha256(actual):
        raise ValueError("head execution slice actual cursor hash is invalid")
    bt_updates = _nonnegative_integer(
        actual["bt_mle_completed_updates"],
        name="actual cursor BT updates",
    )
    prorm_updates = _nonnegative_integer(
        actual["prorm_plus_completed_updates"],
        name="actual cursor ProRM+ updates",
    )
    bt_complete = actual["bt_mle_complete"]
    prorm_complete = actual["prorm_plus_complete"]
    if (
        type(bt_complete) is not bool
        or type(prorm_complete) is not bool
        or bt_updates > maximum
        or prorm_updates > maximum
        or bt_updates % cadence
        or prorm_updates % cadence
    ):
        raise ValueError("head execution slice actual cursor is invalid")
    if not bt_complete:
        if prorm_updates != 0 or prorm_complete:
            raise ValueError("head execution slice violates strict BT-to-ProRM order")
        expected_next: PrimaryLearner | None = "bt_mle"
    elif not prorm_complete:
        expected_next = "prorm_plus"
    else:
        expected_next = None
    if actual["next_head"] != expected_next or expected_next != learner:
        raise ValueError("head execution slice does not address the actual next head")
    actual_head_updates = bt_updates if learner == BT_MLE else prorm_updates
    if actual_head_updates != start:
        raise ValueError("head execution slice start differs from the actual cursor")

    mode = result["fresh_or_resume"]
    predecessor = result["predecessor_checkpoint"]
    if mode == "fresh":
        if start != 0 or predecessor is not None:
            raise ValueError(
                "fresh head execution slice must start at zero without predecessor state"
            )
        if learner == PRORM_PLUS and bt_complete is not True:
            raise ValueError("fresh ProRM+ requires durable BT completion")
    elif mode == "resume":
        if start == 0 or not isinstance(predecessor, Mapping):
            raise ValueError("resumed head execution slice requires nonzero predecessor state")
        if set(predecessor) != {"schema_version", "artifact_sha256", "role"}:
            raise ValueError("slice predecessor checkpoint fields are invalid")
        evidence = admission.continuation_evidence
        if evidence is None or dict(predecessor) != evidence.verified_checkpoint.to_dict():
            raise ValueError("slice predecessor checkpoint differs from continuation admission")
    else:
        raise ValueError("head execution slice mode is invalid")
    return result


def _validate_store(
    store: DurableCheckpointStore,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> None:
    if type(store) is not DurableCheckpointStore:
        raise TypeError("checkpoint_store must be an exact DurableCheckpointStore")
    expected = formal_primary_checkpoint_binding(admission, learner)
    if store.objective != learner or store.binding != expected:
        raise ValueError("checkpoint store does not match the admitted R3 logical head")


def _formal_checkpoint_payload(
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    learner: PrimaryLearner,
    head_execution_slice_sha256: str,
    controller_checkpoint: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(controller_checkpoint, Mapping):
        raise TypeError("controller_checkpoint must be a mapping")
    # Tensor-bearing controller state cannot be JSON encoded.  Its existing
    # state-complete digest is therefore bound separately while the exact
    # mapping itself remains in the torch checkpoint envelope.
    controller_sha = controller_checkpoint.get("checkpoint_sha256")
    _training._validate_digest(
        controller_sha,
        name="controller_checkpoint.checkpoint_sha256",
    )
    _training._validate_digest(
        head_execution_slice_sha256,
        name="head_execution_slice_sha256",
    )
    return {
        "schema_version": FORMAL_PRIMARY_CHECKPOINT_PAYLOAD_SCHEMA,
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "logical_run_id": admission.logical_run_id,
        "head_run_id": admission.head_run_ids[R3_PRIMARY_HEADS.index(learner)],
        "scheduler_segment_id": admission.scheduler_segment_id,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "objective": learner,
        "runtime_sha256": runtime.runtime_sha256,
        "head_execution_slice_sha256": head_execution_slice_sha256,
        "controller_checkpoint_sha256": controller_sha,
        "controller_checkpoint": dict(controller_checkpoint),
        "information_boundary": FORMAL_PRIMARY_INFORMATION_BOUNDARY,
    }


def _validate_formal_checkpoint_payload(
    value: object,
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    learner: PrimaryLearner,
    allow_predecessor_segment: bool,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("formal checkpoint payload must be a mapping")
    required = {
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
    if set(value) != required:
        raise ValueError("formal checkpoint payload fields are invalid")
    segment_index = value["segment_index"]
    expected_segment = (
        admission.segment_index - 1 if allow_predecessor_segment else admission.segment_index
    )
    if segment_index != expected_segment:
        raise ValueError("formal checkpoint belongs to the wrong scheduler segment")
    if allow_predecessor_segment:
        evidence = admission.continuation_evidence
        if evidence is None:
            raise ValueError("predecessor checkpoint requires continuation evidence")
        predecessor = evidence.predecessor
        expected_admission_sha = predecessor.admission_sha256
        expected_scheduler_segment_id = predecessor.scheduler_segment_id
        # The predecessor runtime is externally terminal-validated.  Its exact
        # runtime digest is retained in the checkpoint and is not replaced by
        # the current segment's digest.
        expected_runtime_sha: str | None = None
    else:
        expected_admission_sha = admission.admission_sha256
        expected_scheduler_segment_id = admission.scheduler_segment_id
        expected_runtime_sha = runtime.runtime_sha256
    head_run_id = admission.head_run_ids[R3_PRIMARY_HEADS.index(learner)]
    exact = {
        "schema_version": FORMAL_PRIMARY_CHECKPOINT_PAYLOAD_SCHEMA,
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": expected_admission_sha,
        "logical_run_id": admission.logical_run_id,
        "head_run_id": head_run_id,
        "scheduler_segment_id": expected_scheduler_segment_id,
        "segment_index": expected_segment,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "objective": learner,
        "information_boundary": FORMAL_PRIMARY_INFORMATION_BOUNDARY,
    }
    for name, expected in exact.items():
        if value.get(name) != expected:
            raise ValueError(f"formal checkpoint {name} does not match admission")
    _training._validate_digest(
        value.get("runtime_sha256"),
        name="formal_checkpoint.runtime_sha256",
    )
    _training._validate_digest(
        value.get("head_execution_slice_sha256"),
        name="formal_checkpoint.head_execution_slice_sha256",
    )
    if expected_runtime_sha is not None and value["runtime_sha256"] != expected_runtime_sha:
        raise ValueError("formal checkpoint runtime differs from current scheduler segment")
    controller = value.get("controller_checkpoint")
    if not isinstance(controller, Mapping):
        raise TypeError("formal checkpoint lacks controller_checkpoint")
    controller_sha = controller.get("checkpoint_sha256")
    _training._validate_digest(
        controller_sha,
        name="controller_checkpoint.checkpoint_sha256",
    )
    if value.get("controller_checkpoint_sha256") != controller_sha:
        raise ValueError("formal checkpoint controller digest is inconsistent")
    return controller


def _validate_stored_formal_payload(
    value: object,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("stored formal checkpoint payload must be a mapping")
    required = {
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
    if set(value) != required:
        raise ValueError("stored formal checkpoint payload fields are invalid")
    exact = {
        "schema_version": FORMAL_PRIMARY_CHECKPOINT_PAYLOAD_SCHEMA,
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "logical_run_id": admission.logical_run_id,
        "head_run_id": admission.head_run_ids[R3_PRIMARY_HEADS.index(learner)],
        "scheduler_segment_id": admission.scheduler_segment_id,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "objective": learner,
        "information_boundary": FORMAL_PRIMARY_INFORMATION_BOUNDARY,
    }
    for name, expected in exact.items():
        if value.get(name) != expected:
            raise ValueError(f"stored formal checkpoint {name} is invalid")
    _training._validate_digest(
        value.get("runtime_sha256"),
        name="stored formal checkpoint runtime_sha256",
    )
    _training._validate_digest(
        value.get("head_execution_slice_sha256"),
        name="stored formal checkpoint head_execution_slice_sha256",
    )
    controller = value.get("controller_checkpoint")
    if not isinstance(controller, Mapping):
        raise TypeError("stored formal checkpoint lacks controller state")
    controller_sha = controller.get("checkpoint_sha256")
    _training._validate_digest(
        controller_sha,
        name="stored controller checkpoint SHA",
    )
    if value.get("controller_checkpoint_sha256") != controller_sha:
        raise ValueError("stored controller checkpoint digest is inconsistent")
    context = admission.materialization.context
    rank_diagnostic = (
        context.reward_head_identifiability
        if learner == BT_MLE
        else context.prorm_moment_map_identifiability
    )
    expected_identity = _training._first_order_controller_identity(
        objective_name=learner,
        execution_role=FORMAL_PRIMARY_EXECUTION_ROLE,
        spec=context.settings.convergence,
        fixed_snapshot_steps=context.settings.outer_steps,
        rank_diagnostic=rank_diagnostic,
    )
    _training._validated_first_order_controller_checkpoint(
        controller,
        expected_identity=expected_identity,
        spec=context.settings.convergence,
        protocol=context.settings.convergence.optimizer_protocol,
        fixed_snapshot_steps=context.settings.outer_steps,
    )
    return controller_sha


def _formal_selected_terminal_payload(
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    learner: PrimaryLearner,
    head_execution_slice_sha256: str,
    selected_terminal_checkpoint: Mapping[str, object],
    terminal_result_core: Mapping[str, object],
) -> dict[str, object]:
    terminal_sha = selected_terminal_checkpoint.get("terminal_checkpoint_sha256")
    _training._validate_digest(
        terminal_sha,
        name="selected_terminal_checkpoint.terminal_checkpoint_sha256",
    )
    _training._validate_digest(
        head_execution_slice_sha256,
        name="head_execution_slice_sha256",
    )
    core = json.loads(
        json.dumps(
            dict(terminal_result_core),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return {
        "schema_version": FORMAL_PRIMARY_SELECTED_TERMINAL_PAYLOAD_SCHEMA,
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "logical_run_id": admission.logical_run_id,
        "head_run_id": admission.head_run_ids[R3_PRIMARY_HEADS.index(learner)],
        "scheduler_segment_id": admission.scheduler_segment_id,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "objective": learner,
        "runtime_sha256": runtime.runtime_sha256,
        "runtime_identity": runtime.to_dict(),
        "head_execution_slice_sha256": head_execution_slice_sha256,
        "selected_terminal_checkpoint_sha256": terminal_sha,
        "selected_terminal_checkpoint": dict(selected_terminal_checkpoint),
        "terminal_result_core": core,
        "terminal_result_core_sha256": _canonical_sha256(core),
        "information_boundary": FORMAL_PRIMARY_INFORMATION_BOUNDARY,
    }


def _validate_selected_terminal_payload(
    value: object,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
    runtime: SlurmSegmentRuntime | None,
    expected_head_execution_slice_sha256: str | None,
) -> str:
    required = {
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
        "runtime_identity",
        "head_execution_slice_sha256",
        "selected_terminal_checkpoint_sha256",
        "selected_terminal_checkpoint",
        "terminal_result_core",
        "terminal_result_core_sha256",
        "information_boundary",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("formal selected terminal payload fields are invalid")
    exact = {
        "schema_version": FORMAL_PRIMARY_SELECTED_TERMINAL_PAYLOAD_SCHEMA,
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "logical_run_id": admission.logical_run_id,
        "head_run_id": admission.head_run_ids[R3_PRIMARY_HEADS.index(learner)],
        "scheduler_segment_id": admission.scheduler_segment_id,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "objective": learner,
        "information_boundary": FORMAL_PRIMARY_INFORMATION_BOUNDARY,
    }
    for name, expected in exact.items():
        if value.get(name) != expected:
            raise ValueError(f"formal selected terminal {name} is invalid")
    _training._validate_digest(
        value.get("runtime_sha256"),
        name="formal selected terminal runtime_sha256",
    )
    if runtime is not None and value["runtime_sha256"] != runtime.runtime_sha256:
        raise ValueError("formal selected terminal runtime differs from this segment")
    stored_runtime = _runtime_from_terminal_identity(value.get("runtime_identity"))
    if (
        stored_runtime.runtime_sha256 != value["runtime_sha256"]
        or stored_runtime.admission_sha256 != admission.admission_sha256
        or (runtime is not None and stored_runtime.to_dict() != runtime.to_dict())
    ):
        raise ValueError("formal selected terminal runtime identity is inconsistent")
    slice_sha = value.get("head_execution_slice_sha256")
    _training._validate_digest(
        slice_sha,
        name="formal selected terminal head_execution_slice_sha256",
    )
    if (
        expected_head_execution_slice_sha256 is not None
        and slice_sha != expected_head_execution_slice_sha256
    ):
        raise ValueError("formal selected terminal slice binding is invalid")
    terminal = value.get("selected_terminal_checkpoint")
    if not isinstance(terminal, Mapping):
        raise TypeError("formal selected terminal checkpoint must be a mapping")
    context = admission.materialization.context
    rank_diagnostic = (
        context.reward_head_identifiability
        if learner == BT_MLE
        else context.prorm_moment_map_identifiability
    )
    identity = _training._first_order_controller_identity(
        objective_name=learner,
        execution_role=FORMAL_PRIMARY_EXECUTION_ROLE,
        spec=context.settings.convergence,
        fixed_snapshot_steps=context.settings.outer_steps,
        rank_diagnostic=rank_diagnostic,
    )
    validated = _training._validated_selected_primary_terminal_checkpoint(
        terminal,
        expected_identity=identity,
    )
    terminal_sha = validated["terminal_checkpoint_sha256"]
    if value.get("selected_terminal_checkpoint_sha256") != terminal_sha:
        raise ValueError("formal selected terminal checkpoint digest is inconsistent")
    core = value.get("terminal_result_core")
    if not isinstance(core, Mapping):
        raise TypeError("formal selected terminal result core must be a mapping")
    if _training._validate_digest(
        value.get("terminal_result_core_sha256"),
        name="formal selected terminal result-core hash",
    ) != _canonical_sha256(core):
        raise ValueError("formal selected terminal result-core hash is inconsistent")
    _validate_terminal_result_core(
        core,
        admission=admission,
        runtime=stored_runtime,
        learner=learner,
        selected_terminal_checkpoint=validated,
        expected_head_execution_slice_sha256=(expected_head_execution_slice_sha256),
    )
    return terminal_sha  # type: ignore[return-value]


def _latest_generation_artifact_sha256(
    store: DurableCheckpointStore,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
    continuation_required: bool,
) -> str:
    audited = store.audit_generations(verify_all_checkpoint_bytes=True)
    if not audited:
        raise RuntimeError("continuation checkpoint store has no committed generation")
    latest = audited[-1]
    generation = latest.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ValueError("checkpoint generation is invalid")
    generation_path = store.generations_path / f"generation-{generation:08d}"
    metadata_sha = _sha256_file(generation_path / "metadata.json")
    state_sha = _sha256_file(generation_path / "state.pt")
    continuation_receipt_kind: str | None = None
    continuation_receipt_sha256: str | None = None
    if continuation_required:
        if admission.segment_index >= admission.design.max_scheduler_segments:
            raise RuntimeError("the frozen final segment cannot issue continuation state")
        latest_progress_sha = store.latest_progress_sha256()
        candidates: list[tuple[str, Path, Mapping[str, object]]] = []
        for kind, directory, schema in (
            ("scheduler_signal", store.signal_directory, SIGNAL_RECEIPT_SCHEMA),
            (
                "planned_segment_boundary",
                store.planned_boundary_directory,
                PLANNED_BOUNDARY_RECEIPT_SCHEMA,
            ),
        ):
            paths = sorted(directory.glob("event-*.json"))
            if not paths:
                continue
            path = paths[-1]
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{kind} receipt must be a regular non-symlink file")
            with path.open("r", encoding="utf-8") as stream:
                receipt = json.load(stream)
            if not isinstance(receipt, Mapping):
                raise TypeError(f"{kind} receipt must be a mapping")
            if (
                receipt.get("schema_version") == schema
                and receipt.get("checkpoint_metadata_sha256") == metadata_sha
                and receipt.get("last_progress_sha256") == latest_progress_sha
            ):
                candidates.append((kind, path, receipt))
        if len(candidates) != 1:
            raise RuntimeError(
                "continuation requires exactly one latest-generation signal "
                "or planned-boundary receipt"
            )
        continuation_receipt_kind, receipt_path, receipt = candidates[0]
        scheduler = receipt.get("scheduler_identity")
        if not isinstance(scheduler, Mapping):
            raise TypeError("continuation receipt lacks scheduler identity")
        exact_receipt_fields = {
            "objective": learner,
            "binding": store.binding,
            "binding_sha256": store.binding_sha256,
            "completed_steps": latest.get("completed_steps"),
            "checkpoint_metadata_sha256": metadata_sha,
            "checkpoint_verified": True,
            "continuation_checkpoint_usable": True,
            "planned_action": "continue_same_logical_run",
            "terminal_success_claimed": False,
        }
        if continuation_receipt_kind == "scheduler_signal":
            exact_receipt_fields.update(
                {
                    "reached_safe_boundary": True,
                    "checkpoint_flush_succeeded": True,
                }
            )
        else:
            if receipt.get("update_blocks_remaining") != 0:
                raise ValueError("planned-boundary receipt did not exhaust its frozen slice")
            _training._validate_digest(
                receipt.get("execution_slice_sha256"),
                name="planned-boundary execution_slice_sha256",
            )
        for name, expected in exact_receipt_fields.items():
            if receipt.get(name) != expected:
                raise ValueError(f"continuation receipt {name} is invalid")
        if (
            scheduler.get("design_sha256") != admission.design.design_sha256
            or scheduler.get("admission_sha256") != admission.admission_sha256
            or scheduler.get("scheduler_segment_id") != admission.scheduler_segment_id
            or scheduler.get("segment_index") != admission.segment_index
            or scheduler.get("task_id") != admission.task_id
            or scheduler.get("seed") != admission.seed
        ):
            raise ValueError("continuation receipt scheduler identity differs from admission")
        continuation_receipt_sha256 = _sha256_file(receipt_path)
    envelope = torch.load(
        generation_path / "state.pt",
        weights_only=True,
    )
    if not isinstance(envelope, Mapping) or not isinstance(
        envelope.get("payload"),
        Mapping,
    ):
        raise ValueError("formal continuation checkpoint envelope is malformed")
    payload = envelope["payload"]
    if continuation_required:
        state_payload_kind = "resumable_controller"
        state_payload_sha256 = _validate_stored_formal_payload(
            payload,
            admission=admission,
            learner=learner,
        )
    else:
        state_payload_kind = "selected_primary_terminal"
        state_payload_sha256 = _validate_selected_terminal_payload(
            payload,
            admission=admission,
            learner=learner,
            runtime=None,
            expected_head_execution_slice_sha256=None,
        )
    return _canonical_sha256(
        {
            "schema_version": (
                VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA
                if continuation_required
                else FORMAL_PRIMARY_SELECTED_TERMINAL_ARTIFACT_SCHEMA
            ),
            "role": (
                VERIFIED_CONTINUATION_CHECKPOINT_ROLE
                if continuation_required
                else FORMAL_PRIMARY_SELECTED_TERMINAL_ARTIFACT_ROLE
            ),
            "binding_sha256": store.binding_sha256,
            "generation": generation,
            "metadata_sha256": metadata_sha,
            "state_sha256": state_sha,
            "completed_steps": latest.get("completed_steps"),
            "state_payload_kind": state_payload_kind,
            "state_payload_sha256": state_payload_sha256,
            "segment_index": admission.segment_index,
            "continuation_receipt_kind": continuation_receipt_kind,
            "continuation_receipt_sha256": continuation_receipt_sha256,
            "continuation_required": continuation_required,
        }
    )


def continuation_checkpoint_artifact_ref(
    store: DurableCheckpointStore,
    *,
    predecessor: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> ArtifactRef:
    """Audit and content-address the latest state-complete predecessor checkpoint."""

    admitted = _require_admission(predecessor)
    method = _learner(learner)
    _validate_store(store, admission=admitted, learner=method)
    return ArtifactRef(
        schema_version=VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
        artifact_sha256=_latest_generation_artifact_sha256(
            store,
            admission=admitted,
            learner=method,
            continuation_required=True,
        ),
        role=VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    )


def recover_formal_result_from_selected_terminal(
    store: DurableCheckpointStore,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> FormalR3PrimaryHeadResult:
    """Reconstruct a result after terminal commit but before result publication.

    The selected terminal generation contains a self-hashed result core but
    deliberately omits its own generation artifact digest.  Recovery audits
    that generation, validates every inner cross-link, computes the independent
    terminal artifact digest, and deterministically supplies the two remaining
    result hashes.
    """

    current = _require_admission(admission)
    method = _learner(learner)
    _validate_store(store, admission=current, learner=method)
    audited = store.audit_generations(verify_all_checkpoint_bytes=True)
    if not audited:
        raise RuntimeError("selected terminal recovery requires a checkpoint")
    latest = audited[-1]
    generation = latest.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ValueError("selected terminal generation is invalid")
    generation_path = store.generations_path / f"generation-{generation:08d}"
    envelope = torch.load(generation_path / "state.pt", weights_only=True)
    if not isinstance(envelope, Mapping) or not isinstance(
        envelope.get("payload"),
        Mapping,
    ):
        raise ValueError("selected terminal envelope is malformed")
    outer = envelope["payload"]
    if outer.get("schema_version") != FORMAL_PRIMARY_SELECTED_TERMINAL_PAYLOAD_SCHEMA:
        raise ValueError("latest generation is not a selected-primary terminal")
    segment_index = outer.get("segment_index")
    admission_sha = outer.get("admission_sha256")
    if (
        isinstance(segment_index, bool)
        or not isinstance(segment_index, int)
        or segment_index < 1
        or not isinstance(admission_sha, str)
    ):
        raise ValueError("selected terminal completion admission is malformed")
    completion = current
    while completion.segment_index > segment_index:
        evidence = completion.continuation_evidence
        if evidence is None:
            raise ValueError("selected terminal admission chain is incomplete")
        completion = evidence.predecessor
        completion.validate_integrity()
    if completion.segment_index != segment_index or completion.admission_sha256 != admission_sha:
        raise ValueError("selected terminal belongs to an unadmitted segment")
    _validate_selected_terminal_payload(
        outer,
        admission=completion,
        learner=method,
        runtime=None,
        expected_head_execution_slice_sha256=None,
    )
    runtime = _runtime_from_terminal_identity(outer["runtime_identity"])
    selected_terminal = outer["selected_terminal_checkpoint"]
    core = outer["terminal_result_core"]
    if not isinstance(selected_terminal, Mapping) or not isinstance(core, Mapping):
        raise TypeError("selected terminal recovery payload is malformed")
    (
        head,
        selected_step,
        controller_updates,
        slice_sha,
        resumed,
    ) = _validate_terminal_result_core(
        core,
        admission=completion,
        runtime=runtime,
        learner=method,
        selected_terminal_checkpoint=selected_terminal,
        expected_head_execution_slice_sha256=outer["head_execution_slice_sha256"],  # type: ignore[arg-type]
    )
    terminal_artifact_sha = _latest_generation_artifact_sha256(
        store,
        admission=completion,
        learner=method,
        continuation_required=False,
    )
    payload = _result_payload(
        admission=completion,
        runtime=runtime,
        learner=method,
        context=completion.materialization.context,
        head=head,
        selected_primary_step=selected_step,
        controller_updates_executed=controller_updates,
        terminal_checkpoint_artifact_sha256=terminal_artifact_sha,
        head_execution_slice_sha256=slice_sha,
        resumed_from_predecessor=resumed,
    )
    result = FormalR3PrimaryHeadResult(
        admission=completion,
        runtime=runtime,
        learner=method,
        context=completion.materialization.context,
        head=head,
        selected_primary_step=selected_step,
        controller_updates_executed=controller_updates,
        terminal_checkpoint_artifact_sha256=terminal_artifact_sha,
        head_execution_slice_sha256=slice_sha,
        resumed_from_predecessor=resumed,
        result_sha256=_canonical_sha256(payload),
    )
    result.validate_integrity()
    return result


def latest_generation_is_selected_terminal(
    store: DurableCheckpointStore,
    *,
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
) -> bool:
    """Strictly inspect whether the latest generation is a recoverable terminal.

    A non-terminal resumable controller generation returns ``False``.  A
    generation that advertises the selected-terminal schema is fully recovered
    and validated before ``True`` is returned, so malformed terminal state
    never degrades into a resumable-controller path.
    """

    current = _require_admission(admission)
    method = _learner(learner)
    _validate_store(store, admission=current, learner=method)
    audited = store.audit_generations(verify_all_checkpoint_bytes=True)
    if not audited:
        return False
    generation = audited[-1].get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ValueError("latest checkpoint generation is invalid")
    path = store.generations_path / f"generation-{generation:08d}" / "state.pt"
    envelope = torch.load(path, weights_only=True)
    if not isinstance(envelope, Mapping) or not isinstance(
        envelope.get("payload"),
        Mapping,
    ):
        raise ValueError("latest checkpoint envelope is malformed")
    payload = envelope["payload"]
    if payload.get("schema_version") != FORMAL_PRIMARY_SELECTED_TERMINAL_PAYLOAD_SCHEMA:
        return False
    recover_formal_result_from_selected_terminal(
        store,
        admission=current,
        learner=method,
    )
    return True


def _checkpoint_completed_steps(payload: Mapping[str, object]) -> int:
    controller = payload.get("controller_state")
    if not isinstance(controller, Mapping):
        raise ValueError("controller checkpoint is missing controller_state")
    steps = controller.get("completed_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("controller checkpoint completed_steps is invalid")
    return steps


def _controller_updates_executed(
    convergence: _training._ConvergedTrainingRun,
    trainer: object,
) -> int:
    evidence = convergence.evidence
    if not isinstance(evidence, Mapping):
        raise TypeError("convergence evidence must be a mapping")
    candidates = [int(trainer.completed_steps)]
    execution = evidence.get("optimizer_protocol_execution")
    if isinstance(execution, Mapping):
        observed = execution.get("completed_updates_observed")
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise ValueError("optimizer completed_updates_observed is invalid")
        candidates.append(observed)
    for field_name in ("selected_primary_step", "fixed_step_snapshot_steps"):
        value = evidence.get(field_name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"convergence {field_name} is invalid")
            candidates.append(value)
    legacy = evidence.get("legacy_constant_lr_boundary_snapshot")
    if isinstance(legacy, Mapping):
        step = legacy.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("legacy checkpoint step is invalid")
        candidates.append(step)
    return max(candidates)


def _memory_bytes(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        raise RuntimeError("formal R3 primary execution requires CUDA")
    torch.cuda.synchronize(device)
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.max_memory_allocated(device)),
    )


def _signal_state(signal: CheckpointSignal) -> dict[str, object]:
    return {
        "requested": signal.requested,
        "signal_name": signal.signal_name,
        "received_at_utc": signal.received_at_utc,
        "additional_signal_count": signal.additional_signal_count,
    }


def _remaining_allocation_seconds(runtime: SlurmSegmentRuntime) -> float:
    elapsed = (time.monotonic_ns() - runtime.captured_monotonic_ns) / 1.0e9
    return max(0.0, float(runtime.requested_walltime_seconds) - elapsed)


def _terminal_result_core_payload(
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    learner: PrimaryLearner,
    context: NeutralPhase2TrainingContext,
    head: _training.TrainedHeadEvidence,
    selected_primary_step: int,
    controller_updates_executed: int,
    head_execution_slice_sha256: str,
    resumed_from_predecessor: bool,
) -> dict[str, object]:
    return {
        "schema_version": FORMAL_PRIMARY_TERMINAL_RESULT_CORE_SCHEMA,
        "campaign_kind": admission.design.campaign_kind,
        "execution_revision": admission.design.execution_revision,
        "campaign_role": admission.design.campaign_role,
        "execution_role": FORMAL_PRIMARY_EXECUTION_ROLE,
        "design_sha256": admission.design.design_sha256,
        "admission_sha256": admission.admission_sha256,
        "logical_run_id": admission.logical_run_id,
        "head_run_id": admission.head_run_ids[R3_PRIMARY_HEADS.index(learner)],
        "scheduler_segment_id": admission.scheduler_segment_id,
        "runtime_sha256": runtime.runtime_sha256,
        "segment_index": admission.segment_index,
        "task_id": admission.task_id,
        "seed": admission.seed,
        "learner": learner,
        "science_semantic_sha256": admission.design.science.semantic_sha256,
        "science_file_sha256": admission.design.science.file_sha256,
        "materialization_attestation_sha256": (admission.materialization.attestation_sha256),
        "context_sha256": context.context_sha256,
        "input_training_sha256": context.input_training_sha256,
        "prepared_training_sha256": context.primary_training_sha256,
        "oracle_reward_sha256": context.oracle_reward_sha256,
        "label_stream_sha256": context.label_stream.label_stream_sha256,
        "selected_primary_step": selected_primary_step,
        "controller_updates_executed": controller_updates_executed,
        "head": head.to_dict(),
        "head_execution_slice_sha256": head_execution_slice_sha256,
        "resumed_from_predecessor": resumed_from_predecessor,
        "information_boundary": {
            "train_only": True,
            "validation_or_test_data_accessed": False,
            "policy_session_opened": False,
            "policy_rollout_performed": False,
            "beta_outcome_computed": False,
            "controls_executed": False,
        },
        "external_scheduler_terminal_validated": False,
        "formal_r3_evidence": False,
    }


def _trained_head_from_terminal_payload(
    value: object,
    *,
    learner: PrimaryLearner,
) -> _training.TrainedHeadEvidence:
    required = {
        "arm",
        "method",
        "head_weight",
        "head_dtype",
        "initial_head_sha256",
        "head_sha256",
        "initial_objective",
        "final_objective",
        "history_summary",
        "final_pcg",
        "first_order_convergence",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("selected terminal head evidence fields are invalid")
    weights = value["head_weight"]
    if not isinstance(weights, list):
        raise TypeError("selected terminal head_weight must be a list")
    history = value["history_summary"]
    convergence = value["first_order_convergence"]
    final_pcg = value["final_pcg"]
    if not isinstance(history, Mapping) or not isinstance(convergence, Mapping):
        raise TypeError("selected terminal head evidence mappings are malformed")
    if final_pcg is not None and not isinstance(final_pcg, Mapping):
        raise TypeError("selected terminal final_pcg must be a mapping or None")
    result = _training.TrainedHeadEvidence(
        arm=value["arm"],  # type: ignore[arg-type]
        method=value["method"],  # type: ignore[arg-type]
        head_weight=tuple(weights),
        head_dtype=value["head_dtype"],  # type: ignore[arg-type]
        initial_head_sha256=value["initial_head_sha256"],  # type: ignore[arg-type]
        head_sha256=value["head_sha256"],  # type: ignore[arg-type]
        initial_objective=value["initial_objective"],  # type: ignore[arg-type]
        final_objective=value["final_objective"],  # type: ignore[arg-type]
        history_summary=dict(history),
        final_pcg=(None if final_pcg is None else dict(final_pcg)),
        first_order_convergence=dict(convergence),
    )
    if result.arm != _training.PRIMARY_TRAINING_ARM or result.method != learner:
        raise ValueError("selected terminal head is not the primary learner")
    return result


def _validate_terminal_result_core(
    value: object,
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    learner: PrimaryLearner,
    selected_terminal_checkpoint: Mapping[str, object],
    expected_head_execution_slice_sha256: str | None,
) -> tuple[_training.TrainedHeadEvidence, int, int, str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError("selected terminal result core must be a mapping")
    head = _trained_head_from_terminal_payload(
        value.get("head"),
        learner=learner,
    )
    selected_step = _positive_integer(
        value.get("selected_primary_step"),
        name="terminal result-core selected_primary_step",
    )
    controller_updates = _positive_integer(
        value.get("controller_updates_executed"),
        name="terminal result-core controller_updates_executed",
    )
    slice_sha = value.get("head_execution_slice_sha256")
    _training._validate_digest(
        slice_sha,
        name="terminal result-core head execution slice SHA",
    )
    resumed = value.get("resumed_from_predecessor")
    if not isinstance(resumed, bool):
        raise TypeError("terminal result-core resume marker must be bool")
    if (
        selected_terminal_checkpoint.get("selected_primary_step") != selected_step
        or selected_terminal_checkpoint.get("controller_updates_executed") != controller_updates
        or selected_terminal_checkpoint.get("selected_head_sha256") != head.head_sha256
        or selected_terminal_checkpoint.get("convergence_evidence") != head.first_order_convergence
        or (
            expected_head_execution_slice_sha256 is not None
            and slice_sha != expected_head_execution_slice_sha256
        )
    ):
        raise ValueError("terminal result core differs from the selected terminal checkpoint")
    expected = _terminal_result_core_payload(
        admission=admission,
        runtime=runtime,
        learner=learner,
        context=admission.materialization.context,
        head=head,
        selected_primary_step=selected_step,
        controller_updates_executed=controller_updates,
        head_execution_slice_sha256=slice_sha,  # type: ignore[arg-type]
        resumed_from_predecessor=resumed,
    )
    if dict(value) != expected:
        raise ValueError("selected terminal result core identity is invalid")
    return (
        head,
        selected_step,
        controller_updates,
        slice_sha,  # type: ignore[return-value]
        resumed,
    )


def _result_payload(
    *,
    admission: PrimarySegmentAdmission,
    runtime: SlurmSegmentRuntime,
    learner: PrimaryLearner,
    context: NeutralPhase2TrainingContext,
    head: _training.TrainedHeadEvidence,
    selected_primary_step: int,
    controller_updates_executed: int,
    terminal_checkpoint_artifact_sha256: str,
    head_execution_slice_sha256: str,
    resumed_from_predecessor: bool,
) -> dict[str, object]:
    core = _terminal_result_core_payload(
        admission=admission,
        runtime=runtime,
        learner=learner,
        context=context,
        head=head,
        selected_primary_step=selected_primary_step,
        controller_updates_executed=controller_updates_executed,
        head_execution_slice_sha256=head_execution_slice_sha256,
        resumed_from_predecessor=resumed_from_predecessor,
    )
    return {
        **core,
        "schema_version": FORMAL_PRIMARY_HEAD_RESULT_SCHEMA,
        "terminal_checkpoint_artifact_sha256": (terminal_checkpoint_artifact_sha256),
    }


@dataclass(frozen=True, slots=True)
class FormalR3PrimaryHeadResult:
    """Unfinalized head output; external terminal evidence is still mandatory."""

    admission: PrimarySegmentAdmission = field(repr=False, compare=False)
    runtime: SlurmSegmentRuntime
    learner: PrimaryLearner
    context: NeutralPhase2TrainingContext = field(repr=False, compare=False)
    head: _training.TrainedHeadEvidence = field(repr=False)
    selected_primary_step: int
    controller_updates_executed: int
    terminal_checkpoint_artifact_sha256: str
    head_execution_slice_sha256: str
    resumed_from_predecessor: bool
    result_sha256: str

    def __post_init__(self) -> None:
        admission = _require_admission(self.admission)
        self.runtime.validate_integrity()
        method = _learner(self.learner)
        if type(self.context) is not NeutralPhase2TrainingContext:
            raise TypeError("context must be an exact NeutralPhase2TrainingContext")
        self.context.validate_integrity()
        if self.context is not admission.materialization.context:
            raise ValueError("formal result context is not the admitted materialization")
        if type(self.head) is not _training.TrainedHeadEvidence:
            raise TypeError("head must be an exact TrainedHeadEvidence")
        if self.head.method != method or self.head.arm != _training.PRIMARY_TRAINING_ARM:
            raise ValueError("formal result head is not the admitted primary learner")
        convergence = self.head.first_order_convergence
        if (
            convergence.get("schema_version") != "objective-first-order-convergence/v2"
            or convergence.get("converged") is not True
            or convergence.get("test_or_validation_data_accessed") is not False
            or convergence.get("spec") != self.context.settings.convergence.to_dict()
        ):
            raise ValueError("formal result does not contain the frozen R3 convergence proof")
        updates = _positive_integer(
            self.controller_updates_executed,
            name="controller_updates_executed",
        )
        selected_step = _positive_integer(
            self.selected_primary_step,
            name="selected_primary_step",
        )
        if selected_step > updates or convergence.get("selected_primary_step") != selected_step:
            raise ValueError("formal result selected/controller update counts differ")
        _training._validate_digest(
            self.terminal_checkpoint_artifact_sha256,
            name="terminal_checkpoint_artifact_sha256",
        )
        _training._validate_digest(
            self.head_execution_slice_sha256,
            name="head_execution_slice_sha256",
        )
        if not isinstance(self.resumed_from_predecessor, bool):
            raise TypeError("resumed_from_predecessor must be bool")
        if self.resumed_from_predecessor and admission.segment_index == 1:
            raise ValueError("formal result cannot resume in the first segment")
        _training._validate_digest(self.result_sha256, name="result_sha256")
        payload = _result_payload(
            admission=admission,
            runtime=self.runtime,
            learner=method,
            context=self.context,
            head=self.head,
            selected_primary_step=selected_step,
            controller_updates_executed=updates,
            terminal_checkpoint_artifact_sha256=(self.terminal_checkpoint_artifact_sha256),
            head_execution_slice_sha256=self.head_execution_slice_sha256,
            resumed_from_predecessor=self.resumed_from_predecessor,
        )
        if self.result_sha256 != _canonical_sha256(payload):
            raise ValueError("formal primary head result SHA does not match its payload")

    def validate_integrity(self) -> None:
        self.__post_init__()

    def to_dict(self) -> dict[str, object]:
        payload = _result_payload(
            admission=self.admission,
            runtime=self.runtime,
            learner=self.learner,
            context=self.context,
            head=self.head,
            selected_primary_step=self.selected_primary_step,
            controller_updates_executed=self.controller_updates_executed,
            terminal_checkpoint_artifact_sha256=(self.terminal_checkpoint_artifact_sha256),
            head_execution_slice_sha256=self.head_execution_slice_sha256,
            resumed_from_predecessor=self.resumed_from_predecessor,
        )
        return {**payload, "result_sha256": self.result_sha256}


def run_formal_r3_primary_head_segment(
    admission: PrimarySegmentAdmission,
    learner: PrimaryLearner,
    *,
    runtime: SlurmSegmentRuntime,
    resource_plan: VerifiedGatePOperationalBundle,
    checkpoint_store: DurableCheckpointStore,
    checkpoint_signal: CheckpointSignal,
    head_execution_slice: Mapping[str, object],
) -> FormalR3PrimaryHeadResult:
    """Run or resume one formal primary head inside one admitted scheduler segment."""

    admitted = _require_admission(admission)
    method = _learner(learner)
    if type(runtime) is not SlurmSegmentRuntime:
        raise TypeError("runtime must be an exact SlurmSegmentRuntime")
    runtime.validate_integrity()
    if (
        runtime.design_sha256 != admitted.design.design_sha256
        or runtime.admission_sha256 != admitted.admission_sha256
        or runtime.scheduler_segment_id != admitted.scheduler_segment_id
        or runtime.segment_index != admitted.segment_index
        or runtime.task_id != admitted.task_id
        or runtime.seed != admitted.seed
    ):
        raise ValueError("Slurm runtime does not match the admitted segment")
    if type(checkpoint_signal) is not CheckpointSignal:
        raise TypeError("checkpoint_signal must be an exact CheckpointSignal")
    plan = _validate_primary_resource_plan(
        admission=admitted,
        runtime=runtime,
        resource_plan=resource_plan,
    )
    execution_slice = _validate_head_execution_slice(
        head_execution_slice,
        admission=admitted,
        runtime=runtime,
        plan=plan,
        learner=method,
    )
    execution_slice_sha = execution_slice["slice_sha256"]
    _training._validate_digest(
        execution_slice_sha,
        name="head_execution_slice.slice_sha256",
    )
    checkpoint_interval = plan.durable_checkpoint_cadence_updates
    audit_interval = admitted.design.science.settings.convergence.check_interval
    if checkpoint_interval % audit_interval:
        raise ValueError("Gate-P checkpoint cadence must be a positive multiple of 20 updates")
    _validate_store(
        checkpoint_store,
        admission=admitted,
        learner=method,
    )
    if checkpoint_signal.handlers_installed is not True:
        raise RuntimeError(
            "formal R3 primary execution requires installed checkpoint signal handlers"
        )
    context = admitted.materialization.context
    context.validate_integrity()
    device = context.training.reward_features.device
    if (
        device.type != "cuda"
        or context.training.policy_scores.device != device
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("formal R3 primary execution requires one coherent CUDA context")
    if torch.cuda.get_device_name(device) != plan.gpu_name:
        raise RuntimeError("live CUDA device differs from the Gate-P resource plan")

    audited_before = checkpoint_store.audit_generations(verify_all_checkpoint_bytes=True)
    resume_state: Mapping[str, object] | None = None
    resumed = execution_slice["fresh_or_resume"] == "resume"
    if not resumed:
        if audited_before:
            raise RuntimeError("fresh R3 head cannot consume an existing checkpoint")
        checkpoint_store.record_progress(
            status="initialized",
            completed_steps=0,
            details={
                "execution_role": FORMAL_PRIMARY_EXECUTION_ROLE,
                "admission_sha256": admitted.admission_sha256,
                "runtime_sha256": runtime.runtime_sha256,
                "active_named_rng_states": [],
            },
        )
    else:
        if not audited_before or admitted.continuation_evidence is None:
            raise RuntimeError("continuation segment requires committed predecessor state")
        expected_ref = admitted.continuation_evidence.verified_checkpoint
        observed_ref = continuation_checkpoint_artifact_ref(
            checkpoint_store,
            predecessor=admitted.continuation_evidence.predecessor,
            learner=method,
        )
        if observed_ref != expected_ref:
            raise ValueError("continuation capability does not bind this checkpoint store")
        outer_resume = checkpoint_store.load()
        if outer_resume is None:
            raise RuntimeError("continuation checkpoint disappeared after audit")
        resume_state = _validate_formal_checkpoint_payload(
            outer_resume,
            admission=admitted,
            runtime=runtime,
            learner=method,
            allow_predecessor_segment=True,
        )
        if _checkpoint_completed_steps(resume_state) != execution_slice["start_completed_updates"]:
            raise ValueError("resumed checkpoint step differs from the head execution slice")

    trainer = build_primary_core_trainer(context, method)
    initial_head_sha = _training._tensor_sha256(trainer.model.weight)
    cumulative_audit_seconds = 0.0
    cumulative_checkpoint_io_seconds = 0.0
    execution_started_ns = time.monotonic_ns()
    last_checkpoint_metadata_sha256: str | None = (
        None
        if not audited_before
        else _sha256_file(
            checkpoint_store.generations_path
            / f"generation-{len(audited_before):08d}"
            / "metadata.json"
        )
    )

    if method == BT_MLE:

        def raw_audit() -> _training._FirstOrderMeasurement:
            return _training._bt_first_order_measurement(trainer)

        rank_diagnostic = context.reward_head_identifiability
    else:

        def raw_audit() -> _training._FirstOrderMeasurement:
            return _training._prorm_first_order_measurement(trainer)

        rank_diagnostic = context.prorm_moment_map_identifiability

    def timed_audit() -> _training._FirstOrderMeasurement:
        nonlocal cumulative_audit_seconds
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        result = raw_audit()
        torch.cuda.synchronize(device)
        cumulative_audit_seconds += (time.perf_counter_ns() - started) / 1.0e9
        return result

    def checkpoint_hook(
        controller_checkpoint: Mapping[str, object],
        *,
        reason: str,
    ) -> None:
        nonlocal cumulative_checkpoint_io_seconds
        nonlocal last_checkpoint_metadata_sha256
        formal_payload = _formal_checkpoint_payload(
            admission=admitted,
            runtime=runtime,
            learner=method,
            head_execution_slice_sha256=execution_slice_sha,  # type: ignore[arg-type]
            controller_checkpoint=controller_checkpoint,
        )
        started = time.perf_counter_ns()
        manifest = checkpoint_store.save(
            formal_payload,
            completed_steps=_checkpoint_completed_steps(controller_checkpoint),
            reason=reason,  # type: ignore[arg-type]
        )
        audited = checkpoint_store.audit_generations(verify_all_checkpoint_bytes=False)
        if not audited:
            raise RuntimeError("new checkpoint generation was not committed")
        latest = audited[-1]
        generation = latest.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("new checkpoint generation is invalid")
        checkpoint_path = (
            checkpoint_store.generations_path / f"generation-{generation:08d}" / "state.pt"
        )
        if _sha256_file(checkpoint_path) != latest.get("checkpoint_sha256"):
            raise RuntimeError("newly committed checkpoint failed byte verification")
        cumulative_checkpoint_io_seconds += (time.perf_counter_ns() - started) / 1.0e9
        metadata_sha = manifest.get("metadata_sha256")
        _training._validate_digest(
            metadata_sha,
            name="checkpoint manifest metadata_sha256",
        )
        last_checkpoint_metadata_sha256 = metadata_sha

    def after_resume_state_restored() -> None:
        checkpoint_store.restore_pending_rng_state()
        checkpoint_store.record_progress(
            status="resumed",
            completed_steps=(
                0 if resume_state is None else _checkpoint_completed_steps(resume_state)
            ),
            details={
                "execution_role": FORMAL_PRIMARY_EXECUTION_ROLE,
                "admission_sha256": admitted.admission_sha256,
                "runtime_sha256": runtime.runtime_sha256,
                "rng_restored_after_trainer_and_controller_state": True,
                "active_named_rng_states": [],
            },
        )

    def progress_hook(event: Mapping[str, object]) -> None:
        controller_complete = event.get("controller_complete") is True
        completed = event.get("completed_steps")
        next_update = event.get("next_update")
        consecutive = event.get("consecutive_passes")
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or isinstance(consecutive, bool)
            or not isinstance(consecutive, int)
        ):
            raise TypeError("controller safe-boundary progress is malformed")
        current_memory, peak_memory = _memory_bytes(device)
        total_seconds = (time.monotonic_ns() - execution_started_ns) / 1.0e9
        training_seconds = max(
            0.0,
            total_seconds - cumulative_audit_seconds - cumulative_checkpoint_io_seconds,
        )
        active_next_update = (
            next_update
            if isinstance(next_update, int) and not isinstance(next_update, bool)
            else None
        )
        checkpoint_store.record_training_progress(
            status=(
                "finalizing"
                if controller_complete
                else ("running" if active_next_update is not None else "failed")
            ),
            completed_steps=completed,
            next_update=active_next_update,
            learning_rate=(
                float(event["next_learning_rate"])
                if event.get("next_learning_rate") is not None
                else None
            ),
            gradient_ratio=(
                float(event["gradient_ratio"]) if event.get("gradient_ratio") is not None else None
            ),
            consecutive_passes=consecutive,
            pcg=(event["pcg"] if isinstance(event.get("pcg"), Mapping) else None),
            current_gpu_memory_bytes=current_memory,
            peak_gpu_memory_bytes=peak_memory,
            cumulative_training_seconds=training_seconds,
            cumulative_audit_seconds=cumulative_audit_seconds,
            cumulative_checkpoint_io_seconds=cumulative_checkpoint_io_seconds,
            checkpoint_metadata_sha256=last_checkpoint_metadata_sha256,
            signal_state=_signal_state(checkpoint_signal),
            scheduler_segment=admitted.segment_index,
            remaining_allocation_seconds=_remaining_allocation_seconds(runtime),
        )

    def stop_requested() -> str | None:
        if not checkpoint_signal.requested:
            return None
        return checkpoint_signal.signal_name or "scheduler_signal"

    try:
        convergence = _training._run_trainer_to_first_order_convergence(
            trainer,
            audit=timed_audit,
            spec=context.settings.convergence,
            fixed_snapshot_steps=context.settings.outer_steps,
            objective_name=method,
            rank_diagnostic=rank_diagnostic,
            resume_state=resume_state,
            checkpoint_hook=checkpoint_hook,
            checkpoint_interval_steps=checkpoint_interval,
            stop_requested=stop_requested,
            begin_update=checkpoint_signal.begin_update,
            end_update=checkpoint_signal.end_update,
            execution_step_cap=int(execution_slice["end_completed_updates_inclusive"]),
            after_resume_state_restored=(
                after_resume_state_restored if resume_state is not None else None
            ),
            progress_hook=progress_hook,
            execution_role=FORMAL_PRIMARY_EXECUTION_ROLE,
        )
    except PlannedSegmentBoundary as boundary:
        audited = checkpoint_store.audit_generations(verify_all_checkpoint_bytes=True)
        if not audited or last_checkpoint_metadata_sha256 is None:
            raise RuntimeError("planned segment boundary lacks a verified checkpoint") from boundary
        completed = int(audited[-1]["completed_steps"])
        start = int(execution_slice["start_completed_updates"])
        cadence = int(execution_slice["science_audit_cadence_updates"])
        consumed_blocks = (completed - start) // cadence
        expected_blocks = int(execution_slice["safe_update_blocks_available_to_head"])
        if completed < start or (completed - start) % cadence or consumed_blocks != expected_blocks:
            raise RuntimeError(
                "planned segment boundary does not exhaust the frozen head slice"
            ) from boundary
        can_continue = admitted.segment_index < admitted.design.max_scheduler_segments
        checkpoint_store.record_planned_boundary_receipt(
            head_name=method,
            completed_steps=completed,
            checkpoint_metadata_sha256=last_checkpoint_metadata_sha256,
            checkpoint_verified=True,
            last_progress_sha256=checkpoint_store.latest_progress_sha256(),
            scheduler_identity=runtime.to_dict(),
            execution_slice_sha256=execution_slice_sha,  # type: ignore[arg-type]
            update_blocks_consumed=consumed_blocks,
            update_blocks_remaining=0,
            planned_action=("continue_same_logical_run" if can_continue else "fail_closed"),
        )
        raise
    except CheckpointInterruption as interruption:
        audited = checkpoint_store.audit_generations(verify_all_checkpoint_bytes=True)
        if not audited or last_checkpoint_metadata_sha256 is None:
            raise RuntimeError(
                "scheduler interruption lacks a verified checkpoint"
            ) from interruption
        if (
            not isinstance(checkpoint_signal.signal_name, str)
            or not checkpoint_signal.signal_name
            or not isinstance(checkpoint_signal.received_at_utc, str)
            or not checkpoint_signal.received_at_utc.endswith("Z")
        ):
            raise RuntimeError(
                "scheduler interruption lacks an exact signal identity"
            ) from interruption
        completed = int(audited[-1]["completed_steps"])
        last_progress_sha = checkpoint_store.latest_progress_sha256()
        can_continue = admitted.segment_index < admitted.design.max_scheduler_segments
        checkpoint_store.record_signal_receipt(
            head_name=method,
            signal_name=checkpoint_signal.signal_name,
            received_at_utc=checkpoint_signal.received_at_utc,
            additional_signal_count=checkpoint_signal.additional_signal_count,
            completed_steps=completed,
            in_flight_update=checkpoint_signal.received_in_flight_update,
            reached_safe_boundary=True,
            checkpoint_metadata_sha256=last_checkpoint_metadata_sha256,
            checkpoint_flush_succeeded=True,
            checkpoint_verified=True,
            last_progress_sha256=last_progress_sha,
            scheduler_identity=runtime.to_dict(),
            planned_action=("continue_same_logical_run" if can_continue else "fail_closed"),
        )
        raise

    controller_updates = _controller_updates_executed(convergence, trainer)
    selected_terminal = convergence.selected_terminal_checkpoint
    if not isinstance(selected_terminal, Mapping):
        raise RuntimeError(
            "formal R3 convergence did not produce a selected-primary terminal checkpoint"
        )
    selected_primary_step = _positive_integer(
        selected_terminal.get("selected_primary_step"),
        name="selected terminal selected_primary_step",
    )
    if (
        selected_terminal.get("controller_updates_executed") != controller_updates
        or trainer.completed_steps != selected_primary_step
    ):
        raise RuntimeError("selected terminal update counts differ from the live converged run")
    if method == PRORM_PLUS:
        final_solver = convergence.final.inner_solver
        if final_solver is None or final_solver.get("converged") is not True:
            raise RuntimeError("formal ProRM+ final FP64 PCG audit did not converge")
        final_pcg = _training._pcg_evidence(final_solver)
    else:
        final_pcg = None
    head = _training._make_head_evidence(
        arm=_training.PRIMARY_TRAINING_ARM,
        method=method,
        model=trainer.model,
        initial_head_sha256=initial_head_sha,
        initial_objective=convergence.initial.objective,
        final_objective=convergence.final.objective,
        history=convergence.history,
        final_pcg=final_pcg,
        first_order_convergence=convergence.evidence,
    )
    terminal_result_core = _terminal_result_core_payload(
        admission=admitted,
        runtime=runtime,
        learner=method,
        context=context,
        head=head,
        selected_primary_step=selected_primary_step,
        controller_updates_executed=controller_updates,
        head_execution_slice_sha256=execution_slice_sha,  # type: ignore[arg-type]
        resumed_from_predecessor=resumed,
    )
    terminal_payload = _formal_selected_terminal_payload(
        admission=admitted,
        runtime=runtime,
        learner=method,
        head_execution_slice_sha256=execution_slice_sha,  # type: ignore[arg-type]
        selected_terminal_checkpoint=selected_terminal,
        terminal_result_core=terminal_result_core,
    )
    _validate_selected_terminal_payload(
        terminal_payload,
        admission=admitted,
        learner=method,
        runtime=runtime,
        expected_head_execution_slice_sha256=execution_slice_sha,  # type: ignore[arg-type]
    )
    if checkpoint_signal.active_update is not None:
        raise RuntimeError("terminal publication cannot occur during an active update")
    terminal_save_started = time.perf_counter_ns()
    terminal_manifest = checkpoint_store.save(
        terminal_payload,
        completed_steps=selected_primary_step,
        reason="stage_boundary",
    )
    audited_terminal = checkpoint_store.audit_generations(verify_all_checkpoint_bytes=True)
    if not audited_terminal:
        raise RuntimeError("formal primary head completed without a durable checkpoint")
    terminal_generation = audited_terminal[-1].get("generation")
    if (
        isinstance(terminal_generation, bool)
        or not isinstance(terminal_generation, int)
        or audited_terminal[-1].get("completed_steps") != selected_primary_step
    ):
        raise RuntimeError("selected-primary terminal generation is invalid")
    cumulative_checkpoint_io_seconds += (time.perf_counter_ns() - terminal_save_started) / 1.0e9
    terminal_metadata_sha = terminal_manifest.get("metadata_sha256")
    _training._validate_digest(
        terminal_metadata_sha,
        name="selected terminal metadata SHA",
    )
    last_checkpoint_metadata_sha256 = terminal_metadata_sha  # type: ignore[assignment]

    current_memory, peak_memory = _memory_bytes(device)
    total_seconds = (time.monotonic_ns() - execution_started_ns) / 1.0e9
    training_seconds = max(
        0.0,
        total_seconds - cumulative_audit_seconds - cumulative_checkpoint_io_seconds,
    )
    checkpoint_store.record_training_progress(
        status="completed",
        completed_steps=selected_primary_step,
        next_update=None,
        learning_rate=None,
        gradient_ratio=float(
            convergence.evidence["final_gate"]["gradient_ratio_to_zero_initialization"]
        ),
        consecutive_passes=context.settings.convergence.consecutive_checks,
        pcg=None,
        current_gpu_memory_bytes=current_memory,
        peak_gpu_memory_bytes=peak_memory,
        cumulative_training_seconds=_nonnegative_seconds(
            training_seconds,
            name="cumulative_training_seconds",
        ),
        cumulative_audit_seconds=cumulative_audit_seconds,
        cumulative_checkpoint_io_seconds=cumulative_checkpoint_io_seconds,
        checkpoint_metadata_sha256=last_checkpoint_metadata_sha256,
        signal_state=_signal_state(checkpoint_signal),
        scheduler_segment=admitted.segment_index,
        remaining_allocation_seconds=_remaining_allocation_seconds(runtime),
    )
    terminal_signal_serviced = False

    def service_terminal_signal() -> None:
        nonlocal terminal_signal_serviced
        if terminal_signal_serviced or not checkpoint_signal.requested:
            return
        if (
            not isinstance(checkpoint_signal.signal_name, str)
            or not checkpoint_signal.signal_name
            or not isinstance(checkpoint_signal.received_at_utc, str)
            or not checkpoint_signal.received_at_utc.endswith("Z")
        ):
            raise RuntimeError("terminal signal lacks an exact signal identity")
        audited = checkpoint_store.audit_generations(verify_all_checkpoint_bytes=True)
        if not audited or audited[-1].get("generation") != terminal_generation:
            raise RuntimeError("terminal signal does not bind the selected generation")
        checkpoint_store.record_signal_receipt(
            head_name=method,
            signal_name=checkpoint_signal.signal_name,
            received_at_utc=checkpoint_signal.received_at_utc,
            additional_signal_count=checkpoint_signal.additional_signal_count,
            completed_steps=selected_primary_step,
            in_flight_update=checkpoint_signal.received_in_flight_update,
            reached_safe_boundary=True,
            checkpoint_metadata_sha256=last_checkpoint_metadata_sha256,
            checkpoint_flush_succeeded=True,
            checkpoint_verified=True,
            last_progress_sha256=checkpoint_store.latest_progress_sha256(),
            scheduler_identity=runtime.to_dict(),
            planned_action="terminate_completed",
        )
        terminal_signal_serviced = True

    # A signal received during final audit, terminal save, or completed-progress
    # publication terminates safely against this selected terminal generation.
    service_terminal_signal()
    terminal_artifact_sha = _latest_generation_artifact_sha256(
        checkpoint_store,
        admission=admitted,
        learner=method,
        continuation_required=False,
    )
    payload = _result_payload(
        admission=admitted,
        runtime=runtime,
        learner=method,
        context=context,
        head=head,
        selected_primary_step=selected_primary_step,
        controller_updates_executed=controller_updates,
        terminal_checkpoint_artifact_sha256=terminal_artifact_sha,
        head_execution_slice_sha256=execution_slice_sha,  # type: ignore[arg-type]
        resumed_from_predecessor=resumed,
    )
    result = FormalR3PrimaryHeadResult(
        admission=admitted,
        runtime=runtime,
        learner=method,
        context=context,
        head=head,
        selected_primary_step=selected_primary_step,
        controller_updates_executed=controller_updates,
        terminal_checkpoint_artifact_sha256=terminal_artifact_sha,
        head_execution_slice_sha256=execution_slice_sha,  # type: ignore[arg-type]
        resumed_from_predecessor=resumed,
        result_sha256=_canonical_sha256(payload),
    )
    result.validate_integrity()
    # Catch a first signal arriving during result materialization.  No mutable
    # training state exists after this poll.
    service_terminal_signal()
    return result


__all__ = [
    "FORMAL_PRIMARY_CHECKPOINT_BINDING_SCHEMA",
    "FORMAL_PRIMARY_CHECKPOINT_PAYLOAD_SCHEMA",
    "FORMAL_PRIMARY_EXECUTION_ROLE",
    "FORMAL_PRIMARY_HEAD_RESULT_SCHEMA",
    "FORMAL_PRIMARY_SELECTED_TERMINAL_ARTIFACT_ROLE",
    "FORMAL_PRIMARY_SELECTED_TERMINAL_ARTIFACT_SCHEMA",
    "FORMAL_PRIMARY_SELECTED_TERMINAL_PAYLOAD_SCHEMA",
    "FORMAL_PRIMARY_TERMINAL_RESULT_CORE_SCHEMA",
    "HEAD_EXECUTION_SLICE_SCHEMA",
    "FormalR3PrimaryHeadResult",
    "SlurmSegmentRuntime",
    "capture_slurm_segment_runtime",
    "continuation_checkpoint_artifact_ref",
    "formal_primary_checkpoint_binding",
    "latest_generation_is_selected_terminal",
    "recover_formal_result_from_selected_terminal",
    "r3_operational_policy_hashes",
    "run_formal_r3_primary_head_segment",
]
