"""Fail-closed HPC4 evidence plane for the independent R3 Gate-C controls.

This module does not implement any reward-model objective or convergence
criterion.  Family results are accepted only after the scientific
``phase2_r3_controls`` validator reopens them.  The execution plane adds:

* a non-reusable, 100-update-per-family runtime profile;
* one immutable plan for the exact three-family by three-seed matrix;
* result closures and raw Slurm terminal validation for each of the nine jobs;
* an exact 3x3 aggregate; and
* a head-free Gate-C authorization which is useful only together with Gate R.

No object in this module contains a trained parameter, optimizer state,
checkpoint state, validation/test value, policy rollout, utility, or beta.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .phase2_r3_artifacts import (
    CanonicalJsonArtifact,
    canonical_json_bytes,
    publish_canonical_artifact,
)
from .phase2_r3_post_recovery_contract import (
    R3_AUTHORIZED_INFORMATION,
    R3_AUTHORIZED_NEXT_ACTION,
    R3_EXECUTION_REVISION,
    R3_FINAL_AUTHORIZATION_RELATIVE,
    R3_FINAL_AUTHORIZATION_ROLE,
    R3_FINAL_AUTHORIZATION_SCHEMA,
    R3_GATE_C_AGGREGATE_RELATIVE,
    R3_GATE_R_AUTHORIZATION_RELATIVE,
    R3_OPTIMIZER_SCHEDULE_SHA256,
    R3_ORDERED_RECOVERY_SEEDS,
    R3_TRANSPORT_BOUNDARY,
)
from .phase2_r3_sacct_stdlib import (
    _validate_terminal_row,
    inspect_sacct_terminal_bytes,
)

R3_CONTROLS_PROFILE_MEASUREMENT_SCHEMA: Final = (
    "phase2-recovery-r3-gate-c-profile-family-measurement/v1"
)
R3_CONTROLS_PROFILE_COMPUTE_RECEIPT_SCHEMA: Final = (
    "phase2-recovery-r3-gate-c-profile-compute-receipt/v1"
)
R3_CONTROLS_PROFILE_SCHEMA: Final = "phase2-recovery-r3-gate-c-operational-profile/v1"
R3_CONTROLS_PROFILE_ROLE: Final = (
    "train_only_runtime_profile_nonreusable_for_gate_c_resource_freeze"
)
R3_CONTROLS_PLAN_SCHEMA: Final = "phase2-recovery-r3-gate-c-execution-plan/v1"
R3_CONTROLS_TASK_CLOSURE_SCHEMA: Final = "phase2-recovery-r3-gate-c-family-task-closure/v1"
R3_CONTROLS_TERMINAL_SCHEMA: Final = "phase2-recovery-r3-gate-c-external-slurm-terminal/v1"
R3_CONTROLS_AGGREGATE_SCHEMA: Final = "phase2-recovery-r3-gate-c-aggregate/v1"
R3_CONTROLS_AUTHORIZATION_SCHEMA: Final = R3_FINAL_AUTHORIZATION_SCHEMA
R3_CONTROLS_AUTHORIZATION_ROLE: Final = R3_FINAL_AUTHORIZATION_ROLE

R3_CONTROLS_PROFILE_UPDATES_PER_FAMILY: Final = 100
R3_CONTROLS_CHECKPOINT_CADENCE_UPDATES: Final = 200
R3_CONTROLS_PROFILE_CLUSTER: Final = "hpc4"
R3_CONTROLS_PROFILE_ACCOUNT: Final = "sigroup"
R3_CONTROLS_PROFILE_PARTITION: Final = "gpu-l20"
R3_CONTROLS_PROFILE_GPU_NAME: Final = "NVIDIA L20"
R3_CONTROLS_PROFILE_SLURM_GPU_TRES: Final = "gres/gpu:l20"
# ``nvidia-smi`` reports the physical device as 46,068 MiB, while PyTorch
# exposes 47,676,129,280 allocatable-address-space bytes for this allocation.
# Gate-C receipts and admission are defined against the latter because that is
# the capacity observed by the training process.
R3_CONTROLS_L20_PHYSICAL_GPU_MEMORY_BYTES: Final = 46_068 * 1024**2
R3_CONTROLS_PROFILE_TORCH_VISIBLE_GPU_MEMORY_BYTES: Final = 47_676_129_280
R3_CONTROLS_PROFILE_GPU_MEMORY_CAPACITY_BYTES: Final = (
    R3_CONTROLS_PROFILE_TORCH_VISIBLE_GPU_MEMORY_BYTES
)
R3_CONTROLS_PROFILE_GPUS_PER_TASK: Final = 1
R3_CONTROLS_PROFILE_CPUS_PER_TASK: Final = 8
R3_CONTROLS_PROFILE_MEMORY_BYTES: Final = 96 * 1024**3
R3_CONTROLS_PROFILE_NODES: Final = 1
R3_CONTROLS_PROFILE_WALLTIME_SECONDS: Final = 12 * 60 * 60
R3_CONTROLS_MAX_WALLTIME_SECONDS_PER_SEGMENT: Final = 2 * 24 * 60 * 60
R3_CONTROLS_ARRAY_TASKS_PER_FAMILY: Final = 3
R3_CONTROLS_TOTAL_TASKS: Final = 9
R3_CONTROLS_PROJECT_RELATIVE_ROOT: Final = PurePosixPath("runs/phase2-recovery-r3-controls")
R3_CONTROLS_AUTHORIZATION_RELATIVE: Final = PurePosixPath(
    R3_FINAL_AUTHORIZATION_RELATIVE.as_posix()
)
R3_CONTROLS_AGGREGATE_RELATIVE: Final = PurePosixPath(R3_GATE_C_AGGREGATE_RELATIVE.as_posix())

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_POSITIVE_JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
_SAFE_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_PROFILE_STOP_REASON = "predeclared_profile_update_cap"
_PROFILE_INFORMATION_BOUNDARY = "train_only_runtime_measurement"
_FORMAL_INFORMATION_BOUNDARY = "train_only_local_mechanism_evidence"
_TASK_COMPLETION_REASON = "sustained_first_order_gate"

_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "beta",
        "head",
        "head_weight",
        "model",
        "model_state",
        "optimizer_state",
        "checkpoint",
        "policy",
        "rollout",
        "utility",
        "heldout",
        "validation",
        "test",
        "learner_ordering",
        "gradient_direction",
    }
)
_FORBIDDEN_KEY_FRAGMENTS = (
    "head_weight",
    "optimizer_state",
    "model_state",
    "checkpoint_bytes",
    "policy_rollout",
    "finite_policy",
    "heldout_",
    "validation_",
    "test_",
    "beta_",
)

_MEASUREMENT_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "family",
        "seed",
        "completed_updates",
        "stop_reason",
        "information_boundary",
        "result_reusable_for_training",
        "git_commit",
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
        "input_training_sha256",
        "oracle_reward_sha256",
        "setup_wall_seconds",
        "training_wall_seconds",
        "audit_wall_seconds",
        "checkpoint_roundtrip_wall_seconds",
        "peak_gpu_memory_bytes",
        "gpu_total_memory_bytes",
        "scheduler_terminal",
        "measurement_sha256",
    }
)
_PROFILE_COMPUTE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "family",
        "seed",
        "completed_updates",
        "stop_reason",
        "information_boundary",
        "result_reusable_for_training",
        "git_commit",
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
        "input_training_sha256",
        "oracle_reward_sha256",
        "setup_wall_seconds",
        "training_wall_seconds",
        "audit_wall_seconds",
        "checkpoint_roundtrip_wall_seconds",
        "peak_gpu_memory_bytes",
        "gpu_total_memory_bytes",
        "compute_receipt_sha256",
    }
)
_PROFILE_TERMINAL_FIELDS = frozenset(
    {
        "array_job_id",
        "array_task_id",
        "job_id",
        "job_id_raw",
        "raw_sacct_sha256",
        "elapsed_seconds",
    }
)
_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "optimizer_schedule_sha256",
        "git_commit",
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
        "measurements",
        "measurement_set_sha256",
        "resource_plan",
        "profile_sha256",
    }
)
_RESOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "formal_update_cap",
        "profile_updates_per_family",
        "audit_interval_updates",
        "checkpoint_cadence_updates",
        "walltime_safety_margin_fraction",
        "fixed_walltime_margin_seconds",
        "memory_safety_margin_fraction",
        "cluster",
        "account",
        "partition",
        "gpu_name",
        "observed_gpu_memory_capacity_bytes",
        "gpus_per_task",
        "cpus_per_task",
        "memory_bytes",
        "array_concurrency",
        "requested_walltime_seconds_per_segment",
        "signal_lead_seconds",
        "max_scheduler_segments",
        "family_projections",
        "resource_plan_sha256",
    }
)
_PROJECTION_FIELDS = frozenset(
    {
        "family",
        "projected_setup_wall_seconds",
        "projected_update_wall_seconds",
        "projected_audit_wall_seconds",
        "projected_checkpoint_wall_seconds",
        "projected_total_with_margin_seconds",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "profile_sha256",
        "optimizer_schedule_sha256",
        "git_commit",
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
        "resources",
        "arrays",
        "tasks",
        "plan_sha256",
    }
)
_PLAN_RESOURCE_FIELDS = frozenset(
    {
        "cluster",
        "account",
        "partition",
        "gpu_name",
        "observed_gpu_memory_capacity_bytes",
        "slurm_gpu_tres",
        "gpus_per_task",
        "cpus_per_task",
        "memory_bytes",
        "nodes",
        "array_concurrency",
        "requested_walltime_seconds",
        "signal_lead_seconds",
        "checkpoint_cadence_updates",
        "max_scheduler_segments",
    }
)
_ARRAY_FIELDS = frozenset(
    {
        "family_index",
        "family",
        "array_task_range",
        "ordered_seeds",
        "namespace",
    }
)
_TASK_FIELDS = frozenset(
    {
        "task_id",
        "family_index",
        "seed_index",
        "array_task_id",
        "family",
        "seed",
        "namespace",
    }
)
_CLOSURE_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "plan_sha256",
        "profile_sha256",
        "task_id",
        "array_task_id",
        "family",
        "seed",
        "segment_index",
        "family_result_file_sha256",
        "family_result_sha256",
        "information_boundary",
        "compute_complete",
        "completion_reason",
        "result_reusable_for_training",
        "prohibited_channels_accessed",
        "closure_sha256",
    }
)
_PROHIBITED_CHANNEL_FIELDS = frozenset(
    {
        "primary_parameters",
        "primary_optimizer_state",
        "primary_checkpoint",
        "heldout_or_validation",
        "policy_or_rollout",
        "utility_or_outcome",
        "beta",
    }
)
_TERMINAL_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "plan_sha256",
        "closure_sha256",
        "task_id",
        "array_task_id",
        "family",
        "seed",
        "segment_index",
        "array_job_id",
        "job_id",
        "job_id_raw",
        "raw_sacct_sha256",
        "raw_sacct_size_bytes",
        "elapsed_seconds",
        "terminal_sha256",
    }
)


def _core_module() -> Any:
    from . import phase2_r3_controls

    return phase2_r3_controls


def _families_and_seeds() -> tuple[tuple[str, ...], tuple[int, ...]]:
    core = _core_module()
    families = tuple(core.R3_GATE_C_FAMILIES)
    seeds = tuple(core.R3_GATE_C_SEEDS)
    if (
        len(families) != 3
        or len(set(families)) != 3
        or any(type(item) is not str or _SAFE_TOKEN_RE.fullmatch(item) is None for item in families)
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or any(type(item) is not int or item < 1 for item in seeds)
    ):
        raise RuntimeError("the scientific Gate-C API does not expose an exact 3x3 design")
    return families, seeds


def _json_copy(value: object, *, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be strict JSON") from error


def _closed(value: object, *, name: str, fields: frozenset[str]) -> dict[str, Any]:
    copied = _json_copy(value, name=name)
    if not isinstance(copied, dict) or set(copied) != fields:
        raise ValueError(f"{name} has an invalid closed field set")
    return copied


def _semantic_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256_for_mapping(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: object) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError("git_commit must be a full lowercase Git commit")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_real(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_real(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _fraction(value: object, *, name: str) -> float:
    result = _positive_real(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must not exceed one")
    return result


def _positive_job_id(value: object, *, name: str) -> str:
    if type(value) is not str or _POSITIVE_JOB_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a positive decimal Slurm job ID")
    return value


def _reject_forbidden_payload(value: object, *, name: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if type(raw_key) is not str:
                raise TypeError(f"{name} contains a non-string JSON key")
            key = raw_key.lower()
            if (
                key in _FORBIDDEN_EXACT_KEYS
                or any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS)
            ) and child is not False:
                raise ValueError(f"{name} contains forbidden field {raw_key!r}")
            _reject_forbidden_payload(child, name=name)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_payload(child, name=name)


def _self_hashed(
    value: object,
    *,
    name: str,
    fields: frozenset[str],
    hash_field: str,
) -> dict[str, Any]:
    payload = _closed(value, name=name, fields=fields)
    unsigned = dict(payload)
    observed = _digest(unsigned.pop(hash_field), name=f"{name} {hash_field}")
    if _semantic_sha256(unsigned) != observed:
        raise ValueError(f"{name} self-hash is invalid")
    return payload


def _task_design() -> tuple[dict[str, object], ...]:
    families, seeds = _families_and_seeds()
    tasks: list[dict[str, object]] = []
    task_id = 0
    for family_index, family in enumerate(families):
        for seed_index, seed in enumerate(seeds):
            tasks.append(
                {
                    "task_id": task_id,
                    "family_index": family_index,
                    "seed_index": seed_index,
                    "array_task_id": seed_index,
                    "family": family,
                    "seed": seed,
                    "namespace": str(
                        R3_CONTROLS_PROJECT_RELATIVE_ROOT
                        / "formal"
                        / f"family-{family}"
                        / f"seed-{seed}"
                    ),
                }
            )
            task_id += 1
    if len(tasks) != R3_CONTROLS_TOTAL_TASKS:
        raise RuntimeError("internal Gate-C task design is not 3x3")
    return tuple(tasks)


def build_profile_compute_receipt(
    *,
    family: str,
    seed: int,
    git_commit: str,
    container_sha256: str,
    controls_config_file_sha256: str,
    controls_config_semantic_sha256: str,
    input_training_sha256: str,
    oracle_reward_sha256: str,
    setup_wall_seconds: float,
    training_wall_seconds: float,
    audit_wall_seconds: float,
    checkpoint_roundtrip_wall_seconds: float,
    peak_gpu_memory_bytes: int,
    gpu_total_memory_bytes: int,
) -> dict[str, object]:
    """Close disposable profile compute before external Slurm finalization."""

    families, seeds = _families_and_seeds()
    if family not in families or seed != seeds[0]:
        raise ValueError("profile compute must use one frozen family and the first Gate-C seed")
    body = {
        "schema_version": R3_CONTROLS_PROFILE_COMPUTE_RECEIPT_SCHEMA,
        "role": "train_only_100_update_control_family_compute_nonreusable",
        "family": family,
        "seed": seed,
        "completed_updates": R3_CONTROLS_PROFILE_UPDATES_PER_FAMILY,
        "stop_reason": _PROFILE_STOP_REASON,
        "information_boundary": _PROFILE_INFORMATION_BOUNDARY,
        "result_reusable_for_training": False,
        "git_commit": _commit(git_commit),
        "container_sha256": _digest(container_sha256, name="container SHA-256"),
        "controls_config_file_sha256": _digest(
            controls_config_file_sha256,
            name="controls config file SHA-256",
        ),
        "controls_config_semantic_sha256": _digest(
            controls_config_semantic_sha256,
            name="controls config semantic SHA-256",
        ),
        "input_training_sha256": _digest(
            input_training_sha256,
            name="input training SHA-256",
        ),
        "oracle_reward_sha256": _digest(
            oracle_reward_sha256,
            name="oracle reward SHA-256",
        ),
        "setup_wall_seconds": _nonnegative_real(
            setup_wall_seconds,
            name="setup wall seconds",
        ),
        "training_wall_seconds": _positive_real(
            training_wall_seconds,
            name="training wall seconds",
        ),
        "audit_wall_seconds": _positive_real(
            audit_wall_seconds,
            name="audit wall seconds",
        ),
        "checkpoint_roundtrip_wall_seconds": _positive_real(
            checkpoint_roundtrip_wall_seconds,
            name="checkpoint roundtrip wall seconds",
        ),
        "peak_gpu_memory_bytes": _positive_int(
            peak_gpu_memory_bytes,
            name="peak GPU memory bytes",
        ),
        "gpu_total_memory_bytes": _positive_int(
            gpu_total_memory_bytes,
            name="GPU total memory bytes",
        ),
    }
    return validate_profile_compute_receipt(
        {**body, "compute_receipt_sha256": _semantic_sha256(body)}
    )


def validate_profile_compute_receipt(value: object) -> dict[str, object]:
    """Revalidate the scheduler-independent and scientifically non-reusable receipt."""

    payload = _self_hashed(
        value,
        name="Gate-C profile compute receipt",
        fields=_PROFILE_COMPUTE_RECEIPT_FIELDS,
        hash_field="compute_receipt_sha256",
    )
    families, seeds = _families_and_seeds()
    if (
        payload["schema_version"] != R3_CONTROLS_PROFILE_COMPUTE_RECEIPT_SCHEMA
        or payload["role"] != "train_only_100_update_control_family_compute_nonreusable"
        or payload["family"] not in families
        or payload["seed"] != seeds[0]
        or payload["completed_updates"] != R3_CONTROLS_PROFILE_UPDATES_PER_FAMILY
        or payload["stop_reason"] != _PROFILE_STOP_REASON
        or payload["information_boundary"] != _PROFILE_INFORMATION_BOUNDARY
        or payload["result_reusable_for_training"] is not False
    ):
        raise ValueError("Gate-C profile compute receipt exceeds its non-reusable role")
    _commit(payload["git_commit"])
    for name in (
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
        "input_training_sha256",
        "oracle_reward_sha256",
    ):
        _digest(payload[name], name=name)
    _nonnegative_real(payload["setup_wall_seconds"], name="setup wall seconds")
    for name in (
        "training_wall_seconds",
        "audit_wall_seconds",
        "checkpoint_roundtrip_wall_seconds",
    ):
        _positive_real(payload[name], name=name)
    _positive_int(payload["peak_gpu_memory_bytes"], name="peak GPU memory bytes")
    if (
        _positive_int(payload["gpu_total_memory_bytes"], name="GPU total memory bytes")
        != R3_CONTROLS_PROFILE_TORCH_VISIBLE_GPU_MEMORY_BYTES
    ):
        raise ValueError(
            "Gate-C profile did not expose the admitted 47,676,129,280-byte Torch L20 capacity"
        )
    _reject_forbidden_payload(payload, name="Gate-C profile compute receipt")
    return payload


def build_profile_scheduler_terminal(
    raw_sacct_bytes: bytes,
    *,
    expected_raw_sacct_sha256: str,
    family: str,
    array_job_id: str,
    job_id_raw: str,
) -> dict[str, object]:
    """Validate one exact COMPLETED/0:0 fixed HPC4 profile allocation row."""

    families, _ = _families_and_seeds()
    if family not in families:
        raise ValueError("unknown Gate-C profile family")
    task_id = families.index(family)
    parent = _positive_job_id(array_job_id, name="profile array job ID")
    raw_job_id = _positive_job_id(job_id_raw, name="profile raw Slurm job ID")
    expected_job_id = f"{parent}_{task_id}"
    inspection = inspect_sacct_terminal_bytes(
        raw_sacct_bytes,
        expected_raw_sha256=_digest(
            expected_raw_sacct_sha256,
            name="expected raw profile sacct SHA-256",
        ),
    )
    expected_resources = {
        "cluster": R3_CONTROLS_PROFILE_CLUSTER,
        "account": R3_CONTROLS_PROFILE_ACCOUNT,
        "partition": R3_CONTROLS_PROFILE_PARTITION,
        "gpu_name": R3_CONTROLS_PROFILE_GPU_NAME,
        "slurm_gpu_tres": R3_CONTROLS_PROFILE_SLURM_GPU_TRES,
        "gpus_per_task": R3_CONTROLS_PROFILE_GPUS_PER_TASK,
        "cpus_per_task": R3_CONTROLS_PROFILE_CPUS_PER_TASK,
        "memory_bytes": R3_CONTROLS_PROFILE_MEMORY_BYTES,
        "nodes": R3_CONTROLS_PROFILE_NODES,
        "requested_walltime_seconds": R3_CONTROLS_PROFILE_WALLTIME_SECONDS,
    }
    row = _validate_terminal_row(
        inspection,
        expected_job_id=expected_job_id,
        expected_job_id_raw=raw_job_id,
        expected_resources=expected_resources,
        requested_walltime_seconds=R3_CONTROLS_PROFILE_WALLTIME_SECONDS,
    )
    return {
        "array_job_id": parent,
        "array_task_id": task_id,
        "job_id": expected_job_id,
        "job_id_raw": raw_job_id,
        "raw_sacct_sha256": inspection.raw_sacct_sha256,
        "elapsed_seconds": row.elapsed_seconds,
    }


def build_profile_family_measurement_from_compute_receipt(
    compute_receipt: Mapping[str, object],
    *,
    scheduler_terminal: Mapping[str, object],
) -> dict[str, object]:
    """Add external scheduler truth without making profile compute reusable."""

    receipt = validate_profile_compute_receipt(compute_receipt)
    return build_profile_family_measurement(
        family=str(receipt["family"]),
        seed=int(receipt["seed"]),
        git_commit=str(receipt["git_commit"]),
        container_sha256=str(receipt["container_sha256"]),
        controls_config_file_sha256=str(receipt["controls_config_file_sha256"]),
        controls_config_semantic_sha256=str(receipt["controls_config_semantic_sha256"]),
        input_training_sha256=str(receipt["input_training_sha256"]),
        oracle_reward_sha256=str(receipt["oracle_reward_sha256"]),
        setup_wall_seconds=float(receipt["setup_wall_seconds"]),
        training_wall_seconds=float(receipt["training_wall_seconds"]),
        audit_wall_seconds=float(receipt["audit_wall_seconds"]),
        checkpoint_roundtrip_wall_seconds=float(receipt["checkpoint_roundtrip_wall_seconds"]),
        peak_gpu_memory_bytes=int(receipt["peak_gpu_memory_bytes"]),
        gpu_total_memory_bytes=int(receipt["gpu_total_memory_bytes"]),
        scheduler_terminal=scheduler_terminal,
    )


def build_profile_family_measurement(
    *,
    family: str,
    seed: int,
    git_commit: str,
    container_sha256: str,
    controls_config_file_sha256: str,
    controls_config_semantic_sha256: str,
    input_training_sha256: str,
    oracle_reward_sha256: str,
    setup_wall_seconds: float,
    training_wall_seconds: float,
    audit_wall_seconds: float,
    checkpoint_roundtrip_wall_seconds: float,
    peak_gpu_memory_bytes: int,
    gpu_total_memory_bytes: int,
    scheduler_terminal: Mapping[str, object],
) -> dict[str, object]:
    """Build one non-reusable 100-update family measurement."""

    families, seeds = _families_and_seeds()
    if family not in families or seed != seeds[0]:
        raise ValueError("profile measurement must use one frozen family and the first Gate-C seed")
    terminal = _closed(
        scheduler_terminal,
        name="profile scheduler terminal",
        fields=_PROFILE_TERMINAL_FIELDS,
    )
    array_job_id = _positive_job_id(terminal["array_job_id"], name="profile array job ID")
    array_task_id = _nonnegative_int(
        terminal["array_task_id"],
        name="profile array task ID",
    )
    family_index = families.index(family)
    if array_task_id != family_index or terminal["job_id"] != f"{array_job_id}_{array_task_id}":
        raise ValueError("profile scheduler terminal does not match its family task")
    _positive_job_id(terminal["job_id_raw"], name="profile raw job ID")
    _digest(terminal["raw_sacct_sha256"], name="profile raw sacct SHA-256")
    _positive_int(terminal["elapsed_seconds"], name="profile elapsed seconds")
    body = {
        "schema_version": R3_CONTROLS_PROFILE_MEASUREMENT_SCHEMA,
        "role": "train_only_100_update_control_family_runtime_measurement",
        "family": family,
        "seed": seed,
        "completed_updates": R3_CONTROLS_PROFILE_UPDATES_PER_FAMILY,
        "stop_reason": _PROFILE_STOP_REASON,
        "information_boundary": _PROFILE_INFORMATION_BOUNDARY,
        "result_reusable_for_training": False,
        "git_commit": _commit(git_commit),
        "container_sha256": _digest(container_sha256, name="container SHA-256"),
        "controls_config_file_sha256": _digest(
            controls_config_file_sha256,
            name="controls config file SHA-256",
        ),
        "controls_config_semantic_sha256": _digest(
            controls_config_semantic_sha256,
            name="controls config semantic SHA-256",
        ),
        "input_training_sha256": _digest(
            input_training_sha256,
            name="input training SHA-256",
        ),
        "oracle_reward_sha256": _digest(
            oracle_reward_sha256,
            name="oracle reward SHA-256",
        ),
        "setup_wall_seconds": _nonnegative_real(
            setup_wall_seconds,
            name="setup wall seconds",
        ),
        "training_wall_seconds": _positive_real(
            training_wall_seconds,
            name="training wall seconds",
        ),
        "audit_wall_seconds": _positive_real(
            audit_wall_seconds,
            name="audit wall seconds",
        ),
        "checkpoint_roundtrip_wall_seconds": _positive_real(
            checkpoint_roundtrip_wall_seconds,
            name="checkpoint roundtrip wall seconds",
        ),
        "peak_gpu_memory_bytes": _positive_int(
            peak_gpu_memory_bytes,
            name="peak GPU memory bytes",
        ),
        "gpu_total_memory_bytes": _positive_int(
            gpu_total_memory_bytes,
            name="GPU total memory bytes",
        ),
        "scheduler_terminal": terminal,
    }
    result = {**body, "measurement_sha256": _semantic_sha256(body)}
    return validate_profile_family_measurement(result)


def validate_profile_family_measurement(value: object) -> dict[str, object]:
    payload = _self_hashed(
        value,
        name="Gate-C family profile measurement",
        fields=_MEASUREMENT_FIELDS,
        hash_field="measurement_sha256",
    )
    families, seeds = _families_and_seeds()
    if (
        payload["schema_version"] != R3_CONTROLS_PROFILE_MEASUREMENT_SCHEMA
        or payload["role"] != "train_only_100_update_control_family_runtime_measurement"
        or payload["family"] not in families
        or payload["seed"] != seeds[0]
        or payload["completed_updates"] != R3_CONTROLS_PROFILE_UPDATES_PER_FAMILY
        or payload["stop_reason"] != _PROFILE_STOP_REASON
        or payload["information_boundary"] != _PROFILE_INFORMATION_BOUNDARY
        or payload["result_reusable_for_training"] is not False
    ):
        raise ValueError("Gate-C family profile exceeds its frozen non-reusable boundary")
    _commit(payload["git_commit"])
    for name in (
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
        "input_training_sha256",
        "oracle_reward_sha256",
    ):
        _digest(payload[name], name=name)
    _nonnegative_real(payload["setup_wall_seconds"], name="setup wall seconds")
    for name in (
        "training_wall_seconds",
        "audit_wall_seconds",
        "checkpoint_roundtrip_wall_seconds",
    ):
        _positive_real(payload[name], name=name)
    _positive_int(payload["peak_gpu_memory_bytes"], name="peak GPU memory bytes")
    if (
        _positive_int(payload["gpu_total_memory_bytes"], name="GPU total memory bytes")
        != R3_CONTROLS_PROFILE_TORCH_VISIBLE_GPU_MEMORY_BYTES
    ):
        raise ValueError(
            "Gate-C profile did not expose the admitted 47,676,129,280-byte Torch L20 capacity"
        )
    terminal = _closed(
        payload["scheduler_terminal"],
        name="profile scheduler terminal",
        fields=_PROFILE_TERMINAL_FIELDS,
    )
    array_job_id = _positive_job_id(terminal["array_job_id"], name="profile array job ID")
    expected_task = families.index(str(payload["family"]))
    if (
        terminal["array_task_id"] != expected_task
        or terminal["job_id"] != f"{array_job_id}_{expected_task}"
    ):
        raise ValueError("profile scheduler terminal family mapping is invalid")
    _positive_job_id(terminal["job_id_raw"], name="profile raw job ID")
    _digest(terminal["raw_sacct_sha256"], name="profile raw sacct SHA-256")
    _positive_int(terminal["elapsed_seconds"], name="profile elapsed seconds")
    _reject_forbidden_payload(payload, name="Gate-C family profile")
    return payload


def _scientific_schedule(config: object) -> tuple[int, int]:
    """Read formal update/check cadence from the scientific bundle.

    Gate-C owns these values.  The HPC layer intentionally has no fallback
    numeric constants: an unknown bundle shape fails closed.
    """

    direct_maximum = getattr(config, "maximum_updates", None)
    direct_interval = getattr(config, "audit_interval_updates", None)
    if direct_maximum is not None and direct_interval is not None:
        return (
            _positive_int(direct_maximum, name="formal maximum updates"),
            _positive_int(direct_interval, name="formal audit interval"),
        )
    settings = getattr(config, "settings", config)
    convergence = getattr(settings, "convergence", None)
    if convergence is None and isinstance(settings, Mapping):
        convergence = settings.get("convergence")

    def read(source: object, names: tuple[str, ...], label: str) -> int:
        for name in names:
            if isinstance(source, Mapping) and name in source:
                return _positive_int(source[name], name=label)
            if hasattr(source, name):
                return _positive_int(getattr(source, name), name=label)
        raise ValueError(f"controls config does not expose {label}")

    if convergence is None:
        raise ValueError("controls config does not expose its convergence schedule")
    maximum = read(
        convergence,
        ("maximum_updates", "max_updates", "max_steps"),
        "formal maximum updates",
    )
    interval = read(
        convergence,
        ("audit_interval_updates", "check_interval", "check_interval_steps"),
        "formal audit interval",
    )
    return maximum, interval


def _profile_identities(
    measurements: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    families, _ = _families_and_seeds()
    validated = tuple(validate_profile_family_measurement(item) for item in measurements)
    if tuple(item["family"] for item in validated) != families:
        raise ValueError("profile measurements must contain each family once in frozen order")
    common_names = (
        "seed",
        "git_commit",
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
        "gpu_total_memory_bytes",
    )
    first = validated[0]
    for item in validated[1:]:
        if any(item[name] != first[name] for name in common_names):
            raise ValueError("profile measurements do not share one immutable execution identity")
    return validated


def _projection(
    measurement: Mapping[str, object],
    *,
    formal_update_cap: int,
    audit_interval_updates: int,
    checkpoint_cadence_updates: int,
    walltime_safety_margin_fraction: float,
    fixed_walltime_margin_seconds: float,
    signal_lead_seconds: int,
) -> dict[str, object]:
    profile_updates = R3_CONTROLS_PROFILE_UPDATES_PER_FAMILY
    profile_audits = math.ceil(profile_updates / audit_interval_updates)
    formal_audits = math.ceil(formal_update_cap / audit_interval_updates)
    formal_checkpoints = math.ceil(formal_update_cap / checkpoint_cadence_updates) + 3
    setup = float(measurement["setup_wall_seconds"])
    updates = float(measurement["training_wall_seconds"]) * formal_update_cap / profile_updates
    audits = float(measurement["audit_wall_seconds"]) * formal_audits / profile_audits
    checkpoints = float(measurement["checkpoint_roundtrip_wall_seconds"]) * formal_checkpoints
    base = setup + updates + audits + checkpoints
    total = (
        base * (1.0 + walltime_safety_margin_fraction)
        + fixed_walltime_margin_seconds
        + signal_lead_seconds
    )
    return {
        "family": measurement["family"],
        "projected_setup_wall_seconds": setup,
        "projected_update_wall_seconds": updates,
        "projected_audit_wall_seconds": audits,
        "projected_checkpoint_wall_seconds": checkpoints,
        "projected_total_with_margin_seconds": total,
    }


def build_controls_operational_profile(
    measurements: Sequence[Mapping[str, object]],
    *,
    controls_config: object,
    optimizer_schedule_sha256: str,
    checkpoint_cadence_updates: int,
    walltime_safety_margin_fraction: float,
    fixed_walltime_margin_seconds: float,
    memory_safety_margin_fraction: float,
    cluster: str,
    account: str,
    partition: str,
    gpu_name: str,
    cpus_per_task: int,
    memory_bytes: int,
    array_concurrency: int,
    requested_walltime_seconds_per_segment: int,
    signal_lead_seconds: int,
    max_scheduler_segments: int,
) -> dict[str, object]:
    """Freeze resources solely from the three non-reusable family profiles."""

    validated = _profile_identities(measurements)
    formal_update_cap, audit_interval = _scientific_schedule(controls_config)
    cadence = _positive_int(
        checkpoint_cadence_updates,
        name="checkpoint cadence updates",
    )
    if cadence != R3_CONTROLS_CHECKPOINT_CADENCE_UPDATES or cadence % audit_interval != 0:
        raise ValueError(
            "checkpoint cadence must be the frozen 200-update policy "
            "and a multiple of the audit interval"
        )
    margin = _fraction(
        walltime_safety_margin_fraction,
        name="walltime safety margin fraction",
    )
    fixed_margin = _positive_real(
        fixed_walltime_margin_seconds,
        name="fixed walltime margin seconds",
    )
    memory_margin = _fraction(
        memory_safety_margin_fraction,
        name="memory safety margin fraction",
    )
    walltime = _positive_int(
        requested_walltime_seconds_per_segment,
        name="requested walltime seconds per segment",
    )
    if walltime > R3_CONTROLS_MAX_WALLTIME_SECONDS_PER_SEGMENT:
        raise ValueError("Gate-C segment walltime exceeds the frozen two-day ceiling")
    segments = _positive_int(max_scheduler_segments, name="maximum scheduler segments")
    if segments != 1:
        raise ValueError(
            "Gate-C currently requires one profile-covered segment; "
            "multi-segment execution needs a committed scientific continuation API"
        )
    signal = _positive_int(signal_lead_seconds, name="signal lead seconds")
    if signal >= walltime:
        raise ValueError("signal lead time must be strictly smaller than segment walltime")
    concurrency = _positive_int(array_concurrency, name="array concurrency")
    if concurrency != 1:
        raise ValueError("Gate-C family arrays must use rolling concurrency one")
    projections = [
        _projection(
            item,
            formal_update_cap=formal_update_cap,
            audit_interval_updates=audit_interval,
            checkpoint_cadence_updates=cadence,
            walltime_safety_margin_fraction=margin,
            fixed_walltime_margin_seconds=fixed_margin,
            signal_lead_seconds=signal,
        )
        for item in validated
    ]
    if max(float(item["projected_total_with_margin_seconds"]) for item in projections) > (
        walltime * segments
    ):
        raise ValueError("profile projection is not covered by the frozen segment budget")
    required_gpu_memory = max(
        math.ceil(int(item["peak_gpu_memory_bytes"]) * (1.0 + memory_margin)) for item in validated
    )
    admitted_memory = _positive_int(memory_bytes, name="memory bytes")
    observed_gpu_capacity = int(validated[0]["gpu_total_memory_bytes"])
    if required_gpu_memory > observed_gpu_capacity:
        raise ValueError("profile-derived GPU memory requirement exceeds one NVIDIA L20")
    if (
        cluster != R3_CONTROLS_PROFILE_CLUSTER
        or account != R3_CONTROLS_PROFILE_ACCOUNT
        or partition != R3_CONTROLS_PROFILE_PARTITION
        or gpu_name != R3_CONTROLS_PROFILE_GPU_NAME
        or cpus_per_task != R3_CONTROLS_PROFILE_CPUS_PER_TASK
        or admitted_memory != R3_CONTROLS_PROFILE_MEMORY_BYTES
    ):
        raise ValueError(
            "formal Gate-C resources must retain the profiled HPC4 L20, "
            "8-CPU, 96-GiB host allocation"
        )
    first = validated[0]
    resource_body = {
        "schema_version": "phase2-recovery-r3-gate-c-resource-plan/v1",
        "role": "profile_derived_gate_c_scheduler_and_checkpoint_policy",
        "formal_update_cap": formal_update_cap,
        "profile_updates_per_family": R3_CONTROLS_PROFILE_UPDATES_PER_FAMILY,
        "audit_interval_updates": audit_interval,
        "checkpoint_cadence_updates": cadence,
        "walltime_safety_margin_fraction": margin,
        "fixed_walltime_margin_seconds": fixed_margin,
        "memory_safety_margin_fraction": memory_margin,
        "cluster": str(cluster),
        "account": str(account),
        "partition": str(partition),
        "gpu_name": str(gpu_name),
        "observed_gpu_memory_capacity_bytes": observed_gpu_capacity,
        "gpus_per_task": 1,
        "cpus_per_task": _positive_int(cpus_per_task, name="CPUs per task"),
        "memory_bytes": admitted_memory,
        "array_concurrency": concurrency,
        "requested_walltime_seconds_per_segment": walltime,
        "signal_lead_seconds": signal,
        "max_scheduler_segments": segments,
        "family_projections": projections,
    }
    for name in ("cluster", "account", "partition", "gpu_name"):
        if not resource_body[name] or "\n" in str(resource_body[name]):
            raise ValueError(f"{name} is unsafe")
    resource_plan = {
        **resource_body,
        "resource_plan_sha256": _semantic_sha256(resource_body),
    }
    measurement_list = [dict(item) for item in validated]
    body = {
        "schema_version": R3_CONTROLS_PROFILE_SCHEMA,
        "role": R3_CONTROLS_PROFILE_ROLE,
        "optimizer_schedule_sha256": _digest(
            optimizer_schedule_sha256,
            name="optimizer schedule SHA-256",
        ),
        "git_commit": first["git_commit"],
        "container_sha256": first["container_sha256"],
        "controls_config_file_sha256": first["controls_config_file_sha256"],
        "controls_config_semantic_sha256": first["controls_config_semantic_sha256"],
        "measurements": measurement_list,
        "measurement_set_sha256": _semantic_sha256({"measurements": measurement_list}),
        "resource_plan": resource_plan,
    }
    result = {**body, "profile_sha256": _semantic_sha256(body)}
    return validate_controls_operational_profile(result, controls_config=controls_config)


def validate_controls_operational_profile(
    value: object,
    *,
    controls_config: object,
) -> dict[str, object]:
    payload = _self_hashed(
        value,
        name="Gate-C operational profile",
        fields=_PROFILE_FIELDS,
        hash_field="profile_sha256",
    )
    if (
        payload["schema_version"] != R3_CONTROLS_PROFILE_SCHEMA
        or payload["role"] != R3_CONTROLS_PROFILE_ROLE
    ):
        raise ValueError("Gate-C operational profile has the wrong identity")
    _digest(payload["optimizer_schedule_sha256"], name="optimizer schedule SHA-256")
    _commit(payload["git_commit"])
    for name in (
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
        "measurement_set_sha256",
    ):
        _digest(payload[name], name=name)
    config_file_sha = getattr(controls_config, "file_sha256", None)
    config_semantic_sha = getattr(controls_config, "semantic_sha256", None)
    if (
        payload["controls_config_file_sha256"] != config_file_sha
        or payload["controls_config_semantic_sha256"] != config_semantic_sha
    ):
        raise ValueError("Gate-C profile belongs to another scientific config")
    measurements = payload["measurements"]
    if not isinstance(measurements, list):
        raise TypeError("Gate-C measurements must be a list")
    validated = _profile_identities(measurements)
    first = validated[0]
    for name in (
        "git_commit",
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
    ):
        if payload[name] != first[name]:
            raise ValueError(f"Gate-C profile {name} differs from its measurements")
    if payload["measurement_set_sha256"] != _semantic_sha256(
        {"measurements": [dict(item) for item in validated]}
    ):
        raise ValueError("Gate-C profile measurement-set hash is invalid")

    resource = _self_hashed(
        payload["resource_plan"],
        name="Gate-C resource plan",
        fields=_RESOURCE_FIELDS,
        hash_field="resource_plan_sha256",
    )
    maximum, interval = _scientific_schedule(controls_config)
    cadence = _positive_int(
        resource["checkpoint_cadence_updates"],
        name="checkpoint cadence updates",
    )
    if (
        resource["schema_version"] != "phase2-recovery-r3-gate-c-resource-plan/v1"
        or resource["role"] != "profile_derived_gate_c_scheduler_and_checkpoint_policy"
        or resource["formal_update_cap"] != maximum
        or resource["profile_updates_per_family"] != R3_CONTROLS_PROFILE_UPDATES_PER_FAMILY
        or resource["audit_interval_updates"] != interval
        or cadence != R3_CONTROLS_CHECKPOINT_CADENCE_UPDATES
        or cadence % interval != 0
        or resource["gpus_per_task"] != 1
    ):
        raise ValueError("Gate-C resource plan differs from the scientific/profile contract")
    margin = _fraction(
        resource["walltime_safety_margin_fraction"],
        name="walltime safety margin fraction",
    )
    fixed_margin = _positive_real(
        resource["fixed_walltime_margin_seconds"],
        name="fixed walltime margin seconds",
    )
    memory_margin = _fraction(
        resource["memory_safety_margin_fraction"],
        name="memory safety margin fraction",
    )
    walltime = _positive_int(
        resource["requested_walltime_seconds_per_segment"],
        name="requested walltime seconds",
    )
    if walltime > R3_CONTROLS_MAX_WALLTIME_SECONDS_PER_SEGMENT:
        raise ValueError("Gate-C segment walltime exceeds two days")
    signal = _positive_int(resource["signal_lead_seconds"], name="signal lead seconds")
    segments = _positive_int(
        resource["max_scheduler_segments"],
        name="maximum scheduler segments",
    )
    if segments != 1:
        raise ValueError("Gate-C multi-segment execution has no admitted continuation API")
    if signal >= walltime:
        raise ValueError("Gate-C signal lead time is outside its segment")
    cpus = _positive_int(resource["cpus_per_task"], name="CPUs per task")
    memory = _positive_int(resource["memory_bytes"], name="memory bytes")
    concurrency = _positive_int(resource["array_concurrency"], name="array concurrency")
    if concurrency != 1:
        raise ValueError("Gate-C family arrays must use rolling concurrency one")
    for name in ("cluster", "account", "partition", "gpu_name"):
        if type(resource[name]) is not str or not resource[name] or "\n" in resource[name]:
            raise ValueError(f"Gate-C resource {name} is unsafe")
    if (
        resource["cluster"] != R3_CONTROLS_PROFILE_CLUSTER
        or resource["account"] != R3_CONTROLS_PROFILE_ACCOUNT
        or resource["partition"] != R3_CONTROLS_PROFILE_PARTITION
        or resource["gpu_name"] != R3_CONTROLS_PROFILE_GPU_NAME
        or cpus != R3_CONTROLS_PROFILE_CPUS_PER_TASK
        or memory != R3_CONTROLS_PROFILE_MEMORY_BYTES
    ):
        raise ValueError("Gate-C resource plan differs from the fixed profiled host allocation")
    expected_projections = [
        _projection(
            item,
            formal_update_cap=maximum,
            audit_interval_updates=interval,
            checkpoint_cadence_updates=cadence,
            walltime_safety_margin_fraction=margin,
            fixed_walltime_margin_seconds=fixed_margin,
            signal_lead_seconds=signal,
        )
        for item in validated
    ]
    projections = resource["family_projections"]
    if not isinstance(projections, list) or projections != expected_projections:
        raise ValueError("Gate-C family walltime projections are not reproducible")
    for projection in projections:
        _closed(
            projection,
            name="Gate-C family projection",
            fields=_PROJECTION_FIELDS,
        )
    if max(float(item["projected_total_with_margin_seconds"]) for item in projections) > (
        walltime * segments
    ):
        raise ValueError("Gate-C resource plan undercovers its runtime projection")
    required_gpu_memory = max(
        math.ceil(int(item["peak_gpu_memory_bytes"]) * (1.0 + memory_margin)) for item in validated
    )
    observed_gpu_capacity = _positive_int(
        resource["observed_gpu_memory_capacity_bytes"],
        name="observed GPU memory capacity bytes",
    )
    if observed_gpu_capacity != R3_CONTROLS_PROFILE_TORCH_VISIBLE_GPU_MEMORY_BYTES or any(
        item["gpu_total_memory_bytes"] != observed_gpu_capacity for item in validated
    ):
        raise ValueError("Gate-C resource plan GPU capacity differs from profile receipts")
    if required_gpu_memory > observed_gpu_capacity:
        raise ValueError("Gate-C resource plan undercovers measured GPU memory")
    _reject_forbidden_payload(payload, name="Gate-C operational profile")
    return payload


def publish_controls_operational_profile(
    output: str | Path,
    measurements: Sequence[Mapping[str, object]],
    **kwargs: object,
) -> CanonicalJsonArtifact:
    payload = build_controls_operational_profile(measurements, **kwargs)
    return publish_canonical_artifact(output, payload)


def _plan_resources(resource: Mapping[str, object]) -> dict[str, object]:
    partition = str(resource["partition"])
    gpu_token = partition.removeprefix("gpu-").lower()
    if _SAFE_TOKEN_RE.fullmatch(gpu_token) is None:
        raise ValueError("Gate-C partition does not imply a safe Slurm GPU TRES")
    return {
        "cluster": resource["cluster"],
        "account": resource["account"],
        "partition": partition,
        "gpu_name": resource["gpu_name"],
        "observed_gpu_memory_capacity_bytes": resource["observed_gpu_memory_capacity_bytes"],
        "slurm_gpu_tres": f"gres/gpu:{gpu_token}",
        "gpus_per_task": 1,
        "cpus_per_task": resource["cpus_per_task"],
        "memory_bytes": resource["memory_bytes"],
        "nodes": 1,
        "array_concurrency": resource["array_concurrency"],
        "requested_walltime_seconds": resource["requested_walltime_seconds_per_segment"],
        "signal_lead_seconds": resource["signal_lead_seconds"],
        "checkpoint_cadence_updates": resource["checkpoint_cadence_updates"],
        "max_scheduler_segments": resource["max_scheduler_segments"],
    }


def build_controls_execution_plan(
    profile: Mapping[str, object],
    *,
    controls_config: object,
) -> dict[str, object]:
    """Build three separately submitted family arrays with three seeds each."""

    validated = validate_controls_operational_profile(
        profile,
        controls_config=controls_config,
    )
    families, seeds = _families_and_seeds()
    resource = _closed(
        validated["resource_plan"],
        name="Gate-C profile resource plan",
        fields=_RESOURCE_FIELDS,
    )
    resources = _plan_resources(resource)
    concurrency = int(resources["array_concurrency"])
    arrays = [
        {
            "family_index": index,
            "family": family,
            "array_task_range": f"0-2%{concurrency}",
            "ordered_seeds": list(seeds),
            "namespace": str(R3_CONTROLS_PROJECT_RELATIVE_ROOT / "formal" / f"family-{family}"),
        }
        for index, family in enumerate(families)
    ]
    body = {
        "schema_version": R3_CONTROLS_PLAN_SCHEMA,
        "role": "three_independent_family_arrays_exact_three_seeds_each",
        "profile_sha256": validated["profile_sha256"],
        "optimizer_schedule_sha256": validated["optimizer_schedule_sha256"],
        "git_commit": validated["git_commit"],
        "container_sha256": validated["container_sha256"],
        "controls_config_file_sha256": validated["controls_config_file_sha256"],
        "controls_config_semantic_sha256": validated["controls_config_semantic_sha256"],
        "resources": resources,
        "arrays": arrays,
        "tasks": [dict(item) for item in _task_design()],
    }
    result = {**body, "plan_sha256": _semantic_sha256(body)}
    return validate_controls_execution_plan(
        result,
        profile=validated,
        controls_config=controls_config,
    )


def validate_controls_execution_plan(
    value: object,
    *,
    profile: Mapping[str, object],
    controls_config: object,
) -> dict[str, object]:
    payload = _self_hashed(
        value,
        name="Gate-C execution plan",
        fields=_PLAN_FIELDS,
        hash_field="plan_sha256",
    )
    validated_profile = validate_controls_operational_profile(
        profile,
        controls_config=controls_config,
    )
    if (
        payload["schema_version"] != R3_CONTROLS_PLAN_SCHEMA
        or payload["role"] != "three_independent_family_arrays_exact_three_seeds_each"
        or payload["profile_sha256"] != validated_profile["profile_sha256"]
    ):
        raise ValueError("Gate-C execution plan belongs to another profile")
    for name in (
        "optimizer_schedule_sha256",
        "git_commit",
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
    ):
        if payload[name] != validated_profile[name]:
            raise ValueError(f"Gate-C execution plan {name} drifted from its profile")
    resources = _closed(
        payload["resources"],
        name="Gate-C plan resources",
        fields=_PLAN_RESOURCE_FIELDS,
    )
    profile_resource = _closed(
        validated_profile["resource_plan"],
        name="Gate-C profile resources",
        fields=_RESOURCE_FIELDS,
    )
    if resources != _plan_resources(profile_resource):
        raise ValueError("Gate-C plan scheduler resources drifted from the profile")
    families, seeds = _families_and_seeds()
    arrays = payload["arrays"]
    if not isinstance(arrays, list) or len(arrays) != len(families):
        raise ValueError("Gate-C plan must have exactly three family arrays")
    expected_arrays = [
        {
            "family_index": index,
            "family": family,
            "array_task_range": f"0-2%{resources['array_concurrency']}",
            "ordered_seeds": list(seeds),
            "namespace": str(R3_CONTROLS_PROJECT_RELATIVE_ROOT / "formal" / f"family-{family}"),
        }
        for index, family in enumerate(families)
    ]
    for item in arrays:
        _closed(item, name="Gate-C family array", fields=_ARRAY_FIELDS)
    if arrays != expected_arrays:
        raise ValueError("Gate-C family arrays differ from the frozen rolling layout")
    tasks = payload["tasks"]
    expected_tasks = [dict(item) for item in _task_design()]
    if not isinstance(tasks, list):
        raise TypeError("Gate-C plan tasks must be a list")
    for item in tasks:
        _closed(item, name="Gate-C task", fields=_TASK_FIELDS)
    if tasks != expected_tasks:
        raise ValueError("Gate-C execution plan is not the exact 3x3 task matrix")
    _reject_forbidden_payload(payload, name="Gate-C execution plan")
    return payload


def publish_controls_execution_plan(
    output: str | Path,
    profile: Mapping[str, object],
    *,
    controls_config: object,
) -> CanonicalJsonArtifact:
    payload = build_controls_execution_plan(
        profile,
        controls_config=controls_config,
    )
    return publish_canonical_artifact(output, payload)


def _task_from_plan(plan: Mapping[str, object], task_id: int) -> dict[str, object]:
    requested = _nonnegative_int(task_id, name="Gate-C task ID")
    tasks = plan["tasks"]
    if not isinstance(tasks, list) or requested >= len(tasks):
        raise ValueError("Gate-C task ID is outside the exact 3x3 matrix")
    task = _closed(tasks[requested], name="Gate-C task", fields=_TASK_FIELDS)
    if task["task_id"] != requested:
        raise ValueError("Gate-C task index and task ID differ")
    return task


def _validate_core_result(
    result: object,
    *,
    controls_config: object,
    family: str,
    seed: int,
) -> dict[str, object]:
    core = _core_module()
    validated = core.validate_r3_control_family_result(result, controls_config)
    copied = _json_copy(validated, name="Gate-C family result")
    if not isinstance(copied, dict):
        raise TypeError("scientific Gate-C validator did not return a JSON object")
    if copied.get("family") != family or copied.get("seed") != seed:
        raise ValueError("scientific Gate-C result belongs to another family/seed")
    completion = copied.get("completion")
    if (
        not isinstance(completion, dict)
        or completion.get("status") != "completed"
        or completion.get("stop_reason") != _TASK_COMPLETION_REASON
        or completion.get("formal_family_result") is not True
        or completion.get("profile_only") is not False
        or completion.get("head_or_optimizer_state_retained") is not False
    ):
        raise ValueError("scientific Gate-C result did not close the sustained gate")
    result_sha = copied.get("result_sha256")
    _digest(result_sha, name="Gate-C family result SHA-256")
    unsigned = dict(copied)
    del unsigned["result_sha256"]
    if _semantic_sha256(unsigned) != result_sha:
        raise ValueError("Gate-C family result self-hash is invalid")
    return copied


def build_controls_task_closure(
    plan: Mapping[str, object],
    *,
    profile: Mapping[str, object],
    controls_config: object,
    task_id: int,
    segment_index: int,
    family_result: Mapping[str, object],
) -> dict[str, object]:
    validated_plan = validate_controls_execution_plan(
        plan,
        profile=profile,
        controls_config=controls_config,
    )
    task = _task_from_plan(validated_plan, task_id)
    segment = _positive_int(segment_index, name="Gate-C segment index")
    resources = _closed(
        validated_plan["resources"],
        name="Gate-C plan resources",
        fields=_PLAN_RESOURCE_FIELDS,
    )
    if segment > int(resources["max_scheduler_segments"]):
        raise ValueError("Gate-C task exceeded the profile-frozen segment count")
    result = _validate_core_result(
        family_result,
        controls_config=controls_config,
        family=str(task["family"]),
        seed=int(task["seed"]),
    )
    prohibited = {name: False for name in sorted(_PROHIBITED_CHANNEL_FIELDS)}
    body = {
        "schema_version": R3_CONTROLS_TASK_CLOSURE_SCHEMA,
        "role": "one_complete_train_only_gate_c_family_seed_result",
        "plan_sha256": validated_plan["plan_sha256"],
        "profile_sha256": validated_plan["profile_sha256"],
        "task_id": task["task_id"],
        "array_task_id": task["array_task_id"],
        "family": task["family"],
        "seed": task["seed"],
        "segment_index": segment,
        "family_result_file_sha256": _file_sha256_for_mapping(result),
        "family_result_sha256": result["result_sha256"],
        "information_boundary": _FORMAL_INFORMATION_BOUNDARY,
        "compute_complete": True,
        "completion_reason": _TASK_COMPLETION_REASON,
        "result_reusable_for_training": False,
        "prohibited_channels_accessed": prohibited,
    }
    result_closure = {**body, "closure_sha256": _semantic_sha256(body)}
    return validate_controls_task_closure(
        result_closure,
        plan=validated_plan,
        profile=profile,
        controls_config=controls_config,
        family_result=result,
    )


def validate_controls_task_closure(
    value: object,
    *,
    plan: Mapping[str, object],
    profile: Mapping[str, object],
    controls_config: object,
    family_result: Mapping[str, object],
) -> dict[str, object]:
    payload = _self_hashed(
        value,
        name="Gate-C task closure",
        fields=_CLOSURE_FIELDS,
        hash_field="closure_sha256",
    )
    validated_plan = validate_controls_execution_plan(
        plan,
        profile=profile,
        controls_config=controls_config,
    )
    task = _task_from_plan(validated_plan, int(payload["task_id"]))
    result = _validate_core_result(
        family_result,
        controls_config=controls_config,
        family=str(task["family"]),
        seed=int(task["seed"]),
    )
    resources = _closed(
        validated_plan["resources"],
        name="Gate-C plan resources",
        fields=_PLAN_RESOURCE_FIELDS,
    )
    segment = _positive_int(payload["segment_index"], name="Gate-C segment index")
    prohibited = _closed(
        payload["prohibited_channels_accessed"],
        name="Gate-C prohibited-channel evidence",
        fields=_PROHIBITED_CHANNEL_FIELDS,
    )
    if (
        payload["schema_version"] != R3_CONTROLS_TASK_CLOSURE_SCHEMA
        or payload["role"] != "one_complete_train_only_gate_c_family_seed_result"
        or payload["plan_sha256"] != validated_plan["plan_sha256"]
        or payload["profile_sha256"] != validated_plan["profile_sha256"]
        or payload["array_task_id"] != task["array_task_id"]
        or payload["family"] != task["family"]
        or payload["seed"] != task["seed"]
        or segment > resources["max_scheduler_segments"]
        or payload["family_result_file_sha256"] != _file_sha256_for_mapping(result)
        or payload["family_result_sha256"] != result["result_sha256"]
        or payload["information_boundary"] != _FORMAL_INFORMATION_BOUNDARY
        or payload["compute_complete"] is not True
        or payload["completion_reason"] != _TASK_COMPLETION_REASON
        or payload["result_reusable_for_training"] is not False
        or any(item is not False for item in prohibited.values())
    ):
        raise ValueError("Gate-C task closure violates its train-only completion boundary")
    _reject_forbidden_payload(payload, name="Gate-C task closure")
    return payload


def publish_controls_task_closure(
    output: str | Path,
    plan: Mapping[str, object],
    **kwargs: object,
) -> CanonicalJsonArtifact:
    payload = build_controls_task_closure(plan, **kwargs)
    return publish_canonical_artifact(output, payload)


def _slurm_resources(plan: Mapping[str, object]) -> dict[str, object]:
    resources = _closed(
        plan["resources"],
        name="Gate-C plan resources",
        fields=_PLAN_RESOURCE_FIELDS,
    )
    return {
        "cluster": resources["cluster"],
        "account": resources["account"],
        "partition": resources["partition"],
        "gpu_name": resources["gpu_name"],
        "slurm_gpu_tres": resources["slurm_gpu_tres"],
        "gpus_per_task": resources["gpus_per_task"],
        "cpus_per_task": resources["cpus_per_task"],
        "memory_bytes": resources["memory_bytes"],
        "nodes": resources["nodes"],
        "requested_walltime_seconds": resources["requested_walltime_seconds"],
    }


def build_controls_task_terminal(
    raw_sacct_bytes: bytes,
    *,
    expected_raw_sacct_sha256: str,
    plan: Mapping[str, object],
    profile: Mapping[str, object],
    controls_config: object,
    closure: Mapping[str, object],
    family_result: Mapping[str, object],
    array_job_id: str,
    job_id_raw: str,
) -> dict[str, object]:
    """Promote one exact raw Slurm row after reopening compute evidence."""

    validated_plan = validate_controls_execution_plan(
        plan,
        profile=profile,
        controls_config=controls_config,
    )
    validated_closure = validate_controls_task_closure(
        closure,
        plan=validated_plan,
        profile=profile,
        controls_config=controls_config,
        family_result=family_result,
    )
    task = _task_from_plan(validated_plan, int(validated_closure["task_id"]))
    parent = _positive_job_id(array_job_id, name="Gate-C array job ID")
    raw_job_id = _positive_job_id(job_id_raw, name="Gate-C raw Slurm job ID")
    expected_job_id = f"{parent}_{task['array_task_id']}"
    inspection = inspect_sacct_terminal_bytes(
        raw_sacct_bytes,
        expected_raw_sha256=_digest(
            expected_raw_sacct_sha256,
            name="expected raw sacct SHA-256",
        ),
    )
    resources = _slurm_resources(validated_plan)
    row = _validate_terminal_row(
        inspection,
        expected_job_id=expected_job_id,
        expected_job_id_raw=raw_job_id,
        expected_resources=resources,
        requested_walltime_seconds=int(resources["requested_walltime_seconds"]),
    )
    body = {
        "schema_version": R3_CONTROLS_TERMINAL_SCHEMA,
        "role": "external_scheduler_completed_zero_exit_gate_c_family_task",
        "plan_sha256": validated_plan["plan_sha256"],
        "closure_sha256": validated_closure["closure_sha256"],
        "task_id": task["task_id"],
        "array_task_id": task["array_task_id"],
        "family": task["family"],
        "seed": task["seed"],
        "segment_index": validated_closure["segment_index"],
        "array_job_id": parent,
        "job_id": expected_job_id,
        "job_id_raw": raw_job_id,
        "raw_sacct_sha256": inspection.raw_sacct_sha256,
        "raw_sacct_size_bytes": inspection.raw_size_bytes,
        "elapsed_seconds": row.elapsed_seconds,
    }
    result = {**body, "terminal_sha256": _semantic_sha256(body)}
    return validate_controls_task_terminal(
        result,
        raw_sacct_bytes=raw_sacct_bytes,
        plan=validated_plan,
        profile=profile,
        controls_config=controls_config,
        closure=validated_closure,
        family_result=family_result,
    )


def validate_controls_task_terminal(
    value: object,
    *,
    raw_sacct_bytes: bytes,
    plan: Mapping[str, object],
    profile: Mapping[str, object],
    controls_config: object,
    closure: Mapping[str, object],
    family_result: Mapping[str, object],
) -> dict[str, object]:
    payload = _self_hashed(
        value,
        name="Gate-C Slurm terminal",
        fields=_TERMINAL_FIELDS,
        hash_field="terminal_sha256",
    )
    validated_plan = validate_controls_execution_plan(
        plan,
        profile=profile,
        controls_config=controls_config,
    )
    validated_closure = validate_controls_task_closure(
        closure,
        plan=validated_plan,
        profile=profile,
        controls_config=controls_config,
        family_result=family_result,
    )
    task = _task_from_plan(validated_plan, int(payload["task_id"]))
    raw_sha = _digest(payload["raw_sacct_sha256"], name="raw sacct SHA-256")
    inspection = inspect_sacct_terminal_bytes(
        raw_sacct_bytes,
        expected_raw_sha256=raw_sha,
    )
    parent = _positive_job_id(payload["array_job_id"], name="Gate-C array job ID")
    expected_job_id = f"{parent}_{task['array_task_id']}"
    raw_job_id = _positive_job_id(payload["job_id_raw"], name="Gate-C raw job ID")
    resources = _slurm_resources(validated_plan)
    row = _validate_terminal_row(
        inspection,
        expected_job_id=expected_job_id,
        expected_job_id_raw=raw_job_id,
        expected_resources=resources,
        requested_walltime_seconds=int(resources["requested_walltime_seconds"]),
    )
    if (
        payload["schema_version"] != R3_CONTROLS_TERMINAL_SCHEMA
        or payload["role"] != "external_scheduler_completed_zero_exit_gate_c_family_task"
        or payload["plan_sha256"] != validated_plan["plan_sha256"]
        or payload["closure_sha256"] != validated_closure["closure_sha256"]
        or payload["array_task_id"] != task["array_task_id"]
        or payload["family"] != task["family"]
        or payload["seed"] != task["seed"]
        or payload["segment_index"] != validated_closure["segment_index"]
        or payload["job_id"] != expected_job_id
        or payload["raw_sacct_size_bytes"] != inspection.raw_size_bytes
        or payload["elapsed_seconds"] != row.elapsed_seconds
    ):
        raise ValueError("Gate-C terminal differs from its plan, compute, or raw Slurm row")
    _reject_forbidden_payload(payload, name="Gate-C scheduler terminal")
    return payload


def publish_controls_task_terminal(
    output: str | Path,
    raw_sacct_bytes: bytes,
    **kwargs: object,
) -> CanonicalJsonArtifact:
    payload = build_controls_task_terminal(raw_sacct_bytes, **kwargs)
    return publish_canonical_artifact(output, payload)


def _source_record(
    *,
    task: Mapping[str, object],
    family_result: Mapping[str, object],
    closure: Mapping[str, object],
    terminal: Mapping[str, object],
) -> dict[str, object]:
    return {
        "task_id": task["task_id"],
        "family": task["family"],
        "seed": task["seed"],
        "family_result_file_sha256": _file_sha256_for_mapping(family_result),
        "family_result_sha256": family_result["result_sha256"],
        "closure_sha256": closure["closure_sha256"],
        "terminal_sha256": terminal["terminal_sha256"],
        "raw_sacct_sha256": terminal["raw_sacct_sha256"],
    }


def build_controls_aggregate(
    entries: Sequence[
        tuple[
            Mapping[str, object],
            Mapping[str, object],
            Mapping[str, object],
            bytes,
        ]
    ],
    *,
    plan: Mapping[str, object],
    profile: Mapping[str, object],
    controls_config: object,
) -> dict[str, object]:
    """Close Gate C only for all nine ordered family/seed tasks."""

    validated_plan = validate_controls_execution_plan(
        plan,
        profile=profile,
        controls_config=controls_config,
    )
    if not isinstance(entries, Sequence) or len(entries) != R3_CONTROLS_TOTAL_TASKS:
        raise ValueError("Gate-C aggregation requires exactly nine task entries")
    sources: list[dict[str, object]] = []
    family_counts: dict[str, int] = {family: 0 for family in _families_and_seeds()[0]}
    seen_scheduler_jobs: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, tuple) or len(raw_entry) != 4:
            raise TypeError("each Gate-C aggregate entry must be a four-item tuple")
        family_result, closure, terminal, raw_sacct_bytes = raw_entry
        task = _task_from_plan(validated_plan, index)
        result = _validate_core_result(
            family_result,
            controls_config=controls_config,
            family=str(task["family"]),
            seed=int(task["seed"]),
        )
        checked_closure = validate_controls_task_closure(
            closure,
            plan=validated_plan,
            profile=profile,
            controls_config=controls_config,
            family_result=result,
        )
        if checked_closure["task_id"] != index:
            raise ValueError("Gate-C aggregate entries are not in frozen task order")
        checked_terminal = validate_controls_task_terminal(
            terminal,
            raw_sacct_bytes=raw_sacct_bytes,
            plan=validated_plan,
            profile=profile,
            controls_config=controls_config,
            closure=checked_closure,
            family_result=result,
        )
        scheduler_identity = (
            str(checked_terminal["job_id"]),
            str(checked_terminal["job_id_raw"]),
        )
        if scheduler_identity in seen_scheduler_jobs:
            raise ValueError("Gate-C aggregate contains duplicate scheduler evidence")
        seen_scheduler_jobs.add(scheduler_identity)
        family_counts[str(task["family"])] += 1
        sources.append(
            _source_record(
                task=task,
                family_result=result,
                closure=checked_closure,
                terminal=checked_terminal,
            )
        )
    if any(count != 3 for count in family_counts.values()):
        raise ValueError("Gate-C aggregate is not exactly three seeds per family")
    body = {
        "schema_version": R3_CONTROLS_AGGREGATE_SCHEMA,
        "role": "exact_three_families_by_three_seeds_train_only_gate_c_closure",
        "plan_sha256": validated_plan["plan_sha256"],
        "profile_sha256": validated_plan["profile_sha256"],
        "optimizer_schedule_sha256": validated_plan["optimizer_schedule_sha256"],
        "ordered_families": list(_families_and_seeds()[0]),
        "ordered_seeds": list(_families_and_seeds()[1]),
        "matrix_shape": [3, 3],
        "all_nine_compute_complete": True,
        "all_nine_scheduler_success": True,
        "gate_c_passed": True,
        "fresh_calibration_authorized": False,
        "result_reusable_for_training": False,
        "information_boundary": _FORMAL_INFORMATION_BOUNDARY,
        "sources": sources,
        "source_set_sha256": _semantic_sha256({"sources": sources}),
    }
    result = {**body, "aggregate_sha256": _semantic_sha256(body)}
    _reject_forbidden_payload(result, name="Gate-C aggregate")
    return result


def validate_controls_aggregate_structure(value: object) -> dict[str, object]:
    """Validate a head-free aggregate envelope without claiming source liveness."""

    fields = frozenset(
        {
            "schema_version",
            "role",
            "plan_sha256",
            "profile_sha256",
            "optimizer_schedule_sha256",
            "ordered_families",
            "ordered_seeds",
            "matrix_shape",
            "all_nine_compute_complete",
            "all_nine_scheduler_success",
            "gate_c_passed",
            "fresh_calibration_authorized",
            "result_reusable_for_training",
            "information_boundary",
            "sources",
            "source_set_sha256",
            "aggregate_sha256",
        }
    )
    payload = _self_hashed(
        value,
        name="Gate-C aggregate",
        fields=fields,
        hash_field="aggregate_sha256",
    )
    families, seeds = _families_and_seeds()
    sources = payload["sources"]
    if (
        payload["schema_version"] != R3_CONTROLS_AGGREGATE_SCHEMA
        or payload["role"] != "exact_three_families_by_three_seeds_train_only_gate_c_closure"
        or payload["ordered_families"] != list(families)
        or payload["ordered_seeds"] != list(seeds)
        or payload["matrix_shape"] != [3, 3]
        or payload["all_nine_compute_complete"] is not True
        or payload["all_nine_scheduler_success"] is not True
        or payload["gate_c_passed"] is not True
        or payload["fresh_calibration_authorized"] is not False
        or payload["result_reusable_for_training"] is not False
        or payload["information_boundary"] != _FORMAL_INFORMATION_BOUNDARY
        or not isinstance(sources, list)
        or len(sources) != R3_CONTROLS_TOTAL_TASKS
        or payload["source_set_sha256"] != _semantic_sha256({"sources": sources})
    ):
        raise ValueError("Gate-C aggregate is not the exact head-free 3x3 closure")
    for name in (
        "plan_sha256",
        "profile_sha256",
        "optimizer_schedule_sha256",
        "source_set_sha256",
    ):
        _digest(payload[name], name=name)
    expected_tasks = _task_design()
    source_fields = frozenset(
        {
            "task_id",
            "family",
            "seed",
            "family_result_file_sha256",
            "family_result_sha256",
            "closure_sha256",
            "terminal_sha256",
            "raw_sacct_sha256",
        }
    )
    for source, task in zip(sources, expected_tasks, strict=True):
        item = _closed(source, name="Gate-C aggregate source", fields=source_fields)
        if (
            item["task_id"] != task["task_id"]
            or item["family"] != task["family"]
            or item["seed"] != task["seed"]
        ):
            raise ValueError("Gate-C aggregate source ordering is invalid")
        for name in source_fields - {"task_id", "family", "seed"}:
            _digest(item[name], name=f"Gate-C source {name}")
    _reject_forbidden_payload(payload, name="Gate-C aggregate")
    return payload


def _validated_gate_r(value: object) -> dict[str, object]:
    from . import phase2_r3_authorization as gate_r

    validated = gate_r._validate_authorization_structure(value)
    if (
        validated.get("schema_version") != gate_r.R3_SUCCESS_AUTHORIZATION_SCHEMA
        or validated.get("gate_r_passed") is not True
        or validated.get("fresh_calibration_authorized") is not False
    ):
        raise ValueError("Gate-R authorization is not the required R3 schedule-only gate")
    return validated


def build_controls_authorization(
    aggregate: Mapping[str, object],
    *,
    gate_r_authorization: Mapping[str, object],
    gate_r_authorization_file_sha256: str,
) -> dict[str, object]:
    """Combine Gate R and Gate C without transporting any trained state."""

    gate_c = validate_controls_aggregate_structure(aggregate)
    gate_r = _validated_gate_r(gate_r_authorization)
    if gate_r["optimizer_schedule_sha256"] != gate_c["optimizer_schedule_sha256"]:
        raise ValueError("Gate R and Gate C bind different optimizer schedules")
    if (
        gate_r["optimizer_schedule_sha256"] != R3_OPTIMIZER_SCHEDULE_SHA256
        or gate_r["execution_revision"] != R3_EXECUTION_REVISION
        or gate_r["ordered_seeds"] != list(R3_ORDERED_RECOVERY_SEEDS)
        or gate_c["ordered_seeds"] != list(R3_ORDERED_RECOVERY_SEEDS)
    ):
        raise ValueError("Gate R and Gate C do not bind the exact R3 design schedule")
    body = {
        "schema_version": R3_CONTROLS_AUTHORIZATION_SCHEMA,
        "role": R3_CONTROLS_AUTHORIZATION_ROLE,
        "recovery_design_sha256": gate_r["recovery_design_sha256"],
        "optimizer_schedule_sha256": gate_c["optimizer_schedule_sha256"],
        "optimizer_schedule_is_unique": True,
        "execution_revision": gate_r["execution_revision"],
        "ordered_seeds": list(R3_ORDERED_RECOVERY_SEEDS),
        "gate_r_authorization_path": R3_GATE_R_AUTHORIZATION_RELATIVE.as_posix(),
        "gate_r_authorization_file_sha256": _digest(
            gate_r_authorization_file_sha256,
            name="Gate-R authorization file SHA-256",
        ),
        "gate_r_authorization_sha256": gate_r["authorization_sha256"],
        "gate_c_aggregate_path": R3_GATE_C_AGGREGATE_RELATIVE.as_posix(),
        "gate_c_aggregate_file_sha256": _file_sha256_for_mapping(gate_c),
        "gate_c_aggregate_sha256": gate_c["aggregate_sha256"],
        "gate_r_passed": True,
        "gate_c_passed": True,
        "fresh_calibration_authorized": True,
        "authorized_information": R3_AUTHORIZED_INFORMATION,
        "authorized_next_action": R3_AUTHORIZED_NEXT_ACTION,
        "formal_efficacy_claim_authorized": False,
        "recovery_or_control_outputs_reusable": False,
        "validation_or_heldout_access_authorized": False,
        "policy_or_final_utility_access_authorized": False,
        "transport_boundary": dict(R3_TRANSPORT_BOUNDARY),
        "gate_c_source_set_sha256": gate_c["source_set_sha256"],
    }
    result = {**body, "authorization_sha256": _semantic_sha256(body)}
    return validate_controls_authorization_structure(
        result,
        aggregate=gate_c,
        gate_r_authorization=gate_r,
        gate_r_authorization_file_sha256=gate_r_authorization_file_sha256,
    )


def validate_controls_authorization_structure(
    value: object,
    *,
    aggregate: Mapping[str, object],
    gate_r_authorization: Mapping[str, object],
    gate_r_authorization_file_sha256: str,
) -> dict[str, object]:
    fields = frozenset(
        {
            "schema_version",
            "role",
            "recovery_design_sha256",
            "optimizer_schedule_sha256",
            "optimizer_schedule_is_unique",
            "execution_revision",
            "ordered_seeds",
            "gate_r_authorization_path",
            "gate_r_authorization_file_sha256",
            "gate_r_authorization_sha256",
            "gate_c_aggregate_path",
            "gate_c_aggregate_file_sha256",
            "gate_c_aggregate_sha256",
            "gate_r_passed",
            "gate_c_passed",
            "fresh_calibration_authorized",
            "authorized_information",
            "authorized_next_action",
            "formal_efficacy_claim_authorized",
            "recovery_or_control_outputs_reusable",
            "validation_or_heldout_access_authorized",
            "policy_or_final_utility_access_authorized",
            "transport_boundary",
            "gate_c_source_set_sha256",
            "authorization_sha256",
        }
    )
    payload = _self_hashed(
        value,
        name="Gate-C success authorization",
        fields=fields,
        hash_field="authorization_sha256",
    )
    gate_c = validate_controls_aggregate_structure(aggregate)
    gate_r = _validated_gate_r(gate_r_authorization)
    transport = _closed(
        payload["transport_boundary"],
        name="Gate-C authorization transport boundary",
        fields=frozenset(
            {
                "parameters",
                "optimizer_moments",
                "checkpoints",
                "labels_or_data",
                "gradients_or_directions",
                "validation_or_test_values",
                "policy_outputs",
                "utility_values",
                "beta_values",
            }
        ),
    )
    expected_gate_r_file_sha = _digest(
        gate_r_authorization_file_sha256,
        name="Gate-R authorization file SHA-256",
    )
    if (
        payload["schema_version"] != R3_CONTROLS_AUTHORIZATION_SCHEMA
        or payload["role"] != R3_CONTROLS_AUTHORIZATION_ROLE
        or payload["recovery_design_sha256"] != gate_r["recovery_design_sha256"]
        or payload["optimizer_schedule_sha256"] != R3_OPTIMIZER_SCHEDULE_SHA256
        or payload["optimizer_schedule_is_unique"] is not True
        or payload["execution_revision"] != R3_EXECUTION_REVISION
        or payload["ordered_seeds"] != list(R3_ORDERED_RECOVERY_SEEDS)
        or payload["gate_r_authorization_path"] != R3_GATE_R_AUTHORIZATION_RELATIVE.as_posix()
        or payload["gate_r_authorization_file_sha256"] != expected_gate_r_file_sha
        or payload["gate_r_authorization_sha256"] != gate_r["authorization_sha256"]
        or payload["gate_c_aggregate_path"] != R3_GATE_C_AGGREGATE_RELATIVE.as_posix()
        or payload["gate_c_aggregate_file_sha256"] != _file_sha256_for_mapping(gate_c)
        or payload["gate_c_aggregate_sha256"] != gate_c["aggregate_sha256"]
        or payload["optimizer_schedule_sha256"] != gate_c["optimizer_schedule_sha256"]
        or gate_r["optimizer_schedule_sha256"] != payload["optimizer_schedule_sha256"]
        or payload["gate_r_passed"] is not True
        or payload["gate_c_passed"] is not True
        or payload["fresh_calibration_authorized"] is not True
        or payload["authorized_information"] != R3_AUTHORIZED_INFORMATION
        or payload["authorized_next_action"] != R3_AUTHORIZED_NEXT_ACTION
        or payload["formal_efficacy_claim_authorized"] is not False
        or payload["recovery_or_control_outputs_reusable"] is not False
        or payload["validation_or_heldout_access_authorized"] is not False
        or payload["policy_or_final_utility_access_authorized"] is not False
        or transport != R3_TRANSPORT_BOUNDARY
        or payload["gate_c_source_set_sha256"] != gate_c["source_set_sha256"]
    ):
        raise ValueError("Gate-C authorization exceeds the Gate-R plus Gate-C boundary")
    _reject_forbidden_payload(payload, name="Gate-C authorization")
    return payload


def publish_controls_authorization(
    output: str | Path,
    aggregate: Mapping[str, object],
    *,
    gate_r_authorization: Mapping[str, object],
    gate_r_authorization_file_sha256: str,
) -> CanonicalJsonArtifact:
    payload = build_controls_authorization(
        aggregate,
        gate_r_authorization=gate_r_authorization,
        gate_r_authorization_file_sha256=gate_r_authorization_file_sha256,
    )
    return publish_canonical_artifact(output, payload)


__all__ = [
    "R3_CONTROLS_AGGREGATE_SCHEMA",
    "R3_CONTROLS_AGGREGATE_RELATIVE",
    "R3_CONTROLS_ARRAY_TASKS_PER_FAMILY",
    "R3_CONTROLS_AUTHORIZATION_RELATIVE",
    "R3_CONTROLS_AUTHORIZATION_ROLE",
    "R3_CONTROLS_AUTHORIZATION_SCHEMA",
    "R3_CONTROLS_MAX_WALLTIME_SECONDS_PER_SEGMENT",
    "R3_CONTROLS_PLAN_SCHEMA",
    "R3_CONTROLS_PROFILE_COMPUTE_RECEIPT_SCHEMA",
    "R3_CONTROLS_CHECKPOINT_CADENCE_UPDATES",
    "R3_CONTROLS_PROFILE_ACCOUNT",
    "R3_CONTROLS_PROFILE_CLUSTER",
    "R3_CONTROLS_PROFILE_CPUS_PER_TASK",
    "R3_CONTROLS_PROFILE_GPUS_PER_TASK",
    "R3_CONTROLS_PROFILE_GPU_NAME",
    "R3_CONTROLS_PROFILE_GPU_MEMORY_CAPACITY_BYTES",
    "R3_CONTROLS_PROFILE_TORCH_VISIBLE_GPU_MEMORY_BYTES",
    "R3_CONTROLS_L20_PHYSICAL_GPU_MEMORY_BYTES",
    "R3_CONTROLS_PROFILE_MEMORY_BYTES",
    "R3_CONTROLS_PROFILE_MEASUREMENT_SCHEMA",
    "R3_CONTROLS_PROFILE_NODES",
    "R3_CONTROLS_PROFILE_PARTITION",
    "R3_CONTROLS_PROFILE_ROLE",
    "R3_CONTROLS_PROFILE_SCHEMA",
    "R3_CONTROLS_PROFILE_SLURM_GPU_TRES",
    "R3_CONTROLS_PROFILE_UPDATES_PER_FAMILY",
    "R3_CONTROLS_PROFILE_WALLTIME_SECONDS",
    "R3_CONTROLS_PROJECT_RELATIVE_ROOT",
    "R3_CONTROLS_TASK_CLOSURE_SCHEMA",
    "R3_CONTROLS_TERMINAL_SCHEMA",
    "R3_CONTROLS_TOTAL_TASKS",
    "build_controls_aggregate",
    "build_controls_authorization",
    "build_controls_execution_plan",
    "build_controls_operational_profile",
    "build_controls_task_closure",
    "build_controls_task_terminal",
    "build_profile_compute_receipt",
    "build_profile_family_measurement",
    "build_profile_family_measurement_from_compute_receipt",
    "build_profile_scheduler_terminal",
    "publish_controls_authorization",
    "publish_controls_execution_plan",
    "publish_controls_operational_profile",
    "publish_controls_task_closure",
    "publish_controls_task_terminal",
    "validate_controls_aggregate_structure",
    "validate_controls_authorization_structure",
    "validate_controls_execution_plan",
    "validate_controls_operational_profile",
    "validate_controls_task_closure",
    "validate_controls_task_terminal",
    "validate_profile_compute_receipt",
    "validate_profile_family_measurement",
]
