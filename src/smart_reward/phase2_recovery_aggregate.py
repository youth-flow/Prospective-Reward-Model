"""Head-free authorization from the frozen Phase-2 recovery execution.

This module is deliberately campaign-specific.  It accepts only the three
ordered SUCCESS directories produced by recovery array 1648125, execution
revision 2.  The recovery results are identity- and boundary-checked, but
their trained head values are never copied into the aggregate.  The sole
positive authorization is a fresh, schedule-frozen full calibration pilot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECOVERY_SUCCESS_AUTHORIZATION_SCHEMA = "prorm-phase2-recovery-success-authorization/v1"
RECOVERY_AGGREGATION_IDENTITY_SCHEMA = "prorm-phase2-recovery-aggregation-identity/v1"
RECOVERY_SCHEDULER_EVIDENCE_SCHEMA = "prorm-phase2-recovery-scheduler-terminal-evidence/v1"
RECOVERY_NAMESPACE_IDENTITY_SCHEMA = "prorm-phase2-recovery-namespace-identity/v1"
RECOVERY_LIVE_CONTROL_SCHEMA = "prorm-phase2-recovery-live-scontrol-raw/v1"

RECOVERY_DESIGN_SHA256 = "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"
SOURCE_CONFIG_HASH = "81ccbd3bc9d745d3792d1834116ba2480d34a7201d6f4e463a61d8d8ab0baefa"
OPTIMIZER_SCHEDULE_SHA256 = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
TRAINING_SETTINGS_SHA256 = "34574e1b1dc22a9503b89249059596d92aa5c3df074022ecfc8ff008dc4bc3af"
RECOVERY_GIT_COMMIT = "ad7613b7cef3ff536ec62f6f80608ee29e927b1c"
SOURCE_ARRAY_JOB_ID = "1648125"
EXECUTION_REVISION = 2
RETRY_REASON = "pretrainer_hf_datasets_runtime_lock"
ORDERED_SEEDS = (20260801, 20260802, 20260803)

PARENT_DESIGN_SHA256 = "0c8820b67b8ca85c23cd5c31b8d25001018b31a7271f6a861d88cfae2f85d7ca"
PARENT_REGISTRY_SHA256 = "7be4ee90b1f494d32f96214f407a57cbee54be86a77dacc1206d2acd527857dc"
PARENT_PRODUCER_GIT_COMMIT = "ae28e2a10f0bd5762899be01ce66bc5b423374cf"
IMAGE_SHA256 = "d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb"
HF_INVENTORY_SHA256 = "86c7c0fcab9cc0de612c6a5af05778e8b34617822b2e33474df8ed840eef82fd"

# This module authorizes one already-submitted campaign, not a relocatable
# experiment.  Production entry points intentionally expose no root override.
# Tests may monkeypatch this module constant.
PRODUCTION_PROJECT_ROOT = Path("/project/sigroup/smart-reward-model")
_RECOVERY_EXECUTION_RELATIVE = (
    Path("runs/phase2-recovery-pilot") / RECOVERY_DESIGN_SHA256 / f"execution-{EXECUTION_REVISION}"
)
_SCHEDULER_EVIDENCE_RELATIVE = Path("runs/phase2-recovery-pilot/recovery-1648125-terminal.json")
_AUTHORIZATION_RELATIVE = Path("runs/phase2-recovery-pilot/recovery-success-authorization.json")
_LIVE_CONTROL_RELATIVE = _RECOVERY_EXECUTION_RELATIVE / (
    "scheduler-control-live-20260725T153801+0800/scontrol-array-1648125.txt"
)
LIVE_CONTROL_SHA256 = "cb61484f435747d6705ff4567257afff2c447faa16144b697e9f9dcc03f83a5e"
LIVE_CONTROL_SIZE_BYTES = 4817
_LIVE_CONTROL_CAPTURED_AT = "2026-07-25T15:38:01+08:00"
_LIVE_CONTROL_COMMAND = (
    "scontrol show job -o 1648125_0; scontrol show job -o 1648125_1; scontrol show job -o 1648125_2"
)
# These digests bind the exact operator and checkout fields in the immutable
# live receipt without publishing the HPC login identity or home layout in
# source. LIVE_CONTROL_SHA256 independently binds the complete raw receipt.
_LIVE_USER_ID_SHA256 = "1a78fd915165ac22dd2226a847d08516ac778d6098b3dba34c4d898ade7d00a2"
_LIVE_COMMAND_SHA256 = "727a9412ec853ebaf0ddd094870e6e7648637b47429602f98901de5faf4a33df"
_LIVE_WORK_DIR_SHA256 = "c218562be8e6709056eaded46bb4cd4912913534b53c24afcc94f7273ddc0cba"
_LIVE_JOB_IDS = ("1648126", "1648203", "1648125")
_LIVE_STATES = ("RUNNING", "RUNNING", "PENDING")
_PARENT_REGISTRY_RELATIVE = Path("configs/phase2_recovery_parent_failures.json")
_PARENT_VALIDATOR_RELATIVE = Path("scripts/hpc4/validate_phase2_recovery_parent.py")
_PARENT_VALIDATOR_SHA256 = "6ec4746a2b2710be3e87abb4246e644e1ef19da36a60a7b09c93cab8c82bbb39"
_RECOVERY_CONFIG_RELATIVE = Path("configs/common_beta_recovery_pilot.yaml")
_RECOVERY_CONFIG_SHA256 = "a6a924dae429ceb0df11cea128542cae16fb42a2e69a0d2120acb0e4f8f1d80f"
_DEEP_GATE_SOURCE_RELATIVE = Path("src/smart_reward/phase2_aggregate.py")
_TENSOR_HASH_SOURCE_RELATIVE = Path("src/smart_reward/phase2_training.py")
_CONFIG_VALIDATOR_SOURCE_RELATIVE = Path("src/smart_reward/phase2_config.py")
_RECOVERY_TIE_BREAK = "exact_zero_initialized_deterministic_adamw_lr_decay_path"

_HEX = frozenset("0123456789abcdef")
_UTC_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_REQUIRED_EVIDENCE_FILES = frozenset(
    {
        "SUCCESS",
        "phase2-config-check.json",
        "base-config-check.json",
        "infrastructure-failure-verification.json",
        "parent-verification-before.json",
        "parent-verification-after.json",
        "artifact-snapshot-before.json",
        "artifact-snapshot-after.json",
        "parent-run-snapshot-before.json",
        "parent-run-snapshot-after.json",
        "hf-inventory-verification.json",
        "gpu-check.json",
        "run-manifest.json",
        "env-report.log",
        "recovery-train.log",
        "recovery-result.json",
        "recovery-output-verification.json",
    }
)
_REQUIRED_REFERENCE = "parent-artifact"
_SUCCESS_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "workload_exit_code",
        "final_exit_code",
        "array_job_id",
        "array_task_id",
        "seed",
        "execution_revision",
        "retry_reason",
        "recovery_design_sha256",
        "base_config_hash",
        "recovery_git_commit",
        "parent_design_sha256",
        "parent_registry_sha256",
        "parent_producer_git_commit",
        "one_shot_no_further_adaptation",
        "created_at_utc",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "design_stage",
        "evidence_role",
        "formal_eligibility",
        "per_seed_supports_formal_claim",
        "seed",
        "source_config_hash",
        "recovery_design_sha256",
        "recovery_execution_identity",
        "recovery_run_manifest_sha256",
        "parent_failure_binding",
        "artifact_reuse",
        "train_oracle_rescore",
        "head_training",
        "information_boundary",
        "one_shot_no_further_adaptation",
        "failure_action",
    }
)
_RESULT_BOUNDARY = {
    "train_tensors_decoded": True,
    "train_candidate_prefix_decoded": True,
    "validation_tensors_decoded": False,
    "test_tensors_decoded": False,
    "validation_or_test_candidates_decoded": False,
    "policy_session_opened": False,
    "policy_rollout_performed": False,
    "heldout_evaluator_called": False,
    "final_oracle_session_opened": False,
    "downstream_utility_computed": False,
}
_OUTPUT_VERIFICATION_KEYS = frozenset(
    {
        "status",
        "result_sha256",
        "five_head_recovery_protocol_verified",
        "selected_primary_steps",
        "diagnostic_seed_reproduction",
    }
)
_FIVE_HEAD_NAMES = frozenset(
    {
        "primary_bt_mle",
        "primary_prorm_plus",
        "low_dimensional_prorm_plus",
        "exact_margin_prorm_plus",
        "exact_soft_label_bt",
    }
)
_ENVIRONMENT_KEYS = frozenset(
    {
        "formal",
        "git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "account",
        "partition",
        "gpu_models",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "created_at_utc",
        "config_hash",
        "normalized_config",
        "seed",
        "selected_seed",
        "named_seeds",
        "git",
        "python",
        "platform",
        "torch",
        "revisions",
        "packages",
        "slurm",
    }
)
_MANIFEST_SLURM_REQUIRED_KEYS = frozenset(
    {
        "PRORM_GIT_COMMIT",
        "PRORM_IMAGE_SHA256",
        "PRORM_HF_INVENTORY_SHA256",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ID",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS_ON_NODE",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "CUDA_VISIBLE_DEVICES",
    }
)
_MANIFEST_SLURM_OPTIONAL_KEYS = frozenset(
    {
        "SLURM_GPUS",
        "NVIDIA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    }
)
_SACCT_REQUEST_TRES = "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1"
_SACCT_ALLOCATED_TRES = "billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1"
_SACCT_FORMAT = (
    "JobID,JobIDRaw,State,ExitCode,DerivedExitCode,Cluster,Account,Partition,"
    "NNodes,NCPUS,ReqTRES,AllocTRES"
)
_SACCT_COMMAND = (
    "sacct",
    "-X",
    "-n",
    "-P",
    "-j",
    SOURCE_ARRAY_JOB_ID,
    f"--format={_SACCT_FORMAT}",
)
_SCHEDULER_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "source_command",
        "array_job_id",
        "captured_at_utc",
        "raw_sacct",
        "rows",
    }
)
_SCHEDULER_ROW_KEYS = frozenset(
    {
        "job_id",
        "job_id_raw",
        "array_job_id",
        "array_task_id",
        "seed",
        "state",
        "exit_code",
        "derived_exit_code",
        "cluster",
        "account",
        "partition",
        "n_nodes",
        "n_cpus",
        "requested_tres",
        "allocated_tres",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version",
        "source_config_hash",
        "recovery_design_sha256",
        "optimizer_schedule_sha256",
        "training_settings_sha256",
        "source_array_job_id",
        "execution_revision",
        "ordered_seeds",
        "recovery_status",
        "full_calibration_authorized",
        "authorized_information",
        "authorized_next_action",
        "recovery_outputs_reusable",
        "validation_or_heldout_access_authorized",
        "policy_or_final_utility_access_authorized",
        "formal_efficacy_claim_authorized",
        "recovery_output_reuse",
        "information_boundary",
        "campaign_namespace",
        "scheduler_terminal",
        "supplementary_submission_control",
        "environment_identity",
        "parent_failure_identity",
        "aggregation_identity",
        "scheduler_source",
        "sources",
    }
)
_NAMESPACE_KEYS = frozenset(
    {
        "schema_version",
        "project_root",
        "recovery_execution_relative",
        "scheduler_terminal_relative",
        "supplementary_control_relative",
        "parent_registry_repository_relative",
        "authorization_relative",
    }
)
_LIVE_CONTROL_KEYS = frozenset(
    {
        "schema_version",
        "sha256",
        "size_bytes",
        "captured_at",
        "evidence_role",
        "terminal_status_authority",
        "terminal_status_source",
        "all_tasks_requeue_zero",
        "all_tasks_restarts_zero_at_capture",
        "rows",
    }
)
_LIVE_ROW_KEYS = frozenset(
    {
        "array_job_id",
        "array_task_id",
        "job_id_raw",
        "state_at_capture",
        "requeue",
        "restarts_at_capture",
        "account",
        "partition",
        "qos",
        "num_nodes",
        "num_cpus",
        "num_tasks",
        "cpus_per_task",
        "generic_gpu_count",
        "allocated_l20_count",
    }
)
_AGGREGATE_BOUNDARY = {
    "source_results_train_only": True,
    "validation_tensors_decoded": False,
    "test_tensors_decoded": False,
    "validation_or_test_candidates_decoded": False,
    "policy_rollout_performed": False,
    "heldout_evaluator_called": False,
    "final_oracle_session_opened": False,
    "downstream_utility_computed": False,
    "trained_parameters_extracted_for_aggregation": False,
    "trained_parameters_serialized_in_aggregate": False,
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _exact_keys(
    value: object,
    *,
    name: str,
    expected: set[str] | frozenset[str],
) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    if set(result) != set(expected):
        raise ValueError(
            f"{name} keys differ from the locked schema; "
            f"missing={sorted(set(expected) - set(result))!r}, "
            f"unknown={sorted(set(result) - set(expected))!r}"
        )
    return result


def _digest(
    value: object,
    *,
    name: str,
    lengths: frozenset[int] = frozenset({64}),
) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal digest")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).removesuffix(b"\n")).hexdigest()


def _read_stable_file(
    path: Path,
    *,
    name: str,
    maximum_bytes: int | None = None,
) -> bytes:
    """Read one regular file through a no-follow descriptor and verify stability."""

    _require_real_file(path, name=name)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} must be a regular non-symlink file: {path}")
        if maximum_bytes is not None and (before.st_size <= 0 or before.st_size > maximum_bytes):
            raise ValueError(f"{name} has an invalid byte length")
        chunks: list[bytes] = []
        remaining = maximum_bytes
        while True:
            read_size = 1024 * 1024
            if remaining is not None:
                read_size = min(read_size, remaining + 1)
            chunk = os.read(descriptor, read_size)
            if not chunk:
                break
            chunks.append(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError(f"{name} has an invalid byte length")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"{name} changed while it was being read")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise ValueError(f"{name} changed length while it was being read")
    return raw


def _sha256_file(path: Path, *, name: str = "source file") -> str:
    _require_real_file(path, name=name)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    count = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} must be a regular non-symlink file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or count != after.st_size:
        raise ValueError(f"{name} changed while it was being hashed")
    return digest.hexdigest()


def _require_real_file(path: Path, *, name: str) -> None:
    try:
        resolved = path.resolve(strict=True)
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{name} is missing: {path}") from error
    if not path.is_absolute() or path != resolved or not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")


def _require_real_directory(path: Path, *, name: str) -> None:
    try:
        resolved = path.resolve(strict=True)
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{name} is missing: {path}") from error
    if not path.is_absolute() or path != resolved or not stat.S_ISDIR(mode):
        raise ValueError(f"{name} must be a real non-symlink directory: {path}")


def _read_json(
    path: Path,
    *,
    name: str,
    require_canonical: bool = False,
    maximum_bytes: int = 256 * 1024 * 1024,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_stable_file(path, name=name, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {item!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    if require_canonical and raw != _canonical_bytes(value):
        raise ValueError(f"{name} is not canonical JSON")
    return value, raw


def _parse_success(path: Path) -> tuple[dict[str, str], bytes]:
    raw = _read_stable_file(
        path,
        name="SUCCESS marker",
        maximum_bytes=64 * 1024,
    )
    if not raw.endswith(b"\n"):
        raise ValueError("SUCCESS marker must be non-empty, bounded, and newline-terminated")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("SUCCESS marker is not UTF-8") from error
    fields: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line or "=" not in line:
            raise ValueError(f"SUCCESS marker line {line_number} is malformed")
        key, value = line.split("=", 1)
        if not key or key in fields:
            raise ValueError(f"SUCCESS marker line {line_number} has a duplicate/empty key")
        fields[key] = value
    _exact_keys(fields, name="SUCCESS marker", expected=_SUCCESS_KEYS)
    return fields, raw


def _validate_success(
    fields: Mapping[str, str],
    *,
    seed: int,
    array_task_id: int,
) -> None:
    expected = {
        "schema_version": "prorm-phase2-recovery-run-status/v1",
        "status": "SUCCESS",
        "workload_exit_code": "0",
        "final_exit_code": "0",
        "array_job_id": SOURCE_ARRAY_JOB_ID,
        "array_task_id": str(array_task_id),
        "seed": str(seed),
        "execution_revision": str(EXECUTION_REVISION),
        "retry_reason": RETRY_REASON,
        "recovery_design_sha256": RECOVERY_DESIGN_SHA256,
        "base_config_hash": SOURCE_CONFIG_HASH,
        "recovery_git_commit": RECOVERY_GIT_COMMIT,
        "parent_design_sha256": PARENT_DESIGN_SHA256,
        "parent_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_producer_git_commit": PARENT_PRODUCER_GIT_COMMIT,
        "one_shot_no_further_adaptation": "true",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ValueError(f"SUCCESS marker {key} differs from recovery execution 2")
    if _UTC_TIMESTAMP.fullmatch(fields.get("created_at_utc", "")) is None:
        raise ValueError("SUCCESS marker created_at_utc is invalid")


def _parse_sacct_raw(raw: bytes) -> list[dict[str, object]]:
    if not raw or len(raw) > 1024 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("raw sacct evidence must be non-empty, bounded, and newline-terminated")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("raw sacct evidence is not UTF-8") from error
    if len(lines) != len(ORDERED_SEEDS):
        raise ValueError("raw sacct evidence must contain exactly three allocation rows")

    rows: list[dict[str, object]] = []
    raw_job_ids: set[str] = set()
    for array_task_id, (seed, line) in enumerate(zip(ORDERED_SEEDS, lines, strict=True)):
        fields = line.split("|")
        if len(fields) != 12:
            raise ValueError(f"raw sacct row {array_task_id} has an invalid parsable layout")
        (
            job_id,
            job_id_raw,
            state,
            exit_code,
            derived_exit_code,
            cluster,
            account,
            partition,
            n_nodes,
            n_cpus,
            requested_tres,
            allocated_tres,
        ) = fields
        expected = (
            f"{SOURCE_ARRAY_JOB_ID}_{array_task_id}",
            "COMPLETED",
            "0:0",
            "0:0",
            "hpc4",
            "sigroup",
            "gpu-l20",
            "1",
            "8",
            _SACCT_REQUEST_TRES,
            _SACCT_ALLOCATED_TRES,
        )
        if (
            (
                job_id,
                state,
                exit_code,
                derived_exit_code,
                cluster,
                account,
                partition,
                n_nodes,
                n_cpus,
                requested_tres,
                allocated_tres,
            )
            != expected
            or re.fullmatch(r"[1-9][0-9]*", job_id_raw) is None
            or job_id_raw in raw_job_ids
        ):
            raise ValueError(
                f"raw sacct row {array_task_id} is not the exact successful task allocation"
            )
        raw_job_ids.add(job_id_raw)
        rows.append(
            {
                "job_id": job_id,
                "job_id_raw": job_id_raw,
                "array_job_id": SOURCE_ARRAY_JOB_ID,
                "array_task_id": array_task_id,
                "seed": seed,
                "state": state,
                "exit_code": exit_code,
                "derived_exit_code": derived_exit_code,
                "cluster": cluster,
                "account": account,
                "partition": partition,
                "n_nodes": int(n_nodes),
                "n_cpus": int(n_cpus),
                "requested_tres": requested_tres,
                "allocated_tres": allocated_tres,
            }
        )
    return rows


def _project_root() -> Path:
    root = Path(PRODUCTION_PROJECT_ROOT)
    _require_real_directory(root, name="frozen production project root")
    return root


def _exact_project_path(relative: Path, *, name: str, must_exist: bool) -> Path:
    root = _project_root()
    expected = root / relative
    if not expected.is_absolute() or expected.parts[: len(root.parts)] != root.parts:
        raise ValueError(f"{name} escaped the frozen production project root")
    if must_exist:
        resolved = expected.resolve(strict=True)
        if resolved != expected:
            raise ValueError(f"{name} is not canonical: {expected}")
    return expected


def _require_exact_project_path(
    path: Path,
    relative: Path,
    *,
    name: str,
    must_exist: bool = True,
) -> Path:
    expected = _exact_project_path(relative, name=name, must_exist=must_exist)
    if path != expected:
        raise ValueError(f"{name} is outside its frozen campaign namespace")
    return expected


def _campaign_namespace_identity() -> dict[str, object]:
    root = _project_root()
    return {
        "schema_version": RECOVERY_NAMESPACE_IDENTITY_SCHEMA,
        "project_root": os.fspath(root),
        "recovery_execution_relative": _RECOVERY_EXECUTION_RELATIVE.as_posix(),
        "scheduler_terminal_relative": _SCHEDULER_EVIDENCE_RELATIVE.as_posix(),
        "supplementary_control_relative": _LIVE_CONTROL_RELATIVE.as_posix(),
        "parent_registry_repository_relative": _PARENT_REGISTRY_RELATIVE.as_posix(),
        "authorization_relative": _AUTHORIZATION_RELATIVE.as_posix(),
    }


def _parse_scontrol_line(line: str, *, task: int) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            raise ValueError(f"live scontrol task {task} contains a malformed token")
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise ValueError(f"live scontrol task {task} has a duplicate/empty field")
        fields[key] = value
    return fields


def _utf8_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_live_control(raw: bytes) -> dict[str, object]:
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("live scontrol receipt is empty, oversized, or not newline-terminated")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("live scontrol receipt is not UTF-8") from error
    if len(lines) != 6:
        raise ValueError("live scontrol receipt must contain three headers and three task rows")
    if lines[:3] != [
        f"schema={RECOVERY_LIVE_CONTROL_SCHEMA}",
        f"captured_at={_LIVE_CONTROL_CAPTURED_AT}",
        f"command={_LIVE_CONTROL_COMMAND}",
    ]:
        raise ValueError("live scontrol receipt headers differ from the frozen capture")

    rows: list[dict[str, object]] = []
    for task, line in enumerate(lines[3:]):
        fields = _parse_scontrol_line(line, task=task)
        expected_nodes = "1-1" if task == 2 else "1"
        expected_l20 = 0 if task == 2 else 1
        tres = fields.get("TRES", "").split(",")
        if (
            fields.get("JobId") != _LIVE_JOB_IDS[task]
            or fields.get("ArrayJobId") != SOURCE_ARRAY_JOB_ID
            or fields.get("ArrayTaskId") != str(task)
            or fields.get("ArrayTaskThrottle") != "3"
            or fields.get("JobName") != "prorm-p2-recovery"
            or _utf8_sha256(fields.get("UserId", "")) != _LIVE_USER_ID_SHA256
            or fields.get("Account") != "sigroup"
            or fields.get("QOS") != "l20_qos"
            or fields.get("JobState") != _LIVE_STATES[task]
            or fields.get("Requeue") != "0"
            or fields.get("Restarts") != "0"
            or fields.get("BatchFlag") != "1"
            or fields.get("Reboot") != "0"
            or fields.get("TimeLimit") != "12:00:00"
            or fields.get("Partition") != "gpu-l20"
            or fields.get("NumNodes") != expected_nodes
            or fields.get("NumCPUs") != "8"
            or fields.get("NumTasks") != "1"
            or fields.get("CPUs/Task") != "8"
            or "gres/gpu=1" not in tres
            or ("gres/gpu:l20=1" in tres) is not (task != 2)
            or fields.get("TresPerNode") != "gres:gpu:1"
            or _utf8_sha256(fields.get("Command", "")) != _LIVE_COMMAND_SHA256
            or _utf8_sha256(fields.get("WorkDir", "")) != _LIVE_WORK_DIR_SHA256
        ):
            raise ValueError(f"live scontrol task {task} violates the frozen submission receipt")
        rows.append(
            {
                "array_job_id": SOURCE_ARRAY_JOB_ID,
                "array_task_id": task,
                "job_id_raw": _LIVE_JOB_IDS[task],
                "state_at_capture": _LIVE_STATES[task],
                "requeue": 0,
                "restarts_at_capture": 0,
                "account": "sigroup",
                "partition": "gpu-l20",
                "qos": "l20_qos",
                "num_nodes": 1,
                "num_cpus": 8,
                "num_tasks": 1,
                "cpus_per_task": 8,
                "generic_gpu_count": 1,
                "allocated_l20_count": expected_l20,
            }
        )
    return {
        "schema_version": RECOVERY_LIVE_CONTROL_SCHEMA,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "captured_at": _LIVE_CONTROL_CAPTURED_AT,
        "evidence_role": "supplementary_submission_and_live_control_receipt",
        "terminal_status_authority": False,
        "terminal_status_source": "terminal_sacct_and_success_receipts_only",
        "all_tasks_requeue_zero": True,
        "all_tasks_restarts_zero_at_capture": True,
        "rows": rows,
    }


def _expected_live_control() -> dict[str, object]:
    """Return the complete immutable projection safe for offline consumers."""

    rows = []
    for task, (job_id_raw, state) in enumerate(zip(_LIVE_JOB_IDS, _LIVE_STATES, strict=True)):
        rows.append(
            {
                "array_job_id": SOURCE_ARRAY_JOB_ID,
                "array_task_id": task,
                "job_id_raw": job_id_raw,
                "state_at_capture": state,
                "requeue": 0,
                "restarts_at_capture": 0,
                "account": "sigroup",
                "partition": "gpu-l20",
                "qos": "l20_qos",
                "num_nodes": 1,
                "num_cpus": 8,
                "num_tasks": 1,
                "cpus_per_task": 8,
                "generic_gpu_count": 1,
                "allocated_l20_count": 0 if task == 2 else 1,
            }
        )
    return {
        "schema_version": RECOVERY_LIVE_CONTROL_SCHEMA,
        "sha256": LIVE_CONTROL_SHA256,
        "size_bytes": LIVE_CONTROL_SIZE_BYTES,
        "captured_at": _LIVE_CONTROL_CAPTURED_AT,
        "evidence_role": "supplementary_submission_and_live_control_receipt",
        "terminal_status_authority": False,
        "terminal_status_source": "terminal_sacct_and_success_receipts_only",
        "all_tasks_requeue_zero": True,
        "all_tasks_restarts_zero_at_capture": True,
        "rows": rows,
    }


def _load_live_control() -> dict[str, object]:
    path = _exact_project_path(
        _LIVE_CONTROL_RELATIVE,
        name="supplementary live scontrol receipt",
        must_exist=True,
    )
    _require_real_file(path, name="supplementary live scontrol receipt")
    mode = stat.S_IMODE(path.lstat().st_mode)
    if os.name == "posix" and mode != 0o440:
        raise ValueError("supplementary live scontrol receipt must retain mode 0440")
    raw = _read_stable_file(
        path,
        name="supplementary live scontrol receipt",
        maximum_bytes=64 * 1024,
    )
    if (
        hashlib.sha256(raw).hexdigest() != LIVE_CONTROL_SHA256
        or len(raw) != LIVE_CONTROL_SIZE_BYTES
    ):
        raise ValueError("supplementary live scontrol receipt bytes changed")
    return _parse_live_control(raw)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_bytes(path: Path, payload: bytes, *, label: str) -> str:
    """Publish one immutable file and verify the exact linked inode by descriptor."""

    _require_real_directory(path.parent, name=f"{label} output parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing {label}: {path}")
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    parent_stat = path.parent.stat()
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
    directory_fd: int | None = None
    temporary_name: str | None = None
    temporary_path: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    destination_linked = False
    publication_complete = False

    def require_same_parent() -> None:
        current = path.parent.stat()
        if (current.st_dev, current.st_ino) != parent_identity:
            raise ValueError(f"{label} output parent changed during publication")
        if directory_fd is not None:
            opened = os.fstat(directory_fd)
            if (opened.st_dev, opened.st_ino) != parent_identity:
                raise ValueError(f"{label} held output-directory descriptor changed")

    def unlink_name(name: str) -> None:
        if directory_fd is not None:
            os.unlink(name, dir_fd=directory_fd)
        else:
            (path.parent / name).unlink()

    try:
        if os.name == "posix":
            directory_fd = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            require_same_parent()
            descriptor = -1
            for _ in range(128):
                temporary_name = f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
                try:
                    descriptor = os.open(
                        temporary_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                        0o440,
                        dir_fd=directory_fd,
                    )
                    break
                except FileExistsError:
                    continue
            if descriptor < 0:
                raise FileExistsError(f"could not reserve a temporary {label} inode")
        else:
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary_path = Path(raw_temporary)
            temporary_name = temporary_path.name
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o440)
            os.fsync(stream.fileno())
            temporary_stat = os.fstat(stream.fileno())
            temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
            if temporary_stat.st_size != len(payload):
                raise OSError(f"{label} temporary inode has the wrong byte size")
        require_same_parent()
        try:
            if directory_fd is not None:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            else:
                if temporary_path is None:
                    raise RuntimeError("temporary publication path was not initialized")
                os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite existing {label}: {path}") from error
        destination_linked = True
        require_same_parent()
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        if directory_fd is not None:
            published_fd = os.open(path.name, read_flags, dir_fd=directory_fd)
        else:
            if path.is_symlink():
                raise ValueError(f"published {label} unexpectedly became a symlink")
            published_fd = os.open(path, read_flags)
        try:
            published_stat = os.fstat(published_fd)
            published = bytearray()
            while True:
                chunk = os.read(published_fd, 1024 * 1024)
                if not chunk:
                    break
                published.extend(chunk)
        finally:
            os.close(published_fd)
        if (
            temporary_identity is None
            or (published_stat.st_dev, published_stat.st_ino) != temporary_identity
            or published_stat.st_size != len(payload)
            or bytes(published) != payload
            or hashlib.sha256(published).hexdigest() != expected_sha256
            or (os.name == "posix" and stat.S_IMODE(published_stat.st_mode) != 0o440)
        ):
            raise ValueError(f"published {label} inode failed byte/inode verification")
        unlink_name(temporary_name)
        temporary_name = None
        require_same_parent()
        if directory_fd is not None:
            os.fsync(directory_fd)
        else:
            _fsync_directory(path.parent)
        publication_complete = True
        return expected_sha256
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                unlink_name(temporary_name)
        if destination_linked and not publication_complete and temporary_identity is not None:
            # Publication failed after link creation. Remove only our own inode;
            # never unlink an attacker-substituted destination.
            with suppress(FileNotFoundError):
                destination_stat = path.lstat()
                if (
                    not stat.S_ISLNK(destination_stat.st_mode)
                    and (destination_stat.st_dev, destination_stat.st_ino) == temporary_identity
                ):
                    if directory_fd is not None:
                        os.unlink(path.name, dir_fd=directory_fd)
                    else:
                        path.unlink()
        if directory_fd is not None:
            with suppress(OSError):
                os.fsync(directory_fd)
            os.close(directory_fd)
        elif temporary_name is not None or destination_linked:
            _fsync_directory(path.parent)


def capture_phase2_recovery_scheduler_evidence_with_digest(
    output_json: str | os.PathLike[str],
    *,
    now: datetime | None = None,
) -> tuple[dict[str, object], str]:
    """Run the locked unfiltered ``sacct`` query and preserve its raw bytes."""

    destination = Path(output_json).absolute()
    _require_exact_project_path(
        destination,
        _SCHEDULER_EVIDENCE_RELATIVE,
        name="scheduler evidence destination",
        must_exist=False,
    )
    # This is deliberately supplementary: it proves submission/live
    # Requeue/Restarts/resources but never substitutes for terminal sacct.
    _load_live_control()
    raw_path = destination.with_name(f"{destination.stem}.sacct.psv")
    if destination == raw_path:
        raise ValueError("scheduler JSON and raw evidence paths must be distinct")
    if (
        destination.exists()
        or destination.is_symlink()
        or raw_path.exists()
        or raw_path.is_symlink()
    ):
        raise FileExistsError("refusing to overwrite recovery scheduler evidence")
    try:
        completed = subprocess.run(
            list(_SACCT_COMMAND),
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not execute the locked sacct query") from error
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("the locked sacct query did not complete without stderr")
    raw = bytes(completed.stdout)
    rows = _parse_sacct_raw(raw)
    timestamp = datetime.now(timezone.utc) if now is None else now
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("scheduler capture timestamp must be timezone-aware")
    created_at = (
        timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    payload: dict[str, object] = {
        "schema_version": RECOVERY_SCHEDULER_EVIDENCE_SCHEMA,
        "source_command": list(_SACCT_COMMAND),
        "array_job_id": SOURCE_ARRAY_JOB_ID,
        "captured_at_utc": created_at,
        "raw_sacct": {
            "filename": raw_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
        "rows": rows,
    }
    # Publish the raw source first and its canonical interpretation last.  A
    # JSON file can therefore never exist without its exact raw byte source.
    _write_exclusive_bytes(raw_path, raw, label="raw sacct evidence")
    try:
        digest = _write_exclusive_bytes(
            destination,
            _canonical_bytes(payload),
            label="scheduler evidence JSON",
        )
    except BaseException:
        with suppress(FileNotFoundError):
            raw_path.unlink()
        _fsync_directory(raw_path.parent)
        raise
    return payload, digest


def capture_phase2_recovery_scheduler_evidence(
    output_json: str | os.PathLike[str],
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Capture terminal scheduler evidence; retain the historical dict API."""

    payload, _ = capture_phase2_recovery_scheduler_evidence_with_digest(
        output_json,
        now=now,
    )
    return payload


def _validate_scheduler_evidence(
    path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    _require_exact_project_path(
        path,
        _SCHEDULER_EVIDENCE_RELATIVE,
        name="recovery scheduler evidence",
    )
    value, encoded = _read_json(
        path,
        name="recovery scheduler evidence",
        require_canonical=True,
        maximum_bytes=1024 * 1024,
    )
    evidence = _exact_keys(
        value,
        name="recovery scheduler evidence",
        expected=_SCHEDULER_EVIDENCE_KEYS,
    )
    raw_binding = _exact_keys(
        evidence["raw_sacct"],
        name="recovery scheduler evidence.raw_sacct",
        expected={"filename", "sha256", "size_bytes"},
    )
    expected_raw_name = f"{path.stem}.sacct.psv"
    if (
        evidence["schema_version"] != RECOVERY_SCHEDULER_EVIDENCE_SCHEMA
        or evidence["source_command"] != list(_SACCT_COMMAND)
        or evidence["array_job_id"] != SOURCE_ARRAY_JOB_ID
        or _UTC_TIMESTAMP.fullmatch(str(evidence["captured_at_utc"])) is None
        or raw_binding["filename"] != expected_raw_name
    ):
        raise ValueError("scheduler evidence does not bind the locked sacct capture")
    raw_path = path.with_name(expected_raw_name)
    _require_real_file(raw_path, name="raw sacct evidence")
    raw = _read_stable_file(
        raw_path,
        name="raw sacct evidence",
        maximum_bytes=1024 * 1024,
    )
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_binding["sha256"] != raw_sha256 or raw_binding["size_bytes"] != len(raw):
        raise ValueError("scheduler evidence does not bind its raw sacct bytes")
    rows = _parse_sacct_raw(raw)
    if tuple(str(row["job_id_raw"]) for row in rows) != _LIVE_JOB_IDS:
        raise ValueError("terminal sacct allocation IDs differ from live control evidence")
    claimed_rows = evidence["rows"]
    if not isinstance(claimed_rows, list) or claimed_rows != rows:
        raise ValueError("scheduler evidence rows differ from the raw sacct bytes")
    for index, row in enumerate(claimed_rows):
        _exact_keys(
            row,
            name=f"recovery scheduler evidence.rows[{index}]",
            expected=_SCHEDULER_ROW_KEYS,
        )
    source = {
        "scheduler_evidence_sha256": hashlib.sha256(encoded).hexdigest(),
        "raw_sacct_sha256": raw_sha256,
        "raw_sacct_size_bytes": len(raw),
    }
    return {"rows": rows, **source}, source


def _validate_run_path(path: Path, *, seed: int, array_task_id: int) -> None:
    expected_names = (
        f"job-{SOURCE_ARRAY_JOB_ID}_{array_task_id}",
        f"seed-{seed}",
        f"execution-{EXECUTION_REVISION}",
        RECOVERY_DESIGN_SHA256,
        "phase2-recovery-pilot",
    )
    current = path
    for depth, expected in enumerate(expected_names):
        _require_real_directory(current, name=f"run namespace component {depth}")
        if current.name != expected:
            raise ValueError(
                f"ordered run directory has invalid namespace component: "
                f"expected {expected!r}, observed {current.name!r}"
            )
        current = current.parent


def _snapshot_evidence(path: Path) -> list[dict[str, object]]:
    entries = {entry.name: entry for entry in path.iterdir()}
    allowed = set(_REQUIRED_EVIDENCE_FILES) | {_REQUIRED_REFERENCE}
    if set(entries) - allowed:
        raise ValueError(
            f"recovery run contains unexpected evidence: {sorted(set(entries) - allowed)!r}"
        )
    missing = allowed - set(entries)
    if missing:
        raise ValueError(f"recovery run is missing required evidence: {sorted(missing)!r}")

    inventory: list[dict[str, object]] = []
    for name in sorted(_REQUIRED_EVIDENCE_FILES):
        evidence = entries[name]
        _require_real_file(evidence, name=f"source evidence {name}")
        size = evidence.stat().st_size
        inventory.append(
            {
                "name": name,
                "kind": "regular_file",
                "size_bytes": size,
                "sha256": _sha256_file(evidence),
            }
        )
    reference = entries[_REQUIRED_REFERENCE]
    if not reference.is_symlink():
        raise ValueError("parent-artifact must be the required symlink reference")
    target = os.readlink(reference)
    if os.path.isabs(target):
        raise ValueError("parent-artifact must retain the job's relative reference")
    target_bytes = os.fsencode(target)
    inventory.append(
        {
            "name": _REQUIRED_REFERENCE,
            "kind": "symlink_reference",
            "size_bytes": len(target_bytes),
            "sha256": hashlib.sha256(target_bytes).hexdigest(),
        }
    )
    return sorted(inventory, key=lambda item: str(item["name"]))


def _inventory_file(
    inventory: Sequence[Mapping[str, object]],
    name: str,
) -> Mapping[str, object]:
    matches = [entry for entry in inventory if entry.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"evidence inventory does not contain exactly one {name}")
    return matches[0]


def _validate_manifest(
    value: object,
    *,
    seed: int,
    array_task_id: int,
    scheduler_row: Mapping[str, object],
) -> dict[str, object]:
    manifest = _exact_keys(value, name="run-manifest.json", expected=_MANIFEST_KEYS)
    git = _exact_keys(
        manifest["git"],
        name="run-manifest.json.git",
        expected={"commit", "dirty"},
    )
    slurm = _mapping(manifest["slurm"], name="run-manifest.json.slurm")
    if not _MANIFEST_SLURM_REQUIRED_KEYS.issubset(slurm) or not set(slurm).issubset(
        _MANIFEST_SLURM_REQUIRED_KEYS | _MANIFEST_SLURM_OPTIONAL_KEYS
    ):
        raise ValueError("run-manifest.json Slurm fields differ from the real allowlist schema")
    torch_state = _mapping(manifest["torch"], name="run-manifest.json.torch")
    gpus = torch_state.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != 1:
        raise ValueError("run-manifest.json must contain exactly one GPU record")
    gpu = _exact_keys(
        gpus[0],
        name="run-manifest.json.torch.gpus[0]",
        expected={"index", "name", "total_memory_bytes", "compute_capability"},
    )
    if set(torch_state) != {
        "installed",
        "version",
        "cuda_available",
        "cuda_version",
        "cudnn_version",
        "gpu_count",
        "gpus",
    }:
        raise ValueError("run-manifest.json torch fields differ from the real manifest schema")
    raw_job_id = scheduler_row.get("job_id_raw")
    n_nodes = scheduler_row.get("n_nodes")
    n_cpus = scheduler_row.get("n_cpus")
    if (
        manifest["schema_version"] != "smart-reward-run/v1"
        or manifest["config_hash"] != SOURCE_CONFIG_HASH
        or manifest["seed"] != list(ORDERED_SEEDS)
        or manifest["selected_seed"] != seed
        or git["commit"] != RECOVERY_GIT_COMMIT
        or git["dirty"] is not False
        or slurm.get("PRORM_GIT_COMMIT") != RECOVERY_GIT_COMMIT
        or slurm.get("PRORM_IMAGE_SHA256") != IMAGE_SHA256
        or slurm.get("PRORM_HF_INVENTORY_SHA256") != HF_INVENTORY_SHA256
        or slurm.get("SLURM_ARRAY_JOB_ID") != SOURCE_ARRAY_JOB_ID
        or slurm.get("SLURM_ARRAY_TASK_ID") != str(array_task_id)
        or slurm.get("SLURM_JOB_ID") != raw_job_id
        or slurm.get("SLURM_CLUSTER_NAME") != "hpc4"
        or slurm.get("SLURM_JOB_ACCOUNT") != "sigroup"
        or slurm.get("SLURM_JOB_PARTITION") != "gpu-l20"
        or slurm.get("SLURM_JOB_NAME") != "prorm-p2-recovery"
        or slurm.get("SLURM_NNODES") != str(n_nodes)
        or slurm.get("SLURM_NTASKS") != "1"
        or slurm.get("SLURM_CPUS_PER_TASK") != "8"
        or slurm.get("SLURM_GPUS_ON_NODE") != "1"
        or slurm.get("CUDA_VISIBLE_DEVICES") != "0"
        or not slurm.get("SLURM_JOB_NODELIST")
        or slurm.get("SLURM_PROCID") != "0"
        or slurm.get("SLURM_LOCALID") != "0"
        or slurm.get("SLURM_NODEID") != "0"
        or n_nodes != 1
        or n_cpus != 8
        or scheduler_row.get("requested_tres") != _SACCT_REQUEST_TRES
        or scheduler_row.get("allocated_tres") != _SACCT_ALLOCATED_TRES
        or torch_state.get("installed") is not True
        or torch_state.get("version") != "2.7.1+cu126"
        or torch_state.get("cuda_available") is not True
        or torch_state.get("cuda_version") != "12.6"
        or isinstance(torch_state.get("cudnn_version"), bool)
        or not isinstance(torch_state.get("cudnn_version"), int)
        or int(torch_state["cudnn_version"]) <= 0
        or torch_state.get("gpu_count") != 1
        or gpu
        != {
            "index": 0,
            "name": "NVIDIA L20",
            "total_memory_bytes": 47676129280,
            "compute_capability": "8.9",
        }
    ):
        raise ValueError("run-manifest.json does not bind the frozen recovery execution")
    return {
        "formal": True,
        "git_commit": RECOVERY_GIT_COMMIT,
        "image_sha256": IMAGE_SHA256,
        "hf_inventory_sha256": HF_INVENTORY_SHA256,
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_models": ["NVIDIA L20"],
    }


def _validate_gpu_check(
    value: object,
    *,
    manifest_environment: Mapping[str, object],
    scheduler_row: Mapping[str, object],
) -> None:
    gpu = _exact_keys(
        value,
        name="gpu-check.json",
        expected={"status", "gpu_model", "cuda_device_count"},
    )
    if (
        gpu
        != {
            "status": "ok",
            "gpu_model": "NVIDIA L20",
            "cuda_device_count": 1,
        }
        or manifest_environment.get("gpu_models") != [gpu["gpu_model"]]
        or scheduler_row.get("allocated_tres") != _SACCT_ALLOCATED_TRES
        or scheduler_row.get("n_nodes") != 1
        or scheduler_row.get("n_cpus") != 8
    ):
        raise ValueError("gpu-check.json, manifest, and terminal Slurm resources do not agree")


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    _require_real_directory(root, name="recovery authorization repository root")
    return root


def _git_blob(commit: str, relative: Path, *, name: str) -> bytes:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(_repository_root()),
                "cat-file",
                "blob",
                f"{commit}:{relative.as_posix()}",
            ],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"could not inspect {name} Git blob") from error
    if completed.returncode != 0 or completed.stderr:
        raise ValueError(f"{name} is not available as the exact claimed Git blob")
    return bytes(completed.stdout)


def _verify_frozen_parent_support() -> tuple[Path, Path]:
    root = _repository_root()
    registry = root / _PARENT_REGISTRY_RELATIVE
    validator = root / _PARENT_VALIDATOR_RELATIVE
    _require_real_file(registry, name="frozen parent registry")
    _require_real_file(validator, name="frozen parent validator")
    registry_raw = _read_stable_file(
        registry,
        name="frozen parent registry",
        maximum_bytes=4 * 1024 * 1024,
    )
    validator_raw = _read_stable_file(
        validator,
        name="frozen parent validator",
        maximum_bytes=4 * 1024 * 1024,
    )
    registry_blob = _git_blob(
        RECOVERY_GIT_COMMIT,
        _PARENT_REGISTRY_RELATIVE,
        name="frozen parent registry",
    )
    validator_blob = _git_blob(
        RECOVERY_GIT_COMMIT,
        _PARENT_VALIDATOR_RELATIVE,
        name="frozen parent validator",
    )
    if (
        registry_raw != registry_blob
        or hashlib.sha256(registry_raw).hexdigest() != PARENT_REGISTRY_SHA256
        or validator_raw != validator_blob
        or hashlib.sha256(validator_raw).hexdigest() != _PARENT_VALIDATOR_SHA256
    ):
        raise ValueError("frozen parent registry/validator differs from recovery Git blobs")
    return registry, validator


def _derive_parent_verification(seed: int) -> tuple[dict[str, object], bytes]:
    registry, validator = _verify_frozen_parent_support()
    command = [
        sys.executable,
        "-I",
        "-S",
        os.fspath(validator),
        os.fspath(registry),
        "--project-root",
        os.fspath(_project_root()),
        "--expected-registry-sha256",
        PARENT_REGISTRY_SHA256,
        "--expected-parent-design-sha256",
        PARENT_DESIGN_SHA256,
        "--expected-base-config-hash",
        SOURCE_CONFIG_HASH,
        "--seed",
        str(seed),
        "--verify-sources",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not rerun the frozen parent validator") from error
    raw = bytes(completed.stdout)
    if completed.returncode != 0 or completed.stderr or not raw or len(raw) > 16 * 1024 * 1024:
        raise ValueError("frozen parent validator did not produce one clean receipt")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {item!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen parent validator output is not strict JSON") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError("frozen parent validator output is not canonical")
    return value, raw


def _validate_snapshot_record_digest(
    record: Mapping[str, object],
    *,
    expected_name: str,
    expected_sha256: str,
    location: str,
) -> Mapping[str, object]:
    entry = _exact_keys(
        record,
        name=location,
        expected={"name", "sha256", "size", "mtime_ns"},
    )
    if entry["name"] != expected_name or entry["sha256"] != expected_sha256:
        raise ValueError(f"{location} differs from the frozen parent registry")
    _integer(entry["size"], name=f"{location}.size")
    _integer(entry["mtime_ns"], name=f"{location}.mtime_ns")
    return entry


def _validate_parent_snapshots(
    run_path: Path,
    *,
    selected: Mapping[str, object],
) -> None:
    artifact_hashes = _mapping(
        selected.get("artifact_sha256"),
        name="parent selected_seed.artifact_sha256",
    )
    evidence_hashes = _mapping(
        selected.get("evidence_sha256"),
        name="parent selected_seed.evidence_sha256",
    )
    artifact_before, artifact_before_raw = _read_json(
        run_path / "artifact-snapshot-before.json",
        name="artifact snapshot before",
        require_canonical=True,
        maximum_bytes=1024 * 1024,
    )
    artifact_after, artifact_after_raw = _read_json(
        run_path / "artifact-snapshot-after.json",
        name="artifact snapshot after",
        require_canonical=True,
        maximum_bytes=1024 * 1024,
    )
    if artifact_before_raw != artifact_after_raw:
        raise ValueError("parent artifact snapshot changed during recovery")
    artifact_snapshot = _exact_keys(
        artifact_before,
        name="parent artifact snapshot",
        expected={"schema_version", "records"},
    )
    records = artifact_snapshot["records"]
    expected_artifact_names = {
        "metadata.json",
        "tensors.safetensors",
        "candidates.jsonl",
        "prompts.jsonl",
        "training_edges.jsonl",
        "evaluation_edges.jsonl",
    }
    if (
        artifact_snapshot["schema_version"] != "prorm-read-only-artifact-snapshot/v1"
        or not isinstance(records, list)
        or [item.get("name") for item in records if isinstance(item, Mapping)]
        != sorted(expected_artifact_names)
        or len(records) != len(expected_artifact_names)
    ):
        raise ValueError("parent artifact snapshot schema is invalid")
    for index, filename in enumerate(sorted(expected_artifact_names)):
        expected_digest = _digest(
            artifact_hashes.get(filename),
            name=f"parent artifact {filename} registry SHA256",
        )
        entry = _validate_snapshot_record_digest(
            _mapping(records[index], name=f"artifact snapshot records[{index}]"),
            expected_name=filename,
            expected_sha256=expected_digest,
            location=f"artifact snapshot records[{index}]",
        )
        artifact_file = Path(str(selected.get("source_artifact_resolved"))) / filename
        _require_real_file(artifact_file, name=f"current parent artifact {filename}")
        current = artifact_file.stat()
        if entry["size"] != current.st_size or entry["mtime_ns"] != current.st_mtime_ns:
            raise ValueError(f"current parent artifact {filename} changed after recovery")

    run_before, run_before_raw = _read_json(
        run_path / "parent-run-snapshot-before.json",
        name="parent run snapshot before",
        require_canonical=True,
        maximum_bytes=1024 * 1024,
    )
    run_after, run_after_raw = _read_json(
        run_path / "parent-run-snapshot-after.json",
        name="parent run snapshot after",
        require_canonical=True,
        maximum_bytes=1024 * 1024,
    )
    if run_before_raw != run_after_raw:
        raise ValueError("parent run snapshot changed during recovery")
    run_snapshot = _exact_keys(
        run_before,
        name="parent run snapshot",
        expected={"schema_version", "records"},
    )
    run_records = run_snapshot["records"]
    expected_run_names = {
        "FAILED",
        "run-manifest.json",
        "artifact-materialization.json",
        "artifact-verification.json",
        "phase2-run.log",
    }
    if (
        run_snapshot["schema_version"] != "prorm-read-only-parent-run-snapshot/v1"
        or not isinstance(run_records, list)
        or len(run_records) != len(expected_run_names) + 1
    ):
        raise ValueError("parent run snapshot schema is invalid")
    by_name = {str(item.get("name")): item for item in run_records if isinstance(item, Mapping)}
    if set(by_name) != expected_run_names | {"artifact"}:
        raise ValueError("parent run snapshot files differ from the frozen registry")
    for filename in expected_run_names:
        expected_digest = _digest(
            evidence_hashes.get(filename),
            name=f"parent run {filename} registry SHA256",
        )
        entry = _validate_snapshot_record_digest(
            _mapping(by_name[filename], name=f"parent run snapshot {filename}"),
            expected_name=filename,
            expected_sha256=expected_digest,
            location=f"parent run snapshot {filename}",
        )
        parent_file = Path(str(selected.get("source_run_resolved"))) / filename
        _require_real_file(parent_file, name=f"current parent run {filename}")
        current = parent_file.stat()
        if entry["size"] != current.st_size or entry["mtime_ns"] != current.st_mtime_ns:
            raise ValueError(f"current parent run {filename} changed after recovery")
    link = _exact_keys(
        by_name["artifact"],
        name="parent run snapshot artifact link",
        expected={"name", "symlink_target", "mtime_ns"},
    )
    source_run = Path(str(selected.get("source_run_resolved")))
    actual_parent_link = source_run / "artifact"
    if (
        link["name"] != "artifact"
        or not actual_parent_link.is_symlink()
        or link["symlink_target"] != os.readlink(actual_parent_link)
    ):
        raise ValueError("parent run snapshot artifact link is not the frozen binding")
    link_mtime = _integer(
        link["mtime_ns"],
        name="parent run snapshot artifact link mtime_ns",
    )
    if actual_parent_link.lstat().st_mtime_ns != link_mtime:
        raise ValueError("parent run artifact link changed after recovery")


def _validate_parent_artifact_reference(
    run_path: Path,
    *,
    selected: Mapping[str, object],
) -> None:
    reference = run_path / _REQUIRED_REFERENCE
    if not reference.is_symlink():
        raise ValueError("recovery run is missing its required parent-artifact symlink")
    artifact = Path(str(selected.get("source_artifact_resolved")))
    _require_real_directory(artifact, name="frozen parent artifact")
    expected_target = os.path.relpath(artifact, start=run_path)
    if os.readlink(reference) != expected_target or reference.resolve(strict=True) != artifact:
        raise ValueError("parent-artifact does not resolve to the frozen registry artifact")


def _validate_parent_receipts(
    run_path: Path,
    *,
    seed: int,
) -> Mapping[str, object]:
    derived, derived_raw = _derive_parent_verification(seed)
    before, before_raw = _read_json(
        run_path / "parent-verification-before.json",
        name="parent verification before",
        require_canonical=True,
        maximum_bytes=16 * 1024 * 1024,
    )
    after, after_raw = _read_json(
        run_path / "parent-verification-after.json",
        name="parent verification after",
        require_canonical=True,
        maximum_bytes=16 * 1024 * 1024,
    )
    if (
        before_raw != derived_raw
        or after_raw != derived_raw
        or before != derived
        or after != derived
    ):
        raise ValueError("published parent verification differs from a fresh frozen revalidation")
    receipt = _exact_keys(
        derived,
        name="fresh parent verification",
        expected={
            "status",
            "schema_version",
            "registry_sha256",
            "campaign",
            "selected_seed",
            "all_three_sources_verified",
        },
    )
    if (
        receipt["status"] != "ok"
        or receipt["schema_version"] != "prorm-phase2-recovery-parent-failures/v1"
        or receipt["registry_sha256"] != PARENT_REGISTRY_SHA256
        or receipt["all_three_sources_verified"] is not True
    ):
        raise ValueError("fresh parent verification does not authorize this recovery source")
    selected = _mapping(receipt["selected_seed"], name="fresh parent selected_seed")
    if selected.get("seed") != seed:
        raise ValueError("fresh parent verification selected the wrong seed")
    _validate_parent_artifact_reference(run_path, selected=selected)
    _validate_parent_snapshots(run_path, selected=selected)
    return selected


def _validate_result(
    value: object,
    *,
    result_sha256: str,
    manifest_sha256: str,
    environment: Mapping[str, object],
    seed: int,
    parent_verification: Mapping[str, object],
) -> None:
    result = _exact_keys(value, name="recovery-result.json", expected=_RESULT_KEYS)
    boundary = _exact_keys(
        result["information_boundary"],
        name="recovery-result.json.information_boundary",
        expected=set(_RESULT_BOUNDARY),
    )
    execution_identity = _exact_keys(
        result["recovery_execution_identity"],
        name="recovery-result.json.recovery_execution_identity",
        expected=_ENVIRONMENT_KEYS,
    )
    parent = _exact_keys(
        result["parent_failure_binding"],
        name="recovery-result.json.parent_failure_binding",
        expected={
            "registry_sha256",
            "parent_phase2_design_sha256",
            "parent_source_job_array_id",
            "parent_seed_entry",
            "parent_artifact_producer",
            "parent_failure_aggregate_present",
            "exact_three_seed_failure_registry_used",
            "optimizer_diagnostic",
        },
    )
    parent_producer = _exact_keys(
        parent["parent_artifact_producer"],
        name="recovery-result.json.parent_failure_binding.parent_artifact_producer",
        expected={"git_commit", "image_sha256", "hf_inventory_sha256"},
    )
    artifact_reuse = _exact_keys(
        result["artifact_reuse"],
        name="recovery-result.json.artifact_reuse",
        expected={
            "mode",
            "metadata_sha256",
            "tensor_file_sha256",
            "candidate_file_sha256",
            "producer_identity_separate_from_recovery_training_identity",
            "materialized_or_mutated_by_recovery",
        },
    )
    train_rescore = _exact_keys(
        result["train_oracle_rescore"],
        name="recovery-result.json.train_oracle_rescore",
        expected={
            "source",
            "num_prompts",
            "num_candidates",
            "transformed_rewards_sha256",
            "oracle_chat_template_sha256",
            "frozen_transform",
            "raw_oracle_logits_serialized",
        },
    )
    # Deliberately check only that the bound head payload is a mapping.  Its
    # values are verified by recovery-output-verification.json and are never
    # selected, copied, or returned by this aggregate path.
    _mapping(result["head_training"], name="recovery-result.json.head_training")
    selected_artifact_hashes = _mapping(
        parent_verification.get("artifact_sha256"),
        name="fresh parent selected artifact hashes",
    )
    selected_entry = {
        key: value
        for key, value in parent_verification.items()
        if key not in {"source_run_resolved", "source_artifact_resolved"}
    }
    if (
        result["schema_version"] != "prorm-phase2-recovery-train-only-result/v1"
        or result["status"] != "SUCCESS"
        or result["design_stage"] != "pilot"
        or result["evidence_role"] != "one_shot_optimizer_recovery_train_only"
        or result["formal_eligibility"] is not False
        or result["per_seed_supports_formal_claim"] is not False
        or result["seed"] != seed
        or result["source_config_hash"] != SOURCE_CONFIG_HASH
        or result["recovery_design_sha256"] != RECOVERY_DESIGN_SHA256
        or result["recovery_run_manifest_sha256"] != manifest_sha256
        or dict(execution_identity) != dict(environment)
        or dict(boundary) != _RESULT_BOUNDARY
        or result["one_shot_no_further_adaptation"] is not True
        or result["failure_action"] != "hard_fail_no_second_recovery"
        or parent.get("registry_sha256") != PARENT_REGISTRY_SHA256
        or parent.get("parent_phase2_design_sha256") != PARENT_DESIGN_SHA256
        or parent.get("parent_source_job_array_id") != "1647491"
        or parent.get("parent_failure_aggregate_present") is not False
        or parent.get("exact_three_seed_failure_registry_used") is not True
        or parent.get("parent_seed_entry") != selected_entry
        or parent_producer.get("git_commit") != PARENT_PRODUCER_GIT_COMMIT
        or parent_producer.get("image_sha256") != IMAGE_SHA256
        or parent_producer.get("hf_inventory_sha256") != HF_INVENTORY_SHA256
        or artifact_reuse.get("mode") != "immutable_parent_materialization_only"
        or artifact_reuse.get("producer_identity_separate_from_recovery_training_identity")
        is not True
        or artifact_reuse.get("materialized_or_mutated_by_recovery") is not False
        or artifact_reuse.get("metadata_sha256") != selected_artifact_hashes.get("metadata.json")
        or artifact_reuse.get("tensor_file_sha256")
        != selected_artifact_hashes.get("tensors.safetensors")
        or artifact_reuse.get("candidate_file_sha256")
        != selected_artifact_hashes.get("candidates.jsonl")
        or train_rescore.get("source") != "saved_train_candidate_prefix_only"
        or train_rescore.get("num_prompts") != 1536
        or train_rescore.get("num_candidates") != 4
        or train_rescore.get("frozen_transform") != {"b": 0.0, "tau": 1.0}
        or train_rescore.get("raw_oracle_logits_serialized") is not False
    ):
        raise ValueError("recovery-result.json violates the frozen train-only contract")
    _digest(result_sha256, name="recovery-result.json SHA256")


def _finite_number(
    value: object,
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _tensor_sha256_float32(value: object, *, name: str) -> tuple[str, int]:
    """Reproduce phase2_training._tensor_sha256 for a serialized FP32 vector."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty serialized vector")
    normalized = [_finite_number(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    try:
        import torch

        from .phase2_training import _tensor_sha256
    except (ImportError, OSError) as error:
        raise RuntimeError("could not load the frozen tensor-hash implementation") from error
    tensor = torch.tensor(normalized, dtype=torch.float32)
    return _tensor_sha256(tensor), len(normalized)


def _learning_rate_for_update(
    protocol: Mapping[str, object],
    update: int,
    *,
    name: str,
) -> float:
    schedule = _exact_keys(
        protocol.get("learning_rate_schedule"),
        name=f"{name}.learning_rate_schedule",
        expected={"update_indexing", "application", "stages", "schedule_sha256"},
    )
    if (
        schedule["update_indexing"] != "one_indexed_inclusive"
        or schedule["application"] != "set_learning_rate_immediately_before_optimizer_update"
        or schedule["schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256
    ):
        raise ValueError(f"{name} learning-rate schedule identity is invalid")
    stages = schedule["stages"]
    if not isinstance(stages, list):
        raise TypeError(f"{name}.learning_rate_schedule.stages must be a list")
    expected_stages = (
        (1, 5760, 1.0e-3),
        (5761, 6760, 3.0e-4),
        (6761, 8760, 1.0e-4),
        (8761, 10760, 3.0e-5),
        (10761, 12760, 1.0e-5),
    )
    if len(stages) != len(expected_stages):
        raise ValueError(f"{name} does not contain the exact five recovery LR stages")
    selected: float | None = None
    for index, (raw_stage, expected) in enumerate(zip(stages, expected_stages, strict=True)):
        stage = _exact_keys(
            raw_stage,
            name=f"{name}.learning_rate_schedule.stages[{index}]",
            expected={"first_update", "last_update", "learning_rate"},
        )
        observed = (
            stage["first_update"],
            stage["last_update"],
            _finite_number(
                stage["learning_rate"],
                name=f"{name}.learning_rate_schedule.stages[{index}].learning_rate",
                nonnegative=True,
            ),
        )
        if observed != expected:
            raise ValueError(f"{name} recovery LR stage {index} is not frozen")
        if expected[0] <= update <= expected[1]:
            selected = expected[2]
    if selected is None:
        raise ValueError(f"{name} update is outside the frozen LR schedule")
    return selected


def _validate_measurement(
    value: object,
    *,
    solver: str,
    name: str,
) -> tuple[float, float, Mapping[str, object] | None]:
    measurement = _exact_keys(
        value,
        name=name,
        expected={"objective", "gradient_l2_norm", "inner_solver", "audit_dtype"},
    )
    objective = _finite_number(
        measurement["objective"],
        name=f"{name}.objective",
        nonnegative=True,
    )
    gradient = _finite_number(
        measurement["gradient_l2_norm"],
        name=f"{name}.gradient_l2_norm",
        nonnegative=True,
    )
    if measurement["audit_dtype"] != "float64":
        raise ValueError(f"{name} is not the serialized FP64 recovery audit")
    inner = measurement["inner_solver"]
    if solver == "none":
        if inner is not None:
            raise ValueError(f"{name}.inner_solver must be null for this objective")
        return objective, gradient, None
    if solver == "pcg":
        evidence = _exact_keys(
            inner,
            name=f"{name}.inner_solver",
            expected={
                "method",
                "dtype",
                "cold_start",
                "warm_start_used",
                "iterations",
                "residual_norm",
                "relative_residual",
                "converged",
            },
        )
        if (
            evidence["method"] != "pcg"
            or evidence["dtype"] != "float64"
            or evidence["cold_start"] is not True
            or evidence["warm_start_used"] is not False
            or evidence["converged"] is not True
        ):
            raise ValueError(f"{name}.inner_solver violates the cold FP64 PCG contract")
        _integer(
            evidence["iterations"],
            name=f"{name}.inner_solver.iterations",
        )
        _finite_number(
            evidence["residual_norm"],
            name=f"{name}.inner_solver.residual_norm",
            nonnegative=True,
        )
        relative = _finite_number(
            evidence["relative_residual"],
            name=f"{name}.inner_solver.relative_residual",
            nonnegative=True,
        )
        if relative > 1.0e-5:
            raise ValueError(f"{name}.inner_solver exceeds the frozen PCG tolerance")
        return objective, gradient, evidence
    if solver != "pseudoinverse":
        raise RuntimeError(f"unsupported recovery measurement solver {solver!r}")
    evidence = _exact_keys(
        inner,
        name=f"{name}.inner_solver",
        expected={
            "method",
            "dtype",
            "cold_start",
            "warm_start_used",
            "numerical_rank",
            "relative_eigenvalue_tolerance",
            "solve_residual_norm",
            "solve_relative_residual",
            "converged",
        },
    )
    if (
        evidence["method"] != "truncated_moore_penrose_pseudoinverse"
        or evidence["dtype"] != "float64"
        or evidence["cold_start"] is not True
        or evidence["warm_start_used"] is not False
        or evidence["converged"] is not True
        or _finite_number(
            evidence["relative_eigenvalue_tolerance"],
            name=f"{name}.inner_solver.relative_eigenvalue_tolerance",
            nonnegative=True,
        )
        != 1.0e-10
    ):
        raise ValueError(f"{name}.inner_solver violates the frozen pseudoinverse contract")
    _integer(
        evidence["numerical_rank"],
        name=f"{name}.inner_solver.numerical_rank",
        minimum=1,
    )
    _finite_number(
        evidence["solve_residual_norm"],
        name=f"{name}.inner_solver.solve_residual_norm",
        nonnegative=True,
    )
    relative = _finite_number(
        evidence["solve_relative_residual"],
        name=f"{name}.inner_solver.solve_relative_residual",
        nonnegative=True,
    )
    if relative > 1.0e-6:
        raise ValueError(f"{name}.inner_solver exceeds the pseudoinverse residual gate")
    return objective, gradient, evidence


def _validate_history_summary(
    value: object,
    *,
    expected_steps: int,
    solver: str,
    name: str,
) -> None:
    history = _exact_keys(
        value,
        name=name,
        expected={
            "num_steps",
            "history_objective_timing",
            "stored_checkpoint_steps",
            "checkpoints",
            "objective",
            "gradient_l2_norm",
            "pcg",
        },
    )
    expected_checkpoint_steps = sorted(
        {
            1,
            max(1, expected_steps // 4),
            max(1, expected_steps // 2),
            max(1, (3 * expected_steps) // 4),
            expected_steps,
        }
    )
    checkpoints = history["checkpoints"]
    if (
        history["num_steps"] != expected_steps
        or history["history_objective_timing"] != "pre_update"
        or history["stored_checkpoint_steps"] != expected_checkpoint_steps
        or not isinstance(checkpoints, list)
        or len(checkpoints) != len(expected_checkpoint_steps)
    ):
        raise ValueError(f"{name} is not the exact selected-iterate history summary")
    checkpoint_keys = {
        "step",
        "objective",
        "gradient_norm",
        "dual_loss",
        "dual_saddle_value",
        "dual_refresh",
        "pcg_iterations",
        "pcg_residual_norm",
        "pcg_relative_residual",
        "pcg_converged",
    }
    for index, (raw, step) in enumerate(zip(checkpoints, expected_checkpoint_steps, strict=True)):
        checkpoint = _exact_keys(
            raw,
            name=f"{name}.checkpoints[{index}]",
            expected=checkpoint_keys,
        )
        if checkpoint["step"] != step:
            raise ValueError(f"{name}.checkpoints[{index}] has the wrong step")
        _finite_number(checkpoint["objective"], name=f"{name}.checkpoints[{index}].objective")
        _finite_number(
            checkpoint["gradient_norm"],
            name=f"{name}.checkpoints[{index}].gradient_norm",
            nonnegative=True,
        )
        if solver in {"pcg", "pseudoinverse"}:
            for field in ("dual_loss", "dual_saddle_value", "pcg_residual_norm"):
                if solver == "pseudoinverse" and field == "pcg_residual_norm":
                    continue
                _finite_number(
                    checkpoint[field],
                    name=f"{name}.checkpoints[{index}].{field}",
                    nonnegative=(field == "pcg_residual_norm"),
                )
            _integer(
                checkpoint["dual_refresh"],
                name=f"{name}.checkpoints[{index}].dual_refresh",
            )
            if solver == "pcg":
                _integer(
                    checkpoint["pcg_iterations"],
                    name=f"{name}.checkpoints[{index}].pcg_iterations",
                )
                relative = _finite_number(
                    checkpoint["pcg_relative_residual"],
                    name=f"{name}.checkpoints[{index}].pcg_relative_residual",
                    nonnegative=True,
                )
                if checkpoint["pcg_converged"] is not True or relative > 1.0e-5:
                    raise ValueError(f"{name}.checkpoints[{index}] has invalid PCG evidence")
            elif any(
                checkpoint[field] is not None
                for field in (
                    "pcg_iterations",
                    "pcg_residual_norm",
                    "pcg_relative_residual",
                    "pcg_converged",
                )
            ):
                raise ValueError(f"{name}.checkpoints[{index}] has unexpected iterative-PCG fields")
        elif any(
            checkpoint[field] is not None
            for field in (
                "dual_loss",
                "dual_saddle_value",
                "dual_refresh",
                "pcg_iterations",
                "pcg_residual_norm",
                "pcg_relative_residual",
                "pcg_converged",
            )
        ):
            raise ValueError(f"{name}.checkpoints[{index}] has unexpected PCG fields")
    for field in ("objective", "gradient_l2_norm"):
        summary = _exact_keys(
            history[field],
            name=f"{name}.{field}",
            expected={"first", "last_pre_update", "minimum", "maximum"},
        )
        values = {
            key: _finite_number(
                item,
                name=f"{name}.{field}.{key}",
                nonnegative=(field == "gradient_l2_norm"),
            )
            for key, item in summary.items()
        }
        if values["minimum"] > values["maximum"]:
            raise ValueError(f"{name}.{field} has invalid extrema")
    if solver == "pcg":
        pcg = _exact_keys(
            history["pcg"],
            name=f"{name}.pcg",
            expected={
                "num_fresh_solves",
                "all_converged",
                "maximum_relative_residual",
                "maximum_iterations",
            },
        )
        if (
            pcg["num_fresh_solves"] != expected_steps
            or pcg["all_converged"] is not True
            or _finite_number(
                pcg["maximum_relative_residual"],
                name=f"{name}.pcg.maximum_relative_residual",
                nonnegative=True,
            )
            > 1.0e-5
        ):
            raise ValueError(f"{name}.pcg does not cover every selected update")
        _integer(
            pcg["maximum_iterations"],
            name=f"{name}.pcg.maximum_iterations",
        )
    elif history["pcg"] is not None:
        raise ValueError(f"{name}.pcg must be null for this trainer")


def _validate_optimizer_execution(
    value: object,
    *,
    protocol: Mapping[str, object],
    selected_step: int,
    selected_head_sha256: str,
    name: str,
) -> None:
    execution = _exact_keys(
        value,
        name=name,
        expected={
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
            "selected_primary_optimizer_state_restored_without_reconstruction",
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
        },
    )
    expected_scalars = {
        "schema_version": "deterministic-adamw-lr-decay-execution/v2",
        "optimizer_class": "torch.optim.AdamW",
        "parameter_count": 1,
        "fresh_optimizer_state_before_first_update": True,
        "reward_head_dtype_observed": "torch.float32",
        "first_order_audit_dtype_required": "float64",
        "microbatch_order": "canonical_edge_order_contiguous_ascending_no_shuffle",
        "one_optimizer_update_per_step": True,
        "learning_rate_set_immediately_before_every_update": True,
        "single_optimizer_instance_for_all_updates": True,
        "optimizer_state_reset_at_lr_milestone": False,
        "adamw_moments_preserved_at_learning_rate_boundaries": True,
        "selected_primary_optimizer_state_restored_without_reconstruction": True,
        "selected_primary_optimizer_state_restored_and_verified": True,
        "selected_optimizer_object_identity_preserved": True,
        "selected_optimizer_moments_restored_and_verified": True,
        "test_or_validation_data_accessed": False,
    }
    if any(execution[key] != expected for key, expected in expected_scalars.items()):
        raise ValueError(f"{name} violates the frozen AdamW execution contract")
    if dict(_mapping(execution["protocol"], name=f"{name}.protocol")) != dict(protocol):
        raise ValueError(f"{name}.protocol differs from the frozen recovery protocol")
    completed = _integer(
        execution["completed_updates_observed"],
        name=f"{name}.completed_updates_observed",
        minimum=5760,
    )
    if completed != max(selected_step, 720, 5760):
        raise ValueError(f"{name} executed more or fewer updates than required")
    state_checks = _exact_keys(
        execution["per_update_state_checks"],
        name=f"{name}.per_update_state_checks",
        expected={
            "schema_version",
            "before_update_checks",
            "after_update_checks",
            "first_pre_update_state_empty",
            "completed_updates_covered",
            "check_sequence_sha256",
            "all_updates_checked_before_and_after",
            "all_subsequent_pre_update_scalar_steps_exact",
            "all_post_update_scalar_steps_exact",
            "exp_avg_and_exp_avg_sq_shape_dtype_device_valid",
        },
    )
    if (
        state_checks["schema_version"] != "recovery-adamw-per-update-state-checks/v1"
        or state_checks["before_update_checks"] != completed
        or state_checks["after_update_checks"] != completed
        or state_checks["completed_updates_covered"] != completed
        or state_checks["first_pre_update_state_empty"] is not True
        or state_checks["all_updates_checked_before_and_after"] is not True
        or state_checks["all_subsequent_pre_update_scalar_steps_exact"] is not True
        or state_checks["all_post_update_scalar_steps_exact"] is not True
        or state_checks["exp_avg_and_exp_avg_sq_shape_dtype_device_valid"] is not True
    ):
        raise ValueError(f"{name}.per_update_state_checks is incomplete")
    _digest(
        state_checks["check_sequence_sha256"],
        name=f"{name}.per_update_state_checks.check_sequence_sha256",
    )
    transitions = execution["boundary_transitions"]
    if not isinstance(transitions, list):
        raise TypeError(f"{name}.boundary_transitions must be a list")
    expected_transitions = [
        (5761, 1.0e-3, 3.0e-4),
        (6761, 3.0e-4, 1.0e-4),
        (8761, 1.0e-4, 3.0e-5),
        (10761, 3.0e-5, 1.0e-5),
    ]
    expected_transitions = [
        transition for transition in expected_transitions if transition[0] <= completed
    ]
    if len(transitions) != len(expected_transitions):
        raise ValueError(f"{name}.boundary_transitions does not match completed updates")
    for index, (raw, expected) in enumerate(zip(transitions, expected_transitions, strict=True)):
        transition = _exact_keys(
            raw,
            name=f"{name}.boundary_transitions[{index}]",
            expected={
                "next_update",
                "previous_learning_rate",
                "new_learning_rate",
                "moment_state_sha256_before_lr_assignment",
                "moment_state_sha256_after_lr_assignment",
                "same_optimizer_instance",
                "moments_preserved",
            },
        )
        before = _digest(
            transition["moment_state_sha256_before_lr_assignment"],
            name=f"{name}.boundary_transitions[{index}].moment_before",
        )
        after = _digest(
            transition["moment_state_sha256_after_lr_assignment"],
            name=f"{name}.boundary_transitions[{index}].moment_after",
        )
        if (
            transition["next_update"] != expected[0]
            or transition["previous_learning_rate"] != expected[1]
            or transition["new_learning_rate"] != expected[2]
            or before != after
            or transition["same_optimizer_instance"] is not True
            or transition["moments_preserved"] is not True
        ):
            raise ValueError(f"{name}.boundary_transitions[{index}] is invalid")
    selected_optimizer = _digest(
        execution["selected_optimizer_state_sha256"],
        name=f"{name}.selected_optimizer_state_sha256",
    )
    checkpoint_optimizer = _digest(
        execution["selected_checkpoint_optimizer_state_dict_sha256"],
        name=f"{name}.selected_checkpoint_optimizer_state_dict_sha256",
    )
    if (
        execution["selected_head_sha256"] != selected_head_sha256
        or execution["restored_head_sha256"] != selected_head_sha256
        or execution["restored_optimizer_state_sha256"] != selected_optimizer
        or execution["restored_optimizer_state_dict_sha256"] != checkpoint_optimizer
    ):
        raise ValueError(f"{name} selected/restored state hashes disagree")
    expected_checkpoint = _canonical_sha256(
        {
            "schema_version": "selected-recovery-state-binding/v1",
            "completed_updates": selected_step,
            "head_sha256": selected_head_sha256,
            "optimizer_state_sha256": selected_optimizer,
            "optimizer_state_dict_sha256": checkpoint_optimizer,
        }
    )
    if execution["selected_checkpoint_sha256"] != expected_checkpoint:
        raise ValueError(f"{name}.selected_checkpoint_sha256 is not the composite binding")


def _validate_recovery_head(
    value: object,
    *,
    expected_arm: str,
    expected_method: str,
    expected_objective: str,
    solver: str,
    expected_protocol: Mapping[str, object],
    expected_weight: object | None,
    name: str,
) -> tuple[int, str, int]:
    head = _exact_keys(
        value,
        name=name,
        expected={
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
        },
    )
    if (
        head["arm"] != expected_arm
        or head["method"] != expected_method
        or head["head_dtype"] != "torch.float32"
    ):
        raise ValueError(f"{name} has the wrong arm, method, or FP32 dtype")
    if expected_weight is not None and head["head_weight"] != expected_weight:
        raise ValueError(f"{name}.head_weight differs from the primary serialized head")
    head_sha, dimension = _tensor_sha256_float32(
        head["head_weight"],
        name=f"{name}.head_weight",
    )
    try:
        import torch

        from .phase2_training import _tensor_sha256
    except (ImportError, OSError) as error:
        raise RuntimeError("could not load the frozen tensor-hash implementation") from error
    zero_sha = _tensor_sha256(torch.zeros(dimension, dtype=torch.float32))
    if head["initial_head_sha256"] != zero_sha or head["head_sha256"] != head_sha:
        raise ValueError(f"{name} tensor SHA256 does not bind its serialized FP32 vector")

    convergence = _exact_keys(
        head["first_order_convergence"],
        name=f"{name}.first_order_convergence",
        expected={
            "schema_version",
            "objective",
            "converged",
            "fail_closed",
            "spec",
            "gradient_ratio_formula",
            "initial_zero_head_measurement",
            "checks",
            "selected_primary_step",
            "selected_primary_head_sha256",
            "consecutive_threshold_passes_at_selection",
            "final_gate",
            "fixed_step_compute_matched_snapshot",
            "fixed_step_snapshot_steps",
            "fixed_step_snapshot_is_not_primary_selection",
            "solution_identification",
            "test_or_validation_data_accessed",
            "legacy_constant_lr_boundary_snapshot",
            "optimizer_protocol_execution",
        },
    )
    spec = _exact_keys(
        convergence["spec"],
        name=f"{name}.first_order_convergence.spec",
        expected={
            "schema_version",
            "gradient_ratio_tolerance",
            "min_steps",
            "max_steps",
            "check_interval",
            "consecutive_checks",
            "gradient_norm_denominator_floor",
            "fail_closed",
            "gradient",
            "denominator",
            "validation_or_test_selection",
            "optimizer_protocol",
        },
    )
    if (
        convergence["schema_version"] != "objective-first-order-convergence/v2"
        or convergence["objective"] != expected_objective
        or convergence["converged"] is not True
        or convergence["fail_closed"] is not True
        or convergence["gradient_ratio_formula"]
        != (
            "||full_data_unclipped_gradient(w_t)||_2 / "
            "max(||full_data_unclipped_gradient(w_zero)||_2, denominator_floor)"
        )
        or convergence["selected_primary_head_sha256"] != head_sha
        or convergence["consecutive_threshold_passes_at_selection"] != 3
        or convergence["fixed_step_snapshot_steps"] != 720
        or convergence["fixed_step_snapshot_is_not_primary_selection"] is not True
        or convergence["test_or_validation_data_accessed"] is not False
        or spec["schema_version"] != "objective-first-order-convergence-spec/v2"
        or spec["gradient_ratio_tolerance"] != 1.0e-3
        or spec["min_steps"] != 100
        or spec["max_steps"] != 12760
        or spec["check_interval"] != 20
        or spec["consecutive_checks"] != 3
        or spec["gradient_norm_denominator_floor"] != 1.0e-30
        or spec["fail_closed"] is not True
        or spec["gradient"] != "full_data_post_update_unclipped"
        or spec["denominator"] != "exact_zero_initialization_gradient_l2_norm"
        or spec["validation_or_test_selection"] is not False
        or dict(_mapping(spec["optimizer_protocol"], name=f"{name}.spec.protocol"))
        != dict(expected_protocol)
    ):
        raise ValueError(f"{name} first-order spec/protocol is not the frozen recovery gate")
    # This also validates every stage, even if the selected iterate occurs
    # before the later stages.
    _learning_rate_for_update(expected_protocol, 12760, name=f"{name}.protocol")
    selected_step = _integer(
        convergence["selected_primary_step"],
        name=f"{name}.selected_primary_step",
        minimum=100,
    )
    if selected_step > 12760 or selected_step % 20:
        raise ValueError(f"{name}.selected_primary_step is not a scheduled check")
    initial_objective, initial_gradient, _ = _validate_measurement(
        convergence["initial_zero_head_measurement"],
        solver=solver,
        name=f"{name}.initial_zero_head_measurement",
    )
    denominator = max(initial_gradient, 1.0e-30)
    if head["initial_objective"] != initial_objective:
        raise ValueError(f"{name}.initial_objective is not bound to the zero-head audit")
    checks = convergence["checks"]
    expected_steps = list(range(20, selected_step + 1, 20))
    if not isinstance(checks, list) or len(checks) != len(expected_steps):
        raise ValueError(f"{name}.checks must cover every scheduled pre-selection audit")
    consecutive = 0
    first_selection: int | None = None
    for index, (raw_check, step) in enumerate(zip(checks, expected_steps, strict=True)):
        check = _exact_keys(
            raw_check,
            name=f"{name}.checks[{index}]",
            expected={
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
            },
        )
        _, gradient, _ = _validate_measurement(
            check["measurement"],
            solver=solver,
            name=f"{name}.checks[{index}].measurement",
        )
        ratio = gradient / denominator
        eligible = step >= 100
        passed = eligible and ratio <= 1.0e-3
        consecutive = consecutive + 1 if passed else 0
        if consecutive >= 3 and first_selection is None:
            first_selection = step
        if (
            check["step"] != step
            or check["post_update"] is not True
            or check["full_data"] is not True
            or check["gradient_clipping_applied"] is not False
            or not math.isclose(
                _finite_number(
                    check["gradient_ratio_to_zero_initialization"],
                    name=f"{name}.checks[{index}].gradient_ratio",
                    nonnegative=True,
                ),
                ratio,
                rel_tol=1.0e-10,
                abs_tol=1.0e-14,
            )
            or check["eligible_after_min_steps"] is not eligible
            or check["threshold_passed"] is not passed
            or check["consecutive_threshold_passes"] != consecutive
            or check["learning_rate_used_for_update"]
            != _learning_rate_for_update(expected_protocol, step, name=f"{name}.protocol")
            or check["learning_rate_schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256
        ):
            raise ValueError(f"{name}.checks[{index}] has invalid gate arithmetic")
    if first_selection != selected_step or consecutive != 3:
        raise ValueError(f"{name} was not selected at the first sustained three-pass gate")

    final = _exact_keys(
        convergence["final_gate"],
        name=f"{name}.final_gate",
        expected={
            "step",
            "measurement",
            "gradient_ratio_to_zero_initialization",
            "threshold_passed",
            "fresh_post_restore_audit",
            "learning_rate_at_selected_iterate",
        },
    )
    final_objective, final_gradient, final_solver = _validate_measurement(
        final["measurement"],
        solver=solver,
        name=f"{name}.final_gate.measurement",
    )
    final_ratio = final_gradient / denominator
    if (
        final["step"] != selected_step
        or final["threshold_passed"] is not True
        or final["fresh_post_restore_audit"] is not True
        or final["learning_rate_at_selected_iterate"]
        != _learning_rate_for_update(
            expected_protocol,
            selected_step,
            name=f"{name}.protocol",
        )
        or not math.isclose(
            _finite_number(
                final["gradient_ratio_to_zero_initialization"],
                name=f"{name}.final_gate.gradient_ratio",
                nonnegative=True,
            ),
            final_ratio,
            rel_tol=1.0e-10,
            abs_tol=1.0e-14,
        )
        or final_ratio > 1.0e-3
        or head["final_objective"] != final_objective
    ):
        raise ValueError(f"{name}.final_gate does not bind the restored selected iterate")
    if solver == "pcg":
        final_pcg = _exact_keys(
            head["final_pcg"],
            name=f"{name}.final_pcg",
            expected={
                "method",
                "dtype",
                "iterations",
                "residual_norm",
                "relative_residual",
                "converged",
                "cold_start",
                "warm_start_used",
            },
        )
        if final_solver is None or dict(final_pcg) != dict(final_solver):
            raise ValueError(f"{name}.final_pcg differs from the fresh final audit")
    elif head["final_pcg"] is not None:
        raise ValueError(f"{name}.final_pcg must be null for this objective")

    fixed = _exact_keys(
        convergence["fixed_step_compute_matched_snapshot"],
        name=f"{name}.fixed_step_compute_matched_snapshot",
        expected={
            "schema_version",
            "step",
            "head_sha256",
            "measurement",
            "gradient_ratio_to_zero_initialization",
            "history_summary",
            "role",
            "used_as_primary_selection_rule",
            "coincides_with_selected_primary_iterate",
        },
    )
    legacy = _exact_keys(
        convergence["legacy_constant_lr_boundary_snapshot"],
        name=f"{name}.legacy_constant_lr_boundary_snapshot",
        expected={
            "schema_version",
            "step",
            "head_sha256",
            "measurement",
            "gradient_ratio_to_zero_initialization",
            "history_summary",
            "learning_rate_used_for_update",
            "learning_rate_schedule_sha256",
            "role",
            "used_as_primary_selection_rule",
            "coincides_with_selected_primary_iterate",
            "test_or_validation_data_accessed",
        },
    )
    for snapshot, step, role, snapshot_name in (
        (
            fixed,
            720,
            "compute_matched_and_pilot_diagnostic_only",
            "fixed_step_compute_matched_snapshot",
        ),
        (
            legacy,
            5760,
            "immutable_legacy_constant_lr_failure_boundary_diagnostic",
            "legacy_constant_lr_boundary_snapshot",
        ),
    ):
        _, gradient, _ = _validate_measurement(
            snapshot["measurement"],
            solver=solver,
            name=f"{name}.{snapshot_name}.measurement",
        )
        expected_schema = (
            "fixed-step-compute-matched-snapshot/v1"
            if step == 720
            else "legacy-constant-lr-boundary-snapshot/v1"
        )
        if (
            snapshot["schema_version"] != expected_schema
            or snapshot["step"] != step
            or snapshot["role"] != role
            or snapshot["used_as_primary_selection_rule"] is not False
            or snapshot["coincides_with_selected_primary_iterate"] is not (selected_step == step)
            or not math.isclose(
                _finite_number(
                    snapshot["gradient_ratio_to_zero_initialization"],
                    name=f"{name}.{snapshot_name}.gradient_ratio",
                    nonnegative=True,
                ),
                gradient / denominator,
                rel_tol=1.0e-10,
                abs_tol=1.0e-14,
            )
        ):
            raise ValueError(f"{name}.{snapshot_name} violates its frozen role")
        snapshot_sha = _digest(
            snapshot["head_sha256"],
            name=f"{name}.{snapshot_name}.head_sha256",
        )
        if selected_step == step and snapshot_sha != head_sha:
            raise ValueError(f"{name}.{snapshot_name} does not bind the selected head")
        _validate_history_summary(
            snapshot["history_summary"],
            expected_steps=step,
            solver=solver,
            name=f"{name}.{snapshot_name}.history_summary",
        )
    if (
        legacy["learning_rate_used_for_update"]
        != _learning_rate_for_update(expected_protocol, 5760, name=f"{name}.protocol")
        or legacy["learning_rate_schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256
        or legacy["test_or_validation_data_accessed"] is not False
    ):
        raise ValueError(f"{name}.legacy_constant_lr_boundary_snapshot is not schedule-bound")
    identification = _exact_keys(
        convergence["solution_identification"],
        name=f"{name}.solution_identification",
        expected={
            "initialization",
            "tie_break",
            "primary_iterate_selection",
            "validation_or_test_checkpoint_selection",
            "objective_value_checkpoint_selection",
            "minimum_norm_projection_applied",
            "minimum_norm_solution_claimed",
            "unique_reward_head_solution_claimed",
            "optional_objective_rank_diagnostic",
            "minimum_norm_note",
        },
    )
    rank = _exact_keys(
        identification["optional_objective_rank_diagnostic"],
        name=f"{name}.solution_identification.optional_objective_rank_diagnostic",
        expected={"evaluated", "evidence"},
    )
    if (
        identification["initialization"] != "exact_zero_head"
        or identification["tie_break"] != _RECOVERY_TIE_BREAK
        or identification["primary_iterate_selection"]
        != "first_scheduled_iterate_completing_the_sustained_first_order_gate"
        or identification["validation_or_test_checkpoint_selection"] is not False
        or identification["objective_value_checkpoint_selection"] is not False
        or identification["minimum_norm_projection_applied"] is not False
        or identification["minimum_norm_solution_claimed"] is not False
        or identification["unique_reward_head_solution_claimed"] is not False
        or rank["evaluated"] is not True
        or not isinstance(rank["evidence"], Mapping)
    ):
        raise ValueError(f"{name}.solution_identification is invalid")
    _validate_optimizer_execution(
        convergence["optimizer_protocol_execution"],
        protocol=expected_protocol,
        selected_step=selected_step,
        selected_head_sha256=head_sha,
        name=f"{name}.optimizer_protocol_execution",
    )
    _validate_history_summary(
        head["history_summary"],
        expected_steps=selected_step,
        solver=solver,
        name=f"{name}.history_summary",
    )
    return selected_step, head_sha, dimension


def _load_frozen_recovery_config() -> Mapping[str, object]:
    repository = _repository_root()
    source = repository / _RECOVERY_CONFIG_RELATIVE
    raw = _read_stable_file(
        source,
        name="frozen recovery configuration",
        maximum_bytes=512 * 1024,
    )
    blob = _git_blob(
        RECOVERY_GIT_COMMIT,
        _RECOVERY_CONFIG_RELATIVE,
        name="frozen recovery configuration",
    )
    if raw != blob or hashlib.sha256(raw).hexdigest() != _RECOVERY_CONFIG_SHA256:
        raise ValueError("loaded recovery configuration differs from its producer Git blob")
    try:
        from .phase2_config import load_phase2_config, phase2_design_identity
    except ImportError as error:
        raise RuntimeError("could not load the Phase-2 configuration validator") from error
    config = load_phase2_config(source)
    if phase2_design_identity(config) != RECOVERY_DESIGN_SHA256:
        raise ValueError("frozen recovery configuration has the wrong design identity")
    return config


def _deep_validate_recovery_training(
    value: object,
    *,
    seed: int,
    train_oracle_reward_sha256: str,
) -> dict[str, int]:
    training = _exact_keys(
        value,
        name="head_training",
        expected={"schema_version", "heads", "audit", "test_data_accessed"},
    )
    if (
        training["schema_version"] != "phase2-fresh-head-training/v3"
        or training["test_data_accessed"] is not False
    ):
        raise ValueError("head_training is not the recovery v3 train-only schema")
    heads = _exact_keys(
        training["heads"],
        name="head_training.heads",
        expected={"bt_mle", "prorm_plus"},
    )
    audit = _mapping(training["audit"], name="head_training.audit")
    config = _load_frozen_recovery_config()
    try:
        from . import phase2_aggregate
        from .phase2_training import compile_phase2_training_settings
    except ImportError as error:
        raise RuntimeError("could not load the bound Phase-2 deep-gate implementation") from error
    compiled = compile_phase2_training_settings(config)
    if compiled.sha256 != TRAINING_SETTINGS_SHA256:
        raise ValueError("compiled recovery training settings differ from the producer identity")
    if audit.get("training_settings_sha256") != compiled.sha256:
        raise ValueError("head_training audit has the wrong compiled settings SHA256")
    run = _mapping(config["run"], name="recovery config.run")
    split_sizes = _mapping(run["split_sizes"], name="recovery config.run.split_sizes")
    data = _mapping(config["data"], name="recovery config.data")
    reward_model = _mapping(config["reward_model"], name="recovery config.reward_model")
    protocol = _mapping(
        reward_model["optimizer_protocol"],
        name="recovery config.reward_model.optimizer_protocol",
    )
    controls = _mapping(
        config["positive_controls"],
        name="recovery config.positive_controls",
    )
    low_config = _mapping(
        controls["low_dimensional_tangent"],
        name="recovery config.positive_controls.low_dimensional_tangent",
    )
    objective = _mapping(config["objective"], name="recovery config.objective")
    ridge = _mapping(
        _mapping(
            objective["full_tangent"],
            name="recovery config.objective.full_tangent",
        )["ridge"],
        name="recovery config.objective.full_tangent.ridge",
    )
    annotations = _mapping(config["annotations"], name="recovery config.annotations")
    primary = _mapping(audit["primary_heads"], name="head_training.audit.primary_heads")
    low = _mapping(
        audit["low_dimensional_control"],
        name="head_training.audit.low_dimensional_control",
    )
    exact = _mapping(
        audit["exact_margin_control"],
        name="head_training.audit.exact_margin_control",
    )
    exact_soft = _mapping(
        audit["exact_soft_label_bt_control"],
        name="head_training.audit.exact_soft_label_bt_control",
    )
    entries = (
        (
            "primary_bt_mle",
            primary["bt_mle"],
            "r4_independent_gamma_0.9",
            "bt_mle",
            "bt_mle",
            "none",
            heads["bt_mle"],
        ),
        (
            "primary_prorm_plus",
            primary["prorm_plus"],
            "r4_independent_gamma_0.9",
            "prorm_plus",
            "prorm_plus",
            "pcg",
            heads["prorm_plus"],
        ),
        (
            "low_dimensional_prorm_plus",
            low["head"],
            "low_dimensional_tangent_positive_control",
            "prorm_plus",
            "low_dimensional_prorm_plus",
            "pseudoinverse",
            None,
        ),
        (
            "exact_margin_prorm_plus",
            exact["head"],
            "exact_margin_positive_control",
            "prorm_plus",
            "exact_margin_prorm_plus",
            "pcg",
            None,
        ),
        (
            "exact_soft_label_bt",
            exact_soft["head"],
            "exact_soft_label_bt_secondary_diagnostic",
            "bt_mle",
            "exact_soft_label_bt_cross_entropy",
            "none",
            None,
        ),
    )
    selected_steps: dict[str, int] = {}
    head_hashes: dict[str, str] = {}
    dimensions: set[int] = set()
    for (
        head_name,
        raw_head,
        arm,
        method,
        objective_name,
        solver,
        expected_weight,
    ) in entries:
        selected, head_sha, dimension = _validate_recovery_head(
            raw_head,
            expected_arm=arm,
            expected_method=method,
            expected_objective=objective_name,
            solver=solver,
            expected_protocol=protocol,
            expected_weight=expected_weight,
            name=f"head_training.{head_name}",
        )
        selected_steps[head_name] = selected
        head_hashes[head_name] = head_sha
        dimensions.add(dimension)
    if len(dimensions) != 1:
        raise ValueError("all five recovery reward heads must share one feature dimension")

    # The frozen ad7613 recovery producer serialized the complete fresh-inner
    # eight-field PCG audit as ``final_pcg``.  Validate that raw evidence above
    # without mutation, then project only the in-memory common-gate adapter to
    # the newer five-field ``training_final`` contract.  This compatibility
    # bridge is producer-identity-specific and cannot admit either schema at
    # the raw evidence boundary.
    adapted_audit = copy.deepcopy(dict(audit))
    adapted_primary = _mapping(
        adapted_audit["primary_heads"],
        name="head_training.deep_gate.audit.primary_heads",
    )
    adapted_exact = _mapping(
        adapted_audit["exact_margin_control"],
        name="head_training.deep_gate.audit.exact_margin_control",
    )
    adapted_pcg_heads = (
        _mapping(
            adapted_primary["prorm_plus"],
            name="head_training.deep_gate.audit.primary_heads.prorm_plus",
        ),
        _mapping(
            adapted_exact["head"],
            name="head_training.deep_gate.audit.exact_margin_control.head",
        ),
    )
    for index, adapted_head in enumerate(adapted_pcg_heads):
        if not isinstance(adapted_head, dict):
            raise TypeError(
                f"head_training deep-gate PCG adapter {index} is not a mutable JSON object"
            )
        rich_final = _mapping(
            adapted_head["final_pcg"],
            name=f"head_training.deep_gate.pcg_heads[{index}].final_pcg",
        )
        adapted_head["final_pcg"] = {
            key: rich_final[key]
            for key in (
                "iterations",
                "residual_norm",
                "relative_residual",
                "converged",
                "cold_start",
            )
        }
    adapter = {
        "training_arm": adapted_audit.get("training_arm"),
        "training_design_sha256": adapted_audit.get("training_design_sha256"),
        "heads_sha256": _canonical_sha256(dict(heads)),
        "head_weights": copy.deepcopy(dict(heads)),
        "audit": adapted_audit,
        "source": "trained_after_train_oracle_rescore",
        "old_phase1_comparison_heads_reused": False,
        "test_data_accessed": False,
    }
    # Reuse the complete production gate for identifiability, exact-margin,
    # exact-soft-BT, low-dimensional geometry, objective binding, and all
    # cross-control relationships after the identity-specific projection.
    phase2_aggregate._validate_head_training(
        adapter,
        design_sha256=RECOVERY_DESIGN_SHA256,
        seed=seed,
        train_oracle_reward_sha256=train_oracle_reward_sha256,
        expected_train_prompts=int(split_sizes["train"]),
        expected_candidates=int(data["num_candidates"]),
        expected_outer_steps=int(reward_model["outer_steps"]),
        expected_low_dimension=int(low_config["dimension"]),
        expected_projection_namespace=str(low_config["seed_namespace"]),
        expected_eigenvalue_tolerance=float(low_config["relative_eigenvalue_tolerance"]),
        expected_pcg_tolerance=float(ridge["pcg_tolerance"]),
        prohibit_label_clipping=annotations.get("prohibit_clipping") is True,
        numeric_tolerances=_mapping(
            controls["numeric_gate_tolerances"],
            name="recovery config.positive_controls.numeric_gate_tolerances",
        ),
        adaptive_convergence=_mapping(
            reward_model["adaptive_convergence"],
            name="recovery config.reward_model.adaptive_convergence",
        ),
        identifiability_config=_mapping(
            reward_model["identifiability"],
            name="recovery config.reward_model.identifiability",
        ),
        reward_model_config=reward_model,
        expected_optimizer_protocol=protocol,
        exact_soft_label_bt_config=_mapping(
            controls["exact_soft_label_bt"],
            name="recovery config.positive_controls.exact_soft_label_bt",
        ),
        design_stage="pilot",
        name="head_training.deep_gate",
    )

    label = _mapping(audit["label_stream"], name="head_training.audit.label_stream")
    primary_audit = _mapping(
        audit["primary_optimization_audit"],
        name="head_training.audit.primary_optimization_audit",
    )
    projection = _mapping(low["projection"], name="head_training.low.projection")
    direct = _mapping(
        audit["direct_oracle_identity"],
        name="head_training.audit.direct_oracle_identity",
    )
    native = _mapping(
        direct["native_oracle_direction"],
        name="head_training.audit.direct_oracle_identity.native_oracle_direction",
    )
    training_instance = _canonical_sha256(
        {
            "schema_version": "phase2-training-instance/v1",
            "phase2_config_hash": RECOVERY_DESIGN_SHA256,
            "settings_sha256": TRAINING_SETTINGS_SHA256,
            "input_training_sha256": audit["input_training_sha256"],
            "oracle_reward_sha256": train_oracle_reward_sha256,
            "seed": seed,
            "label_stream_sha256": label["label_stream_sha256"],
            "reward_head_identifiability_sha256": _canonical_sha256(
                _mapping(
                    primary_audit["reward_head_identifiability"],
                    name="head_training.reward_head_identifiability",
                )
            ),
            "prorm_moment_map_identifiability_sha256": _canonical_sha256(
                _mapping(
                    primary_audit["prorm_moment_map_identifiability"],
                    name="head_training.prorm_moment_map_identifiability",
                )
            ),
            "bt_head_sha256": head_hashes["primary_bt_mle"],
            "prorm_plus_head_sha256": head_hashes["primary_prorm_plus"],
            "low_dimensional_head_sha256": head_hashes["low_dimensional_prorm_plus"],
            "low_dimensional_projection_sha256": projection["projection_sha256"],
            "low_dimensional_moment_map_identifiability_sha256": _canonical_sha256(
                _mapping(
                    low["projected_prorm_moment_map_identifiability"],
                    name="head_training.low.projected_moment_map_identifiability",
                )
            ),
            "exact_margin_head_sha256": head_hashes["exact_margin_prorm_plus"],
            "exact_soft_label_bt_head_sha256": head_hashes["exact_soft_label_bt"],
            "direct_oracle_direction_sha256": native["direction_sha256"],
        }
    )
    if audit["training_instance_sha256"] != training_instance:
        raise ValueError("head_training.training_instance_sha256 is not rederived")
    return selected_steps


def _derive_recovery_output_verification(
    result: Mapping[str, object],
    *,
    result_sha256: str,
    seed: int,
) -> dict[str, object]:
    """Recompute the exact receipt produced by the frozen ad7613 sbatch."""

    boundary = _mapping(result.get("information_boundary"), name="information_boundary")
    training = _mapping(result.get("head_training"), name="head_training")
    audit = _mapping(training.get("audit"), name="head_training.audit")
    low_control = _mapping(
        audit.get("low_dimensional_control"),
        name="audit.low_dimensional_control",
    )
    rescore = _mapping(result.get("train_oracle_rescore"), name="train_oracle_rescore")
    transformed_reward_sha = _digest(
        rescore.get("transformed_rewards_sha256"),
        name="train_oracle_rescore.transformed_rewards_sha256",
    )
    selected_steps = _deep_validate_recovery_training(
        training,
        seed=seed,
        train_oracle_reward_sha256=transformed_reward_sha,
    )
    optimization = _mapping(
        audit.get("primary_optimization_audit"),
        name="primary_optimization_audit",
    )
    reward_rank = _mapping(
        optimization.get("reward_head_identifiability"),
        name="reward_head_identifiability",
    )
    moment_rank = _mapping(
        optimization.get("prorm_moment_map_identifiability"),
        name="prorm_moment_map_identifiability",
    )
    low_rank = _mapping(
        low_control.get("projected_prorm_moment_map_identifiability"),
        name="projected_prorm_moment_map_identifiability",
    )
    if (
        reward_rank.get("schema_version") != "reward-head-identifiability/v2"
        or reward_rank.get("algorithmic_tie_break") != _RECOVERY_TIE_BREAK
        or reward_rank.get("full_design_rank_proves_finite_bt_minimizer_exists") is not False
        or not isinstance(
            reward_rank.get("mixed_outcome_edge_coercivity_diagnostic"),
            Mapping,
        )
        or moment_rank.get("schema_version") != "prorm-moment-map-identifiability/v2"
        or moment_rank.get("algorithmic_tie_break") != _RECOVERY_TIE_BREAK
        or low_rank.get("schema_version") != "projected-prorm-moment-map-identifiability/v2"
        or low_rank.get("algorithmic_tie_break") != _RECOVERY_TIE_BREAK
    ):
        raise ValueError("recovery identifiability/tie-break evidence is invalid")

    label = _mapping(audit.get("label_stream"), name="label_stream")
    if (
        rescore.get("raw_oracle_logits_serialized") is not False
        or label.get("oracle_reward_sha256") != transformed_reward_sha
    ):
        raise ValueError("recovery oracle/label reward identity is invalid")

    anchor_applicable = seed == ORDERED_SEEDS[0]
    anchor_passed: bool | None = None
    if anchor_applicable:
        expected_anchor = {
            "namespace": "prorm-common-beta-r4-labels-v1",
            "base_seed": 20260801,
            "derived_seed": 2443486425476852717,
            "derivation_sha256": (
                "a1e901f534096bedb757ba978e2ba9838031aeacab4e86265557279953b236ae"
            ),
            "initial_state_sha256": (
                "6f8e7260e641e4f52990e6f28c6558f333133bd0286549dbca3de426bf51a3d1"
            ),
            "final_state_sha256": (
                "7dbc6a6143c98a995abda1baab73de8867267121c1299d80b6c10f8165f6ce83"
            ),
            "mean_h_sha256": ("524eb0c9936dffe8d0ef807d4b4181cb3c4bbad556a51c7578a16c06b1e13cf0"),
            "replicate_count_sha256": (
                "b905ce98d6ec87a03bbda10405e3ffc766ff913bf9613e58bb50bf1ffa7b63c8"
            ),
            "replicate_win_sha256": (
                "de68ae122cfe40a09146fdd00f367b640bdcc03a12aff46e6713fe7e88cafb13"
            ),
            "replicate_h_sha256": (
                "92a99a227ce5679049c856b3fbd92005d0d9a8460760bc346a5ef266d7d3350d"
            ),
            "realized_total_annotations": 61011,
        }
        if (
            transformed_reward_sha
            != "7a7d7b005ec7e377205d6f40743bed950ad38154dec6f54516f7ced8ffca0b1a"
            or any(label.get(key) != expected for key, expected in expected_anchor.items())
        ):
            raise ValueError("diagnostic seed oracle/label reproducibility anchor failed")
        anchor_passed = True

    execution_identity = _mapping(
        result.get("recovery_execution_identity"),
        name="recovery_execution_identity",
    )
    parent_binding = _mapping(
        result.get("parent_failure_binding"),
        name="parent_failure_binding",
    )
    parent_producer = _mapping(
        parent_binding.get("parent_artifact_producer"),
        name="parent_artifact_producer",
    )
    heads = training.get("heads")
    if (
        result.get("schema_version") != "prorm-phase2-recovery-train-only-result/v1"
        or result.get("status") != "SUCCESS"
        or result.get("seed") != seed
        or result.get("recovery_design_sha256") != RECOVERY_DESIGN_SHA256
        or result.get("source_config_hash") != SOURCE_CONFIG_HASH
        or execution_identity.get("formal") is not True
        or execution_identity.get("git_commit") != RECOVERY_GIT_COMMIT
        or execution_identity.get("image_sha256") != IMAGE_SHA256
        or execution_identity.get("hf_inventory_sha256") != HF_INVENTORY_SHA256
        or execution_identity.get("account") != "sigroup"
        or execution_identity.get("partition") != "gpu-l20"
        or execution_identity.get("gpu_models") != ["NVIDIA L20"]
        or parent_binding.get("parent_phase2_design_sha256") != PARENT_DESIGN_SHA256
        or parent_binding.get("registry_sha256") != PARENT_REGISTRY_SHA256
        or parent_producer.get("git_commit") != PARENT_PRODUCER_GIT_COMMIT
        or parent_producer.get("image_sha256") != IMAGE_SHA256
        or parent_producer.get("hf_inventory_sha256") != HF_INVENTORY_SHA256
        or result.get("one_shot_no_further_adaptation") is not True
        or training.get("schema_version") != "phase2-fresh-head-training/v3"
        or audit.get("schema_version") != "phase2-fresh-head-training/v3"
        or audit.get("training_design_sha256") != RECOVERY_DESIGN_SHA256
        or audit.get("training_settings_sha256") != TRAINING_SETTINGS_SHA256
        or not isinstance(audit.get("training_instance_sha256"), str)
        or not isinstance(audit.get("input_training_sha256"), str)
        or not isinstance(heads, Mapping)
        or set(heads) != {"bt_mle", "prorm_plus"}
        or boundary.get("validation_tensors_decoded") is not False
        or boundary.get("test_tensors_decoded") is not False
        or boundary.get("validation_or_test_candidates_decoded") is not False
        or boundary.get("policy_session_opened") is not False
        or boundary.get("policy_rollout_performed") is not False
        or boundary.get("heldout_evaluator_called") is not False
        or boundary.get("final_oracle_session_opened") is not False
        or boundary.get("downstream_utility_computed") is not False
    ):
        raise ValueError("recovery output identity or train-only boundary is invalid")
    _digest(audit.get("training_instance_sha256"), name="training_instance_sha256")
    _digest(audit.get("input_training_sha256"), name="input_training_sha256")
    return {
        "status": "ok",
        "result_sha256": result_sha256,
        "five_head_recovery_protocol_verified": True,
        "selected_primary_steps": selected_steps,
        "diagnostic_seed_reproduction": {
            "anchor_seed": ORDERED_SEEDS[0],
            "applicable": anchor_applicable,
            "passed": anchor_passed,
        },
    }


def _validate_output_verification(
    value: object,
    *,
    result_sha256: str,
    seed: int,
    result: Mapping[str, object],
) -> None:
    verification = _exact_keys(
        value,
        name="recovery-output-verification.json",
        expected=_OUTPUT_VERIFICATION_KEYS,
    )
    steps = _exact_keys(
        verification["selected_primary_steps"],
        name="recovery-output-verification.json.selected_primary_steps",
        expected=_FIVE_HEAD_NAMES,
    )
    for name, value in steps.items():
        step = _integer(value, name=f"selected_primary_steps.{name}", minimum=100)
        if step > 12760 or step % 20:
            raise ValueError(f"selected_primary_steps.{name} exceeds the frozen schedule")
    anchor = _exact_keys(
        verification["diagnostic_seed_reproduction"],
        name="recovery-output-verification.json.diagnostic_seed_reproduction",
        expected={"anchor_seed", "applicable", "passed"},
    )
    anchor_applicable = seed == ORDERED_SEEDS[0]
    derived = _derive_recovery_output_verification(
        result,
        result_sha256=result_sha256,
        seed=seed,
    )
    if (
        verification["status"] != "ok"
        or verification["result_sha256"] != result_sha256
        or verification["five_head_recovery_protocol_verified"] is not True
        or anchor["anchor_seed"] != ORDERED_SEEDS[0]
        or anchor["applicable"] is not anchor_applicable
        or anchor["passed"] is not (True if anchor_applicable else None)
        or dict(verification) != derived
    ):
        raise ValueError("recovery-output-verification.json is not the rederived five-head receipt")


def _validate_seed(
    path: Path,
    *,
    seed: int,
    array_task_id: int,
    scheduler_row: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    _validate_run_path(path, seed=seed, array_task_id=array_task_id)
    for forbidden in ("FAILED", "recovery-failure-evidence.json"):
        candidate = path / forbidden
        if candidate.exists() or candidate.is_symlink():
            raise ValueError(f"successful recovery run also contains forbidden {forbidden}")

    before = _snapshot_evidence(path)
    success, _ = _parse_success(path / "SUCCESS")
    _validate_success(success, seed=seed, array_task_id=array_task_id)
    parent_verification = _validate_parent_receipts(path, seed=seed)

    manifest, manifest_raw = _read_json(
        path / "run-manifest.json",
        name="run-manifest.json",
    )
    environment = _validate_manifest(
        manifest,
        seed=seed,
        array_task_id=array_task_id,
        scheduler_row=scheduler_row,
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    gpu_check, _ = _read_json(
        path / "gpu-check.json",
        name="gpu-check.json",
        maximum_bytes=64 * 1024,
    )
    _validate_gpu_check(
        gpu_check,
        manifest_environment=environment,
        scheduler_row=scheduler_row,
    )

    result, result_raw = _read_json(
        path / "recovery-result.json",
        name="recovery-result.json",
        require_canonical=True,
    )
    result_sha256 = hashlib.sha256(result_raw).hexdigest()
    _validate_result(
        result,
        result_sha256=result_sha256,
        manifest_sha256=manifest_sha256,
        environment=environment,
        seed=seed,
        parent_verification=parent_verification,
    )

    verification, verification_raw = _read_json(
        path / "recovery-output-verification.json",
        name="recovery-output-verification.json",
        require_canonical=True,
        maximum_bytes=1024 * 1024,
    )
    _validate_output_verification(
        verification,
        result_sha256=result_sha256,
        seed=seed,
        result=result,
    )

    if _read_stable_file(
        path / "artifact-snapshot-before.json",
        name="artifact snapshot before",
        maximum_bytes=1024 * 1024,
    ) != _read_stable_file(
        path / "artifact-snapshot-after.json",
        name="artifact snapshot after",
        maximum_bytes=1024 * 1024,
    ) or _read_stable_file(
        path / "parent-run-snapshot-before.json",
        name="parent run snapshot before",
        maximum_bytes=1024 * 1024,
    ) != _read_stable_file(
        path / "parent-run-snapshot-after.json",
        name="parent run snapshot after",
        maximum_bytes=1024 * 1024,
    ):
        raise ValueError("parent recovery evidence changed during the train-only execution")

    after = _snapshot_evidence(path)
    if before != after:
        raise ValueError("recovery source evidence changed during aggregation")
    result_inventory = _inventory_file(after, "recovery-result.json")
    verification_inventory = _inventory_file(after, "recovery-output-verification.json")
    success_inventory = _inventory_file(after, "SUCCESS")
    if (
        result_inventory["sha256"] != result_sha256
        or verification_inventory["sha256"] != hashlib.sha256(verification_raw).hexdigest()
        or success_inventory["sha256"]
        != hashlib.sha256(
            _read_stable_file(
                path / "SUCCESS",
                name="SUCCESS marker",
                maximum_bytes=64 * 1024,
            )
        ).hexdigest()
    ):
        raise ValueError("source evidence inventory disagrees with parsed recovery evidence")

    source = {
        "seed": seed,
        "array_task_id": array_task_id,
        "job_id": f"{SOURCE_ARRAY_JOB_ID}_{array_task_id}",
        "success_marker_sha256": success_inventory["sha256"],
        "recovery_result_sha256": result_sha256,
        "output_verification_sha256": verification_inventory["sha256"],
        "evidence_inventory_sha256": _canonical_sha256(after),
        "evidence": after,
    }
    return source, environment


def _validator_source_sha256() -> str:
    return _sha256_file(Path(__file__), name="loaded recovery authorization validator")


def _loaded_dependency_sha256(relative: Path, *, name: str) -> str:
    return _sha256_file(_repository_root() / relative, name=name)


def _git_text(arguments: Sequence[str], *, name: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(_repository_root()), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"could not inspect {name}") from error
    if completed.returncode != 0 or completed.stderr:
        raise ValueError(f"{name} Git inspection failed")
    return completed.stdout


def _validate_claimed_aggregator_git_identity(
    *,
    aggregator_git_commit: str,
    validator_sha256: str,
    deep_gate_source_sha256: str,
    tensor_hash_source_sha256: str,
    config_validator_source_sha256: str,
    recovery_config_sha256: str,
) -> None:
    _digest(
        aggregator_git_commit,
        name="aggregation_identity.aggregator_git_commit",
        lengths=frozenset({40, 64}),
    )
    relative = Path("src/smart_reward/phase2_recovery_aggregate.py")
    blob = _git_blob(
        aggregator_git_commit,
        relative,
        name="claimed recovery authorization validator",
    )
    if (
        hashlib.sha256(blob).hexdigest() != validator_sha256
        or _read_stable_file(
            Path(__file__),
            name="loaded recovery authorization validator",
            maximum_bytes=8 * 1024 * 1024,
        )
        != blob
    ):
        raise ValueError("claimed aggregation commit does not bind the loaded validator")
    dependencies = (
        (
            _DEEP_GATE_SOURCE_RELATIVE,
            deep_gate_source_sha256,
            "loaded Phase-2 deep gate",
        ),
        (
            _TENSOR_HASH_SOURCE_RELATIVE,
            tensor_hash_source_sha256,
            "loaded Phase-2 tensor-hash/training source",
        ),
        (
            _CONFIG_VALIDATOR_SOURCE_RELATIVE,
            config_validator_source_sha256,
            "loaded Phase-2 configuration validator",
        ),
    )
    for dependency, recorded, name in dependencies:
        digest = _digest(recorded, name=f"aggregation_identity.{dependency.name}")
        dependency_blob = _git_blob(
            aggregator_git_commit,
            dependency,
            name=name,
        )
        if (
            hashlib.sha256(dependency_blob).hexdigest() != digest
            or _read_stable_file(
                _repository_root() / dependency,
                name=name,
                maximum_bytes=8 * 1024 * 1024,
            )
            != dependency_blob
        ):
            raise ValueError(
                "claimed aggregation commit does not bind every loaded deep-gate source"
            )
    frozen_config_digest = _digest(
        recovery_config_sha256,
        name="aggregation_identity.recovery_config_sha256",
    )
    recovery_config_blob = _git_blob(
        RECOVERY_GIT_COMMIT,
        _RECOVERY_CONFIG_RELATIVE,
        name="recovery producer configuration",
    )
    if (
        frozen_config_digest != _RECOVERY_CONFIG_SHA256
        or hashlib.sha256(recovery_config_blob).hexdigest() != frozen_config_digest
    ):
        raise ValueError("aggregation identity does not bind the recovery producer config")
    head = _git_text(["rev-parse", "--verify", "HEAD"], name="consumer HEAD").strip()
    try:
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(_repository_root()),
                "merge-base",
                "--is-ancestor",
                aggregator_git_commit,
                head,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not inspect aggregation commit ancestry") from error
    if ancestor.returncode != 0 or ancestor.stderr:
        raise ValueError("aggregation commit is not an ancestor of the consumer checkout")


def _reject_forbidden_aggregate_keys(value: object, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} contains a non-string key")
            lowered = key.lower()
            if any(forbidden in lowered for forbidden in ("head", "vector", "path")):
                raise ValueError(f"{location}.{key} is forbidden in the head-free aggregate")
            _reject_forbidden_aggregate_keys(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_aggregate_keys(item, location=f"{location}[{index}]")


def _validate_scheduler_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or len(value) != len(ORDERED_SEEDS):
        raise ValueError("authorization scheduler rows are not the exact terminal array")
    raw_job_ids: set[str] = set()
    rows: list[Mapping[str, object]] = []
    for task, (seed, raw_row) in enumerate(zip(ORDERED_SEEDS, value, strict=True)):
        row = _exact_keys(
            raw_row,
            name=f"recovery success authorization.scheduler_terminal.rows[{task}]",
            expected=_SCHEDULER_ROW_KEYS,
        )
        raw_job_id = row["job_id_raw"]
        if (
            row["job_id"] != f"{SOURCE_ARRAY_JOB_ID}_{task}"
            or not isinstance(raw_job_id, str)
            or re.fullmatch(r"[1-9][0-9]*", raw_job_id) is None
            or raw_job_id in raw_job_ids
            or row["array_job_id"] != SOURCE_ARRAY_JOB_ID
            or row["array_task_id"] != task
            or row["seed"] != seed
            or row["state"] != "COMPLETED"
            or row["exit_code"] != "0:0"
            or row["derived_exit_code"] != "0:0"
            or row["cluster"] != "hpc4"
            or row["account"] != "sigroup"
            or row["partition"] != "gpu-l20"
            or row["n_nodes"] != 1
            or row["n_cpus"] != 8
            or row["requested_tres"] != _SACCT_REQUEST_TRES
            or row["allocated_tres"] != _SACCT_ALLOCATED_TRES
        ):
            raise ValueError("authorization scheduler rows are not the exact terminal array")
        raw_job_ids.add(raw_job_id)
        rows.append(row)
    return rows


def _validate_authorization_payload(
    value: object,
    *,
    require_current_validator: bool,
) -> dict[str, object]:
    _reject_forbidden_aggregate_keys(value)
    authorization = _exact_keys(
        value,
        name="recovery success authorization",
        expected=_AUTHORIZATION_KEYS,
    )
    reuse = _exact_keys(
        authorization["recovery_output_reuse"],
        name="recovery success authorization.recovery_output_reuse",
        expected={"beta", "reward_model_parameters", "policy"},
    )
    boundary = _exact_keys(
        authorization["information_boundary"],
        name="recovery success authorization.information_boundary",
        expected=set(_AGGREGATE_BOUNDARY),
    )
    namespace = _exact_keys(
        authorization["campaign_namespace"],
        name="recovery success authorization.campaign_namespace",
        expected=_NAMESPACE_KEYS,
    )
    live_control = _exact_keys(
        authorization["supplementary_submission_control"],
        name="recovery success authorization.supplementary_submission_control",
        expected=_LIVE_CONTROL_KEYS,
    )
    environment = _exact_keys(
        authorization["environment_identity"],
        name="recovery success authorization.environment_identity",
        expected=_ENVIRONMENT_KEYS,
    )
    parent = _exact_keys(
        authorization["parent_failure_identity"],
        name="recovery success authorization.parent_failure_identity",
        expected={"phase2_design_sha256", "registry_sha256", "producer_git_commit"},
    )
    aggregation_identity = _exact_keys(
        authorization["aggregation_identity"],
        name="recovery success authorization.aggregation_identity",
        expected={
            "schema_version",
            "aggregator_git_commit",
            "recovery_producer_git_commit",
            "validator_source_sha256",
            "deep_gate_source_sha256",
            "tensor_hash_source_sha256",
            "config_validator_source_sha256",
            "recovery_config_sha256",
        },
    )
    scheduler_terminal = _exact_keys(
        authorization["scheduler_terminal"],
        name="recovery success authorization.scheduler_terminal",
        expected={
            "rows",
            "scheduler_evidence_sha256",
            "raw_sacct_sha256",
            "raw_sacct_size_bytes",
        },
    )
    scheduler_source = _exact_keys(
        authorization["scheduler_source"],
        name="recovery success authorization.scheduler_source",
        expected={
            "scheduler_evidence_sha256",
            "raw_sacct_sha256",
            "raw_sacct_size_bytes",
        },
    )
    expected_environment = {
        "formal": True,
        "git_commit": RECOVERY_GIT_COMMIT,
        "image_sha256": IMAGE_SHA256,
        "hf_inventory_sha256": HF_INVENTORY_SHA256,
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_models": ["NVIDIA L20"],
    }
    expected_parent = {
        "phase2_design_sha256": PARENT_DESIGN_SHA256,
        "registry_sha256": PARENT_REGISTRY_SHA256,
        "producer_git_commit": PARENT_PRODUCER_GIT_COMMIT,
    }
    if (
        authorization["schema_version"] != RECOVERY_SUCCESS_AUTHORIZATION_SCHEMA
        or authorization["source_config_hash"] != SOURCE_CONFIG_HASH
        or authorization["recovery_design_sha256"] != RECOVERY_DESIGN_SHA256
        or authorization["optimizer_schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256
        or authorization["training_settings_sha256"] != TRAINING_SETTINGS_SHA256
        or authorization["source_array_job_id"] != SOURCE_ARRAY_JOB_ID
        or authorization["execution_revision"] != EXECUTION_REVISION
        or authorization["ordered_seeds"] != list(ORDERED_SEEDS)
        or authorization["recovery_status"] != "all_three_seeds_success"
        or authorization["full_calibration_authorized"] is not True
        or authorization["authorized_information"] != "optimizer_schedule_only"
        or authorization["authorized_next_action"]
        != "issue_schedule_frozen_full_common_beta_calibration_pilot"
        or authorization["recovery_outputs_reusable"] is not False
        or authorization["validation_or_heldout_access_authorized"] is not False
        or authorization["policy_or_final_utility_access_authorized"] is not False
        or authorization["formal_efficacy_claim_authorized"] is not False
        or dict(reuse)
        != {
            "beta": False,
            "reward_model_parameters": False,
            "policy": False,
        }
        or dict(boundary) != _AGGREGATE_BOUNDARY
        or dict(namespace) != _campaign_namespace_identity()
        or dict(live_control) != _expected_live_control()
        or dict(environment) != expected_environment
        or dict(parent) != expected_parent
        or aggregation_identity["schema_version"] != RECOVERY_AGGREGATION_IDENTITY_SCHEMA
        or aggregation_identity["recovery_producer_git_commit"] != RECOVERY_GIT_COMMIT
    ):
        raise ValueError(
            "recovery success authorization grants more than the one schedule-only action"
        )
    aggregator_git_commit = _digest(
        aggregation_identity["aggregator_git_commit"],
        name="aggregation_identity.aggregator_git_commit",
        lengths=frozenset({40, 64}),
    )
    validator_sha256 = _digest(
        aggregation_identity["validator_source_sha256"],
        name="aggregation_identity.validator_source_sha256",
    )
    deep_gate_source_sha256 = _digest(
        aggregation_identity["deep_gate_source_sha256"],
        name="aggregation_identity.deep_gate_source_sha256",
    )
    tensor_hash_source_sha256 = _digest(
        aggregation_identity["tensor_hash_source_sha256"],
        name="aggregation_identity.tensor_hash_source_sha256",
    )
    config_validator_source_sha256 = _digest(
        aggregation_identity["config_validator_source_sha256"],
        name="aggregation_identity.config_validator_source_sha256",
    )
    recovery_config_sha256 = _digest(
        aggregation_identity["recovery_config_sha256"],
        name="aggregation_identity.recovery_config_sha256",
    )
    if require_current_validator:
        if validator_sha256 != _validator_source_sha256():
            raise ValueError("authorization does not bind the loaded aggregate validator")
        _validate_claimed_aggregator_git_identity(
            aggregator_git_commit=aggregator_git_commit,
            validator_sha256=validator_sha256,
            deep_gate_source_sha256=deep_gate_source_sha256,
            tensor_hash_source_sha256=tensor_hash_source_sha256,
            config_validator_source_sha256=config_validator_source_sha256,
            recovery_config_sha256=recovery_config_sha256,
        )

    scheduler_rows = _validate_scheduler_rows(scheduler_terminal["rows"])
    live_rows = live_control["rows"]
    if not isinstance(live_rows, list) or len(live_rows) != len(scheduler_rows):
        raise ValueError("supplementary live control receipt has invalid task rows")
    for task, (scheduler_row, raw_live_row) in enumerate(
        zip(scheduler_rows, live_rows, strict=True)
    ):
        live_row = _exact_keys(
            raw_live_row,
            name=f"supplementary live control row {task}",
            expected=_LIVE_ROW_KEYS,
        )
        if (
            live_row["job_id_raw"] != scheduler_row["job_id_raw"]
            or live_row["array_job_id"] != scheduler_row["array_job_id"]
            or live_row["array_task_id"] != scheduler_row["array_task_id"]
            or live_row["state_at_capture"] != _LIVE_STATES[task]
            or live_row["requeue"] != 0
            or live_row["restarts_at_capture"] != 0
            or live_row["num_nodes"] != scheduler_row["n_nodes"]
            or live_row["num_cpus"] != scheduler_row["n_cpus"]
            or live_row["generic_gpu_count"] != 1
            or scheduler_row["requested_tres"] != _SACCT_REQUEST_TRES
            or scheduler_row["allocated_tres"] != _SACCT_ALLOCATED_TRES
        ):
            raise ValueError("live control identity disagrees with terminal scheduler rows")
    for field in ("scheduler_evidence_sha256", "raw_sacct_sha256"):
        _digest(scheduler_terminal[field], name=f"scheduler_terminal.{field}")
        if scheduler_terminal[field] != scheduler_source[field]:
            raise ValueError("authorization scheduler source hashes disagree")
    raw_size = _integer(
        scheduler_terminal["raw_sacct_size_bytes"],
        name="scheduler_terminal.raw_sacct_size_bytes",
        minimum=1,
    )
    if scheduler_source["raw_sacct_size_bytes"] != raw_size:
        raise ValueError("authorization scheduler source byte sizes disagree")

    raw_sources = authorization["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != len(ORDERED_SEEDS):
        raise ValueError("authorization must bind exactly three ordered seed sources")
    expected_direct_files = {
        "SUCCESS": "success_marker_sha256",
        "recovery-result.json": "recovery_result_sha256",
        "recovery-output-verification.json": "output_verification_sha256",
    }
    for task, (seed, raw_source) in enumerate(zip(ORDERED_SEEDS, raw_sources, strict=True)):
        source = _exact_keys(
            raw_source,
            name=f"recovery success authorization.sources[{task}]",
            expected={
                "seed",
                "array_task_id",
                "job_id",
                "success_marker_sha256",
                "recovery_result_sha256",
                "output_verification_sha256",
                "evidence_inventory_sha256",
                "evidence",
            },
        )
        evidence = source["evidence"]
        if not isinstance(evidence, list):
            raise ValueError(f"authorization source {task} evidence must be a list")
        expected_names = set(_REQUIRED_EVIDENCE_FILES) | {_REQUIRED_REFERENCE}
        observed_names: set[str] = set()
        for index, raw_entry in enumerate(evidence):
            entry = _exact_keys(
                raw_entry,
                name=f"recovery success authorization.sources[{task}].evidence[{index}]",
                expected={"name", "kind", "size_bytes", "sha256"},
            )
            name = entry["name"]
            if not isinstance(name, str) or name in observed_names:
                raise ValueError(f"authorization source {task} has invalid evidence names")
            observed_names.add(name)
            expected_kind = "symlink_reference" if name == _REQUIRED_REFERENCE else "regular_file"
            if name not in expected_names or entry["kind"] != expected_kind:
                raise ValueError(f"authorization source {task} has invalid evidence kind")
            _integer(
                entry["size_bytes"],
                name=f"authorization source {task} evidence size",
            )
            _digest(
                entry["sha256"],
                name=f"authorization source {task} evidence SHA256",
            )
        if (
            observed_names != expected_names
            or [entry["name"] for entry in evidence] != sorted(observed_names)
            or source["seed"] != seed
            or source["array_task_id"] != task
            or source["job_id"] != f"{SOURCE_ARRAY_JOB_ID}_{task}"
        ):
            raise ValueError(f"authorization source {task} identity/inventory is invalid")
        inventory_sha = _digest(
            source["evidence_inventory_sha256"],
            name=f"authorization source {task} inventory SHA256",
        )
        if inventory_sha != _canonical_sha256(evidence):
            raise ValueError(f"authorization source {task} inventory hash is invalid")
        by_name = {str(entry["name"]): entry for entry in evidence}
        for filename, source_field in expected_direct_files.items():
            _digest(
                source[source_field],
                name=f"authorization source {task} {source_field}",
            )
            if source[source_field] != by_name[filename]["sha256"]:
                raise ValueError(f"authorization source {task} direct evidence hash disagrees")

    for key, item in authorization.items():
        if (
            isinstance(item, bool)
            and item is True
            and "authorized" in key
            and key != "full_calibration_authorized"
        ):
            raise ValueError(f"unexpected positive authorization: {key}")
    return dict(authorization)


def verify_phase2_recovery_authorization(
    path: str | os.PathLike[str],
    expected_sha256: str,
) -> dict[str, object]:
    """Verify one canonical authorization artifact and its external SHA256."""

    expected = _digest(expected_sha256, name="expected authorization SHA256")
    source = Path(path).absolute()
    _require_exact_project_path(
        source,
        _AUTHORIZATION_RELATIVE,
        name="recovery success authorization",
    )
    value, raw = _read_json(
        source,
        name="recovery success authorization",
        require_canonical=True,
        maximum_bytes=16 * 1024 * 1024,
    )
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("recovery success authorization SHA256 mismatch")
    return _validate_authorization_payload(value, require_current_validator=True)


def build_phase2_recovery_authorization(
    run_directories: Sequence[str | os.PathLike[str]],
    *,
    scheduler_evidence: str | os.PathLike[str],
    aggregator_git_commit: str,
) -> dict[str, object]:
    """Validate three recovery SUCCESS runs and build a head-free authorization."""

    _digest(
        aggregator_git_commit,
        name="aggregator_git_commit",
        lengths=frozenset({40, 64}),
    )
    if len(run_directories) != len(ORDERED_SEEDS):
        raise ValueError("recovery authorization requires exactly three ordered run directories")
    paths = tuple(Path(item).absolute() for item in run_directories)
    if len(set(paths)) != len(paths):
        raise ValueError("recovery authorization run directories must be distinct")
    expected_execution_root = _exact_project_path(
        _RECOVERY_EXECUTION_RELATIVE,
        name="frozen recovery execution root",
        must_exist=True,
    )
    for path in paths:
        _require_real_directory(path, name="ordered recovery run directory")
    recovery_root = paths[0].parents[1]
    if recovery_root != expected_execution_root or any(
        path.parents[1] != recovery_root for path in paths[1:]
    ):
        raise ValueError("ordered recovery sources are outside the frozen production execution")
    scheduler_path = Path(scheduler_evidence).absolute()
    scheduler_terminal, scheduler_source = _validate_scheduler_evidence(scheduler_path)
    live_control = _load_live_control()
    scheduler_rows = scheduler_terminal["rows"]
    if not isinstance(scheduler_rows, list):
        raise ValueError("terminal scheduler evidence has no task rows")

    sources: list[dict[str, object]] = []
    environments: list[dict[str, object]] = []
    for array_task_id, (seed, path, raw_scheduler_row) in enumerate(
        zip(ORDERED_SEEDS, paths, scheduler_rows, strict=True)
    ):
        scheduler_row = _mapping(
            raw_scheduler_row,
            name=f"terminal scheduler row {array_task_id}",
        )
        source, environment = _validate_seed(
            path,
            seed=seed,
            array_task_id=array_task_id,
            scheduler_row=scheduler_row,
        )
        sources.append(source)
        environments.append(environment)
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("recovery seeds do not share one exact execution environment identity")
    for task, (path, source) in enumerate(zip(paths, sources, strict=True)):
        if _snapshot_evidence(path) != source["evidence"]:
            raise ValueError(
                f"recovery source evidence for array task {task} changed during aggregation"
            )
    scheduler_terminal_after, scheduler_source_after = _validate_scheduler_evidence(scheduler_path)
    if scheduler_terminal_after != scheduler_terminal or scheduler_source_after != scheduler_source:
        raise ValueError("scheduler source evidence changed during aggregation")

    payload: dict[str, object] = {
        "schema_version": RECOVERY_SUCCESS_AUTHORIZATION_SCHEMA,
        "source_config_hash": SOURCE_CONFIG_HASH,
        "recovery_design_sha256": RECOVERY_DESIGN_SHA256,
        "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
        "training_settings_sha256": TRAINING_SETTINGS_SHA256,
        "source_array_job_id": SOURCE_ARRAY_JOB_ID,
        "execution_revision": EXECUTION_REVISION,
        "ordered_seeds": list(ORDERED_SEEDS),
        "recovery_status": "all_three_seeds_success",
        "full_calibration_authorized": True,
        "authorized_information": "optimizer_schedule_only",
        "authorized_next_action": ("issue_schedule_frozen_full_common_beta_calibration_pilot"),
        "recovery_outputs_reusable": False,
        "validation_or_heldout_access_authorized": False,
        "policy_or_final_utility_access_authorized": False,
        "formal_efficacy_claim_authorized": False,
        "recovery_output_reuse": {
            "beta": False,
            "reward_model_parameters": False,
            "policy": False,
        },
        "information_boundary": {
            "source_results_train_only": True,
            "validation_tensors_decoded": False,
            "test_tensors_decoded": False,
            "validation_or_test_candidates_decoded": False,
            "policy_rollout_performed": False,
            "heldout_evaluator_called": False,
            "final_oracle_session_opened": False,
            "downstream_utility_computed": False,
            "trained_parameters_extracted_for_aggregation": False,
            "trained_parameters_serialized_in_aggregate": False,
        },
        "campaign_namespace": _campaign_namespace_identity(),
        "scheduler_terminal": scheduler_terminal,
        "supplementary_submission_control": live_control,
        "environment_identity": environments[0],
        "parent_failure_identity": {
            "phase2_design_sha256": PARENT_DESIGN_SHA256,
            "registry_sha256": PARENT_REGISTRY_SHA256,
            "producer_git_commit": PARENT_PRODUCER_GIT_COMMIT,
        },
        "aggregation_identity": {
            "schema_version": RECOVERY_AGGREGATION_IDENTITY_SCHEMA,
            "aggregator_git_commit": aggregator_git_commit,
            "recovery_producer_git_commit": RECOVERY_GIT_COMMIT,
            "validator_source_sha256": _validator_source_sha256(),
            "deep_gate_source_sha256": _loaded_dependency_sha256(
                _DEEP_GATE_SOURCE_RELATIVE,
                name="loaded Phase-2 deep gate",
            ),
            "tensor_hash_source_sha256": _loaded_dependency_sha256(
                _TENSOR_HASH_SOURCE_RELATIVE,
                name="loaded Phase-2 tensor-hash/training source",
            ),
            "config_validator_source_sha256": _loaded_dependency_sha256(
                _CONFIG_VALIDATOR_SOURCE_RELATIVE,
                name="loaded Phase-2 configuration validator",
            ),
            "recovery_config_sha256": _RECOVERY_CONFIG_SHA256,
        },
        "scheduler_source": {
            **scheduler_source,
        },
        "sources": sources,
    }
    return _validate_authorization_payload(
        payload,
        require_current_validator=True,
    )


def _write_phase2_recovery_authorization_with_digest(
    run_directories: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    scheduler_evidence: str | os.PathLike[str],
    aggregator_git_commit: str,
) -> tuple[dict[str, object], str]:
    """Build and atomically publish canonical JSON without overwriting."""

    destination = Path(output).absolute()
    _require_exact_project_path(
        destination,
        _AUTHORIZATION_RELATIVE,
        name="recovery authorization output",
        must_exist=False,
    )
    _require_real_directory(destination.parent, name="authorization output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite existing recovery authorization: {destination}"
        )
    payload = build_phase2_recovery_authorization(
        run_directories,
        scheduler_evidence=scheduler_evidence,
        aggregator_git_commit=aggregator_git_commit,
    )
    encoded = _canonical_bytes(payload)
    expected_digest = hashlib.sha256(encoded).hexdigest()
    for task, (raw_path, source) in enumerate(
        zip(run_directories, payload["sources"], strict=True)
    ):
        path = Path(raw_path).absolute()
        _validate_parent_receipts(path, seed=ORDERED_SEEDS[task])
        if not isinstance(source, Mapping) or _snapshot_evidence(path) != source["evidence"]:
            raise ValueError(
                f"recovery source evidence for array task {task} changed before publication"
            )
    scheduler_terminal, scheduler_source = _validate_scheduler_evidence(
        Path(scheduler_evidence).absolute()
    )
    if (
        scheduler_terminal != payload["scheduler_terminal"]
        or scheduler_source != payload["scheduler_source"]
        or _load_live_control() != payload["supplementary_submission_control"]
        or _campaign_namespace_identity() != payload["campaign_namespace"]
    ):
        raise ValueError("scheduler evidence changed before authorization publication")
    _verify_cli_checkout(aggregator_git_commit)
    published_digest = _write_exclusive_bytes(
        destination,
        encoded,
        label="recovery authorization",
    )
    if published_digest != expected_digest:
        raise RuntimeError("published authorization digest differs from canonical bytes")
    return payload, expected_digest


def write_phase2_recovery_authorization(
    run_directories: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    scheduler_evidence: str | os.PathLike[str],
    aggregator_git_commit: str,
) -> dict[str, object]:
    """Publish canonical authorization while retaining the historical dict API."""

    payload, _ = _write_phase2_recovery_authorization_with_digest(
        run_directories,
        output,
        scheduler_evidence=scheduler_evidence,
        aggregator_git_commit=aggregator_git_commit,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the head-free authorization for the frozen Phase-2 recovery execution."
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "run_directories",
        type=Path,
        nargs=3,
        metavar="ORDERED_RUN_DIR",
    )
    parser.add_argument("--scheduler-evidence", type=Path, required=True)
    parser.add_argument("--aggregator-git-commit", required=True)
    return parser


def _verify_cli_checkout(aggregator_git_commit: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    def git(arguments: Sequence[str], *, text: bool) -> str | bytes:
        try:
            completed = subprocess.run(
                ["git", "-C", os.fspath(repo_root), *arguments],
                check=False,
                capture_output=True,
                text=text,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("could not inspect the recovery aggregation checkout") from error
        if completed.returncode != 0 or completed.stderr:
            raise RuntimeError("recovery aggregation checkout inspection failed")
        return completed.stdout

    head = str(git(["rev-parse", "--verify", "HEAD"], text=True)).strip()
    status = str(git(["status", "--porcelain", "--untracked-files=normal"], text=True))
    committed_sources = {
        Path("src/smart_reward/phase2_recovery_aggregate.py"): (_validator_source_sha256()),
        _DEEP_GATE_SOURCE_RELATIVE: _loaded_dependency_sha256(
            _DEEP_GATE_SOURCE_RELATIVE,
            name="loaded Phase-2 deep gate",
        ),
        _TENSOR_HASH_SOURCE_RELATIVE: _loaded_dependency_sha256(
            _TENSOR_HASH_SOURCE_RELATIVE,
            name="loaded Phase-2 tensor-hash/training source",
        ),
        _CONFIG_VALIDATOR_SOURCE_RELATIVE: _loaded_dependency_sha256(
            _CONFIG_VALIDATOR_SOURCE_RELATIVE,
            name="loaded Phase-2 configuration validator",
        ),
    }
    source_blobs = {
        relative: git(
            [
                "cat-file",
                "blob",
                f"{aggregator_git_commit}:{relative.as_posix()}",
            ],
            text=False,
        )
        for relative in committed_sources
    }
    if (
        head != aggregator_git_commit
        or status
        or any(
            not isinstance(source_blobs[relative], bytes)
            or hashlib.sha256(source_blobs[relative]).hexdigest() != expected
            for relative, expected in committed_sources.items()
        )
    ):
        raise ValueError(
            "recovery aggregation requires the exact clean committed validator checkout"
        )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    _digest(
        arguments.aggregator_git_commit,
        name="aggregator_git_commit",
        lengths=frozenset({40, 64}),
    )
    _verify_cli_checkout(arguments.aggregator_git_commit)
    _, digest = _write_phase2_recovery_authorization_with_digest(
        arguments.run_directories,
        arguments.output,
        scheduler_evidence=arguments.scheduler_evidence,
        aggregator_git_commit=arguments.aggregator_git_commit,
    )
    print(
        json.dumps(
            {
                "status": "authorized",
                "output": os.fspath(arguments.output),
                "sha256": digest,
                "authorized_next_action": (
                    "issue_schedule_frozen_full_common_beta_calibration_pilot"
                ),
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXECUTION_REVISION",
    "OPTIMIZER_SCHEDULE_SHA256",
    "ORDERED_SEEDS",
    "RECOVERY_DESIGN_SHA256",
    "RECOVERY_SUCCESS_AUTHORIZATION_SCHEMA",
    "RECOVERY_SCHEDULER_EVIDENCE_SCHEMA",
    "SOURCE_ARRAY_JOB_ID",
    "build_phase2_recovery_authorization",
    "capture_phase2_recovery_scheduler_evidence",
    "capture_phase2_recovery_scheduler_evidence_with_digest",
    "verify_phase2_recovery_authorization",
    "write_phase2_recovery_authorization",
]
