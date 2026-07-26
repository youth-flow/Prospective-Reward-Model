#!/usr/bin/env python3
"""Derive the immutable R3 primary segment-1 Slurm plan from successful Gate-P.

This is deliberately a pure-data command.  It does not load model tensors,
initialize CUDA, or accept resource/science overrides.  Every resource emitted
for ``sbatch`` is recovered from one caller-pinned Gate-P operational bundle
after the corresponding successful profile terminal is revalidated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from smart_reward.phase2_r3_artifacts import (
    canonical_json_bytes,
    publish_canonical_artifact,
    read_canonical_artifact,
)
from smart_reward.phase2_r3_profile_artifacts import (
    reopen_verified_gate_p_operational_bundle,
)
from smart_reward.phase2_r3_terminal import (
    reopen_profile_allocation_intent,
    reopen_profile_slurm_runtime_receipt,
    revalidate_successful_profile_terminal,
)

_SCHEMA = "phase2-recovery-r3-primary-submission-plan/v1"
_ROLE = "formal_primary_segment_1_submission_plan"
_TASK_IDS = [0, 1, 2]
_MIB = 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "segment_index",
        "array_task_ids",
        "array_concurrency",
        "slurm_account",
        "partition",
        "gpu_name",
        "gpus_per_task",
        "cpus_per_task",
        "memory_bytes",
        "memory_mib",
        "requested_walltime_seconds",
        "slurm_walltime",
        "advance_signal_lead_seconds",
        "max_scheduler_segments",
        "audit_cadence_updates",
        "durable_checkpoint_cadence_updates",
        "resource_plan_sha256",
        "operational_bundle_path",
        "operational_bundle_file_sha256",
        "operational_bundle_semantic_sha256",
        "profile_allocation_intent_path",
        "profile_allocation_intent_file_sha256",
        "profile_runtime_receipt_path",
        "profile_runtime_receipt_file_sha256",
        "profile_terminal_evidence_directory",
        "profile_terminal_manifest_file_sha256",
        "profile_terminal_raw_sacct_sha256",
        "profile_terminal_sha256",
        "submission_plan_sha256",
    }
)
_SBATCH_FIELD_ORDER = (
    "submission_plan_sha256",
    "resource_plan_sha256",
    "slurm_account",
    "partition",
    "gpu_name",
    "gpus_per_task",
    "cpus_per_task",
    "memory_bytes",
    "memory_mib",
    "requested_walltime_seconds",
    "slurm_walltime",
    "array_concurrency",
    "max_scheduler_segments",
    "advance_signal_lead_seconds",
    "audit_cadence_updates",
    "durable_checkpoint_cadence_updates",
)


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _absolute_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a non-empty single-line path")
    if not Path(value).is_absolute():
        raise ValueError(f"{name} must be absolute")
    return value


def _slurm_walltime(seconds: int) -> str:
    value = _positive_int(seconds, name="requested walltime seconds")
    days, remainder = divmod(value, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{final_seconds:02d}"


def _semantic_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validated_plan(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _PLAN_KEYS:
        raise ValueError("primary submission plan fields are invalid")
    plan = dict(value)
    if plan["schema_version"] != _SCHEMA or plan["role"] != _ROLE:
        raise ValueError("primary submission plan schema or role is invalid")
    if plan["segment_index"] != 1 or plan["array_task_ids"] != _TASK_IDS:
        raise ValueError("primary submission plan must be fixed segment-1 array 0-2")

    for name in (
        "array_concurrency",
        "gpus_per_task",
        "cpus_per_task",
        "memory_bytes",
        "memory_mib",
        "requested_walltime_seconds",
        "advance_signal_lead_seconds",
        "max_scheduler_segments",
        "audit_cadence_updates",
        "durable_checkpoint_cadence_updates",
    ):
        _positive_int(plan[name], name=name)
    if plan["array_concurrency"] > len(_TASK_IDS):
        raise ValueError("array concurrency exceeds the fixed three-seed wave")
    if plan["gpus_per_task"] != 1:
        raise ValueError("formal primary execution requires exactly one GPU per task")
    if int(plan["memory_bytes"]) % _MIB:
        raise ValueError("memory bytes cannot be represented exactly as Slurm MiB")
    if int(plan["memory_mib"]) * _MIB != plan["memory_bytes"]:
        raise ValueError("memory MiB differs from the exact resource-plan bytes")
    if plan["slurm_walltime"] != _slurm_walltime(int(plan["requested_walltime_seconds"])):
        raise ValueError("Slurm walltime differs from exact resource-plan seconds")

    for name in ("slurm_account", "partition", "gpu_name", "slurm_walltime"):
        item = plan[name]
        if type(item) is not str or not item or "\n" in item or "\r" in item:
            raise ValueError(f"{name} must be a non-empty single-line string")
    if plan["slurm_account"] != "sigroup":
        raise ValueError("formal R3 primary plan must use the sigroup account")
    if not str(plan["partition"]).startswith("gpu-"):
        raise ValueError("formal R3 primary plan requires a GPU partition")

    for name in (
        "resource_plan_sha256",
        "operational_bundle_file_sha256",
        "operational_bundle_semantic_sha256",
        "profile_allocation_intent_file_sha256",
        "profile_runtime_receipt_file_sha256",
        "profile_terminal_manifest_file_sha256",
        "profile_terminal_raw_sacct_sha256",
        "profile_terminal_sha256",
        "submission_plan_sha256",
    ):
        _digest(plan[name], name=name)
    for name in (
        "operational_bundle_path",
        "profile_allocation_intent_path",
        "profile_runtime_receipt_path",
        "profile_terminal_evidence_directory",
    ):
        _absolute_path(plan[name], name=name)

    semantic = plan.pop("submission_plan_sha256")
    if semantic != _semantic_sha256(plan):
        raise ValueError("primary submission plan SHA-256 is invalid")
    plan["submission_plan_sha256"] = semantic
    return plan


def _build_plan(arguments: argparse.Namespace) -> dict[str, object]:
    bundle_sha = _digest(
        arguments.operational_bundle_file_sha256,
        name="operational bundle file SHA-256",
    )
    intent_sha = _digest(
        arguments.profile_allocation_intent_file_sha256,
        name="profile allocation intent file SHA-256",
    )
    runtime_sha = _digest(
        arguments.profile_runtime_receipt_file_sha256,
        name="profile runtime receipt file SHA-256",
    )
    terminal_manifest_sha = _digest(
        arguments.profile_terminal_manifest_file_sha256,
        name="profile terminal manifest file SHA-256",
    )
    terminal_raw_sha = _digest(
        arguments.profile_terminal_raw_sacct_sha256,
        name="profile terminal raw sacct SHA-256",
    )

    bundle = reopen_verified_gate_p_operational_bundle(
        arguments.operational_bundle,
        expected_file_sha256=bundle_sha,
    )
    intent = reopen_profile_allocation_intent(
        arguments.profile_allocation_intent,
        expected_file_sha256=intent_sha,
    )
    runtime = reopen_profile_slurm_runtime_receipt(
        arguments.profile_runtime_receipt,
        expected_file_sha256=runtime_sha,
        operational_bundle=bundle,
        allocation_intent=intent,
    )
    terminal = revalidate_successful_profile_terminal(
        bundle,
        runtime_receipt=runtime,
        evidence_directory=arguments.profile_terminal_evidence_directory,
        expected_manifest_file_sha256=terminal_manifest_sha,
        expected_raw_sacct_sha256=terminal_raw_sha,
    )
    resource_plan = bundle.resource_plan
    memory_bytes = int(bundle.memory_bytes)
    if memory_bytes % _MIB:
        raise ValueError("Gate-P memory bytes cannot be represented exactly by Slurm --mem")
    body: dict[str, object] = {
        "schema_version": _SCHEMA,
        "role": _ROLE,
        "segment_index": 1,
        "array_task_ids": _TASK_IDS,
        "array_concurrency": int(resource_plan["array_concurrency"]),
        "slurm_account": bundle.slurm_account,
        "partition": bundle.partition,
        "gpu_name": bundle.gpu_name,
        "gpus_per_task": bundle.gpus_per_task,
        "cpus_per_task": bundle.cpus_per_task,
        "memory_bytes": memory_bytes,
        "memory_mib": memory_bytes // _MIB,
        "requested_walltime_seconds": (bundle.requested_walltime_seconds_per_segment),
        "slurm_walltime": _slurm_walltime(bundle.requested_walltime_seconds_per_segment),
        "advance_signal_lead_seconds": bundle.advance_signal_lead_seconds,
        "max_scheduler_segments": bundle.max_scheduler_segments,
        "audit_cadence_updates": bundle.audit_cadence_updates,
        "durable_checkpoint_cadence_updates": (bundle.durable_checkpoint_cadence_updates),
        "resource_plan_sha256": bundle.resource_plan_sha256,
        "operational_bundle_path": str(bundle.artifact_path),
        "operational_bundle_file_sha256": bundle.file_sha256,
        "operational_bundle_semantic_sha256": bundle.bundle_semantic_sha256,
        "profile_allocation_intent_path": str(intent.artifact_path),
        "profile_allocation_intent_file_sha256": intent.file_sha256,
        "profile_runtime_receipt_path": str(runtime.artifact_path),
        "profile_runtime_receipt_file_sha256": runtime.file_sha256,
        "profile_terminal_evidence_directory": str(terminal.evidence_directory),
        "profile_terminal_manifest_file_sha256": terminal.manifest_file_sha256,
        "profile_terminal_raw_sacct_sha256": terminal.inspection.raw_sacct_sha256,
        "profile_terminal_sha256": terminal.terminal_sha256,
    }
    return _validated_plan({**body, "submission_plan_sha256": _semantic_sha256(body)})


def _emit(value: Mapping[str, object]) -> None:
    print(
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser(
        "create",
        help="revalidate successful Gate-P and publish an exact segment-1 plan",
    )
    create.add_argument("--operational-bundle", type=Path, required=True)
    create.add_argument("--operational-bundle-file-sha256", required=True)
    create.add_argument("--profile-allocation-intent", type=Path, required=True)
    create.add_argument(
        "--profile-allocation-intent-file-sha256",
        required=True,
    )
    create.add_argument("--profile-runtime-receipt", type=Path, required=True)
    create.add_argument("--profile-runtime-receipt-file-sha256", required=True)
    create.add_argument(
        "--profile-terminal-evidence-directory",
        type=Path,
        required=True,
    )
    create.add_argument("--profile-terminal-manifest-file-sha256", required=True)
    create.add_argument("--profile-terminal-raw-sacct-sha256", required=True)
    create.add_argument("--output", type=Path, required=True)

    inspect = commands.add_parser(
        "inspect",
        help="caller-pin and validate a published plan before sbatch or execution",
    )
    inspect.add_argument("--plan", type=Path, required=True)
    inspect.add_argument("--plan-file-sha256", required=True)
    inspect.add_argument(
        "--format",
        choices=("json", "sbatch-lines"),
        default="json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "create":
        plan = _build_plan(arguments)
        artifact = publish_canonical_artifact(arguments.output, plan)
        _emit(
            {
                "status": "r3_primary_segment_1_submission_plan_published",
                "submission_plan_sha256": plan["submission_plan_sha256"],
                "resource_plan_sha256": plan["resource_plan_sha256"],
                "file_sha256": artifact.file_sha256,
            }
        )
        return 0

    artifact = read_canonical_artifact(
        arguments.plan,
        expected_file_sha256=_digest(
            arguments.plan_file_sha256,
            name="primary submission plan file SHA-256",
        ),
    )
    plan = _validated_plan(artifact.payload)
    if arguments.format == "sbatch-lines":
        for name in _SBATCH_FIELD_ORDER:
            print(f"{name}={plan[name]}", flush=True)
    else:
        _emit(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
