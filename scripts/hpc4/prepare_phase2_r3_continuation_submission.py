#!/usr/bin/env python3
"""Build a fixed R3 continuation wave from sealed predecessor terminals."""

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
    reopen_primary_segment_runtime_closure,
    revalidate_completed_primary_terminal,
    revalidate_continuable_primary_terminal,
)

_SCHEMA = "phase2-recovery-r3-primary-continuation-wave-plan/v1"
_ROLE = "sealed_predecessor_terminal_derived_continuation_wave"
_BASE_SCHEMA = "phase2-recovery-r3-primary-submission-plan/v1"
_TASK_SEED_MAP = {0: 20260801, 1: 20260802, 2: 20260803}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "base_primary_submission_plan_path",
        "base_primary_submission_plan_file_sha256",
        "base_primary_submission_plan_sha256",
        "previous_continuation_plan_path",
        "previous_continuation_plan_file_sha256",
        "operational_bundle_path",
        "operational_bundle_file_sha256",
        "operational_bundle_semantic_sha256",
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
        "array_task_ids",
        "array_concurrency",
        "max_scheduler_segments",
        "advance_signal_lead_seconds",
        "audit_cadence_updates",
        "durable_checkpoint_cadence_updates",
        "dependency_array_job_ids",
        "continuation_array_required",
        "all_tasks_complete",
        "task_routes",
        "continuation_plan_sha256",
    }
)
_ROUTE_KEYS = frozenset(
    {
        "task_id",
        "seed",
        "action",
        "predecessor_segment_index",
        "next_segment_index",
        "history",
    }
)
_ENTRY_KEYS = frozenset(
    {
        "segment_index",
        "runtime_closure_path",
        "runtime_closure_file_sha256",
        "runtime_closure_sha256",
        "terminal_kind",
        "terminal_evidence_directory",
        "terminal_manifest_file_sha256",
        "terminal_raw_sacct_sha256",
        "terminal_sha256",
        "array_job_id",
        "job_id",
        "selected_checkpoint",
    }
)
_SBATCH_FIELDS = (
    "continuation_plan_sha256",
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
    "continuation_array_required",
    "all_tasks_complete",
)


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _semantic_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _absolute_path(value: object, *, name: str) -> str:
    if type(value) is not str or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a single-line path")
    return value


def _base_plan(path: Path, *, expected_file_sha256: str) -> dict[str, object]:
    artifact = read_canonical_artifact(
        path,
        expected_file_sha256=_digest(
            expected_file_sha256,
            name="base primary submission-plan file SHA-256",
        ),
    )
    plan = artifact.payload
    if (
        plan.get("schema_version") != _BASE_SCHEMA
        or plan.get("segment_index") != 1
        or plan.get("array_task_ids") != [0, 1, 2]
    ):
        raise ValueError("base primary submission plan is not the fixed segment-1 plan")
    observed_sha = plan.get("submission_plan_sha256")
    _digest(observed_sha, name="base primary submission plan SHA-256")
    unsigned = dict(plan)
    unsigned.pop("submission_plan_sha256")
    if observed_sha != _semantic_sha256(unsigned):
        raise ValueError("base primary submission plan self-hash is invalid")
    return plan


def _validate_entry(value: object, *, task_id: int) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ENTRY_KEYS:
        raise ValueError(f"task {task_id} continuation history entry fields are invalid")
    entry = dict(value)
    _positive_int(entry["segment_index"], name="history segment_index")
    for name in (
        "runtime_closure_path",
        "terminal_evidence_directory",
    ):
        _absolute_path(entry[name], name=name)
    for name in (
        "runtime_closure_file_sha256",
        "runtime_closure_sha256",
        "terminal_manifest_file_sha256",
        "terminal_raw_sacct_sha256",
        "terminal_sha256",
    ):
        _digest(entry[name], name=name)
    if type(entry["array_job_id"]) is not str or _JOB_ID.fullmatch(entry["array_job_id"]) is None:
        raise ValueError("history array_job_id is invalid")
    if type(entry["job_id"]) is not str or _JOB_ID.fullmatch(entry["job_id"]) is None:
        raise ValueError("history job_id is invalid")
    kind = entry["terminal_kind"]
    checkpoint = entry["selected_checkpoint"]
    if kind == "continuable":
        if not isinstance(checkpoint, Mapping):
            raise ValueError("continuable history requires a selected checkpoint")
    elif kind == "completed":
        if checkpoint is not None:
            raise ValueError("completed history cannot select a continuation checkpoint")
    else:
        raise ValueError("history terminal kind is invalid")
    return entry


def _validate_route(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ROUTE_KEYS:
        raise ValueError("continuation task route fields are invalid")
    route = dict(value)
    task_id = route["task_id"]
    if type(task_id) is not int or task_id not in _TASK_SEED_MAP:
        raise ValueError("continuation route task_id is invalid")
    if route["seed"] != _TASK_SEED_MAP[task_id]:
        raise ValueError("continuation route seed differs from the frozen map")
    history_raw = route["history"]
    if not isinstance(history_raw, list) or not history_raw:
        raise ValueError("continuation route history must be non-empty")
    history = [_validate_entry(entry, task_id=task_id) for entry in history_raw]
    if [entry["segment_index"] for entry in history] != list(range(1, len(history) + 1)):
        raise ValueError("continuation route history is not consecutive from segment 1")
    if any(entry["terminal_kind"] != "continuable" for entry in history[:-1]):
        raise ValueError("only a final history entry may be completed")
    last = history[-1]
    if route["predecessor_segment_index"] != last["segment_index"]:
        raise ValueError("route predecessor segment differs from its history")
    if route["action"] == "continue":
        if (
            last["terminal_kind"] != "continuable"
            or route["next_segment_index"] != last["segment_index"] + 1
        ):
            raise ValueError("continuation route does not follow its sealed terminal")
    elif route["action"] == "complete":
        if last["terminal_kind"] != "completed" or route["next_segment_index"] is not None:
            raise ValueError("completed route must not authorize another segment")
    else:
        raise ValueError("continuation route action is invalid")
    route["history"] = history
    return route


def _validated_plan(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _PLAN_KEYS:
        raise ValueError("continuation wave plan fields are invalid")
    plan = dict(value)
    if plan["schema_version"] != _SCHEMA or plan["role"] != _ROLE:
        raise ValueError("continuation wave plan schema or role is invalid")
    if plan["array_task_ids"] != [0, 1, 2]:
        raise ValueError("continuation wave must retain fixed array task IDs 0-2")
    for name in (
        "gpus_per_task",
        "cpus_per_task",
        "memory_bytes",
        "memory_mib",
        "requested_walltime_seconds",
        "array_concurrency",
        "max_scheduler_segments",
        "advance_signal_lead_seconds",
        "audit_cadence_updates",
        "durable_checkpoint_cadence_updates",
    ):
        _positive_int(plan[name], name=name)
    if plan["gpus_per_task"] != 1 or plan["array_concurrency"] > 3:
        raise ValueError("continuation wave GPU/concurrency contract is invalid")
    for name in (
        "base_primary_submission_plan_path",
        "operational_bundle_path",
    ):
        _absolute_path(plan[name], name=name)
    for name in (
        "base_primary_submission_plan_file_sha256",
        "base_primary_submission_plan_sha256",
        "operational_bundle_file_sha256",
        "operational_bundle_semantic_sha256",
        "resource_plan_sha256",
        "continuation_plan_sha256",
    ):
        _digest(plan[name], name=name)
    previous_path = plan["previous_continuation_plan_path"]
    previous_sha = plan["previous_continuation_plan_file_sha256"]
    if (previous_path is None) != (previous_sha is None):
        raise ValueError("previous continuation plan path/SHA must appear together")
    if previous_path is not None:
        _absolute_path(previous_path, name="previous continuation plan path")
        _digest(previous_sha, name="previous continuation plan file SHA-256")
    routes_raw = plan["task_routes"]
    if not isinstance(routes_raw, list) or len(routes_raw) != 3:
        raise ValueError("continuation wave requires exactly three task routes")
    routes = [_validate_route(route) for route in routes_raw]
    if [route["task_id"] for route in routes] != [0, 1, 2]:
        raise ValueError("continuation task routes must remain ordered 0-2")
    all_complete = all(route["action"] == "complete" for route in routes)
    required = any(route["action"] == "continue" for route in routes)
    if plan["all_tasks_complete"] is not all_complete:
        raise ValueError("all_tasks_complete differs from task routes")
    if plan["continuation_array_required"] is not required:
        raise ValueError("continuation_array_required differs from task routes")
    dependencies = plan["dependency_array_job_ids"]
    if (
        not isinstance(dependencies, list)
        or any(type(item) is not str or _JOB_ID.fullmatch(item) is None for item in dependencies)
        or dependencies != sorted(set(dependencies), key=int)
    ):
        raise ValueError("continuation dependency job IDs are invalid")
    expected_dependencies = sorted(
        {
            str(route["history"][-1]["array_job_id"])
            for route in routes
            if route["action"] == "continue"
        },
        key=int,
    )
    if dependencies != expected_dependencies:
        raise ValueError("continuation dependencies differ from route terminals")
    semantic = plan.pop("continuation_plan_sha256")
    if semantic != _semantic_sha256(plan):
        raise ValueError("continuation wave plan SHA-256 is invalid")
    plan["task_routes"] = routes
    plan["continuation_plan_sha256"] = semantic
    return plan


def reopen_continuation_plan(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> dict[str, object]:
    artifact = read_canonical_artifact(
        path,
        expected_file_sha256=_digest(
            expected_file_sha256,
            name="continuation plan file SHA-256",
        ),
    )
    return _validated_plan(artifact.payload)


def _task_arguments(arguments: argparse.Namespace, task_id: int) -> dict[str, object]:
    return {
        "closure_path": getattr(arguments, f"task_{task_id}_runtime_closure"),
        "closure_sha": getattr(
            arguments,
            f"task_{task_id}_runtime_closure_file_sha256",
        ),
        "terminal_directory": getattr(
            arguments,
            f"task_{task_id}_terminal_evidence_directory",
        ),
        "terminal_manifest_sha": getattr(
            arguments,
            f"task_{task_id}_terminal_manifest_file_sha256",
        ),
        "terminal_raw_sha": getattr(
            arguments,
            f"task_{task_id}_terminal_raw_sacct_sha256",
        ),
    }


def _latest_entry(
    *,
    task_id: int,
    bundle: object,
    arguments: Mapping[str, object],
) -> dict[str, object]:
    closure = reopen_primary_segment_runtime_closure(
        arguments["closure_path"],
        expected_file_sha256=_digest(
            arguments["closure_sha"],
            name=f"task {task_id} runtime closure file SHA-256",
        ),
        operational_bundle=bundle,
    )
    admission = closure.admission_payload
    runtime = closure.runtime_payload
    outcome = closure.outcome_payload
    if (
        admission.get("task_id") != task_id
        or admission.get("seed") != _TASK_SEED_MAP[task_id]
        or runtime.get("task_id") != task_id
        or runtime.get("array_task_id") != task_id
        or outcome.get("task_id") != task_id
    ):
        raise ValueError(f"task {task_id} closure identity differs from the fixed task map")
    segment_index = admission.get("segment_index")
    _positive_int(segment_index, name=f"task {task_id} predecessor segment")
    if (
        runtime.get("segment_index") != segment_index
        or outcome.get("segment_index") != segment_index
    ):
        raise ValueError(f"task {task_id} closure segment identities differ")
    terminal_kwargs = {
        "runtime_closure": closure,
        "evidence_directory": arguments["terminal_directory"],
        "expected_manifest_file_sha256": _digest(
            arguments["terminal_manifest_sha"],
            name=f"task {task_id} terminal manifest file SHA-256",
        ),
        "expected_raw_sacct_sha256": _digest(
            arguments["terminal_raw_sha"],
            name=f"task {task_id} terminal raw sacct SHA-256",
        ),
    }
    if closure.status == "continuation_required_after_safe_checkpoint":
        terminal = revalidate_continuable_primary_terminal(
            bundle,
            **terminal_kwargs,
        )
        kind = "continuable"
        checkpoint = outcome["continuation_checkpoint"]
    elif closure.status == "compute_complete_pending_external_scheduler_terminal":
        terminal = revalidate_completed_primary_terminal(
            bundle,
            **terminal_kwargs,
        )
        kind = "completed"
        checkpoint = None
    else:
        raise ValueError("primary closure is neither continuable nor completed")
    return _validate_entry(
        {
            "segment_index": segment_index,
            "runtime_closure_path": str(closure.artifact_path),
            "runtime_closure_file_sha256": closure.file_sha256,
            "runtime_closure_sha256": closure.closure_sha256,
            "terminal_kind": kind,
            "terminal_evidence_directory": str(terminal.evidence_directory),
            "terminal_manifest_file_sha256": terminal.manifest_file_sha256,
            "terminal_raw_sacct_sha256": terminal.inspection.raw_sacct_sha256,
            "terminal_sha256": terminal.terminal_sha256,
            "array_job_id": str(runtime["array_job_id"]),
            "job_id": str(runtime["job_id"]),
            "selected_checkpoint": checkpoint,
        },
        task_id=task_id,
    )


def _build_plan(arguments: argparse.Namespace) -> dict[str, object]:
    base_file_sha = _digest(
        arguments.primary_submission_plan_file_sha256,
        name="base primary submission plan file SHA-256",
    )
    base = _base_plan(
        arguments.primary_submission_plan,
        expected_file_sha256=base_file_sha,
    )
    bundle = reopen_verified_gate_p_operational_bundle(
        base["operational_bundle_path"],
        expected_file_sha256=base["operational_bundle_file_sha256"],
    )
    if (
        bundle.resource_plan_sha256 != base["resource_plan_sha256"]
        or bundle.bundle_semantic_sha256 != base["operational_bundle_semantic_sha256"]
    ):
        raise ValueError("base submission plan differs from its Gate-P bundle")

    previous_path = arguments.previous_continuation_plan
    previous_sha = arguments.previous_continuation_plan_file_sha256
    if (previous_path is None) != (previous_sha is None):
        raise ValueError("previous continuation plan path/SHA must be supplied together")
    previous = (
        None
        if previous_path is None
        else reopen_continuation_plan(
            previous_path,
            expected_file_sha256=previous_sha,
        )
    )
    if previous is not None and (
        previous["base_primary_submission_plan_file_sha256"] != base_file_sha
        or previous["base_primary_submission_plan_sha256"] != base["submission_plan_sha256"]
        or previous["resource_plan_sha256"] != bundle.resource_plan_sha256
    ):
        raise ValueError("previous continuation plan belongs to another primary design")

    routes: list[dict[str, object]] = []
    for task_id in range(3):
        latest = _latest_entry(
            task_id=task_id,
            bundle=bundle,
            arguments=_task_arguments(arguments, task_id),
        )
        prior_route = None if previous is None else previous["task_routes"][task_id]
        if prior_route is None:
            if latest["segment_index"] != 1:
                raise ValueError("initial continuation plan must consume segment-1 terminals")
            history = [latest]
        elif prior_route["action"] == "complete":
            if latest != prior_route["history"][-1]:
                raise ValueError("completed task terminal cannot be replaced or advanced")
            history = list(prior_route["history"])
        else:
            if latest["segment_index"] != prior_route["next_segment_index"]:
                raise ValueError("latest terminal is not the next consecutive segment")
            history = [*prior_route["history"], latest]
        if latest["terminal_kind"] == "continuable":
            if latest["segment_index"] >= bundle.max_scheduler_segments:
                raise ValueError("continuable terminal exhausts the Gate-P segment limit")
            action = "continue"
            next_segment = int(latest["segment_index"]) + 1
        else:
            action = "complete"
            next_segment = None
        routes.append(
            _validate_route(
                {
                    "task_id": task_id,
                    "seed": _TASK_SEED_MAP[task_id],
                    "action": action,
                    "predecessor_segment_index": latest["segment_index"],
                    "next_segment_index": next_segment,
                    "history": history,
                }
            )
        )
    dependencies = sorted(
        {
            str(route["history"][-1]["array_job_id"])
            for route in routes
            if route["action"] == "continue"
        },
        key=int,
    )
    body: dict[str, object] = {
        "schema_version": _SCHEMA,
        "role": _ROLE,
        "base_primary_submission_plan_path": str(
            Path(arguments.primary_submission_plan).resolve(strict=True)
        ),
        "base_primary_submission_plan_file_sha256": base_file_sha,
        "base_primary_submission_plan_sha256": base["submission_plan_sha256"],
        "previous_continuation_plan_path": (
            None if previous_path is None else str(Path(previous_path).resolve(strict=True))
        ),
        "previous_continuation_plan_file_sha256": previous_sha,
        "operational_bundle_path": str(bundle.artifact_path),
        "operational_bundle_file_sha256": bundle.file_sha256,
        "operational_bundle_semantic_sha256": bundle.bundle_semantic_sha256,
        "resource_plan_sha256": bundle.resource_plan_sha256,
        "slurm_account": base["slurm_account"],
        "partition": base["partition"],
        "gpu_name": base["gpu_name"],
        "gpus_per_task": base["gpus_per_task"],
        "cpus_per_task": base["cpus_per_task"],
        "memory_bytes": base["memory_bytes"],
        "memory_mib": base["memory_mib"],
        "requested_walltime_seconds": base["requested_walltime_seconds"],
        "slurm_walltime": base["slurm_walltime"],
        "array_task_ids": [0, 1, 2],
        "array_concurrency": base["array_concurrency"],
        "max_scheduler_segments": base["max_scheduler_segments"],
        "advance_signal_lead_seconds": base["advance_signal_lead_seconds"],
        "audit_cadence_updates": base["audit_cadence_updates"],
        "durable_checkpoint_cadence_updates": base["durable_checkpoint_cadence_updates"],
        "dependency_array_job_ids": dependencies,
        "continuation_array_required": any(route["action"] == "continue" for route in routes),
        "all_tasks_complete": all(route["action"] == "complete" for route in routes),
        "task_routes": routes,
    }
    return _validated_plan({**body, "continuation_plan_sha256": _semantic_sha256(body)})


def _add_task_arguments(parser: argparse.ArgumentParser, task_id: int) -> None:
    prefix = f"--task-{task_id}"
    destination = f"task_{task_id}"
    parser.add_argument(
        f"{prefix}-runtime-closure",
        dest=f"{destination}_runtime_closure",
        type=Path,
        required=True,
    )
    parser.add_argument(
        f"{prefix}-runtime-closure-file-sha256",
        dest=f"{destination}_runtime_closure_file_sha256",
        required=True,
    )
    parser.add_argument(
        f"{prefix}-terminal-evidence-directory",
        dest=f"{destination}_terminal_evidence_directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        f"{prefix}-terminal-manifest-file-sha256",
        dest=f"{destination}_terminal_manifest_file_sha256",
        required=True,
    )
    parser.add_argument(
        f"{prefix}-terminal-raw-sacct-sha256",
        dest=f"{destination}_terminal_raw_sacct_sha256",
        required=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--primary-submission-plan", type=Path, required=True)
    create.add_argument("--primary-submission-plan-file-sha256", required=True)
    create.add_argument("--previous-continuation-plan", type=Path)
    create.add_argument("--previous-continuation-plan-file-sha256")
    for task_id in range(3):
        _add_task_arguments(create, task_id)
    create.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--plan", type=Path, required=True)
    inspect.add_argument("--plan-file-sha256", required=True)
    inspect.add_argument(
        "--format",
        choices=("json", "sbatch-lines", "task-lines"),
        default="json",
    )
    inspect.add_argument("--task-id", type=int, choices=(0, 1, 2))
    return parser


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


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "create":
        plan = _build_plan(arguments)
        artifact = publish_canonical_artifact(arguments.output, plan)
        _emit(
            {
                "status": (
                    "r3_primary_complete_no_continuation"
                    if plan["all_tasks_complete"]
                    else "r3_primary_continuation_wave_plan_published"
                ),
                "continuation_array_required": plan["continuation_array_required"],
                "continuation_plan_sha256": plan["continuation_plan_sha256"],
                "file_sha256": artifact.file_sha256,
            }
        )
        return 0
    plan = reopen_continuation_plan(
        arguments.plan,
        expected_file_sha256=arguments.plan_file_sha256,
    )
    if arguments.format == "sbatch-lines":
        for name in _SBATCH_FIELDS:
            print(f"{name}={str(plan[name]).lower() if type(plan[name]) is bool else plan[name]}")
        dependency_ids = ":".join(plan["dependency_array_job_ids"])
        print(f"dependency_afterok={dependency_ids}")
        print(f"dependency_afternotok={dependency_ids}")
    elif arguments.format == "task-lines":
        if arguments.task_id is None:
            raise ValueError("--task-id is required with --format task-lines")
        route = plan["task_routes"][arguments.task_id]
        print(f"task_id={route['task_id']}")
        print(f"seed={route['seed']}")
        print(f"action={route['action']}")
        print(f"predecessor_segment_index={route['predecessor_segment_index']}")
        print(
            "next_segment_index="
            f"{'' if route['next_segment_index'] is None else route['next_segment_index']}"
        )
    else:
        if arguments.task_id is not None:
            raise ValueError("--task-id is only valid with --format task-lines")
        _emit(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
