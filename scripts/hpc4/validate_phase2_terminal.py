#!/usr/bin/env python3
"""Fail-closed validation of one formal Phase-2 seed's complete attempt history."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

HEX = re.compile(r"[0-9a-f]{64}")
COMPUTE_MARKER_SCHEMA = "prorm-phase2-confirmatory-run-status/v1"
SCHEDULER_MARKER_SCHEMA = "prorm-phase2-scheduler-terminal-status/v1"
LEDGER_SCHEMAS = {"phase2-seed-attempt-ledger/v3"}
RETRY_POLICY = "single_predeclared_attempt_no_retry"


def die(message: str) -> None:
    raise SystemExit(message)


def _strict_object(
    pairs: list[tuple[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{context} repeats JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path, *, canonical: bool = False) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        die(f"JSON evidence is missing or unsafe: {path}")
    raw = path.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {item}")
        ),
    )
    if not isinstance(value, dict):
        die(f"JSON evidence must contain one object: {path}")
    if canonical:
        expected = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
        if raw != expected:
            die(f"JSON evidence is not canonical: {path}")
    return value, raw


def validate_ledger(
    value: Any,
    *,
    terminal_status: str,
    expected_length: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "retry_policy",
        "replacement_seed_allowed",
        "attempts",
    }:
        die("attempt ledger fields are invalid")
    attempts = value.get("attempts")
    if (
        value.get("schema_version") not in LEDGER_SCHEMAS
        or value.get("retry_policy") != RETRY_POLICY
        or value.get("replacement_seed_allowed") is not False
        or not isinstance(attempts, list)
        or len(attempts) != expected_length
    ):
        die("attempt ledger contract is invalid")
    jobs: set[str] = set()
    for index, attempt in enumerate(attempts, 1):
        expected_keys = {
            "attempt_index",
            "slurm_job_id",
            "status",
            "final_outcome_reveal_started",
            "log_sha256",
        }
        if value.get("schema_version") == "phase2-seed-attempt-ledger/v3":
            expected_keys |= {
                "cluster_name",
                "array_job_id",
                "array_task_id",
            }
        if not isinstance(attempt, dict) or set(attempt) != expected_keys:
            die("attempt ledger entry fields are invalid")
        expected_status = terminal_status
        job = attempt.get("slurm_job_id")
        if (
            attempt.get("attempt_index") != index
            or not isinstance(job, str)
            or not job
            or job in jobs
            or attempt.get("status") != expected_status
            or not isinstance(attempt.get("final_outcome_reveal_started"), bool)
            or not isinstance(attempt.get("log_sha256"), str)
            or HEX.fullmatch(attempt["log_sha256"]) is None
        ):
            die("attempt ledger sequence is invalid")
        if value.get("schema_version") == "phase2-seed-attempt-ledger/v3" and (
            not isinstance(attempt.get("cluster_name"), str)
            or not attempt["cluster_name"]
            or not isinstance(attempt.get("array_job_id"), str)
            or re.fullmatch(r"[1-9][0-9]*", attempt["array_job_id"]) is None
            or not isinstance(attempt.get("array_task_id"), int)
            or attempt["array_task_id"] < 0
        ):
            die("attempt ledger scheduler identity is invalid")
        if (
            expected_status == "success_result"
            and attempt["final_outcome_reveal_started"] is not True
        ):
            die("successful attempt did not cross the outcome boundary")
        jobs.add(job)
    return attempts


def parse_key_value(path: Path, *, name: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        die(f"{name} is missing or unsafe: {path}")
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            die(f"{name} contains a malformed line")
        key, value = line.split("=", 1)
        if not key or key in fields:
            die(f"{name} contains a duplicate or empty key")
        fields[key] = value
    return fields


def parse_compute_marker(path: Path) -> dict[str, str]:
    fields = parse_key_value(path, name="compute marker")
    required = {
        "schema_version",
        "status",
        "workload_exit_code",
        "final_exit_code",
        "array_job_id",
        "array_task_id",
        "cluster_name",
        "slurm_job_id",
        "slurm_restart_count",
        "attempt_index",
        "seed",
        "phase2_design_sha256",
        "base_config_hash",
        "git_commit",
        "accepted_freeze_aggregate_sha256",
        "registry_submission_sha256",
        "registry_execution_sha256",
        "attempt_claim_sha256",
        "outcome_reveal_marker_sha256",
        "attempt_ledger_sha256",
        "final_outcome_reveal_started",
        "created_at_utc",
    }
    optional = {"terminal_manifest_sha256", "result_sha256"}
    if not required <= set(fields) or set(fields) - required - optional:
        die("compute marker fields differ from the locked schema")
    return fields


def validate_compute_registry(
    *,
    attempt_dir: Path,
    claim: dict[str, str],
) -> None:
    design_root = attempt_dir.parent.parent
    registry = design_root / "campaign-registry"
    submissions = registry / "submissions"
    executions = registry / "executions"
    recoveries = registry / "recoveries"
    scheduler_terminals = registry / "scheduler-terminals"
    for directory in (
        registry,
        submissions,
        executions,
        recoveries,
        scheduler_terminals,
    ):
        if directory.is_symlink() or not directory.is_dir():
            die("formal attempt lacks its canonical campaign registry")
    submission_matches: list[tuple[Path, dict[str, Any]]] = []
    for path in submissions.glob("array-*.json"):
        value, raw = load_json(path, canonical=True)
        if hashlib.sha256(raw).hexdigest() == claim["registry_submission_sha256"]:
            submission_matches.append((path, value))
    if len(submission_matches) != 1:
        die("attempt claim lacks one unique immutable submission registry record")
    _, submission = submission_matches[0]
    entries = submission.get("entries")
    selected = (
        [
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("seed") == int(claim["seed"])
            and item.get("attempt_index") == int(claim["attempt_index"])
        ]
        if isinstance(entries, list)
        else []
    )
    if (
        submission.get("schema_version") != "prorm-phase2-campaign-submission/v1"
        or submission.get("status") != "committed_while_slurm_held"
        or submission.get("phase2_design_sha256") != claim["phase2_design_sha256"]
        or submission.get("base_config_hash") != claim["base_config_hash"]
        or submission.get("git_commit") != claim["git_commit"]
        or submission.get("accepted_freeze_aggregate_sha256")
        != claim["accepted_freeze_aggregate_sha256"]
        or len(selected) != 1
        or selected[0].get("array_job_id") != claim["array_job_id"]
        or str(selected[0].get("array_task_id")) != claim["array_task_id"]
    ):
        die("attempt claim disagrees with its held-array submission registry")
    execution_path = executions / f"seed-{claim['seed']}-attempt-{claim['attempt_index']}.json"
    execution, execution_raw = load_json(execution_path, canonical=True)
    if (
        hashlib.sha256(execution_raw).hexdigest() != claim["registry_execution_sha256"]
        or execution.get("schema_version") != "prorm-phase2-campaign-execution/v1"
        or execution.get("status") != "compute_started_no_requeue"
        or execution.get("seed") != int(claim["seed"])
        or execution.get("attempt_index") != int(claim["attempt_index"])
        or execution.get("cluster_name") != claim["cluster_name"]
        or execution.get("array_job_id") != claim["array_job_id"]
        or str(execution.get("array_task_id")) != claim["array_task_id"]
        or execution.get("slurm_job_id") != claim["slurm_job_id"]
        or execution.get("slurm_restart_count") != 0
        or execution.get("replacement_seed_allowed") is not False
        or execution.get("submission", {}).get("sha256") != claim["registry_submission_sha256"]
    ):
        die("attempt claim disagrees with its execution registry record")
    scheduler_record = (
        scheduler_terminals / f"seed-{claim['seed']}-attempt-{claim['attempt_index']}.json"
    )
    if scheduler_record.exists() or scheduler_record.is_symlink():
        die("compute-owned terminal attempt also has scheduler-terminal ownership")
    recovery_path = recoveries / f"seed-{claim['seed']}-attempt-{claim['attempt_index']}.json"
    if recovery_path.exists() or recovery_path.is_symlink():
        die("formal no-retry attempt unexpectedly has recovery authorization")


def validate_scheduler_registry(
    *,
    attempt_dir: Path,
    claim: dict[str, str],
    scheduler_registry_sha: str,
    scheduler_raw_sha: str,
    scheduler_state: str,
    exit_code: str,
) -> None:
    design_root = attempt_dir.parent.parent
    registry = design_root / "campaign-registry"
    submissions = registry / "submissions"
    executions = registry / "executions"
    recoveries = registry / "recoveries"
    scheduler_terminals = registry / "scheduler-terminals"
    for directory in (
        registry,
        submissions,
        executions,
        recoveries,
        scheduler_terminals,
    ):
        if directory.is_symlink() or not directory.is_dir():
            die("scheduler-reconciled attempt lacks its canonical campaign registry")
    record_path = (
        scheduler_terminals / f"seed-{claim['seed']}-attempt-{claim['attempt_index']}.json"
    )
    record, record_raw = load_json(record_path, canonical=True)
    expected_record_fields = {
        "schema_version",
        "status",
        "seed",
        "attempt_index",
        "phase2_design_sha256",
        "base_config_hash",
        "phase2_runtime_contract_sha256",
        "git_commit",
        "accepted_freeze_aggregate_sha256",
        "cluster_name",
        "array_job_id",
        "array_task_id",
        "slurm_job_id",
        "slurm_restart_count",
        "scheduler_state",
        "exit_code",
        "scheduler_raw_evidence_sha256",
        "registry_submission_sha256",
        "registry_execution_sha256",
        "retry_authorized",
        "replacement_seed_allowed",
    }
    expected_execution_sha: str | None = (
        None if claim["registry_execution_sha256"] == "none" else claim["registry_execution_sha256"]
    )
    if (
        (
            claim.get("schema_version") == "prorm-phase2-formal-scheduler-attempt-claim/v1"
            and claim.get("registry_scheduler_terminal_sha256") != scheduler_registry_sha
        )
        or set(record) != expected_record_fields
        or hashlib.sha256(record_raw).hexdigest() != scheduler_registry_sha
        or record.get("schema_version") != "prorm-phase2-campaign-scheduler-terminal/v1"
        or record.get("status") != "terminal_non_success_no_retry"
        or record.get("seed") != int(claim["seed"])
        or record.get("attempt_index") != int(claim["attempt_index"])
        or record.get("phase2_design_sha256") != claim["phase2_design_sha256"]
        or record.get("base_config_hash") != claim["base_config_hash"]
        or not isinstance(record.get("phase2_runtime_contract_sha256"), str)
        or HEX.fullmatch(record["phase2_runtime_contract_sha256"]) is None
        or record.get("git_commit") != claim["git_commit"]
        or record.get("accepted_freeze_aggregate_sha256")
        != claim["accepted_freeze_aggregate_sha256"]
        or record.get("cluster_name") != claim["cluster_name"]
        or record.get("array_job_id") != claim["array_job_id"]
        or str(record.get("array_task_id")) != claim["array_task_id"]
        or record.get("slurm_job_id") != claim["slurm_job_id"]
        or record.get("slurm_restart_count") != 0
        or record.get("scheduler_state") != scheduler_state
        or record.get("exit_code") != exit_code
        or record.get("scheduler_raw_evidence_sha256") != scheduler_raw_sha
        or record.get("registry_submission_sha256") != claim["registry_submission_sha256"]
        or record.get("registry_execution_sha256") != expected_execution_sha
        or record.get("retry_authorized") is not False
        or record.get("replacement_seed_allowed") is not False
    ):
        die("scheduler-terminal registry record is malformed or identity-mismatched")
    submission_matches: list[tuple[Path, dict[str, Any]]] = []
    for path in submissions.glob("array-*.json"):
        value, raw = load_json(path, canonical=True)
        if hashlib.sha256(raw).hexdigest() == claim["registry_submission_sha256"]:
            submission_matches.append((path, value))
    if len(submission_matches) != 1:
        die("scheduler attempt lacks one unique held-array submission registry record")
    submission_path, submission = submission_matches[0]
    entries = submission.get("entries")
    selected = (
        [
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("seed") == int(claim["seed"])
            and item.get("attempt_index") == int(claim["attempt_index"])
            and item.get("array_job_id") == claim["array_job_id"]
            and str(item.get("array_task_id")) == claim["array_task_id"]
        ]
        if isinstance(entries, list)
        else []
    )
    if (
        submission_path.name != f"array-{claim['array_job_id']}.json"
        or submission.get("schema_version") != "prorm-phase2-campaign-submission/v1"
        or submission.get("status") != "committed_while_slurm_held"
        or submission.get("phase2_design_sha256") != claim["phase2_design_sha256"]
        or submission.get("base_config_hash") != claim["base_config_hash"]
        or submission.get("git_commit") != claim["git_commit"]
        or submission.get("accepted_freeze_aggregate_sha256")
        != claim["accepted_freeze_aggregate_sha256"]
        or submission.get("submitted_cluster") != claim["cluster_name"]
        or submission.get("replacement_seed_allowed") is not False
        or len(selected) != 1
    ):
        die("scheduler attempt disagrees with its held-array submission registry")
    execution_path = executions / f"seed-{claim['seed']}-attempt-{claim['attempt_index']}.json"
    if expected_execution_sha is None:
        if execution_path.exists() or execution_path.is_symlink():
            die("scheduler-only attempt unexpectedly has a compute execution record")
    else:
        execution, execution_raw = load_json(execution_path, canonical=True)
        if (
            hashlib.sha256(execution_raw).hexdigest() != expected_execution_sha
            or execution.get("schema_version") != "prorm-phase2-campaign-execution/v1"
            or execution.get("status") != "compute_started_no_requeue"
            or execution.get("seed") != int(claim["seed"])
            or execution.get("attempt_index") != int(claim["attempt_index"])
            or execution.get("phase2_design_sha256") != claim["phase2_design_sha256"]
            or execution.get("base_config_hash") != claim["base_config_hash"]
            or execution.get("git_commit") != claim["git_commit"]
            or execution.get("accepted_freeze_aggregate_sha256")
            != claim["accepted_freeze_aggregate_sha256"]
            or execution.get("cluster_name") != claim["cluster_name"]
            or execution.get("array_job_id") != claim["array_job_id"]
            or str(execution.get("array_task_id")) != claim["array_task_id"]
            or execution.get("slurm_job_id") != claim["slurm_job_id"]
            or execution.get("slurm_restart_count") != 0
            or execution.get("replacement_seed_allowed") is not False
            or execution.get("submission", {}).get("sha256") != claim["registry_submission_sha256"]
        ):
            die("scheduler attempt disagrees with its compute execution registry")
    recovery_path = recoveries / f"seed-{claim['seed']}-attempt-{claim['attempt_index']}.json"
    if recovery_path.exists() or recovery_path.is_symlink():
        die("scheduler-terminal failure can never carry retry authorization")


def environment_available(
    value: dict[str, Any],
    *,
    schema: str,
) -> dict[str, Any] | None:
    if schema in {
        "phase2-seed-terminal-success/v2",
        "phase2-seed-terminal-failure/v1",
    }:
        environment = value.get("environment_identity")
    else:
        availability = value.get("evidence_availability")
        if (
            not isinstance(availability, dict)
            or availability.get("schema_version") != "phase2-seed-failure-evidence-availability/v1"
        ):
            die("failure v2 evidence availability is invalid")
        slot = availability.get("environment_identity")
        if not isinstance(slot, dict) or slot.get("status") not in {
            "available",
            "unavailable",
        }:
            die("failure v2 environment availability is invalid")
        if slot["status"] == "unavailable":
            if set(slot) != {"status", "reason"} or slot["reason"] not in {
                "not_produced_before_failure",
                "not_published_before_hard_termination",
                "not_recoverable_from_scheduler_evidence",
            }:
                die("failure v2 unavailable environment reason is invalid")
            return None
        if set(slot) != {"status", "value"}:
            die("failure v2 available environment slot is invalid")
        environment = slot["value"]
    if not isinstance(environment, dict):
        die("available terminal environment identity is invalid")
    return environment


def validate_terminal(argv: list[str]) -> None:
    if len(argv) != 7:
        die(
            "usage: validate_phase2_terminal.py "
            "<terminal> <seed> <design-sha> <base-hash> <git-commit> "
            "<image-sha> <inventory-sha>"
        )
    (
        terminal_raw,
        seed_raw,
        design_sha,
        base_hash,
        git_commit,
        image_sha,
        inventory_sha,
    ) = argv
    expected_seed = int(seed_raw)
    terminal = Path(terminal_raw)
    value, terminal_raw_bytes = load_json(terminal)
    terminal = terminal.resolve()
    schema = value.get("schema_version")
    if schema == "phase2-seed-terminal-success/v2":
        expected_status = "success_result"
        terminal_attempt_status = "success_result"
        capture = "compute_exit_trap"
    elif schema == "phase2-seed-terminal-failure/v2":
        expected_status = "failed"
        terminal_attempt_status = "terminal_failure"
        capture = value.get("capture_method")
        if capture not in {
            "compute_exit_trap",
            "scheduler_terminal_reconciliation",
        }:
            die("failure v2 capture method is invalid")
    else:
        die("terminal input must be a supported success or failure manifest")
    if (
        value.get("terminal_status") != expected_status
        or value.get("terminal") is not True
        or value.get("supports_formal_claim") is not False
        or value.get("seed_replacement_allowed") is not False
        or value.get("seed") != expected_seed
        or value.get("source_config_hash") != base_hash
        or value.get("phase2_design_sha256") != design_sha
    ):
        die("terminal manifest identity is invalid")
    environment = environment_available(value, schema=schema)
    if environment is not None and (
        environment.get("formal") is not True
        or environment.get("git_commit") != git_commit
        or environment.get("image_sha256") != image_sha
        or environment.get("hf_inventory_sha256") != inventory_sha
        or environment.get("account") != "sigroup"
        or environment.get("partition") != "gpu-l20"
        or not isinstance(environment.get("gpu_models"), list)
        or len(environment["gpu_models"]) != 1
        or "l20" not in str(environment["gpu_models"][0]).lower()
    ):
        die("available terminal environment identity is invalid")

    ledger_path = terminal.parent / "phase2-attempt-ledger.json"
    ledger, ledger_raw = load_json(ledger_path, canonical=True)
    if ledger != value.get("attempt_ledger"):
        die("terminal manifest and canonical attempt ledger sidecar disagree")
    attempts = validate_ledger(
        ledger,
        terminal_status=terminal_attempt_status,
        expected_length=len(ledger.get("attempts", [])),
    )
    if len(attempts) != 1:
        die("formal Phase-2 retries are disabled; terminal ledger must be attempt-1")
    ledger_sha = hashlib.sha256(ledger_raw).hexdigest()
    last = attempts[-1]

    scheduler_marker: dict[str, Any] | None = None
    compute_fields: dict[str, str] | None = None
    if capture == "scheduler_terminal_reconciliation":
        marker = terminal.parent / "SCHEDULER_FAILED"
        scheduler_marker, _ = load_json(marker, canonical=True)
        attestation_path = terminal.parent / "scheduler-terminal-attestation.json"
        attestation, attestation_raw = load_json(attestation_path, canonical=True)
        scheduler_claim = parse_key_value(
            terminal.parent.parent / "CLAIM",
            name="scheduler attempt claim",
        )
        scheduler_registry_path = (
            terminal.parent.parent.parent.parent
            / "campaign-registry"
            / "scheduler-terminals"
            / f"seed-{expected_seed}-attempt-{len(attempts)}.json"
        )
        scheduler_registry_value, scheduler_registry_raw = load_json(
            scheduler_registry_path,
            canonical=True,
        )
        scheduler_registry_sha = hashlib.sha256(scheduler_registry_raw).hexdigest()
        scheduler_outcome_marker = terminal.parent.parent / "OUTCOME_REVEAL_STARTED"
        if scheduler_outcome_marker.is_symlink():
            die("scheduler outcome marker is unsafe")
        scheduler_outcome_sha = (
            hashlib.sha256(scheduler_outcome_marker.read_bytes()).hexdigest()
            if scheduler_outcome_marker.is_file()
            else "none"
        )
        raw_evidence = terminal.parent / "scheduler-terminal-attestation.raw"
        if raw_evidence.is_symlink() or not raw_evidence.is_file():
            die("scheduler raw evidence is missing or unsafe")
        raw_sha = hashlib.sha256(raw_evidence.read_bytes()).hexdigest()
        terminal_sha = hashlib.sha256(terminal_raw_bytes).hexdigest()
        attestation_sha = hashlib.sha256(attestation_raw).hexdigest()
        if (
            set(scheduler_marker)
            != {
                "schema_version",
                "status",
                "terminal",
                "supports_formal_claim",
                "seed",
                "attempt_index",
                "cluster_name",
                "slurm_job_id",
                "array_job_id",
                "array_task_id",
                "phase2_design_sha256",
                "base_config_hash",
                "phase2_runtime_contract_sha256",
                "registry_submission_sha256",
                "registry_execution_sha256",
                "registry_scheduler_terminal_sha256",
                "attempt_claim_sha256",
                "outcome_reveal_marker_sha256",
                "final_outcome_reveal_started",
                "scheduler_terminal_attestation_sha256",
                "scheduler_raw_evidence_sha256",
                "attempt_ledger_sha256",
                "terminal_manifest_sha256",
            }
            or scheduler_marker.get("schema_version") != SCHEDULER_MARKER_SCHEMA
            or scheduler_marker.get("status") != "SCHEDULER_FAILED"
            or scheduler_marker.get("terminal") is not True
            or scheduler_marker.get("supports_formal_claim") is not False
            or scheduler_marker.get("seed") != expected_seed
            or scheduler_marker.get("attempt_index") != len(attempts)
            or scheduler_marker.get("cluster_name") != attestation.get("cluster_name")
            or scheduler_marker.get("slurm_job_id") != last["slurm_job_id"]
            or scheduler_marker.get("array_job_id") != attestation.get("array_job_id")
            or scheduler_marker.get("array_task_id") != attestation.get("array_task_id")
            or scheduler_marker.get("phase2_design_sha256") != design_sha
            or scheduler_marker.get("base_config_hash") != base_hash
            or scheduler_marker.get("phase2_runtime_contract_sha256")
            != value.get("phase2_runtime_contract_sha256")
            or scheduler_marker.get("registry_submission_sha256")
            != attestation.get("registry_submission_sha256")
            or scheduler_marker.get("registry_execution_sha256")
            != attestation.get("registry_execution_sha256")
            or scheduler_marker.get("registry_scheduler_terminal_sha256") != scheduler_registry_sha
            or scheduler_marker.get("attempt_claim_sha256")
            != hashlib.sha256((terminal.parent.parent / "CLAIM").read_bytes()).hexdigest()
            or scheduler_marker.get("final_outcome_reveal_started")
            is not last["final_outcome_reveal_started"]
            or scheduler_marker.get("outcome_reveal_marker_sha256") != scheduler_outcome_sha
            or scheduler_marker.get("attempt_ledger_sha256") != ledger_sha
            or scheduler_marker.get("terminal_manifest_sha256") != terminal_sha
            or scheduler_marker.get("scheduler_terminal_attestation_sha256") != attestation_sha
            or scheduler_marker.get("scheduler_raw_evidence_sha256") != raw_sha
            or set(attestation)
            != {
                "schema_version",
                "terminal",
                "supports_formal_claim",
                "seed",
                "attempt_index",
                "cluster_name",
                "slurm_job_id",
                "array_job_id",
                "array_task_id",
                "scheduler_state",
                "exit_code",
                "scheduler_evidence_sha256",
                "source_config_hash",
                "phase2_design_sha256",
                "phase2_runtime_contract_sha256",
                "git_commit",
                "accepted_freeze_aggregate_sha256",
                "registry_submission_sha256",
                "registry_execution_sha256",
                "final_outcome_reveal_started",
            }
            or attestation.get("schema_version") != "phase2-scheduler-terminal-attestation/v1"
            or attestation.get("terminal") is not True
            or attestation.get("supports_formal_claim") is not False
            or attestation.get("seed") != expected_seed
            or attestation.get("attempt_index") != len(attempts)
            or attestation.get("slurm_job_id") != last["slurm_job_id"]
            or not isinstance(attestation.get("cluster_name"), str)
            or not attestation["cluster_name"]
            or not isinstance(attestation.get("array_job_id"), str)
            or re.fullmatch(r"[1-9][0-9]*", attestation["array_job_id"]) is None
            or attestation.get("array_task_id") != expected_seed - 20260901
            or attestation.get("source_config_hash") != base_hash
            or attestation.get("phase2_design_sha256") != design_sha
            or attestation.get("phase2_runtime_contract_sha256")
            != value.get("phase2_runtime_contract_sha256")
            or attestation.get("git_commit") != git_commit
            or attestation.get("registry_submission_sha256")
            != scheduler_marker.get("registry_submission_sha256")
            or attestation.get("registry_execution_sha256")
            != scheduler_marker.get("registry_execution_sha256")
            or scheduler_registry_value.get("schema_version")
            != "prorm-phase2-campaign-scheduler-terminal/v1"
            or scheduler_registry_value.get("status") != "terminal_non_success_no_retry"
            or scheduler_registry_value.get("seed") != expected_seed
            or scheduler_registry_value.get("attempt_index") != len(attempts)
            or scheduler_registry_value.get("retry_authorized") is not False
            or scheduler_registry_value.get("replacement_seed_allowed") is not False
            or scheduler_registry_value.get("scheduler_raw_evidence_sha256") != raw_sha
            or attestation.get("scheduler_evidence_sha256") != raw_sha
            or value.get("evidence_sha256_by_role", {}).get("scheduler_terminal_attestation")
            != raw_sha
        ):
            die("scheduler-terminal marker or attestation binding is invalid")
        freeze_sha = attestation.get("accepted_freeze_aggregate_sha256")
        if not isinstance(freeze_sha, str) or HEX.fullmatch(freeze_sha) is None:
            die("scheduler attestation freeze identity is invalid")
        validate_scheduler_registry(
            attempt_dir=terminal.parent.parent,
            claim=scheduler_claim,
            scheduler_registry_sha=scheduler_registry_sha,
            scheduler_raw_sha=raw_sha,
            scheduler_state=str(attestation.get("scheduler_state")),
            exit_code=str(attestation.get("exit_code")),
        )
    else:
        marker_name = "SUCCESS" if expected_status == "success_result" else "FAILED"
        marker = terminal.parent / marker_name
        compute_fields = parse_compute_marker(marker)
        try:
            final_exit = int(compute_fields["final_exit_code"])
        except ValueError:
            die("compute marker exit code is invalid")
        if (
            compute_fields.get("schema_version") != COMPUTE_MARKER_SCHEMA
            or compute_fields.get("status") != marker_name
            or compute_fields.get("seed") != str(expected_seed)
            or compute_fields.get("phase2_design_sha256") != design_sha
            or compute_fields.get("base_config_hash") != base_hash
            or compute_fields.get("git_commit") != git_commit
            or not compute_fields.get("cluster_name")
            or compute_fields.get("slurm_restart_count") != "0"
            or HEX.fullmatch(compute_fields.get("registry_submission_sha256", "")) is None
            or HEX.fullmatch(compute_fields.get("registry_execution_sha256", "")) is None
            or HEX.fullmatch(compute_fields.get("attempt_claim_sha256", "")) is None
            or compute_fields.get("attempt_ledger_sha256") != ledger_sha
            or compute_fields.get("attempt_index") != str(len(attempts))
            or compute_fields.get("slurm_job_id") != last["slurm_job_id"]
            or (marker_name == "SUCCESS" and final_exit != 0)
            or (marker_name == "FAILED" and final_exit == 0)
        ):
            die("compute terminal marker binding is invalid")
        terminal_sha = hashlib.sha256(terminal_raw_bytes).hexdigest()
        if (
            marker_name == "FAILED"
            and compute_fields.get("terminal_manifest_sha256") != terminal_sha
        ):
            die("terminal FAILED marker does not bind its failure manifest SHA256")
        freeze_sha = compute_fields.get("accepted_freeze_aggregate_sha256")
        if not isinstance(freeze_sha, str) or HEX.fullmatch(freeze_sha) is None:
            die("compute marker freeze identity is invalid")

    result_path: Path | None = None
    result_sha: str | None = None
    rollout_path: Path | None = None
    rollout_sha: str | None = None
    if schema == "phase2-seed-terminal-success/v2":
        result = value.get("result")
        if not isinstance(result, dict) or set(result) != {
            "path",
            "sha256",
            "schema_version",
        }:
            die("success terminal result binding is invalid")
        relative = result.get("path")
        result_sha = result.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(result_sha, str)
            or HEX.fullmatch(result_sha) is None
        ):
            die("success result path/hash is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or len(pure.parts) != 1
            or any(part in ("", ".", "..") for part in pure.parts)
            or "\\" in relative
            or ":" in relative
        ):
            die("success result path must be one POSIX basename")
        candidate = terminal.parent / relative
        result_value, result_raw = load_json(candidate)
        result_path = candidate.resolve()
        if (
            result_path.parent != terminal.parent
            or hashlib.sha256(result_raw).hexdigest() != result_sha
            or result_value.get("seed") != expected_seed
            or result_value.get("source_config_hash") != base_hash
            or result_value.get("phase2_design_sha256") != design_sha
        ):
            die("success result identity or SHA256 is invalid")
        rollout = value.get("rollout")
        if not isinstance(rollout, dict) or set(rollout) != {
            "path",
            "sha256",
            "schema_version",
        }:
            die("success terminal rollout binding is invalid")
        rollout_relative = rollout.get("path")
        rollout_sha = rollout.get("sha256")
        if (
            not isinstance(rollout_relative, str)
            or not isinstance(rollout_sha, str)
            or HEX.fullmatch(rollout_sha) is None
            or rollout.get("schema_version") != "common-beta-trajectory/v2"
            or result_value.get("rollouts_jsonl") != rollout_relative
            or result_value.get("rollouts_sha256") != rollout_sha
        ):
            die("success rollout identity is invalid")
        rollout_pure = PurePosixPath(rollout_relative)
        if (
            rollout_pure.is_absolute()
            or len(rollout_pure.parts) != 1
            or any(part in ("", ".", "..") for part in rollout_pure.parts)
            or "\\" in rollout_relative
            or ":" in rollout_relative
        ):
            die("success rollout path must be one POSIX basename")
        rollout_candidate = terminal.parent / rollout_relative
        if rollout_candidate.is_symlink() or not rollout_candidate.is_file():
            die("success rollout must be a sibling regular file")
        rollout_path = rollout_candidate.resolve()
        if (
            rollout_path.parent != terminal.parent
            or hashlib.sha256(rollout_path.read_bytes()).hexdigest() != rollout_sha
            or rollout_path.name != f"{result_path.stem}.rollouts.jsonl"
        ):
            die("success rollout SHA256 is invalid")
        try:
            rollout_lines = rollout_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            die(f"success rollout JSONL is not UTF-8: {exc}")
        if not rollout_lines or any(not line for line in rollout_lines):
            die("success rollout JSONL must be non-empty and contain no blank lines")
        for line_number, line in enumerate(rollout_lines, 1):
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=lambda pairs, line_number=line_number: _strict_object(
                        pairs,
                        context=f"rollout line {line_number}",
                    ),
                    parse_constant=lambda item: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant: {item}")
                    ),
                )
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                die(f"success rollout JSONL line {line_number} is invalid: {exc}")
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != "common-beta-trajectory/v2"
            ):
                die("success rollout JSONL rows must use common-beta-trajectory/v2")

    validate_physical_attempts(
        terminal=terminal,
        terminal_schema=schema,
        final_capture=capture,
        final_ledger=ledger,
        design_sha=design_sha,
        base_hash=base_hash,
        git_commit=git_commit,
        expected_seed=expected_seed,
    )
    print(marker)
    print(ledger_path)
    print(ledger_sha)
    print("-" if result_path is None else result_path)
    print("-" if result_sha is None else result_sha)
    print(freeze_sha)
    print("-" if rollout_path is None else rollout_path)
    print("-" if rollout_sha is None else rollout_sha)


def validate_physical_attempts(
    *,
    terminal: Path,
    terminal_schema: str,
    final_capture: str,
    final_ledger: dict[str, Any],
    design_sha: str,
    base_hash: str,
    git_commit: str,
    expected_seed: int,
) -> None:
    job_dir = terminal.parent
    attempt_dir = job_dir.parent
    seed_dir = attempt_dir.parent
    if (
        not re.fullmatch(r"job-[A-Za-z0-9][A-Za-z0-9_-]*", job_dir.name)
        or not re.fullmatch(r"attempt-[1-9][0-9]*", attempt_dir.name)
        or seed_dir.name != f"seed-{expected_seed}"
        or seed_dir.parent.name != design_sha
        or job_dir.is_symlink()
        or attempt_dir.is_symlink()
        or seed_dir.is_symlink()
        or seed_dir.parent.is_symlink()
        or job_dir.resolve().parent != attempt_dir.resolve()
        or attempt_dir.resolve().parent != seed_dir.resolve()
        or seed_dir.resolve().parent != seed_dir.parent.resolve()
    ):
        die("terminal path is outside its canonical seed/attempt/job hierarchy")
    attempts = final_ledger["attempts"]
    expected_attempt_names = {f"attempt-{index}" for index in range(1, len(attempts) + 1)}
    observed_attempts = {
        path.name for path in seed_dir.iterdir() if path.name.startswith("attempt-")
    }
    if observed_attempts != expected_attempt_names:
        die("physical attempt directories are missing, extra, or noncontiguous")
    for index, expected in enumerate(attempts, 1):
        current_attempt = seed_dir / f"attempt-{index}"
        if current_attempt.is_symlink() or not current_attempt.is_dir():
            die("physical attempt directory is unsafe")
        for entry in current_attempt.iterdir():
            if entry.name in {"CLAIM", "OUTCOME_REVEAL_STARTED"} or entry.name.startswith("job-"):
                continue
            if re.fullmatch(
                r"\.(?:CLAIM|OUTCOME_REVEAL_STARTED)\.tmp\.[A-Za-z0-9]+",
                entry.name,
            ):
                if entry.is_symlink() or not entry.is_file():
                    die("attempt root contains an unsafe marker staging residue")
                continue
            if (
                re.fullmatch(
                    r"\.job-[A-Za-z0-9_-]+\.(?:in-progress-[A-Za-z0-9_-]+|"
                    r"scheduler\.tmp\.[A-Za-z0-9]+)",
                    entry.name,
                )
                is None
                or entry.is_symlink()
                or not entry.is_dir()
            ):
                die("attempt root contains an unexpected or unsafe entry")
        job_entries = [path for path in current_attempt.iterdir() if path.name.startswith("job-")]
        if len(job_entries) != 1 or job_entries[0].is_symlink() or not job_entries[0].is_dir():
            die("each attempt must have exactly one atomically claimed job directory")
        current_job = job_entries[0]
        marker_names = {
            name
            for name in ("SUCCESS", "FAILED", "SCHEDULER_FAILED")
            if (current_job / name).exists() or (current_job / name).is_symlink()
        }
        if len(marker_names) != 1:
            die("the formal attempt must have exactly one terminal marker")
        terminal_files = [
            path
            for path in current_job.iterdir()
            if path.name
            in {
                "phase2-success-terminal.json",
                "phase2-failure-terminal.json",
            }
        ]
        ledger_path = current_job / "phase2-attempt-ledger.json"
        current_ledger, current_raw = load_json(ledger_path, canonical=True)
        is_final = index == len(attempts)
        claim = parse_key_value(current_attempt / "CLAIM", name="attempt claim")
        common_claim_fields = {
            "schema_version",
            "status",
            "cluster_name",
            "array_job_id",
            "array_task_id",
            "slurm_job_id",
            "slurm_restart_count",
            "attempt_index",
            "seed",
            "phase2_design_sha256",
            "base_config_hash",
            "git_commit",
            "accepted_freeze_aggregate_sha256",
            "registry_submission_sha256",
            "registry_execution_sha256",
            "created_at_utc",
        }
        scheduler_claim = (
            claim.get("schema_version") == "prorm-phase2-formal-scheduler-attempt-claim/v1"
        )
        expected_claim_fields = (
            common_claim_fields | {"registry_scheduler_terminal_sha256"}
            if scheduler_claim
            else common_claim_fields
        )
        if set(claim) != expected_claim_fields:
            die("attempt claim fields differ from the locked schema")
        if (
            (scheduler_claim and claim.get("status") != "CLAIMED_BY_SCHEDULER_RECONCILIATION")
            or (
                not scheduler_claim
                and (
                    claim.get("schema_version") != "prorm-phase2-formal-attempt-claim/v1"
                    or claim.get("status") != "CLAIMED"
                )
            )
            or not claim.get("cluster_name")
            or re.fullmatch(r"[1-9][0-9]*", claim.get("array_job_id", "")) is None
            or claim.get("array_task_id") != str(expected_seed - 20260901)
            or claim.get("slurm_job_id") != expected["slurm_job_id"]
            or claim.get("slurm_restart_count") != "0"
            or claim.get("attempt_index") != str(index)
            or claim.get("seed") != str(expected_seed)
            or claim.get("phase2_design_sha256") != design_sha
            or claim.get("base_config_hash") != base_hash
            or claim.get("git_commit") != git_commit
            or HEX.fullmatch(claim.get("accepted_freeze_aggregate_sha256", "")) is None
            or HEX.fullmatch(claim.get("registry_submission_sha256", "")) is None
            or (
                claim.get("registry_execution_sha256") != "none"
                and HEX.fullmatch(claim.get("registry_execution_sha256", "")) is None
            )
            or (
                scheduler_claim
                and HEX.fullmatch(claim.get("registry_scheduler_terminal_sha256", "")) is None
            )
            or re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                claim.get("created_at_utc", ""),
            )
            is None
            or current_job.name != f"job-{claim['array_job_id']}_{claim['array_task_id']}"
        ):
            die("attempt claim identity is invalid")
        claim_sha = hashlib.sha256((current_attempt / "CLAIM").read_bytes()).hexdigest()
        if final_ledger.get("schema_version") == "phase2-seed-attempt-ledger/v3" and (
            expected.get("cluster_name") != claim["cluster_name"]
            or expected.get("array_job_id") != claim["array_job_id"]
            or str(expected.get("array_task_id")) != claim["array_task_id"]
        ):
            die("attempt ledger scheduler identity disagrees with its claim")
        outcome_marker = current_attempt / "OUTCOME_REVEAL_STARTED"
        expected_reveal = bool(expected["final_outcome_reveal_started"])
        if expected_reveal:
            if outcome_marker.is_symlink() or not outcome_marker.is_file():
                die("revealed attempt lacks its immutable outcome-boundary marker")
            outcome_fields = parse_key_value(
                outcome_marker,
                name="outcome-reveal boundary marker",
            )
            if (
                set(outcome_fields)
                != {
                    "schema_version",
                    "status",
                    "cluster_name",
                    "array_job_id",
                    "array_task_id",
                    "slurm_job_id",
                    "slurm_restart_count",
                    "attempt_index",
                    "seed",
                    "phase2_design_sha256",
                    "base_config_hash",
                    "git_commit",
                    "accepted_freeze_aggregate_sha256",
                    "registry_submission_sha256",
                    "registry_execution_sha256",
                    "attempt_claim_sha256",
                    "created_at_utc",
                }
                or outcome_fields.get("schema_version") != "prorm-phase2-outcome-reveal-boundary/v1"
                or outcome_fields.get("status") != "OUTCOME_REVEAL_STARTED"
                or outcome_fields.get("cluster_name") != claim["cluster_name"]
                or outcome_fields.get("array_job_id") != claim["array_job_id"]
                or outcome_fields.get("array_task_id") != claim["array_task_id"]
                or outcome_fields.get("slurm_job_id") != claim["slurm_job_id"]
                or outcome_fields.get("slurm_restart_count") != "0"
                or outcome_fields.get("attempt_index") != "1"
                or outcome_fields.get("seed") != str(expected_seed)
                or outcome_fields.get("phase2_design_sha256") != design_sha
                or outcome_fields.get("base_config_hash") != base_hash
                or outcome_fields.get("git_commit") != git_commit
                or outcome_fields.get("accepted_freeze_aggregate_sha256")
                != claim["accepted_freeze_aggregate_sha256"]
                or outcome_fields.get("registry_submission_sha256")
                != claim["registry_submission_sha256"]
                or outcome_fields.get("registry_execution_sha256")
                != claim["registry_execution_sha256"]
                or outcome_fields.get("attempt_claim_sha256") != claim_sha
                or re.fullmatch(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                    outcome_fields.get("created_at_utc", ""),
                )
                is None
            ):
                die("outcome-reveal boundary marker identity is invalid")
            outcome_sha = hashlib.sha256(outcome_marker.read_bytes()).hexdigest()
        else:
            if outcome_marker.exists() or outcome_marker.is_symlink():
                die("pre-outcome attempt unexpectedly has an outcome-boundary marker")
            outcome_sha = "none"
        if is_final:
            if current_job != job_dir or len(terminal_files) != 1 or terminal_files[0] != terminal:
                die("submitted terminal must be the unique final-attempt terminal")
            if current_ledger != final_ledger:
                die("final physical ledger differs from the terminal ledger")
            expected_marker = (
                "SCHEDULER_FAILED"
                if final_capture == "scheduler_terminal_reconciliation"
                else (
                    "SUCCESS" if terminal_schema == "phase2-seed-terminal-success/v2" else "FAILED"
                )
            )
            if marker_names != {expected_marker}:
                die("final physical marker disagrees with terminal capture")
            if expected_marker == "SCHEDULER_FAILED":
                final_marker, _ = load_json(
                    current_job / "SCHEDULER_FAILED",
                    canonical=True,
                )
                marker_cluster = final_marker.get("cluster_name")
                marker_array_job = str(final_marker.get("array_job_id"))
                marker_array_task = str(final_marker.get("array_task_id"))
            else:
                final_marker = parse_compute_marker(current_job / expected_marker)
                marker_cluster = final_marker.get("cluster_name")
                marker_array_job = final_marker.get("array_job_id")
                marker_array_task = final_marker.get("array_task_id")
            if (
                marker_cluster != claim["cluster_name"]
                or marker_array_job != claim["array_job_id"]
                or marker_array_task != claim["array_task_id"]
                or (
                    expected_marker != "SCHEDULER_FAILED"
                    and final_marker.get("attempt_claim_sha256") != claim_sha
                )
                or (
                    expected_marker != "SCHEDULER_FAILED"
                    and final_marker.get("outcome_reveal_marker_sha256") != outcome_sha
                )
            ):
                die("final marker disagrees with the authoritative attempt claim")
            if expected_marker != "SCHEDULER_FAILED":
                attempt_evidence = current_job / "attempt-evidence.log"
                if attempt_evidence.is_symlink() or not attempt_evidence.is_file():
                    die("compute attempt evidence is missing or unsafe")
                attempt_evidence_sha = hashlib.sha256(attempt_evidence.read_bytes()).hexdigest()
                if attempt_evidence_sha != expected["log_sha256"]:
                    die("compute attempt log differs from its ledger binding")
                validate_compute_registry(
                    attempt_dir=current_attempt,
                    claim=claim,
                )
            continue


if __name__ == "__main__":
    validate_terminal(sys.argv[1:])
