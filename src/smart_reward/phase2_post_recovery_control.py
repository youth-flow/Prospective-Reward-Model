"""Fail-closed control-plane evidence for post-recovery Phase-2 calibration.

This module deliberately contains no training logic. It binds the one
head-free recovery authorization to a fresh three-seed pilot array, captures
the terminal Slurm allocation rows, and validates the run receipts that are
allowed to enter aggregation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .config import config_hash
from .phase2_config import (
    PHASE2_BUDGETED_END_TO_END_SEEDS,
    PHASE2_BUDGETED_END_TO_END_STAGE,
    PHASE2_POST_RECOVERY_SCHEMA_VERSION,
    load_phase2_config_bundle,
    validate_post_recovery_authorization_reference,
)
from .phase2_recovery_aggregate import verify_phase2_recovery_authorization

ORDERED_SEEDS = (20260801, 20260802, 20260803)
RECOVERY_ARRAY_JOB_ID = "1648125"
RECOVERY_EXECUTION_REVISION = 2
OPTIMIZER_SCHEDULE_SHA256 = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
POST_RECOVERY_CONFIG_SCHEMA = PHASE2_POST_RECOVERY_SCHEMA_VERSION
POST_RECOVERY_DESIGN_NAME = "common-beta-post-recovery-calibration-v1"
POST_RECOVERY_TERMINAL_SCHEMA = "prorm-phase2-post-recovery-pilot-terminal/v1"
POST_RECOVERY_RUN_STATUS_SCHEMA = "prorm-phase2-post-recovery-pilot-run-status/v1"
POST_RECOVERY_OUTPUT_VERIFICATION_SCHEMA = "prorm-phase2-post-recovery-pilot-output-verification/v1"
POST_RECOVERY_AGGREGATE_PUBLICATION_SCHEMA = "prorm-phase2-post-recovery-aggregate-publication/v1"
POST_RECOVERY_AGGREGATE_TERMINAL_SCHEMA = "prorm-phase2-post-recovery-aggregate-terminal/v1"
POST_RECOVERY_AGGREGATE_ATTEMPT_READY_SCHEMA = (
    "prorm-phase2-post-recovery-aggregate-attempt-ready/v1"
)
POST_RECOVERY_AGGREGATE_PUBLICATION_OWNER_SCHEMA = (
    "prorm-phase2-post-recovery-aggregate-publication-owner/v1"
)
POST_RECOVERY_AGGREGATE_AUTHORITY_SCHEMA = (
    "prorm-phase2-post-recovery-aggregate-submission-authority/v1"
)
POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM_SCHEMA = (
    "prorm-phase2-post-recovery-aggregate-evidence-claim/v1"
)
POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM = "EVIDENCE_CLAIM.json"
POST_RECOVERY_AGGREGATE_SUCCESS_SCHEMA = "prorm-phase2-post-recovery-aggregate-status/v3"
POST_RECOVERY_AGGREGATE_SUBMIT_INTENT_SCHEMA = (
    "prorm-phase2-post-recovery-aggregate-submit-intent/v2"
)
POST_RECOVERY_AGGREGATE_SUBMIT_ATTEMPT_SCHEMA = (
    "prorm-phase2-post-recovery-aggregate-submit-attempt/v2"
)
POST_RECOVERY_AGGREGATE_HELD_REQUEST_SCHEMA = "prorm-phase2-post-recovery-aggregate-held-request/v2"
POST_RECOVERY_AGGREGATE_SCRIPT_BINDING_SCHEMA = (
    "prorm-phase2-post-recovery-aggregate-batch-script/v1"
)
POST_RECOVERY_AGGREGATE_SCRIPT_EVIDENCE = "script.sbatch"
POST_RECOVERY_AGGREGATE_CONTROLLER_READBACKS = "controller"
POST_RECOVERY_AGGREGATE_SCRIPT_TRANSPORT = "sbatch-stdin"
POST_RECOVERY_ARRAY_INTENT_SCHEMA = "prorm-phase2-post-recovery-array-intent/v1"
POST_RECOVERY_ARRAY_SUBMISSION_SCHEMA = "prorm-phase2-post-recovery-array-submission/v1"
POST_RECOVERY_HELD_SCHEDULER_SCHEMA = "prorm-phase2-post-recovery-held-scheduler-request/v1"
POST_RECOVERY_PILOT_PHASES = frozenset({"calibration", "freeze"})

_HEX = frozenset("0123456789abcdef")
_SACCT_FIELDS = (
    "JobID",
    "JobIDRaw",
    "State",
    "ExitCode",
    "DerivedExitCode",
    "Cluster",
    "Account",
    "Partition",
    "NNodes",
    "NCPUS",
    "ReqTRES",
    "AllocTRES",
)
_SACCT_FORMAT_FIELDS = (
    "JobID%32",
    "JobIDRaw%32",
    "State%64",
    "ExitCode%32",
    "DerivedExitCode%32",
    "Cluster%64",
    "Account%64",
    "Partition%64",
    "NNodes%16",
    "NCPUS%16",
    "ReqTRES%512",
    "AllocTRES%512",
)
_EXPECTED_REQ_TRES = "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1"
_EXPECTED_ALLOC_TRES = "billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1"
_TERMINAL_ROW_KEYS = frozenset(
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
        "req_tres",
        "alloc_tres",
    }
)
_SUCCESS_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "pilot_phase",
        "workload_exit_code",
        "final_exit_code",
        "slurm_job_id",
        "allocation_job_id_raw",
        "slurm_array_task_job_id",
        "array_job_id",
        "array_task_id",
        "seed",
        "cluster",
        "account",
        "partition",
        "restart_count",
        "phase2_design_sha256",
        "base_config_hash",
        "git_commit",
        "recovery_authorization_sha256",
        "optimizer_schedule_sha256",
        "submission_intent_sha256",
        "submission_ledger_sha256",
        "materialization_mode",
        "recovery_outputs_mounted",
        "hf_root_mount_mode",
        "datasets_cache_scope",
        "artifact_metadata_sha256",
        "phase2_result_sha256",
        "phase2_output_verification_sha256",
        "post_recovery_output_verification_sha256",
        "created_at_utc",
    }
)
_AGGREGATE_PUBLICATION_BASE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "slurm_job_id",
        "slurm_job_is_array",
        "cluster",
        "account",
        "partition",
        "restart_count",
        "pilot_array_job_id",
        "pilot_phase",
        "phase2_design_sha256",
        "base_config_hash",
        "recovery_authorization_sha256",
        "optimizer_schedule_sha256",
        "pilot_terminal_evidence_sha256",
        "submission_intent_sha256",
        "submission_ledger_sha256",
        "aggregate_submission_intent_sha256",
        "aggregate_submission_attempt_sha256",
        "aggregate_submission_attempt_index",
        "aggregate_submission_failure_chain_sha256",
        "phase2_overlay_sha256",
        "phase2_base_sha256",
        "aggregator_git_commit",
        "producer_git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "aggregate_sha256",
        "created_at_utc",
    }
)
_AGGREGATE_PUBLICATION_KEYS = frozenset(
    _AGGREGATE_PUBLICATION_BASE_KEYS
    | {
        "aggregate_attempt_ready_sha256",
        "aggregate_submission_authority_sha256",
        "aggregate_evidence_manifest_sha256",
    }
)
_AGGREGATE_SUCCESS_KEYS = frozenset(
    _AGGREGATE_PUBLICATION_KEYS
    | {
        "aggregate_publication_receipt_sha256",
        "aggregation_terminal_evidence_sha256",
    }
)
_AGGREGATE_TERMINAL_ROW_KEYS = frozenset(
    {
        "job_id",
        "job_id_raw",
        "state",
        "exit_code",
        "derived_exit_code",
        "cluster",
        "account",
        "partition",
        "n_nodes",
        "n_cpus",
        "req_tres",
        "alloc_tres",
    }
)
_AGGREGATE_ATTEMPT_READY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "slurm_job_id",
        "slurm_job_is_array",
        "cluster",
        "account",
        "partition",
        "restart_count",
        "pilot_array_job_id",
        "pilot_phase",
        "phase2_design_sha256",
        "base_config_hash",
        "recovery_authorization_sha256",
        "optimizer_schedule_sha256",
        "pilot_terminal_evidence_sha256",
        "submission_intent_sha256",
        "submission_ledger_sha256",
        "aggregate_submission_intent_sha256",
        "aggregate_submission_attempt_sha256",
        "aggregate_submission_attempt_index",
        "aggregate_submission_failure_chain_sha256",
        "phase2_overlay_sha256",
        "phase2_base_sha256",
        "aggregator_git_commit",
        "producer_git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "final_output",
        "final_evidence_root",
        "attempt_aggregate",
        "attempt_evidence",
        "aggregate_sha256",
        "final_namespace_untouched",
        "created_at_utc",
    }
)
_AGGREGATE_PUBLICATION_OWNER_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "attempt_slurm_job_id",
        "attempt_ready_sha256",
        "aggregate_sha256",
        "created_at_utc",
    }
)
_EXPECTED_AGGREGATE_REQ_TRES = "billing=4,cpu=4,mem=16G,node=1"
_EXPECTED_AGGREGATE_ALLOC_TRES = "billing=4,cpu=4,mem=16G,node=1"
_ARRAY_SPEC = "0-2%2"
_SUBMISSION_INTENT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "pilot_phase",
        "phase2_design_sha256",
        "base_config_hash",
        "recovery_authorization_sha256",
        "optimizer_schedule_sha256",
        "git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "phase2_overlay_sha256",
        "phase2_base_sha256",
        "sbatch_script",
        "export_spec",
        "export_spec_sha256",
        "ordered_seeds",
        "array_spec",
        "max_running_tasks",
        "job_name",
        "project_root",
        "repository_root",
        "submitter_user",
        "cluster",
        "account",
        "partition",
        "qos",
        "nodes",
        "tasks",
        "cpus_per_task",
        "memory",
        "gpus_per_node",
        "walltime",
        "requeue",
        "same_design_resubmission_allowed",
        "replacement_array_allowed",
        "created_at_utc",
    }
)
_SUBMISSION_LEDGER_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "intent_sha256",
        "pilot_phase",
        "phase2_design_sha256",
        "base_config_hash",
        "recovery_authorization_sha256",
        "optimizer_schedule_sha256",
        "git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "phase2_overlay_sha256",
        "phase2_base_sha256",
        "ordered_seeds",
        "array_spec",
        "array_job_id",
        "cluster",
        "scheduler_request",
        "same_design_resubmission_allowed",
        "replacement_array_allowed",
        "released_only_after_ledger_fsync",
        "created_at_utc",
    }
)
_SCHEDULER_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "captured_while_held",
        "raw_scontrol_record",
        "raw_scontrol_sha256",
        "normalized",
    }
)
_NORMALIZED_SCHEDULER_KEYS = frozenset(
    {
        "array_job_id",
        "job_name",
        "array_spec",
        "array_task_throttle",
        "cluster",
        "account",
        "partition",
        "qos",
        "nodes",
        "tasks",
        "cpus",
        "cpus_per_task",
        "memory",
        "gpus_per_node",
        "walltime",
        "tres",
        "tres_per_node",
        "requeue",
        "restarts",
        "command",
        "work_dir",
    }
)
_AGGREGATE_SUBMIT_INTENT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "pilot_phase",
        "phase2_design_sha256",
        "pilot_array_job_id",
        "aggregator_git_commit",
        "project_root",
        "repository_root",
        "final_output",
        "partition",
        "walltime",
        "workload_export_spec",
        "workload_export_spec_sha256",
        "sbatch_script",
        "submitter_user",
        "job_name",
        "cluster",
        "account",
        "nodes",
        "tasks",
        "cpus_per_task",
        "memory",
        "requeue",
        "retry_only_after_exact_terminal_failure",
        "created_at_utc",
    }
)
_AGGREGATE_SUBMIT_ATTEMPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "intent_sha256",
        "attempt_index",
        "slurm_job_id",
        "cluster",
        "scheduler_export_spec",
        "scheduler_export_spec_sha256",
        "scheduler_request",
        "batch_script",
        "released_only_after_attempt_ledger_fsync",
        "created_at_utc",
    }
)
_AGGREGATE_SUBMIT_FAILURE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "intent_sha256",
        "attempt_index",
        "slurm_job_id",
        "attempt_ledger_sha256",
        "row",
        "query",
        "raw_sacct",
        "retry_authorized",
        "captured_at_utc",
    }
)
_AGGREGATE_SUBMIT_SACCT_FIELDS = (
    "JobIDRaw",
    "JobID",
    "JobName",
    "State",
    "ExitCode",
    "DerivedExitCode",
    "Cluster",
    "Account",
    "Partition",
    "NNodes",
    "NCPUS",
    "Submit",
    "Timelimit",
    "ReqTRES",
    "AllocTRES",
)
_AGGREGATE_SUBMIT_SACCT_FORMAT_FIELDS = (
    "JobIDRaw%32",
    "JobID%32",
    "JobName%128",
    "State%64",
    "ExitCode%32",
    "DerivedExitCode%32",
    "Cluster%64",
    "Account%64",
    "Partition%64",
    "NNodes%16",
    "NCPUS%16",
    "Submit%32",
    "Timelimit%32",
    "ReqTRES%512",
    "AllocTRES%512",
)
_AGGREGATE_SUBMIT_SACCT_KEYS = frozenset(_AGGREGATE_SUBMIT_SACCT_FIELDS)
_AGGREGATE_SUBMIT_TERMINAL_FAILURE_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _exact_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str] | set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != set(keys):
        missing = sorted(set(keys) - set(value))
        extra = sorted(set(value) - set(keys))
        raise ValueError(f"{name} fields differ; missing={missing!r}, extra={extra!r}")
    return value


def _digest(value: object, *, name: str, lengths: frozenset[int] = frozenset({64})) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_real_file(path: Path, *, name: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is missing or inaccessible") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} cannot be resolved") from error
    if resolved != path.absolute():
        raise ValueError(f"{name} must use its canonical absolute path")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, raw: bytes, *, name: str) -> None:
    parent = path.parent
    try:
        parent_metadata = parent.lstat()
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} parent is missing or inaccessible") from error
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent.is_symlink()
        or resolved_parent != parent.absolute()
    ):
        raise ValueError(f"{name} parent must be an existing canonical real directory")
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.staged-",
        dir=parent,
    )
    temporary = Path(temporary_text).absolute()
    os.chmod(temporary, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A crash or injected failure may leave only this uniquely named
        # forensic staging file.  The final path is never partially visible.
        raise
    try:
        os.link(
            temporary,
            path,
            src_dir_fd=None,
            dst_dir_fd=None,
            follow_symlinks=False,
        )
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {name}: {path}") from None
    except BaseException:
        # The complete staged inode is retained.  Retrying creates a new
        # staging inode and still cannot clobber the final name.
        raise
    # The final name already points to the completely fsynced inode; an
    # extra staging hard link is harmless forensic residue.
    with suppress(OSError):
        temporary.unlink()
    _fsync_directory(parent)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize crash-resumable terminal publication for one aggregate."""

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o640)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("aggregate terminal lock must be a regular file")
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _valid_utc(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return parsed.tzinfo is None


def sacct_command(array_job_id: str) -> tuple[str, ...]:
    """Return the exact HPC4-compatible allocation query."""

    if re.fullmatch(r"[1-9][0-9]*", array_job_id) is None:
        raise ValueError("array_job_id must be a positive decimal Slurm job ID")
    return (
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        array_job_id,
        f"--format={','.join(_SACCT_FORMAT_FIELDS)}",
    )


def _parse_sacct_raw(raw: bytes, *, array_job_id: str) -> list[dict[str, object]]:
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("raw sacct bytes must be non-empty, bounded, and newline-terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("raw sacct bytes must be UTF-8") from error
    lines = text.splitlines()
    if len(lines) != len(ORDERED_SEEDS) or any(not line for line in lines):
        raise ValueError("raw sacct evidence must contain exactly three allocation rows")

    rows: list[dict[str, object]] = []
    raw_ids: set[str] = set()
    for task, (seed, line) in enumerate(zip(ORDERED_SEEDS, lines, strict=True)):
        fields = line.split("|")
        if len(fields) != len(_SACCT_FIELDS):
            raise ValueError(f"raw sacct row {task} does not have the locked twelve columns")
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
            req_tres,
            alloc_tres,
        ) = fields
        if (
            job_id != f"{array_job_id}_{task}"
            or re.fullmatch(r"[1-9][0-9]*", job_id_raw) is None
            or job_id_raw in raw_ids
            or state != "COMPLETED"
            or exit_code != "0:0"
            or derived_exit_code != "0:0"
            or cluster != "hpc4"
            or account != "sigroup"
            or partition != "gpu-l20"
            or n_nodes != "1"
            or n_cpus != "8"
            or req_tres != _EXPECTED_REQ_TRES
            or alloc_tres != _EXPECTED_ALLOC_TRES
        ):
            raise ValueError(f"raw sacct row {task} is not the exact successful HPC4 allocation")
        raw_ids.add(job_id_raw)
        rows.append(
            {
                "job_id": job_id,
                "job_id_raw": job_id_raw,
                "array_job_id": array_job_id,
                "array_task_id": task,
                "seed": seed,
                "state": state,
                "exit_code": exit_code,
                "derived_exit_code": derived_exit_code,
                "cluster": cluster,
                "account": account,
                "partition": partition,
                "n_nodes": 1,
                "n_cpus": 8,
                "req_tres": req_tres,
                "alloc_tres": alloc_tres,
            }
        )
    return rows


def capture_post_recovery_terminal_evidence(
    array_job_id: str,
    destination: str | os.PathLike[str],
    *,
    pilot_phase: str,
) -> dict[str, object]:
    """Capture exact terminal allocation rows for one post-recovery pilot array."""

    if pilot_phase not in POST_RECOVERY_PILOT_PHASES:
        raise ValueError("pilot_phase must be calibration or freeze")
    command = sacct_command(array_job_id)
    output = Path(destination).absolute()
    if output.name in {"", ".", ".."}:
        raise ValueError("terminal evidence output filename is invalid")
    raw_path = output.with_name(f"{output.stem}.sacct.psv")
    if output.exists() or output.is_symlink() or raw_path.exists() or raw_path.is_symlink():
        raise FileExistsError("refusing to overwrite terminal scheduler evidence")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not execute the locked sacct query") from error
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("the locked sacct query failed or emitted stderr")
    raw = completed.stdout
    rows = _parse_sacct_raw(raw, array_job_id=array_job_id)
    payload = {
        "schema_version": POST_RECOVERY_TERMINAL_SCHEMA,
        "pilot_phase": pilot_phase,
        "captured_at_utc": _utc_now(),
        "query": list(command),
        "array_job_id": array_job_id,
        "ordered_seeds": list(ORDERED_SEEDS),
        "rows": rows,
        "raw_sacct": {
            "filename": raw_path.name,
            "sha256": _sha256_bytes(raw),
            "size_bytes": len(raw),
        },
    }
    _write_exclusive(raw_path, raw, name="raw sacct evidence")
    try:
        _write_exclusive(output, _canonical_json(payload), name="terminal scheduler evidence")
    except BaseException:
        # The pair is one publication unit.  Remove only the exact raw file
        # created above if the canonical envelope cannot be installed.
        raw_path.unlink(missing_ok=True)
        raise
    return payload


def verify_post_recovery_terminal_evidence(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_array_job_id: str,
    expected_pilot_phase: str,
) -> dict[str, object]:
    """Verify canonical JSON and the raw three-row Slurm allocation capture."""

    expected_digest = _digest(expected_sha256, name="terminal evidence SHA256")
    evidence_path = Path(path).absolute()
    _require_real_file(evidence_path, name="terminal scheduler evidence")
    raw_json = evidence_path.read_bytes()
    if _sha256_bytes(raw_json) != expected_digest:
        raise ValueError("terminal scheduler evidence SHA256 mismatch")
    try:
        value = json.loads(
            raw_json.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("terminal scheduler evidence is not strict JSON") from error
    if raw_json != _canonical_json(value):
        raise ValueError("terminal scheduler evidence must use canonical JSON bytes")
    evidence = _exact_mapping(
        value,
        name="terminal scheduler evidence",
        keys={
            "schema_version",
            "pilot_phase",
            "captured_at_utc",
            "query",
            "array_job_id",
            "ordered_seeds",
            "rows",
            "raw_sacct",
        },
    )
    if (
        evidence["schema_version"] != POST_RECOVERY_TERMINAL_SCHEMA
        or evidence["pilot_phase"] != expected_pilot_phase
        or expected_pilot_phase not in POST_RECOVERY_PILOT_PHASES
        or evidence["array_job_id"] != expected_array_job_id
        or evidence["ordered_seeds"] != list(ORDERED_SEEDS)
        or evidence["query"] != list(sacct_command(expected_array_job_id))
        or not _valid_utc(evidence["captured_at_utc"])
    ):
        raise ValueError("terminal scheduler evidence identity is invalid")
    raw_binding = _exact_mapping(
        evidence["raw_sacct"],
        name="terminal scheduler evidence.raw_sacct",
        keys={"filename", "sha256", "size_bytes"},
    )
    expected_raw_name = f"{evidence_path.stem}.sacct.psv"
    if raw_binding["filename"] != expected_raw_name:
        raise ValueError("terminal scheduler evidence names an unexpected raw file")
    raw_path = evidence_path.with_name(expected_raw_name)
    _require_real_file(raw_path, name="raw sacct evidence")
    raw = raw_path.read_bytes()
    if raw_binding["sha256"] != _sha256_bytes(raw) or raw_binding["size_bytes"] != len(raw):
        raise ValueError("terminal scheduler evidence does not bind its raw bytes")
    rows = _parse_sacct_raw(raw, array_job_id=expected_array_job_id)
    if evidence["rows"] != rows:
        raise ValueError("terminal scheduler rows differ from the raw sacct bytes")
    return dict(evidence)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def verify_recovery_authorization_file(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> dict[str, object]:
    """Expose the recovery module's canonical, head-free authorization gate."""

    return verify_phase2_recovery_authorization(path, expected_sha256)


def verify_recovery_authorization_config_binding(
    authorization_path: str | os.PathLike[str],
    overlay_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_pilot_phase: str | None = None,
    expected_stage: str = "pilot",
) -> dict[str, object]:
    """Verify the actual receipt and its exact hash-bound config projection."""

    payload = verify_recovery_authorization_file(
        authorization_path,
        expected_sha256=expected_sha256,
    )
    bundle = load_phase2_config_bundle(overlay_path)
    config = bundle.config
    design_value = config["design"]
    if not isinstance(design_value, Mapping):
        raise TypeError("post-recovery design must be a mapping")
    design = design_value
    stage = design.get("stage")
    pilot_phase = design.get("pilot_phase")
    if expected_stage not in {
        "pilot",
        PHASE2_BUDGETED_END_TO_END_STAGE,
        "confirmatory",
    }:
        raise ValueError("expected_stage must be pilot, budgeted_end_to_end, or confirmatory")
    if (
        config["schema_version"] != POST_RECOVERY_CONFIG_SCHEMA
        or stage != expected_stage
        or (expected_stage == "pilot" and pilot_phase not in POST_RECOVERY_PILOT_PHASES)
        or (expected_stage == PHASE2_BUDGETED_END_TO_END_STAGE and pilot_phase is not None)
        or (expected_stage == "confirmatory" and pilot_phase is not None)
        or "post-recovery" not in str(design.get("name", "")).lower()
        or (expected_stage == "pilot" and tuple(config["run"]["seeds"]) != ORDERED_SEEDS)
        or (
            expected_stage == PHASE2_BUDGETED_END_TO_END_STAGE
            and tuple(config["run"]["seeds"]) != PHASE2_BUDGETED_END_TO_END_SEEDS
        )
        or (expected_pilot_phase is not None and pilot_phase != expected_pilot_phase)
    ):
        raise ValueError("overlay is not the expected locked post-recovery design")
    reference = validate_post_recovery_authorization_reference(
        config["recovery_success_reference"],
        authorization_payload_sha256=expected_sha256,
        authorization_payload=payload,
    )
    protocol = config["reward_model"]["optimizer_protocol"]
    if (
        protocol.get("schema_version") != "deterministic-adamw-lr-decay/v1"
        or protocol.get("role") != "frozen_post_recovery_phase2_optimizer"
        or protocol.get("source_recovery_authorization_sha256") != expected_sha256
        or protocol.get("learning_rate_schedule", {}).get("schedule_sha256")
        != OPTIMIZER_SCHEDULE_SHA256
        or "one_time_recovery" in protocol
    ):
        raise ValueError("overlay does not bind the adopted recovery optimizer schedule")
    return {
        "authorization": payload,
        "authorization_reference": reference,
        "authorization_sha256": expected_sha256,
        "phase2_design_sha256": bundle.design_identity,
        "base_config_hash": config_hash(bundle.base_config),
        "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
        "stage": stage,
        "pilot_phase": pilot_phase,
    }


def parse_post_recovery_success_marker(
    path: str | os.PathLike[str],
) -> dict[str, str]:
    marker_path = Path(path).absolute()
    _require_real_file(marker_path, name="post-recovery SUCCESS marker")
    raw = marker_path.read_bytes()
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("post-recovery SUCCESS marker is malformed")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("post-recovery SUCCESS marker must be UTF-8") from error
    fields: dict[str, str] = {}
    for line in lines:
        if line.count("=") != 1:
            raise ValueError("post-recovery SUCCESS marker line is malformed")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            raise ValueError("post-recovery SUCCESS marker has duplicate or empty fields")
        fields[key] = value
    if set(fields) != _SUCCESS_KEYS:
        missing = sorted(_SUCCESS_KEYS - set(fields))
        extra = sorted(set(fields) - _SUCCESS_KEYS)
        raise ValueError(
            f"post-recovery SUCCESS marker fields differ; missing={missing!r}, extra={extra!r}"
        )
    return fields


def _parse_post_recovery_aggregate_receipt(
    path: str | os.PathLike[str],
    *,
    name: str,
    keys: frozenset[str],
) -> dict[str, str]:
    receipt_path = Path(path).absolute()
    _require_real_file(receipt_path, name=name)
    raw = receipt_path.read_bytes()
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise ValueError(f"{name} is malformed")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    fields: dict[str, str] = {}
    for line in lines:
        if line.count("=") != 1:
            raise ValueError(f"{name} line is malformed")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            raise ValueError(f"{name} has duplicate or empty fields")
        fields[key] = value
    if set(fields) != keys:
        missing = sorted(keys - set(fields))
        extra = sorted(set(fields) - keys)
        raise ValueError(f"{name} fields differ; missing={missing!r}, extra={extra!r}")
    return fields


def parse_post_recovery_aggregate_publication_receipt(
    path: str | os.PathLike[str],
) -> dict[str, str]:
    """Parse the in-job durable aggregate publication receipt."""

    return _parse_post_recovery_aggregate_receipt(
        path,
        name="post-recovery aggregate PUBLISHED receipt",
        keys=_AGGREGATE_PUBLICATION_KEYS,
    )


def parse_post_recovery_aggregate_attempt_ready(
    path: str | os.PathLike[str],
) -> dict[str, str]:
    """Parse one durable CPU-attempt READY receipt."""

    return _parse_post_recovery_aggregate_receipt(
        path,
        name="post-recovery aggregate attempt READY receipt",
        keys=_AGGREGATE_ATTEMPT_READY_KEYS,
    )


def parse_post_recovery_aggregate_publication_owner(
    path: str | os.PathLike[str],
) -> dict[str, str]:
    """Parse the post-terminal attempt claim on one final namespace."""

    return _parse_post_recovery_aggregate_receipt(
        path,
        name="post-recovery aggregate ATTEMPT receipt",
        keys=_AGGREGATE_PUBLICATION_OWNER_KEYS,
    )


def parse_post_recovery_aggregate_success_receipt(
    path: str | os.PathLike[str],
) -> dict[str, str]:
    """Parse the post-job terminally proven aggregate success receipt."""

    return _parse_post_recovery_aggregate_receipt(
        path,
        name="post-recovery aggregate SUCCESS receipt",
        keys=_AGGREGATE_SUCCESS_KEYS,
    )


def _strict_json_file(path: Path, *, name: str) -> dict[str, object]:
    _require_real_file(path, name=name)
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _strict_canonical_json_file(path: Path, *, name: str) -> dict[str, object]:
    value = _strict_json_file(path, name=name)
    if path.read_bytes() != _canonical_json(value):
        raise ValueError(f"{name} must use canonical JSON bytes")
    return value


def _strict_canonical_json_value(path: Path, *, name: str) -> object:
    _require_real_file(path, name=name)
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if raw != _canonical_json(value):
        raise ValueError(f"{name} must use canonical JSON bytes")
    return value


def _parse_export_spec(raw: object) -> dict[str, str]:
    if not isinstance(raw, str) or not raw or "\n" in raw or "\r" in raw:
        raise ValueError("post-recovery export specification is malformed")
    result: dict[str, str] = {}
    for item in raw.split(","):
        if item.count("=") != 1:
            raise ValueError("post-recovery export specification item is malformed")
        key, value = item.split("=", 1)
        if not key or not value or key in result:
            raise ValueError("post-recovery export specification has duplicate or empty fields")
        result[key] = value
    return result


def verify_post_recovery_submission_evidence(
    intent_path: str | os.PathLike[str],
    submission_path: str | os.PathLike[str],
    *,
    expected_pilot_phase: str,
    expected_design_sha256: str,
    expected_base_config_hash: str,
    expected_authorization_sha256: str,
    expected_optimizer_schedule_sha256: str,
    expected_git_commit: str,
    expected_image_sha256: str,
    expected_inventory_sha256: str,
    expected_overlay_sha256: str,
    expected_base_sha256: str,
    expected_overlay_repo_relative: str,
    expected_base_repo_relative: str,
    expected_array_job_id: str,
    expected_intent_sha256: str,
    expected_submission_sha256: str,
    expected_project_root: str,
) -> dict[str, object]:
    """Deep-verify the self-contained pre-release array ledger evidence."""

    for value, name, lengths in (
        (expected_design_sha256, "expected design SHA256", frozenset({64})),
        (expected_base_config_hash, "expected base config hash", frozenset({64})),
        (
            expected_authorization_sha256,
            "expected authorization SHA256",
            frozenset({64}),
        ),
        (
            expected_optimizer_schedule_sha256,
            "expected optimizer schedule SHA256",
            frozenset({64}),
        ),
        (expected_git_commit, "expected Git commit", frozenset({40, 64})),
        (expected_image_sha256, "expected image SHA256", frozenset({64})),
        (expected_inventory_sha256, "expected inventory SHA256", frozenset({64})),
        (expected_overlay_sha256, "expected overlay SHA256", frozenset({64})),
        (expected_base_sha256, "expected base SHA256", frozenset({64})),
        (expected_intent_sha256, "expected intent SHA256", frozenset({64})),
        (
            expected_submission_sha256,
            "expected submission SHA256",
            frozenset({64}),
        ),
    ):
        _digest(value, name=name, lengths=lengths)
    if (
        expected_pilot_phase not in POST_RECOVERY_PILOT_PHASES
        or re.fullmatch(r"[1-9][0-9]*", expected_array_job_id) is None
        or not isinstance(expected_project_root, str)
        or not Path(expected_project_root).is_absolute()
        or Path(expected_project_root) == Path(Path(expected_project_root).anchor)
    ):
        raise ValueError("expected post-recovery submission identity is invalid")
    intent_file = Path(intent_path).absolute()
    submission_file = Path(submission_path).absolute()
    intent = _exact_mapping(
        _strict_canonical_json_file(
            intent_file,
            name="post-recovery submission intent evidence",
        ),
        name="post-recovery submission intent evidence",
        keys=_SUBMISSION_INTENT_KEYS,
    )
    submission = _exact_mapping(
        _strict_canonical_json_file(
            submission_file,
            name="post-recovery submission ledger evidence",
        ),
        name="post-recovery submission ledger evidence",
        keys=_SUBMISSION_LEDGER_KEYS,
    )
    if (
        _sha256_file(intent_file) != expected_intent_sha256
        or _sha256_file(submission_file) != expected_submission_sha256
    ):
        raise ValueError("post-recovery submission evidence SHA256 mismatch")
    job_name = f"prorm-p2-post-{expected_pilot_phase}-{expected_design_sha256[:12]}"
    expected_identity = {
        "pilot_phase": expected_pilot_phase,
        "phase2_design_sha256": expected_design_sha256,
        "base_config_hash": expected_base_config_hash,
        "recovery_authorization_sha256": expected_authorization_sha256,
        "optimizer_schedule_sha256": expected_optimizer_schedule_sha256,
        "git_commit": expected_git_commit,
        "image_sha256": expected_image_sha256,
        "hf_inventory_sha256": expected_inventory_sha256,
        "phase2_overlay_sha256": expected_overlay_sha256,
        "phase2_base_sha256": expected_base_sha256,
    }
    for key, expected_value in expected_identity.items():
        if intent[key] != expected_value or submission[key] != expected_value:
            raise ValueError(f"post-recovery submission evidence {key} differs")
    script = _exact_mapping(
        intent["sbatch_script"],
        name="post-recovery submission intent sbatch_script",
        keys={"repo_relative_path", "sha256"},
    )
    project_root = intent["project_root"]
    repository_root = intent["repository_root"]
    submitter_user = intent["submitter_user"]
    walltime = intent["walltime"]
    if (
        intent["schema_version"] != POST_RECOVERY_ARRAY_INTENT_SCHEMA
        or intent["status"] != "precommitted_before_first_scheduler_submission"
        or intent["ordered_seeds"] != list(ORDERED_SEEDS)
        or intent["array_spec"] != _ARRAY_SPEC
        or intent["max_running_tasks"] != 2
        or intent["job_name"] != job_name
        or intent["cluster"] != "hpc4"
        or intent["account"] != "sigroup"
        or intent["partition"] != "gpu-l20"
        or intent["qos"] != "l20_qos"
        or intent["nodes"] != 1
        or intent["tasks"] != 1
        or intent["cpus_per_task"] != 8
        or intent["memory"] != "96G"
        or intent["gpus_per_node"] != 1
        or intent["requeue"] is not False
        or intent["same_design_resubmission_allowed"] is not False
        or intent["replacement_array_allowed"] is not False
        or not _valid_utc(intent["created_at_utc"])
        or project_root != expected_project_root
        or script["repo_relative_path"] != "scripts/hpc4/phase2_post_recovery_calibration.sbatch"
        or re.fullmatch(
            r"(?:[1-9][0-9]*-[0-9]{2}:[0-9]{2}:[0-9]{2}|"
            r"[0-9]{2}:[0-9]{2}:[0-9]{2})",
            str(walltime),
        )
        is None
        or not isinstance(project_root, str)
        or not Path(project_root).is_absolute()
        or Path(project_root) == Path(Path(project_root).anchor)
        or not isinstance(repository_root, str)
        or not Path(repository_root).is_absolute()
        or Path(repository_root) == Path(Path(repository_root).anchor)
        or not isinstance(submitter_user, str)
        or re.fullmatch(r"[A-Za-z0-9._-]+", submitter_user) is None
    ):
        raise ValueError("post-recovery submission intent policy is invalid")
    _digest(script["sha256"], name="post-recovery sbatch source SHA256")
    export_spec = intent["export_spec"]
    export_spec_sha256 = _digest(
        intent["export_spec_sha256"],
        name="post-recovery export specification SHA256",
    )
    if _sha256_bytes(str(export_spec).encode("utf-8")) != export_spec_sha256:
        raise ValueError("post-recovery export specification SHA256 mismatch")
    exported = _parse_export_spec(export_spec)
    required_exports = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PRORM_PROJECT_ROOT": project_root,
        "PRORM_REPO_ROOT": repository_root,
        "PRORM_IMAGE_SHA256": expected_image_sha256,
        "PRORM_HF_INVENTORY_SHA256": expected_inventory_sha256,
        "PRORM_POST_RECOVERY_OVERLAY_REL": expected_overlay_repo_relative,
        "PRORM_PHASE2_BASE_REL": expected_base_repo_relative,
        "PRORM_POST_RECOVERY_OVERLAY_SHA256": expected_overlay_sha256,
        "PRORM_PHASE2_BASE_SHA256": expected_base_sha256,
        "PRORM_POST_RECOVERY_DESIGN_SHA256": expected_design_sha256,
        "PRORM_PHASE2_BASE_CONFIG_HASH": expected_base_config_hash,
        "PRORM_RECOVERY_AUTHORIZATION_SHA256": expected_authorization_sha256,
        "PRORM_OPTIMIZER_SCHEDULE_SHA256": expected_optimizer_schedule_sha256,
        "PRORM_GIT_COMMIT": expected_git_commit,
        "PRORM_POST_RECOVERY_PILOT_PHASE": expected_pilot_phase,
        "PRORM_POST_RECOVERY_NAMESPACE": expected_pilot_phase,
    }
    for key, expected_value in required_exports.items():
        if exported.get(key) != expected_value:
            raise ValueError(f"post-recovery export specification {key} differs")
    for key in (
        "PRORM_SCRATCH_ROOT",
        "PRORM_IMAGE",
        "PRORM_HF_CACHE",
        "PRORM_HF_INVENTORY",
        "PRORM_RECOVERY_AUTHORIZATION",
    ):
        if not exported.get(key):
            raise ValueError(f"post-recovery export specification lacks {key}")
    if exported["PRORM_HF_INVENTORY"] != (
        f"{exported['PRORM_HF_CACHE']}/inventories/{expected_base_config_hash}.json"
    ):
        raise ValueError("post-recovery export inventory projection differs")
    allowed_exports = set(required_exports) | {
        "PRORM_PROJECT_ROOT",
        "PRORM_SCRATCH_ROOT",
        "PRORM_IMAGE",
        "PRORM_HF_CACHE",
        "PRORM_HF_INVENTORY",
        "PRORM_RECOVERY_AUTHORIZATION",
        "PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT",
        "PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT",
        "PRORM_PHASE2_BETA_SOURCE_AGGREGATE",
        "PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256",
        "PRORM_PHASE2_HORIZON_PARENT_AGGREGATE",
        "PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256",
    }
    if set(exported) - allowed_exports:
        raise ValueError("post-recovery export specification has unexpected fields")
    for prefix in ("BETA_SOURCE", "HORIZON_PARENT"):
        presence = exported.get(f"PRORM_PHASE2_{prefix}_AGGREGATE_PRESENT")
        path = exported.get(f"PRORM_PHASE2_{prefix}_AGGREGATE")
        digest = exported.get(f"PRORM_PHASE2_{prefix}_AGGREGATE_SHA256")
        if presence == "0":
            if path is not None or digest is not None:
                raise ValueError("absent post-recovery predecessor leaked a binding")
        elif presence == "1":
            if not path:
                raise ValueError("present post-recovery predecessor lacks a path")
            _digest(digest, name=f"post-recovery {prefix} predecessor SHA256")
        else:
            raise ValueError("post-recovery predecessor presence is invalid")

    if (
        submission["schema_version"] != POST_RECOVERY_ARRAY_SUBMISSION_SCHEMA
        or submission["status"] != "committed_while_scheduler_held"
        or submission["intent_sha256"] != expected_intent_sha256
        or submission["ordered_seeds"] != list(ORDERED_SEEDS)
        or submission["array_spec"] != _ARRAY_SPEC
        or submission["array_job_id"] != expected_array_job_id
        or submission["cluster"] != "hpc4"
        or submission["same_design_resubmission_allowed"] is not False
        or submission["replacement_array_allowed"] is not False
        or submission["released_only_after_ledger_fsync"] is not True
        or not _valid_utc(submission["created_at_utc"])
    ):
        raise ValueError("post-recovery submission ledger policy is invalid")
    scheduler = _exact_mapping(
        submission["scheduler_request"],
        name="post-recovery held scheduler request",
        keys=_SCHEDULER_REQUEST_KEYS,
    )
    normalized = _exact_mapping(
        scheduler["normalized"],
        name="post-recovery normalized held scheduler request",
        keys=_NORMALIZED_SCHEDULER_KEYS,
    )
    raw_scontrol = scheduler["raw_scontrol_record"]
    raw_scontrol_sha256 = _digest(
        scheduler["raw_scontrol_sha256"],
        name="post-recovery raw scontrol SHA256",
    )
    if (
        scheduler["schema_version"] != POST_RECOVERY_HELD_SCHEDULER_SCHEMA
        or scheduler["captured_while_held"] is not True
        or not isinstance(raw_scontrol, str)
        or _sha256_bytes(raw_scontrol.encode("utf-8")) != raw_scontrol_sha256
    ):
        raise ValueError("post-recovery held scheduler request identity is invalid")
    lines = [line for line in raw_scontrol.splitlines() if line]
    if len(lines) != 1 or "\r" in raw_scontrol:
        raise ValueError("post-recovery raw scontrol evidence must contain one safe row")
    raw_fields: dict[str, str] = {}
    for token in lines[0].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in raw_fields:
            raise ValueError(f"duplicate raw scontrol field: {key}")
        raw_fields[key] = value
    raw_tres: dict[str, str] = {}
    for item in raw_fields.get("TRES", "").split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key in raw_tres:
            raise ValueError(f"duplicate raw scontrol TRES field: {key}")
        raw_tres[key] = value
    expected_command = os.fspath(
        Path(repository_root) / "scripts" / "hpc4" / "phase2_post_recovery_calibration.sbatch"
    )
    if (
        raw_fields.get("ArrayJobId", raw_fields.get("JobId")) != expected_array_job_id
        or raw_fields.get("ArrayTaskId") != _ARRAY_SPEC
        or raw_fields.get("JobName") != job_name
        or not raw_fields.get("UserId", "").startswith(f"{submitter_user}(")
        or raw_fields.get("Account") != "sigroup"
        or raw_fields.get("Partition") != "gpu-l20"
        or raw_fields.get("QOS") != "l20_qos"
        or raw_fields.get("Requeue") != "0"
        or raw_fields.get("Restarts") != "0"
        or raw_fields.get("ArrayTaskThrottle") != "2"
        or raw_fields.get("NumNodes") not in {"1", "1-1"}
        or raw_fields.get("NumTasks") != "1"
        or raw_fields.get("NumCPUs") != "8"
        or raw_fields.get("CPUs/Task") != "8"
        or raw_fields.get("MinMemoryNode") != "96G"
        or raw_fields.get("TimeLimit") != walltime
        or raw_fields.get("JobState") != "PENDING"
        or raw_fields.get("Reason") != "JobHeldUser"
        or raw_fields.get("Command") != expected_command
        or raw_fields.get("WorkDir") != repository_root
        or raw_tres.get("cpu") != "8"
        or raw_tres.get("mem") != "96G"
        or raw_tres.get("node") != "1"
        or raw_tres.get("gres/gpu") != "1"
    ):
        raise ValueError("post-recovery raw held scheduler request differs")
    expected_normalized = {
        "array_job_id": expected_array_job_id,
        "job_name": job_name,
        "array_spec": _ARRAY_SPEC,
        "array_task_throttle": 2,
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "gpu-l20",
        "qos": "l20_qos",
        "nodes": 1,
        "tasks": 1,
        "cpus": 8,
        "cpus_per_task": 8,
        "memory": "96G",
        "gpus_per_node": 1,
        "walltime": walltime,
        "tres": {"cpu": "8", "mem": "96G", "node": "1", "gres/gpu": "1"},
        "tres_per_node": raw_fields.get("TresPerNode"),
        "requeue": False,
        "restarts": 0,
        "command": expected_command,
        "work_dir": repository_root,
    }
    if (
        dict(normalized) != expected_normalized
        or re.fullmatch(
            r"gres(?::|/)gpu(?::[A-Za-z0-9_.-]+)?:1",
            str(normalized["tres_per_node"]),
        )
        is None
    ):
        raise ValueError("post-recovery normalized held scheduler request differs")
    return {
        "intent": dict(intent),
        "submission": dict(submission),
        "intent_sha256": expected_intent_sha256,
        "submission_sha256": expected_submission_sha256,
        "array_job_id": expected_array_job_id,
        "export_spec_sha256": export_spec_sha256,
    }


def _aggregate_evidence_file(
    aggregate_path: Path,
    reference: object,
    *,
    name: str,
) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{name} must be a non-empty relative POSIX path")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or "\\" in reference or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"{name} is not a safe canonical relative POSIX path")
    candidate = aggregate_path.parent.joinpath(*pure.parts).absolute()
    _require_real_file(candidate, name=name)
    expected_root = Path(f"{aggregate_path}.evidence").absolute()
    try:
        candidate.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(f"{name} leaves the aggregate evidence bundle") from error
    return candidate


def _host_absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} is invalid")
    path = Path(value)
    if not path.is_absolute() or path == Path(path.anchor):
        raise ValueError(f"{name} must be an absolute non-root host path")
    return path


def _aggregate_submit_failure_row(
    value: object,
    *,
    name: str,
) -> dict[str, str]:
    row = _exact_mapping(
        value,
        name=name,
        keys=_AGGREGATE_SUBMIT_SACCT_KEYS,
    )
    result = {str(key): str(item) for key, item in row.items()}
    state = result["State"].split("+", 1)[0]
    if (
        re.fullmatch(r"[1-9][0-9]*", result["JobIDRaw"]) is None
        or result["JobID"] != result["JobIDRaw"]
        or state not in _AGGREGATE_SUBMIT_TERMINAL_FAILURE_STATES
        or result["Cluster"] != "hpc4"
        or result["Account"] != "sigroup"
        or result["NNodes"] not in {"0", "1"}
        or result["NCPUS"] not in {"0", "4"}
        or result["ReqTRES"] != _EXPECTED_AGGREGATE_REQ_TRES
        or result["AllocTRES"] not in {"", _EXPECTED_AGGREGATE_ALLOC_TRES}
    ):
        raise ValueError(f"{name} is not an exact terminal CPU failure")
    return result


def _verify_aggregate_submit_held_request(
    *,
    scheduler: Mapping[str, object],
    intent: Mapping[str, object],
    intent_sha256: str,
    attempt_index: int,
    job_id: str,
) -> None:
    if set(scheduler) != {
        "schema_version",
        "captured_while_held",
        "raw_scontrol_record",
        "raw_scontrol_sha256",
        "normalized",
    }:
        raise ValueError("aggregate submission scheduler fields differ")
    raw = scheduler["raw_scontrol_record"]
    if (
        scheduler["schema_version"] != POST_RECOVERY_AGGREGATE_HELD_REQUEST_SCHEMA
        or scheduler["captured_while_held"] is not True
        or not isinstance(raw, str)
        or _sha256_bytes(raw.encode()) != scheduler["raw_scontrol_sha256"]
    ):
        raise ValueError("aggregate held scheduler evidence is invalid")
    lines = [line for line in raw.splitlines() if line]
    if len(lines) != 1 or "\r" in raw:
        raise ValueError("aggregate held scheduler evidence must contain one row")
    fields: dict[str, str] = {}
    for token in lines[0].split():
        if "=" not in token:
            continue
        key, item = token.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate aggregate scontrol field: {key}")
        fields[key] = item
    tres: dict[str, str] = {}
    for item in fields.get("TRES", "").split(","):
        if "=" not in item:
            continue
        key, entry = item.split("=", 1)
        if key in tres:
            raise ValueError(f"duplicate aggregate scontrol TRES field: {key}")
        tres[key] = entry
    comment = f"prorm-aggregate:{intent_sha256}:attempt-{attempt_index}"
    repository_root = str(intent["repository_root"])
    script = _exact_mapping(
        intent["sbatch_script"],
        name="aggregate submission intent sbatch_script",
        keys={
            "repo_relative_path",
            "sha256",
            "git_blob_sha1",
            "size_bytes",
            "git_object",
            "evidence_filename",
            "transport",
        },
    )
    if (
        script["repo_relative_path"] != "scripts/hpc4/phase2_post_recovery_aggregate.sbatch"
        or script["evidence_filename"] != POST_RECOVERY_AGGREGATE_SCRIPT_EVIDENCE
        or script["transport"] != POST_RECOVERY_AGGREGATE_SCRIPT_TRANSPORT
    ):
        raise ValueError("aggregate submission intent sbatch identity differs")
    command = "(null)"
    if (
        fields.get("JobId") != job_id
        or fields.get("JobName") != intent["job_name"]
        or not fields.get("UserId", "").startswith(f"{intent['submitter_user']}(")
        or fields.get("Account") != "sigroup"
        or fields.get("Partition") != intent["partition"]
        or fields.get("Requeue") != "0"
        or fields.get("Restarts") != "0"
        or fields.get("NumNodes") not in {"1", "1-1"}
        or fields.get("NumTasks") != "1"
        or fields.get("NumCPUs") != "4"
        or fields.get("CPUs/Task") != "4"
        or fields.get("MinMemoryNode") != "16G"
        or fields.get("TimeLimit") != intent["walltime"]
        or fields.get("Command") != command
        or fields.get("WorkDir") != repository_root
        or fields.get("Comment") != comment
        or fields.get("BatchFlag") != "1"
        or fields.get("JobState") != "PENDING"
        or fields.get("Reason") != "JobHeldUser"
        or tres.get("cpu") != "4"
        or tres.get("mem") != "16G"
        or tres.get("node") != "1"
        or any("gpu" in key.lower() for key in tres)
        or "gpu" in fields.get("TresPerNode", "").lower()
    ):
        raise ValueError("aggregate held scheduler request differs from intent")
    normalized = _exact_mapping(
        scheduler["normalized"],
        name="aggregate normalized held scheduler request",
        keys={
            "job_id",
            "job_name",
            "cluster",
            "account",
            "partition",
            "nodes",
            "tasks",
            "cpus",
            "cpus_per_task",
            "memory",
            "walltime",
            "requeue",
            "restarts",
            "comment",
            "command",
            "work_dir",
        },
    )
    expected_normalized = {
        "job_id": job_id,
        "job_name": intent["job_name"],
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": intent["partition"],
        "nodes": 1,
        "tasks": 1,
        "cpus": 4,
        "cpus_per_task": 4,
        "memory": "16G",
        "walltime": intent["walltime"],
        "requeue": False,
        "restarts": 0,
        "comment": comment,
        "command": command,
        "work_dir": repository_root,
    }
    if dict(normalized) != expected_normalized:
        raise ValueError("aggregate normalized held scheduler request differs")


def _aggregate_controller_readback_relative(attempt_index: int) -> str:
    if attempt_index <= 0:
        raise ValueError("aggregate attempt index must be positive")
    return f"{POST_RECOVERY_AGGREGATE_CONTROLLER_READBACKS}/attempt-{attempt_index:04d}.sbatch"


def _aggregate_controller_readback_query(job_id: str) -> list[str]:
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ValueError("aggregate controller readback job ID is invalid")
    return ["scontrol", "write", "batch_script", job_id, "-"]


def _aggregate_sbatch_command(
    *,
    intent: Mapping[str, object],
    intent_sha256: str,
    attempt_index: int,
    scheduler_export: str,
    aggregate_file: Path,
) -> list[str]:
    repository_root = str(intent["repository_root"])
    log_root = (
        aggregate_file.parent.parent
        / "slurm-logs"
        / "phase2-post-recovery-aggregate"
        / aggregate_file.name
    )
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={intent['job_name']}",
        "--clusters=hpc4",
        "--account=sigroup",
        f"--partition={intent['partition']}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=16G",
        f"--time={intent['walltime']}",
        "--no-requeue",
        f"--comment=prorm-aggregate:{intent_sha256}:attempt-{attempt_index}",
        f"--chdir={repository_root}",
        f"--output={log_root}/%x-%j.out",
        f"--error={log_root}/%x-%j.err",
        f"--export={scheduler_export}",
    ]


def _verify_aggregate_attempt_script_binding(
    *,
    attempt: Mapping[str, object],
    intent: Mapping[str, object],
    intent_sha256: str,
    attempt_index: int,
    job_id: str,
    scheduler_export: str,
    bundle: Path,
    aggregate_file: Path,
    committed_script: bytes,
) -> str:
    binding = _exact_mapping(
        attempt["batch_script"],
        name=f"aggregate CPU attempt {attempt_index} batch-script binding",
        keys={
            "schema_version",
            "transport",
            "submission_command",
            "stdin_sha256",
            "stdin_size_bytes",
            "controller_readback",
            "controller_matches_committed",
        },
    )
    controller = _exact_mapping(
        binding["controller_readback"],
        name=f"aggregate CPU attempt {attempt_index} controller readback",
        keys={"query", "relative_path", "sha256", "size_bytes"},
    )
    relative = _aggregate_controller_readback_relative(attempt_index)
    controller_file = bundle.joinpath(*PurePosixPath(relative).parts)
    _require_real_file(
        controller_file,
        name=f"aggregate CPU attempt {attempt_index} controller script evidence",
    )
    raw = controller_file.read_bytes()
    expected_command = _aggregate_sbatch_command(
        intent=intent,
        intent_sha256=intent_sha256,
        attempt_index=attempt_index,
        scheduler_export=scheduler_export,
        aggregate_file=aggregate_file,
    )
    if (
        binding["schema_version"] != POST_RECOVERY_AGGREGATE_SCRIPT_BINDING_SCHEMA
        or binding["transport"] != POST_RECOVERY_AGGREGATE_SCRIPT_TRANSPORT
        or binding["submission_command"] != expected_command
        or binding["stdin_sha256"] != _sha256_bytes(committed_script)
        or binding["stdin_size_bytes"] != len(committed_script)
        or binding["controller_matches_committed"] is not True
        or controller["query"] != _aggregate_controller_readback_query(job_id)
        or controller["relative_path"] != relative
        or controller["sha256"] != _sha256_bytes(raw)
        or controller["size_bytes"] != len(raw)
        or raw != committed_script
    ):
        raise ValueError(f"aggregate CPU attempt {attempt_index} script binding differs")
    return relative


def _git_committed_script_bytes(
    repository_root: Path,
    *,
    git_object: str,
) -> bytes:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(repository_root),
                "cat-file",
                "blob",
                git_object,
            ],
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not read committed aggregate sbatch script") from error
    if completed.returncode != 0 or completed.stderr or not completed.stdout:
        raise RuntimeError("committed aggregate sbatch script query failed")
    return completed.stdout


def _verify_aggregate_submission_bundle(
    *,
    attempt_evidence: Path,
    ready: Mapping[str, str],
    aggregate_file: Path,
    project_root: Path,
    require_live_registry: bool,
) -> dict[str, object]:
    bundle = attempt_evidence / "aggregate-submission"
    _real_directory(bundle, name="aggregate submission evidence bundle")
    intent_file = bundle / "intent.json"
    selected_attempt_file = bundle / "attempt.json"
    failure_chain_file = bundle / "failure-chain.json"
    committed_script_file = bundle / POST_RECOVERY_AGGREGATE_SCRIPT_EVIDENCE
    controller_directory = bundle / POST_RECOVERY_AGGREGATE_CONTROLLER_READBACKS
    intent = _exact_mapping(
        _strict_canonical_json_file(
            intent_file,
            name="aggregate CPU submission intent evidence",
        ),
        name="aggregate CPU submission intent evidence",
        keys=_AGGREGATE_SUBMIT_INTENT_KEYS,
    )
    attempt = _exact_mapping(
        _strict_canonical_json_file(
            selected_attempt_file,
            name="aggregate selected CPU attempt evidence",
        ),
        name="aggregate selected CPU attempt evidence",
        keys=_AGGREGATE_SUBMIT_ATTEMPT_KEYS,
    )
    intent_sha256 = _sha256_file(intent_file)
    attempt_sha256 = _sha256_file(selected_attempt_file)
    attempt_index = int(ready["aggregate_submission_attempt_index"])
    if (
        ready["aggregate_submission_intent_sha256"] != intent_sha256
        or ready["aggregate_submission_attempt_sha256"] != attempt_sha256
        or re.fullmatch(
            r"[1-9][0-9]*",
            ready["aggregate_submission_attempt_index"],
        )
        is None
        or intent["schema_version"] != POST_RECOVERY_AGGREGATE_SUBMIT_INTENT_SCHEMA
        or intent["status"] != "precommitted_before_first_cpu_attempt"
        or intent["pilot_phase"] != ready["pilot_phase"]
        or intent["phase2_design_sha256"] != ready["phase2_design_sha256"]
        or intent["pilot_array_job_id"] != ready["pilot_array_job_id"]
        or intent["aggregator_git_commit"] != ready["aggregator_git_commit"]
        or intent["project_root"] != os.fspath(project_root)
        or intent["final_output"] != os.fspath(aggregate_file)
        or intent["partition"] != ready["partition"]
        or intent["cluster"] != "hpc4"
        or intent["account"] != "sigroup"
        or intent["nodes"] != 1
        or intent["tasks"] != 1
        or intent["cpus_per_task"] != 4
        or intent["memory"] != "16G"
        or intent["requeue"] is not False
        or intent["retry_only_after_exact_terminal_failure"] is not True
        or not _valid_utc(intent["created_at_utc"])
    ):
        raise ValueError("aggregate CPU submission intent identity is invalid")
    _host_absolute_path(
        intent["repository_root"],
        name="aggregate CPU submission repository root",
    )
    _digest(
        intent["aggregator_git_commit"],
        name="aggregate CPU submission Git commit",
        lengths=frozenset({40, 64}),
    )
    script = _exact_mapping(
        intent["sbatch_script"],
        name="aggregate CPU submission sbatch_script",
        keys={
            "repo_relative_path",
            "sha256",
            "git_blob_sha1",
            "size_bytes",
            "git_object",
            "evidence_filename",
            "transport",
        },
    )
    if (
        script["repo_relative_path"] != "scripts/hpc4/phase2_post_recovery_aggregate.sbatch"
        or script["git_object"]
        != f"{intent['aggregator_git_commit']}:{script['repo_relative_path']}"
        or script["evidence_filename"] != POST_RECOVERY_AGGREGATE_SCRIPT_EVIDENCE
        or script["transport"] != POST_RECOVERY_AGGREGATE_SCRIPT_TRANSPORT
    ):
        raise ValueError("aggregate CPU submission sbatch source differs")
    _require_real_file(
        committed_script_file,
        name="aggregate committed sbatch script evidence",
    )
    committed_script = committed_script_file.read_bytes()
    if (
        not committed_script
        or len(committed_script) > 1024 * 1024
        or b"\0" in committed_script
        or _digest(
            script["sha256"],
            name="aggregate CPU submission sbatch SHA256",
        )
        != _sha256_bytes(committed_script)
        or _digest(
            script["git_blob_sha1"],
            name="aggregate CPU submission sbatch Git blob SHA1",
            lengths=frozenset({40}),
        )
        != _git_blob_sha1(committed_script)
        or _integer(
            script["size_bytes"],
            name="aggregate CPU submission sbatch size",
            minimum=1,
        )
        != len(committed_script)
    ):
        raise ValueError("aggregate committed sbatch script evidence differs")
    _real_directory(
        controller_directory,
        name="aggregate controller batch-script evidence directory",
    )
    workload_export = str(intent["workload_export_spec"])
    workload_sha256 = _digest(
        intent["workload_export_spec_sha256"],
        name="aggregate workload export SHA256",
    )
    if _sha256_bytes(workload_export.encode()) != workload_sha256:
        raise ValueError("aggregate workload export SHA256 mismatch")
    exported = _parse_export_spec(workload_export)
    expected_exports = {
        "PRORM_PROJECT_ROOT": os.fspath(project_root),
        "PRORM_REPO_ROOT": str(intent["repository_root"]),
        "PRORM_POST_RECOVERY_DESIGN_SHA256": ready["phase2_design_sha256"],
        "PRORM_POST_RECOVERY_ARRAY_JOB_ID": ready["pilot_array_job_id"],
        "PRORM_AGGREGATOR_GIT_COMMIT": ready["aggregator_git_commit"],
        "PRORM_POST_RECOVERY_AGGREGATE_OUTPUT": os.fspath(aggregate_file),
        "PRORM_POST_RECOVERY_PILOT_PHASE": ready["pilot_phase"],
    }
    for key, expected_value in expected_exports.items():
        if exported.get(key) != expected_value:
            raise ValueError(f"aggregate workload export {key} differs")
    registry = (
        project_root
        / "runs"
        / "phase2-post-recovery-aggregate-attempts"
        / aggregate_file.name
        / "submission-registry"
    )
    control_suffix = (
        f",PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY={registry}"
        f",PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256={intent_sha256}"
        f",PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX={attempt_index}"
        ",PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256="
        f"{workload_sha256}"
    )
    scheduler_export = f"{workload_export}{control_suffix}"
    if (
        attempt["schema_version"] != POST_RECOVERY_AGGREGATE_SUBMIT_ATTEMPT_SCHEMA
        or attempt["status"] != "committed_while_scheduler_held"
        or attempt["intent_sha256"] != intent_sha256
        or attempt["attempt_index"] != attempt_index
        or attempt["slurm_job_id"] != ready["slurm_job_id"]
        or attempt["cluster"] != "hpc4"
        or attempt["scheduler_export_spec"] != scheduler_export
        or attempt["scheduler_export_spec_sha256"] != _sha256_bytes(scheduler_export.encode())
        or attempt["released_only_after_attempt_ledger_fsync"] is not True
        or not _valid_utc(attempt["created_at_utc"])
    ):
        raise ValueError("aggregate selected CPU attempt identity is invalid")
    scheduler = _exact_mapping(
        attempt["scheduler_request"],
        name="aggregate selected CPU attempt scheduler request",
        keys={
            "schema_version",
            "captured_while_held",
            "raw_scontrol_record",
            "raw_scontrol_sha256",
            "normalized",
        },
    )
    _verify_aggregate_submit_held_request(
        scheduler=scheduler,
        intent=intent,
        intent_sha256=intent_sha256,
        attempt_index=attempt_index,
        job_id=ready["slurm_job_id"],
    )
    _verify_aggregate_attempt_script_binding(
        attempt=attempt,
        intent=intent,
        intent_sha256=intent_sha256,
        attempt_index=attempt_index,
        job_id=ready["slurm_job_id"],
        scheduler_export=scheduler_export,
        bundle=bundle,
        aggregate_file=aggregate_file,
        committed_script=committed_script,
    )

    chain = _strict_canonical_json_value(
        failure_chain_file,
        name="aggregate CPU failure chain evidence",
    )
    if (
        not isinstance(chain, list)
        or _sha256_file(failure_chain_file) != ready["aggregate_submission_failure_chain_sha256"]
        or len(chain) != attempt_index - 1
    ):
        raise ValueError("aggregate CPU failure chain identity is invalid")
    attempts_directory = bundle / "attempts"
    failures_directory = bundle / "failures"
    _real_directory(
        attempts_directory,
        name="aggregate CPU attempt ledger evidence directory",
    )
    _real_directory(
        failures_directory,
        name="aggregate CPU failure evidence directory",
    )
    expected_attempt_names = {f"attempt-{index:04d}.json" for index in range(1, attempt_index + 1)}
    observed_attempt_names = {
        child.name for child in attempts_directory.iterdir() if not child.name.startswith(".")
    }
    if observed_attempt_names != expected_attempt_names:
        raise ValueError("aggregate CPU attempt ledger evidence is non-contiguous")
    expected_controller_names = {
        f"attempt-{index:04d}.sbatch" for index in range(1, attempt_index + 1)
    }
    observed_controller_names = {
        child.name for child in controller_directory.iterdir() if not child.name.startswith(".")
    }
    if observed_controller_names != expected_controller_names:
        raise ValueError("aggregate CPU controller readback evidence is non-contiguous")
    selected_chain_attempt = attempts_directory / f"attempt-{attempt_index:04d}.json"
    _require_real_file(
        selected_chain_attempt,
        name="selected aggregate CPU attempt chain evidence",
    )
    if selected_chain_attempt.read_bytes() != selected_attempt_file.read_bytes():
        raise ValueError("selected aggregate CPU attempt aliases different bytes")
    expected_failure_names: set[str] = set()
    previous_rows: dict[str, dict[str, str]] = {}
    for expected_index, raw_entry in enumerate(chain, start=1):
        entry = _exact_mapping(
            raw_entry,
            name=f"aggregate CPU failure chain entry {expected_index}",
            keys={
                "attempt_index",
                "slurm_job_id",
                "attempt_ledger_sha256",
                "filename",
                "sha256",
            },
        )
        filename = f"job-{entry['slurm_job_id']}.json"
        attempt_name = f"attempt-{expected_index:04d}.json"
        attempt_path = attempts_directory / attempt_name
        failure_path = failures_directory / filename
        _require_real_file(
            attempt_path,
            name=f"aggregate CPU prior attempt {expected_index}",
        )
        prior_attempt = _exact_mapping(
            _strict_canonical_json_file(
                attempt_path,
                name=f"aggregate CPU prior attempt {expected_index}",
            ),
            name=f"aggregate CPU prior attempt {expected_index}",
            keys=_AGGREGATE_SUBMIT_ATTEMPT_KEYS,
        )
        failure = _exact_mapping(
            _strict_canonical_json_file(
                failure_path,
                name=f"aggregate CPU failure {expected_index}",
            ),
            name=f"aggregate CPU failure {expected_index}",
            keys=_AGGREGATE_SUBMIT_FAILURE_KEYS,
        )
        row = _aggregate_submit_failure_row(
            failure["row"],
            name=f"aggregate CPU failure row {expected_index}",
        )
        raw_binding = _exact_mapping(
            failure["raw_sacct"],
            name=f"aggregate CPU failure raw binding {expected_index}",
            keys={"filename", "sha256", "size_bytes"},
        )
        raw_path = failures_directory / f"job-{entry['slurm_job_id']}.sacct.psv"
        _require_real_file(
            raw_path,
            name=f"aggregate CPU failure raw evidence {expected_index}",
        )
        raw = raw_path.read_bytes()
        failure_query = (
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            str(entry["slurm_job_id"]),
            f"--format={','.join(_AGGREGATE_SUBMIT_SACCT_FORMAT_FIELDS)}",
        )
        try:
            raw_lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError("aggregate CPU failure raw evidence is not UTF-8") from error
        if not raw.endswith(b"\n") or len(raw_lines) != 1 or not raw_lines[0]:
            raise ValueError("aggregate CPU failure raw evidence must contain exactly one row")
        raw_values = raw_lines[0].split("|")
        if len(raw_values) != len(_AGGREGATE_SUBMIT_SACCT_FIELDS):
            raise ValueError("aggregate CPU failure raw evidence fields differ")
        raw_row = dict(zip(_AGGREGATE_SUBMIT_SACCT_FIELDS, raw_values, strict=True))
        prior_scheduler_export = (
            f"{workload_export}"
            f",PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY={registry}"
            f",PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256={intent_sha256}"
            f",PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX={expected_index}"
            ",PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256="
            f"{workload_sha256}"
        )
        if (
            prior_attempt["schema_version"] != POST_RECOVERY_AGGREGATE_SUBMIT_ATTEMPT_SCHEMA
            or prior_attempt["status"] != "committed_while_scheduler_held"
            or prior_attempt["intent_sha256"] != intent_sha256
            or prior_attempt["attempt_index"] != expected_index
            or prior_attempt["slurm_job_id"] != entry["slurm_job_id"]
            or prior_attempt["cluster"] != "hpc4"
            or prior_attempt["scheduler_export_spec"] != prior_scheduler_export
            or prior_attempt["scheduler_export_spec_sha256"]
            != _sha256_bytes(prior_scheduler_export.encode())
            or prior_attempt["released_only_after_attempt_ledger_fsync"] is not True
            or not _valid_utc(prior_attempt["created_at_utc"])
        ):
            raise ValueError(f"aggregate CPU prior attempt {expected_index} identity differs")
        prior_scheduler = _exact_mapping(
            prior_attempt["scheduler_request"],
            name=f"aggregate CPU prior scheduler {expected_index}",
            keys={
                "schema_version",
                "captured_while_held",
                "raw_scontrol_record",
                "raw_scontrol_sha256",
                "normalized",
            },
        )
        _verify_aggregate_submit_held_request(
            scheduler=prior_scheduler,
            intent=intent,
            intent_sha256=intent_sha256,
            attempt_index=expected_index,
            job_id=str(entry["slurm_job_id"]),
        )
        _verify_aggregate_attempt_script_binding(
            attempt=prior_attempt,
            intent=intent,
            intent_sha256=intent_sha256,
            attempt_index=expected_index,
            job_id=str(entry["slurm_job_id"]),
            scheduler_export=prior_scheduler_export,
            bundle=bundle,
            aggregate_file=aggregate_file,
            committed_script=committed_script,
        )
        if (
            entry["attempt_index"] != expected_index
            or entry["filename"] != filename
            or entry["attempt_ledger_sha256"] != _sha256_file(attempt_path)
            or entry["sha256"] != _sha256_file(failure_path)
            or failure["schema_version"]
            != "prorm-phase2-post-recovery-aggregate-attempt-failure/v1"
            or failure["status"] != "TERMINAL_FAILURE"
            or failure["intent_sha256"] != intent_sha256
            or failure["attempt_index"] != expected_index
            or failure["slurm_job_id"] != entry["slurm_job_id"]
            or failure["attempt_ledger_sha256"] != entry["attempt_ledger_sha256"]
            or failure["query"] != list(failure_query)
            or raw_binding["filename"] != raw_path.name
            or raw_binding["sha256"] != _sha256_bytes(raw)
            or raw_binding["size_bytes"] != len(raw)
            or raw_row != row
            or failure["retry_authorized"] is not True
            or not _valid_utc(failure["captured_at_utc"])
            or row["JobIDRaw"] != entry["slurm_job_id"]
            or row["JobName"] != intent["job_name"]
            or row["Partition"] != intent["partition"]
            or row["Timelimit"] != intent["walltime"]
        ):
            raise ValueError(f"aggregate CPU failure chain entry {expected_index} differs")
        expected_failure_names.add(filename)
        previous_rows[str(entry["slurm_job_id"])] = row
    observed_failure_names = {
        child.name
        for child in failures_directory.iterdir()
        if not child.name.startswith(".") and child.suffix == ".json"
    }
    if observed_failure_names != expected_failure_names:
        raise ValueError("aggregate CPU failure evidence set differs")
    expected_failure_raw_names = {
        name.removesuffix(".json") + ".sacct.psv" for name in expected_failure_names
    }
    observed_failure_raw_names = {
        child.name
        for child in failures_directory.iterdir()
        if not child.name.startswith(".") and child.name.endswith(".sacct.psv")
    }
    if observed_failure_raw_names != expected_failure_raw_names:
        raise ValueError("aggregate CPU failure raw evidence set differs")
    if {
        child.name for child in failures_directory.iterdir() if not child.name.startswith(".")
    } != expected_failure_names | expected_failure_raw_names:
        raise ValueError("aggregate CPU failure evidence contains unknown files")

    result = {
        "intent": dict(intent),
        "attempt": dict(attempt),
        "intent_sha256": intent_sha256,
        "attempt_sha256": attempt_sha256,
        "attempt_index": attempt_index,
        "previous_rows": previous_rows,
        "registry": registry,
    }
    if not require_live_registry:
        return result
    _real_directory(registry, name="live aggregate submission registry")
    live_committed = registry / POST_RECOVERY_AGGREGATE_SCRIPT_EVIDENCE
    _require_real_file(
        live_committed,
        name="live aggregate committed sbatch evidence",
    )
    if live_committed.read_bytes() != committed_script_file.read_bytes():
        raise ValueError("copied and live aggregate committed sbatch evidence differ")
    repository_root = _host_absolute_path(
        intent["repository_root"],
        name="aggregate CPU submission repository root",
    )
    git_script = _git_committed_script_bytes(
        repository_root,
        git_object=str(script["git_object"]),
    )
    if git_script != committed_script:
        raise ValueError("aggregate committed sbatch evidence differs from Git")
    live_controller_directory = registry / POST_RECOVERY_AGGREGATE_CONTROLLER_READBACKS
    _real_directory(
        live_controller_directory,
        name="live aggregate controller readback directory",
    )
    live_controller_names = {
        child.name
        for child in live_controller_directory.iterdir()
        if not child.name.startswith(".")
    }
    if live_controller_names != expected_controller_names:
        raise ValueError("live aggregate controller readback chain differs")
    for controller_name in expected_controller_names:
        copied_controller = controller_directory / controller_name
        live_controller = live_controller_directory / controller_name
        _require_real_file(
            live_controller,
            name="live aggregate controller batch-script evidence",
        )
        if copied_controller.read_bytes() != live_controller.read_bytes():
            raise ValueError("copied aggregate controller batch-script evidence differs")
    live_intent = registry / "intent.json"
    live_selected = registry / "attempts" / f"attempt-{attempt_index:04d}.json"
    for copied, live, name in (
        (intent_file, live_intent, "aggregate submission intent"),
        (
            selected_attempt_file,
            live_selected,
            "selected aggregate submission attempt",
        ),
    ):
        _require_real_file(live, name=f"live {name}")
        if copied.read_bytes() != live.read_bytes():
            raise ValueError(f"copied and live {name} differ")
    live_attempt_names = {
        child.name
        for child in (registry / "attempts").iterdir()
        if child.name.startswith("attempt-") and child.suffix == ".json"
    }
    if live_attempt_names != expected_attempt_names:
        raise ValueError("selected aggregate attempt is not the latest registry attempt")
    for index in range(1, attempt_index + 1):
        copied_attempt = attempts_directory / f"attempt-{index:04d}.json"
        live_attempt = registry / "attempts" / copied_attempt.name
        _require_real_file(live_attempt, name="live aggregate attempt ledger")
        if copied_attempt.read_bytes() != live_attempt.read_bytes():
            raise ValueError("copied aggregate attempt ledger chain differs")
    for failure_name in expected_failure_names:
        copied_failure = failures_directory / failure_name
        live_failure = registry / "failures" / failure_name
        _require_real_file(live_failure, name="live aggregate failure evidence")
        if copied_failure.read_bytes() != live_failure.read_bytes():
            raise ValueError("copied aggregate failure evidence differs")
        raw_name = failure_name.removesuffix(".json") + ".sacct.psv"
        copied_raw = failures_directory / raw_name
        live_raw = registry / "failures" / raw_name
        _require_real_file(live_raw, name="live aggregate failure raw evidence")
        if copied_raw.read_bytes() != live_raw.read_bytes():
            raise ValueError("copied aggregate failure raw evidence differs")
    live_failure_names = {
        child.name
        for child in (registry / "failures").iterdir()
        if child.name.startswith("job-") and child.suffix == ".json"
    }
    if live_failure_names != expected_failure_names:
        raise ValueError("live aggregate failure chain differs")
    live_failure_raw_names = {
        child.name
        for child in (registry / "failures").iterdir()
        if child.name.startswith("job-") and child.name.endswith(".sacct.psv")
    }
    if live_failure_raw_names != expected_failure_raw_names:
        raise ValueError("live aggregate failure raw chain differs")
    if {
        child.name for child in (registry / "failures").iterdir() if not child.name.startswith(".")
    } != expected_failure_names | expected_failure_raw_names:
        raise ValueError("live aggregate failure registry contains unknown files")
    return result


def _verify_aggregate_submission_scheduler_authority(
    submission: Mapping[str, object],
) -> dict[str, object]:
    intent = submission["intent"]
    attempt = submission["attempt"]
    previous_rows = submission["previous_rows"]
    if (
        not isinstance(intent, Mapping)
        or not isinstance(attempt, Mapping)
        or not isinstance(previous_rows, Mapping)
    ):
        raise TypeError("aggregate CPU scheduler authority context is invalid")
    user = str(intent["submitter_user"])
    job_name = str(intent["job_name"])
    squeue_query = (
        "squeue",
        "--noheader",
        f"--user={user}",
        f"--name={job_name}",
        "--format=%A",
    )
    squeue = subprocess.run(
        squeue_query,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if squeue.returncode != 0 or squeue.stderr:
        raise RuntimeError("aggregate CPU live scheduler authority query failed")
    live_ids = {line.strip() for line in squeue.stdout.splitlines() if line.strip()}
    if live_ids or any(re.fullmatch(r"[1-9][0-9]*", item) is None for item in live_ids):
        raise ValueError("aggregate CPU terminalization requires no live same-identity job")
    fields = ",".join(_AGGREGATE_SUBMIT_SACCT_FORMAT_FIELDS)
    starttime = str(intent["created_at_utc"])[:-1]
    sacct_query = (
        "sacct",
        "-X",
        f"--starttime={starttime}",
        f"--name={job_name}",
        "--noheader",
        "--parsable2",
        f"--format={fields}",
    )
    sacct = subprocess.run(
        sacct_query,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if sacct.returncode != 0 or sacct.stderr:
        raise RuntimeError("aggregate CPU historical scheduler authority query failed")
    if not sacct.stdout or not sacct.stdout.endswith("\n") or "\r" in sacct.stdout:
        raise ValueError("aggregate CPU historical scheduler authority bytes are unsafe")
    names = _AGGREGATE_SUBMIT_SACCT_FIELDS
    rows: dict[str, dict[str, str]] = {}
    for line in sacct.stdout.splitlines():
        values = line.split("|")
        if not line or "\r" in line or len(values) != len(names):
            raise ValueError("aggregate CPU scheduler history is malformed")
        row = dict(zip(names, values, strict=True))
        job_id = row["JobIDRaw"]
        if (
            re.fullmatch(r"[1-9][0-9]*", job_id) is None
            or row["JobID"] != job_id
            or row["JobName"] != job_name
            or row["Cluster"] != "hpc4"
            or row["Account"] != "sigroup"
            or row["Partition"] != intent["partition"]
            or row["Timelimit"] != intent["walltime"]
            or row["ReqTRES"] != _EXPECTED_AGGREGATE_REQ_TRES
            or row["AllocTRES"] not in {"", _EXPECTED_AGGREGATE_ALLOC_TRES}
            or job_id in rows
        ):
            raise ValueError("aggregate CPU scheduler history differs from intent")
        rows[job_id] = row
    selected_job_id = str(attempt["slurm_job_id"])
    expected_ids = set(previous_rows) | {selected_job_id}
    if set(rows) != expected_ids:
        raise ValueError("aggregate CPU scheduler history has an unregistered or missing attempt")
    for job_id, expected_row in previous_rows.items():
        if rows[job_id] != expected_row:
            raise ValueError("aggregate CPU prior failure history changed")
    selected = rows[selected_job_id]
    if (
        selected["State"] != "COMPLETED"
        or selected["ExitCode"] != "0:0"
        or selected["DerivedExitCode"] != "0:0"
        or selected["NNodes"] != "1"
        or selected["NCPUS"] != "4"
        or selected["AllocTRES"] != _EXPECTED_AGGREGATE_ALLOC_TRES
    ):
        raise ValueError("selected aggregate CPU attempt is not the unique exact success")
    return {
        "squeue_query": squeue_query,
        "squeue_raw": squeue.stdout.encode(),
        "sacct_query": sacct_query,
        "sacct_raw": sacct.stdout.encode(),
        "rows": [rows[job_id] for job_id in sorted(rows, key=int)],
        "selected_job_id": selected_job_id,
        "intent_sha256": str(submission["intent_sha256"]),
        "attempt_sha256": str(submission["attempt_sha256"]),
        "attempt_index": int(submission["attempt_index"]),
    }


def _aggregate_submission_authority_artifacts(
    authority: Mapping[str, object],
    *,
    attempt_ready_sha256: str,
    captured_at_utc: str,
) -> dict[str, bytes]:
    squeue_raw = authority["squeue_raw"]
    sacct_raw = authority["sacct_raw"]
    rows = authority["rows"]
    if (
        not isinstance(squeue_raw, bytes)
        or squeue_raw
        or not isinstance(sacct_raw, bytes)
        or not sacct_raw
        or not isinstance(rows, list)
    ):
        raise ValueError("aggregate submission authority capture is invalid")
    squeue_name = "AUTHORITY.squeue.txt"
    sacct_name = "AUTHORITY.sacct.psv"
    if not _valid_utc(captured_at_utc):
        raise ValueError("aggregate authority capture timestamp is invalid")
    payload = {
        "schema_version": POST_RECOVERY_AGGREGATE_AUTHORITY_SCHEMA,
        "captured_at_utc": captured_at_utc,
        "attempt_ready_sha256": _digest(
            attempt_ready_sha256,
            name="aggregate attempt READY SHA256",
        ),
        "aggregate_submission_intent_sha256": authority["intent_sha256"],
        "aggregate_submission_attempt_sha256": authority["attempt_sha256"],
        "aggregate_submission_attempt_index": authority["attempt_index"],
        "selected_slurm_job_id": authority["selected_job_id"],
        "squeue": {
            "query": list(authority["squeue_query"]),
            "raw_filename": squeue_name,
            "raw_sha256": _sha256_bytes(squeue_raw),
            "raw_size_bytes": len(squeue_raw),
            "live_job_ids": [],
        },
        "sacct": {
            "query": list(authority["sacct_query"]),
            "raw_filename": sacct_name,
            "raw_sha256": _sha256_bytes(sacct_raw),
            "raw_size_bytes": len(sacct_raw),
            "rows": rows,
        },
    }
    return {
        "AUTHORITY.json": _canonical_json(payload),
        squeue_name: squeue_raw,
        sacct_name: sacct_raw,
    }


def _verify_aggregate_submission_authority_evidence(
    evidence_root: Path,
    *,
    ready_sha256: str,
    submission: Mapping[str, object],
    expected_captured_at_utc: str,
    fresh_authority: Mapping[str, object] | None = None,
) -> dict[str, object]:
    authority_root = evidence_root / "aggregation-attempt"
    authority_path = authority_root / "AUTHORITY.json"
    squeue_path = authority_root / "AUTHORITY.squeue.txt"
    sacct_path = authority_root / "AUTHORITY.sacct.psv"
    payload = _exact_mapping(
        _strict_canonical_json_file(
            authority_path,
            name="aggregate submission authority evidence",
        ),
        name="aggregate submission authority evidence",
        keys={
            "schema_version",
            "captured_at_utc",
            "attempt_ready_sha256",
            "aggregate_submission_intent_sha256",
            "aggregate_submission_attempt_sha256",
            "aggregate_submission_attempt_index",
            "selected_slurm_job_id",
            "squeue",
            "sacct",
        },
    )
    intent = submission["intent"]
    attempt = submission["attempt"]
    previous_rows = submission["previous_rows"]
    if (
        not isinstance(intent, Mapping)
        or not isinstance(attempt, Mapping)
        or not isinstance(previous_rows, Mapping)
    ):
        raise TypeError("aggregate submission authority context is invalid")
    if (
        payload["schema_version"] != POST_RECOVERY_AGGREGATE_AUTHORITY_SCHEMA
        or not _valid_utc(expected_captured_at_utc)
        or payload["captured_at_utc"] != expected_captured_at_utc
        or payload["attempt_ready_sha256"] != ready_sha256
        or payload["aggregate_submission_intent_sha256"] != submission["intent_sha256"]
        or payload["aggregate_submission_attempt_sha256"] != submission["attempt_sha256"]
        or payload["aggregate_submission_attempt_index"] != submission["attempt_index"]
        or payload["selected_slurm_job_id"] != attempt["slurm_job_id"]
    ):
        raise ValueError("aggregate submission authority identity is invalid")
    squeue = _exact_mapping(
        payload["squeue"],
        name="aggregate submission authority squeue binding",
        keys={
            "query",
            "raw_filename",
            "raw_sha256",
            "raw_size_bytes",
            "live_job_ids",
        },
    )
    sacct = _exact_mapping(
        payload["sacct"],
        name="aggregate submission authority sacct binding",
        keys={
            "query",
            "raw_filename",
            "raw_sha256",
            "raw_size_bytes",
            "rows",
        },
    )
    _require_real_file(squeue_path, name="aggregate authority raw squeue evidence")
    _require_real_file(sacct_path, name="aggregate authority raw sacct evidence")
    squeue_raw = squeue_path.read_bytes()
    sacct_raw = sacct_path.read_bytes()
    expected_squeue_query = [
        "squeue",
        "--noheader",
        f"--user={intent['submitter_user']}",
        f"--name={intent['job_name']}",
        "--format=%A",
    ]
    fields = ",".join(_AGGREGATE_SUBMIT_SACCT_FORMAT_FIELDS)
    expected_sacct_query = [
        "sacct",
        "-X",
        f"--starttime={str(intent['created_at_utc'])[:-1]}",
        f"--name={intent['job_name']}",
        "--noheader",
        "--parsable2",
        f"--format={fields}",
    ]
    if (
        squeue["query"] != expected_squeue_query
        or squeue["raw_filename"] != squeue_path.name
        or squeue["raw_sha256"] != _sha256_bytes(squeue_raw)
        or squeue["raw_size_bytes"] != len(squeue_raw)
        or squeue["live_job_ids"] != []
        or squeue_raw
        or sacct["query"] != expected_sacct_query
        or sacct["raw_filename"] != sacct_path.name
        or sacct["raw_sha256"] != _sha256_bytes(sacct_raw)
        or sacct["raw_size_bytes"] != len(sacct_raw)
        or not sacct_raw.endswith(b"\n")
    ):
        raise ValueError("aggregate submission authority raw bindings differ")
    try:
        lines = sacct_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("aggregate authority sacct evidence must be UTF-8") from error
    rows: dict[str, dict[str, str]] = {}
    for line in lines:
        values = line.split("|")
        if not line or "\r" in line or len(values) != len(_AGGREGATE_SUBMIT_SACCT_FIELDS):
            raise ValueError("aggregate authority sacct row is malformed")
        row = dict(zip(_AGGREGATE_SUBMIT_SACCT_FIELDS, values, strict=True))
        job_id = row["JobIDRaw"]
        if (
            re.fullmatch(r"[1-9][0-9]*", job_id) is None
            or row["JobID"] != job_id
            or row["JobName"] != intent["job_name"]
            or row["Cluster"] != "hpc4"
            or row["Account"] != "sigroup"
            or row["Partition"] != intent["partition"]
            or row["Timelimit"] != intent["walltime"]
            or row["ReqTRES"] != _EXPECTED_AGGREGATE_REQ_TRES
            or row["AllocTRES"] not in {"", _EXPECTED_AGGREGATE_ALLOC_TRES}
            or job_id in rows
        ):
            raise ValueError("aggregate authority sacct row differs from intent")
        rows[job_id] = row
    selected_job_id = str(attempt["slurm_job_id"])
    if set(rows) != set(previous_rows) | {selected_job_id}:
        raise ValueError("aggregate authority evidence has an ambiguous attempt set")
    for job_id, expected_row in previous_rows.items():
        if rows[job_id] != expected_row:
            raise ValueError("aggregate authority prior failure row differs")
    selected = rows[selected_job_id]
    if (
        selected["State"] != "COMPLETED"
        or selected["ExitCode"] != "0:0"
        or selected["DerivedExitCode"] != "0:0"
        or selected["NNodes"] != "1"
        or selected["NCPUS"] != "4"
        or selected["AllocTRES"] != _EXPECTED_AGGREGATE_ALLOC_TRES
    ):
        raise ValueError("aggregate authority selected attempt is not exact success")
    normalized_rows = [rows[job_id] for job_id in sorted(rows, key=int)]
    if sacct["rows"] != normalized_rows:
        raise ValueError("aggregate authority parsed rows differ from raw evidence")
    if fresh_authority is not None and (
        fresh_authority["squeue_raw"] != squeue_raw
        or fresh_authority["sacct_raw"] != sacct_raw
        or fresh_authority["rows"] != normalized_rows
        or list(fresh_authority["squeue_query"]) != expected_squeue_query
        or list(fresh_authority["sacct_query"]) != expected_sacct_query
    ):
        raise ValueError(
            "published aggregate authority evidence differs from fresh scheduler state"
        )
    return {
        "path": authority_path,
        "sha256": _sha256_file(authority_path),
        "payload": dict(payload),
    }


def verify_post_recovery_aggregate_attempt_ready(
    aggregate_path: str | os.PathLike[str],
    *,
    attempt_job_id: str,
) -> dict[str, object]:
    """Verify a persistent READY attempt before any final publication."""

    if re.fullmatch(r"[1-9][0-9]*", attempt_job_id) is None:
        raise ValueError("aggregate attempt job ID must be a positive integer")
    aggregate_file = Path(aggregate_path).absolute()
    if aggregate_file.parent.name != "aggregates":
        raise ValueError("final aggregate path must use the project aggregates namespace")
    project_root = aggregate_file.parent.parent
    _canonical_project = project_root.resolve(strict=True)
    if project_root.is_symlink() or not project_root.is_dir() or _canonical_project != project_root:
        raise ValueError("aggregate attempt project root is unsafe")
    attempt_root = (
        project_root
        / "runs"
        / "phase2-post-recovery-aggregate-attempts"
        / aggregate_file.name
        / f"job-{attempt_job_id}"
    )
    try:
        attempt_metadata = attempt_root.lstat()
    except OSError as error:
        raise ValueError("aggregate attempt directory is missing") from error
    if (
        not stat.S_ISDIR(attempt_metadata.st_mode)
        or attempt_root.is_symlink()
        or attempt_root.resolve(strict=True) != attempt_root
    ):
        raise ValueError("aggregate attempt directory is unsafe")
    ready_path = attempt_root / "READY"
    ready = parse_post_recovery_aggregate_attempt_ready(ready_path)
    staged_aggregate = attempt_root / "aggregate.json"
    attempt_evidence = attempt_root / "evidence"
    _require_real_file(staged_aggregate, name="staged post-recovery aggregate")
    try:
        evidence_metadata = attempt_evidence.lstat()
    except OSError as error:
        raise ValueError("aggregate attempt evidence directory is missing") from error
    if (
        not stat.S_ISDIR(evidence_metadata.st_mode)
        or attempt_evidence.is_symlink()
        or attempt_evidence.resolve(strict=True) != attempt_evidence
    ):
        raise ValueError("aggregate attempt evidence directory is unsafe")
    aggregate_sha256 = _sha256_file(staged_aggregate)
    if (
        ready["schema_version"] != POST_RECOVERY_AGGREGATE_ATTEMPT_READY_SCHEMA
        or ready["status"] != "READY"
        or ready["slurm_job_id"] != attempt_job_id
        or ready["slurm_job_is_array"] != "false"
        or ready["cluster"] != "hpc4"
        or ready["account"] != "sigroup"
        or ready["partition"] not in {"amd", "intel"}
        or ready["restart_count"] != "0"
        or ready["final_output"] != os.fspath(aggregate_file)
        or ready["final_evidence_root"] != f"{aggregate_file}.evidence"
        or ready["attempt_aggregate"] != "aggregate.json"
        or ready["attempt_evidence"] != "evidence"
        or ready["aggregate_sha256"] != aggregate_sha256
        or ready["final_namespace_untouched"] != "true"
        or not _valid_utc(ready["created_at_utc"])
    ):
        raise ValueError("aggregate attempt READY identity is invalid")
    staged = _strict_json_file(
        staged_aggregate,
        name="staged post-recovery aggregate",
    )
    if staged.get("schema_version") != "common-beta-pilot-selection-aggregate/v3":
        raise ValueError("staged post-recovery aggregate schema is invalid")
    pilot_phase = staged.get("pilot_phase")
    if pilot_phase not in POST_RECOVERY_PILOT_PHASES:
        raise ValueError("staged post-recovery aggregate pilot phase is invalid")
    design_sha256 = _digest(
        staged.get("phase2_design_sha256"),
        name="staged post-recovery aggregate design SHA256",
    )
    base_config_hash = _digest(
        staged.get("source_config_hash"),
        name="staged post-recovery aggregate base config hash",
    )
    control = _exact_mapping(
        staged.get("post_recovery_control"),
        name="staged post-recovery aggregate.post_recovery_control",
        keys={
            "schema_version",
            "pilot_phase",
            "phase2_overlay",
            "phase2_overlay_repo_relative",
            "phase2_overlay_sha256",
            "phase2_overlay_git_blob_sha1",
            "phase2_overlay_git_commit",
            "normalized_phase2_config",
            "normalized_phase2_config_sha256",
            "recovery_authorization",
            "recovery_authorization_sha256",
            "optimizer_schedule_sha256",
            "submission_intent",
            "submission_intent_sha256",
            "submission_ledger",
            "submission_ledger_sha256",
            "pilot_terminal_evidence",
            "pilot_terminal_evidence_sha256",
            "pilot_array_job_id",
            "ordered_seeds",
            "materialization_mode",
            "recovery_outputs_reused",
            "all_tasks_terminal_completed_zero_exit",
            "post_recovery_validator_source_sha256",
            "phase2_deep_validator_source_sha256",
        },
    )
    normalized = control["normalized_phase2_config"]
    if not isinstance(normalized, Mapping):
        raise TypeError("staged post-recovery normalized config must be a mapping")
    design = normalized.get("design")
    if not isinstance(design, Mapping):
        raise TypeError("staged post-recovery design must be a mapping")
    base_relative = design.get("source_config")
    overlay_relative = control["phase2_overlay_repo_relative"]
    if (
        not isinstance(base_relative, str)
        or not isinstance(overlay_relative, str)
        or not base_relative.startswith("configs/")
        or not overlay_relative.startswith("configs/")
    ):
        raise ValueError("staged post-recovery config references are invalid")
    overlay_file = attempt_evidence.joinpath(*PurePosixPath(overlay_relative).parts)
    base_file = attempt_evidence.joinpath(*PurePosixPath(base_relative).parts)
    intent_file = attempt_evidence / "submission-registry" / "intent.json"
    submission_file = attempt_evidence / "submission-registry" / "submission.json"
    for evidence_file, name in (
        (overlay_file, "staged overlay evidence"),
        (base_file, "staged base evidence"),
        (intent_file, "staged submission intent evidence"),
        (submission_file, "staged submission ledger evidence"),
    ):
        _require_real_file(evidence_file, name=name)
    expected_final_prefix = f"{aggregate_file.name}.evidence"
    expected_references = {
        "phase2_overlay": f"{expected_final_prefix}/{overlay_relative}",
        "submission_intent": (f"{expected_final_prefix}/submission-registry/intent.json"),
        "submission_ledger": (f"{expected_final_prefix}/submission-registry/submission.json"),
    }
    for key, expected_value in expected_references.items():
        if control[key] != expected_value:
            raise ValueError(f"staged post-recovery aggregate {key} differs")
    identity = _exact_mapping(
        staged.get("aggregation_identity"),
        name="staged post-recovery aggregate.aggregation_identity",
        keys={
            "schema_version",
            "aggregator_git_commit",
            "producer_git_commit",
            "image_sha256",
            "hf_inventory_sha256",
            "validator_source_sha256",
        },
    )
    expected_ready = {
        "pilot_array_job_id": str(control["pilot_array_job_id"]),
        "pilot_phase": str(pilot_phase),
        "phase2_design_sha256": design_sha256,
        "base_config_hash": base_config_hash,
        "recovery_authorization_sha256": str(control["recovery_authorization_sha256"]),
        "optimizer_schedule_sha256": str(control["optimizer_schedule_sha256"]),
        "pilot_terminal_evidence_sha256": str(control["pilot_terminal_evidence_sha256"]),
        "submission_intent_sha256": str(control["submission_intent_sha256"]),
        "submission_ledger_sha256": str(control["submission_ledger_sha256"]),
        "phase2_overlay_sha256": str(control["phase2_overlay_sha256"]),
        "phase2_base_sha256": _sha256_file(base_file),
        "aggregator_git_commit": str(identity["aggregator_git_commit"]),
        "producer_git_commit": str(identity["producer_git_commit"]),
        "image_sha256": str(identity["image_sha256"]),
        "hf_inventory_sha256": str(identity["hf_inventory_sha256"]),
    }
    for key, expected_value in expected_ready.items():
        if ready[key] != expected_value:
            raise ValueError(f"aggregate attempt READY {key} differs")
    if (
        _sha256_file(overlay_file) != ready["phase2_overlay_sha256"]
        or _sha256_file(intent_file) != ready["submission_intent_sha256"]
        or _sha256_file(submission_file) != ready["submission_ledger_sha256"]
    ):
        raise ValueError("aggregate attempt READY does not bind its evidence files")
    verify_post_recovery_submission_evidence(
        intent_file,
        submission_file,
        expected_pilot_phase=str(pilot_phase),
        expected_design_sha256=design_sha256,
        expected_base_config_hash=base_config_hash,
        expected_authorization_sha256=ready["recovery_authorization_sha256"],
        expected_optimizer_schedule_sha256=ready["optimizer_schedule_sha256"],
        expected_git_commit=ready["producer_git_commit"],
        expected_image_sha256=ready["image_sha256"],
        expected_inventory_sha256=ready["hf_inventory_sha256"],
        expected_overlay_sha256=ready["phase2_overlay_sha256"],
        expected_base_sha256=ready["phase2_base_sha256"],
        expected_overlay_repo_relative=overlay_relative,
        expected_base_repo_relative=base_relative,
        expected_array_job_id=ready["pilot_array_job_id"],
        expected_intent_sha256=ready["submission_intent_sha256"],
        expected_submission_sha256=ready["submission_ledger_sha256"],
        expected_project_root=os.fspath(project_root),
    )
    for key in (
        "aggregate_submission_intent_sha256",
        "aggregate_submission_attempt_sha256",
        "aggregate_submission_failure_chain_sha256",
    ):
        _digest(ready[key], name=f"aggregate attempt READY {key}")
    aggregate_submission = _verify_aggregate_submission_bundle(
        attempt_evidence=attempt_evidence,
        ready=ready,
        aggregate_file=aggregate_file,
        project_root=project_root,
        require_live_registry=True,
    )
    expected_files, expected_directories = _expected_aggregate_evidence_paths(
        overlay_relative=overlay_relative,
        base_relative=base_relative,
        submission=aggregate_submission,
        published=False,
    )
    evidence_tree = _verify_exact_evidence_tree(
        attempt_evidence,
        expected_files=expected_files,
        expected_directories=expected_directories,
        name="post-recovery aggregate attempt evidence",
    )
    return {
        "ready": ready,
        "ready_path": ready_path,
        "ready_sha256": _sha256_file(ready_path),
        "attempt_root": attempt_root,
        "attempt_aggregate": staged_aggregate,
        "attempt_evidence": attempt_evidence,
        "aggregate_sha256": aggregate_sha256,
        "aggregate": staged,
        "aggregate_submission": aggregate_submission,
        "evidence_tree": evidence_tree,
    }


def _post_recovery_aggregate_context(
    aggregate_path: str | os.PathLike[str],
) -> dict[str, object]:
    aggregate_file = Path(aggregate_path).absolute()
    aggregate = _strict_json_file(
        aggregate_file,
        name="post-recovery aggregate",
    )
    if aggregate.get("schema_version") != "common-beta-pilot-selection-aggregate/v3":
        raise ValueError("post-recovery aggregate receipt requires the native v3 schema")
    pilot_phase = aggregate.get("pilot_phase")
    if pilot_phase not in POST_RECOVERY_PILOT_PHASES:
        raise ValueError("post-recovery aggregate pilot phase is invalid")
    design_sha256 = _digest(
        aggregate.get("phase2_design_sha256"),
        name="post-recovery aggregate phase2_design_sha256",
    )
    base_config_hash = _digest(
        aggregate.get("source_config_hash"),
        name="post-recovery aggregate source_config_hash",
    )
    control = _exact_mapping(
        aggregate.get("post_recovery_control"),
        name="post-recovery aggregate.post_recovery_control",
        keys={
            "schema_version",
            "pilot_phase",
            "phase2_overlay",
            "phase2_overlay_repo_relative",
            "phase2_overlay_sha256",
            "phase2_overlay_git_blob_sha1",
            "phase2_overlay_git_commit",
            "normalized_phase2_config",
            "normalized_phase2_config_sha256",
            "recovery_authorization",
            "recovery_authorization_sha256",
            "optimizer_schedule_sha256",
            "submission_intent",
            "submission_intent_sha256",
            "submission_ledger",
            "submission_ledger_sha256",
            "pilot_terminal_evidence",
            "pilot_terminal_evidence_sha256",
            "pilot_array_job_id",
            "ordered_seeds",
            "materialization_mode",
            "recovery_outputs_reused",
            "all_tasks_terminal_completed_zero_exit",
            "post_recovery_validator_source_sha256",
            "phase2_deep_validator_source_sha256",
        },
    )
    normalized = control["normalized_phase2_config"]
    if not isinstance(normalized, Mapping):
        raise TypeError("post-recovery aggregate normalized config must be a mapping")
    design = normalized.get("design")
    if not isinstance(design, Mapping):
        raise TypeError("post-recovery aggregate normalized config design must be a mapping")
    source_config = design.get("source_config")
    if not isinstance(source_config, str) or not source_config:
        raise ValueError("post-recovery aggregate source config path is invalid")
    source_pure = PurePosixPath(source_config)
    if (
        source_pure.is_absolute()
        or "\\" in source_config
        or ".." in source_pure.parts
        or "." in source_pure.parts
        or not source_pure.parts
        or source_pure.parts[0] != "configs"
    ):
        raise ValueError("post-recovery aggregate source config path is unsafe")
    pilot_array_job_id = str(control["pilot_array_job_id"])
    if (
        control["pilot_phase"] != pilot_phase
        or control["ordered_seeds"] != list(ORDERED_SEEDS)
        or control["materialization_mode"] != "fresh"
        or control["recovery_outputs_reused"] is not False
        or control["all_tasks_terminal_completed_zero_exit"] is not True
        or control["optimizer_schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256
        or re.fullmatch(r"[1-9][0-9]*", pilot_array_job_id) is None
    ):
        raise ValueError("post-recovery aggregate control identity is invalid")

    overlay_file = _aggregate_evidence_file(
        aggregate_file,
        control["phase2_overlay"],
        name="post-recovery aggregate overlay evidence",
    )
    evidence_root = Path(f"{aggregate_file}.evidence").absolute()
    base_file = evidence_root.joinpath(*source_pure.parts)
    _require_real_file(base_file, name="post-recovery aggregate base evidence")
    try:
        base_file.relative_to(evidence_root)
    except ValueError as error:
        raise ValueError("post-recovery aggregate base evidence leaves its bundle") from error
    overlay_sha256 = _digest(
        control["phase2_overlay_sha256"],
        name="post-recovery aggregate overlay SHA256",
    )
    if _sha256_file(overlay_file) != overlay_sha256:
        raise ValueError("post-recovery aggregate overlay evidence SHA256 mismatch")
    submission_intent_file = _aggregate_evidence_file(
        aggregate_file,
        control["submission_intent"],
        name="post-recovery submission intent evidence",
    )
    submission_ledger_file = _aggregate_evidence_file(
        aggregate_file,
        control["submission_ledger"],
        name="post-recovery submission ledger evidence",
    )
    expected_intent_file = evidence_root / "submission-registry" / "intent.json"
    expected_ledger_file = evidence_root / "submission-registry" / "submission.json"
    if (
        submission_intent_file != expected_intent_file
        or submission_ledger_file != expected_ledger_file
    ):
        raise ValueError("post-recovery submission evidence does not use its locked bundle paths")
    submission_intent_sha256 = _digest(
        control["submission_intent_sha256"],
        name="post-recovery submission intent SHA256",
    )
    submission_ledger_sha256 = _digest(
        control["submission_ledger_sha256"],
        name="post-recovery submission ledger SHA256",
    )

    identity = _exact_mapping(
        aggregate.get("aggregation_identity"),
        name="post-recovery aggregate.aggregation_identity",
        keys={
            "schema_version",
            "aggregator_git_commit",
            "producer_git_commit",
            "image_sha256",
            "hf_inventory_sha256",
            "validator_source_sha256",
        },
    )
    verify_post_recovery_submission_evidence(
        submission_intent_file,
        submission_ledger_file,
        expected_pilot_phase=str(pilot_phase),
        expected_design_sha256=design_sha256,
        expected_base_config_hash=base_config_hash,
        expected_authorization_sha256=str(control["recovery_authorization_sha256"]),
        expected_optimizer_schedule_sha256=str(control["optimizer_schedule_sha256"]),
        expected_git_commit=str(identity["producer_git_commit"]),
        expected_image_sha256=str(identity["image_sha256"]),
        expected_inventory_sha256=str(identity["hf_inventory_sha256"]),
        expected_overlay_sha256=overlay_sha256,
        expected_base_sha256=_sha256_file(base_file),
        expected_overlay_repo_relative=str(control["phase2_overlay_repo_relative"]),
        expected_base_repo_relative=source_config,
        expected_array_job_id=pilot_array_job_id,
        expected_intent_sha256=submission_intent_sha256,
        expected_submission_sha256=submission_ledger_sha256,
        expected_project_root=os.fspath(aggregate_file.parent.parent),
    )
    expected = {
        "slurm_job_is_array": "false",
        "cluster": "hpc4",
        "account": "sigroup",
        "restart_count": "0",
        "pilot_array_job_id": pilot_array_job_id,
        "pilot_phase": str(pilot_phase),
        "phase2_design_sha256": design_sha256,
        "base_config_hash": base_config_hash,
        "recovery_authorization_sha256": str(control["recovery_authorization_sha256"]),
        "optimizer_schedule_sha256": str(control["optimizer_schedule_sha256"]),
        "submission_intent_sha256": submission_intent_sha256,
        "submission_ledger_sha256": submission_ledger_sha256,
        "pilot_terminal_evidence_sha256": str(control["pilot_terminal_evidence_sha256"]),
        "phase2_overlay_sha256": overlay_sha256,
        "phase2_base_sha256": _sha256_file(base_file),
        "aggregator_git_commit": str(identity["aggregator_git_commit"]),
        "producer_git_commit": str(identity["producer_git_commit"]),
        "image_sha256": str(identity["image_sha256"]),
        "hf_inventory_sha256": str(identity["hf_inventory_sha256"]),
        "aggregate_sha256": _sha256_file(aggregate_file),
    }
    for key, lengths in (
        ("phase2_design_sha256", frozenset({64})),
        ("base_config_hash", frozenset({64})),
        ("recovery_authorization_sha256", frozenset({64})),
        ("optimizer_schedule_sha256", frozenset({64})),
        ("submission_intent_sha256", frozenset({64})),
        ("submission_ledger_sha256", frozenset({64})),
        ("pilot_terminal_evidence_sha256", frozenset({64})),
        ("phase2_overlay_sha256", frozenset({64})),
        ("phase2_base_sha256", frozenset({64})),
        ("aggregator_git_commit", frozenset({40, 64})),
        ("producer_git_commit", frozenset({40, 64})),
        ("image_sha256", frozenset({64})),
        ("hf_inventory_sha256", frozenset({64})),
        ("aggregate_sha256", frozenset({64})),
    ):
        _digest(
            expected[key],
            name=f"post-recovery aggregate receipt {key}",
            lengths=lengths,
        )
    return {
        "aggregate_file": aggregate_file,
        "aggregate": aggregate,
        "expected": expected,
        "pilot_phase": pilot_phase,
        "phase2_design_sha256": design_sha256,
        "base_config_hash": base_config_hash,
        "overlay_relative": str(control["phase2_overlay_repo_relative"]),
        "base_relative": source_config,
    }


def _verify_post_recovery_aggregate_publication_receipt(
    context: Mapping[str, object],
) -> tuple[Path, dict[str, str]]:
    aggregate_file = context["aggregate_file"]
    if not isinstance(aggregate_file, Path):
        raise TypeError("post-recovery aggregate context path is invalid")
    owner_path = Path(f"{aggregate_file}.ATTEMPT").absolute()
    owner = parse_post_recovery_aggregate_publication_owner(owner_path)
    ready_evidence_path = (
        Path(f"{aggregate_file}.evidence").absolute() / "aggregation-attempt" / "READY"
    )
    _require_real_file(
        ready_evidence_path,
        name="post-recovery aggregate READY evidence",
    )
    ready_sha256 = _sha256_file(ready_evidence_path)
    if (
        owner["schema_version"] != POST_RECOVERY_AGGREGATE_PUBLICATION_OWNER_SCHEMA
        or owner["status"] != "CLAIMED"
        or re.fullmatch(r"[1-9][0-9]*", owner["attempt_slurm_job_id"]) is None
        or owner["attempt_ready_sha256"] != ready_sha256
        or owner["aggregate_sha256"] != context["expected"]["aggregate_sha256"]
        or not _valid_utc(owner["created_at_utc"])
    ):
        raise ValueError("post-recovery aggregate ATTEMPT receipt is invalid")
    ready = parse_post_recovery_aggregate_attempt_ready(ready_evidence_path)
    if (
        ready["slurm_job_id"] != owner["attempt_slurm_job_id"]
        or ready["aggregate_sha256"] != owner["aggregate_sha256"]
        or ready["final_output"] != os.fspath(aggregate_file)
        or ready["final_evidence_root"] != f"{aggregate_file}.evidence"
    ):
        raise ValueError("post-recovery aggregate READY evidence identity is invalid")
    submission = _verify_aggregate_submission_bundle(
        attempt_evidence=Path(f"{aggregate_file}.evidence").absolute(),
        ready=ready,
        aggregate_file=aggregate_file,
        project_root=aggregate_file.parent.parent,
        require_live_registry=False,
    )
    evidence_root = Path(f"{aggregate_file}.evidence").absolute()
    authority = _verify_aggregate_submission_authority_evidence(
        evidence_root,
        ready_sha256=ready_sha256,
        submission=submission,
        expected_captured_at_utc=owner["created_at_utc"],
        fresh_authority=None,
    )
    overlay_relative = context.get("overlay_relative")
    base_relative = context.get("base_relative")
    if not isinstance(overlay_relative, str) or not isinstance(base_relative, str):
        raise TypeError("post-recovery aggregate evidence paths are invalid")
    expected_files, expected_directories = _expected_aggregate_evidence_paths(
        overlay_relative=overlay_relative,
        base_relative=base_relative,
        submission=submission,
        published=True,
    )
    evidence_tree = _verify_exact_evidence_tree(
        evidence_root,
        expected_files=expected_files,
        expected_directories=expected_directories,
        name="published post-recovery aggregate evidence",
    )
    _verify_aggregate_evidence_claim(
        evidence_root,
        evidence_tree=evidence_tree,
        publication_owner_path=owner_path,
        publication_owner=owner,
        attempt_ready_sha256=ready_sha256,
        aggregate_sha256=owner["aggregate_sha256"],
    )
    evidence_manifest_sha256 = _evidence_tree_manifest_sha256(evidence_tree)
    publication_path = Path(f"{aggregate_file}.PUBLISHED").absolute()
    fields = parse_post_recovery_aggregate_publication_receipt(publication_path)
    expected = context["expected"]
    if not isinstance(expected, Mapping):
        raise TypeError("post-recovery aggregate receipt context is invalid")
    locked = {
        "schema_version": POST_RECOVERY_AGGREGATE_PUBLICATION_SCHEMA,
        "status": "PUBLISHED",
        **expected,
        "slurm_job_id": owner["attempt_slurm_job_id"],
        "aggregate_attempt_ready_sha256": ready_sha256,
        "aggregate_submission_authority_sha256": authority["sha256"],
        "aggregate_evidence_manifest_sha256": evidence_manifest_sha256,
        "aggregate_submission_intent_sha256": ready["aggregate_submission_intent_sha256"],
        "aggregate_submission_attempt_sha256": ready["aggregate_submission_attempt_sha256"],
        "aggregate_submission_attempt_index": ready["aggregate_submission_attempt_index"],
        "aggregate_submission_failure_chain_sha256": ready[
            "aggregate_submission_failure_chain_sha256"
        ],
    }
    for key, expected_value in locked.items():
        if fields.get(key) != expected_value:
            raise ValueError(f"post-recovery aggregate PUBLISHED receipt {key} is invalid")
    if not _valid_utc(fields["created_at_utc"]):
        raise ValueError("post-recovery aggregate PUBLISHED timestamp is invalid")
    return publication_path, fields


def post_recovery_aggregate_sacct_command(slurm_job_id: str) -> tuple[str, ...]:
    """Return the locked HPC4 query for one CPU aggregation allocation."""

    if re.fullmatch(r"[1-9][0-9]*", slurm_job_id) is None:
        raise ValueError("aggregation Slurm job ID must be a positive decimal integer")
    return (
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        slurm_job_id,
        f"--format={','.join(_SACCT_FORMAT_FIELDS)}",
    )


def _parse_post_recovery_aggregate_sacct_raw(
    raw: bytes,
    *,
    expected_job_id: str,
    expected_partition: str,
) -> dict[str, object]:
    if not raw or len(raw) > 16 * 1024 or not raw.endswith(b"\n"):
        raise ValueError(
            "raw aggregation sacct bytes must be non-empty, bounded, and newline-terminated"
        )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("raw aggregation sacct bytes must be UTF-8") from error
    if len(lines) != 1 or not lines[0]:
        raise ValueError("raw aggregation sacct evidence must contain exactly one row")
    fields = lines[0].split("|")
    if len(fields) != len(_SACCT_FIELDS):
        raise ValueError("raw aggregation sacct row lacks the locked twelve columns")
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
        req_tres,
        alloc_tres,
    ) = fields
    if (
        job_id != expected_job_id
        or job_id_raw != expected_job_id
        or state != "COMPLETED"
        or exit_code != "0:0"
        or derived_exit_code != "0:0"
        or cluster != "hpc4"
        or account != "sigroup"
        or partition != expected_partition
        or partition not in {"amd", "intel"}
        or n_nodes != "1"
        or n_cpus != "4"
        or req_tres != _EXPECTED_AGGREGATE_REQ_TRES
        or alloc_tres != _EXPECTED_AGGREGATE_ALLOC_TRES
    ):
        raise ValueError("raw aggregation sacct row is not the exact successful HPC4 allocation")
    return {
        "job_id": job_id,
        "job_id_raw": job_id_raw,
        "state": state,
        "exit_code": exit_code,
        "derived_exit_code": derived_exit_code,
        "cluster": cluster,
        "account": account,
        "partition": partition,
        "n_nodes": 1,
        "n_cpus": 4,
        "req_tres": req_tres,
        "alloc_tres": alloc_tres,
    }


def verify_post_recovery_aggregate_terminal_evidence(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_aggregate_path: str | os.PathLike[str],
    expected_aggregate_sha256: str,
    expected_publication_path: str | os.PathLike[str],
    expected_publication_sha256: str,
    expected_slurm_job_id: str,
    expected_partition: str,
) -> dict[str, object]:
    """Verify one exact post-job CPU aggregation allocation capture."""

    evidence_path = Path(path).absolute()
    aggregate_path = Path(expected_aggregate_path).absolute()
    publication_path = Path(expected_publication_path).absolute()
    expected_path = Path(f"{aggregate_path}.TERMINAL.json").absolute()
    if evidence_path != expected_path:
        raise ValueError(
            "post-recovery aggregate terminal evidence must be adjacent to its aggregate"
        )
    _require_real_file(evidence_path, name="post-recovery aggregate terminal evidence")
    _require_real_file(aggregate_path, name="post-recovery aggregate")
    _require_real_file(
        publication_path,
        name="post-recovery aggregate PUBLISHED receipt",
    )
    evidence_digest = _digest(
        expected_sha256,
        name="post-recovery aggregate terminal evidence SHA256",
    )
    aggregate_digest = _digest(
        expected_aggregate_sha256,
        name="post-recovery aggregate SHA256",
    )
    publication_digest = _digest(
        expected_publication_sha256,
        name="post-recovery aggregate PUBLISHED receipt SHA256",
    )
    raw_json = evidence_path.read_bytes()
    if _sha256_bytes(raw_json) != evidence_digest:
        raise ValueError("post-recovery aggregate terminal evidence SHA256 mismatch")
    try:
        value = json.loads(
            raw_json.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("post-recovery aggregate terminal evidence is not strict JSON") from error
    if raw_json != _canonical_json(value):
        raise ValueError("post-recovery aggregate terminal evidence must use canonical JSON bytes")
    evidence = _exact_mapping(
        value,
        name="post-recovery aggregate terminal evidence",
        keys={
            "schema_version",
            "captured_at_utc",
            "query",
            "aggregation_slurm_job_id",
            "row",
            "aggregate",
            "publication_receipt",
            "raw_sacct",
        },
    )
    if (
        evidence["schema_version"] != POST_RECOVERY_AGGREGATE_TERMINAL_SCHEMA
        or evidence["query"] != list(post_recovery_aggregate_sacct_command(expected_slurm_job_id))
        or evidence["aggregation_slurm_job_id"] != expected_slurm_job_id
        or not _valid_utc(evidence["captured_at_utc"])
    ):
        raise ValueError("post-recovery aggregate terminal evidence identity is invalid")
    aggregate_binding = _exact_mapping(
        evidence["aggregate"],
        name="post-recovery aggregate terminal evidence.aggregate",
        keys={"filename", "sha256", "size_bytes"},
    )
    publication_binding = _exact_mapping(
        evidence["publication_receipt"],
        name="post-recovery aggregate terminal evidence.publication_receipt",
        keys={"filename", "sha256", "size_bytes"},
    )
    raw_binding = _exact_mapping(
        evidence["raw_sacct"],
        name="post-recovery aggregate terminal evidence.raw_sacct",
        keys={"filename", "sha256", "size_bytes"},
    )
    raw_path = Path(f"{aggregate_path}.TERMINAL.sacct.psv").absolute()
    expected_bindings = (
        (
            aggregate_binding,
            aggregate_path,
            aggregate_digest,
            "post-recovery aggregate",
        ),
        (
            publication_binding,
            publication_path,
            publication_digest,
            "post-recovery aggregate PUBLISHED receipt",
        ),
    )
    for binding, bound_path, expected_digest, name in expected_bindings:
        if (
            binding["filename"] != bound_path.name
            or binding["sha256"] != expected_digest
            or binding["size_bytes"] != bound_path.stat().st_size
            or _sha256_file(bound_path) != expected_digest
        ):
            raise ValueError(f"terminal evidence does not bind its {name}")
    _require_real_file(raw_path, name="raw aggregation sacct evidence")
    raw = raw_path.read_bytes()
    if (
        raw_binding["filename"] != raw_path.name
        or raw_binding["sha256"] != _sha256_bytes(raw)
        or raw_binding["size_bytes"] != len(raw)
    ):
        raise ValueError("terminal evidence does not bind its raw aggregation sacct bytes")
    row = _parse_post_recovery_aggregate_sacct_raw(
        raw,
        expected_job_id=expected_slurm_job_id,
        expected_partition=expected_partition,
    )
    parsed_row = _exact_mapping(
        evidence["row"],
        name="post-recovery aggregate terminal evidence.row",
        keys=_AGGREGATE_TERMINAL_ROW_KEYS,
    )
    if parsed_row != row:
        raise ValueError("post-recovery aggregate terminal row differs from the raw sacct bytes")
    return dict(evidence)


def _aggregate_receipt_bytes(fields: Mapping[str, str]) -> bytes:
    if any(
        not key
        or not value
        or "=" in key
        or "\n" in key
        or "\r" in key
        or "\n" in value
        or "\r" in value
        for key, value in fields.items()
    ):
        raise ValueError("aggregate receipt contains an unsafe key or value")
    return "".join(f"{key}={fields[key]}\n" for key in sorted(fields)).encode("utf-8")


def _real_directory(path: Path, *, name: str) -> None:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is missing or inaccessible") from error
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink() or resolved != path.absolute():
        raise ValueError(f"{name} must be a canonical non-symlink directory")


def _directory_manifest(
    root: Path,
    *,
    name: str,
) -> dict[str, tuple[int, str]]:
    """Compatibility file manifest derived from the exact tree manifest."""

    tree = _directory_tree_manifest(root, name=name)
    files = tree["files"]
    if not isinstance(files, Mapping):
        raise TypeError(f"{name} tree manifest files are invalid")
    return {
        relative: (
            int(binding["size_bytes"]),
            str(binding["sha256"]),
        )
        for relative, binding in files.items()
        if isinstance(binding, Mapping)
    }


def _directory_tree_manifest(
    root: Path,
    *,
    name: str,
) -> dict[str, object]:
    """Bind every directory and regular file, including hidden and empty entries."""

    _real_directory(root, name=name)
    directories: list[str] = []
    files: dict[str, dict[str, object]] = {}

    def walk_error(error: OSError) -> None:
        raise ValueError(f"{name} cannot be traversed exactly") from error

    for directory_text, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=walk_error,
        followlinks=False,
    ):
        directory = Path(directory_text).absolute()
        _real_directory(directory, name=f"{name} directory")
        if directory != root:
            relative_directory = directory.relative_to(root).as_posix()
            if (
                not relative_directory
                or relative_directory in directories
                or "\\" in relative_directory
                or "\n" in relative_directory
                or "\r" in relative_directory
            ):
                raise ValueError(f"{name} contains an unsafe directory name")
            directories.append(relative_directory)
        directory_names.sort()
        file_names.sort()
        for child_name in directory_names:
            child = directory / child_name
            try:
                metadata = child.lstat()
            except OSError as error:
                raise ValueError(f"{name} directory entry is inaccessible") from error
            if not stat.S_ISDIR(metadata.st_mode) or child.is_symlink():
                raise ValueError(f"{name} contains a linked or special directory")
        for child_name in file_names:
            child = directory / child_name
            _require_real_file(child, name=f"{name} file")
            relative = child.relative_to(root).as_posix()
            if (
                not relative
                or relative in files
                or "\\" in relative
                or "\n" in relative
                or "\r" in relative
            ):
                raise ValueError(f"{name} contains an unsafe file name")
            files[relative] = {
                "size_bytes": child.stat().st_size,
                "sha256": _sha256_file(child),
            }
    return {
        "schema_version": "prorm-exact-evidence-tree-manifest/v1",
        "directories": sorted(directories),
        "files": {relative: files[relative] for relative in sorted(files)},
    }


def _evidence_tree_manifest_sha256(tree: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_json(dict(tree)))


def _relative_parent_directories(relative_paths: set[str]) -> set[str]:
    directories: set[str] = set()
    for relative in relative_paths:
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or "\\" in relative
            or "." in pure.parts
            or ".." in pure.parts
        ):
            raise ValueError("aggregate evidence relative path is unsafe")
        parent = pure.parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _expected_aggregate_evidence_paths(
    *,
    overlay_relative: str,
    base_relative: str,
    submission: Mapping[str, object],
    published: bool,
) -> tuple[set[str], set[str]]:
    attempt_index = int(submission["attempt_index"])
    previous_rows = submission["previous_rows"]
    if attempt_index < 1 or not isinstance(previous_rows, Mapping):
        raise ValueError("aggregate submission chain is invalid")
    files = {
        overlay_relative,
        base_relative,
        "submission-registry/intent.json",
        "submission-registry/submission.json",
        "aggregate-submission/intent.json",
        "aggregate-submission/attempt.json",
        "aggregate-submission/failure-chain.json",
        (f"aggregate-submission/{POST_RECOVERY_AGGREGATE_SCRIPT_EVIDENCE}"),
        *{
            f"aggregate-submission/attempts/attempt-{index:04d}.json"
            for index in range(1, attempt_index + 1)
        },
        *{
            "aggregate-submission/"
            f"{POST_RECOVERY_AGGREGATE_CONTROLLER_READBACKS}/"
            f"attempt-{index:04d}.sbatch"
            for index in range(1, attempt_index + 1)
        },
        *{f"aggregate-submission/failures/job-{job_id}.json" for job_id in previous_rows},
        *{f"aggregate-submission/failures/job-{job_id}.sacct.psv" for job_id in previous_rows},
    }
    directories = _relative_parent_directories(files)
    directories.add("aggregate-submission/failures")
    if published:
        files.update(
            {
                "aggregation-attempt/READY",
                "aggregation-attempt/AUTHORITY.json",
                "aggregation-attempt/AUTHORITY.squeue.txt",
                "aggregation-attempt/AUTHORITY.sacct.psv",
                (f"aggregation-attempt/{POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM}"),
            }
        )
        directories.update(_relative_parent_directories(files))
        directories.add("aggregation-attempt")
    return files, directories


def _verify_exact_evidence_tree(
    root: Path,
    *,
    expected_files: set[str],
    expected_directories: set[str],
    name: str,
) -> dict[str, object]:
    tree = _directory_tree_manifest(root, name=name)
    files = tree["files"]
    directories = tree["directories"]
    if (
        not isinstance(files, Mapping)
        or not isinstance(directories, list)
        or set(files) != expected_files
        or set(directories) != expected_directories
        or len(directories) != len(expected_directories)
    ):
        raise ValueError(f"{name} contains an unexpected file or directory")
    return tree


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
    except PermissionError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []

    def walk_error(error: OSError) -> None:
        raise OSError("published aggregate evidence cannot be traversed for fsync") from error

    for directory_text, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=walk_error,
        followlinks=False,
    ):
        directory = Path(directory_text)
        directories.append(directory)
        for child_name in sorted(file_names):
            child = directory / child_name
            _require_real_file(child, name="published aggregate evidence file")
            open_flags = (
                os.O_RDWR if os.name == "nt" else os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(
                child,
                open_flags,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for child_name in directory_names:
            child = directory / child_name
            if child.is_symlink():
                raise ValueError("published aggregate evidence contains a symlink")
    for directory in reversed(directories):
        _fsync_directory(directory)


def _ensure_evidence_directory(path: Path, *, name: str) -> bool:
    """Atomically claim one directory name or verify the existing real directory."""

    created = False
    try:
        path.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        pass
    _real_directory(path, name=name)
    if created:
        _fsync_directory(path.parent)
    return created


def _install_evidence_file_noreplace(
    source: Path,
    destination: Path,
    *,
    expected: Mapping[str, object],
    name: str,
) -> bool:
    """Install one completely staged inode with hard-link create-if-absent."""

    _require_real_file(source, name=f"{name} staged source")
    _real_directory(destination.parent, name=f"{name} destination parent")
    if source.stat().st_size != _integer(
        expected.get("size_bytes"),
        name=f"{name} expected size",
    ) or _sha256_file(source) != _digest(
        expected.get("sha256"),
        name=f"{name} expected SHA256",
    ):
        raise ValueError(f"{name} staged source differs from its exact manifest")
    installed = False
    try:
        os.link(
            source,
            destination,
            src_dir_fd=None,
            dst_dir_fd=None,
            follow_symlinks=False,
        )
        installed = True
    except FileExistsError:
        pass
    _require_real_file(destination, name=name)
    if (
        destination.stat().st_size != expected["size_bytes"]
        or _sha256_file(destination) != expected["sha256"]
        or destination.read_bytes() != source.read_bytes()
    ):
        raise ValueError(f"existing {name} differs; refusing to overwrite it")
    if installed:
        _fsync_directory(destination.parent)
    return installed


def _validate_partial_evidence_tree(
    root: Path,
    *,
    expected: Mapping[str, object],
    claim_relative: str,
) -> dict[str, object]:
    """Accept only a durable exact prefix of the claimed evidence tree."""

    observed = _directory_tree_manifest(
        root,
        name="partial post-recovery aggregate evidence",
    )
    observed_directories = observed.get("directories")
    observed_files = observed.get("files")
    expected_directories = expected.get("directories")
    expected_files = expected.get("files")
    if (
        not isinstance(observed_directories, list)
        or not isinstance(observed_files, Mapping)
        or not isinstance(expected_directories, list)
        or not isinstance(expected_files, Mapping)
        or not set(observed_directories).issubset(expected_directories)
        or not set(observed_files).issubset(expected_files)
        or any(observed_files[path] != expected_files[path] for path in observed_files)
    ):
        raise ValueError("partial aggregate evidence is not an exact expected prefix")
    if claim_relative not in observed_files and (
        observed_files or set(observed_directories) - {"aggregation-attempt"}
    ):
        raise ValueError("partial aggregate evidence contains payload before its claim")
    return observed


def _publish_exact_file(path: Path, raw: bytes, *, name: str) -> None:
    if path.exists() or path.is_symlink():
        _require_real_file(path, name=name)
        if path.read_bytes() != raw:
            raise ValueError(f"existing {name} differs from the selected attempt")
        return
    _write_exclusive(path, raw, name=name)
    _require_real_file(path, name=name)
    if path.read_bytes() != raw:
        raise ValueError(f"published {name} differs from the selected attempt")


def _expected_published_evidence_manifest(
    attempt: Mapping[str, object],
    *,
    authority_artifacts: Mapping[str, bytes] | None = None,
    evidence_claim: bytes | None = None,
) -> dict[str, object]:
    attempt_evidence = attempt["attempt_evidence"]
    ready_path = attempt["ready_path"]
    if not isinstance(attempt_evidence, Path) or not isinstance(ready_path, Path):
        raise TypeError("aggregate attempt evidence paths are invalid")
    tree = _directory_tree_manifest(
        attempt_evidence,
        name="post-recovery aggregate attempt evidence",
    )
    raw_directories = tree["directories"]
    raw_files = tree["files"]
    if not isinstance(raw_directories, list) or not isinstance(raw_files, Mapping):
        raise TypeError("aggregate attempt evidence tree is invalid")
    directories = set(raw_directories)
    files = {
        relative: dict(binding)
        for relative, binding in raw_files.items()
        if isinstance(binding, Mapping)
    }
    ready_relative = "aggregation-attempt/READY"
    if ready_relative in files or any(
        relative == "aggregation-attempt" or relative.startswith("aggregation-attempt/")
        for relative in directories | set(files)
    ):
        raise ValueError("aggregate attempt evidence illegally occupies the terminalizer namespace")
    directories.add("aggregation-attempt")
    files[ready_relative] = {
        "size_bytes": ready_path.stat().st_size,
        "sha256": _sha256_file(ready_path),
    }
    for filename, raw in (authority_artifacts or {}).items():
        relative = f"aggregation-attempt/{filename}"
        if "/" in filename or "\\" in filename or relative in files or not isinstance(raw, bytes):
            raise ValueError("aggregate authority artifact identity is unsafe")
        files[relative] = {
            "size_bytes": len(raw),
            "sha256": _sha256_bytes(raw),
        }
    if evidence_claim is not None:
        if not isinstance(evidence_claim, bytes):
            raise TypeError("aggregate evidence claim must contain bytes")
        claim_relative = f"aggregation-attempt/{POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM}"
        if claim_relative in files:
            raise ValueError("aggregate evidence claim path is already occupied")
        files[claim_relative] = {
            "size_bytes": len(evidence_claim),
            "sha256": _sha256_bytes(evidence_claim),
        }
    return {
        "schema_version": "prorm-exact-evidence-tree-manifest/v1",
        "directories": sorted(directories),
        "files": {relative: files[relative] for relative in sorted(files)},
    }


def _aggregate_evidence_claim_bytes(
    *,
    publication_owner_receipt_sha256: str,
    attempt_slurm_job_id: str,
    attempt_ready_sha256: str,
    aggregate_sha256: str,
    payload_manifest: Mapping[str, object],
    created_at_utc: str,
) -> bytes:
    if re.fullmatch(r"[1-9][0-9]*", attempt_slurm_job_id) is None or not _valid_utc(created_at_utc):
        raise ValueError("aggregate evidence claim identity is invalid")
    payload = {
        "schema_version": POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM_SCHEMA,
        "status": "CLAIMED",
        "publication_owner_receipt_sha256": _digest(
            publication_owner_receipt_sha256,
            name="aggregate publication owner receipt SHA256",
        ),
        "attempt_slurm_job_id": attempt_slurm_job_id,
        "attempt_ready_sha256": _digest(
            attempt_ready_sha256,
            name="aggregate attempt READY SHA256",
        ),
        "aggregate_sha256": _digest(
            aggregate_sha256,
            name="aggregate evidence claim aggregate SHA256",
        ),
        "payload_exact_tree_manifest_sha256": _evidence_tree_manifest_sha256(payload_manifest),
        "created_at_utc": created_at_utc,
    }
    return _canonical_json(payload)


def _evidence_tree_without_claim(tree: Mapping[str, object]) -> dict[str, object]:
    directories = tree.get("directories")
    files = tree.get("files")
    if not isinstance(directories, list) or not isinstance(files, Mapping):
        raise TypeError("aggregate evidence tree manifest is invalid")
    claim_relative = f"aggregation-attempt/{POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM}"
    if claim_relative not in files:
        raise ValueError("aggregate evidence tree lacks its publication claim")
    payload_files = {
        str(relative): dict(binding)
        for relative, binding in files.items()
        if relative != claim_relative and isinstance(binding, Mapping)
    }
    if len(payload_files) != len(files) - 1:
        raise TypeError("aggregate evidence tree contains an invalid file binding")
    return {
        "schema_version": "prorm-exact-evidence-tree-manifest/v1",
        "directories": list(directories),
        "files": {relative: payload_files[relative] for relative in sorted(payload_files)},
    }


def _verify_aggregate_evidence_claim(
    evidence_root: Path,
    *,
    evidence_tree: Mapping[str, object],
    publication_owner_path: Path,
    publication_owner: Mapping[str, str],
    attempt_ready_sha256: str,
    aggregate_sha256: str,
) -> dict[str, object]:
    claim_path = evidence_root / "aggregation-attempt" / POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM
    claim = _exact_mapping(
        _strict_canonical_json_file(
            claim_path,
            name="post-recovery aggregate evidence claim",
        ),
        name="post-recovery aggregate evidence claim",
        keys={
            "schema_version",
            "status",
            "publication_owner_receipt_sha256",
            "attempt_slurm_job_id",
            "attempt_ready_sha256",
            "aggregate_sha256",
            "payload_exact_tree_manifest_sha256",
            "created_at_utc",
        },
    )
    expected = _aggregate_evidence_claim_bytes(
        publication_owner_receipt_sha256=_sha256_file(publication_owner_path),
        attempt_slurm_job_id=publication_owner["attempt_slurm_job_id"],
        attempt_ready_sha256=attempt_ready_sha256,
        aggregate_sha256=aggregate_sha256,
        payload_manifest=_evidence_tree_without_claim(evidence_tree),
        created_at_utc=publication_owner["created_at_utc"],
    )
    if (
        claim_path.read_bytes() != expected
        or claim["schema_version"] != POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM_SCHEMA
        or claim["status"] != "CLAIMED"
    ):
        raise ValueError("post-recovery aggregate evidence claim differs")
    return dict(claim)


def _publish_attempt_evidence(
    attempt: Mapping[str, object],
    final_evidence: Path,
    *,
    authority: Mapping[str, object],
    publication_owner: Mapping[str, str],
    publication_owner_path: Path,
) -> dict[str, object]:
    if (
        publication_owner["attempt_slurm_job_id"] != attempt["ready"]["slurm_job_id"]
        or publication_owner["attempt_ready_sha256"] != attempt["ready_sha256"]
        or publication_owner["aggregate_sha256"] != attempt["aggregate_sha256"]
        or not _valid_utc(publication_owner["created_at_utc"])
    ):
        raise ValueError("aggregate evidence publication owner differs from the attempt")
    _require_real_file(
        publication_owner_path,
        name="aggregate evidence publication owner receipt",
    )
    authority_artifacts = _aggregate_submission_authority_artifacts(
        authority,
        attempt_ready_sha256=str(attempt["ready_sha256"]),
        captured_at_utc=publication_owner["created_at_utc"],
    )
    payload_manifest = _expected_published_evidence_manifest(
        attempt,
        authority_artifacts=authority_artifacts,
    )
    claim_raw = _aggregate_evidence_claim_bytes(
        publication_owner_receipt_sha256=_sha256_file(publication_owner_path),
        attempt_slurm_job_id=publication_owner["attempt_slurm_job_id"],
        attempt_ready_sha256=publication_owner["attempt_ready_sha256"],
        aggregate_sha256=publication_owner["aggregate_sha256"],
        payload_manifest=payload_manifest,
        created_at_utc=publication_owner["created_at_utc"],
    )
    expected_manifest = _expected_published_evidence_manifest(
        attempt,
        authority_artifacts=authority_artifacts,
        evidence_claim=claim_raw,
    )
    expected_directories = expected_manifest["directories"]
    expected_files = expected_manifest["files"]
    if not isinstance(expected_directories, list) or not isinstance(expected_files, Mapping):
        raise TypeError("aggregate expected evidence manifest is invalid")
    parent = final_evidence.parent
    _real_directory(parent, name="aggregate publication parent")
    attempt_evidence = attempt["attempt_evidence"]
    ready_path = attempt["ready_path"]
    if not isinstance(attempt_evidence, Path) or not isinstance(ready_path, Path):
        raise TypeError("aggregate attempt evidence paths are invalid")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{final_evidence.name}.publishing-",
            dir=parent,
        )
    ).absolute()
    try:
        shutil.copytree(
            attempt_evidence,
            temporary,
            copy_function=shutil.copyfile,
            dirs_exist_ok=True,
            symlinks=False,
        )
        ready_parent = temporary / "aggregation-attempt"
        ready_parent.mkdir(mode=0o750)
        _write_exclusive(
            ready_parent / "READY",
            ready_path.read_bytes(),
            name="published post-recovery aggregate READY evidence",
        )
        for filename, raw in authority_artifacts.items():
            _write_exclusive(
                ready_parent / filename,
                raw,
                name=f"published aggregate authority {filename}",
            )
        claim_relative = f"aggregation-attempt/{POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM}"
        _write_exclusive(
            temporary.joinpath(*PurePosixPath(claim_relative).parts),
            claim_raw,
            name="post-recovery aggregate evidence claim",
        )
        for directory_text, _, filenames in os.walk(
            temporary,
            topdown=True,
            followlinks=False,
        ):
            directory = Path(directory_text)
            for filename in filenames:
                if os.name != "nt":
                    os.chmod(directory / filename, 0o440)
        _fsync_tree(temporary)
        if (
            _expected_published_evidence_manifest(
                attempt,
                authority_artifacts=authority_artifacts,
                evidence_claim=claim_raw,
            )
            != expected_manifest
            or _directory_tree_manifest(
                temporary,
                name="temporary post-recovery aggregate evidence",
            )
            != expected_manifest
        ):
            raise ValueError("aggregate attempt evidence changed during publication")

        _ensure_evidence_directory(
            final_evidence,
            name="claimed post-recovery aggregate evidence root",
        )
        _validate_partial_evidence_tree(
            final_evidence,
            expected=expected_manifest,
            claim_relative=claim_relative,
        )
        claim_parent = final_evidence / "aggregation-attempt"
        _ensure_evidence_directory(
            claim_parent,
            name="aggregate evidence claim directory",
        )
        _install_evidence_file_noreplace(
            temporary.joinpath(*PurePosixPath(claim_relative).parts),
            final_evidence.joinpath(*PurePosixPath(claim_relative).parts),
            expected=expected_files[claim_relative],
            name="post-recovery aggregate evidence claim",
        )
        _validate_partial_evidence_tree(
            final_evidence,
            expected=expected_manifest,
            claim_relative=claim_relative,
        )

        for relative in sorted(
            expected_directories,
            key=lambda value: (len(PurePosixPath(value).parts), value),
        ):
            _ensure_evidence_directory(
                final_evidence.joinpath(*PurePosixPath(relative).parts),
                name=f"aggregate evidence directory {relative}",
            )
        for relative in sorted(expected_files):
            if relative == claim_relative:
                continue
            _install_evidence_file_noreplace(
                temporary.joinpath(*PurePosixPath(relative).parts),
                final_evidence.joinpath(*PurePosixPath(relative).parts),
                expected=expected_files[relative],
                name=f"aggregate evidence file {relative}",
            )
        _fsync_tree(final_evidence)
        _fsync_directory(parent)
        if (
            _directory_tree_manifest(
                final_evidence,
                name="published post-recovery aggregate evidence",
            )
            != expected_manifest
        ):
            raise ValueError("published aggregate evidence failed final verification")
    except BaseException:
        # The complete sibling source tree is retained as forensic evidence.
        # A visible final tree is accepted only as an exact claim-bound prefix
        # and never authorizes consumption without the later PUBLISHED receipt.
        raise
    with suppress(OSError):
        shutil.rmtree(temporary)
    submission = attempt["aggregate_submission"]
    if not isinstance(submission, Mapping):
        raise TypeError("aggregate CPU submission context is invalid")
    return _verify_aggregate_submission_authority_evidence(
        final_evidence,
        ready_sha256=str(attempt["ready_sha256"]),
        submission=submission,
        expected_captured_at_utc=publication_owner["created_at_utc"],
        fresh_authority=authority,
    )


def _verify_publication_owner_for_attempt(
    aggregate_file: Path,
    attempt: Mapping[str, object],
) -> dict[str, str]:
    owner_path = Path(f"{aggregate_file}.ATTEMPT").absolute()
    owner = parse_post_recovery_aggregate_publication_owner(owner_path)
    if (
        owner["schema_version"] != POST_RECOVERY_AGGREGATE_PUBLICATION_OWNER_SCHEMA
        or owner["status"] != "CLAIMED"
        or owner["attempt_slurm_job_id"] != attempt["ready"]["slurm_job_id"]
        or owner["attempt_ready_sha256"] != attempt["ready_sha256"]
        or owner["aggregate_sha256"] != attempt["aggregate_sha256"]
        or not _valid_utc(owner["created_at_utc"])
    ):
        raise ValueError("final aggregate namespace is owned by a different or invalid attempt")
    return owner


def _verify_completed_aggregate_for_attempt(
    aggregate_file: Path,
    *,
    attempt_job_id: str,
) -> dict[str, object]:
    """Replay a completed publication without consulting staging or live state."""

    owner_path = Path(f"{aggregate_file}.ATTEMPT").absolute()
    owner = parse_post_recovery_aggregate_publication_owner(owner_path)
    if owner["attempt_slurm_job_id"] != attempt_job_id:
        raise ValueError("completed aggregate belongs to a different CPU attempt")
    return verify_post_recovery_aggregate_success_receipt(aggregate_file)


def _query_post_recovery_aggregate_sacct(
    *,
    attempt_job_id: str,
    expected_partition: str,
) -> tuple[tuple[str, ...], bytes]:
    command = post_recovery_aggregate_sacct_command(attempt_job_id)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not execute the locked aggregation sacct query") from error
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("the locked aggregation sacct query failed or emitted stderr")
    raw = completed.stdout
    _parse_post_recovery_aggregate_sacct_raw(
        raw,
        expected_job_id=attempt_job_id,
        expected_partition=expected_partition,
    )
    return command, raw


def _capture_post_recovery_aggregate_terminal_evidence_locked(
    aggregate_path: str | os.PathLike[str],
    *,
    attempt_job_id: str,
    queried_command: tuple[str, ...],
    queried_raw: bytes,
) -> dict[str, object]:
    """Claim, publish, and terminalize one already successful CPU attempt."""

    aggregate_file = Path(aggregate_path).absolute()
    success_path = Path(f"{aggregate_file}.SUCCESS").absolute()
    if success_path.exists() or success_path.is_symlink():
        return _verify_completed_aggregate_for_attempt(
            aggregate_file,
            attempt_job_id=attempt_job_id,
        )
    attempt = verify_post_recovery_aggregate_attempt_ready(
        aggregate_file,
        attempt_job_id=attempt_job_id,
    )
    aggregate_submission = attempt["aggregate_submission"]
    if not isinstance(aggregate_submission, Mapping):
        raise TypeError("aggregate CPU submission context is invalid")
    authority = _verify_aggregate_submission_scheduler_authority(aggregate_submission)
    if attempt["ready"]["partition"] not in {
        "amd",
        "intel",
    } or queried_command != post_recovery_aggregate_sacct_command(attempt_job_id):
        raise ValueError("aggregate attempt terminal query identity is invalid")
    _parse_post_recovery_aggregate_sacct_raw(
        queried_raw,
        expected_job_id=attempt_job_id,
        expected_partition=attempt["ready"]["partition"],
    )

    owner_path = Path(f"{aggregate_file}.ATTEMPT").absolute()
    final_evidence = Path(f"{aggregate_file}.evidence").absolute()
    publication_path = Path(f"{aggregate_file}.PUBLISHED").absolute()
    terminal_path = Path(f"{aggregate_file}.TERMINAL.json").absolute()
    raw_path = Path(f"{aggregate_file}.TERMINAL.sacct.psv").absolute()

    phases = {
        "owner": owner_path.exists() or owner_path.is_symlink(),
        "evidence": final_evidence.exists() or final_evidence.is_symlink(),
        "aggregate": aggregate_file.exists() or aggregate_file.is_symlink(),
        "publication": publication_path.exists() or publication_path.is_symlink(),
        "raw": raw_path.exists() or raw_path.is_symlink(),
        "terminal": terminal_path.exists() or terminal_path.is_symlink(),
        "success": success_path.exists() or success_path.is_symlink(),
    }
    if not phases["owner"] and any(phases[name] for name in phases if name != "owner"):
        raise ValueError("unclaimed final aggregate namespace is already occupied")
    if (
        (phases["aggregate"] and not phases["evidence"])
        or (phases["publication"] and not phases["aggregate"])
        or (phases["raw"] and not phases["publication"])
        or (phases["terminal"] and not phases["raw"])
        or (phases["success"] and not phases["terminal"])
    ):
        raise ValueError("final aggregate publication phases are out of order")

    if phases["owner"]:
        owner = _verify_publication_owner_for_attempt(aggregate_file, attempt)
    else:
        owner = {
            "schema_version": POST_RECOVERY_AGGREGATE_PUBLICATION_OWNER_SCHEMA,
            "status": "CLAIMED",
            "attempt_slurm_job_id": attempt_job_id,
            "attempt_ready_sha256": str(attempt["ready_sha256"]),
            "aggregate_sha256": str(attempt["aggregate_sha256"]),
            "created_at_utc": _utc_now(),
        }
        _write_exclusive(
            owner_path,
            _aggregate_receipt_bytes(owner),
            name="post-recovery aggregate ATTEMPT receipt",
        )
        owner = _verify_publication_owner_for_attempt(aggregate_file, attempt)

    authority_evidence = _publish_attempt_evidence(
        attempt,
        final_evidence,
        authority=authority,
        publication_owner=owner,
        publication_owner_path=owner_path,
    )
    evidence_manifest_sha256 = _evidence_tree_manifest_sha256(
        _directory_tree_manifest(
            final_evidence,
            name="published post-recovery aggregate evidence",
        )
    )
    attempt_aggregate = attempt["attempt_aggregate"]
    if not isinstance(attempt_aggregate, Path):
        raise TypeError("aggregate attempt staged aggregate path is invalid")
    _publish_exact_file(
        aggregate_file,
        attempt_aggregate.read_bytes(),
        name="post-recovery aggregate",
    )
    context = _post_recovery_aggregate_context(aggregate_file)
    expected = context["expected"]
    if not isinstance(expected, Mapping):
        raise TypeError("post-recovery aggregate receipt context is invalid")
    if (
        expected["aggregate_sha256"] != attempt["aggregate_sha256"]
        or expected["phase2_design_sha256"] != attempt["ready"]["phase2_design_sha256"]
    ):
        raise ValueError("published aggregate differs from the selected attempt")
    if publication_path.exists() or publication_path.is_symlink():
        publication_path, publication = _verify_post_recovery_aggregate_publication_receipt(context)
    else:
        publication_fields = {
            "schema_version": POST_RECOVERY_AGGREGATE_PUBLICATION_SCHEMA,
            "status": "PUBLISHED",
            **expected,
            "slurm_job_id": attempt_job_id,
            "partition": str(attempt["ready"]["partition"]),
            "aggregate_attempt_ready_sha256": str(attempt["ready_sha256"]),
            "aggregate_submission_authority_sha256": str(authority_evidence["sha256"]),
            "aggregate_evidence_manifest_sha256": evidence_manifest_sha256,
            "aggregate_submission_intent_sha256": str(
                attempt["ready"]["aggregate_submission_intent_sha256"]
            ),
            "aggregate_submission_attempt_sha256": str(
                attempt["ready"]["aggregate_submission_attempt_sha256"]
            ),
            "aggregate_submission_attempt_index": str(
                attempt["ready"]["aggregate_submission_attempt_index"]
            ),
            "aggregate_submission_failure_chain_sha256": str(
                attempt["ready"]["aggregate_submission_failure_chain_sha256"]
            ),
            "created_at_utc": _utc_now(),
        }
        _write_exclusive(
            publication_path,
            _aggregate_receipt_bytes(publication_fields),
            name="post-recovery aggregate PUBLISHED receipt",
        )
        publication_path, publication = _verify_post_recovery_aggregate_publication_receipt(context)

    terminal_present = terminal_path.exists() or terminal_path.is_symlink()
    raw_present = raw_path.exists() or raw_path.is_symlink()
    publication_sha256 = _sha256_file(publication_path)
    aggregate_sha256 = _sha256_file(aggregate_file)
    if terminal_present and not raw_present:
        raise ValueError(
            "post-recovery aggregate terminal JSON exists without its raw sacct evidence"
        )
    if raw_present:
        _require_real_file(raw_path, name="raw aggregation sacct evidence")
        if raw_path.read_bytes() != queried_raw:
            raise ValueError("existing raw aggregation sacct evidence differs from the fresh query")
    if terminal_present:
        terminal_sha256 = _sha256_file(terminal_path)
        verify_post_recovery_aggregate_terminal_evidence(
            terminal_path,
            expected_sha256=terminal_sha256,
            expected_aggregate_path=aggregate_file,
            expected_aggregate_sha256=aggregate_sha256,
            expected_publication_path=publication_path,
            expected_publication_sha256=publication_sha256,
            expected_slurm_job_id=publication["slurm_job_id"],
            expected_partition=publication["partition"],
        )
    else:
        if raw_present:
            _require_real_file(raw_path, name="raw aggregation sacct evidence")
            raw = raw_path.read_bytes()
        else:
            raw = queried_raw
        row = _parse_post_recovery_aggregate_sacct_raw(
            raw,
            expected_job_id=publication["slurm_job_id"],
            expected_partition=publication["partition"],
        )
        terminal = {
            "schema_version": POST_RECOVERY_AGGREGATE_TERMINAL_SCHEMA,
            "captured_at_utc": _utc_now(),
            "query": list(queried_command),
            "aggregation_slurm_job_id": publication["slurm_job_id"],
            "row": row,
            "aggregate": {
                "filename": aggregate_file.name,
                "sha256": aggregate_sha256,
                "size_bytes": aggregate_file.stat().st_size,
            },
            "publication_receipt": {
                "filename": publication_path.name,
                "sha256": publication_sha256,
                "size_bytes": publication_path.stat().st_size,
            },
            "raw_sacct": {
                "filename": raw_path.name,
                "sha256": _sha256_bytes(raw),
                "size_bytes": len(raw),
            },
        }
        if not raw_present:
            _write_exclusive(raw_path, raw, name="raw aggregation sacct evidence")
        _write_exclusive(
            terminal_path,
            _canonical_json(terminal),
            name="post-recovery aggregate terminal evidence",
        )
        terminal_sha256 = _sha256_file(terminal_path)
        verify_post_recovery_aggregate_terminal_evidence(
            terminal_path,
            expected_sha256=terminal_sha256,
            expected_aggregate_path=aggregate_file,
            expected_aggregate_sha256=aggregate_sha256,
            expected_publication_path=publication_path,
            expected_publication_sha256=publication_sha256,
            expected_slurm_job_id=publication["slurm_job_id"],
            expected_partition=publication["partition"],
        )
    success = dict(publication)
    success.update(
        {
            "schema_version": POST_RECOVERY_AGGREGATE_SUCCESS_SCHEMA,
            "status": "SUCCESS",
            "aggregate_publication_receipt_sha256": publication_sha256,
            "aggregation_terminal_evidence_sha256": terminal_sha256,
            "created_at_utc": _utc_now(),
        }
    )
    _write_exclusive(
        success_path,
        _aggregate_receipt_bytes(success),
        name="post-recovery aggregate SUCCESS receipt",
    )
    return verify_post_recovery_aggregate_success_receipt(aggregate_file)


def capture_post_recovery_aggregate_terminal_evidence(
    aggregate_path: str | os.PathLike[str],
    *,
    attempt_job_id: str,
) -> dict[str, object]:
    """Publish only after exact terminal success; resume every publication phase."""

    aggregate_file = Path(aggregate_path).absolute()
    lock_path = Path(f"{aggregate_file}.TERMINAL.lock").absolute()
    success_path = Path(f"{aggregate_file}.SUCCESS").absolute()
    if success_path.exists() or success_path.is_symlink():
        with _exclusive_file_lock(lock_path):
            return _verify_completed_aggregate_for_attempt(
                aggregate_file,
                attempt_job_id=attempt_job_id,
            )
    attempt = verify_post_recovery_aggregate_attempt_ready(
        aggregate_file,
        attempt_job_id=attempt_job_id,
    )
    aggregate_submission = attempt["aggregate_submission"]
    if not isinstance(aggregate_submission, Mapping):
        raise TypeError("aggregate CPU submission context is invalid")
    _verify_aggregate_submission_scheduler_authority(aggregate_submission)
    ready = attempt["ready"]
    if not isinstance(ready, Mapping):
        raise TypeError("aggregate attempt READY context is invalid")
    command, raw = _query_post_recovery_aggregate_sacct(
        attempt_job_id=attempt_job_id,
        expected_partition=str(ready["partition"]),
    )
    with _exclusive_file_lock(lock_path):
        return _capture_post_recovery_aggregate_terminal_evidence_locked(
            aggregate_file,
            attempt_job_id=attempt_job_id,
            queried_command=command,
            queried_raw=raw,
        )


def verify_post_recovery_aggregate_success_receipt(
    aggregate_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Require publication plus exact post-job terminal CPU evidence."""

    context = _post_recovery_aggregate_context(aggregate_path)
    aggregate_file = context["aggregate_file"]
    if not isinstance(aggregate_file, Path):
        raise TypeError("post-recovery aggregate context path is invalid")
    publication_path, publication = _verify_post_recovery_aggregate_publication_receipt(context)
    expected_receipt_path = Path(f"{aggregate_file}.SUCCESS").absolute()
    actual_receipt_path = (
        expected_receipt_path if receipt_path is None else Path(receipt_path).absolute()
    )
    if actual_receipt_path != expected_receipt_path:
        raise ValueError(
            "post-recovery aggregate SUCCESS receipt must be adjacent to its aggregate"
        )
    fields = parse_post_recovery_aggregate_success_receipt(actual_receipt_path)
    expected = context["expected"]
    if not isinstance(expected, Mapping):
        raise TypeError("post-recovery aggregate receipt context is invalid")
    locked = {
        "schema_version": POST_RECOVERY_AGGREGATE_SUCCESS_SCHEMA,
        "status": "SUCCESS",
        **expected,
        "slurm_job_id": publication["slurm_job_id"],
        "partition": publication["partition"],
        "aggregate_publication_receipt_sha256": _sha256_file(publication_path),
    }
    for key, expected_value in locked.items():
        if fields.get(key) != expected_value:
            raise ValueError(f"post-recovery aggregate SUCCESS receipt {key} is invalid")
    for key in _AGGREGATE_PUBLICATION_KEYS - {
        "schema_version",
        "status",
        "created_at_utc",
    }:
        if fields[key] != publication[key]:
            raise ValueError(f"post-recovery aggregate SUCCESS receipt changed published {key}")
    if not _valid_utc(fields["created_at_utc"]):
        raise ValueError("post-recovery aggregate SUCCESS receipt created_at_utc is invalid")
    terminal_sha256 = _digest(
        fields["aggregation_terminal_evidence_sha256"],
        name="post-recovery aggregate terminal evidence SHA256",
    )
    publication_sha256 = _digest(
        fields["aggregate_publication_receipt_sha256"],
        name="post-recovery aggregate PUBLISHED receipt SHA256",
    )
    terminal_path = Path(f"{aggregate_file}.TERMINAL.json").absolute()
    terminal = verify_post_recovery_aggregate_terminal_evidence(
        terminal_path,
        expected_sha256=terminal_sha256,
        expected_aggregate_path=aggregate_file,
        expected_aggregate_sha256=fields["aggregate_sha256"],
        expected_publication_path=publication_path,
        expected_publication_sha256=publication_sha256,
        expected_slurm_job_id=fields["slurm_job_id"],
        expected_partition=fields["partition"],
    )
    return {
        "receipt": fields,
        "receipt_sha256": _sha256_file(actual_receipt_path),
        "publication_receipt_sha256": publication_sha256,
        "terminal_evidence_sha256": terminal_sha256,
        "terminal_evidence": terminal,
        "aggregate_sha256": fields["aggregate_sha256"],
        "phase2_design_sha256": context["phase2_design_sha256"],
        "base_config_hash": context["base_config_hash"],
        "pilot_phase": context["pilot_phase"],
        "pilot_array_job_id": fields["pilot_array_job_id"],
        "aggregation_slurm_job_id": fields["slurm_job_id"],
    }


def verify_post_recovery_success_marker(
    path: str | os.PathLike[str],
    *,
    expected_array_job_id: str,
    expected_task_id: int,
    expected_seed: int,
    expected_design_sha256: str,
    expected_base_config_hash: str,
    expected_git_commit: str,
    expected_authorization_sha256: str,
    expected_submission_intent_sha256: str,
    expected_submission_ledger_sha256: str,
    expected_allocation_job_id_raw: str,
    expected_pilot_phase: str,
) -> dict[str, str]:
    """Validate one immutable SUCCESS receipt before pilot aggregation."""

    fields = parse_post_recovery_success_marker(path)
    expected_task = _integer(expected_task_id, name="expected_task_id")
    if expected_task >= len(ORDERED_SEEDS) or ORDERED_SEEDS[expected_task] != expected_seed:
        raise ValueError("expected task/seed mapping is not the locked pilot order")
    for value, name, lengths in (
        (expected_design_sha256, "expected design SHA256", frozenset({64})),
        (expected_base_config_hash, "expected base config hash", frozenset({64})),
        (expected_git_commit, "expected Git commit", frozenset({40, 64})),
        (expected_authorization_sha256, "expected authorization SHA256", frozenset({64})),
        (
            expected_submission_intent_sha256,
            "expected submission intent SHA256",
            frozenset({64}),
        ),
        (
            expected_submission_ledger_sha256,
            "expected submission ledger SHA256",
            frozenset({64}),
        ),
    ):
        _digest(value, name=name, lengths=lengths)
    if re.fullmatch(r"[1-9][0-9]*", expected_allocation_job_id_raw) is None:
        raise ValueError("expected allocation JobIDRaw must be a positive integer")
    if expected_pilot_phase not in POST_RECOVERY_PILOT_PHASES:
        raise ValueError("expected_pilot_phase must be calibration or freeze")
    composite_job_id = f"{expected_array_job_id}_{expected_task}"
    expected = {
        "schema_version": POST_RECOVERY_RUN_STATUS_SCHEMA,
        "status": "SUCCESS",
        "pilot_phase": expected_pilot_phase,
        "workload_exit_code": "0",
        "final_exit_code": "0",
        "slurm_job_id": expected_allocation_job_id_raw,
        "allocation_job_id_raw": expected_allocation_job_id_raw,
        "slurm_array_task_job_id": composite_job_id,
        "array_job_id": expected_array_job_id,
        "array_task_id": str(expected_task),
        "seed": str(expected_seed),
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "gpu-l20",
        "restart_count": "0",
        "phase2_design_sha256": expected_design_sha256,
        "base_config_hash": expected_base_config_hash,
        "git_commit": expected_git_commit,
        "recovery_authorization_sha256": expected_authorization_sha256,
        "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
        "submission_intent_sha256": expected_submission_intent_sha256,
        "submission_ledger_sha256": expected_submission_ledger_sha256,
        "materialization_mode": "fresh",
        "recovery_outputs_mounted": "false",
        "hf_root_mount_mode": "read_only",
        "datasets_cache_scope": "job_local",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ValueError(f"post-recovery SUCCESS marker {key} is invalid")
    for key in (
        "artifact_metadata_sha256",
        "phase2_result_sha256",
        "phase2_output_verification_sha256",
        "post_recovery_output_verification_sha256",
    ):
        _digest(fields[key], name=f"post-recovery SUCCESS marker {key}")
    if not _valid_utc(fields["created_at_utc"]):
        raise ValueError("post-recovery SUCCESS marker created_at_utc is invalid")
    return fields


__all__ = [
    "OPTIMIZER_SCHEDULE_SHA256",
    "ORDERED_SEEDS",
    "POST_RECOVERY_CONFIG_SCHEMA",
    "POST_RECOVERY_DESIGN_NAME",
    "POST_RECOVERY_AGGREGATE_PUBLICATION_SCHEMA",
    "POST_RECOVERY_AGGREGATE_SUCCESS_SCHEMA",
    "POST_RECOVERY_AGGREGATE_TERMINAL_SCHEMA",
    "POST_RECOVERY_OUTPUT_VERIFICATION_SCHEMA",
    "POST_RECOVERY_PILOT_PHASES",
    "POST_RECOVERY_RUN_STATUS_SCHEMA",
    "POST_RECOVERY_TERMINAL_SCHEMA",
    "RECOVERY_ARRAY_JOB_ID",
    "RECOVERY_EXECUTION_REVISION",
    "capture_post_recovery_aggregate_terminal_evidence",
    "capture_post_recovery_terminal_evidence",
    "parse_post_recovery_aggregate_publication_receipt",
    "parse_post_recovery_aggregate_success_receipt",
    "parse_post_recovery_success_marker",
    "post_recovery_aggregate_sacct_command",
    "sacct_command",
    "verify_post_recovery_aggregate_terminal_evidence",
    "verify_post_recovery_success_marker",
    "verify_post_recovery_aggregate_success_receipt",
    "verify_post_recovery_terminal_evidence",
    "verify_recovery_authorization_config_binding",
    "verify_recovery_authorization_file",
]
