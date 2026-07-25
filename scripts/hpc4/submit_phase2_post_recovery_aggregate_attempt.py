#!/usr/bin/env python3
"""Submit one controlled post-recovery aggregate attempt at a time.

Every CPU job is created held.  Its exact request is committed to an immutable
attempt ledger before release.  A retry is possible only after the previous
registered attempt has exact terminal non-zero Slurm evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

INTENT_SCHEMA = "prorm-phase2-post-recovery-aggregate-submit-intent/v2"
ATTEMPT_SCHEMA = "prorm-phase2-post-recovery-aggregate-submit-attempt/v2"
FAILURE_SCHEMA = "prorm-phase2-post-recovery-aggregate-attempt-failure/v1"
SCHEDULER_SCHEMA = "prorm-phase2-post-recovery-aggregate-held-request/v2"
SCRIPT_BINDING_SCHEMA = "prorm-phase2-post-recovery-aggregate-batch-script/v1"
SCRIPT_EVIDENCE_FILENAME = "script.sbatch"
CONTROLLER_READBACK_DIRECTORY = "controller"
SCRIPT_TRANSPORT = "sbatch-stdin"
_MAX_SCRIPT_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_TERMINAL_FAILURE_STATES = frozenset(
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
_CONTROL_EXPORT_KEYS = frozenset(
    {
        "PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY",
        "PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256",
        "PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX",
        "PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256",
    }
)
_SACCT_FIELDS = (
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
_SACCT_FORMAT_FIELDS = (
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


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def _validate_script_bytes(raw: bytes, *, name: str) -> bytes:
    if not raw or len(raw) > _MAX_SCRIPT_BYTES or b"\0" in raw:
        raise ValueError(f"{name} bytes are malformed")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, *, name: str, lengths: set[int] | None = None) -> str:
    allowed = {64} if lengths is None else lengths
    if (
        not isinstance(value, str)
        or len(value) not in allowed
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _effective_user() -> str:
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise RuntimeError("aggregate submission requires POSIX effective-user identity")
    import pwd

    user = pwd.getpwuid(os.geteuid()).pw_name
    if re.fullmatch(r"[A-Za-z0-9._-]+", user) is None:
        raise ValueError("effective submitter user is unsafe")
    return user


def _valid_utc(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("timestamp must be UTC")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("timestamp must be UTC") from error
    return value


def _canonical_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    metadata = absolute.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical non-symlink directory")
    return absolute


def _canonical_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    metadata = absolute.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical regular non-symlink file")
    return absolute


def _ensure_directory(path: Path, *, root: Path, name: str) -> Path:
    absolute = path.absolute()
    absolute.relative_to(root)
    absolute.mkdir(parents=True, exist_ok=True)
    current = root
    for component in absolute.relative_to(root).parts:
        current /= component
        _canonical_directory(current, name=name)
    return absolute


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(path: Path, raw: bytes, *, name: str) -> str:
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=".staged-",
        dir=path.parent,
    )
    temporary = Path(temporary_text)
    os.chmod(temporary, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {name}: {path}") from None
    except BaseException:
        raise
    with suppress(OSError):
        temporary.unlink()
    _fsync_directory(path.parent)
    return _sha256_bytes(raw)


def _write_exclusive(path: Path, value: Mapping[str, object], *, name: str) -> str:
    return _write_bytes_exclusive(
        path,
        _canonical_json(value),
        name=name,
    )


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path, *, name: str) -> tuple[dict[str, object], str]:
    _canonical_file(path, name=name)
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise ValueError(f"{name} must contain canonical JSON object bytes")
    return value, _sha256_bytes(raw)


def _run(
    arguments: Sequence[str],
    *,
    name: str,
    text: bool = True,
    input_bytes: bytes | None = None,
    require_empty_stderr: bool = False,
) -> str | bytes:
    if text and input_bytes is not None:
        raise TypeError("binary subprocess input requires binary output mode")
    completed = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=text,
        input=input_bytes,
        timeout=60,
    )
    stderr_present = bool(completed.stderr)
    if completed.returncode != 0 or (require_empty_stderr and stderr_present):
        detail = (
            completed.stderr.strip()
            if text
            else completed.stderr.decode("utf-8", errors="replace").strip()
        )
        raise RuntimeError(f"{name} failed{': ' + detail if detail else ''}")
    if require_empty_stderr and not completed.stdout:
        raise RuntimeError(f"{name} returned empty stdout")
    return completed.stdout


def _parse_export_spec(raw: str) -> dict[str, str]:
    if not raw or "\n" in raw or "\r" in raw:
        raise ValueError("workload export specification is malformed")
    result: dict[str, str] = {}
    for item in raw.split(","):
        if item.count("=") != 1:
            raise ValueError("workload export specification item is malformed")
        key, value = item.split("=", 1)
        if not key or not value or key in result:
            raise ValueError("workload export specification has duplicate fields")
        result[key] = value
    if set(result) & _CONTROL_EXPORT_KEYS:
        raise ValueError("workload export specification occupies control fields")
    return result


def _intent_payload(
    *,
    pilot_phase: str,
    design_sha256: str,
    pilot_array_job_id: str,
    aggregator_git_commit: str,
    project_root: Path,
    repository_root: Path,
    output: Path,
    partition: str,
    walltime: str,
    workload_export_spec: str,
    script_relative: str,
    script_sha256: str,
    script_git_blob_sha1: str,
    script_size_bytes: int,
    submitter_user: str,
    job_name: str,
    created_at_utc: str,
) -> dict[str, object]:
    if pilot_phase not in {"calibration", "freeze"}:
        raise ValueError("pilot phase is invalid")
    _digest(design_sha256, name="design SHA256")
    _digest(aggregator_git_commit, name="aggregator commit", lengths={40, 64})
    _digest(script_sha256, name="aggregate sbatch SHA256")
    _digest(script_git_blob_sha1, name="aggregate sbatch Git blob SHA1", lengths={40})
    if (
        re.fullmatch(r"[1-9][0-9]*", pilot_array_job_id) is None
        or not isinstance(script_size_bytes, int)
        or not 0 < script_size_bytes <= _MAX_SCRIPT_BYTES
        or partition not in {"amd", "intel"}
        or re.fullmatch(
            r"(?:[1-9][0-9]*-[0-9]{2}:[0-9]{2}:[0-9]{2}|"
            r"[0-9]{2}:[0-9]{2}:[0-9]{2})",
            walltime,
        )
        is None
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", job_name) is None
        or re.fullmatch(r"[A-Za-z0-9._-]+", submitter_user) is None
    ):
        raise ValueError("aggregate submission identity is invalid")
    exports = _parse_export_spec(workload_export_spec)
    required_exports = {
        "PRORM_PROJECT_ROOT": os.fspath(project_root),
        "PRORM_REPO_ROOT": os.fspath(repository_root),
        "PRORM_POST_RECOVERY_DESIGN_SHA256": design_sha256,
        "PRORM_POST_RECOVERY_ARRAY_JOB_ID": pilot_array_job_id,
        "PRORM_AGGREGATOR_GIT_COMMIT": aggregator_git_commit,
        "PRORM_POST_RECOVERY_AGGREGATE_OUTPUT": os.fspath(output),
        "PRORM_POST_RECOVERY_PILOT_PHASE": pilot_phase,
    }
    for key, expected in required_exports.items():
        if exports.get(key) != expected:
            raise ValueError(f"aggregate workload export {key} differs")
    return {
        "schema_version": INTENT_SCHEMA,
        "status": "precommitted_before_first_cpu_attempt",
        "pilot_phase": pilot_phase,
        "phase2_design_sha256": design_sha256,
        "pilot_array_job_id": pilot_array_job_id,
        "aggregator_git_commit": aggregator_git_commit,
        "project_root": os.fspath(project_root),
        "repository_root": os.fspath(repository_root),
        "final_output": os.fspath(output),
        "partition": partition,
        "walltime": walltime,
        "workload_export_spec": workload_export_spec,
        "workload_export_spec_sha256": _sha256_bytes(workload_export_spec.encode("utf-8")),
        "sbatch_script": {
            "repo_relative_path": script_relative,
            "sha256": script_sha256,
            "git_blob_sha1": script_git_blob_sha1,
            "size_bytes": script_size_bytes,
            "git_object": f"{aggregator_git_commit}:{script_relative}",
            "evidence_filename": SCRIPT_EVIDENCE_FILENAME,
            "transport": SCRIPT_TRANSPORT,
        },
        "submitter_user": submitter_user,
        "job_name": job_name,
        "cluster": "hpc4",
        "account": "sigroup",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 4,
        "memory": "16G",
        "requeue": False,
        "retry_only_after_exact_terminal_failure": True,
        "created_at_utc": _valid_utc(created_at_utc),
    }


def _validate_intent(
    value: Mapping[str, object],
    *,
    expected: Mapping[str, object],
) -> None:
    if set(value) != set(expected):
        raise ValueError("aggregate submission intent fields differ")
    observed = dict(value)
    template = dict(expected)
    _valid_utc(observed.pop("created_at_utc", None))
    template.pop("created_at_utc", None)
    if observed != template:
        raise ValueError("aggregate submission intent identity differs")


def _comment(intent_sha256: str, attempt_index: int) -> str:
    return f"prorm-aggregate:{intent_sha256}:attempt-{attempt_index}"


def _parse_scontrol(
    raw: str,
    *,
    job_id: str,
    intent: Mapping[str, object],
    intent_sha256: str,
    attempt_index: int,
    repository_root: Path,
) -> tuple[str, dict[str, object] | None]:
    lines = [line for line in raw.splitlines() if line]
    if len(lines) != 1 or "\r" in raw:
        raise ValueError("aggregate scontrol evidence must contain one safe row")
    fields: dict[str, str] = {}
    for token in lines[0].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate scontrol field: {key}")
        fields[key] = value
    tres: dict[str, str] = {}
    for item in fields.get("TRES", "").split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key in tres:
            raise ValueError(f"duplicate aggregate TRES field: {key}")
        tres[key] = value
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
        or fields.get("Command") != "(null)"
        or fields.get("WorkDir") != os.fspath(repository_root)
        or fields.get("Comment") != _comment(intent_sha256, attempt_index)
        or fields.get("BatchFlag") != "1"
        or tres.get("cpu") != "4"
        or tres.get("mem") != "16G"
        or tres.get("node") != "1"
        or any("gpu" in key.lower() for key in tres)
        or "gpu" in fields.get("TresPerNode", "").lower()
        or not fields.get("JobState")
    ):
        raise ValueError("CPU job differs from the immutable aggregate intent")
    held = fields["JobState"] == "PENDING" and fields.get("Reason") == "JobHeldUser"
    if held:
        scheduler = {
            "schema_version": SCHEDULER_SCHEMA,
            "captured_while_held": True,
            "raw_scontrol_record": raw,
            "raw_scontrol_sha256": _sha256_bytes(raw.encode("utf-8")),
            "normalized": {
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
                "comment": _comment(intent_sha256, attempt_index),
                "command": "(null)",
                "work_dir": os.fspath(repository_root),
            },
        }
        return "HELD", scheduler
    if fields["JobState"] == "PENDING" and str(fields.get("Reason", "")).startswith("JobHeld"):
        raise ValueError("aggregate job is held by an unexpected authority")
    return str(fields["JobState"]), None


def _controller_readback_relative(attempt_index: int) -> str:
    if attempt_index <= 0:
        raise ValueError("aggregate attempt index must be positive")
    return f"{CONTROLLER_READBACK_DIRECTORY}/attempt-{attempt_index:04d}.sbatch"


def _controller_readback_query(job_id: str) -> tuple[str, ...]:
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ValueError("aggregate controller readback job ID is invalid")
    return ("scontrol", "write", "batch_script", job_id, "-")


def _sbatch_command(
    *,
    intent: Mapping[str, object],
    intent_sha256: str,
    attempt_index: int,
    scheduler_export_spec: str,
    repository_root: Path,
    log_root: Path,
) -> tuple[str, ...]:
    return (
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
        f"--comment={_comment(intent_sha256, attempt_index)}",
        f"--chdir={repository_root}",
        f"--output={log_root}/%x-%j.out",
        f"--error={log_root}/%x-%j.err",
        f"--export={scheduler_export_spec}",
    )


def _read_controller_batch_script(job_id: str) -> tuple[tuple[str, ...], bytes]:
    query = _controller_readback_query(job_id)
    raw = _run(
        query,
        name="aggregate controller batch-script readback",
        text=False,
        require_empty_stderr=True,
    )
    assert isinstance(raw, bytes)
    return query, _validate_script_bytes(
        raw,
        name="aggregate controller batch-script readback",
    )


def _capture_controller_batch_script(
    *,
    registry: Path,
    attempt_index: int,
    job_id: str,
    submission_command: Sequence[str],
    committed_script: bytes,
) -> dict[str, object]:
    query, controller_raw = _read_controller_batch_script(job_id)
    if controller_raw != committed_script:
        raise RuntimeError("controller batch script differs from the committed Git blob")
    relative = _controller_readback_relative(attempt_index)
    evidence = registry.joinpath(*relative.split("/"))
    if evidence.exists() or evidence.is_symlink():
        _canonical_file(evidence, name="aggregate controller batch-script evidence")
        if evidence.read_bytes() != controller_raw:
            raise ValueError("aggregate controller batch-script evidence changed")
    else:
        _write_bytes_exclusive(
            evidence,
            controller_raw,
            name="aggregate controller batch-script evidence",
        )
    return {
        "schema_version": SCRIPT_BINDING_SCHEMA,
        "transport": SCRIPT_TRANSPORT,
        "submission_command": list(submission_command),
        "stdin_sha256": _sha256_bytes(committed_script),
        "stdin_size_bytes": len(committed_script),
        "controller_readback": {
            "query": list(query),
            "relative_path": relative,
            "sha256": _sha256_bytes(controller_raw),
            "size_bytes": len(controller_raw),
        },
        "controller_matches_committed": True,
    }


def _verify_controller_batch_script_fresh(
    *,
    registry: Path,
    attempt_index: int,
    job_id: str,
    committed_script: bytes,
) -> None:
    query, controller_raw = _read_controller_batch_script(job_id)
    evidence = registry.joinpath(*_controller_readback_relative(attempt_index).split("/"))
    _canonical_file(evidence, name="aggregate controller batch-script evidence")
    if (
        query != _controller_readback_query(job_id)
        or controller_raw != committed_script
        or evidence.read_bytes() != controller_raw
    ):
        raise RuntimeError("aggregate controller batch script changed after capture")


def _attempt_payload(
    *,
    intent: Mapping[str, object],
    intent_sha256: str,
    attempt_index: int,
    job_id: str,
    scheduler_export_spec: str,
    scheduler_request: Mapping[str, object],
    batch_script: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "committed_while_scheduler_held",
        "intent_sha256": intent_sha256,
        "attempt_index": attempt_index,
        "slurm_job_id": job_id,
        "cluster": "hpc4",
        "scheduler_export_spec": scheduler_export_spec,
        "scheduler_export_spec_sha256": _sha256_bytes(scheduler_export_spec.encode("utf-8")),
        "scheduler_request": dict(scheduler_request),
        "batch_script": dict(batch_script),
        "released_only_after_attempt_ledger_fsync": True,
        "created_at_utc": _utc_now(),
    }


def _validate_attempt(
    value: Mapping[str, object],
    *,
    intent: Mapping[str, object],
    intent_sha256: str,
    expected_index: int,
    committed_script: bytes,
    expected_submission_command: Sequence[str],
    repository_root: Path,
    registry: Path,
) -> str:
    expected_keys = {
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
    if set(value) != expected_keys:
        raise ValueError("aggregate attempt ledger fields differ")
    job_id = value["slurm_job_id"]
    if (
        value["schema_version"] != ATTEMPT_SCHEMA
        or value["status"] != "committed_while_scheduler_held"
        or value["intent_sha256"] != intent_sha256
        or value["attempt_index"] != expected_index
        or not isinstance(job_id, str)
        or re.fullmatch(r"[1-9][0-9]*", job_id) is None
        or value["cluster"] != "hpc4"
        or value["released_only_after_attempt_ledger_fsync"] is not True
    ):
        raise ValueError("aggregate attempt ledger identity is invalid")
    _valid_utc(value["created_at_utc"])
    control_suffix = (
        f",PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY={registry}"
        f",PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256={intent_sha256}"
        f",PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX={expected_index}"
        ",PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256="
        f"{intent['workload_export_spec_sha256']}"
    )
    expected_export = f"{intent['workload_export_spec']}{control_suffix}"
    if value["scheduler_export_spec"] != expected_export or value[
        "scheduler_export_spec_sha256"
    ] != _sha256_bytes(expected_export.encode("utf-8")):
        raise ValueError("aggregate attempt scheduler export differs")
    scheduler = value["scheduler_request"]
    if not isinstance(scheduler, Mapping):
        raise TypeError("aggregate attempt scheduler request must be a mapping")
    raw = scheduler.get("raw_scontrol_record")
    if (
        scheduler.get("schema_version") != SCHEDULER_SCHEMA
        or scheduler.get("captured_while_held") is not True
        or not isinstance(raw, str)
        or scheduler.get("raw_scontrol_sha256") != _sha256_bytes(raw.encode("utf-8"))
    ):
        raise ValueError("aggregate held scheduler evidence is invalid")
    state, reparsed = _parse_scontrol(
        raw,
        job_id=job_id,
        intent=intent,
        intent_sha256=intent_sha256,
        attempt_index=expected_index,
        repository_root=repository_root,
    )
    if state != "HELD" or reparsed != dict(scheduler):
        raise ValueError("aggregate held scheduler evidence does not reparse")
    batch_script = value["batch_script"]
    if not isinstance(batch_script, Mapping) or set(batch_script) != {
        "schema_version",
        "transport",
        "submission_command",
        "stdin_sha256",
        "stdin_size_bytes",
        "controller_readback",
        "controller_matches_committed",
    }:
        raise ValueError("aggregate attempt batch-script binding fields differ")
    controller = batch_script["controller_readback"]
    if not isinstance(controller, Mapping) or set(controller) != {
        "query",
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("aggregate attempt controller readback fields differ")
    expected_relative = _controller_readback_relative(expected_index)
    controller_path = registry.joinpath(*expected_relative.split("/"))
    _canonical_file(
        controller_path,
        name="aggregate controller batch-script evidence",
    )
    controller_raw = controller_path.read_bytes()
    if (
        batch_script["schema_version"] != SCRIPT_BINDING_SCHEMA
        or batch_script["transport"] != SCRIPT_TRANSPORT
        or batch_script["submission_command"] != list(expected_submission_command)
        or batch_script["stdin_sha256"] != _sha256_bytes(committed_script)
        or batch_script["stdin_size_bytes"] != len(committed_script)
        or batch_script["controller_matches_committed"] is not True
        or controller["query"] != list(_controller_readback_query(job_id))
        or controller["relative_path"] != expected_relative
        or controller["sha256"] != _sha256_bytes(controller_raw)
        or controller["size_bytes"] != len(controller_raw)
        or controller_raw != committed_script
    ):
        raise ValueError("aggregate attempt batch-script binding differs")
    return job_id


def _squeue_ids(job_name: str, user: str) -> tuple[str, ...]:
    raw = _run(
        (
            "squeue",
            "--noheader",
            f"--user={user}",
            f"--name={job_name}",
            "--format=%A",
        ),
        name="aggregate live-name query",
    )
    assert isinstance(raw, str)
    values = {line.strip() for line in raw.splitlines() if line.strip()}
    if any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in values):
        raise ValueError("squeue returned an invalid aggregate job ID")
    return tuple(sorted(values, key=int))


def _sacct_rows(
    *,
    job_name: str,
    starttime: str,
    intent: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    fields = ",".join(_SACCT_FORMAT_FIELDS)
    raw = _run(
        (
            "sacct",
            "-X",
            f"--starttime={starttime}",
            f"--name={job_name}",
            "--noheader",
            "--parsable2",
            f"--format={fields}",
        ),
        name="aggregate historical-name query",
    )
    assert isinstance(raw, str)
    rows: dict[str, dict[str, str]] = {}
    for line in raw.splitlines():
        if not line or "\r" in line:
            raise ValueError("sacct returned unsafe aggregate history")
        values = line.split("|")
        if len(values) != len(_SACCT_FIELDS):
            raise ValueError("aggregate sacct history field count differs")
        row = dict(zip(_SACCT_FIELDS, values, strict=True))
        job_id = row["JobIDRaw"]
        if (
            re.fullmatch(r"[1-9][0-9]*", job_id) is None
            or row["JobID"] != job_id
            or row["JobName"] != job_name
            or row["Cluster"] != "hpc4"
            or row["Account"] != "sigroup"
            or row["Partition"] != intent["partition"]
            or row["NNodes"] not in {"0", "1"}
            or row["NCPUS"] not in {"0", "4"}
            or row["Timelimit"] != intent["walltime"]
            or row["ReqTRES"] != "billing=4,cpu=4,mem=16G,node=1"
            or (row["AllocTRES"] not in {"", "billing=4,cpu=4,mem=16G,node=1"})
        ):
            raise ValueError("aggregate sacct history differs from intent")
        if job_id in rows:
            raise ValueError("aggregate sacct history has duplicate root jobs")
        rows[job_id] = row
    return rows


def _verify_release_collision_snapshot(
    *,
    job_name: str,
    user: str,
    intent: Mapping[str, object],
    intent_sha256: str,
    attempt_index: int,
    job_id: str,
    registered_job_ids: set[str],
    registry: Path,
    committed_script: bytes,
    repository_root: Path,
) -> None:
    """Recheck the full deterministic-name namespace immediately before release."""

    live_ids = set(_squeue_ids(job_name, user))
    rows = _sacct_rows(
        job_name=job_name,
        starttime=_valid_utc(intent["created_at_utc"])[:-1],
        intent=intent,
    )
    expected_ids = set(registered_job_ids)
    expected_ids.add(job_id)
    observed_ids = live_ids | set(rows)
    if observed_ids != expected_ids:
        raise RuntimeError("aggregate scheduler identity changed after ledger commit")
    if job_id not in live_ids:
        raise RuntimeError("ledger-bound aggregate attempt disappeared before release")
    raw = _run(
        ("scontrol", "show", "job", "--oneliner", job_id),
        name="pre-release aggregate held attempt query",
    )
    assert isinstance(raw, str)
    state, scheduler = _parse_scontrol(
        raw,
        job_id=job_id,
        intent=intent,
        intent_sha256=intent_sha256,
        attempt_index=attempt_index,
        repository_root=repository_root,
    )
    if state != "HELD" or scheduler is None:
        raise RuntimeError(
            "ledger-bound aggregate attempt was released before final collision check"
        )
    _verify_controller_batch_script_fresh(
        registry=registry,
        attempt_index=attempt_index,
        job_id=job_id,
        committed_script=committed_script,
    )


def _failure_sacct_query(job_id: str) -> tuple[str, ...]:
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ValueError("aggregate failure job ID is invalid")
    return (
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        job_id,
        f"--format={','.join(_SACCT_FORMAT_FIELDS)}",
    )


def _parse_single_sacct_raw(
    raw: bytes,
    *,
    expected_job_id: str,
    expected_name: str,
    expected_partition: str,
    expected_walltime: str,
) -> dict[str, str]:
    if not raw or len(raw) > 16 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("aggregate failure raw sacct bytes are malformed")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("aggregate failure raw sacct bytes must be UTF-8") from error
    if len(lines) != 1 or not lines[0]:
        raise ValueError("aggregate failure raw sacct must contain exactly one row")
    values = lines[0].split("|")
    if len(values) != len(_SACCT_FIELDS):
        raise ValueError("aggregate failure raw sacct fields differ")
    row = dict(zip(_SACCT_FIELDS, values, strict=True))
    if (
        row["JobIDRaw"] != expected_job_id
        or row["JobID"] != expected_job_id
        or row["JobName"] != expected_name
        or row["Cluster"] != "hpc4"
        or row["Account"] != "sigroup"
        or row["Partition"] != expected_partition
        or row["NNodes"] not in {"0", "1"}
        or row["NCPUS"] not in {"0", "4"}
        or row["Timelimit"] != expected_walltime
        or row["ReqTRES"] != "billing=4,cpu=4,mem=16G,node=1"
        or row["AllocTRES"] not in {"", "billing=4,cpu=4,mem=16G,node=1"}
        or not _terminal_failure(row)
    ):
        raise ValueError("aggregate failure raw sacct row is not exact terminal failure")
    return row


def _query_single_failure_row(
    *,
    job_id: str,
    intent: Mapping[str, object],
) -> tuple[tuple[str, ...], bytes, dict[str, str]]:
    query = _failure_sacct_query(job_id)
    completed = subprocess.run(
        list(query),
        check=False,
        capture_output=True,
        timeout=60,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("aggregate failure sacct query failed")
    raw = completed.stdout
    row = _parse_single_sacct_raw(
        raw,
        expected_job_id=job_id,
        expected_name=str(intent["job_name"]),
        expected_partition=str(intent["partition"]),
        expected_walltime=str(intent["walltime"]),
    )
    return query, raw, row


def _successful(row: Mapping[str, str]) -> bool:
    return (
        row["State"] == "COMPLETED"
        and row["ExitCode"] == "0:0"
        and row["DerivedExitCode"] == "0:0"
        and row["NNodes"] == "1"
        and row["NCPUS"] == "4"
        and row["AllocTRES"] == "billing=4,cpu=4,mem=16G,node=1"
    )


def _terminal_failure(row: Mapping[str, str]) -> bool:
    state = row["State"].split("+", 1)[0]
    return state in _TERMINAL_FAILURE_STATES


def _failure_payload(
    *,
    intent_sha256: str,
    attempt_index: int,
    job_id: str,
    attempt_ledger_sha256: str,
    row: Mapping[str, str],
    query: Sequence[str],
    raw_filename: str,
    raw_sha256: str,
    raw_size_bytes: int,
) -> dict[str, object]:
    if not _terminal_failure(row):
        raise ValueError("retry requires exact terminal failure evidence")
    return {
        "schema_version": FAILURE_SCHEMA,
        "status": "TERMINAL_FAILURE",
        "intent_sha256": intent_sha256,
        "attempt_index": attempt_index,
        "slurm_job_id": job_id,
        "attempt_ledger_sha256": _digest(
            attempt_ledger_sha256,
            name="failed aggregate attempt ledger SHA256",
        ),
        "row": dict(row),
        "query": list(query),
        "raw_sacct": {
            "filename": raw_filename,
            "sha256": _digest(
                raw_sha256,
                name="aggregate failure raw sacct SHA256",
            ),
            "size_bytes": raw_size_bytes,
        },
        "retry_authorized": True,
        "captured_at_utc": _utc_now(),
    }


def _validate_failure(
    value: Mapping[str, object],
    *,
    intent_sha256: str,
    attempt_index: int,
    job_id: str,
    attempt_ledger_sha256: str,
    row: Mapping[str, str],
    expected_query: Sequence[str],
    raw_path: Path,
) -> None:
    if set(value) != {
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
    }:
        raise ValueError("aggregate failure receipt fields differ")
    if (
        value["schema_version"] != FAILURE_SCHEMA
        or value["status"] != "TERMINAL_FAILURE"
        or value["intent_sha256"] != intent_sha256
        or value["attempt_index"] != attempt_index
        or value["slurm_job_id"] != job_id
        or value["attempt_ledger_sha256"] != attempt_ledger_sha256
        or value["row"] != dict(row)
        or value["query"] != list(expected_query)
        or value["retry_authorized"] is not True
        or not _terminal_failure(row)
    ):
        raise ValueError("aggregate retry failure evidence is invalid")
    _valid_utc(value["captured_at_utc"])
    raw_binding = value["raw_sacct"]
    if not isinstance(raw_binding, Mapping) or set(raw_binding) != {
        "filename",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("aggregate failure raw sacct binding is invalid")
    _canonical_file(raw_path, name="aggregate failure raw sacct evidence")
    raw = raw_path.read_bytes()
    if (
        raw_binding["filename"] != raw_path.name
        or raw_binding["sha256"] != _sha256_bytes(raw)
        or raw_binding["size_bytes"] != len(raw)
    ):
        raise ValueError("aggregate failure raw sacct binding differs")
    parsed = _parse_single_sacct_raw(
        raw,
        expected_job_id=job_id,
        expected_name=str(row["JobName"]),
        expected_partition=str(row["Partition"]),
        expected_walltime=str(row["Timelimit"]),
    )
    if parsed != dict(row):
        raise ValueError("aggregate failure parsed row differs from raw sacct")


def _attempt_paths(registry: Path) -> list[Path]:
    attempts = registry / "attempts"
    result = sorted(attempts.glob("attempt-*.json"))
    expected = [attempts / f"attempt-{index:04d}.json" for index in range(1, len(result) + 1)]
    if result != expected:
        raise ValueError("aggregate attempt registry is non-contiguous")
    return result


def verify_aggregate_submission_registry(
    registry: Path,
    *,
    expected_intent_sha256: str,
    expected_attempt_index: int,
    expected_job_id: str,
    expected_project_root: Path,
    expected_repository_root: Path,
    expected_output: Path,
    expected_workload_export_sha256: str,
) -> dict[str, object]:
    """Deep-verify the CPU job's own held-and-ledgered submission."""

    project_root = _canonical_directory(
        expected_project_root,
        name="aggregate project root",
    )
    repository_root = _canonical_directory(
        expected_repository_root,
        name="aggregate repository root",
    )
    output = expected_output.absolute()
    if output.parent != project_root / "aggregates":
        raise ValueError("aggregate output leaves the project aggregates namespace")
    expected_registry = (
        project_root
        / "runs"
        / "phase2-post-recovery-aggregate-attempts"
        / output.name
        / "submission-registry"
    )
    registry = _canonical_directory(registry, name="aggregate submission registry")
    if registry != expected_registry:
        raise ValueError("aggregate submission registry path differs")
    intent, intent_sha256 = _strict_json(
        registry / "intent.json",
        name="aggregate submission intent",
    )
    if (
        intent_sha256
        != _digest(
            expected_intent_sha256,
            name="expected aggregate intent SHA256",
        )
        or intent.get("schema_version") != INTENT_SCHEMA
        or intent.get("project_root") != os.fspath(project_root)
        or intent.get("repository_root") != os.fspath(repository_root)
        or intent.get("final_output") != os.fspath(output)
        or intent.get("workload_export_spec_sha256")
        != _digest(
            expected_workload_export_sha256,
            name="expected workload export SHA256",
        )
    ):
        raise ValueError("aggregate submission intent binding differs")
    script_value = intent.get("sbatch_script")
    if not isinstance(script_value, Mapping) or set(script_value) != {
        "repo_relative_path",
        "sha256",
        "git_blob_sha1",
        "size_bytes",
        "git_object",
        "evidence_filename",
        "transport",
    }:
        raise TypeError("aggregate submission sbatch binding must be a mapping")
    script_relative = str(script_value["repo_relative_path"])
    if (
        script_relative != "scripts/hpc4/phase2_post_recovery_aggregate.sbatch"
        or script_value["git_object"] != f"{intent['aggregator_git_commit']}:{script_relative}"
        or script_value["evidence_filename"] != SCRIPT_EVIDENCE_FILENAME
        or script_value["transport"] != SCRIPT_TRANSPORT
    ):
        raise ValueError("aggregate submission sbatch identity differs")
    committed_file = _canonical_file(
        registry / SCRIPT_EVIDENCE_FILENAME,
        name="aggregate committed sbatch evidence",
    )
    committed_script = _validate_script_bytes(
        committed_file.read_bytes(),
        name="aggregate committed sbatch evidence",
    )
    committed_git = _run(
        (
            "git",
            "-C",
            os.fspath(repository_root),
            "cat-file",
            "blob",
            str(script_value["git_object"]),
        ),
        name="committed aggregate sbatch source query",
        text=False,
        require_empty_stderr=True,
    )
    assert isinstance(committed_git, bytes)
    if (
        committed_git != committed_script
        or script_value["sha256"] != _sha256_bytes(committed_script)
        or script_value["git_blob_sha1"] != _git_blob_sha1(committed_script)
        or script_value["size_bytes"] != len(committed_script)
    ):
        raise ValueError("aggregate committed sbatch evidence differs from Git")
    log_root = project_root / "slurm-logs" / "phase2-post-recovery-aggregate" / output.name
    attempt_paths = _attempt_paths(registry)
    if len(attempt_paths) != expected_attempt_index:
        raise ValueError("running CPU job is not the latest authorized aggregate attempt")
    registered: list[tuple[int, str, str]] = []
    for index, registered_path in enumerate(attempt_paths, start=1):
        registered_attempt, registered_sha256 = _strict_json(
            registered_path,
            name=f"aggregate submission attempt {index}",
        )
        scheduler_export = (
            f"{intent['workload_export_spec']}"
            f",PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY={registry}"
            f",PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256={intent_sha256}"
            f",PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX={index}"
            ",PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256="
            f"{intent['workload_export_spec_sha256']}"
        )
        registered_job_id = _validate_attempt(
            registered_attempt,
            intent=intent,
            intent_sha256=intent_sha256,
            expected_index=index,
            committed_script=committed_script,
            expected_submission_command=_sbatch_command(
                intent=intent,
                intent_sha256=intent_sha256,
                attempt_index=index,
                scheduler_export_spec=scheduler_export,
                repository_root=repository_root,
                log_root=log_root,
            ),
            repository_root=repository_root,
            registry=registry,
        )
        registered.append((index, registered_job_id, registered_sha256))
    controller_root = _canonical_directory(
        registry / CONTROLLER_READBACK_DIRECTORY,
        name="aggregate controller readback registry",
    )
    expected_controller_names = {
        f"attempt-{index:04d}.sbatch" for index in range(1, expected_attempt_index + 1)
    }
    observed_controller_names = {
        child.name for child in controller_root.iterdir() if not child.name.startswith(".")
    }
    if observed_controller_names != expected_controller_names:
        raise ValueError("aggregate controller readback registry is non-contiguous")
    attempt_path = registry / "attempts" / f"attempt-{expected_attempt_index:04d}.json"
    attempt, attempt_sha256 = _strict_json(
        attempt_path,
        name="aggregate submission attempt ledger",
    )
    job_id = _validate_attempt(
        attempt,
        intent=intent,
        intent_sha256=intent_sha256,
        expected_index=expected_attempt_index,
        committed_script=committed_script,
        expected_submission_command=_sbatch_command(
            intent=intent,
            intent_sha256=intent_sha256,
            attempt_index=expected_attempt_index,
            scheduler_export_spec=str(attempt["scheduler_export_spec"]),
            repository_root=repository_root,
            log_root=log_root,
        ),
        repository_root=repository_root,
        registry=registry,
    )
    if job_id != expected_job_id:
        raise ValueError("running CPU job is not its registered aggregate attempt")
    failure_entries: list[dict[str, object]] = []
    expected_failure_names: set[str] = set()
    for index, previous_job_id, previous_attempt_sha256 in registered[:-1]:
        failure_path = registry / "failures" / f"job-{previous_job_id}.json"
        failure_raw_path = registry / "failures" / f"job-{previous_job_id}.sacct.psv"
        failure, failure_sha256 = _strict_json(
            failure_path,
            name=f"aggregate attempt {index} terminal failure",
        )
        row = failure.get("row")
        if not isinstance(row, Mapping):
            raise TypeError("aggregate terminal failure row must be a mapping")
        _validate_failure(
            failure,
            intent_sha256=intent_sha256,
            attempt_index=index,
            job_id=previous_job_id,
            attempt_ledger_sha256=previous_attempt_sha256,
            row={str(key): str(value) for key, value in row.items()},
            expected_query=_failure_sacct_query(previous_job_id),
            raw_path=failure_raw_path,
        )
        expected_failure_names.add(failure_path.name)
        failure_entries.append(
            {
                "attempt_index": index,
                "slurm_job_id": previous_job_id,
                "attempt_ledger_sha256": previous_attempt_sha256,
                "filename": failure_path.name,
                "sha256": failure_sha256,
            }
        )
    observed_failure_names = {
        path.name
        for path in (registry / "failures").iterdir()
        if path.name.startswith("job-") and path.suffix == ".json"
    }
    if observed_failure_names != expected_failure_names:
        raise ValueError("aggregate failure registry does not match prior attempts")
    expected_failure_raw_names = {
        name.removesuffix(".json") + ".sacct.psv" for name in expected_failure_names
    }
    observed_failure_raw_names = {
        path.name
        for path in (registry / "failures").iterdir()
        if path.name.startswith("job-") and path.name.endswith(".sacct.psv")
    }
    if observed_failure_raw_names != expected_failure_raw_names:
        raise ValueError("aggregate raw failure registry does not match prior attempts")
    visible_failure_names = {
        path.name for path in (registry / "failures").iterdir() if not path.name.startswith(".")
    }
    if visible_failure_names != expected_failure_names | expected_failure_raw_names:
        raise ValueError("aggregate failure registry contains unknown files")
    failure_chain_raw = _canonical_json(failure_entries)
    return {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "verified",
        "intent": intent,
        "intent_path": registry / "intent.json",
        "intent_sha256": intent_sha256,
        "attempt": attempt,
        "attempt_path": attempt_path,
        "attempt_sha256": attempt_sha256,
        "attempt_index": expected_attempt_index,
        "slurm_job_id": job_id,
        "failure_entries": failure_entries,
        "failure_chain_raw": failure_chain_raw,
        "failure_chain_sha256": _sha256_bytes(failure_chain_raw),
        "script_path": committed_file,
        "script_sha256": _sha256_bytes(committed_script),
        "script_size_bytes": len(committed_script),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pilot-phase", choices=("calibration", "freeze"), required=True)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--pilot-array-job-id", required=True)
    parser.add_argument("--aggregator-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("amd", "intel"), required=True)
    parser.add_argument("--walltime", required=True)
    parser.add_argument("--export-spec", required=True)
    parser.add_argument("--sbatch-script", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = _canonical_directory(arguments.project_root, name="project root")
    repository_root = _canonical_directory(
        arguments.repo_root,
        name="repository root",
    )
    output = arguments.output.absolute()
    if output.parent != project_root / "aggregates":
        raise ValueError("aggregate output must use the exact aggregates namespace")
    script = _canonical_file(arguments.sbatch_script, name="aggregate sbatch script")
    script_relative = script.relative_to(repository_root).as_posix()
    committed = _run(
        (
            "git",
            "-C",
            os.fspath(repository_root),
            "cat-file",
            "blob",
            f"{arguments.aggregator_git_commit}:{script_relative}",
        ),
        name="committed aggregate sbatch source query",
        text=False,
        require_empty_stderr=True,
    )
    assert isinstance(committed, bytes)
    committed = _validate_script_bytes(committed, name="committed aggregate sbatch script")
    identity = f"{arguments.pilot_phase}\0{arguments.design_sha256}\0{output}".encode()
    job_name = f"prorm-p2-post-agg-{arguments.design_sha256[:12]}-{_sha256_bytes(identity)[:10]}"
    user = _effective_user()
    intent_template = _intent_payload(
        pilot_phase=arguments.pilot_phase,
        design_sha256=arguments.design_sha256,
        pilot_array_job_id=arguments.pilot_array_job_id,
        aggregator_git_commit=arguments.aggregator_git_commit,
        project_root=project_root,
        repository_root=repository_root,
        output=output,
        partition=arguments.partition,
        walltime=arguments.walltime,
        workload_export_spec=arguments.export_spec,
        script_relative=script_relative,
        script_sha256=_sha256_bytes(committed),
        script_git_blob_sha1=_git_blob_sha1(committed),
        script_size_bytes=len(committed),
        submitter_user=user,
        job_name=job_name,
        created_at_utc=_utc_now(),
    )
    root = _ensure_directory(
        project_root / "runs" / "phase2-post-recovery-aggregate-attempts" / output.name,
        root=project_root,
        name="aggregate attempt root",
    )
    registry = _ensure_directory(
        root / "submission-registry",
        root=project_root,
        name="aggregate submission registry",
    )
    attempts_root = _ensure_directory(
        registry / "attempts",
        root=project_root,
        name="aggregate submission attempts",
    )
    failures_root = _ensure_directory(
        registry / "failures",
        root=project_root,
        name="aggregate submission failures",
    )
    controller_root = _ensure_directory(
        registry / CONTROLLER_READBACK_DIRECTORY,
        root=project_root,
        name="aggregate controller readback registry",
    )
    log_root = _ensure_directory(
        project_root / "slurm-logs" / "phase2-post-recovery-aggregate" / output.name,
        root=project_root,
        name="aggregate Slurm logs",
    )
    lock_descriptor = os.open(
        registry / "LOCK",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise ValueError("aggregate submission lock is not a regular file")
        import fcntl

        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        intent_path = registry / "intent.json"
        if intent_path.exists() or intent_path.is_symlink():
            intent, intent_sha256 = _strict_json(
                intent_path,
                name="aggregate submission intent",
            )
            _validate_intent(intent, expected=intent_template)
        else:
            intent = dict(intent_template)
            intent_sha256 = _write_exclusive(
                intent_path,
                intent,
                name="aggregate submission intent",
            )
        committed_evidence = registry / SCRIPT_EVIDENCE_FILENAME
        if committed_evidence.exists() or committed_evidence.is_symlink():
            _canonical_file(
                committed_evidence,
                name="aggregate committed sbatch evidence",
            )
            if committed_evidence.read_bytes() != committed:
                raise ValueError("aggregate committed sbatch evidence changed")
        else:
            _write_bytes_exclusive(
                committed_evidence,
                committed,
                name="aggregate committed sbatch evidence",
            )
        attempt_files = _attempt_paths(registry)
        registered: list[tuple[int, str, dict[str, object], str]] = []
        for index, attempt_path in enumerate(attempt_files, start=1):
            attempt, attempt_sha256 = _strict_json(
                attempt_path,
                name=f"aggregate attempt {index}",
            )
            registered_export = (
                f"{intent['workload_export_spec']}"
                f",PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY={registry}"
                f",PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256={intent_sha256}"
                f",PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX={index}"
                ",PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256="
                f"{intent['workload_export_spec_sha256']}"
            )
            job_id = _validate_attempt(
                attempt,
                intent=intent,
                intent_sha256=intent_sha256,
                expected_index=index,
                committed_script=committed,
                expected_submission_command=_sbatch_command(
                    intent=intent,
                    intent_sha256=intent_sha256,
                    attempt_index=index,
                    scheduler_export_spec=registered_export,
                    repository_root=repository_root,
                    log_root=log_root,
                ),
                repository_root=repository_root,
                registry=registry,
            )
            registered.append((index, job_id, attempt, attempt_sha256))
        expected_controller_names = {
            f"attempt-{index:04d}.sbatch" for index in range(1, len(registered) + 1)
        }
        next_controller_name = f"attempt-{len(registered) + 1:04d}.sbatch"
        observed_controller_names = {
            child.name for child in controller_root.iterdir() if not child.name.startswith(".")
        }
        if not expected_controller_names.issubset(
            observed_controller_names
        ) or not observed_controller_names.issubset(
            expected_controller_names | {next_controller_name}
        ):
            raise ValueError("aggregate controller readback registry is non-contiguous")
        controller_residue = observed_controller_names - expected_controller_names
        live_ids = _squeue_ids(job_name, user)
        rows = _sacct_rows(
            job_name=job_name,
            starttime=_valid_utc(intent["created_at_utc"])[:-1],
            intent=intent,
        )
        registered_ids = {item[1] for item in registered}
        unknown_ids = (set(live_ids) | set(rows)) - registered_ids
        next_index = len(registered) + 1
        retry_allowed = not registered
        if registered:
            latest_index, latest_job_id, _, latest_sha256 = registered[-1]
            earlier = registered[:-1]
            for index, job_id, _, previous_attempt_sha256 in earlier:
                row = rows.get(job_id)
                failure_path = failures_root / f"job-{job_id}.json"
                failure_raw_path = failures_root / f"job-{job_id}.sacct.psv"
                if row is None or not failure_path.exists():
                    raise RuntimeError("earlier aggregate attempt lacks terminal failure evidence")
                failure, _ = _strict_json(
                    failure_path,
                    name="aggregate attempt failure evidence",
                )
                _validate_failure(
                    failure,
                    intent_sha256=intent_sha256,
                    attempt_index=index,
                    job_id=job_id,
                    attempt_ledger_sha256=previous_attempt_sha256,
                    row=row,
                    expected_query=_failure_sacct_query(job_id),
                    raw_path=failure_raw_path,
                )
            if latest_job_id in live_ids:
                if unknown_ids:
                    raise RuntimeError("unregistered aggregate job collides with active attempt")
                raw = _run(
                    ("scontrol", "show", "job", "--oneliner", latest_job_id),
                    name="registered aggregate attempt query",
                )
                assert isinstance(raw, str)
                state, _ = _parse_scontrol(
                    raw,
                    job_id=latest_job_id,
                    intent=intent,
                    intent_sha256=intent_sha256,
                    attempt_index=latest_index,
                    repository_root=repository_root,
                )
                if state == "HELD":
                    _verify_release_collision_snapshot(
                        job_name=job_name,
                        user=user,
                        intent=intent,
                        intent_sha256=intent_sha256,
                        attempt_index=latest_index,
                        job_id=latest_job_id,
                        registered_job_ids=registered_ids,
                        registry=registry,
                        committed_script=committed,
                        repository_root=repository_root,
                    )
                    _run(
                        ("scontrol", "release", latest_job_id),
                        name="registered aggregate attempt release",
                    )
                print(f"{latest_job_id};hpc4;{latest_index};{intent_sha256};{latest_sha256}")
                return 0
            latest_row = rows.get(latest_job_id)
            if latest_row is None:
                raise RuntimeError("registered aggregate attempt is absent from squeue and sacct")
            if _successful(latest_row):
                if unknown_ids:
                    raise RuntimeError("replacement exists after successful aggregate attempt")
                print(f"{latest_job_id};hpc4;{latest_index};{intent_sha256};{latest_sha256}")
                return 0
            if not _terminal_failure(latest_row):
                raise RuntimeError(
                    "registered aggregate attempt is not exact success or terminal failure"
                )
            failure_path = failures_root / f"job-{latest_job_id}.json"
            failure_raw_path = failures_root / f"job-{latest_job_id}.sacct.psv"
            failure_query, failure_raw, failure_row = _query_single_failure_row(
                job_id=latest_job_id,
                intent=intent,
            )
            if failure_row != latest_row:
                raise RuntimeError("aggregate failure row differs between locked scheduler queries")
            if failure_path.exists() or failure_path.is_symlink():
                failure, _ = _strict_json(
                    failure_path,
                    name="latest aggregate failure evidence",
                )
                _validate_failure(
                    failure,
                    intent_sha256=intent_sha256,
                    attempt_index=latest_index,
                    job_id=latest_job_id,
                    attempt_ledger_sha256=latest_sha256,
                    row=failure_row,
                    expected_query=failure_query,
                    raw_path=failure_raw_path,
                )
            else:
                if failure_raw_path.exists() or failure_raw_path.is_symlink():
                    _canonical_file(
                        failure_raw_path,
                        name="aggregate failure raw sacct evidence",
                    )
                    if failure_raw_path.read_bytes() != failure_raw:
                        raise ValueError("existing aggregate failure raw sacct bytes differ")
                else:
                    _write_bytes_exclusive(
                        failure_raw_path,
                        failure_raw,
                        name="aggregate failure raw sacct evidence",
                    )
                _write_exclusive(
                    failure_path,
                    _failure_payload(
                        intent_sha256=intent_sha256,
                        attempt_index=latest_index,
                        job_id=latest_job_id,
                        attempt_ledger_sha256=latest_sha256,
                        row=failure_row,
                        query=failure_query,
                        raw_filename=failure_raw_path.name,
                        raw_sha256=_sha256_bytes(failure_raw),
                        raw_size_bytes=len(failure_raw),
                    ),
                    name="aggregate terminal failure evidence",
                )
            retry_allowed = True
        if len(unknown_ids) > 1:
            raise RuntimeError("multiple unregistered aggregate scheduler jobs exist")
        if unknown_ids:
            orphan_id = next(iter(unknown_ids))
            if not retry_allowed or orphan_id not in live_ids:
                raise RuntimeError("historical unregistered aggregate job forbids retry")
            raw = _run(
                ("scontrol", "show", "job", "--oneliner", orphan_id),
                name="orphan aggregate attempt query",
            )
            assert isinstance(raw, str)
            state, scheduler = _parse_scontrol(
                raw,
                job_id=orphan_id,
                intent=intent,
                intent_sha256=intent_sha256,
                attempt_index=next_index,
                repository_root=repository_root,
            )
            if state != "HELD" or scheduler is None:
                raise RuntimeError("unregistered aggregate attempt was externally released")
            job_id = orphan_id
            scheduler_export = (
                f"{intent['workload_export_spec']}"
                f",PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY={registry}"
                f",PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256={intent_sha256}"
                f",PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX={next_index}"
                ",PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256="
                f"{intent['workload_export_spec_sha256']}"
            )
            submission_command = _sbatch_command(
                intent=intent,
                intent_sha256=intent_sha256,
                attempt_index=next_index,
                scheduler_export_spec=scheduler_export,
                repository_root=repository_root,
                log_root=log_root,
            )
        else:
            if not retry_allowed:
                raise RuntimeError("aggregate retry is not authorized")
            if controller_residue:
                raise RuntimeError(
                    "aggregate controller readback exists without a scheduler attempt"
                )
            scheduler_export = (
                f"{intent['workload_export_spec']}"
                f",PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY={registry}"
                f",PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256={intent_sha256}"
                f",PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX={next_index}"
                ",PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256="
                f"{intent['workload_export_spec_sha256']}"
            )
            submission_command = _sbatch_command(
                intent=intent,
                intent_sha256=intent_sha256,
                attempt_index=next_index,
                scheduler_export_spec=scheduler_export,
                repository_root=repository_root,
                log_root=log_root,
            )
            submitted = _run(
                submission_command,
                name="held aggregate attempt submission",
                text=False,
                input_bytes=committed,
                require_empty_stderr=True,
            )
            assert isinstance(submitted, bytes)
            try:
                submitted_text = submitted.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError("aggregate sbatch returned non-UTF-8 output") from error
            if (
                not submitted_text.endswith("\n")
                or submitted_text.count("\n") != 1
                or "\r" in submitted_text
            ):
                raise RuntimeError("aggregate sbatch returned non-canonical output")
            job_id, separator, cluster = submitted_text[:-1].partition(";")
            if (
                re.fullmatch(r"[1-9][0-9]*", job_id) is None
                or separator != ";"
                or cluster != "hpc4"
            ):
                raise RuntimeError("aggregate sbatch returned an invalid identity")
            raw = _run(
                ("scontrol", "show", "job", "--oneliner", job_id),
                name="new held aggregate attempt query",
            )
            assert isinstance(raw, str)
            state, scheduler = _parse_scontrol(
                raw,
                job_id=job_id,
                intent=intent,
                intent_sha256=intent_sha256,
                attempt_index=next_index,
                repository_root=repository_root,
            )
            if state != "HELD" or scheduler is None:
                raise RuntimeError("new aggregate attempt was not held")
        batch_script = _capture_controller_batch_script(
            registry=registry,
            attempt_index=next_index,
            job_id=job_id,
            submission_command=submission_command,
            committed_script=committed,
        )
        attempt = _attempt_payload(
            intent=intent,
            intent_sha256=intent_sha256,
            attempt_index=next_index,
            job_id=job_id,
            scheduler_export_spec=scheduler_export,
            scheduler_request=scheduler,
            batch_script=batch_script,
        )
        attempt_path = attempts_root / f"attempt-{next_index:04d}.json"
        attempt_sha256 = _write_exclusive(
            attempt_path,
            attempt,
            name="aggregate submission attempt ledger",
        )
        installed, installed_sha256 = _strict_json(
            attempt_path,
            name="aggregate submission attempt ledger",
        )
        _validate_attempt(
            installed,
            intent=intent,
            intent_sha256=intent_sha256,
            expected_index=next_index,
            committed_script=committed,
            expected_submission_command=submission_command,
            repository_root=repository_root,
            registry=registry,
        )
        if installed_sha256 != attempt_sha256:
            raise ValueError("aggregate attempt ledger changed after fsync")
        _verify_release_collision_snapshot(
            job_name=job_name,
            user=user,
            intent=intent,
            intent_sha256=intent_sha256,
            attempt_index=next_index,
            job_id=job_id,
            registered_job_ids=registered_ids,
            registry=registry,
            committed_script=committed,
            repository_root=repository_root,
        )
        _run(
            ("scontrol", "release", job_id),
            name="ledger-bound aggregate attempt release",
        )
        print(f"{job_id};hpc4;{next_index};{intent_sha256};{attempt_sha256}")
        return 0
    finally:
        try:
            import fcntl

            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
