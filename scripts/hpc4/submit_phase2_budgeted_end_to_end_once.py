#!/usr/bin/env python3
"""Submit the fixed five-seed budgeted Phase-2 array exactly once.

This is an exploratory-only control plane.  It is deliberately independent of
the formal fixed-wave and post-recovery pilot submission registries.  A new
request is submitted held, its complete scheduler identity is verified, and
immutable intent and submission ledgers are installed and fsync'd before the
request can be released.

The deterministic job name plus strict squeue/sacct collision scans recover a
crash-created held orphan.  A historical orphan is never replaced: absence of
recoverable held scheduler state is a fail-closed terminal condition.
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
from datetime import datetime, timezone
from pathlib import Path

INTENT_SCHEMA = "prorm-phase2-budgeted-end-to-end-array-intent/v1"
SUBMISSION_SCHEMA = "prorm-phase2-budgeted-end-to-end-array-submission/v1"
SCHEDULER_REQUEST_SCHEMA = "prorm-phase2-budgeted-end-to-end-held-scheduler-request/v2"
ORDERED_SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005)
ARRAY_SPEC = "0-4%2"
SBATCH_SCRIPT_RELATIVE = "scripts/hpc4/phase2_budgeted_end_to_end.sbatch"
OPTIMIZER_SCHEDULE_SHA256 = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
SCHEDULER_COMMENT_PREFIX = "prorm-budgeted:"
COLLISION_SEARCH_START = "2026-01-01T00:00:00"

_HEX = frozenset("0123456789abcdef")
_EXPECTED_TRES = {
    "billing": "8",
    "cpu": "8",
    "gres/gpu": "1",
    "mem": "96G",
    "node": "1",
}
_EXPECTED_ALLOC_TRES = {
    **_EXPECTED_TRES,
    "gres/gpu:l20": "1",
}
_SQUEUE_FORMAT = "%F|%K|%j|%u|%P|%q"
_SACCT_FORMAT = (
    "JobIDRaw%128,JobID%128,JobName%128,User%128,Cluster%64,"
    "Account%64,Partition%64,QOS%64,State%64,"
    "Submit%32,Timelimit%32,ReqTRES%256,AllocTRES%256"
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
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _scheduler_comment(export_spec_sha256: object) -> str:
    """Return the only scheduler comment admissible for orphan adoption."""

    return SCHEDULER_COMMENT_PREFIX + _digest(
        export_spec_sha256,
        name="scheduler export specification SHA256",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _valid_utc(value: object, *, name: str = "created_at_utc") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UTC timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"{name} must be a UTC timestamp") from error
    return value


def _effective_user() -> str:
    """Resolve the effective uid rather than trusting ambient user aliases."""

    import pwd

    value = pwd.getpwuid(os.geteuid()).pw_name
    if re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise ValueError("effective submitter user is invalid")
    return value


def _canonical_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    metadata = absolute.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical real directory")
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
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} leaves the project root") from error
    absolute.mkdir(parents=True, exist_ok=True)
    current = root
    for component in relative.parts:
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


def _write_exclusive(path: Path, value: Mapping[str, object], *, name: str) -> str:
    """Install canonical JSON without an overwrite window and fsync its parent."""

    raw = _canonical_json(value)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite {name}: {path}") from None
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_bytes(raw)


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
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
    timeout: int = 60,
) -> str:
    environment = os.environ.copy()
    environment["SLURM_TIME_FORMAT"] = "standard"
    completed = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise RuntimeError(f"{name} failed{': ' + detail if detail else ''}")
    return completed.stdout


def _run_bytes(
    arguments: Sequence[str],
    *,
    name: str,
    timeout: int = 60,
) -> bytes:
    environment = os.environ.copy()
    environment["SLURM_TIME_FORMAT"] = "standard"
    completed = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        env=environment,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{name} failed{': ' + detail if detail else ''}")
    return completed.stdout


def _parse_tres(raw: str, *, name: str, allow_empty: bool = False) -> dict[str, str]:
    if raw == "" and allow_empty:
        return {}
    if not raw or any(character in raw for character in "\r\n\x00"):
        raise ValueError(f"{name} is unsafe")
    result: dict[str, str] = {}
    for entry in raw.split(","):
        if entry.count("=") != 1:
            raise ValueError(f"{name} has an invalid entry")
        key, value = entry.split("=", 1)
        if (
            not key
            or not value
            or key in result
            or re.fullmatch(r"[A-Za-z0-9_./:-]+", key) is None
            or re.fullmatch(r"[A-Za-z0-9_.:/+-]+", value) is None
        ):
            raise ValueError(f"{name} has an invalid or duplicate entry")
        result[key] = value
    return result


def _intent_payload(
    *,
    design_sha256: str,
    base_config_hash: str,
    authorization_sha256: str,
    optimizer_schedule_sha256: str,
    git_commit: str,
    image_sha256: str,
    inventory_sha256: str,
    overlay_sha256: str,
    base_file_sha256: str,
    sbatch_script_sha256: str,
    export_spec: str,
    export_spec_sha256: str,
    walltime: str,
    job_name: str,
    project_root: str,
    repository_root: str,
    submitter_user: str,
    created_at_utc: str,
) -> dict[str, object]:
    for value, name, lengths in (
        (design_sha256, "design_sha256", frozenset({64})),
        (base_config_hash, "base_config_hash", frozenset({64})),
        (authorization_sha256, "authorization_sha256", frozenset({64})),
        (optimizer_schedule_sha256, "optimizer_schedule_sha256", frozenset({64})),
        (git_commit, "git_commit", frozenset({40, 64})),
        (image_sha256, "image_sha256", frozenset({64})),
        (inventory_sha256, "inventory_sha256", frozenset({64})),
        (overlay_sha256, "overlay_sha256", frozenset({64})),
        (base_file_sha256, "base_file_sha256", frozenset({64})),
        (sbatch_script_sha256, "sbatch_script_sha256", frozenset({64})),
        (export_spec_sha256, "export_spec_sha256", frozenset({64})),
    ):
        _digest(value, name=name, lengths=lengths)
    if optimizer_schedule_sha256 != OPTIMIZER_SCHEDULE_SHA256:
        raise ValueError("optimizer schedule is not the adopted recovery schedule")
    if (
        not export_spec
        or "\n" in export_spec
        or "\r" in export_spec
        or "\x00" in export_spec
        or _sha256_bytes(export_spec.encode("utf-8")) != export_spec_sha256
    ):
        raise ValueError("export specification does not match its SHA256")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", job_name) is None:
        raise ValueError("deterministic Slurm job name is invalid")
    if re.fullmatch(r"[A-Za-z0-9._-]+", submitter_user) is None:
        raise ValueError("submitter user is invalid")
    project = Path(project_root)
    repository = Path(repository_root)
    if (
        not project.is_absolute()
        or project == Path("/")
        or any(character in project_root for character in "\r\n\x00")
        or not repository.is_absolute()
        or repository == Path("/")
        or any(character in repository_root for character in "\r\n\x00")
    ):
        raise ValueError("project or repository root is invalid")
    if (
        re.fullmatch(
            r"(?:[1-9][0-9]*-[0-9]{2}:[0-9]{2}:[0-9]{2}|[0-9]{2}:[0-9]{2}:[0-9]{2})",
            walltime,
        )
        is None
    ):
        raise ValueError("walltime is invalid")
    return {
        "schema_version": INTENT_SCHEMA,
        "status": "committed_while_scheduler_held",
        "experiment_stage": "budgeted_end_to_end",
        "formal_eligibility": False,
        "supports_formal_claim": False,
        "evidence_role": "budgeted_end_to_end_exploratory_only",
        "phase2_design_sha256": design_sha256,
        "base_config_hash": base_config_hash,
        "recovery_authorization_sha256": authorization_sha256,
        "optimizer_schedule_sha256": optimizer_schedule_sha256,
        "git_commit": git_commit,
        "image_sha256": image_sha256,
        "hf_inventory_sha256": inventory_sha256,
        "phase2_overlay_sha256": overlay_sha256,
        "phase2_base_sha256": base_file_sha256,
        "sbatch_script": {
            "repo_relative_path": SBATCH_SCRIPT_RELATIVE,
            "sha256": sbatch_script_sha256,
        },
        "export_spec": export_spec,
        "export_spec_sha256": export_spec_sha256,
        "ordered_seeds": list(ORDERED_SEEDS),
        "array_spec": ARRAY_SPEC,
        "max_running_tasks": 2,
        "job_name": job_name,
        "project_root": project_root,
        "repository_root": repository_root,
        "submitter_user": submitter_user,
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "gpu-l20",
        "qos": "l20_qos",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 8,
        "memory": "96G",
        "gpus_per_node": 1,
        "walltime": walltime,
        "requeue": False,
        "same_design_resubmission_allowed": False,
        "replacement_array_allowed": False,
        "replacement_seed_allowed": False,
        "collision_search_start": COLLISION_SEARCH_START,
        "created_at_utc": _valid_utc(created_at_utc),
    }


def _validate_intent(
    value: Mapping[str, object],
    *,
    expected: Mapping[str, object],
) -> None:
    if set(value) != set(expected):
        raise ValueError("budgeted end-to-end intent fields differ")
    observed = dict(value)
    expected_value = dict(expected)
    observed_created = observed.pop("created_at_utc", None)
    expected_value.pop("created_at_utc", None)
    _valid_utc(observed_created)
    if observed != expected_value:
        raise ValueError("budgeted end-to-end intent identity differs")


def _parse_squeue_ids(
    raw: str,
    *,
    expected_name: str,
    expected_user: str,
) -> tuple[str, ...]:
    if "\r" in raw or "\x00" in raw:
        raise ValueError("squeue returned unsafe bytes")
    if not raw:
        return ()
    roots: set[str] = set()
    for line in raw.splitlines():
        if not line:
            raise ValueError("squeue returned an empty scheduler row")
        fields = line.split("|")
        if len(fields) != 6:
            raise ValueError("squeue row differs from the locked field set")
        job_id, task_expression, job_name, user, partition, qos = fields
        if (
            re.fullmatch(r"[1-9][0-9]*", job_id) is None
            or not _parse_array_task_set(task_expression)
            or job_name != expected_name
            or user != expected_user
            or partition != "gpu-l20"
            or qos != "l20_qos"
        ):
            raise ValueError("squeue scheduler identity differs from the intent")
        roots.add(job_id)
    return tuple(sorted(roots, key=int))


def _parse_array_task_set(raw: str) -> frozenset[int]:
    """Parse one Slurm array-index expression and require a fixed-wave subset."""

    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw or any(character in raw for character in "\r\n\x00"):
        raise ValueError("scheduler array task expression is unsafe")
    body, separator, throttle = raw.partition("%")
    if separator and throttle != "2":
        raise ValueError("scheduler array throttle differs from the fixed wave")
    if "%" in throttle:
        raise ValueError("scheduler array task expression has multiple throttles")
    tasks: set[int] = set()
    for component in body.split(","):
        if re.fullmatch(r"[0-4]", component):
            values: Sequence[int] = (int(component),)
        else:
            match = re.fullmatch(r"([0-4])-([0-4])", component)
            if match is None or int(match.group(1)) > int(match.group(2)):
                raise ValueError("scheduler array task expression is invalid")
            values = range(int(match.group(1)), int(match.group(2)) + 1)
        for value in values:
            if value in tasks:
                raise ValueError("scheduler array task expression repeats an index")
            tasks.add(value)
    if not tasks:
        raise ValueError("scheduler array task expression is empty")
    return frozenset(tasks)


def _array_root(value: str) -> str | None:
    match = re.fullmatch(
        r"([1-9][0-9]*)(?:_(?:[0-4]|\[(?:[0-4](?:[-,][0-4])*)%?2?\]))?",
        value,
    )
    return None if match is None else match.group(1)


def _parse_sacct_ids(
    raw: str,
    *,
    expected_name: str,
    expected_user: str,
    expected_walltime: str,
) -> tuple[str, ...]:
    if "\r" in raw or "\x00" in raw:
        raise ValueError("sacct returned unsafe bytes")
    if not raw:
        return ()
    roots: set[str] = set()
    for line in raw.splitlines():
        if not line:
            raise ValueError("sacct returned an empty scheduler row")
        fields = line.split("|")
        if len(fields) != 13:
            raise ValueError("sacct row differs from the locked field set")
        (
            raw_id,
            job_id,
            job_name,
            user,
            cluster,
            account,
            partition,
            qos,
            state,
            submitted,
            timelimit,
            req_tres_raw,
            alloc_tres_raw,
        ) = fields
        if (
            job_name != expected_name
            or user != expected_user
            or cluster != "hpc4"
            or account != "sigroup"
            or partition != "gpu-l20"
            or qos != "l20_qos"
            or re.fullmatch(r"[A-Z][A-Z_]*(?:\+)?(?: by [0-9]+)?", state) is None
            or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", submitted)
            is None
            or timelimit != expected_walltime
            or _parse_tres(req_tres_raw, name="sacct ReqTRES") != _EXPECTED_TRES
        ):
            raise ValueError("historical scheduler identity differs from the intent")
        allocated = _parse_tres(
            alloc_tres_raw,
            name="sacct AllocTRES",
            allow_empty=True,
        )
        if allocated not in ({}, _EXPECTED_ALLOC_TRES):
            raise ValueError("historical scheduler allocation differs from the intent")
        root = _array_root(job_id) or _array_root(raw_id)
        if root is None:
            raise ValueError("sacct returned an unrecognized array identity")
        roots.add(root)
    return tuple(sorted(roots, key=int))


def _parse_scontrol_records(
    raw: str,
    *,
    array_job_id: str,
    expected_name: str,
    expected_export_spec_sha256: str,
    expected_walltime: str,
    expected_command: Path,
    expected_workdir: Path,
    expected_user: str,
) -> tuple[str, dict[str, object] | None]:
    if "\r" in raw or "\x00" in raw:
        raise ValueError("scontrol returned unsafe bytes")
    lines = raw.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("scontrol returned no complete scheduler record")
    parsed: list[dict[str, str]] = []
    parsed_tres: list[dict[str, str]] = []
    record_identities: set[str] = set()
    expected_user_id = re.compile(rf"{re.escape(expected_user)}\([0-9]+\)")
    expected_comment = _scheduler_comment(expected_export_spec_sha256)
    for line in lines:
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                raise ValueError("scontrol returned a token outside key=value form")
            key, value = token.split("=", 1)
            if key in fields:
                raise ValueError(f"duplicate scontrol job field: {key}")
            fields[key] = value
        task_id = fields.get("ArrayTaskId")
        tres = _parse_tres(fields.get("TRES", ""), name="scontrol TRES")
        if (
            fields.get("ArrayJobId", fields.get("JobId")) != array_job_id
            or fields.get("JobName") != expected_name
            or fields.get("Comment") != expected_comment
            or expected_user_id.fullmatch(str(fields.get("UserId", ""))) is None
            or fields.get("Account") != "sigroup"
            or fields.get("Partition") != "gpu-l20"
            or fields.get("QOS") != "l20_qos"
            or fields.get("Requeue") != "0"
            or fields.get("Restarts") != "0"
            or fields.get("ArrayTaskThrottle") != "2"
            or fields.get("NumNodes") not in {"1", "1-1"}
            or fields.get("NumTasks") != "1"
            or fields.get("NumCPUs") != "8"
            or fields.get("CPUs/Task") != "8"
            or fields.get("MinMemoryNode") != "96G"
            or fields.get("TimeLimit") != expected_walltime
            or tres not in (_EXPECTED_TRES, _EXPECTED_ALLOC_TRES)
            or re.fullmatch(
                r"gres(?::|/)gpu(?::[A-Za-z0-9_.-]+)?:1",
                fields.get("TresPerNode", ""),
            )
            is None
            or fields.get("Command") != os.fspath(expected_command)
            or fields.get("WorkDir") != os.fspath(expected_workdir)
            or not fields.get("JobState")
        ):
            raise ValueError("Slurm job differs from the immutable budgeted end-to-end intent")
        _parse_array_task_set(str(task_id))
        record_identity = f"{fields.get('JobId')}|{task_id}"
        if record_identity in record_identities:
            raise ValueError("scontrol returned a duplicate job record")
        record_identities.add(record_identity)
        parsed.append(fields)
        parsed_tres.append(tres)

    held_master = (
        len(parsed) == 1
        and parsed[0].get("ArrayTaskId") == ARRAY_SPEC
        and parsed[0].get("JobState") == "PENDING"
        and parsed[0].get("Reason") == "JobHeldUser"
        and parsed_tres[0] == _EXPECTED_TRES
    )
    if held_master:
        fields = parsed[0]
        evidence = {
            "schema_version": SCHEDULER_REQUEST_SCHEMA,
            "captured_while_held": True,
            "raw_scontrol_record": raw,
            "raw_scontrol_sha256": _sha256_bytes(raw.encode("utf-8")),
            "normalized": {
                "array_job_id": array_job_id,
                "job_name": expected_name,
                "comment": expected_comment,
                "export_spec_sha256": expected_export_spec_sha256,
                "array_spec": ARRAY_SPEC,
                "array_task_throttle": 2,
                "cluster": "hpc4",
                "account": "sigroup",
                "partition": "gpu-l20",
                "qos": "l20_qos",
                "submitter_user": expected_user,
                "user_id": fields["UserId"],
                "nodes": 1,
                "tasks": 1,
                "cpus": 8,
                "cpus_per_task": 8,
                "memory": "96G",
                "gpus_per_node": 1,
                "walltime": expected_walltime,
                "tres": dict(_EXPECTED_TRES),
                "tres_per_node": fields["TresPerNode"],
                "requeue": False,
                "restarts": 0,
                "job_state": "PENDING",
                "reason": "JobHeldUser",
                "command": os.fspath(expected_command),
                "work_dir": os.fspath(expected_workdir),
            },
        }
        return "HELD", evidence
    if any(
        fields.get("JobState") == "PENDING" and str(fields.get("Reason", "")).startswith("JobHeld")
        for fields in parsed
    ):
        raise ValueError("array is held by an unexpected scheduler authority")
    return "ALREADY_RELEASED", None


def _submission_payload(
    *,
    intent: Mapping[str, object],
    intent_sha256: str,
    array_job_id: str,
    scheduler_request: Mapping[str, object],
    created_at_utc: str | None = None,
) -> dict[str, object]:
    if re.fullmatch(r"[1-9][0-9]*", array_job_id) is None:
        raise ValueError("array_job_id must be positive")
    return {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "committed_while_scheduler_held",
        "intent_sha256": _digest(intent_sha256, name="intent_sha256"),
        "experiment_stage": "budgeted_end_to_end",
        "formal_eligibility": False,
        "supports_formal_claim": False,
        "phase2_design_sha256": intent["phase2_design_sha256"],
        "base_config_hash": intent["base_config_hash"],
        "recovery_authorization_sha256": intent["recovery_authorization_sha256"],
        "optimizer_schedule_sha256": intent["optimizer_schedule_sha256"],
        "git_commit": intent["git_commit"],
        "image_sha256": intent["image_sha256"],
        "hf_inventory_sha256": intent["hf_inventory_sha256"],
        "phase2_overlay_sha256": intent["phase2_overlay_sha256"],
        "phase2_base_sha256": intent["phase2_base_sha256"],
        "ordered_seeds": list(ORDERED_SEEDS),
        "array_spec": ARRAY_SPEC,
        "array_job_id": array_job_id,
        "cluster": "hpc4",
        "scheduler_request": dict(scheduler_request),
        "same_design_resubmission_allowed": False,
        "replacement_array_allowed": False,
        "replacement_seed_allowed": False,
        "released_only_after_ledger_fsync": True,
        "created_at_utc": _valid_utc(created_at_utc or _utc_now()),
    }


def _validate_submission(
    value: Mapping[str, object],
    *,
    intent: Mapping[str, object],
    intent_sha256: str,
) -> str:
    expected_keys = set(
        _submission_payload(
            intent=intent,
            intent_sha256=intent_sha256,
            array_job_id="1",
            scheduler_request={},
            created_at_utc="2026-01-01T00:00:00Z",
        )
    )
    if set(value) != expected_keys:
        raise ValueError("budgeted end-to-end submission ledger fields differ")
    for field in (
        "phase2_design_sha256",
        "base_config_hash",
        "recovery_authorization_sha256",
        "optimizer_schedule_sha256",
        "git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "phase2_overlay_sha256",
        "phase2_base_sha256",
    ):
        if value.get(field) != intent[field]:
            raise ValueError(f"budgeted end-to-end submission {field} differs")
    array_job_id = value.get("array_job_id")
    if (
        value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("status") != "committed_while_scheduler_held"
        or value.get("intent_sha256") != intent_sha256
        or value.get("experiment_stage") != "budgeted_end_to_end"
        or value.get("formal_eligibility") is not False
        or value.get("supports_formal_claim") is not False
        or value.get("ordered_seeds") != list(ORDERED_SEEDS)
        or value.get("array_spec") != ARRAY_SPEC
        or re.fullmatch(r"[1-9][0-9]*", str(array_job_id)) is None
        or value.get("cluster") != "hpc4"
        or value.get("same_design_resubmission_allowed") is not False
        or value.get("replacement_array_allowed") is not False
        or value.get("replacement_seed_allowed") is not False
        or value.get("released_only_after_ledger_fsync") is not True
    ):
        raise ValueError("budgeted end-to-end submission policy is invalid")
    _valid_utc(value.get("created_at_utc"))

    scheduler = value.get("scheduler_request")
    if not isinstance(scheduler, Mapping) or set(scheduler) != {
        "schema_version",
        "captured_while_held",
        "raw_scontrol_record",
        "raw_scontrol_sha256",
        "normalized",
    }:
        raise ValueError("budgeted end-to-end scheduler request fields differ")
    normalized = scheduler.get("normalized")
    expected_normalized_fields = {
        "array_job_id",
        "job_name",
        "comment",
        "export_spec_sha256",
        "array_spec",
        "array_task_throttle",
        "cluster",
        "account",
        "partition",
        "qos",
        "submitter_user",
        "user_id",
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
        "job_state",
        "reason",
        "command",
        "work_dir",
    }
    if not isinstance(normalized, Mapping) or set(normalized) != expected_normalized_fields:
        raise ValueError("budgeted end-to-end normalized scheduler fields differ")
    script = intent.get("sbatch_script")
    repository_root = intent.get("repository_root")
    submitter_user = intent.get("submitter_user")
    if (
        not isinstance(script, Mapping)
        or set(script) != {"repo_relative_path", "sha256"}
        or script.get("repo_relative_path") != SBATCH_SCRIPT_RELATIVE
        or not isinstance(repository_root, str)
        or not isinstance(submitter_user, str)
    ):
        raise ValueError("budgeted end-to-end intent source identity is invalid")
    expected_command = Path(repository_root) / SBATCH_SCRIPT_RELATIVE
    export_spec_sha256 = _digest(
        intent.get("export_spec_sha256"),
        name="intent export_spec_sha256",
    )
    expected_normalized = {
        "array_job_id": str(array_job_id),
        "job_name": intent["job_name"],
        "comment": _scheduler_comment(export_spec_sha256),
        "export_spec_sha256": export_spec_sha256,
        "array_spec": ARRAY_SPEC,
        "array_task_throttle": 2,
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "gpu-l20",
        "qos": "l20_qos",
        "submitter_user": submitter_user,
        "nodes": 1,
        "tasks": 1,
        "cpus": 8,
        "cpus_per_task": 8,
        "memory": "96G",
        "gpus_per_node": 1,
        "walltime": intent["walltime"],
        "tres": dict(_EXPECTED_TRES),
        "requeue": False,
        "restarts": 0,
        "job_state": "PENDING",
        "reason": "JobHeldUser",
        "command": os.fspath(expected_command),
        "work_dir": repository_root,
    }
    for field, expected_value in expected_normalized.items():
        if normalized.get(field) != expected_value:
            raise ValueError(f"budgeted end-to-end scheduler {field} differs")
    if (
        re.fullmatch(
            rf"{re.escape(submitter_user)}\([0-9]+\)",
            str(normalized.get("user_id", "")),
        )
        is None
        or re.fullmatch(
            r"gres(?::|/)gpu(?::[A-Za-z0-9_.-]+)?:1",
            str(normalized.get("tres_per_node", "")),
        )
        is None
        or scheduler.get("schema_version") != SCHEDULER_REQUEST_SCHEMA
        or scheduler.get("captured_while_held") is not True
    ):
        raise ValueError("budgeted end-to-end held scheduler evidence is invalid")
    raw = scheduler.get("raw_scontrol_record")
    raw_sha256 = _digest(
        scheduler.get("raw_scontrol_sha256"),
        name="scheduler_request.raw_scontrol_sha256",
    )
    if not isinstance(raw, str) or _sha256_bytes(raw.encode("utf-8")) != raw_sha256:
        raise ValueError("budgeted end-to-end raw scheduler evidence changed")
    state, reparsed = _parse_scontrol_records(
        raw,
        array_job_id=str(array_job_id),
        expected_name=str(intent["job_name"]),
        expected_export_spec_sha256=export_spec_sha256,
        expected_walltime=str(intent["walltime"]),
        expected_command=expected_command,
        expected_workdir=Path(repository_root),
        expected_user=submitter_user,
    )
    if state != "HELD" or reparsed != dict(scheduler):
        raise ValueError("budgeted end-to-end scheduler evidence does not reparse exactly")
    return str(array_job_id)


def _squeue_ids(*, job_name: str, user: str) -> tuple[str, ...]:
    return _parse_squeue_ids(
        _run(
            (
                "squeue",
                "--noheader",
                f"--user={user}",
                f"--name={job_name}",
                f"--format={_SQUEUE_FORMAT}",
            ),
            name="budgeted end-to-end live collision query",
        ),
        expected_name=job_name,
        expected_user=user,
    )


def _sacct_ids_since(
    *,
    job_name: str,
    user: str,
    walltime: str,
) -> tuple[str, ...]:
    return _parse_sacct_ids(
        _run(
            (
                "sacct",
                "-X",
                "--clusters=hpc4",
                f"--starttime={COLLISION_SEARCH_START}",
                f"--user={user}",
                f"--name={job_name}",
                "--noheader",
                "--parsable2",
                f"--format={_SACCT_FORMAT}",
            ),
            name="budgeted end-to-end historical collision query",
        ),
        expected_name=job_name,
        expected_user=user,
        expected_walltime=walltime,
    )


def _collision_snapshot(
    *,
    job_name: str,
    user: str,
    walltime: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        _squeue_ids(job_name=job_name, user=user),
        _sacct_ids_since(job_name=job_name, user=user, walltime=walltime),
    )


def _exact_script(
    *,
    repo_root: Path,
    requested: Path,
    git_commit: str,
) -> tuple[Path, str]:
    expected = repo_root / SBATCH_SCRIPT_RELATIVE
    script = _canonical_file(requested, name="budgeted end-to-end sbatch script")
    if script != expected:
        raise ValueError("budgeted end-to-end submission requires the exact locked sbatch path")
    committed = _run_bytes(
        (
            "git",
            "-C",
            os.fspath(repo_root),
            "cat-file",
            "blob",
            f"{git_commit}:{SBATCH_SCRIPT_RELATIVE}",
        ),
        name="committed budgeted end-to-end sbatch source query",
    )
    local = script.read_bytes()
    if committed != local:
        raise ValueError("budgeted end-to-end sbatch script differs from the submitted Git commit")
    return script, _sha256_bytes(local)


def verify_submission_ledger(
    ledger_path: Path,
    *,
    project_root: Path,
    repo_root: Path,
    design_sha256: str,
    base_config_hash: str,
    authorization_sha256: str,
    optimizer_schedule_sha256: str,
    git_commit: str,
    image_sha256: str,
    inventory_sha256: str,
    overlay_sha256: str,
    base_file_sha256: str,
    export_spec_sha256: str,
    array_job_id: str,
    submitter_user: str,
) -> dict[str, object]:
    """Deep-verify an immutable budgeted submission ledger without Slurm."""

    _digest(design_sha256, name="design_sha256")
    _digest(git_commit, name="git_commit", lengths=frozenset({40, 64}))
    project = _canonical_directory(project_root, name="project root")
    repository = _canonical_directory(repo_root, name="repository root")
    expected_ledger = (
        project / "runs" / "phase2-budgeted-end-to-end" / design_sha256 / "submission-ledger"
    )
    ledger = _canonical_directory(ledger_path, name="submission ledger")
    if ledger != expected_ledger:
        raise ValueError("submission ledger path differs from the design identity")
    intent, intent_sha256 = _strict_json(
        ledger / "intent.json",
        name="budgeted end-to-end intent",
    )
    submission, submission_sha256 = _strict_json(
        ledger / "submission.json",
        name="budgeted end-to-end submission",
    )
    script = intent.get("sbatch_script")
    if (
        not isinstance(script, Mapping)
        or set(script) != {"repo_relative_path", "sha256"}
        or script.get("repo_relative_path") != SBATCH_SCRIPT_RELATIVE
    ):
        raise ValueError("budgeted end-to-end sbatch source binding is invalid")
    committed = _run_bytes(
        (
            "git",
            "-C",
            os.fspath(repository),
            "cat-file",
            "blob",
            f"{git_commit}:{SBATCH_SCRIPT_RELATIVE}",
        ),
        name="registered budgeted end-to-end sbatch source query",
    )
    if script.get("sha256") != _sha256_bytes(committed):
        raise ValueError("budgeted end-to-end sbatch Git binding changed")
    export_spec = intent.get("export_spec")
    if not isinstance(export_spec, str):
        raise ValueError("budgeted end-to-end export specification is invalid")
    expected = _intent_payload(
        design_sha256=design_sha256,
        base_config_hash=base_config_hash,
        authorization_sha256=authorization_sha256,
        optimizer_schedule_sha256=optimizer_schedule_sha256,
        git_commit=git_commit,
        image_sha256=image_sha256,
        inventory_sha256=inventory_sha256,
        overlay_sha256=overlay_sha256,
        base_file_sha256=base_file_sha256,
        sbatch_script_sha256=str(script["sha256"]),
        export_spec=export_spec,
        export_spec_sha256=_digest(
            export_spec_sha256,
            name="expected export_spec_sha256",
        ),
        walltime=str(intent.get("walltime")),
        job_name=f"prorm-p2-budgeted-{design_sha256[:12]}",
        project_root=os.fspath(project),
        repository_root=os.fspath(repository),
        submitter_user=submitter_user,
        created_at_utc=str(intent.get("created_at_utc")),
    )
    _validate_intent(intent, expected=expected)
    registered_id = _validate_submission(
        submission,
        intent=intent,
        intent_sha256=intent_sha256,
    )
    if registered_id != array_job_id:
        raise ValueError("array is not the immutable registered submission")
    return {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "verified",
        "ledger": os.fspath(ledger),
        "intent_sha256": intent_sha256,
        "submission_sha256": submission_sha256,
        "array_job_id": registered_id,
        "phase2_design_sha256": design_sha256,
        "ordered_seeds": list(ORDERED_SEEDS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--base-config-hash", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--optimizer-schedule-sha256", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--overlay-sha256", required=True)
    parser.add_argument("--base-file-sha256", required=True)
    parser.add_argument("--walltime", required=True)
    parser.add_argument("--export-spec", required=True)
    parser.add_argument("--sbatch-script", type=Path, required=True)
    return parser


def _show_array(
    *,
    array_job_id: str,
    job_name: str,
    export_spec_sha256: str,
    walltime: str,
    sbatch_script: Path,
    repo_root: Path,
    user: str,
    name: str,
) -> tuple[str, dict[str, object] | None]:
    raw = _run(
        ("scontrol", "show", "job", "--oneliner", array_job_id),
        name=name,
    )
    return _parse_scontrol_records(
        raw,
        array_job_id=array_job_id,
        expected_name=job_name,
        expected_export_spec_sha256=export_spec_sha256,
        expected_walltime=walltime,
        expected_command=sbatch_script,
        expected_workdir=repo_root,
        expected_user=user,
    )


def _require_only_registered(
    *,
    array_job_id: str,
    job_name: str,
    user: str,
    walltime: str,
) -> None:
    live, accounted = _collision_snapshot(
        job_name=job_name,
        user=user,
        walltime=walltime,
    )
    identities = set(live) | set(accounted)
    if array_job_id not in identities:
        raise RuntimeError("ledger-bound held array disappeared before release")
    if identities != {array_job_id}:
        raise RuntimeError("another scheduler array appeared before the ledger-bound release")


def _submit_held(
    *,
    job_name: str,
    walltime: str,
    export_spec: str,
    export_spec_sha256: str,
    log_root: Path,
    repo_root: Path,
    sbatch_script: Path,
) -> str:
    raw = _run(
        (
            "sbatch",
            "--parsable",
            "--hold",
            f"--job-name={job_name}",
            f"--comment={_scheduler_comment(export_spec_sha256)}",
            "--clusters=hpc4",
            "--account=sigroup",
            "--partition=gpu-l20",
            "--qos=l20_qos",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=8",
            "--mem=96G",
            "--gpus-per-node=1",
            f"--time={walltime}",
            f"--array={ARRAY_SPEC}",
            "--no-requeue",
            f"--chdir={repo_root}",
            f"--output={log_root}/%x-%A_%a.out",
            f"--error={log_root}/%x-%A_%a.err",
            f"--export={export_spec}",
            os.fspath(sbatch_script),
        ),
        name="held budgeted end-to-end submission",
    )
    if "\r" in raw or "\x00" in raw:
        raise RuntimeError("sbatch --parsable returned unsafe bytes")
    match = re.fullmatch(r"([1-9][0-9]*);hpc4\n?", raw)
    if match is None:
        raise RuntimeError("sbatch did not return exactly one hpc4 array identity")
    return match.group(1)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = _canonical_directory(arguments.project_root, name="project root")
    repo_root = _canonical_directory(arguments.repo_root, name="repository root")
    _digest(arguments.design_sha256, name="design_sha256")
    _digest(arguments.git_commit, name="git_commit", lengths=frozenset({40, 64}))
    sbatch_script, sbatch_script_sha256 = _exact_script(
        repo_root=repo_root,
        requested=arguments.sbatch_script,
        git_commit=arguments.git_commit,
    )
    user = _effective_user()
    job_name = f"prorm-p2-budgeted-{arguments.design_sha256[:12]}"
    export_spec_sha256 = _sha256_bytes(arguments.export_spec.encode("utf-8"))
    intent_template = _intent_payload(
        design_sha256=arguments.design_sha256,
        base_config_hash=arguments.base_config_hash,
        authorization_sha256=arguments.authorization_sha256,
        optimizer_schedule_sha256=arguments.optimizer_schedule_sha256,
        git_commit=arguments.git_commit,
        image_sha256=arguments.image_sha256,
        inventory_sha256=arguments.inventory_sha256,
        overlay_sha256=arguments.overlay_sha256,
        base_file_sha256=arguments.base_file_sha256,
        sbatch_script_sha256=sbatch_script_sha256,
        export_spec=arguments.export_spec,
        export_spec_sha256=export_spec_sha256,
        walltime=arguments.walltime,
        job_name=job_name,
        project_root=os.fspath(project_root),
        repository_root=os.fspath(repo_root),
        submitter_user=user,
        created_at_utc=_utc_now(),
    )
    design_root = _ensure_directory(
        project_root / "runs" / "phase2-budgeted-end-to-end" / arguments.design_sha256,
        root=project_root,
        name="budgeted end-to-end design root",
    )
    ledger_root = _ensure_directory(
        design_root / "submission-ledger",
        root=project_root,
        name="budgeted end-to-end submission ledger",
    )
    log_root = _ensure_directory(
        project_root / "slurm-logs" / "phase2-budgeted-end-to-end" / arguments.design_sha256,
        root=project_root,
        name="budgeted end-to-end Slurm log root",
    )
    lock_path = ledger_root / "LOCK"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise ValueError("budgeted end-to-end submission lock is not a regular file")
        import fcntl

        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        intent_path = ledger_root / "intent.json"
        submission_path = ledger_root / "submission.json"
        if (submission_path.exists() or submission_path.is_symlink()) and not (
            intent_path.exists() or intent_path.is_symlink()
        ):
            raise ValueError("submission ledger exists without its immutable intent")
        intent: dict[str, object] | None = None
        intent_sha256: str | None = None
        if intent_path.exists() or intent_path.is_symlink():
            intent, intent_sha256 = _strict_json(
                intent_path,
                name="budgeted end-to-end intent",
            )
            _validate_intent(intent, expected=intent_template)

        live_ids, accounted_ids = _collision_snapshot(
            job_name=job_name,
            user=user,
            walltime=arguments.walltime,
        )
        all_ids = set(live_ids) | set(accounted_ids)

        if submission_path.exists() or submission_path.is_symlink():
            assert intent is not None and intent_sha256 is not None
            submission, _ = _strict_json(
                submission_path,
                name="budgeted end-to-end submission",
            )
            array_job_id = _validate_submission(
                submission,
                intent=intent,
                intent_sha256=intent_sha256,
            )
            if all_ids - {array_job_id}:
                raise RuntimeError(
                    "another scheduler array shares the registered budgeted design identity"
                )
            try:
                state, _ = _show_array(
                    array_job_id=array_job_id,
                    job_name=job_name,
                    export_spec_sha256=export_spec_sha256,
                    walltime=arguments.walltime,
                    sbatch_script=sbatch_script,
                    repo_root=repo_root,
                    user=user,
                    name="registered budgeted end-to-end array query",
                )
            except RuntimeError as error:
                fresh_live, fresh_accounted = _collision_snapshot(
                    job_name=job_name,
                    user=user,
                    walltime=arguments.walltime,
                )
                fresh_ids = set(fresh_live) | set(fresh_accounted)
                if fresh_ids - {array_job_id}:
                    raise RuntimeError(
                        "another scheduler array shares the registered budgeted design identity"
                    ) from error
                if array_job_id in fresh_live:
                    raise RuntimeError(
                        "registered live budgeted array could not be verified by scontrol"
                    ) from error
                if array_job_id not in fresh_accounted:
                    raise RuntimeError(
                        "registered budgeted array is absent from scontrol and sacct"
                    ) from error
            else:
                if state == "HELD":
                    _require_only_registered(
                        array_job_id=array_job_id,
                        job_name=job_name,
                        user=user,
                        walltime=arguments.walltime,
                    )
                    _run(
                        ("scontrol", "release", array_job_id),
                        name="registered held budgeted array release",
                    )
            print(f"{array_job_id};hpc4")
            return 0

        if len(all_ids) > 1:
            raise RuntimeError(
                "multiple scheduler arrays match one budgeted end-to-end design identity"
            )
        if not live_ids and accounted_ids:
            raise RuntimeError(
                "historical unregistered budgeted array exists; replacement is forbidden"
            )
        if live_ids and all_ids != {live_ids[0]}:
            raise RuntimeError("ambiguous historical budgeted scheduler identity forbids recovery")

        scheduler_request: dict[str, object] | None
        if live_ids:
            array_job_id = live_ids[0]
            state, scheduler_request = _show_array(
                array_job_id=array_job_id,
                job_name=job_name,
                export_spec_sha256=export_spec_sha256,
                walltime=arguments.walltime,
                sbatch_script=sbatch_script,
                repo_root=repo_root,
                user=user,
                name="orphan held budgeted array query",
            )
            if state != "HELD" or scheduler_request is None:
                raise RuntimeError(
                    "unregistered budgeted array was externally released; replacement is forbidden"
                )
        else:
            array_job_id = _submit_held(
                job_name=job_name,
                walltime=arguments.walltime,
                export_spec=arguments.export_spec,
                export_spec_sha256=export_spec_sha256,
                log_root=log_root,
                repo_root=repo_root,
                sbatch_script=sbatch_script,
            )
            state, scheduler_request = _show_array(
                array_job_id=array_job_id,
                job_name=job_name,
                export_spec_sha256=export_spec_sha256,
                walltime=arguments.walltime,
                sbatch_script=sbatch_script,
                repo_root=repo_root,
                user=user,
                name="new held budgeted end-to-end array query",
            )
            if state != "HELD" or scheduler_request is None:
                raise RuntimeError("new budgeted array was not held before ledger commitment")

        if intent is None:
            intent = dict(intent_template)
            intent_sha256 = _write_exclusive(
                intent_path,
                intent,
                name="budgeted end-to-end intent",
            )
        assert intent_sha256 is not None and scheduler_request is not None
        submission = _submission_payload(
            intent=intent,
            intent_sha256=intent_sha256,
            array_job_id=array_job_id,
            scheduler_request=scheduler_request,
        )
        _write_exclusive(
            submission_path,
            submission,
            name="budgeted end-to-end submission",
        )
        installed_intent, installed_intent_sha256 = _strict_json(
            intent_path,
            name="budgeted end-to-end intent",
        )
        _validate_intent(installed_intent, expected=intent_template)
        installed_submission, _ = _strict_json(
            submission_path,
            name="budgeted end-to-end submission",
        )
        registered_id = _validate_submission(
            installed_submission,
            intent=installed_intent,
            intent_sha256=installed_intent_sha256,
        )
        if registered_id != array_job_id:
            raise RuntimeError("installed submission ledger changed its array identity")
        _require_only_registered(
            array_job_id=array_job_id,
            job_name=job_name,
            user=user,
            walltime=arguments.walltime,
        )
        _run(
            ("scontrol", "release", array_job_id),
            name="ledger-bound held budgeted array release",
        )
        print(f"{array_job_id};hpc4")
        return 0
    finally:
        try:
            import fcntl

            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
