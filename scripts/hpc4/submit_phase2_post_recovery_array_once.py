#!/usr/bin/env python3
"""Submit exactly one immutable post-recovery pilot array per design identity.

The scheduler request is created held.  Its exact Slurm identity is captured
and an immutable, fsync-durable ledger is published before release.  A
deterministic job name plus squeue/sacct collision checks closes the crash
window between ``sbatch`` and ledger publication without ever creating a
replacement array.
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

INTENT_SCHEMA = "prorm-phase2-post-recovery-array-intent/v1"
SUBMISSION_SCHEMA = "prorm-phase2-post-recovery-array-submission/v1"
SCHEDULER_REQUEST_SCHEMA = "prorm-phase2-post-recovery-held-scheduler-request/v1"
ORDERED_SEEDS = (20260801, 20260802, 20260803)
ARRAY_SPEC = "0-2%2"
OPTIMIZER_SCHEDULE_SHA256 = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
_HEX = frozenset("0123456789abcdef")
_EXPECTED_REQ_TRES = "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1"
_EXPECTED_ALLOC_TRES = "billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _effective_user() -> str:
    """Return the account bound to the effective uid, ignoring ambient aliases."""

    import pwd

    value = pwd.getpwuid(os.geteuid()).pw_name
    if re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise ValueError("effective submitter user is invalid")
    return value


def _valid_utc(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("created_at_utc must be a UTC timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("created_at_utc must be a UTC timestamp") from error
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
        absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} leaves the project root") from error
    absolute.mkdir(parents=True, exist_ok=True)
    current = root
    for component in absolute.relative_to(root).parts:
        current = current / component
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
    timeout: int = 60,
) -> str:
    completed = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
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
    completed = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{name} failed{': ' + detail if detail else ''}")
    return completed.stdout


def _intent_payload(
    *,
    pilot_phase: str,
    design_sha256: str,
    base_config_hash: str,
    authorization_sha256: str,
    optimizer_schedule_sha256: str,
    git_commit: str,
    image_sha256: str,
    inventory_sha256: str,
    overlay_sha256: str,
    base_file_sha256: str,
    sbatch_script_relative: str,
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
    if pilot_phase not in {"calibration", "freeze"}:
        raise ValueError("pilot_phase must be calibration or freeze")
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
        or _sha256_bytes(export_spec.encode("utf-8")) != export_spec_sha256
    ):
        raise ValueError("export specification does not match its SHA256")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", job_name) is None:
        raise ValueError("deterministic Slurm job name is invalid")
    project_path = Path(project_root)
    repository_path = Path(repository_root)
    if (
        not project_path.is_absolute()
        or project_path == Path("/")
        or "\n" in project_root
        or "\r" in project_root
        or not repository_path.is_absolute()
        or repository_path == Path("/")
        or "\n" in repository_root
        or "\r" in repository_root
    ):
        raise ValueError("project or repository root is invalid")
    if re.fullmatch(r"[A-Za-z0-9._-]+", submitter_user) is None:
        raise ValueError("submitter user is invalid")
    if (
        not sbatch_script_relative.startswith("scripts/hpc4/")
        or "\\" in sbatch_script_relative
        or ".." in Path(sbatch_script_relative).parts
    ):
        raise ValueError("sbatch script repository path is invalid")
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
        "status": "precommitted_before_first_scheduler_submission",
        "pilot_phase": pilot_phase,
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
            "repo_relative_path": sbatch_script_relative,
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
        "created_at_utc": _valid_utc(created_at_utc),
    }


def _validate_intent(
    value: Mapping[str, object],
    *,
    expected: Mapping[str, object],
) -> None:
    if set(value) != set(expected):
        raise ValueError("post-recovery array intent fields differ")
    observed = dict(value)
    expected_value = dict(expected)
    observed_created = observed.pop("created_at_utc", None)
    expected_value.pop("created_at_utc", None)
    _valid_utc(observed_created)
    if observed != expected_value:
        raise ValueError("post-recovery array intent identity differs")


def _parse_squeue_ids(raw: str) -> tuple[str, ...]:
    values = {line.strip() for line in raw.splitlines() if line.strip()}
    if any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in values):
        raise ValueError("squeue returned an invalid array identity")
    return tuple(sorted(values, key=int))


def _array_root(value: str) -> str | None:
    match = re.fullmatch(
        r"([1-9][0-9]*)(?:_(?:[0-9]+|\[[0-9,%\-]+\]))?",
        value,
    )
    return None if match is None else match.group(1)


def _parse_sacct_ids(
    raw: str,
    *,
    expected_name: str,
    expected_walltime: str,
) -> tuple[str, ...]:
    roots: set[str] = set()
    for line in raw.splitlines():
        if not line or "\r" in line:
            raise ValueError("sacct returned an unsafe historical row")
        fields = line.split("|")
        if len(fields) != 8:
            raise ValueError("sacct historical row differs from the locked field set")
        (
            raw_id,
            job_id,
            job_name,
            state,
            submitted,
            timelimit,
            req_tres,
            alloc_tres,
        ) = fields
        if (
            job_name != expected_name
            or not state
            or not submitted
            or timelimit != expected_walltime
            or req_tres != _EXPECTED_REQ_TRES
            or alloc_tres not in {"", _EXPECTED_ALLOC_TRES}
        ):
            raise ValueError("historical scheduler identity differs from the intent")
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
    expected_walltime: str,
    expected_command: Path,
    expected_workdir: Path,
    expected_user: str,
) -> tuple[str, dict[str, object] | None]:
    lines = [line for line in raw.splitlines() if line]
    if not lines or "\r" in raw:
        raise ValueError("scontrol returned no safe scheduler record")
    parsed: list[dict[str, str]] = []
    for line in lines:
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key in fields:
                raise ValueError(f"duplicate scontrol job field: {key}")
            fields[key] = value
        tres: dict[str, str] = {}
        for entry in fields.get("TRES", "").split(","):
            if "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            if key in tres:
                raise ValueError(f"duplicate scheduler TRES field: {key}")
            tres[key] = value
        if (
            fields.get("ArrayJobId", fields.get("JobId")) != array_job_id
            or fields.get("JobName") != expected_name
            or not str(fields.get("UserId", "")).startswith(f"{expected_user}(")
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
            or tres.get("cpu") != "8"
            or tres.get("mem") != "96G"
            or tres.get("node") != "1"
            or tres.get("gres/gpu") != "1"
            or re.fullmatch(
                r"gres(?::|/)gpu(?::[A-Za-z0-9_.-]+)?:1",
                fields.get("TresPerNode", ""),
            )
            is None
            or fields.get("Command") != os.fspath(expected_command)
            or fields.get("WorkDir") != os.fspath(expected_workdir)
            or not fields.get("JobState")
        ):
            raise ValueError("Slurm job differs from the immutable pilot intent")
        parsed.append(fields)
    held_master = (
        len(parsed) == 1
        and parsed[0].get("ArrayTaskId") == ARRAY_SPEC
        and parsed[0].get("JobState") == "PENDING"
        and parsed[0].get("Reason") == "JobHeldUser"
    )
    if held_master:
        fields = parsed[0]
        normalized_tres = {
            key: value
            for key, value in (
                entry.split("=", 1) for entry in fields["TRES"].split(",") if "=" in entry
            )
            if key in {"cpu", "mem", "node", "gres/gpu"}
        }
        evidence = {
            "schema_version": SCHEDULER_REQUEST_SCHEMA,
            "captured_while_held": True,
            "raw_scontrol_record": raw,
            "raw_scontrol_sha256": _sha256_bytes(raw.encode("utf-8")),
            "normalized": {
                "array_job_id": array_job_id,
                "job_name": expected_name,
                "array_spec": ARRAY_SPEC,
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
                "walltime": expected_walltime,
                "tres": normalized_tres,
                "tres_per_node": fields["TresPerNode"],
                "requeue": False,
                "restarts": 0,
                "command": os.fspath(expected_command),
                "work_dir": os.fspath(expected_workdir),
            },
        }
        return "HELD", evidence
    if any(
        fields.get("JobState") == "PENDING" and str(fields.get("Reason", "")).startswith("JobHeld")
        for fields in parsed
    ):
        raise ValueError("pilot array is held by an unexpected scheduler authority")
    return "ALREADY_RELEASED", None


def _submission_payload(
    *,
    intent: Mapping[str, object],
    intent_sha256: str,
    array_job_id: str,
    submitted_cluster: str,
    scheduler_request: Mapping[str, object],
) -> dict[str, object]:
    if re.fullmatch(r"[1-9][0-9]*", array_job_id) is None:
        raise ValueError("array_job_id must be positive")
    if submitted_cluster != "hpc4":
        raise ValueError("submitted cluster must be hpc4")
    return {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "committed_while_scheduler_held",
        "intent_sha256": _digest(intent_sha256, name="intent_sha256"),
        "pilot_phase": intent["pilot_phase"],
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
        "cluster": submitted_cluster,
        "scheduler_request": dict(scheduler_request),
        "same_design_resubmission_allowed": False,
        "replacement_array_allowed": False,
        "released_only_after_ledger_fsync": True,
        "created_at_utc": _utc_now(),
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
            submitted_cluster="hpc4",
            scheduler_request={},
        )
    )
    if set(value) != expected_keys:
        raise ValueError("post-recovery submission ledger fields differ")
    for field in (
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
    ):
        if value.get(field) != intent[field]:
            raise ValueError(f"post-recovery submission ledger {field} differs")
    array_job_id = value.get("array_job_id")
    if (
        value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("status") != "committed_while_scheduler_held"
        or value.get("intent_sha256") != intent_sha256
        or value.get("ordered_seeds") != list(ORDERED_SEEDS)
        or value.get("array_spec") != ARRAY_SPEC
        or re.fullmatch(r"[1-9][0-9]*", str(array_job_id)) is None
        or value.get("cluster") != "hpc4"
        or value.get("same_design_resubmission_allowed") is not False
        or value.get("replacement_array_allowed") is not False
        or value.get("released_only_after_ledger_fsync") is not True
    ):
        raise ValueError("post-recovery submission ledger policy is invalid")
    _valid_utc(value.get("created_at_utc"))
    scheduler = value.get("scheduler_request")
    if not isinstance(scheduler, Mapping):
        raise ValueError("post-recovery submission scheduler request is missing")
    if set(scheduler) != {
        "schema_version",
        "captured_while_held",
        "raw_scontrol_record",
        "raw_scontrol_sha256",
        "normalized",
    }:
        raise ValueError("post-recovery submission scheduler request fields differ")
    normalized = scheduler.get("normalized")
    if not isinstance(normalized, Mapping) or set(normalized) != {
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
    }:
        raise ValueError("post-recovery normalized scheduler request fields differ")
    script = intent.get("sbatch_script")
    repository_root = intent.get("repository_root")
    submitter_user = intent.get("submitter_user")
    if (
        not isinstance(script, Mapping)
        or set(script) != {"repo_relative_path", "sha256"}
        or not isinstance(repository_root, str)
        or not isinstance(submitter_user, str)
    ):
        raise ValueError("post-recovery intent source identity is invalid")
    expected_command = Path(repository_root) / str(script["repo_relative_path"])
    expected_normalized = {
        "array_job_id": str(array_job_id),
        "job_name": intent["job_name"],
        "array_spec": ARRAY_SPEC,
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
        "walltime": intent["walltime"],
        "tres": {"cpu": "8", "mem": "96G", "node": "1", "gres/gpu": "1"},
        "requeue": False,
        "restarts": 0,
        "command": os.fspath(expected_command),
        "work_dir": repository_root,
    }
    for field, expected_value in expected_normalized.items():
        if normalized.get(field) != expected_value:
            raise ValueError(f"post-recovery normalized scheduler request {field} differs")
    if (
        not isinstance(normalized.get("tres_per_node"), str)
        or re.fullmatch(
            r"gres(?::|/)gpu(?::[A-Za-z0-9_.-]+)?:1",
            normalized["tres_per_node"],
        )
        is None
    ):
        raise ValueError("post-recovery normalized GPU request differs")
    if (
        scheduler.get("schema_version") != SCHEDULER_REQUEST_SCHEMA
        or scheduler.get("captured_while_held") is not True
    ):
        raise ValueError("post-recovery submission held scheduler evidence is invalid")
    _digest(
        scheduler.get("raw_scontrol_sha256"),
        name="scheduler_request.raw_scontrol_sha256",
    )
    raw = scheduler.get("raw_scontrol_record")
    if (
        not isinstance(raw, str)
        or _sha256_bytes(raw.encode("utf-8")) != scheduler["raw_scontrol_sha256"]
    ):
        raise ValueError("post-recovery submission raw scheduler evidence changed")
    state, reparsed = _parse_scontrol_records(
        raw,
        array_job_id=str(array_job_id),
        expected_name=str(intent["job_name"]),
        expected_walltime=str(intent["walltime"]),
        expected_command=expected_command,
        expected_workdir=Path(repository_root),
        expected_user=submitter_user,
    )
    if state != "HELD" or reparsed != dict(scheduler):
        raise ValueError("post-recovery submission scheduler evidence does not reparse exactly")
    return str(array_job_id)


def verify_submission_registry(
    registry_path: Path,
    *,
    project_root: Path,
    repo_root: Path,
    pilot_phase: str,
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
    """Deep-verify the immutable pre-release submission registry."""

    canonical_project = _canonical_directory(project_root, name="project root")
    canonical_repo = _canonical_directory(repo_root, name="repository root")
    expected_registry = (
        canonical_project
        / "runs"
        / f"phase2-post-recovery-{pilot_phase}"
        / design_sha256
        / "submission-registry"
    )
    registry = _canonical_directory(registry_path, name="submission registry")
    if registry != expected_registry:
        raise ValueError("submission registry path differs from the design identity")
    intent, intent_sha256 = _strict_json(
        registry / "intent.json",
        name="post-recovery array intent",
    )
    submission, submission_sha256 = _strict_json(
        registry / "submission.json",
        name="post-recovery array submission ledger",
    )
    script = intent.get("sbatch_script")
    if not isinstance(script, Mapping) or set(script) != {
        "repo_relative_path",
        "sha256",
    }:
        raise ValueError("post-recovery intent sbatch source binding is invalid")
    relative = script["repo_relative_path"]
    expected_relative = "scripts/hpc4/phase2_post_recovery_calibration.sbatch"
    if relative != expected_relative:
        raise ValueError("post-recovery intent names an unexpected sbatch source")
    committed_source = _run_bytes(
        (
            "git",
            "-C",
            os.fspath(canonical_repo),
            "cat-file",
            "blob",
            f"{git_commit}:{expected_relative}",
        ),
        name="registered post-recovery sbatch source query",
    )
    if script["sha256"] != _sha256_bytes(committed_source):
        raise ValueError("post-recovery intent sbatch source Git binding changed")
    observed_template = _intent_payload(
        pilot_phase=pilot_phase,
        design_sha256=design_sha256,
        base_config_hash=base_config_hash,
        authorization_sha256=authorization_sha256,
        optimizer_schedule_sha256=optimizer_schedule_sha256,
        git_commit=git_commit,
        image_sha256=image_sha256,
        inventory_sha256=inventory_sha256,
        overlay_sha256=overlay_sha256,
        base_file_sha256=base_file_sha256,
        sbatch_script_relative=expected_relative,
        sbatch_script_sha256=str(script["sha256"]),
        export_spec=str(intent.get("export_spec")),
        export_spec_sha256=_digest(
            export_spec_sha256,
            name="expected post-recovery export_spec_sha256",
        ),
        walltime=str(intent.get("walltime")),
        job_name=f"prorm-p2-post-{pilot_phase}-{design_sha256[:12]}",
        project_root=os.fspath(canonical_project),
        repository_root=os.fspath(canonical_repo),
        submitter_user=submitter_user,
        created_at_utc=str(intent.get("created_at_utc")),
    )
    _validate_intent(intent, expected=observed_template)
    registered_array = _validate_submission(
        submission,
        intent=intent,
        intent_sha256=intent_sha256,
    )
    if registered_array != array_job_id:
        raise ValueError("running array is not the immutable registered submission")
    return {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "verified",
        "registry": os.fspath(registry),
        "intent_sha256": intent_sha256,
        "submission_sha256": submission_sha256,
        "array_job_id": registered_array,
        "phase2_design_sha256": design_sha256,
        "pilot_phase": pilot_phase,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pilot-phase", choices=("calibration", "freeze"), required=True)
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


def _sacct_ids_since(
    *,
    starttime: str,
    job_name: str,
    walltime: str,
) -> tuple[str, ...]:
    raw = _run(
        (
            "sacct",
            "-X",
            f"--starttime={starttime}",
            f"--name={job_name}",
            "--noheader",
            "--parsable2",
            (
                "--format=JobIDRaw%128,JobID%128,JobName%128,State%64,"
                "Submit%32,Timelimit%32,ReqTRES%256,AllocTRES%256"
            ),
        ),
        name="sacct deterministic-name collision query",
    )
    return _parse_sacct_ids(
        raw,
        expected_name=job_name,
        expected_walltime=walltime,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = _canonical_directory(arguments.project_root, name="project root")
    repo_root = _canonical_directory(arguments.repo_root, name="repository root")
    sbatch_script = _canonical_file(arguments.sbatch_script, name="pilot sbatch script")
    try:
        sbatch_relative = sbatch_script.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise ValueError("pilot sbatch script leaves the repository") from error
    committed = _run_bytes(
        (
            "git",
            "-C",
            os.fspath(repo_root),
            "cat-file",
            "blob",
            f"{arguments.git_commit}:{sbatch_relative}",
        ),
        name="committed pilot sbatch source query",
    )
    if committed != sbatch_script.read_bytes():
        raise ValueError("pilot sbatch script differs from the submitted Git commit")
    export_spec_sha256 = _sha256_bytes(arguments.export_spec.encode("utf-8"))
    job_name = f"prorm-p2-post-{arguments.pilot_phase}-{arguments.design_sha256[:12]}"
    user = _effective_user()
    intent_template = _intent_payload(
        pilot_phase=arguments.pilot_phase,
        design_sha256=arguments.design_sha256,
        base_config_hash=arguments.base_config_hash,
        authorization_sha256=arguments.authorization_sha256,
        optimizer_schedule_sha256=arguments.optimizer_schedule_sha256,
        git_commit=arguments.git_commit,
        image_sha256=arguments.image_sha256,
        inventory_sha256=arguments.inventory_sha256,
        overlay_sha256=arguments.overlay_sha256,
        base_file_sha256=arguments.base_file_sha256,
        sbatch_script_relative=sbatch_relative,
        sbatch_script_sha256=_sha256_file(sbatch_script),
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
        project_root
        / "runs"
        / f"phase2-post-recovery-{arguments.pilot_phase}"
        / arguments.design_sha256,
        root=project_root,
        name="post-recovery design root",
    )
    registry = _ensure_directory(
        design_root / "submission-registry",
        root=project_root,
        name="post-recovery submission registry",
    )
    log_root = _ensure_directory(
        project_root
        / "slurm-logs"
        / f"phase2-post-recovery-{arguments.pilot_phase}"
        / arguments.design_sha256,
        root=project_root,
        name="post-recovery Slurm log root",
    )
    lock_path = registry / "LOCK"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise ValueError("post-recovery submission lock is not a regular file")
        import fcntl

        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        intent_path = registry / "intent.json"
        submission_path = registry / "submission.json"
        if intent_path.exists() or intent_path.is_symlink():
            intent, intent_sha256 = _strict_json(
                intent_path,
                name="post-recovery array intent",
            )
            _validate_intent(intent, expected=intent_template)
        else:
            intent = dict(intent_template)
            intent_sha256 = _write_exclusive(
                intent_path,
                intent,
                name="post-recovery array intent",
            )
        created_at = _valid_utc(intent["created_at_utc"])
        starttime = created_at[:-1]
        live_ids = _parse_squeue_ids(
            _run(
                (
                    "squeue",
                    "--noheader",
                    f"--user={user}",
                    f"--name={job_name}",
                    "--format=%A",
                ),
                name="live deterministic-name collision query",
            )
        )
        accounted_ids = _sacct_ids_since(
            starttime=starttime,
            job_name=job_name,
            walltime=arguments.walltime,
        )

        if submission_path.exists() or submission_path.is_symlink():
            submission, _ = _strict_json(
                submission_path,
                name="post-recovery array submission ledger",
            )
            array_job_id = _validate_submission(
                submission,
                intent=intent,
                intent_sha256=intent_sha256,
            )
            unexpected_ids = (set(live_ids) | set(accounted_ids)) - {array_job_id}
            if unexpected_ids:
                raise RuntimeError(
                    "another scheduler array shares the registered post-recovery design identity"
                )
            try:
                scheduler_record = _run(
                    ("scontrol", "show", "job", "--oneliner", array_job_id),
                    name="registered pilot array query",
                )
            except RuntimeError:
                accounted = _sacct_ids_since(
                    starttime=starttime,
                    job_name=job_name,
                    walltime=arguments.walltime,
                )
                if array_job_id not in accounted:
                    raise RuntimeError(
                        "registered pilot array is absent from both scontrol and sacct"
                    ) from None
            else:
                state, _ = _parse_scontrol_records(
                    scheduler_record,
                    array_job_id=array_job_id,
                    expected_name=job_name,
                    expected_walltime=arguments.walltime,
                    expected_command=sbatch_script,
                    expected_workdir=repo_root,
                    expected_user=user,
                )
                if state == "HELD":
                    _run(
                        ("scontrol", "release", array_job_id),
                        name="registered held pilot array release",
                    )
            print(f"{array_job_id};hpc4")
            return 0

        if len(live_ids) > 1:
            raise RuntimeError("multiple scheduler arrays match one post-recovery design identity")
        if not live_ids and accounted_ids:
            raise RuntimeError(
                "historical unregistered post-recovery array exists; replacement is forbidden"
            )
        if live_ids and any(value != live_ids[0] for value in accounted_ids):
            raise RuntimeError(
                "ambiguous historical post-recovery scheduler identity forbids recovery"
            )

        submitted_cluster = "hpc4"
        if live_ids:
            array_job_id = live_ids[0]
            scheduler_record = _run(
                ("scontrol", "show", "job", "--oneliner", array_job_id),
                name="orphan held pilot array query",
            )
            state, scheduler_request = _parse_scontrol_records(
                scheduler_record,
                array_job_id=array_job_id,
                expected_name=job_name,
                expected_walltime=arguments.walltime,
                expected_command=sbatch_script,
                expected_workdir=repo_root,
                expected_user=user,
            )
            if state != "HELD" or scheduler_request is None:
                raise RuntimeError(
                    "unregistered post-recovery array was externally released; "
                    "replacement is forbidden"
                )
        else:
            raw_submission = _run(
                (
                    "sbatch",
                    "--parsable",
                    "--hold",
                    f"--job-name={job_name}",
                    "--clusters=hpc4",
                    "--account=sigroup",
                    "--partition=gpu-l20",
                    "--qos=l20_qos",
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=8",
                    "--mem=96G",
                    "--gpus-per-node=1",
                    f"--time={arguments.walltime}",
                    f"--array={ARRAY_SPEC}",
                    "--no-requeue",
                    f"--chdir={repo_root}",
                    f"--output={log_root}/%x-%A_%a.out",
                    f"--error={log_root}/%x-%A_%a.err",
                    f"--export={arguments.export_spec}",
                    os.fspath(sbatch_script),
                ),
                name="held post-recovery pilot submission",
            ).strip()
            if "\n" in raw_submission or "\r" in raw_submission:
                raise RuntimeError("sbatch --parsable returned multiple lines")
            array_job_id, separator, raw_cluster = raw_submission.partition(";")
            if re.fullmatch(r"[1-9][0-9]*", array_job_id) is None:
                raise RuntimeError("sbatch did not return one numeric array job ID")
            if separator:
                submitted_cluster = raw_cluster
            if submitted_cluster != "hpc4":
                raise RuntimeError("sbatch returned an unexpected Slurm cluster")
            scheduler_record = _run(
                ("scontrol", "show", "job", "--oneliner", array_job_id),
                name="new held pilot array query",
            )
            state, scheduler_request = _parse_scontrol_records(
                scheduler_record,
                array_job_id=array_job_id,
                expected_name=job_name,
                expected_walltime=arguments.walltime,
                expected_command=sbatch_script,
                expected_workdir=repo_root,
                expected_user=user,
            )
            if state != "HELD" or scheduler_request is None:
                raise RuntimeError("new post-recovery array was not held before ledger commitment")

        submission = _submission_payload(
            intent=intent,
            intent_sha256=intent_sha256,
            array_job_id=array_job_id,
            submitted_cluster=submitted_cluster,
            scheduler_request=scheduler_request,
        )
        _write_exclusive(
            submission_path,
            submission,
            name="post-recovery array submission ledger",
        )
        installed, _ = _strict_json(
            submission_path,
            name="post-recovery array submission ledger",
        )
        _validate_submission(
            installed,
            intent=intent,
            intent_sha256=intent_sha256,
        )
        release_live_ids = _parse_squeue_ids(
            _run(
                (
                    "squeue",
                    "--noheader",
                    f"--user={user}",
                    f"--name={job_name}",
                    "--format=%A",
                ),
                name="pre-release live deterministic-name collision query",
            )
        )
        release_accounted_ids = _sacct_ids_since(
            starttime=starttime,
            job_name=job_name,
            walltime=arguments.walltime,
        )
        release_ids = set(release_live_ids) | set(release_accounted_ids)
        if array_job_id not in release_ids:
            raise RuntimeError("ledger-bound held pilot array disappeared before release")
        unexpected_release_ids = release_ids - {array_job_id}
        if unexpected_release_ids:
            raise RuntimeError(
                "another scheduler array appeared before the ledger-bound post-recovery release"
            )
        _run(
            ("scontrol", "release", array_job_id),
            name="ledger-bound held pilot array release",
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
