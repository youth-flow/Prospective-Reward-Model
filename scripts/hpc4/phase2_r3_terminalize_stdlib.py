#!/usr/bin/env python3
"""Route and capture R3 terminal evidence without importing the model stack.

The fixed HPC4 host Python does not contain torch or PyYAML, so the login-side
part of terminalization must remain stdlib-only.  This helper performs no
scientific validation and mints no capability.  It only:

* derives the one scheduler selector and canonical output paths from an
  immutable Gate-P receipt or Gate-R runtime closure; and
* captures the exact allocation-only ``sacct`` bytes after the target has left
  ``squeue -r``.

The compute-side SIF process must subsequently re-open these bytes through
``capture_phase2_r3_terminal.py`` and the sealed R3 validators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

_POSITIVE_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_JOB_SELECTOR = re.compile(r"[1-9][0-9]*(?:_(?:0|[1-9][0-9]*))?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GATEP_ATTEMPT = re.compile(r"gatep-attempt-(?!000)[0-9]{3}\Z")
_PRIMARY_STATUSES: Final = {
    "continuation_required_after_safe_checkpoint": "primary-continuable-finalize",
    "compute_complete_pending_external_scheduler_terminal": "primary-completed-finalize",
}
_SACCT_FORMAT_FIELDS: Final = (
    "JobID%64",
    "JobIDRaw%32",
    "State%64",
    "ExitCode%32",
    "DerivedExitCode%32",
    "Cluster%64",
    "Account%64",
    "Partition%64",
    "QOS%64",
    "NNodes%16",
    "NCPUS%16",
    "ReqTRES%512",
    "AllocTRES%512",
    "ElapsedRaw%32",
)


def _strict_json(path: Path, *, name: str) -> tuple[dict[str, object], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    raw = path.read_bytes()

    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{name} contains non-finite constant {token!r}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    canonical = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if raw != canonical:
        raise ValueError(f"{name} is not canonical JSON")
    return value, raw


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _canonical_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    if not absolute.is_dir() or absolute.is_symlink():
        raise ValueError(f"{name} must be a real directory")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise ValueError(f"{name} must already be canonical")
    return resolved


def _canonical_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    if not absolute.is_file() or absolute.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise ValueError(f"{name} must already be canonical")
    return resolved


def _contained(path: Path, root: Path, *, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escaped its canonical root") from error


def _safe_job_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _POSITIVE_JOB_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a positive decimal Slurm job ID")
    return value


def _embedded_payload(value: object, *, name: str) -> dict[str, object]:
    embedded = _mapping(value, name=name)
    if set(embedded) != {"encoding", "file_sha256", "size_bytes", "payload"}:
        raise ValueError(f"{name} has an invalid embedded-canonical field set")
    payload = _mapping(embedded["payload"], name=f"{name}.payload")
    raw = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if (
        embedded["encoding"] != "canonical-json-utf8-newline"
        or embedded["size_bytes"] != len(raw)
        or embedded["file_sha256"] != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError(f"{name} embedded byte binding is invalid")
    return payload


def plan_gatep(*, project_root: Path, attempt_root: Path) -> dict[str, object]:
    project = _canonical_directory(project_root, name="project root")
    attempt = _canonical_directory(attempt_root, name="Gate-P attempt root")
    expected_prefix = project / "runs" / "phase2-recovery-r3" / "gatep"
    _contained(attempt, expected_prefix, name="Gate-P attempt root")
    relative = attempt.relative_to(expected_prefix)
    if (
        len(relative.parts) != 2
        or not relative.parts[0]
        or _GATEP_ATTEMPT.fullmatch(relative.parts[1]) is None
    ):
        raise ValueError(
            "Gate-P attempt must be "
            "<project>/runs/phase2-recovery-r3/gatep/<identity>/gatep-attempt-NNN"
        )
    operational_bundle = _canonical_file(
        attempt / "gatep-operational-bundle.json",
        name="Gate-P operational bundle",
    )
    allocation_intent = _canonical_file(
        attempt / "profile-allocation-intent.json",
        name="Gate-P allocation intent",
    )
    runtime_receipt = _canonical_file(
        attempt / "profile-runtime-receipt.json",
        name="Gate-P runtime receipt",
    )
    receipt, _ = _strict_json(runtime_receipt, name="Gate-P runtime receipt")
    runtime = _mapping(receipt.get("slurm_runtime"), name="Gate-P Slurm runtime")
    job_id = _safe_job_id(runtime.get("job_id"), name="Gate-P runtime job_id")
    array_job_id = runtime.get("array_job_id")
    array_task_id = runtime.get("array_task_id")
    if array_job_id is None:
        if array_task_id is not None:
            raise ValueError("Gate-P non-array runtime retained an array task ID")
        selector = job_id
    else:
        array = _safe_job_id(array_job_id, name="Gate-P runtime array_job_id")
        if type(array_task_id) is not int or array_task_id < 0:
            raise ValueError("Gate-P runtime array_task_id is invalid")
        selector = f"{array}_{array_task_id}"
    return {
        "mode": "gatep",
        "job_selector": selector,
        "route_status": "profile",
        "finalizer_command": "profile-finalize",
        "attempt_root": os.fspath(attempt),
        "operational_bundle": os.fspath(operational_bundle),
        "allocation_intent": os.fspath(allocation_intent),
        "runtime_receipt": os.fspath(runtime_receipt),
        "runtime_closure": None,
        "raw_sacct": os.fspath(attempt / "terminal-raw" / "profile.sacct.psv"),
        "evidence_directory": os.fspath(attempt / "terminal-evidence" / "profile"),
        "task_id": None,
        "segment_index": None,
    }


def plan_primary(
    *,
    project_root: Path,
    attempt_root: Path,
    task_id: int,
    operational_bundle: Path,
) -> dict[str, object]:
    project = _canonical_directory(project_root, name="project root")
    attempt = _canonical_directory(attempt_root, name="Gate-R attempt root")
    r3_root = project / "runs" / "phase2-recovery-r3"
    _contained(attempt, r3_root, name="Gate-R attempt root")
    if attempt == r3_root:
        raise ValueError("Gate-R attempt root cannot equal the campaign root")
    if type(task_id) is not int or task_id not in {0, 1, 2}:
        raise ValueError("Gate-R task ID must be 0, 1, or 2")
    bundle = _canonical_file(operational_bundle, name="Gate-P operational bundle")
    _contained(bundle, r3_root / "gatep", name="Gate-P operational bundle")
    if (
        bundle.name != "gatep-operational-bundle.json"
        or _GATEP_ATTEMPT.fullmatch(bundle.parent.name) is None
    ):
        raise ValueError("Gate-P operational bundle is not at its fixed attempt path")
    closure = _canonical_file(
        attempt / "runtime-closures" / f"task-{task_id}.json",
        name="Gate-R runtime closure",
    )
    payload, _ = _strict_json(closure, name="Gate-R runtime closure")
    admission = _embedded_payload(payload.get("admission"), name="closure admission")
    runtime = _embedded_payload(payload.get("runtime"), name="closure runtime")
    segment_index = admission.get("segment_index")
    if admission.get("task_id") != task_id or type(segment_index) is not int or segment_index < 1:
        raise ValueError("Gate-R closure task/segment routing is invalid")
    if runtime.get("task_id") != task_id or runtime.get("segment_index") != segment_index:
        raise ValueError("Gate-R runtime differs from its admission route")
    array_job_id = _safe_job_id(
        runtime.get("array_job_id"),
        name="Gate-R runtime array_job_id",
    )
    job_id = _safe_job_id(runtime.get("job_id"), name="Gate-R runtime job_id")
    if runtime.get("array_task_id") != task_id:
        raise ValueError("Gate-R runtime array task differs from the requested task")
    status = payload.get("status")
    if not isinstance(status, str) or status not in _PRIMARY_STATUSES:
        raise ValueError("Gate-R closure status is not terminalizable")
    selector = f"{array_job_id}_{task_id}"
    return {
        "mode": "primary",
        "job_selector": selector,
        "job_id_raw": job_id,
        "route_status": status,
        "finalizer_command": _PRIMARY_STATUSES[status],
        "attempt_root": os.fspath(attempt),
        "operational_bundle": os.fspath(bundle),
        "allocation_intent": None,
        "runtime_receipt": None,
        "runtime_closure": os.fspath(closure),
        "raw_sacct": os.fspath(
            attempt / "terminal-raw" / f"task-{task_id}-segment-{segment_index}.sacct.psv"
        ),
        "evidence_directory": os.fspath(
            attempt / "terminal-evidence" / f"task-{task_id}-segment-{segment_index}"
        ),
        "task_id": task_id,
        "segment_index": segment_index,
    }


def sacct_terminal_command(job_selector: str) -> tuple[str, ...]:
    if _JOB_SELECTOR.fullmatch(job_selector) is None:
        raise ValueError("job selector must identify one exact job or array task")
    return (
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        job_selector,
        f"--format={','.join(_SACCT_FORMAT_FIELDS)}",
    )


def _run_command(command: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, check=False, capture_output=True)


def _stable_existing(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError("existing raw sacct target is not a regular non-symlink file")
    if os.name == "posix" and stat.S_IMODE(before.st_mode) != 0o440:
        raise ValueError("existing raw sacct target does not retain mode 0440")
    raw = path.read_bytes()
    after = path.lstat()
    if (before.st_dev, before.st_ino) != (
        after.st_dev,
        after.st_ino,
    ) or before.st_size != after.st_size:
        raise ValueError("existing raw sacct target changed while being read")
    return raw


def _publish_exclusive(path: Path, raw: bytes) -> None:
    if path.is_symlink():
        raise ValueError("raw sacct output must not be a symbolic link")
    parent = _canonical_directory(path.parent, name="raw sacct output parent")
    destination = parent / path.name
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o440,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing raw sacct evidence")
            view = view[written:]
        if os.name == "posix":
            # Submitters run under umask 077, but terminal evidence has a
            # fixed group-readable 0440 contract.
            os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except PermissionError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def capture_raw_sacct(
    *,
    job_selector: str,
    output: Path,
    user: str,
    attempts: int,
    interval_seconds: float,
) -> dict[str, object]:
    if _JOB_SELECTOR.fullmatch(job_selector) is None:
        raise ValueError("job selector must identify one exact job or array task")
    if not user or any(character in user for character in "\r\n,"):
        raise ValueError("scheduler user is invalid")
    if attempts < 1:
        raise ValueError("attempt count must be positive")
    if interval_seconds < 0:
        raise ValueError("retry interval cannot be negative")
    queue = _run_command(("squeue", "-r", "-h", "-u", user, "-o", "%i"))
    if queue.returncode != 0 or queue.stderr:
        raise RuntimeError("locked squeue query failed or wrote stderr")
    try:
        queue_text = queue.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("squeue returned non-UTF-8 output") from error
    active: set[str] = set()
    for line in queue_text.splitlines():
        item = line.strip()
        if not item:
            continue
        if _JOB_SELECTOR.fullmatch(item) is None:
            raise ValueError("squeue returned an unsafe expanded job selector")
        active.add(item)
    if job_selector in active:
        raise RuntimeError("target job selector is still present in squeue -r")

    command = sacct_terminal_command(job_selector)
    raw: bytes | None = None
    for index in range(attempts):
        completed = _run_command(command)
        candidate = completed.stdout
        if (
            completed.returncode == 0
            and not completed.stderr
            and candidate.endswith(b"\n")
            and candidate.count(b"\n") == 1
            and candidate.strip()
        ):
            raw = candidate
            break
        if index + 1 < attempts:
            time.sleep(interval_seconds)
    if raw is None:
        raise RuntimeError("sacct did not return one stable allocation row")

    destination = output.absolute()
    reused = destination.exists() or destination.is_symlink()
    if reused:
        existing = _stable_existing(destination)
        if existing != raw:
            raise ValueError("existing raw sacct bytes differ from the live locked query")
    else:
        _publish_exclusive(destination, raw)
        if _stable_existing(destination) != raw:
            raise ValueError("published raw sacct bytes changed after fsync")
    return {
        "status": (
            "r3_raw_sacct_revalidated_exact"
            if reused
            else "r3_raw_sacct_published_pending_sif_validation"
        ),
        "command": list(command),
        "job_selector": job_selector,
        "output": os.fspath(destination),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "reused": reused,
    }


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

    gatep = commands.add_parser("plan-gatep")
    gatep.add_argument("--project-root", type=Path, required=True)
    gatep.add_argument("--attempt-root", type=Path, required=True)

    primary = commands.add_parser("plan-primary")
    primary.add_argument("--project-root", type=Path, required=True)
    primary.add_argument("--attempt-root", type=Path, required=True)
    primary.add_argument("--task-id", type=int, required=True)
    primary.add_argument("--operational-bundle", type=Path, required=True)

    capture = commands.add_parser("capture")
    capture.add_argument("--job-selector", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--user", required=True)
    capture.add_argument("--attempts", type=int, default=30)
    capture.add_argument("--interval-seconds", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "plan-gatep":
        result = plan_gatep(
            project_root=arguments.project_root,
            attempt_root=arguments.attempt_root,
        )
    elif arguments.command == "plan-primary":
        result = plan_primary(
            project_root=arguments.project_root,
            attempt_root=arguments.attempt_root,
            task_id=arguments.task_id,
            operational_bundle=arguments.operational_bundle,
        )
    else:
        result = capture_raw_sacct(
            job_selector=arguments.job_selector,
            output=arguments.output,
            user=arguments.user,
            attempts=arguments.attempts,
            interval_seconds=arguments.interval_seconds,
        )
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
