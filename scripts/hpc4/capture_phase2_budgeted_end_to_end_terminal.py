#!/usr/bin/env python3
"""Capture and verify terminal Slurm evidence for the fixed-five budgeted wave."""

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

SCHEMA_VERSION = "prorm-phase2-budgeted-end-to-end-terminal/v1"
ORDERED_SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005)

_SACCT_FIELDS = (
    "JobID",
    "JobIDRaw",
    "State",
    "ExitCode",
    "DerivedExitCode",
    "Cluster",
    "Account",
    "Partition",
    "QOS",
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
    "QOS%64",
    "NNodes%16",
    "NCPUS%16",
    "ReqTRES%512",
    "AllocTRES%512",
)
_EXPECTED_REQ_TRES = {
    "billing": "8",
    "cpu": "8",
    "gres/gpu": "1",
    "mem": "96G",
    "node": "1",
}
_EXPECTED_ALLOC_TRES = {
    **_EXPECTED_REQ_TRES,
    "gres/gpu:l20": "1",
}
_ROW_KEYS = {
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
    "qos",
    "n_nodes",
    "n_cpus",
    "req_tres",
    "alloc_tres",
}


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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _array_job_id(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError("array_job_id must be a positive decimal Slurm job ID")
    return value


def _valid_utc(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _strict_json(raw: bytes, *, name: str) -> object:
    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def _real_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is missing or inaccessible") from error
    if not stat.S_ISDIR(metadata.st_mode) or absolute.is_symlink() or resolved != absolute:
        raise ValueError(f"{name} must be a canonical real directory")
    return absolute


def _real_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is missing or inaccessible") from error
    if not stat.S_ISREG(metadata.st_mode) or absolute.is_symlink() or resolved != absolute:
        raise ValueError(f"{name} must be a canonical regular non-symlink file")
    return absolute


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except PermissionError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, raw: bytes, *, name: str) -> None:
    destination = path.absolute()
    parent = _real_directory(destination.parent, name=f"{name} parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {name}: {destination}")
    descriptor, staged_text = tempfile.mkstemp(prefix=f".{destination.name}.staged-", dir=parent)
    staged = Path(staged_text).absolute()
    os.chmod(staged, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, destination, follow_symlinks=False)
    except FileExistsError:
        staged.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {name}: {destination}") from None
    finally:
        with suppress(OSError):
            staged.unlink()
    _fsync_directory(parent)


def sacct_command(array_job_id: str) -> tuple[str, ...]:
    """Return the one locked allocation-only query used for this wave."""

    checked = _array_job_id(array_job_id)
    return (
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        checked,
        f"--format={','.join(_SACCT_FORMAT_FIELDS)}",
    )


def _parse_tres(value: str, *, name: str) -> dict[str, str]:
    if not value or "\r" in value or "\n" in value or "\x00" in value:
        raise ValueError(f"{name} is empty or unsafe")
    result: dict[str, str] = {}
    for item in value.split(","):
        if item.count("=") != 1:
            raise ValueError(f"{name} is not strict comma-separated key=value data")
        key, scalar = item.split("=", 1)
        if not key or not scalar or key in result:
            raise ValueError(f"{name} has an empty or duplicate field")
        result[key] = scalar
    return result


def _parse_sacct(raw: bytes, *, array_job_id: str) -> list[dict[str, object]]:
    checked_id = _array_job_id(array_job_id)
    if not raw or len(raw) > 128 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("raw sacct bytes must be non-empty, bounded, and newline-terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("raw sacct bytes must be UTF-8") from error
    if "\r" in text or "\x00" in text:
        raise ValueError("raw sacct bytes contain unsafe characters")
    lines = text.splitlines()
    if len(lines) != len(ORDERED_SEEDS) or any(not line for line in lines):
        raise ValueError("raw sacct evidence must contain exactly five allocation rows")

    rows: list[dict[str, object]] = []
    allocation_ids: set[str] = set()
    for task, (seed, line) in enumerate(zip(ORDERED_SEEDS, lines, strict=True)):
        fields = line.split("|")
        if len(fields) != len(_SACCT_FIELDS):
            raise ValueError(f"raw sacct row {task} does not have thirteen locked columns")
        (
            job_id,
            job_id_raw,
            state,
            exit_code,
            derived_exit_code,
            cluster,
            account,
            partition,
            qos,
            n_nodes,
            n_cpus,
            req_tres,
            alloc_tres,
        ) = fields
        if (
            job_id != f"{checked_id}_{task}"
            or re.fullmatch(r"[1-9][0-9]*", job_id_raw) is None
            or job_id_raw in allocation_ids
            or state != "COMPLETED"
            or exit_code != "0:0"
            or derived_exit_code != "0:0"
            or cluster != "hpc4"
            or account != "sigroup"
            or partition != "gpu-l20"
            or qos != "l20_qos"
            or n_nodes != "1"
            or n_cpus != "8"
            or _parse_tres(req_tres, name=f"raw sacct row {task} ReqTRES") != _EXPECTED_REQ_TRES
            or _parse_tres(alloc_tres, name=f"raw sacct row {task} AllocTRES")
            != _EXPECTED_ALLOC_TRES
        ):
            raise ValueError(f"raw sacct row {task} is not the exact successful HPC4 allocation")
        allocation_ids.add(job_id_raw)
        rows.append(
            {
                "job_id": job_id,
                "job_id_raw": job_id_raw,
                "array_job_id": checked_id,
                "array_task_id": task,
                "seed": seed,
                "state": state,
                "exit_code": exit_code,
                "derived_exit_code": derived_exit_code,
                "cluster": cluster,
                "account": account,
                "partition": partition,
                "qos": qos,
                "n_nodes": 1,
                "n_cpus": 8,
                "req_tres": req_tres,
                "alloc_tres": alloc_tres,
            }
        )
    return rows


def capture_terminal_evidence(
    array_job_id: str,
    destination: str | os.PathLike[str],
) -> dict[str, object]:
    """Capture raw sacct bytes and their canonical fixed-five envelope."""

    checked_id = _array_job_id(array_job_id)
    output = Path(destination).absolute()
    if output.name in {"", ".", ".."}:
        raise ValueError("terminal evidence output filename is invalid")
    raw_path = output.with_name(f"{output.stem}.sacct.psv")
    if output.exists() or output.is_symlink() or raw_path.exists() or raw_path.is_symlink():
        raise FileExistsError("refusing to overwrite terminal scheduler evidence")
    command = sacct_command(checked_id)
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
    rows = _parse_sacct(raw, array_job_id=checked_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "fixed_five_exploratory_execution_evidence",
        "formal_claim_eligible": False,
        "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": list(command),
        "array_job_id": checked_id,
        "ordered_seeds": list(ORDERED_SEEDS),
        "rows": rows,
        "raw_sacct": {
            "filename": raw_path.name,
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
        },
    }
    _write_exclusive(raw_path, raw, name="raw sacct evidence")
    try:
        _write_exclusive(output, _canonical_json(payload), name="terminal scheduler evidence")
    except BaseException:
        raw_path.unlink(missing_ok=True)
        raise
    return payload


def verify_terminal_evidence(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_array_job_id: str,
) -> dict[str, object]:
    """Verify canonical JSON, raw sacct bytes, and all five terminal rows."""

    expected_digest = _digest(expected_sha256, name="terminal evidence SHA256")
    checked_id = _array_job_id(expected_array_job_id)
    evidence_path = _real_file(Path(path), name="terminal scheduler evidence")
    raw_json = evidence_path.read_bytes()
    if _sha256(raw_json) != expected_digest:
        raise ValueError("terminal scheduler evidence SHA256 mismatch")
    decoded = _strict_json(raw_json, name="terminal scheduler evidence")
    if raw_json != _canonical_json(decoded):
        raise ValueError("terminal scheduler evidence must use canonical JSON bytes")
    if not isinstance(decoded, Mapping) or set(decoded) != {
        "schema_version",
        "analysis_role",
        "formal_claim_eligible",
        "captured_at_utc",
        "query",
        "array_job_id",
        "ordered_seeds",
        "rows",
        "raw_sacct",
    }:
        raise ValueError("terminal scheduler evidence fields are invalid")
    if (
        decoded["schema_version"] != SCHEMA_VERSION
        or decoded["analysis_role"] != "fixed_five_exploratory_execution_evidence"
        or decoded["formal_claim_eligible"] is not False
        or decoded["array_job_id"] != checked_id
        or decoded["ordered_seeds"] != list(ORDERED_SEEDS)
        or decoded["query"] != list(sacct_command(checked_id))
        or not _valid_utc(decoded["captured_at_utc"])
    ):
        raise ValueError("terminal scheduler evidence identity is invalid")
    binding = decoded["raw_sacct"]
    if not isinstance(binding, Mapping) or set(binding) != {"filename", "sha256", "size_bytes"}:
        raise ValueError("terminal scheduler raw-byte binding fields are invalid")
    expected_raw_name = f"{evidence_path.stem}.sacct.psv"
    if binding["filename"] != expected_raw_name:
        raise ValueError("terminal scheduler evidence names an unexpected raw file")
    raw_path = _real_file(evidence_path.with_name(expected_raw_name), name="raw sacct evidence")
    raw = raw_path.read_bytes()
    if (
        binding["sha256"] != _sha256(raw)
        or isinstance(binding["size_bytes"], bool)
        or binding["size_bytes"] != len(raw)
    ):
        raise ValueError("terminal scheduler evidence does not bind its raw bytes")
    rows = _parse_sacct(raw, array_job_id=checked_id)
    encoded_rows = decoded["rows"]
    if (
        not isinstance(encoded_rows, list)
        or any(not isinstance(row, Mapping) or set(row) != _ROW_KEYS for row in encoded_rows)
        or encoded_rows != rows
    ):
        raise ValueError("terminal scheduler rows differ from the raw sacct bytes")
    return dict(decoded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("array_job_id")
    capture.add_argument("output", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("evidence", type=Path)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--array-job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "capture":
        payload = capture_terminal_evidence(arguments.array_job_id, arguments.output)
        digest = _sha256(arguments.output.read_bytes())
        result = {
            "status": "captured",
            "array_job_id": payload["array_job_id"],
            "output": str(arguments.output),
            "sha256": digest,
        }
    else:
        payload = verify_terminal_evidence(
            arguments.evidence,
            expected_sha256=arguments.expected_sha256,
            expected_array_job_id=arguments.array_job_id,
        )
        result = {
            "status": "verified",
            "array_job_id": payload["array_job_id"],
            "sha256": arguments.expected_sha256,
        }
    print(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
