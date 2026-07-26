#!/usr/bin/env python3
"""Capture or verify post-recovery pilot terminal evidence using stdlib only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

ORDERED_SEEDS = (20260801, 20260802, 20260803)
PILOT_PHASES = frozenset({"calibration", "freeze"})
TERMINAL_SCHEMA = "prorm-phase2-post-recovery-pilot-terminal/v1"
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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _real_file(path: Path, *, name: str, maximum_bytes: int = 64 * 1024) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
        or metadata.st_size < 1
        or metadata.st_size > maximum_bytes
    ):
        raise ValueError(f"{name} must be a bounded canonical regular file")
    return absolute


def _valid_utc(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return parsed.tzinfo is None


def _array_job_id(value: str) -> str:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError("array_job_id must be a positive decimal Slurm job ID")
    return value


def sacct_command(array_job_id: str) -> tuple[str, ...]:
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


def _parse_sacct(raw: bytes, *, array_job_id: str) -> list[dict[str, object]]:
    checked_id = _array_job_id(array_job_id)
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("raw sacct bytes must be non-empty, bounded, and newline-terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("raw sacct bytes must be UTF-8") from error
    if "\r" in text or any(ord(character) < 32 and character != "\n" for character in text):
        raise ValueError("raw sacct bytes contain forbidden control characters")
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
            job_id != f"{checked_id}_{task}"
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
                "array_job_id": checked_id,
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


def _write_exclusive(path: Path, raw: bytes, *, name: str) -> None:
    destination = path.absolute()
    parent = destination.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
        or destination.exists()
        or destination.is_symlink()
    ):
        raise FileExistsError(f"refusing unsafe or existing {name}: {destination}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(parent)


def capture_terminal(
    array_job_id: str,
    output_path: Path,
    *,
    pilot_phase: str,
) -> dict[str, object]:
    checked_id = _array_job_id(array_job_id)
    if pilot_phase not in PILOT_PHASES:
        raise ValueError("pilot_phase must be calibration or freeze")
    output = output_path.absolute()
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
        "schema_version": TERMINAL_SCHEMA,
        "pilot_phase": pilot_phase,
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


def verify_terminal(
    path: Path,
    *,
    expected_sha256: str,
    array_job_id: str,
    pilot_phase: str,
) -> dict[str, object]:
    expected = _digest(expected_sha256, name="terminal evidence SHA256")
    checked_id = _array_job_id(array_job_id)
    if pilot_phase not in PILOT_PHASES:
        raise ValueError("pilot_phase must be calibration or freeze")
    evidence_path = _real_file(
        path,
        name="terminal scheduler evidence",
        maximum_bytes=1024 * 1024,
    )
    raw_json = evidence_path.read_bytes()
    if _sha256(raw_json) != expected:
        raise ValueError("terminal scheduler evidence SHA256 mismatch")
    try:
        value = json.loads(
            raw_json.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"terminal evidence contains non-finite constant {token!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("terminal scheduler evidence is not strict JSON") from error
    if not isinstance(value, dict) or raw_json != _canonical_json(value):
        raise ValueError("terminal scheduler evidence must be canonical JSON")
    if set(value) != {
        "schema_version",
        "pilot_phase",
        "captured_at_utc",
        "query",
        "array_job_id",
        "ordered_seeds",
        "rows",
        "raw_sacct",
    }:
        raise ValueError("terminal scheduler evidence fields differ")
    if (
        value["schema_version"] != TERMINAL_SCHEMA
        or value["pilot_phase"] != pilot_phase
        or value["array_job_id"] != checked_id
        or value["ordered_seeds"] != list(ORDERED_SEEDS)
        or value["query"] != list(sacct_command(checked_id))
        or not _valid_utc(value["captured_at_utc"])
    ):
        raise ValueError("terminal scheduler evidence identity is invalid")
    binding = value["raw_sacct"]
    if not isinstance(binding, dict) or set(binding) != {"filename", "sha256", "size_bytes"}:
        raise ValueError("terminal raw-sacct binding fields differ")
    raw_name = f"{evidence_path.stem}.sacct.psv"
    if binding["filename"] != raw_name:
        raise ValueError("terminal scheduler evidence names an unexpected raw file")
    raw_path = _real_file(
        evidence_path.with_name(raw_name),
        name="raw sacct evidence",
        maximum_bytes=64 * 1024,
    )
    raw = raw_path.read_bytes()
    if binding["sha256"] != _sha256(raw) or binding["size_bytes"] != len(raw):
        raise ValueError("terminal scheduler evidence does not bind its raw bytes")
    rows = _parse_sacct(raw, array_job_id=checked_id)
    if value["rows"] != rows:
        raise ValueError("terminal scheduler rows differ from the raw sacct bytes")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("array_job_id")
    capture.add_argument("output", type=Path)
    capture.add_argument("--pilot-phase", choices=tuple(sorted(PILOT_PHASES)), required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("evidence", type=Path)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--array-job-id", required=True)
    verify.add_argument("--pilot-phase", choices=tuple(sorted(PILOT_PHASES)), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "capture":
        payload = capture_terminal(
            arguments.array_job_id,
            arguments.output,
            pilot_phase=arguments.pilot_phase,
        )
    else:
        payload = verify_terminal(
            arguments.evidence,
            expected_sha256=arguments.expected_sha256,
            array_job_id=arguments.array_job_id,
            pilot_phase=arguments.pilot_phase,
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "array_job_id": payload["array_job_id"],
                "pilot_phase": payload["pilot_phase"],
                "ordered_seeds": payload["ordered_seeds"],
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
