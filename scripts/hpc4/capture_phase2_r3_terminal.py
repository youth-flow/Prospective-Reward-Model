#!/usr/bin/env python3
"""Prepare and finalize R3 scheduler evidence without loading model state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from smart_reward.phase2_r3_terminal import (
    finalize_completed_primary_terminal_from_files,
    finalize_continuable_primary_terminal_from_files,
    finalize_successful_profile_terminal_from_files,
    publish_profile_allocation_intent,
    sacct_terminal_command,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _publish_exclusive(path: Path, raw: bytes) -> str:
    if path.is_symlink():
        raise ValueError("output must not be a symbolic link")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("output parent must be a directory")
    destination = parent / path.name
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
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
            os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest()


def _emit(value: dict[str, object]) -> None:
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _add_primary_finalize_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--operational-bundle", type=Path, required=True)
    parser.add_argument("--operational-bundle-file-sha256", required=True)
    parser.add_argument("--runtime-closure", type=Path, required=True)
    parser.add_argument("--runtime-closure-file-sha256", required=True)
    parser.add_argument("--raw-sacct", type=Path, required=True)
    parser.add_argument("--raw-sacct-sha256", required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    intent = commands.add_parser(
        "profile-intent",
        help="publish the exact Gate-P allocation request before sbatch",
    )
    intent.add_argument("--output", type=Path, required=True)
    intent.add_argument("--cluster", required=True)
    intent.add_argument("--account", required=True)
    intent.add_argument("--partition", required=True)
    intent.add_argument("--gpu-name", required=True)
    intent.add_argument("--gpus-per-task", type=_positive_int, required=True)
    intent.add_argument("--cpus-per-task", type=_positive_int, required=True)
    intent.add_argument("--memory-bytes", type=_positive_int, required=True)
    intent.add_argument("--walltime-seconds", type=_positive_int, required=True)

    sacct = commands.add_parser(
        "capture-sacct",
        help="capture the exact single-allocation sacct row after job termination",
    )
    sacct.add_argument("--job-selector", required=True)
    sacct.add_argument("--output", type=Path, required=True)

    finalize = commands.add_parser(
        "profile-finalize",
        help="issue a successful Gate-P terminal capability from pure-data files",
    )
    finalize.add_argument("--operational-bundle", type=Path, required=True)
    finalize.add_argument("--operational-bundle-file-sha256", required=True)
    finalize.add_argument("--allocation-intent", type=Path, required=True)
    finalize.add_argument("--allocation-intent-file-sha256", required=True)
    finalize.add_argument("--runtime-receipt", type=Path, required=True)
    finalize.add_argument("--runtime-receipt-file-sha256", required=True)
    finalize.add_argument("--raw-sacct", type=Path, required=True)
    finalize.add_argument("--raw-sacct-sha256", required=True)
    finalize.add_argument("--evidence-directory", type=Path, required=True)

    primary_continuable = commands.add_parser(
        "primary-continuable-finalize",
        help="issue continuable primary scheduler evidence from pure-data files",
    )
    _add_primary_finalize_arguments(primary_continuable)

    primary_completed = commands.add_parser(
        "primary-completed-finalize",
        help="issue completed primary scheduler evidence from pure-data files",
    )
    _add_primary_finalize_arguments(primary_completed)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "profile-intent":
        intent = publish_profile_allocation_intent(
            arguments.output,
            cluster=arguments.cluster,
            account=arguments.account,
            partition=arguments.partition,
            gpu_name=arguments.gpu_name,
            gpus_per_task=arguments.gpus_per_task,
            cpus_per_task=arguments.cpus_per_task,
            memory_bytes=arguments.memory_bytes,
            requested_walltime_seconds=arguments.walltime_seconds,
        )
        _emit(
            {
                "status": "r3_gate_p_allocation_intent_published",
                "allocation_intent_sha256": intent.allocation_intent_sha256,
                "file_sha256": intent.file_sha256,
            }
        )
        return 0

    if arguments.command == "capture-sacct":
        command = sacct_terminal_command(arguments.job_selector)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "sacct failed with "
                f"exit {completed.returncode}: "
                f"{completed.stderr.decode('utf-8', errors='replace')}"
            )
        raw_sha256 = _publish_exclusive(arguments.output, completed.stdout)
        _emit(
            {
                "status": "r3_raw_sacct_published_pending_validation",
                "command": list(command),
                "raw_sha256": raw_sha256,
                "size_bytes": len(completed.stdout),
            }
        )
        return 0

    if arguments.command == "profile-finalize":
        terminal = finalize_successful_profile_terminal_from_files(
            operational_bundle_path=arguments.operational_bundle,
            expected_operational_bundle_file_sha256=(arguments.operational_bundle_file_sha256),
            allocation_intent_path=arguments.allocation_intent,
            expected_allocation_intent_file_sha256=(arguments.allocation_intent_file_sha256),
            runtime_receipt_path=arguments.runtime_receipt,
            expected_runtime_receipt_file_sha256=(arguments.runtime_receipt_file_sha256),
            raw_sacct_path=arguments.raw_sacct,
            expected_raw_sacct_sha256=arguments.raw_sacct_sha256,
            evidence_directory=arguments.evidence_directory,
        )
        status = "r3_gate_p_scheduler_terminal_validated"
    else:
        primary_finalize = (
            finalize_continuable_primary_terminal_from_files
            if arguments.command == "primary-continuable-finalize"
            else finalize_completed_primary_terminal_from_files
        )
        terminal = primary_finalize(
            operational_bundle_path=arguments.operational_bundle,
            expected_operational_bundle_file_sha256=(arguments.operational_bundle_file_sha256),
            runtime_closure_path=arguments.runtime_closure,
            expected_runtime_closure_file_sha256=(arguments.runtime_closure_file_sha256),
            raw_sacct_path=arguments.raw_sacct,
            expected_raw_sacct_sha256=arguments.raw_sacct_sha256,
            evidence_directory=arguments.evidence_directory,
        )
        status = (
            "r3_primary_continuable_scheduler_terminal_validated"
            if arguments.command == "primary-continuable-finalize"
            else "r3_primary_completed_scheduler_terminal_validated"
        )
    _emit(
        {
            "status": status,
            "manifest_file_sha256": terminal.manifest_file_sha256,
            "terminal_sha256": terminal.terminal_sha256,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
