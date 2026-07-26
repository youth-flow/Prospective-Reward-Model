#!/usr/bin/env python3
"""Publish and close the R3 Gate-C HPC4 evidence chain.

The CLI is intentionally compute-agnostic.  It accepts only scientific family
results already accepted by ``phase2_r3_controls``; it never implements or
changes a reward-model objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from contextlib import suppress
from pathlib import Path

from smart_reward.phase2_r3_artifacts import (
    publish_canonical_artifact,
    read_canonical_artifact,
)
from smart_reward.phase2_r3_controls import load_r3_controls_config
from smart_reward.phase2_r3_controls_hpc4 import (
    build_controls_aggregate,
    build_controls_execution_plan,
    build_controls_operational_profile,
    build_controls_task_closure,
    build_controls_task_terminal,
    build_profile_family_measurement_from_compute_receipt,
    build_profile_scheduler_terminal,
    validate_controls_execution_plan,
    validate_controls_operational_profile,
)
from smart_reward.phase2_r3_post_recovery_authorization import (
    publish_r3_final_authorization,
)
from smart_reward.phase2_r3_sacct_stdlib import (
    inspect_sacct_terminal_bytes,
    sacct_terminal_command,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _read(path: Path, digest: str) -> dict[str, object]:
    return read_canonical_artifact(
        path.resolve(strict=True),
        expected_file_sha256=digest,
    ).payload


def _read_raw(path: Path, digest: str) -> bytes:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("raw evidence must be a regular non-symlink file")
    before = resolved.stat()
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("raw evidence changed or differs from its expected SHA-256")
    return raw


def _publish_raw(path: Path, raw: bytes) -> str:
    destination = path.absolute()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(destination, flags, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o440)
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            destination.unlink()
        raise
    return hashlib.sha256(raw).hexdigest()


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--controls-config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-file-sha256", required=True)


def _plan_common(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile-finalize")
    profile.add_argument("--controls-config", type=Path, required=True)
    profile.add_argument(
        "--measurement",
        nargs=2,
        action="append",
        metavar=("PATH", "FILE_SHA256"),
        required=True,
    )
    profile.add_argument("--optimizer-schedule-sha256", required=True)
    profile.add_argument("--checkpoint-cadence-updates", type=_positive_int, required=True)
    profile.add_argument("--walltime-safety-margin-fraction", type=float, required=True)
    profile.add_argument("--fixed-walltime-margin-seconds", type=float, required=True)
    profile.add_argument("--memory-safety-margin-fraction", type=float, required=True)
    profile.add_argument("--cluster", required=True)
    profile.add_argument("--account", required=True)
    profile.add_argument("--partition", required=True)
    profile.add_argument("--gpu-name", required=True)
    profile.add_argument("--cpus-per-task", type=_positive_int, required=True)
    profile.add_argument("--memory-bytes", type=_positive_int, required=True)
    profile.add_argument("--array-concurrency", type=_positive_int, required=True)
    profile.add_argument("--walltime-seconds", type=_positive_int, required=True)
    profile.add_argument("--signal-lead-seconds", type=_positive_int, required=True)
    profile.add_argument("--max-scheduler-segments", type=_positive_int, required=True)
    profile.add_argument("--output", type=Path, required=True)

    profile_measurement = commands.add_parser("profile-measurement-finalize")
    profile_measurement.add_argument("--compute-receipt", type=Path, required=True)
    profile_measurement.add_argument(
        "--compute-receipt-file-sha256",
        required=True,
    )
    profile_measurement.add_argument("--raw-sacct", type=Path, required=True)
    profile_measurement.add_argument("--raw-sacct-sha256", required=True)
    profile_measurement.add_argument("--family", required=True)
    profile_measurement.add_argument("--array-job-id", required=True)
    profile_measurement.add_argument("--job-id-raw", required=True)
    profile_measurement.add_argument("--output", type=Path, required=True)

    plan = commands.add_parser("plan")
    _common(plan)
    plan.add_argument("--output", type=Path, required=True)

    inspect = commands.add_parser("inspect-plan")
    _plan_common(inspect)

    close = commands.add_parser("close-task")
    _plan_common(close)
    close.add_argument("--result", type=Path, required=True)
    close.add_argument("--result-file-sha256", required=True)
    close.add_argument("--task-id", type=_nonnegative_int, required=True)
    close.add_argument("--segment-index", type=_positive_int, required=True)
    close.add_argument("--output", type=Path, required=True)

    capture_raw = commands.add_parser("capture-sacct")
    capture_raw.add_argument("--job-selector", required=True)
    capture_raw.add_argument("--output", type=Path, required=True)

    publish_raw = commands.add_parser("publish-captured-sacct")
    publish_raw.add_argument("--source", type=Path, required=True)
    publish_raw.add_argument("--source-sha256", required=True)
    publish_raw.add_argument("--output", type=Path, required=True)

    terminal = commands.add_parser("terminal")
    _plan_common(terminal)
    terminal.add_argument("--result", type=Path, required=True)
    terminal.add_argument("--result-file-sha256", required=True)
    terminal.add_argument("--closure", type=Path, required=True)
    terminal.add_argument("--closure-file-sha256", required=True)
    terminal.add_argument("--raw-sacct", type=Path, required=True)
    terminal.add_argument("--raw-sacct-sha256", required=True)
    terminal.add_argument("--array-job-id", required=True)
    terminal.add_argument("--job-id-raw", required=True)
    terminal.add_argument("--output", type=Path, required=True)

    aggregate = commands.add_parser("aggregate")
    _plan_common(aggregate)
    aggregate.add_argument(
        "--entry",
        nargs=8,
        action="append",
        metavar=(
            "RESULT",
            "RESULT_SHA",
            "CLOSURE",
            "CLOSURE_SHA",
            "TERMINAL",
            "TERMINAL_SHA",
            "RAW",
            "RAW_SHA",
        ),
        required=True,
    )
    aggregate.add_argument("--output", type=Path, required=True)

    authorize = commands.add_parser("authorize")
    authorize.add_argument("--aggregate", type=Path, required=True)
    authorize.add_argument("--aggregate-file-sha256", required=True)
    authorize.add_argument("--gate-r-authorization", type=Path, required=True)
    authorize.add_argument("--gate-r-authorization-file-sha256", required=True)
    authorize.add_argument("--project-root", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    return parser


def _load_context(
    arguments: argparse.Namespace,
) -> tuple[
    object,
    dict[str, object],
    dict[str, object] | None,
]:
    config = load_r3_controls_config(arguments.controls_config)
    profile = _read(arguments.profile, arguments.profile_file_sha256)
    validate_controls_operational_profile(profile, controls_config=config)
    plan: dict[str, object] | None = None
    if hasattr(arguments, "plan"):
        plan = _read(arguments.plan, arguments.plan_file_sha256)
        validate_controls_execution_plan(
            plan,
            profile=profile,
            controls_config=config,
        )
    return config, profile, plan


def _emit(value: dict[str, object]) -> None:
    print(json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "profile-finalize":
        if len(arguments.measurement) != 3:
            raise ValueError("profile-finalize requires exactly three measurements")
        config = load_r3_controls_config(arguments.controls_config)
        measurements = [_read(Path(path), digest) for path, digest in arguments.measurement]
        payload = build_controls_operational_profile(
            measurements,
            controls_config=config,
            optimizer_schedule_sha256=arguments.optimizer_schedule_sha256,
            checkpoint_cadence_updates=arguments.checkpoint_cadence_updates,
            walltime_safety_margin_fraction=arguments.walltime_safety_margin_fraction,
            fixed_walltime_margin_seconds=arguments.fixed_walltime_margin_seconds,
            memory_safety_margin_fraction=arguments.memory_safety_margin_fraction,
            cluster=arguments.cluster,
            account=arguments.account,
            partition=arguments.partition,
            gpu_name=arguments.gpu_name,
            cpus_per_task=arguments.cpus_per_task,
            memory_bytes=arguments.memory_bytes,
            array_concurrency=arguments.array_concurrency,
            requested_walltime_seconds_per_segment=arguments.walltime_seconds,
            signal_lead_seconds=arguments.signal_lead_seconds,
            max_scheduler_segments=arguments.max_scheduler_segments,
        )
        artifact = publish_canonical_artifact(arguments.output.absolute(), payload)
        _emit({"status": "profile_published", "file_sha256": artifact.file_sha256})
        return 0

    if arguments.command == "profile-measurement-finalize":
        receipt = _read(
            arguments.compute_receipt,
            arguments.compute_receipt_file_sha256,
        )
        raw = _read_raw(arguments.raw_sacct, arguments.raw_sacct_sha256)
        terminal = build_profile_scheduler_terminal(
            raw,
            expected_raw_sacct_sha256=arguments.raw_sacct_sha256,
            family=arguments.family,
            array_job_id=arguments.array_job_id,
            job_id_raw=arguments.job_id_raw,
        )
        payload = build_profile_family_measurement_from_compute_receipt(
            receipt,
            scheduler_terminal=terminal,
        )
        artifact = publish_canonical_artifact(arguments.output.absolute(), payload)
        _emit(
            {
                "status": "profile_measurement_published",
                "family": payload["family"],
                "measurement_sha256": payload["measurement_sha256"],
                "file_sha256": artifact.file_sha256,
                "result_reusable_for_training": False,
            }
        )
        return 0

    if arguments.command == "plan":
        config, profile, _ = _load_context(arguments)
        payload = build_controls_execution_plan(profile, controls_config=config)
        artifact = publish_canonical_artifact(arguments.output.absolute(), payload)
        _emit({"status": "plan_published", "file_sha256": artifact.file_sha256})
        return 0

    if arguments.command == "capture-sacct":
        completed = subprocess.run(
            sacct_terminal_command(arguments.job_selector),
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"sacct failed with exit {completed.returncode}: "
                f"{completed.stderr.decode('utf-8', errors='replace')}"
            )
        digest = _publish_raw(arguments.output, completed.stdout)
        _emit({"status": "raw_sacct_published", "raw_sacct_sha256": digest})
        return 0

    if arguments.command == "publish-captured-sacct":
        raw = _read_raw(arguments.source, arguments.source_sha256)
        inspect_sacct_terminal_bytes(
            raw,
            expected_raw_sha256=arguments.source_sha256,
        )
        digest = _publish_raw(arguments.output, raw)
        _emit({"status": "raw_sacct_published", "raw_sacct_sha256": digest})
        return 0

    if arguments.command == "authorize":
        artifact = publish_r3_final_authorization(
            gate_r_authorization=arguments.gate_r_authorization,
            gate_r_authorization_file_sha256=(arguments.gate_r_authorization_file_sha256),
            gate_c_aggregate=arguments.aggregate,
            gate_c_aggregate_file_sha256=arguments.aggregate_file_sha256,
            output=arguments.output,
            project_root=arguments.project_root,
        )
        _emit(
            {
                "status": "r3_final_authorization_published",
                "file_sha256": artifact.file_sha256,
            }
        )
        return 0

    config, profile, plan = _load_context(arguments)
    if plan is None:
        raise RuntimeError("internal plan loading failed")
    if arguments.command == "inspect-plan":
        _emit(
            {
                "status": "plan_valid",
                "plan_sha256": plan["plan_sha256"],
                "git_commit": plan["git_commit"],
                "container_sha256": plan["container_sha256"],
                "controls_config_file_sha256": plan["controls_config_file_sha256"],
                "controls_config_semantic_sha256": plan["controls_config_semantic_sha256"],
                "resources": plan["resources"],
                "arrays": plan["arrays"],
            }
        )
        return 0

    if arguments.command == "aggregate":
        if len(arguments.entry) != 9:
            raise ValueError("aggregate requires exactly nine ordered entries")
        entries = []
        for (
            result_path,
            result_sha,
            closure_path,
            closure_sha,
            terminal_path,
            terminal_sha,
            raw_path,
            raw_sha,
        ) in arguments.entry:
            entries.append(
                (
                    _read(Path(result_path), result_sha),
                    _read(Path(closure_path), closure_sha),
                    _read(Path(terminal_path), terminal_sha),
                    _read_raw(Path(raw_path), raw_sha),
                )
            )
        payload = build_controls_aggregate(
            entries,
            plan=plan,
            profile=profile,
            controls_config=config,
        )
    else:
        result = _read(arguments.result, arguments.result_file_sha256)
        if arguments.command == "close-task":
            payload = build_controls_task_closure(
                plan,
                profile=profile,
                controls_config=config,
                task_id=arguments.task_id,
                segment_index=arguments.segment_index,
                family_result=result,
            )
        else:
            closure = _read(arguments.closure, arguments.closure_file_sha256)
            raw = _read_raw(arguments.raw_sacct, arguments.raw_sacct_sha256)
            payload = build_controls_task_terminal(
                raw,
                expected_raw_sacct_sha256=arguments.raw_sacct_sha256,
                plan=plan,
                profile=profile,
                controls_config=config,
                closure=closure,
                family_result=result,
                array_job_id=arguments.array_job_id,
                job_id_raw=arguments.job_id_raw,
            )
    artifact = publish_canonical_artifact(arguments.output.absolute(), payload)
    _emit({"status": f"{arguments.command}_published", "file_sha256": artifact.file_sha256})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
