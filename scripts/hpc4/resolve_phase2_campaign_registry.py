#!/usr/bin/env python3
"""Validate the fixed-wave formal registry and resolve its terminal heads."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_phase2_terminal import validate_terminal

SEEDS = tuple(range(20260901, 20260931))
WAVE_TASKS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (16, 17, 18, 19),
    (20, 21, 22, 23),
    (24, 25, 26, 27),
    (28, 29),
)
PLAN_SCHEMA = "prorm-phase2-fixed-wave-campaign-plan/v1"
ADMISSION_SCHEMA = "prorm-phase2-wave-admission/v1"
SUBMISSION_SCHEMA = "prorm-phase2-campaign-submission/v3"
RETRY_POLICY = "single_predeclared_attempt_no_retry"
ADMISSION_RULE = "predecessor_terminal_completeness_only_outcome_independent"
HEX64 = re.compile(r"[0-9a-f]{64}")
TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


def die(message: str) -> None:
    raise SystemExit(message)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def load(path: Path, *, canonical: bool = False) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        die(f"registry record is missing or unsafe: {path}")
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
        die(f"registry record must contain one object: {path}")
    if canonical and raw != _canonical_bytes(value):
        die(f"registry record is not canonical JSON: {path}")
    return value, raw


def _wave_payload(index: int) -> dict[str, object]:
    tasks = WAVE_TASKS[index]
    return {
        "wave_index": index,
        "array_spec": f"{tasks[0]}-{tasks[-1]}%2",
        "array_task_ids": list(tasks),
        "seeds": [SEEDS[task] for task in tasks],
    }


def _validate_plan(
    path: Path,
    *,
    design: str,
    base: str,
    commit: str,
    image: str,
    inventory: str,
) -> tuple[dict[str, Any], str]:
    value, raw = load(path, canonical=True)
    expected_fields = {
        "schema_version",
        "status",
        "phase2_design_sha256",
        "base_config_hash",
        "git_commit",
        "accepted_freeze_aggregate_sha256",
        "ordered_seeds",
        "attempt_index",
        "retry_policy",
        "replacement_seed_allowed",
        "optional_stopping_allowed",
        "max_submitted_tasks",
        "max_running_tasks",
        "waves",
        "job_tuple",
        "producer",
        "created_at_utc",
    }
    expected_job_fields = {
        "account",
        "partition",
        "qos",
        "nodes",
        "tasks",
        "cpus_per_task",
        "memory",
        "gpus_per_node",
        "walltime",
        "no_requeue",
        "held_before_registry_commit",
        "script",
        "script_file_sha256",
    }
    expected_producer_fields = {
        "overlay_file_sha256",
        "base_file_sha256",
        "identities_file_sha256",
        "image_sha256",
        "hf_inventory_sha256",
    }
    job = value.get("job_tuple")
    producer = value.get("producer")
    expected_waves = [_wave_payload(index) for index in range(len(WAVE_TASKS))]
    job_file_sha = hashlib.sha256(
        Path(__file__).with_name("phase2_confirmatory.sbatch").read_bytes()
    ).hexdigest()
    if (
        set(value) != expected_fields
        or value.get("schema_version") != PLAN_SCHEMA
        or value.get("status") != "precommitted_before_first_slurm_submission"
        or value.get("phase2_design_sha256") != design
        or value.get("base_config_hash") != base
        or value.get("git_commit") != commit
        or HEX64.fullmatch(str(value.get("accepted_freeze_aggregate_sha256", ""))) is None
        or value.get("ordered_seeds") != list(SEEDS)
        or value.get("attempt_index") != 1
        or value.get("retry_policy") != RETRY_POLICY
        or value.get("replacement_seed_allowed") is not False
        or value.get("optional_stopping_allowed") is not False
        or value.get("max_submitted_tasks") != 4
        or value.get("max_running_tasks") != 2
        or value.get("waves") != expected_waves
        or TIMESTAMP.fullmatch(str(value.get("created_at_utc", ""))) is None
        or not isinstance(job, dict)
        or set(job) != expected_job_fields
        or job.get("account") != "sigroup"
        or job.get("partition") != "gpu-l20"
        or job.get("qos") != "l20_qos"
        or job.get("nodes") != 1
        or job.get("tasks") != 1
        or job.get("cpus_per_task") != 8
        or job.get("memory") != "64G"
        or job.get("gpus_per_node") != 1
        or re.fullmatch(
            r"(?:[1-9][0-9]*-)?[0-9]{2}:[0-9]{2}:[0-9]{2}",
            str(job.get("walltime", "")),
        )
        is None
        or job.get("no_requeue") is not True
        or job.get("held_before_registry_commit") is not True
        or job.get("script") != "scripts/hpc4/phase2_confirmatory.sbatch"
        or job.get("script_file_sha256") != job_file_sha
        or not isinstance(producer, dict)
        or set(producer) != expected_producer_fields
        or producer.get("image_sha256") != image
        or producer.get("hf_inventory_sha256") != inventory
        or any(
            HEX64.fullmatch(str(producer.get(field, ""))) is None
            for field in (
                "overlay_file_sha256",
                "base_file_sha256",
                "identities_file_sha256",
            )
        )
    ):
        die("formal fixed-wave campaign plan is invalid")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_registry_root(root: Path, design: str) -> dict[str, Path]:
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.name != design
        or HEX64.fullmatch(design) is None
    ):
        die("formal design root is unsafe or identity-mismatched")
    registry = root / "campaign-registry"
    directories = {
        "registry": registry,
        "admissions": registry / "admissions",
        "submissions": registry / "submissions",
        "executions": registry / "executions",
        "recoveries": registry / "recoveries",
        "scheduler_terminals": registry / "scheduler-terminals",
        "staging": registry / ".staging",
    }
    for directory in directories.values():
        if directory.is_symlink() or not directory.is_dir():
            die(f"formal campaign registry directory is missing or unsafe: {directory}")
    allowed_registry_entries = {
        ".staging",
        "admissions",
        "campaign-plan.json",
        "submissions",
        "executions",
        "recoveries",
        "scheduler-terminals",
        "registry.lock",
    }
    if {path.name for path in registry.iterdir()} != allowed_registry_entries:
        die("campaign registry root contains an unexpected entry")
    lock = registry / "registry.lock"
    if lock.is_symlink() or not lock.is_file():
        die("campaign registry lock is missing or unsafe")
    staging_pattern = re.compile(
        r"(?:campaign-plan|admission-wave-[0-7]|scheduler-request-wave-[0-7]|"
        r"submission-array-[1-9][0-9]*|"
        r"execution-seed-[1-9][0-9]*-attempt-1)\.[A-Za-z0-9]+"
    )
    for path in directories["staging"].iterdir():
        if staging_pattern.fullmatch(path.name) is None or path.is_symlink() or not path.is_file():
            die(f"unsafe non-authoritative registry staging residue: {path}")
    return directories


def _validate_scheduler_request(
    value: object,
    *,
    expected_sha256: str,
    plan: dict[str, Any],
    plan_sha256: str,
    wave_index: int,
    array_job_id: str,
) -> None:
    expected_fields = {
        "schema_version",
        "captured_while_held",
        "raw_scontrol_record",
        "raw_scontrol_sha256",
        "normalized",
    }
    expected_normalized_fields = {
        "array_job_id",
        "job_name",
        "array_spec",
        "array_task_throttle",
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
    if not isinstance(value, dict) or set(value) != expected_fields:
        die("held scheduler request evidence fields are invalid")
    normalized = value.get("normalized")
    raw = value.get("raw_scontrol_record")
    expected_command = str(
        Path(__file__).resolve().parents[2] / "scripts" / "hpc4" / "phase2_confirmatory.sbatch"
    )
    expected_work_dir = str(Path(__file__).resolve().parents[2])
    wave = _wave_payload(wave_index)
    if (
        hashlib.sha256(_canonical_bytes(value)).hexdigest() != expected_sha256
        or value.get("schema_version") != "prorm-phase2-held-scheduler-request/v1"
        or value.get("captured_while_held") is not True
        or not isinstance(raw, str)
        or not raw
        or "\r" in raw
        or "\n" in raw
        or hashlib.sha256(raw.encode()).hexdigest() != value.get("raw_scontrol_sha256")
        or not isinstance(normalized, dict)
        or set(normalized) != expected_normalized_fields
        or normalized.get("array_job_id") != array_job_id
        or normalized.get("job_name") != f"prorm-p2-{plan_sha256[:12]}-w{wave_index}"
        or normalized.get("array_spec") != wave["array_spec"]
        or normalized.get("array_task_throttle") != 2
        or normalized.get("account") != "sigroup"
        or normalized.get("partition") != "gpu-l20"
        or normalized.get("qos") != "l20_qos"
        or normalized.get("nodes") != 1
        or normalized.get("tasks") != 1
        or normalized.get("cpus") != 8
        or normalized.get("cpus_per_task") != 8
        or normalized.get("memory") != "64G"
        or normalized.get("gpus_per_node") != 1
        or normalized.get("walltime") != plan["job_tuple"]["walltime"]
        or normalized.get("tres") != {"cpu": "8", "gres/gpu": "1", "mem": "64G", "node": "1"}
        or re.fullmatch(
            r"gres(?::|/)gpu(?::[A-Za-z0-9_.-]+)?:1",
            str(normalized.get("tres_per_node", "")),
        )
        is None
        or normalized.get("requeue") is not False
        or normalized.get("restarts") != 0
        or normalized.get("command") != expected_command
        or normalized.get("work_dir") != expected_work_dir
    ):
        die("held scheduler request evidence disagrees with the campaign plan")
    fields: dict[str, str] = {}
    for token in raw.split():
        if "=" not in token:
            continue
        key, item = token.split("=", 1)
        if key in fields:
            die(f"held scheduler request repeats scontrol field {key}")
        fields[key] = item
    tres: dict[str, str] = {}
    for entry in fields.get("TRES", "").split(","):
        if "=" not in entry:
            continue
        key, item = entry.split("=", 1)
        if key in tres:
            die(f"held scheduler request repeats TRES field {key}")
        tres[key] = item
    if (
        fields.get("ArrayJobId", fields.get("JobId")) != array_job_id
        or fields.get("JobName") != normalized["job_name"]
        or fields.get("ArrayTaskId") != wave["array_spec"]
        or fields.get("ArrayTaskThrottle") != "2"
        or fields.get("JobState") != "PENDING"
        or fields.get("Reason") != "JobHeldUser"
        or fields.get("Account") != "sigroup"
        or fields.get("Partition") != "gpu-l20"
        or fields.get("QOS") != "l20_qos"
        or fields.get("Requeue") != "0"
        or fields.get("Restarts") != "0"
        or fields.get("NumNodes") not in {"1", "1-1"}
        or fields.get("NumTasks") != "1"
        or fields.get("NumCPUs") != "8"
        or fields.get("CPUs/Task") != "8"
        or fields.get("MinMemoryNode") != "64G"
        or fields.get("TimeLimit") != normalized["walltime"]
        or {key: tres.get(key) for key in normalized["tres"]} != normalized["tres"]
        or fields.get("TresPerNode") != normalized["tres_per_node"]
        or fields.get("Command") != expected_command
        or fields.get("WorkDir") != expected_work_dir
    ):
        die("raw held scontrol evidence disagrees with its normalized identity")


def _validate_submissions(
    submissions: Path,
    *,
    plan: dict[str, Any],
    plan_sha256: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    paths = sorted(submissions.iterdir())
    by_wave: dict[int, dict[str, Any]] = {}
    array_ids: set[str] = set()
    expected_fields = {
        "schema_version",
        "status",
        "campaign_plan_sha256",
        "wave_admission_sha256",
        "scheduler_request_sha256",
        "scheduler_request",
        "wave_index",
        "phase2_design_sha256",
        "base_config_hash",
        "git_commit",
        "accepted_freeze_aggregate_sha256",
        "array_job_id",
        "submitted_cluster",
        "array_spec",
        "attempt_index",
        "entries",
        "job_tuple",
        "producer",
        "replacement_seed_allowed",
        "created_at_utc",
    }
    for path in paths:
        if re.fullmatch(r"array-[1-9][0-9]*\.json", path.name) is None:
            die(f"unexpected submission-registry entry: {path}")
        value, raw = load(path, canonical=True)
        wave_index = value.get("wave_index")
        array_id = value.get("array_job_id")
        if (
            set(value) != expected_fields
            or value.get("schema_version") != SUBMISSION_SCHEMA
            or value.get("status") != "committed_while_slurm_held"
            or value.get("campaign_plan_sha256") != plan_sha256
            or HEX64.fullmatch(str(value.get("wave_admission_sha256", ""))) is None
            or HEX64.fullmatch(str(value.get("scheduler_request_sha256", ""))) is None
            or not isinstance(wave_index, int)
            or isinstance(wave_index, bool)
            or not 0 <= wave_index < len(WAVE_TASKS)
            or value.get("phase2_design_sha256") != plan["phase2_design_sha256"]
            or value.get("base_config_hash") != plan["base_config_hash"]
            or value.get("git_commit") != plan["git_commit"]
            or value.get("accepted_freeze_aggregate_sha256")
            != plan["accepted_freeze_aggregate_sha256"]
            or not isinstance(array_id, str)
            or re.fullmatch(r"[1-9][0-9]*", array_id) is None
            or path.name != f"array-{array_id}.json"
            or value.get("submitted_cluster") != "hpc4"
            or value.get("array_spec") != _wave_payload(wave_index)["array_spec"]
            or value.get("attempt_index") != 1
            or value.get("job_tuple") != plan["job_tuple"]
            or value.get("producer") != plan["producer"]
            or value.get("replacement_seed_allowed") is not False
            or TIMESTAMP.fullmatch(str(value.get("created_at_utc", ""))) is None
            or wave_index in by_wave
            or array_id in array_ids
        ):
            die(f"fixed-wave submission registry identity is invalid: {path}")
        tasks = WAVE_TASKS[wave_index]
        expected_entries = [
            {
                "seed": SEEDS[task],
                "attempt_index": 1,
                "array_job_id": array_id,
                "array_task_id": task,
            }
            for task in tasks
        ]
        if value.get("entries") != expected_entries:
            die(f"fixed-wave submission entries are invalid: {path}")
        _validate_scheduler_request(
            value.get("scheduler_request"),
            expected_sha256=str(value["scheduler_request_sha256"]),
            plan=plan,
            plan_sha256=plan_sha256,
            wave_index=wave_index,
            array_job_id=array_id,
        )
        value["_path"] = path
        value["_sha256"] = hashlib.sha256(raw).hexdigest()
        by_wave[wave_index] = value
        array_ids.add(array_id)
    if sorted(by_wave) != list(range(len(by_wave))):
        die("fixed-wave submissions must be one gap-free ordered prefix")
    return paths, by_wave


def _terminal_snapshot_entry(
    root: Path,
    *,
    seed: int,
    terminal: Path,
) -> dict[str, object]:
    resolved_root = root.resolve()
    resolved_terminal = terminal.resolve()
    try:
        relative = resolved_terminal.relative_to(resolved_root)
    except ValueError:
        die(f"terminal for seed {seed} escapes the formal design root")
    marker_candidates = [
        terminal.parent / name
        for name in ("SUCCESS", "FAILED", "SCHEDULER_FAILED")
        if (terminal.parent / name).exists() or (terminal.parent / name).is_symlink()
    ]
    if (
        terminal.is_symlink()
        or not resolved_terminal.is_file()
        or f"seed-{seed}" not in relative.parts
        or len(marker_candidates) != 1
        or marker_candidates[0].is_symlink()
        or not marker_candidates[0].is_file()
    ):
        die(f"terminal snapshot path is unsafe or seed-mismatched for seed {seed}")
    marker = marker_candidates[0].resolve()
    try:
        marker_relative = marker.relative_to(resolved_root)
    except ValueError:
        die(f"terminal marker for seed {seed} escapes the formal design root")
    return {
        "seed": seed,
        "terminal_relative_path": relative.as_posix(),
        "terminal_sha256": hashlib.sha256(resolved_terminal.read_bytes()).hexdigest(),
        "marker_relative_path": marker_relative.as_posix(),
        "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
    }


def _snapshot_sha256(snapshot: list[dict[str, object]]) -> str:
    return hashlib.sha256(_canonical_bytes(snapshot)).hexdigest()


def _validate_admissions(
    *,
    root: Path,
    admissions: Path,
    plan: dict[str, Any],
    plan_sha256: str,
    submissions: dict[int, dict[str, Any]],
    terminals_by_seed: dict[int, Path],
) -> dict[int, dict[str, Any]]:
    paths = sorted(admissions.iterdir())
    by_wave: dict[int, dict[str, Any]] = {}
    expected_fields = {
        "schema_version",
        "status",
        "campaign_plan_sha256",
        "wave_index",
        "wave",
        "admission_rule",
        "predecessor_wave_index",
        "predecessor_admission_sha256",
        "predecessor_submission_sha256",
        "predecessor_terminal_snapshot",
        "predecessor_terminal_snapshot_sha256",
        "created_at_utc",
    }
    for path in paths:
        match = re.fullmatch(r"wave-([0-7])\.json", path.name)
        if match is None:
            die(f"unexpected wave-admission registry entry: {path}")
        wave_index = int(match.group(1))
        value, raw = load(path, canonical=True)
        predecessor_index = None if wave_index == 0 else wave_index - 1
        predecessor_admission_sha = (
            None if predecessor_index is None else by_wave.get(predecessor_index, {}).get("_sha256")
        )
        predecessor_submission_sha = (
            None
            if predecessor_index is None
            else submissions.get(predecessor_index, {}).get("_sha256")
        )
        predecessor_tasks = () if predecessor_index is None else WAVE_TASKS[predecessor_index]
        if any(SEEDS[task] not in terminals_by_seed for task in predecessor_tasks):
            die(
                f"wave {wave_index} admission was committed before its predecessor "
                "had complete terminal evidence"
            )
        snapshot = [
            _terminal_snapshot_entry(
                root,
                seed=SEEDS[task],
                terminal=terminals_by_seed[SEEDS[task]],
            )
            for task in predecessor_tasks
        ]
        if (
            set(value) != expected_fields
            or value.get("schema_version") != ADMISSION_SCHEMA
            or value.get("status") != "committed_before_current_wave_scheduler_submission"
            or value.get("campaign_plan_sha256") != plan_sha256
            or value.get("wave_index") != wave_index
            or value.get("wave") != _wave_payload(wave_index)
            or value.get("admission_rule") != ADMISSION_RULE
            or value.get("predecessor_wave_index") != predecessor_index
            or value.get("predecessor_admission_sha256") != predecessor_admission_sha
            or value.get("predecessor_submission_sha256") != predecessor_submission_sha
            or value.get("predecessor_terminal_snapshot") != snapshot
            or value.get("predecessor_terminal_snapshot_sha256") != _snapshot_sha256(snapshot)
            or TIMESTAMP.fullmatch(str(value.get("created_at_utc", ""))) is None
            or wave_index in by_wave
        ):
            die(f"wave-admission registry identity is invalid: {path}")
        value["_path"] = path
        value["_sha256"] = hashlib.sha256(raw).hexdigest()
        by_wave[wave_index] = value
    if sorted(by_wave) != list(range(len(by_wave))):
        die("wave admissions must be one gap-free ordered prefix")
    if len(by_wave) not in {len(submissions), len(submissions) + 1}:
        die("wave admissions must cover every submission and at most one ready wave")
    for wave_index, submission in submissions.items():
        admission = by_wave.get(wave_index)
        if admission is None or submission.get("wave_admission_sha256") != admission.get("_sha256"):
            die("fixed-wave submission does not bind its immutable admission receipt")
    return by_wave


def _registry_keys(
    directory: Path,
    *,
    label: str,
    registered_seeds: set[int],
) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    pattern = re.compile(r"seed-([1-9][0-9]*)-attempt-([1-9][0-9]*)\.json")
    for path in sorted(directory.iterdir()):
        match = pattern.fullmatch(path.name)
        if match is None:
            die(f"unexpected {label}-registry entry: {path}")
        key = (int(match.group(1)), int(match.group(2)))
        if key in keys or key[0] not in registered_seeds or key[1] != 1:
            die(f"{label}-registry record has no unique committed attempt: {key}")
        load(path, canonical=True)
        keys.add(key)
    return keys


def _terminal_for_seed(
    root: Path,
    *,
    seed: int,
    design: str,
    base: str,
    commit: str,
    image: str,
    inventory: str,
    scheduler_terminal_keys: set[tuple[int, int]],
) -> Path | None:
    seed_root = root / f"seed-{seed}"
    if not seed_root.exists() and not seed_root.is_symlink():
        return None
    if (
        seed_root.is_symlink()
        or not seed_root.is_dir()
        or seed_root.resolve() != seed_root.absolute()
    ):
        die(f"formal seed root is unsafe for seed {seed}")
    attempt_names = {path.name for path in seed_root.iterdir() if path.name.startswith("attempt-")}
    if attempt_names != {"attempt-1"}:
        die(f"physical attempts disagree with attempt-1 for seed {seed}")
    attempt_root = seed_root / "attempt-1"
    jobs = [path for path in attempt_root.iterdir() if path.name.startswith("job-")]
    if len(jobs) != 1 or jobs[0].is_symlink() or not jobs[0].is_dir():
        die(f"registered attempt lacks one unique claimed job for seed {seed}")
    terminal_candidates = [
        path
        for path in jobs[0].iterdir()
        if path.name
        in {
            "phase2-success-terminal.json",
            "phase2-failure-terminal.json",
        }
    ]
    if not terminal_candidates:
        return None
    if len(terminal_candidates) != 1:
        die(f"registered attempt has multiple terminal manifests for seed {seed}")
    scheduler_marker = jobs[0] / "SCHEDULER_FAILED"
    scheduler_captured = scheduler_marker.exists() or scheduler_marker.is_symlink()
    if scheduler_captured != ((seed, 1) in scheduler_terminal_keys):
        die(f"scheduler-terminal registry/marker ownership disagrees for seed {seed}")
    terminal = terminal_candidates[0]
    with contextlib.redirect_stdout(io.StringIO()):
        validate_terminal(
            [
                str(terminal),
                str(seed),
                design,
                base,
                commit,
                image,
                inventory,
            ]
        )
    return terminal.resolve()


def _resolve(
    root: Path,
    *,
    design: str,
    base: str,
    commit: str,
    image: str,
    inventory: str,
) -> tuple[dict[str, object], list[Path]]:
    directories = _validate_registry_root(root, design)
    plan, plan_sha = _validate_plan(
        directories["registry"] / "campaign-plan.json",
        design=design,
        base=base,
        commit=commit,
        image=image,
        inventory=inventory,
    )
    _, submissions = _validate_submissions(
        directories["submissions"],
        plan=plan,
        plan_sha256=plan_sha,
    )
    registered_seeds = {
        seed
        for submission in submissions.values()
        for seed in _wave_payload(int(submission["wave_index"]))["seeds"]
    }
    _registry_keys(
        directories["executions"],
        label="execution",
        registered_seeds=registered_seeds,
    )
    recovery_keys = _registry_keys(
        directories["recoveries"],
        label="recovery",
        registered_seeds=registered_seeds,
    )
    scheduler_keys = _registry_keys(
        directories["scheduler_terminals"],
        label="scheduler-terminal",
        registered_seeds=registered_seeds,
    )
    if recovery_keys:
        die("formal no-retry campaign must not contain recovery authorization")

    observed_seed_roots = {path.name for path in root.iterdir() if path.name.startswith("seed-")}
    allowed_seed_roots = {f"seed-{seed}" for seed in registered_seeds}
    if not observed_seed_roots <= allowed_seed_roots:
        die("physical seed roots contain an unregistered or replacement seed")

    terminals: list[Path] = []
    terminal_seeds: set[int] = set()
    terminals_by_seed: dict[int, Path] = {}
    for wave_index in sorted(submissions):
        tasks = WAVE_TASKS[wave_index]
        wave_terminals: list[Path] = []
        for task in tasks:
            terminal = _terminal_for_seed(
                root,
                seed=SEEDS[task],
                design=design,
                base=base,
                commit=commit,
                image=image,
                inventory=inventory,
                scheduler_terminal_keys=scheduler_keys,
            )
            if terminal is not None:
                wave_terminals.append(terminal)
                terminal_seeds.add(SEEDS[task])
                terminals_by_seed[SEEDS[task]] = terminal
        if len(wave_terminals) not in {0, len(tasks)} and wave_index != len(submissions) - 1:
            die("a non-final submitted wave is only partially terminalized")
        if wave_index < len(submissions) - 1 and len(wave_terminals) != len(tasks):
            die("a later wave was submitted before its predecessor became terminal")
        terminals.extend(wave_terminals)

    admissions = _validate_admissions(
        root=root,
        admissions=directories["admissions"],
        plan=plan,
        plan_sha256=plan_sha,
        submissions=submissions,
        terminals_by_seed=terminals_by_seed,
    )

    if submissions:
        latest_index = len(submissions) - 1
        latest_tasks = WAVE_TASKS[latest_index]
        latest_terminal = sum(SEEDS[task] in terminal_seeds for task in latest_tasks)
        if latest_terminal != len(latest_tasks):
            latest = submissions[latest_index]
            return (
                {
                    "status": "active",
                    "campaign_plan_sha256": plan_sha,
                    "wave_index": latest_index,
                    "array_spec": latest["array_spec"],
                    "array_job_id": latest["array_job_id"],
                    "wave_admission_sha256": admissions[latest_index]["_sha256"],
                    "walltime": plan["job_tuple"]["walltime"],
                    "plan_created_at_utc": plan["created_at_utc"],
                },
                terminals,
            )

    if len(submissions) < len(WAVE_TASKS):
        next_index = len(submissions)
        wave = _wave_payload(next_index)
        return (
            {
                "status": "ready",
                "campaign_plan_sha256": plan_sha,
                "wave_admission_sha256": (
                    admissions[next_index]["_sha256"] if next_index in admissions else None
                ),
                "walltime": plan["job_tuple"]["walltime"],
                "plan_created_at_utc": plan["created_at_utc"],
                **wave,
            },
            terminals,
        )

    if len(terminals) != len(SEEDS) or len(set(terminals)) != len(SEEDS):
        die("complete fixed-wave registry did not resolve exactly 30 unique terminals")
    expected_seed_roots = {f"seed-{seed}" for seed in SEEDS}
    if observed_seed_roots != expected_seed_roots:
        die("complete fixed-wave registry is missing physical seed roots")
    return (
        {
            "status": "complete",
            "campaign_plan_sha256": plan_sha,
            "num_waves": len(WAVE_TASKS),
            "num_terminal_seeds": len(terminals),
            "walltime": plan["job_tuple"]["walltime"],
            "plan_created_at_utc": plan["created_at_utc"],
        },
        terminals,
    )


def _materialize_admission(
    output: Path,
    *,
    root: Path,
    state: dict[str, object],
    terminals: list[Path],
) -> str:
    if (
        state.get("status") != "ready"
        or state.get("wave_admission_sha256") is not None
        or not isinstance(state.get("wave_index"), int)
    ):
        die("a new wave admission can only be materialized for one unadmitted ready wave")
    wave_index = int(state["wave_index"])
    registry = root / "campaign-registry"
    staging = registry / ".staging"
    if (
        output.is_symlink()
        or not output.is_file()
        or output.stat().st_size != 0
        or output.parent.resolve() != staging.resolve()
        or re.fullmatch(
            rf"admission-wave-{wave_index}\.[A-Za-z0-9]+",
            output.name,
        )
        is None
    ):
        die("wave admission staging path is unsafe")
    predecessor_index = None if wave_index == 0 else wave_index - 1
    predecessor_admission_sha: str | None = None
    predecessor_submission_sha: str | None = None
    snapshot: list[dict[str, object]] = []
    if predecessor_index is not None:
        admission_path = registry / "admissions" / f"wave-{predecessor_index}.json"
        _, admission_raw = load(admission_path, canonical=True)
        predecessor_admission_sha = hashlib.sha256(admission_raw).hexdigest()
        submission_matches: list[bytes] = []
        for path in (registry / "submissions").iterdir():
            value, raw = load(path, canonical=True)
            if value.get("wave_index") == predecessor_index:
                submission_matches.append(raw)
        if len(submission_matches) != 1:
            die("ready wave lacks one unique predecessor submission")
        predecessor_submission_sha = hashlib.sha256(submission_matches[0]).hexdigest()
        predecessor_tasks = WAVE_TASKS[predecessor_index]
        if len(terminals) < len(predecessor_tasks):
            die("ready wave lacks its complete predecessor terminal set")
        predecessor_terminals = terminals[-len(predecessor_tasks) :]
        snapshot = [
            _terminal_snapshot_entry(root, seed=SEEDS[task], terminal=terminal)
            for task, terminal in zip(
                predecessor_tasks,
                predecessor_terminals,
                strict=True,
            )
        ]
    payload = {
        "schema_version": ADMISSION_SCHEMA,
        "status": "committed_before_current_wave_scheduler_submission",
        "campaign_plan_sha256": state["campaign_plan_sha256"],
        "wave_index": wave_index,
        "wave": _wave_payload(wave_index),
        "admission_rule": ADMISSION_RULE,
        "predecessor_wave_index": predecessor_index,
        "predecessor_admission_sha256": predecessor_admission_sha,
        "predecessor_submission_sha256": predecessor_submission_sha,
        "predecessor_terminal_snapshot": snapshot,
        "predecessor_terminal_snapshot_sha256": _snapshot_sha256(snapshot),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    raw = _canonical_bytes(payload)
    with output.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def main(argv: list[str]) -> None:
    state_only = False
    admission_output: Path | None = None
    if argv and argv[-1] == "--state":
        state_only = True
        argv = argv[:-1]
    elif len(argv) >= 2 and argv[-2] == "--admit":
        admission_output = Path(argv[-1])
        argv = argv[:-2]
    if len(argv) != 6:
        die(
            "usage: resolve_phase2_campaign_registry.py "
            "<design-root> <design-sha> <base-hash> <git-commit> "
            "<image-sha> <inventory-sha> [--state | --admit <staging-path>]"
        )
    root_raw, design, base, commit, image, inventory = argv
    state, terminals = _resolve(
        Path(root_raw),
        design=design,
        base=base,
        commit=commit,
        image=image,
        inventory=inventory,
    )
    if state_only:
        print(json.dumps(state, allow_nan=False, sort_keys=True, separators=(",", ":")))
        return
    if admission_output is not None:
        print(
            _materialize_admission(
                admission_output,
                root=Path(root_raw),
                state=state,
                terminals=terminals,
            )
        )
        return
    if state["status"] != "complete":
        die("formal campaign registry is not terminal across all fixed waves")
    for terminal in terminals:
        print(terminal)


if __name__ == "__main__":
    main(sys.argv[1:])
