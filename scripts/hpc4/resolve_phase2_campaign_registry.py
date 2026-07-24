#!/usr/bin/env python3
"""Resolve exact-30 terminal heads from the immutable formal campaign registry."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

from validate_phase2_terminal import validate_terminal

SEEDS = tuple(range(20260901, 20260931))


def die(message: str) -> None:
    raise SystemExit(message)


def load(path: Path) -> tuple[dict[str, Any], str]:
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
    return value, hashlib.sha256(raw).hexdigest()


def main(argv: list[str]) -> None:
    if len(argv) != 6:
        die(
            "usage: resolve_phase2_campaign_registry.py "
            "<design-root> <design-sha> <base-hash> <git-commit> "
            "<image-sha> <inventory-sha>"
        )
    root_raw, design, base, commit, image, inventory = argv
    root = Path(root_raw)
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.name != design
        or re.fullmatch(r"[0-9a-f]{64}", design) is None
    ):
        die("formal design root is unsafe or identity-mismatched")
    registry = root / "campaign-registry"
    submissions = registry / "submissions"
    executions = registry / "executions"
    recoveries = registry / "recoveries"
    scheduler_terminals = registry / "scheduler-terminals"
    staging = registry / ".staging"
    for directory in (
        registry,
        submissions,
        executions,
        recoveries,
        scheduler_terminals,
        staging,
    ):
        if directory.is_symlink() or not directory.is_dir():
            die(f"formal campaign registry directory is missing or unsafe: {directory}")
    allowed_registry_entries = {
        ".staging",
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
        r"(?:submission-array-[1-9][0-9]*|"
        r"execution-seed-[1-9][0-9]*-attempt-1)\.[A-Za-z0-9]+"
    )
    for path in staging.iterdir():
        if staging_pattern.fullmatch(path.name) is None or path.is_symlink() or not path.is_file():
            die(f"unsafe non-authoritative registry staging residue: {path}")

    attempts_by_seed: dict[int, dict[int, dict[str, Any]]] = {seed: {} for seed in SEEDS}
    initial_submissions: list[tuple[Path, dict[str, Any]]] = []
    expected_submission_fields = {
        "schema_version",
        "status",
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
    expected_job_fields = {
        "account",
        "partition",
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
    job_file_sha = hashlib.sha256(
        Path(__file__).with_name("phase2_confirmatory.sbatch").read_bytes()
    ).hexdigest()
    submission_paths = sorted(submissions.iterdir())
    if not submission_paths:
        die("formal campaign registry has no committed submissions")
    for path in submission_paths:
        if not re.fullmatch(r"array-[1-9][0-9]*\.json", path.name):
            die(f"unexpected submission-registry entry: {path}")
        value, sha = load(path)
        entries = value.get("entries")
        job_tuple = value.get("job_tuple")
        producer = value.get("producer")
        if (
            set(value) != expected_submission_fields
            or value.get("schema_version") != "prorm-phase2-campaign-submission/v1"
            or value.get("status") != "committed_while_slurm_held"
            or value.get("phase2_design_sha256") != design
            or value.get("base_config_hash") != base
            or value.get("git_commit") != commit
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(value.get("accepted_freeze_aggregate_sha256", "")),
            )
            is None
            or value.get("replacement_seed_allowed") is not False
            or value.get("array_job_id") != path.stem.removeprefix("array-")
            or value.get("submitted_cluster") != "hpc4"
            or not isinstance(value.get("attempt_index"), int)
            or isinstance(value.get("attempt_index"), bool)
            or value["attempt_index"] != 1
            or value.get("array_spec") not in {"0-29%1", "0-29%2"}
            or re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                str(value.get("created_at_utc", "")),
            )
            is None
            or not isinstance(job_tuple, dict)
            or set(job_tuple) != expected_job_fields
            or job_tuple.get("account") != "sigroup"
            or job_tuple.get("partition") != "gpu-l20"
            or job_tuple.get("nodes") != 1
            or job_tuple.get("tasks") != 1
            or job_tuple.get("cpus_per_task") != 8
            or job_tuple.get("memory") != "64G"
            or job_tuple.get("gpus_per_node") != 1
            or re.fullmatch(
                r"(?:[1-9][0-9]*-)?[0-9]{2}:[0-9]{2}:[0-9]{2}",
                str(job_tuple.get("walltime", "")),
            )
            is None
            or job_tuple.get("no_requeue") is not True
            or job_tuple.get("held_before_registry_commit") is not True
            or job_tuple.get("script") != "scripts/hpc4/phase2_confirmatory.sbatch"
            or job_tuple.get("script_file_sha256") != job_file_sha
            or not isinstance(producer, dict)
            or set(producer) != expected_producer_fields
            or producer.get("image_sha256") != image
            or producer.get("hf_inventory_sha256") != inventory
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(producer.get(field, ""))) is None
                for field in (
                    "overlay_file_sha256",
                    "base_file_sha256",
                    "identities_file_sha256",
                )
            )
            or not isinstance(entries, list)
            or not entries
        ):
            die(f"submission registry identity is invalid: {path}")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "seed",
                "attempt_index",
                "array_job_id",
                "array_task_id",
            }:
                die(f"submission registry entry fields are invalid: {path}")
            seed = entry["seed"]
            attempt = entry["attempt_index"]
            if (
                seed not in attempts_by_seed
                or not isinstance(attempt, int)
                or isinstance(attempt, bool)
                or attempt < 1
                or attempt != value["attempt_index"]
                or entry.get("array_job_id") != value["array_job_id"]
                or not isinstance(entry.get("array_task_id"), int)
                or isinstance(entry.get("array_task_id"), bool)
                or entry["array_task_id"] != seed - 20260901
                or attempt in attempts_by_seed[seed]
            ):
                die("campaign registry has an unexpected or duplicate formal attempt")
            attempts_by_seed[seed][attempt] = {
                **entry,
                "submission_path": path,
                "submission_sha256": sha,
            }
        if value["attempt_index"] == 1:
            initial_submissions.append((path, value))
    expected_initial_entries = [
        {
            "seed": seed,
            "attempt_index": 1,
            "array_job_id": initial_submissions[0][1]["array_job_id"]
            if len(initial_submissions) == 1
            else "",
            "array_task_id": seed - 20260901,
        }
        for seed in SEEDS
    ]
    if (
        len(initial_submissions) != 1
        or re.fullmatch(
            r"0-29%[12]",
            str(initial_submissions[0][1].get("array_spec", "")),
        )
        is None
        or initial_submissions[0][1].get("entries") != expected_initial_entries
    ):
        die("attempt-1 must be one ordered exact-30 held-array submission")
    if any(set(attempts) != {1} for attempts in attempts_by_seed.values()):
        die("formal Phase-2 retries are disabled; registry must contain only attempt-1")

    observed_seed_roots = {path.name for path in root.iterdir() if path.name.startswith("seed-")}
    expected_seed_roots = {f"seed-{seed}" for seed in SEEDS}
    if observed_seed_roots != expected_seed_roots:
        die("physical seed roots are missing or contain an unregistered extra seed")

    def registry_keys(directory: Path, *, label: str) -> set[tuple[int, int]]:
        keys: set[tuple[int, int]] = set()
        pattern = re.compile(r"seed-([1-9][0-9]*)-attempt-([1-9][0-9]*)\.json")
        for path in sorted(directory.iterdir()):
            match = pattern.fullmatch(path.name)
            if match is None:
                die(f"unexpected {label}-registry entry: {path}")
            key = (int(match.group(1)), int(match.group(2)))
            if key in keys:
                die(f"duplicate {label}-registry identity: {key}")
            if key[0] not in attempts_by_seed or key[1] not in attempts_by_seed[key[0]]:
                die(f"{label}-registry record has no committed submission: {key}")
            load(path)
            keys.add(key)
        return keys

    registry_keys(executions, label="execution")
    recovery_keys = registry_keys(recoveries, label="recovery")
    scheduler_terminal_keys = registry_keys(
        scheduler_terminals,
        label="scheduler-terminal",
    )
    if recovery_keys:
        die("formal no-retry campaign must not contain recovery authorization")

    terminals: list[Path] = []
    for seed in SEEDS:
        registered = attempts_by_seed[seed]
        if set(registered) != {1}:
            die(f"campaign registry must contain exactly attempt-1 for seed {seed}")
        head = 1
        seed_scheduler_keys = {key for key in scheduler_terminal_keys if key[0] == seed}
        if seed_scheduler_keys not in (set(), {(seed, head)}):
            die(f"scheduler-terminal registry is not confined to seed-{seed}'s head")
        seed_root = root / f"seed-{seed}"
        if (
            seed_root.is_symlink()
            or not seed_root.is_dir()
            or seed_root.resolve() != seed_root.absolute()
        ):
            die(f"formal seed root is unsafe for seed {seed}")
        attempt_names = {
            path.name for path in seed_root.iterdir() if path.name.startswith("attempt-")
        }
        if attempt_names != {f"attempt-{index}" for index in range(1, head + 1)}:
            die(f"physical attempts disagree with registry head for seed {seed}")
        head_root = seed_root / f"attempt-{head}"
        if (
            head_root.is_symlink()
            or not head_root.is_dir()
            or head_root.resolve().parent != seed_root.resolve()
        ):
            die(f"registered head attempt is unsafe for seed {seed}")
        jobs = [path for path in head_root.iterdir() if path.name.startswith("job-")]
        if len(jobs) != 1 or jobs[0].is_symlink() or not jobs[0].is_dir():
            die(f"registered head lacks one unique claimed job for seed {seed}")
        terminal_candidates = [
            path
            for path in jobs[0].iterdir()
            if path.name
            in {
                "phase2-success-terminal.json",
                "phase2-failure-terminal.json",
            }
        ]
        if len(terminal_candidates) != 1:
            die(f"registered head is not terminalized for seed {seed}")
        scheduler_marker = jobs[0] / "SCHEDULER_FAILED"
        scheduler_captured = scheduler_marker.exists() or scheduler_marker.is_symlink()
        if scheduler_captured != ((seed, head) in scheduler_terminal_keys):
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
        terminals.append(terminal.resolve())

    if len(set(terminals)) != len(SEEDS):
        die("registry resolver produced duplicate terminal paths")
    for terminal in terminals:
        print(terminal)


if __name__ == "__main__":
    main(sys.argv[1:])
