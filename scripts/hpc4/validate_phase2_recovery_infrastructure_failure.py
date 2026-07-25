#!/usr/bin/env python3
"""Verify the exact pre-trainer failure that authorizes recovery execution 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from collections.abc import Mapping
from pathlib import Path

DESIGN = "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"
BASE = "81ccbd3bc9d745d3792d1834116ba2480d34a7201d6f4e463a61d8d8ab0baefa"
COMMIT = "734d2a27473f974431b96d5d196f9793e14b2755"
ARRAY_JOB_ID = "1648094"
FORBIDDEN_TRAINING_FILES = frozenset(
    {
        "SUCCESS",
        "run-manifest.json",
        "env-report.log",
        "gpu-check.json",
        "recovery-train.log",
        "recovery-result.json",
        "recovery-failure-evidence.json",
        "recovery-output-verification.json",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _require_real_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"expected a regular non-symlink file: {path}")
    if path.resolve(strict=True) != path:
        raise ValueError(f"file path must be canonical with no symlink component: {path}")


def _require_real_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"expected a real non-symlink directory: {path}")
    if path.resolve(strict=True) != path:
        raise ValueError(f"directory path must be canonical with no symlink component: {path}")


def _parse_marker(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"invalid recovery status marker: {path}")
        values[key] = value
    return values


def validate(
    registry_path: Path,
    *,
    project_root: Path,
    expected_registry_sha256: str,
) -> dict[str, object]:
    _require_real_file(registry_path)
    observed_registry_sha256 = _sha256(registry_path)
    if observed_registry_sha256 != expected_registry_sha256:
        raise ValueError("infrastructure-failure registry SHA256 mismatch")
    _require_real_directory(project_root)
    if project_root.resolve(strict=True) != project_root:
        raise ValueError("project root must be canonical")

    registry = _strict_json(registry_path)
    expected_root = {
        "schema_version": "prorm-phase2-recovery-infrastructure-failure/v1",
        "execution_revision": 1,
        "next_execution_revision": 2,
        "next_execution_reason": "pretrainer_hf_datasets_runtime_lock",
        "recovery_design_sha256": DESIGN,
        "recovery_git_commit": COMMIT,
        "source_array_job_id": ARRAY_JOB_ID,
        "trainer_entered": False,
    }
    for key, expected in expected_root.items():
        if registry.get(key) != expected:
            raise ValueError(f"invalid infrastructure-failure registry field: {key}")
    tasks = registry.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("infrastructure-failure registry must contain exactly three tasks")

    evidence_root = project_root / "runs" / "phase2-recovery-pilot" / DESIGN
    log_root = project_root / "slurm-logs" / "phase2-recovery-pilot" / DESIGN
    _require_real_directory(evidence_root)
    _require_real_directory(log_root)
    summaries: list[dict[str, object]] = []
    for index, raw_task in enumerate(tasks):
        if not isinstance(raw_task, Mapping):
            raise TypeError("infrastructure-failure task must be an object")
        seed = 20260801 + index
        expected_class = (
            "hf_datasets_read_only_runtime_lock"
            if index < 2
            else "cancelled_before_training_after_sibling_preflight_failure"
        )
        expected_task_fields = {
            "seed": seed,
            "array_task_id": index,
            "failure_class": expected_class,
            "workload_exit_code": 1 if index < 2 else 0,
            "final_exit_code": 1,
        }
        for key, expected in expected_task_fields.items():
            if raw_task.get(key) != expected:
                raise ValueError(f"invalid task {index} infrastructure evidence field: {key}")

        run_dir = evidence_root / f"seed-{seed}" / f"job-{ARRAY_JOB_ID}_{index}"
        _require_real_directory(run_dir.parent)
        _require_real_directory(run_dir)
        raw_files = raw_task.get("files")
        if not isinstance(raw_files, Mapping) or not raw_files:
            raise ValueError(f"task {index} has no frozen file inventory")
        expected_names = set(raw_files)
        observed_names = {entry.name for entry in run_dir.iterdir()}
        if observed_names != expected_names:
            raise ValueError(f"task {index} recovery evidence file set changed")
        if observed_names & FORBIDDEN_TRAINING_FILES:
            raise ValueError(f"task {index} contains forbidden trainer evidence")
        for name, expected_hash in raw_files.items():
            if not isinstance(name, str) or not isinstance(expected_hash, str):
                raise TypeError(f"task {index} file inventory is invalid")
            path = run_dir / name
            _require_real_file(path)
            if _sha256(path) != expected_hash:
                raise ValueError(f"task {index} recovery evidence bytes changed: {name}")

        marker = _parse_marker(run_dir / "FAILED")
        expected_marker = {
            "schema_version": "prorm-phase2-recovery-run-status/v1",
            "status": "FAILED",
            "workload_exit_code": str(expected_task_fields["workload_exit_code"]),
            "final_exit_code": "1",
            "array_job_id": ARRAY_JOB_ID,
            "array_task_id": str(index),
            "seed": str(seed),
            "recovery_design_sha256": DESIGN,
            "base_config_hash": BASE,
            "recovery_git_commit": COMMIT,
            "one_shot_no_further_adaptation": "true",
        }
        for key, expected in expected_marker.items():
            if marker.get(key) != expected:
                raise ValueError(f"task {index} FAILED marker field changed: {key}")

        stdout = log_root / f"prorm-p2-recovery-{ARRAY_JOB_ID}_{index}.out"
        stderr = log_root / f"prorm-p2-recovery-{ARRAY_JOB_ID}_{index}.err"
        for stream_name, path in (("stdout", stdout), ("stderr", stderr)):
            _require_real_file(path)
            expected_hash = raw_task.get(f"{stream_name}_sha256")
            if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
                raise ValueError(f"task {index} {stream_name} log bytes changed")
        stderr_text = stderr.read_text(encoding="utf-8")
        if index < 2:
            if (
                "OSError: [Errno 30] Read-only file system:" not in stderr_text
                or "/hf-cache/datasets/" not in stderr_text
                or ".lock" not in stderr_text
            ):
                raise ValueError(f"task {index} does not prove the Datasets lock failure")
        elif "CANCELLED" not in stderr_text:
            raise ValueError("task 2 does not prove pre-training cancellation")

        summaries.append(
            {
                "array_task_id": index,
                "failure_class": expected_class,
                "seed": seed,
                "trainer_entered": False,
            }
        )

    return {
        "execution_revision": 1,
        "next_execution_reason": "pretrainer_hf_datasets_runtime_lock",
        "next_execution_revision": 2,
        "recovery_design_sha256": DESIGN,
        "registry_sha256": observed_registry_sha256,
        "source_array_job_id": ARRAY_JOB_ID,
        "status": "ok",
        "tasks": summaries,
        "trainer_entered": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the frozen pre-trainer failure authorizing recovery execution 2."
    )
    parser.add_argument("registry")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--output")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    result = validate(
        Path(arguments.registry).resolve(strict=True),
        project_root=Path(arguments.project_root),
        expected_registry_sha256=arguments.expected_registry_sha256,
    )
    payload = json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    if arguments.output:
        destination = Path(arguments.output)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
