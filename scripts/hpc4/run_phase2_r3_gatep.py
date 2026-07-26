#!/usr/bin/env python3
"""Run the fixed R3 Gate-P workload inside one admitted HPC4 GPU allocation.

This program is deliberately train-only.  It verifies Gate 0 and Gate 1,
materializes only the frozen training split for seed 20260801, executes exactly
100 updates for each primary head, projects the primary resource plan, and
publishes the operational bundle plus the in-job Slurm receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from smart_reward.phase2_r3_config import load_r3_science_config
from smart_reward.phase2_r3_gate0 import verify_live_r3_gate0_in_container
from smart_reward.phase2_r3_gate1 import verify_live_r3_gate1_in_container
from smart_reward.phase2_r3_gatep_failure import reopen_gate_p_attempt_lineage
from smart_reward.phase2_r3_identity import (
    create_gate_p_admission,
    create_validated_gate_p_run,
)
from smart_reward.phase2_r3_inputs import materialize_r3_train_only_from_parent
from smart_reward.phase2_r3_profile import (
    build_gate_p_resource_plan,
    freeze_profile_safety_margin_policy,
    record_profile_preparation_from_train_input,
    run_formal_gate_p_cuda_profile,
    validate_scheduler_resource_envelope,
)
from smart_reward.phase2_r3_profile_artifacts import (
    publish_verified_gate_p_operational_bundle,
)
from smart_reward.phase2_r3_terminal import (
    capture_profile_slurm_runtime_receipt,
    reopen_profile_allocation_intent,
)

_PROFILE_SEED = 20260801
_DIGEST_LENGTH = 64
PRODUCTION_REPO_ROOT = Path("/home/yyangjo/Smart-Reward-Model")
PRODUCTION_PROJECT_ROOT = Path("/project/sigroup/smart-reward-model")


def _require_production_layout() -> tuple[Path, Path]:
    """Return the two fixed HPC4 roots after rejecting aliases and role swaps."""

    for root, name in (
        (PRODUCTION_REPO_ROOT, "production Git repository root"),
        (PRODUCTION_PROJECT_ROOT, "persistent project root"),
    ):
        if root.is_symlink():
            raise RuntimeError(f"{name} must not be a symbolic link")
        try:
            resolved = root.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(f"{name} does not exist") from error
        if resolved != root or not resolved.is_dir():
            raise RuntimeError(f"{name} must be its fixed canonical directory")

    repo = PRODUCTION_REPO_ROOT
    project = PRODUCTION_PROJECT_ROOT
    if repo == project or repo in project.parents or project in repo.parents:
        raise RuntimeError("production repository and project roots must not overlap")
    git_metadata = repo / ".git"
    if git_metadata.is_symlink() or not git_metadata.exists():
        raise RuntimeError("production repository root is not a Git checkout")
    project_git_metadata = project / ".git"
    if project_git_metadata.exists() or project_git_metadata.is_symlink():
        raise RuntimeError("persistent project root must not be a Git checkout")
    expected_runner = repo / "scripts" / "hpc4" / "run_phase2_r3_gatep.py"
    if Path(__file__).resolve(strict=True) != expected_runner:
        raise RuntimeError("Gate-P runner did not originate from the fixed Git checkout")
    return repo, project


def _existing_file_in(
    value: str | os.PathLike[str],
    *,
    root: Path,
    name: str,
    required_mode: int | None = None,
) -> Path:
    candidate = Path(value)
    if candidate.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must be inside {root}") from error
    if candidate != resolved:
        raise ValueError(f"{name} must use its canonical absolute path")
    if required_mode is not None and resolved.stat().st_mode & 0o777 != required_mode:
        raise ValueError(f"{name} must have mode {required_mode:04o}")
    return resolved


def _future_file_in(
    value: str | os.PathLike[str],
    *,
    root: Path,
    name: str,
) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ValueError(f"{name} must be a canonical, non-symlink absolute path")
    parent = candidate.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must be inside {root}") from error
    canonical = parent / candidate.name
    if candidate != canonical:
        raise ValueError(f"{name} must use its canonical absolute path")
    if canonical.exists() and not canonical.is_file():
        raise ValueError(f"{name} exists but is not a regular file")
    return canonical


def _digest(value: str, *, name: str) -> str:
    if (
        len(value) != _DIGEST_LENGTH
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _stable_file_sha256(
    value: str | os.PathLike[str],
    *,
    expected_sha256: str,
    name: str,
) -> str:
    path = Path(value)
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    hasher = hashlib.sha256()
    with resolved.open("rb", buffering=0) as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"{name} changed while it was being hashed")
    observed = hasher.hexdigest()
    if observed != _digest(expected_sha256, name=f"expected {name} SHA-256"):
        raise ValueError(f"{name} SHA-256 mismatch")
    return observed


def _validate_lineage_cross_binding(
    *,
    intent_lineage_file_sha256: str,
    intent_lineage_sha256: str,
    actual_lineage_file_sha256: str,
    actual_lineage_sha256: str,
    exported_lineage_file_sha256: str,
    exported_lineage_sha256: str,
) -> None:
    """Require the intent, reopened artifact, and scheduler export to agree."""

    bindings = {
        "profile intent lineage file SHA-256": intent_lineage_file_sha256,
        "profile intent lineage semantic SHA-256": intent_lineage_sha256,
        "actual lineage file SHA-256": actual_lineage_file_sha256,
        "actual lineage semantic SHA-256": actual_lineage_sha256,
        "exported lineage file SHA-256": exported_lineage_file_sha256,
        "exported lineage semantic SHA-256": exported_lineage_sha256,
    }
    checked = {name: _digest(value, name=name) for name, value in bindings.items()}
    if (
        len(
            {
                checked["profile intent lineage file SHA-256"],
                checked["actual lineage file SHA-256"],
                checked["exported lineage file SHA-256"],
            }
        )
        != 1
    ):
        raise ValueError("profile intent lineage file SHA-256 differs from the actual lineage")
    if (
        len(
            {
                checked["profile intent lineage semantic SHA-256"],
                checked["actual lineage semantic SHA-256"],
                checked["exported lineage semantic SHA-256"],
            }
        )
        != 1
    ):
        raise ValueError("profile intent lineage semantic SHA-256 differs from the actual lineage")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science-config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--parent-registry", type=Path, required=True)
    parser.add_argument("--parent-registry-file-sha256", required=True)
    parser.add_argument("--gate0-file-sha256", required=True)
    parser.add_argument("--gate1-file-sha256", required=True)
    parser.add_argument("--source-test-receipt-file-sha256", required=True)

    parser.add_argument("--scheduler-raw-evidence", type=Path, required=True)
    parser.add_argument("--scheduler-raw-evidence-sha256", required=True)
    parser.add_argument("--resource-raw-evidence", type=Path, required=True)
    parser.add_argument("--resource-raw-evidence-sha256", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--gpu-total-memory-bytes", type=_positive_int, required=True)
    parser.add_argument("--max-allocation-wall-seconds", type=_positive_int, required=True)
    parser.add_argument("--max-array-concurrency", type=_positive_int, required=True)
    parser.add_argument("--max-scheduler-segments", type=_positive_int, required=True)
    parser.add_argument("--max-gpus-per-task", type=_positive_int, required=True)
    parser.add_argument("--max-cpus-per-task", type=_positive_int, required=True)
    parser.add_argument("--max-memory-bytes", type=_positive_int, required=True)

    parser.add_argument("--walltime-margin-fraction", type=_positive_float, required=True)
    parser.add_argument(
        "--fixed-walltime-margin-seconds",
        type=_positive_float,
        required=True,
    )
    parser.add_argument("--memory-margin-fraction", type=_positive_float, required=True)
    parser.add_argument("--signal-margin-seconds", type=_positive_float, required=True)
    parser.add_argument(
        "--durable-checkpoint-cadence-updates",
        type=_positive_int,
        required=True,
    )

    parser.add_argument(
        "--requested-walltime-seconds-per-segment",
        type=_positive_int,
        required=True,
    )
    parser.add_argument("--array-concurrency", type=_positive_int, required=True)
    parser.add_argument("--cpus-per-task", type=_positive_int, required=True)
    parser.add_argument("--memory-bytes", type=_positive_int, required=True)

    parser.add_argument("--io-probe-directory", type=Path, required=True)
    parser.add_argument("--attempt-lineage", type=Path, required=True)
    parser.add_argument("--attempt-lineage-file-sha256", required=True)
    parser.add_argument("--attempt-lineage-sha256", required=True)
    parser.add_argument("--operational-bundle", type=Path, required=True)
    parser.add_argument("--allocation-intent", type=Path, required=True)
    parser.add_argument("--allocation-intent-file-sha256", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repo_root, project_root = _require_production_layout()
    commit = os.environ.get("PRORM_R3_GIT_COMMIT", "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("Gate-P requires the exact clean Git commit identity")
    input_root = project_root / "runs" / "phase2-recovery-r3" / "inputs" / commit

    science_config = _existing_file_in(
        arguments.science_config,
        root=repo_root,
        name="science config",
    )
    if science_config != repo_root / "configs" / "phase2_recovery_r3_science.yaml":
        raise ValueError("science config is not the frozen R3 production config")
    source_config = _existing_file_in(
        arguments.source_config,
        root=project_root,
        name="source config",
        required_mode=0o440,
    )
    if source_config != input_root / "common_beta_pilot_base.yaml":
        raise ValueError("source config is not the retained clean-commit copy")
    parent_registry = _existing_file_in(
        arguments.parent_registry,
        root=project_root,
        name="parent registry",
        required_mode=0o440,
    )
    if parent_registry != input_root / "phase2_recovery_parent_failures.json":
        raise ValueError("parent registry is not the retained clean-commit copy")
    scheduler_raw_evidence = _existing_file_in(
        arguments.scheduler_raw_evidence,
        root=project_root,
        name="scheduler raw evidence",
    )
    resource_raw_evidence = _existing_file_in(
        arguments.resource_raw_evidence,
        root=project_root,
        name="resource raw evidence",
    )
    allocation_intent = _existing_file_in(
        arguments.allocation_intent,
        root=project_root,
        name="allocation intent",
    )
    attempt_lineage = _existing_file_in(
        arguments.attempt_lineage,
        root=project_root,
        name="Gate-P attempt lineage",
    )
    if attempt_lineage.parent != allocation_intent.parent:
        raise ValueError("Gate-P attempt lineage and profile allocation intent differ in attempt")
    intent = reopen_profile_allocation_intent(
        allocation_intent,
        expected_file_sha256=_digest(
            arguments.allocation_intent_file_sha256,
            name="allocation-intent file SHA-256",
        ),
    )
    lineage = reopen_gate_p_attempt_lineage(
        attempt_lineage,
        project_root=project_root,
        expected_file_sha256=_digest(
            arguments.attempt_lineage_file_sha256,
            name="Gate-P attempt lineage file SHA-256",
        ),
    )
    _validate_lineage_cross_binding(
        intent_lineage_file_sha256=intent.attempt_lineage_file_sha256,
        intent_lineage_sha256=intent.attempt_lineage_sha256,
        actual_lineage_file_sha256=lineage.file_sha256,
        actual_lineage_sha256=lineage.lineage_sha256,
        exported_lineage_file_sha256=arguments.attempt_lineage_file_sha256,
        exported_lineage_sha256=arguments.attempt_lineage_sha256,
    )
    operational_bundle = _future_file_in(
        arguments.operational_bundle,
        root=project_root,
        name="operational bundle",
    )
    runtime_receipt_path = _future_file_in(
        arguments.runtime_receipt,
        root=project_root,
        name="runtime receipt",
    )

    scheduler_sha256 = _stable_file_sha256(
        scheduler_raw_evidence,
        expected_sha256=arguments.scheduler_raw_evidence_sha256,
        name="scheduler raw evidence",
    )
    resource_sha256 = _stable_file_sha256(
        resource_raw_evidence,
        expected_sha256=arguments.resource_raw_evidence_sha256,
        name="resource raw evidence",
    )

    science = load_r3_science_config(science_config)
    gate0 = verify_live_r3_gate0_in_container(
        expected_file_sha256=_digest(
            arguments.gate0_file_sha256,
            name="Gate-0 file SHA-256",
        )
    )
    gate1 = verify_live_r3_gate1_in_container(
        expected_file_sha256=_digest(
            arguments.gate1_file_sha256,
            name="Gate-1 file SHA-256",
        ),
        expected_source_test_receipt_file_sha256=_digest(
            arguments.source_test_receipt_file_sha256,
            name="source-test receipt file SHA-256",
        ),
    )
    admission = create_gate_p_admission(
        gate0_capability=gate0,
        gate1_capabilities=gate1,
        science=science,
    )
    materialized = materialize_r3_train_only_from_parent(
        project_root=project_root,
        parent_registry_path=parent_registry,
        expected_parent_registry_file_sha256=_digest(
            arguments.parent_registry_file_sha256,
            name="parent registry file SHA-256",
        ),
        source_config_path=source_config,
        science_bundle=science,
        seed=_PROFILE_SEED,
        device="cuda",
    )
    profile_run = create_validated_gate_p_run(
        materialization_capability=materialized.capability,
        science=science,
        admission=admission,
    )
    safety = freeze_profile_safety_margin_policy(
        profile_run,
        walltime_margin_fraction=arguments.walltime_margin_fraction,
        fixed_walltime_margin_seconds=arguments.fixed_walltime_margin_seconds,
        memory_margin_fraction=arguments.memory_margin_fraction,
        signal_margin_seconds=arguments.signal_margin_seconds,
        durable_checkpoint_cadence_updates=(arguments.durable_checkpoint_cadence_updates),
    )
    envelope = validate_scheduler_resource_envelope(
        profile_run,
        scheduler_raw_evidence_sha256=scheduler_sha256,
        resource_raw_evidence_sha256=resource_sha256,
        partition=arguments.partition,
        gpu_name=arguments.gpu_name,
        gpu_total_memory_bytes=arguments.gpu_total_memory_bytes,
        max_allocation_wall_seconds=arguments.max_allocation_wall_seconds,
        max_array_concurrency=arguments.max_array_concurrency,
        max_scheduler_segments=arguments.max_scheduler_segments,
        max_gpus_per_task=arguments.max_gpus_per_task,
        max_cpus_per_task=arguments.max_cpus_per_task,
        max_memory_bytes=arguments.max_memory_bytes,
    )
    preparation = record_profile_preparation_from_train_input(
        profile_run,
        materialized.timings,
    )
    formal_result = run_formal_gate_p_cuda_profile(
        profile_run,
        safety_policy=safety,
        envelope=envelope,
        preparation=preparation,
        io_probe_directory=arguments.io_probe_directory,
    )
    plan = build_gate_p_resource_plan(
        formal_result,
        safety_policy=safety,
        envelope=envelope,
        requested_walltime_seconds_per_segment=(arguments.requested_walltime_seconds_per_segment),
        array_concurrency=arguments.array_concurrency,
        cpus_per_task=arguments.cpus_per_task,
        memory_bytes=arguments.memory_bytes,
    )
    bundle = publish_verified_gate_p_operational_bundle(
        operational_bundle,
        profile_run=profile_run,
        safety_policy=safety,
        envelope=envelope,
        formal_result=formal_result,
        resource_plan=plan,
    )
    runtime_receipt = capture_profile_slurm_runtime_receipt(
        bundle,
        intent,
        runtime_receipt_path,
    )
    print(
        json.dumps(
            {
                "status": "r3_gate_p_compute_completed_pending_scheduler_terminal",
                "profile_seed": _PROFILE_SEED,
                "operational_bundle_file_sha256": bundle.file_sha256,
                "operational_bundle_semantic_sha256": bundle.bundle_semantic_sha256,
                "profile_run_sha256": bundle.profile_run_sha256,
                "formal_profile_sha256": bundle.formal_profile_sha256,
                "resource_plan_sha256": bundle.resource_plan_sha256,
                "runtime_receipt_file_sha256": runtime_receipt.file_sha256,
                "slurm_job_id": runtime_receipt.job_id,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
