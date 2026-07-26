#!/usr/bin/env python3
"""Pure-stdlib, fail-closed pre-sbatch inspection for the R3 Gate-C plan.

This entrypoint is intentionally import-isolated.  HPC4's canonical host
Python does not contain the project dependencies, and login-node Apptainer is
not an admitted dependency.  The constants below are the committed Gate-C
science/resource contract; the full in-container validators re-open the same
artifacts again inside every Slurm allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

FAMILIES = (
    "exact_margin_prorm_plus",
    "exact_soft_label_bt",
    "low_dimensional_prorm_plus",
)
SEEDS = (20260801, 20260802, 20260803)
CONFIG_FILE_SHA256 = "8283a742107023417143257222a0366ce1f116a585d139c3fe19b6cc5a145803"
CONFIG_SEMANTIC_SHA256 = "b59819d1d2e84b03641b190798444dcb5a0ff1dd848a7cc76387337e5132793a"
OPTIMIZER_SCHEDULE_SHA256 = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
FORMAL_UPDATE_CAP = 12760
AUDIT_INTERVAL = 20
PROFILE_UPDATES = 100
CHECKPOINT_CADENCE_UPDATES = 200
MAX_WALLTIME_SECONDS = 2 * 24 * 60 * 60
L20_PHYSICAL_GPU_MEMORY_BYTES = 46_068 * 1024**2
TORCH_VISIBLE_GPU_MEMORY_BYTES = 47_676_129_280
# Gate-C admission follows the capacity visible to its PyTorch process.
GPU_MEMORY_CAPACITY_BYTES = TORCH_VISIBLE_GPU_MEMORY_BYTES
HOST_MEMORY_BYTES = 96 * 1024**3

PROFILE_FIELDS = {
    "schema_version",
    "role",
    "optimizer_schedule_sha256",
    "git_commit",
    "container_sha256",
    "controls_config_file_sha256",
    "controls_config_semantic_sha256",
    "measurements",
    "measurement_set_sha256",
    "resource_plan",
    "profile_sha256",
}
MEASUREMENT_FIELDS = {
    "schema_version",
    "role",
    "family",
    "seed",
    "completed_updates",
    "stop_reason",
    "information_boundary",
    "result_reusable_for_training",
    "git_commit",
    "container_sha256",
    "controls_config_file_sha256",
    "controls_config_semantic_sha256",
    "input_training_sha256",
    "oracle_reward_sha256",
    "setup_wall_seconds",
    "training_wall_seconds",
    "audit_wall_seconds",
    "checkpoint_roundtrip_wall_seconds",
    "peak_gpu_memory_bytes",
    "gpu_total_memory_bytes",
    "scheduler_terminal",
    "measurement_sha256",
}
TERMINAL_FIELDS = {
    "array_job_id",
    "array_task_id",
    "job_id",
    "job_id_raw",
    "raw_sacct_sha256",
    "elapsed_seconds",
}
RESOURCE_FIELDS = {
    "schema_version",
    "role",
    "formal_update_cap",
    "profile_updates_per_family",
    "audit_interval_updates",
    "checkpoint_cadence_updates",
    "walltime_safety_margin_fraction",
    "fixed_walltime_margin_seconds",
    "memory_safety_margin_fraction",
    "cluster",
    "account",
    "partition",
    "gpu_name",
    "observed_gpu_memory_capacity_bytes",
    "gpus_per_task",
    "cpus_per_task",
    "memory_bytes",
    "array_concurrency",
    "requested_walltime_seconds_per_segment",
    "signal_lead_seconds",
    "max_scheduler_segments",
    "family_projections",
    "resource_plan_sha256",
}
PROJECTION_FIELDS = {
    "family",
    "projected_setup_wall_seconds",
    "projected_update_wall_seconds",
    "projected_audit_wall_seconds",
    "projected_checkpoint_wall_seconds",
    "projected_total_with_margin_seconds",
}
PLAN_FIELDS = {
    "schema_version",
    "role",
    "profile_sha256",
    "optimizer_schedule_sha256",
    "git_commit",
    "container_sha256",
    "controls_config_file_sha256",
    "controls_config_semantic_sha256",
    "resources",
    "arrays",
    "tasks",
    "plan_sha256",
}
PLAN_RESOURCE_FIELDS = {
    "cluster",
    "account",
    "partition",
    "gpu_name",
    "observed_gpu_memory_capacity_bytes",
    "slurm_gpu_tres",
    "gpus_per_task",
    "cpus_per_task",
    "memory_bytes",
    "nodes",
    "array_concurrency",
    "requested_walltime_seconds",
    "signal_lead_seconds",
    "checkpoint_cadence_updates",
    "max_scheduler_segments",
}
ARRAY_FIELDS = {
    "family_index",
    "family",
    "array_task_range",
    "ordered_seeds",
    "namespace",
}
TASK_FIELDS = {
    "task_id",
    "family_index",
    "seed_index",
    "array_task_id",
    "family",
    "seed",
    "namespace",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _semantic_sha256(value: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _digest(value: object, name: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _real(value: object, name: str, *, positive: bool) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} is outside its admitted range")
    return result


def _closed(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} has an invalid closed field set")
    return value


def _self_hashed(
    value: object,
    fields: set[str],
    hash_field: str,
    name: str,
) -> dict[str, Any]:
    payload = _closed(value, fields, name)
    unsigned = dict(payload)
    observed = _digest(unsigned.pop(hash_field), f"{name} self-hash")
    if _semantic_sha256(unsigned) != observed:
        raise ValueError(f"{name} self-hash is invalid")
    return payload


def _read_stable(path: Path) -> bytes:
    resolved = path.resolve(strict=True)
    before = resolved.lstat()
    if resolved.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{path} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = resolved.lstat()
    identities = {
        (before.st_dev, before.st_ino),
        (opened.st_dev, opened.st_ino),
        (after_open.st_dev, after_open.st_ino),
        (after.st_dev, after.st_ino),
    }
    if (
        len(identities) != 1
        or before.st_size != opened.st_size
        or opened.st_size != after_open.st_size
        or after_open.st_size != after.st_size
    ):
        raise ValueError(f"{path} changed while it was read")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise ValueError(f"{path} changed byte length while it was read")
    return raw


def _read_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = _digest(expected_sha256, "expected artifact file SHA-256")
    raw = _read_stable(path)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("artifact file SHA-256 differs from caller binding")
    try:
        payload = json.loads(
            raw.decode(),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact is not strict UTF-8 JSON") from error
    if type(payload) is not dict or raw != _canonical_bytes(payload):
        raise ValueError("artifact is not canonical JSON")
    return payload


def _validate_measurement(
    value: object,
    *,
    family: str,
    common: dict[str, object],
) -> dict[str, Any]:
    item = _self_hashed(
        value,
        MEASUREMENT_FIELDS,
        "measurement_sha256",
        "profile measurement",
    )
    if (
        item["schema_version"] != "phase2-recovery-r3-gate-c-profile-family-measurement/v1"
        or item["role"] != "train_only_100_update_control_family_runtime_measurement"
        or item["family"] != family
        or item["seed"] != SEEDS[0]
        or item["completed_updates"] != PROFILE_UPDATES
        or item["stop_reason"] != "predeclared_profile_update_cap"
        or item["information_boundary"] != "train_only_runtime_measurement"
        or item["result_reusable_for_training"] is not False
    ):
        raise ValueError("profile measurement exceeds the frozen profiling role")
    if COMMIT_RE.fullmatch(str(item["git_commit"])) is None:
        raise ValueError("profile measurement has an invalid Git commit")
    for name in (
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
        "input_training_sha256",
        "oracle_reward_sha256",
    ):
        _digest(item[name], name)
    for name in (
        "training_wall_seconds",
        "audit_wall_seconds",
        "checkpoint_roundtrip_wall_seconds",
    ):
        _real(item[name], name, positive=True)
    _real(item["setup_wall_seconds"], "setup wall seconds", positive=False)
    _positive_int(item["peak_gpu_memory_bytes"], "peak GPU memory")
    if item["gpu_total_memory_bytes"] != TORCH_VISIBLE_GPU_MEMORY_BYTES:
        raise ValueError("profile measurement is not from the admitted HPC4 L20")
    terminal = _closed(
        item["scheduler_terminal"],
        TERMINAL_FIELDS,
        "profile scheduler terminal",
    )
    family_index = FAMILIES.index(family)
    parent = terminal["array_job_id"]
    if (
        type(parent) is not str
        or JOB_ID_RE.fullmatch(parent) is None
        or terminal["array_task_id"] != family_index
        or terminal["job_id"] != f"{parent}_{family_index}"
        or type(terminal["job_id_raw"]) is not str
        or JOB_ID_RE.fullmatch(terminal["job_id_raw"]) is None
    ):
        raise ValueError("profile scheduler terminal mapping is invalid")
    _digest(terminal["raw_sacct_sha256"], "profile raw sacct SHA-256")
    _positive_int(terminal["elapsed_seconds"], "profile elapsed seconds")
    if common and any(item[name] != expected for name, expected in common.items()):
        raise ValueError("profile measurements have different execution identities")
    return item


def _projection(
    measurement: dict[str, Any],
    *,
    cadence: int,
    walltime_margin: float,
    fixed_margin: float,
    signal: int,
) -> dict[str, object]:
    profile_audits = math.ceil(PROFILE_UPDATES / AUDIT_INTERVAL)
    formal_audits = math.ceil(FORMAL_UPDATE_CAP / AUDIT_INTERVAL)
    formal_checkpoints = math.ceil(FORMAL_UPDATE_CAP / cadence) + 3
    setup = float(measurement["setup_wall_seconds"])
    updates = float(measurement["training_wall_seconds"]) * FORMAL_UPDATE_CAP / PROFILE_UPDATES
    audits = float(measurement["audit_wall_seconds"]) * formal_audits / profile_audits
    checkpoints = float(measurement["checkpoint_roundtrip_wall_seconds"]) * formal_checkpoints
    return {
        "family": measurement["family"],
        "projected_setup_wall_seconds": setup,
        "projected_update_wall_seconds": updates,
        "projected_audit_wall_seconds": audits,
        "projected_checkpoint_wall_seconds": checkpoints,
        "projected_total_with_margin_seconds": (
            (setup + updates + audits + checkpoints) * (1 + walltime_margin) + fixed_margin + signal
        ),
    }


def _expected_tasks() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    task_id = 0
    for family_index, family in enumerate(FAMILIES):
        for seed_index, seed in enumerate(SEEDS):
            result.append(
                {
                    "task_id": task_id,
                    "family_index": family_index,
                    "seed_index": seed_index,
                    "array_task_id": seed_index,
                    "family": family,
                    "seed": seed,
                    "namespace": (
                        f"runs/phase2-recovery-r3-controls/formal/family-{family}/seed-{seed}"
                    ),
                }
            )
            task_id += 1
    return result


def _validate_profile(value: object) -> dict[str, Any]:
    profile = _self_hashed(
        value,
        PROFILE_FIELDS,
        "profile_sha256",
        "operational profile",
    )
    if (
        profile["schema_version"] != "phase2-recovery-r3-gate-c-operational-profile/v1"
        or profile["role"] != "train_only_runtime_profile_nonreusable_for_gate_c_resource_freeze"
        or profile["optimizer_schedule_sha256"] != OPTIMIZER_SCHEDULE_SHA256
        or profile["controls_config_file_sha256"] != CONFIG_FILE_SHA256
        or profile["controls_config_semantic_sha256"] != CONFIG_SEMANTIC_SHA256
        or COMMIT_RE.fullmatch(str(profile["git_commit"])) is None
    ):
        raise ValueError("operational profile identity is invalid")
    _digest(profile["container_sha256"], "profile container SHA-256")
    measurements = profile["measurements"]
    if type(measurements) is not list or len(measurements) != 3:
        raise ValueError("profile must contain exactly three measurements")
    common = {
        name: profile[name]
        for name in (
            "git_commit",
            "container_sha256",
            "controls_config_file_sha256",
            "controls_config_semantic_sha256",
        )
    }
    validated = [
        _validate_measurement(item, family=family, common=common)
        for family, item in zip(FAMILIES, measurements, strict=True)
    ]
    if profile["measurement_set_sha256"] != _semantic_sha256({"measurements": validated}):
        raise ValueError("profile measurement-set hash is invalid")
    resource = _self_hashed(
        profile["resource_plan"],
        RESOURCE_FIELDS,
        "resource_plan_sha256",
        "resource plan",
    )
    cadence = _positive_int(
        resource["checkpoint_cadence_updates"],
        "checkpoint cadence",
    )
    walltime = _positive_int(
        resource["requested_walltime_seconds_per_segment"],
        "formal walltime",
    )
    signal = _positive_int(resource["signal_lead_seconds"], "signal lead")
    walltime_margin = _real(
        resource["walltime_safety_margin_fraction"],
        "walltime margin",
        positive=True,
    )
    memory_margin = _real(
        resource["memory_safety_margin_fraction"],
        "memory margin",
        positive=True,
    )
    fixed_margin = _real(
        resource["fixed_walltime_margin_seconds"],
        "fixed walltime margin",
        positive=True,
    )
    if (
        resource["schema_version"] != "phase2-recovery-r3-gate-c-resource-plan/v1"
        or resource["role"] != "profile_derived_gate_c_scheduler_and_checkpoint_policy"
        or resource["formal_update_cap"] != FORMAL_UPDATE_CAP
        or resource["profile_updates_per_family"] != PROFILE_UPDATES
        or resource["audit_interval_updates"] != AUDIT_INTERVAL
        or cadence != CHECKPOINT_CADENCE_UPDATES
        or cadence % AUDIT_INTERVAL
        or resource["cluster"] != "hpc4"
        or resource["account"] != "sigroup"
        or resource["partition"] != "gpu-l20"
        or resource["gpu_name"] != "NVIDIA L20"
        or resource["observed_gpu_memory_capacity_bytes"] != TORCH_VISIBLE_GPU_MEMORY_BYTES
        or resource["gpus_per_task"] != 1
        or resource["cpus_per_task"] != 8
        or resource["memory_bytes"] != HOST_MEMORY_BYTES
        or resource["array_concurrency"] != 1
        or resource["max_scheduler_segments"] != 1
        or not 0 < walltime_margin <= 1
        or not 0 < memory_margin <= 1
        or not signal < walltime <= MAX_WALLTIME_SECONDS
    ):
        raise ValueError("resource plan differs from the frozen HPC4 contract")
    expected_projections = [
        _projection(
            item,
            cadence=cadence,
            walltime_margin=walltime_margin,
            fixed_margin=fixed_margin,
            signal=signal,
        )
        for item in validated
    ]
    projections = resource["family_projections"]
    if type(projections) is not list or projections != expected_projections:
        raise ValueError("profile projections are not reproducible")
    for projection in projections:
        _closed(projection, PROJECTION_FIELDS, "family projection")
    if max(item["projected_total_with_margin_seconds"] for item in projections) > walltime:
        raise ValueError("formal walltime undercovers the measured projection")
    required_gpu = max(
        math.ceil(item["peak_gpu_memory_bytes"] * (1 + memory_margin)) for item in validated
    )
    if required_gpu > TORCH_VISIBLE_GPU_MEMORY_BYTES:
        raise ValueError("profile-derived GPU memory exceeds one NVIDIA L20")
    return profile


def _validate_plan(value: object, profile: dict[str, Any]) -> dict[str, Any]:
    plan = _self_hashed(value, PLAN_FIELDS, "plan_sha256", "execution plan")
    if (
        plan["schema_version"] != "phase2-recovery-r3-gate-c-execution-plan/v1"
        or plan["role"] != "three_independent_family_arrays_exact_three_seeds_each"
        or plan["profile_sha256"] != profile["profile_sha256"]
    ):
        raise ValueError("execution plan belongs to another profile")
    for name in (
        "optimizer_schedule_sha256",
        "git_commit",
        "container_sha256",
        "controls_config_file_sha256",
        "controls_config_semantic_sha256",
    ):
        if plan[name] != profile[name]:
            raise ValueError(f"execution plan {name} drifted from profile")
    resources = _closed(plan["resources"], PLAN_RESOURCE_FIELDS, "plan resources")
    profile_resource = profile["resource_plan"]
    expected_resources = {
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_name": "NVIDIA L20",
        "observed_gpu_memory_capacity_bytes": TORCH_VISIBLE_GPU_MEMORY_BYTES,
        "slurm_gpu_tres": "gres/gpu:l20",
        "gpus_per_task": 1,
        "cpus_per_task": 8,
        "memory_bytes": HOST_MEMORY_BYTES,
        "nodes": 1,
        "array_concurrency": 1,
        "requested_walltime_seconds": profile_resource["requested_walltime_seconds_per_segment"],
        "signal_lead_seconds": profile_resource["signal_lead_seconds"],
        "checkpoint_cadence_updates": profile_resource["checkpoint_cadence_updates"],
        "max_scheduler_segments": 1,
    }
    if resources != expected_resources:
        raise ValueError("plan scheduler resources drifted from profile")
    expected_arrays = [
        {
            "family_index": index,
            "family": family,
            "array_task_range": "0-2%1",
            "ordered_seeds": list(SEEDS),
            "namespace": (f"runs/phase2-recovery-r3-controls/formal/family-{family}"),
        }
        for index, family in enumerate(FAMILIES)
    ]
    arrays = plan["arrays"]
    if type(arrays) is not list:
        raise ValueError("plan arrays must be a list")
    for item in arrays:
        _closed(item, ARRAY_FIELDS, "family array")
    if arrays != expected_arrays:
        raise ValueError("plan arrays differ from exact rolling design")
    tasks = plan["tasks"]
    if type(tasks) is not list:
        raise ValueError("plan tasks must be a list")
    for item in tasks:
        _closed(item, TASK_FIELDS, "formal task")
    if tasks != _expected_tasks():
        raise ValueError("plan is not the exact 3x3 task matrix")
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-file-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    controls_raw = _read_stable(arguments.controls_config)
    if hashlib.sha256(controls_raw).hexdigest() != CONFIG_FILE_SHA256:
        raise ValueError("controls config bytes differ from the frozen science contract")
    profile = _validate_profile(_read_artifact(arguments.profile, arguments.profile_file_sha256))
    plan = _validate_plan(
        _read_artifact(arguments.plan, arguments.plan_file_sha256),
        profile,
    )
    print(
        json.dumps(
            {
                "status": "plan_validated_by_pure_stdlib_preflight",
                "plan_sha256": plan["plan_sha256"],
                "git_commit": plan["git_commit"],
                "container_sha256": plan["container_sha256"],
                "controls_config_file_sha256": plan["controls_config_file_sha256"],
                "controls_config_semantic_sha256": plan["controls_config_semantic_sha256"],
                "resources": plan["resources"],
                "arrays": plan["arrays"],
            },
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
