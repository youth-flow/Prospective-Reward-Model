#!/usr/bin/env python3
"""Run one independent formal R3 Gate-C family/seed task on HPC4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from smart_reward.phase2_r3_artifacts import (
    publish_canonical_artifact,
    read_canonical_artifact,
)
from smart_reward.phase2_r3_config import load_r3_science_config
from smart_reward.phase2_r3_control_training import run_r3_control_family
from smart_reward.phase2_r3_controls import (
    R3_GATE_C_FAMILIES,
    R3_GATE_C_SEEDS,
    load_r3_controls_config,
)
from smart_reward.phase2_r3_controls_hpc4 import (
    validate_controls_execution_plan,
    validate_controls_operational_profile,
)
from smart_reward.phase2_r3_inputs import (
    materialize_r3_control_train_only_from_parent,
)


def _digest(value: str, *, name: str) -> str:
    if (
        len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--science-config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--parent-registry", type=Path, required=True)
    parser.add_argument("--parent-registry-file-sha256", required=True)
    parser.add_argument("--controls-config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-file-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--task-id", type=_nonnegative_int, required=True)
    parser.add_argument("--family", choices=R3_GATE_C_FAMILIES, required=True)
    parser.add_argument("--seed", type=int, choices=R3_GATE_C_SEEDS, required=True)
    parser.add_argument(
        "--checkpoint-cadence-updates",
        type=_positive_int,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


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


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    array_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_text not in {"0", "1", "2"}:
        raise RuntimeError("formal Gate-C execution requires SLURM_ARRAY_TASK_ID in {0,1,2}")
    if R3_GATE_C_SEEDS[int(array_text)] != arguments.seed:
        raise ValueError("Gate-C CLI seed differs from the Slurm array mapping")

    controls = load_r3_controls_config(arguments.controls_config)
    profile = read_canonical_artifact(
        arguments.profile.resolve(strict=True),
        expected_file_sha256=_digest(
            arguments.profile_file_sha256,
            name="profile file SHA-256",
        ),
    ).payload
    profile = validate_controls_operational_profile(
        profile,
        controls_config=controls,
    )
    plan = read_canonical_artifact(
        arguments.plan.resolve(strict=True),
        expected_file_sha256=_digest(
            arguments.plan_file_sha256,
            name="plan file SHA-256",
        ),
    ).payload
    plan = validate_controls_execution_plan(
        plan,
        profile=profile,
        controls_config=controls,
    )
    tasks = plan["tasks"]
    if (
        not isinstance(tasks, list)
        or arguments.task_id >= len(tasks)
        or not isinstance(tasks[arguments.task_id], dict)
    ):
        raise ValueError("Gate-C task ID is outside the exact formal matrix")
    task = tasks[arguments.task_id]
    if (
        task.get("task_id") != arguments.task_id
        or task.get("array_task_id") != int(array_text)
        or task.get("family") != arguments.family
        or task.get("seed") != arguments.seed
    ):
        raise ValueError("Gate-C CLI task differs from the sealed execution plan")
    resources = plan["resources"]
    if (
        not isinstance(resources, dict)
        or resources.get("checkpoint_cadence_updates") != arguments.checkpoint_cadence_updates
        or resources.get("max_scheduler_segments") != 1
    ):
        raise ValueError("Gate-C checkpoint/segment policy differs from the sealed plan")

    science = load_r3_science_config(arguments.science_config)
    materialized = materialize_r3_control_train_only_from_parent(
        project_root=arguments.project_root,
        parent_registry_path=arguments.parent_registry,
        expected_parent_registry_file_sha256=_digest(
            arguments.parent_registry_file_sha256,
            name="parent registry file SHA-256",
        ),
        source_config_path=arguments.source_config,
        science_bundle=science,
        seed=arguments.seed,
        device="cuda",
    )
    result = run_r3_control_family(
        materialized.capability,
        arguments.family,
        controls_config=controls,
    )
    artifact = publish_canonical_artifact(arguments.output.absolute(), result)
    _emit(
        {
            "status": "gate_c_family_result_published",
            "task_id": arguments.task_id,
            "family": arguments.family,
            "seed": arguments.seed,
            "completed_updates": result["completion"]["completed_updates"],
            "result_sha256": result["result_sha256"],
            "file_sha256": artifact.file_sha256,
            "primary_label_stream_constructed": False,
            "external_scheduler_terminal_required": True,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
