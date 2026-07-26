#!/usr/bin/env python3
"""Run one disposable 100-update Gate-C family profile on HPC4."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from smart_reward.phase2_r3_artifacts import publish_canonical_artifact
from smart_reward.phase2_r3_config import load_r3_science_config
from smart_reward.phase2_r3_control_training import (
    profile_r3_control_family,
    validate_r3_control_profile_observation,
)
from smart_reward.phase2_r3_controls import (
    R3_GATE_C_FAMILIES,
    R3_GATE_C_SEEDS,
    load_r3_controls_config,
)
from smart_reward.phase2_r3_controls_hpc4 import build_profile_compute_receipt
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


def _commit(value: str) -> str:
    if (
        len(value) != 40
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("git commit must be a full lowercase Git SHA")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--science-config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--parent-registry", type=Path, required=True)
    parser.add_argument("--parent-registry-file-sha256", required=True)
    parser.add_argument("--controls-config", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--container-sha256", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
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
    task_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_text not in {"0", "1", "2"}:
        raise RuntimeError("Gate-C profile requires SLURM_ARRAY_TASK_ID in {0,1,2}")
    task_id = int(task_text)
    family = R3_GATE_C_FAMILIES[task_id]
    seed = R3_GATE_C_SEEDS[0]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Gate-C profile requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    gpu_total_memory_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    if gpu_total_memory_bytes != 46_068 * 1024**2:
        raise RuntimeError("Gate-C profile requires the observed 46,068-MiB HPC4 L20 capacity")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    input_started = time.perf_counter()

    controls = load_r3_controls_config(arguments.controls_config)
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
        seed=seed,
        device=device,
    )
    torch.cuda.synchronize(device)
    input_preparation_wall_seconds = time.perf_counter() - input_started
    observation = validate_r3_control_profile_observation(
        profile_r3_control_family(
            materialized.capability,
            family,
            controls_config=controls,
            checkpoint_directory=arguments.checkpoint_directory,
        )
    )
    torch.cuda.synchronize(device)
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    if peak_gpu_memory_bytes < 1:
        raise RuntimeError("Gate-C profile observed no allocated CUDA memory")
    setup_wall_seconds = (
        input_preparation_wall_seconds
        + float(observation["family_setup_wall_seconds"])
        + float(observation["one_time_audit_wall_seconds"])
    )
    receipt = build_profile_compute_receipt(
        family=family,
        seed=seed,
        git_commit=_commit(arguments.git_commit),
        container_sha256=_digest(
            arguments.container_sha256,
            name="container SHA-256",
        ),
        controls_config_file_sha256=controls.file_sha256,
        controls_config_semantic_sha256=controls.semantic_sha256,
        input_training_sha256=str(observation["input_training_sha256"]),
        oracle_reward_sha256=str(observation["train_oracle_rewards_sha256"]),
        setup_wall_seconds=setup_wall_seconds,
        training_wall_seconds=float(observation["training_wall_seconds"]),
        audit_wall_seconds=float(observation["scheduled_audit_wall_seconds"]),
        checkpoint_roundtrip_wall_seconds=float(observation["checkpoint_roundtrip_wall_seconds"]),
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        gpu_total_memory_bytes=gpu_total_memory_bytes,
    )
    artifact = publish_canonical_artifact(arguments.output.absolute(), receipt)
    _emit(
        {
            "status": "gate_c_profile_compute_receipt_published",
            "family": family,
            "seed": seed,
            "completed_updates": receipt["completed_updates"],
            "compute_receipt_sha256": receipt["compute_receipt_sha256"],
            "file_sha256": artifact.file_sha256,
            "formal_result_issued": False,
            "primary_label_stream_constructed": False,
            "external_scheduler_terminal_required": True,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
