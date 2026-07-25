#!/usr/bin/env python3
"""Fail-closed validation of the one-shot Phase-2 recovery parent evidence.

This control-plane program intentionally has no dependency on torch or the
project package.  It verifies the tracked three-seed registry, every immutable
FAILED-run evidence file, and every materialized-artifact content identity
before a recovery GPU job is allowed to open the oracle or train a head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA = "prorm-phase2-recovery-parent-failures/v1"
MARKER_SCHEMA = "prorm-phase2-run-status/v1"
ARTIFACT_BINDING_SCHEMA = "prorm-phase2-artifact-binding/v1"
ARTIFACT_SCHEMA = "controlled-feature-artifact/v1"
MATERIALIZATION_SCHEMA = "phase1-materialization/v1"
EXPECTED_SEEDS = (20260801, 20260802, 20260803)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")

TOP_LEVEL_KEYS = {
    "schema_version",
    "campaign",
    "common_artifact_identities",
    "optimizer_diagnostic",
    "seeds",
}
CAMPAIGN_KEYS = {
    "source_job_array_id",
    "parent_phase2_design_sha256",
    "base_config_hash",
    "producer",
    "failure_class",
    "failed_optimizer_updates",
    "first_order_tolerance",
    "consecutive_passes_required",
    "failure_aggregate",
    "one_shot_no_further_adaptation",
    "allowed_recovery_scope",
}
SEED_KEYS = {
    "seed",
    "array_task_id",
    "source_run",
    "source_artifact",
    "evidence_sha256",
    "artifact_sha256",
}
EVIDENCE_FILES = {
    "FAILED",
    "run-manifest.json",
    "artifact-materialization.json",
    "artifact-verification.json",
    "phase2-run.log",
}
ARTIFACT_FILES = {
    "metadata.json",
    "tensors.safetensors",
    "candidates.jsonl",
    "prompts.jsonl",
    "training_edges.jsonl",
    "evaluation_edges.jsonl",
}
ARTIFACT_DERIVED_DIGESTS = {
    "policy_prompt_semantics_records",
    "selected_prompt_ids",
}


class RecoveryParentError(ValueError):
    """The parent run or artifact does not match the frozen recovery registry."""


def _keys(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise RecoveryParentError(
            f"{name} fields differ from schema; "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise RecoveryParentError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, name: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RecoveryParentError(f"missing {name}: {path}") from error
    if not stat.S_ISREG(mode):
        raise RecoveryParentError(f"{name} must be a regular non-symlink file: {path}")


def _inside(root: Path, relative: object, name: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RecoveryParentError(f"{name} must be a non-empty project-relative path")
    raw = root / relative
    cursor = root
    for component in Path(relative).parts:
        if component in {"", ".", ".."}:
            raise RecoveryParentError(f"{name} contains an unsafe path component")
        cursor = cursor / component
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError as error:
            raise RecoveryParentError(f"{name} does not exist: {cursor}") from error
        if stat.S_ISLNK(mode):
            raise RecoveryParentError(f"{name} path contains a symlink: {cursor}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise RecoveryParentError(f"{name} does not exist: {raw}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RecoveryParentError(f"{name} escapes the project root") from error
    if raw.is_symlink() or not resolved.is_dir():
        raise RecoveryParentError(f"{name} must be a real directory, not a symlink")
    return resolved


def _json(path: Path, name: str) -> dict[str, Any]:
    _regular_file(path, name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryParentError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RecoveryParentError(f"{name} root must be an object")
    return value


def _verify_file(path: Path, expected: object, name: str) -> str:
    _regular_file(path, name)
    digest = _digest(expected, f"{name} expected SHA256")
    observed = _sha256(path)
    if observed != digest:
        raise RecoveryParentError(f"{name} SHA256 mismatch: expected {digest}, observed {observed}")
    return observed


def _marker(path: Path, *, seed: int, campaign: dict[str, Any]) -> None:
    _regular_file(path, "parent FAILED marker")
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            raise RecoveryParentError("parent FAILED marker contains a malformed line")
        key, value = line.split("=", 1)
        if not key or key in fields:
            raise RecoveryParentError("parent FAILED marker has an invalid or duplicate key")
        fields[key] = value
    required = {
        "schema_version": MARKER_SCHEMA,
        "status": "FAILED",
        "seed": str(seed),
        "phase2_design_sha256": campaign["parent_phase2_design_sha256"],
        "base_config_hash": campaign["base_config_hash"],
        "git_commit": campaign["producer"]["git_commit"],
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            raise RecoveryParentError(f"parent FAILED marker {key} mismatch")
    try:
        if int(fields["workload_exit_code"]) == 0 or int(fields["final_exit_code"]) == 0:
            raise RecoveryParentError("parent FAILED marker records a zero exit code")
    except (KeyError, ValueError) as error:
        raise RecoveryParentError("parent FAILED marker has invalid exit codes") from error


def _manifest(path: Path, *, seed: int, campaign: dict[str, Any]) -> None:
    value = _json(path, "parent run manifest")
    if (
        value.get("schema_version") != "smart-reward-run/v1"
        or value.get("config_hash") != campaign["base_config_hash"]
        or value.get("selected_seed") != seed
    ):
        raise RecoveryParentError("parent run manifest base identity or seed mismatch")
    git = value.get("git")
    slurm = value.get("slurm")
    if (
        not isinstance(git, dict)
        or git.get("commit") != campaign["producer"]["git_commit"]
        or git.get("dirty") is not False
    ):
        raise RecoveryParentError("parent run manifest producer commit mismatch")
    if not isinstance(slurm, dict):
        raise RecoveryParentError("parent run manifest lacks Slurm identity")
    expected_env = {
        "PRORM_GIT_COMMIT": campaign["producer"]["git_commit"],
        "PRORM_IMAGE_SHA256": campaign["producer"]["image_sha256"],
        "PRORM_HF_INVENTORY_SHA256": campaign["producer"]["hf_inventory_sha256"],
    }
    for key, expected in expected_env.items():
        if slurm.get(key) != expected:
            raise RecoveryParentError(f"parent run manifest {key} mismatch")
    if (
        slurm.get("SLURM_JOB_ACCOUNT") != "sigroup"
        or slurm.get("SLURM_JOB_PARTITION") != "gpu-l20"
        or slurm.get("SLURM_CLUSTER_NAME") != "hpc4"
        or slurm.get("SLURM_ARRAY_JOB_ID") != campaign["source_job_array_id"]
        or slurm.get("SLURM_ARRAY_TASK_ID") != str(EXPECTED_SEEDS.index(seed))
        or slurm.get("SLURM_GPUS_ON_NODE") != "1"
        or slurm.get("SLURM_NNODES") != "1"
        or slurm.get("SLURM_NTASKS") != "1"
        or slurm.get("CUDA_VISIBLE_DEVICES") != "0"
    ):
        raise RecoveryParentError("parent run manifest is not the selected HPC4 L20 array task")
    torch_state = value.get("torch")
    gpus = torch_state.get("gpus") if isinstance(torch_state, dict) else None
    if (
        not isinstance(torch_state, dict)
        or torch_state.get("cuda_available") is not True
        or torch_state.get("gpu_count") != 1
        or torch_state.get("version") != "2.7.1+cu126"
        or torch_state.get("cuda_version") != "12.6"
        or not isinstance(gpus, list)
        or len(gpus) != 1
        or gpus[0]
        != {
            "index": 0,
            "name": "NVIDIA L20",
            "total_memory_bytes": 47676129280,
            "compute_capability": "8.9",
        }
    ):
        raise RecoveryParentError("parent run manifest lacks its single NVIDIA L20 identity")


def _artifact_binding(
    path: Path,
    *,
    seed: int,
    campaign: dict[str, Any],
    metadata_sha256: str,
) -> None:
    value = _json(path, "parent artifact binding")
    expected = {
        "schema_version": ARTIFACT_BINDING_SCHEMA,
        "base_config_hash": campaign["base_config_hash"],
        "phase2_design_sha256": campaign["parent_phase2_design_sha256"],
        "seed": seed,
        "artifact_metadata_sha256": metadata_sha256,
        "producer": campaign["producer"],
    }
    if set(value) != {*expected, "mode"}:
        raise RecoveryParentError("parent artifact binding fields differ from schema")
    if value.get("mode") not in {"materialized", "reused"}:
        raise RecoveryParentError("parent artifact binding mode is invalid")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RecoveryParentError(f"parent artifact binding {key} mismatch")


def _artifact_verification(
    path: Path,
    *,
    seed: int,
    campaign: dict[str, Any],
    metadata_sha256: str,
) -> None:
    value = _json(path, "parent artifact verification")
    if (
        value.get("status") != "ok"
        or value.get("formal_environment") is not True
        or value.get("seed") != seed
        or value.get("base_config_hash") != campaign["base_config_hash"]
        or value.get("phase2_design_sha256") != campaign["parent_phase2_design_sha256"]
        or value.get("artifact_metadata_sha256") != metadata_sha256
    ):
        raise RecoveryParentError("parent artifact verification identity mismatch")


def _artifact_metadata(
    path: Path,
    *,
    seed: int,
    campaign: dict[str, Any],
    hashes: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    value = _json(path, "parent artifact metadata")
    if (
        value.get("schema") != ARTIFACT_SCHEMA
        or value.get("config_hash") != campaign["base_config_hash"]
        or value.get("seed") != seed
        or value.get("tensor_sha256") != hashes["tensors.safetensors"]
    ):
        raise RecoveryParentError("artifact metadata base, seed, or tensor identity mismatch")
    evidence = value.get("evidence")
    if not isinstance(evidence, dict):
        raise RecoveryParentError("artifact metadata evidence must be an object")
    if (
        evidence.get("schema") != MATERIALIZATION_SCHEMA
        or evidence.get("config_sha256") != campaign["base_config_hash"]
        or evidence.get("seed") != seed
        or evidence.get("producer") != campaign["producer"]
    ):
        raise RecoveryParentError("artifact producer/materialization identity mismatch")
    jsonl = evidence.get("jsonl_sha256")
    if not isinstance(jsonl, dict) or set(jsonl) != ARTIFACT_FILES - {
        "metadata.json",
        "tensors.safetensors",
    }:
        raise RecoveryParentError("artifact JSONL inventory is incomplete")
    for filename in jsonl:
        if jsonl[filename] != hashes[filename]:
            raise RecoveryParentError(f"artifact metadata {filename} SHA256 mismatch")
    prompt_semantics = evidence.get("policy_prompt_semantics")
    pool = evidence.get("prompt_pool_selection")
    if (
        not isinstance(prompt_semantics, dict)
        or prompt_semantics.get("records_sha256") != hashes["policy_prompt_semantics_records"]
        or not isinstance(pool, dict)
        or pool.get("selected_prompt_ids_sha256") != hashes["selected_prompt_ids"]
        or pool.get("eligible_prompt_ids_sha256") != common["eligible_prompt_ids_sha256"]
    ):
        raise RecoveryParentError("artifact prompt/candidate selection identity mismatch")
    tensor_specs = value.get("tensors")
    schema = common.get("tensor_schema")
    if not isinstance(tensor_specs, dict) or not isinstance(schema, dict):
        raise RecoveryParentError("artifact tensor schema is missing")
    if len(tensor_specs) != schema.get("num_tensor_keys"):
        raise RecoveryParentError("artifact tensor-key count mismatch")
    expected_shapes = {
        "train.policy_scores": schema["train_policy_scores_shape"],
        "train.reward_features": schema["train_reward_features_shape"],
        "validation.policy_scores": schema["validation_policy_scores_shape"],
        "validation.reward_features": schema["validation_reward_features_shape"],
        "test.policy_scores": schema["test_policy_scores_shape"],
        "test.reward_features": schema["test_reward_features_shape"],
    }
    for key, expected_shape in expected_shapes.items():
        spec = tensor_specs.get(key)
        if not isinstance(spec, dict) or spec.get("shape") != expected_shape:
            raise RecoveryParentError(f"artifact tensor shape mismatch for {key}")
    return value


def load_and_validate_registry(
    registry_path: Path,
    *,
    project_root: Path | None,
    expected_registry_sha256: str,
    expected_parent_design_sha256: str,
    expected_base_config_hash: str,
    seed: int | None,
    verify_sources: bool,
) -> dict[str, Any]:
    """Validate the tracked registry and optionally all referenced source bytes."""

    registry_path = registry_path.resolve(strict=True)
    _regular_file(registry_path, "parent failure registry")
    observed_registry_sha = _sha256(registry_path)
    if observed_registry_sha != _digest(
        expected_registry_sha256, "expected parent failure registry SHA256"
    ):
        raise RecoveryParentError("parent failure registry SHA256 mismatch")
    registry = _keys(_json(registry_path, "parent failure registry"), TOP_LEVEL_KEYS, "registry")
    if registry["schema_version"] != REGISTRY_SCHEMA:
        raise RecoveryParentError("unsupported parent failure registry schema")
    campaign = _keys(registry["campaign"], CAMPAIGN_KEYS, "campaign")
    producer = _keys(
        campaign["producer"],
        {"git_commit", "image_sha256", "hf_inventory_sha256"},
        "campaign.producer",
    )
    if HEX40.fullmatch(str(producer["git_commit"])) is None:
        raise RecoveryParentError("parent producer git commit is invalid")
    _digest(producer["image_sha256"], "parent image SHA256")
    _digest(producer["hf_inventory_sha256"], "parent inventory SHA256")
    if (
        campaign["parent_phase2_design_sha256"] != expected_parent_design_sha256
        or campaign["base_config_hash"] != expected_base_config_hash
    ):
        raise RecoveryParentError("registry parent design/base identity mismatch")
    _digest(campaign["parent_phase2_design_sha256"], "parent design SHA256")
    _digest(campaign["base_config_hash"], "base config hash")
    if (
        campaign["source_job_array_id"] != "1647491"
        or campaign["failure_class"] != "primary_bt_mle_first_order_convergence_gate_not_met"
        or campaign["failed_optimizer_updates"] != 5760
        or campaign["first_order_tolerance"] != 0.001
        or campaign["consecutive_passes_required"] != 3
        or campaign["failure_aggregate"]
        != {
            "present": False,
            "reason": "the_failed_job_array_predates_structured_failure_aggregation",
            "replacement_evidence": (
                "exact_three_seed_registry_binds_each_failed_terminal_and_phase2_run_log"
            ),
        }
        or campaign["one_shot_no_further_adaptation"] is not True
        or campaign["allowed_recovery_scope"] != "train_only_same_materialized_artifacts"
    ):
        raise RecoveryParentError("registry campaign is not the authorized one-shot recovery")

    root = project_root.resolve(strict=True) if project_root is not None else None
    common = _keys(
        registry["common_artifact_identities"],
        {"eligible_prompt_ids_sha256", "tensor_schema"},
        "common_artifact_identities",
    )
    _digest(common["eligible_prompt_ids_sha256"], "eligible prompt IDs SHA256")
    diagnostic = _keys(
        registry["optimizer_diagnostic"],
        {
            "schema_version",
            "source_job_id",
            "source_git_commit",
            "path",
            "sha256",
            "seed",
            "artifact_metadata_sha256",
            "configured_maximum_steps",
            "diagnostic_only",
            "train_only",
            "nonconfirmatory",
            "tested_decay_schedule",
            "decay_schedule_selected_step",
            "lbfgs_gate_passed",
        },
        "optimizer_diagnostic",
    )
    if (
        diagnostic["schema_version"] != "phase2-bt-convergence-diagnostic/v1"
        or diagnostic["source_job_id"] != "1647982"
        or HEX40.fullmatch(str(diagnostic["source_git_commit"])) is None
        or diagnostic["seed"] != EXPECTED_SEEDS[0]
        or diagnostic["artifact_metadata_sha256"]
        != "83924663eefd089deeb29dd87cc13629eb566bd4e38df0b5ceb57355ae7f343f"
        or diagnostic["configured_maximum_steps"] != 5760
        or diagnostic["diagnostic_only"] is not True
        or diagnostic["train_only"] is not True
        or diagnostic["nonconfirmatory"] is not True
        or diagnostic["tested_decay_schedule"]
        != [
            {"additional_updates": 1000, "learning_rate": 0.0003},
            {"additional_updates": 2000, "learning_rate": 0.0001},
            {"additional_updates": 2000, "learning_rate": 0.00003},
            {"additional_updates": 2000, "learning_rate": 0.00001},
        ]
        or diagnostic["decay_schedule_selected_step"] != 6900
        or diagnostic["lbfgs_gate_passed"] is not True
    ):
        raise RecoveryParentError("optimizer diagnostic registry binding is invalid")
    _digest(diagnostic["sha256"], "optimizer diagnostic SHA256")
    if verify_sources:
        if root is None:
            raise RecoveryParentError("project_root is required to verify optimizer diagnostic")
        diagnostic_path = root / str(diagnostic["path"])
        try:
            resolved_diagnostic = diagnostic_path.resolve(strict=True)
            resolved_diagnostic.relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise RecoveryParentError("optimizer diagnostic path is missing or unsafe") from error
        _verify_file(
            resolved_diagnostic,
            diagnostic["sha256"],
            "optimizer diagnostic",
        )
        diagnostic_value = _json(resolved_diagnostic, "optimizer diagnostic")
        expected_top = {
            "adamw_decay_probe",
            "artifact_metadata_sha256",
            "configured_adamw",
            "elapsed_seconds",
            "evidence_role",
            "information_boundary",
            "input_training_sha256",
            "label_stream_identity",
            "lbfgs_probe",
            "oracle_rescore",
            "phase2_design_sha256",
            "run_manifest_sha256",
            "schema_version",
            "seed",
            "source_config_hash",
            "training_settings_sha256",
        }
        if set(diagnostic_value) != expected_top:
            raise RecoveryParentError("optimizer diagnostic top-level schema is invalid")
        if (
            diagnostic_value["schema_version"] != diagnostic["schema_version"]
            or diagnostic_value["evidence_role"]
            != "nonconfirmatory_train_only_optimizer_diagnostic"
            or diagnostic_value["seed"] != diagnostic["seed"]
            or diagnostic_value["phase2_design_sha256"] != campaign["parent_phase2_design_sha256"]
            or diagnostic_value["source_config_hash"] != campaign["base_config_hash"]
            or diagnostic_value["artifact_metadata_sha256"]
            != diagnostic["artifact_metadata_sha256"]
            or diagnostic_value["information_boundary"]
            != {
                "train_only": True,
                "validation_or_test_targets_accessed": False,
                "raw_oracle_values_serialized": False,
                "raw_labels_serialized": False,
                "head_vectors_serialized": False,
                "eligible_for_primary_claim": False,
            }
        ):
            raise RecoveryParentError("optimizer diagnostic identity/boundary is invalid")
        label_identity = diagnostic_value["label_stream_identity"]
        if not isinstance(label_identity, dict):
            raise RecoveryParentError("optimizer diagnostic label identity is invalid")
        required_label_values = {
            "base_seed": 20260801,
            "derivation_sha256": "a1e901f534096bedb757ba978e2ba9838031aeacab4e86265557279953b236ae",
            "initial_generator_state_sha256": (
                "6f8e7260e641e4f52990e6f28c6558f333133bd0286549dbca3de426bf51a3d1"
            ),
            "final_generator_state_sha256": (
                "7dbc6a6143c98a995abda1baab73de8867267121c1299d80b6c10f8165f6ce83"
            ),
            "mean_h_sha256": "524eb0c9936dffe8d0ef807d4b4181cb3c4bbad556a51c7578a16c06b1e13cf0",
            "replicate_count_sha256": (
                "b905ce98d6ec87a03bbda10405e3ffc766ff913bf9613e58bb50bf1ffa7b63c8"
            ),
            "replicate_win_sha256": (
                "de68ae122cfe40a09146fdd00f367b640bdcc03a12aff46e6713fe7e88cafb13"
            ),
            "replicate_h_sha256": (
                "92a99a227ce5679049c856b3fbd92005d0d9a8460760bc346a5ef266d7d3350d"
            ),
            "realized_total_annotations": 61011,
        }
        if not any(
            label_identity.get(key) == "prorm-common-beta-r4-labels-v1"
            for key in ("namespace", "rng_namespace")
        ):
            raise RecoveryParentError("optimizer diagnostic label namespace is invalid")
        for key, expected in required_label_values.items():
            if label_identity.get(key) != expected:
                raise RecoveryParentError(f"optimizer diagnostic label {key} mismatch")
        if (
            diagnostic_value["run_manifest_sha256"]
            != "5feceba80c4717a539a3d2d86a308f5eb20411075007643a4ceb48c43987218a"
            or diagnostic_value["training_settings_sha256"]
            != "67a8f945bbe761b88d5583d6c1785bb6a6a745c41727df882352766aee4409f2"
            or diagnostic_value["input_training_sha256"]
            != "adb87d1ee4b8d42dcd1ae6475d5b13d972af8a99ef398384ab58ef8a7d4edd97"
        ):
            raise RecoveryParentError("optimizer diagnostic input identity is invalid")
        oracle_rescore = diagnostic_value["oracle_rescore"]
        if (
            not isinstance(oracle_rescore, dict)
            or oracle_rescore.get("transformed_rewards_sha256")
            != "7a7d7b005ec7e377205d6f40743bed950ad38154dec6f54516f7ced8ffca0b1a"
            or oracle_rescore.get("raw_values_serialized") is not False
        ):
            raise RecoveryParentError("optimizer diagnostic oracle identity is invalid")
        configured = diagnostic_value["configured_adamw"]
        failure = configured.get("failure") if isinstance(configured, dict) else None
        failure_spec = failure.get("spec") if isinstance(failure, dict) else None
        if (
            not isinstance(configured, dict)
            or configured.get("converged") is not False
            or configured.get("success") is not None
            or not isinstance(failure, dict)
            or failure.get("converged") is not False
            or failure.get("fail_closed") is not True
            or failure.get("selected_primary_step") is not None
            or failure.get("final_gate") is not None
            or failure.get("test_or_validation_data_accessed") is not False
            or not isinstance(failure_spec, dict)
            or failure_spec.get("gradient_ratio_tolerance") != 0.001
            or failure_spec.get("min_steps") != 100
            or failure_spec.get("max_steps") != 5760
            or failure_spec.get("check_interval") != 20
            or failure_spec.get("consecutive_checks") != 3
            or failure_spec.get("gradient") != "full_data_post_update_unclipped"
            or failure_spec.get("denominator") != "exact_zero_initialization_gradient_l2_norm"
            or failure_spec.get("fail_closed") is not True
            or failure_spec.get("validation_or_test_selection") is not False
        ):
            raise RecoveryParentError("optimizer diagnostic configured AdamW failure is invalid")
        decay = diagnostic_value["adamw_decay_probe"]
        if (
            not isinstance(decay, dict)
            or decay.get("schema_version") != "phase2-bt-adamw-decay-probe/v1"
            or decay.get("starts_after_configured_max_steps") is not True
            or decay.get("stages")
            != [
                {"learning_rate": 0.0003, "maximum_updates": 1000},
                {"learning_rate": 0.0001, "maximum_updates": 2000},
                {"learning_rate": 0.00003, "maximum_updates": 2000},
                {"learning_rate": 0.00001, "maximum_updates": 2000},
            ]
            or decay.get("check_interval") != 20
            or decay.get("gradient_ratio_tolerance") != 0.001
            or decay.get("required_consecutive_checks") != 3
            or decay.get("converged") is not True
            or decay.get("selected_step") != 6900
        ):
            raise RecoveryParentError("optimizer diagnostic AdamW decay probe is invalid")
        lbfgs = diagnostic_value["lbfgs_probe"]
        if (
            not isinstance(lbfgs, dict)
            or lbfgs.get("schema_version") != "phase2-bt-lbfgs-probe/v1"
            or lbfgs.get("configured_gate_passed") is not True
            or lbfgs.get("initialization") != "exact_zero_head"
            or lbfgs.get("dtype") != "float64"
            or lbfgs.get("full_batch") is not True
            or lbfgs.get("line_search") != "strong_wolfe"
            or lbfgs.get("gradient_ratio_to_zero_initialization") != 3.9269623845303005e-05
            or lbfgs.get("final_objective") != 0.6716634777372092
            or lbfgs.get("head_vector_serialized") is not False
        ):
            raise RecoveryParentError("optimizer diagnostic L-BFGS probe is invalid")
    entries = registry["seeds"]
    if (
        not isinstance(entries, list)
        or tuple(entry.get("seed") for entry in entries) != EXPECTED_SEEDS
    ):
        raise RecoveryParentError("registry must contain the exact ordered three parent seeds")
    selected: dict[str, Any] | None = None
    for index, raw_entry in enumerate(entries):
        entry = _keys(raw_entry, SEED_KEYS, f"seeds[{index}]")
        entry_seed = entry["seed"]
        if entry["array_task_id"] != index:
            raise RecoveryParentError("registry seed-to-array-task mapping is invalid")
        evidence_hashes = _keys(
            entry["evidence_sha256"], EVIDENCE_FILES, f"seed {entry_seed} evidence"
        )
        artifact_hashes = _keys(
            entry["artifact_sha256"],
            ARTIFACT_FILES | ARTIFACT_DERIVED_DIGESTS,
            f"seed {entry_seed} artifact",
        )
        for name, digest in {**evidence_hashes, **artifact_hashes}.items():
            _digest(digest, f"seed {entry_seed} {name}")
        expected_run = (
            "runs/phase2-pilot/"
            f"{campaign['parent_phase2_design_sha256']}/seed-{entry_seed}/"
            f"job-{campaign['source_job_array_id']}_{index}"
        )
        expected_artifact = (
            f"artifacts/{campaign['base_config_hash']}/"
            f"{producer['image_sha256']}/{producer['hf_inventory_sha256']}/"
            f"{producer['git_commit']}/seed-{entry_seed}"
        )
        if entry["source_run"] != expected_run or entry["source_artifact"] != expected_artifact:
            raise RecoveryParentError("registry source run/artifact address is not canonical")
        enriched = dict(entry)
        if verify_sources:
            if root is None:
                raise RecoveryParentError("project_root is required to verify source bytes")
            run_dir = _inside(root, entry["source_run"], "source_run")
            artifact_dir = _inside(root, entry["source_artifact"], "source_artifact")
            if (run_dir / "SUCCESS").exists() or (run_dir / "SUCCESS").is_symlink():
                raise RecoveryParentError("parent FAILED run must not contain SUCCESS")
            artifact_link = run_dir / "artifact"
            try:
                link_mode = artifact_link.lstat().st_mode
            except FileNotFoundError as error:
                raise RecoveryParentError("parent run is missing its artifact symlink") from error
            if not stat.S_ISLNK(link_mode):
                raise RecoveryParentError("parent run artifact must be a terminal symlink")
            if artifact_link.resolve(strict=True) != artifact_dir:
                raise RecoveryParentError(
                    "parent run artifact symlink does not resolve to the canonical artifact"
                )
            observed_artifact_entries = {child.name for child in artifact_dir.iterdir()}
            if observed_artifact_entries != ARTIFACT_FILES:
                raise RecoveryParentError(
                    "parent artifact directory entries differ from the fixed six-file schema"
                )
            for filename in ARTIFACT_FILES:
                _regular_file(artifact_dir / filename, f"artifact {filename}")
            for filename, expected in evidence_hashes.items():
                _verify_file(run_dir / filename, expected, f"parent {filename}")
            for filename in ARTIFACT_FILES:
                _verify_file(
                    artifact_dir / filename,
                    artifact_hashes[filename],
                    f"artifact {filename}",
                )
            _marker(run_dir / "FAILED", seed=entry_seed, campaign=campaign)
            expected_log = (
                "error: bt_mle did not satisfy the sustained first-order "
                "gradient-ratio gate by 5760 steps\n"
            )
            try:
                log_text = (run_dir / "phase2-run.log").read_text(encoding="utf-8")
            except UnicodeError as error:
                raise RecoveryParentError("parent phase2-run.log is not UTF-8") from error
            if log_text != expected_log:
                raise RecoveryParentError(
                    "parent phase2-run.log does not prove the exact BT convergence failure"
                )
            _manifest(run_dir / "run-manifest.json", seed=entry_seed, campaign=campaign)
            _artifact_binding(
                run_dir / "artifact-materialization.json",
                seed=entry_seed,
                campaign=campaign,
                metadata_sha256=artifact_hashes["metadata.json"],
            )
            _artifact_verification(
                run_dir / "artifact-verification.json",
                seed=entry_seed,
                campaign=campaign,
                metadata_sha256=artifact_hashes["metadata.json"],
            )
            _artifact_metadata(
                artifact_dir / "metadata.json",
                seed=entry_seed,
                campaign=campaign,
                hashes=artifact_hashes,
                common=common,
            )
            enriched["source_run_resolved"] = os.fspath(run_dir)
            enriched["source_artifact_resolved"] = os.fspath(artifact_dir)
        if seed is not None and entry_seed == seed:
            selected = enriched
    if seed is not None and selected is None:
        raise RecoveryParentError("requested seed is not in the parent failure registry")
    return {
        "status": "ok",
        "schema_version": REGISTRY_SCHEMA,
        "registry_sha256": observed_registry_sha,
        "campaign": campaign,
        "selected_seed": selected,
        "all_three_sources_verified": verify_sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-parent-design-sha256", required=True)
    parser.add_argument("--expected-base-config-hash", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify_sources and args.project_root is None:
        parser.error("--verify-sources requires --project-root")
    result = load_and_validate_registry(
        args.registry,
        project_root=args.project_root,
        expected_registry_sha256=args.expected_registry_sha256,
        expected_parent_design_sha256=args.expected_parent_design_sha256,
        expected_base_config_hash=args.expected_base_config_hash,
        seed=args.seed,
        verify_sources=args.verify_sources,
    )
    rendered = json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
