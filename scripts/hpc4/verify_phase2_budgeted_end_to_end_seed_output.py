#!/usr/bin/env python3
"""Fail-closed verifier for one budgeted Phase-2 end-to-end seed.

The verifier is deliberately separate from both the workload and the
fixed-three descriptive aggregate.  It proves that the raw result, rollout
sidecar, run manifest, freshly materialized artifact metadata, and local
materialization receipt form one immutable seed/design/environment closure.
It then admits the result through the shared budgeted normalizer and publishes
one deterministic, non-formal verification record without replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from smart_reward.config import config_hash
from smart_reward.phase2_config import (
    PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
    PHASE2_BUDGETED_END_TO_END_SEEDS,
    PHASE2_BUDGETED_END_TO_END_STAGE,
    load_phase2_config_bundle,
)
from smart_reward.phase2_exploratory_aggregate import (
    normalize_budgeted_end_to_end_seed_result,
)
from smart_reward.phase2_inputs import prepare_phase2_inputs
from smart_reward.phase2_rollout import (
    PHASE2_ARM_ORDER,
    PHASE2_BUDGETED_RESULT_SCHEMA,
    PHASE2_BUDGETED_ROLLOUT_SCHEMA,
    Phase2Design,
)
from smart_reward.seeding import SeedBundle, derive_seed

_VERIFICATION_SCHEMA = "prorm-phase2-budgeted-fixed-three-seed-output-verification/v1"
_MANIFEST_SCHEMA = "smart-reward-run/v1"
_ARTIFACT_SCHEMA = "controlled-feature-artifact/v1"
_ARTIFACT_BINDING_SCHEMA = "prorm-phase2-budgeted-artifact-binding/v1"
_PROMPT_SEMANTICS_SCHEMA = "full-policy-prompt-semantics/v1"
_RESULT_FILENAME = "phase2-result.json"
_ROLLOUT_FILENAME = "phase2-result.rollouts.jsonl"
_VERIFICATION_FILENAME = "phase2-budgeted-output-verification.json"
_MANIFEST_FILENAME = "run-manifest.json"
_ARTIFACT_DIRECTORY = "artifact"
_ARTIFACT_METADATA_FILENAME = "metadata.json"
_ARTIFACT_BINDING_FILENAME = "artifact-materialization.json"
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_BYTES = 2 * 1024 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")

_ROLLOUT_KEYS = frozenset(
    {
        "schema_version",
        "design_stage",
        "evidence_role",
        "formal_claim_eligible",
        "supports_formal_claim",
        "arm",
        "policy_source",
        "beta_common",
        "prompt_id",
        "candidate_index",
        "prompt",
        "prompt_semantics",
        "response",
        "token_ids",
        "response_mask",
        "response_token_count",
        "terminated_by_eos",
        "reached_max_length",
        "prompt_rollout_seed",
        "kl_orientation",
        "kl_history_source",
        "on_policy_kl_pi_updated_to_pi0",
        "transformed_oracle_reward",
        "target_utility",
        "raw_oracle_logit_serialized",
    }
)
_PROMPT_SEMANTICS_KEYS = frozenset(
    {
        "schema_version",
        "raw_prompt_sha256",
        "policy_chat_token_count",
        "policy_prompt_token_ids_sha256",
        "max_prompt_tokens",
        "truncated",
        "raw_prompt_preserved",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "created_at_utc",
        "config_hash",
        "normalized_config",
        "seed",
        "selected_seed",
        "named_seeds",
        "git",
        "python",
        "platform",
        "torch",
        "revisions",
        "packages",
        "slurm",
    }
)
_ARTIFACT_KEYS = frozenset(
    {
        "schema",
        "config_hash",
        "seed",
        "splits",
        "tensors",
        "tensor_sha256",
        "evidence",
    }
)
_ENDPOINT_KEYS = frozenset(
    {
        "heldout_local_regret",
        "finite_policy_utility",
        "oracle_pairwise_cross_entropy",
        "oracle_probability_mae",
        "pairwise_order_accuracy",
    }
)
_LEARNERS = frozenset({"bt_mle", "prorm_plus"})
_NUMERICAL_EVENT_SEQUENCE = [
    "freeze_heldout_evaluation_state",
    "policy_rollouts_and_on_policy_kl",
    "enforced_nonformal_pre_oracle_safety",
    "final_operational_oracle_rollout_scoring",
    "deferred_heldout_oracle_scoring_and_metrics",
]


def _digest(value: object, *, name: str, lengths: frozenset[int] = frozenset({64})) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal digest")
    return value


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _canonical_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{name} does not exist") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical real directory")
    return absolute


def _canonical_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{name} does not exist") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise ValueError(f"{name} must be a canonical regular non-symlink file")
    return absolute


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_nonfinite(value: object, *, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, name=f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, name=f"{name}[{index}]")


def _decode_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite constant {token!r}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error
    result = _mapping(value, name=name)
    _reject_nonfinite(result, name=name)
    return result


def _strict_json(path: Path, *, name: str) -> tuple[dict[str, Any], bytes, str]:
    source = _canonical_file(path, name=name)
    size = source.stat().st_size
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise ValueError(f"{name} has an invalid byte size")
    raw = source.read_bytes()
    return _decode_json_object(raw, name=name), raw, _sha256_bytes(raw)


def _strict_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    source = _canonical_file(path, name="rollout JSONL")
    size = source.stat().st_size
    if size <= 0 or size > _MAX_JSONL_BYTES:
        raise ValueError("rollout JSONL has an invalid byte size")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            if not raw_line.endswith(b"\n"):
                raise ValueError("rollout JSONL must end every row with LF")
            line = raw_line[:-1]
            if not line or b"\r" in line or b"\0" in line:
                raise ValueError(f"rollout JSONL row {line_number} is empty or non-canonical")
            records.append(_decode_json_object(line, name=f"rollout JSONL row {line_number}"))
    if not records:
        raise ValueError("rollout JSONL must contain at least one row")
    return records, digest.hexdigest()


def _relative_file_reference(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a non-empty POSIX relative filename")
    reference = PurePosixPath(value)
    if reference.is_absolute() or len(reference.parts) != 1 or reference.name in {".", ".."}:
        raise ValueError(f"{name} must be one relative filename without traversal")
    return value


def _relative_directory_reference(value: object, *, name: str) -> str:
    reference = _relative_file_reference(value, name=name)
    if "." in PurePosixPath(reference).name:
        raise ValueError(f"{name} must be one relative directory name")
    return reference


def _require_result_root(
    result: Mapping[str, object],
    *,
    expected_seed: int,
    expected_design_sha256: str,
    expected_base_config_hash: str,
    expected_freeze_sha256: str,
) -> float:
    exact = {
        "schema_version": PHASE2_BUDGETED_RESULT_SCHEMA,
        "design_stage": PHASE2_BUDGETED_END_TO_END_STAGE,
        "formal_eligibility": False,
        "formal_claim_eligible": False,
        "supports_formal_claim": False,
        "per_seed_supports_formal_claim": False,
        "excluded_from_confirmatory_evidence": True,
        "confirmatory_authorization_created": False,
        "evidence_role": PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
        "seed": expected_seed,
        "phase2_design_sha256": expected_design_sha256,
        "source_config_hash": expected_base_config_hash,
    }
    for key, expected in exact.items():
        if result.get(key) != expected:
            raise ValueError(f"result field {key!r} is not bound to the expected budgeted seed")
    runtime = _mapping(result.get("phase2_runtime_contract"), name="result runtime contract")
    runtime_sha256 = _digest(
        result.get("phase2_runtime_contract_sha256"),
        name="result phase2_runtime_contract_sha256",
    )
    runtime_raw = json.dumps(
        runtime,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if _sha256_bytes(runtime_raw) != runtime_sha256:
        raise ValueError("result runtime-contract bytes do not match their SHA256")
    frozen = _mapping(
        result.get("common_beta_frozen_evidence"),
        name="result common_beta_frozen_evidence",
    )
    if (
        frozen.get("schema_version") != "common-beta-frozen-global-budgeted/v1"
        or frozen.get("evidence_role") != PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE
        or frozen.get("formal_eligibility") is not False
        or frozen.get("supports_formal_claim") is not False
        or frozen.get("beta_source_aggregate_sha256") != expected_freeze_sha256
        or runtime.get("beta_source_aggregate_sha256") != expected_freeze_sha256
    ):
        raise ValueError("result is not bound to the accepted freeze evidence")
    frozen_beta = _finite(
        frozen.get("frozen_global_beta"),
        name="result frozen_global_beta",
        positive=True,
    )
    if (
        _finite(frozen.get("beta_common"), name="result frozen beta_common", positive=True)
        != frozen_beta
        or _finite(runtime.get("frozen_global_beta"), name="runtime frozen beta", positive=True)
        != frozen_beta
    ):
        raise ValueError("result reports inconsistent frozen beta values")
    rollout_seed = SeedBundle.from_base_seed(expected_seed).rollout
    if result.get("common_random_numbers") != {
        "named_stream": "rollout",
        "seed": rollout_seed,
        "same_per_prompt_seed_reset_across_arms": True,
        "candidate_index_alignment": True,
    }:
        raise ValueError("result common-random-number evidence is invalid")
    if (
        result.get("numerical_event_sequence") != _NUMERICAL_EVENT_SEQUENCE
        or result.get("numerical_event_sequence_matches_confirmatory") is not True
    ):
        raise ValueError("result numerical event sequence is invalid")
    boundary = _mapping(result.get("information_boundary"), name="result information boundary")
    expected_boundary = {
        "current_seed_train_curvature_role": "predicted_kl_diagnostic_only",
        "new_rollout_prompts_used_for_calibration": False,
        "source_materialization_heldout_scores_used_for_calibration": False,
        "new_rollout_oracle_scoring_after_heads_beta_and_directions_frozen": True,
        "heldout_candidate_rescore_after_heads_beta_and_directions_frozen": True,
        "heldout_directions_used_for_policy": False,
        "source_artifact_may_contain_prior_heldout_candidate_scores": True,
    }
    for key, expected in expected_boundary.items():
        if boundary.get(key) != expected:
            raise ValueError(f"result information-boundary field {key!r} is invalid")
    if boundary.get("beta_selection_split") != runtime.get("common_beta_calibration_split"):
        raise ValueError("result beta selection split differs from the runtime contract")
    return frozen_beta


def _validate_overlay(
    overlay: Path,
    *,
    expected_seed: int,
    expected_design_sha256: str,
    expected_base_config_hash: str,
    expected_freeze_sha256: str,
    frozen_beta: float,
) -> tuple[dict[str, Any], dict[str, Any], str, int]:
    _canonical_file(overlay, name="budgeted overlay")
    bundle = load_phase2_config_bundle(overlay)
    config = _mapping(bundle.config, name="validated budgeted overlay")
    base = _mapping(bundle.base_config, name="validated budgeted base config")
    if bundle.design_identity != expected_design_sha256:
        raise ValueError("overlay design identity differs from --design-sha256")
    if config_hash(base) != expected_base_config_hash:
        raise ValueError("overlay base config differs from --base-config-hash")
    design = _mapping(config.get("design"), name="overlay design")
    run = _mapping(config.get("run"), name="overlay run")
    objective = _mapping(config.get("objective"), name="overlay objective")
    common_beta = _mapping(objective.get("common_beta"), name="overlay common beta")
    evaluation = _mapping(config.get("evaluation"), name="overlay evaluation")
    maximum = _mapping(evaluation.get("max_length"), name="overlay max_length")
    if (
        design.get("stage") != PHASE2_BUDGETED_END_TO_END_STAGE
        or design.get("formal_eligibility") is not False
        or design.get("evidence_role") != PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE
        or design.get("source_config_hash") != expected_base_config_hash
        or tuple(run.get("seeds", ())) != PHASE2_BUDGETED_END_TO_END_SEEDS
        or expected_seed not in PHASE2_BUDGETED_END_TO_END_SEEDS
        or run.get("confirmatory") is not False
        or common_beta.get("beta_source_aggregate_sha256") != expected_freeze_sha256
        or maximum.get("parent_pilot_aggregate_sha256") != expected_freeze_sha256
        or _finite(
            common_beta.get("frozen_global_beta"),
            name="overlay frozen_global_beta",
            positive=True,
        )
        != frozen_beta
    ):
        raise ValueError("overlay is not the fixed-three accepted-freeze budgeted design")
    candidates = _integer(
        evaluation.get("rollout_candidates_per_prompt"),
        name="overlay rollout_candidates_per_prompt",
        minimum=1,
    )
    return config, base, _sha256_file(overlay), candidates


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    expected_seed: int,
    expected_base_config_hash: str,
    expected_base_config: Mapping[str, object],
    expected_git_commit: str,
    expected_image_sha256: str,
    expected_inventory_sha256: str,
    expected_slurm_job_id_raw: str,
    expected_array_job_id: str,
    expected_array_task_id: int,
) -> dict[str, object]:
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("run manifest does not have the exact smart-reward-run/v1 schema")
    if (
        manifest.get("schema_version") != _MANIFEST_SCHEMA
        or manifest.get("config_hash") != expected_base_config_hash
        or manifest.get("normalized_config") != expected_base_config
        or manifest.get("selected_seed") != expected_seed
        or tuple(manifest.get("seed", ())) != PHASE2_BUDGETED_END_TO_END_SEEDS
    ):
        raise ValueError("run manifest config or seed identity is cross-bound")
    git = _mapping(manifest.get("git"), name="run manifest git")
    slurm = _mapping(manifest.get("slurm"), name="run manifest slurm")
    torch_state = _mapping(manifest.get("torch"), name="run manifest torch")
    gpus = _sequence(torch_state.get("gpus"), name="run manifest torch.gpus")
    if len(gpus) != 1:
        raise ValueError("run manifest must record exactly one GPU")
    gpu = _mapping(gpus[0], name="run manifest GPU")
    gpu_name = gpu.get("name")
    if (
        git.get("commit") != expected_git_commit
        or git.get("dirty") is not False
        or slurm.get("PRORM_GIT_COMMIT") != expected_git_commit
        or slurm.get("PRORM_IMAGE_SHA256") != expected_image_sha256
        or slurm.get("PRORM_HF_INVENTORY_SHA256") != expected_inventory_sha256
        or slurm.get("SLURM_JOB_ID") != expected_slurm_job_id_raw
        or slurm.get("SLURM_ARRAY_JOB_ID") != expected_array_job_id
        or slurm.get("SLURM_ARRAY_TASK_ID") != str(expected_array_task_id)
        or slurm.get("SLURM_JOB_ACCOUNT") != "sigroup"
        or slurm.get("SLURM_JOB_PARTITION") != "gpu-l20"
        or slurm.get("SLURM_CLUSTER_NAME") != "hpc4"
        or torch_state.get("cuda_available") is not True
        or torch_state.get("gpu_count") != 1
        or gpu_name != "NVIDIA L20"
    ):
        raise ValueError("run manifest environment or Slurm identity is invalid")
    expected_index = PHASE2_BUDGETED_END_TO_END_SEEDS.index(expected_seed)
    if expected_array_task_id != expected_index:
        raise ValueError("Slurm array task is cross-bound to the wrong fixed seed")
    return {
        "formal": True,
        "git_commit": expected_git_commit,
        "image_sha256": expected_image_sha256,
        "hf_inventory_sha256": expected_inventory_sha256,
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_models": ["NVIDIA L20"],
    }


def _test_prompt_ids(metadata: Mapping[str, object]) -> list[str]:
    splits = _mapping(metadata.get("splits"), name="artifact splits")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("artifact splits must be exactly train, validation, and test")
    all_ids: list[str] = []
    test_ids: list[str] = []
    for split in ("train", "validation", "test"):
        value = _mapping(splits.get(split), name=f"artifact split {split}")
        if set(value) != {"prompt_ids"}:
            raise ValueError(f"artifact split {split} has an invalid schema")
        raw_ids = _sequence(value.get("prompt_ids"), name=f"artifact split {split} prompt_ids")
        if not raw_ids or any(not isinstance(item, str) or not item for item in raw_ids):
            raise ValueError(f"artifact split {split} prompt IDs must be non-empty strings")
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError(f"artifact split {split} contains duplicate prompt IDs")
        if set(all_ids).intersection(raw_ids):
            raise ValueError("artifact split prompt IDs overlap")
        all_ids.extend(raw_ids)
        if split == "test":
            test_ids = list(raw_ids)
    return test_ids


def _validate_artifact(
    metadata: Mapping[str, object],
    materialization: Mapping[str, object],
    *,
    expected_seed: int,
    expected_design_sha256: str,
    expected_base_config_hash: str,
    expected_git_commit: str,
    expected_image_sha256: str,
    expected_inventory_sha256: str,
    expected_metadata_sha256: str,
) -> list[str]:
    if set(metadata) != _ARTIFACT_KEYS:
        raise ValueError("artifact metadata does not have the exact controlled artifact schema")
    evidence = _mapping(metadata.get("evidence"), name="artifact evidence")
    producer = _mapping(evidence.get("producer"), name="artifact producer")
    if (
        metadata.get("schema") != _ARTIFACT_SCHEMA
        or metadata.get("config_hash") != expected_base_config_hash
        or metadata.get("seed") != expected_seed
        or producer
        != {
            "git_commit": expected_git_commit,
            "image_sha256": expected_image_sha256,
            "hf_inventory_sha256": expected_inventory_sha256,
        }
    ):
        raise ValueError("artifact metadata is cross-bound to another seed/config/producer")
    _digest(metadata.get("tensor_sha256"), name="artifact tensor_sha256")
    expected_binding = {
        "schema_version": _ARTIFACT_BINDING_SCHEMA,
        "mode": "fresh",
        "seed": expected_seed,
        "phase2_design_sha256": expected_design_sha256,
        "base_config_hash": expected_base_config_hash,
        "artifact_metadata_sha256": expected_metadata_sha256,
        "recovery_artifact_reused": False,
        "recovery_reward_heads_reused": False,
        "recovery_optimizer_state_reused": False,
    }
    if materialization != expected_binding:
        raise ValueError("artifact materialization receipt does not bind this fresh seed artifact")
    return _test_prompt_ids(metadata)


def _validate_prepared_inputs(
    overlay: Path,
    artifact_dir: Path,
    manifest_file: Path,
    *,
    expected_seed: int,
    expected_design_sha256: str,
    expected_base_config_hash: str,
    expected_artifact_metadata_sha256: str,
    expected_manifest_sha256: str,
    expected_environment_identity: Mapping[str, object],
    expected_test_prompt_ids: Sequence[str],
) -> tuple[dict[str, tuple[object, ...]], str, str]:
    """Re-open the complete artifact graph through the production input gate."""

    prepared = prepare_phase2_inputs(
        overlay,
        seed=expected_seed,
        artifact_dir=artifact_dir,
        run_manifest=manifest_file,
        require_formal=True,
        match_current_environment=True,
    )
    observed_prompt_ids = tuple(
        getattr(prompt, "prompt_id", None) for prompt in prepared.test_prompts
    )
    if (
        prepared.seed != expected_seed
        or prepared.phase2_config_hash != expected_design_sha256
        or prepared.source_config_hash != expected_base_config_hash
        or prepared.artifact_metadata_sha256 != expected_artifact_metadata_sha256
        or prepared.run_manifest_sha256 != expected_manifest_sha256
        or dict(prepared.environment_identity) != dict(expected_environment_identity)
        or prepared.artifact_dir.resolve(strict=True) != artifact_dir
        or prepared.run_manifest.resolve(strict=True) != manifest_file
        or observed_prompt_ids != tuple(expected_test_prompt_ids)
    ):
        raise ValueError("production Phase-2 input preparation returned a cross-bound identity")
    raw_semantics = getattr(prepared, "materialization_prompt_semantics", None)
    if not isinstance(raw_semantics, Mapping):
        raise ValueError("production inputs do not expose materialization prompt semantics")
    raw_records = raw_semantics.get("records")
    if isinstance(raw_records, (str, bytes, bytearray)) or not isinstance(raw_records, Sequence):
        raise ValueError("materialization prompt-semantics records are invalid")
    records_by_id: dict[str, Mapping[str, object]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("materialization prompt-semantics record is invalid")
        prompt_id = raw_record.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in records_by_id:
            raise ValueError("materialization prompt-semantics IDs are invalid or duplicated")
        records_by_id[prompt_id] = raw_record

    contracts: dict[str, tuple[object, ...]] = {}
    expected_record_keys = (set(_PROMPT_SEMANTICS_KEYS) - {"schema_version"}) | {"prompt_id"}
    for prompt in prepared.test_prompts:
        prompt_id = getattr(prompt, "prompt_id", None)
        messages = getattr(prompt, "messages", None)
        if (
            not isinstance(prompt_id, str)
            or not isinstance(messages, tuple)
            or len(messages) != 1
            or getattr(messages[0], "role", None) != "user"
            or not isinstance(getattr(messages[0], "content", None), str)
            or not messages[0].content
        ):
            raise ValueError("prepared test prompt does not preserve one raw user message")
        record = records_by_id.get(prompt_id)
        if record is None or set(record) != expected_record_keys:
            raise ValueError("prepared test prompt lacks exact materialization semantics")
        prompt_text = messages[0].content
        raw_prompt_sha256 = _digest(
            record.get("raw_prompt_sha256"),
            name=f"materialization prompt {prompt_id} raw_prompt_sha256",
        )
        prompt_count = _integer(
            record.get("policy_chat_token_count"),
            name=f"materialization prompt {prompt_id} policy_chat_token_count",
            minimum=1,
        )
        token_sha256 = _digest(
            record.get("policy_prompt_token_ids_sha256"),
            name=f"materialization prompt {prompt_id} token SHA256",
        )
        max_prompt_tokens = _integer(
            record.get("max_prompt_tokens"),
            name=f"materialization prompt {prompt_id} max_prompt_tokens",
            minimum=1,
        )
        if (
            record.get("prompt_id") != prompt_id
            or raw_prompt_sha256 != _sha256_bytes(prompt_text.encode("utf-8"))
            or prompt_count > max_prompt_tokens
            or record.get("truncated") is not False
            or record.get("raw_prompt_preserved") is not True
        ):
            raise ValueError("prepared prompt differs from materialization semantics")
        contracts[prompt_id] = (
            prompt_text,
            raw_prompt_sha256,
            prompt_count,
            token_sha256,
            max_prompt_tokens,
            False,
            True,
        )
    if tuple(contracts) != tuple(expected_test_prompt_ids):
        raise ValueError("prepared test prompt semantics differ from artifact test order")
    policy_template_sha256 = _digest(
        getattr(prepared, "policy_chat_template_sha256", None),
        name="prepared policy chat-template SHA256",
    )
    oracle_template_sha256 = _digest(
        getattr(prepared, "oracle_chat_template_sha256", None),
        name="prepared oracle chat-template SHA256",
    )
    if policy_template_sha256 == oracle_template_sha256:
        raise ValueError("policy and operational-oracle chat templates must remain distinct")
    return contracts, policy_template_sha256, oracle_template_sha256


def _validate_oracle_template_closure(
    result: Mapping[str, object],
    *,
    expected_policy_template_sha256: str,
    expected_oracle_template_sha256: str,
) -> None:
    train_rescore = _mapping(
        result.get("train_oracle_rescore"),
        name="result train oracle rescore",
    )
    heldout = _mapping(
        result.get("heldout_fixed_beta"),
        name="result held-out evidence",
    )
    heldout_rescore = _mapping(
        heldout.get("oracle_rescore"),
        name="result held-out oracle rescore",
    )
    boundary = _mapping(
        result.get("information_boundary"),
        name="result information boundary",
    )
    prompt_semantics = _mapping(
        boundary.get("prompt_semantics"),
        name="result prompt-semantics continuity",
    )
    oracle = _mapping(
        prompt_semantics.get("oracle"),
        name="result oracle prompt-semantics continuity",
    )
    if (
        train_rescore.get("oracle_chat_template_sha256") != expected_oracle_template_sha256
        or heldout_rescore.get("oracle_chat_template_sha256") != expected_oracle_template_sha256
        or oracle.get("policy_chat_template_sha256") != expected_policy_template_sha256
        or oracle.get("oracle_chat_template_sha256") != expected_oracle_template_sha256
        or oracle.get("input_text") != "same_raw_prompt_plus_assistant_response"
        or oracle.get("rerendered_with_independent_oracle_chat_template") is not True
        or oracle.get("policy_chat_tokens_reused_by_oracle") is not False
        or oracle.get("policy_and_oracle_chat_template_sha256_distinct") is not True
    ):
        raise ValueError("Qwen policy and operational-oracle template closure is invalid")


def _validate_prompt_semantics(
    row: Mapping[str, object],
    *,
    row_number: int,
) -> tuple[object, ...]:
    semantics = _mapping(
        row.get("prompt_semantics"),
        name=f"rollout row {row_number} prompt_semantics",
    )
    if set(semantics) != _PROMPT_SEMANTICS_KEYS:
        raise ValueError(f"rollout row {row_number} prompt semantics schema is invalid")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"rollout row {row_number} prompt must be non-empty")
    raw_prompt_sha256 = _digest(
        semantics.get("raw_prompt_sha256"),
        name=f"rollout row {row_number} raw_prompt_sha256",
    )
    prompt_count = _integer(
        semantics.get("policy_chat_token_count"),
        name=f"rollout row {row_number} policy_chat_token_count",
        minimum=1,
    )
    max_prompt_tokens = _integer(
        semantics.get("max_prompt_tokens"),
        name=f"rollout row {row_number} max_prompt_tokens",
        minimum=1,
    )
    if (
        semantics.get("schema_version") != _PROMPT_SEMANTICS_SCHEMA
        or raw_prompt_sha256 != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        or prompt_count > max_prompt_tokens
        or semantics.get("truncated") is not False
        or semantics.get("raw_prompt_preserved") is not True
    ):
        raise ValueError(f"rollout row {row_number} does not prove full prompt preservation")
    prompt_token_sha256 = _digest(
        semantics.get("policy_prompt_token_ids_sha256"),
        name=f"rollout row {row_number} policy_prompt_token_ids_sha256",
    )
    return (
        prompt,
        raw_prompt_sha256,
        prompt_count,
        prompt_token_sha256,
        max_prompt_tokens,
        False,
        True,
    )


def _validate_rollouts(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int,
    prompt_ids: Sequence[str],
    prompt_contracts: Mapping[str, tuple[object, ...]],
    candidates_per_prompt: int,
    frozen_beta: float,
) -> tuple[dict[str, int], dict[str, dict[str, object]]]:
    if tuple(prompt_contracts) != tuple(prompt_ids):
        raise ValueError("rollout prompt contracts differ from the artifact test order")
    per_arm = len(prompt_ids) * candidates_per_prompt
    expected_count = per_arm * len(PHASE2_ARM_ORDER)
    if len(records) != expected_count:
        raise ValueError(
            f"rollout row count mismatch: expected {expected_count}, observed {len(records)}"
        )
    cross_arm: dict[tuple[str, int], tuple[object, ...]] = {}
    arm_values: dict[str, dict[str, list[float]]] = {
        arm: {"kl": [], "reward": [], "utility": []} for arm in PHASE2_ARM_ORDER
    }
    arm_lengths: dict[str, list[int]] = {arm: [] for arm in PHASE2_ARM_ORDER}
    arm_eos: dict[str, list[bool]] = {arm: [] for arm in PHASE2_ARM_ORDER}
    arm_max_length: dict[str, list[bool]] = {arm: [] for arm in PHASE2_ARM_ORDER}
    base_rollout_seed = SeedBundle.from_base_seed(
        _integer(seed, name="rollout seed identity", minimum=0)
    ).rollout
    for row_offset, row in enumerate(records):
        row_number = row_offset + 1
        if set(row) != _ROLLOUT_KEYS:
            raise ValueError(f"rollout row {row_number} has missing or extra fields")
        arm_index, within_arm = divmod(row_offset, per_arm)
        prompt_index, candidate_index = divmod(within_arm, candidates_per_prompt)
        expected_arm = PHASE2_ARM_ORDER[arm_index]
        expected_prompt_id = prompt_ids[prompt_index]
        expected_source = (
            "zero_b_reference" if expected_arm == "zero_b" else "direct_common_beta_displacement"
        )
        if (
            row.get("schema_version") != PHASE2_BUDGETED_ROLLOUT_SCHEMA
            or row.get("design_stage") != PHASE2_BUDGETED_END_TO_END_STAGE
            or row.get("evidence_role") != PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE
            or row.get("formal_claim_eligible") is not False
            or row.get("supports_formal_claim") is not False
            or row.get("arm") != expected_arm
            or row.get("policy_source") != expected_source
            or row.get("prompt_id") != expected_prompt_id
            or row.get("candidate_index") != candidate_index
            or _finite(
                row.get("beta_common"),
                name=f"rollout row {row_number} beta_common",
                positive=True,
            )
            != frozen_beta
            or row.get("kl_orientation") != "pi_updated_to_pi0"
            or row.get("kl_history_source") != "updated_policy"
            or row.get("raw_oracle_logit_serialized") is not False
        ):
            raise ValueError(f"rollout row {row_number} identity/order contract is invalid")
        response = row.get("response")
        if not isinstance(response, str):
            raise ValueError(f"rollout row {row_number} response must be a string")
        token_ids = _sequence(row.get("token_ids"), name=f"rollout row {row_number} token_ids")
        mask = _sequence(row.get("response_mask"), name=f"rollout row {row_number} response_mask")
        if (
            not token_ids
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in token_ids
            )
            or len(mask) != len(token_ids)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item not in (0, 1)
                for item in mask
            )
            or not any(mask)
            or row.get("response_token_count") != sum(mask)
        ):
            raise ValueError(f"rollout row {row_number} token/mask evidence is invalid")
        active = [index for index, item in enumerate(mask) if item]
        if active != list(range(active[0], active[-1] + 1)):
            raise ValueError(f"rollout row {row_number} response mask is not contiguous")
        semantics_identity = _validate_prompt_semantics(row, row_number=row_number)
        if semantics_identity != prompt_contracts[expected_prompt_id]:
            raise ValueError(
                f"rollout row {row_number} differs from materialization prompt semantics"
            )
        prompt_count = int(semantics_identity[2])
        encoded_prompt_ids = json.dumps(
            token_ids[:prompt_count],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if (
            active[0] != prompt_count
            or _sha256_bytes(encoded_prompt_ids) != semantics_identity[3]
            or not isinstance(row.get("terminated_by_eos"), bool)
            or not isinstance(row.get("reached_max_length"), bool)
            or (row.get("terminated_by_eos") is True and row.get("reached_max_length") is True)
        ):
            raise ValueError(f"rollout row {row_number} generation boundary is invalid")
        prompt_seed = _integer(
            row.get("prompt_rollout_seed"),
            name=f"rollout row {row_number} prompt_rollout_seed",
            minimum=0,
        )
        expected_prompt_seed = derive_seed(
            base_rollout_seed,
            f"phase2-test-prompt:{expected_prompt_id}",
        )
        if prompt_seed != expected_prompt_seed:
            raise ValueError(f"rollout row {row_number} has a cross-seed prompt rollout seed")
        kl = _finite(
            row.get("on_policy_kl_pi_updated_to_pi0"),
            name=f"rollout row {row_number} on-policy KL",
        )
        if kl < 0.0:
            raise ValueError(f"rollout row {row_number} on-policy KL is negative")
        reward = _finite(
            row.get("transformed_oracle_reward"),
            name=f"rollout row {row_number} transformed oracle reward",
        )
        utility = _finite(
            row.get("target_utility"),
            name=f"rollout row {row_number} target utility",
        )
        if not math.isclose(
            utility,
            reward - frozen_beta * kl,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"rollout row {row_number} target utility is inconsistent")
        arm_values[expected_arm]["kl"].append(kl)
        arm_values[expected_arm]["reward"].append(reward)
        arm_values[expected_arm]["utility"].append(utility)
        arm_lengths[expected_arm].append(int(row["response_token_count"]))
        arm_eos[expected_arm].append(bool(row["terminated_by_eos"]))
        arm_max_length[expected_arm].append(bool(row["reached_max_length"]))
        identity = (
            expected_prompt_id,
            candidate_index,
            *semantics_identity,
            prompt_seed,
        )
        key = (expected_prompt_id, candidate_index)
        if key in cross_arm and cross_arm[key] != identity:
            raise ValueError(f"rollout row {row_number} breaks cross-arm common random numbers")
        cross_arm[key] = identity

    def linear_quantile(values: Sequence[float], quantile: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    summaries: dict[str, dict[str, object]] = {}
    for arm, metrics in arm_values.items():
        kl_values = metrics["kl"]
        prompt_means = [
            math.fsum(
                kl_values[
                    prompt_index * candidates_per_prompt : (prompt_index + 1)
                    * candidates_per_prompt
                ]
            )
            / candidates_per_prompt
            for prompt_index in range(len(prompt_ids))
        ]
        lengths = arm_lengths[arm]
        eos = arm_eos[arm]
        reached = arm_max_length[arm]
        summaries[arm] = {
            **{metric: math.fsum(values) / len(values) for metric, values in metrics.items()},
            "kl_tail": {
                "num_prompts": len(prompt_ids),
                "candidates_per_prompt": candidates_per_prompt,
                "mean": math.fsum(prompt_means) / len(prompt_means),
                "p50": linear_quantile(prompt_means, 0.50),
                "p90": linear_quantile(prompt_means, 0.90),
                "p95": linear_quantile(prompt_means, 0.95),
                "p99": linear_quantile(prompt_means, 0.99),
                "maximum": max(prompt_means),
                "per_sequence_maximum": max(kl_values),
            },
            "length": {
                "num_trajectories": len(lengths),
                "terminated_by_eos_count": sum(eos),
                "terminated_by_eos_rate": sum(eos) / len(eos),
                "reached_max_length_count": sum(reached),
                "reached_max_length_rate": sum(reached) / len(reached),
                "response_token_count_mean": math.fsum(lengths) / len(lengths),
                "response_token_count_minimum": min(lengths),
                "response_token_count_maximum": max(lengths),
            },
        }
    return (
        {
            "row_count": expected_count,
            "rows_per_arm": per_arm,
            "test_prompt_count": len(prompt_ids),
            "candidates_per_prompt": candidates_per_prompt,
        },
        summaries,
    )


def _validate_rollout_result_summaries(
    result: Mapping[str, object],
    summaries: Mapping[str, Mapping[str, object]],
) -> None:
    arms = _mapping(result.get("arms"), name="result arms")
    expected_arms = set(PHASE2_ARM_ORDER)
    if set(arms) != expected_arms or set(summaries) != expected_arms:
        raise ValueError("result and rollout sidecar arm sets differ")
    for arm_name in PHASE2_ARM_ORDER:
        arm = _mapping(arms.get(arm_name), name=f"result arm {arm_name}")
        utility = _mapping(
            arm.get("utility"),
            name=f"result arm {arm_name} utility",
        )
        expected = summaries[arm_name]
        observed = {
            "outer_kl": _finite(
                arm.get("mean_on_policy_kl_pi_updated_to_pi0"),
                name=f"result arm {arm_name} mean KL",
            ),
            "utility_kl": _finite(
                utility.get("mean_on_policy_kl_pi_updated_to_pi0"),
                name=f"result arm {arm_name} utility mean KL",
            ),
            "reward": _finite(
                utility.get("mean_target_reward"),
                name=f"result arm {arm_name} mean target reward",
            ),
            "utility": _finite(
                utility.get("mean_target_utility"),
                name=f"result arm {arm_name} mean target utility",
            ),
        }
        expected_values = {
            "outer_kl": expected["kl"],
            "utility_kl": expected["kl"],
            "reward": expected["reward"],
            "utility": expected["utility"],
        }
        if any(
            not math.isclose(
                observed[name],
                expected_value,
                rel_tol=1.0e-6,
                abs_tol=1.0e-7,
            )
            for name, expected_value in expected_values.items()
        ):
            raise ValueError(f"result arm {arm_name} summaries differ from the rollout sidecar")
        expected_tail = _mapping(
            expected["kl_tail"],
            name=f"computed arm {arm_name} KL tail",
        )
        observed_tail = _mapping(
            arm.get("on_policy_kl_tail"),
            name=f"result arm {arm_name} KL tail",
        )
        if (
            observed_tail.get("schema_version") != "on-policy-kl-tail-summary/v1"
            or observed_tail.get("unit") != "prompt_mean_over_candidates"
            or observed_tail.get("num_prompts") != expected_tail["num_prompts"]
            or observed_tail.get("candidates_per_prompt") != expected_tail["candidates_per_prompt"]
            or observed_tail.get("pilot_selection_role") != "locality_tail_measurement"
            or observed_tail.get("formal_gate_applied") is not False
        ):
            raise ValueError(f"result arm {arm_name} KL-tail identity is invalid")
        for metric in (
            "mean",
            "p50",
            "p90",
            "p95",
            "p99",
            "maximum",
            "per_sequence_maximum",
        ):
            if not math.isclose(
                _finite(
                    observed_tail.get(metric),
                    name=f"result arm {arm_name} KL tail {metric}",
                ),
                float(expected_tail[metric]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"result arm {arm_name} KL tail differs from the rollout sidecar")

        expected_length = _mapping(
            expected["length"],
            name=f"computed arm {arm_name} length summary",
        )
        observed_length = _mapping(
            arm.get("rollout"),
            name=f"result arm {arm_name} rollout summary",
        )
        if any(
            observed_length.get(key) != expected_length[key]
            for key in (
                "num_trajectories",
                "terminated_by_eos_count",
                "reached_max_length_count",
            )
        ) or any(
            not math.isclose(
                _finite(
                    observed_length.get(key),
                    name=f"result arm {arm_name} rollout {key}",
                ),
                float(expected_length[key]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            for key in ("terminated_by_eos_rate", "reached_max_length_rate")
        ):
            raise ValueError(
                f"result arm {arm_name} length summary differs from the rollout sidecar"
            )
        observed_response_tokens = _mapping(
            observed_length.get("response_token_count"),
            name=f"result arm {arm_name} response-token summary",
        )
        expected_response_tokens = {
            "mean": expected_length["response_token_count_mean"],
            "minimum": expected_length["response_token_count_minimum"],
            "maximum": expected_length["response_token_count_maximum"],
        }
        for metric, expected_value in expected_response_tokens.items():
            if not math.isclose(
                _finite(
                    observed_response_tokens.get(metric),
                    name=f"result arm {arm_name} response tokens {metric}",
                ),
                float(expected_value),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"result arm {arm_name} response-token summary differs from sidecar"
                )


def _validate_normalized(
    normalized: object,
    *,
    expected_seed: int,
    expected_design_sha256: str,
    expected_runtime_sha256: str,
    expected_freeze_sha256: str,
    expected_frozen_beta: float,
) -> dict[str, Any]:
    value = _mapping(normalized, name="normalized budgeted seed result")
    endpoints = _mapping(value.get("endpoints"), name="normalized endpoints")
    if (
        value.get("admissible") is not True
        or value.get("seed") != expected_seed
        or value.get("phase2_design_sha256") != expected_design_sha256
        or value.get("phase2_runtime_contract_sha256") != expected_runtime_sha256
        or value.get("beta_source_aggregate_sha256") != expected_freeze_sha256
        or _finite(
            value.get("frozen_global_beta"),
            name="normalized frozen_global_beta",
            positive=True,
        )
        != expected_frozen_beta
        or set(endpoints) != _ENDPOINT_KEYS
    ):
        raise ValueError("budgeted normalizer rejected the seed result")
    for endpoint, learner_values in endpoints.items():
        values = _mapping(learner_values, name=f"normalized endpoint {endpoint}")
        if set(values) != _LEARNERS:
            raise ValueError(f"normalized endpoint {endpoint} has invalid learners")
        for learner, metric in values.items():
            _finite(metric, name=f"normalized endpoint {endpoint}.{learner}")
    return value


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_write(path: Path, payload: Mapping[str, object]) -> None:
    destination = path.absolute()
    parent = _canonical_directory(destination.parent, name="verification output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite verification output: {destination}")
    raw = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partially created path is intentionally left occupied.  A retry
        # must be explicit and must never mistake a partial publication for an
        # absent target.
        raise
    try:
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
    except PermissionError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def verify_seed_output(
    overlay_path: Path,
    result_path: Path,
    rollouts_path: Path,
    output_path: Path,
    *,
    seed: int,
    design_sha256: str,
    base_config_hash: str,
    git_commit: str,
    image_sha256: str,
    hf_inventory_sha256: str,
    artifact_metadata_sha256: str,
    freeze_evidence_sha256: str,
    slurm_job_id_raw: str,
    array_job_id: str,
    array_task_id: int,
) -> dict[str, object]:
    """Verify and exclusively publish one non-formal seed record."""

    expected_seed = _integer(seed, name="seed", minimum=0)
    if expected_seed not in PHASE2_BUDGETED_END_TO_END_SEEDS:
        raise ValueError("seed is outside the fixed-three budgeted design")
    expected_design = _digest(design_sha256, name="design_sha256")
    expected_base = _digest(base_config_hash, name="base_config_hash")
    expected_git = _digest(
        git_commit,
        name="git_commit",
        lengths=frozenset({40, 64}),
    )
    expected_image = _digest(image_sha256, name="image_sha256")
    expected_inventory = _digest(hf_inventory_sha256, name="hf_inventory_sha256")
    expected_artifact = _digest(
        artifact_metadata_sha256,
        name="artifact_metadata_sha256",
    )
    expected_freeze = _digest(freeze_evidence_sha256, name="freeze_evidence_sha256")
    task_id = _integer(array_task_id, name="array_task_id", minimum=0)
    for name, value in (
        ("slurm_job_id_raw", slurm_job_id_raw),
        ("array_job_id", array_job_id),
    ):
        if not isinstance(value, str) or not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"{name} must be a positive decimal Slurm ID")

    overlay = _canonical_file(overlay_path, name="budgeted overlay")
    result_file = _canonical_file(result_path, name="phase2 result")
    rollouts_file = _canonical_file(rollouts_path, name="phase2 rollout JSONL")
    if result_file.name != _RESULT_FILENAME or rollouts_file.name != _ROLLOUT_FILENAME:
        raise ValueError("result and rollout filenames differ from the locked sbatch contract")
    if result_file.parent != rollouts_file.parent:
        raise ValueError("result and rollout sidecar must share one canonical job directory")
    job_dir = result_file.parent
    expected_output = job_dir / _VERIFICATION_FILENAME
    if output_path.absolute() != expected_output:
        raise ValueError(
            "verification output must use the locked filename in the canonical job directory"
        )
    manifest_file = _canonical_file(job_dir / _MANIFEST_FILENAME, name="run manifest")
    materialization_file = _canonical_file(
        job_dir / _ARTIFACT_BINDING_FILENAME,
        name="artifact materialization receipt",
    )
    artifact_metadata_file = _canonical_file(
        job_dir / _ARTIFACT_DIRECTORY / _ARTIFACT_METADATA_FILENAME,
        name="artifact metadata",
    )

    result, _result_raw, result_sha256 = _strict_json(result_file, name="phase2 result")
    manifest, _manifest_raw, manifest_sha256 = _strict_json(
        manifest_file,
        name="run manifest",
    )
    artifact, _artifact_raw, observed_artifact_sha256 = _strict_json(
        artifact_metadata_file,
        name="artifact metadata",
    )
    materialization, _materialization_raw, materialization_sha256 = _strict_json(
        materialization_file,
        name="artifact materialization receipt",
    )
    if observed_artifact_sha256 != expected_artifact:
        raise ValueError("artifact metadata bytes differ from the expected SHA256")

    if (
        _relative_file_reference(result.get("rollouts_jsonl"), name="result rollouts_jsonl")
        != _ROLLOUT_FILENAME
        or _relative_file_reference(result.get("run_manifest"), name="result run_manifest")
        != _MANIFEST_FILENAME
        or _relative_directory_reference(result.get("artifact_dir"), name="result artifact_dir")
        != _ARTIFACT_DIRECTORY
        or result.get("rollouts_sha256") is None
        or result.get("run_manifest_sha256") != manifest_sha256
        or result.get("artifact_metadata_sha256") != observed_artifact_sha256
    ):
        raise ValueError("result relative filenames or input hashes are inconsistent")

    frozen_beta = _require_result_root(
        result,
        expected_seed=expected_seed,
        expected_design_sha256=expected_design,
        expected_base_config_hash=expected_base,
        expected_freeze_sha256=expected_freeze,
    )
    config, base, overlay_sha256, candidates_per_prompt = _validate_overlay(
        overlay,
        expected_seed=expected_seed,
        expected_design_sha256=expected_design,
        expected_base_config_hash=expected_base,
        expected_freeze_sha256=expected_freeze,
        frozen_beta=frozen_beta,
    )
    expected_runtime = Phase2Design.from_phase2_config(config)
    if (
        result.get("phase2_runtime_contract") != expected_runtime.to_dict()
        or result.get("phase2_runtime_contract_sha256") != expected_runtime.sha256
    ):
        raise ValueError("result runtime contract differs from the validated overlay")
    environment_identity = _validate_manifest(
        manifest,
        expected_seed=expected_seed,
        expected_base_config_hash=expected_base,
        expected_base_config=base,
        expected_git_commit=expected_git,
        expected_image_sha256=expected_image,
        expected_inventory_sha256=expected_inventory,
        expected_slurm_job_id_raw=slurm_job_id_raw,
        expected_array_job_id=array_job_id,
        expected_array_task_id=task_id,
    )
    if (
        result.get("environment_identity") != environment_identity
        or result.get("current_process_identity") != environment_identity
    ):
        raise ValueError("result environment/current-process identity differs from its manifest")
    prompt_ids = _validate_artifact(
        artifact,
        materialization,
        expected_seed=expected_seed,
        expected_design_sha256=expected_design,
        expected_base_config_hash=expected_base,
        expected_git_commit=expected_git,
        expected_image_sha256=expected_image,
        expected_inventory_sha256=expected_inventory,
        expected_metadata_sha256=observed_artifact_sha256,
    )
    (
        prompt_contracts,
        policy_template_sha256,
        oracle_template_sha256,
    ) = _validate_prepared_inputs(
        overlay,
        artifact_metadata_file.parent,
        manifest_file,
        expected_seed=expected_seed,
        expected_design_sha256=expected_design,
        expected_base_config_hash=expected_base,
        expected_artifact_metadata_sha256=observed_artifact_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_environment_identity=environment_identity,
        expected_test_prompt_ids=prompt_ids,
    )
    _validate_oracle_template_closure(
        result,
        expected_policy_template_sha256=policy_template_sha256,
        expected_oracle_template_sha256=oracle_template_sha256,
    )
    if (
        _sha256_file(manifest_file) != manifest_sha256
        or _sha256_file(artifact_metadata_file) != observed_artifact_sha256
    ):
        raise RuntimeError("manifest or artifact metadata changed during complete input validation")
    records, rollouts_sha256 = _strict_jsonl(rollouts_file)
    if result.get("rollouts_sha256") != rollouts_sha256:
        raise ValueError("rollout bytes differ from result.rollouts_sha256")
    geometry, rollout_summaries = _validate_rollouts(
        records,
        seed=expected_seed,
        prompt_ids=prompt_ids,
        prompt_contracts=prompt_contracts,
        candidates_per_prompt=candidates_per_prompt,
        frozen_beta=frozen_beta,
    )
    _validate_rollout_result_summaries(result, rollout_summaries)
    normalized = _validate_normalized(
        normalize_budgeted_end_to_end_seed_result(result),
        expected_seed=expected_seed,
        expected_design_sha256=expected_design,
        expected_runtime_sha256=str(result["phase2_runtime_contract_sha256"]),
        expected_freeze_sha256=expected_freeze,
        expected_frozen_beta=frozen_beta,
    )

    verification: dict[str, object] = {
        "schema_version": _VERIFICATION_SCHEMA,
        "status": "verified",
        "design_stage": PHASE2_BUDGETED_END_TO_END_STAGE,
        "evidence_role": PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
        "formal_eligibility": False,
        "formal_claim_eligible": False,
        "supports_formal_claim": False,
        "inferential_or_significance_claim_produced": False,
        "seed": expected_seed,
        "phase2_design_sha256": expected_design,
        "base_config_hash": expected_base,
        "accepted_freeze_aggregate_sha256": expected_freeze,
        "frozen_global_beta": frozen_beta,
        "phase2_runtime_contract_sha256": result["phase2_runtime_contract_sha256"],
        "git_commit": expected_git,
        "image_sha256": expected_image,
        "hf_inventory_sha256": expected_inventory,
        "slurm_job_id_raw": slurm_job_id_raw,
        "array_job_id": array_job_id,
        "array_task_id": task_id,
        "result_sha256": result_sha256,
        "rollouts_sha256": rollouts_sha256,
        "run_manifest_sha256": manifest_sha256,
        "artifact_metadata_sha256": observed_artifact_sha256,
        "artifact_materialization_sha256": materialization_sha256,
        "slurm": {
            "job_id_raw": slurm_job_id_raw,
            "array_job_id": array_job_id,
            "array_task_id": task_id,
            "account": "sigroup",
            "cluster": "hpc4",
            "partition": "gpu-l20",
        },
        "relative_files": {
            "result": _RESULT_FILENAME,
            "rollouts": _ROLLOUT_FILENAME,
            "run_manifest": _MANIFEST_FILENAME,
            "artifact_metadata": (f"{_ARTIFACT_DIRECTORY}/{_ARTIFACT_METADATA_FILENAME}"),
            "artifact_materialization": _ARTIFACT_BINDING_FILENAME,
        },
        "input_sha256": {
            "overlay": overlay_sha256,
            "result": result_sha256,
            "rollouts": rollouts_sha256,
            "run_manifest": manifest_sha256,
            "artifact_metadata": observed_artifact_sha256,
            "artifact_materialization": materialization_sha256,
        },
        "rollout_geometry": {
            **geometry,
            "arm_order": list(PHASE2_ARM_ORDER),
        },
        "environment_identity": environment_identity,
        "normalized_seed_record": normalized,
    }
    _exclusive_write(output_path, verification)
    return verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--base-config-hash", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--hf-inventory-sha256", required=True)
    parser.add_argument("--artifact-metadata-sha256", required=True)
    parser.add_argument("--freeze-evidence-sha256", required=True)
    parser.add_argument("--slurm-job-id-raw", required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--array-task-id", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    verification = verify_seed_output(
        arguments.overlay,
        arguments.result,
        arguments.rollouts,
        arguments.output,
        seed=arguments.seed,
        design_sha256=arguments.design_sha256,
        base_config_hash=arguments.base_config_hash,
        git_commit=arguments.git_commit,
        image_sha256=arguments.image_sha256,
        hf_inventory_sha256=arguments.hf_inventory_sha256,
        artifact_metadata_sha256=arguments.artifact_metadata_sha256,
        freeze_evidence_sha256=arguments.freeze_evidence_sha256,
        slurm_job_id_raw=arguments.slurm_job_id_raw,
        array_job_id=arguments.array_job_id,
        array_task_id=arguments.array_task_id,
    )
    print(
        json.dumps(
            {
                "schema_version": _VERIFICATION_SCHEMA,
                "status": verification["status"],
                "formal_claim_eligible": False,
                "seed": verification["seed"],
                "output": os.fspath(arguments.output),
                "output_sha256": _sha256_file(arguments.output.absolute()),
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
