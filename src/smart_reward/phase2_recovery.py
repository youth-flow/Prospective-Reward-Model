"""One-shot, train-only Phase-2 recovery on immutable parent artifacts.

The recovery process deliberately does not call ``prepare_phase2_inputs``:
that general runner verifies and reconstructs held-out tensors for later
rollouts.  Here only the five reward-free train tensors and the canonical
train-candidate prefix are decoded.  The policy runtime is never opened.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .cli import _run_environment_identity
from .config import config_hash
from .data import CandidateNode
from .experiment import TrainingTensorData
from .oracle import RobustOracleTransform
from .phase2_config import Phase2ConfigBundle, load_phase2_config_bundle
from .phase2_hf import HuggingFacePhase2Backend
from .phase2_training import OptimizationConvergenceError, train_phase2_heads

RECOVERY_RESULT_SCHEMA = "prorm-phase2-recovery-train-only-result/v1"
RECOVERY_FAILURE_SCHEMA = "prorm-phase2-recovery-train-only-failure/v1"
PARENT_REGISTRY_SCHEMA = "prorm-phase2-recovery-parent-failures/v1"
TRAIN_TENSOR_KEYS = (
    "train.policy_scores",
    "train.reward_features",
    "train.h",
    "train.left_wins",
    "train.num_annotations",
)
HEX = frozenset("0123456789abcdef")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous().clone()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def _digest(value: object, name: str, *, lengths: frozenset[int] = frozenset({64})) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"expected a regular non-symlink JSON file: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _recovery_control(bundle: Phase2ConfigBundle) -> dict[str, Any]:
    value = bundle.config.get("recovery_control")
    if not isinstance(value, dict):
        raise ValueError("recovery overlay lacks recovery_control")
    required = {
        "schema_version": "prorm-phase2-recovery-control/v1",
        "parent_terminal_status": "FAILED",
        "parent_failure_aggregate_present": False,
        "artifact_reuse": "immutable_parent_materialization_only",
        "artifact_producer_identity_separate_from_recovery_training_identity": True,
        "execution_scope": "train_only",
        "policy_rollout_allowed": False,
        "validation_or_test_access_allowed": False,
        "final_oracle_allowed": False,
        "downstream_utility_allowed": False,
        "one_shot_no_further_adaptation": True,
        "failure_action": "hard_fail_no_second_recovery",
        "optimizer_diagnostic_role": "train_only_nonconfirmatory_schedule_selection",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"recovery_control.{key} violates the train-only contract")
    return value


def _load_registry_entry(
    registry_path: Path,
    *,
    bundle: Phase2ConfigBundle,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    control = _recovery_control(bundle)
    registry_sha = _sha256_file(registry_path)
    if registry_sha != _digest(
        control.get("parent_failure_registry_sha256"),
        "recovery_control.parent_failure_registry_sha256",
    ):
        raise ValueError("parent failure registry SHA256 changed")
    registry = _read_json(registry_path)
    if (
        set(registry)
        != {
            "schema_version",
            "campaign",
            "common_artifact_identities",
            "optimizer_diagnostic",
            "seeds",
        }
        or registry.get("schema_version") != PARENT_REGISTRY_SCHEMA
    ):
        raise ValueError("parent failure registry schema is invalid")
    campaign = registry.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("parent failure registry campaign is invalid")
    if (
        campaign.get("parent_phase2_design_sha256") != control.get("parent_phase2_design_sha256")
        or campaign.get("base_config_hash") != config_hash(bundle.base_config)
        or campaign.get("source_job_array_id") != control.get("parent_source_job_array_id")
        or campaign.get("one_shot_no_further_adaptation") is not True
        or campaign.get("allowed_recovery_scope") != "train_only_same_materialized_artifacts"
    ):
        raise ValueError("parent registry campaign differs from the recovery overlay")
    raw_entries = registry.get("seeds")
    if not isinstance(raw_entries, list):
        raise ValueError("parent failure registry seeds must be a list")
    if [entry.get("seed") for entry in raw_entries if isinstance(entry, dict)] != list(
        control.get("parent_seeds", [])
    ):
        raise ValueError("parent registry seeds differ from the recovery overlay")
    for entry in raw_entries:
        if isinstance(entry, dict) and entry.get("seed") == seed:
            diagnostic = registry.get("optimizer_diagnostic")
            if not isinstance(diagnostic, dict):
                raise ValueError("parent registry optimizer diagnostic is invalid")
            if (
                diagnostic.get("sha256") != control.get("optimizer_diagnostic_sha256")
                or diagnostic.get("path") != control.get("optimizer_diagnostic_path")
                or diagnostic.get("source_git_commit")
                != control.get("optimizer_diagnostic_source_git_commit")
                or diagnostic.get("source_job_id")
                != control.get("optimizer_diagnostic_source_job_id")
            ):
                raise ValueError("optimizer diagnostic differs from the recovery overlay")
            return campaign, entry, diagnostic, registry_sha
    raise ValueError("selected seed is absent from the parent failure registry")


def _validate_metadata_and_specs(
    artifact_dir: Path,
    *,
    campaign: Mapping[str, object],
    entry: Mapping[str, object],
    seed: int,
) -> tuple[dict[str, Any], dict[str, tuple[tuple[int, ...], str]]]:
    hashes = entry.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("registry artifact hashes are invalid")
    metadata_path = artifact_dir / "metadata.json"
    if _sha256_file(metadata_path) != hashes.get("metadata.json"):
        raise ValueError("artifact metadata SHA256 differs from the parent registry")
    metadata = _read_json(metadata_path)
    if (
        metadata.get("schema") != "controlled-feature-artifact/v1"
        or metadata.get("config_hash") != campaign.get("base_config_hash")
        or metadata.get("seed") != seed
        or metadata.get("tensor_sha256") != hashes.get("tensors.safetensors")
    ):
        raise ValueError("artifact metadata identity differs from the parent registry")
    evidence = metadata.get("evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema") != "phase1-materialization/v1"
        or evidence.get("producer") != campaign.get("producer")
        or evidence.get("jsonl_sha256", {}).get("candidates.jsonl")
        != hashes.get("candidates.jsonl")
    ):
        raise ValueError("artifact evidence producer/candidate identity is invalid")
    raw_specs = metadata.get("tensors")
    if not isinstance(raw_specs, dict):
        raise ValueError("artifact tensor metadata is invalid")
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for key in TRAIN_TENSOR_KEYS:
        raw = raw_specs.get(key)
        if (
            not isinstance(raw, dict)
            or set(raw) != {"shape", "dtype"}
            or not isinstance(raw.get("shape"), list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in raw["shape"]
            )
            or not isinstance(raw.get("dtype"), str)
        ):
            raise ValueError(f"invalid train tensor metadata for {key}")
        specs[key] = (tuple(raw["shape"]), raw["dtype"])
    return metadata, specs


def _load_train_tensors_only(
    artifact_dir: Path,
    *,
    metadata: Mapping[str, object],
    specs: Mapping[str, tuple[tuple[int, ...], str]],
    entry: Mapping[str, object],
) -> TrainingTensorData:
    """Decode only the five train keys; held-out tensor values remain unopened."""

    hashes = entry["artifact_sha256"]
    tensor_path = artifact_dir / "tensors.safetensors"
    if _sha256_file(tensor_path) != hashes["tensors.safetensors"]:
        raise ValueError("artifact tensor-file SHA256 differs from the parent registry")
    try:
        from safetensors import safe_open
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError("recovery requires safetensors") from error
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        for key in TRAIN_TENSOR_KEYS:
            tensor = handle.get_tensor(key)
            expected_shape, expected_dtype = specs[key]
            actual_dtype = str(tensor.dtype).removeprefix("torch.")
            if tuple(tensor.shape) != expected_shape or actual_dtype != expected_dtype:
                raise ValueError(f"train tensor payload differs from metadata for {key}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"train tensor {key} contains NaN or infinity")
            tensors[key] = tensor.detach().contiguous()
    splits = metadata.get("splits")
    train_split = splits.get("train") if isinstance(splits, dict) else None
    prompt_ids = train_split.get("prompt_ids") if isinstance(train_split, dict) else None
    if not isinstance(prompt_ids, list):
        raise ValueError("artifact train prompt IDs are invalid")
    return TrainingTensorData(
        prompt_ids=tuple(prompt_ids),
        policy_scores=tensors["train.policy_scores"],
        reward_features=tensors["train.reward_features"],
        h=tensors["train.h"],
        left_wins=tensors["train.left_wins"],
        num_annotations=tensors["train.num_annotations"],
    )


def _load_train_candidate_prefix_only(
    artifact_dir: Path,
    *,
    train: TrainingTensorData,
    entry: Mapping[str, object],
) -> tuple[CandidateNode, ...]:
    """Parse exactly P*M leading records and never decode the held-out suffix."""

    path = artifact_dir / "candidates.jsonl"
    expected = entry["artifact_sha256"]["candidates.jsonl"]
    if _sha256_file(path) != expected:
        raise ValueError("candidate JSONL SHA256 differs from the parent registry")
    count = train.num_prompts * train.num_candidates
    records: list[CandidateNode] = []
    # buffering=0 prevents a training-prefix readline from prefetching held-out
    # candidate bytes into a user-space buffered stream.
    with path.open("rb", buffering=0) as stream:
        for index in range(count):
            raw = stream.readline()
            if not raw or not raw.endswith(b"\n"):
                raise ValueError("candidate JSONL ended within the train prefix")
            try:
                value = json.loads(
                    raw.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_pairs,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid train candidate JSONL record {index}") from error
            records.append(CandidateNode.from_dict(value))
    for flat_index, candidate in enumerate(records):
        prompt_index, candidate_index = divmod(flat_index, train.num_candidates)
        prompt_id = train.prompt_ids[prompt_index]
        if not isinstance(prompt_id, str):
            raise ValueError("controlled recovery requires string train prompt IDs")
        if (
            candidate.prompt_id != prompt_id
            or candidate.candidate_id != f"{prompt_id}::candidate::{candidate_index}"
        ):
            raise ValueError("train candidate prefix is not in canonical tensor order")
    return tuple(records)


def _oracle_contract(metadata: Mapping[str, object]) -> tuple[str, RobustOracleTransform]:
    evidence = metadata.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("artifact evidence is invalid")
    template = _digest(
        evidence.get("oracle_chat_template_sha256"),
        "oracle chat template SHA256",
    )
    raw_transform = evidence.get("oracle_transform")
    if not isinstance(raw_transform, Mapping) or set(raw_transform) != {"b", "tau"}:
        raise ValueError("artifact oracle transform is invalid")
    return template, RobustOracleTransform(
        b=raw_transform["b"],
        tau=raw_transform["tau"],
    )


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite recovery evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_phase2_recovery_train_only(
    overlay: str | os.PathLike[str],
    *,
    registry: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    current_run_manifest: str | os.PathLike[str],
    output_json: str | os.PathLike[str],
    seed: int,
    device: str = "cuda",
) -> dict[str, object]:
    """Execute the authorized one-shot recovery without held-out or policy access."""

    bundle = load_phase2_config_bundle(overlay)
    control = _recovery_control(bundle)
    if seed not in bundle.config["run"]["seeds"]:
        raise ValueError("seed is not declared by the recovery overlay")
    source_hash = config_hash(bundle.base_config)
    if source_hash != bundle.config["design"]["source_config_hash"]:
        raise ValueError("recovery overlay base identity changed")
    registry_path = Path(registry).resolve(strict=True)
    campaign, entry, registry_diagnostic, registry_sha = _load_registry_entry(
        registry_path,
        bundle=bundle,
        seed=seed,
    )
    if bundle.design_identity == campaign["parent_phase2_design_sha256"]:
        raise ValueError("recovery design identity must differ from its failed parent")
    manifest_sha, current_identity = _run_environment_identity(
        current_run_manifest,
        expected_config_hash=source_hash,
        expected_seed=seed,
        require_formal=True,
        match_current_environment=True,
    )
    producer = campaign.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("parent producer identity is invalid")
    if current_identity["git_commit"] == producer.get("git_commit"):
        raise ValueError("recovery training commit must be distinct from the parent producer")

    artifact = Path(artifact_dir).resolve(strict=True)
    metadata, specs = _validate_metadata_and_specs(
        artifact,
        campaign=campaign,
        entry=entry,
        seed=seed,
    )
    train_cpu = _load_train_tensors_only(
        artifact,
        metadata=metadata,
        specs=specs,
        entry=entry,
    )
    candidates = _load_train_candidate_prefix_only(
        artifact,
        train=train_cpu,
        entry=entry,
    )
    oracle_chat_sha, transform = _oracle_contract(metadata)
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal recovery training requires one allocated CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("formal recovery training requires exactly one visible GPU")
    backend = HuggingFacePhase2Backend(
        bundle.base_config,
        device=target_device,
        local_files_only=True,
    )
    prompts = tuple(candidate.prompt for candidate in candidates)
    responses = tuple(candidate.response for candidate in candidates)
    batch_size = min(16, int(bundle.base_config["reward_model"]["microbatch_size"]))
    with backend.oracle_session(expected_chat_template_sha256=oracle_chat_sha) as oracle:
        flat_rewards = oracle.score_transformed(
            prompts,
            responses,
            transform=transform,
            batch_size=batch_size,
        )
    expected_shape = (train_cpu.num_prompts * train_cpu.num_candidates,)
    if (
        not isinstance(flat_rewards, torch.Tensor)
        or tuple(flat_rewards.shape) != expected_shape
        or not flat_rewards.is_floating_point()
        or flat_rewards.requires_grad
        or not bool(torch.isfinite(flat_rewards).all())
    ):
        raise ValueError("train-only oracle rescore returned malformed values")
    train = train_cpu.to(target_device)
    train_rewards = (
        flat_rewards.detach()
        .to(device=target_device, dtype=train.policy_scores.dtype)
        .reshape(train.num_prompts, train.num_candidates)
        .clone()
    )
    result = train_phase2_heads(train, train_rewards, seed=seed, settings=bundle)
    payload: dict[str, object] = {
        "schema_version": RECOVERY_RESULT_SCHEMA,
        "status": "SUCCESS",
        "design_stage": "pilot",
        "evidence_role": "one_shot_optimizer_recovery_train_only",
        "formal_eligibility": False,
        "per_seed_supports_formal_claim": False,
        "seed": seed,
        "source_config_hash": source_hash,
        "recovery_design_sha256": bundle.design_identity,
        "recovery_execution_identity": current_identity,
        "recovery_run_manifest_sha256": manifest_sha,
        "parent_failure_binding": {
            "registry_sha256": registry_sha,
            "parent_phase2_design_sha256": campaign["parent_phase2_design_sha256"],
            "parent_source_job_array_id": campaign["source_job_array_id"],
            "parent_seed_entry": entry,
            "parent_artifact_producer": producer,
            "parent_failure_aggregate_present": False,
            "exact_three_seed_failure_registry_used": True,
            "optimizer_diagnostic": registry_diagnostic,
        },
        "artifact_reuse": {
            "mode": "immutable_parent_materialization_only",
            "metadata_sha256": entry["artifact_sha256"]["metadata.json"],
            "tensor_file_sha256": entry["artifact_sha256"]["tensors.safetensors"],
            "candidate_file_sha256": entry["artifact_sha256"]["candidates.jsonl"],
            "producer_identity_separate_from_recovery_training_identity": True,
            "materialized_or_mutated_by_recovery": False,
        },
        "train_oracle_rescore": {
            "source": "saved_train_candidate_prefix_only",
            "num_prompts": train.num_prompts,
            "num_candidates": train.num_candidates,
            "transformed_rewards_sha256": _tensor_sha256(train_rewards),
            "oracle_chat_template_sha256": oracle_chat_sha,
            "frozen_transform": {"b": transform.b, "tau": transform.tau},
            "raw_oracle_logits_serialized": False,
        },
        "head_training": result.to_dict(),
        "information_boundary": {
            "train_tensors_decoded": True,
            "train_candidate_prefix_decoded": True,
            "validation_tensors_decoded": False,
            "test_tensors_decoded": False,
            "validation_or_test_candidates_decoded": False,
            "policy_session_opened": False,
            "policy_rollout_performed": False,
            "heldout_evaluator_called": False,
            "final_oracle_session_opened": False,
            "downstream_utility_computed": False,
        },
        "one_shot_no_further_adaptation": control["one_shot_no_further_adaptation"],
        "failure_action": control["failure_action"],
    }
    _atomic_json(Path(output_json), payload)
    return payload


def write_recovery_failure(
    destination: str | os.PathLike[str],
    *,
    error: BaseException,
    seed: int,
    recovery_design_sha256: str | None,
    registry_sha256: str | None,
) -> None:
    evidence = error.evidence if isinstance(error, OptimizationConvergenceError) else None
    payload: dict[str, object] = {
        "schema_version": RECOVERY_FAILURE_SCHEMA,
        "status": "FAILED",
        "seed": seed,
        "recovery_design_sha256": recovery_design_sha256,
        "parent_failure_registry_sha256": registry_sha256,
        "error_type": type(error).__name__,
        "message": str(error),
        "optimization_convergence_evidence": evidence,
        "one_shot_no_further_adaptation": True,
        "retry_or_second_recovery_authorized": False,
        "information_boundary": {
            "policy_rollout_performed": False,
            "heldout_evaluator_called": False,
            "final_oracle_session_opened": False,
            "downstream_utility_computed": False,
        },
    }
    _atomic_json(Path(destination), payload)


__all__ = [
    "PARENT_REGISTRY_SCHEMA",
    "RECOVERY_FAILURE_SCHEMA",
    "RECOVERY_RESULT_SCHEMA",
    "TRAIN_TENSOR_KEYS",
    "run_phase2_recovery_train_only",
    "write_recovery_failure",
]
