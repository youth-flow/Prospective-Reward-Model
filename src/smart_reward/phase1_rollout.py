"""Matched-KL Phase-1 policy rollouts under the frozen LoRA tangent.

This stage is intentionally downstream of the immutable Phase-1 artifact and
the fixed-step reward-head comparison.  It never retrains a reward head and it
never reads an oracle value from the training split.  Instead, it

* reconstructs both policy directions from train-only features and scores;
* reloads the exact pinned, FP32, fixed-A/zero-B policy coordinate system;
* matches BT-MLE and ProRM+ independently to the same measured forward-KL
  budget on a small shared set of saved reference candidates;
* samples test responses with per-prompt common random numbers; and
* releases the policy before loading the oracle once and applying the frozen
  train-fitted robust transform.

Every externally supplied identity is checked before a model is loaded.  The
two output files are new-file-only, and raw oracle logits are never serialized.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch

from . import hf as _hf
from . import phase1 as _phase1
from .artifacts import load_controlled_feature_artifact
from .config import config_hash, validate_config
from .contracts import (
    CANONICAL_LEARNERS,
    CONTROLLED_COMPARISON_SCHEMA_V1,
    CONTROLLED_COMPARISON_SCHEMA_V2,
    LEGACY_V1_LEARNERS,
    MATCHED_KL_ROLLOUT_SCHEMA_V2,
    UPDATED_ROLLOUT_SCHEMA_V2,
)
from .data import CandidateNode, load_jsonl
from .experiment import ControlledFeatureExperiment
from .hf import ExactTokenCandidates, FixedALoRASetup
from .linear import resolve_fisher_solve_dtype
from .oracle import RobustOracleTransform
from .paths import relative_posix_reference
from .prompts import PromptRecord, load_prompt_jsonl
from .repro import collect_execution_identity
from .rollout import (
    match_fixed_a_measured_kl,
    oracle_rollout_improvement,
    policy_direction_from_head,
)
from .scores import ParameterLayout
from .seeding import SeedBundle, derive_seed

_COMPARISON_SCHEMAS = {
    CONTROLLED_COMPARISON_SCHEMA_V1,
    CONTROLLED_COMPARISON_SCHEMA_V2,
}
_RESULT_SCHEMA = MATCHED_KL_ROLLOUT_SCHEMA_V2
_ROLLOUT_SCHEMA = UPDATED_ROLLOUT_SCHEMA_V2
_ARTIFACT_EVIDENCE_SCHEMA = "phase1-materialization/v1"
_LEARNERS = CANONICAL_LEARNERS
_ROLLOUT_POLICIES = ("reference", *_LEARNERS)
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_JSON_BYTES = 64 * 1024 * 1024


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    return seed


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_object(source: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(source)
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"JSON input exceeds {_MAX_JSON_BYTES} bytes: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head_sha256(weight: torch.Tensor) -> str:
    value = weight.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(bytes(value.view(torch.uint8).tolist()))
    return digest.hexdigest()


def _validate_environment_identity(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    fields = {
        "formal",
        "git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "account",
        "partition",
        "gpu_models",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has an invalid schema")
    if not isinstance(value["formal"], bool):
        raise TypeError(f"{name}.formal must be boolean")
    if value["formal"] is not True:
        raise ValueError("matched-KL rollout requires a formal comparison environment")
    image = _validate_digest(value["image_sha256"], name=f"{name} image_sha256")
    hf_inventory = _validate_digest(
        value["hf_inventory_sha256"],
        name=f"{name} hf_inventory_sha256",
    )
    git_commit = value["git_commit"]
    if (
        not isinstance(git_commit, str)
        or len(git_commit) not in {40, 64}
        or any(character not in _HEX_DIGITS for character in git_commit)
    ):
        raise ValueError(f"{name} git_commit must be a lowercase Git digest")
    if value["account"] != "sigroup":
        raise ValueError(f"{name} account must equal 'sigroup'")
    partition = value["partition"]
    if not isinstance(partition, str) or not partition:
        raise ValueError(f"{name} partition must be non-empty")
    gpu_models = value["gpu_models"]
    if (
        not isinstance(gpu_models, Sequence)
        or isinstance(gpu_models, (str, bytes, bytearray))
        or len(gpu_models) != 1
        or not isinstance(gpu_models[0], str)
        or not gpu_models[0]
    ):
        raise ValueError(f"{name} gpu_models must contain exactly one model name")
    return {
        "formal": True,
        "git_commit": git_commit,
        "image_sha256": image,
        "hf_inventory_sha256": hf_inventory,
        "account": "sigroup",
        "partition": partition,
        "gpu_models": [gpu_models[0]],
    }


def parse_comparison_heads(
    source: str | os.PathLike[str] | Mapping[str, object],
    *,
    expected_config_hash: str,
    expected_seed: int,
    expected_artifact_metadata_sha256: str,
    expected_dimension: int | None = None,
) -> dict[str, tuple[float, ...]]:
    """Parse canonical multiplier-one BT/ProRM+ heads from comparison v1/v2.

    The comparison identity, learner names, finite float32 head bytes, and
    recorded head digests are all checked.  Sensitivity runs cannot
    accidentally supply the policy update.
    """

    expected_digest = _validate_digest(expected_config_hash, name="expected_config_hash")
    expected_artifact_digest = _validate_digest(
        expected_artifact_metadata_sha256,
        name="expected_artifact_metadata_sha256",
    )
    validated_seed = _validate_seed(expected_seed)
    if expected_dimension is not None and (
        isinstance(expected_dimension, bool)
        or not isinstance(expected_dimension, int)
        or expected_dimension < 1
    ):
        raise ValueError("expected_dimension must be a positive integer or None")
    value = dict(source) if isinstance(source, Mapping) else _read_json_object(source)

    required = {
        "schema_version",
        "config_hash",
        "seed",
        "artifact_dir",
        "artifact_metadata_sha256",
        "run_manifest",
        "run_manifest_sha256",
        "environment_identity",
        "damping_runs",
    }
    if set(value) != required:
        raise ValueError(
            "comparison fields do not match the controlled-comparison contract: "
            f"missing={sorted(required - set(value))!r}, "
            f"extra={sorted(set(value) - required)!r}"
        )
    schema_version = value["schema_version"]
    if schema_version not in _COMPARISON_SCHEMAS:
        raise ValueError(f"unsupported comparison schema: {schema_version!r}")
    serialized_learners = (
        LEGACY_V1_LEARNERS
        if schema_version == CONTROLLED_COMPARISON_SCHEMA_V1
        else CANONICAL_LEARNERS
    )
    if value["config_hash"] != expected_digest:
        raise ValueError("comparison config hash does not match the validated configuration")
    if value["seed"] != validated_seed:
        raise ValueError("comparison seed does not match the requested seed")
    recorded_artifact_digest = _validate_digest(
        value["artifact_metadata_sha256"],
        name="comparison artifact_metadata_sha256",
    )
    if recorded_artifact_digest != expected_artifact_digest:
        raise ValueError("comparison was not trained from this artifact metadata")
    if not isinstance(value["artifact_dir"], str) or not value["artifact_dir"]:
        raise ValueError("comparison artifact_dir must be a non-empty string")
    if not isinstance(value["run_manifest"], str) or not value["run_manifest"]:
        raise ValueError("comparison run_manifest must be a non-empty string")
    _validate_digest(value["run_manifest_sha256"], name="comparison run_manifest_sha256")
    _validate_environment_identity(
        value["environment_identity"],
        name="comparison environment_identity",
    )
    runs = value["damping_runs"]
    if isinstance(runs, (str, bytes, bytearray)) or not isinstance(runs, Sequence):
        raise TypeError("comparison damping_runs must be a sequence")
    main_runs: list[Mapping[str, object]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping) or set(run) != {"damping_multiplier", "result"}:
            raise ValueError(f"comparison damping_runs[{index}] has an invalid schema")
        multiplier = _finite_float(
            run["damping_multiplier"], name=f"damping_runs[{index}].damping_multiplier"
        )
        if multiplier == 1.0:
            main_runs.append(run)
    if len(main_runs) != 1:
        raise ValueError("comparison must contain exactly one damping_multiplier=1 run")
    result = main_runs[0]["result"]
    if not isinstance(result, Mapping):
        raise ValueError("the multiplier-one comparison result must be an object")

    heads: dict[str, tuple[float, ...]] = {}
    for learner_name, serialized_name in zip(_LEARNERS, serialized_learners, strict=True):
        learner = result.get(serialized_name)
        if not isinstance(learner, Mapping):
            raise ValueError(f"comparison is missing learner {serialized_name!r}")
        if learner.get("method") != serialized_name:
            raise ValueError(f"comparison learner {serialized_name!r} has the wrong method")
        raw_weight = learner.get("head_weight")
        if isinstance(raw_weight, (str, bytes, bytearray)) or not isinstance(raw_weight, Sequence):
            raise TypeError(f"{serialized_name}.head_weight must be a sequence")
        weight = tuple(
            _finite_float(item, name=f"{serialized_name}.head_weight[{index}]")
            for index, item in enumerate(raw_weight)
        )
        if not weight:
            raise ValueError(f"{serialized_name}.head_weight must not be empty")
        if expected_dimension is not None and len(weight) != expected_dimension:
            raise ValueError(
                f"{serialized_name}.head_weight has length {len(weight)}, "
                f"expected {expected_dimension}"
            )
        recorded_sha = _validate_digest(
            learner.get("head_sha256"), name=f"{serialized_name}.head_sha256"
        )
        tensor = torch.tensor(weight, dtype=torch.float32)
        if _head_sha256(tensor) != recorded_sha:
            raise ValueError(f"{serialized_name}.head_weight does not match head_sha256")
        heads[learner_name] = weight
    return heads


def _probe_rank(seed: int, candidate: CandidateNode) -> tuple[bytes, str, str]:
    payload = (
        f"smart-reward-model/kl-probe/v1\0{seed}\0"
        f"{candidate.prompt_id!r}\0{candidate.candidate_id!r}"
    ).encode()
    return (
        hashlib.sha256(payload).digest(),
        repr(candidate.prompt_id),
        repr(candidate.candidate_id),
    )


def select_kl_probe_nodes(
    candidates: Sequence[CandidateNode],
    train_prompt_ids: Sequence[str | int],
    *,
    count: int,
    seed: int,
) -> tuple[CandidateNode, ...]:
    """Select an input-order-invariant shared subset of train candidates."""

    validated_seed = _validate_seed(seed)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence")
    values = tuple(candidates)
    if not values or not all(isinstance(candidate, CandidateNode) for candidate in values):
        raise TypeError("candidates must contain at least one CandidateNode")
    train_ids = tuple(train_prompt_ids)
    if not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("train_prompt_ids must be non-empty and unique")
    train_set = set(train_ids)
    eligible = [candidate for candidate in values if candidate.prompt_id in train_set]
    identities = [(candidate.prompt_id, candidate.candidate_id) for candidate in eligible]
    if len(set(identities)) != len(identities):
        raise ValueError("candidate identities must be unique")
    if count > len(eligible):
        raise ValueError(
            f"requested {count} KL probes but only {len(eligible)} train candidates exist"
        )
    return tuple(sorted(eligible, key=lambda item: _probe_rank(validated_seed, item))[:count])


def pad_reference_candidates(
    candidates: Sequence[CandidateNode],
    *,
    pad_token_id: int,
    device: str | torch.device = "cpu",
    source_model_id: int | None = None,
    source_trainable_sha256: str | None = None,
) -> ExactTokenCandidates:
    """Safely right-pad rows while preserving every original token index.

    Phase 1 serialized one prompt at a time, so every position before the
    first response-mask one is an active prompt token.  Rows are truncated
    immediately after their final active response token and padded only on the
    right.  Consequently the causal model sees exactly the same absolute token
    and response positions as in the saved candidate; this avoids relying on a
    model-specific reconstruction of ``position_ids`` from an attention mask.

    ``ExactTokenCandidates.prompt_width`` is the minimum response start.  Its
    contract only requires all earlier positions to be non-response positions;
    individual rows may start their response later.
    """

    if (
        isinstance(pad_token_id, bool)
        or not isinstance(pad_token_id, Integral)
        or int(pad_token_id) < 0
    ):
        raise ValueError("pad_token_id must be a non-negative integer")
    values = tuple(candidates)
    if not values or not all(isinstance(value, CandidateNode) for value in values):
        raise TypeError("candidates must contain at least one CandidateNode")
    if source_trainable_sha256 is not None:
        _validate_digest(source_trainable_sha256, name="source_trainable_sha256")

    rows: list[tuple[tuple[int, ...], tuple[int, ...], int, CandidateNode]] = []
    for candidate in values:
        active = tuple(index for index, bit in enumerate(candidate.response_mask) if bit)
        if not active:
            raise ValueError("every probe candidate must contain a response")
        start = active[0]
        stop = active[-1] + 1
        if start < 1:
            raise ValueError("every probe candidate must retain at least one prompt token")
        if active != tuple(range(start, stop)):
            raise ValueError("serialized response_mask must select one contiguous span")
        active_tokens = candidate.token_ids[:stop]
        active_response_mask = candidate.response_mask[:stop]
        if not active_tokens or not any(active_response_mask):
            raise ValueError("probe candidate prompt and response must both be non-empty")
        rows.append((active_tokens, active_response_mask, start, candidate))

    prompt_width = min(start for _, _, start, _ in rows)
    sequence_width = max(len(tokens) for tokens, _, _, _ in rows)
    target_device = torch.device(device)
    input_ids = torch.full(
        (len(rows), sequence_width),
        int(pad_token_id),
        dtype=torch.int64,
        device=target_device,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    terminated = torch.empty(len(rows), dtype=torch.bool, device=target_device)
    reached_limit = torch.empty(len(rows), dtype=torch.bool, device=target_device)
    for row_index, (tokens, mask, _, candidate) in enumerate(rows):
        input_ids[row_index, : len(tokens)] = torch.tensor(
            tokens, dtype=torch.int64, device=target_device
        )
        attention_mask[row_index, : len(tokens)] = True
        response_mask[row_index, : len(mask)] = torch.tensor(
            mask, dtype=torch.bool, device=target_device
        )
        terminated[row_index] = candidate.terminated_by_eos
        reached_limit[row_index] = candidate.reached_max_length

    return ExactTokenCandidates(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        terminated_by_eos=terminated,
        reached_max_length=reached_limit,
        prompt_width=prompt_width,
        source_model_id=source_model_id,
        source_trainable_sha256=source_trainable_sha256,
    )


def select_and_pad_kl_probe_candidates(
    candidates: Sequence[CandidateNode],
    train_prompt_ids: Sequence[str | int],
    *,
    count: int,
    seed: int,
    pad_token_id: int,
    device: str | torch.device = "cpu",
    source_model_id: int | None = None,
    source_trainable_sha256: str | None = None,
) -> tuple[tuple[CandidateNode, ...], ExactTokenCandidates]:
    """Compose deterministic train-probe selection and safe tensor padding."""

    selected = select_kl_probe_nodes(
        candidates,
        train_prompt_ids,
        count=count,
        seed=seed,
    )
    batch = pad_reference_candidates(
        selected,
        pad_token_id=pad_token_id,
        device=device,
        source_model_id=source_model_id,
        source_trainable_sha256=source_trainable_sha256,
    )
    return selected, batch


def _finite_reward_vector(value: object, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.float64)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            tensor = torch.tensor(tuple(value), dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must contain real scalars") from error
    else:
        raise TypeError(f"{name} must be a tensor or sequence")
    if tensor.ndim != 1 or tensor.numel() < 1 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return tensor


def assemble_rollout_result(
    *,
    config_sha256: str,
    seed: int,
    artifact_dir: str | os.PathLike[str],
    comparison_json: str | os.PathLike[str],
    updated_rollouts_jsonl: str | os.PathLike[str],
    reference_base: str | os.PathLike[str],
    artifact_metadata_sha256: str,
    comparison_sha256: str,
    updated_rollouts_sha256: str,
    run_manifest_sha256: str,
    environment_identity: Mapping[str, object],
    kl_probe_candidate_ids: Sequence[str | int],
    reference_rollout_rewards: torch.Tensor | Sequence[float],
    artifact_test_rewards: torch.Tensor | Sequence[float],
    learner_direction_evidence: Mapping[str, Mapping[str, object]],
    learner_update_evidence: Mapping[str, Mapping[str, object]],
    learner_transformed_rewards: Mapping[str, torch.Tensor | Sequence[float]],
    rollout_seed: int,
    num_test_prompts: int,
    candidates_per_prompt: int,
) -> dict[str, object]:
    """Assemble the strict JSON result without model inference or file reads."""

    digest = _validate_digest(config_sha256, name="config_sha256")
    artifact_digest = _validate_digest(artifact_metadata_sha256, name="artifact_metadata_sha256")
    comparison_digest = _validate_digest(comparison_sha256, name="comparison_sha256")
    rollouts_digest = _validate_digest(updated_rollouts_sha256, name="updated_rollouts_sha256")
    manifest_digest = _validate_digest(run_manifest_sha256, name="run_manifest_sha256")
    environment = _validate_environment_identity(
        environment_identity,
        name="environment_identity",
    )
    validated_seed = _validate_seed(seed)
    validated_rollout_seed = _validate_seed(rollout_seed)
    for name, value in (
        ("num_test_prompts", num_test_prompts),
        ("candidates_per_prompt", candidates_per_prompt),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if num_test_prompts < 2:
        raise ValueError("num_test_prompts must be at least two for a prompt-level sample SE")
    probe_ids = tuple(kl_probe_candidate_ids)
    if not probe_ids or len(set(probe_ids)) != len(probe_ids):
        raise ValueError("kl_probe_candidate_ids must be non-empty and unique")
    for name, mapping in (
        ("learner_direction_evidence", learner_direction_evidence),
        ("learner_update_evidence", learner_update_evidence),
        ("learner_transformed_rewards", learner_transformed_rewards),
    ):
        if not isinstance(mapping, Mapping) or set(mapping) != set(_LEARNERS):
            raise ValueError(f"{name} must contain exactly {_LEARNERS!r}")

    reference = _finite_reward_vector(reference_rollout_rewards, name="reference_rollout_rewards")
    artifact_reference = _finite_reward_vector(artifact_test_rewards, name="artifact_test_rewards")
    if artifact_reference.numel() != reference.numel():
        raise ValueError(
            "artifact test and zero-B reference rollout must have the same candidate count"
        )
    expected_rollouts = num_test_prompts * candidates_per_prompt
    if reference.numel() != expected_rollouts:
        raise ValueError(
            "reference rollout count must equal num_test_prompts*candidates_per_prompt"
        )
    reference_mean = float(reference.mean().item())
    artifact_reference_mean = float(artifact_reference.mean().item())
    learners: dict[str, object] = {}
    for learner_name in _LEARNERS:
        direction = learner_direction_evidence[learner_name]
        update = learner_update_evidence[learner_name]
        if not isinstance(direction, Mapping) or direction.get("schema_version") != (
            "policy-direction/v1"
        ):
            raise ValueError(f"{learner_name} direction evidence has the wrong schema")
        if not isinstance(update, Mapping) or update.get("schema_version") != (
            "measured-kl-update/v1"
        ):
            raise ValueError(f"{learner_name} measured-KL evidence has the wrong schema")
        if update.get("converged") is not True or update.get("applied") is not True:
            raise ValueError(f"{learner_name} did not converge to an applied KL step")
        rewards = _finite_reward_vector(
            learner_transformed_rewards[learner_name],
            name=f"{learner_name}_transformed_rewards",
        )
        if rewards.numel() != reference.numel():
            raise ValueError(
                f"{learner_name} rollout count must match the zero-B reference rollout"
            )
        updated_mean = float(rewards.mean().item())
        # Candidate outcomes within one prompt share context and generation
        # history.  The experimental unit is therefore the prompt: average
        # candidate differences inside each prompt before computing the sample
        # standard error across prompts.
        prompt_reference = reference.reshape(num_test_prompts, candidates_per_prompt).mean(dim=1)
        prompt_updated = rewards.reshape(num_test_prompts, candidates_per_prompt).mean(dim=1)
        paired = oracle_rollout_improvement(prompt_reference, prompt_updated)
        learners[learner_name] = {
            "direction": dict(direction),
            "measured_kl_update": dict(update),
            "test_rollout_candidates": rewards.numel(),
            "test_transformed_oracle_mean": updated_mean,
            "paired_improvement_over_zero_b_reference": paired.to_dict(),
        }

    return {
        "schema_version": _RESULT_SCHEMA,
        "config_hash": digest,
        "seed": validated_seed,
        "artifact_dir": relative_posix_reference(artifact_dir, base=reference_base),
        "comparison_json": relative_posix_reference(comparison_json, base=reference_base),
        "updated_rollouts_jsonl": relative_posix_reference(
            updated_rollouts_jsonl,
            base=reference_base,
        ),
        "artifact_metadata_sha256": artifact_digest,
        "comparison_sha256": comparison_digest,
        "updated_rollouts_sha256": rollouts_digest,
        "run_manifest_sha256": manifest_digest,
        "environment_identity": environment,
        "kl_probe": {
            "candidate_ids": list(probe_ids),
            "num_candidates": len(probe_ids),
            "selection_seed": derive_seed(validated_seed, "kl_probe_selection"),
            "shared_reference_distribution": True,
        },
        "test_reference": {
            "source": "zero_b_common_random_number_rollout",
            "num_candidates": reference.numel(),
            "num_prompts": num_test_prompts,
            "candidates_per_prompt": candidates_per_prompt,
            "transformed_oracle_mean": reference_mean,
        },
        "artifact_test_descriptive_sanity": {
            "source": "phase1_artifact.test.true_rewards",
            "paired_with_updated_rollouts": False,
            "num_candidates": artifact_reference.numel(),
            "transformed_oracle_mean": artifact_reference_mean,
        },
        "common_random_numbers": {
            "named_stream": "rollout",
            "seed": validated_rollout_seed,
            "reset_per_learner_and_prompt": True,
        },
        "learners": learners,
        "train_oracle_values_accessed": False,
        "raw_oracle_values_serialized": False,
    }


@dataclass(frozen=True, slots=True)
class _ArtifactContract:
    a_state_sha256: str
    layout: ParameterLayout
    policy_chat_template_sha256: str
    oracle_chat_template_sha256: str
    oracle_transform: RobustOracleTransform
    jsonl_sha256: Mapping[str, str]


def _validate_producer_identity(value: object) -> dict[str, str]:
    fields = {"git_commit", "image_sha256", "hf_inventory_sha256"}
    if not isinstance(value, Mapping) or not set(value).issubset(fields):
        raise ValueError("artifact producer must contain only validated digest fields")
    producer: dict[str, str] = {}
    for name, raw_digest in value.items():
        allowed_lengths = {40, 64} if name == "git_commit" else {64}
        if (
            not isinstance(raw_digest, str)
            or len(raw_digest) not in allowed_lengths
            or any(character not in _HEX_DIGITS for character in raw_digest)
        ):
            raise ValueError(f"artifact producer {name} is not a lowercase digest")
        producer[name] = raw_digest
    formal_producer = _phase1._producer_identity_from_environment()
    if formal_producer and producer != formal_producer:
        raise ValueError(
            "artifact producer identity does not match the formal Git/image/HF-inventory identity"
        )
    return producer


def _artifact_contract(
    artifact_dir: Path,
    *,
    normalized_config: Mapping[str, object],
    expected_config_hash: str,
    expected_seed: int,
) -> _ArtifactContract:
    metadata = _read_json_object(artifact_dir / "metadata.json")
    if metadata.get("config_hash") != expected_config_hash:
        raise ValueError("artifact metadata config hash does not match the configuration")
    if metadata.get("seed") != expected_seed:
        raise ValueError("artifact metadata seed does not match the requested seed")
    evidence = metadata.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("artifact evidence must be an object")
    if evidence.get("schema") != _ARTIFACT_EVIDENCE_SCHEMA:
        raise ValueError("artifact is not a phase1-materialization/v1 artifact")
    if evidence.get("config_sha256") != expected_config_hash:
        raise ValueError("artifact evidence config hash is inconsistent")
    if evidence.get("seed") != expected_seed:
        raise ValueError("artifact evidence seed is inconsistent")
    expected_named_seeds = SeedBundle.from_base_seed(expected_seed).to_dict()
    if evidence.get("named_seeds") != expected_named_seeds:
        raise ValueError("artifact named seeds are inconsistent with the requested seed")

    a_sha = _validate_digest(evidence.get("policy_a_sha256"), name="policy_a_sha256")
    policy_config = normalized_config.get("policy")
    if not isinstance(policy_config, Mapping):
        raise TypeError("normalized policy config must be a mapping")
    if policy_config.get("fixed_lora_a") is not None:
        try:
            expected_lora_a_evidence = _phase1._policy_lora_a_initialization_evidence(
                policy_config,
                run_seed=expected_seed,
                observed_sha256=a_sha,
            )
        except RuntimeError as error:
            raise ValueError(
                "artifact LoRA-A fingerprint differs from the configured global identity"
            ) from error
        if evidence.get("policy_lora_a_initialization") != expected_lora_a_evidence:
            raise ValueError(
                "artifact global LoRA-A initialization/provenance does not match "
                "the source configuration"
            )
    policy_chat_sha = _validate_digest(
        evidence.get("chat_template_sha256"), name="chat_template_sha256"
    )
    oracle_chat_sha = _validate_digest(
        evidence.get("oracle_chat_template_sha256"),
        name="oracle_chat_template_sha256",
    )
    raw_layout = evidence.get("policy_layout")
    if isinstance(raw_layout, (str, bytes, bytearray)) or not isinstance(raw_layout, Sequence):
        raise TypeError("artifact policy_layout must be a sequence")
    for index, entry in enumerate(raw_layout):
        if not isinstance(entry, Mapping) or set(entry) != {
            "name",
            "shape",
            "offset",
            "numel",
        }:
            raise ValueError(f"artifact policy_layout[{index}] has an invalid schema")
    layout = ParameterLayout.from_metadata(raw_layout)

    transform_value = evidence.get("oracle_transform")
    if not isinstance(transform_value, Mapping) or set(transform_value) != {"b", "tau"}:
        raise ValueError("artifact oracle_transform must contain exactly b and tau")
    transform = RobustOracleTransform(
        b=_finite_float(transform_value["b"], name="oracle_transform.b"),
        tau=_finite_float(transform_value["tau"], name="oracle_transform.tau"),
    )
    oracle_config = normalized_config.get("oracle")
    if not isinstance(oracle_config, Mapping):
        raise TypeError("normalized oracle config must be a mapping")
    expected_transform_calibration = oracle_config.get("transform_calibration")
    if expected_transform_calibration is not None:
        if not isinstance(expected_transform_calibration, Mapping):
            raise TypeError("oracle.transform_calibration must be a mapping")
        if evidence.get("oracle_transform_calibration") != expected_transform_calibration:
            raise ValueError(
                "artifact frozen oracle-transform calibration/provenance does not "
                "match the source configuration"
            )
        if (
            evidence.get("oracle_fit_split") != "excluded_phase1_train_artifacts"
            or evidence.get("oracle_transform_fitted_on_current_seed") is not False
        ):
            raise ValueError(
                "artifact does not prove that the frozen oracle transform avoided "
                "current-seed refitting"
            )
        if transform.b != float(expected_transform_calibration["b"]) or transform.tau != float(
            expected_transform_calibration["tau"]
        ):
            raise ValueError(
                "artifact oracle_transform differs from the configured frozen global values"
            )
    revisions = evidence.get("revisions")
    expected_revisions = {
        "prompt_dataset": normalized_config["data"]["prompt_revision"],
        "policy_model": normalized_config["policy"]["revision"],
        "reward_feature_model": normalized_config["reward_model"]["revision"],
        "oracle_model": normalized_config["oracle"]["revision"],
    }
    if revisions != expected_revisions:
        raise ValueError("artifact pinned revisions do not match the configuration")

    _validate_producer_identity(evidence.get("producer"))

    json_hashes = evidence.get("jsonl_sha256")
    expected_json_names = {
        "prompts.jsonl",
        "candidates.jsonl",
        "training_edges.jsonl",
        "evaluation_edges.jsonl",
    }
    if not isinstance(json_hashes, Mapping) or set(json_hashes) != expected_json_names:
        raise ValueError("artifact jsonl_sha256 inventory is incomplete or has extra files")
    validated_hashes = {
        name: _validate_digest(value, name=f"jsonl_sha256[{name!r}]")
        for name, value in json_hashes.items()
    }
    for name, expected in validated_hashes.items():
        path = artifact_dir / name
        if _sha256_file(path) != expected:
            raise ValueError(f"artifact JSONL SHA256 mismatch: {name}")
    return _ArtifactContract(
        a_state_sha256=a_sha,
        layout=layout,
        policy_chat_template_sha256=policy_chat_sha,
        oracle_chat_template_sha256=oracle_chat_sha,
        oracle_transform=transform,
        jsonl_sha256=validated_hashes,
    )


@dataclass(slots=True)
class _PolicyRuntime:
    tokenizer: object
    setup: FixedALoRASetup


@dataclass(slots=True)
class _OracleRuntime:
    tokenizer: object
    model: torch.nn.Module


def _load_policy_runtime(
    config: Mapping[str, object],
    *,
    seed: int,
    device: torch.device,
    local_files_only: bool,
) -> _PolicyRuntime:
    transformers = _phase1._require_module("transformers")
    peft = _phase1._require_module("peft")
    policy = config["policy"]
    tokenizer = _phase1._load_pretrained(
        transformers.AutoTokenizer,
        policy["model"],
        policy["revision"],
        local_files_only=local_files_only,
        kind="policy tokenizer",
        use_fast=True,
    )
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("policy tokenizer must provide a non-empty chat_template")
    tokenizer.truncation_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError("policy tokenizer must expose a pad or EOS token id")

    lora_a_initialization = _phase1._policy_lora_a_initialization(
        policy,
        run_seed=seed,
    )
    lora_seed = int(lora_a_initialization["initialization_seed"])
    with _phase1._fork_torch_seed(lora_seed, device):
        model = _phase1._load_pretrained(
            transformers.AutoModelForCausalLM,
            policy["model"],
            policy["revision"],
            local_files_only=local_files_only,
            kind="policy model",
            torch_dtype=torch.float32,
        )
    model.to(device)
    model.eval()
    lora_config = peft.LoraConfig(
        r=policy["lora_rank"],
        lora_alpha=policy["lora_alpha"],
        lora_dropout=policy["lora_dropout"],
        target_modules=list(policy["lora_modules"]),
        layers_to_transform=list(policy["lora_layers"]),
        bias="none",
        init_lora_weights=True,
        task_type="CAUSAL_LM",
    )
    with _phase1._fork_torch_seed(lora_seed, device):
        setup = _hf.configure_fixed_a_lora(model, lora_config)
    _phase1._policy_lora_a_initialization_evidence(
        policy,
        run_seed=seed,
        observed_sha256=setup.a_state_sha256,
    )
    setup.model.eval()
    return _PolicyRuntime(tokenizer=tokenizer, setup=setup)


def _load_oracle_runtime(
    config: Mapping[str, object],
    *,
    device: torch.device,
    local_files_only: bool,
) -> _OracleRuntime:
    transformers = _phase1._require_module("transformers")
    oracle = config["oracle"]
    tokenizer = _phase1._load_pretrained(
        transformers.AutoTokenizer,
        oracle["model"],
        oracle["revision"],
        local_files_only=local_files_only,
        kind="oracle tokenizer",
        use_fast=True,
    )
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("oracle tokenizer must provide a non-empty chat_template")
    model = _phase1._load_pretrained(
        transformers.AutoModelForSequenceClassification,
        oracle["model"],
        oracle["revision"],
        local_files_only=local_files_only,
        kind="oracle model",
        torch_dtype=torch.float32,
        num_labels=1,
    )
    model.to(device)
    model.eval()
    return _OracleRuntime(tokenizer=tokenizer, model=model)


def _template_sha256(tokenizer: object) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template in (None, ""):
        raise ValueError("tokenizer must provide a non-empty chat_template")
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def _zero_b_(setup: FixedALoRASetup) -> None:
    with torch.no_grad():
        for _, parameter in setup.named_tangent_parameters():
            parameter.zero_()


def _validate_prompt_candidate_join(
    experiment: ControlledFeatureExperiment,
    prompts: Sequence[PromptRecord],
    candidates: Sequence[CandidateNode],
    *,
    num_candidates: int,
) -> tuple[PromptRecord, ...]:
    expected_ids = (
        *experiment.train.prompt_ids,
        *experiment.validation.prompt_ids,
        *experiment.test.prompt_ids,
    )
    if len(prompts) != len(expected_ids):
        raise ValueError("prompts.jsonl count does not match the tensor artifact")
    by_id = {prompt.prompt_id: prompt for prompt in prompts}
    if len(by_id) != len(prompts) or set(by_id) != set(expected_ids):
        raise ValueError("prompts.jsonl IDs do not match the tensor artifact")
    ordered = tuple(by_id[prompt_id] for prompt_id in expected_ids)
    expected_splits = (
        ("train", experiment.train.prompt_ids),
        ("validation", experiment.validation.prompt_ids),
        ("test", experiment.test.prompt_ids),
    )
    for split, prompt_ids in expected_splits:
        if any(by_id[prompt_id].split != split for prompt_id in prompt_ids):
            raise ValueError(f"prompts.jsonl has an incorrect {split} assignment")

    grouped: dict[str | int, list[CandidateNode]] = defaultdict(list)
    identities: set[tuple[str | int, str | int]] = set()
    for candidate in candidates:
        identity = (candidate.prompt_id, candidate.candidate_id)
        if identity in identities:
            raise ValueError("candidates.jsonl contains a duplicate candidate identity")
        identities.add(identity)
        grouped[candidate.prompt_id].append(candidate)
    if set(grouped) != set(expected_ids):
        raise ValueError("candidates.jsonl prompt IDs do not match the tensor artifact")
    if any(len(grouped[prompt_id]) != num_candidates for prompt_id in expected_ids):
        raise ValueError("each artifact prompt must have exactly data.num_candidates nodes")
    for prompt_id, nodes in grouped.items():
        record = by_id[prompt_id]
        prompt_text = _phase1._prompt_text(record)
        if any(node.prompt != prompt_text for node in nodes):
            raise ValueError("candidate prompt text does not match prompts.jsonl")
    return ordered


def _generate_updated_rollouts(
    runtime: _PolicyRuntime,
    test_prompts: Sequence[PromptRecord],
    *,
    policy_name: str,
    config: Mapping[str, object],
    rollout_seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    if policy_name not in _ROLLOUT_POLICIES:
        raise ValueError(f"unknown rollout policy: {policy_name!r}")
    policy = config["policy"]
    num_candidates = int(config["evaluation"]["rollout_candidates_per_prompt"])
    generation_kwargs = {
        **dict(policy["sampling"]),
        "num_return_sequences": num_candidates,
        "max_new_tokens": int(policy["max_response_tokens"]),
        "eos_token_id": runtime.tokenizer.eos_token_id,
        "pad_token_id": runtime.tokenizer.pad_token_id,
        "use_cache": True,
    }
    records: list[dict[str, object]] = []
    for prompt in test_prompts:
        encoded = runtime.tokenizer.apply_chat_template(
            [message.to_dict() for message in prompt.messages],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=int(policy["max_prompt_tokens"]),
            return_tensors="pt",
            return_dict=True,
        )
        prompt_inputs = _phase1._model_inputs(encoded, device)
        prompt_seed = derive_seed(rollout_seed, f"test-prompt:{prompt.prompt_id}")
        with _phase1._fork_torch_seed(prompt_seed, device):
            generated = _hf.generate_exact_candidates(
                runtime.setup.model,
                prompt_inputs["input_ids"],
                prompt_attention_mask=prompt_inputs["attention_mask"],
                generation_kwargs=generation_kwargs,
            )
        if generated.input_ids.shape[0] != num_candidates:
            raise RuntimeError("policy returned an unexpected number of test candidates")
        prompt_text = _phase1._prompt_text(prompt)
        for candidate_index in range(num_candidates):
            response_ids = generated.input_ids[candidate_index][
                generated.response_mask[candidate_index].bool()
            ]
            records.append(
                {
                    "schema_version": _ROLLOUT_SCHEMA,
                    "policy": policy_name,
                    "learner": policy_name,
                    "policy_source": (
                        "zero_b_reference" if policy_name == "reference" else "matched_kl_update"
                    ),
                    "prompt_id": prompt.prompt_id,
                    "candidate_index": candidate_index,
                    "prompt": prompt_text,
                    "response": _phase1._decode_response(runtime.tokenizer, response_ids),
                    "token_ids": [
                        int(value) for value in generated.input_ids[candidate_index].tolist()
                    ],
                    "response_mask": [
                        int(value) for value in generated.response_mask[candidate_index].tolist()
                    ],
                    "terminated_by_eos": bool(generated.terminated_by_eos[candidate_index].item()),
                    "reached_max_length": bool(
                        generated.reached_max_length[candidate_index].item()
                    ),
                    "prompt_rollout_seed": prompt_seed,
                }
            )
        del generated, prompt_inputs
    return records


def _score_updated_rollouts(
    runtime: _OracleRuntime,
    records: Sequence[Mapping[str, object]],
    *,
    transform: RobustOracleTransform,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, torch.Tensor]]:
    if batch_size < 1:
        raise ValueError("oracle batch_size must be positive")
    raw_batches: list[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        raw_batches.append(
            _hf.score_oracle_chats(
                runtime.model,
                runtime.tokenizer,
                [str(record["prompt"]) for record in batch],
                [str(record["response"]) for record in batch],
                device=device,
            ).to(device="cpu", dtype=torch.float32)
        )
    # Raw logits remain transient and are transformed before any record is
    # assembled.  In particular, no raw value is written to the output JSONL.
    raw_scores = torch.cat(raw_batches)
    transformed = transform(raw_scores).detach().to(device="cpu", dtype=torch.float32)
    if transformed.shape != (len(records),):
        raise RuntimeError("oracle score count does not match updated rollouts")
    output_records: list[dict[str, object]] = []
    by_policy: dict[str, list[float]] = {name: [] for name in _ROLLOUT_POLICIES}
    for record, reward in zip(records, transformed.tolist(), strict=True):
        policy_name = record.get("policy")
        if policy_name not in by_policy:
            raise ValueError("updated rollout contains an unknown policy")
        value = float(reward)
        by_policy[policy_name].append(value)
        output_records.append({**dict(record), "transformed_oracle_reward": value})
    del raw_scores, raw_batches, transformed
    return output_records, {
        policy_name: torch.tensor(values, dtype=torch.float32)
        for policy_name, values in by_policy.items()
    }


def _stage_jsonl(
    destination: Path,
    records: Sequence[Mapping[str, object]],
) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        for record in records:
            json.dump(
                record,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _stage_json(destination: Path, payload: Mapping[str, object]) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _publish_staged_pair(
    rollouts_path: Path,
    staged_rollouts: Path,
    result_path: Path,
    staged_result: Path,
) -> None:
    """Publish two staged files with rollback on a partial rename."""

    for target in (rollouts_path, result_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {target}")
    staged = [staged_rollouts, staged_result]
    published: list[Path] = []
    try:
        for temporary, target in (
            (staged_rollouts, rollouts_path),
            (staged_result, result_path),
        ):
            if target.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {target}")
            os.replace(temporary, target)
            staged.remove(temporary)
            published.append(target)
    except BaseException:
        # Preflight established that neither target existed, so every path in
        # ``published`` was created by this transaction and is safe to remove.
        for target in reversed(published):
            with suppress(FileNotFoundError):
                target.unlink()
        raise
    finally:
        for temporary in staged:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _publish_output_pair(
    rollouts_path: Path,
    rollout_records: Sequence[Mapping[str, object]],
    result_path: Path,
    result: Mapping[str, object],
) -> None:
    """Convenience wrapper that stages and transactionally publishes both files."""

    staged_rollouts: Path | None = None
    staged_result: Path | None = None
    try:
        staged_rollouts = _stage_jsonl(rollouts_path, rollout_records)
        staged_result = _stage_json(result_path, result)
        _publish_staged_pair(
            rollouts_path,
            staged_rollouts,
            result_path,
            staged_result,
        )
        staged_rollouts = None
        staged_result = None
    finally:
        for temporary in (staged_rollouts, staged_result):
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()


def evaluate_matched_kl_rollouts(
    config: Mapping[str, object],
    *,
    seed: int,
    artifact_dir: str | os.PathLike[str],
    comparison_json: str | os.PathLike[str],
    output_json: str | os.PathLike[str],
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, object]:
    """Run the real matched-KL BT/ProRM+ test-rollout evaluation stage.

    Outputs are ``output_json`` and a sibling ``updated_rollouts.jsonl``.
    Existing targets are refused before optional dependencies, model cache, or
    CUDA state are touched.
    """

    validated_seed = _validate_seed(seed)
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be bool")
    normalized = validate_config(config)
    configured_seeds = (
        [normalized["run"]["seed"]] if "seed" in normalized["run"] else normalized["run"]["seeds"]
    )
    if validated_seed not in configured_seeds:
        raise ValueError("seed must be one of the explicitly configured run seeds")
    if any(normalized[section]["dtype"] != "float32" for section in ("policy", "oracle")):
        raise ValueError("matched-KL policy and oracle evaluation requires float32")
    digest = config_hash(normalized)

    destination = Path(output_json)
    rollouts_path = destination.parent / "updated_rollouts.jsonl"
    if destination.resolve() == rollouts_path.resolve():
        raise ValueError("output_json cannot be named updated_rollouts.jsonl")
    for target in (destination, rollouts_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {target}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    artifact_path = Path(artifact_dir)
    comparison_path = Path(comparison_json)
    artifact_metadata_digest = _sha256_file(artifact_path / "metadata.json")
    experiment = load_controlled_feature_artifact(
        artifact_path,
        expected_config_hash=digest,
        expected_seed=validated_seed,
    )
    contract = _artifact_contract(
        artifact_path,
        normalized_config=normalized,
        expected_config_hash=digest,
        expected_seed=validated_seed,
    )
    if contract.layout.dimension != experiment.train.policy_dimension:
        raise ValueError("artifact policy layout dimension does not match train policy scores")
    if _sha256_file(artifact_path / "metadata.json") != artifact_metadata_digest:
        raise RuntimeError("artifact metadata changed while its contract was being validated")
    comparison_digest = _sha256_file(comparison_path)
    comparison_value = _read_json_object(comparison_path)
    heads = parse_comparison_heads(
        comparison_value,
        expected_config_hash=digest,
        expected_seed=validated_seed,
        expected_artifact_metadata_sha256=artifact_metadata_digest,
        expected_dimension=experiment.train.reward_dimension,
    )
    if _sha256_file(comparison_path) != comparison_digest:
        raise RuntimeError("comparison JSON changed while it was being validated")

    prompts = load_prompt_jsonl(artifact_path / "prompts.jsonl")
    candidates = load_jsonl(artifact_path / "candidates.jsonl", CandidateNode)
    ordered_prompts = _validate_prompt_candidate_join(
        experiment,
        prompts,
        candidates,
        num_candidates=int(normalized["data"]["num_candidates"]),
    )
    test_prompt_ids = set(experiment.test.prompt_ids)
    test_prompts = tuple(
        prompt for prompt in ordered_prompts if prompt.prompt_id in test_prompt_ids
    )
    if tuple(prompt.prompt_id for prompt in test_prompts) != experiment.test.prompt_ids:
        raise ValueError("test prompt order does not match the tensor artifact")

    objective = normalized["objective"]
    direction_results = {
        learner: policy_direction_from_head(
            experiment.train,
            head,
            relative_damping=float(objective["damping_relative_to_mean_fisher_diagonal"]),
            beta=float(objective["beta"]),
            pcg_dtype=objective["pcg_dtype"],
            pcg_max_iterations=int(objective["pcg_max_iterations"]),
            pcg_tolerance=float(objective["pcg_tolerance"]),
            require_pcg_convergence=True,
        )
        for learner, head in heads.items()
    }

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if target_device.type == "cuda" and collect_execution_identity() != dict(
        comparison_value["environment_identity"]
    ):
        raise RuntimeError("rollout execution environment does not match the comparison")
    policy_runtime = _load_policy_runtime(
        normalized,
        seed=validated_seed,
        device=target_device,
        local_files_only=local_files_only,
    )
    if policy_runtime.setup.a_state_sha256 != contract.a_state_sha256:
        raise RuntimeError("reloaded LoRA-A SHA256 does not match the Phase-1 artifact")
    if policy_runtime.setup.layout.to_metadata() != contract.layout.to_metadata():
        raise RuntimeError("reloaded LoRA-B coordinate layout does not match Phase 1")
    if _template_sha256(policy_runtime.tokenizer) != contract.policy_chat_template_sha256:
        raise RuntimeError("reloaded policy chat template does not match Phase 1")
    _zero_b_(policy_runtime.setup)
    zero_b_sha = _hf._fingerprint_named_tensors(policy_runtime.setup.named_tangent_parameters())
    solve_dtype = resolve_fisher_solve_dtype(objective["pcg_dtype"])
    train_node_scores = experiment.train.policy_scores.to(
        device=target_device,
        dtype=solve_dtype,
    )
    selected_probes, reference_candidates = select_and_pad_kl_probe_candidates(
        candidates,
        experiment.train.prompt_ids,
        count=int(normalized["evaluation"]["kl_probe_candidates"]),
        seed=derive_seed(validated_seed, "kl_probe_selection"),
        pad_token_id=int(policy_runtime.tokenizer.pad_token_id),
        device=target_device,
        source_model_id=id(policy_runtime.setup.model),
        source_trainable_sha256=zero_b_sha,
    )

    update_results: dict[str, object] = {}
    seeds = SeedBundle.from_base_seed(validated_seed)
    # Generate the zero-B baseline first.  Every subsequent learner resets the
    # same per-prompt named seed, giving candidate-index-aligned common random
    # numbers rather than comparing against Phase-1's candidate-generation
    # stream.
    unscored_rollouts = _generate_updated_rollouts(
        policy_runtime,
        test_prompts,
        policy_name="reference",
        config=normalized,
        rollout_seed=seeds.rollout,
        device=target_device,
    )
    try:
        for learner in _LEARNERS:
            _zero_b_(policy_runtime.setup)
            if (
                _hf._fingerprint_named_tensors(policy_runtime.setup.named_tangent_parameters())
                != zero_b_sha
            ):
                raise RuntimeError("LoRA-B did not return to the common zero origin")
            direction = direction_results[learner].direction.to(
                device=target_device,
                dtype=solve_dtype,
            )
            update = match_fixed_a_measured_kl(
                policy_runtime.setup.model,
                policy_runtime.setup,
                reference_candidates,
                direction,
                target_kl=float(normalized["evaluation"]["kl_budget"]),
                train_node_scores=train_node_scores,
                relative_tolerance=float(normalized["evaluation"]["kl_relative_tolerance"]),
            )
            if not update.converged or not update.applied:
                _zero_b_(policy_runtime.setup)
                raise RuntimeError(
                    f"{learner} measured-KL line search did not converge; policy restored to zero-B"
                )
            update_results[learner] = update
            unscored_rollouts.extend(
                _generate_updated_rollouts(
                    policy_runtime,
                    test_prompts,
                    policy_name=learner,
                    config=normalized,
                    rollout_seed=seeds.rollout,
                    device=target_device,
                )
            )
            _zero_b_(policy_runtime.setup)
    finally:
        _zero_b_(policy_runtime.setup)

    # The oracle and policy never coexist after this boundary.
    del direction, reference_candidates, train_node_scores, policy_runtime
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    oracle_runtime = _load_oracle_runtime(
        normalized,
        device=target_device,
        local_files_only=local_files_only,
    )
    try:
        if _template_sha256(oracle_runtime.tokenizer) != (contract.oracle_chat_template_sha256):
            raise RuntimeError("reloaded oracle chat template does not match Phase 1")
        scored_rollouts, transformed_by_learner = _score_updated_rollouts(
            oracle_runtime,
            unscored_rollouts,
            transform=contract.oracle_transform,
            batch_size=min(16, int(normalized["reward_model"]["microbatch_size"])),
            device=target_device,
        )
    finally:
        del oracle_runtime
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    staged_rollouts: Path | None = None
    staged_result: Path | None = None
    try:
        # The main result binds the exact canonical JSONL bytes that will be
        # published, not a reserialization guessed in advance.
        staged_rollouts = _stage_jsonl(rollouts_path, scored_rollouts)
        rollouts_digest = _sha256_file(staged_rollouts)
        payload = assemble_rollout_result(
            config_sha256=digest,
            seed=validated_seed,
            artifact_dir=artifact_path,
            comparison_json=comparison_path,
            updated_rollouts_jsonl=rollouts_path,
            reference_base=destination.parent,
            artifact_metadata_sha256=artifact_metadata_digest,
            comparison_sha256=comparison_digest,
            updated_rollouts_sha256=rollouts_digest,
            run_manifest_sha256=str(comparison_value["run_manifest_sha256"]),
            environment_identity=comparison_value["environment_identity"],
            kl_probe_candidate_ids=[candidate.candidate_id for candidate in selected_probes],
            reference_rollout_rewards=transformed_by_learner["reference"],
            artifact_test_rewards=experiment.test.true_rewards.reshape(-1),
            learner_direction_evidence={
                learner: direction_results[learner].to_dict() for learner in _LEARNERS
            },
            learner_update_evidence={
                learner: update_results[learner].to_dict() for learner in _LEARNERS
            },
            learner_transformed_rewards={
                learner: transformed_by_learner[learner] for learner in _LEARNERS
            },
            rollout_seed=seeds.rollout,
            num_test_prompts=experiment.test.num_prompts,
            candidates_per_prompt=experiment.test.num_candidates,
        )
        staged_result = _stage_json(destination, payload)
        _publish_staged_pair(
            rollouts_path,
            staged_rollouts,
            destination,
            staged_result,
        )
        staged_rollouts = None
        staged_result = None
        return payload
    finally:
        for temporary in (staged_rollouts, staged_result):
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()


__all__ = [
    "assemble_rollout_result",
    "evaluate_matched_kl_rollouts",
    "pad_reference_candidates",
    "parse_comparison_heads",
    "select_and_pad_kl_probe_candidates",
    "select_kl_probe_nodes",
]
