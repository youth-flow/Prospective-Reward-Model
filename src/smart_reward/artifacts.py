"""Integrity-checked Phase-1 tensor artifacts.

The artifact boundary is intentionally narrow.  It serializes only the tensors
accepted by :class:`~smart_reward.experiment.ControlledFeatureExperiment`:
training contains observable preference statistics, while true rewards exist
only in validation and test.  Runtime environment variables, host details, and
credentials are never collected.

``safetensors`` is an optional dependency and is imported only when one of the
public save/load functions is called.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .experiment import (
    ControlledFeatureExperiment,
    EvaluationTensorData,
    TrainingTensorData,
)

SCHEMA = "controlled-feature-artifact/v1"
METADATA_FILENAME = "metadata.json"
TENSORS_FILENAME = "tensors.safetensors"

_SPLITS = ("train", "validation", "test")
_EXPECTED_TENSOR_KEYS = frozenset(
    {
        "train.policy_scores",
        "train.reward_features",
        "train.h",
        "train.left_wins",
        "train.num_annotations",
        "validation.policy_scores",
        "validation.reward_features",
        "validation.true_rewards",
        "test.policy_scores",
        "test.reward_features",
        "test.true_rewards",
    }
)
_TOP_LEVEL_KEYS = frozenset(
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
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_DTYPE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_SENSITIVE_EVIDENCE_SEGMENTS = frozenset(
    {
        "api_key",
        "access_key",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "credentials",
        "env",
        "environ",
        "environment",
        "password",
        "passwd",
        "private_key",
        "secret",
    }
)
_SAFE_TOKEN_METADATA_KEYS = frozenset(
    {
        "fail_closed_above_max_prompt_tokens",
        "max_prompt_tokens",
        "policy_chat_token_count",
        "policy_prompt_token_ids_sha256",
        "selected_maximum_policy_chat_token_count",
        "selected_minimum_policy_chat_token_count",
    }
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\Ahf_[A-Za-z0-9]{20,}\Z"),
    re.compile(r"\Agh[opsu]_[A-Za-z0-9]{20,}\Z"),
    re.compile(r"\Agithub_pat_[A-Za-z0-9_]{20,}\Z"),
    re.compile(r"\Ask-[A-Za-z0-9_-]{20,}\Z"),
)
_MAX_METADATA_BYTES = 16 * 1024 * 1024


class ArtifactError(ValueError):
    """Base class for an invalid or unverifiable controlled-feature artifact."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when serialized bytes do not match their recorded digest."""


class ArtifactDependencyError(ImportError):
    """Raised when the optional safetensors dependency is unavailable."""


def _require_safetensors_torch() -> Any:
    try:
        return importlib.import_module("safetensors.torch")
    except (ImportError, ModuleNotFoundError) as error:
        raise ArtifactDependencyError(
            "Phase-1 artifact I/O requires the optional dependency 'safetensors'; "
            "install smart-reward-model[llm]"
        ) from error


def _validate_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ArtifactError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _validate_seed(seed: object, *, name: str = "seed") -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2**63 - 1:
        raise ArtifactError(f"{name} must be an integer in [0, 2**63 - 1]")
    return seed


def _key_segments(key: str) -> set[str]:
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.lower()).strip("_")
    pieces = set(part for part in normalized.split("_") if part)
    pieces.add(normalized)
    for width in (2,):
        split = normalized.split("_")
        pieces.update("_".join(split[index : index + width]) for index in range(len(split)))
    return pieces


def _is_sensitive_evidence_key(key: str) -> bool:
    """Distinguish credential tokens from ordinary tokenizer metadata.

    Every key containing the standalone segment ``token`` or ``tokens`` fails
    closed unless it is one of the explicitly reviewed tokenizer-metadata
    fields in the artifact schema.  This keeps counts and digests serializable
    without letting credential namespaces hide behind structural suffixes.
    """

    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.lower()).strip("_")
    segments = _key_segments(key)
    if segments.intersection(_SENSITIVE_EVIDENCE_SEGMENTS):
        return True
    if not {"token", "tokens"}.intersection(segments):
        return False
    return normalized not in _SAFE_TOKEN_METADATA_KEYS


def _validate_evidence(value: object, *, path: str = "evidence") -> Any:
    """Return a JSON-safe copy while rejecting likely environment/secret fields."""

    if value is None or isinstance(value, (bool, str)):
        looks_secret = isinstance(value, str) and any(
            pattern.fullmatch(value) for pattern in _SECRET_VALUE_PATTERNS
        )
        if looks_secret:
            raise ArtifactError(f"{path} appears to contain a credential")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ArtifactError(f"{path} keys must be non-empty strings")
            if _is_sensitive_evidence_key(key):
                raise ArtifactError(
                    f"{path}.{key} is an environment/credential field and cannot be stored"
                )
            result[key] = _validate_evidence(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _validate_evidence(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise ArtifactError(
        f"{path} must contain JSON primitives, lists, or mappings; got {type(value).__name__}"
    )


def _canonical_evidence(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {}
    if not isinstance(evidence, Mapping):
        raise ArtifactError("evidence must be a mapping or None")
    result = _validate_evidence(evidence)
    if not isinstance(result, dict):  # Kept explicit for static and runtime safety.
        raise ArtifactError("evidence must be a mapping")
    return result


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    if not name.startswith("torch."):
        raise ArtifactError(f"unsupported torch dtype representation: {name!r}")
    return name.removeprefix("torch.")


def _tensor_payload(experiment: ControlledFeatureExperiment) -> dict[str, torch.Tensor]:
    train = experiment.train
    validation = experiment.validation
    test = experiment.test
    tensors = {
        "train.policy_scores": train.policy_scores,
        "train.reward_features": train.reward_features,
        "train.h": train.h,
        "train.left_wins": train.left_wins,
        "train.num_annotations": train.num_annotations,
        "validation.policy_scores": validation.policy_scores,
        "validation.reward_features": validation.reward_features,
        "validation.true_rewards": validation.true_rewards,
        "test.policy_scores": test.policy_scores,
        "test.reward_features": test.reward_features,
        "test.true_rewards": test.true_rewards,
    }
    if frozenset(tensors) != _EXPECTED_TENSOR_KEYS:
        raise RuntimeError("internal controlled-feature tensor schema mismatch")
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ArtifactError(f"experiment field {key!r} is not a torch.Tensor")
    # Saving detached CPU copies makes artifacts device-independent and prevents
    # caller mutation while safetensors is writing.
    return {
        key: tensor.detach().to(device="cpu").contiguous().clone()
        for key, tensor in tensors.items()
    }


def _tensor_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(tensor.shape),
            "dtype": _dtype_name(tensor.dtype),
        }
        for key, tensor in sorted(tensors.items())
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    # Windows requires a writable descriptor for ``os.fsync``.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _temporary_path(directory: Path, *, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    os.close(descriptor)
    return Path(name)


def _metadata_payload(
    experiment: ControlledFeatureExperiment,
    tensors: Mapping[str, torch.Tensor],
    *,
    config_hash: str,
    seed: int,
    tensor_sha256: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "config_hash": config_hash,
        "seed": seed,
        "splits": {
            "train": {"prompt_ids": list(experiment.train.prompt_ids)},
            "validation": {"prompt_ids": list(experiment.validation.prompt_ids)},
            "test": {"prompt_ids": list(experiment.test.prompt_ids)},
        },
        "tensors": _tensor_specs(tensors),
        "tensor_sha256": tensor_sha256,
        "evidence": evidence,
    }


def save_controlled_feature_artifact(
    experiment: ControlledFeatureExperiment,
    directory: str | os.PathLike[str],
    *,
    config_hash: str,
    seed: int,
    evidence: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically save a leakage-safe controlled-feature artifact.

    ``metadata.json`` is installed last and therefore serves as the commit
    marker.  Existing target files are refused unless ``overwrite=True``.
    """

    if not isinstance(experiment, ControlledFeatureExperiment):
        raise TypeError("experiment must be ControlledFeatureExperiment")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be bool")
    validated_config_hash = _validate_digest("config_hash", config_hash)
    validated_seed = _validate_seed(seed)
    validated_evidence = _canonical_evidence(evidence)
    safetensors_torch = _require_safetensors_torch()

    target_directory = Path(directory)
    if target_directory.exists() and not target_directory.is_dir():
        raise NotADirectoryError(f"artifact path is not a directory: {target_directory}")
    target_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = target_directory / METADATA_FILENAME
    tensors_path = target_directory / TENSORS_FILENAME
    existing = [
        path.name for path in (metadata_path, tensors_path) if path.exists() or path.is_symlink()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing artifact target(s): {sorted(existing)!r}"
        )
    for target in (metadata_path, tensors_path):
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"artifact target is a directory: {target}")

    tensors = _tensor_payload(experiment)
    # Tensors are mutable even inside frozen dataclasses.  Reconstructing from
    # the detached snapshot reruns all shape, dtype, finiteness, and split
    # invariants immediately before persistence.
    snapshot = _rebuild_experiment(
        {
            "train": experiment.train.prompt_ids,
            "validation": experiment.validation.prompt_ids,
            "test": experiment.test.prompt_ids,
        },
        tensors,
    )
    tensor_temp = _temporary_path(target_directory, prefix=".tensors-")
    metadata_temp = _temporary_path(target_directory, prefix=".metadata-")
    try:
        safetensors_torch.save_file(tensors, str(tensor_temp))
        _fsync_file(tensor_temp)
        tensor_sha256 = _sha256_file(tensor_temp)
        metadata = _metadata_payload(
            snapshot,
            tensors,
            config_hash=validated_config_hash,
            seed=validated_seed,
            tensor_sha256=tensor_sha256,
            evidence=validated_evidence,
        )
        serialized_metadata = (
            json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        if len(serialized_metadata.encode("utf-8")) > _MAX_METADATA_BYTES:
            raise ArtifactError("metadata.json exceeds the 16 MiB safety limit")
        with metadata_temp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized_metadata)
            stream.flush()
            os.fsync(stream.fileno())

        # Metadata is the commit marker: a crash before its replacement cannot
        # make a newly created artifact look complete.
        os.replace(tensor_temp, tensors_path)
        os.replace(metadata_temp, metadata_path)
    finally:
        tensor_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return target_directory


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"metadata contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except FileNotFoundError as error:
        raise ArtifactError(f"missing artifact metadata: {path}") from error
    if size > _MAX_METADATA_BYTES:
        raise ArtifactError("metadata.json exceeds the 16 MiB safety limit")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError("metadata.json is not valid canonical UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ArtifactError("metadata.json root must be an object")
    return value


def _validate_prompt_ids(value: object, *, split: str) -> tuple[str | int, ...]:
    if not isinstance(value, list):
        raise ArtifactError(f"splits.{split}.prompt_ids must be a JSON array")
    result: list[str | int] = []
    for index, prompt_id in enumerate(value):
        if isinstance(prompt_id, bool) or not isinstance(prompt_id, (str, int)):
            raise ArtifactError(f"splits.{split}.prompt_ids[{index}] must be a string or integer")
        if isinstance(prompt_id, str) and not prompt_id:
            raise ArtifactError(f"splits.{split}.prompt_ids[{index}] must be non-empty")
        result.append(prompt_id)
    if not result:
        raise ArtifactError(f"split {split!r} must contain at least one prompt ID")
    if len(set(result)) != len(result):
        raise ArtifactError(f"split {split!r} contains duplicate prompt IDs")
    return tuple(result)


def _validate_splits(value: object) -> dict[str, tuple[str | int, ...]]:
    if not isinstance(value, dict) or set(value) != set(_SPLITS):
        raise ArtifactError("splits must contain exactly train, validation, and test")
    result: dict[str, tuple[str | int, ...]] = {}
    for split in _SPLITS:
        split_value = value[split]
        if not isinstance(split_value, dict) or set(split_value) != {"prompt_ids"}:
            raise ArtifactError(f"splits.{split} must contain exactly prompt_ids")
        result[split] = _validate_prompt_ids(split_value["prompt_ids"], split=split)
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = set(result[left]).intersection(result[right])
        if overlap:
            raise ArtifactError(
                f"prompt ID leakage between {left} and {right}: "
                f"{sorted(repr(item) for item in overlap)!r}"
            )
    return result


def _validate_tensor_specs(value: object) -> dict[str, tuple[tuple[int, ...], str]]:
    if not isinstance(value, dict):
        raise ArtifactError("tensors metadata must be an object")
    keys = frozenset(value)
    if keys != _EXPECTED_TENSOR_KEYS:
        missing = sorted(_EXPECTED_TENSOR_KEYS - keys)
        extra = sorted(keys - _EXPECTED_TENSOR_KEYS)
        raise ArtifactError(
            f"tensor metadata keys do not match schema; missing={missing!r}, extra={extra!r}"
        )
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for key in sorted(_EXPECTED_TENSOR_KEYS):
        spec = value[key]
        if not isinstance(spec, dict) or set(spec) != {"shape", "dtype"}:
            raise ArtifactError(f"tensor metadata for {key!r} must contain shape and dtype")
        shape = spec["shape"]
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape
        ):
            raise ArtifactError(f"tensor metadata shape for {key!r} is invalid")
        dtype = spec["dtype"]
        if not isinstance(dtype, str) or _DTYPE_NAME.fullmatch(dtype) is None:
            raise ArtifactError(f"tensor metadata dtype for {key!r} is invalid")
        result[key] = (tuple(shape), dtype)
    return result


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    expected_config_hash: str | None,
    expected_seed: int | None,
) -> tuple[
    dict[str, tuple[str | int, ...]],
    dict[str, tuple[tuple[int, ...], str]],
    str,
]:
    if set(metadata) != _TOP_LEVEL_KEYS:
        missing = sorted(_TOP_LEVEL_KEYS - set(metadata))
        extra = sorted(set(metadata) - _TOP_LEVEL_KEYS)
        raise ArtifactError(
            f"metadata fields do not match schema; missing={missing!r}, extra={extra!r}"
        )
    if metadata["schema"] != SCHEMA:
        raise ArtifactError(f"unsupported artifact schema: {metadata['schema']!r}")
    config_hash = _validate_digest("metadata config_hash", metadata["config_hash"])
    seed = _validate_seed(metadata["seed"], name="metadata seed")
    tensor_sha256 = _validate_digest("metadata tensor_sha256", metadata["tensor_sha256"])
    if expected_config_hash is not None:
        expected = _validate_digest("expected_config_hash", expected_config_hash)
        if config_hash != expected:
            raise ArtifactError(
                f"config hash mismatch: expected {expected}, artifact records {config_hash}"
            )
    if expected_seed is not None:
        expected_value = _validate_seed(expected_seed, name="expected_seed")
        if seed != expected_value:
            raise ArtifactError(
                f"seed mismatch: expected {expected_value}, artifact records {seed}"
            )
    _canonical_evidence(metadata["evidence"])
    prompt_ids = _validate_splits(metadata["splits"])
    specs = _validate_tensor_specs(metadata["tensors"])
    return prompt_ids, specs, tensor_sha256


def _validate_loaded_tensors(
    tensors: object,
    specs: Mapping[str, tuple[tuple[int, ...], str]],
) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, dict):
        raise ArtifactError("safetensors payload did not decode to a tensor mapping")
    keys = frozenset(tensors)
    if keys != _EXPECTED_TENSOR_KEYS:
        missing = sorted(_EXPECTED_TENSOR_KEYS - keys)
        extra = sorted(keys - _EXPECTED_TENSOR_KEYS)
        raise ArtifactError(
            f"safetensors keys do not match schema; missing={missing!r}, extra={extra!r}"
        )
    result: dict[str, torch.Tensor] = {}
    for key in sorted(_EXPECTED_TENSOR_KEYS):
        tensor = tensors[key]
        if not isinstance(tensor, torch.Tensor):
            raise ArtifactError(f"payload entry {key!r} is not a torch.Tensor")
        expected_shape, expected_dtype = specs[key]
        if tuple(tensor.shape) != expected_shape:
            raise ArtifactError(
                f"shape mismatch for {key!r}: metadata={expected_shape!r}, "
                f"payload={tuple(tensor.shape)!r}"
            )
        actual_dtype = _dtype_name(tensor.dtype)
        if actual_dtype != expected_dtype:
            raise ArtifactError(
                f"dtype mismatch for {key!r}: metadata={expected_dtype!r}, payload={actual_dtype!r}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ArtifactError(f"payload tensor {key!r} contains NaN or infinity")
        result[key] = tensor.detach().to(device="cpu").contiguous()
    return result


def _rebuild_experiment(
    prompt_ids: Mapping[str, tuple[str | int, ...]],
    tensors: Mapping[str, torch.Tensor],
) -> ControlledFeatureExperiment:
    try:
        train = TrainingTensorData(
            prompt_ids=prompt_ids["train"],
            policy_scores=tensors["train.policy_scores"],
            reward_features=tensors["train.reward_features"],
            h=tensors["train.h"],
            left_wins=tensors["train.left_wins"],
            num_annotations=tensors["train.num_annotations"],
        )
        validation = EvaluationTensorData(
            prompt_ids=prompt_ids["validation"],
            policy_scores=tensors["validation.policy_scores"],
            reward_features=tensors["validation.reward_features"],
            true_rewards=tensors["validation.true_rewards"],
        )
        test = EvaluationTensorData(
            prompt_ids=prompt_ids["test"],
            policy_scores=tensors["test.policy_scores"],
            reward_features=tensors["test.reward_features"],
            true_rewards=tensors["test.true_rewards"],
        )
        return ControlledFeatureExperiment(train=train, validation=validation, test=test)
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"artifact violates controlled experiment schema: {error}") from error


def load_controlled_feature_artifact(
    directory: str | os.PathLike[str],
    *,
    expected_config_hash: str | None = None,
    expected_seed: int | None = None,
) -> ControlledFeatureExperiment:
    """Verify and load a Phase-1 artifact as experiment dataclasses.

    The raw tensor file is SHA-256 checked *before* safetensors parses it.  Key,
    shape, dtype, finiteness, and prompt-disjointness checks then run before the
    experiment dataclasses are reconstructed.
    """

    target_directory = Path(directory)
    if not target_directory.is_dir():
        raise ArtifactError(f"artifact directory does not exist: {target_directory}")
    metadata = _read_metadata(target_directory / METADATA_FILENAME)
    prompt_ids, specs, expected_tensor_hash = _validate_metadata(
        metadata,
        expected_config_hash=expected_config_hash,
        expected_seed=expected_seed,
    )
    tensors_path = target_directory / TENSORS_FILENAME
    try:
        before = tensors_path.stat()
    except FileNotFoundError as error:
        raise ArtifactError(f"missing artifact tensor file: {tensors_path}") from error
    actual_tensor_hash = _sha256_file(tensors_path)
    if actual_tensor_hash != expected_tensor_hash:
        raise ArtifactIntegrityError(
            "tensors.safetensors SHA-256 mismatch: "
            f"expected {expected_tensor_hash}, computed {actual_tensor_hash}"
        )

    # Import and deserialize only after byte-level integrity succeeds.
    safetensors_torch = _require_safetensors_torch()
    try:
        raw_tensors = safetensors_torch.load_file(str(tensors_path), device="cpu")
    except Exception as error:
        raise ArtifactError("tensors.safetensors could not be decoded") from error
    after = tensors_path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ArtifactIntegrityError("tensors.safetensors changed while it was being loaded")
    tensors = _validate_loaded_tensors(raw_tensors, specs)
    return _rebuild_experiment(prompt_ids, tensors)


def artifact_metadata_sha256(
    directory: str | os.PathLike[str],
    *,
    expected_config_hash: str | None = None,
    expected_seed: int | None = None,
) -> str:
    """Return the validated metadata-byte identity of one artifact.

    The canonical metadata records the tensor SHA-256 and, for Phase 1, the
    four JSONL SHA-256 values.  Binding a downstream result to this digest
    therefore prevents a head trained on another same-shaped artifact from
    being silently reused.  Callers that consume tensors must still use
    :func:`load_controlled_feature_artifact`, which verifies the tensor bytes.
    """

    target = Path(directory) / METADATA_FILENAME
    try:
        before = target.stat()
    except FileNotFoundError as error:
        raise ArtifactError(f"missing artifact metadata: {target}") from error
    metadata = _read_metadata(target)
    _validate_metadata(
        metadata,
        expected_config_hash=expected_config_hash,
        expected_seed=expected_seed,
    )
    digest = _sha256_file(target)
    after = target.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ArtifactIntegrityError("metadata.json changed while it was being identified")
    return digest


__all__ = [
    "ArtifactDependencyError",
    "ArtifactError",
    "ArtifactIntegrityError",
    "METADATA_FILENAME",
    "SCHEMA",
    "TENSORS_FILENAME",
    "artifact_metadata_sha256",
    "load_controlled_feature_artifact",
    "save_controlled_feature_artifact",
]
