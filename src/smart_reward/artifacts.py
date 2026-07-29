"""Atomic, integrity-checked experiment tensor artifacts."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .exact import ExactDeltaExperiment, ExactSplitData

SCHEMA = "prorm-experiment-artifact/v1"
METADATA_FILENAME = "metadata.json"
TENSORS_FILENAME = "tensors.safetensors"
_SPLITS = ("train", "validation", "test")
_TENSOR_NAMES = {
    f"{split}.{name}"
    for split in _SPLITS
    for name in ("policy_scores", "reward_features", "true_rewards")
}


class ArtifactError(ValueError):
    pass


class ArtifactIntegrityError(ArtifactError):
    pass


def _safetensors() -> Any:
    try:
        return importlib.import_module("safetensors.torch")
    except ImportError as error:
        raise ImportError("safetensors is required for experiment artifacts") from error


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ArtifactError("seed must be an integer in [0, 2**63)")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(experiment: ExactDeltaExperiment) -> dict[str, torch.Tensor]:
    return {
        f"{split}.{name}": getattr(getattr(experiment, split), name).detach().cpu().contiguous()
        for split in _SPLITS
        for name in ("policy_scores", "reward_features", "true_rewards")
    }


def _specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(tensors.items())
    }


def _rebuild(
    prompt_ids: Mapping[str, tuple[str, ...]],
    tensors: Mapping[str, torch.Tensor],
) -> ExactDeltaExperiment:
    splits = {
        split: ExactSplitData(
            prompt_ids=prompt_ids[split],
            policy_scores=tensors[f"{split}.policy_scores"],
            reward_features=tensors[f"{split}.reward_features"],
            true_rewards=tensors[f"{split}.true_rewards"],
        )
        for split in _SPLITS
    }
    return ExactDeltaExperiment(**splits)


def save_exact_delta_artifact(
    experiment: ExactDeltaExperiment,
    directory: str | os.PathLike[str],
    *,
    config_hash: str,
    seed: int,
    evidence: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    if not isinstance(experiment, ExactDeltaExperiment):
        raise TypeError("experiment must be ExactDeltaExperiment")
    config_digest = _digest(config_hash, "config_hash")
    seed_value = _seed(seed)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    metadata_path = target / METADATA_FILENAME
    tensors_path = target / TENSORS_FILENAME
    if not overwrite and (metadata_path.exists() or tensors_path.exists()):
        raise FileExistsError("refusing to overwrite an existing artifact")
    tensors = _payload(experiment)
    tensor_handle, tensor_name = tempfile.mkstemp(prefix=".tensors-", dir=target)
    metadata_handle, metadata_name = tempfile.mkstemp(prefix=".metadata-", dir=target)
    os.close(tensor_handle)
    os.close(metadata_handle)
    try:
        _safetensors().save_file(tensors, tensor_name)
        tensor_digest = _sha256(Path(tensor_name))
        metadata = {
            "schema": SCHEMA,
            "config_hash": config_digest,
            "seed": seed_value,
            "prompt_ids": {split: list(getattr(experiment, split).prompt_ids) for split in _SPLITS},
            "tensor_specs": _specs(tensors),
            "tensor_sha256": tensor_digest,
            "evidence": dict(evidence or {}),
        }
        with Path(metadata_name).open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                metadata,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tensor_name, tensors_path)
        os.replace(metadata_name, metadata_path)
    finally:
        Path(tensor_name).unlink(missing_ok=True)
        Path(metadata_name).unlink(missing_ok=True)
    return target


def _read_metadata(directory: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(directory) / METADATA_FILENAME
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    expected = {
        "schema",
        "config_hash",
        "seed",
        "prompt_ids",
        "tensor_specs",
        "tensor_sha256",
        "evidence",
    }
    if not isinstance(value, dict) or set(value) != expected or value["schema"] != SCHEMA:
        raise ArtifactError("malformed artifact metadata")
    _digest(value["config_hash"], "metadata config_hash")
    _digest(value["tensor_sha256"], "metadata tensor_sha256")
    _seed(value["seed"])
    if not isinstance(value["evidence"], dict):
        raise ArtifactError("metadata evidence must be an object")
    return value


def load_exact_delta_artifact(
    directory: str | os.PathLike[str],
    *,
    expected_config_hash: str | None = None,
    expected_seed: int | None = None,
) -> ExactDeltaExperiment:
    metadata = _read_metadata(directory)
    if expected_config_hash is not None and metadata["config_hash"] != _digest(
        expected_config_hash, "expected_config_hash"
    ):
        raise ArtifactError("config hash mismatch")
    if expected_seed is not None and metadata["seed"] != _seed(expected_seed):
        raise ArtifactError("seed mismatch")
    tensors_path = Path(directory) / TENSORS_FILENAME
    if _sha256(tensors_path) != metadata["tensor_sha256"]:
        raise ArtifactIntegrityError("tensor SHA-256 mismatch")
    tensors = _safetensors().load_file(str(tensors_path), device="cpu")
    if set(tensors) != _TENSOR_NAMES:
        raise ArtifactError("artifact tensor names mismatch")
    specs = metadata["tensor_specs"]
    if not isinstance(specs, dict) or set(specs) != _TENSOR_NAMES:
        raise ArtifactError("artifact tensor specifications mismatch")
    for name, tensor in tensors.items():
        if specs[name] != {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}:
            raise ArtifactError(f"tensor specification mismatch for {name}")
    prompt_ids_raw = metadata["prompt_ids"]
    if not isinstance(prompt_ids_raw, dict) or set(prompt_ids_raw) != set(_SPLITS):
        raise ArtifactError("prompt split metadata mismatch")
    prompt_ids = {split: tuple(str(value) for value in prompt_ids_raw[split]) for split in _SPLITS}
    return _rebuild(prompt_ids, tensors)


def exact_delta_artifact_metadata_sha256(
    directory: str | os.PathLike[str],
    *,
    expected_config_hash: str | None = None,
    expected_seed: int | None = None,
) -> str:
    metadata = _read_metadata(directory)
    if expected_config_hash is not None and metadata["config_hash"] != _digest(
        expected_config_hash, "expected_config_hash"
    ):
        raise ArtifactError("config hash mismatch")
    if expected_seed is not None and metadata["seed"] != _seed(expected_seed):
        raise ArtifactError("seed mismatch")
    return _sha256(Path(directory) / METADATA_FILENAME)


__all__ = [
    "ArtifactError",
    "ArtifactIntegrityError",
    "exact_delta_artifact_metadata_sha256",
    "load_exact_delta_artifact",
    "save_exact_delta_artifact",
]
