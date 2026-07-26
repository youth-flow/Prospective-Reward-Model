"""Train-only R3 materialization from immutable Phase-1 parent artifacts.

This module is deliberately narrower than the general Phase-2 input loader:

* the parent registry and every consumed source file are content verified;
* only the five reward-free ``train.*`` safetensors entries are decoded;
* only the exact leading ``P * M`` candidate records are parsed, with an
  unbuffered stream that cannot prefetch the held-out suffix;
* the only model session opened by the production entry point is a local-only
  oracle session on exactly one visible CUDA device; and
* the result carries a validator-specific sealed
  :class:`R3TrainMaterializationCapability`, never a Gate-P admission or
  authorization capability.

Hashing the complete candidate file is a control-plane byte-integrity check.
The data-plane JSON decoder sees only the train prefix.  Consequently invalid
UTF-8 or malicious JSON in the held-out suffix cannot cross the train-only
information boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import torch

from .config import config_hash, load_config
from .data import CandidateNode
from .experiment import TrainingTensorData
from .oracle import RobustOracleTransform
from .phase2_hf import HuggingFacePhase2Backend
from .phase2_primary import prepare_neutral_phase2_context
from .phase2_r3_config import R3ScienceConfigBundle
from .phase2_r3_materialization import (
    TRAIN_TENSOR_KEYS,
    TrainMaterializationProvenance,
    ValidatedR3Materialization,
    validate_r3_materialization,
)

PARENT_REGISTRY_SCHEMA: Final = "prorm-phase2-recovery-parent-failures/v1"
ARTIFACT_SCHEMA: Final = "controlled-feature-artifact/v1"
MATERIALIZATION_SCHEMA: Final = "phase1-materialization/v1"
ARTIFACT_BINDING_SCHEMA: Final = "prorm-phase2-artifact-binding/v1"
R3_INPUT_PREPARATION_TIMINGS_SCHEMA: Final = "phase2-recovery-r3-train-input-preparation-timings/v1"
R3_TRAIN_MATERIALIZATION_CAPABILITY_SCHEMA: Final = (
    "phase2-recovery-r3-validated-input-materialization-capability/v1"
)

_STORAGE_TRAIN_KEYS: Final = (
    "train.policy_scores",
    "train.reward_features",
    "train.h",
    "train.left_wins",
    "train.num_annotations",
)
_CONTEXT_TO_STORAGE: Final = dict(zip(TRAIN_TENSOR_KEYS, _STORAGE_TRAIN_KEYS, strict=True))
_ALL_TENSOR_KEYS: Final = frozenset(
    {
        *_STORAGE_TRAIN_KEYS,
        "validation.policy_scores",
        "validation.reward_features",
        "validation.true_rewards",
        "test.policy_scores",
        "test.reward_features",
        "test.true_rewards",
    }
)
_JSONL_FILES: Final = frozenset(
    {
        "candidates.jsonl",
        "prompts.jsonl",
        "training_edges.jsonl",
        "evaluation_edges.jsonl",
    }
)
_EVIDENCE_FILES: Final = frozenset(
    {
        "FAILED",
        "run-manifest.json",
        "artifact-materialization.json",
        "artifact-verification.json",
        "phase2-run.log",
    }
)
_ARTIFACT_FILES: Final = frozenset(
    {
        "metadata.json",
        "tensors.safetensors",
        *_JSONL_FILES,
    }
)
_ARTIFACT_DERIVED_DIGESTS: Final = frozenset(
    {"policy_prompt_semantics_records", "selected_prompt_ids"}
)
_REGISTRY_KEYS: Final = frozenset(
    {
        "schema_version",
        "campaign",
        "common_artifact_identities",
        "optimizer_diagnostic",
        "seeds",
    }
)
_CAMPAIGN_KEYS: Final = frozenset(
    {
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
)
_SEED_KEYS: Final = frozenset(
    {
        "seed",
        "array_task_id",
        "source_run",
        "source_artifact",
        "evidence_sha256",
        "artifact_sha256",
    }
)
_METADATA_KEYS: Final = frozenset(
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
_COMMON_TENSOR_SCHEMA_KEYS: Final = frozenset(
    {
        "num_tensor_keys",
        "train_policy_scores_shape",
        "train_reward_features_shape",
        "validation_policy_scores_shape",
        "validation_reward_features_shape",
        "test_policy_scores_shape",
        "test_reward_features_shape",
    }
)
_DTYPE_RE: Final = re.compile(r"[a-z][a-z0-9_]*\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT_RE: Final = re.compile(r"[0-9a-f]{40,64}\Z")
_MAX_JSON_BYTES: Final = 16 * 1024 * 1024
_MAX_ARTIFACT_BYTES: Final = 64 * 1024 * 1024 * 1024
_CAPABILITY_SEAL = object()

_FileIdentity = tuple[int, int, int, int, int]


class _SafeTensorHandle(Protocol):
    def __enter__(self) -> _SafeTensorHandle: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> object: ...

    def keys(self) -> Sequence[str]: ...

    def get_tensor(self, name: str) -> torch.Tensor: ...


class _BinaryLineStream(Protocol):
    def __enter__(self) -> _BinaryLineStream: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> object: ...

    def fileno(self) -> int: ...

    def readline(self, size: int = -1) -> bytes: ...


_SafeOpenFactory = Callable[..., _SafeTensorHandle]
_LineStreamFactory = Callable[[Path], _BinaryLineStream]
_Clock = Callable[[], int]
_OracleRescorer = Callable[..., torch.Tensor]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("R3 input evidence must contain strict JSON data") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _positive_seconds(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _exact_mapping(
    value: object,
    *,
    name: str,
    keys: set[str] | frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != set(keys):
        missing = sorted(set(keys) - set(value))
        extra = sorted(set(value) - set(keys))
        raise ValueError(f"{name} fields differ; missing={missing!r}, extra={extra!r}")
    return dict(value)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _decode_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} root must be an object")
    return value


def _file_identity(value: os.stat_result) -> _FileIdentity:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_mode),
    )


def _require_regular_file(path: Path, *, name: str) -> _FileIdentity:
    try:
        state = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"missing {name}: {path}") from error
    if not stat.S_ISREG(state.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    return _file_identity(state)


def _open_readonly_nofollow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def _stable_file_digest(
    path: Path,
    *,
    name: str,
    maximum_bytes: int = _MAX_ARTIFACT_BYTES,
) -> tuple[str, _FileIdentity]:
    before = _require_regular_file(path, name=name)
    if before[2] > maximum_bytes:
        raise ValueError(f"{name} exceeds its byte limit")
    descriptor = _open_readonly_nofollow(path)
    digest = hashlib.sha256()
    observed = 0
    try:
        opened = _file_identity(os.fstat(descriptor))
        if not stat.S_ISREG(opened[4]) or opened[:2] != before[:2]:
            raise ValueError(f"{name} changed while it was opened")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed += len(block)
            if observed > maximum_bytes:
                raise ValueError(f"{name} exceeds its byte limit")
            digest.update(block)
        after_descriptor = _file_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = _require_regular_file(path, name=name)
    if (
        observed != before[2]
        or opened != before
        or after_descriptor != before
        or after_path != before
    ):
        raise ValueError(f"{name} changed while it was hashed")
    return digest.hexdigest(), before


def _stable_file_bytes(
    path: Path,
    *,
    name: str,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> tuple[bytes, str, _FileIdentity]:
    before = _require_regular_file(path, name=name)
    if before[2] > maximum_bytes:
        raise ValueError(f"{name} exceeds its byte limit")
    descriptor = _open_readonly_nofollow(path)
    chunks: list[bytes] = []
    observed = 0
    try:
        opened = _file_identity(os.fstat(descriptor))
        if not stat.S_ISREG(opened[4]) or opened[:2] != before[:2]:
            raise ValueError(f"{name} changed while it was opened")
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            if not block:
                break
            observed += len(block)
            if observed > maximum_bytes:
                raise ValueError(f"{name} exceeds its byte limit")
            chunks.append(block)
        after_descriptor = _file_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = _require_regular_file(path, name=name)
    if (
        observed != before[2]
        or opened != before
        or after_descriptor != before
        or after_path != before
    ):
        raise ValueError(f"{name} changed while it was read")
    raw = b"".join(chunks)
    return raw, hashlib.sha256(raw).hexdigest(), before


def _read_json_with_digest(
    path: Path,
    *,
    name: str,
    expected_sha256: object,
) -> tuple[dict[str, Any], str, _FileIdentity]:
    expected = _digest(expected_sha256, name=f"{name} expected SHA256")
    raw, observed, identity = _stable_file_bytes(path, name=name)
    if observed != expected:
        raise ValueError(f"{name} SHA256 mismatch: expected {expected}, observed {observed}")
    return _decode_json_object(raw, name=name), observed, identity


def _require_real_root(path: str | os.PathLike[str]) -> Path:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("project_root must not be a symbolic link")
    try:
        root = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("project_root does not exist") from error
    if not root.is_dir() or root.is_symlink():
        raise ValueError("project_root must be a real directory")
    return root


def _safe_relative(value: object, *, name: str) -> Path:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty project-relative path")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{name} is not a safe project-relative path")
    return relative


def _resolve_inside(
    root: Path,
    relative: object,
    *,
    name: str,
    directory: bool,
) -> Path:
    rel = _safe_relative(relative, name=name)
    cursor = root
    for component in rel.parts:
        cursor = cursor / component
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError as error:
            raise ValueError(f"{name} does not exist: {cursor}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} path contains a symbolic link: {cursor}")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes project_root") from error
    if directory:
        if not resolved.is_dir() or resolved.is_symlink():
            raise ValueError(f"{name} must be a real directory")
    else:
        _require_regular_file(resolved, name=name)
    return resolved


def _resolve_file_argument(
    root: Path,
    value: str | os.PathLike[str],
    *,
    name: str,
) -> Path:
    supplied = Path(value)
    if supplied.is_absolute():
        try:
            # Normalize ``..`` without resolving symlinks: resolving first
            # would erase the very path component this boundary must reject.
            lexical = Path(os.path.abspath(os.fspath(supplied)))
            relative = lexical.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} must be an existing file inside project_root") from error
    else:
        relative = supplied
    return _resolve_inside(root, os.fspath(relative), name=name, directory=False)


def _validate_hash_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str],
) -> dict[str, str]:
    mapping = _exact_mapping(value, name=name, keys=keys)
    return {key: _digest(mapping[key], name=f"{name}[{key!r}]") for key in sorted(keys)}


def _validate_registry(
    value: object,
    *,
    science: R3ScienceConfigBundle,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry = _exact_mapping(value, name="parent registry", keys=_REGISTRY_KEYS)
    if registry["schema_version"] != PARENT_REGISTRY_SCHEMA:
        raise ValueError("parent registry schema is invalid")
    campaign = _exact_mapping(
        registry["campaign"],
        name="parent registry campaign",
        keys=_CAMPAIGN_KEYS,
    )
    producer = _exact_mapping(
        campaign["producer"],
        name="parent producer",
        keys={"git_commit", "image_sha256", "hf_inventory_sha256"},
    )
    if type(producer["git_commit"]) is not str or _GIT_RE.fullmatch(producer["git_commit"]) is None:
        raise ValueError("parent producer git commit is invalid")
    _digest(producer["image_sha256"], name="parent producer image SHA256")
    _digest(producer["hf_inventory_sha256"], name="parent producer inventory SHA256")
    _digest(
        campaign["parent_phase2_design_sha256"],
        name="parent Phase-2 design SHA256",
    )
    if campaign["base_config_hash"] != science.settings.source_config_hash:
        raise ValueError("parent registry base config differs from R3 science")
    _digest(campaign["base_config_hash"], name="parent base config hash")
    if (
        type(campaign["source_job_array_id"]) is not str
        or not campaign["source_job_array_id"].isdigit()
        or campaign["failure_class"] != "primary_bt_mle_first_order_convergence_gate_not_met"
        or campaign["failed_optimizer_updates"] != 5760
        or campaign["first_order_tolerance"] != 0.001
        or campaign["consecutive_passes_required"] != 3
        or campaign["one_shot_no_further_adaptation"] is not True
        or campaign["allowed_recovery_scope"] != "train_only_same_materialized_artifacts"
    ):
        raise ValueError("parent registry campaign is not the frozen train-only parent")
    failure_aggregate = _exact_mapping(
        campaign["failure_aggregate"],
        name="parent failure aggregate",
        keys={"present", "reason", "replacement_evidence"},
    )
    if (
        failure_aggregate["present"] is not False
        or type(failure_aggregate["reason"]) is not str
        or not failure_aggregate["reason"]
        or type(failure_aggregate["replacement_evidence"]) is not str
        or not failure_aggregate["replacement_evidence"]
    ):
        raise ValueError("parent failure aggregate policy is invalid")

    common = _exact_mapping(
        registry["common_artifact_identities"],
        name="common artifact identities",
        keys={"eligible_prompt_ids_sha256", "tensor_schema"},
    )
    _digest(
        common["eligible_prompt_ids_sha256"],
        name="eligible prompt IDs SHA256",
    )
    tensor_schema = _exact_mapping(
        common["tensor_schema"],
        name="common tensor schema",
        keys=_COMMON_TENSOR_SCHEMA_KEYS,
    )
    if tensor_schema["num_tensor_keys"] != len(_ALL_TENSOR_KEYS):
        raise ValueError("common tensor schema has an invalid key count")
    for name in _COMMON_TENSOR_SCHEMA_KEYS - {"num_tensor_keys"}:
        shape = tensor_schema[name]
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"common tensor schema {name} is invalid")

    # The diagnostic is not consumed by R3 input materialization, but it is
    # part of the caller-hash-bound registry and must at least remain strict
    # JSON with its content identities well formed.
    diagnostic = registry["optimizer_diagnostic"]
    if not isinstance(diagnostic, Mapping):
        raise TypeError("optimizer diagnostic registry entry must be a mapping")
    for name in ("sha256", "artifact_metadata_sha256"):
        _digest(diagnostic.get(name), name=f"optimizer diagnostic {name}")

    entries = registry["seeds"]
    if not isinstance(entries, list):
        raise TypeError("parent registry seeds must be a list")
    expected_seeds = tuple(science.settings.seeds)
    observed_seeds = tuple(
        entry.get("seed") if isinstance(entry, Mapping) else None for entry in entries
    )
    if observed_seeds != expected_seeds:
        raise ValueError("parent registry seeds differ from the R3 science seed order")
    selected: dict[str, Any] | None = None
    for index, raw_entry in enumerate(entries):
        entry = _exact_mapping(
            raw_entry,
            name=f"parent seed entry {index}",
            keys=_SEED_KEYS,
        )
        if entry["array_task_id"] != index:
            raise ValueError("parent seed array-task mapping is invalid")
        _safe_relative(entry["source_run"], name=f"seed {entry['seed']} source_run")
        _safe_relative(
            entry["source_artifact"],
            name=f"seed {entry['seed']} source_artifact",
        )
        entry["evidence_sha256"] = _validate_hash_mapping(
            entry["evidence_sha256"],
            name=f"seed {entry['seed']} evidence",
            keys=_EVIDENCE_FILES,
        )
        entry["artifact_sha256"] = _validate_hash_mapping(
            entry["artifact_sha256"],
            name=f"seed {entry['seed']} artifact",
            keys=_ARTIFACT_FILES | _ARTIFACT_DERIVED_DIGESTS,
        )
        if entry["seed"] == seed:
            selected = entry
    if selected is None:
        raise ValueError("requested seed is absent from the parent registry")
    return campaign, common, selected


def _validate_manifest(
    value: object,
    *,
    campaign: Mapping[str, object],
    seed: int,
    task_index: int,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("parent run manifest must be a mapping")
    if (
        value.get("schema_version") != "smart-reward-run/v1"
        or value.get("config_hash") != campaign["base_config_hash"]
        or value.get("selected_seed") != seed
    ):
        raise ValueError("parent run manifest identity is invalid")
    producer = campaign["producer"]
    git = value.get("git")
    slurm = value.get("slurm")
    if (
        not isinstance(git, Mapping)
        or git.get("commit") != producer["git_commit"]  # type: ignore[index]
        or git.get("dirty") is not False
        or not isinstance(slurm, Mapping)
    ):
        raise ValueError("parent run manifest producer is invalid")
    expected = {
        "PRORM_GIT_COMMIT": producer["git_commit"],  # type: ignore[index]
        "PRORM_IMAGE_SHA256": producer["image_sha256"],  # type: ignore[index]
        "PRORM_HF_INVENTORY_SHA256": producer["hf_inventory_sha256"],  # type: ignore[index]
        "SLURM_CLUSTER_NAME": "hpc4",
        "SLURM_JOB_ACCOUNT": "sigroup",
        "SLURM_JOB_PARTITION": "gpu-l20",
        "SLURM_ARRAY_JOB_ID": campaign["source_job_array_id"],
        "SLURM_ARRAY_TASK_ID": str(task_index),
        "SLURM_GPUS_ON_NODE": "1",
        "SLURM_NNODES": "1",
        "SLURM_NTASKS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    if any(slurm.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError("parent run manifest is not the bound single-GPU HPC4 task")


def _validate_artifact_binding(
    value: object,
    *,
    campaign: Mapping[str, object],
    seed: int,
    metadata_sha256: str,
) -> None:
    binding = _exact_mapping(
        value,
        name="artifact materialization binding",
        keys={
            "schema_version",
            "mode",
            "base_config_hash",
            "phase2_design_sha256",
            "seed",
            "artifact_metadata_sha256",
            "producer",
        },
    )
    expected = {
        "schema_version": ARTIFACT_BINDING_SCHEMA,
        "base_config_hash": campaign["base_config_hash"],
        "phase2_design_sha256": campaign["parent_phase2_design_sha256"],
        "seed": seed,
        "artifact_metadata_sha256": metadata_sha256,
        "producer": campaign["producer"],
    }
    if binding["mode"] not in {"materialized", "reused"} or any(
        binding.get(name) != expected_value for name, expected_value in expected.items()
    ):
        raise ValueError("artifact materialization binding is invalid")


def _validate_artifact_verification(
    value: object,
    *,
    campaign: Mapping[str, object],
    seed: int,
    metadata_sha256: str,
) -> None:
    verification = _exact_mapping(
        value,
        name="artifact verification",
        keys={
            "status",
            "seed",
            "phase2_design_sha256",
            "base_config_hash",
            "formal_environment",
            "artifact_metadata_sha256",
        },
    )
    expected = {
        "status": "ok",
        "seed": seed,
        "phase2_design_sha256": campaign["parent_phase2_design_sha256"],
        "base_config_hash": campaign["base_config_hash"],
        "formal_environment": True,
        "artifact_metadata_sha256": metadata_sha256,
    }
    if verification != expected:
        raise ValueError("artifact verification identity is invalid")


def _validate_prompt_ids(value: object, *, split: str) -> tuple[str | int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"metadata split {split} prompt_ids must be a non-empty list")
    result: list[str | int] = []
    for index, prompt_id in enumerate(value):
        if isinstance(prompt_id, bool) or not isinstance(prompt_id, (str, int)):
            raise TypeError(f"metadata {split} prompt ID {index} is invalid")
        if isinstance(prompt_id, str) and not prompt_id:
            raise ValueError(f"metadata {split} prompt ID {index} is empty")
        result.append(prompt_id)
    if len(set(result)) != len(result):
        raise ValueError(f"metadata split {split} contains duplicate prompt IDs")
    return tuple(result)


def _validate_metadata(
    value: object,
    *,
    campaign: Mapping[str, object],
    common: Mapping[str, object],
    artifact_hashes: Mapping[str, str],
    seed: int,
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[tuple[int, ...], str]],
    str,
    RobustOracleTransform,
]:
    metadata = _exact_mapping(value, name="artifact metadata", keys=_METADATA_KEYS)
    if (
        metadata["schema"] != ARTIFACT_SCHEMA
        or metadata["config_hash"] != campaign["base_config_hash"]
        or metadata["seed"] != seed
        or metadata["tensor_sha256"] != artifact_hashes["tensors.safetensors"]
    ):
        raise ValueError("artifact metadata base identity is invalid")
    splits = _exact_mapping(
        metadata["splits"],
        name="artifact metadata splits",
        keys={"train", "validation", "test"},
    )
    prompt_ids: dict[str, tuple[str | int, ...]] = {}
    for split in ("train", "validation", "test"):
        split_value = _exact_mapping(
            splits[split],
            name=f"artifact metadata split {split}",
            keys={"prompt_ids"},
        )
        prompt_ids[split] = _validate_prompt_ids(
            split_value["prompt_ids"],
            split=split,
        )
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if set(prompt_ids[left]).intersection(prompt_ids[right]):
            raise ValueError("artifact metadata prompt splits overlap")
    if any(type(prompt_id) is not str for prompt_id in prompt_ids["train"]):
        raise ValueError("R3 train-only materialization requires string train prompt IDs")

    raw_specs = _exact_mapping(
        metadata["tensors"],
        name="artifact tensor metadata",
        keys=_ALL_TENSOR_KEYS,
    )
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for key in sorted(_ALL_TENSOR_KEYS):
        raw = _exact_mapping(
            raw_specs[key],
            name=f"artifact tensor metadata {key}",
            keys={"shape", "dtype"},
        )
        shape = raw["shape"]
        dtype = raw["dtype"]
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"artifact tensor metadata {key} shape is invalid")
        if type(dtype) is not str or _DTYPE_RE.fullmatch(dtype) is None:
            raise ValueError(f"artifact tensor metadata {key} dtype is invalid")
        specs[key] = (tuple(shape), dtype)

    common_schema = common["tensor_schema"]
    expected_shapes = {
        "train.policy_scores": common_schema["train_policy_scores_shape"],
        "train.reward_features": common_schema["train_reward_features_shape"],
        "validation.policy_scores": common_schema["validation_policy_scores_shape"],
        "validation.reward_features": common_schema["validation_reward_features_shape"],
        "test.policy_scores": common_schema["test_policy_scores_shape"],
        "test.reward_features": common_schema["test_reward_features_shape"],
    }
    for key, expected_shape in expected_shapes.items():
        if list(specs[key][0]) != expected_shape:
            raise ValueError(f"artifact tensor shape differs from registry for {key}")

    evidence = metadata["evidence"]
    if not isinstance(evidence, Mapping):
        raise TypeError("artifact metadata evidence must be a mapping")
    jsonl_hashes = _validate_hash_mapping(
        evidence.get("jsonl_sha256"),
        name="artifact metadata JSONL inventory",
        keys=_JSONL_FILES,
    )
    if any(jsonl_hashes[name] != artifact_hashes[name] for name in _JSONL_FILES):
        raise ValueError("artifact metadata JSONL hashes differ from the registry")
    if (
        evidence.get("schema") != MATERIALIZATION_SCHEMA
        or evidence.get("config_sha256") != campaign["base_config_hash"]
        or evidence.get("seed") != seed
        or evidence.get("producer") != campaign["producer"]
    ):
        raise ValueError("artifact materialization evidence identity is invalid")
    oracle_template_sha = _digest(
        evidence.get("oracle_chat_template_sha256"),
        name="oracle chat template SHA256",
    )
    raw_transform = _exact_mapping(
        evidence.get("oracle_transform"),
        name="artifact oracle transform",
        keys={"b", "tau"},
    )
    transform = RobustOracleTransform(
        b=raw_transform["b"],
        tau=raw_transform["tau"],
    )
    return (
        tuple(prompt_ids["train"]),  # type: ignore[arg-type]
        specs,
        oracle_template_sha,
        transform,
    )


def _default_safe_open(*args: object, **kwargs: object) -> _SafeTensorHandle:
    try:
        from safetensors import safe_open
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError("R3 train-only materialization requires safetensors") from error
    return safe_open(*args, **kwargs)


def _default_line_stream(path: Path) -> _BinaryLineStream:
    # FileIO's buffering=0 makes each readline consume exactly the requested
    # prefix bytes instead of prefetching held-out candidate records.
    return path.open("rb", buffering=0)


def _load_train_tensors_only(
    path: Path,
    *,
    expected_identity: _FileIdentity,
    prompt_ids: tuple[str, ...],
    specs: Mapping[str, tuple[tuple[int, ...], str]],
    safe_open_factory: _SafeOpenFactory,
) -> TrainingTensorData:
    if _require_regular_file(path, name="artifact tensors") != expected_identity:
        raise ValueError("artifact tensors changed after byte verification")
    decoded: dict[str, torch.Tensor] = {}
    try:
        with safe_open_factory(
            os.fspath(path),
            framework="pt",
            device="cpu",
        ) as handle:
            if frozenset(handle.keys()) != _ALL_TENSOR_KEYS:
                raise ValueError("safetensors keys differ from the Phase-1 schema")
            for key in _STORAGE_TRAIN_KEYS:
                tensor = handle.get_tensor(key)
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"safetensors entry {key} is not a tensor")
                expected_shape, expected_dtype = specs[key]
                actual_dtype = str(tensor.dtype).removeprefix("torch.")
                if tuple(tensor.shape) != expected_shape or actual_dtype != expected_dtype:
                    raise ValueError(f"train tensor {key} differs from artifact metadata")
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"train tensor {key} contains NaN or infinity")
                decoded[key] = tensor.detach().to(device="cpu").contiguous().clone()
    except (TypeError, ValueError):
        raise
    except Exception as error:
        raise ValueError("tensors.safetensors could not be selectively decoded") from error
    if _require_regular_file(path, name="artifact tensors") != expected_identity:
        raise ValueError("artifact tensors changed while train keys were decoded")
    return TrainingTensorData(
        prompt_ids=prompt_ids,
        policy_scores=decoded["train.policy_scores"],
        reward_features=decoded["train.reward_features"],
        h=decoded["train.h"],
        left_wins=decoded["train.left_wins"],
        num_annotations=decoded["train.num_annotations"],
    )


def _load_train_candidate_prefix_only(
    path: Path,
    *,
    expected_identity: _FileIdentity,
    train: TrainingTensorData,
    line_stream_factory: _LineStreamFactory,
) -> tuple[tuple[CandidateNode, ...], str]:
    if _require_regular_file(path, name="artifact candidates") != expected_identity:
        raise ValueError("artifact candidates changed after byte verification")
    count = train.num_prompts * train.num_candidates
    records: list[CandidateNode] = []
    prefix_digest = hashlib.sha256()
    with line_stream_factory(path) as stream:
        opened = _file_identity(os.fstat(stream.fileno()))
        if opened != expected_identity:
            raise ValueError("artifact candidates changed while opening the train prefix")
        for index in range(count):
            raw = stream.readline()
            if not raw or not raw.endswith(b"\n"):
                raise ValueError("candidate JSONL ended inside the train prefix")
            prefix_digest.update(raw)
            try:
                value = _decode_json_object(
                    raw,
                    name=f"train candidate record {index}",
                )
                records.append(CandidateNode.from_dict(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid train candidate record {index}") from error
        closed_identity = _file_identity(os.fstat(stream.fileno()))
        if closed_identity != expected_identity:
            raise ValueError("artifact candidates changed while reading the train prefix")
    if _require_regular_file(path, name="artifact candidates") != expected_identity:
        raise ValueError("artifact candidates changed after reading the train prefix")

    for flat_index, candidate in enumerate(records):
        prompt_index, candidate_index = divmod(flat_index, train.num_candidates)
        prompt_id = train.prompt_ids[prompt_index]
        if (
            candidate.prompt_id != prompt_id
            or candidate.candidate_id != f"{prompt_id}::candidate::{candidate_index}"
        ):
            raise ValueError("train candidate prefix is not in canonical tensor order")
    return tuple(records), prefix_digest.hexdigest()


def _load_source_config(path: Path, *, expected_hash: str) -> dict[str, Any]:
    before = _require_regular_file(path, name="source Phase-1 config")
    source_config = load_config(path)
    after = _require_regular_file(path, name="source Phase-1 config")
    if before != after:
        raise ValueError("source Phase-1 config changed while it was loaded")
    observed_hash = config_hash(source_config)
    if observed_hash != expected_hash:
        raise ValueError("source Phase-1 config semantic hash differs from the parent registry")
    return source_config


def _clock_value(clock_ns: _Clock, *, name: str) -> int:
    value = clock_ns()
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must return a nonnegative integer nanosecond value")
    return value


def _elapsed_seconds(start: int, end: int, *, name: str) -> float:
    if end <= start:
        raise ValueError(f"{name} timing must be strictly monotonic")
    return (end - start) / 1.0e9


def _timing_payload(
    *,
    materialization_attestation_sha256: str,
    artifact_verification_wall_seconds: float,
    oracle_rescore_wall_seconds: float,
    label_reconstruction_wall_seconds: float,
) -> dict[str, object]:
    artifact = _positive_seconds(
        artifact_verification_wall_seconds,
        name="artifact_verification_wall_seconds",
    )
    oracle = _positive_seconds(
        oracle_rescore_wall_seconds,
        name="oracle_rescore_wall_seconds",
    )
    labels = _positive_seconds(
        label_reconstruction_wall_seconds,
        name="label_reconstruction_wall_seconds",
    )
    return {
        "schema_version": R3_INPUT_PREPARATION_TIMINGS_SCHEMA,
        "materialization_attestation_sha256": _digest(
            materialization_attestation_sha256,
            name="materialization_attestation_sha256",
        ),
        "artifact_verification_wall_seconds": artifact,
        "oracle_rescore_wall_seconds": oracle,
        "label_reconstruction_wall_seconds": labels,
        "total_preparation_wall_seconds": math.fsum((artifact, oracle, labels)),
        "source_artifacts_reverified": True,
        "local_only_oracle_session": True,
        "policy_session_opened": False,
        "heldout_bytes_decoded": False,
    }


def _capability_payload(
    *,
    materialization_attestation_sha256: str,
    seed: int,
    source_config_hash: str,
    parent_registry_file_sha256: str,
    parent_seed_entry_sha256: str,
    artifact_metadata_sha256: str,
    artifact_tensors_sha256: str,
    artifact_candidates_sha256: str,
    candidate_train_prefix_sha256: str,
    artifact_materialization_sha256: str,
    artifact_verification_sha256: str,
    source_run_manifest_sha256: str,
    source_producer_identity_sha256: str,
    oracle_chat_template_sha256: str,
    oracle_transform_sha256: str,
    oracle_reward_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": R3_TRAIN_MATERIALIZATION_CAPABILITY_SCHEMA,
        "materialization_attestation_sha256": _digest(
            materialization_attestation_sha256,
            name="materialization_attestation_sha256",
        ),
        "seed": _exact_int(seed, name="capability seed"),
        "source_config_hash": _digest(
            source_config_hash,
            name="source_config_hash",
        ),
        "parent_registry_file_sha256": _digest(
            parent_registry_file_sha256,
            name="parent_registry_file_sha256",
        ),
        "parent_seed_entry_sha256": _digest(
            parent_seed_entry_sha256,
            name="parent_seed_entry_sha256",
        ),
        "artifact_metadata_sha256": _digest(
            artifact_metadata_sha256,
            name="artifact_metadata_sha256",
        ),
        "artifact_tensors_sha256": _digest(
            artifact_tensors_sha256,
            name="artifact_tensors_sha256",
        ),
        "artifact_candidates_sha256": _digest(
            artifact_candidates_sha256,
            name="artifact_candidates_sha256",
        ),
        "candidate_train_prefix_sha256": _digest(
            candidate_train_prefix_sha256,
            name="candidate_train_prefix_sha256",
        ),
        "artifact_materialization_sha256": _digest(
            artifact_materialization_sha256,
            name="artifact_materialization_sha256",
        ),
        "artifact_verification_sha256": _digest(
            artifact_verification_sha256,
            name="artifact_verification_sha256",
        ),
        "source_run_manifest_sha256": _digest(
            source_run_manifest_sha256,
            name="source_run_manifest_sha256",
        ),
        "source_producer_identity_sha256": _digest(
            source_producer_identity_sha256,
            name="source_producer_identity_sha256",
        ),
        "oracle_chat_template_sha256": _digest(
            oracle_chat_template_sha256,
            name="oracle_chat_template_sha256",
        ),
        "oracle_transform_sha256": _digest(
            oracle_transform_sha256,
            name="oracle_transform_sha256",
        ),
        "oracle_reward_sha256": _digest(
            oracle_reward_sha256,
            name="oracle_reward_sha256",
        ),
        "byte_sources_reverified": True,
        "train_tensor_keys_decoded": list(TRAIN_TENSOR_KEYS),
        "heldout_tensor_values_decoded": False,
        "candidate_suffix_decoded": False,
        "policy_session_opened": False,
    }


@dataclass(frozen=True, slots=True)
class R3TrainMaterializationCapability:
    """Process-local authority issued only by the real R3 input validator.

    ``_seal`` is deliberately ``init=False``.  A normal constructor call and
    ``dataclasses.replace`` therefore receive the default invalid seal and
    fail.  Cross-process code must re-run this module's byte validators rather
    than deserialize a capability claim.
    """

    materialization: ValidatedR3Materialization = field(repr=False, compare=False)
    schema_version: str
    materialization_attestation_sha256: str
    seed: int
    source_config_hash: str
    parent_registry_file_sha256: str
    parent_seed_entry_sha256: str
    artifact_metadata_sha256: str
    artifact_tensors_sha256: str
    artifact_candidates_sha256: str
    candidate_train_prefix_sha256: str
    artifact_materialization_sha256: str
    artifact_verification_sha256: str
    source_run_manifest_sha256: str
    source_producer_identity_sha256: str
    oracle_chat_template_sha256: str
    oracle_transform_sha256: str
    oracle_reward_sha256: str
    byte_sources_reverified: bool
    train_tensor_keys_decoded: tuple[str, ...]
    heldout_tensor_values_decoded: bool
    candidate_suffix_decoded: bool
    policy_session_opened: bool
    capability_sha256: str
    _seal: object = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if self._seal is not _CAPABILITY_SEAL:
            raise TypeError(
                "R3TrainMaterializationCapability must be issued by the train-only byte validator"
            )
        if type(self.materialization) is not ValidatedR3Materialization:
            raise TypeError("capability materialization has an invalid type")
        self.materialization.validate_integrity()
        expected = _capability_payload(
            materialization_attestation_sha256=(self.materialization_attestation_sha256),
            seed=self.seed,
            source_config_hash=self.source_config_hash,
            parent_registry_file_sha256=self.parent_registry_file_sha256,
            parent_seed_entry_sha256=self.parent_seed_entry_sha256,
            artifact_metadata_sha256=self.artifact_metadata_sha256,
            artifact_tensors_sha256=self.artifact_tensors_sha256,
            artifact_candidates_sha256=self.artifact_candidates_sha256,
            candidate_train_prefix_sha256=self.candidate_train_prefix_sha256,
            artifact_materialization_sha256=(self.artifact_materialization_sha256),
            artifact_verification_sha256=self.artifact_verification_sha256,
            source_run_manifest_sha256=self.source_run_manifest_sha256,
            source_producer_identity_sha256=(self.source_producer_identity_sha256),
            oracle_chat_template_sha256=self.oracle_chat_template_sha256,
            oracle_transform_sha256=self.oracle_transform_sha256,
            oracle_reward_sha256=self.oracle_reward_sha256,
        )
        observed = {
            "schema_version": self.schema_version,
            "materialization_attestation_sha256": (self.materialization_attestation_sha256),
            "seed": self.seed,
            "source_config_hash": self.source_config_hash,
            "parent_registry_file_sha256": self.parent_registry_file_sha256,
            "parent_seed_entry_sha256": self.parent_seed_entry_sha256,
            "artifact_metadata_sha256": self.artifact_metadata_sha256,
            "artifact_tensors_sha256": self.artifact_tensors_sha256,
            "artifact_candidates_sha256": self.artifact_candidates_sha256,
            "candidate_train_prefix_sha256": (self.candidate_train_prefix_sha256),
            "artifact_materialization_sha256": (self.artifact_materialization_sha256),
            "artifact_verification_sha256": (self.artifact_verification_sha256),
            "source_run_manifest_sha256": self.source_run_manifest_sha256,
            "source_producer_identity_sha256": (self.source_producer_identity_sha256),
            "oracle_chat_template_sha256": self.oracle_chat_template_sha256,
            "oracle_transform_sha256": self.oracle_transform_sha256,
            "oracle_reward_sha256": self.oracle_reward_sha256,
            "byte_sources_reverified": self.byte_sources_reverified,
            "train_tensor_keys_decoded": list(self.train_tensor_keys_decoded),
            "heldout_tensor_values_decoded": (self.heldout_tensor_values_decoded),
            "candidate_suffix_decoded": self.candidate_suffix_decoded,
            "policy_session_opened": self.policy_session_opened,
        }
        if observed != expected:
            raise ValueError("R3 train materialization capability is not closed")
        provenance = self.materialization.provenance
        exact_bindings = {
            "materialization_attestation_sha256": (self.materialization.attestation_sha256),
            "seed": self.materialization.seed,
            "source_config_hash": (self.materialization.science_bundle.settings.source_config_hash),
            "parent_registry_file_sha256": (provenance.parent_artifact_registry_sha256),
            "artifact_metadata_sha256": provenance.artifact_metadata_sha256,
            "artifact_tensors_sha256": provenance.artifact_tensors_sha256,
            "artifact_candidates_sha256": provenance.artifact_candidates_sha256,
            "candidate_train_prefix_sha256": (provenance.candidate_train_prefix_sha256),
            "artifact_materialization_sha256": (provenance.artifact_materialization_sha256),
            "artifact_verification_sha256": (provenance.artifact_verification_sha256),
            "source_run_manifest_sha256": (provenance.source_run_manifest_sha256),
            "source_producer_identity_sha256": (provenance.source_producer_identity_sha256),
            "oracle_reward_sha256": provenance.oracle_reward_sha256,
        }
        if any(
            observed.get(name) != expected_value for name, expected_value in exact_bindings.items()
        ):
            raise ValueError(
                "R3 train materialization capability differs from its validated materialization"
            )
        _digest(self.capability_sha256, name="capability_sha256")
        if self.capability_sha256 != _canonical_sha256(expected):
            raise ValueError("R3 train materialization capability self-hash is invalid")

    def validate_integrity(self) -> None:
        self.__post_init__()

    def to_dict(self) -> dict[str, object]:
        payload = _capability_payload(
            materialization_attestation_sha256=(self.materialization_attestation_sha256),
            seed=self.seed,
            source_config_hash=self.source_config_hash,
            parent_registry_file_sha256=self.parent_registry_file_sha256,
            parent_seed_entry_sha256=self.parent_seed_entry_sha256,
            artifact_metadata_sha256=self.artifact_metadata_sha256,
            artifact_tensors_sha256=self.artifact_tensors_sha256,
            artifact_candidates_sha256=self.artifact_candidates_sha256,
            candidate_train_prefix_sha256=self.candidate_train_prefix_sha256,
            artifact_materialization_sha256=(self.artifact_materialization_sha256),
            artifact_verification_sha256=self.artifact_verification_sha256,
            source_run_manifest_sha256=self.source_run_manifest_sha256,
            source_producer_identity_sha256=(self.source_producer_identity_sha256),
            oracle_chat_template_sha256=self.oracle_chat_template_sha256,
            oracle_transform_sha256=self.oracle_transform_sha256,
            oracle_reward_sha256=self.oracle_reward_sha256,
        )
        return {**payload, "capability_sha256": self.capability_sha256}


def _issue_train_materialization_capability(
    *,
    materialization: ValidatedR3Materialization,
    source_config_hash: str,
    parent_registry_file_sha256: str,
    parent_seed_entry_sha256: str,
    artifact_metadata_sha256: str,
    artifact_tensors_sha256: str,
    artifact_candidates_sha256: str,
    candidate_train_prefix_sha256: str,
    artifact_materialization_sha256: str,
    artifact_verification_sha256: str,
    source_run_manifest_sha256: str,
    source_producer_identity_sha256: str,
    oracle_chat_template_sha256: str,
    oracle_transform_sha256: str,
) -> R3TrainMaterializationCapability:
    payload = _capability_payload(
        materialization_attestation_sha256=materialization.attestation_sha256,
        seed=materialization.seed,
        source_config_hash=source_config_hash,
        parent_registry_file_sha256=parent_registry_file_sha256,
        parent_seed_entry_sha256=parent_seed_entry_sha256,
        artifact_metadata_sha256=artifact_metadata_sha256,
        artifact_tensors_sha256=artifact_tensors_sha256,
        artifact_candidates_sha256=artifact_candidates_sha256,
        candidate_train_prefix_sha256=candidate_train_prefix_sha256,
        artifact_materialization_sha256=artifact_materialization_sha256,
        artifact_verification_sha256=artifact_verification_sha256,
        source_run_manifest_sha256=source_run_manifest_sha256,
        source_producer_identity_sha256=source_producer_identity_sha256,
        oracle_chat_template_sha256=oracle_chat_template_sha256,
        oracle_transform_sha256=oracle_transform_sha256,
        oracle_reward_sha256=materialization.oracle_reward_sha256,
    )
    capability = object.__new__(R3TrainMaterializationCapability)
    object.__setattr__(capability, "materialization", materialization)
    for name, value in payload.items():
        if name == "train_tensor_keys_decoded":
            value = tuple(value)  # type: ignore[arg-type]
        object.__setattr__(capability, name, value)
    object.__setattr__(
        capability,
        "capability_sha256",
        _canonical_sha256(payload),
    )
    object.__setattr__(capability, "_seal", _CAPABILITY_SEAL)
    capability.validate_integrity()
    return capability


@dataclass(frozen=True, slots=True)
class R3InputPreparationTimings:
    """Three upstream preparation components needed by formal profiling."""

    schema_version: str
    materialization_attestation_sha256: str
    artifact_verification_wall_seconds: float
    oracle_rescore_wall_seconds: float
    label_reconstruction_wall_seconds: float
    total_preparation_wall_seconds: float
    source_artifacts_reverified: bool
    local_only_oracle_session: bool
    policy_session_opened: bool
    heldout_bytes_decoded: bool
    timings_sha256: str

    def __post_init__(self) -> None:
        expected = _timing_payload(
            materialization_attestation_sha256=(self.materialization_attestation_sha256),
            artifact_verification_wall_seconds=(self.artifact_verification_wall_seconds),
            oracle_rescore_wall_seconds=self.oracle_rescore_wall_seconds,
            label_reconstruction_wall_seconds=(self.label_reconstruction_wall_seconds),
        )
        observed = {
            "schema_version": self.schema_version,
            "materialization_attestation_sha256": (self.materialization_attestation_sha256),
            "artifact_verification_wall_seconds": (self.artifact_verification_wall_seconds),
            "oracle_rescore_wall_seconds": self.oracle_rescore_wall_seconds,
            "label_reconstruction_wall_seconds": (self.label_reconstruction_wall_seconds),
            "total_preparation_wall_seconds": (self.total_preparation_wall_seconds),
            "source_artifacts_reverified": self.source_artifacts_reverified,
            "local_only_oracle_session": self.local_only_oracle_session,
            "policy_session_opened": self.policy_session_opened,
            "heldout_bytes_decoded": self.heldout_bytes_decoded,
        }
        if observed != expected:
            raise ValueError("R3 input preparation timings are not closed")
        _digest(self.timings_sha256, name="timings_sha256")
        if self.timings_sha256 != _canonical_sha256(expected):
            raise ValueError("R3 input preparation timings self-hash is invalid")

    def validate_integrity(self) -> None:
        self.__post_init__()

    def to_dict(self) -> dict[str, object]:
        payload = _timing_payload(
            materialization_attestation_sha256=(self.materialization_attestation_sha256),
            artifact_verification_wall_seconds=(self.artifact_verification_wall_seconds),
            oracle_rescore_wall_seconds=self.oracle_rescore_wall_seconds,
            label_reconstruction_wall_seconds=(self.label_reconstruction_wall_seconds),
        )
        return {**payload, "timings_sha256": self.timings_sha256}


@dataclass(frozen=True, slots=True)
class R3TrainOnlyMaterializationResult:
    """Validated train materialization plus non-authorizing timing evidence."""

    capability: R3TrainMaterializationCapability
    preparation_timings: R3InputPreparationTimings

    def __post_init__(self) -> None:
        if type(self.capability) is not R3TrainMaterializationCapability:
            raise TypeError("capability must be an exact R3TrainMaterializationCapability")
        if type(self.preparation_timings) is not R3InputPreparationTimings:
            raise TypeError("preparation_timings must be exact R3 timings")
        self.capability.validate_integrity()
        self.preparation_timings.validate_integrity()
        if (
            self.preparation_timings.materialization_attestation_sha256
            != self.capability.materialization.attestation_sha256
        ):
            raise ValueError("preparation timings bind another materialization")

    @property
    def materialization(self) -> ValidatedR3Materialization:
        """Expose tensors for compute; authority remains the sealed capability."""

        return self.capability.materialization

    def validate_integrity(self) -> None:
        self.__post_init__()

    def to_dict(self) -> dict[str, object]:
        return {
            "materialization_capability": self.capability.to_dict(),
            "preparation_timings": self.preparation_timings.to_dict(),
            "gate_p_capability_issued": False,
        }


def _materialize_r3_train_only_from_parent(
    *,
    project_root: str | os.PathLike[str],
    parent_registry_path: str | os.PathLike[str],
    expected_parent_registry_file_sha256: str,
    source_config_path: str | os.PathLike[str],
    science_bundle: R3ScienceConfigBundle,
    seed: int,
    target_device: torch.device,
    oracle_rescorer: _OracleRescorer,
    clock_ns: _Clock,
    safe_open_factory: _SafeOpenFactory,
    line_stream_factory: _LineStreamFactory,
) -> R3TrainOnlyMaterializationResult:
    """Dependency-injected implementation; production wraps it with CUDA."""

    if type(science_bundle) is not R3ScienceConfigBundle:
        raise TypeError("science_bundle must be an exact R3ScienceConfigBundle")
    science_bundle.validate_integrity()
    requested_seed = _exact_int(seed, name="seed")
    if requested_seed not in science_bundle.settings.seeds:
        raise ValueError("seed is not declared by the R3 science contract")
    if type(target_device) is not torch.device:
        raise TypeError("target_device must be an exact torch.device")
    if not callable(oracle_rescorer):
        raise TypeError("oracle_rescorer must be callable")
    if not callable(clock_ns):
        raise TypeError("clock_ns must be callable")

    root = _require_real_root(project_root)
    registry_path = _resolve_file_argument(
        root,
        parent_registry_path,
        name="parent registry",
    )
    config_path = _resolve_file_argument(
        root,
        source_config_path,
        name="source Phase-1 config",
    )
    expected_registry_sha = _digest(
        expected_parent_registry_file_sha256,
        name="expected parent registry file SHA256",
    )

    t0 = _clock_value(clock_ns, name="preparation clock")
    registry, registry_sha, _ = _read_json_with_digest(
        registry_path,
        name="parent registry",
        expected_sha256=expected_registry_sha,
    )
    campaign, common, entry = _validate_registry(
        registry,
        science=science_bundle,
        seed=requested_seed,
    )
    source_config = _load_source_config(
        config_path,
        expected_hash=campaign["base_config_hash"],
    )
    run_root = _resolve_inside(
        root,
        entry["source_run"],
        name="parent source run",
        directory=True,
    )
    artifact_root = _resolve_inside(
        root,
        entry["source_artifact"],
        name="parent source artifact",
        directory=True,
    )
    evidence_hashes = entry["evidence_sha256"]
    artifact_hashes = entry["artifact_sha256"]

    manifest, manifest_sha, _ = _read_json_with_digest(
        run_root / "run-manifest.json",
        name="parent run manifest",
        expected_sha256=evidence_hashes["run-manifest.json"],
    )
    _validate_manifest(
        manifest,
        campaign=campaign,
        seed=requested_seed,
        task_index=entry["array_task_id"],
    )
    metadata, metadata_sha, _ = _read_json_with_digest(
        artifact_root / "metadata.json",
        name="artifact metadata",
        expected_sha256=artifact_hashes["metadata.json"],
    )
    prompt_ids, specs, oracle_template_sha, transform = _validate_metadata(
        metadata,
        campaign=campaign,
        common=common,
        artifact_hashes=artifact_hashes,
        seed=requested_seed,
    )
    materialization_binding, binding_sha, _ = _read_json_with_digest(
        run_root / "artifact-materialization.json",
        name="artifact materialization binding",
        expected_sha256=evidence_hashes["artifact-materialization.json"],
    )
    _validate_artifact_binding(
        materialization_binding,
        campaign=campaign,
        seed=requested_seed,
        metadata_sha256=metadata_sha,
    )
    verification, verification_sha, _ = _read_json_with_digest(
        run_root / "artifact-verification.json",
        name="artifact verification",
        expected_sha256=evidence_hashes["artifact-verification.json"],
    )
    _validate_artifact_verification(
        verification,
        campaign=campaign,
        seed=requested_seed,
        metadata_sha256=metadata_sha,
    )

    tensor_path = artifact_root / "tensors.safetensors"
    tensor_sha, tensor_identity = _stable_file_digest(
        tensor_path,
        name="artifact tensors",
    )
    if tensor_sha != artifact_hashes["tensors.safetensors"]:
        raise ValueError("artifact tensor SHA256 differs from the parent registry")
    train_cpu = _load_train_tensors_only(
        tensor_path,
        expected_identity=tensor_identity,
        prompt_ids=prompt_ids,
        specs=specs,
        safe_open_factory=safe_open_factory,
    )

    candidate_path = artifact_root / "candidates.jsonl"
    candidate_sha, candidate_identity = _stable_file_digest(
        candidate_path,
        name="artifact candidates",
    )
    if candidate_sha != artifact_hashes["candidates.jsonl"]:
        raise ValueError("artifact candidate SHA256 differs from the parent registry")
    candidates, candidate_prefix_sha = _load_train_candidate_prefix_only(
        candidate_path,
        expected_identity=candidate_identity,
        train=train_cpu,
        line_stream_factory=line_stream_factory,
    )
    t1 = _clock_value(clock_ns, name="preparation clock")

    reward_model = source_config.get("reward_model")
    if not isinstance(reward_model, Mapping):
        raise ValueError("source config lacks reward_model")
    microbatch_size = _exact_int(
        reward_model.get("microbatch_size"),
        name="source oracle microbatch_size",
        minimum=1,
    )
    flat_rewards = oracle_rescorer(
        source_config=source_config,
        candidates=candidates,
        expected_chat_template_sha256=oracle_template_sha,
        transform=transform,
        batch_size=min(16, microbatch_size),
        device=target_device,
    )
    expected_count = train_cpu.num_prompts * train_cpu.num_candidates
    if (
        type(flat_rewards) is not torch.Tensor
        or tuple(flat_rewards.shape) != (expected_count,)
        or not flat_rewards.is_floating_point()
        or flat_rewards.requires_grad
        or not bool(torch.isfinite(flat_rewards).all())
    ):
        raise ValueError("train-only oracle rescore returned malformed values")
    frozen_rewards = flat_rewards.detach().to(
        device=target_device,
        dtype=train_cpu.policy_scores.dtype,
    )
    train_rewards = frozen_rewards.reshape(
        train_cpu.num_prompts,
        train_cpu.num_candidates,
    ).clone()
    del flat_rewards, frozen_rewards
    t2 = _clock_value(clock_ns, name="preparation clock")

    train = train_cpu.to(target_device)
    context = prepare_neutral_phase2_context(
        train,
        train_rewards,
        seed=requested_seed,
        settings=science_bundle.settings,
    )
    provenance = TrainMaterializationProvenance.from_context(
        context,
        parent_artifact_registry_sha256=registry_sha,
        artifact_metadata_sha256=metadata_sha,
        artifact_tensors_sha256=tensor_sha,
        artifact_candidates_sha256=candidate_sha,
        artifact_materialization_sha256=binding_sha,
        artifact_verification_sha256=verification_sha,
        source_run_manifest_sha256=manifest_sha,
        source_producer_identity_sha256=_canonical_sha256(campaign["producer"]),
        candidate_train_prefix_sha256=candidate_prefix_sha,
        candidate_train_prefix_count=len(candidates),
    )
    materialization = validate_r3_materialization(
        context,
        science_bundle=science_bundle,
        provenance=provenance,
    )
    producer_sha = _canonical_sha256(campaign["producer"])
    capability = _issue_train_materialization_capability(
        materialization=materialization,
        source_config_hash=campaign["base_config_hash"],
        parent_registry_file_sha256=registry_sha,
        parent_seed_entry_sha256=_canonical_sha256(entry),
        artifact_metadata_sha256=metadata_sha,
        artifact_tensors_sha256=tensor_sha,
        artifact_candidates_sha256=candidate_sha,
        candidate_train_prefix_sha256=candidate_prefix_sha,
        artifact_materialization_sha256=binding_sha,
        artifact_verification_sha256=verification_sha,
        source_run_manifest_sha256=manifest_sha,
        source_producer_identity_sha256=producer_sha,
        oracle_chat_template_sha256=oracle_template_sha,
        oracle_transform_sha256=_canonical_sha256({"b": transform.b, "tau": transform.tau}),
    )
    t3 = _clock_value(clock_ns, name="preparation clock")
    timing_payload = _timing_payload(
        materialization_attestation_sha256=materialization.attestation_sha256,
        artifact_verification_wall_seconds=_elapsed_seconds(
            t0,
            t1,
            name="artifact verification",
        ),
        oracle_rescore_wall_seconds=_elapsed_seconds(
            t1,
            t2,
            name="oracle rescore",
        ),
        label_reconstruction_wall_seconds=_elapsed_seconds(
            t2,
            t3,
            name="label reconstruction",
        ),
    )
    timings = R3InputPreparationTimings(
        **timing_payload,
        timings_sha256=_canonical_sha256(timing_payload),
    )
    result = R3TrainOnlyMaterializationResult(
        capability=capability,
        preparation_timings=timings,
    )
    result.validate_integrity()
    return result


def _production_oracle_rescore(
    *,
    source_config: Mapping[str, object],
    candidates: Sequence[CandidateNode],
    expected_chat_template_sha256: str,
    transform: RobustOracleTransform,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if device.type != "cuda":
        raise RuntimeError("production R3 oracle rescore requires CUDA")
    backend = HuggingFacePhase2Backend(
        source_config,
        device=device,
        local_files_only=True,
    )
    if backend.local_files_only is not True:
        raise RuntimeError("production R3 oracle backend is not local-only")
    prompts = tuple(candidate.prompt for candidate in candidates)
    responses = tuple(candidate.response for candidate in candidates)
    with backend.oracle_session(
        expected_chat_template_sha256=expected_chat_template_sha256,
    ) as oracle:
        return oracle.score_transformed(
            prompts,
            responses,
            transform=transform,
            batch_size=batch_size,
        )


def _require_single_cuda(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type != "cuda" or target.index not in {None, 0}:
        raise RuntimeError("formal R3 input materialization requires cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("formal R3 input materialization requires allocated CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "formal R3 input materialization requires exactly one visible CUDA device"
        )
    return torch.device("cuda:0")


def materialize_r3_train_only_from_parent(
    *,
    project_root: str | os.PathLike[str],
    parent_registry_path: str | os.PathLike[str],
    expected_parent_registry_file_sha256: str,
    source_config_path: str | os.PathLike[str],
    science_bundle: R3ScienceConfigBundle,
    seed: int,
    device: str | torch.device = "cuda",
) -> R3TrainOnlyMaterializationResult:
    """Materialize one formal R3 train input on exactly one visible CUDA GPU.

    This entry point does not accept an R2 recovery overlay, a policy backend,
    a held-out loader, or a caller-controlled ``formal`` flag.  The returned
    object is evidence for train materialization only; it cannot authorize
    Gate P.
    """

    target = _require_single_cuda(device)
    return _materialize_r3_train_only_from_parent(
        project_root=project_root,
        parent_registry_path=parent_registry_path,
        expected_parent_registry_file_sha256=(expected_parent_registry_file_sha256),
        source_config_path=source_config_path,
        science_bundle=science_bundle,
        seed=seed,
        target_device=target,
        oracle_rescorer=_production_oracle_rescore,
        clock_ns=time.perf_counter_ns,
        safe_open_factory=_default_safe_open,
        line_stream_factory=_default_line_stream,
    )


__all__ = [
    "R3_INPUT_PREPARATION_TIMINGS_SCHEMA",
    "R3_TRAIN_MATERIALIZATION_CAPABILITY_SCHEMA",
    "R3InputPreparationTimings",
    "R3TrainMaterializationCapability",
    "R3TrainOnlyMaterializationResult",
    "materialize_r3_train_only_from_parent",
]
