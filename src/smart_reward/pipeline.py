"""Validated, idempotent stages for local and Slurm execution."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import load_exact_delta_artifact
from .checkpoints import validate_stage_receipt, write_stage_receipt
from .config import config_hash, validate_config
from .exact_phase import materialize_exact_delta
from .exact_policy import SCHEMA as ADAPTER_SCHEMA
from .exact_policy import export_exact_ngd_adapters
from .exact_run import load_exact_reward_comparison, run_exact_reward_comparison
from .rollout import (
    assemble_policy_rollouts,
    evaluate_single_policy_rollout,
    policy_instance_names,
    validate_single_policy_rollout,
)
from .runtime import producer_identity, sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _receipt(root: Path, stage: str) -> Path:
    return root / "stage_receipts" / f"{stage}.json"


def _artifact_outputs(config: Mapping[str, object], artifact: Path, *, seed: int) -> dict[str, str]:
    digest = config_hash(config)
    _ = load_exact_delta_artifact(
        artifact,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    metadata = _read_json(artifact / "metadata.json")
    if metadata.get("evidence", {}).get("producer") != producer_identity():
        raise ValueError("artifact producer identity mismatch")
    recorded = metadata.get("evidence", {}).get("jsonl_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("artifact is missing JSONL identities")
    outputs = {
        "artifact_metadata": sha256_file(artifact / "metadata.json"),
        "artifact_tensors": sha256_file(artifact / "tensors.safetensors"),
    }
    for name in ("prompts.jsonl", "candidates.jsonl", "edges.jsonl"):
        observed = sha256_file(artifact / name)
        if recorded.get(name) != observed:
            raise ValueError(f"artifact JSONL digest mismatch: {name}")
        outputs[name.removesuffix(".jsonl")] = observed
    return outputs


def _ensure_receipt(
    path: Path,
    config: Mapping[str, object],
    *,
    stage: str,
    seed: int,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
) -> None:
    if path.exists():
        validate_stage_receipt(
            path,
            config,
            stage=stage,
            seed=seed,
            inputs=inputs,
            outputs=outputs,
        )
    else:
        write_stage_receipt(
            path,
            config,
            stage=stage,
            seed=seed,
            inputs=inputs,
            outputs=outputs,
        )


def run_materialization_stage(
    config: Mapping[str, object],
    seed_root: str | os.PathLike[str],
    *,
    seed: int,
    device: str = "cuda",
    local_files_only: bool = True,
) -> dict[str, str]:
    normalized = validate_config(config)
    root = Path(seed_root)
    artifact = root / "artifact"
    if not artifact.exists():
        print("stage=materialize status=running", flush=True)
        materialize_exact_delta(
            normalized,
            seed=seed,
            artifact_dir=artifact,
            device=device,
            local_files_only=local_files_only,
        )
    outputs = _artifact_outputs(normalized, artifact, seed=seed)
    _ensure_receipt(
        _receipt(root, "materialize"),
        normalized,
        stage="materialize",
        seed=seed,
        inputs={},
        outputs=outputs,
    )
    work = artifact.parent / f".{artifact.name}.materialize-work"
    if work.exists():
        shutil.rmtree(work)
    print("stage=materialize status=complete", flush=True)
    return outputs


def _validated_materialization(
    config: Mapping[str, object], root: Path, *, seed: int
) -> dict[str, str]:
    outputs = _artifact_outputs(config, root / "artifact", seed=seed)
    validate_stage_receipt(
        _receipt(root, "materialize"),
        config,
        stage="materialize",
        seed=seed,
        inputs={},
        outputs=outputs,
    )
    return outputs


def run_reward_stage(
    config: Mapping[str, object],
    seed_root: str | os.PathLike[str],
    *,
    seed: int,
    device: str = "cuda",
) -> dict[str, Any]:
    normalized = validate_config(config)
    root = Path(seed_root)
    artifact_outputs = _validated_materialization(normalized, root, seed=seed)
    result_path = root / "reward_result.json"
    if not result_path.exists():
        print("stage=reward status=running", flush=True)
        run_exact_reward_comparison(
            normalized,
            root / "artifact",
            result_path,
            seed=seed,
            device=device,
        )
    result = load_exact_reward_comparison(
        result_path,
        expected_config_hash=config_hash(normalized),
        expected_seed=seed,
    )
    if result.get("producer") != producer_identity():
        raise ValueError("reward result producer identity mismatch")
    if result["artifact_metadata_sha256"] != artifact_outputs["artifact_metadata"]:
        raise ValueError("reward result artifact identity mismatch")
    outputs = {"reward_result": sha256_file(result_path)}
    inputs = {"artifact_metadata": artifact_outputs["artifact_metadata"]}
    _ensure_receipt(
        _receipt(root, "reward"),
        normalized,
        stage="reward",
        seed=seed,
        inputs=inputs,
        outputs=outputs,
    )
    print("stage=reward status=complete", flush=True)
    return result


def _validated_reward(
    config: Mapping[str, object], root: Path, *, seed: int
) -> tuple[dict[str, Any], str]:
    artifact_outputs = _validated_materialization(config, root, seed=seed)
    path = root / "reward_result.json"
    result = load_exact_reward_comparison(
        path,
        expected_config_hash=config_hash(config),
        expected_seed=seed,
    )
    if result.get("producer") != producer_identity():
        raise ValueError("reward result producer identity mismatch")
    identity = sha256_file(path)
    validate_stage_receipt(
        _receipt(root, "reward"),
        config,
        stage="reward",
        seed=seed,
        inputs={"artifact_metadata": artifact_outputs["artifact_metadata"]},
        outputs={"reward_result": identity},
    )
    return result, identity


def _adapter_outputs(adapters: Path) -> dict[str, str]:
    metadata_path = adapters / "metadata.json"
    metadata = _read_json(metadata_path)
    if metadata.get("schema") != ADAPTER_SCHEMA:
        raise ValueError("unsupported adapter metadata")
    if metadata.get("producer") != producer_identity():
        raise ValueError("adapter producer identity mismatch")
    records = metadata.get("adapters")
    if not isinstance(records, dict) or len(records) != 9:
        raise ValueError("adapter metadata must describe exactly nine adapters")
    for name, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
            raise ValueError(f"adapter file inventory is missing: {name}")
        for relative, expected in record["files"].items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"adapter file inventory escapes its directory: {name}")
            path = adapters / name / relative
            if sha256_file(path) != expected:
                raise ValueError(f"adapter file digest mismatch: {name}/{relative}")
    return {"adapter_metadata": sha256_file(metadata_path)}


def run_adapter_stage(
    config: Mapping[str, object],
    seed_root: str | os.PathLike[str],
    *,
    seed: int,
    device: str = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    normalized = validate_config(config)
    root = Path(seed_root)
    artifact_outputs = _validated_materialization(normalized, root, seed=seed)
    _, reward_identity = _validated_reward(normalized, root, seed=seed)
    adapters = root / "adapters"
    if not adapters.exists():
        print("stage=adapters status=running", flush=True)
        export_exact_ngd_adapters(
            normalized,
            root / "artifact",
            root / "reward_result.json",
            adapters,
            seed=seed,
            device=device,
            local_files_only=local_files_only,
        )
    outputs = _adapter_outputs(adapters)
    metadata = _read_json(adapters / "metadata.json")
    if metadata.get("artifact_metadata_sha256") != artifact_outputs["artifact_metadata"]:
        raise ValueError("adapter artifact identity mismatch")
    if metadata.get("reward_result_sha256") != reward_identity:
        raise ValueError("adapter reward identity mismatch")
    _ensure_receipt(
        _receipt(root, "adapters"),
        normalized,
        stage="adapters",
        seed=seed,
        inputs={
            "artifact_metadata": artifact_outputs["artifact_metadata"],
            "reward_result": reward_identity,
        },
        outputs=outputs,
    )
    print("stage=adapters status=complete", flush=True)
    return metadata


def _validated_adapters(config: Mapping[str, object], root: Path, *, seed: int) -> dict[str, str]:
    artifact_outputs = _validated_materialization(config, root, seed=seed)
    _, reward_identity = _validated_reward(config, root, seed=seed)
    outputs = _adapter_outputs(root / "adapters")
    validate_stage_receipt(
        _receipt(root, "adapters"),
        config,
        stage="adapters",
        seed=seed,
        inputs={
            "artifact_metadata": artifact_outputs["artifact_metadata"],
            "reward_result": reward_identity,
        },
        outputs=outputs,
    )
    return outputs


def run_policy_rollout_stage(
    config: Mapping[str, object],
    seed_root: str | os.PathLike[str],
    *,
    policy_name: str,
    seed: int,
    device: str = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    normalized = validate_config(config)
    root = Path(seed_root)
    _validated_adapters(normalized, root, seed=seed)
    return evaluate_single_policy_rollout(
        normalized,
        root / "artifact",
        root / "adapters",
        root / "policy_rollout_parts" / policy_name,
        policy_name=policy_name,
        seed=seed,
        device=device,
        local_files_only=local_files_only,
    )


def _rollout_aggregate_inputs(
    config: Mapping[str, object], root: Path, *, seed: int
) -> dict[str, str]:
    normalized = validate_config(config)
    artifact_outputs = _validated_materialization(normalized, root, seed=seed)
    adapter_outputs = _validated_adapters(normalized, root, seed=seed)
    result = {
        "artifact_metadata": artifact_outputs["artifact_metadata"],
        "adapter_metadata": adapter_outputs["adapter_metadata"],
    }
    for name in policy_instance_names(normalized):
        validate_single_policy_rollout(
            normalized,
            root / "artifact",
            root / "adapters",
            root / "policy_rollout_parts" / name,
            policy_name=name,
            seed=seed,
        )
        result[f"policy_{name}"] = sha256_file(
            root / "policy_rollout_parts" / name / "receipt.json"
        )
    return result


def run_rollout_aggregate_stage(
    config: Mapping[str, object], seed_root: str | os.PathLike[str], *, seed: int
) -> dict[str, Any]:
    normalized = validate_config(config)
    root = Path(seed_root)
    inputs = _rollout_aggregate_inputs(normalized, root, seed=seed)
    target = root / "policy_utility"
    if not target.exists():
        print("stage=rollout-aggregate status=running", flush=True)
        payload = assemble_policy_rollouts(
            normalized,
            root / "artifact",
            root / "adapters",
            root / "policy_rollout_parts",
            target,
            seed=seed,
        )
    else:
        payload = _read_json(target / "metrics.json")
    outputs = {
        "metrics": sha256_file(target / "metrics.json"),
        "rollouts": sha256_file(target / "rollouts.jsonl"),
    }
    validate_stage_receipt(
        target / "receipt.json",
        normalized,
        stage="rollout-aggregate",
        seed=seed,
        inputs=inputs,
        outputs=outputs,
    )
    if payload.get("config_sha256") != config_hash(normalized) or payload.get("seed") != seed:
        raise ValueError("rollout aggregate identity mismatch")
    print("stage=rollout-aggregate status=complete", flush=True)
    return payload


__all__ = [
    "run_adapter_stage",
    "run_materialization_stage",
    "run_policy_rollout_stage",
    "run_reward_stage",
    "run_rollout_aggregate_stage",
]
