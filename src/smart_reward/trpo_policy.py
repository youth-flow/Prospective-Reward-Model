"""Export the nine initial matched-quadratic-KL TRPO LoRA-B adapters."""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .artifacts import exact_delta_artifact_metadata_sha256, load_exact_delta_artifact
from .config import TRPO_PROTOCOL, config_hash, validate_config
from .exact_policy import (
    _atomic_json,
    _canonical_sha256,
    _load_policy,
    _quarantine_adapter_component,
    _zero_b,
)
from .policy_update import set_tangent_update_
from .runtime import producer_identity, sha256_file
from .trpo_run import load_trpo_reward_comparison

SCHEMA = "prorm-trpo-adapters/v1"
COMPONENT_SCHEMA = "prorm-trpo-adapter-component/v1"


def adapter_name(method: str, kl_target: float) -> str:
    target_text = format(float(kl_target), "g").replace(".", "p")
    return f"{method}__kappa_{target_text}"


def _fixed_fields(
    *,
    config_sha256: str,
    artifact_metadata_sha256: str,
    reward_result_sha256: str,
    seed: int,
    name: str,
    reward_source: str,
    kl_target: float,
    update: torch.Tensor,
    initial_step_scale: float,
    lora_a_sha256: str,
    lora_layout: object,
    producer: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": COMPONENT_SCHEMA,
        "status": "complete",
        "calibration_status": "pending",
        "config_sha256": config_sha256,
        "artifact_metadata_sha256": artifact_metadata_sha256,
        "reward_result_sha256": reward_result_sha256,
        "seed": seed,
        "adapter_name": name,
        "reward_source": reward_source,
        "kl_target": kl_target,
        "initial_step_scale": initial_step_scale,
        "update_sha256": _canonical_sha256(
            update.detach().to(dtype=torch.float64, device="cpu").tolist()
        ),
        "update_norm": float(torch.linalg.vector_norm(update).item()),
        "lora_a_sha256": lora_a_sha256,
        "lora_layout_sha256": _canonical_sha256(lora_layout),
        "producer": dict(producer),
    }


def _component_record(receipt: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "reward_source": receipt["reward_source"],
        "kl_target": receipt["kl_target"],
        "initial_step_scale": receipt["initial_step_scale"],
        "update_norm": receipt["update_norm"],
        "calibration_status": receipt["calibration_status"],
        "files": dict(receipt["files"]),
        "component_receipt_sha256": digest,
    }


def _validate_component(
    target: Path,
    name: str,
    expected_fixed: Mapping[str, Any],
) -> dict[str, Any]:
    directory = target / name
    receipt_path = target / ".checkpoints" / f"{name}.json"
    if not directory.is_dir() or not receipt_path.is_file():
        raise FileNotFoundError(f"TRPO adapter checkpoint is incomplete: {name}")
    with receipt_path.open("r", encoding="utf-8") as stream:
        receipt = json.load(stream)
    if not isinstance(receipt, dict):
        raise ValueError(f"TRPO adapter receipt must be an object: {name}")
    files = receipt.get("files")
    fixed = {key: value for key, value in receipt.items() if key != "files"}
    if fixed != dict(expected_fixed):
        raise ValueError(f"TRPO adapter identity mismatch: {name}")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"TRPO adapter file inventory is missing: {name}")
    for relative, expected in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"TRPO adapter file escapes its directory: {name}")
        if sha256_file(directory / relative_path) != expected:
            raise ValueError(f"TRPO adapter digest mismatch: {name}/{relative}")
    return _component_record(receipt, sha256_file(receipt_path))


def validate_trpo_adapter_metadata(
    adapters: str | os.PathLike[str],
    *,
    expected_producer: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(adapters)
    with (root / "metadata.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported TRPO adapter metadata")
    if metadata.get("protocol") != TRPO_PROTOCOL:
        raise ValueError("TRPO adapter protocol mismatch")
    if metadata.get("producer") != dict(expected_producer):
        raise ValueError("TRPO adapter producer mismatch")
    records = metadata.get("adapters")
    if not isinstance(records, dict) or len(records) != 9:
        raise ValueError("TRPO metadata must describe exactly nine adapters")
    for name, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
            raise ValueError(f"TRPO adapter record is malformed: {name}")
        receipt_path = root / ".checkpoints" / f"{name}.json"
        if sha256_file(receipt_path) != record.get("component_receipt_sha256"):
            raise ValueError(f"TRPO adapter receipt digest mismatch: {name}")
        with receipt_path.open("r", encoding="utf-8") as stream:
            receipt = json.load(stream)
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != COMPONENT_SCHEMA
            or receipt.get("adapter_name") != name
            or receipt.get("producer") != dict(expected_producer)
            or _component_record(receipt, sha256_file(receipt_path)) != record
        ):
            raise ValueError(f"TRPO adapter component mismatch: {name}")
        for relative, expected in record["files"].items():
            if sha256_file(root / name / relative) != expected:
                raise ValueError(f"TRPO adapter file digest mismatch: {name}/{relative}")
    return metadata


def export_trpo_adapters(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    comparison_json: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Export nine initial adapters; realized-KL calibration is downstream."""

    normalized = validate_config(config)
    if normalized["protocol"] != TRPO_PROTOCOL or seed not in normalized["run"]["seeds"]:
        raise ValueError("TRPO adapter protocol or seed mismatch")
    digest = config_hash(normalized)
    artifact_identity = exact_delta_artifact_metadata_sha256(
        artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    _ = load_exact_delta_artifact(
        artifact_dir,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    comparison_path = Path(comparison_json)
    comparison = load_trpo_reward_comparison(
        comparison_path,
        expected_config_sha256=digest,
        expected_seed=seed,
    )
    if comparison["artifact_metadata_sha256"] != artifact_identity:
        raise ValueError("TRPO reward comparison belongs to another artifact")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    evidence = json.loads((Path(artifact_dir) / "metadata.json").read_text(encoding="utf-8"))[
        "evidence"
    ]
    lora_a_sha256 = evidence["policy_a_sha256"]
    lora_layout = evidence["policy_layout"]
    current_producer = producer_identity()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / ".checkpoints").mkdir(exist_ok=True)
    records: dict[str, Any] = {}
    missing: list[tuple[str, torch.Tensor, dict[str, Any]]] = []
    for method in ("mle_rm", "pro_rm", "oracle"):
        for target_raw in normalized["policy_update"]["kl_targets"]:
            kl_target = float(target_raw)
            update_record = comparison["policy_updates"][method][str(kl_target)]
            update = torch.tensor(
                update_record["update"],
                dtype=torch.float64,
                device=target_device,
            )
            name = adapter_name(method, kl_target)
            fixed = _fixed_fields(
                config_sha256=digest,
                artifact_metadata_sha256=artifact_identity,
                reward_result_sha256=sha256_file(comparison_path),
                seed=seed,
                name=name,
                reward_source=method,
                kl_target=kl_target,
                update=update,
                initial_step_scale=float(update_record["step_scale"]),
                lora_a_sha256=lora_a_sha256,
                lora_layout=lora_layout,
                producer=current_producer,
            )
            try:
                records[name] = _validate_component(target, name, fixed)
                print(f"adapter name={name} status=reused", flush=True)
            except (FileNotFoundError, ValueError):
                _quarantine_adapter_component(target, name)
                missing.append((name, update, fixed))
    setup: Any | None = None
    try:
        if missing:
            setup = _load_policy(normalized, seed, target_device, local_files_only)
            if setup.a_state_sha256 != lora_a_sha256:
                raise RuntimeError("reloaded fixed LoRA-A does not match materialization")
            if setup.layout.to_metadata() != lora_layout:
                raise RuntimeError("reloaded LoRA-B layout does not match materialization")
        for name, update, fixed in missing:
            assert setup is not None
            _zero_b(setup)
            set_tangent_update_(
                setup.named_tangent_parameters(),
                setup.layout,
                update,
                step_size=1.0,
            )
            staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=target))
            try:
                setup.model.save_pretrained(staging, safe_serialization=True)
                files = {
                    path.relative_to(staging).as_posix(): sha256_file(path)
                    for path in sorted(staging.rglob("*"))
                    if path.is_file()
                }
                if not files:
                    raise RuntimeError(f"TRPO adapter serialization produced no files: {name}")
                os.replace(staging, target / name)
                receipt = {**fixed, "files": files}
                receipt_path = target / ".checkpoints" / f"{name}.json"
                _atomic_json(receipt_path, receipt)
                records[name] = _component_record(receipt, sha256_file(receipt_path))
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        metadata = {
            "schema": SCHEMA,
            "protocol": TRPO_PROTOCOL,
            "config_sha256": digest,
            "artifact_metadata_sha256": artifact_identity,
            "reward_result_sha256": sha256_file(comparison_path),
            "seed": seed,
            "kl_targets": [float(value) for value in normalized["policy_update"]["kl_targets"]],
            "calibration_status": "pending",
            "updated_adapter_count": len(records),
            "lora_a_sha256": lora_a_sha256,
            "lora_layout": lora_layout,
            "adapters": records,
            "producer": current_producer,
        }
        metadata_path = target / "metadata.json"
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing == metadata:
                return metadata
            rejected = Path(tempfile.mkdtemp(prefix="metadata.", dir=target / ".rejected"))
            os.replace(metadata_path, rejected / "metadata.json")
        _atomic_json(metadata_path, metadata)
    finally:
        if setup is not None:
            _zero_b(setup)
            del setup
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    return metadata


__all__ = [
    "COMPONENT_SCHEMA",
    "SCHEMA",
    "adapter_name",
    "export_trpo_adapters",
    "validate_trpo_adapter_metadata",
]
