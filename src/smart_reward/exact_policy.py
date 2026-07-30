"""One-step common-beta NGD updates in the fixed LoRA-B tangent."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    exact_delta_artifact_metadata_sha256,
    load_exact_delta_artifact,
)
from .config import PROTOCOL, config_hash, validate_config
from .exact_run import load_exact_reward_comparison
from .hf import configure_fixed_a_lora
from .policy_update import set_tangent_update_
from .runtime import (
    fork_torch_seed,
    load_pretrained,
    producer_identity,
    require_module,
    sha256_file,
)
from .seeding import SeedBundle

SCHEMA = "prorm-ngd-adapters/v1"
COMPONENT_SCHEMA = "prorm-ngd-adapter-component/v1"


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


@torch.no_grad()
def _zero_b(setup: Any) -> None:
    for _, parameter in setup.named_tangent_parameters():
        parameter.zero_()


def _load_policy(config: Mapping[str, Any], seed: int, device: torch.device, local: bool) -> Any:
    transformers = require_module("transformers")
    peft = require_module("peft")
    policy = config["policy"]
    seeds = SeedBundle.from_base_seed(seed)
    with fork_torch_seed(seeds.policy_lora_a, device):
        model = load_pretrained(
            transformers.AutoModelForCausalLM,
            policy["model"],
            policy["revision"],
            local_files_only=local,
            kind="policy model",
            torch_dtype=_dtype(policy["dtype"]),
        )
        lora = peft.LoraConfig(
            r=policy["lora_rank"],
            lora_alpha=policy["lora_alpha"],
            lora_dropout=policy["lora_dropout"],
            target_modules=list(policy["lora_modules"]),
            layers_to_transform=list(policy["lora_layers"]),
            bias="none",
            init_lora_weights=True,
            task_type="CAUSAL_LM",
        )
        setup = configure_fixed_a_lora(model, lora)
    setup.model.to(device).eval()
    return setup


def _method_directory(method: str, beta: float) -> str:
    beta_text = format(beta, "g").replace(".", "p")
    return f"{method}__beta_{beta_text}"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _component_fixed_fields(
    *,
    config_sha256: str,
    artifact_metadata_sha256: str,
    reward_result_sha256: str,
    seed: int,
    name: str,
    reward_source: str,
    beta: float,
    direction: torch.Tensor,
    lora_a_sha256: str,
    lora_layout: Mapping[str, Any],
    producer: Mapping[str, str],
) -> dict[str, Any]:
    direction_values = direction.detach().to(dtype=torch.float64, device="cpu").tolist()
    return {
        "schema": COMPONENT_SCHEMA,
        "status": "complete",
        "config_sha256": config_sha256,
        "artifact_metadata_sha256": artifact_metadata_sha256,
        "reward_result_sha256": reward_result_sha256,
        "seed": seed,
        "adapter_name": name,
        "reward_source": reward_source,
        "beta": beta,
        "step_scale": 1.0 / beta,
        "direction_sha256": _canonical_sha256(direction_values),
        "direction_norm": float(torch.linalg.vector_norm(direction).item()),
        "lora_a_sha256": lora_a_sha256,
        "lora_layout_sha256": _canonical_sha256(lora_layout),
        "producer": dict(producer),
    }


def _component_record(receipt: Mapping[str, Any], receipt_sha256: str) -> dict[str, Any]:
    return {
        "reward_source": receipt["reward_source"],
        "beta": receipt["beta"],
        "step_scale": receipt["step_scale"],
        "direction_norm": receipt["direction_norm"],
        "files": dict(receipt["files"]),
        "component_receipt_sha256": receipt_sha256,
    }


def _validate_adapter_component(
    target: Path,
    name: str,
    expected_fixed: Mapping[str, Any],
) -> dict[str, Any]:
    directory = target / name
    receipt_path = target / ".checkpoints" / f"{name}.json"
    if not directory.is_dir() or not receipt_path.is_file():
        raise FileNotFoundError(f"adapter checkpoint is incomplete: {name}")
    with receipt_path.open("r", encoding="utf-8") as stream:
        receipt = json.load(stream)
    if not isinstance(receipt, dict):
        raise ValueError(f"adapter checkpoint must be an object: {name}")
    files = receipt.get("files")
    fixed = {key: value for key, value in receipt.items() if key != "files"}
    if fixed != dict(expected_fixed):
        raise ValueError(f"adapter checkpoint identity mismatch: {name}")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"adapter checkpoint file inventory is missing: {name}")
    for relative, expected_sha256 in files.items():
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise ValueError(f"adapter checkpoint file inventory is malformed: {name}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"adapter checkpoint file escapes its directory: {name}")
        if sha256_file(directory / relative_path) != expected_sha256:
            raise ValueError(f"adapter checkpoint file digest mismatch: {name}/{relative}")
    return _component_record(receipt, sha256_file(receipt_path))


def _quarantine_adapter_component(target: Path, name: str) -> None:
    directory = target / name
    receipt = target / ".checkpoints" / f"{name}.json"
    if not directory.exists() and not receipt.exists():
        return
    rejected_root = target / ".rejected"
    rejected_root.mkdir(parents=True, exist_ok=True)
    rejected = Path(tempfile.mkdtemp(prefix=f"{name}.", dir=rejected_root))
    if directory.exists():
        os.replace(directory, rejected / "adapter")
    if receipt.exists():
        os.replace(receipt, rejected / "component.json")


def validate_adapter_metadata(
    adapters: str | os.PathLike[str],
    *,
    expected_producer: Mapping[str, str],
) -> dict[str, Any]:
    """Validate final metadata, all nine component receipts, and every adapter file."""

    root = Path(adapters)
    metadata_path = root / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported adapter metadata")
    if metadata.get("producer") != dict(expected_producer):
        raise ValueError("adapter producer identity mismatch")
    records = metadata.get("adapters")
    if not isinstance(records, dict) or len(records) != 9:
        raise ValueError("adapter metadata must describe exactly nine adapters")
    lora_layout = metadata.get("lora_layout")
    directions = metadata.get("directions")
    if not isinstance(lora_layout, dict) or not isinstance(directions, dict):
        raise ValueError("adapter metadata geometry is missing")
    for name, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
            raise ValueError(f"adapter file inventory is missing: {name}")
        receipt_path = root / ".checkpoints" / f"{name}.json"
        if sha256_file(receipt_path) != record.get("component_receipt_sha256"):
            raise ValueError(f"adapter component receipt digest mismatch: {name}")
        with receipt_path.open("r", encoding="utf-8") as stream:
            receipt = json.load(stream)
        reward_source = receipt.get("reward_source") if isinstance(receipt, dict) else None
        direction_record = directions.get(reward_source) if isinstance(reward_source, str) else None
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != COMPONENT_SCHEMA
            or receipt.get("status") != "complete"
            or receipt.get("adapter_name") != name
            or receipt.get("config_sha256") != metadata.get("config_sha256")
            or receipt.get("artifact_metadata_sha256") != metadata.get("artifact_metadata_sha256")
            or receipt.get("reward_result_sha256") != metadata.get("reward_result_sha256")
            or receipt.get("seed") != metadata.get("seed")
            or receipt.get("lora_a_sha256") != metadata.get("lora_a_sha256")
            or receipt.get("lora_layout_sha256") != _canonical_sha256(lora_layout)
            or receipt.get("producer") != dict(expected_producer)
            or not isinstance(direction_record, dict)
            or receipt.get("direction_norm") != direction_record.get("norm")
            or _component_record(receipt, sha256_file(receipt_path)) != record
        ):
            raise ValueError(f"adapter component receipt mismatch: {name}")
        for relative, expected_sha256 in record["files"].items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"adapter file inventory escapes its directory: {name}")
            if sha256_file(root / name / relative_path) != expected_sha256:
                raise ValueError(f"adapter file digest mismatch: {name}/{relative}")
    return metadata


def export_exact_ngd_adapters(
    config: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    comparison_json: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load the three train-fitted directions and export all beta-scaled adapters."""

    normalized = validate_config(config)
    if normalized["protocol"] != PROTOCOL or seed not in normalized["run"]["seeds"]:
        raise ValueError("protocol or seed mismatch")
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
    comparison = load_exact_reward_comparison(
        comparison_json,
        expected_config_hash=digest,
        expected_seed=seed,
    )
    if comparison["artifact_metadata_sha256"] != artifact_identity:
        raise ValueError("reward comparison belongs to another artifact")
    comparison_identity = sha256_file(Path(comparison_json))
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    directions = {
        method: torch.tensor(
            values,
            device=target_device,
            dtype=torch.float64,
        )
        for method, values in comparison["policy_directions"].items()
    }
    evidence = json.loads((Path(artifact_dir) / "metadata.json").read_text(encoding="utf-8"))[
        "evidence"
    ]
    lora_a_sha256 = evidence["policy_a_sha256"]
    lora_layout = evidence["policy_layout"]
    if not isinstance(lora_a_sha256, str) or not isinstance(lora_layout, dict):
        raise ValueError("materialization LoRA evidence is malformed")
    current_producer = producer_identity()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / ".checkpoints").mkdir(exist_ok=True)
    adapters: dict[str, Any] = {}
    missing: list[tuple[str, str, float, torch.Tensor, dict[str, Any]]] = []
    for method in ("mle_rm", "pro_rm", "oracle"):
        direction = directions[method]
        for beta_raw in normalized["policy_update"]["beta_grid"]:
            beta = float(beta_raw)
            name = _method_directory(method, beta)
            fixed = _component_fixed_fields(
                config_sha256=digest,
                artifact_metadata_sha256=artifact_identity,
                reward_result_sha256=comparison_identity,
                seed=seed,
                name=name,
                reward_source=method,
                beta=beta,
                direction=direction,
                lora_a_sha256=lora_a_sha256,
                lora_layout=lora_layout,
                producer=current_producer,
            )
            try:
                adapters[name] = _validate_adapter_component(target, name, fixed)
                print(f"adapter name={name} status=reused", flush=True)
            except (FileNotFoundError, ValueError):
                _quarantine_adapter_component(target, name)
                missing.append((name, method, beta, direction, fixed))
    setup: Any | None = None
    try:
        if missing:
            setup = _load_policy(normalized, seed, target_device, local_files_only)
            if setup.a_state_sha256 != lora_a_sha256:
                raise RuntimeError("reloaded fixed LoRA-A does not match materialization")
            if setup.layout.to_metadata() != lora_layout:
                raise RuntimeError("reloaded LoRA-B layout does not match materialization")
        for name, _method, beta, direction, fixed in missing:
            assert setup is not None
            _zero_b(setup)
            set_tangent_update_(
                setup.named_tangent_parameters(),
                setup.layout,
                direction,
                step_size=1.0 / beta,
            )
            staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=target))
            try:
                setup.model.save_pretrained(staging, safe_serialization=True)
                saved_files = {
                    path.relative_to(staging).as_posix(): sha256_file(path)
                    for path in sorted(staging.rglob("*"))
                    if path.is_file()
                }
                if not saved_files:
                    raise RuntimeError(f"adapter serialization produced no files: {name}")
                os.replace(staging, target / name)
                receipt = {**fixed, "files": saved_files}
                receipt_path = target / ".checkpoints" / f"{name}.json"
                _atomic_json(receipt_path, receipt)
                adapters[name] = _component_record(receipt, sha256_file(receipt_path))
                print(f"adapter name={name} status=checkpointed", flush=True)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        metadata = {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "config_sha256": digest,
            "artifact_metadata_sha256": artifact_identity,
            "reward_result_sha256": comparison_identity,
            "seed": seed,
            "beta_grid": [float(value) for value in normalized["policy_update"]["beta_grid"]],
            "policy_families": ["pi0", "mle_ngd", "pro_ngd", "oracle_ngd"],
            "updated_adapter_count": len(adapters),
            "lora_a_sha256": lora_a_sha256,
            "lora_layout": lora_layout,
            "directions": {
                method: {
                    "fit_split": "train",
                    "norm": float(torch.linalg.vector_norm(value).item()),
                }
                for method, value in directions.items()
            },
            "reward_heads": {
                method: comparison["methods"][serialized]["head_sha256"]
                for method, serialized in (("mle_rm", "MLE-RM"), ("pro_rm", "Pro-RM"))
            },
            "adapters": adapters,
            "producer": current_producer,
        }
        metadata_path = target / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as stream:
                existing_metadata = json.load(stream)
            if existing_metadata == metadata:
                return metadata
            rejected_root = target / ".rejected"
            rejected_root.mkdir(exist_ok=True)
            rejected = Path(tempfile.mkdtemp(prefix="metadata.", dir=rejected_root))
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
    "export_exact_ngd_adapters",
    "validate_adapter_metadata",
]
