"""Small immutable receipts for resumable production stages."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import config_hash, validate_config
from .runtime import producer_identity

SCHEMA = "prorm-stage-receipt/v1"
PROVENANCE_SCHEMA = "prorm-stage-provenance/v1"


def _digest_map(value: Mapping[str, str], *, name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name}.{key} must be a lowercase SHA-256")
        result[key] = digest
    return dict(sorted(result.items()))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _producer_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("producer must be an object")
    allowed = {"git_commit", "image_sha256", "hf_inventory_sha256"}
    if not set(value).issubset(allowed):
        raise ValueError("producer contains unsupported identities")
    result: dict[str, str] = {}
    for key, digest in value.items():
        expected_length = 40 if key == "git_commit" else 64
        if (
            not isinstance(digest, str)
            or len(digest) != expected_length
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"producer.{key} must be an immutable lowercase digest")
        result[key] = digest
    return dict(sorted(result.items()))


def _receipt_payload_with_producer(
    config: Mapping[str, object],
    *,
    stage: str,
    seed: int,
    producer: Mapping[str, str],
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    normalized = validate_config(config)
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    if seed not in normalized["run"]["seeds"]:
        raise ValueError("seed must be configured")
    return {
        "schema": SCHEMA,
        "protocol": normalized["protocol"],
        "stage": stage,
        "status": "complete",
        "config_sha256": config_hash(normalized),
        "seed": seed,
        "producer": _producer_map(dict(producer)),
        "inputs": _digest_map(inputs, name="inputs"),
        "outputs": _digest_map(outputs, name="outputs"),
    }


def stage_receipt_payload(
    config: Mapping[str, object],
    *,
    stage: str,
    seed: int,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    return _receipt_payload_with_producer(
        config,
        stage=stage,
        seed=seed,
        producer=producer_identity(),
        inputs=inputs,
        outputs=outputs,
    )


def write_stage_receipt(
    path: str | os.PathLike[str],
    config: Mapping[str, object],
    *,
    stage: str,
    seed: int,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite stage receipt: {target}")
    payload = stage_receipt_payload(
        config,
        stage=stage,
        seed=seed,
        inputs=inputs,
        outputs=outputs,
    )
    _atomic_json(target, payload)
    return payload


def validate_stage_receipt(
    path: str | os.PathLike[str],
    config: Mapping[str, object],
    *,
    stage: str,
    seed: int,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    expected = stage_receipt_payload(
        config,
        stage=stage,
        seed=seed,
        inputs=inputs,
        outputs=outputs,
    )
    with Path(path).open("r", encoding="utf-8") as stream:
        observed = json.load(stream)
    if observed != expected:
        raise ValueError(f"stage receipt mismatch: {path}")
    return observed


def validate_source_stage_receipt(
    path: str | os.PathLike[str],
    config: Mapping[str, object],
    *,
    stage: str,
    seed: int,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        observed = json.load(stream)
    if not isinstance(observed, dict):
        raise ValueError(f"expected receipt object: {path}")
    expected = _receipt_payload_with_producer(
        config,
        stage=stage,
        seed=seed,
        producer=_producer_map(observed.get("producer")),
        inputs=inputs,
        outputs=outputs,
    )
    if observed != expected:
        raise ValueError(f"source stage receipt mismatch: {path}")
    return observed


def provenance_bridge_payload(
    config: Mapping[str, object],
    *,
    stage: str,
    seed: int,
    source_receipt_sha256: str,
    source_producer: Mapping[str, str],
    outputs: Mapping[str, str],
    affected_stage_analysis_sha256: str,
) -> dict[str, Any]:
    digests = _digest_map(
        {
            "source_receipt": source_receipt_sha256,
            "affected_stage_analysis": affected_stage_analysis_sha256,
        },
        name="provenance",
    )
    normalized = validate_config(config)
    if seed not in normalized["run"]["seeds"]:
        raise ValueError("seed must be configured")
    return {
        "schema": PROVENANCE_SCHEMA,
        "protocol": normalized["protocol"],
        "stage": stage,
        "status": "accepted",
        "config_sha256": config_hash(normalized),
        "seed": seed,
        "source": {
            "receipt_sha256": digests["source_receipt"],
            "producer": _producer_map(dict(source_producer)),
        },
        "consumer": _producer_map(producer_identity()),
        "outputs": _digest_map(outputs, name="outputs"),
        "affected_stage_analysis_sha256": digests["affected_stage_analysis"],
    }


def write_provenance_bridge(
    path: str | os.PathLike[str],
    config: Mapping[str, object],
    **kwargs: Any,
) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite provenance bridge: {target}")
    payload = provenance_bridge_payload(config, **kwargs)
    _atomic_json(target, payload)
    return payload


def validate_provenance_bridge(
    path: str | os.PathLike[str],
    config: Mapping[str, object],
    **kwargs: Any,
) -> dict[str, Any]:
    expected = provenance_bridge_payload(config, **kwargs)
    with Path(path).open("r", encoding="utf-8") as stream:
        observed = json.load(stream)
    if observed != expected:
        raise ValueError(f"provenance bridge mismatch: {path}")
    return observed


__all__ = [
    "PROVENANCE_SCHEMA",
    "SCHEMA",
    "provenance_bridge_payload",
    "stage_receipt_payload",
    "validate_provenance_bridge",
    "validate_source_stage_receipt",
    "validate_stage_receipt",
    "write_provenance_bridge",
    "write_stage_receipt",
]
