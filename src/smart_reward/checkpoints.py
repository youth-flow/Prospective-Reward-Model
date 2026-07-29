"""Small immutable receipts for resumable production stages."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import PROTOCOL, config_hash, validate_config
from .runtime import producer_identity

SCHEMA = "prorm-stage-receipt/v1"


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


def stage_receipt_payload(
    config: Mapping[str, object],
    *,
    stage: str,
    seed: int,
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
        "protocol": PROTOCOL,
        "stage": stage,
        "status": "complete",
        "config_sha256": config_hash(normalized),
        "seed": seed,
        "producer": producer_identity(),
        "inputs": _digest_map(inputs, name="inputs"),
        "outputs": _digest_map(outputs, name="outputs"),
    }


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


__all__ = [
    "SCHEMA",
    "stage_receipt_payload",
    "validate_stage_receipt",
    "write_stage_receipt",
]
