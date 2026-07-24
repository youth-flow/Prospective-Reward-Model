"""Immutable terminal evidence for a formal Phase-2 campaign.

The ordinary Phase-2 aggregate intentionally requires one valid result for
every declared seed.  This module handles the complementary terminal case:
one or more predeclared seeds failed and therefore have no admissible outcome
result.  A failed seed may never be dropped or replaced.  Instead, an
identity-bound failure manifest occupies that seed's exact campaign slot and
the finalizer publishes a terminal, non-inferential campaign record.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final

from .paths import relative_posix_reference
from .phase2_aggregate import build_common_beta_seed_aggregate
from .phase2_config import phase2_design_identity, validate_phase2_config
from .phase2_rollout import PHASE2_RESULT_SCHEMA, PHASE2_ROLLOUT_SCHEMA, Phase2Design
from .repro import atomic_write_json

PHASE2_SEED_FAILURE_SCHEMA_V1: Final = "phase2-seed-terminal-failure/v1"
PHASE2_SEED_FAILURE_SCHEMA: Final = "phase2-seed-terminal-failure/v2"
PHASE2_SEED_SUCCESS_SCHEMA: Final = "phase2-seed-terminal-success/v2"
PHASE2_CAMPAIGN_TERMINAL_SCHEMA: Final = "phase2-campaign-terminal/v2"
# Historical identifiers remain exported for offline provenance readers only;
# the formal builders/finalizer below accept PHASE2_ATTEMPT_LEDGER_SCHEMA (v3).
PHASE2_ATTEMPT_LEDGER_SCHEMA_V1: Final = "phase2-seed-attempt-ledger/v1"
PHASE2_ATTEMPT_LEDGER_SCHEMA_V2: Final = "phase2-seed-attempt-ledger/v2"
PHASE2_ATTEMPT_LEDGER_SCHEMA: Final = "phase2-seed-attempt-ledger/v3"
PHASE2_FAILURE_EVIDENCE_AVAILABILITY_SCHEMA: Final = "phase2-seed-failure-evidence-availability/v1"
PHASE2_FORMAL_SEED_COUNT: Final = 30

_HEX = frozenset("0123456789abcdef")
_ENVIRONMENT_KEYS = frozenset(
    {
        "formal",
        "git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "account",
        "partition",
        "gpu_models",
    }
)
_FAILURE_CLASSES = frozenset(
    {
        "infrastructure",
        "scientific",
        "safety",
        "identity",
        "numerical",
        "software",
    }
)
_EVIDENCE_UNAVAILABLE_REASONS = frozenset(
    {
        "not_produced_before_failure",
        "not_published_before_hard_termination",
        "not_recoverable_from_scheduler_evidence",
    }
)
_FAILURE_CAPTURE_METHODS = frozenset(
    {
        "compute_exit_trap",
        "scheduler_terminal_reconciliation",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "attempt_index",
        "slurm_job_id",
        "status",
        "final_outcome_reveal_started",
        "log_sha256",
    }
)
_ATTEMPT_V2_KEYS = _ATTEMPT_KEYS | {
    "cluster_name",
    "array_job_id",
    "array_task_id",
}


def _digest(value: object, *, name: str, lengths: frozenset[int] = frozenset({64})) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in _HEX for character in value)
    ):
        rendered = " or ".join(str(length) for length in sorted(lengths))
        raise ValueError(f"{name} must be a lowercase hexadecimal digest of length {rendered}")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str] | set[str],
) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    if set(result) != set(keys):
        missing = sorted(set(keys) - set(result))
        unknown = sorted(set(result) - set(keys))
        raise ValueError(f"{name} fields differ: missing={missing!r}, unknown={unknown!r}")
    return result


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_strict_json(path: str | os.PathLike[str], *, name: str) -> tuple[dict[str, object], str]:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {unresolved}")
    source = unresolved.resolve()
    if not source.is_file():
        raise ValueError(f"{name} must be a regular file: {source}")
    raw = source.read_bytes()

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _read_regular_bytes(path: Path, *, name: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {path}")
    if not path.is_file():
        raise ValueError(f"{name} must be a regular file: {path}")
    return path.read_bytes()


def _strict_json_object_bytes(raw: bytes, *, name: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _resolve_sibling_posix_reference(
    value: object,
    *,
    source_path: Path,
    field_name: str,
    expected_name: str | None = None,
) -> Path:
    reference = _string(value, name=f"{source_path}:{field_name}")
    if "\\" in reference:
        raise ValueError(f"{source_path} {field_name} must use POSIX separators")
    pure = PurePosixPath(reference)
    if (
        pure.is_absolute()
        or pure.drive
        or len(pure.parts) != 1
        or pure.parts[0] in {"", ".", ".."}
        or reference != pure.name
        or ":" in pure.parts[0]
    ):
        raise ValueError(f"{source_path} {field_name} must be one POSIX basename")
    if expected_name is not None and reference != expected_name:
        raise ValueError(f"{source_path} {field_name} must equal {expected_name!r}")
    candidate = source_path.parent.joinpath(*pure.parts)
    if candidate.parent.resolve() != source_path.parent.resolve():
        raise ValueError(f"{source_path} {field_name} escaped its source directory")
    return candidate


def _load_result_rollout_evidence(
    result: Mapping[str, object],
    *,
    result_path: Path,
) -> tuple[Path, dict[str, object]]:
    expected_name = f"{result_path.stem}.rollouts.jsonl"
    rollout_path = _resolve_sibling_posix_reference(
        result.get("rollouts_jsonl"),
        source_path=result_path,
        field_name="rollouts_jsonl",
        expected_name=expected_name,
    )
    raw = _read_regular_bytes(
        rollout_path,
        name=f"{result_path} sibling rollout JSONL",
    )
    actual_sha = hashlib.sha256(raw).hexdigest()
    expected_sha = _digest(
        result.get("rollouts_sha256"),
        name=f"{result_path}:rollouts_sha256",
    )
    if actual_sha != expected_sha:
        raise ValueError(f"{result_path} sibling rollout JSONL SHA-256 changed")
    lines = raw.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"{rollout_path} must be non-empty JSONL without blank rows")
    for index, line in enumerate(lines, start=1):
        row = _strict_json_object_bytes(
            line,
            name=f"{rollout_path}:{index}",
        )
        if row.get("schema_version") != PHASE2_ROLLOUT_SCHEMA:
            raise ValueError(f"{rollout_path}:{index} has an unsupported rollout trajectory schema")
    return (
        rollout_path.resolve(),
        {
            "path": rollout_path.name,
            "sha256": actual_sha,
            "schema_version": PHASE2_ROLLOUT_SCHEMA,
        },
    )


def _validate_environment(value: object, *, name: str) -> dict[str, object]:
    identity = _strict_mapping(value, name=name, keys=_ENVIRONMENT_KEYS)
    if identity["formal"] is not True:
        raise ValueError(f"{name}.formal must be true")
    commit = _digest(
        identity["git_commit"],
        name=f"{name}.git_commit",
        lengths=frozenset({40, 64}),
    )
    image = _digest(identity["image_sha256"], name=f"{name}.image_sha256")
    inventory = _digest(
        identity["hf_inventory_sha256"],
        name=f"{name}.hf_inventory_sha256",
    )
    if identity["account"] != "sigroup":
        raise ValueError(f"{name}.account must equal 'sigroup'")
    partition = _string(identity["partition"], name=f"{name}.partition")
    gpu_models = identity["gpu_models"]
    if (
        not isinstance(gpu_models, list)
        or len(gpu_models) != 1
        or not isinstance(gpu_models[0], str)
        or not gpu_models[0]
    ):
        raise ValueError(f"{name}.gpu_models must contain exactly one model")
    return {
        "formal": True,
        "git_commit": commit,
        "image_sha256": image,
        "hf_inventory_sha256": inventory,
        "account": "sigroup",
        "partition": partition,
        "gpu_models": list(gpu_models),
    }


def _formal_contract(
    overlay_config: Mapping[str, object],
) -> tuple[dict[str, object], tuple[int, ...], str, str, str]:
    validated = validate_phase2_config(overlay_config)
    design = _mapping(validated["design"], name="design")
    run = _mapping(validated["run"], name="run")
    if (
        design.get("stage") != "confirmatory"
        or design.get("formal_eligibility") is not True
        or run.get("confirmatory") is not True
        or run.get("formal_eligibility") is not True
    ):
        raise ValueError("campaign terminal evidence requires a formal confirmatory overlay")
    raw_seeds = run.get("seeds")
    if not isinstance(raw_seeds, list):
        raise TypeError("run.seeds must be a list")
    seeds = tuple(_integer(seed, name="run.seeds[]") for seed in raw_seeds)
    if len(seeds) != PHASE2_FORMAL_SEED_COUNT or len(set(seeds)) != len(seeds):
        raise ValueError(
            f"formal terminal campaign requires exactly {PHASE2_FORMAL_SEED_COUNT} unique seeds"
        )
    source_hash = _digest(
        design.get("source_config_hash"),
        name="design.source_config_hash",
    )
    design_sha = phase2_design_identity(validated)
    runtime_sha = Phase2Design.from_phase2_config(validated).sha256
    return validated, seeds, source_hash, design_sha, runtime_sha


def _validate_attempt_ledger(
    value: object,
    *,
    terminal_status: str,
) -> dict[str, object]:
    if terminal_status not in {"terminal_failure", "success_result"}:
        raise ValueError("attempt ledger terminal_status is unsupported")
    ledger = _strict_mapping(
        value,
        name="attempt_ledger",
        keys={
            "schema_version",
            "retry_policy",
            "replacement_seed_allowed",
            "attempts",
        },
    )
    if (
        ledger["schema_version"] != PHASE2_ATTEMPT_LEDGER_SCHEMA
        or ledger["retry_policy"] != "single_predeclared_attempt_no_retry"
        or ledger["replacement_seed_allowed"] is not False
    ):
        raise ValueError("attempt_ledger has an invalid retry or replacement policy")
    raw_attempts = ledger["attempts"]
    if (
        isinstance(raw_attempts, (str, bytes, bytearray))
        or not isinstance(raw_attempts, Sequence)
        or len(raw_attempts) != 1
    ):
        raise ValueError("attempt_ledger.attempts must contain exactly one formal attempt")
    attempts: list[dict[str, object]] = []
    job_ids: set[str] = set()
    ledger_schema = str(ledger["schema_version"])
    for index, raw_attempt in enumerate(raw_attempts, start=1):
        attempt = _strict_mapping(
            raw_attempt,
            name=f"attempt_ledger.attempts[{index - 1}]",
            keys=_ATTEMPT_V2_KEYS,
        )
        if attempt["attempt_index"] != index:
            raise ValueError("attempt indices must be contiguous and one-based")
        job_id = _string(
            attempt["slurm_job_id"],
            name=f"attempt_ledger.attempts[{index - 1}].slurm_job_id",
        )
        if job_id in job_ids:
            raise ValueError("attempt ledger repeats a Slurm job identity")
        job_ids.add(job_id)
        status = attempt["status"]
        expected_status = terminal_status
        if status != expected_status:
            raise ValueError("the single formal attempt must carry the terminal status")
        reveal_started = attempt["final_outcome_reveal_started"]
        if not isinstance(reveal_started, bool):
            raise TypeError("attempt final_outcome_reveal_started must be bool")
        if status == "success_result" and not reveal_started:
            raise ValueError("a successful terminal attempt must reveal its final outcome")
        normalized_attempt: dict[str, object] = {
            "attempt_index": index,
            "slurm_job_id": job_id,
            "status": status,
            "final_outcome_reveal_started": reveal_started,
            "log_sha256": _digest(
                attempt["log_sha256"],
                name=f"attempt_ledger.attempts[{index - 1}].log_sha256",
            ),
        }
        cluster_name = _string(
            attempt["cluster_name"],
            name=f"attempt_ledger.attempts[{index - 1}].cluster_name",
        )
        array_job_id = _string(
            attempt["array_job_id"],
            name=f"attempt_ledger.attempts[{index - 1}].array_job_id",
        )
        if not array_job_id.isdecimal() or array_job_id.startswith("0"):
            raise ValueError("attempt array_job_id must be a positive decimal string")
        array_task_id = _integer(
            attempt["array_task_id"],
            name=f"attempt_ledger.attempts[{index - 1}].array_task_id",
        )
        normalized_attempt.update(
            {
                "cluster_name": cluster_name,
                "array_job_id": array_job_id,
                "array_task_id": array_task_id,
            }
        )
        attempts.append(normalized_attempt)
    return {
        "schema_version": ledger_schema,
        "retry_policy": "single_predeclared_attempt_no_retry",
        "replacement_seed_allowed": False,
        "attempts": attempts,
    }


def _validate_evidence_hashes(value: object) -> dict[str, str]:
    evidence = _mapping(value, name="evidence_sha256_by_role")
    if not evidence:
        raise ValueError("evidence_sha256_by_role must contain at least one evidence hash")
    normalized: dict[str, str] = {}
    for role, digest in evidence.items():
        _string(role, name="evidence_sha256_by_role role")
        normalized[role] = _digest(
            digest,
            name=f"evidence_sha256_by_role.{role}",
        )
    return normalized


def _available_digest_slot(value: str) -> dict[str, object]:
    return {
        "status": "available",
        "sha256": _digest(value, name="available evidence SHA256"),
    }


def _available_environment_slot(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": "available",
        "value": _validate_environment(value, name="environment_identity"),
    }


def _validate_digest_availability_slot(value: object, *, name: str) -> dict[str, object]:
    slot = _mapping(value, name=name)
    status = slot.get("status")
    if status == "available":
        strict = _strict_mapping(slot, name=name, keys={"status", "sha256"})
        return {
            "status": "available",
            "sha256": _digest(strict["sha256"], name=f"{name}.sha256"),
        }
    if status == "unavailable":
        strict = _strict_mapping(slot, name=name, keys={"status", "reason"})
        reason = _string(strict["reason"], name=f"{name}.reason")
        if reason not in _EVIDENCE_UNAVAILABLE_REASONS:
            raise ValueError(
                f"{name}.reason must be one of {sorted(_EVIDENCE_UNAVAILABLE_REASONS)!r}"
            )
        return {
            "status": "unavailable",
            "reason": reason,
        }
    raise ValueError(f"{name}.status must equal 'available' or 'unavailable'")


def _validate_environment_availability_slot(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    slot = _mapping(value, name=name)
    status = slot.get("status")
    if status == "available":
        strict = _strict_mapping(slot, name=name, keys={"status", "value"})
        return {
            "status": "available",
            "value": _validate_environment(
                strict["value"],
                name=f"{name}.value",
            ),
        }
    if status == "unavailable":
        strict = _strict_mapping(slot, name=name, keys={"status", "reason"})
        reason = _string(strict["reason"], name=f"{name}.reason")
        if reason not in _EVIDENCE_UNAVAILABLE_REASONS:
            raise ValueError(
                f"{name}.reason must be one of {sorted(_EVIDENCE_UNAVAILABLE_REASONS)!r}"
            )
        return {
            "status": "unavailable",
            "reason": reason,
        }
    raise ValueError(f"{name}.status must equal 'available' or 'unavailable'")


def _validate_failure_evidence_availability(value: object) -> dict[str, object]:
    availability = _strict_mapping(
        value,
        name="evidence_availability",
        keys={
            "schema_version",
            "run_manifest",
            "artifact_metadata",
            "environment_identity",
        },
    )
    if availability["schema_version"] != PHASE2_FAILURE_EVIDENCE_AVAILABILITY_SCHEMA:
        raise ValueError("evidence_availability has an unsupported schema")
    return {
        "schema_version": PHASE2_FAILURE_EVIDENCE_AVAILABILITY_SCHEMA,
        "run_manifest": _validate_digest_availability_slot(
            availability["run_manifest"],
            name="evidence_availability.run_manifest",
        ),
        "artifact_metadata": _validate_digest_availability_slot(
            availability["artifact_metadata"],
            name="evidence_availability.artifact_metadata",
        ),
        "environment_identity": _validate_environment_availability_slot(
            availability["environment_identity"],
            name="evidence_availability.environment_identity",
        ),
    }


def _fully_available_failure_evidence(
    *,
    run_manifest_sha256: str,
    artifact_metadata_sha256: str,
    environment_identity: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": PHASE2_FAILURE_EVIDENCE_AVAILABILITY_SCHEMA,
        "run_manifest": _available_digest_slot(run_manifest_sha256),
        "artifact_metadata": _available_digest_slot(artifact_metadata_sha256),
        "environment_identity": _available_environment_slot(environment_identity),
    }


def build_phase2_seed_failure_manifest(
    overlay_config: Mapping[str, object],
    *,
    seed: int,
    failure_stage: str,
    failure_class: str,
    failure_type: str,
    failure_message_sha256: str,
    final_outcome_reveal_started: bool,
    attempt_ledger: Mapping[str, object],
    evidence_sha256_by_role: Mapping[str, object],
    evidence_availability: Mapping[str, object] | None = None,
    capture_method: str = "compute_exit_trap",
    run_manifest_sha256: str | None = None,
    artifact_metadata_sha256: str | None = None,
    environment_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one immutable terminal failure record for a predeclared seed."""

    _, seeds, source_hash, design_sha, runtime_sha = _formal_contract(overlay_config)
    selected_seed = _integer(seed, name="seed")
    if selected_seed not in seeds:
        raise ValueError("failure manifest seed is not predeclared by the overlay")
    failure_category = _string(failure_class, name="failure_class")
    if failure_category not in _FAILURE_CLASSES:
        raise ValueError(f"failure_class must be one of {sorted(_FAILURE_CLASSES)!r}")
    if not isinstance(final_outcome_reveal_started, bool):
        raise TypeError("final_outcome_reveal_started must be bool")
    ledger = _validate_attempt_ledger(
        attempt_ledger,
        terminal_status="terminal_failure",
    )
    attempts = ledger["attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError("validated attempt ledger lost its attempts")
    if attempts[-1]["final_outcome_reveal_started"] is not final_outcome_reveal_started:
        raise ValueError("failure reveal boundary disagrees with the terminal attempt")
    capture = _string(capture_method, name="capture_method")
    if capture not in _FAILURE_CAPTURE_METHODS:
        raise ValueError(f"capture_method must be one of {sorted(_FAILURE_CAPTURE_METHODS)!r}")
    if evidence_availability is None:
        if (
            run_manifest_sha256 is None
            or artifact_metadata_sha256 is None
            or environment_identity is None
        ):
            raise ValueError(
                "failure evidence must provide evidence_availability or all three "
                "legacy evidence values"
            )
        availability = _fully_available_failure_evidence(
            run_manifest_sha256=run_manifest_sha256,
            artifact_metadata_sha256=artifact_metadata_sha256,
            environment_identity=environment_identity,
        )
    else:
        if (
            run_manifest_sha256 is not None
            or artifact_metadata_sha256 is not None
            or environment_identity is not None
        ):
            raise ValueError("evidence_availability cannot be mixed with legacy evidence fields")
        availability = _validate_failure_evidence_availability(evidence_availability)
    evidence_hashes = _validate_evidence_hashes(evidence_sha256_by_role)
    if (
        capture == "scheduler_terminal_reconciliation"
        and "scheduler_terminal_attestation" not in evidence_hashes
    ):
        raise ValueError(
            "scheduler reconciliation requires scheduler_terminal_attestation evidence"
        )
    return {
        "schema_version": PHASE2_SEED_FAILURE_SCHEMA,
        "terminal_status": "failed",
        "terminal": True,
        "supports_formal_claim": False,
        "seed": selected_seed,
        "source_config_hash": source_hash,
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract_sha256": runtime_sha,
        "capture_method": capture,
        "evidence_availability": availability,
        "failure": {
            "stage": _string(failure_stage, name="failure_stage"),
            "class": failure_category,
            "type": _string(failure_type, name="failure_type"),
            "message_sha256": _digest(
                failure_message_sha256,
                name="failure_message_sha256",
            ),
            "final_outcome_reveal_started": final_outcome_reveal_started,
            "scientific_result_published": False,
        },
        "attempt_ledger": ledger,
        "evidence_sha256_by_role": evidence_hashes,
        "seed_replacement_allowed": False,
    }


def build_phase2_seed_failure_manifest_from_spec(
    overlay_config: Mapping[str, object],
    spec: Mapping[str, object],
) -> dict[str, object]:
    """Validate a compact CLI spec and bind it to the formal overlay identity."""

    common_keys = {
        "seed",
        "failure_stage",
        "failure_class",
        "failure_type",
        "failure_message_sha256",
        "final_outcome_reveal_started",
        "attempt_ledger",
        "evidence_sha256_by_role",
    }
    value = _mapping(spec, name="failure spec")
    legacy_keys = common_keys | {
        "run_manifest_sha256",
        "artifact_metadata_sha256",
        "environment_identity",
    }
    availability_keys = common_keys | {"evidence_availability", "capture_method"}
    if set(value) == legacy_keys:
        legacy_arguments: dict[str, object] = {
            "run_manifest_sha256": value["run_manifest_sha256"],
            "artifact_metadata_sha256": value["artifact_metadata_sha256"],
            "environment_identity": _mapping(
                value["environment_identity"],
                name="failure spec.environment_identity",
            ),
        }
    elif set(value) == availability_keys:
        legacy_arguments = {
            "evidence_availability": _mapping(
                value["evidence_availability"],
                name="failure spec.evidence_availability",
            ),
            "capture_method": value["capture_method"],
        }
    else:
        missing_legacy = sorted(legacy_keys - set(value))
        missing_availability = sorted(availability_keys - set(value))
        unknown = sorted(set(value) - (legacy_keys | availability_keys))
        raise ValueError(
            "failure spec fields differ from both accepted forms: "
            f"legacy_missing={missing_legacy!r}, "
            f"availability_missing={missing_availability!r}, unknown={unknown!r}"
        )
    return build_phase2_seed_failure_manifest(
        overlay_config,
        seed=value["seed"],
        failure_stage=value["failure_stage"],
        failure_class=value["failure_class"],
        failure_type=value["failure_type"],
        failure_message_sha256=value["failure_message_sha256"],
        final_outcome_reveal_started=value["final_outcome_reveal_started"],
        attempt_ledger=_mapping(
            value["attempt_ledger"],
            name="failure spec.attempt_ledger",
        ),
        evidence_sha256_by_role=_mapping(
            value["evidence_sha256_by_role"],
            name="failure spec.evidence_sha256_by_role",
        ),
        **legacy_arguments,
    )


def write_phase2_seed_failure_manifest(
    overlay_config: Mapping[str, object],
    spec: Mapping[str, object],
    output_json: str | os.PathLike[str],
) -> dict[str, object]:
    """Publish a failure manifest once; an existing terminal record always wins."""

    payload = build_phase2_seed_failure_manifest_from_spec(overlay_config, spec)
    atomic_write_json(output_json, payload, overwrite=False)
    return payload


_FAILURE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "terminal_status",
        "terminal",
        "supports_formal_claim",
        "seed",
        "source_config_hash",
        "phase2_design_sha256",
        "phase2_runtime_contract_sha256",
        "capture_method",
        "evidence_availability",
        "failure",
        "attempt_ledger",
        "evidence_sha256_by_role",
        "seed_replacement_allowed",
    }
)
_FAILURE_MANIFEST_V1_KEYS = _FAILURE_MANIFEST_KEYS - {
    "capture_method",
    "evidence_availability",
} | {
    "run_manifest_sha256",
    "artifact_metadata_sha256",
    "environment_identity",
}


def _load_failure_manifest(
    value: Mapping[str, object],
    *,
    source_sha256: str,
    source_path: Path,
    source_hash: str,
    design_sha: str,
    runtime_sha: str,
    reference_base: Path,
) -> dict[str, object]:
    schema = value.get("schema_version")
    if schema == PHASE2_SEED_FAILURE_SCHEMA:
        keys = _FAILURE_MANIFEST_KEYS
    elif schema == PHASE2_SEED_FAILURE_SCHEMA_V1:
        keys = _FAILURE_MANIFEST_V1_KEYS
    else:
        raise ValueError(f"{source_path} has an unsupported failure terminal schema")
    manifest = _strict_mapping(
        value,
        name=str(source_path),
        keys=keys,
    )
    if (
        manifest["terminal_status"] != "failed"
        or manifest["terminal"] is not True
        or manifest["supports_formal_claim"] is not False
        or manifest["seed_replacement_allowed"] is not False
        or manifest["source_config_hash"] != source_hash
        or manifest["phase2_design_sha256"] != design_sha
        or manifest["phase2_runtime_contract_sha256"] != runtime_sha
    ):
        raise ValueError(f"{source_path} has an invalid failure terminal identity")
    failure = _strict_mapping(
        manifest["failure"],
        name=f"{source_path}:failure",
        keys={
            "stage",
            "class",
            "type",
            "message_sha256",
            "final_outcome_reveal_started",
            "scientific_result_published",
        },
    )
    if (
        _string(failure["stage"], name=f"{source_path}:failure.stage") == ""
        or failure["class"] not in _FAILURE_CLASSES
        or _string(failure["type"], name=f"{source_path}:failure.type") == ""
        or not isinstance(failure["final_outcome_reveal_started"], bool)
        or failure["scientific_result_published"] is not False
    ):
        raise ValueError(f"{source_path} has invalid failure evidence")
    _digest(failure["message_sha256"], name=f"{source_path}:failure.message_sha256")
    ledger = _validate_attempt_ledger(
        manifest["attempt_ledger"],
        terminal_status="terminal_failure",
    )
    attempts = ledger["attempts"]
    if (
        not isinstance(attempts, list)
        or attempts[-1]["final_outcome_reveal_started"]
        is not failure["final_outcome_reveal_started"]
    ):
        raise ValueError(f"{source_path} failure boundary disagrees with its attempt ledger")
    if schema == PHASE2_SEED_FAILURE_SCHEMA:
        availability = _validate_failure_evidence_availability(manifest["evidence_availability"])
        capture = _string(
            manifest["capture_method"],
            name=f"{source_path}:capture_method",
        )
        if capture not in _FAILURE_CAPTURE_METHODS:
            raise ValueError(
                f"{source_path}:capture_method must be one of {sorted(_FAILURE_CAPTURE_METHODS)!r}"
            )
    else:
        capture = "compute_exit_trap"
        availability = _fully_available_failure_evidence(
            run_manifest_sha256=_digest(
                manifest["run_manifest_sha256"],
                name=f"{source_path}:run_manifest_sha256",
            ),
            artifact_metadata_sha256=_digest(
                manifest["artifact_metadata_sha256"],
                name=f"{source_path}:artifact_metadata_sha256",
            ),
            environment_identity=_mapping(
                manifest["environment_identity"],
                name=f"{source_path}:environment_identity",
            ),
        )
    evidence_hashes = _validate_evidence_hashes(manifest["evidence_sha256_by_role"])
    if (
        capture == "scheduler_terminal_reconciliation"
        and "scheduler_terminal_attestation" not in evidence_hashes
    ):
        raise ValueError(f"{source_path} scheduler failure lacks its terminal attestation hash")
    return {
        "seed": _integer(manifest["seed"], name=f"{source_path}:seed"),
        "terminal_status": "failed",
        "source_path": relative_posix_reference(source_path, base=reference_base),
        "source_sha256": source_sha256,
        "capture_method": capture,
        "evidence_availability": availability,
        "failure": dict(failure),
        "attempt_ledger": ledger,
        "attempt_ledger_sha256": _canonical_sha256(ledger),
        "evidence_sha256_by_role": evidence_hashes,
    }


def _load_success_result(
    value: Mapping[str, object],
    *,
    source_sha256: str,
    source_path: Path,
    source_hash: str,
    design_sha: str,
    runtime_sha: str,
    reference_base: Path,
) -> dict[str, object]:
    required = {
        "schema_version",
        "design_stage",
        "formal_eligibility",
        "per_seed_supports_formal_claim",
        "source_config_hash",
        "phase2_design_sha256",
        "phase2_runtime_contract_sha256",
        "seed",
        "artifact_metadata_sha256",
        "run_manifest_sha256",
        "environment_identity",
        "pre_oracle_safety_gate",
        "rollouts_jsonl",
        "rollouts_sha256",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"{source_path} result is missing required fields {sorted(missing)!r}")
    if (
        value["schema_version"] != PHASE2_RESULT_SCHEMA
        or value["design_stage"] != "confirmatory"
        or value["formal_eligibility"] is not True
        or value["per_seed_supports_formal_claim"] is not False
        or value["source_config_hash"] != source_hash
        or value["phase2_design_sha256"] != design_sha
        or value["phase2_runtime_contract_sha256"] != runtime_sha
    ):
        raise ValueError(f"{source_path} is not an identity-bound confirmatory result")
    gate = _mapping(
        value["pre_oracle_safety_gate"],
        name=f"{source_path}:pre_oracle_safety_gate",
    )
    if (
        gate.get("schema_version") != "phase2-pre-oracle-safety-gate/v1"
        or gate.get("formal_gate") is not True
        or gate.get("measure_only") is not False
        or gate.get("passed") is not True
        or gate.get("violations") != []
        or gate.get("beta_retuned") is not False
    ):
        raise ValueError(f"{source_path} did not terminate with a passed pre-oracle safety gate")
    return {
        "seed": _integer(value["seed"], name=f"{source_path}:seed"),
        "terminal_status": "success_result",
        "source_path": relative_posix_reference(source_path, base=reference_base),
        "source_sha256": source_sha256,
        "run_manifest_sha256": _digest(
            value["run_manifest_sha256"],
            name=f"{source_path}:run_manifest_sha256",
        ),
        "artifact_metadata_sha256": _digest(
            value["artifact_metadata_sha256"],
            name=f"{source_path}:artifact_metadata_sha256",
        ),
        "environment_identity": _validate_environment(
            value["environment_identity"],
            name=f"{source_path}:environment_identity",
        ),
        "pre_oracle_safety_gate_passed": True,
    }


_SUCCESS_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "terminal_status",
        "terminal",
        "supports_formal_claim",
        "seed",
        "source_config_hash",
        "phase2_design_sha256",
        "phase2_runtime_contract_sha256",
        "result",
        "rollout",
        "run_manifest_sha256",
        "artifact_metadata_sha256",
        "environment_identity",
        "pre_oracle_safety_gate_passed",
        "attempt_ledger",
        "seed_replacement_allowed",
    }
)


def build_phase2_seed_success_manifest(
    overlay_config: Mapping[str, object],
    result_json: str | os.PathLike[str],
    attempt_ledger: Mapping[str, object],
    *,
    reference_base: str | os.PathLike[str],
) -> dict[str, object]:
    """Bind one validated successful result to its complete immutable attempt history."""

    _, seeds, source_hash, design_sha, runtime_sha = _formal_contract(overlay_config)
    unresolved_result_path = Path(result_json)
    result, result_sha = _read_strict_json(
        unresolved_result_path,
        name="Phase-2 successful result",
    )
    result_path = unresolved_result_path.resolve()
    base = Path(reference_base).resolve()
    if result_path.parent != base:
        raise ValueError("successful result and terminal sidecar must be sibling files")
    loaded = _load_success_result(
        result,
        source_sha256=result_sha,
        source_path=result_path,
        source_hash=source_hash,
        design_sha=design_sha,
        runtime_sha=runtime_sha,
        reference_base=base,
    )
    _, rollout_evidence = _load_result_rollout_evidence(
        result,
        result_path=result_path,
    )
    seed = int(loaded["seed"])
    if seed not in seeds:
        raise ValueError("successful result seed is not predeclared by the overlay")
    ledger = _validate_attempt_ledger(
        attempt_ledger,
        terminal_status="success_result",
    )
    return {
        "schema_version": PHASE2_SEED_SUCCESS_SCHEMA,
        "terminal_status": "success_result",
        "terminal": True,
        "supports_formal_claim": False,
        "seed": seed,
        "source_config_hash": source_hash,
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract_sha256": runtime_sha,
        "result": {
            "path": relative_posix_reference(
                result_path,
                base=reference_base,
            ),
            "sha256": result_sha,
            "schema_version": PHASE2_RESULT_SCHEMA,
        },
        "rollout": rollout_evidence,
        "run_manifest_sha256": loaded["run_manifest_sha256"],
        "artifact_metadata_sha256": loaded["artifact_metadata_sha256"],
        "environment_identity": loaded["environment_identity"],
        "pre_oracle_safety_gate_passed": True,
        "attempt_ledger": ledger,
        "seed_replacement_allowed": False,
    }


def load_phase2_seed_success_spec(
    spec_json: str | os.PathLike[str],
) -> dict[str, object]:
    """Read a duplicate-key-free success spec containing only an attempt ledger."""

    value, _ = _read_strict_json(
        spec_json,
        name="Phase-2 success spec",
    )
    spec = _strict_mapping(
        value,
        name="success spec",
        keys={"attempt_ledger"},
    )
    return {
        "attempt_ledger": dict(
            _mapping(
                spec["attempt_ledger"],
                name="success spec.attempt_ledger",
            )
        )
    }


def write_phase2_seed_success_manifest(
    overlay_config: Mapping[str, object],
    result_json: str | os.PathLike[str],
    attempt_ledger: Mapping[str, object],
    output_json: str | os.PathLike[str],
) -> dict[str, object]:
    """Publish one success terminal sidecar once, without changing the result schema."""

    destination = Path(output_json)
    payload = build_phase2_seed_success_manifest(
        overlay_config,
        result_json,
        attempt_ledger,
        reference_base=destination.parent,
    )
    atomic_write_json(destination, payload, overwrite=False)
    return payload


def _resolve_success_result_reference(
    value: object,
    *,
    source_path: Path,
) -> Path:
    return _resolve_sibling_posix_reference(
        value,
        source_path=source_path,
        field_name="result.path",
    )


def _load_success_manifest(
    value: Mapping[str, object],
    *,
    source_sha256: str,
    source_path: Path,
    source_hash: str,
    design_sha: str,
    runtime_sha: str,
    reference_base: Path,
) -> tuple[dict[str, object], Path, Path]:
    manifest = _strict_mapping(
        value,
        name=str(source_path),
        keys=_SUCCESS_MANIFEST_KEYS,
    )
    if (
        manifest["schema_version"] != PHASE2_SEED_SUCCESS_SCHEMA
        or manifest["terminal_status"] != "success_result"
        or manifest["terminal"] is not True
        or manifest["supports_formal_claim"] is not False
        or manifest["seed_replacement_allowed"] is not False
        or manifest["source_config_hash"] != source_hash
        or manifest["phase2_design_sha256"] != design_sha
        or manifest["phase2_runtime_contract_sha256"] != runtime_sha
        or manifest["pre_oracle_safety_gate_passed"] is not True
    ):
        raise ValueError(f"{source_path} has an invalid success terminal identity")
    result_evidence = _strict_mapping(
        manifest["result"],
        name=f"{source_path}:result",
        keys={"path", "sha256", "schema_version"},
    )
    if result_evidence["schema_version"] != PHASE2_RESULT_SCHEMA:
        raise ValueError(f"{source_path} binds an unsupported result schema")
    expected_result_sha = _digest(
        result_evidence["sha256"],
        name=f"{source_path}:result.sha256",
    )
    unresolved_result_path = _resolve_success_result_reference(
        result_evidence["path"],
        source_path=source_path,
    )
    result, result_sha = _read_strict_json(
        unresolved_result_path,
        name=f"{source_path} bound success result",
    )
    result_path = unresolved_result_path.resolve()
    if result_path.parent != source_path.parent:
        raise ValueError(f"{source_path} bound success result escaped its manifest directory")
    if result_sha != expected_result_sha:
        raise ValueError(f"{source_path} bound success result SHA-256 changed")
    loaded = _load_success_result(
        result,
        source_sha256=result_sha,
        source_path=result_path,
        source_hash=source_hash,
        design_sha=design_sha,
        runtime_sha=runtime_sha,
        reference_base=reference_base,
    )
    rollout_path, rollout_from_result = _load_result_rollout_evidence(
        result,
        result_path=result_path,
    )
    rollout_evidence = _strict_mapping(
        manifest["rollout"],
        name=f"{source_path}:rollout",
        keys={"path", "sha256", "schema_version"},
    )
    _resolve_sibling_posix_reference(
        rollout_evidence["path"],
        source_path=source_path,
        field_name="rollout.path",
        expected_name=str(rollout_from_result["path"]),
    )
    rollout_sha = _digest(
        rollout_evidence["sha256"],
        name=f"{source_path}:rollout.sha256",
    )
    if (
        rollout_evidence["schema_version"] != PHASE2_ROLLOUT_SCHEMA
        or rollout_sha != rollout_from_result["sha256"]
    ):
        raise ValueError(f"{source_path} rollout sidecar disagrees with its bound result")
    seed = _integer(manifest["seed"], name=f"{source_path}:seed")
    environment = _validate_environment(
        manifest["environment_identity"],
        name=f"{source_path}:environment_identity",
    )
    run_manifest_sha = _digest(
        manifest["run_manifest_sha256"],
        name=f"{source_path}:run_manifest_sha256",
    )
    artifact_metadata_sha = _digest(
        manifest["artifact_metadata_sha256"],
        name=f"{source_path}:artifact_metadata_sha256",
    )
    if (
        seed != loaded["seed"]
        or run_manifest_sha != loaded["run_manifest_sha256"]
        or artifact_metadata_sha != loaded["artifact_metadata_sha256"]
        or environment != loaded["environment_identity"]
    ):
        raise ValueError(f"{source_path} success sidecar disagrees with its bound result")
    ledger = _validate_attempt_ledger(
        manifest["attempt_ledger"],
        terminal_status="success_result",
    )
    return (
        {
            "seed": seed,
            "terminal_status": "success_result",
            "source_path": relative_posix_reference(
                source_path,
                base=reference_base,
            ),
            "source_sha256": source_sha256,
            "result_path": relative_posix_reference(
                result_path,
                base=reference_base,
            ),
            "result_sha256": result_sha,
            "rollout_path": relative_posix_reference(
                rollout_path,
                base=reference_base,
            ),
            "rollout_sha256": rollout_sha,
            "rollout_schema_version": PHASE2_ROLLOUT_SCHEMA,
            "run_manifest_sha256": run_manifest_sha,
            "artifact_metadata_sha256": artifact_metadata_sha,
            "environment_identity": environment,
            "pre_oracle_safety_gate_passed": True,
            "attempt_ledger": ledger,
            "attempt_ledger_sha256": _canonical_sha256(ledger),
        },
        result_path,
        rollout_path,
    )


def _validate_campaign_job_id_uniqueness(
    seeds: tuple[int, ...],
    entries: Mapping[int, Mapping[str, object]],
) -> None:
    owner_by_job_id: dict[str, tuple[int, int]] = {}
    for seed in seeds:
        ledger = _mapping(
            entries[seed].get("attempt_ledger"),
            name=f"seed {seed} terminal attempt ledger",
        )
        attempts = ledger.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"seed {seed} terminal attempt ledger is empty")
        for attempt in attempts:
            normalized = _mapping(attempt, name=f"seed {seed} terminal attempt")
            job_id = _string(
                normalized.get("slurm_job_id"),
                name=f"seed {seed} attempt slurm_job_id",
            )
            attempt_index = _integer(
                normalized.get("attempt_index"),
                name=f"seed {seed} attempt_index",
                minimum=1,
            )
            owner = owner_by_job_id.get(job_id)
            if owner is not None:
                raise ValueError(
                    "campaign attempt ledgers repeat Slurm job identity "
                    f"{job_id!r} across seed/attempt {owner!r} and "
                    f"{(seed, attempt_index)!r}"
                )
            owner_by_job_id[job_id] = (seed, attempt_index)


def _verify_terminal_manifest_hashes(
    seeds: tuple[int, ...],
    *,
    terminal_paths: Mapping[int, Path],
    entries: Mapping[int, Mapping[str, object]],
) -> None:
    for seed in seeds:
        path = terminal_paths[seed]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"seed {seed} terminal manifest is no longer a regular file")
        observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_sha != entries[seed]["source_sha256"]:
            raise ValueError(f"seed {seed} terminal manifest changed during finalization")


def _verify_success_evidence_hashes(
    *,
    success_paths: Mapping[int, Path],
    rollout_paths: Mapping[int, Path],
    entries: Mapping[int, Mapping[str, object]],
) -> None:
    if set(success_paths) != set(rollout_paths):
        raise ValueError("successful result and rollout evidence seed sets differ")
    for seed, path in success_paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"seed {seed} successful result is no longer a regular file")
        observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_sha != entries[seed]["result_sha256"]:
            raise ValueError(f"seed {seed} successful result changed during finalization")
        rollout_path = rollout_paths[seed]
        if rollout_path.parent.resolve() != path.parent.resolve():
            raise ValueError(f"seed {seed} rollout JSONL escaped its result directory")
        if rollout_path.is_symlink() or not rollout_path.is_file():
            raise ValueError(f"seed {seed} rollout JSONL is no longer a regular file")
        observed_rollout_sha = hashlib.sha256(rollout_path.read_bytes()).hexdigest()
        if observed_rollout_sha != entries[seed]["rollout_sha256"]:
            raise ValueError(f"seed {seed} rollout JSONL changed during finalization")


def _failure_campaign_payload(
    *,
    seeds: tuple[int, ...],
    source_hash: str,
    design_sha: str,
    runtime_sha: str,
    entries: Mapping[int, Mapping[str, object]],
    failed_seeds: Sequence[int],
) -> dict[str, object]:
    ordered_failed = sorted(failed_seeds)
    terminal_failed = sorted(
        seed for seed in seeds if entries[seed].get("terminal_status") == "failed"
    )
    if ordered_failed != terminal_failed or not ordered_failed:
        raise ValueError("failed_seeds must exactly name explicit failed terminal manifests")
    return {
        "schema_version": PHASE2_CAMPAIGN_TERMINAL_SCHEMA,
        "status": "not_passed_due_to_seed_failure",
        "terminal": True,
        "supports_formal_claim": False,
        "source_config_hash": source_hash,
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract_sha256": runtime_sha,
        "declared_seeds": list(seeds),
        "num_declared_seeds": len(seeds),
        "terminal_seed_set_complete": True,
        "seed_replacement_allowed": False,
        "retry_policy": "single_predeclared_attempt_no_retry",
        "successful_result_seeds": sorted(set(seeds) - set(ordered_failed)),
        "failed_seeds": ordered_failed,
        "entries": [dict(entries[seed]) for seed in seeds],
        "aggregate_validation_failure": None,
        "primary_ci_computed": False,
        "primary_aggregate": None,
    }


def write_phase2_campaign_terminal(
    overlay_config: Mapping[str, object],
    terminal_inputs: Sequence[str | os.PathLike[str]],
    output_json: str | os.PathLike[str],
    *,
    aggregate_output_json: str | os.PathLike[str],
) -> dict[str, object]:
    """Finalize exactly 30 immutable seed slots and publish no partial CI."""

    validated, seeds, source_hash, design_sha, runtime_sha = _formal_contract(overlay_config)
    if isinstance(terminal_inputs, (str, bytes, bytearray)) or not isinstance(
        terminal_inputs, Sequence
    ):
        raise TypeError("terminal_inputs must be a sequence of result/failure paths")
    if len(terminal_inputs) != len(seeds):
        raise ValueError("campaign finalization requires exactly one terminal input per seed")
    destination = Path(output_json).resolve()
    aggregate_destination = Path(aggregate_output_json).resolve()
    if destination == aggregate_destination:
        raise ValueError("campaign terminal and primary aggregate outputs must be distinct")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing JSON: {destination}")
    if aggregate_destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing primary aggregate: {aggregate_destination}"
        )
    reference_base = destination.parent
    entries: dict[int, dict[str, object]] = {}
    terminal_paths: dict[int, Path] = {}
    success_paths: dict[int, Path] = {}
    rollout_paths: dict[int, Path] = {}
    failed_seeds: list[int] = []
    for raw_path in terminal_inputs:
        unresolved_source_path = Path(raw_path)
        value, source_sha = _read_strict_json(
            unresolved_source_path,
            name="Phase-2 terminal input",
        )
        source_path = unresolved_source_path.resolve()
        schema = value.get("schema_version")
        if schema == PHASE2_SEED_FAILURE_SCHEMA:
            entry = _load_failure_manifest(
                value,
                source_sha256=source_sha,
                source_path=source_path,
                source_hash=source_hash,
                design_sha=design_sha,
                runtime_sha=runtime_sha,
                reference_base=reference_base,
            )
            failed_seeds.append(int(entry["seed"]))
        elif schema == PHASE2_SEED_SUCCESS_SCHEMA:
            entry, result_path, rollout_path = _load_success_manifest(
                value,
                source_sha256=source_sha,
                source_path=source_path,
                source_hash=source_hash,
                design_sha=design_sha,
                runtime_sha=runtime_sha,
                reference_base=reference_base,
            )
            success_paths[int(entry["seed"])] = result_path
            rollout_paths[int(entry["seed"])] = rollout_path
        elif schema == PHASE2_RESULT_SCHEMA:
            raise ValueError(
                f"{source_path} is a bare result; formal finalization requires "
                f"a {PHASE2_SEED_SUCCESS_SCHEMA} sidecar with a complete attempt ledger"
            )
        else:
            raise ValueError(
                f"{source_path} is neither a success terminal manifest nor "
                "a failure terminal manifest"
            )
        seed = int(entry["seed"])
        if seed in entries:
            raise ValueError(f"duplicate terminal input for seed {seed}")
        entries[seed] = entry
        terminal_paths[seed] = source_path
    if set(entries) != set(seeds):
        raise ValueError(
            "terminal inputs must exactly match the predeclared seed set; "
            f"missing={sorted(set(seeds) - set(entries))!r}, "
            f"unexpected={sorted(set(entries) - set(seeds))!r}"
        )
    _validate_campaign_job_id_uniqueness(seeds, entries)

    if failed_seeds:
        payload = _failure_campaign_payload(
            seeds=seeds,
            source_hash=source_hash,
            design_sha=design_sha,
            runtime_sha=runtime_sha,
            entries=entries,
            failed_seeds=failed_seeds,
        )
        _verify_success_evidence_hashes(
            success_paths=success_paths,
            rollout_paths=rollout_paths,
            entries=entries,
        )
        _verify_terminal_manifest_hashes(
            seeds,
            terminal_paths=terminal_paths,
            entries=entries,
        )
        atomic_write_json(destination, payload, overwrite=False)
        return payload

    try:
        aggregate = build_common_beta_seed_aggregate(
            validated,
            [success_paths[seed] for seed in seeds],
            reference_base=reference_base,
        )
    except Exception:
        # This branch performs integrity checks only.  The original aggregate
        # exception is never reclassified as a failed seed or published.
        _verify_success_evidence_hashes(
            success_paths=success_paths,
            rollout_paths=rollout_paths,
            entries=entries,
        )
        _verify_terminal_manifest_hashes(
            seeds,
            terminal_paths=terminal_paths,
            entries=entries,
        )
        raise

    _verify_success_evidence_hashes(
        success_paths=success_paths,
        rollout_paths=rollout_paths,
        entries=entries,
    )
    _verify_terminal_manifest_hashes(
        seeds,
        terminal_paths=terminal_paths,
        entries=entries,
    )
    aggregate_payload = aggregate.to_dict()
    atomic_write_json(aggregate_destination, aggregate_payload, overwrite=False)
    aggregate_sha = hashlib.sha256(aggregate_destination.read_bytes()).hexdigest()
    evidence = aggregate.evidence.to_dict()
    payload = {
        "schema_version": PHASE2_CAMPAIGN_TERMINAL_SCHEMA,
        "status": "primary_aggregate_completed",
        "terminal": True,
        "supports_formal_claim": evidence["supports_pre_registered_claim"],
        "source_config_hash": source_hash,
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract_sha256": runtime_sha,
        "declared_seeds": list(seeds),
        "num_declared_seeds": len(seeds),
        "terminal_seed_set_complete": True,
        "seed_replacement_allowed": False,
        "retry_policy": "single_predeclared_attempt_no_retry",
        "successful_result_seeds": list(seeds),
        "failed_seeds": [],
        "entries": [dict(entries[seed]) for seed in seeds],
        "aggregate_validation_failure": None,
        "primary_ci_computed": True,
        "primary_aggregate": {
            "path": relative_posix_reference(
                aggregate_destination,
                base=reference_base,
            ),
            "sha256": aggregate_sha,
            "schema_version": aggregate_payload["schema_version"],
            "evidence_status": evidence["status"],
        },
    }
    atomic_write_json(destination, payload, overwrite=False)
    return payload


__all__ = [
    "PHASE2_ATTEMPT_LEDGER_SCHEMA",
    "PHASE2_ATTEMPT_LEDGER_SCHEMA_V1",
    "PHASE2_ATTEMPT_LEDGER_SCHEMA_V2",
    "PHASE2_CAMPAIGN_TERMINAL_SCHEMA",
    "PHASE2_FORMAL_SEED_COUNT",
    "PHASE2_FAILURE_EVIDENCE_AVAILABILITY_SCHEMA",
    "PHASE2_SEED_FAILURE_SCHEMA",
    "PHASE2_SEED_FAILURE_SCHEMA_V1",
    "PHASE2_SEED_SUCCESS_SCHEMA",
    "build_phase2_seed_success_manifest",
    "build_phase2_seed_failure_manifest",
    "build_phase2_seed_failure_manifest_from_spec",
    "load_phase2_seed_success_spec",
    "write_phase2_campaign_terminal",
    "write_phase2_seed_failure_manifest",
    "write_phase2_seed_success_manifest",
]
