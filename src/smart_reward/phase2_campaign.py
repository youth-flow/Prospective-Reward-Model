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
from pathlib import Path
from typing import Final

from .paths import relative_posix_reference
from .phase2_aggregate import build_common_beta_seed_aggregate
from .phase2_config import phase2_design_identity, validate_phase2_config
from .phase2_rollout import PHASE2_RESULT_SCHEMA, Phase2Design
from .repro import atomic_write_json

PHASE2_SEED_FAILURE_SCHEMA: Final = "phase2-seed-terminal-failure/v1"
PHASE2_CAMPAIGN_TERMINAL_SCHEMA: Final = "phase2-campaign-terminal/v1"
PHASE2_ATTEMPT_LEDGER_SCHEMA: Final = "phase2-seed-attempt-ledger/v1"
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
_ATTEMPT_KEYS = frozenset(
    {
        "attempt_index",
        "slurm_job_id",
        "status",
        "final_outcome_reveal_started",
        "log_sha256",
    }
)


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


def _validate_attempt_ledger(value: object) -> dict[str, object]:
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
        or ledger["retry_policy"] != "same_predeclared_seed_pre_outcome_infrastructure_only"
        or ledger["replacement_seed_allowed"] is not False
    ):
        raise ValueError("attempt_ledger has an invalid retry or replacement policy")
    raw_attempts = ledger["attempts"]
    if (
        isinstance(raw_attempts, (str, bytes, bytearray))
        or not isinstance(raw_attempts, Sequence)
        or not raw_attempts
    ):
        raise ValueError("attempt_ledger.attempts must be a non-empty sequence")
    attempts: list[dict[str, object]] = []
    job_ids: set[str] = set()
    for index, raw_attempt in enumerate(raw_attempts, start=1):
        attempt = _strict_mapping(
            raw_attempt,
            name=f"attempt_ledger.attempts[{index - 1}]",
            keys=_ATTEMPT_KEYS,
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
        expected_status = (
            "terminal_failure"
            if index == len(raw_attempts)
            else "infrastructure_failure_pre_outcome"
        )
        if status != expected_status:
            raise ValueError(
                "only pre-outcome infrastructure attempts may precede the terminal attempt"
            )
        reveal_started = attempt["final_outcome_reveal_started"]
        if not isinstance(reveal_started, bool):
            raise TypeError("attempt final_outcome_reveal_started must be bool")
        if status == "infrastructure_failure_pre_outcome" and reveal_started:
            raise ValueError("a retryable infrastructure attempt cannot reveal outcomes")
        attempts.append(
            {
                "attempt_index": index,
                "slurm_job_id": job_id,
                "status": status,
                "final_outcome_reveal_started": reveal_started,
                "log_sha256": _digest(
                    attempt["log_sha256"],
                    name=f"attempt_ledger.attempts[{index - 1}].log_sha256",
                ),
            }
        )
    return {
        "schema_version": PHASE2_ATTEMPT_LEDGER_SCHEMA,
        "retry_policy": "same_predeclared_seed_pre_outcome_infrastructure_only",
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


def build_phase2_seed_failure_manifest(
    overlay_config: Mapping[str, object],
    *,
    seed: int,
    run_manifest_sha256: str,
    artifact_metadata_sha256: str,
    environment_identity: Mapping[str, object],
    failure_stage: str,
    failure_class: str,
    failure_type: str,
    failure_message_sha256: str,
    final_outcome_reveal_started: bool,
    attempt_ledger: Mapping[str, object],
    evidence_sha256_by_role: Mapping[str, object],
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
    ledger = _validate_attempt_ledger(attempt_ledger)
    attempts = ledger["attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError("validated attempt ledger lost its attempts")
    if attempts[-1]["final_outcome_reveal_started"] is not final_outcome_reveal_started:
        raise ValueError("failure reveal boundary disagrees with the terminal attempt")
    return {
        "schema_version": PHASE2_SEED_FAILURE_SCHEMA,
        "terminal_status": "failed",
        "terminal": True,
        "supports_formal_claim": False,
        "seed": selected_seed,
        "source_config_hash": source_hash,
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract_sha256": runtime_sha,
        "run_manifest_sha256": _digest(
            run_manifest_sha256,
            name="run_manifest_sha256",
        ),
        "artifact_metadata_sha256": _digest(
            artifact_metadata_sha256,
            name="artifact_metadata_sha256",
        ),
        "environment_identity": _validate_environment(
            environment_identity,
            name="environment_identity",
        ),
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
        "evidence_sha256_by_role": _validate_evidence_hashes(evidence_sha256_by_role),
        "seed_replacement_allowed": False,
    }


def build_phase2_seed_failure_manifest_from_spec(
    overlay_config: Mapping[str, object],
    spec: Mapping[str, object],
) -> dict[str, object]:
    """Validate a compact CLI spec and bind it to the formal overlay identity."""

    value = _strict_mapping(
        spec,
        name="failure spec",
        keys={
            "seed",
            "run_manifest_sha256",
            "artifact_metadata_sha256",
            "environment_identity",
            "failure_stage",
            "failure_class",
            "failure_type",
            "failure_message_sha256",
            "final_outcome_reveal_started",
            "attempt_ledger",
            "evidence_sha256_by_role",
        },
    )
    return build_phase2_seed_failure_manifest(
        overlay_config,
        seed=value["seed"],
        run_manifest_sha256=value["run_manifest_sha256"],
        artifact_metadata_sha256=value["artifact_metadata_sha256"],
        environment_identity=_mapping(
            value["environment_identity"],
            name="failure spec.environment_identity",
        ),
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
        "run_manifest_sha256",
        "artifact_metadata_sha256",
        "environment_identity",
        "failure",
        "attempt_ledger",
        "evidence_sha256_by_role",
        "seed_replacement_allowed",
    }
)


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
    manifest = _strict_mapping(
        value,
        name=str(source_path),
        keys=_FAILURE_MANIFEST_KEYS,
    )
    if (
        manifest["schema_version"] != PHASE2_SEED_FAILURE_SCHEMA
        or manifest["terminal_status"] != "failed"
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
    ledger = _validate_attempt_ledger(manifest["attempt_ledger"])
    attempts = ledger["attempts"]
    if (
        not isinstance(attempts, list)
        or attempts[-1]["final_outcome_reveal_started"]
        is not failure["final_outcome_reveal_started"]
    ):
        raise ValueError(f"{source_path} failure boundary disagrees with its attempt ledger")
    return {
        "seed": _integer(manifest["seed"], name=f"{source_path}:seed"),
        "terminal_status": "failed",
        "source_path": relative_posix_reference(source_path, base=reference_base),
        "source_sha256": source_sha256,
        "run_manifest_sha256": _digest(
            manifest["run_manifest_sha256"],
            name=f"{source_path}:run_manifest_sha256",
        ),
        "artifact_metadata_sha256": _digest(
            manifest["artifact_metadata_sha256"],
            name=f"{source_path}:artifact_metadata_sha256",
        ),
        "environment_identity": _validate_environment(
            manifest["environment_identity"],
            name=f"{source_path}:environment_identity",
        ),
        "failure": dict(failure),
        "attempt_ledger_sha256": _canonical_sha256(ledger),
        "evidence_sha256_by_role": _validate_evidence_hashes(manifest["evidence_sha256_by_role"]),
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


def _failure_campaign_payload(
    *,
    seeds: tuple[int, ...],
    source_hash: str,
    design_sha: str,
    runtime_sha: str,
    entries: Mapping[int, Mapping[str, object]],
    failed_seeds: Sequence[int],
    aggregate_validation_failure: Mapping[str, object] | None,
) -> dict[str, object]:
    ordered_failed = sorted(failed_seeds)
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
        "retry_policy": "same_predeclared_seed_pre_outcome_infrastructure_only",
        "successful_result_seeds": sorted(set(seeds) - set(ordered_failed)),
        "failed_seeds": ordered_failed,
        "entries": [dict(entries[seed]) for seed in seeds],
        "aggregate_validation_failure": (
            None if aggregate_validation_failure is None else dict(aggregate_validation_failure)
        ),
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
    success_paths: dict[int, Path] = {}
    failed_seeds: list[int] = []
    for raw_path in terminal_inputs:
        source_path = Path(raw_path).resolve()
        value, source_sha = _read_strict_json(
            source_path,
            name="Phase-2 terminal input",
        )
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
        elif schema == PHASE2_RESULT_SCHEMA:
            seed_hint = _integer(value.get("seed"), name=f"{source_path}:seed")
            if seed_hint not in seeds:
                raise ValueError(
                    f"{source_path} attempts to substitute undeclared seed {seed_hint}"
                )
            try:
                entry = _load_success_result(
                    value,
                    source_sha256=source_sha,
                    source_path=source_path,
                    source_hash=source_hash,
                    design_sha=design_sha,
                    runtime_sha=runtime_sha,
                    reference_base=reference_base,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                entry = {
                    "seed": seed_hint,
                    "terminal_status": "invalid_result",
                    "source_path": relative_posix_reference(
                        source_path,
                        base=reference_base,
                    ),
                    "source_sha256": source_sha,
                    "validation_failure": {
                        "schema_version": "phase2-seed-result-validation-failure/v1",
                        "error_type": type(error).__name__,
                        "message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
                        "scientific_result_published": False,
                    },
                }
                failed_seeds.append(seed_hint)
            else:
                success_paths[int(entry["seed"])] = source_path
        else:
            raise ValueError(
                f"{source_path} is neither a confirmatory result nor a failure manifest"
            )
        seed = int(entry["seed"])
        if seed in entries:
            raise ValueError(f"duplicate terminal input for seed {seed}")
        entries[seed] = entry
    if set(entries) != set(seeds):
        raise ValueError(
            "terminal inputs must exactly match the predeclared seed set; "
            f"missing={sorted(set(seeds) - set(entries))!r}, "
            f"unexpected={sorted(set(entries) - set(seeds))!r}"
        )

    if failed_seeds:
        payload = _failure_campaign_payload(
            seeds=seeds,
            source_hash=source_hash,
            design_sha=design_sha,
            runtime_sha=runtime_sha,
            entries=entries,
            failed_seeds=failed_seeds,
            aggregate_validation_failure=None,
        )
        atomic_write_json(destination, payload, overwrite=False)
        return payload

    try:
        aggregate = build_common_beta_seed_aggregate(
            validated,
            [success_paths[seed] for seed in seeds],
            reference_base=reference_base,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        message_sha = hashlib.sha256(str(error).encode("utf-8")).hexdigest()
        implicated = [
            seed
            for seed in seeds
            if str(success_paths[seed]) in str(error)
            or str(success_paths[seed].resolve()) in str(error)
        ]
        conservatively_failed = implicated if implicated else list(seeds)
        payload = _failure_campaign_payload(
            seeds=seeds,
            source_hash=source_hash,
            design_sha=design_sha,
            runtime_sha=runtime_sha,
            entries=entries,
            failed_seeds=conservatively_failed,
            aggregate_validation_failure={
                "schema_version": "phase2-aggregate-validation-failure/v1",
                "error_type": type(error).__name__,
                "message_sha256": message_sha,
                "scope": ("identified_seed_inputs" if implicated else "cross_seed_or_unattributed"),
                "scientific_result_published": False,
            },
        )
        atomic_write_json(destination, payload, overwrite=False)
        return payload

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
        "retry_policy": "same_predeclared_seed_pre_outcome_infrastructure_only",
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
    "PHASE2_CAMPAIGN_TERMINAL_SCHEMA",
    "PHASE2_FORMAL_SEED_COUNT",
    "PHASE2_SEED_FAILURE_SCHEMA",
    "build_phase2_seed_failure_manifest",
    "build_phase2_seed_failure_manifest_from_spec",
    "write_phase2_campaign_terminal",
    "write_phase2_seed_failure_manifest",
]
