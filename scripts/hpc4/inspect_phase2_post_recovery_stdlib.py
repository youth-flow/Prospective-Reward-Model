#!/usr/bin/env python3
"""Dependency-free transport inspection for post-recovery HPC4 entrypoints.

This module intentionally imports only the Python standard library.  It does
not replace the scientific validators: GPU/CPU jobs must still reopen the
authorization, configuration, and predecessor lineage inside the frozen SIF.
Its narrower job is to make the login-node submission plane fail closed on
canonical paths, registered config identities, hashes, and the exact R2/R3
authorization transport needed by that later in-container verification.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final

R2_AUTHORIZATION_SCHEMA: Final = "prorm-phase2-recovery-success-authorization/v1"
R2_AUTHORIZATION_RELATIVE: Final = Path(
    "runs/phase2-recovery-pilot/recovery-success-authorization.json"
)
R2_REFERENCE_SCHEMA: Final = "prorm-phase2-recovery-success-reference/v1"
R2_PROJECTION_SCHEMA: Final = "prorm-phase2-recovery-success-projection/v1"

R3_AUTHORIZATION_SCHEMA: Final = "phase2-recovery-r3-gate-c-success-authorization/v1"
R3_AUTHORIZATION_RELATIVE: Final = Path(
    "runs/phase2-recovery-r3-controls/gate-c-success-authorization.json"
)
R3_REFERENCE_SCHEMA: Final = "phase2-recovery-r3-final-authorization-reference/v1"
R3_PROJECTION_SCHEMA: Final = "phase2-recovery-r3-final-authorization-projection/v1"
R3_GATE_R_SCHEMA: Final = "phase2-recovery-r3-success-authorization/v1"
R3_GATE_R_RELATIVE: Final = Path("runs/phase2-recovery-r3/recovery-success-authorization.json")
R3_GATE_C_RELATIVE: Final = Path("runs/phase2-recovery-r3-controls/gate-c-aggregate.json")
R3_GATE_R_ROLE: Final = "three_seed_all_scheduler_segments_audited_gate_r_capability"
R3_FINAL_ROLE: Final = "head_free_exact_three_by_three_gate_c_success_capability"
R3_EXECUTION_REVISION: Final = 3
R3_RECOVERY_STATUS: Final = "all_three_seeds_all_scheduler_segments_success"
R3_GATE_R_NEXT_ACTION: Final = "await_separate_gate_c_authorization"
R3_R2_DESIGN_SHA256: Final = "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"
R3_CONTINUABLE_TERMINAL_SCHEMA: Final = "phase2-recovery-r3-external-primary-segment-terminal/v1"
R3_CONTINUABLE_TERMINAL_ROLE: Final = (
    "external_scheduler_terminal_completed_zero_exit_continuation_required"
)
R3_COMPLETED_TERMINAL_SCHEMA: Final = (
    "phase2-recovery-r3-external-primary-segment-completed-terminal/v1"
)
R3_COMPLETED_TERMINAL_ROLE: Final = (
    "external_scheduler_terminal_completed_zero_exit_compute_complete"
)
R3_TERMINAL_MANIFEST_SCHEMA: Final = "phase2-recovery-r3-sacct-terminal-evidence-manifest/v1"

POST_RECOVERY_CONFIG_SCHEMA: Final = "prorm-common-beta-post-recovery-experiment/v1"
ADOPTED_SCHEDULE_SHA256: Final = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
ORDERED_SEEDS: Final = (20260801, 20260802, 20260803)
_HEX_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_RELATIVE_RE: Final = re.compile(r"[A-Za-z0-9._/-]+\Z")
_YAML_KEY_RE: Final = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):(.*)\Z")
_MAX_JSON_BYTES: Final = 128 * 1024 * 1024
_MAX_YAML_BYTES: Final = 16 * 1024 * 1024
_TERMINAL_EVIDENCE_FILENAMES: Final = frozenset(
    {
        "raw-sacct.psv",
        "parsed-sacct.json",
        "terminal-manifest.json",
    }
)
_R3_FINAL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "role",
        "recovery_design_sha256",
        "optimizer_schedule_sha256",
        "optimizer_schedule_is_unique",
        "execution_revision",
        "ordered_seeds",
        "gate_r_authorization_path",
        "gate_r_authorization_file_sha256",
        "gate_r_authorization_sha256",
        "gate_c_aggregate_path",
        "gate_c_aggregate_file_sha256",
        "gate_c_aggregate_sha256",
        "gate_c_source_set_sha256",
        "gate_r_passed",
        "gate_c_passed",
        "fresh_calibration_authorized",
        "authorized_information",
        "authorized_next_action",
        "formal_efficacy_claim_authorized",
        "recovery_or_control_outputs_reusable",
        "validation_or_heldout_access_authorized",
        "policy_or_final_utility_access_authorized",
        "transport_boundary",
        "authorization_sha256",
    }
)
_R3_GATE_R_FIELDS: Final = frozenset(
    {
        "schema_version",
        "role",
        "recovery_design_sha256",
        "optimizer_schedule_sha256",
        "execution_revision",
        "ordered_seeds",
        "recovery_status",
        "gate_r_passed",
        "fresh_calibration_authorized",
        "authorized_information",
        "authorized_next_action",
        "recovery_outputs_reusable",
        "validation_or_heldout_access_authorized",
        "policy_or_final_utility_access_authorized",
        "formal_efficacy_claim_authorized",
        "recovery_output_reuse",
        "transport_boundary",
        "operational_bundle",
        "terminal_set_sha256",
        "sources",
        "authorization_sha256",
    }
)
_R3_BUNDLE_FIELDS: Final = frozenset(
    {
        "path",
        "file_sha256",
        "bundle_semantic_sha256",
        "profile_run_sha256",
        "formal_profile_sha256",
        "resource_plan_sha256",
    }
)
_R3_SOURCE_FIELDS: Final = frozenset(
    {
        "task_id",
        "seed",
        "design_sha256",
        "logical_run_id",
        "materialization_attestation_sha256",
        "final_segment_index",
        "completion_receipt_sha256s",
        "segments",
    }
)
_R3_SEGMENT_FIELDS: Final = frozenset(
    {
        "segment_index",
        "scheduler_segment_id",
        "scheduler_array_job_id",
        "scheduler_job_id",
        "scheduler_job_selector",
        "runtime_closure_path",
        "runtime_closure_file_sha256",
        "runtime_closure_sha256",
        "segment_outcome_sha256",
        "terminal_kind",
        "terminal_schema_version",
        "terminal_role",
        "terminal_evidence_directory",
        "terminal_manifest_file_sha256",
        "terminal_raw_sacct_sha256",
        "terminal_sha256",
    }
)
_R3_TERMINAL_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema_version",
        "producer_kind",
        "json_role_is_authority",
        "locked_sacct_command",
        "raw_sacct",
        "parsed_evidence",
        "terminal_artifact",
        "manifest_sha256",
    }
)
_R3_FINAL_TRANSPORT_BOUNDARY: Final = {
    "parameters": False,
    "optimizer_moments": False,
    "checkpoints": False,
    "labels_or_data": False,
    "gradients_or_directions": False,
    "validation_or_test_values": False,
    "policy_outputs": False,
    "utility_values": False,
    "beta_values": False,
}
_R3_GATE_R_RECOVERY_REUSE: Final = {
    "beta": False,
    "reward_model_parameters": False,
    "policy": False,
}
_R3_GATE_R_TRANSPORT_BOUNDARY: Final = {
    "trained_parameter_payload_included": False,
    "checkpoint_payload_included": False,
    "optimizer_state_payload_included": False,
    "training_data_payload_included": False,
    "label_payload_included": False,
}
_R3_TASK_SEED_MAP: Final = ((0, 20260801), (1, 20260802), (2, 20260803))
_POSITIVE_JOB_ID_RE: Final = re.compile(r"[1-9][0-9]*\Z")
_SAFE_COMPONENT_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_GATEP_ATTEMPT_RE: Final = re.compile(r"gatep-attempt-(?!000)[0-9]{3}\Z")


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _HEX_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
        or absolute == Path(absolute.anchor)
    ):
        raise ValueError(f"{name} must be a canonical real directory")
    return absolute


def _canonical_file(path: Path, *, name: str, maximum_bytes: int) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
        or metadata.st_size < 1
        or metadata.st_size > maximum_bytes
    ):
        raise ValueError(f"{name} must be a bounded canonical regular file")
    return absolute


def _reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(
    path: Path,
    *,
    name: str,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> tuple[dict[str, object], bytes]:
    source = _canonical_file(path, name=name, maximum_bytes=maximum_bytes)
    raw = source.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite JSON constant {token!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value, raw


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _closed_mapping(
    value: object,
    *,
    name: str,
    fields: frozenset[str],
) -> dict[str, object]:
    result = _mapping(value, name=name)
    if set(result) != fields:
        raise ValueError(f"{name} has an invalid closed field set")
    return result


def _sequence(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _semantic_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _identity_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _positive_job_id(value: object, *, name: str) -> str:
    if type(value) is not str or _POSITIVE_JOB_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a positive decimal Slurm job ID")
    return value


def _safe_relative(value: object, *, name: str) -> PurePosixPath:
    if type(value) is not str or _SAFE_RELATIVE_RE.fullmatch(value) is None:
        raise ValueError(f"{name} relative path is invalid")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.as_posix() != value
        or "." in relative.parts
        or ".." in relative.parts
        or any(_SAFE_COMPONENT_RE.fullmatch(part) is None for part in relative.parts)
    ):
        raise ValueError(f"{name} must be a safe canonical POSIX relative path")
    return relative


def _require_bundle_namespace(value: object) -> PurePosixPath:
    relative = _safe_relative(value, name="Gate-P operational bundle")
    parts = relative.parts
    if (
        len(parts) != 6
        or parts[:3] != ("runs", "phase2-recovery-r3", "gatep")
        or _GATEP_ATTEMPT_RE.fullmatch(parts[4]) is None
        or parts[5] != "gatep-operational-bundle.json"
    ):
        raise ValueError("Gate-P bundle path is outside its exact R3 evidence namespace")
    return relative


def _require_closure_namespace(
    value: object,
    *,
    task_id: int,
) -> PurePosixPath:
    relative = _safe_relative(value, name="R3 runtime closure")
    parts = relative.parts
    if (
        len(parts) < 4
        or parts[:2] != ("runs", "phase2-recovery-r3")
        or parts[-2] != "runtime-closures"
        or parts[-1] != f"task-{task_id}.json"
    ):
        raise ValueError("R3 runtime closure path is outside its exact evidence namespace")
    return relative


def _require_terminal_namespace(
    value: object,
    *,
    task_id: int,
    segment_index: int,
) -> PurePosixPath:
    relative = _safe_relative(value, name="R3 terminal evidence directory")
    parts = relative.parts
    if (
        len(parts) < 4
        or parts[:2] != ("runs", "phase2-recovery-r3")
        or parts[-2] != "terminal-evidence"
        or parts[-1] != f"task-{task_id}-segment-{segment_index}"
    ):
        raise ValueError("R3 terminal path is outside its exact evidence namespace")
    return relative


def _canonical_project_path(
    root: Path,
    relative_value: object,
    *,
    name: str,
    kind: str,
    expected_relative: Path | None = None,
) -> Path:
    if not isinstance(relative_value, str) or _SAFE_RELATIVE_RE.fullmatch(relative_value) is None:
        raise ValueError(f"{name} relative path is invalid")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"{name} relative path escapes the project root")
    relative_path = Path(*relative.parts)
    if expected_relative is not None and relative_path != expected_relative:
        raise ValueError(f"{name} differs from its exact production path")
    candidate = root / relative_path
    if kind == "file":
        return _canonical_file(candidate, name=name, maximum_bytes=_MAX_JSON_BYTES)
    if kind == "directory":
        return _canonical_directory(candidate, name=name)
    raise AssertionError("invalid canonical project path kind")


def _r3_terminal_directory(
    root: Path,
    relative: PurePosixPath,
    *,
    segment: Mapping[str, object],
    expected_producer_kind: str,
    name: str,
) -> Path:
    terminal = _canonical_project_path(
        root,
        relative.as_posix(),
        name=name,
        kind="directory",
    )
    entries = list(os.scandir(terminal))
    if (
        len(entries) != len(_TERMINAL_EVIDENCE_FILENAMES)
        or {entry.name for entry in entries} != _TERMINAL_EVIDENCE_FILENAMES
    ):
        raise ValueError(
            f"{name} must contain exactly "
            f"{sorted(_TERMINAL_EVIDENCE_FILENAMES)!r} and no other entries"
        )
    files: dict[str, Path] = {}
    for entry in entries:
        path = terminal / entry.name
        files[entry.name] = _canonical_file(
            path,
            name=f"{name}/{entry.name}",
            maximum_bytes=(1024 * 1024 if entry.name == "raw-sacct.psv" else _MAX_JSON_BYTES),
        )

    raw_sha256 = _digest(
        segment["terminal_raw_sacct_sha256"],
        name=f"{name} raw sacct SHA256",
    )
    raw_bytes = files["raw-sacct.psv"].read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != raw_sha256:
        raise ValueError(f"{name} raw sacct bytes changed")

    manifest_sha256 = _digest(
        segment["terminal_manifest_file_sha256"],
        name=f"{name} manifest file SHA256",
    )
    manifest, manifest_raw = _strict_json(
        files["terminal-manifest.json"],
        name=f"{name} manifest",
    )
    if hashlib.sha256(
        manifest_raw
    ).hexdigest() != manifest_sha256 or manifest_raw != _canonical_json_bytes(manifest):
        raise ValueError(f"{name} manifest transport is invalid")
    manifest = _closed_mapping(
        manifest,
        name=f"{name} manifest",
        fields=_R3_TERMINAL_MANIFEST_FIELDS,
    )
    unsigned_manifest = dict(manifest)
    manifest_semantic_sha256 = _digest(
        unsigned_manifest.pop("manifest_sha256"),
        name=f"{name} manifest semantic SHA256",
    )
    if (
        manifest["schema_version"] != R3_TERMINAL_MANIFEST_SCHEMA
        or manifest["producer_kind"] != expected_producer_kind
        or manifest["json_role_is_authority"] is not False
        or _identity_sha256(unsigned_manifest) != manifest_semantic_sha256
    ):
        raise ValueError(f"{name} manifest identity/self-hash is invalid")

    raw_binding = _closed_mapping(
        manifest["raw_sacct"],
        name=f"{name} raw sacct binding",
        fields=frozenset({"filename", "sha256", "size_bytes"}),
    )
    parsed_binding = _closed_mapping(
        manifest["parsed_evidence"],
        name=f"{name} parsed evidence binding",
        fields=frozenset({"filename", "file_sha256", "size_bytes", "inspection_sha256"}),
    )
    if (
        raw_binding["filename"] != "raw-sacct.psv"
        or raw_binding["sha256"] != raw_sha256
        or type(raw_binding["size_bytes"]) is not int
        or raw_binding["size_bytes"] != len(raw_bytes)
        or parsed_binding["filename"] != "parsed-sacct.json"
        or type(parsed_binding["size_bytes"]) is not int
        or parsed_binding["size_bytes"] < 1
    ):
        raise ValueError(f"{name} manifest file bindings are invalid")
    _digest(
        parsed_binding["inspection_sha256"],
        name=f"{name} parsed inspection SHA256",
    )
    parsed_sha256 = _digest(
        parsed_binding["file_sha256"],
        name=f"{name} parsed evidence file SHA256",
    )
    parsed_bytes = files["parsed-sacct.json"].read_bytes()
    parsed_payload, parsed_raw = _strict_json(
        files["parsed-sacct.json"],
        name=f"{name} parsed evidence",
    )
    if (
        hashlib.sha256(parsed_bytes).hexdigest() != parsed_sha256
        or len(parsed_bytes) != parsed_binding["size_bytes"]
        or parsed_raw != _canonical_json_bytes(parsed_payload)
    ):
        raise ValueError(f"{name} parsed evidence transport is invalid")

    terminal_artifact = _mapping(
        manifest["terminal_artifact"],
        name=f"{name} terminal artifact",
    )
    unsigned_terminal = dict(terminal_artifact)
    terminal_sha256 = _digest(
        unsigned_terminal.pop("terminal_sha256", None),
        name=f"{name} terminal semantic SHA256",
    )
    scheduler_evidence = _mapping(
        terminal_artifact.get("scheduler_evidence"),
        name=f"{name} scheduler evidence",
    )
    if (
        terminal_sha256
        != _digest(segment["terminal_sha256"], name=f"{name} Gate-R terminal SHA256")
        or _identity_sha256(unsigned_terminal) != terminal_sha256
        or terminal_artifact.get("schema_version") != segment["terminal_schema_version"]
        or terminal_artifact.get("role") != segment["terminal_role"]
        or manifest["locked_sacct_command"] != scheduler_evidence.get("locked_sacct_command")
    ):
        raise ValueError(f"{name} terminal artifact binding is invalid")
    return terminal


def _r3_bindings(
    payload: Mapping[str, object],
    *,
    root: Path,
) -> list[tuple[str, Path, str | None]]:
    final = _closed_mapping(
        payload,
        name="R3 final authorization",
        fields=_R3_FINAL_FIELDS,
    )
    final_transport = _closed_mapping(
        final["transport_boundary"],
        name="R3 final transport boundary",
        fields=frozenset(_R3_FINAL_TRANSPORT_BOUNDARY),
    )
    design_sha256 = _digest(
        final["recovery_design_sha256"],
        name="R3 final recovery design SHA256",
    )
    if (
        final["schema_version"] != R3_AUTHORIZATION_SCHEMA
        or final["role"] != R3_FINAL_ROLE
        or final["optimizer_schedule_sha256"] != ADOPTED_SCHEDULE_SHA256
        or final["optimizer_schedule_is_unique"] is not True
        or final["execution_revision"] != R3_EXECUTION_REVISION
        or final["ordered_seeds"] != list(ORDERED_SEEDS)
        or final["gate_r_passed"] is not True
        or final["gate_c_passed"] is not True
        or final["fresh_calibration_authorized"] is not True
        or final["authorized_information"]
        != "gate_r_design_optimizer_schedule_and_gate_source_hashes_only"
        or final["authorized_next_action"] != "materialize_fresh_common_beta_calibration"
        or final["formal_efficacy_claim_authorized"] is not False
        or final["recovery_or_control_outputs_reusable"] is not False
        or final["validation_or_heldout_access_authorized"] is not False
        or final["policy_or_final_utility_access_authorized"] is not False
        or final_transport != _R3_FINAL_TRANSPORT_BOUNDARY
    ):
        raise ValueError("R3 final authorization crosses its exact head-free boundary")
    for field in (
        "gate_r_authorization_file_sha256",
        "gate_r_authorization_sha256",
        "gate_c_aggregate_file_sha256",
        "gate_c_aggregate_sha256",
        "gate_c_source_set_sha256",
    ):
        _digest(final[field], name=f"R3 final {field}")

    gate_r_file_sha256 = str(final["gate_r_authorization_file_sha256"])
    gate_c_file_sha256 = str(final["gate_c_aggregate_file_sha256"])
    gate_r = _canonical_project_path(
        root,
        final["gate_r_authorization_path"],
        name="R3 Gate-R authorization",
        kind="file",
        expected_relative=R3_GATE_R_RELATIVE,
    )
    gate_c = _canonical_project_path(
        root,
        final["gate_c_aggregate_path"],
        name="R3 Gate-C aggregate",
        kind="file",
        expected_relative=R3_GATE_C_RELATIVE,
    )
    if _sha256(gate_r) != gate_r_file_sha256:
        raise ValueError("R3 Gate-R authorization bytes changed")
    if _sha256(gate_c) != gate_c_file_sha256:
        raise ValueError("R3 Gate-C aggregate bytes changed")

    gate_r_payload, gate_r_raw = _strict_json(gate_r, name="R3 Gate-R authorization")
    gate_c_payload, gate_c_raw = _strict_json(gate_c, name="R3 Gate-C aggregate")
    if gate_r_raw != _canonical_json_bytes(gate_r_payload) or gate_c_raw != _canonical_json_bytes(
        gate_c_payload
    ):
        raise ValueError("R3 Gate-R/Gate-C source transport is not canonical")
    gate_r_payload = _closed_mapping(
        gate_r_payload,
        name="R3 Gate-R authorization",
        fields=_R3_GATE_R_FIELDS,
    )
    gate_r_unsigned = dict(gate_r_payload)
    gate_r_semantic_sha256 = _digest(
        gate_r_unsigned.pop("authorization_sha256"),
        name="R3 Gate-R authorization semantic SHA256",
    )
    gate_r_reuse = _closed_mapping(
        gate_r_payload["recovery_output_reuse"],
        name="R3 Gate-R recovery-output reuse boundary",
        fields=frozenset(_R3_GATE_R_RECOVERY_REUSE),
    )
    gate_r_transport = _closed_mapping(
        gate_r_payload["transport_boundary"],
        name="R3 Gate-R transport boundary",
        fields=frozenset(_R3_GATE_R_TRANSPORT_BOUNDARY),
    )
    if (
        gate_r_payload["schema_version"] != R3_GATE_R_SCHEMA
        or gate_r_payload["role"] != R3_GATE_R_ROLE
        or gate_r_payload["recovery_design_sha256"] != design_sha256
        or design_sha256 == R3_R2_DESIGN_SHA256
        or gate_r_payload["optimizer_schedule_sha256"] != ADOPTED_SCHEDULE_SHA256
        or gate_r_payload["execution_revision"] != R3_EXECUTION_REVISION
        or gate_r_payload["ordered_seeds"] != list(ORDERED_SEEDS)
        or gate_r_payload["recovery_status"] != R3_RECOVERY_STATUS
        or gate_r_payload["gate_r_passed"] is not True
        or gate_r_payload["fresh_calibration_authorized"] is not False
        or gate_r_payload["authorized_information"] != "optimizer_schedule_only"
        or gate_r_payload["authorized_next_action"] != R3_GATE_R_NEXT_ACTION
        or gate_r_payload["recovery_outputs_reusable"] is not False
        or gate_r_payload["validation_or_heldout_access_authorized"] is not False
        or gate_r_payload["policy_or_final_utility_access_authorized"] is not False
        or gate_r_payload["formal_efficacy_claim_authorized"] is not False
        or gate_r_reuse != _R3_GATE_R_RECOVERY_REUSE
        or gate_r_transport != _R3_GATE_R_TRANSPORT_BOUNDARY
        or _semantic_sha256(gate_r_unsigned) != gate_r_semantic_sha256
        or gate_r_semantic_sha256 != final["gate_r_authorization_sha256"]
    ):
        raise ValueError("R3 Gate-R authorization identity/boundary is invalid")
    gate_c_unsigned = dict(gate_c_payload)
    gate_c_semantic_sha256 = _digest(
        gate_c_unsigned.pop("aggregate_sha256", None),
        name="R3 Gate-C aggregate semantic SHA256",
    )
    gate_c_sources = _sequence(
        gate_c_payload.get("sources"),
        name="R3 Gate-C aggregate sources",
    )
    if (
        _identity_sha256(gate_c_unsigned) != gate_c_semantic_sha256
        or gate_c_semantic_sha256 != final["gate_c_aggregate_sha256"]
        or _digest(
            gate_c_payload.get("source_set_sha256"),
            name="R3 Gate-C aggregate source-set SHA256",
        )
        != _identity_sha256({"sources": gate_c_sources})
        or gate_c_payload.get("source_set_sha256") != final["gate_c_source_set_sha256"]
    ):
        raise ValueError("R3 Gate-C aggregate semantic binding is invalid")

    bindings: list[tuple[str, Path, str | None]] = [
        ("file", gate_r, gate_r_file_sha256),
        ("file", gate_c, gate_c_file_sha256),
    ]
    bundle = _closed_mapping(
        gate_r_payload["operational_bundle"],
        name="R3 Gate-R operational bundle",
        fields=_R3_BUNDLE_FIELDS,
    )
    bundle_relative = _require_bundle_namespace(bundle["path"])
    for field in _R3_BUNDLE_FIELDS - {"path"}:
        _digest(bundle[field], name=f"R3 Gate-P operational bundle {field}")
    bundle_path = _canonical_project_path(
        root,
        bundle_relative.as_posix(),
        name="R3 Gate-P operational bundle",
        kind="file",
    )
    bundle_sha256 = str(bundle["file_sha256"])
    if _sha256(bundle_path) != bundle_sha256:
        raise ValueError("R3 Gate-P operational bundle bytes changed")
    bindings.append(("file", bundle_path, bundle_sha256))

    sources = _sequence(gate_r_payload["sources"], name="R3 Gate-R sources")
    if len(sources) != len(_R3_TASK_SEED_MAP):
        raise ValueError("R3 Gate-R authorization must contain exactly three sources")
    logical_run_ids: set[str] = set()
    for source_index, (source_value, (task_id, seed)) in enumerate(
        zip(sources, _R3_TASK_SEED_MAP, strict=True)
    ):
        source = _closed_mapping(
            source_value,
            name=f"R3 Gate-R source {source_index}",
            fields=_R3_SOURCE_FIELDS,
        )
        logical_run_id = _digest(
            source["logical_run_id"],
            name=f"R3 Gate-R source {source_index} logical run SHA256",
        )
        if logical_run_id in logical_run_ids:
            raise ValueError("R3 Gate-R sources reuse a logical run identity")
        logical_run_ids.add(logical_run_id)
        if (
            type(source["task_id"]) is not int
            or type(source["seed"]) is not int
            or source["task_id"] != task_id
            or source["seed"] != seed
            or source["design_sha256"] != design_sha256
        ):
            raise ValueError("R3 Gate-R source violates ordered task/seed identity")
        _digest(
            source["materialization_attestation_sha256"],
            name=f"R3 Gate-R source {source_index} materialization SHA256",
        )
        receipts = _sequence(
            source["completion_receipt_sha256s"],
            name=f"R3 Gate-R source {source_index} completion receipts",
        )
        if len(receipts) != 2:
            raise ValueError("R3 Gate-R source must bind both completion receipts")
        for receipt in receipts:
            _digest(receipt, name="R3 Gate-R completion receipt SHA256")

        segments = _sequence(
            source["segments"],
            name=f"R3 Gate-R source {source_index} segments",
        )
        if (
            not segments
            or type(source["final_segment_index"]) is not int
            or source["final_segment_index"] != len(segments)
        ):
            raise ValueError("R3 Gate-R final segment index is invalid")
        for expected_segment, segment_value in enumerate(segments, start=1):
            segment = _closed_mapping(
                segment_value,
                name=f"R3 Gate-R source {source_index} segment {expected_segment}",
                fields=_R3_SEGMENT_FIELDS,
            )
            final_segment = expected_segment == len(segments)
            expected_kind = "completed" if final_segment else "continuable"
            expected_schema = (
                R3_COMPLETED_TERMINAL_SCHEMA if final_segment else R3_CONTINUABLE_TERMINAL_SCHEMA
            )
            expected_role = (
                R3_COMPLETED_TERMINAL_ROLE if final_segment else R3_CONTINUABLE_TERMINAL_ROLE
            )
            if (
                type(segment["segment_index"]) is not int
                or segment["segment_index"] != expected_segment
                or segment["terminal_kind"] != expected_kind
                or segment["terminal_schema_version"] != expected_schema
                or segment["terminal_role"] != expected_role
            ):
                raise ValueError("R3 Gate-R scheduler segment sequence is invalid")
            for field in (
                "scheduler_segment_id",
                "runtime_closure_file_sha256",
                "runtime_closure_sha256",
                "segment_outcome_sha256",
                "terminal_manifest_file_sha256",
                "terminal_raw_sacct_sha256",
                "terminal_sha256",
            ):
                _digest(segment[field], name=f"R3 Gate-R segment {field}")
            array_job_id = _positive_job_id(
                segment["scheduler_array_job_id"],
                name="R3 scheduler array job ID",
            )
            _positive_job_id(
                segment["scheduler_job_id"],
                name="R3 scheduler allocation job ID",
            )
            if segment["scheduler_job_selector"] != f"{array_job_id}_{task_id}":
                raise ValueError("R3 scheduler selector is invalid")

            closure_relative = _require_closure_namespace(
                segment["runtime_closure_path"],
                task_id=task_id,
            )
            terminal_relative = _require_terminal_namespace(
                segment["terminal_evidence_directory"],
                task_id=task_id,
                segment_index=expected_segment,
            )
            closure = _canonical_project_path(
                root,
                closure_relative.as_posix(),
                name="R3 runtime closure",
                kind="file",
            )
            closure_sha256 = str(segment["runtime_closure_file_sha256"])
            if _sha256(closure) != closure_sha256:
                raise ValueError("R3 runtime closure bytes changed")
            terminal = _r3_terminal_directory(
                root,
                terminal_relative,
                segment=segment,
                expected_producer_kind=(
                    "completed_primary_terminal"
                    if final_segment
                    else "continuable_primary_terminal"
                ),
                name=(f"R3 source {source_index} segment {expected_segment} terminal evidence"),
            )
            bindings.extend(
                (
                    ("file", closure, closure_sha256),
                    ("directory", terminal, None),
                )
            )

    terminal_set_sha256 = _digest(
        gate_r_payload["terminal_set_sha256"],
        name="R3 Gate-R terminal-set SHA256",
    )
    if terminal_set_sha256 != _semantic_sha256({"sources": sources}):
        raise ValueError("R3 Gate-R terminal-set SHA256 is invalid")

    paths = [path for _, path, _ in bindings]
    if len(paths) != len(set(paths)):
        raise ValueError("R3 authorization bind plan contains duplicate path roles")
    return bindings


def inspect_authorization(
    authorization_path: Path,
    *,
    expected_sha256: str,
    project_root: Path,
) -> dict[str, object]:
    root = _canonical_directory(project_root, name="project root")
    expected = _digest(expected_sha256, name="authorization SHA256")
    source = _canonical_file(
        authorization_path,
        name="recovery authorization",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    if _sha256(source) != expected:
        raise ValueError("recovery authorization SHA256 mismatch")
    payload, raw = _strict_json(source, name="recovery authorization")
    schema = payload.get("schema_version")
    if schema == R2_AUTHORIZATION_SCHEMA:
        expected_path = root / R2_AUTHORIZATION_RELATIVE
        bindings: list[tuple[str, Path, str | None]] = []
    elif schema == R3_AUTHORIZATION_SCHEMA:
        expected_path = root / R3_AUTHORIZATION_RELATIVE
        if raw != _canonical_json_bytes(payload):
            raise ValueError("R3 final authorization is not canonical JSON")
        unsigned = dict(payload)
        semantic_sha256 = _digest(
            unsigned.pop("authorization_sha256", None),
            name="R3 final authorization semantic SHA256",
        )
        if _identity_sha256(unsigned) != semantic_sha256:
            raise ValueError("R3 final authorization semantic SHA256 is invalid")
        bindings = _r3_bindings(payload, root=root)
    else:
        raise ValueError("authorization schema is neither exact R2 nor exact R3")
    if source != expected_path:
        raise ValueError("authorization is not at its schema-specific canonical path")
    schedule = _digest(
        payload.get("optimizer_schedule_sha256"),
        name="authorization optimizer schedule SHA256",
    )
    if schedule != ADOPTED_SCHEDULE_SHA256:
        raise ValueError("authorization does not carry the adopted optimizer schedule")
    return {
        "schema_version": schema,
        "authorization_sha256": expected,
        "optimizer_schedule_sha256": schedule,
        "bindings": [
            {
                "kind": kind,
                "path": os.fspath(path),
                "sha256": digest,
            }
            for kind, path, digest in bindings
        ],
    }


def _yaml_scalar(raw: str, *, name: str) -> object:
    value = raw.strip()
    if not value:
        raise ValueError(f"{name} is not a scalar")
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if value[:1] in {"'", '"'}:
        try:
            decoded = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{name} has an invalid quoted scalar") from error
        if not isinstance(decoded, str):
            raise ValueError(f"{name} quoted scalar is not a string")
        return decoded
    if any(token in value for token in (" #", "\t", "\r", "\n")):
        raise ValueError(f"{name} plain scalar is unsafe")
    return value


def _selected_yaml(path: Path) -> tuple[dict[tuple[str, ...], object], list[int]]:
    source = _canonical_file(path, name="post-recovery overlay", maximum_bytes=_MAX_YAML_BYTES)
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("post-recovery overlay must be UTF-8") from error
    if "\t" in text or "\r" in text:
        raise ValueError("post-recovery overlay uses forbidden whitespace")

    targets = {
        ("schema_version",),
        ("design", "name"),
        ("design", "stage"),
        ("design", "pilot_phase"),
        ("design", "source_config"),
        ("design", "source_config_hash"),
        ("recovery_success_reference", "schema_version"),
        ("recovery_success_reference", "artifact_sha256"),
        (
            "recovery_success_reference",
            "authorization_projection",
            "schema_version",
        ),
        (
            "recovery_success_reference",
            "authorization_projection",
            "source_schema_version",
        ),
        (
            "recovery_success_reference",
            "authorization_projection",
            "optimizer_schedule_sha256",
        ),
        (
            "reward_model",
            "optimizer_protocol",
            "source_recovery_authorization_sha256",
        ),
        (
            "reward_model",
            "optimizer_protocol",
            "learning_rate_schedule",
            "schedule_sha256",
        ),
        ("objective", "common_beta", "beta_source_aggregate_sha256"),
        ("evaluation", "max_length", "horizon_grid_index"),
        ("evaluation", "max_length", "parent_pilot_aggregate_sha256"),
    }
    values: dict[tuple[str, ...], object] = {}
    contexts: list[tuple[int, str]] = []
    seeds: list[int] = []
    in_seed_list = False
    seed_indent = -1

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"overlay line {line_number} has non-canonical indentation")
        stripped = line[indent:]
        while contexts and contexts[-1][0] >= indent:
            contexts.pop()
        if in_seed_list and indent <= seed_indent:
            in_seed_list = False
        if stripped.startswith("- "):
            if not in_seed_list or indent != seed_indent + 2:
                continue
            seed = _yaml_scalar(stripped[2:], name=f"overlay seed line {line_number}")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("post-recovery seeds must be integers")
            seeds.append(seed)
            continue
        match = _YAML_KEY_RE.fullmatch(stripped)
        if match is None:
            continue
        key, remainder = match.groups()
        path_key = tuple(item[1] for item in contexts) + (key,)
        if not remainder.strip():
            contexts.append((indent, key))
            if path_key == ("run", "seeds"):
                in_seed_list = True
                seed_indent = indent
            continue
        if path_key in targets:
            if path_key in values:
                raise ValueError(f"post-recovery overlay repeats {'.'.join(path_key)}")
            values[path_key] = _yaml_scalar(
                remainder,
                name=f"overlay {'.'.join(path_key)}",
            )
    return values, seeds


def _required(
    values: Mapping[tuple[str, ...], object],
    path: tuple[str, ...],
) -> object:
    if path not in values:
        raise ValueError(f"post-recovery overlay lacks {'.'.join(path)}")
    return values[path]


def _registered_identity(
    identities: Mapping[str, object],
    *,
    relative: str,
    path: Path,
    seed_count: int,
) -> str:
    configs = _mapping(identities.get("configs"), name="config identity registry")
    entry = _mapping(configs.get(relative), name=f"config identity {relative}")
    if entry.get("seed_count") != seed_count:
        raise ValueError(f"registered seed count differs for {relative}")
    if entry.get("file_sha256") != _sha256(path):
        raise ValueError(f"registered file SHA256 differs for {relative}")
    return _digest(entry.get("config_hash"), name=f"registered config hash for {relative}")


def inspect_overlay(
    overlay_path: Path,
    *,
    repo_root: Path,
    authorization_schema: str,
    authorization_sha256: str,
    beta_source_sha256: str | None,
    horizon_parent_sha256: str | None,
) -> dict[str, object]:
    repository = _canonical_directory(repo_root, name="repository root")
    overlay = _canonical_file(
        overlay_path,
        name="post-recovery overlay",
        maximum_bytes=_MAX_YAML_BYTES,
    )
    try:
        relative_path = overlay.relative_to(repository)
    except ValueError as error:
        raise ValueError("post-recovery overlay is outside the repository") from error
    if relative_path.parent != Path("configs") or relative_path.suffix != ".yaml":
        raise ValueError("post-recovery overlay must be a direct configs/*.yaml file")
    relative = relative_path.as_posix()

    values, seeds = _selected_yaml(overlay)
    if _required(values, ("schema_version",)) != POST_RECOVERY_CONFIG_SCHEMA:
        raise ValueError("overlay is not the post-recovery schema")
    if _required(values, ("design", "stage")) != "pilot":
        raise ValueError("overlay is not a post-recovery pilot")
    pilot_phase = _required(values, ("design", "pilot_phase"))
    if pilot_phase not in {"calibration", "freeze"}:
        raise ValueError("post-recovery pilot phase is invalid")
    if tuple(seeds) != ORDERED_SEEDS:
        raise ValueError("post-recovery overlay does not use the fixed three seeds")

    source_config_value = _required(values, ("design", "source_config"))
    if not isinstance(source_config_value, str):
        raise ValueError("overlay design.source_config must be a string")
    source_relative = PurePosixPath(source_config_value)
    if (
        source_relative.is_absolute()
        or source_relative.parent != PurePosixPath("configs")
        or source_relative.suffix != ".yaml"
    ):
        raise ValueError("overlay source config must be a direct configs/*.yaml file")
    base_relative = source_relative.as_posix()
    base = _canonical_file(
        repository / Path(*source_relative.parts),
        name="post-recovery base config",
        maximum_bytes=_MAX_YAML_BYTES,
    )

    identities, _ = _strict_json(
        repository / "configs" / "identities.json",
        name="config identity registry",
        maximum_bytes=16 * 1024 * 1024,
    )
    if identities.get("schema_version") != "prorm-config-identities/v1":
        raise ValueError("config identity registry schema is invalid")
    design_sha256 = _registered_identity(
        identities,
        relative=relative,
        path=overlay,
        seed_count=3,
    )
    base_hash = _registered_identity(
        identities,
        relative=base_relative,
        path=base,
        seed_count=3,
    )
    if _required(values, ("design", "source_config_hash")) != base_hash:
        raise ValueError("overlay source_config_hash differs from the registered base")

    expected_auth_sha256 = _digest(
        authorization_sha256,
        name="expected authorization SHA256",
    )
    reference_map = {
        R2_AUTHORIZATION_SCHEMA: (R2_REFERENCE_SCHEMA, R2_PROJECTION_SCHEMA),
        R3_AUTHORIZATION_SCHEMA: (R3_REFERENCE_SCHEMA, R3_PROJECTION_SCHEMA),
    }
    if authorization_schema not in reference_map:
        raise ValueError("authorization schema is neither exact R2 nor exact R3")
    reference_schema, projection_schema = reference_map[authorization_schema]
    if (
        _required(values, ("recovery_success_reference", "schema_version")) != reference_schema
        or _required(
            values,
            ("recovery_success_reference", "authorization_projection", "schema_version"),
        )
        != projection_schema
        or _required(
            values,
            (
                "recovery_success_reference",
                "authorization_projection",
                "source_schema_version",
            ),
        )
        != authorization_schema
        or _required(values, ("recovery_success_reference", "artifact_sha256"))
        != expected_auth_sha256
        or _required(
            values,
            (
                "recovery_success_reference",
                "authorization_projection",
                "optimizer_schedule_sha256",
            ),
        )
        != ADOPTED_SCHEDULE_SHA256
        or _required(
            values,
            (
                "reward_model",
                "optimizer_protocol",
                "source_recovery_authorization_sha256",
            ),
        )
        != expected_auth_sha256
        or _required(
            values,
            (
                "reward_model",
                "optimizer_protocol",
                "learning_rate_schedule",
                "schedule_sha256",
            ),
        )
        != ADOPTED_SCHEDULE_SHA256
    ):
        raise ValueError("overlay lost its authorization/optimizer transport binding")

    name = _required(values, ("design", "name"))
    horizon_index = _required(values, ("evaluation", "max_length", "horizon_grid_index"))
    if isinstance(horizon_index, bool) or not isinstance(horizon_index, int) or horizon_index < 0:
        raise ValueError("overlay horizon grid index is invalid")
    beta_binding = _required(
        values,
        ("objective", "common_beta", "beta_source_aggregate_sha256"),
    )
    horizon_binding = _required(
        values,
        ("evaluation", "max_length", "parent_pilot_aggregate_sha256"),
    )
    supplied_beta = (
        None
        if beta_source_sha256 is None
        else _digest(
            beta_source_sha256,
            name="beta-source aggregate SHA256",
        )
    )
    supplied_horizon = (
        None
        if horizon_parent_sha256 is None
        else _digest(
            horizon_parent_sha256,
            name="horizon-parent aggregate SHA256",
        )
    )
    if beta_binding != supplied_beta or horizon_binding != supplied_horizon:
        raise ValueError("overlay predecessor hashes differ from submitted predecessor files")

    if pilot_phase == "calibration":
        if supplied_beta is not None:
            raise ValueError("calibration must not bind a beta-source aggregate")
        if horizon_index == 0:
            expected_name = "common-beta-post-recovery-calibration-v1"
            expected_relative = "configs/common_beta_post_recovery_calibration.yaml"
            aggregate_name = "phase2-post-recovery-calibration-aggregate.json"
            if supplied_horizon is not None:
                raise ValueError("initial calibration must not bind a horizon parent")
        else:
            expected_name = f"common-beta-post-recovery-calibration-horizon-{horizon_index}-v1"
            expected_relative = (
                f"configs/common_beta_post_recovery_calibration_horizon_{horizon_index}.yaml"
            )
            aggregate_name = (
                f"phase2-post-recovery-calibration-horizon-{horizon_index}-aggregate.json"
            )
            if supplied_horizon is None:
                raise ValueError("new-horizon calibration lacks its parent aggregate")
    else:
        if supplied_beta is None or supplied_horizon is None:
            raise ValueError("freeze must bind beta-source and horizon-parent aggregates")
        if name == "common-beta-post-recovery-freeze-v1":
            beta_index = 0
            expected_relative = "configs/common_beta_post_recovery_freeze.yaml"
            aggregate_name = "phase2-post-recovery-freeze-aggregate.json"
        else:
            match = re.fullmatch(
                r"common-beta-post-recovery-freeze-retry-([1-9][0-9]*)-v1",
                str(name),
            )
            if match is None:
                raise ValueError("freeze design name is invalid")
            beta_index = int(match.group(1))
            expected_relative = f"configs/common_beta_post_recovery_freeze_retry_{beta_index}.yaml"
            aggregate_name = f"phase2-post-recovery-freeze-retry-{beta_index}-aggregate.json"
        expected_name = str(name)
    if name != expected_name or relative != expected_relative:
        raise ValueError("overlay path differs from its semantic post-recovery identity")

    return {
        "pilot_phase": pilot_phase,
        "phase2_design_sha256": design_sha256,
        "base_config_hash": base_hash,
        "base_path": os.fspath(base),
        "base_relative": base_relative,
        "optimizer_schedule_sha256": ADOPTED_SCHEDULE_SHA256,
        "semantic_aggregate_filename": aggregate_name,
    }


def _emit_authorization(value: Mapping[str, object], *, output_format: str) -> None:
    if output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": "prorm-post-recovery-auth-transport-inspection/v1",
                    "status": "passed",
                    **value,
                },
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return
    bindings = _sequence(value["bindings"], name="authorization bindings")
    print(value["schema_version"])
    print(value["optimizer_schedule_sha256"])
    print(len(bindings))
    for raw_binding in bindings:
        binding = _mapping(raw_binding, name="authorization binding")
        digest = binding["sha256"]
        print(
            "\t".join(
                (
                    str(binding["kind"]),
                    str(binding["path"]),
                    "-" if digest is None else str(digest),
                )
            )
        )


def _emit_overlay(value: Mapping[str, object]) -> None:
    for key in (
        "pilot_phase",
        "phase2_design_sha256",
        "base_config_hash",
        "base_path",
        "base_relative",
        "optimizer_schedule_sha256",
        "semantic_aggregate_filename",
    ):
        print(value[key])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorization = subparsers.add_parser("authorization")
    authorization.add_argument("path", type=Path)
    authorization.add_argument("--expected-sha256", required=True)
    authorization.add_argument("--project-root", type=Path, required=True)
    authorization.add_argument("--format", choices=("lines", "json"), default="lines")

    overlay = subparsers.add_parser("overlay")
    overlay.add_argument("path", type=Path)
    overlay.add_argument("--repo-root", type=Path, required=True)
    overlay.add_argument("--authorization-schema", required=True)
    overlay.add_argument("--authorization-sha256", required=True)
    overlay.add_argument("--beta-source-sha256")
    overlay.add_argument("--horizon-parent-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "authorization":
        _emit_authorization(
            inspect_authorization(
                arguments.path,
                expected_sha256=arguments.expected_sha256,
                project_root=arguments.project_root,
            ),
            output_format=arguments.format,
        )
    else:
        _emit_overlay(
            inspect_overlay(
                arguments.path,
                repo_root=arguments.repo_root,
                authorization_schema=arguments.authorization_schema,
                authorization_sha256=arguments.authorization_sha256,
                beta_source_sha256=arguments.beta_source_sha256,
                horizon_parent_sha256=arguments.horizon_parent_sha256,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
