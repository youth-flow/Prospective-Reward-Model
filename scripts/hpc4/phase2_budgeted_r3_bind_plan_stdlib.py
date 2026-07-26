#!/usr/bin/env python3
"""Build the minimal read-only SIF bind allowlist for GateE R3 verification.

This module is intentionally Python-stdlib-only.  It does not authorize
training.  It authenticates the byte-linked path envelope needed by the real
R3 verifier and emits only:

* the combined Gate-R/Gate-C authorization,
* the exact Gate-R authorization and Gate-C aggregate,
* the Gate-P operational bundle,
* every retained scheduler-segment runtime closure, and
* every closed three-file terminal-evidence directory.

No recovery head, checkpoint, optimizer-state, or task-output directory is
admissible as a bind target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

_FINAL_SCHEMA: Final = "phase2-recovery-r3-gate-c-success-authorization/v1"
_FINAL_ROLE: Final = "head_free_exact_three_by_three_gate_c_success_capability"
_GATE_R_SCHEMA: Final = "phase2-recovery-r3-success-authorization/v1"
_GATE_R_ROLE: Final = "three_seed_all_scheduler_segments_audited_gate_r_capability"
_GATE_C_SCHEMA: Final = "phase2-recovery-r3-gate-c-aggregate/v1"
_GATE_C_ROLE: Final = "exact_three_families_by_three_seeds_train_only_gate_c_closure"
_SCHEDULE_SHA256: Final = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
_PROJECT_ROOT: Final = Path("/project/sigroup/smart-reward-model")
_FINAL_RELATIVE: Final = Path("runs/phase2-recovery-r3-controls/gate-c-success-authorization.json")
_GATE_R_RELATIVE: Final = Path("runs/phase2-recovery-r3/recovery-success-authorization.json")
_GATE_C_RELATIVE: Final = Path("runs/phase2-recovery-r3-controls/gate-c-aggregate.json")
_TERMINAL_FILES: Final = frozenset({"raw-sacct.psv", "parsed-sacct.json", "terminal-manifest.json"})
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_FIELDS: Final = frozenset(
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
        "gate_c_source_set_sha256",
        "authorization_sha256",
    }
)
_FINAL_TRANSPORT: Final = {
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

_GATE_R_FIELDS: Final = frozenset(
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
_BUNDLE_FIELDS: Final = frozenset(
    {
        "path",
        "file_sha256",
        "bundle_semantic_sha256",
        "profile_run_sha256",
        "formal_profile_sha256",
        "resource_plan_sha256",
    }
)
_SOURCE_FIELDS: Final = frozenset(
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
_SEGMENT_FIELDS: Final = frozenset(
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
_MANIFEST_FIELDS: Final = frozenset(
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
_GATE_R_REUSE: Final = {
    "beta": False,
    "reward_model_parameters": False,
    "policy": False,
}
_GATE_R_TRANSPORT: Final = {
    "trained_parameter_payload_included": False,
    "checkpoint_payload_included": False,
    "optimizer_state_payload_included": False,
    "training_data_payload_included": False,
    "label_payload_included": False,
}


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _closed_mapping(
    value: object,
    *,
    name: str,
    fields: frozenset[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has an invalid closed field set")
    return dict(value)


def _semantic_sha256(
    value: Mapping[str, object],
    *,
    newline: bool,
) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if newline:
        encoded += "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_self_hash(
    value: Mapping[str, object],
    *,
    field: str,
    name: str,
    newline: bool,
) -> None:
    unsigned = dict(value)
    declared = _digest(unsigned.pop(field, None), name=f"{name} semantic SHA-256")
    if declared != _semantic_sha256(unsigned, newline=newline):
        raise ValueError(f"{name} semantic self-hash is invalid")


def _canonical_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("project root must be an absolute non-symlink directory")
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("project root is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or resolved != root:
        raise ValueError("project root must be canonical")
    return root


def _canonical_file(path: Path, *, root: Path, name: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink file")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        relative = path.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{name} is unavailable outside the project root") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or resolved != path
        or not relative.parts
        or root not in resolved.parents
    ):
        raise ValueError(f"{name} must be a canonical retained project file")
    return path


def _safe_relative(value: object, *, name: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    path = Path(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts or not path.parts:
        raise ValueError(f"{name} must be a safe project-relative path")
    return path


def _retained_file(
    value: object,
    *,
    root: Path,
    name: str,
    expected_sha256: object,
) -> Path:
    relative = _safe_relative(value, name=name)
    path = _canonical_file(root.joinpath(*relative.parts), root=root, name=name)
    expected = _digest(expected_sha256, name=f"{name} SHA-256")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"{name} byte hash mismatch")
    return path


def _r3_relative(path: Path, *, root: Path, name: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escaped the project root") from error
    if relative.parts[:2] != ("runs", "phase2-recovery-r3"):
        raise ValueError(f"{name} is outside the canonical R3 evidence namespace")
    forbidden = {
        "head",
        "heads",
        "checkpoint",
        "checkpoints",
        "optimizer",
        "optimizers",
        "model",
        "models",
        "task-output",
        "task-outputs",
    }
    if any(component.lower() in forbidden for component in relative.parts):
        raise ValueError(f"{name} crosses the no-trained-output namespace boundary")
    return relative


def _validate_bundle_namespace(path: Path, *, root: Path) -> None:
    relative = _r3_relative(path, root=root, name="Gate-P operational bundle")
    if (
        path.name != "gatep-operational-bundle.json"
        or "gatep" not in relative.parts
        or re.fullmatch(r"gatep-attempt-(?!000)[0-9]{3}", path.parent.name) is None
    ):
        raise ValueError("Gate-P operational bundle path is outside its canonical namespace")


def _validate_closure_namespace(
    path: Path,
    *,
    root: Path,
    task_id: int,
) -> None:
    _r3_relative(path, root=root, name="R3 runtime closure")
    if path.parent.name != "runtime-closures" or path.name != f"task-{task_id}.json":
        raise ValueError("runtime closure path is outside its canonical task namespace")


def _validate_terminal_namespace(
    path: Path,
    *,
    root: Path,
    task_id: int,
    segment_index: int,
) -> None:
    relative = _r3_relative(path, root=root, name="R3 terminal evidence")
    if (
        "terminal-evidence" not in relative.parts
        or path.name != f"task-{task_id}-segment-{segment_index}"
    ):
        raise ValueError("terminal evidence path is outside its canonical segment namespace")


def _decode_strict_json(
    raw: bytes,
    *,
    name: str,
    canonical: bool,
) -> dict[str, object]:
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{name} must be nonempty newline-terminated JSON")

    def reject_duplicates(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    if canonical:
        expected = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if raw != expected:
            raise ValueError(f"{name} must use canonical deterministic JSON bytes")
    return value


def _strict_json_file(
    path: Path,
    *,
    root: Path,
    name: str,
    expected_sha256: str,
) -> dict[str, object]:
    source = _canonical_file(path, root=root, name=name)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _digest(expected_sha256, name=f"{name} SHA-256"):
        raise ValueError(f"{name} byte hash mismatch")
    return _decode_strict_json(raw, name=name, canonical=True)


def _terminal_directory(
    value: object,
    *,
    root: Path,
    segment: Mapping[str, object],
    name: str,
    task_id: int,
    segment_index: int,
) -> Path:
    relative = _safe_relative(value, name=name)
    directory = root.joinpath(*relative.parts)
    if directory.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    try:
        metadata = directory.lstat()
        resolved = directory.resolve(strict=True)
        entries = list(os.scandir(directory))
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or resolved != directory or root not in resolved.parents:
        raise ValueError(f"{name} must be a canonical retained project directory")
    _validate_terminal_namespace(
        directory,
        root=root,
        task_id=task_id,
        segment_index=segment_index,
    )
    observed_names = {entry.name for entry in entries}
    if observed_names != _TERMINAL_FILES or len(entries) != len(_TERMINAL_FILES):
        raise ValueError(
            f"{name} must contain exactly {sorted(_TERMINAL_FILES)!r} and no other entries"
        )
    files: dict[str, Path] = {}
    for entry in entries:
        path = directory / entry.name
        info = path.lstat()
        if (
            entry.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or path.resolve(strict=True) != path
        ):
            raise ValueError(f"{name}/{entry.name} must be a canonical non-symlink file")
        files[entry.name] = path

    raw_sha256 = _digest(
        segment["terminal_raw_sacct_sha256"],
        name=f"{name} raw sacct SHA-256",
    )
    if hashlib.sha256(files["raw-sacct.psv"].read_bytes()).hexdigest() != raw_sha256:
        raise ValueError(f"{name} raw sacct byte hash mismatch")
    manifest_sha256 = _digest(
        segment["terminal_manifest_file_sha256"],
        name=f"{name} manifest SHA-256",
    )
    manifest = _strict_json_file(
        files["terminal-manifest.json"],
        root=root,
        name=f"{name} manifest",
        expected_sha256=manifest_sha256,
    )
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError(f"{name} manifest has an invalid closed field set")
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
        or parsed_binding["filename"] != "parsed-sacct.json"
    ):
        raise ValueError(f"{name} manifest filenames or raw binding are invalid")
    parsed_sha256 = _digest(
        parsed_binding["file_sha256"],
        name=f"{name} parsed evidence SHA-256",
    )
    if hashlib.sha256(files["parsed-sacct.json"].read_bytes()).hexdigest() != parsed_sha256:
        raise ValueError(f"{name} parsed evidence byte hash mismatch")
    return directory


def build_bind_plan(
    authorization: str | os.PathLike[str],
    *,
    expected_sha256: str,
    project_root: str | os.PathLike[str] = _PROJECT_ROOT,
) -> dict[str, object]:
    """Return an exact, hash-checked minimal bind plan."""

    root = _canonical_root(project_root)
    authorization_path = Path(authorization)
    if not authorization_path.is_absolute():
        authorization_path = authorization_path.absolute()
    expected_authorization = root / _FINAL_RELATIVE
    if authorization_path != expected_authorization:
        raise ValueError("combined R3 authorization is not at its exact production path")
    combined = _strict_json_file(
        authorization_path,
        root=root,
        name="combined R3 authorization",
        expected_sha256=expected_sha256,
    )
    if set(combined) != _FINAL_FIELDS:
        raise ValueError("combined R3 authorization has an invalid closed field set")
    _validate_self_hash(
        combined,
        field="authorization_sha256",
        name="combined R3 authorization",
        newline=False,
    )
    transport = combined.get("transport_boundary")
    if (
        combined.get("schema_version") != _FINAL_SCHEMA
        or combined.get("role") != _FINAL_ROLE
        or combined.get("optimizer_schedule_sha256") != _SCHEDULE_SHA256
        or combined.get("optimizer_schedule_is_unique") is not True
        or combined.get("execution_revision") != 3
        or combined.get("ordered_seeds") != [20260801, 20260802, 20260803]
        or combined.get("gate_r_authorization_path") != _GATE_R_RELATIVE.as_posix()
        or combined.get("gate_c_aggregate_path") != _GATE_C_RELATIVE.as_posix()
        or combined.get("gate_r_passed") is not True
        or combined.get("gate_c_passed") is not True
        or combined.get("fresh_calibration_authorized") is not True
        or combined.get("formal_efficacy_claim_authorized") is not False
        or combined.get("recovery_or_control_outputs_reusable") is not False
        or combined.get("validation_or_heldout_access_authorized") is not False
        or combined.get("policy_or_final_utility_access_authorized") is not False
        or transport != _FINAL_TRANSPORT
    ):
        raise ValueError("combined R3 authorization source-path envelope is invalid")

    gate_r_sha256 = _digest(
        combined.get("gate_r_authorization_file_sha256"),
        name="Gate-R authorization file SHA-256",
    )
    gate_c_sha256 = _digest(
        combined.get("gate_c_aggregate_file_sha256"),
        name="Gate-C aggregate file SHA-256",
    )
    gate_r_path = root / _GATE_R_RELATIVE
    gate_c_path = root / _GATE_C_RELATIVE
    gate_r = _strict_json_file(
        gate_r_path,
        root=root,
        name="Gate-R authorization",
        expected_sha256=gate_r_sha256,
    )
    gate_c = _strict_json_file(
        gate_c_path,
        root=root,
        name="Gate-C aggregate",
        expected_sha256=gate_c_sha256,
    )
    if set(gate_r) != _GATE_R_FIELDS or gate_r.get("schema_version") != _GATE_R_SCHEMA:
        raise ValueError("Gate-R authorization has an invalid closed identity")
    _validate_self_hash(
        gate_r,
        field="authorization_sha256",
        name="Gate-R authorization",
        newline=True,
    )
    gate_r_reuse = gate_r.get("recovery_output_reuse")
    gate_r_transport = gate_r.get("transport_boundary")
    if (
        gate_r.get("role") != _GATE_R_ROLE
        or gate_r.get("optimizer_schedule_sha256") != _SCHEDULE_SHA256
        or gate_r.get("execution_revision") != 3
        or gate_r.get("ordered_seeds") != [20260801, 20260802, 20260803]
        or gate_r.get("recovery_status") != "all_three_seeds_all_scheduler_segments_success"
        or gate_r.get("gate_r_passed") is not True
        or gate_r.get("fresh_calibration_authorized") is not False
        or gate_r.get("authorized_information") != "optimizer_schedule_only"
        or gate_r.get("authorized_next_action") != "await_separate_gate_c_authorization"
        or gate_r.get("recovery_outputs_reusable") is not False
        or gate_r.get("validation_or_heldout_access_authorized") is not False
        or gate_r.get("policy_or_final_utility_access_authorized") is not False
        or gate_r.get("formal_efficacy_claim_authorized") is not False
        or gate_r_reuse != _GATE_R_REUSE
        or gate_r_transport != _GATE_R_TRANSPORT
        or combined.get("gate_r_authorization_sha256") != gate_r.get("authorization_sha256")
    ):
        raise ValueError("Gate-R authorization crosses its schedule-only head-free boundary")
    _validate_self_hash(
        gate_c,
        field="aggregate_sha256",
        name="Gate-C aggregate",
        newline=False,
    )
    if (
        gate_c.get("schema_version") != _GATE_C_SCHEMA
        or gate_c.get("role") != _GATE_C_ROLE
        or gate_c.get("optimizer_schedule_sha256") != _SCHEDULE_SHA256
        or gate_c.get("ordered_seeds") != [20260801, 20260802, 20260803]
        or gate_c.get("matrix_shape") != [3, 3]
        or gate_c.get("gate_c_passed") is not True
        or gate_c.get("fresh_calibration_authorized") is not False
        or gate_c.get("result_reusable_for_training") is not False
        or combined.get("gate_c_aggregate_sha256") != gate_c.get("aggregate_sha256")
    ):
        raise ValueError("Gate-C aggregate crosses its head-free 3x3 boundary")

    bundle = _closed_mapping(
        gate_r["operational_bundle"],
        name="Gate-R operational bundle",
        fields=_BUNDLE_FIELDS,
    )
    bundle_path = _retained_file(
        bundle["path"],
        root=root,
        name="Gate-P operational bundle",
        expected_sha256=bundle["file_sha256"],
    )
    _validate_bundle_namespace(bundle_path, root=root)

    dependencies: list[dict[str, str]] = [
        {
            "kind": "file",
            "path": os.fspath(authorization_path),
            "sha256": _digest(expected_sha256, name="combined authorization SHA-256"),
        },
        {"kind": "file", "path": os.fspath(gate_r_path), "sha256": gate_r_sha256},
        {"kind": "file", "path": os.fspath(gate_c_path), "sha256": gate_c_sha256},
        {
            "kind": "file",
            "path": os.fspath(bundle_path),
            "sha256": _digest(bundle["file_sha256"], name="Gate-P bundle SHA-256"),
        },
    ]
    sources = gate_r["sources"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("Gate-R authorization must contain exactly three sources")
    expected_task_seed = ((0, 20260801), (1, 20260802), (2, 20260803))
    for source_index, (raw_source, expected) in enumerate(
        zip(sources, expected_task_seed, strict=True)
    ):
        source = _closed_mapping(
            raw_source,
            name=f"Gate-R source {source_index}",
            fields=_SOURCE_FIELDS,
        )
        if (source["task_id"], source["seed"]) != expected:
            raise ValueError("Gate-R source task/seed order is invalid")
        segments = source["segments"]
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"Gate-R source {source_index} has no scheduler segments")
        if source["final_segment_index"] != len(segments):
            raise ValueError(f"Gate-R source {source_index} final segment index is invalid")
        for expected_segment, raw_segment in enumerate(segments, start=1):
            segment = _closed_mapping(
                raw_segment,
                name=f"Gate-R source {source_index} segment {expected_segment}",
                fields=_SEGMENT_FIELDS,
            )
            if segment["segment_index"] != expected_segment:
                raise ValueError("Gate-R scheduler segment order is invalid")
            expected_kind = "completed" if expected_segment == len(segments) else "continuable"
            expected_schema = (
                "phase2-recovery-r3-external-primary-segment-completed-terminal/v1"
                if expected_kind == "completed"
                else "phase2-recovery-r3-external-primary-segment-terminal/v1"
            )
            expected_role = (
                "external_scheduler_terminal_completed_zero_exit_compute_complete"
                if expected_kind == "completed"
                else "external_scheduler_terminal_completed_zero_exit_continuation_required"
            )
            if (
                segment["terminal_kind"] != expected_kind
                or segment["terminal_schema_version"] != expected_schema
                or segment["terminal_role"] != expected_role
            ):
                raise ValueError("Gate-R scheduler terminal-kind sequence is invalid")
            closure = _retained_file(
                segment["runtime_closure_path"],
                root=root,
                name=f"source {source_index} segment {expected_segment} runtime closure",
                expected_sha256=segment["runtime_closure_file_sha256"],
            )
            _validate_closure_namespace(
                closure,
                root=root,
                task_id=source_index,
            )
            terminal = _terminal_directory(
                segment["terminal_evidence_directory"],
                root=root,
                segment=segment,
                name=f"source {source_index} segment {expected_segment} terminal evidence",
                task_id=source_index,
                segment_index=expected_segment,
            )
            dependencies.extend(
                (
                    {
                        "kind": "file",
                        "path": os.fspath(closure),
                        "sha256": _digest(
                            segment["runtime_closure_file_sha256"],
                            name="runtime closure SHA-256",
                        ),
                    },
                    {"kind": "directory", "path": os.fspath(terminal)},
                )
            )

    paths = [item["path"] for item in dependencies]
    if len(paths) != len(set(paths)):
        raise ValueError("R3 verification bind plan contains duplicate path roles")
    return {
        "schema_version": "phase2-budgeted-r3-minimal-bind-plan/v1",
        "project_root": os.fspath(root),
        "authorization_sha256": expected_sha256,
        "dependencies": dependencies,
        "bind_paths": paths,
        "trained_outputs_bound": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("authorization", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    parser.add_argument(
        "--emit",
        choices=("json", "paths"),
        default="json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    plan = build_bind_plan(
        arguments.authorization,
        expected_sha256=arguments.expected_sha256,
        project_root=arguments.project_root,
    )
    if arguments.emit == "paths":
        for path in plan["bind_paths"]:
            print(path)
    else:
        print(
            json.dumps(
                plan,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
