"""Final three-seed Gate-R authorization for Phase-2 recovery revision 3.

The authorization in this module is deliberately narrower than a recovery
result.  It proves that every scheduler segment for all three frozen seeds was
revalidated from canonical runtime closures and raw Slurm terminal evidence.
It exposes only the frozen optimizer-schedule identity to a future combined
Gate-R + Gate-C decision; model parameters, checkpoints, optimizer state,
labels, and data are never embedded. Gate R alone does not authorize fresh
calibration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, TypeAlias

from .phase2_r3_artifacts import (
    CanonicalJsonArtifact,
    canonical_json_bytes,
    publish_canonical_artifact,
    read_canonical_artifact,
)
from .phase2_r3_config import R3_RECOVERY_SCHEDULE_SHA256
from .phase2_r3_identity import (
    CONTINUABLE_PRIMARY_TERMINAL_ROLE,
    CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
    R2_RECOVERY_DESIGN_SHA256,
    R3_CONTINUATION_EVIDENCE_SCHEMA,
    R3_EXECUTION_REVISION,
    R3_ORDERED_SEEDS,
    R3_PRIMARY_HEADS,
    R3_TASK_SEED_MAP,
)
from .phase2_r3_profile_artifacts import (
    VerifiedGatePOperationalBundle,
    reopen_verified_gate_p_operational_bundle,
)
from .phase2_r3_terminal import (
    COMPLETED_PRIMARY_TERMINAL_ROLE,
    COMPLETED_PRIMARY_TERMINAL_SCHEMA,
    CompletedPrimaryTerminalCapability,
    ContinuablePrimaryTerminalCapability,
    reopen_primary_segment_runtime_closure,
    revalidate_completed_primary_terminal,
    revalidate_continuable_primary_terminal,
)

R3_SUCCESS_AUTHORIZATION_SCHEMA: Final = "phase2-recovery-r3-success-authorization/v1"
R3_SUCCESS_AUTHORIZATION_ROLE: Final = "three_seed_all_scheduler_segments_audited_gate_r_capability"
R3_SUCCESS_AUTHORIZATION_RELATIVE: Final = Path(
    "runs/phase2-recovery-r3/recovery-success-authorization.json"
)
PRODUCTION_PROJECT_ROOT: Final = Path("/project/sigroup/smart-reward-model")
_EXPECTED_PRODUCTION_PROJECT_ROOT: Final = Path("/project/sigroup/smart-reward-model")

R3_GATE_R_NEXT_ACTION: Final = "await_separate_gate_c_authorization"
R3_RECOVERY_STATUS: Final = "all_three_seeds_all_scheduler_segments_success"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_POSITIVE_JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")

_AUTHORIZATION_FIELDS: Final = frozenset(
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
_OPERATIONAL_BUNDLE_FIELDS: Final = frozenset(
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
_RECOVERY_OUTPUT_REUSE: Final = {
    "beta": False,
    "reward_model_parameters": False,
    "policy": False,
}
_TRANSPORT_BOUNDARY: Final = {
    "trained_parameter_payload_included": False,
    "checkpoint_payload_included": False,
    "optimizer_state_payload_included": False,
    "training_data_payload_included": False,
    "label_payload_included": False,
}

PrimaryTerminalCapability: TypeAlias = (
    ContinuablePrimaryTerminalCapability | CompletedPrimaryTerminalCapability
)


def _identity_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _artifact_semantic_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_job_id(value: object, *, name: str) -> str:
    if type(value) is not str or _POSITIVE_JOB_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a positive decimal Slurm job ID")
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


def _project_root(project_root: str | os.PathLike[str] | None) -> Path:
    if project_root is None:
        if PRODUCTION_PROJECT_ROOT != _EXPECTED_PRODUCTION_PROJECT_ROOT:
            raise RuntimeError("R3 authorization production root is not the frozen HPC4 path")
        root = PRODUCTION_PROJECT_ROOT
    else:
        root = Path(project_root)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("R3 authorization project root must be an absolute real directory")
    try:
        resolved = root.resolve(strict=True)
        info = root.stat()
    except OSError as error:
        raise ValueError("R3 authorization project root is unavailable") from error
    if resolved != root or not stat.S_ISDIR(info.st_mode):
        raise ValueError("R3 authorization project root must be canonical")
    return root


def _relative_existing_path(
    value: Path,
    *,
    root: Path,
    name: str,
    directory: bool,
) -> str:
    if not value.is_absolute() or value.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink path")
    try:
        resolved = value.resolve(strict=True)
        info = value.stat()
        relative = value.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{name} must be retained under the project root") from error
    expected_kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if resolved != value or not expected_kind or not relative.parts:
        kind = "directory" if directory else "file"
        raise ValueError(f"{name} must be a canonical retained {kind}")
    return relative.as_posix()


def _safe_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    path = Path(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts or not path.parts:
        raise ValueError(f"{name} must be a safe POSIX project-relative path")
    return value


def _resolve_retained_path(
    value: object,
    *,
    root: Path,
    name: str,
    directory: bool,
) -> Path:
    relative = _safe_relative_path(value, name=name)
    path = root.joinpath(*Path(relative).parts)
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
        info = path.stat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    expected_kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if resolved != path or root not in resolved.parents or not expected_kind:
        raise ValueError(f"{name} escapes its retained canonical object")
    return path


def _bundle_identity(bundle: VerifiedGatePOperationalBundle) -> dict[str, str]:
    bundle.validate_integrity()
    return {
        "file_sha256": bundle.file_sha256,
        "bundle_semantic_sha256": bundle.bundle_semantic_sha256,
        "profile_run_sha256": bundle.profile_run_sha256,
        "formal_profile_sha256": bundle.formal_profile_sha256,
        "resource_plan_sha256": bundle.resource_plan_sha256,
    }


def _expected_scheduler_segment_id(*, logical_run_id: str, segment_index: int) -> str:
    return _identity_sha256(
        {
            "namespace": "phase2-recovery-r3-scheduler-segment/v1",
            "logical_run_id": logical_run_id,
            "segment_index": segment_index,
        }
    )


def _expected_continuation_evidence_sha256(
    predecessor: PrimaryTerminalCapability,
) -> str:
    if type(predecessor) is not ContinuablePrimaryTerminalCapability:
        raise TypeError("only a continuable terminal can authorize another segment")
    predecessor.validate_integrity()
    closure = predecessor.runtime_closure
    admission = closure.admission_payload
    outcome = closure.outcome_payload
    checkpoint = outcome["continuation_checkpoint"]
    if not isinstance(checkpoint, Mapping):
        raise ValueError("continuable predecessor has no verified checkpoint reference")
    return _identity_sha256(
        {
            "schema_version": R3_CONTINUATION_EVIDENCE_SCHEMA,
            "design_sha256": admission["design_sha256"],
            "predecessor_admission_sha256": admission["admission_sha256"],
            "logical_run_id": admission["logical_run_id"],
            "task_id": admission["task_id"],
            "seed": admission["seed"],
            "predecessor_segment_index": admission["segment_index"],
            "materialization_attestation_sha256": (admission["materialization_attestation_sha256"]),
            "scheduler_terminal": {
                "schema_version": CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
                "artifact_sha256": predecessor.terminal_sha256,
                "role": CONTINUABLE_PRIMARY_TERMINAL_ROLE,
            },
            "verified_checkpoint": dict(checkpoint),
        }
    )


def _completion_receipts(
    capability: CompletedPrimaryTerminalCapability,
) -> list[str]:
    payload = capability.to_dict()
    receipts = payload["completed_head_receipts"]
    admission = capability.runtime_closure.admission_payload
    runs = admission["head_runs"]
    if (
        not isinstance(receipts, list)
        or len(receipts) != len(R3_PRIMARY_HEADS)
        or not isinstance(runs, list)
        or len(runs) != len(R3_PRIMARY_HEADS)
    ):
        raise ValueError("completed terminal does not bind both primary completion receipts")
    result: list[str] = []
    for index, (learner, raw_receipt, raw_run) in enumerate(
        zip(R3_PRIMARY_HEADS, receipts, runs, strict=True)
    ):
        receipt = _closed_mapping(
            raw_receipt,
            name=f"completion receipt {index}",
            fields=frozenset({"learner", "head_run_id", "completion_receipt_sha256"}),
        )
        run = _closed_mapping(
            raw_run,
            name=f"primary run {index}",
            fields=frozenset({"head", "head_run_id"}),
        )
        if (
            receipt["learner"] != learner
            or run["head"] != learner
            or receipt["head_run_id"] != run["head_run_id"]
        ):
            raise ValueError("completed terminal receipt order or run binding is invalid")
        result.append(
            _digest(
                receipt["completion_receipt_sha256"],
                name=f"completion receipt {index} SHA-256",
            )
        )
    return result


def _history_source(
    history: Sequence[PrimaryTerminalCapability],
    *,
    task_id: int,
    seed: int,
    root: Path,
) -> tuple[dict[str, object], VerifiedGatePOperationalBundle]:
    if isinstance(history, (str, bytes, bytearray)) or not isinstance(history, Sequence):
        raise TypeError(f"task {task_id} terminal history must be a sequence")
    if not history:
        raise ValueError(f"task {task_id} terminal history must not be empty")
    if any(
        type(capability)
        not in {ContinuablePrimaryTerminalCapability, CompletedPrimaryTerminalCapability}
        for capability in history
    ):
        raise TypeError(f"task {task_id} history contains an unvalidated terminal capability")
    if any(
        type(capability) is not ContinuablePrimaryTerminalCapability for capability in history[:-1]
    ):
        raise ValueError(f"task {task_id} only its final terminal may be completed")
    if type(history[-1]) is not CompletedPrimaryTerminalCapability:
        raise ValueError(f"task {task_id} history is not terminally completed")

    first = history[0]
    first.validate_integrity()
    bundle = first.operational_bundle
    bundle_identity = _bundle_identity(bundle)
    if len(history) > bundle.max_scheduler_segments:
        raise ValueError(f"task {task_id} history exceeds the Gate-P segment limit")

    segments: list[dict[str, object]] = []
    design_sha256: str | None = None
    logical_run_id: str | None = None
    materialization_sha256: str | None = None
    primary_runs: object = None
    previous: PrimaryTerminalCapability | None = None
    for expected_segment_index, capability in enumerate(history, start=1):
        capability.validate_integrity()
        if _bundle_identity(capability.operational_bundle) != bundle_identity:
            raise ValueError(f"task {task_id} terminal history crosses operational bundles")
        closure = capability.runtime_closure
        closure.validate_integrity()
        admission = closure.admission_payload
        runtime = closure.runtime_payload
        outcome = closure.outcome_payload
        if (
            admission["task_id"] != task_id
            or admission["seed"] != seed
            or runtime["task_id"] != task_id
            or runtime["seed"] != seed
            or runtime["array_task_id"] != task_id
            or outcome["task_id"] != task_id
            or outcome["seed"] != seed
            or admission["segment_index"] != expected_segment_index
            or runtime["segment_index"] != expected_segment_index
            or outcome["segment_index"] != expected_segment_index
        ):
            raise ValueError(f"task {task_id} history violates the frozen task/seed/segment map")
        current_design = _digest(admission["design_sha256"], name="R3 recovery design SHA-256")
        current_logical_run = _digest(admission["logical_run_id"], name="logical_run_id")
        current_materialization = _digest(
            admission["materialization_attestation_sha256"],
            name="materialization attestation SHA-256",
        )
        current_runs = admission["head_runs"]
        if expected_segment_index == 1:
            design_sha256 = current_design
            logical_run_id = current_logical_run
            materialization_sha256 = current_materialization
            primary_runs = current_runs
        elif (
            current_design != design_sha256
            or current_logical_run != logical_run_id
            or current_materialization != materialization_sha256
            or current_runs != primary_runs
            or previous is None
            or admission["continuation_evidence_sha256"]
            != _expected_continuation_evidence_sha256(previous)
        ):
            raise ValueError(f"task {task_id} continuation chain is not identity-complete")
        expected_segment_id = _expected_scheduler_segment_id(
            logical_run_id=current_logical_run,
            segment_index=expected_segment_index,
        )
        if (
            admission["scheduler_segment_id"] != expected_segment_id
            or runtime["scheduler_segment_id"] != expected_segment_id
            or outcome["scheduler_segment_id"] != expected_segment_id
        ):
            raise ValueError(f"task {task_id} scheduler segment identity is invalid")

        terminal_kind = (
            "continuable"
            if type(capability) is ContinuablePrimaryTerminalCapability
            else "completed"
        )
        terminal_schema = (
            CONTINUABLE_PRIMARY_TERMINAL_SCHEMA
            if terminal_kind == "continuable"
            else COMPLETED_PRIMARY_TERMINAL_SCHEMA
        )
        terminal_role = (
            CONTINUABLE_PRIMARY_TERMINAL_ROLE
            if terminal_kind == "continuable"
            else COMPLETED_PRIMARY_TERMINAL_ROLE
        )
        segments.append(
            {
                "segment_index": expected_segment_index,
                "scheduler_segment_id": expected_segment_id,
                "scheduler_array_job_id": _positive_job_id(
                    runtime["array_job_id"],
                    name="scheduler array job ID",
                ),
                "scheduler_job_id": _positive_job_id(
                    runtime["job_id"],
                    name="scheduler allocation job ID",
                ),
                "scheduler_job_selector": closure.job_selector,
                "runtime_closure_path": _relative_existing_path(
                    closure.artifact_path,
                    root=root,
                    name=f"task {task_id} segment {expected_segment_index} runtime closure",
                    directory=False,
                ),
                "runtime_closure_file_sha256": closure.file_sha256,
                "runtime_closure_sha256": closure.closure_sha256,
                "segment_outcome_sha256": _digest(
                    outcome["outcome_sha256"],
                    name="segment outcome SHA-256",
                ),
                "terminal_kind": terminal_kind,
                "terminal_schema_version": terminal_schema,
                "terminal_role": terminal_role,
                "terminal_evidence_directory": _relative_existing_path(
                    capability.evidence_directory,
                    root=root,
                    name=f"task {task_id} segment {expected_segment_index} terminal evidence",
                    directory=True,
                ),
                "terminal_manifest_file_sha256": capability.manifest_file_sha256,
                "terminal_raw_sacct_sha256": capability.inspection.raw_sacct_sha256,
                "terminal_sha256": capability.terminal_sha256,
            }
        )
        previous = capability

    completed = history[-1]
    if type(completed) is not CompletedPrimaryTerminalCapability:
        raise AssertionError("validated history lost its completed terminal")
    if design_sha256 is None or logical_run_id is None or materialization_sha256 is None:
        raise AssertionError("validated history did not establish its identity")
    return (
        {
            "task_id": task_id,
            "seed": seed,
            "design_sha256": design_sha256,
            "logical_run_id": logical_run_id,
            "materialization_attestation_sha256": materialization_sha256,
            "final_segment_index": len(history),
            "completion_receipt_sha256s": _completion_receipts(completed),
            "segments": segments,
        },
        bundle,
    )


def build_r3_success_authorization(
    terminal_histories: Sequence[Sequence[PrimaryTerminalCapability]],
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Build the sole head-free authorization from three complete histories."""

    root = _project_root(project_root)
    if (
        isinstance(terminal_histories, (str, bytes, bytearray))
        or not isinstance(terminal_histories, Sequence)
        or len(terminal_histories) != len(R3_TASK_SEED_MAP)
    ):
        raise ValueError("R3 success authorization requires exactly three ordered histories")

    sources: list[dict[str, object]] = []
    bundles: list[VerifiedGatePOperationalBundle] = []
    for history, (task_id, seed) in zip(terminal_histories, R3_TASK_SEED_MAP, strict=True):
        source, bundle = _history_source(
            history,
            task_id=task_id,
            seed=seed,
            root=root,
        )
        sources.append(source)
        bundles.append(bundle)

    bundle_identity = _bundle_identity(bundles[0])
    if any(_bundle_identity(bundle) != bundle_identity for bundle in bundles[1:]):
        raise ValueError("R3 seed histories do not share one Gate-P operational bundle")
    design_sha256 = str(sources[0]["design_sha256"])
    if design_sha256 == R2_RECOVERY_DESIGN_SHA256 or any(
        source["design_sha256"] != design_sha256 for source in sources[1:]
    ):
        raise ValueError("R3 seed histories do not share one non-R2 recovery design")
    logical_runs = [source["logical_run_id"] for source in sources]
    if len(set(logical_runs)) != len(logical_runs):
        raise ValueError("R3 seed histories must use distinct logical run identities")
    segment_refs = [
        (
            segment["runtime_closure_path"],
            segment["terminal_evidence_directory"],
            segment["terminal_sha256"],
        )
        for source in sources
        for segment in source["segments"]  # type: ignore[union-attr]
    ]
    if len(set(segment_refs)) != len(segment_refs):
        raise ValueError("R3 seed histories reuse a scheduler-segment evidence identity")

    operational_bundle = {
        "path": _relative_existing_path(
            bundles[0].artifact_path,
            root=root,
            name="Gate-P operational bundle",
            directory=False,
        ),
        **bundle_identity,
    }
    body: dict[str, object] = {
        "schema_version": R3_SUCCESS_AUTHORIZATION_SCHEMA,
        "role": R3_SUCCESS_AUTHORIZATION_ROLE,
        "recovery_design_sha256": design_sha256,
        "optimizer_schedule_sha256": R3_RECOVERY_SCHEDULE_SHA256,
        "execution_revision": R3_EXECUTION_REVISION,
        "ordered_seeds": list(R3_ORDERED_SEEDS),
        "recovery_status": R3_RECOVERY_STATUS,
        "gate_r_passed": True,
        "fresh_calibration_authorized": False,
        "authorized_information": "optimizer_schedule_only",
        "authorized_next_action": R3_GATE_R_NEXT_ACTION,
        "recovery_outputs_reusable": False,
        "validation_or_heldout_access_authorized": False,
        "policy_or_final_utility_access_authorized": False,
        "formal_efficacy_claim_authorized": False,
        "recovery_output_reuse": dict(_RECOVERY_OUTPUT_REUSE),
        "transport_boundary": dict(_TRANSPORT_BOUNDARY),
        "operational_bundle": operational_bundle,
        "terminal_set_sha256": _artifact_semantic_sha256({"sources": sources}),
        "sources": sources,
    }
    return {
        **body,
        "authorization_sha256": _artifact_semantic_sha256(body),
    }


def _validate_authorization_structure(value: object) -> dict[str, object]:
    payload = _closed_mapping(
        value,
        name="R3 success authorization",
        fields=_AUTHORIZATION_FIELDS,
    )
    unsigned = dict(payload)
    authorization_sha256 = _digest(
        unsigned.pop("authorization_sha256"),
        name="R3 success authorization semantic SHA-256",
    )
    if authorization_sha256 != _artifact_semantic_sha256(unsigned):
        raise ValueError("R3 success authorization self-hash is invalid")
    design_sha256 = _digest(
        payload["recovery_design_sha256"],
        name="R3 recovery design SHA-256",
    )
    recovery_output_reuse = _closed_mapping(
        payload["recovery_output_reuse"],
        name="R3 recovery-output reuse boundary",
        fields=frozenset(_RECOVERY_OUTPUT_REUSE),
    )
    transport_boundary = _closed_mapping(
        payload["transport_boundary"],
        name="R3 authorization transport boundary",
        fields=frozenset(_TRANSPORT_BOUNDARY),
    )
    if (
        payload["schema_version"] != R3_SUCCESS_AUTHORIZATION_SCHEMA
        or payload["role"] != R3_SUCCESS_AUTHORIZATION_ROLE
        or design_sha256 == R2_RECOVERY_DESIGN_SHA256
        or payload["optimizer_schedule_sha256"] != R3_RECOVERY_SCHEDULE_SHA256
        or type(payload["execution_revision"]) is not int
        or payload["execution_revision"] != R3_EXECUTION_REVISION
        or payload["ordered_seeds"] != list(R3_ORDERED_SEEDS)
        or payload["recovery_status"] != R3_RECOVERY_STATUS
        or payload["gate_r_passed"] is not True
        or payload["fresh_calibration_authorized"] is not False
        or payload["authorized_information"] != "optimizer_schedule_only"
        or payload["authorized_next_action"] != R3_GATE_R_NEXT_ACTION
        or payload["recovery_outputs_reusable"] is not False
        or payload["validation_or_heldout_access_authorized"] is not False
        or payload["policy_or_final_utility_access_authorized"] is not False
        or payload["formal_efficacy_claim_authorized"] is not False
        or any(value is not False for value in recovery_output_reuse.values())
        or any(value is not False for value in transport_boundary.values())
    ):
        raise ValueError("R3 success authorization exceeds its Gate-R-only boundary")

    bundle = _closed_mapping(
        payload["operational_bundle"],
        name="R3 authorization operational bundle",
        fields=_OPERATIONAL_BUNDLE_FIELDS,
    )
    _safe_relative_path(bundle["path"], name="operational bundle path")
    for name in _OPERATIONAL_BUNDLE_FIELDS - {"path"}:
        _digest(bundle[name], name=f"operational bundle {name}")

    sources = payload["sources"]
    if not isinstance(sources, list) or len(sources) != len(R3_TASK_SEED_MAP):
        raise ValueError("R3 success authorization must contain exactly three sources")
    for source_index, (raw_source, (task_id, seed)) in enumerate(
        zip(sources, R3_TASK_SEED_MAP, strict=True)
    ):
        source = _closed_mapping(
            raw_source,
            name=f"R3 authorization source {source_index}",
            fields=_SOURCE_FIELDS,
        )
        if (
            type(source["task_id"]) is not int
            or type(source["seed"]) is not int
            or source["task_id"] != task_id
            or source["seed"] != seed
            or source["design_sha256"] != design_sha256
        ):
            raise ValueError("R3 authorization source violates its ordered task/seed design")
        for name in (
            "design_sha256",
            "logical_run_id",
            "materialization_attestation_sha256",
        ):
            _digest(source[name], name=f"source {source_index} {name}")
        receipts = source["completion_receipt_sha256s"]
        if not isinstance(receipts, list) or len(receipts) != len(R3_PRIMARY_HEADS):
            raise ValueError("R3 authorization source must bind both completion receipts")
        for receipt in receipts:
            _digest(receipt, name="completion receipt SHA-256")
        segments = source["segments"]
        if not isinstance(segments, list) or not segments:
            raise ValueError("R3 authorization source has no scheduler segments")
        if type(source["final_segment_index"]) is not int or source["final_segment_index"] != len(
            segments
        ):
            raise ValueError("R3 authorization final segment index is inconsistent")
        for segment_index, raw_segment in enumerate(segments, start=1):
            segment = _closed_mapping(
                raw_segment,
                name=f"R3 source {source_index} segment {segment_index}",
                fields=_SEGMENT_FIELDS,
            )
            expected_kind = "completed" if segment_index == len(segments) else "continuable"
            expected_schema = (
                COMPLETED_PRIMARY_TERMINAL_SCHEMA
                if expected_kind == "completed"
                else CONTINUABLE_PRIMARY_TERMINAL_SCHEMA
            )
            expected_role = (
                COMPLETED_PRIMARY_TERMINAL_ROLE
                if expected_kind == "completed"
                else CONTINUABLE_PRIMARY_TERMINAL_ROLE
            )
            if (
                type(segment["segment_index"]) is not int
                or segment["segment_index"] != segment_index
                or segment["terminal_kind"] != expected_kind
                or segment["terminal_schema_version"] != expected_schema
                or segment["terminal_role"] != expected_role
            ):
                raise ValueError("R3 authorization scheduler segment sequence is invalid")
            for name in (
                "scheduler_segment_id",
                "runtime_closure_file_sha256",
                "runtime_closure_sha256",
                "segment_outcome_sha256",
                "terminal_manifest_file_sha256",
                "terminal_raw_sacct_sha256",
                "terminal_sha256",
            ):
                _digest(segment[name], name=f"segment {segment_index} {name}")
            array_job_id = _positive_job_id(
                segment["scheduler_array_job_id"],
                name="scheduler array job ID",
            )
            _positive_job_id(segment["scheduler_job_id"], name="scheduler allocation job ID")
            if segment["scheduler_job_selector"] != f"{array_job_id}_{task_id}":
                raise ValueError("R3 authorization scheduler selector is invalid")
            _safe_relative_path(
                segment["runtime_closure_path"],
                name="runtime closure path",
            )
            _safe_relative_path(
                segment["terminal_evidence_directory"],
                name="terminal evidence directory",
            )
    terminal_set_sha256 = _digest(
        payload["terminal_set_sha256"],
        name="R3 authorization terminal-set SHA-256",
    )
    if terminal_set_sha256 != _artifact_semantic_sha256({"sources": sources}):
        raise ValueError("R3 authorization terminal-set hash is invalid")
    return payload


def _reopen_histories(
    payload: Mapping[str, object],
    *,
    root: Path,
) -> tuple[tuple[PrimaryTerminalCapability, ...], ...]:
    bundle_record = _closed_mapping(
        payload["operational_bundle"],
        name="R3 authorization operational bundle",
        fields=_OPERATIONAL_BUNDLE_FIELDS,
    )
    bundle_path = _resolve_retained_path(
        bundle_record["path"],
        root=root,
        name="Gate-P operational bundle",
        directory=False,
    )
    bundle = reopen_verified_gate_p_operational_bundle(
        bundle_path,
        expected_file_sha256=str(bundle_record["file_sha256"]),
    )
    if {
        "path": str(bundle_record["path"]),
        **_bundle_identity(bundle),
    } != bundle_record:
        raise ValueError("R3 authorization operational bundle changed after publication")

    histories: list[tuple[PrimaryTerminalCapability, ...]] = []
    sources = payload["sources"]
    if not isinstance(sources, list):
        raise TypeError("validated R3 sources lost their list type")
    for source_index, raw_source in enumerate(sources):
        source = _closed_mapping(
            raw_source,
            name=f"R3 authorization source {source_index}",
            fields=_SOURCE_FIELDS,
        )
        segments = source["segments"]
        if not isinstance(segments, list):
            raise TypeError("validated R3 segments lost their list type")
        history: list[PrimaryTerminalCapability] = []
        for raw_segment in segments:
            segment = _closed_mapping(
                raw_segment,
                name=f"R3 authorization source {source_index} segment",
                fields=_SEGMENT_FIELDS,
            )
            closure_path = _resolve_retained_path(
                segment["runtime_closure_path"],
                root=root,
                name="R3 runtime closure",
                directory=False,
            )
            evidence_directory = _resolve_retained_path(
                segment["terminal_evidence_directory"],
                root=root,
                name="R3 terminal evidence directory",
                directory=True,
            )
            closure = reopen_primary_segment_runtime_closure(
                closure_path,
                expected_file_sha256=str(segment["runtime_closure_file_sha256"]),
                operational_bundle=bundle,
            )
            common = {
                "runtime_closure": closure,
                "evidence_directory": evidence_directory,
                "expected_manifest_file_sha256": str(segment["terminal_manifest_file_sha256"]),
                "expected_raw_sacct_sha256": str(segment["terminal_raw_sacct_sha256"]),
            }
            capability: PrimaryTerminalCapability
            if segment["terminal_kind"] == "continuable":
                capability = revalidate_continuable_primary_terminal(bundle, **common)
            else:
                capability = revalidate_completed_primary_terminal(bundle, **common)
            history.append(capability)
        histories.append(tuple(history))
    return tuple(histories)


def publish_r3_success_authorization(
    terminal_histories: Sequence[Sequence[PrimaryTerminalCapability]],
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> CanonicalJsonArtifact:
    """Publish the exact production artifact without replacing any prior file."""

    root = _project_root(project_root)
    output = root / R3_SUCCESS_AUTHORIZATION_RELATIVE
    if output.parent.resolve(strict=True) != output.parent:
        raise ValueError("R3 authorization output parent is not canonical")
    payload = build_r3_success_authorization(
        terminal_histories,
        project_root=root,
    )
    return publish_canonical_artifact(output, payload)


def verify_r3_success_authorization(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Strictly reopen the authorization and every referenced segment."""

    root = _project_root(project_root)
    expected_path = root / R3_SUCCESS_AUTHORIZATION_RELATIVE
    source = Path(path)
    if not source.is_absolute():
        source = source.absolute()
    if source != expected_path:
        raise ValueError("R3 success authorization is not at its exact production path")
    transport = read_canonical_artifact(
        source,
        expected_file_sha256=_digest(
            expected_sha256,
            name="expected R3 authorization file SHA-256",
        ),
    )
    payload = _validate_authorization_structure(transport.payload)
    histories = _reopen_histories(payload, root=root)
    rebuilt = build_r3_success_authorization(histories, project_root=root)
    if rebuilt != payload:
        raise ValueError("R3 success authorization differs from fresh terminal revalidation")
    transport.validate_integrity()
    return payload


__all__ = [
    "PRODUCTION_PROJECT_ROOT",
    "R3_GATE_R_NEXT_ACTION",
    "R3_RECOVERY_STATUS",
    "R3_SUCCESS_AUTHORIZATION_RELATIVE",
    "R3_SUCCESS_AUTHORIZATION_ROLE",
    "R3_SUCCESS_AUTHORIZATION_SCHEMA",
    "PrimaryTerminalCapability",
    "build_r3_success_authorization",
    "publish_r3_success_authorization",
    "verify_r3_success_authorization",
]
