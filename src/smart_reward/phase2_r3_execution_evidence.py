"""Retained, head-free execution evidence for the fixed R3 three-seed wave.

The training process owns rich process-local capabilities which intentionally
cannot be deserialized.  This module does not recreate those capabilities.
Instead it publishes closed canonical receipts while the capabilities are
live, and later reopens the exact design/materialization/admission identities
plus a byte manifest of every immutable task artifact.

The receipts contain no tensor, checkpoint, label, or model payload.  A
segment receipt points at the task root and hashes the payload bytes there;
Gate R must re-read those bytes before it can accept the receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from .phase2_r3_artifacts import (
    CanonicalJsonArtifact,
    canonical_json_bytes,
    publish_canonical_artifact,
    read_canonical_artifact,
)
from .phase2_r3_identity import (
    R3_ORDERED_SEEDS,
    R3_PRIMARY_HEADS,
    R3_TASK_SEED_MAP,
    PrimarySegmentAdmission,
    R3PrimaryDesign,
)
from .phase2_r3_inputs import R3TrainMaterializationCapability
from .phase2_r3_materialization import (
    TRAIN_MATERIALIZATION_PROVENANCE_SCHEMA,
    TRAIN_SPLIT_NAME,
    TRAIN_TENSOR_KEYS,
    VALIDATED_R3_MATERIALIZATION_SCHEMA,
)
from .phase2_r3_primary import SlurmSegmentRuntime
from .phase2_r3_terminal import PrimarySegmentRuntimeClosure

PRIMARY_IDENTITY_RECEIPT_SCHEMA: Final = "phase2-recovery-r3-primary-identity-receipt/v1"
PRIMARY_IDENTITY_RECEIPT_ROLE: Final = (
    "retained_exact_design_materialization_and_segment1_admission"
)
SEGMENT_EVIDENCE_RECEIPT_SCHEMA: Final = "phase2-recovery-r3-segment-evidence-receipt/v1"
SEGMENT_EVIDENCE_RECEIPT_ROLE: Final = "all_immutable_task_bytes_rehashed_after_segment_closure"
EXECUTION_EVIDENCE_RELATIVE: Final = Path("runs/phase2-recovery-r3/primary-execution-evidence")

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MUTABLE_FILENAMES = frozenset(
    {
        "bt_mle.progress.latest.json",
        "prorm_plus.progress.latest.json",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "task_id",
        "seed",
        "base_primary_submission_plan",
        "design",
        "materialization",
        "materialization_capability",
        "segment_1_admission",
        "information_boundary",
        "identity_receipt_sha256",
    }
)
_PLAN_REF_FIELDS = frozenset({"path", "file_sha256", "submission_plan_sha256"})
_SEGMENT_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "task_id",
        "seed",
        "segment_index",
        "identity_receipt",
        "runtime_closure",
        "task_root",
        "immutable_file_manifest",
        "immutable_file_manifest_sha256",
        "completed_head_receipts",
        "information_boundary",
        "segment_evidence_receipt_sha256",
    }
)
_ARTIFACT_REF_FIELDS = frozenset({"path", "file_sha256", "semantic_sha256"})
_MANIFEST_ENTRY_FIELDS = frozenset({"relative_path", "size_bytes", "file_sha256"})
_COMPLETION_REF_FIELDS = frozenset(
    {
        "learner",
        "head_run_id",
        "receipt_relative_path",
        "receipt_file_sha256",
        "receipt_sha256",
        "result_relative_path",
        "result_file_sha256",
        "terminal_checkpoint_artifact_sha256",
    }
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _semantic_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identity_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256(raw)


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _closed(
    value: object,
    *,
    name: str,
    fields: frozenset[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has an invalid closed field set")
    return dict(value)


def _canonical_root(value: str | os.PathLike[str], *, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
        info = path.stat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if resolved != path or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{name} must be a canonical directory")
    return path


def _canonical_file(value: str | os.PathLike[str], *, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        info = path.stat()
    except OSError as error:
        raise ValueError(f"{name} is unavailable") from error
    if resolved != path or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a canonical regular file")
    return path


def _relative_to_root(path: Path, *, project_root: Path, name: str) -> str:
    path = _canonical_file(path, name=name)
    try:
        relative = path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"{name} must be retained under the project root") from error
    if not relative.parts:
        raise ValueError(f"{name} cannot be the project root")
    return relative.as_posix()


def _resolve_project_file(value: object, *, project_root: Path, name: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty project-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or value != relative.as_posix():
        raise ValueError(f"{name} is not a safe project-relative path")
    path = project_root.joinpath(*relative.parts)
    path = _canonical_file(path, name=name)
    if project_root not in path.parents:
        raise ValueError(f"{name} escapes the project root")
    return path


def _reopen_base_plan_binding(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_semantic_sha256: str,
) -> dict[str, object]:
    artifact = read_canonical_artifact(
        path,
        expected_file_sha256=_digest(
            expected_file_sha256,
            name="base primary submission plan file SHA-256",
        ),
    )
    plan = artifact.payload
    required = {
        "schema_version",
        "segment_index",
        "array_task_ids",
        "science_config_file_sha256",
        "parent_registry_file_sha256",
        "submission_plan_sha256",
    }
    if (
        not required.issubset(plan)
        or plan["schema_version"] != "phase2-recovery-r3-primary-submission-plan/v2"
        or plan["segment_index"] != 1
        or plan["array_task_ids"] != [0, 1, 2]
    ):
        raise ValueError("base primary submission plan is not the fixed R3 v2 plan")
    unsigned = dict(plan)
    semantic = unsigned.pop("submission_plan_sha256")
    if semantic != _digest(
        expected_semantic_sha256,
        name="base primary submission plan semantic SHA-256",
    ) or semantic != _semantic_sha256(unsigned):
        raise ValueError("base primary submission plan semantic hash is invalid")
    _digest(plan["science_config_file_sha256"], name="science config file SHA-256")
    _digest(plan["parent_registry_file_sha256"], name="parent registry file SHA-256")
    return plan


def identity_receipt_path(project_root: Path, *, task_id: int) -> Path:
    if type(task_id) is not int or task_id not in dict(R3_TASK_SEED_MAP):
        raise ValueError("identity receipt task_id must be one of 0, 1, 2")
    return project_root / EXECUTION_EVIDENCE_RELATIVE / f"task-{task_id}" / "primary-identity.json"


def segment_evidence_receipt_path(
    project_root: Path,
    *,
    task_id: int,
    segment_index: int,
) -> Path:
    if type(task_id) is not int or task_id not in dict(R3_TASK_SEED_MAP):
        raise ValueError("segment evidence task_id must be one of 0, 1, 2")
    if type(segment_index) is not int or segment_index < 1:
        raise ValueError("segment evidence segment_index must be positive")
    return (
        project_root
        / EXECUTION_EVIDENCE_RELATIVE
        / f"task-{task_id}"
        / f"segment-{segment_index:04d}.json"
    )


def _validate_materialization_payload(
    value: object,
    *,
    design: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    fields = frozenset(
        {
            "schema_version",
            "context_sha256",
            "settings_sha256",
            "science_semantic_sha256",
            "science_file_sha256",
            "seed",
            "provenance",
            "provenance_sha256",
            "input_training_sha256",
            "prepared_training_sha256",
            "oracle_reward_sha256",
            "label_stream_sha256",
            "heldout_bytes_decoded",
            "attestation_sha256",
        }
    )
    materialization = _closed(value, name="materialization", fields=fields)
    provenance_fields = frozenset(
        {
            "schema_version",
            "seed",
            "parent_artifact_registry_sha256",
            "artifact_metadata_sha256",
            "artifact_tensors_sha256",
            "artifact_candidates_sha256",
            "artifact_materialization_sha256",
            "artifact_verification_sha256",
            "source_run_manifest_sha256",
            "source_producer_identity_sha256",
            "split_name",
            "ordered_train_prompt_ids_sha256",
            "train_tensor_sha256",
            "candidate_train_prefix_sha256",
            "candidate_train_prefix_count",
            "input_training_sha256",
            "prepared_training_sha256",
            "oracle_reward_sha256",
            "label_stream_sha256",
            "heldout_bytes_decoded",
            "provenance_sha256",
        }
    )
    provenance = _closed(
        materialization["provenance"],
        name="materialization provenance",
        fields=provenance_fields,
    )
    tensor_hashes = provenance["train_tensor_sha256"]
    if not isinstance(tensor_hashes, Mapping) or set(tensor_hashes) != set(TRAIN_TENSOR_KEYS):
        raise ValueError("materialization provenance train tensor hashes are invalid")
    for name, digest in tensor_hashes.items():
        _digest(digest, name=f"train tensor {name} SHA-256")
    if (
        materialization["schema_version"] != VALIDATED_R3_MATERIALIZATION_SCHEMA
        or materialization["seed"] != seed
        or materialization["heldout_bytes_decoded"] is not False
        or provenance["schema_version"] != TRAIN_MATERIALIZATION_PROVENANCE_SCHEMA
        or provenance["seed"] != seed
        or provenance["split_name"] != TRAIN_SPLIT_NAME
        or provenance["heldout_bytes_decoded"] is not False
        or type(provenance["candidate_train_prefix_count"]) is not int
        or provenance["candidate_train_prefix_count"] < 1
    ):
        raise ValueError("materialization violates the frozen train-only boundary")
    science = design.get("science_semantic_sha256")
    science_file = design.get("science_file_sha256")
    if (
        materialization["science_semantic_sha256"] != science
        or materialization["science_file_sha256"] != science_file
    ):
        raise ValueError("materialization belongs to another science design")
    for name in provenance_fields - {
        "schema_version",
        "seed",
        "split_name",
        "train_tensor_sha256",
        "candidate_train_prefix_count",
        "heldout_bytes_decoded",
    }:
        _digest(provenance[name], name=f"materialization provenance {name}")
    provenance_unsigned = dict(provenance)
    provenance_sha = provenance_unsigned.pop("provenance_sha256")
    if provenance_sha != _identity_sha256(provenance_unsigned):
        raise ValueError("materialization provenance self-hash is invalid")
    exact_links = {
        "provenance_sha256": provenance_sha,
        "input_training_sha256": provenance["input_training_sha256"],
        "prepared_training_sha256": provenance["prepared_training_sha256"],
        "oracle_reward_sha256": provenance["oracle_reward_sha256"],
        "label_stream_sha256": provenance["label_stream_sha256"],
    }
    if any(materialization[name] != expected for name, expected in exact_links.items()):
        raise ValueError("materialization differs from its retained provenance")
    for name in fields - {
        "schema_version",
        "seed",
        "provenance",
        "heldout_bytes_decoded",
    }:
        _digest(materialization[name], name=f"materialization {name}")
    attestation_unsigned = {
        name: materialization[name]
        for name in (
            "schema_version",
            "context_sha256",
            "settings_sha256",
            "science_semantic_sha256",
            "science_file_sha256",
            "seed",
            "provenance_sha256",
            "input_training_sha256",
            "prepared_training_sha256",
            "oracle_reward_sha256",
            "label_stream_sha256",
            "heldout_bytes_decoded",
        )
    }
    if materialization["attestation_sha256"] != _identity_sha256(attestation_unsigned):
        raise ValueError("materialization attestation self-hash is invalid")
    materialization["provenance"] = provenance
    return materialization


def _validate_capability_payload(
    value: object,
    *,
    materialization: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    fields = frozenset(
        {
            "schema_version",
            "materialization_attestation_sha256",
            "seed",
            "source_config_hash",
            "parent_registry_file_sha256",
            "parent_seed_entry_sha256",
            "artifact_metadata_sha256",
            "artifact_tensors_sha256",
            "artifact_candidates_sha256",
            "candidate_train_prefix_sha256",
            "artifact_materialization_sha256",
            "artifact_verification_sha256",
            "source_run_manifest_sha256",
            "source_producer_identity_sha256",
            "oracle_chat_template_sha256",
            "oracle_transform_sha256",
            "oracle_reward_sha256",
            "byte_sources_reverified",
            "train_tensor_keys_decoded",
            "heldout_tensor_values_decoded",
            "candidate_suffix_decoded",
            "policy_session_opened",
            "capability_sha256",
        }
    )
    capability = _closed(value, name="materialization capability", fields=fields)
    if (
        capability["seed"] != seed
        or capability["materialization_attestation_sha256"] != materialization["attestation_sha256"]
        or capability["byte_sources_reverified"] is not True
        or capability["train_tensor_keys_decoded"] != list(TRAIN_TENSOR_KEYS)
        or capability["heldout_tensor_values_decoded"] is not False
        or capability["candidate_suffix_decoded"] is not False
        or capability["policy_session_opened"] is not False
    ):
        raise ValueError("materialization capability violates its train-only authority")
    provenance = materialization["provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("validated materialization provenance lost its mapping type")
    links = {
        "parent_registry_file_sha256": provenance["parent_artifact_registry_sha256"],
        "artifact_metadata_sha256": provenance["artifact_metadata_sha256"],
        "artifact_tensors_sha256": provenance["artifact_tensors_sha256"],
        "artifact_candidates_sha256": provenance["artifact_candidates_sha256"],
        "candidate_train_prefix_sha256": provenance["candidate_train_prefix_sha256"],
        "artifact_materialization_sha256": provenance["artifact_materialization_sha256"],
        "artifact_verification_sha256": provenance["artifact_verification_sha256"],
        "source_run_manifest_sha256": provenance["source_run_manifest_sha256"],
        "source_producer_identity_sha256": provenance["source_producer_identity_sha256"],
        "oracle_reward_sha256": provenance["oracle_reward_sha256"],
    }
    if any(capability[name] != expected for name, expected in links.items()):
        raise ValueError("materialization capability differs from retained provenance")
    for name in fields - {
        "schema_version",
        "seed",
        "byte_sources_reverified",
        "train_tensor_keys_decoded",
        "heldout_tensor_values_decoded",
        "candidate_suffix_decoded",
        "policy_session_opened",
    }:
        _digest(capability[name], name=f"materialization capability {name}")
    unsigned = dict(capability)
    capability_sha = unsigned.pop("capability_sha256")
    if capability_sha != _identity_sha256(unsigned):
        raise ValueError("materialization capability self-hash is invalid")
    return capability


def _validate_admission_payload(
    value: object,
    *,
    design_sha256: str,
    materialization_sha256: str,
    task_id: int,
    seed: int,
) -> dict[str, object]:
    fields = frozenset(
        {
            "schema_version",
            "design_sha256",
            "materialization_attestation_sha256",
            "task_id",
            "seed",
            "segment_index",
            "logical_run_id",
            "head_runs",
            "scheduler_segment_id",
            "start_mode",
            "continuation_evidence_sha256",
            "admission_sha256",
        }
    )
    admission = _closed(value, name="segment-1 admission", fields=fields)
    expected_logical = _identity_sha256(
        {
            "namespace": "phase2-recovery-r3-logical-run/v1",
            "design_sha256": design_sha256,
            "task_id": task_id,
            "seed": seed,
            "materialization_attestation_sha256": materialization_sha256,
        }
    )
    expected_runs = [
        {
            "head": head,
            "head_run_id": _identity_sha256(
                {
                    "namespace": "phase2-recovery-r3-head-run/v1",
                    "logical_run_id": expected_logical,
                    "head": head,
                }
            ),
        }
        for head in R3_PRIMARY_HEADS
    ]
    expected_segment = _identity_sha256(
        {
            "namespace": "phase2-recovery-r3-scheduler-segment/v1",
            "logical_run_id": expected_logical,
            "segment_index": 1,
        }
    )
    if (
        admission["design_sha256"] != design_sha256
        or admission["materialization_attestation_sha256"] != materialization_sha256
        or admission["task_id"] != task_id
        or admission["seed"] != seed
        or admission["segment_index"] != 1
        or admission["logical_run_id"] != expected_logical
        or admission["head_runs"] != expected_runs
        or admission["scheduler_segment_id"] != expected_segment
        or admission["start_mode"] != "fresh_zero_head_fresh_adamw"
        or admission["continuation_evidence_sha256"] is not None
    ):
        raise ValueError("segment-1 admission is not the exact derived R3 identity")
    unsigned = dict(admission)
    admission_sha = unsigned.pop("admission_sha256")
    _digest(admission_sha, name="segment-1 admission SHA-256")
    if admission_sha != _identity_sha256(unsigned):
        raise ValueError("segment-1 admission self-hash is invalid")
    return admission


def publish_primary_identity_receipt(
    project_root: str | os.PathLike[str],
    *,
    base_primary_submission_plan_path: str | os.PathLike[str],
    base_primary_submission_plan_file_sha256: str,
    base_primary_submission_plan_sha256: str,
    design: R3PrimaryDesign,
    materialization_capability: R3TrainMaterializationCapability,
    admission: PrimarySegmentAdmission,
) -> CanonicalJsonArtifact:
    """Publish the sole retained identity receipt for one fixed seed."""

    root = _canonical_root(project_root, name="project root")
    design.validate_integrity()
    materialization_capability.validate_integrity()
    admission.validate_integrity()
    if admission.segment_index != 1:
        raise ValueError("primary identity receipt can only be published from segment 1")
    task_id = admission.task_id
    seed = admission.seed
    if dict(R3_TASK_SEED_MAP).get(task_id) != seed:
        raise ValueError("primary identity receipt violates the frozen task/seed map")
    plan_path = _canonical_file(
        base_primary_submission_plan_path,
        name="base primary submission plan",
    )
    plan_ref = {
        "path": _relative_to_root(
            plan_path,
            project_root=root,
            name="base primary submission plan",
        ),
        "file_sha256": _digest(
            base_primary_submission_plan_file_sha256,
            name="base primary submission plan file SHA-256",
        ),
        "submission_plan_sha256": _digest(
            base_primary_submission_plan_sha256,
            name="base primary submission plan semantic SHA-256",
        ),
    }
    plan = _reopen_base_plan_binding(
        plan_path,
        expected_file_sha256=str(plan_ref["file_sha256"]),
        expected_semantic_sha256=str(plan_ref["submission_plan_sha256"]),
    )
    if (
        plan["science_config_file_sha256"] != design.science.file_sha256
        or plan["parent_registry_file_sha256"]
        != materialization_capability.parent_registry_file_sha256
    ):
        raise ValueError("base primary plan differs from live science/materialization inputs")
    body: dict[str, object] = {
        "schema_version": PRIMARY_IDENTITY_RECEIPT_SCHEMA,
        "role": PRIMARY_IDENTITY_RECEIPT_ROLE,
        "task_id": task_id,
        "seed": seed,
        "base_primary_submission_plan": plan_ref,
        "design": design.to_dict(),
        "materialization": materialization_capability.materialization.to_dict(),
        "materialization_capability": materialization_capability.to_dict(),
        "segment_1_admission": admission.to_dict(),
        "information_boundary": {
            "train_only": True,
            "heldout_bytes_decoded": False,
            "policy_session_opened": False,
            "tensor_payload_included": False,
            "checkpoint_payload_included": False,
        },
    }
    payload = {**body, "identity_receipt_sha256": _semantic_sha256(body)}
    output = identity_receipt_path(root, task_id=task_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    return publish_canonical_artifact(output, payload)


def reopen_primary_identity_receipt(
    project_root: str | os.PathLike[str],
    *,
    task_id: int,
    expected_file_sha256: str,
    expected_design: R3PrimaryDesign | None = None,
    expected_materialization_capability: R3TrainMaterializationCapability | None = None,
) -> dict[str, object]:
    """Reopen and recompute all pure-data identity layers.

    Passing ``expected_design`` upgrades the check from closed-payload
    validation to exact reconstruction against live Gate-0/Gate-1/Gate-P.
    Passing ``expected_materialization_capability`` additionally replays the
    parent-byte materialization validator.
    """

    root = _canonical_root(project_root, name="project root")
    path = identity_receipt_path(root, task_id=task_id)
    artifact = read_canonical_artifact(
        path,
        expected_file_sha256=_digest(
            expected_file_sha256,
            name="identity receipt file SHA-256",
        ),
    )
    payload = _closed(artifact.payload, name="identity receipt", fields=_IDENTITY_FIELDS)
    unsigned = dict(payload)
    receipt_sha = unsigned.pop("identity_receipt_sha256")
    _digest(receipt_sha, name="identity receipt semantic SHA-256")
    if receipt_sha != _semantic_sha256(unsigned):
        raise ValueError("identity receipt self-hash is invalid")
    seed = dict(R3_TASK_SEED_MAP).get(task_id)
    if (
        payload["schema_version"] != PRIMARY_IDENTITY_RECEIPT_SCHEMA
        or payload["role"] != PRIMARY_IDENTITY_RECEIPT_ROLE
        or payload["task_id"] != task_id
        or payload["seed"] != seed
    ):
        raise ValueError("identity receipt violates the frozen task/seed identity")
    boundary = payload["information_boundary"]
    if boundary != {
        "train_only": True,
        "heldout_bytes_decoded": False,
        "policy_session_opened": False,
        "tensor_payload_included": False,
        "checkpoint_payload_included": False,
    }:
        raise ValueError("identity receipt crosses its information boundary")
    plan_ref = _closed(
        payload["base_primary_submission_plan"],
        name="base primary submission plan ref",
        fields=_PLAN_REF_FIELDS,
    )
    plan_path = _resolve_project_file(
        plan_ref["path"],
        project_root=root,
        name="base primary submission plan",
    )
    plan = _reopen_base_plan_binding(
        plan_path,
        expected_file_sha256=str(plan_ref["file_sha256"]),
        expected_semantic_sha256=str(plan_ref["submission_plan_sha256"]),
    )
    design = payload["design"]
    if not isinstance(design, Mapping):
        raise TypeError("identity receipt design must be a mapping")
    design = dict(design)
    design_sha = _digest(design.get("design_sha256"), name="R3 design SHA-256")
    if design.get("ordered_seeds") != list(R3_ORDERED_SEEDS) or design.get("primary_heads") != list(
        R3_PRIMARY_HEADS
    ):
        raise ValueError("identity receipt design is not the fixed three-seed/two-head design")
    if expected_design is not None:
        expected_design.validate_integrity()
        if design != expected_design.to_dict():
            raise ValueError("identity receipt differs from the freshly rebuilt R3 design")
    materialization = _validate_materialization_payload(
        payload["materialization"],
        design=design,
        seed=int(seed),
    )
    capability = _validate_capability_payload(
        payload["materialization_capability"],
        materialization=materialization,
        seed=int(seed),
    )
    if (
        plan["science_config_file_sha256"] != materialization["science_file_sha256"]
        or plan["parent_registry_file_sha256"] != capability["parent_registry_file_sha256"]
    ):
        raise ValueError("retained materialization differs from the base primary inputs")
    if expected_materialization_capability is not None:
        expected_materialization_capability.validate_integrity()
        if (
            materialization != expected_materialization_capability.materialization.to_dict()
            or capability != expected_materialization_capability.to_dict()
        ):
            raise ValueError("identity receipt differs from fresh parent-byte materialization")
    if expected_design is not None and (
        materialization["settings_sha256"] != expected_design.science.settings.sha256
        or capability["source_config_hash"] != expected_design.science.settings.source_config_hash
    ):
        raise ValueError("retained materialization differs from the rebuilt science settings")
    admission = _validate_admission_payload(
        payload["segment_1_admission"],
        design_sha256=design_sha,
        materialization_sha256=str(materialization["attestation_sha256"]),
        task_id=task_id,
        seed=int(seed),
    )
    payload["base_primary_submission_plan"] = plan_ref
    payload["design"] = design
    payload["materialization"] = materialization
    payload["materialization_capability"] = capability
    payload["segment_1_admission"] = admission
    return payload


def _manifest(task_root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(task_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"task evidence contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"task evidence contains a non-regular file: {path}")
        relative = path.relative_to(task_root).as_posix()
        if path.name in _MUTABLE_FILENAMES or path.name.endswith(".latest.json"):
            continue
        file_sha256, size_bytes = _file_sha256(path)
        entries.append(
            {
                "relative_path": relative,
                "size_bytes": size_bytes,
                "file_sha256": file_sha256,
            }
        )
    if not entries:
        raise ValueError("task evidence manifest cannot be empty")
    return entries


def _validate_manifest(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("immutable task manifest must be a non-empty list")
    result: list[dict[str, object]] = []
    previous: str | None = None
    for index, raw_entry in enumerate(value):
        entry = _closed(
            raw_entry,
            name=f"immutable task manifest entry {index}",
            fields=_MANIFEST_ENTRY_FIELDS,
        )
        relative = entry["relative_path"]
        if type(relative) is not str or not relative:
            raise ValueError("manifest relative path must be non-empty")
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or relative != path.as_posix()
            or path.name.endswith(".latest.json")
        ):
            raise ValueError("manifest relative path is unsafe or mutable")
        if previous is not None and relative <= previous:
            raise ValueError("manifest paths must be unique and sorted")
        previous = relative
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 1:
            raise ValueError("manifest file size must be positive")
        _digest(entry["file_sha256"], name="manifest file SHA-256")
        result.append(entry)
    return result


def _completion_refs(
    task_root: Path,
    *,
    closure: PrimarySegmentRuntimeClosure,
    manifest: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    outcome = closure.outcome_payload
    completed = outcome["completed_heads"]
    if not isinstance(completed, list):
        raise TypeError("closure completed-head summary must be a list")
    manifest_by_path = {str(item["relative_path"]): item for item in manifest}
    result: list[dict[str, object]] = []
    for raw_summary in completed:
        if not isinstance(raw_summary, Mapping):
            raise TypeError("closure completion summary must be a mapping")
        summary = dict(raw_summary)
        learner = summary["learner"]
        if learner not in R3_PRIMARY_HEADS:
            raise ValueError("closure completion summary learner is invalid")
        receipt_relative = f"heads/{learner}/head-completion.json"
        result_relative = f"heads/{learner}/internal-head-result.json"
        receipt_entry = manifest_by_path.get(receipt_relative)
        result_entry = manifest_by_path.get(result_relative)
        if receipt_entry is None or result_entry is None:
            raise ValueError("completed head receipt/result is absent from the byte manifest")
        receipt_path = task_root.joinpath(*Path(receipt_relative).parts)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, Mapping):
            raise TypeError("head completion receipt must be a mapping")
        receipt = dict(receipt)
        receipt_unsigned = dict(receipt)
        receipt_sha = receipt_unsigned.pop("receipt_sha256", None)
        if (
            receipt.get("learner") != learner
            or receipt.get("head_run_id") != summary["head_run_id"]
            or receipt_sha != summary["completion_receipt_sha256"]
            or receipt_sha != _identity_sha256(receipt_unsigned)
        ):
            raise ValueError("head completion receipt differs from the segment outcome")
        internal_result = receipt.get("internal_result")
        if (
            not isinstance(internal_result, Mapping)
            or internal_result.get("filename") != "internal-head-result.json"
            or internal_result.get("file_sha256") != result_entry["file_sha256"]
            or internal_result.get("size_bytes") != result_entry["size_bytes"]
        ):
            raise ValueError("head completion receipt differs from its result bytes")
        result.append(
            {
                "learner": learner,
                "head_run_id": summary["head_run_id"],
                "receipt_relative_path": receipt_relative,
                "receipt_file_sha256": receipt_entry["file_sha256"],
                "receipt_sha256": receipt_sha,
                "result_relative_path": result_relative,
                "result_file_sha256": result_entry["file_sha256"],
                "terminal_checkpoint_artifact_sha256": receipt[
                    "terminal_checkpoint_artifact_sha256"
                ],
            }
        )
    return result


def publish_segment_evidence_receipt(
    project_root: str | os.PathLike[str],
    *,
    task_root: str | os.PathLike[str],
    identity_receipt_file_sha256: str,
    closure: PrimarySegmentRuntimeClosure,
    runtime: SlurmSegmentRuntime,
) -> CanonicalJsonArtifact:
    """Hash every immutable task artifact after the runtime closure is sealed."""

    root = _canonical_root(project_root, name="project root")
    task_path = _canonical_root(task_root, name="task root")
    closure.validate_integrity()
    runtime.validate_integrity()
    admission = closure.admission_payload
    if closure.runtime_payload != runtime.to_dict():
        raise ValueError("segment evidence runtime differs from the runtime closure")
    task_id = int(admission["task_id"])
    seed = int(admission["seed"])
    segment_index = int(admission["segment_index"])
    identity_path = identity_receipt_path(root, task_id=task_id)
    identity = reopen_primary_identity_receipt(
        root,
        task_id=task_id,
        expected_file_sha256=identity_receipt_file_sha256,
    )
    if (
        identity["seed"] != seed
        or identity["design"]["design_sha256"] != admission["design_sha256"]  # type: ignore[index]
        or identity["materialization"]["attestation_sha256"]  # type: ignore[index]
        != admission["materialization_attestation_sha256"]
        or identity["segment_1_admission"]["logical_run_id"]  # type: ignore[index]
        != admission["logical_run_id"]
    ):
        raise ValueError("segment closure differs from its retained primary identity")
    manifest = _manifest(task_path)
    # Outcome paths are not serialized in the closure.  Require the canonical
    # task-root outcome filename to be present in the manifest instead.
    expected_outcome = f"segment-outcomes/segment-{segment_index:04d}.json"
    manifest_paths = {str(item["relative_path"]) for item in manifest}
    if expected_outcome not in manifest_paths:
        raise ValueError("segment outcome is absent from the immutable task manifest")
    completions = _completion_refs(task_path, closure=closure, manifest=manifest)
    closure_path = _canonical_file(closure.artifact_path, name="runtime closure")
    body: dict[str, object] = {
        "schema_version": SEGMENT_EVIDENCE_RECEIPT_SCHEMA,
        "role": SEGMENT_EVIDENCE_RECEIPT_ROLE,
        "task_id": task_id,
        "seed": seed,
        "segment_index": segment_index,
        "identity_receipt": {
            "path": _relative_to_root(
                identity_path,
                project_root=root,
                name="identity receipt",
            ),
            "file_sha256": identity_receipt_file_sha256,
            "semantic_sha256": identity["identity_receipt_sha256"],
        },
        "runtime_closure": {
            "path": _relative_to_root(
                closure_path,
                project_root=root,
                name="runtime closure",
            ),
            "file_sha256": closure.file_sha256,
            "semantic_sha256": closure.closure_sha256,
        },
        "task_root": str(task_path),
        "immutable_file_manifest": manifest,
        "immutable_file_manifest_sha256": _semantic_sha256({"files": manifest}),
        "completed_head_receipts": completions,
        "information_boundary": {
            "all_manifest_bytes_rehashed": True,
            "mutable_latest_snapshots_excluded": True,
            "payload_bytes_copied_into_receipt": False,
            "heldout_or_validation_accessed": False,
            "policy_or_final_utility_accessed": False,
        },
    }
    payload = {**body, "segment_evidence_receipt_sha256": _semantic_sha256(body)}
    output = segment_evidence_receipt_path(
        root,
        task_id=task_id,
        segment_index=segment_index,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    return publish_canonical_artifact(output, payload)


def reopen_segment_evidence_receipt(
    project_root: str | os.PathLike[str],
    *,
    task_id: int,
    segment_index: int,
    expected_file_sha256: str,
    runtime_closure: PrimarySegmentRuntimeClosure,
    require_exact_current_manifest: bool,
) -> dict[str, object]:
    """Reopen a segment receipt and rehash all referenced task bytes."""

    root = _canonical_root(project_root, name="project root")
    path = segment_evidence_receipt_path(
        root,
        task_id=task_id,
        segment_index=segment_index,
    )
    artifact = read_canonical_artifact(
        path,
        expected_file_sha256=_digest(
            expected_file_sha256,
            name="segment evidence receipt file SHA-256",
        ),
    )
    payload = _closed(
        artifact.payload,
        name="segment evidence receipt",
        fields=_SEGMENT_FIELDS,
    )
    unsigned = dict(payload)
    semantic = unsigned.pop("segment_evidence_receipt_sha256")
    _digest(semantic, name="segment evidence receipt semantic SHA-256")
    if semantic != _semantic_sha256(unsigned):
        raise ValueError("segment evidence receipt self-hash is invalid")
    seed = dict(R3_TASK_SEED_MAP).get(task_id)
    if (
        payload["schema_version"] != SEGMENT_EVIDENCE_RECEIPT_SCHEMA
        or payload["role"] != SEGMENT_EVIDENCE_RECEIPT_ROLE
        or payload["task_id"] != task_id
        or payload["seed"] != seed
        or payload["segment_index"] != segment_index
        or payload["information_boundary"]
        != {
            "all_manifest_bytes_rehashed": True,
            "mutable_latest_snapshots_excluded": True,
            "payload_bytes_copied_into_receipt": False,
            "heldout_or_validation_accessed": False,
            "policy_or_final_utility_accessed": False,
        }
    ):
        raise ValueError("segment evidence receipt violates its fixed identity/boundary")
    identity_ref = _closed(
        payload["identity_receipt"],
        name="identity receipt ref",
        fields=_ARTIFACT_REF_FIELDS,
    )
    expected_identity_path = identity_receipt_path(root, task_id=task_id)
    if (
        _resolve_project_file(
            identity_ref["path"],
            project_root=root,
            name="identity receipt",
        )
        != expected_identity_path
    ):
        raise ValueError("segment evidence points at a noncanonical identity receipt")
    identity = reopen_primary_identity_receipt(
        root,
        task_id=task_id,
        expected_file_sha256=str(identity_ref["file_sha256"]),
    )
    if identity["identity_receipt_sha256"] != identity_ref["semantic_sha256"]:
        raise ValueError("segment evidence identity receipt semantic hash changed")
    closure_ref = _closed(
        payload["runtime_closure"],
        name="runtime closure ref",
        fields=_ARTIFACT_REF_FIELDS,
    )
    closure_path = _resolve_project_file(
        closure_ref["path"],
        project_root=root,
        name="runtime closure",
    )
    runtime_closure.validate_integrity()
    if (
        closure_path != runtime_closure.artifact_path
        or closure_ref["file_sha256"] != runtime_closure.file_sha256
        or closure_ref["semantic_sha256"] != runtime_closure.closure_sha256
    ):
        raise ValueError("segment evidence points at another runtime closure")
    admission = runtime_closure.admission_payload
    if (
        admission["task_id"] != task_id
        or admission["seed"] != seed
        or admission["segment_index"] != segment_index
        or admission["logical_run_id"] != identity["segment_1_admission"]["logical_run_id"]  # type: ignore[index]
    ):
        raise ValueError("segment evidence closure differs from the retained identity")
    manifest = _validate_manifest(payload["immutable_file_manifest"])
    if payload["immutable_file_manifest_sha256"] != _semantic_sha256({"files": manifest}):
        raise ValueError("immutable task manifest self-hash is invalid")
    task_root = _canonical_root(payload["task_root"], name="retained task root")
    for entry in manifest:
        file_path = task_root.joinpath(*Path(str(entry["relative_path"])).parts)
        file_path = _canonical_file(file_path, name="manifest task artifact")
        if task_root not in file_path.parents:
            raise ValueError("manifest task artifact escapes the task root")
        observed_sha256, observed_size = _file_sha256(file_path)
        if observed_size != entry["size_bytes"] or observed_sha256 != entry["file_sha256"]:
            raise ValueError("manifest task artifact bytes changed")
    if require_exact_current_manifest and manifest != _manifest(task_root):
        raise ValueError("final segment receipt does not cover every immutable task artifact")
    completions_raw = payload["completed_head_receipts"]
    if not isinstance(completions_raw, list):
        raise ValueError("segment completion refs must be a list")
    completions = [
        _closed(item, name="completion evidence ref", fields=_COMPLETION_REF_FIELDS)
        for item in completions_raw
    ]
    if completions != _completion_refs(task_root, closure=runtime_closure, manifest=manifest):
        raise ValueError("segment completion refs differ from live receipt/result bytes")
    payload["immutable_file_manifest"] = manifest
    payload["completed_head_receipts"] = completions
    return payload


__all__ = [
    "EXECUTION_EVIDENCE_RELATIVE",
    "PRIMARY_IDENTITY_RECEIPT_ROLE",
    "PRIMARY_IDENTITY_RECEIPT_SCHEMA",
    "SEGMENT_EVIDENCE_RECEIPT_ROLE",
    "SEGMENT_EVIDENCE_RECEIPT_SCHEMA",
    "identity_receipt_path",
    "publish_primary_identity_receipt",
    "publish_segment_evidence_receipt",
    "reopen_primary_identity_receipt",
    "reopen_segment_evidence_receipt",
    "segment_evidence_receipt_path",
]
