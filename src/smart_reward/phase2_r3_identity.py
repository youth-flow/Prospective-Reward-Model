"""Closed identities and admission capabilities for Phase-2 recovery revision 3.

The objects in this module do not train models, inspect scheduler state, or
read/write artifacts.  They bind capabilities produced by the corresponding
validators.  In particular, an :class:`ArtifactRef` is only a content-addressed
reference; the exact schema/role checks below determine which validated
capability may be consumed at each gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from .phase2_checkpoint import (
    CHECKPOINT_SCHEMA,
    SIGNAL_RECEIPT_SCHEMA,
    TRAINING_PROGRESS_DETAILS_SCHEMA,
)
from .phase2_r3_config import R3ScienceConfigBundle
from .phase2_r3_gate0 import R3Gate0Capability
from .phase2_r3_gate1 import (
    R3Gate1Capabilities,
    r3_container_artifact_ref,
    r3_gate1_artifact_ref,
    r3_source_artifact_ref,
)
from .phase2_r3_inputs import R3TrainMaterializationCapability
from .phase2_r3_materialization import ValidatedR3Materialization

if TYPE_CHECKING:
    from .phase2_r3_profile_artifacts import VerifiedGatePOperationalBundle
    from .phase2_r3_terminal import (
        ContinuablePrimaryTerminalCapability,
        SuccessfulProfileTerminalCapability,
    )

R2_RECOVERY_DESIGN_SHA256: Final = (
    "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"
)

GATE_P_ADMISSION_SCHEMA: Final = "phase2-recovery-r3-gate-p-admission/v1"
GATE_P_RUN_SCHEMA: Final = "phase2-recovery-r3-throughput-profile/v1"
GATE_P_AUTHORIZATION_SCHEMA: Final = "phase2-recovery-r3-gate-p-authorization/v1"
R3_PRIMARY_DESIGN_SCHEMA: Final = "phase2-recovery-r3-primary/v1"
R3_CONTINUATION_EVIDENCE_SCHEMA: Final = "phase2-recovery-r3-primary-continuation-evidence/v1"
R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA: Final = "phase2-recovery-r3-primary-segment-admission/v1"

GATE0_ARTIFACT_SCHEMA: Final = "phase2-recovery-r3-gate0-r2-failure-parent/v1"
GATE0_ARTIFACT_ROLE: Final = "validated_r2_failure_parent"
GATE1_ARTIFACT_SCHEMA: Final = "phase2-recovery-r3-gate1-implementation/v1"
GATE1_ARTIFACT_ROLE: Final = "validated_r3_implementation"
SOURCE_ARTIFACT_SCHEMA: Final = "phase2-recovery-r3-clean-source/v1"
SOURCE_ARTIFACT_ROLE: Final = "validated_clean_source"
CONTAINER_ARTIFACT_SCHEMA: Final = "phase2-recovery-r3-container/v1"
CONTAINER_ARTIFACT_ROLE: Final = "validated_container_image"
CONFIG_ARTIFACT_SCHEMA: Final = "phase2-recovery-r3-science-config/v1"
CONFIG_ARTIFACT_ROLE: Final = "validated_r3_science_config_bytes"

FORMAL_CUDA_PROFILE_RESULT_SCHEMA: Final = "phase2-recovery-r3-formal-cuda-profile-result/v1"
FORMAL_CUDA_PROFILE_RESULT_ROLE: Final = "validated_formal_cuda_profile_result"
SUCCESSFUL_PROFILE_TERMINAL_SCHEMA: Final = "phase2-recovery-r3-external-profile-terminal/v1"
SUCCESSFUL_PROFILE_TERMINAL_ROLE: Final = "external_scheduler_terminal_completed_zero_exit"
RESOURCE_PLAN_SCHEMA: Final = "phase2-recovery-r3-resource-plan/v1"
RESOURCE_PLAN_ROLE: Final = "validated_gate_p_resource_plan"

CONTINUABLE_PRIMARY_TERMINAL_SCHEMA: Final = (
    "phase2-recovery-r3-external-primary-segment-terminal/v1"
)
CONTINUABLE_PRIMARY_TERMINAL_ROLE: Final = (
    "external_scheduler_terminal_completed_zero_exit_continuation_required"
)
VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA: Final = (
    "phase2-recovery-r3-verified-continuation-checkpoint/v1"
)
VERIFIED_CONTINUATION_CHECKPOINT_ROLE: Final = "verified_state_complete_checkpoint_same_logical_run"

R3_CAMPAIGN_KIND: Final = "phase2_recovery_revision3_primary_only"
R3_EXECUTION_REVISION: Final = 3
R3_CAMPAIGN_ROLE: Final = "train_only_optimizer_recovery"
R3_ORDERED_SEEDS: Final = (20260801, 20260802, 20260803)
R3_PRIMARY_HEADS: Final = ("bt_mle", "prorm_plus")
R3_TASK_SEED_MAP: Final = ((0, 20260801), (1, 20260802), (2, 20260803))

GATE_P_SEED: Final = 20260801
GATE_P_HEAD_ORDER: Final = R3_PRIMARY_HEADS
GATE_P_UPDATES_PER_HEAD: Final = 100
GATE_P_AUDIT_UPDATES: Final = (0, 20, 40, 60, 80, 100)
GATE_P_SCHEDULER_SEGMENTS: Final = 1
GATE_P_INFORMATION_BOUNDARY: Final = "train_only_runtime_measurement"
GATE_P_STOP_REASON: Final = "predeclared_profile_update_cap"

HeadName: TypeAlias = Literal["bt_mle", "prorm_plus"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FACTORY_TOKEN = object()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256 digest")
    return value


def _require_exact_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_factory_token(value: object, *, name: str) -> None:
    if value is not _FACTORY_TOKEN:
        raise TypeError(f"{name} must be produced by its validating factory")


def _require_closed_mapping(
    value: object,
    *,
    name: str,
    keys: set[str] | frozenset[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != set(keys):
        raise ValueError(f"{name} has an invalid closed field set")
    try:
        copied = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON data") from error
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must remain a JSON object")
    return copied


def _require_exact_serialized_identity(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    name: str,
) -> None:
    try:
        observed_bytes = json.dumps(
            dict(observed),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        expected_bytes = json.dumps(
            dict(expected),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON data") from error
    if observed_bytes != expected_bytes:
        raise ValueError(f"{name} differs from its rehydrated closed identity")


def _validate_artifact(
    value: object,
    *,
    name: str,
    schema_version: str,
    role: str,
) -> ArtifactRef:
    if type(value) is not ArtifactRef:
        raise TypeError(f"{name} must be ArtifactRef")
    value.validate_integrity()
    if value.schema_version != schema_version or value.role != role:
        raise ValueError(f"{name} must have schema {schema_version!r} and role {role!r}")
    return value


def _validate_science(value: object) -> R3ScienceConfigBundle:
    if type(value) is not R3ScienceConfigBundle:
        raise TypeError("science must be R3ScienceConfigBundle")
    value.validate_integrity()
    _require_digest(value.semantic_sha256, name="science.semantic_sha256")
    _require_digest(value.file_sha256, name="science.file_sha256")
    return value


def _validate_materialization(value: object) -> ValidatedR3Materialization:
    if type(value) is not ValidatedR3Materialization:
        raise TypeError("materialization must be ValidatedR3Materialization")
    value.validate_integrity()
    _require_digest(value.attestation_sha256, name="materialization.attestation_sha256")
    _require_digest(
        value.science_semantic_sha256,
        name="materialization.science_semantic_sha256",
    )
    _require_exact_int(value.seed, name="materialization.seed", minimum=0)
    return value


def _validate_gate0_capability(value: object) -> R3Gate0Capability:
    if type(value) is not R3Gate0Capability:
        raise TypeError("gate0_capability must be exact R3Gate0Capability")
    value.validate_integrity()
    return value


def _validate_gate1_capabilities(value: object) -> R3Gate1Capabilities:
    if type(value) is not R3Gate1Capabilities:
        raise TypeError("gate1_capabilities must be exact R3Gate1Capabilities")
    value.__post_init__()
    return value


def _validate_materialization_capability(
    value: object,
) -> R3TrainMaterializationCapability:
    if type(value) is not R3TrainMaterializationCapability:
        raise TypeError("materialization_capability must be exact R3TrainMaterializationCapability")
    value.validate_integrity()
    return value


def _validate_operational_bundle(
    value: object,
) -> VerifiedGatePOperationalBundle:
    from .phase2_r3_profile_artifacts import VerifiedGatePOperationalBundle

    if type(value) is not VerifiedGatePOperationalBundle:
        raise TypeError("operational_bundle must be exact VerifiedGatePOperationalBundle")
    value.validate_integrity()
    return value


def _validate_successful_profile_terminal(
    value: object,
) -> SuccessfulProfileTerminalCapability:
    from .phase2_r3_terminal import SuccessfulProfileTerminalCapability

    if type(value) is not SuccessfulProfileTerminalCapability:
        raise TypeError("successful_terminal must be exact SuccessfulProfileTerminalCapability")
    value.validate_integrity()
    return value


def _operational_policy_hashes(
    bundle: VerifiedGatePOperationalBundle,
) -> dict[str, str]:
    """Mirror the primary executor's closed policies from one sealed plan."""

    plan = _validate_operational_bundle(bundle)
    shared = {
        "resource_plan_sha256": plan.resource_plan_sha256,
        "information_boundary": "train_only",
    }
    return {
        "checkpoint_policy_sha256": _canonical_sha256(
            {
                "schema_version": "phase2-recovery-r3-checkpoint-policy/v1",
                **shared,
                "state_schema": CHECKPOINT_SCHEMA,
                "durable_checkpoint_cadence_updates": (plan.durable_checkpoint_cadence_updates),
                "mandatory_checkpoint_updates": list(plan.mandatory_checkpoint_updates),
                "mandatory_checkpoint_roles": list(plan.mandatory_checkpoint_roles),
                "atomic_no_overwrite_fsync": True,
            }
        ),
        "progress_policy_sha256": _canonical_sha256(
            {
                "schema_version": "phase2-recovery-r3-progress-policy/v1",
                **shared,
                "details_schema": TRAINING_PROGRESS_DETAILS_SCHEMA,
                "audit_cadence_updates": plan.audit_cadence_updates,
                "publish_after_every_audit_or_checkpoint": True,
                "hash_chain_no_overwrite": True,
            }
        ),
        "signal_policy_sha256": _canonical_sha256(
            {
                "schema_version": "phase2-recovery-r3-signal-policy/v1",
                **shared,
                "receipt_schema": SIGNAL_RECEIPT_SCHEMA,
                "advance_signal_lead_seconds": (plan.advance_signal_lead_seconds),
                "handled_signals": ["USR1", "TERM", "INT"],
                "safe_boundary_checkpoint_required": True,
                "terminal_success_claimed": False,
            }
        ),
        "continuation_policy_sha256": _canonical_sha256(
            {
                "schema_version": "phase2-recovery-r3-continuation-policy/v1",
                **shared,
                "max_scheduler_segments": plan.max_scheduler_segments,
                "segment_boundaries": plan.to_dict()["resource_plan"]["segment_boundaries"],
                "same_logical_run_only": True,
                "fresh_restart_forbidden": True,
                "discrete_replay_exact": True,
                "numeric_replay_relative_tolerance": 1.0e-10,
                "numeric_replay_absolute_tolerance": 1.0e-14,
            }
        ),
    }


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A typed content-addressed reference to an externally validated artifact."""

    schema_version: str
    artifact_sha256: str
    role: str

    def __post_init__(self) -> None:
        _require_text(self.schema_version, name="schema_version")
        _require_digest(self.artifact_sha256, name="artifact_sha256")
        _require_text(self.role, name="role")

    def validate_integrity(self) -> None:
        self.__post_init__()

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "artifact_sha256": self.artifact_sha256,
            "role": self.role,
        }


def _gate_p_admission_payload(
    *,
    gate0: ArtifactRef,
    gate1: ArtifactRef,
    source: ArtifactRef,
    container: ArtifactRef,
    config: ArtifactRef,
) -> dict[str, object]:
    return {
        "schema_version": GATE_P_ADMISSION_SCHEMA,
        "gate0": gate0.to_dict(),
        "gate1": gate1.to_dict(),
        "source": source.to_dict(),
        "container": container.to_dict(),
        "config": config.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class GatePAdmission:
    """Gate-0/Gate-1 capability authorizing the one formal profile run."""

    schema_version: str
    gate0: ArtifactRef
    gate1: ArtifactRef
    source: ArtifactRef
    container: ArtifactRef
    config: ArtifactRef
    admission_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory_token(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        if self.schema_version != GATE_P_ADMISSION_SCHEMA:
            raise ValueError("Gate-P admission schema is not frozen")
        _validate_artifact(
            self.gate0,
            name="gate0",
            schema_version=GATE0_ARTIFACT_SCHEMA,
            role=GATE0_ARTIFACT_ROLE,
        )
        _validate_artifact(
            self.gate1,
            name="gate1",
            schema_version=GATE1_ARTIFACT_SCHEMA,
            role=GATE1_ARTIFACT_ROLE,
        )
        _validate_artifact(
            self.source,
            name="source",
            schema_version=SOURCE_ARTIFACT_SCHEMA,
            role=SOURCE_ARTIFACT_ROLE,
        )
        _validate_artifact(
            self.container,
            name="container",
            schema_version=CONTAINER_ARTIFACT_SCHEMA,
            role=CONTAINER_ARTIFACT_ROLE,
        )
        _validate_artifact(
            self.config,
            name="config",
            schema_version=CONFIG_ARTIFACT_SCHEMA,
            role=CONFIG_ARTIFACT_ROLE,
        )
        _require_digest(self.admission_sha256, name="admission_sha256")
        expected = _canonical_sha256(
            _gate_p_admission_payload(
                gate0=self.gate0,
                gate1=self.gate1,
                source=self.source,
                container=self.container,
                config=self.config,
            )
        )
        if self.admission_sha256 != expected:
            raise ValueError("Gate-P admission SHA does not match its closed payload")

    def validate_integrity(self) -> None:
        _require_factory_token(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _gate_p_admission_payload(
            gate0=self.gate0,
            gate1=self.gate1,
            source=self.source,
            container=self.container,
            config=self.config,
        )
        payload["admission_sha256"] = self.admission_sha256
        return payload


def create_gate_p_admission(
    *,
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    science: R3ScienceConfigBundle,
) -> GatePAdmission:
    """Create Gate-P admission only from current validator-sealed authorities."""

    gate0_capability = _validate_gate0_capability(gate0_capability)
    gate1_capabilities = _validate_gate1_capabilities(gate1_capabilities)
    science = _validate_science(science)
    gate0 = gate0_capability.to_artifact_ref()
    gate1 = r3_gate1_artifact_ref(gate1_capabilities.gate1)
    source = r3_source_artifact_ref(gate1_capabilities.source)
    container = r3_container_artifact_ref(gate1_capabilities.container)
    for name, value in (
        ("gate0", gate0),
        ("gate1", gate1),
        ("source", source),
        ("container", container),
    ):
        if type(value) is not ArtifactRef:
            raise TypeError(f"{name} capability produced an invalid artifact reference")
    config = ArtifactRef(
        schema_version=CONFIG_ARTIFACT_SCHEMA,
        artifact_sha256=science.file_sha256,
        role=CONFIG_ARTIFACT_ROLE,
    )
    payload = _gate_p_admission_payload(
        gate0=gate0,
        gate1=gate1,
        source=source,
        container=container,
        config=config,
    )
    result = GatePAdmission(
        schema_version=GATE_P_ADMISSION_SCHEMA,
        gate0=gate0,
        gate1=gate1,
        source=source,
        container=container,
        config=config,
        admission_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _gate_p_run_payload(
    *,
    materialization_capability: R3TrainMaterializationCapability,
    science: R3ScienceConfigBundle,
    admission: GatePAdmission,
) -> dict[str, object]:
    materialization = materialization_capability.materialization
    return {
        "schema_version": GATE_P_RUN_SCHEMA,
        "materialization_attestation_sha256": materialization.attestation_sha256,
        "science_semantic_sha256": science.semantic_sha256,
        "science_file_sha256": science.file_sha256,
        "gate_p_admission_sha256": admission.admission_sha256,
        "seed": GATE_P_SEED,
        "head_order": list(GATE_P_HEAD_ORDER),
        "completed_updates_per_head": GATE_P_UPDATES_PER_HEAD,
        "audit_updates": list(GATE_P_AUDIT_UPDATES),
        "scheduler_segments": GATE_P_SCHEDULER_SEGMENTS,
        "profile_nonreusable": True,
        "information_boundary": GATE_P_INFORMATION_BOUNDARY,
        "stop_reason": GATE_P_STOP_REASON,
    }


@dataclass(frozen=True, slots=True)
class ValidatedGatePRun:
    """The fixed, one-segment and permanently non-reusable Gate-P run identity."""

    schema_version: str
    materialization_capability: R3TrainMaterializationCapability = field(
        repr=False,
        compare=False,
    )
    science: R3ScienceConfigBundle = field(repr=False)
    admission: GatePAdmission
    seed: int
    head_order: tuple[HeadName, HeadName]
    completed_updates_per_head: int
    audit_updates: tuple[int, ...]
    scheduler_segments: int
    profile_nonreusable: bool
    information_boundary: str
    stop_reason: str
    profile_run_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory_token(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        if self.schema_version != GATE_P_RUN_SCHEMA:
            raise ValueError("Gate-P run schema is not frozen")
        materialization_capability = _validate_materialization_capability(
            self.materialization_capability
        )
        materialization = materialization_capability.materialization
        science = _validate_science(self.science)
        if type(self.admission) is not GatePAdmission:
            raise TypeError("admission must be GatePAdmission")
        self.admission.validate_integrity()
        if materialization.seed != GATE_P_SEED:
            raise ValueError("Gate-P materialization must use seed 20260801")
        if materialization.science_semantic_sha256 != science.semantic_sha256:
            raise ValueError("Gate-P materialization is bound to another science contract")
        if materialization.science_file_sha256 != science.file_sha256:
            raise ValueError("Gate-P materialization is bound to other science bytes")
        if self.admission.config.artifact_sha256 != science.file_sha256:
            raise ValueError("Gate-P admission config does not bind the science file bytes")
        frozen_values = (
            (self.seed, GATE_P_SEED, "seed"),
            (self.head_order, GATE_P_HEAD_ORDER, "head_order"),
            (
                self.completed_updates_per_head,
                GATE_P_UPDATES_PER_HEAD,
                "completed_updates_per_head",
            ),
            (self.audit_updates, GATE_P_AUDIT_UPDATES, "audit_updates"),
            (
                self.scheduler_segments,
                GATE_P_SCHEDULER_SEGMENTS,
                "scheduler_segments",
            ),
            (self.profile_nonreusable, True, "profile_nonreusable"),
            (
                self.information_boundary,
                GATE_P_INFORMATION_BOUNDARY,
                "information_boundary",
            ),
            (self.stop_reason, GATE_P_STOP_REASON, "stop_reason"),
        )
        for observed, expected, name in frozen_values:
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(f"Gate-P {name} is not frozen")
        _require_digest(self.profile_run_sha256, name="profile_run_sha256")
        expected_sha = _canonical_sha256(
            _gate_p_run_payload(
                materialization_capability=materialization_capability,
                science=science,
                admission=self.admission,
            )
        )
        if self.profile_run_sha256 != expected_sha:
            raise ValueError("Gate-P run SHA does not match its closed payload")

    def validate_integrity(self) -> None:
        _require_factory_token(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    @property
    def materialization(self) -> ValidatedR3Materialization:
        """Expose live tensors for profile compute; the sealed capability is authority."""

        return self.materialization_capability.materialization

    def to_dict(self) -> dict[str, object]:
        payload = _gate_p_run_payload(
            materialization_capability=self.materialization_capability,
            science=self.science,
            admission=self.admission,
        )
        payload["profile_run_sha256"] = self.profile_run_sha256
        return payload


def create_validated_gate_p_run(
    *,
    materialization_capability: R3TrainMaterializationCapability,
    science: R3ScienceConfigBundle,
    admission: GatePAdmission,
) -> ValidatedGatePRun:
    """Bind the frozen profile workload to science, materialization and gates."""

    materialization_capability = _validate_materialization_capability(materialization_capability)
    _validate_science(science)
    if type(admission) is not GatePAdmission:
        raise TypeError("admission must be GatePAdmission")
    admission.validate_integrity()
    payload = _gate_p_run_payload(
        materialization_capability=materialization_capability,
        science=science,
        admission=admission,
    )
    result = ValidatedGatePRun(
        schema_version=GATE_P_RUN_SCHEMA,
        materialization_capability=materialization_capability,
        science=science,
        admission=admission,
        seed=GATE_P_SEED,
        head_order=GATE_P_HEAD_ORDER,
        completed_updates_per_head=GATE_P_UPDATES_PER_HEAD,
        audit_updates=GATE_P_AUDIT_UPDATES,
        scheduler_segments=GATE_P_SCHEDULER_SEGMENTS,
        profile_nonreusable=True,
        information_boundary=GATE_P_INFORMATION_BOUNDARY,
        stop_reason=GATE_P_STOP_REASON,
        profile_run_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _gate_p_authorization_payload(
    *,
    operational_bundle_file_sha256: str,
    operational_bundle_semantic_sha256: str,
    profile_run_sha256: str,
    gate_p_admission_sha256: str,
    science_semantic_sha256: str,
    science_file_sha256: str,
    gate0_artifact_sha256: str,
    gate1_artifact_sha256: str,
    source_artifact_sha256: str,
    container_artifact_sha256: str,
    formal_cuda_profile_result: ArtifactRef,
    scheduler_terminal: ArtifactRef,
    resource_plan: ArtifactRef,
    checkpoint_policy_sha256: str,
    progress_policy_sha256: str,
    signal_policy_sha256: str,
    continuation_policy_sha256: str,
    max_scheduler_segments: int,
) -> dict[str, object]:
    return {
        "schema_version": GATE_P_AUTHORIZATION_SCHEMA,
        "operational_bundle_file_sha256": operational_bundle_file_sha256,
        "operational_bundle_semantic_sha256": (operational_bundle_semantic_sha256),
        "profile_run_sha256": profile_run_sha256,
        "gate_p_admission_sha256": gate_p_admission_sha256,
        "science_semantic_sha256": science_semantic_sha256,
        "science_file_sha256": science_file_sha256,
        "gate0_artifact_sha256": gate0_artifact_sha256,
        "gate1_artifact_sha256": gate1_artifact_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "container_artifact_sha256": container_artifact_sha256,
        "formal_cuda_profile_result": formal_cuda_profile_result.to_dict(),
        "scheduler_terminal": scheduler_terminal.to_dict(),
        "resource_plan": resource_plan.to_dict(),
        "checkpoint_policy_sha256": checkpoint_policy_sha256,
        "progress_policy_sha256": progress_policy_sha256,
        "signal_policy_sha256": signal_policy_sha256,
        "continuation_policy_sha256": continuation_policy_sha256,
        "max_scheduler_segments": max_scheduler_segments,
    }


@dataclass(frozen=True, slots=True)
class GatePAuthorization:
    """Tensor-free capability produced from one sealed profile closure."""

    schema_version: str
    operational_bundle_file_sha256: str
    operational_bundle_semantic_sha256: str
    profile_run_sha256: str
    gate_p_admission_sha256: str
    science_semantic_sha256: str
    science_file_sha256: str
    gate0_artifact_sha256: str
    gate1_artifact_sha256: str
    source_artifact_sha256: str
    container_artifact_sha256: str
    formal_cuda_profile_result: ArtifactRef
    scheduler_terminal: ArtifactRef
    resource_plan: ArtifactRef
    checkpoint_policy_sha256: str
    progress_policy_sha256: str
    signal_policy_sha256: str
    continuation_policy_sha256: str
    max_scheduler_segments: int
    authorization_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory_token(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        if self.schema_version != GATE_P_AUTHORIZATION_SCHEMA:
            raise ValueError("Gate-P authorization schema is not frozen")
        for name in (
            "operational_bundle_file_sha256",
            "operational_bundle_semantic_sha256",
            "profile_run_sha256",
            "gate_p_admission_sha256",
            "science_semantic_sha256",
            "science_file_sha256",
            "gate0_artifact_sha256",
            "gate1_artifact_sha256",
            "source_artifact_sha256",
            "container_artifact_sha256",
            "checkpoint_policy_sha256",
            "progress_policy_sha256",
            "signal_policy_sha256",
            "continuation_policy_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        _require_exact_int(
            self.max_scheduler_segments,
            name="max_scheduler_segments",
            minimum=1,
        )
        _validate_artifact(
            self.formal_cuda_profile_result,
            name="formal_cuda_profile_result",
            schema_version=FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
            role=FORMAL_CUDA_PROFILE_RESULT_ROLE,
        )
        _validate_artifact(
            self.scheduler_terminal,
            name="scheduler_terminal",
            schema_version=SUCCESSFUL_PROFILE_TERMINAL_SCHEMA,
            role=SUCCESSFUL_PROFILE_TERMINAL_ROLE,
        )
        _validate_artifact(
            self.resource_plan,
            name="resource_plan",
            schema_version=RESOURCE_PLAN_SCHEMA,
            role=RESOURCE_PLAN_ROLE,
        )
        _require_digest(self.authorization_sha256, name="authorization_sha256")
        expected = _canonical_sha256(
            _gate_p_authorization_payload(
                operational_bundle_file_sha256=(self.operational_bundle_file_sha256),
                operational_bundle_semantic_sha256=(self.operational_bundle_semantic_sha256),
                profile_run_sha256=self.profile_run_sha256,
                gate_p_admission_sha256=self.gate_p_admission_sha256,
                science_semantic_sha256=self.science_semantic_sha256,
                science_file_sha256=self.science_file_sha256,
                gate0_artifact_sha256=self.gate0_artifact_sha256,
                gate1_artifact_sha256=self.gate1_artifact_sha256,
                source_artifact_sha256=self.source_artifact_sha256,
                container_artifact_sha256=self.container_artifact_sha256,
                formal_cuda_profile_result=self.formal_cuda_profile_result,
                scheduler_terminal=self.scheduler_terminal,
                resource_plan=self.resource_plan,
                checkpoint_policy_sha256=self.checkpoint_policy_sha256,
                progress_policy_sha256=self.progress_policy_sha256,
                signal_policy_sha256=self.signal_policy_sha256,
                continuation_policy_sha256=self.continuation_policy_sha256,
                max_scheduler_segments=self.max_scheduler_segments,
            )
        )
        if self.authorization_sha256 != expected:
            raise ValueError("Gate-P authorization SHA does not match its closed payload")

    def validate_integrity(self) -> None:
        _require_factory_token(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _gate_p_authorization_payload(
            operational_bundle_file_sha256=(self.operational_bundle_file_sha256),
            operational_bundle_semantic_sha256=(self.operational_bundle_semantic_sha256),
            profile_run_sha256=self.profile_run_sha256,
            gate_p_admission_sha256=self.gate_p_admission_sha256,
            science_semantic_sha256=self.science_semantic_sha256,
            science_file_sha256=self.science_file_sha256,
            gate0_artifact_sha256=self.gate0_artifact_sha256,
            gate1_artifact_sha256=self.gate1_artifact_sha256,
            source_artifact_sha256=self.source_artifact_sha256,
            container_artifact_sha256=self.container_artifact_sha256,
            formal_cuda_profile_result=self.formal_cuda_profile_result,
            scheduler_terminal=self.scheduler_terminal,
            resource_plan=self.resource_plan,
            checkpoint_policy_sha256=self.checkpoint_policy_sha256,
            progress_policy_sha256=self.progress_policy_sha256,
            signal_policy_sha256=self.signal_policy_sha256,
            continuation_policy_sha256=self.continuation_policy_sha256,
            max_scheduler_segments=self.max_scheduler_segments,
        )
        payload["authorization_sha256"] = self.authorization_sha256
        return payload


def _authorization_identity_from_bundle(
    bundle: VerifiedGatePOperationalBundle,
) -> dict[str, str]:
    profile_identity = bundle.profile_run_identity
    formal = bundle.formal_cuda_profile_result
    bindings = formal.get("identity_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("operational bundle identity bindings are invalid")
    values = {
        "profile_run_sha256": profile_identity.get("profile_run_sha256"),
        "gate_p_admission_sha256": profile_identity.get("gate_p_admission_sha256"),
        "science_semantic_sha256": profile_identity.get("science_semantic_sha256"),
        "science_file_sha256": profile_identity.get("science_file_sha256"),
        "gate0_artifact_sha256": bindings.get("gate0_artifact_sha256"),
        "gate1_artifact_sha256": bindings.get("gate1_artifact_sha256"),
        "source_artifact_sha256": bindings.get("source_artifact_sha256"),
        "container_artifact_sha256": bindings.get("container_artifact_sha256"),
    }
    normalized: dict[str, str] = {}
    for name, value in values.items():
        normalized[name] = _require_digest(value, name=name)
    if bindings.get("config_artifact_sha256") != normalized["science_file_sha256"]:
        raise ValueError("operational bundle config/science bytes differ")
    return normalized


def authorize_gate_p(
    *,
    operational_bundle: VerifiedGatePOperationalBundle,
    successful_terminal: SuccessfulProfileTerminalCapability,
) -> GatePAuthorization:
    """Issue authorization only from one reopened profile and terminal closure."""

    from .phase2_r3_profile_artifacts import (
        formal_profile_artifact_ref,
        resource_plan_artifact_ref,
    )

    bundle = _validate_operational_bundle(operational_bundle)
    terminal = _validate_successful_profile_terminal(successful_terminal)
    terminal_bundle = _validate_operational_bundle(terminal.operational_bundle)
    closure_fields = (
        "file_sha256",
        "bundle_semantic_sha256",
        "profile_run_sha256",
        "formal_profile_sha256",
        "resource_plan_sha256",
    )
    if any(getattr(terminal_bundle, name) != getattr(bundle, name) for name in closure_fields):
        raise ValueError("successful profile terminal belongs to another operational bundle")
    terminal_payload = terminal.to_dict()
    expected_terminal_links = {
        "operational_bundle_file_sha256": bundle.file_sha256,
        "operational_bundle_semantic_sha256": (bundle.bundle_semantic_sha256),
        "profile_run_sha256": bundle.profile_run_sha256,
        "formal_profile_sha256": bundle.formal_profile_sha256,
        "resource_plan_sha256": bundle.resource_plan_sha256,
    }
    if any(
        terminal_payload.get(name) != expected for name, expected in expected_terminal_links.items()
    ):
        raise ValueError("successful profile terminal closure links differ")
    identity = _authorization_identity_from_bundle(bundle)
    policies = _operational_policy_hashes(bundle)
    formal_cuda_profile_result = formal_profile_artifact_ref(bundle)
    scheduler_terminal = terminal.artifact_ref()
    resource_plan = resource_plan_artifact_ref(bundle)
    payload = _gate_p_authorization_payload(
        operational_bundle_file_sha256=bundle.file_sha256,
        operational_bundle_semantic_sha256=bundle.bundle_semantic_sha256,
        **identity,
        formal_cuda_profile_result=formal_cuda_profile_result,
        scheduler_terminal=scheduler_terminal,
        resource_plan=resource_plan,
        **policies,
        max_scheduler_segments=bundle.max_scheduler_segments,
    )
    result = GatePAuthorization(
        schema_version=GATE_P_AUTHORIZATION_SCHEMA,
        operational_bundle_file_sha256=bundle.file_sha256,
        operational_bundle_semantic_sha256=bundle.bundle_semantic_sha256,
        **identity,
        formal_cuda_profile_result=formal_cuda_profile_result,
        scheduler_terminal=scheduler_terminal,
        resource_plan=resource_plan,
        **policies,
        max_scheduler_segments=bundle.max_scheduler_segments,
        authorization_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _primary_design_payload(
    *,
    science: R3ScienceConfigBundle,
    gate0: ArtifactRef,
    gate1: ArtifactRef,
    profile_authorization: GatePAuthorization,
    source: ArtifactRef,
    container: ArtifactRef,
    resource_policy_sha256: str,
    checkpoint_policy_sha256: str,
    progress_policy_sha256: str,
    signal_policy_sha256: str,
    continuation_policy_sha256: str,
    max_scheduler_segments: int,
) -> dict[str, object]:
    return {
        "schema_version": R3_PRIMARY_DESIGN_SCHEMA,
        "campaign_kind": R3_CAMPAIGN_KIND,
        "execution_revision": R3_EXECUTION_REVISION,
        "campaign_role": R3_CAMPAIGN_ROLE,
        "science_semantic_sha256": science.semantic_sha256,
        "science_file_sha256": science.file_sha256,
        "gate0": gate0.to_dict(),
        "gate1": gate1.to_dict(),
        "profile_authorization_sha256": profile_authorization.authorization_sha256,
        "source": source.to_dict(),
        "container": container.to_dict(),
        "ordered_seeds": list(R3_ORDERED_SEEDS),
        "primary_heads": list(R3_PRIMARY_HEADS),
        "task_seed_map": [{"task_id": task_id, "seed": seed} for task_id, seed in R3_TASK_SEED_MAP],
        "resource_policy_sha256": resource_policy_sha256,
        "checkpoint_policy_sha256": checkpoint_policy_sha256,
        "progress_policy_sha256": progress_policy_sha256,
        "signal_policy_sha256": signal_policy_sha256,
        "continuation_policy_sha256": continuation_policy_sha256,
        "max_scheduler_segments": max_scheduler_segments,
    }


@dataclass(frozen=True, slots=True)
class R3PrimaryDesign:
    """Closed primary-only campaign identity admitted by Gate P."""

    schema_version: str
    campaign_kind: str
    execution_revision: int
    campaign_role: str
    science: R3ScienceConfigBundle = field(repr=False)
    gate0: ArtifactRef
    gate1: ArtifactRef
    profile_authorization: GatePAuthorization = field(repr=False)
    source: ArtifactRef
    container: ArtifactRef
    ordered_seeds: tuple[int, int, int]
    primary_heads: tuple[HeadName, HeadName]
    task_seed_map: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    resource_policy_sha256: str
    checkpoint_policy_sha256: str
    progress_policy_sha256: str
    signal_policy_sha256: str
    continuation_policy_sha256: str
    max_scheduler_segments: int
    design_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory_token(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        science = _validate_science(self.science)
        if type(self.profile_authorization) is not GatePAuthorization:
            raise TypeError("profile_authorization must be GatePAuthorization")
        self.profile_authorization.validate_integrity()
        exact_values = (
            (self.schema_version, R3_PRIMARY_DESIGN_SCHEMA, "schema_version"),
            (self.campaign_kind, R3_CAMPAIGN_KIND, "campaign_kind"),
            (self.execution_revision, R3_EXECUTION_REVISION, "execution_revision"),
            (self.campaign_role, R3_CAMPAIGN_ROLE, "campaign_role"),
            (self.ordered_seeds, R3_ORDERED_SEEDS, "ordered_seeds"),
            (self.primary_heads, R3_PRIMARY_HEADS, "primary_heads"),
            (self.task_seed_map, R3_TASK_SEED_MAP, "task_seed_map"),
        )
        for observed, expected, name in exact_values:
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(f"R3 primary {name} is not frozen")
        _validate_artifact(
            self.gate0,
            name="gate0",
            schema_version=GATE0_ARTIFACT_SCHEMA,
            role=GATE0_ARTIFACT_ROLE,
        )
        _validate_artifact(
            self.gate1,
            name="gate1",
            schema_version=GATE1_ARTIFACT_SCHEMA,
            role=GATE1_ARTIFACT_ROLE,
        )
        _validate_artifact(
            self.source,
            name="source",
            schema_version=SOURCE_ARTIFACT_SCHEMA,
            role=SOURCE_ARTIFACT_ROLE,
        )
        _validate_artifact(
            self.container,
            name="container",
            schema_version=CONTAINER_ARTIFACT_SCHEMA,
            role=CONTAINER_ARTIFACT_ROLE,
        )
        authorization = self.profile_authorization
        artifact_links = (
            (self.gate0, authorization.gate0_artifact_sha256, "Gate-0"),
            (self.gate1, authorization.gate1_artifact_sha256, "Gate-1"),
            (self.source, authorization.source_artifact_sha256, "source"),
            (
                self.container,
                authorization.container_artifact_sha256,
                "container",
            ),
        )
        for artifact, expected_sha256, name in artifact_links:
            if artifact.artifact_sha256 != expected_sha256:
                raise ValueError(f"primary design {name} ref differs from Gate P")
        if science.semantic_sha256 != authorization.science_semantic_sha256:
            raise ValueError("primary design science differs from Gate P")
        if science.file_sha256 != authorization.science_file_sha256:
            raise ValueError("primary design science bytes differ from Gate P")
        policy_hashes = (
            "resource_policy_sha256",
            "checkpoint_policy_sha256",
            "progress_policy_sha256",
            "signal_policy_sha256",
            "continuation_policy_sha256",
        )
        for name in policy_hashes:
            _require_digest(getattr(self, name), name=name)
        if self.resource_policy_sha256 != authorization.resource_plan.artifact_sha256:
            raise ValueError("resource policy must be the Gate-P authorized resource plan")
        for name in policy_hashes[1:]:
            if getattr(self, name) != getattr(authorization, name):
                raise ValueError(f"primary design {name} differs from Gate-P derivation")
        _require_exact_int(
            self.max_scheduler_segments,
            name="max_scheduler_segments",
            minimum=1,
        )
        if self.max_scheduler_segments != authorization.max_scheduler_segments:
            raise ValueError("primary max_scheduler_segments differs from Gate-P derivation")
        _require_digest(self.design_sha256, name="design_sha256")
        expected = _canonical_sha256(
            _primary_design_payload(
                science=science,
                gate0=self.gate0,
                gate1=self.gate1,
                profile_authorization=self.profile_authorization,
                source=self.source,
                container=self.container,
                resource_policy_sha256=self.resource_policy_sha256,
                checkpoint_policy_sha256=self.checkpoint_policy_sha256,
                progress_policy_sha256=self.progress_policy_sha256,
                signal_policy_sha256=self.signal_policy_sha256,
                continuation_policy_sha256=self.continuation_policy_sha256,
                max_scheduler_segments=self.max_scheduler_segments,
            )
        )
        if self.design_sha256 != expected:
            raise ValueError("R3 primary design SHA does not match its closed payload")
        if self.design_sha256 == R2_RECOVERY_DESIGN_SHA256:
            raise ValueError("the R2 recovery design hash is forbidden for R3")

    def validate_integrity(self) -> None:
        _require_factory_token(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _primary_design_payload(
            science=self.science,
            gate0=self.gate0,
            gate1=self.gate1,
            profile_authorization=self.profile_authorization,
            source=self.source,
            container=self.container,
            resource_policy_sha256=self.resource_policy_sha256,
            checkpoint_policy_sha256=self.checkpoint_policy_sha256,
            progress_policy_sha256=self.progress_policy_sha256,
            signal_policy_sha256=self.signal_policy_sha256,
            continuation_policy_sha256=self.continuation_policy_sha256,
            max_scheduler_segments=self.max_scheduler_segments,
        )
        payload["design_sha256"] = self.design_sha256
        return payload


def create_r3_primary_design(
    *,
    science: R3ScienceConfigBundle,
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    profile_authorization: GatePAuthorization,
    operational_bundle: VerifiedGatePOperationalBundle,
) -> R3PrimaryDesign:
    """Derive one primary design solely from sealed upstream capabilities."""

    science = _validate_science(science)
    gate0_capability = _validate_gate0_capability(gate0_capability)
    gate1_capabilities = _validate_gate1_capabilities(gate1_capabilities)
    if type(profile_authorization) is not GatePAuthorization:
        raise TypeError("profile_authorization must be GatePAuthorization")
    profile_authorization.validate_integrity()
    bundle = _validate_operational_bundle(operational_bundle)
    authorization_links = {
        "operational_bundle_file_sha256": bundle.file_sha256,
        "operational_bundle_semantic_sha256": (bundle.bundle_semantic_sha256),
        "profile_run_sha256": bundle.profile_run_sha256,
        "formal_profile_sha256": bundle.formal_profile_sha256,
        "resource_plan_sha256": bundle.resource_plan_sha256,
    }
    if (
        profile_authorization.operational_bundle_file_sha256
        != authorization_links["operational_bundle_file_sha256"]
        or profile_authorization.operational_bundle_semantic_sha256
        != authorization_links["operational_bundle_semantic_sha256"]
        or profile_authorization.profile_run_sha256 != authorization_links["profile_run_sha256"]
        or profile_authorization.formal_cuda_profile_result.artifact_sha256
        != authorization_links["formal_profile_sha256"]
        or profile_authorization.resource_plan.artifact_sha256
        != authorization_links["resource_plan_sha256"]
    ):
        raise ValueError("profile authorization belongs to another operational bundle")
    admission = create_gate_p_admission(
        gate0_capability=gate0_capability,
        gate1_capabilities=gate1_capabilities,
        science=science,
    )
    identity = _authorization_identity_from_bundle(bundle)
    if (
        admission.admission_sha256 != identity["gate_p_admission_sha256"]
        or profile_authorization.gate_p_admission_sha256 != admission.admission_sha256
        or science.semantic_sha256 != identity["science_semantic_sha256"]
        or science.file_sha256 != identity["science_file_sha256"]
    ):
        raise ValueError("science or Gate-P admission differs from operational bundle")
    gate0 = admission.gate0
    gate1 = admission.gate1
    source = admission.source
    container = admission.container
    for ref, name in (
        (gate0, "gate0_artifact_sha256"),
        (gate1, "gate1_artifact_sha256"),
        (source, "source_artifact_sha256"),
        (container, "container_artifact_sha256"),
    ):
        if ref.artifact_sha256 != identity[name] or ref.artifact_sha256 != getattr(
            profile_authorization, name
        ):
            raise ValueError("Gate/source/container capability differs from profile closure")
    policy_hashes = _operational_policy_hashes(bundle)
    for name, value in policy_hashes.items():
        if value != getattr(profile_authorization, name):
            raise ValueError(f"profile authorization {name} differs from operational bundle")
    max_scheduler_segments = bundle.max_scheduler_segments
    if max_scheduler_segments != profile_authorization.max_scheduler_segments:
        raise ValueError("profile authorization max segments differs from operational bundle")
    resource_policy_sha256 = profile_authorization.resource_plan.artifact_sha256
    payload = _primary_design_payload(
        science=science,
        gate0=gate0,
        gate1=gate1,
        profile_authorization=profile_authorization,
        source=source,
        container=container,
        resource_policy_sha256=resource_policy_sha256,
        **policy_hashes,
        max_scheduler_segments=max_scheduler_segments,
    )
    result = R3PrimaryDesign(
        schema_version=R3_PRIMARY_DESIGN_SCHEMA,
        campaign_kind=R3_CAMPAIGN_KIND,
        execution_revision=R3_EXECUTION_REVISION,
        campaign_role=R3_CAMPAIGN_ROLE,
        science=science,
        gate0=gate0,
        gate1=gate1,
        profile_authorization=profile_authorization,
        source=source,
        container=container,
        ordered_seeds=R3_ORDERED_SEEDS,
        primary_heads=R3_PRIMARY_HEADS,
        task_seed_map=R3_TASK_SEED_MAP,
        resource_policy_sha256=resource_policy_sha256,
        **policy_hashes,
        max_scheduler_segments=max_scheduler_segments,
        design_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _logical_run_id(
    *,
    design_sha256: str,
    task_id: int,
    seed: int,
    materialization_attestation_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "namespace": "phase2-recovery-r3-logical-run/v1",
            "design_sha256": design_sha256,
            "task_id": task_id,
            "seed": seed,
            "materialization_attestation_sha256": materialization_attestation_sha256,
        }
    )


def _head_run_ids(*, logical_run_id: str) -> tuple[str, str]:
    return tuple(
        _canonical_sha256(
            {
                "namespace": "phase2-recovery-r3-head-run/v1",
                "logical_run_id": logical_run_id,
                "head": head,
            }
        )
        for head in R3_PRIMARY_HEADS
    )  # type: ignore[return-value]


def _scheduler_segment_id(*, logical_run_id: str, segment_index: int) -> str:
    return _canonical_sha256(
        {
            "namespace": "phase2-recovery-r3-scheduler-segment/v1",
            "logical_run_id": logical_run_id,
            "segment_index": segment_index,
        }
    )


def _continuation_evidence_payload(
    *,
    predecessor: PrimarySegmentAdmission,
    scheduler_terminal: ArtifactRef,
    verified_checkpoint: ArtifactRef,
) -> dict[str, object]:
    return {
        "schema_version": R3_CONTINUATION_EVIDENCE_SCHEMA,
        "design_sha256": predecessor.design.design_sha256,
        "predecessor_admission_sha256": predecessor.admission_sha256,
        "logical_run_id": predecessor.logical_run_id,
        "task_id": predecessor.task_id,
        "seed": predecessor.seed,
        "predecessor_segment_index": predecessor.segment_index,
        "materialization_attestation_sha256": (predecessor.materialization.attestation_sha256),
        "scheduler_terminal": scheduler_terminal.to_dict(),
        "verified_checkpoint": verified_checkpoint.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class VerifiedContinuationEvidence:
    """Typed proof that one prior segment may continue the same logical run."""

    schema_version: str
    predecessor: PrimarySegmentAdmission = field(repr=False)
    scheduler_terminal: ArtifactRef
    verified_checkpoint: ArtifactRef
    evidence_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory_token(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        if self.schema_version != R3_CONTINUATION_EVIDENCE_SCHEMA:
            raise ValueError("continuation evidence schema is not frozen")
        if type(self.predecessor) is not PrimarySegmentAdmission:
            raise TypeError("predecessor must be PrimarySegmentAdmission")
        self.predecessor.validate_integrity()
        _validate_artifact(
            self.scheduler_terminal,
            name="scheduler_terminal",
            schema_version=CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
            role=CONTINUABLE_PRIMARY_TERMINAL_ROLE,
        )
        _validate_artifact(
            self.verified_checkpoint,
            name="verified_checkpoint",
            schema_version=VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
            role=VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
        )
        _require_digest(self.evidence_sha256, name="evidence_sha256")
        expected = _canonical_sha256(
            _continuation_evidence_payload(
                predecessor=self.predecessor,
                scheduler_terminal=self.scheduler_terminal,
                verified_checkpoint=self.verified_checkpoint,
            )
        )
        if self.evidence_sha256 != expected:
            raise ValueError("continuation evidence SHA does not match its payload")

    def validate_integrity(self) -> None:
        _require_factory_token(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _continuation_evidence_payload(
            predecessor=self.predecessor,
            scheduler_terminal=self.scheduler_terminal,
            verified_checkpoint=self.verified_checkpoint,
        )
        payload["evidence_sha256"] = self.evidence_sha256
        return payload


def validate_continuation_evidence(
    *,
    predecessor: PrimarySegmentAdmission,
    continuable_terminal: ContinuablePrimaryTerminalCapability,
) -> VerifiedContinuationEvidence:
    """Bind one exact sealed predecessor terminal and its derived checkpoint."""

    if type(predecessor) is not PrimarySegmentAdmission:
        raise TypeError("predecessor must be PrimarySegmentAdmission")
    predecessor.validate_integrity()
    from .phase2_r3_terminal import ContinuablePrimaryTerminalCapability

    if type(continuable_terminal) is not ContinuablePrimaryTerminalCapability:
        raise TypeError("continuable_terminal must be exact ContinuablePrimaryTerminalCapability")
    continuable_terminal.validate_integrity()
    runtime_closure = continuable_terminal.runtime_closure
    runtime_closure.validate_integrity()
    _require_exact_serialized_identity(
        runtime_closure.admission_payload,
        predecessor.to_dict(),
        name="continuable terminal predecessor admission",
    )
    outcome = runtime_closure.outcome_payload
    expected_outcome_identity = {
        "status": "continuation_required_after_safe_checkpoint",
        "design_sha256": predecessor.design.design_sha256,
        "admission_sha256": predecessor.admission_sha256,
        "logical_run_id": predecessor.logical_run_id,
        "scheduler_segment_id": predecessor.scheduler_segment_id,
        "segment_index": predecessor.segment_index,
        "task_id": predecessor.task_id,
        "seed": predecessor.seed,
    }
    if any(outcome.get(name) != expected for name, expected in expected_outcome_identity.items()):
        raise ValueError("continuable terminal outcome differs from the predecessor run")
    terminal_bundle = _validate_operational_bundle(continuable_terminal.operational_bundle)
    authorization = predecessor.design.profile_authorization
    if (
        terminal_bundle.file_sha256 != authorization.operational_bundle_file_sha256
        or terminal_bundle.bundle_semantic_sha256
        != authorization.operational_bundle_semantic_sha256
        or terminal_bundle.profile_run_sha256 != authorization.profile_run_sha256
        or terminal_bundle.formal_profile_sha256
        != authorization.formal_cuda_profile_result.artifact_sha256
        or terminal_bundle.resource_plan_sha256 != authorization.resource_plan.artifact_sha256
    ):
        raise ValueError("continuable terminal belongs to another Gate-P operational closure")
    scheduler_terminal = continuable_terminal.artifact_ref()
    checkpoint_payload = outcome.get("continuation_checkpoint")
    verified_checkpoint = reopen_artifact_ref(checkpoint_payload)
    _validate_artifact(
        scheduler_terminal,
        name="scheduler_terminal",
        schema_version=CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
        role=CONTINUABLE_PRIMARY_TERMINAL_ROLE,
    )
    _validate_artifact(
        verified_checkpoint,
        name="verified_checkpoint",
        schema_version=VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
        role=VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    )
    payload = _continuation_evidence_payload(
        predecessor=predecessor,
        scheduler_terminal=scheduler_terminal,
        verified_checkpoint=verified_checkpoint,
    )
    result = VerifiedContinuationEvidence(
        schema_version=R3_CONTINUATION_EVIDENCE_SCHEMA,
        predecessor=predecessor,
        scheduler_terminal=scheduler_terminal,
        verified_checkpoint=verified_checkpoint,
        evidence_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def _segment_payload(
    *,
    design: R3PrimaryDesign,
    materialization: ValidatedR3Materialization,
    task_id: int,
    seed: int,
    segment_index: int,
    logical_run_id: str,
    head_run_ids: tuple[str, str],
    scheduler_segment_id: str,
    start_mode: str,
    continuation_evidence: VerifiedContinuationEvidence | None,
) -> dict[str, object]:
    return {
        "schema_version": R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA,
        "design_sha256": design.design_sha256,
        "materialization_attestation_sha256": materialization.attestation_sha256,
        "task_id": task_id,
        "seed": seed,
        "segment_index": segment_index,
        "logical_run_id": logical_run_id,
        "head_runs": [
            {"head": head, "head_run_id": head_run_id}
            for head, head_run_id in zip(R3_PRIMARY_HEADS, head_run_ids, strict=True)
        ],
        "scheduler_segment_id": scheduler_segment_id,
        "start_mode": start_mode,
        "continuation_evidence_sha256": (
            None if continuation_evidence is None else continuation_evidence.evidence_sha256
        ),
    }


@dataclass(frozen=True, slots=True)
class PrimarySegmentAdmission:
    """One scheduler segment admitted into a stable per-seed logical run."""

    schema_version: str
    design: R3PrimaryDesign = field(repr=False)
    materialization: ValidatedR3Materialization = field(repr=False)
    task_id: int
    seed: int
    segment_index: int
    logical_run_id: str
    head_run_ids: tuple[str, str]
    scheduler_segment_id: str
    start_mode: str
    continuation_evidence: VerifiedContinuationEvidence | None = field(repr=False)
    admission_sha256: str
    _factory_token: InitVar[object] = None
    _seal: object = field(repr=False, compare=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        _require_factory_token(_factory_token, name=type(self).__name__)
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        if self.schema_version != R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA:
            raise ValueError("primary segment admission schema is not frozen")
        if type(self.design) is not R3PrimaryDesign:
            raise TypeError("design must be R3PrimaryDesign")
        self.design.validate_integrity()
        materialization = _validate_materialization(self.materialization)
        task_id = _require_exact_int(self.task_id, name="task_id", minimum=0)
        seed = _require_exact_int(self.seed, name="seed", minimum=0)
        expected_task_seed = dict(R3_TASK_SEED_MAP).get(task_id)
        if expected_task_seed is None or expected_task_seed != seed:
            raise ValueError("task_id and seed must exactly match the frozen task map")
        if materialization.seed != seed:
            raise ValueError("primary materialization seed differs from the admitted seed")
        if materialization.science_semantic_sha256 != self.design.science.semantic_sha256:
            raise ValueError("primary materialization is bound to another science contract")
        segment_index = _require_exact_int(
            self.segment_index,
            name="segment_index",
            minimum=1,
        )
        if segment_index > self.design.max_scheduler_segments:
            raise ValueError("segment_index exceeds the frozen maximum")
        expected_logical_run_id = _logical_run_id(
            design_sha256=self.design.design_sha256,
            task_id=task_id,
            seed=seed,
            materialization_attestation_sha256=materialization.attestation_sha256,
        )
        _require_digest(self.logical_run_id, name="logical_run_id")
        if self.logical_run_id != expected_logical_run_id:
            raise ValueError("logical_run_id does not match design/task/seed/materialization")
        expected_head_ids = _head_run_ids(logical_run_id=self.logical_run_id)
        if (
            type(self.head_run_ids) is not tuple
            or len(self.head_run_ids) != len(R3_PRIMARY_HEADS)
            or self.head_run_ids != expected_head_ids
        ):
            raise ValueError("head_run_ids do not match the stable logical run")
        for index, digest in enumerate(self.head_run_ids):
            _require_digest(digest, name=f"head_run_ids[{index}]")
        expected_segment_id = _scheduler_segment_id(
            logical_run_id=self.logical_run_id,
            segment_index=segment_index,
        )
        _require_digest(self.scheduler_segment_id, name="scheduler_segment_id")
        if self.scheduler_segment_id != expected_segment_id:
            raise ValueError("scheduler_segment_id does not match the segment")
        if segment_index == 1:
            if self.start_mode != "fresh_zero_head_fresh_adamw":
                raise ValueError("segment 1 must start from zero head and fresh AdamW")
            if self.continuation_evidence is not None:
                raise ValueError("segment 1 cannot consume continuation evidence")
        else:
            if self.start_mode != "verified_state_complete_continuation":
                raise ValueError("later segments must use verified continuation")
            if (
                type(
                    self.continuation_evidence,
                )
                is not VerifiedContinuationEvidence
            ):
                raise TypeError("later segments require VerifiedContinuationEvidence")
            evidence = self.continuation_evidence
            evidence.validate_integrity()
            predecessor = evidence.predecessor
            if predecessor.design.design_sha256 != self.design.design_sha256:
                raise ValueError("continuation predecessor belongs to another design")
            if predecessor.logical_run_id != self.logical_run_id:
                raise ValueError("continuation predecessor belongs to another logical run")
            if (
                predecessor.task_id != task_id
                or predecessor.seed != seed
                or predecessor.materialization.attestation_sha256
                != materialization.attestation_sha256
            ):
                raise ValueError("continuation predecessor task/seed/materialization changed")
            if predecessor.segment_index + 1 != segment_index:
                raise ValueError("continuation segments must be consecutive")
        _require_digest(self.admission_sha256, name="admission_sha256")
        expected = _canonical_sha256(
            _segment_payload(
                design=self.design,
                materialization=materialization,
                task_id=task_id,
                seed=seed,
                segment_index=segment_index,
                logical_run_id=self.logical_run_id,
                head_run_ids=self.head_run_ids,
                scheduler_segment_id=self.scheduler_segment_id,
                start_mode=self.start_mode,
                continuation_evidence=self.continuation_evidence,
            )
        )
        if self.admission_sha256 != expected:
            raise ValueError("primary segment admission SHA does not match its payload")

    def validate_integrity(self) -> None:
        _require_factory_token(getattr(self, "_seal", None), name=type(self).__name__)
        self.__post_init__(_FACTORY_TOKEN)

    def to_dict(self) -> dict[str, object]:
        payload = _segment_payload(
            design=self.design,
            materialization=self.materialization,
            task_id=self.task_id,
            seed=self.seed,
            segment_index=self.segment_index,
            logical_run_id=self.logical_run_id,
            head_run_ids=self.head_run_ids,
            scheduler_segment_id=self.scheduler_segment_id,
            start_mode=self.start_mode,
            continuation_evidence=self.continuation_evidence,
        )
        payload["admission_sha256"] = self.admission_sha256
        return payload


def admit_primary_segment(
    *,
    design: R3PrimaryDesign,
    materialization_capability: R3TrainMaterializationCapability,
    task_id: int,
    seed: int,
    segment_index: int,
    continuation_evidence: VerifiedContinuationEvidence | None = None,
) -> PrimarySegmentAdmission:
    """Admit one exact task/seed segment, fresh first and verified thereafter."""

    if type(design) is not R3PrimaryDesign:
        raise TypeError("design must be R3PrimaryDesign")
    design.validate_integrity()
    capability = _validate_materialization_capability(materialization_capability)
    materialization = _validate_materialization(capability.materialization)
    task_id = _require_exact_int(task_id, name="task_id", minimum=0)
    seed = _require_exact_int(seed, name="seed", minimum=0)
    segment_index = _require_exact_int(segment_index, name="segment_index", minimum=1)
    logical_run_id = _logical_run_id(
        design_sha256=design.design_sha256,
        task_id=task_id,
        seed=seed,
        materialization_attestation_sha256=materialization.attestation_sha256,
    )
    head_run_ids = _head_run_ids(logical_run_id=logical_run_id)
    scheduler_segment_id = _scheduler_segment_id(
        logical_run_id=logical_run_id,
        segment_index=segment_index,
    )
    start_mode = (
        "fresh_zero_head_fresh_adamw"
        if segment_index == 1
        else "verified_state_complete_continuation"
    )
    payload = _segment_payload(
        design=design,
        materialization=materialization,
        task_id=task_id,
        seed=seed,
        segment_index=segment_index,
        logical_run_id=logical_run_id,
        head_run_ids=head_run_ids,
        scheduler_segment_id=scheduler_segment_id,
        start_mode=start_mode,
        continuation_evidence=continuation_evidence,
    )
    result = PrimarySegmentAdmission(
        schema_version=R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA,
        design=design,
        materialization=materialization,
        task_id=task_id,
        seed=seed,
        segment_index=segment_index,
        logical_run_id=logical_run_id,
        head_run_ids=head_run_ids,
        scheduler_segment_id=scheduler_segment_id,
        start_mode=start_mode,
        continuation_evidence=continuation_evidence,
        admission_sha256=_canonical_sha256(payload),
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def reopen_artifact_ref(value: object) -> ArtifactRef:
    """Reopen a generic reference without authorizing its caller-declared role."""

    observed = _require_closed_mapping(
        value,
        name="artifact reference",
        keys={"schema_version", "artifact_sha256", "role"},
    )
    result = ArtifactRef(
        schema_version=observed["schema_version"],  # type: ignore[arg-type]
        artifact_sha256=observed["artifact_sha256"],  # type: ignore[arg-type]
        role=observed["role"],  # type: ignore[arg-type]
    )
    result.validate_integrity()
    _require_exact_serialized_identity(
        observed,
        result.to_dict(),
        name="artifact reference",
    )
    return result


def rehydrate_artifact_ref(value: object) -> ArtifactRef:
    """Alias emphasizing that reopening still yields only a generic reference."""

    return reopen_artifact_ref(value)


def rehydrate_gate_p_admission(
    value: object,
    *,
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    science: R3ScienceConfigBundle,
) -> GatePAdmission:
    """Rebuild admission bytes only against current validator-sealed dependencies."""

    observed = _require_closed_mapping(
        value,
        name="Gate-P admission",
        keys={
            "schema_version",
            "gate0",
            "gate1",
            "source",
            "container",
            "config",
            "admission_sha256",
        },
    )
    result = create_gate_p_admission(
        gate0_capability=gate0_capability,
        gate1_capabilities=gate1_capabilities,
        science=science,
    )
    _require_exact_serialized_identity(
        observed,
        result.to_dict(),
        name="Gate-P admission",
    )
    return result


def rehydrate_validated_gate_p_run(
    value: object,
    *,
    materialization_capability: R3TrainMaterializationCapability,
    science: R3ScienceConfigBundle,
    admission: GatePAdmission,
) -> ValidatedGatePRun:
    """Rebuild a run identity against exact process-local validated dependencies."""

    observed = _require_closed_mapping(
        value,
        name="validated Gate-P run",
        keys={
            "schema_version",
            "materialization_attestation_sha256",
            "science_semantic_sha256",
            "science_file_sha256",
            "gate_p_admission_sha256",
            "seed",
            "head_order",
            "completed_updates_per_head",
            "audit_updates",
            "scheduler_segments",
            "profile_nonreusable",
            "information_boundary",
            "stop_reason",
            "profile_run_sha256",
        },
    )
    result = create_validated_gate_p_run(
        materialization_capability=materialization_capability,
        science=science,
        admission=admission,
    )
    _require_exact_serialized_identity(
        observed,
        result.to_dict(),
        name="validated Gate-P run",
    )
    return result


def rehydrate_gate_p_authorization(
    value: object,
    *,
    operational_bundle: VerifiedGatePOperationalBundle,
    successful_terminal: SuccessfulProfileTerminalCapability,
) -> GatePAuthorization:
    """Rebuild authorization only against current sealed profile evidence."""

    observed = _require_closed_mapping(
        value,
        name="Gate-P authorization",
        keys={
            "schema_version",
            "operational_bundle_file_sha256",
            "operational_bundle_semantic_sha256",
            "profile_run_sha256",
            "gate_p_admission_sha256",
            "science_semantic_sha256",
            "science_file_sha256",
            "gate0_artifact_sha256",
            "gate1_artifact_sha256",
            "source_artifact_sha256",
            "container_artifact_sha256",
            "formal_cuda_profile_result",
            "scheduler_terminal",
            "resource_plan",
            "checkpoint_policy_sha256",
            "progress_policy_sha256",
            "signal_policy_sha256",
            "continuation_policy_sha256",
            "max_scheduler_segments",
            "authorization_sha256",
        },
    )
    result = authorize_gate_p(
        operational_bundle=operational_bundle,
        successful_terminal=successful_terminal,
    )
    _require_exact_serialized_identity(
        observed,
        result.to_dict(),
        name="Gate-P authorization",
    )
    return result


def rehydrate_r3_primary_design(
    value: object,
    *,
    science: R3ScienceConfigBundle,
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    profile_authorization: GatePAuthorization,
    operational_bundle: VerifiedGatePOperationalBundle,
) -> R3PrimaryDesign:
    """Rebuild the primary design against current science and Gate-P authority."""

    observed = _require_closed_mapping(
        value,
        name="R3 primary design",
        keys={
            "schema_version",
            "campaign_kind",
            "execution_revision",
            "campaign_role",
            "science_semantic_sha256",
            "science_file_sha256",
            "gate0",
            "gate1",
            "profile_authorization_sha256",
            "source",
            "container",
            "ordered_seeds",
            "primary_heads",
            "task_seed_map",
            "resource_policy_sha256",
            "checkpoint_policy_sha256",
            "progress_policy_sha256",
            "signal_policy_sha256",
            "continuation_policy_sha256",
            "max_scheduler_segments",
            "design_sha256",
        },
    )
    result = create_r3_primary_design(
        science=science,
        gate0_capability=gate0_capability,
        gate1_capabilities=gate1_capabilities,
        profile_authorization=profile_authorization,
        operational_bundle=operational_bundle,
    )
    _require_exact_serialized_identity(
        observed,
        result.to_dict(),
        name="R3 primary design",
    )
    return result


def rehydrate_verified_continuation_evidence(
    value: object,
    *,
    predecessor: PrimarySegmentAdmission,
    continuable_terminal: ContinuablePrimaryTerminalCapability,
) -> VerifiedContinuationEvidence:
    """Rebuild continuation evidence against exact sealed predecessor evidence."""

    observed = _require_closed_mapping(
        value,
        name="verified continuation evidence",
        keys={
            "schema_version",
            "design_sha256",
            "predecessor_admission_sha256",
            "logical_run_id",
            "task_id",
            "seed",
            "predecessor_segment_index",
            "materialization_attestation_sha256",
            "scheduler_terminal",
            "verified_checkpoint",
            "evidence_sha256",
        },
    )
    result = validate_continuation_evidence(
        predecessor=predecessor,
        continuable_terminal=continuable_terminal,
    )
    _require_exact_serialized_identity(
        observed,
        result.to_dict(),
        name="verified continuation evidence",
    )
    return result


def rehydrate_primary_segment_admission(
    value: object,
    *,
    design: R3PrimaryDesign,
    materialization_capability: R3TrainMaterializationCapability,
    continuation_evidence: VerifiedContinuationEvidence | None = None,
) -> PrimarySegmentAdmission:
    """Rebuild one scheduler-segment admission without serialized tensor state."""

    observed = _require_closed_mapping(
        value,
        name="primary segment admission",
        keys={
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
        },
    )
    result = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capability,
        task_id=observed["task_id"],  # type: ignore[arg-type]
        seed=observed["seed"],  # type: ignore[arg-type]
        segment_index=observed["segment_index"],  # type: ignore[arg-type]
        continuation_evidence=continuation_evidence,
    )
    _require_exact_serialized_identity(
        observed,
        result.to_dict(),
        name="primary segment admission",
    )
    return result


__all__ = [
    "CONFIG_ARTIFACT_ROLE",
    "CONFIG_ARTIFACT_SCHEMA",
    "CONTAINER_ARTIFACT_ROLE",
    "CONTAINER_ARTIFACT_SCHEMA",
    "CONTINUABLE_PRIMARY_TERMINAL_ROLE",
    "CONTINUABLE_PRIMARY_TERMINAL_SCHEMA",
    "FORMAL_CUDA_PROFILE_RESULT_ROLE",
    "FORMAL_CUDA_PROFILE_RESULT_SCHEMA",
    "GATE0_ARTIFACT_ROLE",
    "GATE0_ARTIFACT_SCHEMA",
    "GATE1_ARTIFACT_ROLE",
    "GATE1_ARTIFACT_SCHEMA",
    "GATE_P_ADMISSION_SCHEMA",
    "GATE_P_AUTHORIZATION_SCHEMA",
    "GATE_P_RUN_SCHEMA",
    "RESOURCE_PLAN_ROLE",
    "RESOURCE_PLAN_SCHEMA",
    "R2_RECOVERY_DESIGN_SHA256",
    "R3_CONTINUATION_EVIDENCE_SCHEMA",
    "R3_PRIMARY_DESIGN_SCHEMA",
    "R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA",
    "SOURCE_ARTIFACT_ROLE",
    "SOURCE_ARTIFACT_SCHEMA",
    "SUCCESSFUL_PROFILE_TERMINAL_ROLE",
    "SUCCESSFUL_PROFILE_TERMINAL_SCHEMA",
    "VERIFIED_CONTINUATION_CHECKPOINT_ROLE",
    "VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA",
    "ArtifactRef",
    "GatePAdmission",
    "GatePAuthorization",
    "PrimarySegmentAdmission",
    "R3PrimaryDesign",
    "ValidatedGatePRun",
    "VerifiedContinuationEvidence",
    "admit_primary_segment",
    "authorize_gate_p",
    "create_gate_p_admission",
    "create_r3_primary_design",
    "create_validated_gate_p_run",
    "rehydrate_gate_p_admission",
    "rehydrate_gate_p_authorization",
    "rehydrate_artifact_ref",
    "rehydrate_primary_segment_admission",
    "rehydrate_r3_primary_design",
    "rehydrate_validated_gate_p_run",
    "rehydrate_verified_continuation_evidence",
    "reopen_artifact_ref",
    "validate_continuation_evidence",
]
