"""Pure-data transport and reopening for Gate-P operational evidence.

The live Gate-P objects intentionally retain their validated materialization so
they can protect the profiling process.  Primary jobs must not reconstruct
those tensors merely to consume an already frozen resource plan.  This module
therefore publishes one canonical, self-hashed JSON closure and reopens it into
a factory-only capability using operational data alone.

Reopening verifies the caller-supplied file SHA-256, every nested semantic
hash, all cross-links, and a fresh resource-plan projection.  It never calls a
materialization, oracle, CUDA, or trainer API.  The resulting capability is
evidence for later admission logic; it is not itself Gate-P authorization.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Final

from . import phase2_r3_profile as _profile
from .phase2_profile import (
    PHASE2_PROFILE_AUDIT_UPDATES,
    PHASE2_PROFILE_BINDING_SCHEMA,
    PHASE2_PROFILE_CAMPAIGN_KIND,
    PHASE2_PROFILE_EXECUTION_REVISION,
    PHASE2_PROFILE_LEARNER_ORDER,
    PHASE2_PROFILE_ROLE,
    PHASE2_PROFILE_STOP_REASON,
    PHASE2_PROFILE_UPDATES,
    validate_gate_p_profile_core_result,
)
from .phase2_r3_artifacts import (
    decode_canonical_json_bytes,
    publish_canonical_artifact,
    read_canonical_artifact,
)
from .phase2_r3_identity import (
    FORMAL_CUDA_PROFILE_RESULT_ROLE,
    FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
    GATE_P_AUDIT_UPDATES,
    GATE_P_HEAD_ORDER,
    GATE_P_INFORMATION_BOUNDARY,
    GATE_P_RUN_SCHEMA,
    GATE_P_SCHEDULER_SEGMENTS,
    GATE_P_SEED,
    GATE_P_STOP_REASON,
    GATE_P_UPDATES_PER_HEAD,
    RESOURCE_PLAN_ROLE,
    RESOURCE_PLAN_SCHEMA,
    ArtifactRef,
    ValidatedGatePRun,
)

VERIFIED_GATE_P_OPERATIONAL_BUNDLE_SCHEMA: Final = (
    "phase2-recovery-r3-verified-gate-p-operational-bundle/v1"
)
VERIFIED_GATE_P_OPERATIONAL_BUNDLE_ROLE: Final = (
    "verified_gate_p_operational_bundle_not_authorization"
)

_FACTORY_TOKEN = object()
_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "profile_run_identity",
        "safety_margin_policy",
        "scheduler_resource_envelope",
        "formal_cuda_profile_result",
        "resource_plan",
        "bundle_semantic_sha256",
    }
)
_PROFILE_RUN_FIELDS = frozenset(
    {
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
    }
)
_SAFETY_FIELDS = frozenset(
    {
        "schema_version",
        "profile_run_sha256",
        "declared_before_profile",
        "walltime_margin_fraction",
        "fixed_walltime_margin_seconds",
        "memory_margin_fraction",
        "signal_margin_seconds",
        "durable_checkpoint_cadence_updates",
        "mandatory_checkpoint_updates",
        "mandatory_checkpoint_roles",
        "checkpoint_on_selection",
        "checkpoint_before_head_transition",
        "checkpoint_on_signal_safe_boundary",
        "checkpoint_at_segment_terminal",
        "checkpoint_before_resume",
        "policy_sha256",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "profile_run_sha256",
        "scheduler_raw_evidence_sha256",
        "resource_raw_evidence_sha256",
        "slurm_account",
        "partition",
        "gpu_name",
        "gpu_total_memory_bytes",
        "max_allocation_wall_seconds",
        "max_array_concurrency",
        "max_scheduler_segments",
        "max_gpus_per_task",
        "max_cpus_per_task",
        "max_memory_bytes",
        "envelope_sha256",
    }
)
_PREPARATION_FIELDS = frozenset(
    {
        "schema_version",
        "profile_run_sha256",
        "artifact_verification_wall_seconds",
        "oracle_rescore_wall_seconds",
        "label_reconstruction_wall_seconds",
        "source_artifacts_reverified",
        "labels_reconstructed_from_attested_train_only_source",
        "heldout_bytes_decoded",
        "preparation_sha256",
    }
)
_FORMAL_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "identity_bindings",
        "safety_margin_policy_sha256",
        "scheduler_resource_envelope",
        "scheduler_raw_evidence_sha256",
        "resource_raw_evidence_sha256",
        "preparation",
        "materialization_revalidation_wall_seconds",
        "trainer_enter_wall_seconds",
        "wrapper_wall_seconds",
        "cuda_identity",
        "gpu_utilization_samples",
        "cpu_memory",
        "core_profile",
        "core_profile_sha256",
        "production_checkpoint_io_evidence",
        "stop_reason",
        "information_boundary",
        "formal_profile_sha256",
    }
)
_IDENTITY_BINDING_FIELDS = frozenset(
    {
        "profile_run_sha256",
        "gate_p_admission_sha256",
        "gate0_artifact_sha256",
        "gate1_artifact_sha256",
        "source_artifact_sha256",
        "container_artifact_sha256",
        "config_artifact_sha256",
        "materialization_attestation_sha256",
        "materialization_provenance_sha256",
        "context_sha256",
        "settings_sha256",
        "input_training_sha256",
        "prepared_training_sha256",
        "oracle_reward_sha256",
        "label_stream_sha256",
    }
)


def _exact_mapping(
    value: object,
    fields: frozenset[str],
    *,
    name: str,
) -> dict[str, object]:
    copied = _profile._json_copy(value, name=name)
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must be a JSON object")
    if set(copied) != fields:
        raise ValueError(f"{name} has an invalid closed field set")
    return copied


def _exact_int(value: object, *, name: str, minimum: int | None = None) -> int:
    return _profile._exact_int(value, name=name, minimum=minimum)


def _positive_real(value: object, *, name: str) -> float:
    return _profile._positive_real(value, name=name)


def _digest(value: object, *, name: str) -> str:
    return _profile._digest(value, name=name)


def _semantic_sha256(value: Mapping[str, object]) -> str:
    return _profile._canonical_sha256(dict(value))


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _validated_profile_run_identity(value: object) -> dict[str, object]:
    identity = _exact_mapping(
        value,
        _PROFILE_RUN_FIELDS,
        name="profile_run_identity",
    )
    exact = {
        "schema_version": GATE_P_RUN_SCHEMA,
        "seed": GATE_P_SEED,
        "head_order": list(GATE_P_HEAD_ORDER),
        "completed_updates_per_head": GATE_P_UPDATES_PER_HEAD,
        "audit_updates": list(GATE_P_AUDIT_UPDATES),
        "scheduler_segments": GATE_P_SCHEDULER_SEGMENTS,
        "profile_nonreusable": True,
        "information_boundary": GATE_P_INFORMATION_BOUNDARY,
        "stop_reason": GATE_P_STOP_REASON,
    }
    for name, expected in exact.items():
        if identity[name] != expected:
            raise ValueError(f"profile_run_identity {name} is not frozen")
    for name in (
        "materialization_attestation_sha256",
        "science_semantic_sha256",
        "science_file_sha256",
        "gate_p_admission_sha256",
    ):
        _digest(identity[name], name=f"profile_run_identity.{name}")
    observed = _digest(
        identity["profile_run_sha256"],
        name="profile_run_identity.profile_run_sha256",
    )
    unhashed = dict(identity)
    del unhashed["profile_run_sha256"]
    if _semantic_sha256(unhashed) != observed:
        raise ValueError("profile-run semantic SHA256 is invalid")
    return identity


def _validated_safety_policy(
    value: object,
    *,
    profile_run_sha256: str,
) -> tuple[dict[str, object], SimpleNamespace]:
    policy = _exact_mapping(
        value,
        _SAFETY_FIELDS,
        name="safety_margin_policy",
    )
    if policy["profile_run_sha256"] != profile_run_sha256:
        raise ValueError("safety policy belongs to another profile run")
    if (
        policy["declared_before_profile"] is not True
        or policy["checkpoint_on_selection"] is not True
        or policy["checkpoint_before_head_transition"] is not True
        or policy["checkpoint_on_signal_safe_boundary"] is not True
        or policy["checkpoint_at_segment_terminal"] is not True
        or policy["checkpoint_before_resume"] is not True
    ):
        raise ValueError("safety policy omitted a predeclared mandatory guard")
    wall_fraction = _positive_real(
        policy["walltime_margin_fraction"],
        name="walltime_margin_fraction",
    )
    fixed_wall = _positive_real(
        policy["fixed_walltime_margin_seconds"],
        name="fixed_walltime_margin_seconds",
    )
    memory_fraction = _positive_real(
        policy["memory_margin_fraction"],
        name="memory_margin_fraction",
    )
    signal_margin = _positive_real(
        policy["signal_margin_seconds"],
        name="signal_margin_seconds",
    )
    cadence = _exact_int(
        policy["durable_checkpoint_cadence_updates"],
        name="durable_checkpoint_cadence_updates",
        minimum=_profile.R3_AUDIT_CADENCE_UPDATES,
    )
    if (
        cadence > _profile.R3_MAXIMUM_UPDATES_PER_HEAD
        or cadence % _profile.R3_AUDIT_CADENCE_UPDATES != 0
    ):
        raise ValueError("durable checkpoint cadence must remain audit-aligned and bounded")
    expected = _profile._safety_payload(
        profile_run_sha256=profile_run_sha256,
        walltime_margin_fraction=wall_fraction,
        fixed_walltime_margin_seconds=fixed_wall,
        memory_margin_fraction=memory_fraction,
        signal_margin_seconds=signal_margin,
        durable_checkpoint_cadence_updates=cadence,
        checkpoint_on_selection=True,
        checkpoint_before_head_transition=True,
        checkpoint_on_signal_safe_boundary=True,
        checkpoint_at_segment_terminal=True,
        checkpoint_before_resume=True,
    )
    policy_sha = _digest(policy["policy_sha256"], name="policy_sha256")
    if policy != {**expected, "policy_sha256": policy_sha}:
        raise ValueError("safety policy fields differ from their frozen derivation")
    if _semantic_sha256(expected) != policy_sha:
        raise ValueError("safety policy semantic SHA256 is invalid")
    adapter_values = {
        **expected,
        "mandatory_checkpoint_updates": tuple(expected["mandatory_checkpoint_updates"]),
        "mandatory_checkpoint_roles": tuple(expected["mandatory_checkpoint_roles"]),
        "policy_sha256": policy_sha,
    }
    adapter = SimpleNamespace(**adapter_values)
    return policy, adapter


def _validated_scheduler_envelope(
    value: object,
    *,
    profile_run_sha256: str,
) -> tuple[dict[str, object], SimpleNamespace]:
    envelope = _exact_mapping(
        value,
        _ENVELOPE_FIELDS,
        name="scheduler_resource_envelope",
    )
    if envelope["profile_run_sha256"] != profile_run_sha256:
        raise ValueError("scheduler envelope belongs to another profile run")
    for name in ("scheduler_raw_evidence_sha256", "resource_raw_evidence_sha256"):
        _digest(envelope[name], name=name)
    for name in (
        "gpu_total_memory_bytes",
        "max_allocation_wall_seconds",
        "max_array_concurrency",
        "max_scheduler_segments",
        "max_gpus_per_task",
        "max_cpus_per_task",
        "max_memory_bytes",
    ):
        _exact_int(envelope[name], name=name, minimum=1)
    if type(envelope["partition"]) is not str or not str(envelope["partition"]).startswith("gpu-"):
        raise ValueError("scheduler envelope requires an explicit GPU partition")
    if type(envelope["gpu_name"]) is not str or not envelope["gpu_name"]:
        raise TypeError("scheduler envelope gpu_name must be non-empty")
    expected = _profile._envelope_payload(
        profile_run_sha256=profile_run_sha256,
        scheduler_raw_evidence_sha256=str(envelope["scheduler_raw_evidence_sha256"]),
        resource_raw_evidence_sha256=str(envelope["resource_raw_evidence_sha256"]),
        partition=str(envelope["partition"]),
        gpu_name=str(envelope["gpu_name"]),
        gpu_total_memory_bytes=int(envelope["gpu_total_memory_bytes"]),
        max_allocation_wall_seconds=int(envelope["max_allocation_wall_seconds"]),
        max_array_concurrency=int(envelope["max_array_concurrency"]),
        max_scheduler_segments=int(envelope["max_scheduler_segments"]),
        max_gpus_per_task=int(envelope["max_gpus_per_task"]),
        max_cpus_per_task=int(envelope["max_cpus_per_task"]),
        max_memory_bytes=int(envelope["max_memory_bytes"]),
    )
    envelope_sha = _digest(
        envelope["envelope_sha256"],
        name="envelope_sha256",
    )
    if envelope != {**expected, "envelope_sha256": envelope_sha}:
        raise ValueError("scheduler envelope differs from its frozen derivation")
    if _semantic_sha256(expected) != envelope_sha:
        raise ValueError("scheduler envelope semantic SHA256 is invalid")
    return envelope, SimpleNamespace(**expected, envelope_sha256=envelope_sha)


def _validated_preparation(
    value: object,
    *,
    profile_run_sha256: str,
) -> tuple[dict[str, object], SimpleNamespace]:
    preparation = _exact_mapping(
        value,
        _PREPARATION_FIELDS,
        name="formal_profile.preparation",
    )
    if preparation["profile_run_sha256"] != profile_run_sha256:
        raise ValueError("profile preparation belongs to another profile run")
    artifact_seconds = _positive_real(
        preparation["artifact_verification_wall_seconds"],
        name="artifact_verification_wall_seconds",
    )
    oracle_seconds = _positive_real(
        preparation["oracle_rescore_wall_seconds"],
        name="oracle_rescore_wall_seconds",
    )
    label_seconds = _positive_real(
        preparation["label_reconstruction_wall_seconds"],
        name="label_reconstruction_wall_seconds",
    )
    expected = _profile._preparation_payload(
        profile_run_sha256=profile_run_sha256,
        artifact_verification_wall_seconds=artifact_seconds,
        oracle_rescore_wall_seconds=oracle_seconds,
        label_reconstruction_wall_seconds=label_seconds,
    )
    preparation_sha = _digest(
        preparation["preparation_sha256"],
        name="preparation_sha256",
    )
    if preparation != {**expected, "preparation_sha256": preparation_sha}:
        raise ValueError("profile preparation differs from its frozen derivation")
    if _semantic_sha256(expected) != preparation_sha:
        raise ValueError("profile preparation semantic SHA256 is invalid")
    return preparation, SimpleNamespace(
        **expected,
        preparation_sha256=preparation_sha,
    )


def _validated_formal_profile(
    value: object,
    *,
    profile_run: Mapping[str, object],
    policy: Mapping[str, object],
    envelope: Mapping[str, object],
    envelope_adapter: SimpleNamespace,
) -> tuple[dict[str, object], SimpleNamespace]:
    formal = _exact_mapping(
        value,
        _FORMAL_FIELDS,
        name="formal_cuda_profile_result",
    )
    if (
        formal["schema_version"] != FORMAL_CUDA_PROFILE_RESULT_SCHEMA
        or formal["role"] != FORMAL_CUDA_PROFILE_RESULT_ROLE
    ):
        raise ValueError("formal CUDA profile schema or role changed")
    profile_run_sha = str(profile_run["profile_run_sha256"])
    identity = _exact_mapping(
        formal["identity_bindings"],
        _IDENTITY_BINDING_FIELDS,
        name="formal_profile.identity_bindings",
    )
    for name, content in identity.items():
        _digest(content, name=f"identity_bindings.{name}")
    exact_identity_links = {
        "profile_run_sha256": profile_run_sha,
        "gate_p_admission_sha256": profile_run["gate_p_admission_sha256"],
        "materialization_attestation_sha256": profile_run["materialization_attestation_sha256"],
        "config_artifact_sha256": profile_run["science_file_sha256"],
    }
    for name, expected in exact_identity_links.items():
        if identity[name] != expected:
            raise ValueError(f"formal identity binding {name} is inconsistent")
    if formal["safety_margin_policy_sha256"] != policy["policy_sha256"]:
        raise ValueError("formal profile binds another safety policy")
    if formal["scheduler_resource_envelope"] != envelope:
        raise ValueError("formal profile embeds another scheduler envelope")
    for name in ("scheduler_raw_evidence_sha256", "resource_raw_evidence_sha256"):
        if formal[name] != envelope[name]:
            raise ValueError(f"formal profile {name} cross-link is inconsistent")

    preparation, preparation_adapter = _validated_preparation(
        formal["preparation"],
        profile_run_sha256=profile_run_sha,
    )
    core = _profile._json_copy(formal["core_profile"], name="core_profile")
    if not isinstance(core, dict):
        raise TypeError("formal core profile must be a JSON object")
    validate_gate_p_profile_core_result(core)
    core_links = {
        "seed": profile_run["seed"],
        "context_sha256": identity["context_sha256"],
        "settings_sha256": identity["settings_sha256"],
        "input_training_sha256": identity["input_training_sha256"],
        "learner_order": list(PHASE2_PROFILE_LEARNER_ORDER),
        "update_cap_per_learner": PHASE2_PROFILE_UPDATES,
        "audit_update_indices": list(PHASE2_PROFILE_AUDIT_UPDATES),
        "stop_reason": PHASE2_PROFILE_STOP_REASON,
        "device_type": "cuda",
        "formal_cuda_profile": True,
        "profile_nonreusable": True,
    }
    for name, expected in core_links.items():
        if core[name] != expected:
            raise ValueError(f"formal core profile {name} cross-link is invalid")
    binding = {
        "schema_version": PHASE2_PROFILE_BINDING_SCHEMA,
        "campaign_kind": PHASE2_PROFILE_CAMPAIGN_KIND,
        "execution_revision": PHASE2_PROFILE_EXECUTION_REVISION,
        "role": PHASE2_PROFILE_ROLE,
        "profile_nonreusable": True,
        "seed": profile_run["seed"],
        "context_sha256": identity["context_sha256"],
        "settings_sha256": identity["settings_sha256"],
        "input_training_sha256": identity["input_training_sha256"],
        "learner_order": list(PHASE2_PROFILE_LEARNER_ORDER),
        "update_cap_per_learner": PHASE2_PROFILE_UPDATES,
        "audit_update_indices": list(PHASE2_PROFILE_AUDIT_UPDATES),
        "stop_reason": PHASE2_PROFILE_STOP_REASON,
    }
    if core["binding_sha256"] != _semantic_sha256(binding):
        raise ValueError("formal core profile binding SHA256 is invalid")
    if formal["core_profile_sha256"] != core["profile_sha256"]:
        raise ValueError("formal core-profile hash cross-link is invalid")
    production_io = _profile._validate_production_checkpoint_io_evidence(
        formal["production_checkpoint_io_evidence"],
        core_profile=core,
    )
    cuda_identity = _profile._validate_cuda_identity(
        formal["cuda_identity"],
        envelope=envelope_adapter,
    )
    gpu_samples = _profile._validate_gpu_samples(
        formal["gpu_utilization_samples"],
        cuda_identity=cuda_identity,
    )
    cpu_memory = _profile._validate_cpu_memory(formal["cpu_memory"])
    revalidation_seconds = _positive_real(
        formal["materialization_revalidation_wall_seconds"],
        name="materialization_revalidation_wall_seconds",
    )
    wrapper_seconds = _positive_real(
        formal["wrapper_wall_seconds"],
        name="wrapper_wall_seconds",
    )
    minimum_wrapper = math.fsum(
        (
            revalidation_seconds,
            float(core["setup"]["wall_seconds"]),
            *(float(learner["phase_wall_seconds"]) for learner in core["learners"]),
        )
    )
    if wrapper_seconds < minimum_wrapper:
        raise ValueError("formal wrapper time omits a sequential measured component")
    expected_enter = {
        str(learner["learner"]): learner["build_wall_seconds"] for learner in core["learners"]
    }
    if formal["trainer_enter_wall_seconds"] != expected_enter:
        raise ValueError("formal trainer-enter timing derivation is invalid")
    if (
        formal["stop_reason"] != PHASE2_PROFILE_STOP_REASON
        or formal["information_boundary"] != _profile.FORMAL_PROFILE_INFORMATION_BOUNDARY
    ):
        raise ValueError("formal profile operational boundary changed")
    formal_sha = _digest(
        formal["formal_profile_sha256"],
        name="formal_profile_sha256",
    )
    unhashed = dict(formal)
    del unhashed["formal_profile_sha256"]
    if _semantic_sha256(unhashed) != formal_sha:
        raise ValueError("formal profile semantic SHA256 is invalid")
    _profile._assert_no_sensitive_state(unhashed, path="sealed_formal_profile")
    adapter = SimpleNamespace(
        core_profile=core,
        production_checkpoint_io_evidence=production_io,
        preparation=preparation_adapter,
        cpu_memory=cpu_memory,
        materialization_revalidation_wall_seconds=revalidation_seconds,
        formal_profile_sha256=formal_sha,
        profile_run=SimpleNamespace(profile_run_sha256=profile_run_sha),
        envelope=envelope_adapter,
    )
    # Keep these validations live so malformed pure data cannot pass merely by
    # carrying a correct outer digest copied from another bundle.
    if (
        preparation != formal["preparation"]
        or list(gpu_samples) != formal["gpu_utilization_samples"]
    ):
        raise ValueError("formal profile nested normalization changed its payload")
    return formal, adapter


def _validated_resource_plan(
    value: object,
    *,
    formal: Mapping[str, object],
    formal_adapter: SimpleNamespace,
    profile_run_sha256: str,
    policy_adapter: SimpleNamespace,
    envelope_adapter: SimpleNamespace,
) -> dict[str, object]:
    plan = _profile._json_copy(value, name="resource_plan")
    if not isinstance(plan, dict):
        raise TypeError("resource_plan must be a JSON object")
    resource_sha = _digest(
        plan.get("resource_plan_sha256"),
        name="resource_plan_sha256",
    )
    unhashed = dict(plan)
    del unhashed["resource_plan_sha256"]
    required_inputs = (
        "requested_walltime_seconds_per_segment",
        "array_concurrency",
        "cpus_per_task",
        "memory_bytes",
    )
    for name in required_inputs:
        _exact_int(unhashed.get(name), name=name, minimum=1)
    expected = _profile._resource_plan_payload(
        result=formal_adapter,
        policy=policy_adapter,
        envelope=envelope_adapter,
        requested_walltime_seconds=int(unhashed["requested_walltime_seconds_per_segment"]),
        array_concurrency=int(unhashed["array_concurrency"]),
        cpus_per_task=int(unhashed["cpus_per_task"]),
        memory_bytes=int(unhashed["memory_bytes"]),
    )
    if unhashed != expected:
        raise ValueError(
            "resource plan differs from the pure-data projection and segment derivation"
        )
    if _semantic_sha256(expected) != resource_sha:
        raise ValueError("resource-plan semantic SHA256 is invalid")
    exact_links = {
        "schema_version": RESOURCE_PLAN_SCHEMA,
        "role": RESOURCE_PLAN_ROLE,
        "formal_profile_sha256": formal["formal_profile_sha256"],
        "profile_run_sha256": profile_run_sha256,
        "safety_margin_policy_sha256": policy_adapter.policy_sha256,
        "scheduler_resource_envelope_sha256": (envelope_adapter.envelope_sha256),
    }
    for name, expected_value in exact_links.items():
        if plan[name] != expected_value:
            raise ValueError(f"resource-plan {name} cross-link is invalid")
    _profile._assert_no_sensitive_state(unhashed, path="sealed_resource_plan")
    return plan


def _validated_bundle_payload(value: object) -> dict[str, object]:
    bundle = _exact_mapping(
        value,
        _BUNDLE_FIELDS,
        name="gate_p_operational_bundle",
    )
    if (
        bundle["schema_version"] != VERIFIED_GATE_P_OPERATIONAL_BUNDLE_SCHEMA
        or bundle["role"] != VERIFIED_GATE_P_OPERATIONAL_BUNDLE_ROLE
    ):
        raise ValueError("Gate-P operational bundle schema or role changed")
    profile_run = _validated_profile_run_identity(bundle["profile_run_identity"])
    profile_run_sha = str(profile_run["profile_run_sha256"])
    policy, policy_adapter = _validated_safety_policy(
        bundle["safety_margin_policy"],
        profile_run_sha256=profile_run_sha,
    )
    envelope, envelope_adapter = _validated_scheduler_envelope(
        bundle["scheduler_resource_envelope"],
        profile_run_sha256=profile_run_sha,
    )
    formal, formal_adapter = _validated_formal_profile(
        bundle["formal_cuda_profile_result"],
        profile_run=profile_run,
        policy=policy,
        envelope=envelope,
        envelope_adapter=envelope_adapter,
    )
    _validated_resource_plan(
        bundle["resource_plan"],
        formal=formal,
        formal_adapter=formal_adapter,
        profile_run_sha256=profile_run_sha,
        policy_adapter=policy_adapter,
        envelope_adapter=envelope_adapter,
    )
    semantic_sha = _digest(
        bundle["bundle_semantic_sha256"],
        name="bundle_semantic_sha256",
    )
    unhashed = dict(bundle)
    del unhashed["bundle_semantic_sha256"]
    if _semantic_sha256(unhashed) != semantic_sha:
        raise ValueError("Gate-P operational bundle semantic SHA256 is invalid")
    return bundle


def _bundle_payload(
    *,
    profile_run: ValidatedGatePRun,
    safety_policy: _profile.ProfileSafetyMarginPolicy,
    envelope: _profile.SchedulerResourceEnvelope,
    formal_result: _profile.FormalCudaProfileResult,
    resource_plan: _profile.GatePResourcePlan,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": VERIFIED_GATE_P_OPERATIONAL_BUNDLE_SCHEMA,
        "role": VERIFIED_GATE_P_OPERATIONAL_BUNDLE_ROLE,
        "profile_run_identity": profile_run.to_dict(),
        "safety_margin_policy": safety_policy.to_dict(),
        "scheduler_resource_envelope": envelope.to_dict(),
        "formal_cuda_profile_result": formal_result.to_dict(),
        "resource_plan": resource_plan.to_dict(),
    }
    return {**body, "bundle_semantic_sha256": _semantic_sha256(body)}


@dataclass(frozen=True, slots=True)
class VerifiedGatePOperationalBundle:
    """Factory-only, tensor-free operational capability for primary consumers."""

    artifact_path: Path
    file_sha256: str
    size_bytes: int
    profile_run_sha256: str
    formal_profile_sha256: str
    resource_plan_sha256: str
    bundle_semantic_sha256: str
    _canonical_bytes: bytes = field(repr=False, compare=False)
    _factory_token: InitVar[object]
    _seal: object = field(init=False, repr=False, compare=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise TypeError("VerifiedGatePOperationalBundle must be produced by reopen")
        object.__setattr__(self, "_seal", _FACTORY_TOKEN)
        self._validate_sealed_payload()

    def _validate_sealed_payload(self) -> None:
        if self._seal is not _FACTORY_TOKEN:
            raise TypeError("operational bundle factory seal is invalid")
        if not isinstance(self.artifact_path, Path) or not self.artifact_path.is_absolute():
            raise ValueError("operational bundle path must be an absolute Path")
        for name in (
            "file_sha256",
            "profile_run_sha256",
            "formal_profile_sha256",
            "resource_plan_sha256",
            "bundle_semantic_sha256",
        ):
            _digest(getattr(self, name), name=name)
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("operational bundle size must be positive")
        if type(self._canonical_bytes) is not bytes:
            raise TypeError("operational bundle bytes must be exact bytes")
        if len(self._canonical_bytes) != self.size_bytes:
            raise ValueError("operational bundle byte count is inconsistent")
        if hashlib.sha256(self._canonical_bytes).hexdigest() != self.file_sha256:
            raise ValueError("operational bundle file SHA256 is inconsistent")
        payload = _validated_bundle_payload(decode_canonical_json_bytes(self._canonical_bytes))
        exact = {
            "profile_run_sha256": payload["profile_run_identity"]["profile_run_sha256"],
            "formal_profile_sha256": payload["formal_cuda_profile_result"]["formal_profile_sha256"],
            "resource_plan_sha256": payload["resource_plan"]["resource_plan_sha256"],
            "bundle_semantic_sha256": payload["bundle_semantic_sha256"],
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"operational bundle {name} is inconsistent")

    def _decoded(self) -> dict[str, object]:
        return decode_canonical_json_bytes(self._canonical_bytes)

    @property
    def profile_run_identity(self) -> Mapping[str, object]:
        return _deep_freeze(self._decoded()["profile_run_identity"])  # type: ignore[return-value]

    @property
    def safety_margin_policy(self) -> Mapping[str, object]:
        return _deep_freeze(self._decoded()["safety_margin_policy"])  # type: ignore[return-value]

    @property
    def scheduler_resource_envelope(self) -> Mapping[str, object]:
        return _deep_freeze(  # type: ignore[return-value]
            self._decoded()["scheduler_resource_envelope"]
        )

    @property
    def formal_cuda_profile_result(self) -> Mapping[str, object]:
        return _deep_freeze(  # type: ignore[return-value]
            self._decoded()["formal_cuda_profile_result"]
        )

    @property
    def resource_plan(self) -> Mapping[str, object]:
        return _deep_freeze(self._decoded()["resource_plan"])  # type: ignore[return-value]

    @property
    def slurm_account(self) -> str:
        return str(self.resource_plan["slurm_account"])

    @property
    def partition(self) -> str:
        return str(self.resource_plan["partition"])

    @property
    def gpu_name(self) -> str:
        return str(self.resource_plan["gpu_name"])

    @property
    def gpus_per_task(self) -> int:
        return int(self.resource_plan["gpus_per_task"])

    @property
    def cpus_per_task(self) -> int:
        return int(self.resource_plan["cpus_per_task"])

    @property
    def memory_bytes(self) -> int:
        return int(self.resource_plan["memory_bytes"])

    @property
    def requested_walltime_seconds_per_segment(self) -> int:
        return int(self.resource_plan["requested_walltime_seconds_per_segment"])

    @property
    def advance_signal_lead_seconds(self) -> int:
        return int(self.resource_plan["advance_signal_lead_seconds"])

    @property
    def audit_cadence_updates(self) -> int:
        return int(self.resource_plan["audit_cadence_updates"])

    @property
    def durable_checkpoint_cadence_updates(self) -> int:
        return int(self.resource_plan["durable_checkpoint_cadence_updates"])

    @property
    def mandatory_checkpoint_updates(self) -> tuple[int, ...]:
        value = self.resource_plan["mandatory_checkpoint_updates"]
        if not isinstance(value, Sequence):
            raise TypeError("sealed mandatory checkpoint updates are invalid")
        return tuple(int(item) for item in value)

    @property
    def mandatory_checkpoint_roles(self) -> tuple[str, ...]:
        value = self.resource_plan["mandatory_checkpoint_roles"]
        if not isinstance(value, Sequence):
            raise TypeError("sealed mandatory checkpoint roles are invalid")
        return tuple(str(item) for item in value)

    @property
    def checkpoint_on_selection(self) -> bool:
        return self.resource_plan["checkpoint_on_selection"] is True

    @property
    def checkpoint_before_head_transition(self) -> bool:
        return self.resource_plan["checkpoint_before_head_transition"] is True

    @property
    def checkpoint_on_signal_safe_boundary(self) -> bool:
        return self.resource_plan["checkpoint_on_signal_safe_boundary"] is True

    @property
    def checkpoint_at_segment_terminal(self) -> bool:
        return self.resource_plan["checkpoint_at_segment_terminal"] is True

    @property
    def checkpoint_before_resume(self) -> bool:
        return self.resource_plan["checkpoint_before_resume"] is True

    @property
    def max_scheduler_segments(self) -> int:
        return int(self.resource_plan["max_scheduler_segments"])

    @property
    def segment_boundaries(self) -> tuple[Mapping[str, object], ...]:
        value = self.resource_plan["segment_boundaries"]
        if not isinstance(value, tuple):
            raise TypeError("sealed segment boundaries are invalid")
        return value  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        return self._decoded()

    def validate_integrity(self) -> None:
        self._validate_sealed_payload()
        live = read_canonical_artifact(
            self.artifact_path,
            expected_file_sha256=self.file_sha256,
        )
        if live.canonical_bytes != self._canonical_bytes:
            raise ValueError("live operational bundle differs from sealed bytes")


def reopen_verified_gate_p_operational_bundle(
    artifact_path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
) -> VerifiedGatePOperationalBundle:
    """Reopen a canonical operational closure without training materialization."""

    transport = read_canonical_artifact(
        artifact_path,
        expected_file_sha256=expected_file_sha256,
    )
    payload = _validated_bundle_payload(transport.payload)
    result = VerifiedGatePOperationalBundle(
        artifact_path=transport.artifact_path,
        file_sha256=transport.file_sha256,
        size_bytes=transport.size_bytes,
        profile_run_sha256=str(payload["profile_run_identity"]["profile_run_sha256"]),
        formal_profile_sha256=str(payload["formal_cuda_profile_result"]["formal_profile_sha256"]),
        resource_plan_sha256=str(payload["resource_plan"]["resource_plan_sha256"]),
        bundle_semantic_sha256=str(payload["bundle_semantic_sha256"]),
        _canonical_bytes=transport.canonical_bytes,
        _factory_token=_FACTORY_TOKEN,
    )
    result.validate_integrity()
    return result


def publish_verified_gate_p_operational_bundle(
    artifact_path: str | os.PathLike[str],
    *,
    profile_run: ValidatedGatePRun,
    safety_policy: _profile.ProfileSafetyMarginPolicy,
    envelope: _profile.SchedulerResourceEnvelope,
    formal_result: _profile.FormalCudaProfileResult,
    resource_plan: _profile.GatePResourcePlan,
) -> VerifiedGatePOperationalBundle:
    """Publish and reopen one complete Gate-P operational closure."""

    if type(profile_run) is not ValidatedGatePRun:
        raise TypeError("profile_run must be exactly ValidatedGatePRun")
    profile_run.validate_integrity()
    if type(safety_policy) is not _profile.ProfileSafetyMarginPolicy:
        raise TypeError("safety_policy must be exactly ProfileSafetyMarginPolicy")
    if type(envelope) is not _profile.SchedulerResourceEnvelope:
        raise TypeError("envelope must be exactly SchedulerResourceEnvelope")
    if type(formal_result) is not _profile.FormalCudaProfileResult:
        raise TypeError("formal_result must be exactly FormalCudaProfileResult")
    if type(resource_plan) is not _profile.GatePResourcePlan:
        raise TypeError("resource_plan must be exactly GatePResourcePlan")
    safety_policy.validate_integrity()
    envelope.validate_integrity()
    formal_result.validate_integrity()
    resource_plan.validate_integrity()
    if (
        safety_policy.profile_run_sha256 != profile_run.profile_run_sha256
        or envelope.profile_run_sha256 != profile_run.profile_run_sha256
        or formal_result.profile_run.profile_run_sha256 != profile_run.profile_run_sha256
        or resource_plan.profile_run_sha256 != profile_run.profile_run_sha256
        or resource_plan.formal_profile_sha256 != formal_result.formal_profile_sha256
    ):
        raise ValueError("operational bundle dependencies do not form one closure")
    payload = _bundle_payload(
        profile_run=profile_run,
        safety_policy=safety_policy,
        envelope=envelope,
        formal_result=formal_result,
        resource_plan=resource_plan,
    )
    transport = publish_canonical_artifact(artifact_path, payload)
    return reopen_verified_gate_p_operational_bundle(
        transport.artifact_path,
        expected_file_sha256=transport.file_sha256,
    )


def formal_profile_artifact_ref(
    bundle: VerifiedGatePOperationalBundle,
) -> ArtifactRef:
    """Derive a formal-profile reference only from a reopened sealed bundle."""

    if type(bundle) is not VerifiedGatePOperationalBundle:
        raise TypeError("formal profile reference requires VerifiedGatePOperationalBundle")
    bundle.validate_integrity()
    return ArtifactRef(
        schema_version=FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
        artifact_sha256=bundle.formal_profile_sha256,
        role=FORMAL_CUDA_PROFILE_RESULT_ROLE,
    )


def resource_plan_artifact_ref(
    bundle: VerifiedGatePOperationalBundle,
) -> ArtifactRef:
    """Derive a resource-plan reference only from a reopened sealed bundle."""

    if type(bundle) is not VerifiedGatePOperationalBundle:
        raise TypeError("resource plan reference requires VerifiedGatePOperationalBundle")
    bundle.validate_integrity()
    return ArtifactRef(
        schema_version=RESOURCE_PLAN_SCHEMA,
        artifact_sha256=bundle.resource_plan_sha256,
        role=RESOURCE_PLAN_ROLE,
    )


__all__ = [
    "VERIFIED_GATE_P_OPERATIONAL_BUNDLE_ROLE",
    "VERIFIED_GATE_P_OPERATIONAL_BUNDLE_SCHEMA",
    "VerifiedGatePOperationalBundle",
    "formal_profile_artifact_ref",
    "publish_verified_gate_p_operational_bundle",
    "reopen_verified_gate_p_operational_bundle",
    "resource_plan_artifact_ref",
]
