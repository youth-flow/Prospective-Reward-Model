from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from smart_reward import phase2_r3_gate0 as gate0_module
from smart_reward import phase2_r3_gate1 as gate1_module
from smart_reward import phase2_r3_identity as identity
from smart_reward import phase2_r3_inputs as inputs_module
from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_primary import prepare_neutral_phase2_context
from smart_reward.phase2_r3_artifacts import (
    publish_canonical_artifact,
    read_canonical_artifact,
)
from smart_reward.phase2_r3_config import (
    R3ScienceConfigBundle,
    load_r3_science_config,
)
from smart_reward.phase2_r3_gate0 import R3Gate0Capability
from smart_reward.phase2_r3_gate1 import R3Gate1Capabilities
from smart_reward.phase2_r3_identity import (
    CONFIG_ARTIFACT_ROLE,
    CONFIG_ARTIFACT_SCHEMA,
    CONTAINER_ARTIFACT_ROLE,
    CONTAINER_ARTIFACT_SCHEMA,
    CONTINUABLE_PRIMARY_TERMINAL_ROLE,
    CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
    FORMAL_CUDA_PROFILE_RESULT_ROLE,
    FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
    GATE0_ARTIFACT_ROLE,
    GATE0_ARTIFACT_SCHEMA,
    GATE1_ARTIFACT_ROLE,
    GATE1_ARTIFACT_SCHEMA,
    R2_RECOVERY_DESIGN_SHA256,
    R3_TASK_SEED_MAP,
    RESOURCE_PLAN_ROLE,
    RESOURCE_PLAN_SCHEMA,
    SOURCE_ARTIFACT_ROLE,
    SOURCE_ARTIFACT_SCHEMA,
    SUCCESSFUL_PROFILE_TERMINAL_ROLE,
    SUCCESSFUL_PROFILE_TERMINAL_SCHEMA,
    ArtifactRef,
    GatePAdmission,
    GatePAuthorization,
    R3PrimaryDesign,
    ValidatedGatePRun,
    admit_primary_segment,
    authorize_gate_p,
    create_gate_p_admission,
    create_r3_primary_design,
    create_validated_gate_p_run,
    rehydrate_gate_p_admission,
    rehydrate_gate_p_authorization,
    rehydrate_primary_segment_admission,
    rehydrate_r3_primary_design,
    rehydrate_validated_gate_p_run,
    rehydrate_verified_continuation_evidence,
    reopen_artifact_ref,
    validate_continuation_evidence,
)
from smart_reward.phase2_r3_inputs import R3TrainMaterializationCapability
from smart_reward.phase2_r3_materialization import (
    TrainMaterializationProvenance,
    ValidatedR3Materialization,
    validate_r3_materialization,
)
from smart_reward.phase2_r3_profile_artifacts import (
    VerifiedGatePOperationalBundle,
)
from smart_reward.phase2_r3_terminal import (
    ContinuablePrimaryTerminalCapability,
    SuccessfulProfileTerminalCapability,
)
from tests import conftest as r3_test_support
from tests import test_phase2_r3_profile as profile_test_support

ROOT = Path(__file__).resolve().parents[1]
SCIENCE_PATH = ROOT / "configs" / "phase2_recovery_r3_science.yaml"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _ref(schema: str, role: str, token: str) -> ArtifactRef:
    return ArtifactRef(
        schema_version=schema,
        artifact_sha256=_digest(token),
        role=role,
    )


def _training() -> TrainingTensorData:
    num_prompts, num_candidates = 4, 4
    node = torch.arange(
        num_prompts * num_candidates,
        dtype=torch.float32,
    ).reshape(num_prompts, num_candidates)
    return TrainingTensorData(
        prompt_ids=tuple(f"r3-identity-{index}" for index in range(num_prompts)),
        policy_scores=torch.stack(
            [
                torch.sin((coordinate + 1.0) * 0.11 * node) + 0.1 * coordinate
                for coordinate in range(3)
            ],
            dim=-1,
        ),
        reward_features=torch.stack(
            [
                torch.cos((coordinate + 1.0) * 0.13 * node) - 0.05 * coordinate
                for coordinate in range(2)
            ],
            dim=-1,
        ),
        h=torch.linspace(-0.4, 0.3, num_prompts),
        left_wins=torch.tensor([1, 2, 1, 3], dtype=torch.int64),
        num_annotations=torch.tensor([3, 4, 2, 5], dtype=torch.int64),
    )


def _materialization(
    science: R3ScienceConfigBundle,
    *,
    seed: int,
) -> ValidatedR3Materialization:
    training = _training()
    node = torch.arange(
        training.num_prompts * training.num_candidates,
        dtype=training.policy_scores.dtype,
    ).reshape(training.num_prompts, training.num_candidates)
    context = prepare_neutral_phase2_context(
        training,
        0.2 * torch.sin(0.3 * node),
        seed=seed,
        settings=science.settings,
    )
    provenance = TrainMaterializationProvenance.from_context(
        context,
        parent_artifact_registry_sha256=_digest(f"parent-{seed}"),
        artifact_metadata_sha256=_digest(f"metadata-{seed}"),
        artifact_tensors_sha256=_digest(f"tensors-{seed}"),
        artifact_candidates_sha256=_digest(f"candidates-{seed}"),
        artifact_materialization_sha256=_digest(f"materialization-{seed}"),
        artifact_verification_sha256=_digest(f"verification-{seed}"),
        source_run_manifest_sha256=_digest(f"manifest-{seed}"),
        source_producer_identity_sha256=_digest(f"producer-{seed}"),
        candidate_train_prefix_sha256=_digest(f"candidate-prefix-{seed}"),
        candidate_train_prefix_count=(
            context.training.num_prompts * context.training.num_candidates
        ),
    )
    return validate_r3_materialization(
        context,
        science_bundle=science,
        provenance=provenance,
    )


def _sealed_gate0_capability() -> R3Gate0Capability:
    """White-box producer fixture; ordinary callers cannot supply the token."""

    return R3Gate0Capability(
        schema_version=gate0_module.GATE0_ARTIFACT_SCHEMA,
        role=gate0_module.GATE0_ARTIFACT_ROLE,
        artifact_sha256=_digest("gate0"),
        file_sha256=_digest("gate0-file"),
        production_relative=gate0_module._GATE0_RELATIVE.as_posix(),
        _factory_token=gate0_module._FACTORY_TOKEN,
    )


def _sealed_gate1_capabilities() -> R3Gate1Capabilities:
    """White-box equivalent of one internally consistent live Gate-1 issuance."""

    source_payload = gate1_module._source_capability_payload(
        artifact_sha256=_digest("source"),
        commit="1" * 40,
        tree="2" * 40,
        inventory_sha256=_digest("inventory"),
        formal_path_count=12,
    )
    source = gate1_module.R3SourceCapability(
        **source_payload,
        capability_sha256=gate1_module._canonical_sha256(source_payload),
        _factory_token=gate1_module._SOURCE_FACTORY_TOKEN,
    )
    container_payload = gate1_module._container_capability_payload(
        artifact_sha256=_digest("container"),
        canonical_sif_path=str((ROOT / "fixture.sif").resolve()),
        sif_sha256=_digest("sif"),
        sif_size_bytes=1,
        definition_git_blob_sha256=_digest("definition"),
        requirements_lock_git_blob_sha256=_digest("requirements"),
        runtime_probe_sha256=_digest("runtime"),
        live_runtime_probe_sha256=_digest("runtime"),
    )
    container = gate1_module.R3ContainerCapability(
        **container_payload,
        capability_sha256=gate1_module._canonical_sha256(container_payload),
        _factory_token=gate1_module._CONTAINER_FACTORY_TOKEN,
    )
    gate1_payload = gate1_module._gate1_capability_payload(
        artifact_sha256=_digest("gate1"),
        file_sha256=_digest("gate1-file"),
        source_artifact_sha256=source.artifact_sha256,
        container_artifact_sha256=container.artifact_sha256,
        source_test_receipt_artifact_sha256=_digest("source-test-receipt-artifact"),
        source_test_receipt_file_sha256=_digest("source-test-receipt-file"),
        verification_suite_sha256=_digest("suite"),
        live_reverification_sha256=_digest("live-reverification"),
        production_relative=gate1_module._GATE1_RELATIVE.as_posix(),
    )
    gate1 = gate1_module.R3Gate1Capability(
        **gate1_payload,
        capability_sha256=gate1_module._canonical_sha256(gate1_payload),
        _factory_token=gate1_module._GATE1_FACTORY_TOKEN,
    )
    return R3Gate1Capabilities(gate1=gate1, source=source, container=container)


def _sealed_materialization_capability(
    materialization: ValidatedR3Materialization,
) -> R3TrainMaterializationCapability:
    """White-box use of the real input validator's private issuer."""

    provenance = materialization.provenance
    return inputs_module._issue_train_materialization_capability(
        materialization=materialization,
        source_config_hash=materialization.science_bundle.settings.source_config_hash,
        parent_registry_file_sha256=provenance.parent_artifact_registry_sha256,
        parent_seed_entry_sha256=_digest(f"seed-entry-{materialization.seed}"),
        artifact_metadata_sha256=provenance.artifact_metadata_sha256,
        artifact_tensors_sha256=provenance.artifact_tensors_sha256,
        artifact_candidates_sha256=provenance.artifact_candidates_sha256,
        candidate_train_prefix_sha256=provenance.candidate_train_prefix_sha256,
        artifact_materialization_sha256=provenance.artifact_materialization_sha256,
        artifact_verification_sha256=provenance.artifact_verification_sha256,
        source_run_manifest_sha256=provenance.source_run_manifest_sha256,
        source_producer_identity_sha256=provenance.source_producer_identity_sha256,
        oracle_chat_template_sha256=_digest("oracle-chat-template"),
        oracle_transform_sha256=_digest("oracle-transform"),
    )


@pytest.fixture(scope="module")
def science() -> R3ScienceConfigBundle:
    return load_r3_science_config(SCIENCE_PATH)


@pytest.fixture(scope="module")
def materializations(
    science: R3ScienceConfigBundle,
) -> dict[int, ValidatedR3Materialization]:
    return {seed: _materialization(science, seed=seed) for _, seed in R3_TASK_SEED_MAP}


@pytest.fixture(scope="module")
def materialization_capabilities(
    materializations: dict[int, ValidatedR3Materialization],
) -> dict[int, R3TrainMaterializationCapability]:
    return {
        seed: _sealed_materialization_capability(materialization)
        for seed, materialization in materializations.items()
    }


@pytest.fixture(scope="module")
def gate0_capability() -> R3Gate0Capability:
    return _sealed_gate0_capability()


@pytest.fixture(scope="module")
def gate1_capabilities() -> R3Gate1Capabilities:
    return _sealed_gate1_capabilities()


@pytest.fixture(scope="module")
def refs(science: R3ScienceConfigBundle) -> dict[str, ArtifactRef]:
    return {
        "gate0": _ref(GATE0_ARTIFACT_SCHEMA, GATE0_ARTIFACT_ROLE, "gate0"),
        "gate1": _ref(GATE1_ARTIFACT_SCHEMA, GATE1_ARTIFACT_ROLE, "gate1"),
        "source": _ref(SOURCE_ARTIFACT_SCHEMA, SOURCE_ARTIFACT_ROLE, "source"),
        "container": _ref(
            CONTAINER_ARTIFACT_SCHEMA,
            CONTAINER_ARTIFACT_ROLE,
            "container",
        ),
        "config": ArtifactRef(
            schema_version=CONFIG_ARTIFACT_SCHEMA,
            artifact_sha256=science.file_sha256,
            role=CONFIG_ARTIFACT_ROLE,
        ),
        "formal_profile": _ref(
            FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
            FORMAL_CUDA_PROFILE_RESULT_ROLE,
            "formal-cuda-profile",
        ),
        "profile_terminal": _ref(
            SUCCESSFUL_PROFILE_TERMINAL_SCHEMA,
            SUCCESSFUL_PROFILE_TERMINAL_ROLE,
            "profile-completed-0-0",
        ),
        "resource": _ref(
            RESOURCE_PLAN_SCHEMA,
            RESOURCE_PLAN_ROLE,
            "resource-plan",
        ),
    }


@pytest.fixture(scope="module")
def gate_p_admission(
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    science: R3ScienceConfigBundle,
) -> GatePAdmission:
    return create_gate_p_admission(
        gate0_capability=gate0_capability,
        gate1_capabilities=gate1_capabilities,
        science=science,
    )


@pytest.fixture(scope="module")
def profile_run(
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    science: R3ScienceConfigBundle,
    gate_p_admission: GatePAdmission,
) -> ValidatedGatePRun:
    return create_validated_gate_p_run(
        materialization_capability=materialization_capabilities[20260801],
        science=science,
        admission=gate_p_admission,
    )


@pytest.fixture(scope="module")
def operational_bundle(
    profile_run: ValidatedGatePRun,
    tmp_path_factory: pytest.TempPathFactory,
) -> VerifiedGatePOperationalBundle:
    policy = profile_test_support.safety_policy.__wrapped__(profile_run)
    envelope = profile_test_support.envelope.__wrapped__(profile_run)
    preparation = profile_test_support.preparation.__wrapped__(profile_run)
    bundle, _, _ = profile_test_support.sealed_operational_bundle.__wrapped__(
        profile_run,
        policy,
        envelope,
        preparation,
        tmp_path_factory,
    )
    return bundle


@pytest.fixture(scope="module")
def successful_profile_terminal(
    operational_bundle: VerifiedGatePOperationalBundle,
    tmp_path_factory: pytest.TempPathFactory,
) -> SuccessfulProfileTerminalCapability:
    return profile_test_support._successful_profile_terminal(
        operational_bundle,
        tmp_path_factory.mktemp("identity-successful-profile-terminal"),
    )


@pytest.fixture(scope="module")
def gate_p_authorization(
    operational_bundle: VerifiedGatePOperationalBundle,
    successful_profile_terminal: SuccessfulProfileTerminalCapability,
) -> GatePAuthorization:
    return authorize_gate_p(
        operational_bundle=operational_bundle,
        successful_terminal=successful_profile_terminal,
    )


@pytest.fixture(scope="module")
def design(
    science: R3ScienceConfigBundle,
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    gate_p_authorization: GatePAuthorization,
    operational_bundle: VerifiedGatePOperationalBundle,
) -> R3PrimaryDesign:
    return create_r3_primary_design(
        science=science,
        gate0_capability=gate0_capability,
        gate1_capabilities=gate1_capabilities,
        profile_authorization=gate_p_authorization,
        operational_bundle=operational_bundle,
    )


@pytest.fixture(scope="module")
def seed_one_continuable_terminal(
    design: R3PrimaryDesign,
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    operational_bundle: VerifiedGatePOperationalBundle,
) -> ContinuablePrimaryTerminalCapability:
    predecessor = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260801],
        task_id=0,
        seed=20260801,
        segment_index=1,
    )
    return r3_test_support.make_continuable_primary_terminal(
        predecessor=predecessor,
        operational_bundle=operational_bundle,
    )


def test_every_identity_type_is_frozen_and_slotted(
    refs: dict[str, ArtifactRef],
    gate_p_admission: GatePAdmission,
    profile_run: ValidatedGatePRun,
    gate_p_authorization: GatePAuthorization,
    design: R3PrimaryDesign,
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    seed_one_continuable_terminal: ContinuablePrimaryTerminalCapability,
) -> None:
    segment = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260801],
        task_id=0,
        seed=20260801,
        segment_index=1,
    )
    evidence = validate_continuation_evidence(
        predecessor=segment,
        continuable_terminal=seed_one_continuable_terminal,
    )
    values = (
        refs["gate0"],
        gate_p_admission,
        profile_run,
        gate_p_authorization,
        design,
        segment,
        evidence,
    )
    for value in values:
        assert is_dataclass(value)
        assert type(value).__dataclass_params__.frozen is True
        assert "__slots__" in vars(type(value))
        with pytest.raises(FrozenInstanceError):
            value.schema_version = "tampered"  # type: ignore[misc]
    for capability in values[1:]:
        with pytest.raises(TypeError, match="validating factory"):
            replace(capability)


@pytest.mark.parametrize(
    ("schema", "digest", "role"),
    [
        ("", "a" * 64, "role"),
        ("schema", "A" * 64, "role"),
        ("schema", "a" * 63, "role"),
        ("schema", "a" * 64, ""),
        (b"schema", "a" * 64, "role"),
    ],
)
def test_artifact_ref_rejects_noncanonical_fields(
    schema: object,
    digest: object,
    role: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ArtifactRef(
            schema_version=schema,  # type: ignore[arg-type]
            artifact_sha256=digest,  # type: ignore[arg-type]
            role=role,  # type: ignore[arg-type]
        )


def test_gate_p_admission_recomputes_hash_and_closes_artifact_roles(
    gate_p_admission: GatePAdmission,
    refs: dict[str, ArtifactRef],
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    science: R3ScienceConfigBundle,
) -> None:
    gate_p_admission.validate_integrity()
    assert gate_p_admission.to_dict()["admission_sha256"] == (gate_p_admission.admission_sha256)
    with pytest.raises(ValueError):
        replace(
            gate_p_admission,
            admission_sha256="f" * 64,
            _factory_token=identity._FACTORY_TOKEN,
        )
    with pytest.raises(ValueError):
        replace(
            gate_p_admission,
            gate0=refs["gate1"],
            _factory_token=identity._FACTORY_TOKEN,
        )
    with pytest.raises(TypeError):
        replace(gate_p_admission, _factory_token=None)
    with pytest.raises(TypeError):
        create_gate_p_admission(
            gate0_capability=refs["gate0"],  # type: ignore[arg-type]
            gate1_capabilities=gate1_capabilities,
            science=science,
        )
    with pytest.raises(TypeError, match="R3Gate1Capabilities"):
        create_gate_p_admission(
            gate0_capability=gate0_capability,
            gate1_capabilities=refs["gate1"],  # type: ignore[arg-type]
            science=science,
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("seed", 20260802),
        ("head_order", ("prorm_plus", "bt_mle")),
        ("completed_updates_per_head", 99),
        ("audit_updates", (0, 20, 40, 60, 80)),
        ("scheduler_segments", 2),
        ("profile_nonreusable", False),
        ("information_boundary", "train_and_heldout"),
        ("stop_reason", "gradient_ratio_converged"),
        ("profile_run_sha256", "f" * 64),
    ],
)
def test_gate_p_run_closed_fields_cannot_be_rehashed_by_callers(
    profile_run: ValidatedGatePRun,
    field_name: str,
    replacement: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(
            profile_run,
            **{
                field_name: replacement,
                "_factory_token": identity._FACTORY_TOKEN,
            },
        )


def test_gate_p_run_binds_seed_science_and_exact_config_bytes(
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    materializations: dict[int, ValidatedR3Materialization],
    science: R3ScienceConfigBundle,
    refs: dict[str, ArtifactRef],
    gate_p_admission: GatePAdmission,
) -> None:
    with pytest.raises(TypeError, match="R3TrainMaterializationCapability"):
        create_validated_gate_p_run(
            materialization_capability=materializations[20260801],  # type: ignore[arg-type]
            science=science,
            admission=gate_p_admission,
        )
    with pytest.raises(ValueError, match="seed 20260801"):
        create_validated_gate_p_run(
            materialization_capability=materialization_capabilities[20260802],
            science=science,
            admission=gate_p_admission,
        )
    wrong_config = _ref(
        CONFIG_ARTIFACT_SCHEMA,
        CONFIG_ARTIFACT_ROLE,
        "wrong-config-bytes",
    )
    wrong_payload = identity._gate_p_admission_payload(
        gate0=refs["gate0"],
        gate1=refs["gate1"],
        source=refs["source"],
        container=refs["container"],
        config=wrong_config,
    )
    wrong_admission = GatePAdmission(
        schema_version=identity.GATE_P_ADMISSION_SCHEMA,
        gate0=refs["gate0"],
        gate1=refs["gate1"],
        source=refs["source"],
        container=refs["container"],
        config=wrong_config,
        admission_sha256=identity._canonical_sha256(wrong_payload),
        _factory_token=identity._FACTORY_TOKEN,
    )
    with pytest.raises(ValueError, match="science file bytes"):
        create_validated_gate_p_run(
            materialization_capability=materialization_capabilities[20260801],
            science=science,
            admission=wrong_admission,
        )


def test_gate_p_authorization_rejects_generic_or_live_profile_claims(
    profile_run: ValidatedGatePRun,
    refs: dict[str, ArtifactRef],
    operational_bundle: VerifiedGatePOperationalBundle,
    successful_profile_terminal: SuccessfulProfileTerminalCapability,
) -> None:
    with pytest.raises(TypeError, match="VerifiedGatePOperationalBundle"):
        authorize_gate_p(
            operational_bundle=refs["resource"],  # type: ignore[arg-type]
            successful_terminal=successful_profile_terminal,
        )
    with pytest.raises(TypeError, match="SuccessfulProfileTerminalCapability"):
        authorize_gate_p(
            operational_bundle=operational_bundle,
            successful_terminal=refs["profile_terminal"],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        authorize_gate_p(
            operational_bundle=profile_run,  # type: ignore[arg-type]
            successful_terminal=successful_profile_terminal,
        )


def test_gate_p_authorization_is_tensor_free_and_factory_sealed(
    gate_p_authorization: GatePAuthorization,
) -> None:
    assert not hasattr(gate_p_authorization, "profile_run")
    assert "materialization" not in json.dumps(
        gate_p_authorization.to_dict(),
        sort_keys=True,
    )
    with pytest.raises(ValueError):
        replace(
            gate_p_authorization,
            authorization_sha256="f" * 64,
            _factory_token=identity._FACTORY_TOKEN,
        )
    with pytest.raises(TypeError):
        replace(gate_p_authorization, _factory_token=None)


def test_primary_design_is_new_closed_primary_only_identity(
    design: R3PrimaryDesign,
) -> None:
    design.validate_integrity()
    assert design.design_sha256 != R2_RECOVERY_DESIGN_SHA256
    assert design.ordered_seeds == (20260801, 20260802, 20260803)
    assert design.primary_heads == ("bt_mle", "prorm_plus")
    assert design.task_seed_map == R3_TASK_SEED_MAP
    assert design.resource_policy_sha256 == (
        design.profile_authorization.resource_plan.artifact_sha256
    )
    assert design.to_dict()["design_sha256"] == design.design_sha256
    assert not any(
        field.name.endswith("mapping") or field.name == "design" for field in fields(design)
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("campaign_kind", "phase2_recovery_revision2"),
        ("execution_revision", 2),
        ("campaign_role", "five_head_recovery"),
        ("ordered_seeds", (20260802, 20260801, 20260803)),
        ("primary_heads", ("prorm_plus", "bt_mle")),
        ("task_seed_map", ((0, 20260802), (1, 20260801), (2, 20260803))),
        ("resource_policy_sha256", "e" * 64),
        ("checkpoint_policy_sha256", "e" * 64),
        ("max_scheduler_segments", 0),
        ("max_scheduler_segments", True),
        ("design_sha256", R2_RECOVERY_DESIGN_SHA256),
    ],
)
def test_primary_design_rejects_any_closed_field_or_hash_change(
    design: R3PrimaryDesign,
    field_name: str,
    replacement: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(
            design,
            **{
                field_name: replacement,
                "_factory_token": identity._FACTORY_TOKEN,
            },
        )


def test_primary_design_factory_rejects_mapping_and_gate_p_ref_drift(
    science: R3ScienceConfigBundle,
    refs: dict[str, ArtifactRef],
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    gate_p_authorization: GatePAuthorization,
    operational_bundle: VerifiedGatePOperationalBundle,
) -> None:
    common: dict[str, Any] = {
        "gate0_capability": gate0_capability,
        "gate1_capabilities": gate1_capabilities,
        "profile_authorization": gate_p_authorization,
        "operational_bundle": operational_bundle,
    }
    with pytest.raises(TypeError):
        create_r3_primary_design(
            science={"semantic_sha256": science.semantic_sha256},  # type: ignore[arg-type]
            **common,
        )
    with pytest.raises(TypeError, match="R3Gate0Capability"):
        create_r3_primary_design(
            science=science,
            **{
                **common,
                "gate0_capability": refs["gate0"],
            },
        )
    with pytest.raises(TypeError):
        create_r3_primary_design(
            science=science,
            **common,
            signal_policy_sha256="a" * 64,  # type: ignore[call-arg]
        )


def test_primary_segment_task_seed_map_and_fresh_start_are_exact(
    design: R3PrimaryDesign,
    materializations: dict[int, ValidatedR3Materialization],
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
) -> None:
    logical_ids: set[str] = set()
    for task_id, seed in R3_TASK_SEED_MAP:
        segment = admit_primary_segment(
            design=design,
            materialization_capability=materialization_capabilities[seed],
            task_id=task_id,
            seed=seed,
            segment_index=1,
        )
        segment.validate_integrity()
        assert segment.start_mode == "fresh_zero_head_fresh_adamw"
        assert segment.continuation_evidence is None
        assert segment.head_run_ids[0] != segment.head_run_ids[1]
        assert segment.to_dict()["task_id"] == task_id
        logical_ids.add(segment.logical_run_id)
    assert len(logical_ids) == 3
    with pytest.raises(ValueError, match="task map"):
        admit_primary_segment(
            design=design,
            materialization_capability=materialization_capabilities[20260801],
            task_id=1,
            seed=20260801,
            segment_index=1,
        )
    with pytest.raises(ValueError, match="materialization seed"):
        admit_primary_segment(
            design=design,
            materialization_capability=materialization_capabilities[20260802],
            task_id=0,
            seed=20260801,
            segment_index=1,
        )
    with pytest.raises(TypeError):
        admit_primary_segment(
            design=design,
            materialization_capability=materialization_capabilities[20260801],
            task_id=False,
            seed=20260801,
            segment_index=1,
        )
    with pytest.raises(TypeError, match="exact R3TrainMaterializationCapability"):
        admit_primary_segment(
            design=design,
            materialization_capability=materializations[20260801],  # type: ignore[arg-type]
            task_id=0,
            seed=20260801,
            segment_index=1,
        )


def test_later_segments_require_consecutive_same_logical_run_evidence(
    design: R3PrimaryDesign,
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    seed_one_continuable_terminal: ContinuablePrimaryTerminalCapability,
) -> None:
    first = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260801],
        task_id=0,
        seed=20260801,
        segment_index=1,
    )
    with pytest.raises(TypeError, match="VerifiedContinuationEvidence"):
        admit_primary_segment(
            design=design,
            materialization_capability=materialization_capabilities[20260801],
            task_id=0,
            seed=20260801,
            segment_index=2,
        )
    evidence = validate_continuation_evidence(
        predecessor=first,
        continuable_terminal=seed_one_continuable_terminal,
    )
    second = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260801],
        task_id=0,
        seed=20260801,
        segment_index=2,
        continuation_evidence=evidence,
    )
    assert second.start_mode == "verified_state_complete_continuation"
    assert second.logical_run_id == first.logical_run_id
    assert second.head_run_ids == first.head_run_ids
    assert second.scheduler_segment_id != first.scheduler_segment_id
    with pytest.raises(ValueError, match="consecutive"):
        admit_primary_segment(
            design=design,
            materialization_capability=materialization_capabilities[20260801],
            task_id=0,
            seed=20260801,
            segment_index=3,
            continuation_evidence=evidence,
        )
    with pytest.raises(ValueError, match="segment 1 cannot"):
        admit_primary_segment(
            design=design,
            materialization_capability=materialization_capabilities[20260801],
            task_id=0,
            seed=20260801,
            segment_index=1,
            continuation_evidence=evidence,
        )


def test_continuation_evidence_requires_one_exact_sealed_terminal(
    design: R3PrimaryDesign,
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    seed_one_continuable_terminal: ContinuablePrimaryTerminalCapability,
) -> None:
    first = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260801],
        task_id=0,
        seed=20260801,
        segment_index=1,
    )
    with pytest.raises(TypeError, match="exact ContinuablePrimaryTerminalCapability"):
        validate_continuation_evidence(
            predecessor=first,
            continuable_terminal=_ref(
                CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
                CONTINUABLE_PRIMARY_TERMINAL_ROLE,
                "caller-generic-terminal",
            ),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="private validator"):
        validate_continuation_evidence(
            predecessor=first,
            continuable_terminal=replace(seed_one_continuable_terminal),
        )
    evidence = validate_continuation_evidence(
        predecessor=first,
        continuable_terminal=seed_one_continuable_terminal,
    )
    assert evidence.scheduler_terminal == seed_one_continuable_terminal.artifact_ref()
    assert (
        evidence.verified_checkpoint.to_dict()
        == seed_one_continuable_terminal.runtime_closure.outcome_payload["continuation_checkpoint"]
    )
    with pytest.raises(ValueError):
        replace(
            evidence,
            evidence_sha256="f" * 64,
            _factory_token=identity._FACTORY_TOKEN,
        )
    with pytest.raises(TypeError):
        replace(evidence, _factory_token=None)


def test_cross_seed_evidence_cannot_continue_another_logical_run(
    design: R3PrimaryDesign,
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    operational_bundle: VerifiedGatePOperationalBundle,
) -> None:
    seed_two_first = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260802],
        task_id=1,
        seed=20260802,
        segment_index=1,
    )
    seed_two_terminal = r3_test_support.make_continuable_primary_terminal(
        predecessor=seed_two_first,
        operational_bundle=operational_bundle,
    )
    evidence = validate_continuation_evidence(
        predecessor=seed_two_first,
        continuable_terminal=seed_two_terminal,
    )
    with pytest.raises(ValueError, match="logical run"):
        admit_primary_segment(
            design=design,
            materialization_capability=materialization_capabilities[20260801],
            task_id=0,
            seed=20260801,
            segment_index=2,
            continuation_evidence=evidence,
        )


def test_segment_limit_and_rehashed_identity_tampering_fail_closed(
    design: R3PrimaryDesign,
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
) -> None:
    with pytest.raises(ValueError, match="frozen maximum"):
        admit_primary_segment(
            design=design,
            materialization_capability=materialization_capabilities[20260801],
            task_id=0,
            seed=20260801,
            segment_index=4,
        )
    first = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260801],
        task_id=0,
        seed=20260801,
        segment_index=1,
    )
    with pytest.raises(ValueError):
        replace(
            first,
            logical_run_id="f" * 64,
            _factory_token=identity._FACTORY_TOKEN,
        )
    with pytest.raises(ValueError):
        replace(
            first,
            head_run_ids=("e" * 64, "d" * 64),
            _factory_token=identity._FACTORY_TOKEN,
        )
    with pytest.raises(ValueError):
        replace(
            first,
            scheduler_segment_id="c" * 64,
            _factory_token=identity._FACTORY_TOKEN,
        )
    with pytest.raises(ValueError):
        replace(
            first,
            admission_sha256="b" * 64,
            _factory_token=identity._FACTORY_TOKEN,
        )
    with pytest.raises(TypeError):
        replace(first, _factory_token=None)


def _json_mapping(value: object) -> dict[str, object]:
    copied = json.loads(json.dumps(value))
    assert isinstance(copied, dict)
    return copied


def test_all_identity_mappings_rehydrate_against_exact_live_dependencies(
    science: R3ScienceConfigBundle,
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    gate_p_admission: GatePAdmission,
    profile_run: ValidatedGatePRun,
    gate_p_authorization: GatePAuthorization,
    design: R3PrimaryDesign,
    operational_bundle: VerifiedGatePOperationalBundle,
    successful_profile_terminal: SuccessfulProfileTerminalCapability,
    seed_one_continuable_terminal: ContinuablePrimaryTerminalCapability,
) -> None:
    reopened_gate = rehydrate_gate_p_admission(
        _json_mapping(gate_p_admission.to_dict()),
        gate0_capability=gate0_capability,
        gate1_capabilities=gate1_capabilities,
        science=science,
    )
    reopened_profile = rehydrate_validated_gate_p_run(
        _json_mapping(profile_run.to_dict()),
        materialization_capability=materialization_capabilities[20260801],
        science=science,
        admission=reopened_gate,
    )
    reopened_authorization = rehydrate_gate_p_authorization(
        _json_mapping(gate_p_authorization.to_dict()),
        operational_bundle=operational_bundle,
        successful_terminal=successful_profile_terminal,
    )
    reopened_design = rehydrate_r3_primary_design(
        _json_mapping(design.to_dict()),
        science=science,
        gate0_capability=gate0_capability,
        gate1_capabilities=gate1_capabilities,
        profile_authorization=reopened_authorization,
        operational_bundle=operational_bundle,
    )
    first = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260801],
        task_id=0,
        seed=20260801,
        segment_index=1,
    )
    reopened_first = rehydrate_primary_segment_admission(
        _json_mapping(first.to_dict()),
        design=reopened_design,
        materialization_capability=materialization_capabilities[20260801],
    )
    evidence = validate_continuation_evidence(
        predecessor=first,
        continuable_terminal=seed_one_continuable_terminal,
    )
    reopened_evidence = rehydrate_verified_continuation_evidence(
        _json_mapping(evidence.to_dict()),
        predecessor=reopened_first,
        continuable_terminal=seed_one_continuable_terminal,
    )
    second = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260801],
        task_id=0,
        seed=20260801,
        segment_index=2,
        continuation_evidence=evidence,
    )
    reopened_second = rehydrate_primary_segment_admission(
        _json_mapping(second.to_dict()),
        design=reopened_design,
        materialization_capability=materialization_capabilities[20260801],
        continuation_evidence=reopened_evidence,
    )

    assert (
        reopen_artifact_ref(_json_mapping(gate_p_admission.gate0.to_dict()))
        == gate_p_admission.gate0
    )
    for original, reopened in (
        (gate_p_admission, reopened_gate),
        (profile_run, reopened_profile),
        (gate_p_authorization, reopened_authorization),
        (design, reopened_design),
        (first, reopened_first),
        (evidence, reopened_evidence),
        (second, reopened_second),
    ):
        assert reopened.to_dict() == original.to_dict()
        assert "_factory_token" not in reopened.to_dict()

    serialized = json.dumps(
        [
            reopened_profile.to_dict(),
            reopened_design.to_dict(),
            reopened_second.to_dict(),
        ],
        sort_keys=True,
    )
    assert "policy_scores" not in serialized
    assert "reward_features" not in serialized
    assert "tensor" not in serialized.lower()


def test_published_identity_crosses_process_boundary_as_bytes_not_objects(
    tmp_path: Path,
    gate_p_admission: GatePAdmission,
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    science: R3ScienceConfigBundle,
) -> None:
    path = (tmp_path / "gate-p-admission.json").resolve()
    published = publish_canonical_artifact(path, gate_p_admission.to_dict())

    reopened_bytes = read_canonical_artifact(
        path,
        expected_file_sha256=published.file_sha256,
    )
    reopened_identity = rehydrate_gate_p_admission(
        reopened_bytes.payload,
        gate0_capability=gate0_capability,
        gate1_capabilities=gate1_capabilities,
        science=science,
    )

    assert reopened_identity.to_dict() == gate_p_admission.to_dict()
    assert reopened_bytes.file_sha256 != gate_p_admission.admission_sha256
    assert not hasattr(reopened_bytes, "admission_sha256")


def test_admission_rehydration_rejects_current_capability_drift(
    gate_p_admission: GatePAdmission,
    gate1_capabilities: R3Gate1Capabilities,
    science: R3ScienceConfigBundle,
) -> None:
    drifted_gate0 = R3Gate0Capability(
        schema_version=gate0_module.GATE0_ARTIFACT_SCHEMA,
        role=gate0_module.GATE0_ARTIFACT_ROLE,
        artifact_sha256=_digest("other-gate0"),
        file_sha256=_digest("other-gate0-file"),
        production_relative=gate0_module._GATE0_RELATIVE.as_posix(),
        _factory_token=gate0_module._FACTORY_TOKEN,
    )
    with pytest.raises(ValueError, match="rehydrated closed identity"):
        rehydrate_gate_p_admission(
            _json_mapping(gate_p_admission.to_dict()),
            gate0_capability=drifted_gate0,
            gate1_capabilities=gate1_capabilities,
            science=science,
        )


def test_rehydration_rejects_extra_missing_and_rehashed_fields(
    science: R3ScienceConfigBundle,
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    gate_p_admission: GatePAdmission,
    profile_run: ValidatedGatePRun,
    gate_p_authorization: GatePAuthorization,
    design: R3PrimaryDesign,
    operational_bundle: VerifiedGatePOperationalBundle,
    successful_profile_terminal: SuccessfulProfileTerminalCapability,
) -> None:
    extra = _json_mapping(gate_p_admission.to_dict())
    extra["caller_authorized"] = True
    with pytest.raises(ValueError, match="closed field"):
        rehydrate_gate_p_admission(
            extra,
            gate0_capability=gate0_capability,
            gate1_capabilities=gate1_capabilities,
            science=science,
        )

    missing = _json_mapping(gate_p_admission.to_dict())
    del missing["gate1"]
    with pytest.raises(ValueError, match="closed field"):
        rehydrate_gate_p_admission(
            missing,
            gate0_capability=gate0_capability,
            gate1_capabilities=gate1_capabilities,
            science=science,
        )

    rehashed = _json_mapping(profile_run.to_dict())
    rehashed["profile_run_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="rehydrated closed identity"):
        rehydrate_validated_gate_p_run(
            rehashed,
            materialization_capability=materialization_capabilities[20260801],
            science=science,
            admission=gate_p_admission,
        )

    wrong_authorization_hash = _json_mapping(gate_p_authorization.to_dict())
    wrong_authorization_hash["authorization_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="rehydrated closed identity"):
        rehydrate_gate_p_authorization(
            wrong_authorization_hash,
            operational_bundle=operational_bundle,
            successful_terminal=successful_profile_terminal,
        )

    wrong_design_hash = _json_mapping(design.to_dict())
    wrong_design_hash["design_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="rehydrated closed identity"):
        rehydrate_r3_primary_design(
            wrong_design_hash,
            science=science,
            gate0_capability=gate0_capability,
            gate1_capabilities=gate1_capabilities,
            profile_authorization=gate_p_authorization,
            operational_bundle=operational_bundle,
        )


def test_rehydration_rejects_dependency_drift_and_wrong_runtime_types(
    science: R3ScienceConfigBundle,
    materialization_capabilities: dict[int, R3TrainMaterializationCapability],
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    gate_p_admission: GatePAdmission,
    profile_run: ValidatedGatePRun,
    gate_p_authorization: GatePAuthorization,
    design: R3PrimaryDesign,
    operational_bundle: VerifiedGatePOperationalBundle,
    seed_one_continuable_terminal: ContinuablePrimaryTerminalCapability,
) -> None:
    with pytest.raises(ValueError, match="seed 20260801"):
        rehydrate_validated_gate_p_run(
            _json_mapping(profile_run.to_dict()),
            materialization_capability=materialization_capabilities[20260802],
            science=science,
            admission=gate_p_admission,
        )

    drifted_gate0 = R3Gate0Capability(
        schema_version=gate0_module.GATE0_ARTIFACT_SCHEMA,
        role=gate0_module.GATE0_ARTIFACT_ROLE,
        artifact_sha256=_digest("rehydrate-other-gate0"),
        file_sha256=_digest("rehydrate-other-gate0-file"),
        production_relative=gate0_module._GATE0_RELATIVE.as_posix(),
        _factory_token=gate0_module._FACTORY_TOKEN,
    )
    with pytest.raises(ValueError):
        rehydrate_r3_primary_design(
            _json_mapping(design.to_dict()),
            science=science,
            gate0_capability=drifted_gate0,
            gate1_capabilities=gate1_capabilities,
            profile_authorization=gate_p_authorization,
            operational_bundle=operational_bundle,
        )

    first = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260801],
        task_id=0,
        seed=20260801,
        segment_index=1,
    )
    observed_first = _json_mapping(first.to_dict())
    observed_first["task_id"] = True
    with pytest.raises(TypeError, match="integer"):
        rehydrate_primary_segment_admission(
            observed_first,
            design=design,
            materialization_capability=materialization_capabilities[20260801],
        )

    other_first = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capabilities[20260802],
        task_id=1,
        seed=20260802,
        segment_index=1,
    )
    evidence = validate_continuation_evidence(
        predecessor=first,
        continuable_terminal=seed_one_continuable_terminal,
    )
    with pytest.raises(ValueError, match="rehydrated closed identity"):
        rehydrate_verified_continuation_evidence(
            _json_mapping(evidence.to_dict()),
            predecessor=other_first,
            continuable_terminal=seed_one_continuable_terminal,
        )


def test_reopened_arbitrary_role_is_not_a_gate_capability(
    refs: dict[str, ArtifactRef],
    gate1_capabilities: R3Gate1Capabilities,
    science: R3ScienceConfigBundle,
) -> None:
    caller_ref = reopen_artifact_ref(
        {
            "schema_version": "caller-schema/v1",
            "artifact_sha256": _digest("caller"),
            "role": "caller_says_gate0",
        }
    )
    with pytest.raises(TypeError, match="R3Gate0Capability"):
        create_gate_p_admission(
            gate0_capability=caller_ref,  # type: ignore[arg-type]
            gate1_capabilities=gate1_capabilities,
            science=science,
        )
