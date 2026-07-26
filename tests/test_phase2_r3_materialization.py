from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import smart_reward.phase2_r3_materialization as materialization
from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_primary import (
    NeutralPhase2TrainingContext,
    prepare_neutral_phase2_context,
)
from smart_reward.phase2_r3_config import (
    R3ScienceConfigBundle,
    load_r3_science_config,
)
from smart_reward.phase2_r3_materialization import (
    TRAIN_MATERIALIZATION_PROVENANCE_SCHEMA,
    VALIDATED_R3_MATERIALIZATION_SCHEMA,
    TrainMaterializationProvenance,
    ValidatedR3Materialization,
    validate_r3_materialization,
)

ROOT = Path(__file__).resolve().parents[1]
SCIENCE_CONFIG = ROOT / "configs" / "phase2_recovery_r3_science.yaml"


def _token_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _training() -> TrainingTensorData:
    num_prompts, num_candidates, policy_dimension, reward_dimension = 4, 4, 3, 2
    node = torch.arange(
        num_prompts * num_candidates,
        dtype=torch.float32,
    ).reshape(num_prompts, num_candidates)
    return TrainingTensorData(
        prompt_ids=tuple(f"r3-train-{index}" for index in range(num_prompts)),
        policy_scores=torch.stack(
            [
                torch.sin((coordinate + 1.0) * 0.11 * node) + 0.1 * coordinate
                for coordinate in range(policy_dimension)
            ],
            dim=-1,
        ),
        reward_features=torch.stack(
            [
                torch.cos((coordinate + 1.0) * 0.13 * node) - 0.05 * coordinate
                for coordinate in range(reward_dimension)
            ],
            dim=-1,
        ),
        h=torch.linspace(-0.4, 0.3, num_prompts),
        left_wins=torch.tensor([1, 2, 1, 3], dtype=torch.int64),
        num_annotations=torch.tensor([3, 4, 2, 5], dtype=torch.int64),
    )


def _oracle_rewards(training: TrainingTensorData) -> torch.Tensor:
    node = torch.arange(
        training.num_prompts * training.num_candidates,
        dtype=training.policy_scores.dtype,
        device=training.policy_scores.device,
    ).reshape(training.num_prompts, training.num_candidates)
    return 0.2 * torch.sin(0.3 * node)


def _context(bundle: R3ScienceConfigBundle) -> NeutralPhase2TrainingContext:
    training = _training()
    return prepare_neutral_phase2_context(
        training,
        _oracle_rewards(training),
        seed=bundle.settings.seeds[0],
        settings=bundle.settings,
    )


def _provenance(
    context: NeutralPhase2TrainingContext,
) -> TrainMaterializationProvenance:
    return TrainMaterializationProvenance.from_context(
        context,
        parent_artifact_registry_sha256=_token_sha256("parent-artifact-registry"),
        artifact_metadata_sha256=_token_sha256("metadata.json"),
        artifact_tensors_sha256=_token_sha256("tensors.safetensors"),
        artifact_candidates_sha256=_token_sha256("candidates.jsonl"),
        artifact_materialization_sha256=_token_sha256("artifact-materialization.json"),
        artifact_verification_sha256=_token_sha256("artifact-verification.json"),
        source_run_manifest_sha256=_token_sha256("run-manifest.json"),
        source_producer_identity_sha256=_token_sha256("source-producer-identity"),
        candidate_train_prefix_sha256=_token_sha256("ordered-train-candidate-prefix"),
        candidate_train_prefix_count=(
            context.training.num_prompts * context.training.num_candidates
        ),
    )


def _rehashed_provenance(
    provenance: TrainMaterializationProvenance,
    **changes: object,
) -> TrainMaterializationProvenance:
    payload = provenance.to_dict()
    payload.pop("provenance_sha256")
    payload.update(changes)
    return TrainMaterializationProvenance(
        **payload,
        provenance_sha256=materialization._canonical_sha256(payload),
    )


@pytest.fixture(scope="module")
def science_bundle() -> R3ScienceConfigBundle:
    return load_r3_science_config(SCIENCE_CONFIG)


@pytest.fixture(scope="module")
def context(science_bundle: R3ScienceConfigBundle) -> NeutralPhase2TrainingContext:
    return _context(science_bundle)


@pytest.fixture(scope="module")
def provenance(
    context: NeutralPhase2TrainingContext,
) -> TrainMaterializationProvenance:
    return _provenance(context)


def test_validated_materialization_binds_real_r3_science_and_tiny_train_context(
    science_bundle: R3ScienceConfigBundle,
    context: NeutralPhase2TrainingContext,
    provenance: TrainMaterializationProvenance,
) -> None:
    attestation = validate_r3_materialization(
        context,
        science_bundle=science_bundle,
        provenance=provenance,
    )

    assert type(attestation) is ValidatedR3Materialization
    assert attestation.schema_version == VALIDATED_R3_MATERIALIZATION_SCHEMA
    assert provenance.schema_version == TRAIN_MATERIALIZATION_PROVENANCE_SCHEMA
    assert attestation.science_semantic_sha256 == science_bundle.semantic_sha256
    assert attestation.settings_sha256 == science_bundle.settings.sha256
    assert attestation.seed == context.seed == provenance.seed
    assert attestation.context_sha256 == context.context_sha256
    assert attestation.provenance_sha256 == provenance.provenance_sha256
    assert attestation.heldout_bytes_decoded is False
    assert provenance.split_name == "train"
    assert provenance.heldout_bytes_decoded is False
    assert tuple(provenance.train_tensor_sha256) == materialization.TRAIN_TENSOR_KEYS
    assert provenance.candidate_train_prefix_count == 16
    attestation.validate_integrity()
    serialized = attestation.to_dict()
    assert serialized["provenance"] == provenance.to_dict()
    assert json.loads(json.dumps(serialized, allow_nan=False)) == serialized


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("split_name", "validation"),
        ("artifact_metadata_sha256", "f" * 64),
        ("heldout_bytes_decoded", True),
        ("candidate_train_prefix_count", 17),
    ],
)
def test_provenance_dataclass_replace_cannot_preserve_old_identity(
    provenance: TrainMaterializationProvenance,
    field_name: str,
    replacement: object,
) -> None:
    with pytest.raises(ValueError):
        replace(provenance, **{field_name: replacement})


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 20269999},
        {"ordered_train_prompt_ids_sha256": "1" * 64},
        {"candidate_train_prefix_count": 15},
        {"input_training_sha256": "2" * 64},
        {"prepared_training_sha256": "3" * 64},
        {"oracle_reward_sha256": "4" * 64},
        {"label_stream_sha256": "5" * 64},
    ],
)
def test_rehashed_context_binding_tamper_is_rejected(
    science_bundle: R3ScienceConfigBundle,
    context: NeutralPhase2TrainingContext,
    provenance: TrainMaterializationProvenance,
    change: dict[str, object],
) -> None:
    tampered = _rehashed_provenance(provenance, **change)
    with pytest.raises(ValueError):
        validate_r3_materialization(
            context,
            science_bundle=science_bundle,
            provenance=tampered,
        )


def test_rehashed_per_tensor_tamper_is_rejected(
    science_bundle: R3ScienceConfigBundle,
    context: NeutralPhase2TrainingContext,
    provenance: TrainMaterializationProvenance,
) -> None:
    hashes = dict(provenance.train_tensor_sha256)
    hashes["reward_features"] = "6" * 64
    tampered = _rehashed_provenance(provenance, train_tensor_sha256=hashes)

    with pytest.raises(ValueError, match="prepared train tensors"):
        validate_r3_materialization(
            context,
            science_bundle=science_bundle,
            provenance=tampered,
        )


def test_settings_mismatch_is_rejected_even_with_fresh_context_provenance(
    science_bundle: R3ScienceConfigBundle,
) -> None:
    alternate_settings = replace(
        science_bundle.settings,
        phase2_config_hash="a" * 64,
    )
    training = _training()
    alternate_context = prepare_neutral_phase2_context(
        training,
        _oracle_rewards(training),
        seed=alternate_settings.seeds[0],
        settings=alternate_settings,
    )
    alternate_provenance = _provenance(alternate_context)

    with pytest.raises(ValueError, match="settings differ"):
        validate_r3_materialization(
            alternate_context,
            science_bundle=science_bundle,
            provenance=alternate_provenance,
        )


def test_attestation_replace_and_live_tensor_mutation_fail_closed(
    science_bundle: R3ScienceConfigBundle,
) -> None:
    mutable_context = _context(science_bundle)
    provenance = _provenance(mutable_context)
    attestation = validate_r3_materialization(
        mutable_context,
        science_bundle=science_bundle,
        provenance=provenance,
    )
    with pytest.raises(ValueError):
        replace(attestation, attestation_sha256="7" * 64)
    with pytest.raises(ValueError):
        replace(attestation, oracle_reward_sha256="8" * 64)

    with torch.no_grad():
        mutable_context.training.policy_scores[0, 0, 0].add_(0.25)
    with pytest.raises(ValueError, match="prepared primary training tensors"):
        attestation.validate_integrity()


def test_validator_rejects_untyped_provenance(
    science_bundle: R3ScienceConfigBundle,
    context: NeutralPhase2TrainingContext,
) -> None:
    with pytest.raises(TypeError, match="TrainMaterializationProvenance"):
        validate_r3_materialization(
            context,
            science_bundle=science_bundle,
            provenance={},  # type: ignore[arg-type]
        )
