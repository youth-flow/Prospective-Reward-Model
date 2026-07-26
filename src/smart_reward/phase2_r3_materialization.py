"""Fail-closed attestation for the formal R3 train materialization.

The neutral primary context deliberately carries no campaign claim.  This
module is the narrow promotion boundary that binds that context to the frozen
R3 science configuration and to externally verified parent-artifact bytes.
It does not load artifacts, decode held-out bytes, train a head, or accept a
caller-controlled ``formal`` flag.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from . import phase2_training as _training
from .phase2_primary import NeutralPhase2TrainingContext

if TYPE_CHECKING:
    from .phase2_r3_config import R3ScienceConfigBundle

TRAIN_MATERIALIZATION_PROVENANCE_SCHEMA = "phase2-r3-train-materialization-provenance/v1"
VALIDATED_R3_MATERIALIZATION_SCHEMA = "phase2-r3-validated-train-materialization/v1"
TRAIN_SPLIT_NAME = "train"
TRAIN_TENSOR_KEYS = (
    "policy_scores",
    "reward_features",
    "h",
    "left_wins",
    "num_annotations",
)


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("identity payload must contain strict JSON data") from error
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, *, name: str) -> str:
    return _training._validate_digest(value, name=name)


def _positive_integer(value: object, *, name: str) -> int:
    return _training._positive_integer(value, name=name)


def _ordered_prompt_ids_sha256(prompt_ids: Sequence[str | int]) -> str:
    if isinstance(prompt_ids, (str, bytes, bytearray)) or not isinstance(prompt_ids, Sequence):
        raise TypeError("ordered train prompt IDs must be a sequence")
    normalized: list[str | int] = []
    for index, prompt_id in enumerate(prompt_ids):
        if isinstance(prompt_id, bool) or not isinstance(prompt_id, (str, int)):
            raise TypeError(
                f"ordered train prompt ID {index} must be a string or non-boolean integer"
            )
        if isinstance(prompt_id, str) and not prompt_id:
            raise ValueError(f"ordered train prompt ID {index} must be non-empty")
        normalized.append(prompt_id)
    if not normalized:
        raise ValueError("ordered train prompt IDs must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("ordered train prompt IDs must be unique")
    return _canonical_sha256(
        {
            "schema_version": "phase2-r3-ordered-train-prompt-ids/v1",
            "split_name": TRAIN_SPLIT_NAME,
            "prompt_ids": normalized,
        }
    )


def _context_tensor_sha256(
    context: NeutralPhase2TrainingContext,
) -> dict[str, str]:
    return {
        name: _training._tensor_sha256(getattr(context.training, name))
        for name in TRAIN_TENSOR_KEYS
    }


def _normalize_tensor_hashes(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("train_tensor_sha256 must be a mapping")
    if set(value) != set(TRAIN_TENSOR_KEYS):
        raise ValueError(
            "train_tensor_sha256 must contain exactly the five frozen train tensor keys"
        )
    normalized = {
        name: _digest(value[name], name=f"train_tensor_sha256[{name!r}]")
        for name in TRAIN_TENSOR_KEYS
    }
    return MappingProxyType(normalized)


def _provenance_payload(
    *,
    schema_version: str,
    seed: int,
    parent_artifact_registry_sha256: str,
    artifact_metadata_sha256: str,
    artifact_tensors_sha256: str,
    artifact_candidates_sha256: str,
    artifact_materialization_sha256: str,
    artifact_verification_sha256: str,
    source_run_manifest_sha256: str,
    source_producer_identity_sha256: str,
    split_name: str,
    ordered_train_prompt_ids_sha256: str,
    train_tensor_sha256: Mapping[str, str],
    candidate_train_prefix_sha256: str,
    candidate_train_prefix_count: int,
    input_training_sha256: str,
    prepared_training_sha256: str,
    oracle_reward_sha256: str,
    label_stream_sha256: str,
    heldout_bytes_decoded: bool,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "seed": seed,
        "parent_artifact_registry_sha256": parent_artifact_registry_sha256,
        "artifact_metadata_sha256": artifact_metadata_sha256,
        "artifact_tensors_sha256": artifact_tensors_sha256,
        "artifact_candidates_sha256": artifact_candidates_sha256,
        "artifact_materialization_sha256": artifact_materialization_sha256,
        "artifact_verification_sha256": artifact_verification_sha256,
        "source_run_manifest_sha256": source_run_manifest_sha256,
        "source_producer_identity_sha256": source_producer_identity_sha256,
        "split_name": split_name,
        "ordered_train_prompt_ids_sha256": ordered_train_prompt_ids_sha256,
        "train_tensor_sha256": {name: train_tensor_sha256[name] for name in TRAIN_TENSOR_KEYS},
        "candidate_train_prefix_sha256": candidate_train_prefix_sha256,
        "candidate_train_prefix_count": candidate_train_prefix_count,
        "input_training_sha256": input_training_sha256,
        "prepared_training_sha256": prepared_training_sha256,
        "oracle_reward_sha256": oracle_reward_sha256,
        "label_stream_sha256": label_stream_sha256,
        "heldout_bytes_decoded": heldout_bytes_decoded,
    }


@dataclass(frozen=True, slots=True)
class TrainMaterializationProvenance:
    """Byte identities and train-only decoding evidence from the parent source."""

    schema_version: str
    seed: int
    parent_artifact_registry_sha256: str
    artifact_metadata_sha256: str
    artifact_tensors_sha256: str
    artifact_candidates_sha256: str
    artifact_materialization_sha256: str
    artifact_verification_sha256: str
    source_run_manifest_sha256: str
    source_producer_identity_sha256: str
    split_name: str
    ordered_train_prompt_ids_sha256: str
    train_tensor_sha256: Mapping[str, str]
    candidate_train_prefix_sha256: str
    candidate_train_prefix_count: int
    input_training_sha256: str
    prepared_training_sha256: str
    oracle_reward_sha256: str
    label_stream_sha256: str
    heldout_bytes_decoded: Literal[False]
    provenance_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != TRAIN_MATERIALIZATION_PROVENANCE_SCHEMA:
            raise ValueError("train materialization provenance schema is not the frozen R3 schema")
        seed = _training._validate_seed(self.seed)
        for name in (
            "parent_artifact_registry_sha256",
            "artifact_metadata_sha256",
            "artifact_tensors_sha256",
            "artifact_candidates_sha256",
            "artifact_materialization_sha256",
            "artifact_verification_sha256",
            "source_run_manifest_sha256",
            "source_producer_identity_sha256",
            "ordered_train_prompt_ids_sha256",
            "candidate_train_prefix_sha256",
            "input_training_sha256",
            "prepared_training_sha256",
            "oracle_reward_sha256",
            "label_stream_sha256",
            "provenance_sha256",
        ):
            _digest(getattr(self, name), name=name)
        if self.split_name != TRAIN_SPLIT_NAME:
            raise ValueError("R3 materialization provenance may bind only the exact train split")
        tensor_hashes = _normalize_tensor_hashes(self.train_tensor_sha256)
        object.__setattr__(self, "train_tensor_sha256", tensor_hashes)
        prefix_count = _positive_integer(
            self.candidate_train_prefix_count,
            name="candidate_train_prefix_count",
        )
        if self.heldout_bytes_decoded is not False:
            raise ValueError("R3 train materialization must not decode any held-out bytes")
        payload = _provenance_payload(
            schema_version=self.schema_version,
            seed=seed,
            parent_artifact_registry_sha256=self.parent_artifact_registry_sha256,
            artifact_metadata_sha256=self.artifact_metadata_sha256,
            artifact_tensors_sha256=self.artifact_tensors_sha256,
            artifact_candidates_sha256=self.artifact_candidates_sha256,
            artifact_materialization_sha256=self.artifact_materialization_sha256,
            artifact_verification_sha256=self.artifact_verification_sha256,
            source_run_manifest_sha256=self.source_run_manifest_sha256,
            source_producer_identity_sha256=self.source_producer_identity_sha256,
            split_name=self.split_name,
            ordered_train_prompt_ids_sha256=self.ordered_train_prompt_ids_sha256,
            train_tensor_sha256=tensor_hashes,
            candidate_train_prefix_sha256=self.candidate_train_prefix_sha256,
            candidate_train_prefix_count=prefix_count,
            input_training_sha256=self.input_training_sha256,
            prepared_training_sha256=self.prepared_training_sha256,
            oracle_reward_sha256=self.oracle_reward_sha256,
            label_stream_sha256=self.label_stream_sha256,
            heldout_bytes_decoded=self.heldout_bytes_decoded,
        )
        if _canonical_sha256(payload) != self.provenance_sha256:
            raise ValueError("train materialization provenance SHA256 does not match its contents")

    @classmethod
    def from_context(
        cls,
        context: NeutralPhase2TrainingContext,
        *,
        parent_artifact_registry_sha256: str,
        artifact_metadata_sha256: str,
        artifact_tensors_sha256: str,
        artifact_candidates_sha256: str,
        artifact_materialization_sha256: str,
        artifact_verification_sha256: str,
        source_run_manifest_sha256: str,
        source_producer_identity_sha256: str,
        candidate_train_prefix_sha256: str,
        candidate_train_prefix_count: int,
    ) -> TrainMaterializationProvenance:
        """Build self-hashed provenance after an upstream byte verifier succeeds."""

        if type(context) is not NeutralPhase2TrainingContext:
            raise TypeError("context must be an exact NeutralPhase2TrainingContext")
        context.validate_integrity()
        tensor_hashes = _context_tensor_sha256(context)
        payload = _provenance_payload(
            schema_version=TRAIN_MATERIALIZATION_PROVENANCE_SCHEMA,
            seed=context.seed,
            parent_artifact_registry_sha256=_digest(
                parent_artifact_registry_sha256,
                name="parent_artifact_registry_sha256",
            ),
            artifact_metadata_sha256=_digest(
                artifact_metadata_sha256,
                name="artifact_metadata_sha256",
            ),
            artifact_tensors_sha256=_digest(
                artifact_tensors_sha256,
                name="artifact_tensors_sha256",
            ),
            artifact_candidates_sha256=_digest(
                artifact_candidates_sha256,
                name="artifact_candidates_sha256",
            ),
            artifact_materialization_sha256=_digest(
                artifact_materialization_sha256,
                name="artifact_materialization_sha256",
            ),
            artifact_verification_sha256=_digest(
                artifact_verification_sha256,
                name="artifact_verification_sha256",
            ),
            source_run_manifest_sha256=_digest(
                source_run_manifest_sha256,
                name="source_run_manifest_sha256",
            ),
            source_producer_identity_sha256=_digest(
                source_producer_identity_sha256,
                name="source_producer_identity_sha256",
            ),
            split_name=TRAIN_SPLIT_NAME,
            ordered_train_prompt_ids_sha256=_ordered_prompt_ids_sha256(context.training.prompt_ids),
            train_tensor_sha256=tensor_hashes,
            candidate_train_prefix_sha256=_digest(
                candidate_train_prefix_sha256,
                name="candidate_train_prefix_sha256",
            ),
            candidate_train_prefix_count=_positive_integer(
                candidate_train_prefix_count,
                name="candidate_train_prefix_count",
            ),
            input_training_sha256=context.input_training_sha256,
            prepared_training_sha256=context.primary_training_sha256,
            oracle_reward_sha256=context.oracle_reward_sha256,
            label_stream_sha256=context.label_stream.label_stream_sha256,
            heldout_bytes_decoded=False,
        )
        return cls(
            **payload,
            provenance_sha256=_canonical_sha256(payload),
        )

    def validate_integrity(self) -> None:
        """Recompute the complete provenance identity."""

        self.__post_init__()

    def to_dict(self) -> dict[str, object]:
        payload = _provenance_payload(
            schema_version=self.schema_version,
            seed=self.seed,
            parent_artifact_registry_sha256=self.parent_artifact_registry_sha256,
            artifact_metadata_sha256=self.artifact_metadata_sha256,
            artifact_tensors_sha256=self.artifact_tensors_sha256,
            artifact_candidates_sha256=self.artifact_candidates_sha256,
            artifact_materialization_sha256=self.artifact_materialization_sha256,
            artifact_verification_sha256=self.artifact_verification_sha256,
            source_run_manifest_sha256=self.source_run_manifest_sha256,
            source_producer_identity_sha256=self.source_producer_identity_sha256,
            split_name=self.split_name,
            ordered_train_prompt_ids_sha256=self.ordered_train_prompt_ids_sha256,
            train_tensor_sha256=self.train_tensor_sha256,
            candidate_train_prefix_sha256=self.candidate_train_prefix_sha256,
            candidate_train_prefix_count=self.candidate_train_prefix_count,
            input_training_sha256=self.input_training_sha256,
            prepared_training_sha256=self.prepared_training_sha256,
            oracle_reward_sha256=self.oracle_reward_sha256,
            label_stream_sha256=self.label_stream_sha256,
            heldout_bytes_decoded=self.heldout_bytes_decoded,
        )
        return {**payload, "provenance_sha256": self.provenance_sha256}


def _attestation_payload(
    *,
    schema_version: str,
    context_sha256: str,
    settings_sha256: str,
    science_semantic_sha256: str,
    science_file_sha256: str,
    seed: int,
    provenance_sha256: str,
    input_training_sha256: str,
    prepared_training_sha256: str,
    oracle_reward_sha256: str,
    label_stream_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "context_sha256": context_sha256,
        "settings_sha256": settings_sha256,
        "science_semantic_sha256": science_semantic_sha256,
        "science_file_sha256": science_file_sha256,
        "seed": seed,
        "provenance_sha256": provenance_sha256,
        "input_training_sha256": input_training_sha256,
        "prepared_training_sha256": prepared_training_sha256,
        "oracle_reward_sha256": oracle_reward_sha256,
        "label_stream_sha256": label_stream_sha256,
        "heldout_bytes_decoded": False,
    }


def _r3_bundle_type() -> type[R3ScienceConfigBundle]:
    from .phase2_r3_config import R3ScienceConfigBundle

    return R3ScienceConfigBundle


def _validate_components(
    context: NeutralPhase2TrainingContext,
    *,
    science_bundle: R3ScienceConfigBundle,
    provenance: TrainMaterializationProvenance,
) -> dict[str, object]:
    if type(context) is not NeutralPhase2TrainingContext:
        raise TypeError("context must be an exact NeutralPhase2TrainingContext")
    if type(science_bundle) is not _r3_bundle_type():
        raise TypeError("science_bundle must be an exact R3ScienceConfigBundle")
    if type(provenance) is not TrainMaterializationProvenance:
        raise TypeError("provenance must be an exact TrainMaterializationProvenance")
    context.validate_integrity()
    science_bundle.validate_integrity()
    provenance.validate_integrity()
    if context.settings.sha256 != science_bundle.settings.sha256:
        raise ValueError("neutral context settings differ from the frozen R3 science bundle")
    if provenance.seed != context.seed:
        raise ValueError("materialization provenance seed differs from the neutral context")

    expected_prompt_sha = _ordered_prompt_ids_sha256(context.training.prompt_ids)
    if provenance.ordered_train_prompt_ids_sha256 != expected_prompt_sha:
        raise ValueError("ordered train prompt IDs differ from materialization provenance")
    expected_tensor_hashes = _context_tensor_sha256(context)
    if dict(provenance.train_tensor_sha256) != expected_tensor_hashes:
        raise ValueError("prepared train tensors differ from materialization provenance")
    expected_candidate_count = context.training.num_prompts * context.training.num_candidates
    if provenance.candidate_train_prefix_count != expected_candidate_count:
        raise ValueError("candidate train-prefix count differs from the prepared train geometry")
    if provenance.input_training_sha256 != context.input_training_sha256:
        raise ValueError("input training identity differs from materialization provenance")
    if provenance.prepared_training_sha256 != context.primary_training_sha256:
        raise ValueError("prepared training identity differs from materialization provenance")
    if provenance.oracle_reward_sha256 != context.oracle_reward_sha256:
        raise ValueError("train oracle identity differs from materialization provenance")
    if provenance.label_stream_sha256 != context.label_stream.label_stream_sha256:
        raise ValueError("label stream identity differs from materialization provenance")
    if provenance.heldout_bytes_decoded is not False:
        raise ValueError("materialization provenance crossed the held-out information boundary")

    return _attestation_payload(
        schema_version=VALIDATED_R3_MATERIALIZATION_SCHEMA,
        context_sha256=context.context_sha256,
        settings_sha256=context.settings.sha256,
        science_semantic_sha256=_digest(
            science_bundle.semantic_sha256,
            name="science_bundle.semantic_sha256",
        ),
        science_file_sha256=_digest(
            science_bundle.file_sha256,
            name="science_bundle.file_sha256",
        ),
        seed=context.seed,
        provenance_sha256=provenance.provenance_sha256,
        input_training_sha256=context.input_training_sha256,
        prepared_training_sha256=context.primary_training_sha256,
        oracle_reward_sha256=context.oracle_reward_sha256,
        label_stream_sha256=context.label_stream.label_stream_sha256,
    )


@dataclass(frozen=True, slots=True)
class ValidatedR3Materialization:
    """A revalidatable formal attestation, not a caller-supplied mode claim."""

    context: NeutralPhase2TrainingContext = field(repr=False, compare=False)
    science_bundle: R3ScienceConfigBundle = field(repr=False, compare=False)
    provenance: TrainMaterializationProvenance = field(repr=False, compare=False)
    schema_version: str
    context_sha256: str
    settings_sha256: str
    science_semantic_sha256: str
    science_file_sha256: str
    seed: int
    provenance_sha256: str
    input_training_sha256: str
    prepared_training_sha256: str
    oracle_reward_sha256: str
    label_stream_sha256: str
    heldout_bytes_decoded: Literal[False]
    attestation_sha256: str

    def __post_init__(self) -> None:
        expected = _validate_components(
            self.context,
            science_bundle=self.science_bundle,
            provenance=self.provenance,
        )
        observed = {
            "schema_version": self.schema_version,
            "context_sha256": self.context_sha256,
            "settings_sha256": self.settings_sha256,
            "science_semantic_sha256": self.science_semantic_sha256,
            "science_file_sha256": self.science_file_sha256,
            "seed": self.seed,
            "provenance_sha256": self.provenance_sha256,
            "input_training_sha256": self.input_training_sha256,
            "prepared_training_sha256": self.prepared_training_sha256,
            "oracle_reward_sha256": self.oracle_reward_sha256,
            "label_stream_sha256": self.label_stream_sha256,
            "heldout_bytes_decoded": self.heldout_bytes_decoded,
        }
        if observed != expected:
            raise ValueError("validated R3 materialization fields differ from their live sources")
        _digest(self.attestation_sha256, name="attestation_sha256")
        if _canonical_sha256(expected) != self.attestation_sha256:
            raise ValueError("R3 materialization attestation SHA256 does not match its contents")

    def validate_integrity(self) -> None:
        """Revalidate all live sources and both identity layers."""

        self.__post_init__()

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "context_sha256": self.context_sha256,
            "settings_sha256": self.settings_sha256,
            "science_semantic_sha256": self.science_semantic_sha256,
            "science_file_sha256": self.science_file_sha256,
            "seed": self.seed,
            "provenance": self.provenance.to_dict(),
            "provenance_sha256": self.provenance_sha256,
            "input_training_sha256": self.input_training_sha256,
            "prepared_training_sha256": self.prepared_training_sha256,
            "oracle_reward_sha256": self.oracle_reward_sha256,
            "label_stream_sha256": self.label_stream_sha256,
            "heldout_bytes_decoded": self.heldout_bytes_decoded,
            "attestation_sha256": self.attestation_sha256,
        }
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        return payload


def validate_r3_materialization(
    context: NeutralPhase2TrainingContext,
    *,
    science_bundle: R3ScienceConfigBundle,
    provenance: TrainMaterializationProvenance,
) -> ValidatedR3Materialization:
    """Promote a neutral train context only after every R3 binding revalidates."""

    payload = _validate_components(
        context,
        science_bundle=science_bundle,
        provenance=provenance,
    )
    return ValidatedR3Materialization(
        context=context,
        science_bundle=science_bundle,
        provenance=provenance,
        **payload,
        attestation_sha256=_canonical_sha256(payload),
    )


__all__ = [
    "TRAIN_MATERIALIZATION_PROVENANCE_SCHEMA",
    "TRAIN_SPLIT_NAME",
    "TRAIN_TENSOR_KEYS",
    "VALIDATED_R3_MATERIALIZATION_SCHEMA",
    "TrainMaterializationProvenance",
    "ValidatedR3Materialization",
    "validate_r3_materialization",
]
