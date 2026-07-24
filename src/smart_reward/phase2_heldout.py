"""Leakage-isolated Phase-2 held-out local-regret evaluation.

The source Phase-1 artifact contains validation/test policy geometry and old
oracle targets.  Phase 2 deliberately does not reuse those targets.  This
module carries only immutable candidate text, reward features, and policy
scores across the training boundary.  Fresh transformed oracle rewards are
created inside the final oracle session, after heads, ``beta_common``, and all
deployed policy directions have been frozen.

The primary split metric exactly follows the Phase-1 local-regret definition,
except that its fixed beta is the seed-specific frozen ``beta_common``:

``m_error.T (F_split + lambda_split I)^-1 m_error / (2 beta_common)``.

Both ``F_split`` and ``lambda_split`` are recomputed from all saved nodes in
that split.  Direction error and cosine remain native beta=1 diagnostics.
No held-out direction is returned to the policy runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Protocol

import torch

from .contracts import BT_MLE, CANONICAL_LEARNERS, PRORM_PLUS
from .data import CandidateNode
from .experiment import EvaluationTensorData
from .linear import FisherSolveDType, resolve_fisher_solve_dtype
from .metrics import local_regret, natural_direction_metrics
from .oracle import RobustOracleTransform

PHASE2_HELDOUT_SCHEMA = "phase2-heldout-fixed-beta/v1"
PHASE2_HELDOUT_STATE_SCHEMA = "phase2-heldout-frozen-state/v1"
PHASE2_HELDOUT_INPUT_SCHEMA = "phase2-deferred-heldout-input/v1"
HELDOUT_SPLIT_ORDER = ("validation", "test")
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(bytes(tensor.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


def _head_sha256(values: Sequence[float]) -> str:
    return _canonical_sha256([float(value) for value in values])


@dataclass(frozen=True, slots=True)
class DeferredHeldoutSplit:
    """One target-free saved split whose integrity is checked again at reveal."""

    split: str
    prompt_ids: tuple[str, ...]
    policy_scores: torch.Tensor
    reward_features: torch.Tensor
    candidates: tuple[CandidateNode, ...]
    policy_scores_sha256: str = field(init=False)
    reward_features_sha256: str = field(init=False)
    candidates_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.split not in HELDOUT_SPLIT_ORDER:
            raise ValueError(f"split must be one of {HELDOUT_SPLIT_ORDER!r}")
        if (
            not isinstance(self.prompt_ids, tuple)
            or not self.prompt_ids
            or any(not isinstance(value, str) or not value for value in self.prompt_ids)
            or len(set(self.prompt_ids)) != len(self.prompt_ids)
        ):
            raise ValueError("prompt_ids must be a non-empty tuple of unique strings")
        scores = self.policy_scores
        features = self.reward_features
        if (
            not isinstance(scores, torch.Tensor)
            or scores.ndim != 3
            or scores.shape[0] != len(self.prompt_ids)
            or scores.shape[1] < 2
            or scores.shape[2] < 1
            or not scores.is_floating_point()
            or scores.requires_grad
            or not bool(torch.isfinite(scores).all())
        ):
            raise ValueError("policy_scores must be a finite frozen (P,M,D) floating tensor")
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 3
            or features.shape[:2] != scores.shape[:2]
            or features.shape[2] < 1
            or not features.is_floating_point()
            or features.requires_grad
            or not bool(torch.isfinite(features).all())
            or features.dtype != scores.dtype
            or features.device != scores.device
        ):
            raise ValueError(
                "reward_features must be a finite frozen (P,M,H) tensor matching policy_scores"
            )
        # A private snapshot prevents mutation of the artifact object that was
        # passed to the constructor.  Stored hashes detect later in-place edits.
        scores = scores.detach().to(device="cpu").contiguous().clone()
        features = features.detach().to(device="cpu").contiguous().clone()
        object.__setattr__(self, "policy_scores", scores)
        object.__setattr__(self, "reward_features", features)

        expected_candidates = scores.shape[0] * scores.shape[1]
        if (
            not isinstance(self.candidates, tuple)
            or len(self.candidates) != expected_candidates
            or any(not isinstance(candidate, CandidateNode) for candidate in self.candidates)
        ):
            raise ValueError("candidates must contain exactly one CandidateNode per saved node")
        expected_prompt_order = tuple(
            prompt_id for prompt_id in self.prompt_ids for _ in range(scores.shape[1])
        )
        if tuple(candidate.prompt_id for candidate in self.candidates) != expected_prompt_order:
            raise ValueError("held-out candidates must be prompt-major in tensor order")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("held-out candidate IDs must be unique within a split")

        object.__setattr__(self, "policy_scores_sha256", _tensor_sha256(scores))
        object.__setattr__(self, "reward_features_sha256", _tensor_sha256(features))
        object.__setattr__(
            self,
            "candidates_sha256",
            _canonical_sha256([candidate.to_dict() for candidate in self.candidates]),
        )

    @classmethod
    def from_evaluation_tensor(
        cls,
        split: str,
        tensor_data: EvaluationTensorData,
        candidates: Sequence[CandidateNode],
    ) -> DeferredHeldoutSplit:
        """Copy only target-free geometry from an evaluation artifact object."""

        if not isinstance(tensor_data, EvaluationTensorData):
            raise TypeError("tensor_data must be EvaluationTensorData")
        return cls(
            split=split,
            prompt_ids=tuple(str(value) for value in tensor_data.prompt_ids),
            policy_scores=tensor_data.policy_scores,
            reward_features=tensor_data.reward_features,
            candidates=tuple(candidates),
        )

    @property
    def num_prompts(self) -> int:
        return self.policy_scores.shape[0]

    @property
    def num_candidates(self) -> int:
        return self.policy_scores.shape[1]

    @property
    def policy_dimension(self) -> int:
        return self.policy_scores.shape[2]

    @property
    def reward_dimension(self) -> int:
        return self.reward_features.shape[2]

    def verify_integrity(self) -> None:
        """Reject any tensor or candidate mutation before oracle allocation."""

        if _tensor_sha256(self.policy_scores) != self.policy_scores_sha256:
            raise RuntimeError(f"{self.split} held-out policy scores changed after preparation")
        if _tensor_sha256(self.reward_features) != self.reward_features_sha256:
            raise RuntimeError(f"{self.split} held-out reward features changed after preparation")
        if (
            _canonical_sha256([candidate.to_dict() for candidate in self.candidates])
            != self.candidates_sha256
        ):
            raise RuntimeError(f"{self.split} held-out candidates changed after preparation")

    def identity_payload(self) -> dict[str, object]:
        self.verify_integrity()
        return {
            "split": self.split,
            "num_prompts": self.num_prompts,
            "num_candidates": self.num_candidates,
            "policy_dimension": self.policy_dimension,
            "reward_dimension": self.reward_dimension,
            "prompt_ids_sha256": _canonical_sha256(list(self.prompt_ids)),
            "policy_scores_sha256": self.policy_scores_sha256,
            "reward_features_sha256": self.reward_features_sha256,
            "candidates_sha256": self.candidates_sha256,
            "contains_oracle_targets": False,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())


@dataclass(frozen=True, slots=True)
class DeferredHeldoutInputs:
    """Validation/test payload that makes target leakage structurally impossible."""

    validation: DeferredHeldoutSplit
    test: DeferredHeldoutSplit
    identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.validation, DeferredHeldoutSplit) or (
            self.validation.split != "validation"
        ):
            raise ValueError("validation must be a validation DeferredHeldoutSplit")
        if not isinstance(self.test, DeferredHeldoutSplit) or self.test.split != "test":
            raise ValueError("test must be a test DeferredHeldoutSplit")
        if set(self.validation.prompt_ids).intersection(self.test.prompt_ids):
            raise ValueError("validation and test prompt IDs must be disjoint")
        reference = self.validation
        for name, value in (
            ("num_candidates", self.test.num_candidates),
            ("policy_dimension", self.test.policy_dimension),
            ("reward_dimension", self.test.reward_dimension),
        ):
            if value != getattr(reference, name):
                raise ValueError(f"validation/test {name} must match")
        if (
            self.validation.policy_scores.dtype != self.test.policy_scores.dtype
            or self.validation.reward_features.dtype != self.test.reward_features.dtype
        ):
            raise ValueError("validation/test dtypes must match")
        object.__setattr__(
            self,
            "identity_sha256",
            _canonical_sha256(self.identity_payload()),
        )

    def verify_integrity(self) -> None:
        self.validation.verify_integrity()
        self.test.verify_integrity()
        if _canonical_sha256(self.identity_payload()) != self.identity_sha256:
            raise RuntimeError("deferred held-out identity changed after preparation")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": PHASE2_HELDOUT_INPUT_SCHEMA,
            "split_order": list(HELDOUT_SPLIT_ORDER),
            "validation": self.validation.identity_payload(),
            "test": self.test.identity_payload(),
            "contains_oracle_targets": False,
        }


@dataclass(frozen=True, slots=True)
class FrozenHeldoutEvaluationState:
    """Immutable proof that all train-derived policy quantities already exist."""

    source_config_hash: str
    phase2_design_sha256: str
    phase2_runtime_contract_sha256: str
    seed: int
    heads: Mapping[str, tuple[float, ...]]
    heads_sha256: str
    training_design_sha256: str
    beta_common: float
    deployment_identity: Mapping[str, object]
    deployment_identity_sha256: str = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("source_config_hash", self.source_config_hash),
            ("phase2_design_sha256", self.phase2_design_sha256),
            ("phase2_runtime_contract_sha256", self.phase2_runtime_contract_sha256),
            ("heads_sha256", self.heads_sha256),
            ("training_design_sha256", self.training_design_sha256),
        ):
            _digest(value, name=name)
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.seed > 2**63 - 1
        ):
            raise ValueError("seed must be an integer in [0, 2**63 - 1]")
        if not isinstance(self.heads, Mapping) or set(self.heads) != set(CANONICAL_LEARNERS):
            raise ValueError(f"heads must contain exactly {CANONICAL_LEARNERS!r}")
        copied_heads: dict[str, tuple[float, ...]] = {}
        for learner in CANONICAL_LEARNERS:
            head = self.heads[learner]
            if (
                not isinstance(head, tuple)
                or not head
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    for value in head
                )
            ):
                raise ValueError(f"heads[{learner!r}] must be a finite non-empty tuple")
            copied_heads[learner] = tuple(float(value) for value in head)
        object.__setattr__(self, "heads", copied_heads)
        expected_heads_hash = _canonical_sha256(
            {learner: list(copied_heads[learner]) for learner in CANONICAL_LEARNERS}
        )
        if expected_heads_hash != self.heads_sha256:
            raise ValueError("heads do not match heads_sha256")
        _finite_positive(self.beta_common, name="beta_common")
        if not isinstance(self.deployment_identity, Mapping):
            raise TypeError("deployment_identity must be a mapping")
        try:
            copied_deployment = json.loads(
                json.dumps(
                    dict(self.deployment_identity),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError("deployment_identity must be strict JSON data") from error
        if not isinstance(copied_deployment, dict):
            raise TypeError("deployment_identity must encode a JSON object")
        object.__setattr__(self, "deployment_identity", copied_deployment)
        deployment_hash = _canonical_sha256(copied_deployment)
        object.__setattr__(self, "deployment_identity_sha256", deployment_hash)
        object.__setattr__(self, "state_sha256", _canonical_sha256(self.identity_payload()))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": PHASE2_HELDOUT_STATE_SCHEMA,
            "source_config_hash": self.source_config_hash,
            "phase2_design_sha256": self.phase2_design_sha256,
            "phase2_runtime_contract_sha256": self.phase2_runtime_contract_sha256,
            "seed": self.seed,
            "heads_sha256": self.heads_sha256,
            "training_design_sha256": self.training_design_sha256,
            "beta_common": self.beta_common,
            "deployment_identity_sha256": self.deployment_identity_sha256,
            "heads_frozen": True,
            "beta_common_frozen": True,
            "deployed_directions_frozen": True,
        }


class HeldoutOracleScorer(Protocol):
    """The transformed-score-only surface exposed by the final oracle session."""

    def score_transformed(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        transform: RobustOracleTransform,
        batch_size: int,
    ) -> torch.Tensor: ...


def _absolute_damping(
    policy_scores: torch.Tensor,
    *,
    relative_damping: float,
    pcg_dtype: FisherSolveDType,
) -> float:
    relative = _finite_positive(relative_damping, name="relative_damping")
    flat = policy_scores.reshape(-1, policy_scores.shape[-1]).to(
        dtype=resolve_fisher_solve_dtype(pcg_dtype)
    )
    mean_fisher_diagonal = float(flat.square().mean(dim=0).mean().item())
    if not math.isfinite(mean_fisher_diagonal) or mean_fisher_diagonal <= 0.0:
        raise ValueError("held-out split has a degenerate node Fisher")
    value = relative * mean_fisher_diagonal
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("held-out absolute damping is not finite and positive")
    return value


def _nullable(value: torch.Tensor) -> float | None:
    result = float(value.item())
    return result if math.isfinite(result) else None


def _evaluate_split(
    split: DeferredHeldoutSplit,
    target_rewards: torch.Tensor,
    state: FrozenHeldoutEvaluationState,
    *,
    relative_damping: float,
    pcg_dtype: FisherSolveDType,
    pcg_max_iterations: int,
    pcg_tolerance: float,
) -> dict[str, object]:
    expected_shape = (split.num_prompts, split.num_candidates)
    if target_rewards.shape != expected_shape:
        raise ValueError(
            f"{split.split} transformed oracle targets must have shape {expected_shape!r}"
        )
    targets = target_rewards.detach().to(
        device=split.policy_scores.device,
        dtype=split.policy_scores.dtype,
    )
    if targets.requires_grad or not bool(torch.isfinite(targets).all()):
        raise ValueError(f"{split.split} transformed oracle targets must be finite and detached")
    damping = _absolute_damping(
        split.policy_scores,
        relative_damping=relative_damping,
        pcg_dtype=pcg_dtype,
    )
    learners: dict[str, dict[str, object]] = {}
    for learner in CANONICAL_LEARNERS:
        head = torch.tensor(
            state.heads[learner],
            dtype=split.reward_features.dtype,
            device=split.reward_features.device,
        )
        if head.shape != (split.reward_dimension,):
            raise ValueError(
                f"frozen head {learner!r} has shape {tuple(head.shape)!r}; "
                f"expected ({split.reward_dimension},)"
            )
        predicted = split.reward_features @ head
        regret = local_regret(
            split.policy_scores,
            predicted,
            targets,
            damping=damping,
            beta=state.beta_common,
            pcg_tolerance=pcg_tolerance,
            pcg_max_iterations=pcg_max_iterations,
            pcg_dtype=pcg_dtype,
        )
        directions = natural_direction_metrics(
            split.policy_scores,
            predicted,
            targets,
            damping=damping,
            pcg_tolerance=pcg_tolerance,
            pcg_max_iterations=pcg_max_iterations,
            pcg_dtype=pcg_dtype,
        )
        learners[learner] = {
            "head_sha256": _head_sha256(state.heads[learner]),
            "local_regret_at_frozen_global_beta": float(regret.item()),
            "native_beta1_squared_fisher_direction_error": float(
                directions.squared_fisher_error.item()
            ),
            "native_beta1_fisher_cosine": _nullable(directions.fisher_cosine),
            "native_beta1_predicted_fisher_norm": float(directions.predicted_fisher_norm.item()),
            "native_beta1_target_fisher_norm": float(directions.target_fisher_norm.item()),
            "direction_vectors_serialized": False,
        }

    bt = learners[BT_MLE]
    prorm = learners[PRORM_PLUS]

    def difference(field_name: str) -> float | None:
        left = prorm[field_name]
        right = bt[field_name]
        if left is None or right is None:
            return None
        return float(left) - float(right)

    return {
        "input_identity": split.identity_payload(),
        "input_identity_sha256": split.identity_sha256,
        "transformed_oracle_rewards_sha256": _tensor_sha256(targets),
        "raw_oracle_logits_serialized": False,
        "node_fisher_estimator": "mean_all_saved_split_nodes",
        "moment_estimator": "per_prompt_unbiased_candidate_covariance",
        "relative_damping": float(relative_damping),
        "absolute_damping": damping,
        "fixed_beta": state.beta_common,
        "fixed_beta_source": "pilot_selected_global_beta_frozen_in_confirmatory_design",
        "learners": learners,
        "prorm_plus_minus_bt_mle": {
            "local_regret_at_frozen_global_beta": difference("local_regret_at_frozen_global_beta"),
            "native_beta1_squared_fisher_direction_error": difference(
                "native_beta1_squared_fisher_direction_error"
            ),
            "native_beta1_fisher_cosine": difference("native_beta1_fisher_cosine"),
        },
    }


def score_and_evaluate_deferred_heldout(
    oracle: HeldoutOracleScorer,
    deferred: DeferredHeldoutInputs,
    state: FrozenHeldoutEvaluationState,
    *,
    transform: RobustOracleTransform,
    oracle_chat_template_sha256: str,
    batch_size: int,
    relative_damping: float,
    pcg_dtype: FisherSolveDType,
    pcg_max_iterations: int,
    pcg_tolerance: float,
) -> dict[str, object]:
    """Reveal fresh held-out targets and immediately reduce them to metrics.

    This function must be called from the already-open final oracle session.
    It returns no target vector or policy direction, so neither can flow back
    into calibration or deployment.
    """

    if not isinstance(deferred, DeferredHeldoutInputs):
        raise TypeError("deferred must be DeferredHeldoutInputs")
    if not isinstance(state, FrozenHeldoutEvaluationState):
        raise TypeError("state must be FrozenHeldoutEvaluationState")
    if not isinstance(transform, RobustOracleTransform):
        raise TypeError("transform must be RobustOracleTransform")
    _digest(oracle_chat_template_sha256, name="oracle_chat_template_sha256")
    _positive_integer(batch_size, name="batch_size")
    _finite_positive(relative_damping, name="relative_damping")
    resolve_fisher_solve_dtype(pcg_dtype)
    _positive_integer(pcg_max_iterations, name="pcg_max_iterations")
    _finite_positive(pcg_tolerance, name="pcg_tolerance")
    deferred.verify_integrity()

    splits = (deferred.validation, deferred.test)
    ordered_candidates = tuple(candidate for split in splits for candidate in split.candidates)
    transformed = oracle.score_transformed(
        tuple(candidate.prompt for candidate in ordered_candidates),
        tuple(candidate.response for candidate in ordered_candidates),
        transform=transform,
        batch_size=batch_size,
    )
    if (
        not isinstance(transformed, torch.Tensor)
        or transformed.shape != (len(ordered_candidates),)
        or not transformed.is_floating_point()
        or transformed.requires_grad
        or not bool(torch.isfinite(transformed).all())
    ):
        raise ValueError(
            "final held-out oracle rescore must return one finite detached value per saved node"
        )
    transformed = transformed.detach().to(device="cpu").clone()
    offset = 0
    split_results: dict[str, object] = {}
    for split in splits:
        size = split.num_prompts * split.num_candidates
        targets = transformed[offset : offset + size].reshape(
            split.num_prompts,
            split.num_candidates,
        )
        split_results[split.split] = _evaluate_split(
            split,
            targets,
            state,
            relative_damping=relative_damping,
            pcg_dtype=pcg_dtype,
            pcg_max_iterations=pcg_max_iterations,
            pcg_tolerance=pcg_tolerance,
        )
        offset += size
    if offset != len(ordered_candidates):
        raise RuntimeError("held-out oracle rescore partition did not consume every node")

    payload: dict[str, object] = {
        "schema_version": PHASE2_HELDOUT_SCHEMA,
        "estimand": "frozen_global_common_beta_local_regret",
        "formal_gate_split": "test",
        "descriptive_split": "validation",
        "split_order": list(HELDOUT_SPLIT_ORDER),
        "beta_common": state.beta_common,
        "frozen_state": state.identity_payload(),
        "frozen_state_sha256": state.state_sha256,
        "deferred_input_sha256": deferred.identity_sha256,
        "oracle_rescore": {
            "source": "saved_validation_and_test_candidates_rescored_after_policy_freeze",
            "oracle_chat_template_sha256": oracle_chat_template_sha256,
            "transform": {"b": transform.b, "tau": transform.tau},
            "combined_transformed_rewards_sha256": _tensor_sha256(transformed),
            "raw_oracle_logits_serialized": False,
        },
        "solver": {
            "pcg_dtype": str(pcg_dtype),
            "pcg_max_iterations": pcg_max_iterations,
            "pcg_tolerance": float(pcg_tolerance),
            "relative_damping": float(relative_damping),
            "split_specific_node_fisher_and_damping": True,
        },
        "splits": split_results,
        "information_boundary": {
            "fresh_targets_created_after_heads_beta_and_deployments_frozen": True,
            "validation_or_test_targets_available_to_head_trainer": False,
            "validation_or_test_targets_available_to_beta_calibration": False,
            "validation_or_test_targets_available_to_policy_deployment": False,
            "heldout_direction_used_for_policy": False,
        },
        "raw_oracle_logits_serialized": False,
        "heldout_direction_vectors_serialized": False,
    }
    # Fail now if a future edit adds a non-JSON value or NaN.
    _canonical_sha256(payload)
    return payload


def heldout_evaluation_sha256(payload: Mapping[str, object]) -> str:
    """Return the canonical identity of one complete held-out metric payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    return _canonical_sha256(dict(payload))


def verify_heldout_evaluation_payload(
    payload: Mapping[str, object],
    *,
    expected_sha256: str,
) -> None:
    """Verify the strict top-level contract and canonical result identity."""

    expected = _digest(expected_sha256, name="expected_sha256")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    required = {
        "schema_version",
        "estimand",
        "formal_gate_split",
        "descriptive_split",
        "split_order",
        "beta_common",
        "frozen_state",
        "frozen_state_sha256",
        "deferred_input_sha256",
        "oracle_rescore",
        "solver",
        "splits",
        "information_boundary",
        "raw_oracle_logits_serialized",
        "heldout_direction_vectors_serialized",
    }
    if set(payload) != required:
        raise ValueError("held-out evaluation payload fields do not match the strict schema")
    if payload["schema_version"] != PHASE2_HELDOUT_SCHEMA:
        raise ValueError("held-out evaluation schema version is invalid")
    if payload["formal_gate_split"] != "test" or payload["split_order"] != [
        "validation",
        "test",
    ]:
        raise ValueError("held-out evaluation split contract is invalid")
    splits = payload["splits"]
    if not isinstance(splits, Mapping) or set(splits) != set(HELDOUT_SPLIT_ORDER):
        raise ValueError("held-out evaluation must contain validation and test")
    boundary = payload["information_boundary"]
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not expected_value
        for key, expected_value in {
            "fresh_targets_created_after_heads_beta_and_deployments_frozen": True,
            "validation_or_test_targets_available_to_head_trainer": False,
            "validation_or_test_targets_available_to_beta_calibration": False,
            "validation_or_test_targets_available_to_policy_deployment": False,
            "heldout_direction_used_for_policy": False,
        }.items()
    ):
        raise ValueError("held-out evaluation violates the information boundary")
    if (
        payload["raw_oracle_logits_serialized"] is not False
        or payload["heldout_direction_vectors_serialized"] is not False
    ):
        raise ValueError("held-out evaluation serialized a forbidden raw quantity")
    if heldout_evaluation_sha256(payload) != expected:
        raise ValueError("held-out evaluation payload SHA256 mismatch")


__all__ = [
    "HELDOUT_SPLIT_ORDER",
    "PHASE2_HELDOUT_INPUT_SCHEMA",
    "PHASE2_HELDOUT_SCHEMA",
    "PHASE2_HELDOUT_STATE_SCHEMA",
    "DeferredHeldoutInputs",
    "DeferredHeldoutSplit",
    "FrozenHeldoutEvaluationState",
    "HeldoutOracleScorer",
    "heldout_evaluation_sha256",
    "score_and_evaluate_deferred_heldout",
    "verify_heldout_evaluation_payload",
]
