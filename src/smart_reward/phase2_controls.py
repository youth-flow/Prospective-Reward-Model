"""Pure, leakage-safe positive-control primitives for the common-beta campaign.

This module deliberately stops at immutable tensor/data construction.  It does
not load Hugging Face models, train a reward head, choose ``beta``, or run a
policy.  The three label arms have distinct contracts:

* :func:`build_exact_margin_canonical_arm` replaces only the ProRM target on
  the canonical candidate-0-minus-candidate-1 edge.  Raw train node rewards are
  accepted transiently, fingerprinted, and never retained.
* :func:`sample_canonical_r4_noisy_arm` keeps four independent geometric
  truncation replicates intact.  ProRM receives the mean of the four separate
  unbiased estimates, while BT receives pooled *raw* wins and totals.
* :func:`sample_all_six_prompt_u_stat_arm` uses all six unordered pairs of four
  iid candidates.  The six edges share nodes and are explicitly identified as
  one prompt cluster, never as six independent experimental units.

The low-dimensional tangent helpers require a named full-coordinate layout and
explicit coordinate indices.  They return fresh tensor storage and enforce
``d_selected < n_F``, where ``n_F`` is the number of Fisher nodes.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .annotations import (
    ReplicatedRepeatedLabelBatch,
    sample_replicated_geometric_repeated_labels,
)
from .experiment import TrainingTensorData
from .linear import FisherSolveDType, resolve_fisher_solve_dtype
from .metrics import policy_reward_moment
from .rollout import PolicyDirectionResult, policy_direction_from_node_rewards
from .training import FeatureTrainingBatch

CANONICAL_PAIR: tuple[int, int] = (0, 1)
ALL_SIX_PAIR_LAYOUT: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)
PRIMARY_NUM_REPLICATES = 4
PRIMARY_GAMMA = 0.9
ORTHONORMAL_PROJECTION_ALGORITHM = "gaussian_qr_sign_canonical_v1"

_PromptId = str | int
_CoordinateId = str | int


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's logical dtype, shape, and values without serializing it."""

    value = tensor.detach().to(device="cpu").contiguous().clone()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(bytes(value.untyped_storage()))
    return digest.hexdigest()


def _validate_frozen_float_tensor(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} must be frozen and detached")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_probabilities(
    probabilities: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    value = _validate_frozen_float_tensor(
        "left_win_probabilities",
        probabilities,
        shape=shape,
        device=device,
    )
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError("left_win_probabilities must lie in [0, 1]")
    return value


def _clone_training(
    source: TrainingTensorData,
    *,
    h: torch.Tensor,
    left_wins: torch.Tensor,
    num_annotations: torch.Tensor,
) -> TrainingTensorData:
    """Construct a storage-independent standard training object."""

    return TrainingTensorData(
        prompt_ids=tuple(source.prompt_ids),
        policy_scores=source.policy_scores.detach().clone(),
        reward_features=source.reward_features.detach().clone(),
        h=h.detach().clone(),
        left_wins=left_wins.detach().clone(),
        num_annotations=num_annotations.detach().clone(),
    )


@dataclass(frozen=True, slots=True)
class PromptPairEdge:
    """Immutable identity and orientation for one prompt-local candidate edge."""

    prompt_id: _PromptId
    prompt_index: int
    edge_index_within_prompt: int
    left_candidate_index: int
    right_candidate_index: int

    def __post_init__(self) -> None:
        if isinstance(self.prompt_id, bool) or not isinstance(self.prompt_id, (str, int)):
            raise TypeError("prompt_id must be a string or non-boolean integer")
        for name in (
            "prompt_index",
            "edge_index_within_prompt",
            "left_candidate_index",
            "right_candidate_index",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.left_candidate_index >= self.right_candidate_index:
            raise ValueError("candidate-pair orientation must satisfy left < right")

    @property
    def edge_id(self) -> tuple[_PromptId, int, int]:
        """Collision-free edge ID: ``(prompt_id, left_index, right_index)``."""

        return (
            self.prompt_id,
            self.left_candidate_index,
            self.right_candidate_index,
        )

    @property
    def left_node_id(self) -> tuple[_PromptId, int]:
        return (self.prompt_id, self.left_candidate_index)

    @property
    def right_node_id(self) -> tuple[_PromptId, int]:
        return (self.prompt_id, self.right_candidate_index)


def _edges_for_layout(
    prompt_ids: tuple[_PromptId, ...],
    pair_layout: tuple[tuple[int, int], ...],
) -> tuple[PromptPairEdge, ...]:
    return tuple(
        PromptPairEdge(
            prompt_id=prompt_id,
            prompt_index=prompt_index,
            edge_index_within_prompt=edge_index,
            left_candidate_index=left,
            right_candidate_index=right,
        )
        for prompt_index, prompt_id in enumerate(prompt_ids)
        for edge_index, (left, right) in enumerate(pair_layout)
    )


@dataclass(frozen=True, slots=True)
class ExactMarginAudit:
    """Non-invertible binding evidence for a transient train-oracle input."""

    source_node_rewards_sha256: str
    exact_margin_sha256: str
    source_shape: tuple[int, int]
    orientation: Literal["candidate_0_minus_candidate_1"] = "candidate_0_minus_candidate_1"
    raw_node_rewards_retained: Literal[False] = False
    bt_counts_source: Literal["input_training_passthrough"] = "input_training_passthrough"
    purpose: Literal["zero_label_noise_reward_head_training_control"] = (
        "zero_label_noise_reward_head_training_control"
    )
    reward_head_fit_required: Literal[True] = True
    oracle_direction_identity_expected: Literal[False] = False

    def __post_init__(self) -> None:
        for name in ("source_node_rewards_sha256", "exact_margin_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest")
        if (
            not isinstance(self.source_shape, tuple)
            or len(self.source_shape) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 1
                for size in self.source_shape
            )
        ):
            raise ValueError("source_shape must contain two positive dimensions")
        if self.orientation != "candidate_0_minus_candidate_1":
            raise ValueError("invalid exact-margin orientation")
        if self.raw_node_rewards_retained is not False:
            raise ValueError("raw node rewards must never be retained")
        if self.bt_counts_source != "input_training_passthrough":
            raise ValueError("exact-margin BT counts must be input passthrough")
        if (
            self.purpose != "zero_label_noise_reward_head_training_control"
            or self.reward_head_fit_required is not True
            or self.oracle_direction_identity_expected is not False
        ):
            raise ValueError("invalid exact-margin control interpretation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "exact-margin-audit/v1",
            "source_node_rewards_sha256": self.source_node_rewards_sha256,
            "exact_margin_sha256": self.exact_margin_sha256,
            "source_shape": list(self.source_shape),
            "orientation": self.orientation,
            "raw_node_rewards_retained": self.raw_node_rewards_retained,
            "bt_counts_source": self.bt_counts_source,
            "purpose": self.purpose,
            "reward_head_fit_required": self.reward_head_fit_required,
            "oracle_direction_identity_expected": self.oracle_direction_identity_expected,
        }


@dataclass(frozen=True, slots=True)
class ExactMarginCanonicalArm:
    """Canonical 0-minus-1 training data with an exact ProRM margin target."""

    training: TrainingTensorData
    edges: tuple[PromptPairEdge, ...]
    audit: ExactMarginAudit

    def __post_init__(self) -> None:
        expected_edges = _edges_for_layout(self.training.prompt_ids, (CANONICAL_PAIR,))
        if self.edges != expected_edges:
            raise ValueError("edges must be the prompt-major canonical 0-1 layout")
        if self.audit.source_shape != (
            self.training.num_prompts,
            self.training.num_candidates,
        ):
            raise ValueError("audit source_shape does not match training nodes")
        if self.audit.exact_margin_sha256 != _tensor_sha256(self.training.h):
            raise ValueError("audit exact-margin digest does not match training.h")


def build_exact_margin_canonical_arm(
    training: TrainingTensorData,
    train_node_oracle_rewards: torch.Tensor,
) -> ExactMarginCanonicalArm:
    """Build the zero-label-noise canonical positive control.

    ``train_node_oracle_rewards`` exists only for the duration of this call.
    The returned standard :class:`TrainingTensorData` contains the identifiable
    margin ``r*(x,y_0)-r*(x,y_1)`` but has no node-reward channel.  The input
    BT wins/totals are copied unchanged so this control changes only ProRM's
    target; the audit records that fact explicitly.
    """

    if not isinstance(training, TrainingTensorData):
        raise TypeError("training must be a TrainingTensorData")
    rewards = _validate_frozen_float_tensor(
        "train_node_oracle_rewards",
        train_node_oracle_rewards,
        shape=(training.num_prompts, training.num_candidates),
        device=training.policy_scores.device,
        dtype=training.policy_scores.dtype,
    )
    exact_margin = rewards[:, 0] - rewards[:, 1]
    output = _clone_training(
        training,
        h=exact_margin,
        left_wins=training.left_wins,
        num_annotations=training.num_annotations,
    )
    audit = ExactMarginAudit(
        source_node_rewards_sha256=_tensor_sha256(rewards),
        exact_margin_sha256=_tensor_sha256(exact_margin),
        source_shape=tuple(rewards.shape),
    )
    return ExactMarginCanonicalArm(
        training=output,
        edges=_edges_for_layout(training.prompt_ids, (CANONICAL_PAIR,)),
        audit=audit,
    )


@dataclass(frozen=True, slots=True)
class DirectOracleGeometryControl:
    """Algebraic oracle-moment identity and its native natural direction.

    This is intentionally distinct from :class:`ExactMarginCanonicalArm`.
    Here the restricted reward head is bypassed: the complete-pair moment and
    all-node sample-covariance moment are two algebraically identical
    constructions from the same frozen train nodes.  Consequently this object
    is a geometry/solver positive control, not a claim that a misspecified
    linear reward class must reproduce the oracle direction after training.
    """

    canonical_margins: torch.Tensor
    canonical_pair_moment: torch.Tensor
    complete_pair_u_stat_moment: torch.Tensor
    all_node_covariance_moment: torch.Tensor
    native_oracle_direction: PolicyDirectionResult
    source_node_rewards_sha256: str
    num_prompts: int
    num_candidates: int
    reward_head_bypassed: Literal[True] = True
    complete_pair_identity_is_algebraic: Literal[True] = True
    trained_exact_margin_head_required_to_match: Literal[False] = False
    raw_node_rewards_retained: Literal[False] = False

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_margins, torch.Tensor) or self.canonical_margins.shape != (
            self.num_prompts,
        ):
            raise ValueError("canonical_margins must have shape (num_prompts,)")
        reference = self.all_node_covariance_moment
        if not isinstance(reference, torch.Tensor) or reference.ndim != 1:
            raise ValueError("all_node_covariance_moment must be one-dimensional")
        for name, value in (
            ("canonical_margins", self.canonical_margins),
            ("canonical_pair_moment", self.canonical_pair_moment),
            ("complete_pair_u_stat_moment", self.complete_pair_u_stat_moment),
            ("all_node_covariance_moment", self.all_node_covariance_moment),
        ):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor")
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError(f"{name} must be frozen and detached")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("all direct-oracle tensors must share dtype and device")
        expected_moment_shape = reference.shape
        if self.canonical_pair_moment.shape != expected_moment_shape:
            raise ValueError("canonical_pair_moment has the wrong policy dimension")
        if self.complete_pair_u_stat_moment.shape != expected_moment_shape:
            raise ValueError("complete_pair_u_stat_moment has the wrong policy dimension")
        if self.native_oracle_direction.direction.shape != expected_moment_shape:
            raise ValueError("native oracle direction has the wrong policy dimension")
        if (
            self.native_oracle_direction.direction.dtype != reference.dtype
            or self.native_oracle_direction.direction.device != reference.device
        ):
            raise ValueError("native oracle direction must share moment dtype and device")
        if self.native_oracle_direction.beta != 1.0:
            raise ValueError("native oracle direction must be constructed at beta=1")
        if (
            isinstance(self.num_prompts, bool)
            or not isinstance(self.num_prompts, int)
            or self.num_prompts < 1
            or isinstance(self.num_candidates, bool)
            or not isinstance(self.num_candidates, int)
            or self.num_candidates < 2
        ):
            raise ValueError("direct-oracle node dimensions are invalid")
        if (
            not isinstance(self.source_node_rewards_sha256, str)
            or len(self.source_node_rewards_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.source_node_rewards_sha256
            )
        ):
            raise ValueError("source_node_rewards_sha256 must be a lowercase SHA256 digest")
        if (
            self.reward_head_bypassed is not True
            or self.complete_pair_identity_is_algebraic is not True
            or self.trained_exact_margin_head_required_to_match is not False
            or self.raw_node_rewards_retained is not False
        ):
            raise ValueError("invalid direct-oracle control interpretation")

        tolerance = 2.0e-12 if reference.dtype == torch.float64 else 2.0e-5
        if not torch.allclose(
            self.complete_pair_u_stat_moment,
            reference,
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError(
                "complete-pair U-stat moment must equal the all-node covariance moment"
            )
        expected_moment_norm = float(torch.linalg.vector_norm(reference).item())
        if not math.isclose(
            self.native_oracle_direction.moment_norm,
            expected_moment_norm,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError("native oracle direction evidence uses a different moment")

    @property
    def identity_absolute_error(self) -> float:
        return float(
            torch.linalg.vector_norm(
                self.complete_pair_u_stat_moment - self.all_node_covariance_moment
            ).item()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "direct-oracle-geometry-control/v1",
            "source_node_rewards_sha256": self.source_node_rewards_sha256,
            "num_prompts": self.num_prompts,
            "num_candidates": self.num_candidates,
            "policy_dimension": self.all_node_covariance_moment.numel(),
            "canonical_margin_sha256": _tensor_sha256(self.canonical_margins),
            "canonical_pair_moment": self.canonical_pair_moment.detach().cpu().tolist(),
            "complete_pair_u_stat_moment": (
                self.complete_pair_u_stat_moment.detach().cpu().tolist()
            ),
            "all_node_covariance_moment": (self.all_node_covariance_moment.detach().cpu().tolist()),
            "complete_pair_identity_absolute_error": self.identity_absolute_error,
            "native_oracle_direction": self.native_oracle_direction.to_dict(),
            "reward_head_bypassed": self.reward_head_bypassed,
            "complete_pair_identity_is_algebraic": (self.complete_pair_identity_is_algebraic),
            "trained_exact_margin_head_required_to_match": (
                self.trained_exact_margin_head_required_to_match
            ),
            "raw_node_rewards_retained": self.raw_node_rewards_retained,
        }


def _complete_pair_oracle_moment(
    policy_scores: torch.Tensor,
    node_rewards: torch.Tensor,
) -> torch.Tensor:
    num_candidates = policy_scores.shape[1]
    pair_layout = tuple(
        (left, right) for left in range(num_candidates) for right in range(left + 1, num_candidates)
    )
    left_indices = torch.tensor(
        [left for left, _ in pair_layout],
        dtype=torch.int64,
        device=policy_scores.device,
    )
    right_indices = torch.tensor(
        [right for _, right in pair_layout],
        dtype=torch.int64,
        device=policy_scores.device,
    )
    edge_scores = policy_scores[:, left_indices, :] - policy_scores[:, right_indices, :]
    margins = node_rewards[:, left_indices] - node_rewards[:, right_indices]
    # Half the uniform-unordered-pair moment equals the Bessel-corrected
    # per-prompt sample covariance exactly, for every M >= 2.
    return 0.5 * (edge_scores * margins.unsqueeze(-1)).mean(dim=(0, 1))


def build_direct_oracle_geometry_control(
    training: TrainingTensorData,
    train_node_oracle_rewards: torch.Tensor,
    *,
    relative_damping: float,
    pcg_dtype: FisherSolveDType = "float64",
    pcg_max_iterations: int = 200,
    pcg_tolerance: float = 1.0e-6,
    pcg_absolute_tolerance: float = 0.0,
    pcg_residual_recompute_interval: int = 20,
    require_pcg_convergence: bool = True,
) -> DirectOracleGeometryControl:
    """Build an all-node oracle geometry control without fitting a reward head."""

    if not isinstance(training, TrainingTensorData):
        raise TypeError("training must be a TrainingTensorData")
    rewards = _validate_frozen_float_tensor(
        "train_node_oracle_rewards",
        train_node_oracle_rewards,
        shape=(training.num_prompts, training.num_candidates),
        device=training.policy_scores.device,
        dtype=training.policy_scores.dtype,
    )
    solve_dtype = resolve_fisher_solve_dtype(pcg_dtype)
    scores_for_moment = training.policy_scores.to(dtype=solve_dtype)
    rewards_for_moment = rewards.to(dtype=solve_dtype)
    canonical_margins = (rewards_for_moment[:, 0] - rewards_for_moment[:, 1]).detach()
    canonical_pair_moment = (
        0.5
        * (
            (scores_for_moment[:, 0] - scores_for_moment[:, 1]) * canonical_margins.unsqueeze(1)
        ).mean(dim=0)
    ).detach()
    complete_pair_moment = _complete_pair_oracle_moment(
        scores_for_moment,
        rewards_for_moment,
    ).detach()
    all_node_moment = policy_reward_moment(
        scores_for_moment,
        rewards_for_moment,
        center_candidates=True,
        candidate_dim=1,
    ).detach()
    direction = policy_direction_from_node_rewards(
        training,
        rewards,
        relative_damping=relative_damping,
        beta=1.0,
        pcg_dtype=pcg_dtype,
        pcg_max_iterations=pcg_max_iterations,
        pcg_tolerance=pcg_tolerance,
        pcg_absolute_tolerance=pcg_absolute_tolerance,
        pcg_residual_recompute_interval=pcg_residual_recompute_interval,
        require_pcg_convergence=require_pcg_convergence,
    )
    return DirectOracleGeometryControl(
        canonical_margins=canonical_margins.detach().clone(),
        canonical_pair_moment=canonical_pair_moment.detach().clone(),
        complete_pair_u_stat_moment=complete_pair_moment.detach().clone(),
        all_node_covariance_moment=all_node_moment.detach().clone(),
        native_oracle_direction=direction,
        source_node_rewards_sha256=_tensor_sha256(rewards),
        num_prompts=training.num_prompts,
        num_candidates=training.num_candidates,
    )


@dataclass(frozen=True, slots=True)
class R4LabelAudit:
    """Auditable target routing and annotation cost for a primary noisy arm."""

    probability_sha256: str
    edge_shape: tuple[int, ...]
    realized_total_annotations: int
    expected_annotations_per_edge: float = 40.0
    num_replicates: Literal[4] = 4
    gamma: Literal[0.9] = 0.9
    replicate_boundaries_preserved: Literal[True] = True
    prorm_target: Literal["mean_of_per_replicate_h"] = "mean_of_per_replicate_h"
    bt_target: Literal["pooled_raw_wins_and_totals"] = "pooled_raw_wins_and_totals"
    pooled_counts_reused_as_one_truncation: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.probability_sha256, str)
            or len(self.probability_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.probability_sha256)
        ):
            raise ValueError("probability_sha256 must be a lowercase SHA256 digest")
        if (
            not isinstance(self.edge_shape, tuple)
            or not self.edge_shape
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 1
                for size in self.edge_shape
            )
        ):
            raise ValueError("edge_shape must contain positive dimensions")
        if (
            isinstance(self.realized_total_annotations, bool)
            or not isinstance(self.realized_total_annotations, int)
            or self.realized_total_annotations < math.prod(self.edge_shape) * 4
        ):
            raise ValueError("realized annotation cost is smaller than four labels per edge")
        if self.expected_annotations_per_edge != 40.0:
            raise ValueError("the R=4, gamma=0.9 expected cost must equal 40 per edge")
        if self.num_replicates != PRIMARY_NUM_REPLICATES or self.gamma != PRIMARY_GAMMA:
            raise ValueError("the primary noisy arm is locked to R=4 and gamma=0.9")
        if (
            self.replicate_boundaries_preserved is not True
            or self.pooled_counts_reused_as_one_truncation is not False
            or self.prorm_target != "mean_of_per_replicate_h"
            or self.bt_target != "pooled_raw_wins_and_totals"
        ):
            raise ValueError("invalid R=4 target-routing contract")

    @property
    def realized_annotations_per_edge(self) -> float:
        return self.realized_total_annotations / math.prod(self.edge_shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "r4-label-audit/v1",
            "probability_sha256": self.probability_sha256,
            "edge_shape": list(self.edge_shape),
            "num_replicates": self.num_replicates,
            "gamma": self.gamma,
            "expected_annotations_per_edge": self.expected_annotations_per_edge,
            "realized_total_annotations": self.realized_total_annotations,
            "realized_annotations_per_edge": self.realized_annotations_per_edge,
            "replicate_boundaries_preserved": self.replicate_boundaries_preserved,
            "prorm_target": self.prorm_target,
            "bt_target": self.bt_target,
            "pooled_counts_reused_as_one_truncation": (self.pooled_counts_reused_as_one_truncation),
        }


def _r4_audit(
    probabilities: torch.Tensor,
    repeated_labels: ReplicatedRepeatedLabelBatch,
) -> R4LabelAudit:
    return R4LabelAudit(
        probability_sha256=_tensor_sha256(probabilities),
        edge_shape=tuple(probabilities.shape),
        realized_total_annotations=repeated_labels.total_annotations,
    )


def _validate_primary_repeated_labels(
    repeated_labels: ReplicatedRepeatedLabelBatch,
    *,
    edge_shape: tuple[int, ...],
) -> None:
    if not isinstance(repeated_labels, ReplicatedRepeatedLabelBatch):
        raise TypeError("repeated_labels must be a ReplicatedRepeatedLabelBatch")
    if repeated_labels.num_replicates != PRIMARY_NUM_REPLICATES:
        raise ValueError("primary noisy arms require exactly four independent replicates")
    if repeated_labels.gamma != PRIMARY_GAMMA:
        raise ValueError("primary noisy arms require gamma=0.9")
    if repeated_labels.mean_h.shape != edge_shape:
        raise ValueError("repeated-label edge shape does not match the arm")


@dataclass(frozen=True, slots=True)
class CanonicalR4NoisyArm:
    """Canonical noisy arm with separate ProRM and repeated-label BT targets."""

    training: TrainingTensorData
    repeated_labels: ReplicatedRepeatedLabelBatch
    edges: tuple[PromptPairEdge, ...]
    audit: R4LabelAudit

    def __post_init__(self) -> None:
        edge_shape = (self.training.num_prompts,)
        _validate_primary_repeated_labels(self.repeated_labels, edge_shape=edge_shape)
        if self.edges != _edges_for_layout(self.training.prompt_ids, (CANONICAL_PAIR,)):
            raise ValueError("edges must be the prompt-major canonical 0-1 layout")
        expected_h = self.repeated_labels.mean_h.to(dtype=self.training.h.dtype)
        if not torch.equal(self.training.h, expected_h):
            raise ValueError("training.h must equal the mean of four separate h estimates")
        if not torch.equal(self.training.left_wins, self.repeated_labels.pooled_wins):
            raise ValueError("training.left_wins must pool all raw replicate wins")
        if not torch.equal(
            self.training.num_annotations,
            self.repeated_labels.pooled_totals,
        ):
            raise ValueError("training.num_annotations must pool all raw replicate totals")
        if self.audit.edge_shape != edge_shape:
            raise ValueError("audit edge shape does not match canonical training data")
        if self.audit.realized_total_annotations != self.repeated_labels.total_annotations:
            raise ValueError("audit annotation cost does not match raw labels")


def sample_canonical_r4_noisy_arm(
    training: TrainingTensorData,
    left_win_probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    max_total_annotations: int | None = None,
) -> CanonicalR4NoisyArm:
    """Sample the locked ``R=4``, ``gamma=0.9`` canonical noisy arm."""

    if not isinstance(training, TrainingTensorData):
        raise TypeError("training must be a TrainingTensorData")
    probabilities = _validate_probabilities(
        left_win_probabilities,
        shape=(training.num_prompts,),
        device=training.policy_scores.device,
    )
    repeated = sample_replicated_geometric_repeated_labels(
        probabilities,
        num_replicates=PRIMARY_NUM_REPLICATES,
        gamma=PRIMARY_GAMMA,
        generator=generator,
        max_total_annotations=max_total_annotations,
    )
    output = _clone_training(
        training,
        h=repeated.mean_h.to(dtype=training.policy_scores.dtype),
        left_wins=repeated.pooled_wins,
        num_annotations=repeated.pooled_totals,
    )
    return CanonicalR4NoisyArm(
        training=output,
        repeated_labels=repeated,
        edges=_edges_for_layout(training.prompt_ids, (CANONICAL_PAIR,)),
        audit=_r4_audit(probabilities, repeated),
    )


@dataclass(frozen=True, slots=True)
class PromptUStatisticMetadata:
    """Statistical dependence and work accounting for the all-six arm."""

    prompt_ids: tuple[_PromptId, ...]
    pair_layout: tuple[tuple[int, int], ...] = ALL_SIX_PAIR_LAYOUT
    candidates_per_prompt: Literal[4] = 4
    edges_per_prompt: Literal[6] = 6
    cluster_unit: Literal["prompt"] = "prompt"
    construction: Literal["complete_unordered_pair_u_statistic"] = (
        "complete_unordered_pair_u_statistic"
    )
    candidate_nodes_iid_within_prompt: Literal[True] = True
    edges_share_nodes_within_prompt: Literal[True] = True
    edges_independent_within_prompt: Literal[False] = False
    prompt_is_experimental_unit: Literal[True] = True
    generation_work_multiplier: Literal[1] = 1
    oracle_forward_work_multiplier: Literal[1] = 1
    annotation_work_multiplier: Literal[6] = 6
    reward_edge_work_multiplier: Literal[6] = 6
    independent_sample_multiplier: Literal[1] = 1

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_ids, tuple) or not self.prompt_ids:
            raise ValueError("prompt_ids must be a non-empty immutable tuple")
        if len(set(self.prompt_ids)) != len(self.prompt_ids):
            raise ValueError("prompt_ids must be unique")
        if self.pair_layout != ALL_SIX_PAIR_LAYOUT:
            raise ValueError("all-six metadata must use the canonical pair layout")
        if self.candidates_per_prompt != 4 or self.edges_per_prompt != 6:
            raise ValueError("all-six metadata requires four candidates and six edges")
        if (
            self.cluster_unit != "prompt"
            or self.construction != "complete_unordered_pair_u_statistic"
            or self.candidate_nodes_iid_within_prompt is not True
            or self.edges_share_nodes_within_prompt is not True
            or self.edges_independent_within_prompt is not False
            or self.prompt_is_experimental_unit is not True
            or self.independent_sample_multiplier != 1
        ):
            raise ValueError("invalid prompt-cluster dependence contract")
        if (
            self.generation_work_multiplier != 1
            or self.oracle_forward_work_multiplier != 1
            or self.annotation_work_multiplier != 6
            or self.reward_edge_work_multiplier != 6
        ):
            raise ValueError("invalid all-six work multipliers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "all-six-prompt-u-stat/v1",
            "prompt_ids": list(self.prompt_ids),
            "pair_layout": [list(pair) for pair in self.pair_layout],
            "candidates_per_prompt": self.candidates_per_prompt,
            "edges_per_prompt": self.edges_per_prompt,
            "cluster_unit": self.cluster_unit,
            "construction": self.construction,
            "candidate_nodes_iid_within_prompt": (self.candidate_nodes_iid_within_prompt),
            "edges_share_nodes_within_prompt": self.edges_share_nodes_within_prompt,
            "edges_independent_within_prompt": self.edges_independent_within_prompt,
            "prompt_is_experimental_unit": self.prompt_is_experimental_unit,
            "work_multipliers": {
                "generation": self.generation_work_multiplier,
                "oracle_forward": self.oracle_forward_work_multiplier,
                "annotation": self.annotation_work_multiplier,
                "reward_edge": self.reward_edge_work_multiplier,
                "independent_sample": self.independent_sample_multiplier,
            },
        }


@dataclass(frozen=True, slots=True)
class AllSixPromptUStatisticArm:
    """Complete-pair R=4 training batch with prompt-cluster identities."""

    training_batch: FeatureTrainingBatch
    repeated_labels: ReplicatedRepeatedLabelBatch
    edges: tuple[PromptPairEdge, ...]
    metadata: PromptUStatisticMetadata
    label_audit: R4LabelAudit

    def __post_init__(self) -> None:
        num_prompts = len(self.metadata.prompt_ids)
        edge_shape = (num_prompts, len(ALL_SIX_PAIR_LAYOUT))
        _validate_primary_repeated_labels(self.repeated_labels, edge_shape=edge_shape)
        expected_edges = _edges_for_layout(
            self.metadata.prompt_ids,
            ALL_SIX_PAIR_LAYOUT,
        )
        if self.edges != expected_edges:
            raise ValueError("edges must use prompt-major all-six ordering")
        if self.training_batch.num_edges != num_prompts * 6:
            raise ValueError("training_batch must contain six edges per prompt")
        expected_h = self.repeated_labels.mean_h.reshape(-1).to(dtype=self.training_batch.h.dtype)
        if not torch.equal(self.training_batch.h, expected_h):
            raise ValueError("training_batch.h must be the four-replicate mean")
        if not torch.equal(
            self.training_batch.left_wins,
            self.repeated_labels.pooled_wins.reshape(-1),
        ):
            raise ValueError("training_batch wins must pool the raw replicate labels")
        if not torch.equal(
            self.training_batch.num_annotations,
            self.repeated_labels.pooled_totals.reshape(-1),
        ):
            raise ValueError("training_batch totals must pool the raw replicate labels")
        if self.label_audit.edge_shape != edge_shape:
            raise ValueError("label audit edge shape does not match all-six labels")
        if self.label_audit.realized_total_annotations != (self.repeated_labels.total_annotations):
            raise ValueError("label audit annotation cost does not match raw labels")

    @property
    def edge_prompt_indices(self) -> tuple[int, ...]:
        return tuple(edge.prompt_index for edge in self.edges)

    def prompt_edge_slice(self, prompt_index: int) -> slice:
        if (
            isinstance(prompt_index, bool)
            or not isinstance(prompt_index, int)
            or not 0 <= prompt_index < len(self.metadata.prompt_ids)
        ):
            raise ValueError("prompt_index is out of range")
        start = prompt_index * self.metadata.edges_per_prompt
        return slice(start, start + self.metadata.edges_per_prompt)


def sample_all_six_prompt_u_stat_arm(
    training: TrainingTensorData,
    left_win_probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    max_total_annotations: int | None = None,
) -> AllSixPromptUStatisticArm:
    """Use all six unordered edges of exactly four existing iid candidates.

    ``left_win_probabilities[p, e]`` follows :data:`ALL_SIX_PAIR_LAYOUT`.
    Fisher nodes remain the original ``P*4`` nodes.  Only comparison-edge and
    annotation work grows by six relative to the canonical 0-1 arm.
    """

    if not isinstance(training, TrainingTensorData):
        raise TypeError("training must be a TrainingTensorData")
    if training.num_candidates != 4:
        raise ValueError("all-six construction requires exactly four candidates per prompt")
    probabilities = _validate_probabilities(
        left_win_probabilities,
        shape=(training.num_prompts, 6),
        device=training.policy_scores.device,
    )
    repeated = sample_replicated_geometric_repeated_labels(
        probabilities,
        num_replicates=PRIMARY_NUM_REPLICATES,
        gamma=PRIMARY_GAMMA,
        generator=generator,
        max_total_annotations=max_total_annotations,
    )

    left_indices = torch.tensor(
        [pair[0] for pair in ALL_SIX_PAIR_LAYOUT],
        dtype=torch.int64,
        device=training.policy_scores.device,
    )
    right_indices = torch.tensor(
        [pair[1] for pair in ALL_SIX_PAIR_LAYOUT],
        dtype=torch.int64,
        device=training.policy_scores.device,
    )
    left_features = training.reward_features[:, left_indices, :].reshape(
        -1,
        training.reward_dimension,
    )
    right_features = training.reward_features[:, right_indices, :].reshape(
        -1,
        training.reward_dimension,
    )
    edge_scores = (
        training.policy_scores[:, left_indices, :] - training.policy_scores[:, right_indices, :]
    ).reshape(-1, training.policy_dimension)
    batch = FeatureTrainingBatch(
        left_features=left_features.detach().clone(),
        right_features=right_features.detach().clone(),
        edge_scores=edge_scores.detach().clone(),
        node_scores=training.policy_scores.reshape(
            -1,
            training.policy_dimension,
        )
        .detach()
        .clone(),
        h=repeated.mean_h.reshape(-1).to(dtype=training.policy_scores.dtype).detach().clone(),
        left_wins=repeated.pooled_wins.reshape(-1).detach().clone(),
        num_annotations=repeated.pooled_totals.reshape(-1).detach().clone(),
    )
    metadata = PromptUStatisticMetadata(prompt_ids=tuple(training.prompt_ids))
    return AllSixPromptUStatisticArm(
        training_batch=batch,
        repeated_labels=repeated,
        edges=_edges_for_layout(training.prompt_ids, ALL_SIX_PAIR_LAYOUT),
        metadata=metadata,
        label_audit=_r4_audit(probabilities, repeated),
    )


@dataclass(frozen=True, slots=True)
class TangentCoordinateLayout:
    """Named, ordered identity of every coordinate in a full policy tangent."""

    layout_id: str
    coordinate_ids: tuple[_CoordinateId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("layout_id must be a non-empty string")
        if not isinstance(self.coordinate_ids, tuple) or not self.coordinate_ids:
            raise ValueError("coordinate_ids must be a non-empty immutable tuple")
        for coordinate_id in self.coordinate_ids:
            if isinstance(coordinate_id, bool) or not isinstance(
                coordinate_id,
                (str, int),
            ):
                raise TypeError("coordinate IDs must be strings or non-boolean integers")
            if isinstance(coordinate_id, str) and not coordinate_id:
                raise ValueError("string coordinate IDs must be non-empty")
        if len(set(self.coordinate_ids)) != len(self.coordinate_ids):
            raise ValueError("coordinate IDs must be unique")


@dataclass(frozen=True, slots=True)
class LowDimensionalTangentControl:
    """Storage-independent tangent slice plus its exact coordinate audit."""

    training: TrainingTensorData
    source_layout_id: str
    source_dimension: int
    selected_indices: tuple[int, ...]
    selected_coordinate_ids: tuple[_CoordinateId, ...]
    num_fisher_nodes: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_layout_id, str) or not self.source_layout_id:
            raise ValueError("source_layout_id must be non-empty")
        if (
            isinstance(self.source_dimension, bool)
            or not isinstance(self.source_dimension, int)
            or self.source_dimension < 2
        ):
            raise ValueError("source_dimension must be at least two")
        if len(self.selected_indices) != self.training.policy_dimension:
            raise ValueError("selected index count must equal the sliced policy dimension")
        if len(self.selected_coordinate_ids) != len(self.selected_indices):
            raise ValueError("selected coordinate IDs must align with selected indices")
        if (
            isinstance(self.num_fisher_nodes, bool)
            or not isinstance(self.num_fisher_nodes, int)
            or self.num_fisher_nodes < 1
        ):
            raise ValueError("num_fisher_nodes must be positive")
        if self.training.policy_dimension >= self.num_fisher_nodes:
            raise ValueError("low-dimensional tangent must satisfy d < n_F")
        if self.training.policy_dimension >= self.source_dimension:
            raise ValueError("low-dimensional tangent must be a strict coordinate subset")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "low-dimensional-tangent/v1",
            "source_layout_id": self.source_layout_id,
            "source_dimension": self.source_dimension,
            "selected_indices": list(self.selected_indices),
            "selected_coordinate_ids": list(self.selected_coordinate_ids),
            "selected_dimension": self.training.policy_dimension,
            "num_fisher_nodes": self.num_fisher_nodes,
            "strictly_below_fisher_node_count": True,
        }

    def scatter_direction_to_full(
        self,
        selected_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter a low-dimensional direction back into the full LoRA layout."""

        value = _validate_frozen_float_tensor(
            "selected_direction",
            selected_direction,
            shape=(self.training.policy_dimension,),
            device=self.training.policy_scores.device,
        )
        full = torch.zeros(
            self.source_dimension,
            dtype=value.dtype,
            device=value.device,
        )
        index_tensor = torch.tensor(
            self.selected_indices,
            dtype=torch.int64,
            device=value.device,
        )
        full.index_copy_(0, index_tensor, value)
        return full


def _validate_coordinate_indices(
    coordinate_indices: Sequence[int],
    *,
    source_dimension: int,
    num_fisher_nodes: int,
) -> tuple[int, ...]:
    if isinstance(coordinate_indices, (str, bytes, bytearray)) or not isinstance(
        coordinate_indices,
        Sequence,
    ):
        raise TypeError("coordinate_indices must be an explicit integer sequence")
    indices = tuple(coordinate_indices)
    if not indices:
        raise ValueError("at least one coordinate index must be selected")
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("coordinate indices must be non-boolean integers")
        if not 0 <= index < source_dimension:
            raise ValueError("coordinate index is out of range")
    if len(set(indices)) != len(indices):
        raise ValueError("coordinate indices must be unique")
    if len(indices) >= source_dimension:
        raise ValueError("low-dimensional tangent must be a strict coordinate subset")
    if len(indices) >= num_fisher_nodes:
        raise ValueError("low-dimensional tangent must satisfy d < n_F")
    return indices


def select_low_dimensional_tangent(
    training: TrainingTensorData,
    *,
    coordinate_indices: Sequence[int],
    coordinate_layout: TangentCoordinateLayout,
) -> LowDimensionalTangentControl:
    """Select explicit policy-score coordinates and clone every output tensor."""

    if not isinstance(training, TrainingTensorData):
        raise TypeError("training must be a TrainingTensorData")
    if not isinstance(coordinate_layout, TangentCoordinateLayout):
        raise TypeError("coordinate_layout must be a TangentCoordinateLayout")
    if len(coordinate_layout.coordinate_ids) != training.policy_dimension:
        raise ValueError("coordinate layout dimension does not match policy_scores")
    num_fisher_nodes = training.num_prompts * training.num_candidates
    indices = _validate_coordinate_indices(
        coordinate_indices,
        source_dimension=training.policy_dimension,
        num_fisher_nodes=num_fisher_nodes,
    )
    index_tensor = torch.tensor(
        indices,
        dtype=torch.int64,
        device=training.policy_scores.device,
    )
    sliced = TrainingTensorData(
        prompt_ids=tuple(training.prompt_ids),
        policy_scores=training.policy_scores.index_select(2, index_tensor).detach().clone(),
        reward_features=training.reward_features.detach().clone(),
        h=training.h.detach().clone(),
        left_wins=training.left_wins.detach().clone(),
        num_annotations=training.num_annotations.detach().clone(),
    )
    return LowDimensionalTangentControl(
        training=sliced,
        source_layout_id=coordinate_layout.layout_id,
        source_dimension=training.policy_dimension,
        selected_indices=indices,
        selected_coordinate_ids=tuple(coordinate_layout.coordinate_ids[index] for index in indices),
        num_fisher_nodes=num_fisher_nodes,
    )


def slice_low_dimensional_tangent(
    training: TrainingTensorData,
    *,
    coordinate_indices: Sequence[int],
    coordinate_layout: TangentCoordinateLayout,
) -> TrainingTensorData:
    """Return only the independent :class:`TrainingTensorData` tangent slice."""

    return select_low_dimensional_tangent(
        training,
        coordinate_indices=coordinate_indices,
        coordinate_layout=coordinate_layout,
    ).training


def _validate_projection_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**63:
        raise ValueError("projection seed must be an integer in [0, 2**63)")
    return seed


def _validate_projection_namespace(namespace: str) -> str:
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("projection namespace must be a non-empty string")
    return namespace


def _effective_projection_seed(seed: int, namespace: str) -> int:
    payload = f"{namespace}\0{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def generate_seeded_orthonormal_projection(
    source_dimension: int,
    selected_dimension: int,
    *,
    seed: int,
    namespace: str,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, int]:
    """Generate a sign-canonical Gaussian-QR projection and effective seed.

    The namespace is cryptographically mixed into the declared seed before the
    PyTorch generator is initialized.  Exact cross-device random-number
    equivalence is not assumed; the returned projection's SHA256 is therefore
    part of the downstream control evidence.
    """

    for name, value in (
        ("source_dimension", source_dimension),
        ("selected_dimension", selected_dimension),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if selected_dimension >= source_dimension:
        raise ValueError("orthonormal projection requires selected_dimension < source_dimension")
    declared_seed = _validate_projection_seed(seed)
    namespace_value = _validate_projection_namespace(namespace)
    if not isinstance(dtype, torch.dtype) or dtype not in {torch.float32, torch.float64}:
        raise TypeError("orthonormal projection dtype must be torch.float32 or torch.float64")
    device_value = torch.device(device)
    effective_seed = _effective_projection_seed(declared_seed, namespace_value)
    generator = torch.Generator(device=device_value).manual_seed(effective_seed)
    gaussian = torch.randn(
        source_dimension,
        selected_dimension,
        dtype=dtype,
        device=device_value,
        generator=generator,
    )
    projection, upper = torch.linalg.qr(gaussian, mode="reduced")
    diagonal = torch.diagonal(upper)
    signs = torch.where(
        diagonal < 0.0,
        -torch.ones_like(diagonal),
        torch.ones_like(diagonal),
    )
    projection = (projection * signs.unsqueeze(0)).detach().clone()
    if not bool(torch.isfinite(projection).all()):
        raise FloatingPointError("orthonormal projection generation produced non-finite values")
    return projection, effective_seed


@dataclass(frozen=True, slots=True)
class SeededOrthonormalTangentControl:
    """Projected tangent plus the matrix needed to deploy back in full LoRA space."""

    training: TrainingTensorData
    projection: torch.Tensor
    source_layout_id: str
    source_dimension: int
    num_fisher_nodes: int
    namespace: str
    declared_seed: int
    effective_seed: int
    projection_sha256: str
    algorithm: Literal["gaussian_qr_sign_canonical_v1"] = "gaussian_qr_sign_canonical_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.source_layout_id, str) or not self.source_layout_id:
            raise ValueError("source_layout_id must be non-empty")
        _validate_projection_namespace(self.namespace)
        _validate_projection_seed(self.declared_seed)
        _validate_projection_seed(self.effective_seed)
        if self.effective_seed != _effective_projection_seed(
            self.declared_seed,
            self.namespace,
        ):
            raise ValueError("effective seed does not match namespace-mixed declared seed")
        if self.algorithm != ORTHONORMAL_PROJECTION_ALGORITHM:
            raise ValueError("invalid orthonormal projection algorithm")
        if (
            isinstance(self.source_dimension, bool)
            or not isinstance(self.source_dimension, int)
            or self.source_dimension < 2
        ):
            raise ValueError("source_dimension must be at least two")
        if (
            isinstance(self.num_fisher_nodes, bool)
            or not isinstance(self.num_fisher_nodes, int)
            or self.num_fisher_nodes < 1
        ):
            raise ValueError("num_fisher_nodes must be positive")
        selected_dimension = self.training.policy_dimension
        if selected_dimension >= self.source_dimension:
            raise ValueError("projected tangent must be lower dimensional than its source")
        if selected_dimension >= self.num_fisher_nodes:
            raise ValueError("projected tangent must satisfy d < n_F")
        projection = _validate_frozen_float_tensor(
            "projection",
            self.projection,
            shape=(self.source_dimension, selected_dimension),
            device=self.training.policy_scores.device,
            dtype=torch.float64,
        )
        gram = projection.mT @ projection
        identity = torch.eye(
            selected_dimension,
            dtype=projection.dtype,
            device=projection.device,
        )
        tolerance = 2.0e-12
        if not torch.allclose(gram, identity, rtol=tolerance, atol=tolerance):
            raise ValueError("projection columns must be orthonormal")
        if self.projection_sha256 != _tensor_sha256(projection):
            raise ValueError("projection_sha256 does not match the projection tensor")

    @property
    def selected_dimension(self) -> int:
        return self.training.policy_dimension

    def scatter_direction_to_full(
        self,
        selected_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Map ``u_low`` to the deployable full direction ``P @ u_low``."""

        value = _validate_frozen_float_tensor(
            "selected_direction",
            selected_direction,
            shape=(self.selected_dimension,),
            device=self.projection.device,
        )
        projection = self.projection.to(dtype=value.dtype)
        return (projection @ value).detach().clone()

    def to_dict(self) -> dict[str, Any]:
        gram = self.projection.mT @ self.projection
        identity = torch.eye(
            self.selected_dimension,
            dtype=self.projection.dtype,
            device=self.projection.device,
        )
        orthonormality_max_absolute_error = float(torch.max(torch.abs(gram - identity)).item())
        return {
            "schema_version": "seeded-orthonormal-tangent/v1",
            "source_layout_id": self.source_layout_id,
            "source_dimension": self.source_dimension,
            "selected_dimension": self.selected_dimension,
            "num_fisher_nodes": self.num_fisher_nodes,
            "namespace": self.namespace,
            "declared_seed": self.declared_seed,
            "effective_seed": self.effective_seed,
            "algorithm": self.algorithm,
            "projection_sha256": self.projection_sha256,
            "projection_dtype": str(self.projection.dtype),
            "score_construction": "S_low = cast_fp32(cast_fp64(S_full) @ P_fp64)",
            "deployment_scatter": "u_full = P @ u_low",
            "orthonormal_columns": True,
            "orthonormality_max_absolute_error": (orthonormality_max_absolute_error),
            "strictly_below_fisher_node_count": True,
        }


def select_seeded_orthonormal_tangent(
    training: TrainingTensorData,
    *,
    selected_dimension: int,
    coordinate_layout: TangentCoordinateLayout,
    seed: int,
    namespace: str,
) -> SeededOrthonormalTangentControl:
    """Construct the locked seeded subspace ``S_low = S_full @ P``."""

    if not isinstance(training, TrainingTensorData):
        raise TypeError("training must be a TrainingTensorData")
    if not isinstance(coordinate_layout, TangentCoordinateLayout):
        raise TypeError("coordinate_layout must be a TangentCoordinateLayout")
    if len(coordinate_layout.coordinate_ids) != training.policy_dimension:
        raise ValueError("coordinate layout dimension does not match policy_scores")
    if (
        isinstance(selected_dimension, bool)
        or not isinstance(selected_dimension, int)
        or selected_dimension < 1
    ):
        raise ValueError("selected_dimension must be a positive integer")
    num_fisher_nodes = training.num_prompts * training.num_candidates
    if selected_dimension >= num_fisher_nodes:
        raise ValueError("projected tangent must satisfy d < n_F")
    projection, effective_seed = generate_seeded_orthonormal_projection(
        training.policy_dimension,
        selected_dimension,
        seed=seed,
        namespace=namespace,
        dtype=torch.float64,
        device=training.policy_scores.device,
    )
    projected_scores = (training.policy_scores.to(dtype=torch.float64) @ projection).to(
        dtype=training.policy_scores.dtype
    )
    sliced = TrainingTensorData(
        prompt_ids=tuple(training.prompt_ids),
        policy_scores=projected_scores.detach().clone(),
        reward_features=training.reward_features.detach().clone(),
        h=training.h.detach().clone(),
        left_wins=training.left_wins.detach().clone(),
        num_annotations=training.num_annotations.detach().clone(),
    )
    return SeededOrthonormalTangentControl(
        training=sliced,
        projection=projection.detach().clone(),
        source_layout_id=coordinate_layout.layout_id,
        source_dimension=training.policy_dimension,
        num_fisher_nodes=num_fisher_nodes,
        namespace=_validate_projection_namespace(namespace),
        declared_seed=_validate_projection_seed(seed),
        effective_seed=effective_seed,
        projection_sha256=_tensor_sha256(projection),
    )
