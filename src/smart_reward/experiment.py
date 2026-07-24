"""Leakage-safe Phase-1 comparison on precomputed policy scores and RM features.

This module is deliberately tensor-only: model downloads, response generation,
oracle construction, and repeated-label sampling must happen upstream.  The
training schema cannot carry evaluation rewards, and validation/test prompts are
used only after both heads have completed the same fixed number of updates.

The runner reports the observed BT-MLE and ProRM+ outcomes.  It does **not** encode
or imply a guarantee that ProRM+ wins on any particular finite sample.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .contracts import BT_MLE, PRORM_PLUS, canonical_method
from .linear import FisherSolveDType, resolve_fisher_solve_dtype
from .metrics import local_regret, natural_direction_metrics
from .training import (
    BTMLETrainer,
    BTMLETrainingConfig,
    FeatureTrainingBatch,
    FrozenFeatureLinearReward,
    ProRMPlusTrainer,
    ProRMPlusTrainingConfig,
)

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_PromptId = str | int


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _validate_optional_positive_integer(name: str, value: int | None) -> None:
    if value is not None:
        _positive_integer(name, value)


def _validate_optional_positive_float(name: str, value: float | None) -> None:
    if value is not None:
        _finite_positive(name, value)


def _validate_prompt_ids(prompt_ids: tuple[_PromptId, ...], num_prompts: int) -> None:
    if not isinstance(prompt_ids, tuple):
        raise TypeError("prompt_ids must be an immutable tuple")
    if len(prompt_ids) != num_prompts:
        raise ValueError(f"prompt_ids must contain exactly {num_prompts} entries")
    for prompt_id in prompt_ids:
        if isinstance(prompt_id, bool) or not isinstance(prompt_id, (str, int)):
            raise TypeError("each prompt ID must be a string or non-boolean integer")
        if isinstance(prompt_id, str) and not prompt_id:
            raise ValueError("string prompt IDs must be non-empty")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt IDs must be unique within each split")


def _validate_node_tensors(
    prompt_ids: tuple[_PromptId, ...],
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
) -> tuple[int, int, int, int]:
    if not isinstance(policy_scores, torch.Tensor):
        raise TypeError("policy_scores must be a torch.Tensor")
    if not isinstance(reward_features, torch.Tensor):
        raise TypeError("reward_features must be a torch.Tensor")
    if policy_scores.ndim != 3:
        raise ValueError("policy_scores must have shape (P, M, D)")
    num_prompts, num_candidates, policy_dimension = policy_scores.shape
    if num_prompts < 1:
        raise ValueError("every split must contain at least one prompt")
    if num_candidates < 2:
        raise ValueError("every prompt must contain at least two candidates")
    if policy_dimension < 1:
        raise ValueError("the policy-score dimension must be positive")
    if reward_features.ndim != 3 or reward_features.shape[:2] != (
        num_prompts,
        num_candidates,
    ):
        raise ValueError("reward_features must have shape (P, M, H)")
    reward_dimension = reward_features.shape[2]
    if reward_dimension < 1:
        raise ValueError("the reward-feature dimension must be positive")
    for name, tensor in (
        ("policy_scores", policy_scores),
        ("reward_features", reward_features),
    ):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if tensor.requires_grad:
            raise ValueError(f"{name} must be frozen (requires_grad=False)")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
    if (
        reward_features.dtype != policy_scores.dtype
        or reward_features.device != policy_scores.device
    ):
        raise ValueError("policy_scores and reward_features must share dtype and device")
    _validate_prompt_ids(prompt_ids, num_prompts)
    return num_prompts, num_candidates, policy_dimension, reward_dimension


def _validate_split_scalar(
    name: str,
    tensor: torch.Tensor,
    *,
    num_prompts: int,
    reference: torch.Tensor,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.shape != (num_prompts,):
        raise ValueError(f"{name} must have shape ({num_prompts},)")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if tensor.dtype != reference.dtype or tensor.device != reference.device:
        raise ValueError(f"{name} must share dtype and device with policy_scores")
    if tensor.requires_grad:
        raise ValueError(f"{name} must be frozen (requires_grad=False)")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class TrainingTensorData:
    """The complete and intentionally reward-free Phase-1 training schema.

    There is exactly one canonical edge per prompt: candidate 0 is the left
    endpoint and candidate 1 is the right endpoint.  ``h``, ``left_wins``, and
    ``num_annotations`` all use that orientation.  Evaluation-only target
    rewards are structurally impossible to pass to this constructor.
    """

    prompt_ids: tuple[_PromptId, ...]
    policy_scores: torch.Tensor
    reward_features: torch.Tensor
    h: torch.Tensor
    left_wins: torch.Tensor
    num_annotations: torch.Tensor

    def __post_init__(self) -> None:
        num_prompts, _, _, _ = _validate_node_tensors(
            self.prompt_ids,
            self.policy_scores,
            self.reward_features,
        )
        _validate_split_scalar(
            "h",
            self.h,
            num_prompts=num_prompts,
            reference=self.policy_scores,
        )
        for name, tensor in (
            ("left_wins", self.left_wins),
            ("num_annotations", self.num_annotations),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.shape != (num_prompts,):
                raise ValueError(f"{name} must have shape ({num_prompts},)")
            if tensor.dtype not in _INTEGER_DTYPES:
                raise TypeError(f"{name} must have an integer dtype")
            if tensor.device != self.policy_scores.device:
                raise ValueError(f"{name} must be on the policy_scores device")
        if self.left_wins.dtype != self.num_annotations.dtype:
            raise ValueError("left_wins and num_annotations must have the same dtype")
        if bool((self.num_annotations < 1).any()):
            raise ValueError("every training prompt must have at least one annotation")
        if bool(((self.left_wins < 0) | (self.left_wins > self.num_annotations)).any()):
            raise ValueError("left_wins must satisfy 0 <= left_wins <= num_annotations")

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

    def to(self, device: torch.device | str) -> TrainingTensorData:
        """Return the frozen training tensors on one explicit compute device."""

        return TrainingTensorData(
            prompt_ids=self.prompt_ids,
            policy_scores=self.policy_scores.to(device=device),
            reward_features=self.reward_features.to(device=device),
            h=self.h.to(device=device),
            left_wins=self.left_wins.to(device=device),
            num_annotations=self.num_annotations.to(device=device),
        )

    def to_training_batch(self) -> FeatureTrainingBatch:
        """Materialize the canonical 0-vs-1 edge and all-node Fisher batch."""

        return FeatureTrainingBatch(
            left_features=self.reward_features[:, 0],
            right_features=self.reward_features[:, 1],
            edge_scores=self.policy_scores[:, 0] - self.policy_scores[:, 1],
            node_scores=self.policy_scores.reshape(-1, self.policy_dimension),
            h=self.h,
            left_wins=self.left_wins,
            num_annotations=self.num_annotations,
        )


@dataclass(frozen=True)
class EvaluationTensorData:
    """A held-out prompt/candidate pool containing evaluation targets only."""

    prompt_ids: tuple[_PromptId, ...]
    policy_scores: torch.Tensor
    reward_features: torch.Tensor
    true_rewards: torch.Tensor

    def __post_init__(self) -> None:
        num_prompts, num_candidates, _, _ = _validate_node_tensors(
            self.prompt_ids,
            self.policy_scores,
            self.reward_features,
        )
        if not isinstance(self.true_rewards, torch.Tensor):
            raise TypeError("true_rewards must be a torch.Tensor")
        if self.true_rewards.shape != (num_prompts, num_candidates):
            raise ValueError("true_rewards must have shape (P, M)")
        if not self.true_rewards.is_floating_point():
            raise TypeError("true_rewards must have a floating-point dtype")
        if (
            self.true_rewards.dtype != self.policy_scores.dtype
            or self.true_rewards.device != self.policy_scores.device
        ):
            raise ValueError("true_rewards must share dtype and device with policy_scores")
        if self.true_rewards.requires_grad:
            raise ValueError("true_rewards must be frozen (requires_grad=False)")
        if not bool(torch.isfinite(self.true_rewards).all()):
            raise ValueError("true_rewards must be finite")

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


@dataclass(frozen=True)
class ControlledFeatureExperiment:
    """Leakage-audited train/validation/test tensors for one controlled run."""

    train: TrainingTensorData
    validation: EvaluationTensorData
    test: EvaluationTensorData

    def __post_init__(self) -> None:
        if not isinstance(self.train, TrainingTensorData):
            raise TypeError("train must be TrainingTensorData")
        if not isinstance(self.validation, EvaluationTensorData):
            raise TypeError("validation must be EvaluationTensorData")
        if not isinstance(self.test, EvaluationTensorData):
            raise TypeError("test must be EvaluationTensorData")
        split_ids = {
            "train": set(self.train.prompt_ids),
            "validation": set(self.validation.prompt_ids),
            "test": set(self.test.prompt_ids),
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = split_ids[left].intersection(split_ids[right])
            if overlap:
                rendered_overlap = sorted(repr(item) for item in overlap)
                raise ValueError(
                    f"{left} and {right} prompt IDs must be disjoint; overlap={rendered_overlap!r}"
                )

        reference = self.train
        for name, split in (("validation", self.validation), ("test", self.test)):
            if split.num_candidates != reference.num_candidates:
                raise ValueError(f"{name} must use the same candidate count as train")
            if split.policy_dimension != reference.policy_dimension:
                raise ValueError(f"{name} must use the same policy-score dimension as train")
            if split.reward_dimension != reference.reward_dimension:
                raise ValueError(f"{name} must use the same reward-feature dimension as train")
            if (
                split.policy_scores.dtype != reference.policy_scores.dtype
                or split.policy_scores.device != reference.policy_scores.device
            ):
                raise ValueError(f"{name} must share train dtype and device")


@dataclass(frozen=True)
class FeatureExperimentConfig:
    """Fixed-step fair-comparison settings; weight decay is identically zero."""

    outer_steps: int = 20
    learning_rate: float = 1.0e-2
    beta: float = 1.0
    relative_damping: float = 1.0e-3
    pcg_dtype: FisherSolveDType = "float64"
    pcg_max_iterations: int = 200
    pcg_tolerance: float = 1.0e-5
    pcg_absolute_tolerance: float = 0.0
    pcg_residual_recompute_interval: int = 20
    microbatch_size: int | None = None
    max_grad_norm: float | None = None
    weight_decay: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        _positive_integer("outer_steps", self.outer_steps)
        _finite_positive("learning_rate", self.learning_rate)
        _finite_positive("beta", self.beta)
        _finite_positive("relative_damping", self.relative_damping)
        resolve_fisher_solve_dtype(self.pcg_dtype)
        _positive_integer("pcg_max_iterations", self.pcg_max_iterations)
        _finite_positive("pcg_tolerance", self.pcg_tolerance)
        absolute_tolerance = float(self.pcg_absolute_tolerance)
        if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
            raise ValueError("pcg_absolute_tolerance must be finite and non-negative")
        _positive_integer(
            "pcg_residual_recompute_interval",
            self.pcg_residual_recompute_interval,
        )
        _validate_optional_positive_integer("microbatch_size", self.microbatch_size)
        _validate_optional_positive_float("max_grad_norm", self.max_grad_norm)


@dataclass(frozen=True)
class HeldOutPolicyMetrics:
    """JSON-safe local policy metrics from one untouched held-out split."""

    num_prompts: int
    absolute_damping: float
    local_regret: float
    squared_fisher_error: float
    fisher_cosine: float | None
    pairwise_accuracy: float
    oracle_pairwise_nll: float
    oracle_probability_mae: float
    predicted_fisher_norm: float
    target_fisher_norm: float

    def __post_init__(self) -> None:
        _positive_integer("num_prompts", self.num_prompts)
        for name in (
            "absolute_damping",
            "local_regret",
            "squared_fisher_error",
            "oracle_pairwise_nll",
            "oracle_probability_mae",
            "predicted_fisher_norm",
            "target_fisher_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.absolute_damping <= 0.0:
            raise ValueError("absolute_damping must be strictly positive")
        if self.fisher_cosine is not None:
            cosine = float(self.fisher_cosine)
            if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
                raise ValueError("fisher_cosine must be null or finite in [-1, 1]")
        accuracy = float(self.pairwise_accuracy)
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError("pairwise_accuracy must be finite and lie in [0, 1]")
        probability_mae = float(self.oracle_probability_mae)
        if not 0.0 <= probability_mae <= 1.0:
            raise ValueError("oracle_probability_mae must lie in [0, 1]")


@dataclass(frozen=True)
class PCGEvidence:
    """JSON-safe evidence for the final full-data ProRM+ linear solve."""

    iterations: int
    residual_norm: float
    relative_residual: float
    converged: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 0
        ):
            raise ValueError("iterations must be a non-negative integer")
        for name in ("residual_norm", "relative_residual"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool")


@dataclass(frozen=True)
class FeatureLearnerResult:
    """Train objective, immutable head identity, and held-out metrics."""

    method: str
    initial_train_objective: float
    final_train_objective: float
    initial_head_sha256: str
    head_sha256: str
    head_weight: tuple[float, ...]
    validation: HeldOutPolicyMetrics
    test: HeldOutPolicyMetrics
    final_pcg: PCGEvidence | None

    def __post_init__(self) -> None:
        canonical_method(self.method)
        for name in ("initial_train_objective", "final_train_objective"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("initial_head_sha256", "head_sha256"):
            digest = getattr(self, name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            if any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        if not isinstance(self.head_weight, tuple) or not self.head_weight:
            raise ValueError("head_weight must be a non-empty tuple")
        if not all(math.isfinite(float(value)) for value in self.head_weight):
            raise ValueError("head_weight must contain only finite values")
        if not isinstance(self.validation, HeldOutPolicyMetrics) or not isinstance(
            self.test, HeldOutPolicyMetrics
        ):
            raise TypeError("validation and test must be HeldOutPolicyMetrics")
        if self.method == BT_MLE and self.final_pcg is not None:
            raise ValueError("BT-MLE must not report PCG evidence")
        if self.method == PRORM_PLUS and not isinstance(self.final_pcg, PCGEvidence):
            raise TypeError("ProRM+ must report final PCG evidence")


@dataclass(frozen=True)
class FeatureExperimentResult:
    """JSON-compatible outcome of a leakage-safe fixed-step comparison."""

    config: FeatureExperimentConfig
    train_absolute_damping: float
    bt_mle: FeatureLearnerResult
    prorm_plus: FeatureLearnerResult
    heldout_used_for_training: bool = field(default=False, init=False)
    prorm_plus_win_guaranteed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, FeatureExperimentConfig):
            raise TypeError("config must be FeatureExperimentConfig")
        _finite_positive("train_absolute_damping", self.train_absolute_damping)
        if not isinstance(self.bt_mle, FeatureLearnerResult) or not isinstance(
            self.prorm_plus, FeatureLearnerResult
        ):
            raise TypeError("bt_mle and prorm_plus must be FeatureLearnerResult objects")
        if self.bt_mle.method != BT_MLE or self.prorm_plus.method != PRORM_PLUS:
            raise ValueError("learner results are assigned to the wrong method")
        if self.bt_mle.initial_head_sha256 != self.prorm_plus.initial_head_sha256:
            raise ValueError("both learners must have exactly the same initialization")
        if self.heldout_used_for_training or self.prorm_plus_win_guaranteed:
            raise ValueError("the controlled protocol cannot claim leakage or a guaranteed win")

    @property
    def srm_plus(self) -> FeatureLearnerResult:
        """Compatibility view of the canonical :attr:`prorm_plus` result."""

        return self.prorm_plus

    @property
    def srm_win_guaranteed(self) -> bool:
        """Compatibility view of :attr:`prorm_plus_win_guaranteed`."""

        return self.prorm_plus_win_guaranteed

    def to_dict(self) -> dict[str, Any]:
        """Return a nested structure accepted by strict JSON encoders."""

        return asdict(self)


def _absolute_damping(
    policy_scores: torch.Tensor,
    relative_damping: float,
    pcg_dtype: FisherSolveDType,
) -> float:
    flat_scores = policy_scores.reshape(-1, policy_scores.shape[-1]).to(
        dtype=resolve_fisher_solve_dtype(pcg_dtype)
    )
    fisher_diagonal = flat_scores.square().mean(dim=0)
    scale = float(fisher_diagonal.mean().item())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "mean(diag(F)) must be finite and positive; the policy tangent is degenerate"
        )
    damping = float(relative_damping) * scale
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("absolute damping is not finite and positive")
    return damping


def _head_sha256(weight: torch.Tensor) -> str:
    value = weight.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(bytes(value.view(torch.uint8).tolist()))
    return digest.hexdigest()


@torch.no_grad()
def _heldout_metrics(
    model: FrozenFeatureLinearReward,
    split: EvaluationTensorData,
    config: FeatureExperimentConfig,
) -> HeldOutPolicyMetrics:
    predicted_rewards = model(split.reward_features)
    damping = _absolute_damping(
        split.policy_scores,
        config.relative_damping,
        config.pcg_dtype,
    )
    regret = local_regret(
        split.policy_scores,
        predicted_rewards,
        split.true_rewards,
        damping=damping,
        beta=config.beta,
        pcg_tolerance=config.pcg_tolerance,
        pcg_max_iterations=config.pcg_max_iterations,
        pcg_dtype=config.pcg_dtype,
    )
    directions = natural_direction_metrics(
        split.policy_scores,
        predicted_rewards,
        split.true_rewards,
        damping=damping,
        pcg_tolerance=config.pcg_tolerance,
        pcg_max_iterations=config.pcg_max_iterations,
        pcg_dtype=config.pcg_dtype,
    )
    raw_cosine = float(directions.fisher_cosine.item())
    cosine = raw_cosine if math.isfinite(raw_cosine) else None
    candidate_pairs = torch.combinations(
        torch.arange(split.num_candidates, device=predicted_rewards.device),
        r=2,
    )
    predicted_margins = (
        predicted_rewards[:, candidate_pairs[:, 0]] - predicted_rewards[:, candidate_pairs[:, 1]]
    )
    target_margins = (
        split.true_rewards[:, candidate_pairs[:, 0]] - split.true_rewards[:, candidate_pairs[:, 1]]
    )
    oracle_probabilities = torch.sigmoid(target_margins)
    predicted_probabilities = torch.sigmoid(predicted_margins)
    oracle_pairwise_nll = F.binary_cross_entropy_with_logits(
        predicted_margins,
        oracle_probabilities,
        reduction="mean",
    )
    oracle_probability_mae = torch.mean(torch.abs(predicted_probabilities - oracle_probabilities))
    identifiable = target_margins != 0.0
    if not bool(identifiable.any()):
        raise ValueError("held-out targets contain no identifiable candidate ordering")
    strict_correct = (
        torch.sign(predicted_margins[identifiable]) == torch.sign(target_margins[identifiable])
    ).to(predicted_rewards.dtype)
    predicted_ties = predicted_margins[identifiable] == 0.0
    strict_correct[predicted_ties] = 0.5
    pairwise_accuracy = float(strict_correct.mean().item())
    return HeldOutPolicyMetrics(
        num_prompts=split.num_prompts,
        absolute_damping=damping,
        local_regret=float(regret.item()),
        squared_fisher_error=float(directions.squared_fisher_error.item()),
        fisher_cosine=cosine,
        pairwise_accuracy=pairwise_accuracy,
        oracle_pairwise_nll=float(oracle_pairwise_nll.item()),
        oracle_probability_mae=float(oracle_probability_mae.item()),
        predicted_fisher_norm=float(directions.predicted_fisher_norm.item()),
        target_fisher_norm=float(directions.target_fisher_norm.item()),
    )


class FeatureExperimentRunner:
    """Run the fixed-step BT-MLE/ProRM+ comparison without held-out selection."""

    def __init__(
        self,
        experiment: ControlledFeatureExperiment,
        config: FeatureExperimentConfig | None = None,
    ) -> None:
        if not isinstance(experiment, ControlledFeatureExperiment):
            raise TypeError("experiment must be ControlledFeatureExperiment")
        effective_config = FeatureExperimentConfig() if config is None else config
        if not isinstance(effective_config, FeatureExperimentConfig):
            raise TypeError("config must be FeatureExperimentConfig")
        self.experiment = experiment
        self.config = effective_config

    def run(self) -> FeatureExperimentResult:
        """Train both zero-initialized heads and evaluate untouched splits.

        Validation is descriptive here: it cannot select a checkpoint, change
        the fixed step count, or tune either learner inside this call.  Reported
        comparisons are empirical outcomes and never a guaranteed ProRM+ win.
        """

        train = self.experiment.train
        config = self.config
        training_batch = train.to_training_batch()
        train_damping = _absolute_damping(
            train.policy_scores,
            config.relative_damping,
            config.pcg_dtype,
        )

        zero = torch.zeros(
            train.reward_dimension,
            dtype=train.reward_features.dtype,
            device=train.reward_features.device,
        )
        bt_model = FrozenFeatureLinearReward(train.reward_dimension, zero)
        prorm_model = FrozenFeatureLinearReward(train.reward_dimension, zero)
        if not torch.equal(bt_model.weight.detach(), prorm_model.weight.detach()):
            raise RuntimeError("fair initialization invariant failed")
        bt_initial_hash = _head_sha256(bt_model.weight)
        prorm_initial_hash = _head_sha256(prorm_model.weight)

        bt_trainer = BTMLETrainer(
            bt_model,
            training_batch,
            BTMLETrainingConfig(
                learning_rate=config.learning_rate,
                optimizer="adamw",
                weight_decay=0.0,
                microbatch_size=config.microbatch_size,
                max_grad_norm=config.max_grad_norm,
            ),
        )
        bt_initial_objective = bt_trainer.evaluate()

        prorm_trainer = ProRMPlusTrainer(
            prorm_model,
            training_batch,
            ProRMPlusTrainingConfig(
                learning_rate=config.learning_rate,
                optimizer="adamw",
                weight_decay=0.0,
                microbatch_size=config.microbatch_size,
                max_grad_norm=config.max_grad_norm,
                beta=config.beta,
                damping=train_damping,
                pcg_dtype=config.pcg_dtype,
                pcg_max_iterations=config.pcg_max_iterations,
                pcg_tolerance=config.pcg_tolerance,
                pcg_absolute_tolerance=config.pcg_absolute_tolerance,
                pcg_residual_recompute_interval=config.pcg_residual_recompute_interval,
                require_pcg_convergence=True,
            ),
        )
        prorm_initial_objective = prorm_trainer.evaluate(use_warm_start=False).dual_loss

        # No validation/test tensor is read before both fixed-step fits finish.
        bt_trainer.fit(config.outer_steps)
        prorm_trainer.fit(config.outer_steps)
        bt_final_objective = bt_trainer.evaluate()
        prorm_final = prorm_trainer.evaluate()

        bt_result = FeatureLearnerResult(
            method=BT_MLE,
            initial_train_objective=bt_initial_objective,
            final_train_objective=bt_final_objective,
            initial_head_sha256=bt_initial_hash,
            head_sha256=_head_sha256(bt_model.weight),
            head_weight=tuple(float(value) for value in bt_model.weight.detach().cpu()),
            validation=_heldout_metrics(bt_model, self.experiment.validation, config),
            test=_heldout_metrics(bt_model, self.experiment.test, config),
            final_pcg=None,
        )
        prorm_result = FeatureLearnerResult(
            method=PRORM_PLUS,
            initial_train_objective=prorm_initial_objective,
            final_train_objective=prorm_final.dual_loss,
            initial_head_sha256=prorm_initial_hash,
            head_sha256=_head_sha256(prorm_model.weight),
            head_weight=tuple(float(value) for value in prorm_model.weight.detach().cpu()),
            validation=_heldout_metrics(prorm_model, self.experiment.validation, config),
            test=_heldout_metrics(prorm_model, self.experiment.test, config),
            final_pcg=PCGEvidence(
                iterations=prorm_final.pcg_iterations,
                residual_norm=prorm_final.pcg_residual_norm,
                relative_residual=prorm_final.pcg_relative_residual,
                converged=prorm_final.pcg_converged,
            ),
        )
        return FeatureExperimentResult(
            config=config,
            train_absolute_damping=train_damping,
            bt_mle=bt_result,
            prorm_plus=prorm_result,
        )


def run_feature_experiment(
    experiment: ControlledFeatureExperiment,
    config: FeatureExperimentConfig | None = None,
) -> FeatureExperimentResult:
    """Convenience wrapper around :class:`FeatureExperimentRunner`."""

    return FeatureExperimentRunner(experiment, config).run()


def compile_feature_experiment_config(
    config: Mapping[str, object],
    *,
    damping_multiplier: float = 1.0,
) -> FeatureExperimentConfig:
    """Compile one validated YAML mapping into the only Phase-1 runtime config.

    The YAML stores damping relative to ``mean(diag(F))``.  This function keeps
    it relative; :class:`FeatureExperimentRunner` resolves the absolute value
    separately on train, validation, and test geometry.  No runtime default may
    silently override a declared optimizer setting.
    """

    from .config import validate_config

    normalized = validate_config(config)
    objective = normalized["objective"]
    reward = normalized["reward_model"]
    multiplier = _finite_positive("damping_multiplier", damping_multiplier)
    if reward["optimizer"] != "adamw":  # validate_config currently locks this.
        raise ValueError("the controlled comparison requires optimizer='adamw'")
    return FeatureExperimentConfig(
        outer_steps=int(reward["outer_steps"]),
        learning_rate=float(reward["learning_rate"]),
        beta=float(objective["beta"]),
        relative_damping=(
            float(objective["damping_relative_to_mean_fisher_diagonal"]) * multiplier
        ),
        pcg_dtype=objective["pcg_dtype"],
        pcg_max_iterations=int(objective["pcg_max_iterations"]),
        pcg_tolerance=float(objective["pcg_tolerance"]),
        microbatch_size=int(reward["microbatch_size"]),
        max_grad_norm=(None if "max_grad_norm" not in reward else float(reward["max_grad_norm"])),
    )


__all__ = [
    "ControlledFeatureExperiment",
    "EvaluationTensorData",
    "FeatureExperimentConfig",
    "FeatureExperimentResult",
    "FeatureExperimentRunner",
    "FeatureLearnerResult",
    "HeldOutPolicyMetrics",
    "PCGEvidence",
    "TrainingTensorData",
    "compile_feature_experiment_config",
    "run_feature_experiment",
]
