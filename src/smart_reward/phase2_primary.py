"""Claim-free, isolated and resumable Phase-2 primary-head core.

This module intentionally exposes only the shared R=4 label construction and
one primary learner at a time.  It neither imports nor executes any Phase-2
control arm, held-out evaluator, rollout, policy session, or beta decision.

Objects produced here are deliberately neutral/core evidence.  They are not
R3 campaign identities and are not admissible HPC results.  A formal R3
orchestrator must wrap this core only after validating materialization
provenance, the new R3 design, Gate-P authorization, and scheduler lineage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from . import phase2_training as _training
from .contracts import BT_MLE, CANONICAL_LEARNERS, PRORM_PLUS
from .experiment import TrainingTensorData
from .phase2_checkpoint import CheckpointSignal, DurableCheckpointStore
from .phase2_config import Phase2ConfigBundle
from .phase2_controls import sample_canonical_r4_noisy_arm
from .repeated_label_diagnostics import build_repeated_label_tail_diagnostics
from .training import BTMLETrainer, ProRMPlusTrainer

PHASE2_NEUTRAL_CONTEXT_SCHEMA = "phase2-r4-train-materialization-context/v1"
PHASE2_PRIMARY_CORE_CHECKPOINT_BINDING_SCHEMA = "phase2-r4-primary-core-checkpoint-binding/v1"
PHASE2_PRIMARY_CORE_RESULT_SCHEMA = "phase2-r4-primary-head-core-result/v1"
PHASE2_PRIMARY_CORE_CAMPAIGN_KIND = "phase2_r4_primary_core_unclaimed"
PHASE2_PRIMARY_CORE_EXECUTION_REVISION = 0
PHASE2_PRIMARY_CORE_ROLE = "unclaimed_train_only_core"
PHASE2_PRIMARY_CORE_EXECUTION_ROLE = "phase2_primary_core_unclaimed"
PHASE2_PRIMARY_FIRST_ORDER_CHECK_INTERVAL_STEPS = 20
# The neutral/core runner retains its historical 20-step checkpoint default.
# Formal R3 execution does not consume this constant; its full-state cadence is
# supplied by the Gate-P resource plan.
PHASE2_PRIMARY_CHECKPOINT_INTERVAL_STEPS = 20

PrimaryLearner: TypeAlias = Literal["bt_mle", "prorm_plus"]
PrimaryTrainer: TypeAlias = BTMLETrainer | ProRMPlusTrainer


def _learner(value: object) -> PrimaryLearner:
    if value not in CANONICAL_LEARNERS:
        raise ValueError(f"learner must be one of {CANONICAL_LEARNERS!r}")
    return value


def _context_identity_payload(
    *,
    settings: _training.Phase2TrainingSettings,
    seed: int,
    input_training_sha256: str,
    primary_training_sha256: str,
    oracle_reward_sha256: str,
    absolute_damping: float,
    label_stream: _training.LabelStreamEvidence,
    reward_head_identifiability: Mapping[str, object],
    prorm_moment_map_identifiability: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": PHASE2_NEUTRAL_CONTEXT_SCHEMA,
        "claim_scope": "neutral_materialization_only",
        "phase2_config_hash": settings.phase2_config_hash,
        "settings_sha256": settings.sha256,
        "input_training_sha256": input_training_sha256,
        "primary_training_sha256": primary_training_sha256,
        "oracle_reward_sha256": oracle_reward_sha256,
        "label_stream_sha256": label_stream.label_stream_sha256,
        "label_stream": label_stream.to_dict(),
        "seed": seed,
        "absolute_damping": absolute_damping,
        "reward_head_identifiability_sha256": _training._canonical_sha256(
            dict(reward_head_identifiability)
        ),
        "prorm_moment_map_identifiability_sha256": _training._canonical_sha256(
            dict(prorm_moment_map_identifiability)
        ),
        "active_named_rng_states": [],
    }


@dataclass(frozen=True, slots=True)
class NeutralPhase2TrainingContext:
    """Neutral R=4 train state with no campaign claim or retained target/RNG."""

    settings: _training.Phase2TrainingSettings
    seed: int
    training: TrainingTensorData
    input_training_sha256: str
    primary_training_sha256: str
    oracle_reward_sha256: str
    absolute_damping: float
    label_stream: _training.LabelStreamEvidence
    reward_head_identifiability: Mapping[str, object]
    prorm_moment_map_identifiability: Mapping[str, object]
    context_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.settings, _training.Phase2TrainingSettings):
            raise TypeError("settings must be Phase2TrainingSettings")
        if not isinstance(self.training, TrainingTensorData):
            raise TypeError("training must be TrainingTensorData")
        _training._validate_seed(self.seed)
        if self.seed not in self.settings.seeds:
            raise ValueError("context seed is not declared by its Phase-2 settings")
        for name in (
            "input_training_sha256",
            "primary_training_sha256",
            "oracle_reward_sha256",
            "context_sha256",
        ):
            _training._validate_digest(getattr(self, name), name=name)
        damping = _training._finite_float(
            self.absolute_damping,
            name="absolute_damping",
            minimum=0.0,
            strictly_greater=True,
        )
        if not isinstance(self.label_stream, _training.LabelStreamEvidence):
            raise TypeError("label_stream must be LabelStreamEvidence")
        if (
            self.label_stream.base_seed != self.seed
            or self.label_stream.oracle_reward_sha256 != self.oracle_reward_sha256
        ):
            raise ValueError("context label stream is not bound to its seed and oracle hash")
        reward_rank = _training._strict_json_copy(
            self.reward_head_identifiability,
            name="reward_head_identifiability",
        )
        prorm_rank = _training._strict_json_copy(
            self.prorm_moment_map_identifiability,
            name="prorm_moment_map_identifiability",
        )
        object.__setattr__(self, "reward_head_identifiability", reward_rank)
        object.__setattr__(self, "prorm_moment_map_identifiability", prorm_rank)
        if _training._input_training_sha256(self.training) != self.primary_training_sha256:
            raise ValueError("prepared primary training tensors changed identity")
        payload = _context_identity_payload(
            settings=self.settings,
            seed=self.seed,
            input_training_sha256=self.input_training_sha256,
            primary_training_sha256=self.primary_training_sha256,
            oracle_reward_sha256=self.oracle_reward_sha256,
            absolute_damping=damping,
            label_stream=self.label_stream,
            reward_head_identifiability=reward_rank,
            prorm_moment_map_identifiability=prorm_rank,
        )
        if _training._canonical_sha256(payload) != self.context_sha256:
            raise ValueError("primary context identity does not match its contents")

    def validate_integrity(self) -> None:
        """Revalidate tensor and identity bindings after construction."""

        self.__post_init__()


def prepare_neutral_phase2_context(
    training: TrainingTensorData,
    train_oracle_rewards: torch.Tensor,
    *,
    seed: int,
    settings: (_training.Phase2TrainingSettings | Phase2ConfigBundle | Mapping[str, object]),
) -> NeutralPhase2TrainingContext:
    """Prepare the common R=4 prefix without issuing any campaign identity."""

    if not isinstance(training, TrainingTensorData):
        raise TypeError("training must be TrainingTensorData")
    compiled = _training.compile_phase2_training_settings(settings)
    if compiled.convergence.check_interval != PHASE2_PRIMARY_FIRST_ORDER_CHECK_INTERVAL_STEPS:
        raise ValueError(
            "R3 primary first-order checks must retain the frozen 20-update "
            "cadence; durable full-state checkpoint cadence is a separate "
            "Gate-P operational policy"
        )
    base_seed = _training._validate_seed(seed)
    if base_seed not in compiled.seeds:
        raise ValueError("seed is not one of the configured Phase-2 design seeds")
    rewards = _training._validate_frozen_oracle_rewards(training, train_oracle_rewards)
    input_training_sha = _training._input_training_sha256(training)
    oracle_reward_sha = _training._tensor_sha256(rewards)
    train_fisher_nodes = training.num_prompts * training.num_candidates
    if compiled.pcg_max_iterations < train_fisher_nodes + 1:
        raise ValueError(
            "pcg_max_iterations must cover the train Fisher rank bound plus one "
            f"({train_fisher_nodes + 1})"
        )
    absolute_damping = _training._absolute_damping(training, compiled)

    canonical_margins = rewards[:, 0] - rewards[:, 1]
    probabilities = torch.sigmoid(canonical_margins)
    floor = compiled.probability_floor
    tolerance = 2.0e-6 if probabilities.dtype == torch.float32 else 2.0e-12
    if bool(
        ((probabilities < floor - tolerance) | (probabilities > 1.0 - floor + tolerance)).any()
    ):
        raise ValueError(
            "train_oracle_rewards do not satisfy the locked transformed-oracle "
            "BTL probability range [0.25, 0.75]"
        )
    generator, derived_seed, derivation_sha = _training._generator_for_training(
        training,
        base_seed=base_seed,
        namespace=compiled.label_rng_namespace,
    )
    initial_generator_state_sha = _training._tensor_sha256(generator.get_state())
    noisy_arm = sample_canonical_r4_noisy_arm(
        training,
        probabilities,
        generator=generator,
        max_total_annotations=compiled.max_total_annotations,
    )
    final_generator_state_sha = _training._tensor_sha256(generator.get_state())
    labels = noisy_arm.repeated_labels
    replicate_count_sha = _training._tensor_sha256(labels.counts)
    replicate_win_sha = _training._tensor_sha256(labels.wins)
    replicate_h_sha = _training._tensor_sha256(labels.replicate_h)
    mean_h_sha = _training._tensor_sha256(noisy_arm.training.h)
    repeated_label_tail_diagnostics = build_repeated_label_tail_diagnostics(
        replicate_counts=labels.counts,
        replicate_h=labels.replicate_h,
        mean_h=noisy_arm.training.h,
        replicate_count_sha256=replicate_count_sha,
        replicate_h_sha256=replicate_h_sha,
        mean_h_sha256=mean_h_sha,
    )
    label_payload = {
        "namespace": compiled.label_rng_namespace,
        "base_seed": base_seed,
        "derived_seed": derived_seed,
        "derivation_sha256": derivation_sha,
        "initial_state_sha256": initial_generator_state_sha,
        "final_state_sha256": final_generator_state_sha,
        "probability_sha256": noisy_arm.audit.probability_sha256,
        "replicate_count_sha256": replicate_count_sha,
        "replicate_win_sha256": replicate_win_sha,
        "replicate_h_sha256": replicate_h_sha,
        "mean_h_sha256": mean_h_sha,
        "repeated_label_tail_diagnostics_sha256": repeated_label_tail_diagnostics[
            "diagnostics_sha256"
        ],
        "realized_total_annotations": labels.total_annotations,
    }
    label_stream_sha = _training._canonical_sha256(label_payload)
    label_evidence = _training.LabelStreamEvidence(
        namespace=compiled.label_rng_namespace,
        base_seed=base_seed,
        derived_seed=derived_seed,
        derivation_sha256=derivation_sha,
        generator_device=str(generator.device),
        initial_state_sha256=initial_generator_state_sha,
        final_state_sha256=final_generator_state_sha,
        oracle_reward_sha256=oracle_reward_sha,
        canonical_probability_sha256=noisy_arm.audit.probability_sha256,
        replicate_count_sha256=replicate_count_sha,
        replicate_win_sha256=replicate_win_sha,
        replicate_h_sha256=replicate_h_sha,
        mean_h_sha256=mean_h_sha,
        label_stream_sha256=label_stream_sha,
        repeated_label_tail_diagnostics=repeated_label_tail_diagnostics,
        realized_total_annotations=labels.total_annotations,
        realized_annotations_per_edge=noisy_arm.audit.realized_annotations_per_edge,
        expected_annotations_per_edge=noisy_arm.audit.expected_annotations_per_edge,
        num_edges=training.num_prompts,
    )

    prepared_training = noisy_arm.training
    primary_training_sha = _training._input_training_sha256(prepared_training)
    reward_head_identifiability = _training._reward_head_identifiability(
        prepared_training,
        compiled,
    )
    prorm_moment_map_identifiability = _training._prorm_moment_map_identifiability(
        prepared_training,
        compiled,
    )
    identity = _context_identity_payload(
        settings=compiled,
        seed=base_seed,
        input_training_sha256=input_training_sha,
        primary_training_sha256=primary_training_sha,
        oracle_reward_sha256=oracle_reward_sha,
        absolute_damping=absolute_damping,
        label_stream=label_evidence,
        reward_head_identifiability=reward_head_identifiability,
        prorm_moment_map_identifiability=prorm_moment_map_identifiability,
    )
    return NeutralPhase2TrainingContext(
        settings=compiled,
        seed=base_seed,
        training=prepared_training,
        input_training_sha256=input_training_sha,
        primary_training_sha256=primary_training_sha,
        oracle_reward_sha256=oracle_reward_sha,
        absolute_damping=absolute_damping,
        label_stream=label_evidence,
        reward_head_identifiability=reward_head_identifiability,
        prorm_moment_map_identifiability=prorm_moment_map_identifiability,
        context_sha256=_training._canonical_sha256(identity),
    )


def build_primary_core_trainer(
    context: NeutralPhase2TrainingContext,
    learner: PrimaryLearner,
) -> PrimaryTrainer:
    """Construct exactly one fresh, zero-head AdamW core trainer."""

    if not isinstance(context, NeutralPhase2TrainingContext):
        raise TypeError("context must be NeutralPhase2TrainingContext")
    context.validate_integrity()
    method = _learner(learner)
    model = _training._zero_model(context.training)
    batch = context.training.to_training_batch()
    if method == BT_MLE:
        trainer: PrimaryTrainer = BTMLETrainer(
            model,
            batch,
            _training._bt_config(context.settings),
        )
    else:
        trainer = ProRMPlusTrainer(
            model,
            batch,
            _training._prorm_config(
                context.settings,
                absolute_damping=context.absolute_damping,
            ),
        )
    if trainer.completed_steps != 0 or trainer.history:
        raise RuntimeError("new primary trainer is not at an exact fresh boundary")
    if bool(torch.count_nonzero(trainer.model.weight.detach())):
        raise RuntimeError("new primary trainer head is not exactly zero")
    if not isinstance(trainer.optimizer, torch.optim.AdamW):
        raise TypeError("primary trainer must use AdamW")
    if trainer.optimizer.state:
        raise RuntimeError("new primary AdamW state must be empty")
    parameters = [
        parameter for group in trainer.optimizer.param_groups for parameter in group["params"]
    ]
    if len(parameters) != 1 or parameters[0] is not trainer.model.weight:
        raise RuntimeError("primary AdamW must contain exactly the reward-head parameter")
    return trainer


def primary_core_checkpoint_binding(
    context: NeutralPhase2TrainingContext,
    learner: PrimaryLearner,
) -> dict[str, object]:
    """Return a claim-free binding accepted only by the core runner."""

    if not isinstance(context, NeutralPhase2TrainingContext):
        raise TypeError("context must be NeutralPhase2TrainingContext")
    context.validate_integrity()
    method = _learner(learner)
    return {
        "schema_version": PHASE2_PRIMARY_CORE_CHECKPOINT_BINDING_SCHEMA,
        "campaign_kind": PHASE2_PRIMARY_CORE_CAMPAIGN_KIND,
        "execution_revision": PHASE2_PRIMARY_CORE_EXECUTION_REVISION,
        "role": PHASE2_PRIMARY_CORE_ROLE,
        "objective": method,
        "context_sha256": context.context_sha256,
        "settings_sha256": context.settings.sha256,
        "input_training_sha256": context.input_training_sha256,
        "oracle_reward_sha256": context.oracle_reward_sha256,
        "label_stream_sha256": context.label_stream.label_stream_sha256,
        "seed": context.seed,
        "formal_r3_evidence": False,
        "active_named_rng_states": [],
    }


def _validate_checkpoint_store(
    store: DurableCheckpointStore,
    *,
    expected_binding: Mapping[str, object],
    learner: PrimaryLearner,
) -> None:
    if not isinstance(store, DurableCheckpointStore):
        raise TypeError("checkpoint_store must be DurableCheckpointStore")
    expected = _training._strict_json_copy(
        expected_binding,
        name="expected_primary_checkpoint_binding",
    )
    if store.objective != learner:
        raise ValueError("checkpoint store objective does not match the primary learner")
    if store.binding != expected:
        raise ValueError("checkpoint store binding does not match the primary context")
    if (
        store.binding.get("campaign_kind") != PHASE2_PRIMARY_CORE_CAMPAIGN_KIND
        or store.binding.get("execution_revision") != PHASE2_PRIMARY_CORE_EXECUTION_REVISION
        or store.binding.get("role") != PHASE2_PRIMARY_CORE_ROLE
        or store.binding.get("objective") != learner
        or store.binding.get("formal_r3_evidence") is not False
        or store.binding.get("active_named_rng_states") != []
    ):
        raise ValueError("checkpoint store is not valid claim-free primary-core state")


def _result_identity(
    *,
    learner: PrimaryLearner,
    context_sha256: str,
    training_design_sha256: str,
    training_settings_sha256: str,
    input_training_sha256: str,
    oracle_reward_sha256: str,
    seed: int,
    absolute_damping: float,
    label_stream: _training.LabelStreamEvidence,
    head: _training.TrainedHeadEvidence,
    reward_head_identifiability: Mapping[str, object],
    prorm_moment_map_identifiability: Mapping[str, object],
    checkpoint_store_used: bool,
    resumed_from_checkpoint: bool,
) -> dict[str, object]:
    return {
        "schema_version": PHASE2_PRIMARY_CORE_RESULT_SCHEMA,
        "campaign_kind": PHASE2_PRIMARY_CORE_CAMPAIGN_KIND,
        "execution_revision": PHASE2_PRIMARY_CORE_EXECUTION_REVISION,
        "role": PHASE2_PRIMARY_CORE_ROLE,
        "learner": learner,
        "context_sha256": context_sha256,
        "training_design_sha256": training_design_sha256,
        "training_settings_sha256": training_settings_sha256,
        "input_training_sha256": input_training_sha256,
        "oracle_reward_sha256": oracle_reward_sha256,
        "seed": seed,
        "absolute_damping": absolute_damping,
        "label_stream": label_stream.to_dict(),
        "head": head.to_dict(),
        "identifiability": {
            "reward_head": _training._strict_json_copy(
                reward_head_identifiability,
                name="reward_head_identifiability",
            ),
            "prorm_moment_map": _training._strict_json_copy(
                prorm_moment_map_identifiability,
                name="prorm_moment_map_identifiability",
            ),
        },
        "checkpoint": {
            "store_used": checkpoint_store_used,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "checkpoint_interval_steps": PHASE2_PRIMARY_CHECKPOINT_INTERVAL_STEPS,
        },
        "named_rng_contract": {
            "label_rng_consumed_during_context_preparation": True,
            "label_rng_state_retained": False,
            "active_named_rng_states": [],
        },
        "information_boundary": {
            "train_only": True,
            "raw_oracle_rewards_retained": False,
            "raw_repeated_labels_retained": False,
            "validation_or_test_data_accessed": False,
            "policy_session_opened": False,
            "policy_rollout_performed": False,
            "beta_outcome_computed": False,
            "controls_executed": False,
        },
    }


@dataclass(frozen=True, slots=True)
class PrimaryHeadCoreResult:
    """One claim-free train-only core head; never formal R3 evidence."""

    learner: PrimaryLearner
    context_sha256: str
    training_design_sha256: str
    training_settings_sha256: str
    input_training_sha256: str
    oracle_reward_sha256: str
    seed: int
    absolute_damping: float
    label_stream: _training.LabelStreamEvidence
    head: _training.TrainedHeadEvidence
    reward_head_identifiability: Mapping[str, object]
    prorm_moment_map_identifiability: Mapping[str, object]
    primary_head_result_sha256: str
    checkpoint_store_used: bool
    resumed_from_checkpoint: bool

    def __post_init__(self) -> None:
        method = _learner(self.learner)
        for name in (
            "context_sha256",
            "training_design_sha256",
            "training_settings_sha256",
            "input_training_sha256",
            "oracle_reward_sha256",
            "primary_head_result_sha256",
        ):
            _training._validate_digest(getattr(self, name), name=name)
        _training._validate_seed(self.seed)
        _training._finite_float(
            self.absolute_damping,
            name="absolute_damping",
            minimum=0.0,
            strictly_greater=True,
        )
        if not isinstance(self.label_stream, _training.LabelStreamEvidence):
            raise TypeError("label_stream must be LabelStreamEvidence")
        if not isinstance(self.head, _training.TrainedHeadEvidence):
            raise TypeError("head must be TrainedHeadEvidence")
        if self.head.method != method or self.head.arm != _training.PRIMARY_TRAINING_ARM:
            raise ValueError("result head is not the requested R=4 primary learner")
        if (
            self.label_stream.base_seed != self.seed
            or self.label_stream.oracle_reward_sha256 != self.oracle_reward_sha256
        ):
            raise ValueError("result label evidence is not bound to its seed and oracle")
        reward_rank = _training._strict_json_copy(
            self.reward_head_identifiability,
            name="reward_head_identifiability",
        )
        prorm_rank = _training._strict_json_copy(
            self.prorm_moment_map_identifiability,
            name="prorm_moment_map_identifiability",
        )
        object.__setattr__(self, "reward_head_identifiability", reward_rank)
        object.__setattr__(self, "prorm_moment_map_identifiability", prorm_rank)
        if not isinstance(self.checkpoint_store_used, bool) or not isinstance(
            self.resumed_from_checkpoint,
            bool,
        ):
            raise TypeError("checkpoint result flags must be bool")
        identity = _result_identity(
            learner=method,
            context_sha256=self.context_sha256,
            training_design_sha256=self.training_design_sha256,
            training_settings_sha256=self.training_settings_sha256,
            input_training_sha256=self.input_training_sha256,
            oracle_reward_sha256=self.oracle_reward_sha256,
            seed=self.seed,
            absolute_damping=self.absolute_damping,
            label_stream=self.label_stream,
            head=self.head,
            reward_head_identifiability=reward_rank,
            prorm_moment_map_identifiability=prorm_rank,
            checkpoint_store_used=self.checkpoint_store_used,
            resumed_from_checkpoint=self.resumed_from_checkpoint,
        )
        if _training._canonical_sha256(identity) != self.primary_head_result_sha256:
            raise ValueError("primary result identity does not match its head and context")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PHASE2_PRIMARY_CORE_RESULT_SCHEMA,
            "campaign_kind": PHASE2_PRIMARY_CORE_CAMPAIGN_KIND,
            "execution_revision": PHASE2_PRIMARY_CORE_EXECUTION_REVISION,
            "role": PHASE2_PRIMARY_CORE_ROLE,
            "learner": self.learner,
            "context_sha256": self.context_sha256,
            "training_design_sha256": self.training_design_sha256,
            "training_settings_sha256": self.training_settings_sha256,
            "input_training_sha256": self.input_training_sha256,
            "oracle_reward_sha256": self.oracle_reward_sha256,
            "seed": self.seed,
            "absolute_damping": self.absolute_damping,
            "label_stream": self.label_stream.to_dict(),
            "head": self.head.to_dict(),
            "identifiability": {
                "reward_head": _training._strict_json_copy(
                    self.reward_head_identifiability,
                    name="reward_head_identifiability",
                ),
                "prorm_moment_map": _training._strict_json_copy(
                    self.prorm_moment_map_identifiability,
                    name="prorm_moment_map_identifiability",
                ),
            },
            "primary_head_result_sha256": self.primary_head_result_sha256,
            "checkpoint": {
                "store_used": self.checkpoint_store_used,
                "resumed_from_checkpoint": self.resumed_from_checkpoint,
                "checkpoint_interval_steps": PHASE2_PRIMARY_CHECKPOINT_INTERVAL_STEPS,
            },
            "named_rng_contract": {
                "label_rng_consumed_during_context_preparation": True,
                "label_rng_state_retained": False,
                "active_named_rng_states": [],
            },
            "information_boundary": {
                "train_only": True,
                "raw_oracle_rewards_retained": False,
                "raw_repeated_labels_retained": False,
                "validation_or_test_data_accessed": False,
                "policy_session_opened": False,
                "policy_rollout_performed": False,
                "beta_outcome_computed": False,
                "controls_executed": False,
            },
        }


def _checkpoint_completed_steps(payload: Mapping[str, object]) -> int:
    controller = payload.get("controller_state")
    if not isinstance(controller, Mapping):
        raise ValueError("controller checkpoint is missing controller_state")
    steps = controller.get("completed_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("controller checkpoint completed_steps is invalid")
    return steps


def _controller_updates_executed(
    convergence: _training._ConvergedTrainingRun,
    trainer: PrimaryTrainer,
) -> int:
    """Recover the monotonic work boundary after selected-state restoration."""

    evidence = convergence.evidence
    if not isinstance(evidence, Mapping):
        raise TypeError("convergence evidence must be a mapping")
    candidates = [int(trainer.completed_steps)]
    optimizer_execution = evidence.get("optimizer_protocol_execution")
    if isinstance(optimizer_execution, Mapping):
        observed = optimizer_execution.get("completed_updates_observed")
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError("optimizer completed_updates_observed is invalid")
        candidates.append(observed)
    for field in ("selected_primary_step", "fixed_step_snapshot_steps"):
        value = evidence.get(field)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"convergence {field} is invalid")
            candidates.append(value)
    checks = evidence.get("checks")
    if checks is not None:
        if not isinstance(checks, list):
            raise TypeError("convergence checks must be a list")
        for check in checks:
            if not isinstance(check, Mapping):
                raise TypeError("convergence check must be a mapping")
            step = check.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError("convergence check step is invalid")
            candidates.append(step)
    return max(candidates)


def train_primary_head_core(
    context: NeutralPhase2TrainingContext,
    learner: PrimaryLearner,
    *,
    checkpoint_store: DurableCheckpointStore | None = None,
    checkpoint_signal: CheckpointSignal | None = None,
    stop_requested: Callable[[], str | None] | None = None,
) -> PrimaryHeadCoreResult:
    """Train one claim-free core learner, optionally resuming durable state."""

    if not isinstance(context, NeutralPhase2TrainingContext):
        raise TypeError("context must be NeutralPhase2TrainingContext")
    context.validate_integrity()
    method = _learner(learner)
    if checkpoint_signal is not None and not isinstance(checkpoint_signal, CheckpointSignal):
        raise TypeError("checkpoint_signal must be CheckpointSignal or None")
    if stop_requested is not None and not callable(stop_requested):
        raise TypeError("stop_requested must be callable or None")
    if checkpoint_store is None and (checkpoint_signal is not None or stop_requested is not None):
        raise ValueError("scheduler stop requests require a durable checkpoint store")

    expected_binding = primary_core_checkpoint_binding(context, method)
    resume_state: Mapping[str, object] | None = None
    resumed = False
    if checkpoint_store is not None:
        _validate_checkpoint_store(
            checkpoint_store,
            expected_binding=expected_binding,
            learner=method,
        )
        audited_generations = checkpoint_store.audit_generations(verify_all_checkpoint_bytes=True)
        resume_state = checkpoint_store.load()
        resumed = resume_state is not None
        if resumed:
            if not audited_generations:
                raise RuntimeError("loaded checkpoint has no committed generation")
            payload_steps = _checkpoint_completed_steps(resume_state)
            generation_steps = audited_generations[-1].get("completed_steps")
            if payload_steps != generation_steps:
                raise RuntimeError(
                    "checkpoint generation completed_steps does not match controller state"
                )
        if not resumed:
            checkpoint_store.record_progress(
                status="initialized",
                completed_steps=0,
                details={
                    "execution_role": PHASE2_PRIMARY_CORE_EXECUTION_ROLE,
                    "active_named_rng_states": [],
                },
            )

    trainer = build_primary_core_trainer(context, method)
    initial_head_sha = _training._tensor_sha256(trainer.model.weight)

    def checkpoint_hook(payload: Mapping[str, object], *, reason: str) -> None:
        if checkpoint_store is None:
            raise RuntimeError("durable checkpoint hook has no store")
        steps = _checkpoint_completed_steps(payload)
        checkpoint_store.save(
            payload,
            completed_steps=steps,
            reason=reason,
        )

    def combined_stop_requested() -> str | None:
        if checkpoint_signal is not None and checkpoint_signal.requested:
            return checkpoint_signal.signal_name or "scheduler_signal"
        if stop_requested is None:
            return None
        return stop_requested()

    def after_resume_state_restored() -> None:
        if checkpoint_store is None or resume_state is None:
            raise RuntimeError("resume RNG callback has no loaded durable state")
        checkpoint_store.restore_pending_rng_state()
        checkpoint_store.record_progress(
            status="resumed",
            completed_steps=_checkpoint_completed_steps(resume_state),
            details={
                "execution_role": PHASE2_PRIMARY_CORE_EXECUTION_ROLE,
                "rng_restored_after_trainer_and_controller_state": True,
                "active_named_rng_states": [],
            },
        )

    if method == BT_MLE:

        def audit() -> _training._FirstOrderMeasurement:
            return _training._bt_first_order_measurement(trainer)

        rank_diagnostic = context.reward_head_identifiability
    else:

        def audit() -> _training._FirstOrderMeasurement:
            return _training._prorm_first_order_measurement(trainer)

        rank_diagnostic = context.prorm_moment_map_identifiability

    convergence = _training._run_trainer_to_first_order_convergence(
        trainer,
        audit=audit,
        spec=context.settings.convergence,
        fixed_snapshot_steps=context.settings.outer_steps,
        objective_name=method,
        rank_diagnostic=rank_diagnostic,
        resume_state=resume_state,
        checkpoint_hook=(checkpoint_hook if checkpoint_store is not None else None),
        checkpoint_interval_steps=(
            PHASE2_PRIMARY_CHECKPOINT_INTERVAL_STEPS if checkpoint_store is not None else None
        ),
        stop_requested=(
            combined_stop_requested
            if checkpoint_signal is not None or stop_requested is not None
            else None
        ),
        after_resume_state_restored=(
            after_resume_state_restored if resume_state is not None else None
        ),
        execution_role=PHASE2_PRIMARY_CORE_EXECUTION_ROLE,
    )
    controller_updates_executed = _controller_updates_executed(
        convergence,
        trainer,
    )
    if checkpoint_store is not None:
        terminal_generations = checkpoint_store.audit_generations(verify_all_checkpoint_bytes=True)
        if not terminal_generations:
            raise RuntimeError("durable primary training completed without a checkpoint")
        terminal_steps = terminal_generations[-1].get("completed_steps")
        if terminal_steps != controller_updates_executed:
            raise RuntimeError(
                "terminal checkpoint completed_steps does not match controller execution"
            )

    final_pcg = None
    if method == PRORM_PLUS:
        final_solver = convergence.final.inner_solver
        if final_solver is None or final_solver.get("converged") is not True:
            raise RuntimeError("final ProRM+ cold-start FP64 PCG audit did not converge")
        final_pcg = _training._pcg_evidence(final_solver)
    head = _training._make_head_evidence(
        arm=_training.PRIMARY_TRAINING_ARM,
        method=method,
        model=trainer.model,
        initial_head_sha256=initial_head_sha,
        initial_objective=convergence.initial.objective,
        final_objective=convergence.final.objective,
        history=convergence.history,
        final_pcg=final_pcg,
        first_order_convergence=convergence.evidence,
    )
    context.validate_integrity()
    result_identity = _result_identity(
        learner=method,
        context_sha256=context.context_sha256,
        training_design_sha256=context.settings.phase2_config_hash,
        training_settings_sha256=context.settings.sha256,
        input_training_sha256=context.input_training_sha256,
        oracle_reward_sha256=context.oracle_reward_sha256,
        seed=context.seed,
        absolute_damping=context.absolute_damping,
        label_stream=context.label_stream,
        head=head,
        reward_head_identifiability=context.reward_head_identifiability,
        prorm_moment_map_identifiability=context.prorm_moment_map_identifiability,
        checkpoint_store_used=checkpoint_store is not None,
        resumed_from_checkpoint=resumed,
    )
    result = PrimaryHeadCoreResult(
        learner=method,
        context_sha256=context.context_sha256,
        training_design_sha256=context.settings.phase2_config_hash,
        training_settings_sha256=context.settings.sha256,
        input_training_sha256=context.input_training_sha256,
        oracle_reward_sha256=context.oracle_reward_sha256,
        seed=context.seed,
        absolute_damping=context.absolute_damping,
        label_stream=context.label_stream,
        head=head,
        reward_head_identifiability=context.reward_head_identifiability,
        prorm_moment_map_identifiability=context.prorm_moment_map_identifiability,
        primary_head_result_sha256=_training._canonical_sha256(result_identity),
        checkpoint_store_used=checkpoint_store is not None,
        resumed_from_checkpoint=resumed,
    )
    if checkpoint_store is not None:
        checkpoint_store.record_progress(
            status="completed",
            completed_steps=controller_updates_executed,
            details={
                "execution_role": PHASE2_PRIMARY_CORE_EXECUTION_ROLE,
                "controller_updates_executed": controller_updates_executed,
                "selected_primary_step": int(trainer.completed_steps),
                "primary_head_result_sha256": result.primary_head_result_sha256,
                "head_sha256": result.head.head_sha256,
                "active_named_rng_states": [],
            },
        )
    result.to_dict()
    return result


__all__ = [
    "PHASE2_PRIMARY_CHECKPOINT_INTERVAL_STEPS",
    "PHASE2_PRIMARY_FIRST_ORDER_CHECK_INTERVAL_STEPS",
    "PHASE2_NEUTRAL_CONTEXT_SCHEMA",
    "PHASE2_PRIMARY_CORE_CAMPAIGN_KIND",
    "PHASE2_PRIMARY_CORE_CHECKPOINT_BINDING_SCHEMA",
    "PHASE2_PRIMARY_CORE_EXECUTION_REVISION",
    "PHASE2_PRIMARY_CORE_EXECUTION_ROLE",
    "PHASE2_PRIMARY_CORE_RESULT_SCHEMA",
    "PHASE2_PRIMARY_CORE_ROLE",
    "NeutralPhase2TrainingContext",
    "PrimaryHeadCoreResult",
    "build_primary_core_trainer",
    "prepare_neutral_phase2_context",
    "primary_core_checkpoint_binding",
    "train_primary_head_core",
]
