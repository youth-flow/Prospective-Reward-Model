"""Identity-bound Phase-2 ridge and frozen-beta sensitivity experiments.

The confirmatory primary result is immutable.  This module treats the
``1.0`` ridge/beta cells as byte-bound references to that result and executes
only the pre-registered off-primary cells:

* ridge multipliers ``0.1`` and ``10.0`` retrain a fresh ProRM+ head using the
  exact same deterministic R=4 label stream; BT-MLE is reused because its
  objective does not depend on the policy Fisher or its ridge;
* frozen-beta multipliers ``0.5`` and ``2.0`` reuse both primary reward heads,
  recompute their train-only natural directions, directly redeploy them at the
  multiplied scalar, and generate new on-policy rollouts.

Every cell is retained.  A pre-oracle safety failure is a terminal sensitivity
cell, not a reason to drop a seed or multiplier.  Cross-seed aggregation
therefore either uses all exact-30 observations in a cell or emits no interval
for that cell.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Final

import torch

from .common_beta import (
    bind_frozen_common_beta,
    deploy_with_common_beta,
    summarize_downstream_utility,
)
from .contracts import BT_MLE, CANONICAL_LEARNERS, PRORM_PLUS
from .experiment import TrainingTensorData
from .linear import DampedEmpiricalFisher, resolve_fisher_solve_dtype
from .metrics import local_regret
from .paths import relative_posix_reference
from .phase2_config import (
    PHASE2_CONFIRMATORY_SEEDS,
    Phase2ConfigBundle,
    phase2_design_identity,
    validate_phase2_config,
)
from .phase2_rollout import (
    PHASE2_ARM_ORDER,
    PHASE2_RESULT_SCHEMA,
    Phase2Design,
    Phase2PolicyRollout,
    Phase2PreparedInputs,
    Phase2RuntimeBackend,
    _arm_deployments,
    _kl_tail_summary,
    _length_summary,
    _rollout_policy_arms,
    assess_phase2_pre_oracle_safety,
)
from .phase2_training import (
    Phase2TrainingSettings,
    _generator_for_training,
    _make_head_evidence,
    _prorm_config,
    _prorm_first_order_measurement,
    _prorm_moment_map_identifiability,
    _run_trainer_to_first_order_convergence,
    _tensor_sha256,
    _zero_model,
    compile_phase2_training_settings,
)
from .phase2_training import (
    _canonical_sha256 as _training_canonical_sha256,
)
from .repro import atomic_write_json
from .rollout import (
    PolicyDirectionResult,
    policy_direction_from_head,
    policy_direction_from_node_rewards,
)
from .statistics import aggregate_paired_metrics
from .training import ProRMPlusTrainer

RIDGE_SENSITIVITY_GRID: Final = (0.1, 1.0, 10.0)
BETA_SENSITIVITY_GRID: Final = (0.5, 1.0, 2.0)
OFF_PRIMARY_RIDGE_GRID: Final = (0.1, 10.0)
OFF_PRIMARY_BETA_GRID: Final = (0.5, 2.0)

PHASE2_SENSITIVITY_SEED_SCHEMA: Final = "phase2-confirmatory-sensitivity-seed/v1"
PHASE2_SENSITIVITY_AGGREGATE_SCHEMA: Final = "phase2-confirmatory-sensitivity-aggregate/v1"


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_strict_json(path: str | os.PathLike[str]) -> tuple[dict[str, object], str]:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"JSON source must be a regular non-symlink file: {source}")
    raw = source.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON: {source}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSON source must encode an object: {source}")
    # Reject duplicate/non-canonical object surprises by proving strict JSON
    # serializability again.  Byte-level provenance remains the raw hash.
    json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return value, hashlib.sha256(raw).hexdigest()


def _verify_relative_file_reference(
    raw_reference: object,
    *,
    source: Path,
    expected_sha256: str,
    name: str,
) -> Path:
    if not isinstance(raw_reference, str) or not raw_reference or "\\" in raw_reference:
        raise ValueError(f"{name} must be a non-empty relative POSIX path")
    pure = PurePosixPath(raw_reference)
    if pure.is_absolute() or str(pure) != raw_reference:
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    resolved = (source.parent / Path(*pure.parts)).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must resolve to a regular non-symlink file")
    observed = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"{name} byte hash does not match its recorded SHA256")
    return resolved


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _seed(value: object, *, name: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be in [0, 2**63 - 1]")
    return value


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = " finite and positive" if positive else " finite"
        raise ValueError(f"{name} must be{qualifier}")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must have string keys")
    return value


def _strict_copy(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    try:
        copied = json.loads(
            json.dumps(dict(value), ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be strict JSON data") from error
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must encode an object")
    return copied


def _head_weights(value: object, *, name: str) -> dict[str, tuple[float, ...]]:
    heads = _mapping(value, name=name)
    if set(heads) != set(CANONICAL_LEARNERS):
        raise ValueError(f"{name} must contain exactly {CANONICAL_LEARNERS!r}")
    result: dict[str, tuple[float, ...]] = {}
    for learner in CANONICAL_LEARNERS:
        raw = heads[learner]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence) or not raw:
            raise TypeError(f"{name}.{learner} must be a non-empty sequence")
        result[learner] = tuple(
            _finite(item, name=f"{name}.{learner}[{index}]") for index, item in enumerate(raw)
        )
    return result


def _mean_utility_from_arm(
    arm: Mapping[str, object],
    *,
    name: str,
) -> dict[str, float]:
    utility = _mapping(arm.get("utility"), name=f"{name}.utility")
    result = {
        "mean_target_reward": _finite(
            utility.get("mean_target_reward"),
            name=f"{name}.utility.mean_target_reward",
        ),
        "mean_on_policy_kl": _finite(
            utility.get("mean_on_policy_kl_pi_updated_to_pi0"),
            name=f"{name}.utility.mean_on_policy_kl",
        ),
        "mean_target_utility": _finite(
            utility.get("mean_target_utility"),
            name=f"{name}.utility.mean_target_utility",
        ),
    }
    if result["mean_on_policy_kl"] < 0.0:
        raise ValueError(f"{name} contains a negative KL")
    return result


@dataclass(frozen=True, slots=True)
class PrimarySensitivityBinding:
    """Read-only identity and frozen heads extracted from one primary result."""

    seed: int
    path: Path
    result_sha256: str
    source_config_hash: str
    phase2_design_sha256: str
    runtime_contract_sha256: str
    environment_identity: Mapping[str, object]
    environment_identity_sha256: str
    artifact_metadata_sha256: str
    run_manifest_sha256: str
    heads: Mapping[str, tuple[float, ...]]
    heads_sha256: str
    training_design_sha256: str
    label_stream_sha256: str
    transformed_train_rewards_sha256: str
    beta0: float
    relative_ridge0: float
    primary_arms: Mapping[str, Mapping[str, float]]
    primary_heldout_local_regret: Mapping[str, float]
    primary_rollouts_sha256: str
    raw_result: Mapping[str, object]

    def __post_init__(self) -> None:
        _seed(self.seed)
        for name in (
            "result_sha256",
            "source_config_hash",
            "phase2_design_sha256",
            "runtime_contract_sha256",
            "environment_identity_sha256",
            "artifact_metadata_sha256",
            "run_manifest_sha256",
            "heads_sha256",
            "training_design_sha256",
            "label_stream_sha256",
            "transformed_train_rewards_sha256",
            "primary_rollouts_sha256",
        ):
            _digest(getattr(self, name), name=name)
        _finite(self.beta0, name="beta0", positive=True)
        _finite(self.relative_ridge0, name="relative_ridge0", positive=True)


def load_primary_sensitivity_binding(
    overlay_config: Mapping[str, object],
    primary_result_json: str | os.PathLike[str],
) -> PrimarySensitivityBinding:
    """Validate the immutable primary seed result needed by sensitivities."""

    validated = validate_phase2_config(overlay_config)
    design = _mapping(validated["design"], name="overlay.design")
    if design.get("stage") != "confirmatory" or design.get("formal_eligibility") is not True:
        raise ValueError("sensitivity execution requires a confirmatory formal overlay")
    declared = tuple(int(value) for value in validated["run"]["seeds"])
    if declared != PHASE2_CONFIRMATORY_SEEDS:
        raise ValueError("sensitivity execution requires the exact ordered 30-seed design")
    design_sha = phase2_design_identity(validated)
    runtime = Phase2Design.from_phase2_config(validated)
    value, source_sha = _read_strict_json(primary_result_json)
    source = Path(primary_result_json).resolve()

    required = {
        "schema_version",
        "design_stage",
        "formal_eligibility",
        "per_seed_supports_formal_claim",
        "source_config_hash",
        "phase2_design_sha256",
        "phase2_runtime_contract_sha256",
        "seed",
        "artifact_metadata_sha256",
        "run_manifest_sha256",
        "environment_identity",
        "train_oracle_rescore",
        "head_training",
        "common_beta_calibration",
        "pre_oracle_safety_gate",
        "arms",
        "heldout_fixed_beta",
        "rollouts_sha256",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"primary result is missing required fields {sorted(missing)!r}")
    if (
        value["schema_version"] != PHASE2_RESULT_SCHEMA
        or value["design_stage"] != "confirmatory"
        or value["formal_eligibility"] is not True
        or value["per_seed_supports_formal_claim"] is not False
        or value["source_config_hash"] != design["source_config_hash"]
        or value["phase2_design_sha256"] != design_sha
        or value["phase2_runtime_contract_sha256"] != runtime.sha256
    ):
        raise ValueError("primary result does not match the confirmatory design identity")
    result_seed = _seed(value["seed"])
    if result_seed not in declared:
        raise ValueError("primary result seed is not in the exact confirmatory seed list")

    gate = _mapping(value["pre_oracle_safety_gate"], name="primary.pre_oracle_safety_gate")
    if (
        gate.get("schema_version") != "phase2-pre-oracle-safety-gate/v1"
        or gate.get("formal_gate") is not True
        or gate.get("measure_only") is not False
        or gate.get("passed") is not True
        or gate.get("violations") != []
        or gate.get("beta_retuned") is not False
    ):
        raise ValueError("primary result did not pass its frozen pre-oracle safety gate")

    environment = _strict_copy(
        _mapping(value["environment_identity"], name="primary.environment_identity"),
        name="primary.environment_identity",
    )
    training = _mapping(value["head_training"], name="primary.head_training")
    heads = _head_weights(training.get("head_weights"), name="primary.head_weights")
    heads_sha = _digest(training.get("heads_sha256"), name="primary.heads_sha256")
    expected_heads_sha = _canonical_sha256(
        {learner: list(heads[learner]) for learner in CANONICAL_LEARNERS}
    )
    if heads_sha != expected_heads_sha:
        raise ValueError("primary heads_sha256 does not match serialized head weights")
    training_design_sha = _digest(
        training.get("training_design_sha256"),
        name="primary.training_design_sha256",
    )
    if training_design_sha != design_sha:
        raise ValueError("primary head training is not bound to the complete design")
    audit = _mapping(training.get("audit"), name="primary.head_training.audit")
    label_stream = _mapping(audit.get("label_stream"), name="primary.audit.label_stream")
    label_stream_sha = _digest(
        label_stream.get("label_stream_sha256"),
        name="primary.audit.label_stream.label_stream_sha256",
    )
    train_oracle = _mapping(value["train_oracle_rescore"], name="primary.train_oracle")
    train_rewards_sha = _digest(
        train_oracle.get("transformed_rewards_sha256"),
        name="primary.train_oracle.transformed_rewards_sha256",
    )
    calibration = _mapping(value["common_beta_calibration"], name="primary.calibration")
    beta0 = _finite(calibration.get("beta_common"), name="primary.beta_common", positive=True)
    if beta0 != runtime.frozen_global_beta:
        raise ValueError("primary result did not use the design-frozen global beta")

    raw_arms = _mapping(value["arms"], name="primary.arms")
    if set(raw_arms) != set(PHASE2_ARM_ORDER):
        raise ValueError("primary result must contain the exact four arms in frozen order")
    primary_arms = {
        arm: _mean_utility_from_arm(
            _mapping(raw_arms[arm], name=f"primary.arms.{arm}"),
            name=f"primary.arms.{arm}",
        )
        for arm in PHASE2_ARM_ORDER
    }
    heldout = _mapping(value["heldout_fixed_beta"], name="primary.heldout")
    splits = _mapping(heldout.get("splits"), name="primary.heldout.splits")
    test = _mapping(splits.get("test"), name="primary.heldout.splits.test")
    learners = _mapping(test.get("learners"), name="primary.heldout.test.learners")
    heldout_regret = {
        learner: _finite(
            _mapping(
                learners.get(learner),
                name=f"primary.heldout.test.learners.{learner}",
            ).get("local_regret_at_frozen_global_beta"),
            name=f"primary.heldout.test.learners.{learner}.local_regret",
        )
        for learner in CANONICAL_LEARNERS
    }
    if any(value < 0.0 for value in heldout_regret.values()):
        raise ValueError("primary held-out local regret cannot be negative")

    ridge = validated["objective"]["full_tangent"]["ridge"]
    return PrimarySensitivityBinding(
        seed=result_seed,
        path=source,
        result_sha256=source_sha,
        source_config_hash=str(design["source_config_hash"]),
        phase2_design_sha256=design_sha,
        runtime_contract_sha256=runtime.sha256,
        environment_identity=environment,
        environment_identity_sha256=_canonical_sha256(environment),
        artifact_metadata_sha256=_digest(
            value["artifact_metadata_sha256"],
            name="primary.artifact_metadata_sha256",
        ),
        run_manifest_sha256=_digest(
            value["run_manifest_sha256"],
            name="primary.run_manifest_sha256",
        ),
        heads=heads,
        heads_sha256=heads_sha,
        training_design_sha256=training_design_sha,
        label_stream_sha256=label_stream_sha,
        transformed_train_rewards_sha256=train_rewards_sha,
        beta0=beta0,
        relative_ridge0=float(ridge["relative_coefficient"]),
        primary_arms=primary_arms,
        primary_heldout_local_regret=heldout_regret,
        primary_rollouts_sha256=_digest(
            value["rollouts_sha256"],
            name="primary.rollouts_sha256",
        ),
        raw_result=value,
    )


def _verify_replayed_label_stream(
    training: TrainingTensorData,
    train_oracle_rewards: torch.Tensor,
    binding: PrimarySensitivityBinding,
    settings: Phase2TrainingSettings,
) -> TrainingTensorData:
    """Recreate the primary noisy arm and prove byte-identical label RNG."""

    probabilities = torch.sigmoid(train_oracle_rewards[:, 0] - train_oracle_rewards[:, 1])
    generator, derived_seed, derivation_sha = _generator_for_training(
        training,
        base_seed=binding.seed,
        namespace=settings.label_rng_namespace,
    )
    initial_state_sha = _tensor_sha256(generator.get_state())
    # Imported lazily to keep this module's public surface narrow.
    from .phase2_controls import sample_canonical_r4_noisy_arm

    noisy = sample_canonical_r4_noisy_arm(
        training,
        probabilities,
        generator=generator,
        max_total_annotations=settings.max_total_annotations,
    )
    final_state_sha = _tensor_sha256(generator.get_state())
    labels = noisy.repeated_labels
    payload = {
        "namespace": settings.label_rng_namespace,
        "base_seed": binding.seed,
        "derived_seed": derived_seed,
        "derivation_sha256": derivation_sha,
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": final_state_sha,
        "probability_sha256": noisy.audit.probability_sha256,
        "replicate_count_sha256": _tensor_sha256(labels.counts),
        "replicate_win_sha256": _tensor_sha256(labels.wins),
        "replicate_h_sha256": _tensor_sha256(labels.replicate_h),
        "mean_h_sha256": _tensor_sha256(noisy.training.h),
        "realized_total_annotations": labels.total_annotations,
    }
    observed = _training_canonical_sha256(payload)
    if observed != binding.label_stream_sha256:
        raise ValueError("sensitivity label replay differs from the frozen primary R=4 stream")
    return noisy.training


def _absolute_ridge(
    training: TrainingTensorData,
    *,
    relative_coefficient: float,
    pcg_dtype: str,
) -> tuple[float, float]:
    relative = _finite(
        relative_coefficient,
        name="relative_coefficient",
        positive=True,
    )
    flat = training.policy_scores.reshape(-1, training.policy_dimension).to(
        dtype=resolve_fisher_solve_dtype(pcg_dtype)
    )
    mean_diagonal = float(flat.square().mean(dim=0).mean().item())
    if not math.isfinite(mean_diagonal) or mean_diagonal <= 0.0:
        raise ValueError("sensitivity train Fisher has a degenerate mean diagonal")
    damping = relative * mean_diagonal
    if not math.isfinite(damping) or damping <= 0.0:
        raise FloatingPointError("sensitivity absolute ridge is invalid")
    return mean_diagonal, damping


def _train_ridge_sensitivity_head(
    noisy_training: TrainingTensorData,
    settings: Phase2TrainingSettings,
    *,
    multiplier: float,
) -> dict[str, object]:
    if multiplier not in OFF_PRIMARY_RIDGE_GRID:
        raise ValueError("only off-primary ridge cells may retrain a sensitivity head")
    relative = settings.relative_damping * multiplier
    mean_diagonal, absolute = _absolute_ridge(
        noisy_training,
        relative_coefficient=relative,
        pcg_dtype=str(settings.pcg_dtype),
    )
    rank = _prorm_moment_map_identifiability(noisy_training, settings)
    model = _zero_model(noisy_training)
    initial_sha = _tensor_sha256(model.weight)
    trainer = ProRMPlusTrainer(
        model,
        noisy_training.to_training_batch(),
        _prorm_config(settings, absolute_damping=absolute),
    )
    convergence = _run_trainer_to_first_order_convergence(
        trainer,
        audit=lambda: _prorm_first_order_measurement(trainer),
        spec=settings.convergence,
        fixed_snapshot_steps=settings.outer_steps,
        objective_name=f"ridge_sensitivity_{multiplier:g}_prorm_plus",
        rank_diagnostic=rank,
    )
    final = _prorm_first_order_measurement(trainer)
    if final.inner_solver is None or final.inner_solver.get("converged") is not True:
        raise RuntimeError("ridge sensitivity final cold-start PCG did not converge")
    head = _make_head_evidence(
        arm=f"ridge_sensitivity_{multiplier:g}",
        method=PRORM_PLUS,
        model=model,
        initial_head_sha256=initial_sha,
        initial_objective=convergence.initial.objective,
        final_objective=convergence.final.objective,
        history=convergence.history,
        final_pcg=final.inner_solver,
        first_order_convergence=convergence.evidence,
    )
    return {
        "schema_version": "phase2-ridge-sensitivity-trained-head/v1",
        "ridge_multiplier": multiplier,
        "relative_ridge": relative,
        "train_mean_fisher_diagonal": mean_diagonal,
        "absolute_ridge": absolute,
        "head": head.to_dict(),
        "label_stream_replayed_exactly": True,
        "bt_head_retrained": False,
        "validation_or_test_used_for_training": False,
        "eligible_for_primary_claim": False,
    }


def _heldout_local_regret(
    policy_scores: torch.Tensor,
    reward_features: torch.Tensor,
    target_rewards: torch.Tensor,
    head: Sequence[float],
    *,
    beta: float,
    relative_ridge: float,
    pcg_dtype: str,
    pcg_max_iterations: int,
    pcg_tolerance: float,
) -> tuple[float, float]:
    head_tensor = torch.tensor(
        tuple(float(value) for value in head),
        dtype=reward_features.dtype,
        device=reward_features.device,
    )
    if head_tensor.shape != (reward_features.shape[-1],):
        raise ValueError("sensitivity reward head has the wrong held-out dimension")
    predicted = reward_features @ head_tensor
    flat = policy_scores.reshape(-1, policy_scores.shape[-1]).to(
        dtype=resolve_fisher_solve_dtype(pcg_dtype)
    )
    mean_diagonal = float(flat.square().mean(dim=0).mean().item())
    damping = relative_ridge * mean_diagonal
    value = local_regret(
        policy_scores,
        predicted,
        target_rewards,
        damping=damping,
        beta=beta,
        pcg_dtype=pcg_dtype,
        pcg_max_iterations=pcg_max_iterations,
        pcg_tolerance=pcg_tolerance,
    )
    result = float(value.item())
    if not math.isfinite(result) or result < 0.0:
        raise FloatingPointError("sensitivity held-out local regret is invalid")
    return result, damping


def _native_directions(
    inputs: Phase2PreparedInputs,
    binding: PrimarySensitivityBinding,
    train_oracle_rewards: torch.Tensor,
    design: Phase2Design,
) -> dict[str, PolicyDirectionResult]:
    common = {
        "relative_damping": design.relative_damping,
        "beta": 1.0,
        "pcg_dtype": design.pcg_dtype,
        "pcg_max_iterations": design.pcg_max_iterations,
        "pcg_tolerance": design.pcg_tolerance,
        "require_pcg_convergence": True,
    }
    directions = {
        learner: policy_direction_from_head(
            inputs.train,
            binding.heads[learner],
            **common,
        )
        for learner in CANONICAL_LEARNERS
    }
    directions["oracle_step"] = policy_direction_from_node_rewards(
        inputs.train,
        train_oracle_rewards,
        **common,
    )
    return directions


def _deploy_for_beta(
    inputs: Phase2PreparedInputs,
    directions: Mapping[str, PolicyDirectionResult],
    design: Phase2Design,
    *,
    beta: float,
) -> Mapping[str, object]:
    solve_dtype = resolve_fisher_solve_dtype(design.pcg_dtype)
    flat = inputs.train.policy_scores.to(dtype=solve_dtype).reshape(
        -1,
        inputs.train.policy_dimension,
    )
    fisher = DampedEmpiricalFisher(flat, damping=0.0)
    calibration = bind_frozen_common_beta(
        directions["oracle_step"].direction,
        fisher.matvec,
        frozen_global_beta=beta,
        reference_target_oracle_quadratic_kl=design.target_oracle_quadratic_kl,
    )
    deployed = deploy_with_common_beta(
        {name: direction.direction for name, direction in directions.items()},
        fisher.matvec,
        calibration=calibration,
    )
    return {
        "calibration": calibration,
        "deployments": _arm_deployments(
            inputs,
            calibration,
            directions,
            deployed,
        ),
    }


def _rollout_identity(
    rollouts: Mapping[str, Phase2PolicyRollout],
    *,
    beta: float,
) -> str:
    payload: list[dict[str, object]] = []
    for arm in PHASE2_ARM_ORDER:
        rollout = rollouts[arm]
        for trajectory, kl in zip(
            rollout.trajectories,
            rollout.per_sequence_kl_updated_to_reference.tolist(),
            strict=True,
        ):
            payload.append(
                {
                    **trajectory.to_unscored_dict(beta_common=beta),
                    "on_policy_kl_pi_updated_to_pi0": float(kl),
                }
            )
    return _canonical_sha256(payload)


def _primary_beta_reference(binding: PrimarySensitivityBinding) -> dict[str, object]:
    return {
        "schema_version": "phase2-beta-sensitivity-cell/v1",
        "multiplier": 1.0,
        "beta": binding.beta0,
        "status": "primary_reference",
        "execution": {
            "reward_heads_reused": True,
            "reward_heads_retrained": False,
            "policy_redeployed": False,
            "rollout_reexecuted": False,
            "source_primary_result_sha256": binding.result_sha256,
            "source_primary_rollouts_sha256": binding.primary_rollouts_sha256,
        },
        "pre_oracle_safety": {
            "passed": True,
            "source": "primary_confirmatory_result",
        },
        "arms": {arm: dict(binding.primary_arms[arm]) for arm in PHASE2_ARM_ORDER},
        "eligible_for_primary_claim": False,
        "primary_result_modified": False,
    }


def _primary_ridge_reference(binding: PrimarySensitivityBinding) -> dict[str, object]:
    return {
        "schema_version": "phase2-ridge-sensitivity-cell/v1",
        "multiplier": 1.0,
        "relative_ridge": binding.relative_ridge0,
        "status": "primary_reference",
        "execution": {
            "prorm_plus_head_retrained": False,
            "bt_mle_head_retrained": False,
            "source_primary_result_sha256": binding.result_sha256,
            "source_heads_sha256": binding.heads_sha256,
        },
        "heldout_test": {
            learner: {
                "local_regret": binding.primary_heldout_local_regret[learner],
            }
            for learner in CANONICAL_LEARNERS
        },
        "eligible_for_primary_claim": False,
        "primary_result_modified": False,
    }


def _score_beta_rollouts(
    inputs: Phase2PreparedInputs,
    backend: Phase2RuntimeBackend,
    pending: Mapping[float, Mapping[str, object]],
    *,
    design: Phase2Design,
) -> dict[float, dict[str, object]]:
    safe = {
        multiplier: record
        for multiplier, record in pending.items()
        if record["pre_oracle_safety"].passed
    }
    if not safe:
        return {}
    ordered: list[tuple[float, str, object]] = []
    for multiplier in OFF_PRIMARY_BETA_GRID:
        if multiplier not in safe:
            continue
        rollouts = safe[multiplier]["rollouts"]
        for arm in PHASE2_ARM_ORDER:
            for trajectory in rollouts[arm].trajectories:
                ordered.append((multiplier, arm, trajectory))
    with backend.oracle_session(
        expected_chat_template_sha256=inputs.oracle_chat_template_sha256
    ) as oracle:
        rewards = oracle.score_transformed(
            tuple(item[2].prompt for item in ordered),
            tuple(item[2].response for item in ordered),
            transform=inputs.oracle_transform,
            batch_size=design.oracle_batch_size,
        )
    if (
        rewards.shape != (len(ordered),)
        or rewards.requires_grad
        or not bool(torch.isfinite(rewards).all())
    ):
        raise ValueError("sensitivity oracle returned malformed rollout scores")
    result: dict[float, dict[str, object]] = {}
    offset = 0
    per_arm = len(inputs.test_prompts) * design.rollout_candidates_per_prompt
    shape = (len(inputs.test_prompts), design.rollout_candidates_per_prompt)
    for multiplier in OFF_PRIMARY_BETA_GRID:
        if multiplier not in safe:
            continue
        record = safe[multiplier]
        rollouts = record["rollouts"]
        arm_rewards: dict[str, torch.Tensor] = {}
        for arm in PHASE2_ARM_ORDER:
            arm_rewards[arm] = (
                rewards[offset : offset + per_arm]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .reshape(shape)
            )
            offset += per_arm
        beta = binding_beta = float(record["beta"])
        zero_rewards = arm_rewards["zero_b"]
        oracle_rewards = arm_rewards["oracle_step"]
        oracle_kl = (
            rollouts["oracle_step"]
            .per_sequence_kl_updated_to_reference.to(torch.float32)
            .reshape(shape)
        )
        arms: dict[str, object] = {}
        for arm in PHASE2_ARM_ORDER:
            kl = rollouts[arm].per_sequence_kl_updated_to_reference.to(torch.float32).reshape(shape)
            summary = summarize_downstream_utility(
                arm_rewards[arm],
                kl,
                zero_rewards,
                beta_common=binding_beta,
                oracle_step_transformed_target_rewards=oracle_rewards,
                oracle_step_on_policy_updated_to_reference_kl=oracle_kl,
            )
            arms[arm] = {
                "mean_target_reward": summary.mean_target_reward,
                "mean_on_policy_kl": summary.mean_on_policy_kl,
                "mean_target_utility": summary.mean_target_utility,
                "reached_max_length_rate": _length_summary(rollouts[arm].trajectories)[
                    "reached_max_length_rate"
                ],
                "kl_tail": _kl_tail_summary(
                    kl,
                    formal_gate_applied=False,
                ),
            }
        result[multiplier] = {
            "schema_version": "phase2-beta-sensitivity-cell/v1",
            "multiplier": multiplier,
            "beta": beta,
            "status": "completed",
            "execution": {
                "reward_heads_reused": True,
                "reward_heads_retrained": False,
                "policy_redeployed": True,
                "rollout_reexecuted": True,
                "source_primary_result_sha256": record["primary_result_sha256"],
                "source_heads_sha256": record["heads_sha256"],
                "rollout_identity_sha256": record["rollout_identity_sha256"],
            },
            "pre_oracle_safety": record["pre_oracle_safety"].to_dict(),
            "arms": arms,
            "eligible_for_primary_claim": False,
            "primary_result_modified": False,
        }
    if offset != len(ordered):
        raise RuntimeError("sensitivity oracle scores were not partitioned exactly")
    return result


def run_phase2_sensitivity_seed(
    inputs: Phase2PreparedInputs,
    binding: PrimarySensitivityBinding,
    backend: Phase2RuntimeBackend,
    *,
    settings: Phase2TrainingSettings | Phase2ConfigBundle | Mapping[str, object],
    design: Phase2Design,
    output_json: str | os.PathLike[str],
) -> dict[str, object]:
    """Execute and atomically publish the complete per-seed sensitivity grid."""

    if not isinstance(inputs, Phase2PreparedInputs):
        raise TypeError("inputs must be Phase2PreparedInputs")
    if not isinstance(binding, PrimarySensitivityBinding):
        raise TypeError("binding must be PrimarySensitivityBinding")
    if not isinstance(design, Phase2Design):
        raise TypeError("design must be Phase2Design")
    compiled = compile_phase2_training_settings(settings)
    if (
        inputs.seed != binding.seed
        or inputs.seed not in PHASE2_CONFIRMATORY_SEEDS
        or inputs.source_config_hash != binding.source_config_hash
        or inputs.phase2_config_hash != binding.phase2_design_sha256
        or design.sha256 != binding.runtime_contract_sha256
        or compiled.phase2_config_hash != binding.phase2_design_sha256
        or design.stage != "confirmatory"
        or design.formal_eligibility is not True
    ):
        raise ValueError("sensitivity inputs, primary result, and design identities differ")
    if dict(inputs.environment_identity) != dict(binding.environment_identity):
        raise ValueError("sensitivity runtime environment differs from the primary seed")

    destination = Path(output_json).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite sensitivity artifact: {destination}")

    train_candidates = inputs.train_candidates
    test_candidates = inputs.heldout.test.candidates
    combined = (*train_candidates, *test_candidates)
    with backend.oracle_session(
        expected_chat_template_sha256=inputs.oracle_chat_template_sha256
    ) as oracle:
        transformed = oracle.score_transformed(
            tuple(candidate.prompt for candidate in combined),
            tuple(candidate.response for candidate in combined),
            transform=inputs.oracle_transform,
            batch_size=design.oracle_batch_size,
        )
    expected = len(combined)
    if (
        transformed.shape != (expected,)
        or transformed.requires_grad
        or not bool(torch.isfinite(transformed).all())
    ):
        raise ValueError("sensitivity train/test oracle rescore returned malformed values")
    train_size = len(train_candidates)
    train_rewards = (
        transformed[:train_size]
        .to(device=inputs.train.policy_scores.device, dtype=inputs.train.policy_scores.dtype)
        .reshape(inputs.train.num_prompts, inputs.train.num_candidates)
        .detach()
        .clone()
    )
    if _tensor_sha256(train_rewards) != binding.transformed_train_rewards_sha256:
        raise ValueError("sensitivity train oracle rescore differs from the primary result")
    test_split = inputs.heldout.test
    test_rewards = (
        transformed[train_size:]
        .to(
            device=test_split.policy_scores.device,
            dtype=test_split.policy_scores.dtype,
        )
        .reshape(test_split.num_prompts, test_split.num_candidates)
        .detach()
        .clone()
    )

    noisy_training = _verify_replayed_label_stream(
        inputs.train,
        train_rewards,
        binding,
        compiled,
    )
    ridge_cells: list[dict[str, object]] = []
    for multiplier in RIDGE_SENSITIVITY_GRID:
        if multiplier == 1.0:
            ridge_cells.append(_primary_ridge_reference(binding))
            continue
        trained = _train_ridge_sensitivity_head(
            noisy_training,
            compiled,
            multiplier=multiplier,
        )
        head = _mapping(trained["head"], name="trained.head")
        prorm_regret, damping = _heldout_local_regret(
            test_split.policy_scores,
            test_split.reward_features,
            test_rewards,
            head["head_weight"],
            beta=binding.beta0,
            relative_ridge=binding.relative_ridge0 * multiplier,
            pcg_dtype=design.pcg_dtype,
            pcg_max_iterations=design.pcg_max_iterations,
            pcg_tolerance=design.pcg_tolerance,
        )
        bt_regret, bt_damping = _heldout_local_regret(
            test_split.policy_scores,
            test_split.reward_features,
            test_rewards,
            binding.heads[BT_MLE],
            beta=binding.beta0,
            relative_ridge=binding.relative_ridge0 * multiplier,
            pcg_dtype=design.pcg_dtype,
            pcg_max_iterations=design.pcg_max_iterations,
            pcg_tolerance=design.pcg_tolerance,
        )
        if damping != bt_damping:
            raise RuntimeError("ridge sensitivity learners used different held-out geometry")
        ridge_cells.append(
            {
                "schema_version": "phase2-ridge-sensitivity-cell/v1",
                "multiplier": multiplier,
                "relative_ridge": binding.relative_ridge0 * multiplier,
                "status": "completed",
                "execution": {
                    "prorm_plus_head_retrained": True,
                    "bt_mle_head_retrained": False,
                    "source_primary_result_sha256": binding.result_sha256,
                    "source_heads_sha256": binding.heads_sha256,
                    "replayed_label_stream_sha256": binding.label_stream_sha256,
                    "trained_prorm_plus_head_sha256": head["head_sha256"],
                    "training_evidence_sha256": _canonical_sha256(trained),
                },
                "heldout_test": {
                    BT_MLE: {
                        "local_regret": bt_regret,
                        "head_sha256": _tensor_sha256(
                            torch.tensor(binding.heads[BT_MLE], dtype=torch.float32)
                        ),
                    },
                    PRORM_PLUS: {
                        "local_regret": prorm_regret,
                        "head_sha256": head["head_sha256"],
                    },
                    "absolute_ridge": damping,
                    "transformed_target_sha256": _tensor_sha256(test_rewards),
                },
                "training_evidence": trained,
                "eligible_for_primary_claim": False,
                "primary_result_modified": False,
            }
        )

    directions = _native_directions(inputs, binding, train_rewards, design)
    pending: dict[float, dict[str, object]] = {}
    for multiplier in OFF_PRIMARY_BETA_GRID:
        beta = binding.beta0 * multiplier
        deployed = _deploy_for_beta(
            inputs,
            directions,
            design,
            beta=beta,
        )
        rollouts, _ = _rollout_policy_arms(
            inputs,
            backend,
            deployed["deployments"],
            design=design,
        )
        pre_oracle = assess_phase2_pre_oracle_safety(rollouts, design=design)
        pending[multiplier] = {
            "beta": beta,
            "rollouts": rollouts,
            "pre_oracle_safety": pre_oracle,
            "rollout_identity_sha256": _rollout_identity(rollouts, beta=beta),
            "primary_result_sha256": binding.result_sha256,
            "heads_sha256": binding.heads_sha256,
        }
    scored = _score_beta_rollouts(
        inputs,
        backend,
        pending,
        design=design,
    )
    beta_cells: list[dict[str, object]] = []
    for multiplier in BETA_SENSITIVITY_GRID:
        if multiplier == 1.0:
            beta_cells.append(_primary_beta_reference(binding))
        elif multiplier in scored:
            beta_cells.append(scored[multiplier])
        else:
            record = pending[multiplier]
            pre_oracle = record["pre_oracle_safety"]
            beta_cells.append(
                {
                    "schema_version": "phase2-beta-sensitivity-cell/v1",
                    "multiplier": multiplier,
                    "beta": record["beta"],
                    "status": "pre_oracle_safety_failed",
                    "execution": {
                        "reward_heads_reused": True,
                        "reward_heads_retrained": False,
                        "policy_redeployed": True,
                        "rollout_reexecuted": True,
                        "source_primary_result_sha256": binding.result_sha256,
                        "source_heads_sha256": binding.heads_sha256,
                        "rollout_identity_sha256": record["rollout_identity_sha256"],
                    },
                    "pre_oracle_safety": pre_oracle.to_dict(),
                    "arms": None,
                    "outcome_oracle_called": False,
                    "eligible_for_primary_claim": False,
                    "primary_result_modified": False,
                }
            )

    payload: dict[str, object] = {
        "schema_version": PHASE2_SENSITIVITY_SEED_SCHEMA,
        "seed": binding.seed,
        "source_config_hash": binding.source_config_hash,
        "phase2_design_sha256": binding.phase2_design_sha256,
        "phase2_runtime_contract_sha256": binding.runtime_contract_sha256,
        "environment_identity": dict(binding.environment_identity),
        "environment_identity_sha256": binding.environment_identity_sha256,
        "primary_result": {
            "path": relative_posix_reference(binding.path, base=destination.parent),
            "sha256": binding.result_sha256,
            "artifact_metadata_sha256": binding.artifact_metadata_sha256,
            "run_manifest_sha256": binding.run_manifest_sha256,
            "heads_sha256": binding.heads_sha256,
            "label_stream_sha256": binding.label_stream_sha256,
            "beta0": binding.beta0,
            "relative_ridge0": binding.relative_ridge0,
            "read_only": True,
            "modified": False,
        },
        "grid_contract": {
            "ridge_multipliers": list(RIDGE_SENSITIVITY_GRID),
            "beta_multipliers": list(BETA_SENSITIVITY_GRID),
            "one_factor_at_a_time": True,
            "primary_reference_multiplier": 1.0,
            "all_cells_retained": True,
            "failed_cells_may_not_be_dropped": True,
        },
        "ridge_cells": ridge_cells,
        "beta_cells": beta_cells,
        "role": "required_secondary_sensitivity_only",
        "eligible_for_primary_claim": False,
        "primary_efficacy_status_read": False,
        "primary_efficacy_status_modified": False,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload, overwrite=False)
    return payload


def _validate_seed_sensitivity(
    path: str | os.PathLike[str],
    *,
    expected_design_sha256: str,
    expected_runtime_sha256: str,
    expected_source_config_hash: str,
    expected_beta0: float,
    expected_relative_ridge0: float,
    reference_base: Path,
) -> dict[str, object]:
    value, source_sha = _read_strict_json(path)
    source = Path(path).resolve()
    required = {
        "schema_version",
        "seed",
        "source_config_hash",
        "phase2_design_sha256",
        "phase2_runtime_contract_sha256",
        "environment_identity",
        "environment_identity_sha256",
        "primary_result",
        "grid_contract",
        "ridge_cells",
        "beta_cells",
        "role",
        "eligible_for_primary_claim",
        "primary_efficacy_status_read",
        "primary_efficacy_status_modified",
        "artifact_sha256",
    }
    if set(value) != required:
        raise ValueError(f"{source} sensitivity fields do not match the strict schema")
    recorded_artifact_sha = _digest(
        value.pop("artifact_sha256"),
        name=f"{source}:artifact_sha256",
    )
    if _canonical_sha256(value) != recorded_artifact_sha:
        raise ValueError(f"{source} sensitivity artifact SHA256 mismatch")
    value["artifact_sha256"] = recorded_artifact_sha
    if (
        value["schema_version"] != PHASE2_SENSITIVITY_SEED_SCHEMA
        or value["source_config_hash"] != expected_source_config_hash
        or value["phase2_design_sha256"] != expected_design_sha256
        or value["phase2_runtime_contract_sha256"] != expected_runtime_sha256
        or value["role"] != "required_secondary_sensitivity_only"
        or value["eligible_for_primary_claim"] is not False
        or value["primary_efficacy_status_read"] is not False
        or value["primary_efficacy_status_modified"] is not False
    ):
        raise ValueError(f"{source} is not an identity-bound secondary sensitivity")
    environment = _mapping(value["environment_identity"], name=f"{source}:environment")
    environment_sha = _digest(
        value["environment_identity_sha256"],
        name=f"{source}:environment_identity_sha256",
    )
    if _canonical_sha256(environment) != environment_sha:
        raise ValueError(f"{source} environment identity hash mismatch")
    primary = _mapping(value["primary_result"], name=f"{source}:primary_result")
    if (
        primary.get("read_only") is not True
        or primary.get("modified") is not False
        or _finite(primary.get("beta0"), name=f"{source}:beta0", positive=True) != expected_beta0
    ):
        raise ValueError(f"{source} does not preserve the primary result")
    primary_result_sha = _digest(
        primary.get("sha256"),
        name=f"{source}:primary_result.sha256",
    )
    relative_ridge0 = _finite(
        primary.get("relative_ridge0"),
        name=f"{source}:primary_result.relative_ridge0",
        positive=True,
    )
    if relative_ridge0 != expected_relative_ridge0:
        raise ValueError(f"{source} changed the primary relative ridge")
    _verify_relative_file_reference(
        primary.get("path"),
        source=source,
        expected_sha256=primary_result_sha,
        name=f"{source}:primary_result.path",
    )
    primary_digests = {}
    for field in (
        "artifact_metadata_sha256",
        "run_manifest_sha256",
        "heads_sha256",
        "label_stream_sha256",
    ):
        primary_digests[field] = _digest(
            primary.get(field),
            name=f"{source}:primary_result.{field}",
        )
    grid = _mapping(value["grid_contract"], name=f"{source}:grid_contract")
    if dict(grid) != {
        "ridge_multipliers": list(RIDGE_SENSITIVITY_GRID),
        "beta_multipliers": list(BETA_SENSITIVITY_GRID),
        "one_factor_at_a_time": True,
        "primary_reference_multiplier": 1.0,
        "all_cells_retained": True,
        "failed_cells_may_not_be_dropped": True,
    }:
        raise ValueError(f"{source} sensitivity grid contract is invalid")
    ridge_cells = value["ridge_cells"]
    beta_cells = value["beta_cells"]
    if (
        not isinstance(ridge_cells, list)
        or [cell.get("multiplier") for cell in ridge_cells if isinstance(cell, Mapping)]
        != list(RIDGE_SENSITIVITY_GRID)
        or not isinstance(beta_cells, list)
        or [cell.get("multiplier") for cell in beta_cells if isinstance(cell, Mapping)]
        != list(BETA_SENSITIVITY_GRID)
    ):
        raise ValueError(f"{source} omits, duplicates, or reorders a sensitivity cell")
    ridge_metrics: dict[float, dict[str, float]] = {}
    for multiplier, raw in zip(RIDGE_SENSITIVITY_GRID, ridge_cells, strict=True):
        cell = _mapping(raw, name=f"{source}:ridge[{multiplier}]")
        expected_status = "primary_reference" if multiplier == 1.0 else "completed"
        if (
            cell.get("schema_version") != "phase2-ridge-sensitivity-cell/v1"
            or cell.get("multiplier") != multiplier
            or cell.get("status") != expected_status
            or _finite(
                cell.get("relative_ridge"),
                name=f"{source}:ridge[{multiplier}].relative_ridge",
                positive=True,
            )
            != expected_relative_ridge0 * multiplier
            or cell.get("eligible_for_primary_claim") is not False
            or cell.get("primary_result_modified") is not False
        ):
            raise ValueError(f"{source} ridge cell {multiplier} is invalid")
        execution = _mapping(
            cell.get("execution"),
            name=f"{source}:ridge[{multiplier}].execution",
        )
        if (
            execution.get("source_primary_result_sha256") != primary_result_sha
            or execution.get("source_heads_sha256") != primary_digests["heads_sha256"]
            or execution.get("bt_mle_head_retrained") is not False
            or execution.get("prorm_plus_head_retrained") is not (multiplier != 1.0)
        ):
            raise ValueError(f"{source} ridge cell {multiplier} violates head reuse")
        if multiplier != 1.0:
            if (
                execution.get("replayed_label_stream_sha256")
                != primary_digests["label_stream_sha256"]
            ):
                raise ValueError(f"{source} ridge cell {multiplier} changed the R=4 label stream")
            for field in (
                "trained_prorm_plus_head_sha256",
                "training_evidence_sha256",
            ):
                _digest(
                    execution.get(field),
                    name=f"{source}:ridge[{multiplier}].execution.{field}",
                )
        heldout = _mapping(
            cell.get("heldout_test"),
            name=f"{source}:ridge[{multiplier}].heldout_test",
        )
        ridge_metrics[multiplier] = {
            learner: _finite(
                _mapping(
                    heldout.get(learner),
                    name=f"{source}:ridge[{multiplier}].{learner}",
                ).get("local_regret"),
                name=f"{source}:ridge[{multiplier}].{learner}.local_regret",
            )
            for learner in CANONICAL_LEARNERS
        }
        if any(metric < 0.0 for metric in ridge_metrics[multiplier].values()):
            raise ValueError(f"{source} ridge local regret cannot be negative")

    beta_metrics: dict[float, dict[str, dict[str, float]] | None] = {}
    beta_status: dict[float, str] = {}
    for multiplier, raw in zip(BETA_SENSITIVITY_GRID, beta_cells, strict=True):
        cell = _mapping(raw, name=f"{source}:beta[{multiplier}]")
        allowed = (
            {"primary_reference"}
            if multiplier == 1.0
            else {"completed", "pre_oracle_safety_failed"}
        )
        status = cell.get("status")
        if (
            cell.get("schema_version") != "phase2-beta-sensitivity-cell/v1"
            or cell.get("multiplier") != multiplier
            or status not in allowed
            or cell.get("eligible_for_primary_claim") is not False
            or cell.get("primary_result_modified") is not False
            or _finite(cell.get("beta"), name=f"{source}:beta", positive=True)
            != expected_beta0 * multiplier
        ):
            raise ValueError(f"{source} beta cell {multiplier} is invalid")
        execution = _mapping(
            cell.get("execution"),
            name=f"{source}:beta[{multiplier}].execution",
        )
        if (
            execution.get("source_primary_result_sha256") != primary_result_sha
            or execution.get("reward_heads_reused") is not True
            or execution.get("reward_heads_retrained") is not False
            or execution.get("policy_redeployed") is not (multiplier != 1.0)
            or execution.get("rollout_reexecuted") is not (multiplier != 1.0)
        ):
            raise ValueError(f"{source} beta cell {multiplier} violates frozen-head reuse")
        if multiplier == 1.0:
            _digest(
                execution.get("source_primary_rollouts_sha256"),
                name=f"{source}:beta[{multiplier}].source_primary_rollouts_sha256",
            )
        else:
            if execution.get("source_heads_sha256") != primary_digests["heads_sha256"]:
                raise ValueError(f"{source} beta cell {multiplier} changed the frozen reward heads")
            _digest(
                execution.get("rollout_identity_sha256"),
                name=f"{source}:beta[{multiplier}].rollout_identity_sha256",
            )
        safety = _mapping(
            cell.get("pre_oracle_safety"),
            name=f"{source}:beta[{multiplier}].safety",
        )
        if status == "pre_oracle_safety_failed":
            if (
                safety.get("passed") is not False
                or cell.get("arms") is not None
                or cell.get("outcome_oracle_called") is not False
            ):
                raise ValueError(f"{source} unsafe beta cell leaked outcome metrics")
            beta_metrics[multiplier] = None
        else:
            if safety.get("passed") is not True:
                raise ValueError(f"{source} completed beta cell lacks a passed safety gate")
            arms = _mapping(cell.get("arms"), name=f"{source}:beta[{multiplier}].arms")
            if set(arms) != set(PHASE2_ARM_ORDER):
                raise ValueError(f"{source} beta cell has a selected arm subset")
            beta_metrics[multiplier] = {
                arm: {
                    field: _finite(
                        _mapping(arms[arm], name=f"{source}:beta.{arm}").get(field),
                        name=f"{source}:beta.{arm}.{field}",
                    )
                    for field in (
                        "mean_target_reward",
                        "mean_on_policy_kl",
                        "mean_target_utility",
                    )
                }
                for arm in PHASE2_ARM_ORDER
            }
            if any(
                metrics["mean_on_policy_kl"] < 0.0 for metrics in beta_metrics[multiplier].values()
            ):
                raise ValueError(f"{source} beta cell contains a negative on-policy KL")
        beta_status[multiplier] = str(status)
    return {
        "seed": _seed(value["seed"]),
        "source_path": relative_posix_reference(source, base=reference_base),
        "source_sha256": source_sha,
        "artifact_sha256": recorded_artifact_sha,
        "primary_result_sha256": primary_result_sha,
        "environment_identity": dict(environment),
        "environment_identity_sha256": environment_sha,
        "ridge_metrics": ridge_metrics,
        "beta_metrics": beta_metrics,
        "beta_status": beta_status,
    }


def _validate_primary_aggregate(
    path: str | os.PathLike[str],
    *,
    design_sha256: str,
    runtime_sha256: str,
    source_config_hash: str,
    expected_seeds: tuple[int, ...],
) -> dict[str, object]:
    value, source_sha = _read_strict_json(path)
    from .phase2_aggregate import PHASE2_AGGREGATE_SCHEMA

    if (
        value.get("schema_version") != PHASE2_AGGREGATE_SCHEMA
        or value.get("phase2_design_sha256") != design_sha256
        or value.get("phase2_runtime_contract_sha256") != runtime_sha256
        or value.get("source_config_hash") != source_config_hash
        or tuple(value.get("seeds", ())) != expected_seeds
    ):
        raise ValueError("primary aggregate does not match the exact confirmatory campaign")
    evidence = _mapping(value.get("pre_registered_evidence"), name="primary.evidence")
    status = evidence.get("status")
    if status not in {"passed", "not_passed"}:
        raise ValueError("primary aggregate has an invalid terminal efficacy status")
    supports = evidence.get("supports_pre_registered_claim")
    if not isinstance(supports, bool) or supports is not (status == "passed"):
        raise ValueError("primary aggregate efficacy status/support boolean disagree")
    environment = _strict_copy(
        _mapping(
            value.get("environment_identity"),
            name="primary.environment_identity",
        ),
        name="primary.environment_identity",
    )
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != len(expected_seeds):
        raise ValueError("primary aggregate sources are incomplete")
    result_sha_by_seed: dict[int, str] = {}
    for expected_seed, raw in zip(expected_seeds, sources, strict=True):
        item = _mapping(raw, name="primary.sources")
        if _seed(item.get("seed")) != expected_seed:
            raise ValueError("primary aggregate source order is invalid")
        result_sha_by_seed[expected_seed] = _digest(
            item.get("result_sha256"),
            name="primary.sources.result_sha256",
        )
    return {
        "path": Path(path).resolve(),
        "sha256": source_sha,
        "evidence_status": status,
        "supports_pre_registered_claim": supports,
        "result_sha_by_seed": result_sha_by_seed,
        "environment_identity": environment,
    }


def build_phase2_sensitivity_aggregate(
    overlay_config: Mapping[str, object],
    sensitivity_jsons: Sequence[str | os.PathLike[str]],
    *,
    primary_aggregate_json: str | os.PathLike[str],
    reference_base: str | os.PathLike[str],
) -> dict[str, object]:
    """Validate exact-30/full-grid evidence and aggregate without subsetting."""

    validated = validate_phase2_config(overlay_config)
    design = _mapping(validated["design"], name="overlay.design")
    if design.get("stage") != "confirmatory" or design.get("formal_eligibility") is not True:
        raise ValueError("sensitivity aggregation requires a formal confirmatory overlay")
    seeds = tuple(int(value) for value in validated["run"]["seeds"])
    if seeds != PHASE2_CONFIRMATORY_SEEDS:
        raise ValueError("sensitivity aggregation requires the exact ordered 30 seeds")
    if (
        isinstance(sensitivity_jsons, (str, bytes, bytearray))
        or not isinstance(sensitivity_jsons, Sequence)
        or len(sensitivity_jsons) != len(seeds)
    ):
        raise ValueError("sensitivity aggregation requires exactly 30 explicit artifacts")
    resolved = tuple(str(Path(path).resolve()) for path in sensitivity_jsons)
    if len(set(resolved)) != len(resolved):
        raise ValueError("sensitivity artifact paths must be unique")
    design_sha = phase2_design_identity(validated)
    runtime = Phase2Design.from_phase2_config(validated)
    source_hash = str(design["source_config_hash"])
    beta0 = _finite(runtime.frozen_global_beta, name="frozen beta", positive=True)
    relative_ridge0 = _finite(
        validated["objective"]["full_tangent"]["ridge"]["relative_coefficient"],
        name="primary relative ridge",
        positive=True,
    )
    base = Path(reference_base).resolve()
    primary = _validate_primary_aggregate(
        primary_aggregate_json,
        design_sha256=design_sha,
        runtime_sha256=runtime.sha256,
        source_config_hash=source_hash,
        expected_seeds=seeds,
    )
    loaded: dict[int, dict[str, object]] = {}
    for path in sensitivity_jsons:
        item = _validate_seed_sensitivity(
            path,
            expected_design_sha256=design_sha,
            expected_runtime_sha256=runtime.sha256,
            expected_source_config_hash=source_hash,
            expected_beta0=beta0,
            expected_relative_ridge0=relative_ridge0,
            reference_base=base,
        )
        item_seed = int(item["seed"])
        if item_seed in loaded:
            raise ValueError(f"duplicate sensitivity artifact for seed {item_seed}")
        loaded[item_seed] = item
    if set(loaded) != set(seeds):
        raise ValueError(
            "sensitivity artifacts must exactly match the 30 formal seeds; "
            f"missing={sorted(set(seeds) - set(loaded))!r}, "
            f"unexpected={sorted(set(loaded) - set(seeds))!r}"
        )
    for seed in seeds:
        if loaded[seed]["primary_result_sha256"] != primary["result_sha_by_seed"][seed]:
            raise ValueError(
                f"sensitivity artifact for seed {seed} is not bound to its primary source"
            )
    environment = loaded[seeds[0]]["environment_identity"]
    if any(loaded[seed]["environment_identity"] != environment for seed in seeds[1:]):
        raise ValueError("all sensitivity seeds must share the formal environment identity")
    if environment != primary["environment_identity"]:
        raise ValueError("sensitivity environment identity differs from the primary aggregate")

    bootstrap_seed = int(validated["evaluation"]["paired_bootstrap_seed"])
    bootstrap_resamples = int(validated["evaluation"]["paired_bootstrap_resamples"])
    ridge_aggregates: dict[str, object] = {}
    for multiplier in RIDGE_SENSITIVITY_GRID:
        bt = {
            seed: {"heldout_test_local_regret": loaded[seed]["ridge_metrics"][multiplier][BT_MLE]}
            for seed in seeds
        }
        prorm = {
            seed: {
                "heldout_test_local_regret": loaded[seed]["ridge_metrics"][multiplier][PRORM_PLUS]
            }
            for seed in seeds
        }
        ridge_aggregates[str(multiplier)] = {
            "cell_status": ("primary_reference" if multiplier == 1.0 else "completed_exact_30"),
            "multiplier": multiplier,
            "relative_ridge": float(
                validated["objective"]["full_tangent"]["ridge"]["relative_coefficient"]
            )
            * multiplier,
            "paired_prorm_plus_minus_bt": aggregate_paired_metrics(
                bt,
                prorm,
                directions={"heldout_test_local_regret": "lower_is_better"},
                bootstrap_seed=bootstrap_seed,
                num_resamples=bootstrap_resamples,
            ).to_dict(),
        }

    beta_aggregates: dict[str, object] = {}
    for multiplier in BETA_SENSITIVITY_GRID:
        statuses = [loaded[seed]["beta_status"][multiplier] for seed in seeds]
        if any(status == "pre_oracle_safety_failed" for status in statuses):
            beta_aggregates[str(multiplier)] = {
                "cell_status": "not_estimable_due_to_pre_oracle_safety_failure",
                "multiplier": multiplier,
                "beta": beta0 * multiplier,
                "failed_seeds": [
                    seed
                    for seed in seeds
                    if loaded[seed]["beta_status"][multiplier] == "pre_oracle_safety_failed"
                ],
                "paired_prorm_plus_minus_bt": None,
                "subset_interval_computed": False,
            }
            continue
        expected = "primary_reference" if multiplier == 1.0 else "completed"
        if any(status != expected for status in statuses):
            raise ValueError(f"beta multiplier {multiplier} has an unknown terminal status")
        bt = {
            seed: {
                "mean_target_utility": loaded[seed]["beta_metrics"][multiplier][BT_MLE][
                    "mean_target_utility"
                ]
            }
            for seed in seeds
        }
        prorm = {
            seed: {
                "mean_target_utility": loaded[seed]["beta_metrics"][multiplier][PRORM_PLUS][
                    "mean_target_utility"
                ]
            }
            for seed in seeds
        }
        beta_aggregates[str(multiplier)] = {
            "cell_status": ("primary_reference" if multiplier == 1.0 else "completed_exact_30"),
            "multiplier": multiplier,
            "beta": beta0 * multiplier,
            "paired_prorm_plus_minus_bt": aggregate_paired_metrics(
                bt,
                prorm,
                directions={"mean_target_utility": "higher_is_better"},
                bootstrap_seed=bootstrap_seed,
                num_resamples=bootstrap_resamples,
            ).to_dict(),
            "subset_interval_computed": False,
        }

    payload: dict[str, object] = {
        "schema_version": PHASE2_SENSITIVITY_AGGREGATE_SCHEMA,
        "source_config_hash": source_hash,
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract_sha256": runtime.sha256,
        "seeds": list(seeds),
        "num_seeds": len(seeds),
        "experimental_unit": "seed",
        "environment_identity": dict(environment),
        "primary_aggregate": {
            "path": relative_posix_reference(primary["path"], base=base),
            "sha256": primary["sha256"],
            "efficacy_status": primary["evidence_status"],
            "supports_pre_registered_claim": primary["supports_pre_registered_claim"],
            "read_only": True,
            "modified": False,
        },
        "grid_contract": {
            "ridge_multipliers": list(RIDGE_SENSITIVITY_GRID),
            "beta_multipliers": list(BETA_SENSITIVITY_GRID),
            "exact_seed_set": list(seeds),
            "one_factor_at_a_time": True,
            "missing_cell_allowed": False,
            "selected_seed_subset_allowed": False,
            "unsafe_cell_subset_interval_allowed": False,
            "complete": True,
        },
        "ridge": ridge_aggregates,
        "beta": beta_aggregates,
        "claim_contract": {
            "role": "secondary_sensitivity_only",
            "eligible_to_modify_primary_efficacy_status": False,
            "primary_efficacy_status_copied_without_recomputation": True,
            "post_hoc_multiplier_selection_allowed": False,
        },
        "sources": [
            {
                "seed": seed,
                "path": loaded[seed]["source_path"],
                "file_sha256": loaded[seed]["source_sha256"],
                "artifact_sha256": loaded[seed]["artifact_sha256"],
                "primary_result_sha256": loaded[seed]["primary_result_sha256"],
            }
            for seed in seeds
        ],
    }
    payload["aggregate_sha256"] = _canonical_sha256(payload)
    return payload


def write_phase2_sensitivity_aggregate(
    overlay_config: Mapping[str, object],
    sensitivity_jsons: Sequence[str | os.PathLike[str]],
    output_json: str | os.PathLike[str],
    *,
    primary_aggregate_json: str | os.PathLike[str],
) -> dict[str, object]:
    destination = Path(output_json).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite sensitivity aggregate: {destination}")
    payload = build_phase2_sensitivity_aggregate(
        overlay_config,
        sensitivity_jsons,
        primary_aggregate_json=primary_aggregate_json,
        reference_base=destination.parent,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload, overwrite=False)
    return payload


__all__ = [
    "BETA_SENSITIVITY_GRID",
    "OFF_PRIMARY_BETA_GRID",
    "OFF_PRIMARY_RIDGE_GRID",
    "PHASE2_SENSITIVITY_AGGREGATE_SCHEMA",
    "PHASE2_SENSITIVITY_SEED_SCHEMA",
    "RIDGE_SENSITIVITY_GRID",
    "PrimarySensitivityBinding",
    "build_phase2_sensitivity_aggregate",
    "load_primary_sensitivity_binding",
    "run_phase2_sensitivity_seed",
    "write_phase2_sensitivity_aggregate",
]
