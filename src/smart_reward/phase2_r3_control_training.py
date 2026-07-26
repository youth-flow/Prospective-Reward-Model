"""Independent scientific execution for the three R3 Gate-C families.

Every public entry point consumes the sealed control-only input capability
created before primary R4 labels exist.  The module reuses the production
Phase-2 trainers, first-order controller, direct-oracle identity, seeded
projection, and Moore-Penrose geometry.  It never accepts a primary head,
optimizer state, checkpoint, label artifact, held-out split, policy backend,
or downstream outcome.
"""

from __future__ import annotations

import io
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final

import torch

from .experiment import TrainingTensorData
from .phase2_checkpoint import PlannedSegmentBoundary
from .phase2_controls import (
    TangentCoordinateLayout,
    build_direct_oracle_geometry_control,
    build_exact_margin_canonical_arm,
    sample_canonical_r4_noisy_arm,
    select_seeded_orthonormal_tangent,
)
from .phase2_r3_controls import (
    R3_CONTROL_FIRST_ORDER_GATE_SCHEMA,
    R3_GATE_C_FAMILIES,
    R3_GATE_C_PROFILE_UPDATES,
    R3_GATE_C_SEEDS,
    R3ControlFamily,
    R3ControlsConfigBundle,
    adapt_exact_margin_prorm_plus_result,
    adapt_exact_soft_label_bt_result,
    adapt_low_dimensional_prorm_plus_result,
    validate_r3_control_family_result,
)
from .phase2_r3_inputs import R3ControlTrainInputCapability
from .phase2_training import (
    Phase2TrainingSettings,
    _absolute_damping,
    _bt_config,
    _build_dense_pseudoinverse_geometry,
    _canonical_sha256,
    _compact_direct_oracle_identity,
    _DensePseudoinverseGeometry,
    _DensePseudoinverseProRMTrainer,
    _evaluate_dense_prorm,
    _exact_soft_bt_first_order_measurement,
    _ExactSoftLabelBTBatch,
    _ExactSoftLabelBTTrainer,
    _FirstOrderMeasurement,
    _generator_for_training,
    _prorm_config,
    _prorm_first_order_measurement,
    _run_trainer_to_first_order_convergence,
    _tensor_sha256,
    _validate_frozen_oracle_rewards,
    _zero_model,
)
from .rollout import PolicyDirectionResult, policy_direction_from_head
from .training import ProRMPlusTrainer

R3_CONTROL_PROFILE_OBSERVATION_SCHEMA: Final = "phase2-recovery-r3-control-profile-observation/v1"
_PROFILE_STOP_REASON: Final = "fixed_nonreusable_100_update_profile_boundary"


def _family(value: object) -> R3ControlFamily:
    if value not in R3_GATE_C_FAMILIES:
        raise ValueError(f"family must be one of {R3_GATE_C_FAMILIES!r}")
    return value  # type: ignore[return-value]


def _require_capability(value: object) -> R3ControlTrainInputCapability:
    if type(value) is not R3ControlTrainInputCapability:
        raise TypeError("input_capability must be an exact R3ControlTrainInputCapability")
    value.validate_integrity()
    return value


def _require_controls(value: object) -> R3ControlsConfigBundle:
    if type(value) is not R3ControlsConfigBundle:
        raise TypeError("controls_config must be an exact R3ControlsConfigBundle")
    value.validate_integrity()
    return value


def _closed_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a mapping with string keys")
    return dict(value)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tolerances(config: R3ControlsConfigBundle) -> dict[str, float]:
    raw = _closed_mapping(
        config.normalized.get("numeric_gate_tolerances"),
        name="Gate-C numeric tolerances",
    )
    return {
        name: _finite_nonnegative(value, name=f"Gate-C tolerance {name}")
        for name, value in raw.items()
    }


def _validate_science_binding(
    capability: R3ControlTrainInputCapability,
    controls: R3ControlsConfigBundle,
) -> Phase2TrainingSettings:
    settings = capability.science_bundle.settings
    science = _closed_mapping(
        controls.normalized.get("science_binding"),
        name="Gate-C science binding",
    )
    if (
        science.get("source_semantic_sha256") != capability.science_semantic_sha256
        or science.get("source_file_sha256") != capability.science_file_sha256
    ):
        raise ValueError("Gate-C input belongs to another R3 science identity")
    if capability.seed not in controls.seeds:
        raise ValueError("Gate-C input seed is outside the exact three-seed matrix")
    protocol = settings.convergence.optimizer_protocol
    if protocol is None:
        raise ValueError("Gate-C requires the frozen R3 recovery optimizer protocol")
    if (
        settings.convergence.min_steps != controls.minimum_updates
        or settings.convergence.max_steps != controls.maximum_updates
        or settings.convergence.check_interval != controls.audit_interval_updates
    ):
        raise ValueError("Gate-C convergence schedule differs from its source science")
    return settings


@dataclass(slots=True)
class _PreparedControl:
    family: R3ControlFamily
    full_training: TrainingTensorData
    train_oracle_rewards: torch.Tensor
    settings: Phase2TrainingSettings
    trainer: Any
    audit: Callable[[], _FirstOrderMeasurement]
    initial_head_sha256: str
    canonical_margin_sha256: str
    reward_feature_difference_sha256: str
    exact_arm: object | None = None
    direct_control: object | None = None
    projection_control: object | None = None
    dense_geometry: _DensePseudoinverseGeometry | None = None
    family_local_label_stream_sha256: str | None = None


def _canonical_target_hashes(
    training: TrainingTensorData,
    rewards: torch.Tensor,
) -> tuple[str, str]:
    canonical_margin = (rewards[:, 0] - rewards[:, 1]).detach()
    reward_feature_difference = (
        training.reward_features[:, 0] - training.reward_features[:, 1]
    ).detach()
    return (
        _tensor_sha256(canonical_margin),
        _tensor_sha256(reward_feature_difference),
    )


def _family_local_r4_arm(
    training: TrainingTensorData,
    rewards: torch.Tensor,
    *,
    seed: int,
    settings: Phase2TrainingSettings,
) -> tuple[object, str]:
    probabilities = torch.sigmoid(rewards[:, 0] - rewards[:, 1]).detach()
    tolerance = 2.0e-6 if probabilities.dtype == torch.float32 else 2.0e-12
    floor = settings.probability_floor
    if bool(
        ((probabilities < floor - tolerance) | (probabilities > 1.0 - floor + tolerance)).any()
    ):
        raise ValueError(
            "Gate-C oracle margins violate the frozen transformed-oracle probability range"
        )
    generator, derived_seed, derivation_sha256 = _generator_for_training(
        training,
        base_seed=seed,
        namespace=settings.label_rng_namespace,
    )
    initial_state_sha256 = _tensor_sha256(generator.get_state())
    noisy_arm = sample_canonical_r4_noisy_arm(
        training,
        probabilities,
        generator=generator,
        max_total_annotations=settings.max_total_annotations,
    )
    labels = noisy_arm.repeated_labels
    label_payload = {
        "schema_version": "phase2-r3-low-dimensional-family-local-r4-stream/v1",
        "family": "low_dimensional_prorm_plus",
        "namespace": settings.label_rng_namespace,
        "base_seed": seed,
        "derived_seed": derived_seed,
        "derivation_sha256": derivation_sha256,
        "initial_generator_state_sha256": initial_state_sha256,
        "final_generator_state_sha256": _tensor_sha256(generator.get_state()),
        "oracle_reward_sha256": _tensor_sha256(rewards),
        "canonical_probability_sha256": noisy_arm.audit.probability_sha256,
        "replicate_count_sha256": _tensor_sha256(labels.counts),
        "replicate_win_sha256": _tensor_sha256(labels.wins),
        "replicate_h_sha256": _tensor_sha256(labels.replicate_h),
        "mean_h_sha256": _tensor_sha256(noisy_arm.training.h),
        "independent_replicates_per_edge": settings.num_label_replicates,
        "annotation_gamma": settings.annotation_gamma,
        "primary_label_stream_accessed": False,
    }
    return noisy_arm, _canonical_sha256(label_payload)


def _prepare_control(
    capability: R3ControlTrainInputCapability,
    controls: R3ControlsConfigBundle,
    family: R3ControlFamily,
) -> _PreparedControl:
    settings = _validate_science_binding(capability, controls)
    training = capability.training
    rewards = _validate_frozen_oracle_rewards(
        training,
        capability.train_oracle_rewards,
    )
    if settings.pcg_max_iterations < training.num_prompts * training.num_candidates + 1:
        raise ValueError("Gate-C PCG cap does not cover the train Fisher rank bound plus one")
    canonical_margin_sha256, feature_difference_sha256 = _canonical_target_hashes(
        training,
        rewards,
    )

    if family in {"exact_margin_prorm_plus", "exact_soft_label_bt"}:
        exact_arm = build_exact_margin_canonical_arm(training, rewards)
        if exact_arm.audit.exact_margin_sha256 != canonical_margin_sha256:
            raise RuntimeError("exact-margin arm changed its canonical target")
        model = _zero_model(exact_arm.training)
        initial_head_sha256 = _tensor_sha256(model.weight)
        if family == "exact_margin_prorm_plus":
            absolute_damping = _absolute_damping(training, settings)
            trainer = ProRMPlusTrainer(
                model,
                exact_arm.training.to_training_batch(),
                _prorm_config(settings, absolute_damping=absolute_damping),
            )
            direct = build_direct_oracle_geometry_control(
                training,
                rewards,
                relative_damping=settings.relative_damping,
                pcg_dtype=settings.pcg_dtype,
                pcg_max_iterations=settings.pcg_max_iterations,
                pcg_tolerance=settings.pcg_tolerance,
                pcg_absolute_tolerance=settings.pcg_absolute_tolerance,
                pcg_residual_recompute_interval=(settings.pcg_residual_recompute_interval),
                require_pcg_convergence=True,
            )
            return _PreparedControl(
                family=family,
                full_training=training,
                train_oracle_rewards=rewards,
                settings=settings,
                trainer=trainer,
                audit=partial(_prorm_first_order_measurement, trainer),
                initial_head_sha256=initial_head_sha256,
                canonical_margin_sha256=canonical_margin_sha256,
                reward_feature_difference_sha256=feature_difference_sha256,
                exact_arm=exact_arm,
                direct_control=direct,
            )
        batch = _ExactSoftLabelBTBatch.from_exact_margin_training(exact_arm.training)
        trainer = _ExactSoftLabelBTTrainer(model, batch, _bt_config(settings))
        return _PreparedControl(
            family=family,
            full_training=training,
            train_oracle_rewards=rewards,
            settings=settings,
            trainer=trainer,
            audit=partial(_exact_soft_bt_first_order_measurement, trainer),
            initial_head_sha256=initial_head_sha256,
            canonical_margin_sha256=canonical_margin_sha256,
            reward_feature_difference_sha256=feature_difference_sha256,
            exact_arm=exact_arm,
        )

    selected_dimension = settings.low_dimensional_selected_dimension
    if (
        selected_dimension >= training.policy_dimension
        or selected_dimension >= training.num_prompts * training.num_candidates
    ):
        raise ValueError("Gate-C low-dimensional projection violates d < min(D, n_F)")
    noisy_arm, label_stream_sha256 = _family_local_r4_arm(
        training,
        rewards,
        seed=capability.seed,
        settings=settings,
    )
    coordinate_layout = TangentCoordinateLayout(
        layout_id=settings.low_dimensional_source_layout_id,
        coordinate_ids=tuple(range(training.policy_dimension)),
    )
    projection = select_seeded_orthonormal_tangent(
        noisy_arm.training,
        selected_dimension=selected_dimension,
        coordinate_layout=coordinate_layout,
        seed=capability.seed,
        namespace=settings.low_dimensional_namespace,
    )
    batch, geometry = _build_dense_pseudoinverse_geometry(
        projection.training,
        settings,
    )
    if geometry.rank != selected_dimension:
        raise ValueError("Gate-C low-dimensional Fisher does not have the preregistered exact rank")
    model = _zero_model(projection.training)
    trainer = _DensePseudoinverseProRMTrainer(
        model,
        batch,
        geometry,
        settings,
    )
    return _PreparedControl(
        family=family,
        full_training=training,
        train_oracle_rewards=rewards,
        settings=settings,
        trainer=trainer,
        audit=trainer.audit,
        initial_head_sha256=_tensor_sha256(model.weight),
        canonical_margin_sha256=canonical_margin_sha256,
        reward_feature_difference_sha256=feature_difference_sha256,
        projection_control=projection,
        dense_geometry=geometry,
        family_local_label_stream_sha256=label_stream_sha256,
    )


def _measurement_from_mapping(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    result = _closed_mapping(value, name=name)
    objective = _finite_nonnegative(result.get("objective"), name=f"{name} objective")
    gradient = _finite_nonnegative(
        result.get("gradient_l2_norm"),
        name=f"{name} gradient",
    )
    return {
        **result,
        "objective": objective,
        "gradient_l2_norm": gradient,
    }


def _first_order_gate(
    convergence: object,
    *,
    objective_name: str,
    settings: Phase2TrainingSettings,
) -> dict[str, object]:
    evidence = _closed_mapping(convergence, name="first-order convergence evidence")
    selected_step = evidence.get("selected_primary_step")
    if isinstance(selected_step, bool) or not isinstance(selected_step, int):
        raise TypeError("selected first-order step must be an integer")
    initial = _measurement_from_mapping(
        evidence.get("initial_zero_head_measurement"),
        name="initial first-order measurement",
    )
    final_gate = _closed_mapping(
        evidence.get("final_gate"),
        name="final first-order gate",
    )
    final = _measurement_from_mapping(
        final_gate.get("measurement"),
        name="final first-order measurement",
    )
    checks = evidence.get("checks")
    if not isinstance(checks, list):
        raise TypeError("first-order convergence checks must be a list")
    expected_steps = [
        selected_step - offset * settings.convergence.check_interval
        for offset in range(settings.convergence.consecutive_checks - 1, -1, -1)
    ]
    by_step: dict[int, dict[str, object]] = {}
    for raw in checks:
        check = _closed_mapping(raw, name="first-order check")
        step = check.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("first-order check step must be an integer")
        by_step[step] = check
    sustained: list[dict[str, object]] = []
    for step in expected_steps:
        if step not in by_step:
            raise RuntimeError("selected first-order gate lacks a sustained check")
        check = by_step[step]
        measurement = _measurement_from_mapping(
            check.get("measurement"),
            name=f"first-order check {step}",
        )
        sustained.append(
            {
                "step": step,
                "gradient_l2_norm": measurement["gradient_l2_norm"],
                "gradient_ratio_to_zero_initialization": check.get(
                    "gradient_ratio_to_zero_initialization"
                ),
                "threshold_passed": check.get("threshold_passed"),
            }
        )
    protocol = settings.convergence.optimizer_protocol
    if protocol is None:
        raise RuntimeError("Gate-C first-order evidence lacks the recovery protocol")
    denominator = max(
        float(initial["gradient_l2_norm"]),
        settings.convergence.gradient_norm_denominator_floor,
    )
    return {
        "schema_version": R3_CONTROL_FIRST_ORDER_GATE_SCHEMA,
        "objective": objective_name,
        "learning_rate_schedule_sha256": protocol.schedule_sha256,
        "initial_full_data_unclipped_gradient_l2_norm": initial["gradient_l2_norm"],
        "gradient_norm_denominator": denominator,
        "final_full_data_unclipped_gradient_l2_norm": final["gradient_l2_norm"],
        "gradient_ratio_to_zero_initialization": (float(final["gradient_l2_norm"]) / denominator),
        "selected_step": selected_step,
        "consecutive_passing_checks": settings.convergence.consecutive_checks,
        "sustained_checks": sustained,
        "full_data_post_update_unclipped": True,
        "fresh_zero_initialized": True,
        "fresh_post_restore_audit": final_gate.get("fresh_post_restore_audit"),
        "test_or_validation_data_accessed": False,
        "passed": True,
    }


def _pcg_from_mapping(value: object, *, name: str) -> dict[str, object]:
    pcg = _closed_mapping(value, name=name)
    return {
        "schema_version": "phase2-r3-control-pcg/v1",
        "iterations": pcg.get("iterations"),
        "residual_norm": pcg.get("residual_norm"),
        "relative_residual": pcg.get("relative_residual"),
        "converged": pcg.get("converged"),
        "cold_start": pcg.get("cold_start", True),
        "warm_start_used": pcg.get("warm_start_used", False),
    }


def _pcg_from_direction(value: PolicyDirectionResult) -> dict[str, object]:
    return {
        "schema_version": "phase2-r3-control-pcg/v1",
        "iterations": value.pcg_iterations,
        "residual_norm": value.pcg_residual_norm,
        "relative_residual": value.pcg_relative_residual,
        "converged": value.pcg_converged,
        "cold_start": True,
        "warm_start_used": False,
    }


def _head_audit(
    prepared: _PreparedControl,
    *,
    method: str,
    objective_name: str,
    convergence: object,
    cold: _FirstOrderMeasurement,
) -> dict[str, object]:
    evidence = _closed_mapping(convergence, name="first-order convergence evidence")
    initial = _measurement_from_mapping(
        evidence.get("initial_zero_head_measurement"),
        name="initial first-order measurement",
    )
    final_gate = _closed_mapping(evidence.get("final_gate"), name="final first-order gate")
    final = _measurement_from_mapping(
        final_gate.get("measurement"),
        name="final first-order measurement",
    )
    first_order = _first_order_gate(
        evidence,
        objective_name=objective_name,
        settings=prepared.settings,
    )
    final_objective = float(final["objective"])
    initial_objective = float(initial["objective"])
    binding = math.isclose(
        cold.objective,
        final_objective,
        rel_tol=2.0e-5,
        abs_tol=2.0e-7,
    )
    return {
        "schema_version": "phase2-r3-control-head-audit/v1",
        "method": method,
        "objective": objective_name,
        "initial_head_sha256": prepared.initial_head_sha256,
        "head_sha256": _tensor_sha256(prepared.trainer.model.weight),
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "cold_full_data_audit_objective": cold.objective,
        "cold_full_data_audit_gradient_l2_norm": cold.gradient_l2_norm,
        "objective_decrease_passed": final_objective < initial_objective,
        "objective_binding_passed": binding,
        "first_order_gate": first_order,
        "fresh_zero_initialized": True,
        "raw_head_weight_retained": False,
    }


def _common_target(prepared: _PreparedControl) -> dict[str, object]:
    training = prepared.full_training
    return {
        "split": "train",
        "orientation": "candidate_0_minus_candidate_1",
        "source_node_rewards_sha256": _tensor_sha256(prepared.train_oracle_rewards),
        "canonical_margin_sha256": prepared.canonical_margin_sha256,
        "reward_feature_difference_sha256": (prepared.reward_feature_difference_sha256),
        "num_train_prompts": training.num_prompts,
        "num_candidates": training.num_candidates,
        "reward_dimension": training.reward_dimension,
        "raw_node_rewards_retained": False,
    }


def _exact_margin_evidence(
    prepared: _PreparedControl,
    *,
    convergence: object,
) -> dict[str, object]:
    cold = prepared.audit()
    head = _head_audit(
        prepared,
        method="prorm_plus",
        objective_name="exact_margin_prorm_plus",
        convergence=convergence,
        cold=cold,
    )
    final_inner = _closed_mapping(
        _closed_mapping(convergence, name="convergence").get("final_gate"),
        name="final gate",
    )
    final_measurement = _closed_mapping(
        final_inner.get("measurement"),
        name="final measurement",
    )
    trained_direction = policy_direction_from_head(
        prepared.full_training,
        prepared.trainer.model.weight.detach(),
        relative_damping=prepared.settings.relative_damping,
        beta=1.0,
        pcg_dtype=prepared.settings.pcg_dtype,
        pcg_max_iterations=prepared.settings.pcg_max_iterations,
        pcg_tolerance=prepared.settings.pcg_tolerance,
        pcg_absolute_tolerance=prepared.settings.pcg_absolute_tolerance,
        pcg_residual_recompute_interval=(prepared.settings.pcg_residual_recompute_interval),
        require_pcg_convergence=True,
    )
    if prepared.direct_control is None:
        raise RuntimeError("exact-margin family lacks the direct-oracle control")
    if prepared.exact_arm is None or not torch.allclose(
        prepared.direct_control.canonical_margins,
        prepared.exact_arm.training.h.to(dtype=prepared.direct_control.canonical_margins.dtype),
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise RuntimeError("direct-oracle control uses another canonical margin")
    direct = _compact_direct_oracle_identity(prepared.direct_control)
    native = _closed_mapping(
        direct.pop("native_oracle_direction"),
        name="native oracle direction",
    )
    for redundant_dimension in ("num_prompts", "num_candidates", "policy_dimension"):
        direct.pop(redundant_dimension)
    # This field binds the conceptual canonical target consumed by the FP32
    # reward-head optimizer.  Equality to the FP64 direct-control copy was
    # checked above before replacing the representation-specific tensor hash.
    direct["canonical_margin_sha256"] = prepared.canonical_margin_sha256
    native_pcg = _closed_mapping(native.get("pcg"), name="native oracle PCG")
    relative_error = direct.get("complete_pair_identity_relative_error")
    if relative_error is None:
        if float(direct["complete_pair_identity_absolute_error"]) != 0.0:
            raise RuntimeError("direct identity has no relative denominator")
        relative_error = 0.0
    direct_identity = {
        **direct,
        "complete_pair_identity_relative_error": relative_error,
        "native_oracle_direction_sha256": native.get("direction_sha256"),
        "native_oracle_direction_pcg": _pcg_from_mapping(
            native_pcg,
            name="native oracle PCG",
        ),
    }
    target = {
        "schema_version": "phase2-r3-exact-margin-target/v1",
        **_common_target(prepared),
        "target": "transformed_oracle_reward_difference",
        "sampled_label_stream_accessed": False,
    }
    selected_pcg = _pcg_from_mapping(
        final_measurement.get("inner_solver"),
        name="selected exact-margin PCG",
    )
    cold_pcg = _pcg_from_mapping(
        cold.inner_solver,
        name="cold exact-margin PCG",
    )
    trained_pcg = _pcg_from_direction(trained_direction)
    return {
        "schema_version": "phase2-r3-exact-margin-prorm-plus-evidence/v1",
        "target_audit": target,
        "head_audit": head,
        "pcg_audits": {
            "selected_head_final_inner": selected_pcg,
            "cold_saved_head_audit": cold_pcg,
            "trained_direction": trained_pcg,
        },
        "direct_identity": direct_identity,
        "gates": {
            "exact_margin_objective_decrease": head["objective_decrease_passed"],
            "exact_margin_first_order_convergence": head["first_order_gate"]["passed"],
            "direct_oracle_moment_identity": (
                float(direct_identity["complete_pair_identity_absolute_error"]) <= 1.0e-10
                and float(direct_identity["complete_pair_identity_relative_error"]) <= 1.0e-10
            ),
            "all_required_pcg_solves_converged": all(
                bool(item["converged"])
                for item in (
                    selected_pcg,
                    cold_pcg,
                    trained_pcg,
                    direct_identity["native_oracle_direction_pcg"],
                )
            ),
        },
    }


def _exact_soft_evidence(
    prepared: _PreparedControl,
    *,
    convergence: object,
) -> dict[str, object]:
    cold = prepared.audit()
    head = _head_audit(
        prepared,
        method="bt_mle",
        objective_name="exact_soft_label_bt_cross_entropy",
        convergence=convergence,
        cold=cold,
    )
    batch = prepared.trainer.batch
    target = {
        "schema_version": "phase2-r3-exact-soft-label-bt-target/v1",
        **_common_target(prepared),
        "target": "sigmoid_of_train_transformed_oracle_margin",
        "target_probability_sha256": _tensor_sha256(batch.target_probabilities),
        "noise_free": True,
        "bernoulli_sampling_used": False,
        "sampled_label_stream_accessed": False,
        "raw_target_probabilities_retained": False,
        "raw_oracle_margins_retained": False,
    }
    return {
        "schema_version": "phase2-r3-exact-soft-label-bt-evidence/v1",
        "target_audit": target,
        "head_audit": head,
        "gates": {
            "exact_soft_label_objective_decrease": head["objective_decrease_passed"],
            "exact_soft_label_first_order_convergence": head["first_order_gate"]["passed"],
            "saved_head_objective_binding": head["objective_binding_passed"],
        },
    }


def _low_dimensional_evidence(
    prepared: _PreparedControl,
    controls: R3ControlsConfigBundle,
    *,
    convergence: object,
) -> dict[str, object]:
    projection = prepared.projection_control
    geometry = prepared.dense_geometry
    if projection is None or geometry is None or prepared.family_local_label_stream_sha256 is None:
        raise RuntimeError("low-dimensional family preparation is incomplete")
    final = _evaluate_dense_prorm(
        prepared.trainer.model,
        prepared.trainer.batch,
        geometry,
        prepared.settings,
    )
    cold = prepared.audit()
    head = _head_audit(
        prepared,
        method="prorm_plus",
        objective_name="low_dimensional_prorm_plus",
        convergence=convergence,
        cold=cold,
    )
    thresholds = _tolerances(controls)
    projection_record = projection.to_dict()
    orthonormality_error = float(projection_record["orthonormality_max_absolute_error"])
    selected_direction = final.direction
    scattered = projection.scatter_direction_to_full(selected_direction)
    reference = torch.einsum(
        "ij,j->i",
        projection.projection.to(dtype=selected_direction.dtype),
        selected_direction,
    ).detach()
    scatter_error = scattered - reference
    low_scores = projection.training.policy_scores.reshape(
        -1,
        projection.selected_dimension,
    ).to(dtype=selected_direction.dtype)
    full_scores = prepared.full_training.policy_scores.reshape(
        -1,
        prepared.full_training.policy_dimension,
    ).to(dtype=selected_direction.dtype)
    low_projected = low_scores @ selected_direction
    full_projected = full_scores @ scattered
    score_error = low_projected - full_projected
    scatter_max = float(torch.max(torch.abs(scatter_error)).item())
    score_max = float(torch.max(torch.abs(score_error)).item())
    projection_evidence = {
        "schema_version": "seeded-orthonormal-tangent/v1",
        "algorithm": projection.algorithm,
        "namespace": projection.namespace,
        "source_layout_id": projection.source_layout_id,
        "declared_seed": projection.declared_seed,
        "source_dimension": projection.source_dimension,
        "selected_dimension": projection.selected_dimension,
        "num_fisher_nodes": projection.num_fisher_nodes,
        "projection_sha256": projection.projection_sha256,
        "projection_dtype": str(projection.projection.dtype),
        "orthonormality_max_absolute_error": orthonormality_error,
        "orthonormality_absolute_tolerance": thresholds[
            "low_dimensional_orthonormality_max_absolute_error"
        ],
        "orthonormality_passed": (
            orthonormality_error <= thresholds["low_dimensional_orthonormality_max_absolute_error"]
        ),
    }
    geometry_evidence = {
        "schema_version": "phase2-r3-low-dimensional-pseudoinverse-geometry/v1",
        "regularization": "moore_penrose_pseudoinverse",
        "ridge_enabled": False,
        "solver": "torch.linalg.eigh_truncated_moore_penrose",
        "solver_dtype": prepared.settings.pcg_dtype,
        "selected_dimension": projection.selected_dimension,
        "numerical_rank": geometry.rank,
        "relative_eigenvalue_tolerance": (
            prepared.settings.low_dimensional_relative_eigenvalue_tolerance
        ),
        "fisher_sha256": geometry.fisher_sha256,
        "pseudoinverse_sha256": geometry.pseudoinverse_sha256,
        "pseudoinverse_solve_relative_residual": final.solve_relative_residual,
        "pseudoinverse_relative_residual_tolerance": thresholds[
            "low_dimensional_pseudoinverse_relative_residual"
        ],
        "exact_rank_passed": geometry.rank == projection.selected_dimension,
        "pseudoinverse_residual_passed": (
            final.solve_relative_residual
            <= thresholds["low_dimensional_pseudoinverse_relative_residual"]
        ),
    }
    scatter = {
        "schema_version": "phase2-r3-low-dimensional-scatter-identity/v1",
        "formula": "u_full = P @ u_low",
        "selected_direction_sha256": _tensor_sha256(selected_direction),
        "scattered_full_direction_sha256": _tensor_sha256(scattered),
        "reference_scattered_full_direction_sha256": _tensor_sha256(reference),
        "max_absolute_error": scatter_max,
        "l2_error": float(torch.linalg.vector_norm(scatter_error).item()),
        "absolute_tolerance": thresholds["low_dimensional_scatter_max_absolute_error"],
        "passed": (scatter_max <= thresholds["low_dimensional_scatter_max_absolute_error"]),
    }
    score = {
        "schema_version": "phase2-r3-low-dimensional-score-identity/v1",
        "formula": "(S_full @ P) @ u_low == S_full @ (P @ u_low)",
        "selected_direction_sha256": _tensor_sha256(selected_direction),
        "scattered_full_direction_sha256": _tensor_sha256(scattered),
        "low_projected_score_sha256": _tensor_sha256(low_projected),
        "full_projected_score_sha256": _tensor_sha256(full_projected),
        "max_absolute_error": score_max,
        "l2_error": float(torch.linalg.vector_norm(score_error).item()),
        "absolute_tolerance": thresholds["low_dimensional_score_identity_max_absolute_error"],
        "passed": (score_max <= thresholds["low_dimensional_score_identity_max_absolute_error"]),
    }
    target = {
        "schema_version": "phase2-r3-low-dimensional-r4-target/v1",
        **_common_target(prepared),
        "target": "family_local_r4_mean_h_regenerated_from_train_oracle",
        "family_local_label_stream_sha256": (prepared.family_local_label_stream_sha256),
        "annotation_scheme": "geometric_randomized_truncation",
        "annotation_gamma": prepared.settings.annotation_gamma,
        "independent_replicates_per_edge": prepared.settings.num_label_replicates,
        "replicate_reduction": "arithmetic_mean",
        "label_rng_namespace": prepared.settings.label_rng_namespace,
        "primary_label_stream_accessed": False,
        "raw_labels_retained": False,
    }
    return {
        "schema_version": "phase2-r3-low-dimensional-prorm-plus-evidence/v1",
        "target_audit": target,
        "projection": projection_evidence,
        "geometry": geometry_evidence,
        "head_audit": head,
        "scatter_identity": scatter,
        "score_identity": score,
        "gates": {
            "low_dimensional_objective_decrease": head["objective_decrease_passed"],
            "low_dimensional_first_order_convergence": head["first_order_gate"]["passed"],
            "low_dimensional_exact_rank": geometry_evidence["exact_rank_passed"],
            "low_dimensional_orthonormality": projection_evidence["orthonormality_passed"],
            "low_dimensional_pseudoinverse_residual": geometry_evidence[
                "pseudoinverse_residual_passed"
            ],
            "low_dimensional_scatter_identity": scatter["passed"],
            "low_dimensional_score_identity": score["passed"],
        },
    }


def run_r3_control_family(
    input_capability: R3ControlTrainInputCapability,
    family: str,
    *,
    controls_config: R3ControlsConfigBundle,
) -> dict[str, object]:
    """Run one independent formal family and return compact head-free evidence."""

    capability = _require_capability(input_capability)
    controls = _require_controls(controls_config)
    method = _family(family)
    prepared = _prepare_control(capability, controls, method)
    objective_name = {
        "exact_margin_prorm_plus": "exact_margin_prorm_plus",
        "exact_soft_label_bt": "exact_soft_label_bt_cross_entropy",
        "low_dimensional_prorm_plus": "low_dimensional_prorm_plus",
    }[method]
    convergence = _run_trainer_to_first_order_convergence(
        prepared.trainer,
        audit=prepared.audit,
        spec=prepared.settings.convergence,
        fixed_snapshot_steps=prepared.settings.outer_steps,
        objective_name=objective_name,
        rank_diagnostic=None,
        execution_role="phase2_recovery_r3_mechanism_control",
    )
    if method == "exact_margin_prorm_plus":
        evidence = _exact_margin_evidence(
            prepared,
            convergence=convergence.evidence,
        )
        adapter = adapt_exact_margin_prorm_plus_result
    elif method == "exact_soft_label_bt":
        evidence = _exact_soft_evidence(
            prepared,
            convergence=convergence.evidence,
        )
        adapter = adapt_exact_soft_label_bt_result
    else:
        evidence = _low_dimensional_evidence(
            prepared,
            controls,
            convergence=convergence.evidence,
        )
        adapter = adapt_low_dimensional_prorm_plus_result
    result = adapter(
        seed=capability.seed,
        config=controls,
        input_training_sha256=capability.input_training_sha256,
        train_oracle_rewards_sha256=capability.train_oracle_rewards_sha256,
        input_dimensions=capability.input_dimensions,
        family_evidence=evidence,
    )
    return validate_r3_control_family_result(result, controls)


class _TimedTrainer:
    """Transparent timing proxy used only by the non-reusable profile."""

    def __init__(self, trainer: Any) -> None:
        self._trainer = trainer
        self.step_wall_seconds: list[float] = []

    @property
    def model(self) -> object:
        return self._trainer.model

    @property
    def optimizer(self) -> object:
        return self._trainer.optimizer

    @optimizer.setter
    def optimizer(self, value: object) -> None:
        self._trainer.optimizer = value

    @property
    def completed_steps(self) -> int:
        return int(self._trainer.completed_steps)

    @property
    def history(self) -> object:
        return self._trainer.history

    def step(self) -> object:
        device = self._trainer.model.weight.device
        _synchronize_device(device)
        started = time.perf_counter()
        result = self._trainer.step()
        _synchronize_device(device)
        self.step_wall_seconds.append(time.perf_counter() - started)
        return result

    def state_dict(self) -> Mapping[str, object]:
        return self._trainer.state_dict()

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._trainer.load_state_dict(state)


def _checkpoint_roundtrip(
    value: Mapping[str, object],
    *,
    directory: Path,
) -> float:
    started = time.perf_counter()
    buffer = io.BytesIO()
    torch.save(dict(value), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    if not isinstance(restored, Mapping):
        raise RuntimeError("Gate-C profile checkpoint roundtrip changed the state type")
    # Exercise the target filesystem without retaining reusable state.
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix="r3-gate-c-profile-",
        suffix=".pt",
        dir=directory,
        delete=True,
    ) as stream:
        stream.write(buffer.getbuffer())
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0)
        disk_restored = torch.load(stream, map_location="cpu", weights_only=True)
        if not isinstance(disk_restored, Mapping):
            raise RuntimeError("Gate-C profile disk checkpoint is malformed")
    elapsed = time.perf_counter() - started
    if elapsed <= 0.0:
        raise RuntimeError("Gate-C profile checkpoint timer did not advance")
    return elapsed


def profile_r3_control_family(
    input_capability: R3ControlTrainInputCapability,
    family: str,
    *,
    controls_config: R3ControlsConfigBundle,
    checkpoint_directory: str | os.PathLike[str],
) -> dict[str, object]:
    """Run exactly 100 disposable updates under the production controller.

    The controller stops at a planned segment boundary.  Its checkpoint is
    round-tripped only to measure I/O and is deleted before this function
    returns; no head, optimizer, checkpoint, raw label, or raw reward is
    included in the observation.
    """

    capability = _require_capability(input_capability)
    controls = _require_controls(controls_config)
    if capability.seed != controls.seeds[0]:
        raise ValueError("Gate-C profile must use the first frozen Gate-C seed")
    method = _family(family)
    _synchronize_device(capability.training.policy_scores.device)
    setup_started = time.perf_counter()
    prepared = _prepare_control(capability, controls, method)
    _synchronize_device(capability.training.policy_scores.device)
    family_setup_wall_seconds = time.perf_counter() - setup_started
    if family_setup_wall_seconds <= 0.0:
        raise RuntimeError("Gate-C family setup timer did not advance")
    timed = _TimedTrainer(prepared.trainer)
    audit_wall_seconds: list[float] = []

    def timed_audit() -> _FirstOrderMeasurement:
        device = prepared.trainer.model.weight.device
        _synchronize_device(device)
        started = time.perf_counter()
        result = prepared.audit()
        _synchronize_device(device)
        elapsed = time.perf_counter() - started
        if elapsed <= 0.0:
            raise RuntimeError("Gate-C profile audit timer did not advance")
        audit_wall_seconds.append(elapsed)
        return result

    checkpoint_times: list[float] = []
    checkpoint_root = Path(checkpoint_directory).resolve()

    def checkpoint_hook(state: Mapping[str, object], reason: str) -> None:
        if reason not in {"interval", "stage_boundary"}:
            raise RuntimeError("Gate-C profile checkpoint has an unexpected reason")
        checkpoint_times.append(_checkpoint_roundtrip(state, directory=checkpoint_root))

    objective_name = {
        "exact_margin_prorm_plus": "exact_margin_prorm_plus",
        "exact_soft_label_bt": "exact_soft_label_bt_cross_entropy",
        "low_dimensional_prorm_plus": "low_dimensional_prorm_plus",
    }[method]
    try:
        _run_trainer_to_first_order_convergence(
            timed,
            audit=timed_audit,
            spec=prepared.settings.convergence,
            fixed_snapshot_steps=prepared.settings.outer_steps,
            objective_name=objective_name,
            rank_diagnostic=None,
            checkpoint_hook=checkpoint_hook,
            checkpoint_interval_steps=R3_GATE_C_PROFILE_UPDATES,
            execution_step_cap=R3_GATE_C_PROFILE_UPDATES,
            execution_role="phase2_recovery_r3_profile_nonreusable",
        )
    except PlannedSegmentBoundary:
        pass
    else:
        raise RuntimeError("Gate-C profile was incorrectly promoted to a formal result")
    if timed.completed_steps != R3_GATE_C_PROFILE_UPDATES:
        raise RuntimeError("Gate-C profile did not execute exactly 100 optimizer updates")
    if len(timed.step_wall_seconds) != R3_GATE_C_PROFILE_UPDATES:
        raise RuntimeError("Gate-C profile timing lost an optimizer update")
    if len(audit_wall_seconds) != 7:
        raise RuntimeError(
            "Gate-C profile did not execute one initial, five scheduled, "
            "and one checkpoint-boundary audit"
        )
    # Two disposable cold audits represent the restored-final and evidence
    # binding audits used by a completed formal family.
    timed_audit()
    timed_audit()
    if len(audit_wall_seconds) != 9:
        raise RuntimeError("Gate-C profile one-time audit accounting is incomplete")
    if len(checkpoint_times) != 1:
        raise RuntimeError("Gate-C profile must measure exactly one checkpoint roundtrip")
    body = {
        "schema_version": R3_CONTROL_PROFILE_OBSERVATION_SCHEMA,
        "role": "train_only_nonreusable_100_update_family_observation",
        "family": method,
        "seed": capability.seed,
        "completed_updates": R3_GATE_C_PROFILE_UPDATES,
        "stop_reason": _PROFILE_STOP_REASON,
        "input_training_sha256": capability.input_training_sha256,
        "train_oracle_rewards_sha256": capability.train_oracle_rewards_sha256,
        "family_setup_wall_seconds": family_setup_wall_seconds,
        "one_time_audit_wall_seconds": math.fsum((audit_wall_seconds[0], *audit_wall_seconds[7:])),
        "training_wall_seconds": math.fsum(timed.step_wall_seconds),
        "scheduled_audit_wall_seconds": math.fsum(audit_wall_seconds[1:6]),
        "checkpoint_roundtrip_wall_seconds": math.fsum(
            (audit_wall_seconds[6], checkpoint_times[0])
        ),
        "information_boundary": {
            "train_only": True,
            "primary_label_stream_constructed": False,
            "primary_label_stream_accessed": False,
            "primary_head_accessed": False,
            "heldout_or_validation_accessed": False,
            "policy_optimization_executed": False,
            "result_reusable_for_training": False,
            "head_or_optimizer_state_retained": False,
        },
    }
    return {**body, "observation_sha256": _canonical_sha256(body)}


def validate_r3_control_profile_observation(value: object) -> dict[str, object]:
    """Revalidate a scheduler-independent, non-reusable profile observation."""

    payload = _closed_mapping(value, name="Gate-C profile observation")
    expected_keys = {
        "schema_version",
        "role",
        "family",
        "seed",
        "completed_updates",
        "stop_reason",
        "input_training_sha256",
        "train_oracle_rewards_sha256",
        "family_setup_wall_seconds",
        "one_time_audit_wall_seconds",
        "training_wall_seconds",
        "scheduled_audit_wall_seconds",
        "checkpoint_roundtrip_wall_seconds",
        "information_boundary",
        "observation_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("Gate-C profile observation fields are not closed")
    unsigned = dict(payload)
    observed_sha256 = unsigned.pop("observation_sha256")
    if observed_sha256 != _canonical_sha256(unsigned):
        raise ValueError("Gate-C profile observation self-hash is invalid")
    if (
        payload["schema_version"] != R3_CONTROL_PROFILE_OBSERVATION_SCHEMA
        or payload["role"] != "train_only_nonreusable_100_update_family_observation"
        or payload["family"] not in R3_GATE_C_FAMILIES
        or payload["seed"] != R3_GATE_C_SEEDS[0]
        or payload["completed_updates"] != R3_GATE_C_PROFILE_UPDATES
        or payload["stop_reason"] != _PROFILE_STOP_REASON
    ):
        raise ValueError("Gate-C profile observation exceeds its frozen role")
    for name in (
        "family_setup_wall_seconds",
        "one_time_audit_wall_seconds",
        "training_wall_seconds",
        "scheduled_audit_wall_seconds",
        "checkpoint_roundtrip_wall_seconds",
    ):
        if _finite_nonnegative(payload[name], name=name) <= 0.0:
            raise ValueError(f"{name} must be strictly positive")
    expected_boundary = {
        "train_only": True,
        "primary_label_stream_constructed": False,
        "primary_label_stream_accessed": False,
        "primary_head_accessed": False,
        "heldout_or_validation_accessed": False,
        "policy_optimization_executed": False,
        "result_reusable_for_training": False,
        "head_or_optimizer_state_retained": False,
    }
    if payload["information_boundary"] != expected_boundary:
        raise ValueError("Gate-C profile observation crossed its information boundary")
    for name in ("input_training_sha256", "train_oracle_rewards_sha256"):
        value = payload[name]
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return payload


__all__ = [
    "R3_CONTROL_PROFILE_OBSERVATION_SCHEMA",
    "profile_r3_control_family",
    "run_r3_control_family",
    "validate_r3_control_profile_observation",
]
