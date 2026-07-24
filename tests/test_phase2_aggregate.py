from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import smart_reward.phase2_campaign as phase2_campaign_module
from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.phase2_aggregate import (
    PHASE2_AGGREGATE_SCHEMA,
    build_common_beta_seed_aggregate,
    write_common_beta_seed_aggregate,
)
from smart_reward.phase2_campaign import (
    PHASE2_ATTEMPT_LEDGER_SCHEMA,
    PHASE2_CAMPAIGN_TERMINAL_SCHEMA,
    PHASE2_FAILURE_EVIDENCE_AVAILABILITY_SCHEMA,
    PHASE2_SEED_FAILURE_SCHEMA,
    PHASE2_SEED_FAILURE_SCHEMA_V1,
    PHASE2_SEED_SUCCESS_SCHEMA,
    build_phase2_seed_failure_manifest,
    build_phase2_seed_success_manifest,
    load_phase2_seed_success_spec,
    write_phase2_campaign_terminal,
    write_phase2_seed_failure_manifest,
    write_phase2_seed_success_manifest,
)
from smart_reward.phase2_config import (
    PHASE2_MIN_CONFIRMATORY_SEEDS,
    load_phase2_config,
    phase2_design_identity,
)
from smart_reward.phase2_rollout import PHASE2_ARM_ORDER, Phase2Design

ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_REWARD_DIMENSION = 256


def _fixture_head(first: float, second: float) -> list[float]:
    return [first, second, *([0.0] * (_FIXTURE_REWARD_DIMENSION - 2))]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _token_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _environment(*, gpu: str = "NVIDIA L20") -> dict[str, object]:
    return {
        "formal": True,
        "git_commit": "a" * 40,
        "image_sha256": "b" * 64,
        "hf_inventory_sha256": "c" * 64,
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_models": [gpu],
    }


def _direction(arm_name: str, beta: float) -> dict[str, object]:
    return {
        "direction": {
            "schema_version": "policy-direction/v1",
            "beta": 1.0,
            "pcg": {"converged": True},
        },
        "common_beta_direction": {
            "schema_version": "common-beta-direction/v1",
            "name": arm_name,
            "beta_common": beta,
            "learner_specific_rescaling": False,
        },
    }


def _pcg(*, residual: float = 1.0e-8) -> dict[str, object]:
    return {
        "iterations": 7,
        "residual_norm": residual,
        "relative_residual": residual,
        "converged": True,
        "reason": "tolerance",
    }


def _first_order_convergence(
    *,
    seed: int,
    objective_name: str,
    head_sha256: str,
    initial_objective: float,
    final_objective: float,
    rank_evidence: dict[str, object],
    gradient: float = 1.0e-6,
) -> dict[str, object]:
    selected_step = 140
    return {
        "schema_version": "objective-first-order-convergence/v1",
        "objective": objective_name,
        "converged": True,
        "fail_closed": True,
        "spec": {
            "schema_version": "objective-first-order-convergence-spec/v1",
            "gradient_ratio_tolerance": 1.0e-3,
            "min_steps": 100,
            "max_steps": 5760,
            "check_interval": 20,
            "consecutive_checks": 3,
            "gradient_norm_denominator_floor": 1.0e-30,
            "fail_closed": True,
            "gradient": "full_data_post_update_unclipped",
            "denominator": "exact_zero_initialization_gradient_l2_norm",
            "validation_or_test_selection": False,
        },
        "gradient_ratio_formula": "final_over_initial",
        "initial_zero_head_measurement": {
            "objective": initial_objective,
            "gradient_l2_norm": 1.0,
            "inner_solver": None,
        },
        "checks": [
            {
                "step": step,
                "post_update": True,
                "full_data": True,
                "gradient_clipping_applied": False,
                "threshold_passed": True,
            }
            for step in (100, 120, 140)
        ],
        "selected_primary_step": selected_step,
        "selected_primary_head_sha256": head_sha256,
        "consecutive_threshold_passes_at_selection": 3,
        "final_gate": {
            "step": selected_step,
            "measurement": {
                "objective": final_objective,
                "gradient_l2_norm": gradient,
                "inner_solver": None,
            },
            "gradient_ratio_to_zero_initialization": gradient,
            "threshold_passed": True,
            "fresh_post_restore_audit": True,
        },
        "fixed_step_compute_matched_snapshot": {
            "schema_version": "fixed-step-compute-matched-snapshot/v1",
            "step": 720,
            "head_sha256": _token_sha256(f"{seed}:{objective_name}:fixed-720"),
            "measurement": {
                "objective": final_objective,
                "gradient_l2_norm": gradient,
                "inner_solver": None,
            },
            "gradient_ratio_to_zero_initialization": gradient,
            "history_summary": {"num_steps": 720},
            "role": "compute_matched_and_pilot_diagnostic_only",
            "used_as_primary_selection_rule": False,
            "coincides_with_selected_primary_iterate": False,
        },
        "fixed_step_snapshot_steps": 720,
        "fixed_step_snapshot_is_not_primary_selection": True,
        "solution_identification": {
            "initialization": "exact_zero_head",
            "tie_break": "zero_initialized_adamw_implicit_bias",
            "primary_iterate_selection": (
                "first_scheduled_iterate_completing_the_sustained_first_order_gate"
            ),
            "validation_or_test_checkpoint_selection": False,
            "objective_value_checkpoint_selection": False,
            "minimum_norm_projection_applied": False,
            "minimum_norm_solution_claimed": False,
            "unique_reward_head_solution_claimed": False,
            "optional_objective_rank_diagnostic": {
                "evaluated": True,
                "evidence": rank_evidence,
            },
        },
        "test_or_validation_data_accessed": False,
    }


def _serialized_head(
    *,
    seed: int,
    arm: str,
    method: str,
    weight: list[float],
    initial_objective: float,
    final_objective: float,
    initial_head_sha256: str,
    pcg: bool,
    rank_evidence: dict[str, object],
    objective_name: str | None = None,
) -> dict[str, object]:
    head_sha = _token_sha256(f"{seed}:{arm}:{method}:head")
    return {
        "arm": arm,
        "method": method,
        "head_weight": weight,
        "head_dtype": "torch.float32",
        "initial_head_sha256": initial_head_sha256,
        "head_sha256": head_sha,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "history_summary": {
            "num_steps": 140,
            "history_objective_timing": "pre_update",
        },
        "final_pcg": ({**_pcg(), "cold_start": True} if pcg else None),
        "first_order_convergence": _first_order_convergence(
            seed=seed,
            objective_name=method if objective_name is None else objective_name,
            head_sha256=head_sha,
            initial_objective=initial_objective,
            final_objective=final_objective,
            rank_evidence=rank_evidence,
        ),
    }


def _fresh_inner_audit(
    *,
    objective: float,
    gradient: float = 1.0e-6,
) -> dict[str, object]:
    return {
        "objective": objective,
        "gradient_l2_norm": gradient,
        "gradient": "full_data_unclipped",
        "objective_definition": "damped_fisher_gmm_dual_loss",
        "gradient_definition": "fresh_dual_envelope_gradient",
        "fresh_inner_pcg": {
            **_pcg(),
            "dtype": "float64",
            "warm_start_used": False,
        },
    }


def _identifiability_evidence(
    *,
    seed: int,
    config: dict[str, Any],
) -> dict[str, object]:
    train_prompts = int(config["run"]["split_sizes"]["train"])
    reward_dimension = _FIXTURE_REWARD_DIMENSION
    largest = 2.0
    smallest = 0.5
    tolerance = float(config["reward_model"]["identifiability"]["relative_rank_tolerance"])
    return {
        "schema_version": "reward-head-identifiability/v1",
        "design_matrix": "canonical_edge_reward_feature_differences",
        "split": "train",
        "shape": [train_prompts, reward_dimension],
        "source_dtype": "torch.float32",
        "audit_dtype": "torch.float64",
        "design_matrix_sha256": _token_sha256(f"{seed}:feature-difference-design"),
        "relative_rank_tolerance": tolerance,
        "absolute_singular_value_threshold": tolerance * largest,
        "numerical_rank": reward_dimension,
        "column_dimension": reward_dimension,
        "full_column_rank": True,
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "smallest_retained_singular_value": smallest,
        "retained_condition_number": largest / smallest,
        "role": config["reward_model"]["identifiability"]["role"],
        "require_full_column_rank": config["reward_model"]["identifiability"][
            "require_full_column_rank"
        ],
        "acceptance_gate_passed": True,
        "bt_unique_finite_optimum_sufficient_condition_only": True,
        "prorm_moment_map_full_rank_proved": False,
        "algorithmic_tie_break": "zero_initialized_adamw_implicit_bias",
        "minimum_norm_solution_claimed": False,
        "test_or_validation_data_accessed": False,
    }


def _moment_map_identifiability_evidence(
    *,
    seed: int,
    config: dict[str, Any],
) -> dict[str, object]:
    train_prompts = int(config["run"]["split_sizes"]["train"])
    policy_dimension = 512
    reward_dimension = _FIXTURE_REWARD_DIMENSION
    spectrum = [1.5 - index / (2.0 * (reward_dimension - 1)) for index in range(reward_dimension)]
    tolerance = float(config["reward_model"]["identifiability"]["relative_rank_tolerance"])
    spectrum_payload = {
        "dtype": "torch.float64",
        "shape": [len(spectrum)],
        "values_descending": spectrum,
    }
    block_rows = policy_dimension
    return {
        "schema_version": "prorm-moment-map-identifiability/v1",
        "design_matrix": "canonical_train_edge_moment_jacobian",
        "formula": "J_m = Z^T D / (2 n_edges)",
        "split": "train",
        "orientation": "candidate_0_minus_candidate_1",
        "shape": [policy_dimension, reward_dimension],
        "num_edges": train_prompts,
        "policy_dimension": policy_dimension,
        "column_dimension": reward_dimension,
        "source_policy_score_dtype": "torch.float32",
        "source_reward_feature_dtype": "torch.float32",
        "audit_dtype": "torch.float64",
        "edge_policy_score_difference_sha256": _token_sha256(f"{seed}:edge-policy-score"),
        "edge_reward_feature_difference_sha256": _token_sha256(f"{seed}:edge-reward-feature"),
        "moment_map_sha256": _token_sha256(f"{seed}:moment-map"),
        "computation": {
            "algorithm": "deterministic_blocked_fp64_tsqr",
            "row_block_size": block_rows,
            "num_row_blocks": 1,
            "full_moment_map_materialized": False,
            "randomized_rank_approximation_used": False,
        },
        "relative_rank_tolerance": tolerance,
        "absolute_singular_value_threshold": tolerance * spectrum[0],
        "singular_values_descending": spectrum,
        "singular_values_sha256": _canonical_sha256(spectrum_payload),
        "singular_spectrum_summary": {
            "count": len(spectrum),
            "largest": spectrum[0],
            "smallest": spectrum[-1],
            "smallest_retained": spectrum[-1],
        },
        "numerical_rank": reward_dimension,
        "full_column_rank": True,
        "retained_condition_number": spectrum[0] / spectrum[-1],
        "ridge_geometry": {
            "matrix": "H = F_hat + lambda I",
            "positive_definite": True,
            "reason": "configured_relative_damping_is_strictly_positive",
            "head_hessian": "J_m^T H^{-1} J_m / beta",
            "rank_identity": "rank(J_m^T H^{-1} J_m) = rank(J_m)",
        },
        "unique_ridge_prorm_quadratic_head_iff_full_column_rank": True,
        "observed_unique_ridge_prorm_quadratic_head": True,
        "population_identifiability_theorem_claimed": False,
        "role": config["reward_model"]["identifiability"]["role"],
        "require_full_column_rank": config["reward_model"]["identifiability"][
            "require_full_column_rank"
        ],
        "acceptance_gate_passed": True,
        "algorithmic_tie_break": "zero_initialized_adamw_implicit_bias",
        "minimum_norm_solution_claimed": False,
        "test_or_validation_data_accessed": False,
    }


def _projected_moment_map_identifiability_evidence(
    *,
    seed: int,
    config: dict[str, Any],
    policy_dimension: int,
    projection_sha256: str,
    fisher_sha256: str,
    pseudoinverse_sha256: str,
) -> dict[str, object]:
    evidence = _moment_map_identifiability_evidence(seed=seed, config=config)
    evidence["schema_version"] = "projected-prorm-moment-map-identifiability/v1"
    evidence["design_matrix"] = "canonical_train_edge_projected_moment_jacobian"
    evidence["shape"] = [policy_dimension, _FIXTURE_REWARD_DIMENSION]
    evidence["policy_dimension"] = policy_dimension
    evidence["edge_policy_score_difference_sha256"] = _token_sha256(
        f"{seed}:projected-edge-policy-score"
    )
    evidence["moment_map_sha256"] = _token_sha256(f"{seed}:projected-moment-map")
    evidence["computation"]["row_block_size"] = policy_dimension
    evidence["computation"]["num_row_blocks"] = 1
    evidence["projection_sha256"] = projection_sha256
    evidence.pop("ridge_geometry")
    evidence.pop("unique_ridge_prorm_quadratic_head_iff_full_column_rank")
    evidence.pop("observed_unique_ridge_prorm_quadratic_head")
    evidence["unique_projected_prorm_quadratic_head_iff_full_column_rank"] = True
    evidence["observed_unique_projected_prorm_quadratic_head"] = True
    evidence["require_full_column_rank"] = False
    evidence["row_dimension"] = policy_dimension
    evidence["full_row_rank"] = True
    evidence["require_full_row_rank"] = True
    evidence["acceptance_gate_definition"] = "full_row_rank_for_projected_policy_moment_coverage"
    evidence["projected_geometry"] = {
        "matrix": "H_low = F_hat_low",
        "positive_definite": True,
        "reason": "projected_fisher_numerical_rank_equals_selected_dimension",
        "regularization": "moore_penrose_pseudoinverse_on_full_rank_H_low",
        "fisher_sha256": fisher_sha256,
        "pseudoinverse_sha256": pseudoinverse_sha256,
        "relative_eigenvalue_tolerance": config["positive_controls"]["low_dimensional_tangent"][
            "relative_eigenvalue_tolerance"
        ],
        "head_hessian": "J_m^T H_low^{-1} J_m / beta",
        "rank_identity": "rank(J_m^T H_low^{-1} J_m) = rank(J_m)",
    }
    return evidence


def _head_training_audit(
    *,
    seed: int,
    config: dict[str, Any],
    design_sha: str,
    train_oracle_reward_sha: str,
    head_weights: dict[str, list[float]],
) -> dict[str, object]:
    train_prompts = int(config["run"]["split_sizes"]["train"])
    candidates = int(config["data"]["num_candidates"])
    low_dimension = int(config["positive_controls"]["low_dimensional_tangent"]["dimension"])
    identifiability = _identifiability_evidence(seed=seed, config=config)
    moment_map = _moment_map_identifiability_evidence(seed=seed, config=config)
    projection_sha = _token_sha256(f"{seed}:projection")
    low_fisher_sha = _token_sha256(f"{seed}:low-fisher")
    low_pseudoinverse_sha = _token_sha256(f"{seed}:low-pinv")
    projected_moment_map = _projected_moment_map_identifiability_evidence(
        seed=seed,
        config=config,
        policy_dimension=low_dimension,
        projection_sha256=projection_sha,
        fisher_sha256=low_fisher_sha,
        pseudoinverse_sha256=low_pseudoinverse_sha,
    )
    initial_head_sha = _token_sha256(f"{seed}:zero-head")
    bt = _serialized_head(
        seed=seed,
        arm="r4_independent_gamma_0.9",
        method=BT_MLE,
        weight=head_weights[BT_MLE],
        initial_objective=0.7,
        final_objective=0.5,
        initial_head_sha256=initial_head_sha,
        pcg=False,
        rank_evidence=identifiability,
    )
    prorm = _serialized_head(
        seed=seed,
        arm="r4_independent_gamma_0.9",
        method=PRORM_PLUS,
        weight=head_weights[PRORM_PLUS],
        initial_objective=0.8,
        final_objective=0.4,
        initial_head_sha256=initial_head_sha,
        pcg=True,
        rank_evidence=moment_map,
    )
    exact = _serialized_head(
        seed=seed,
        arm="exact_margin_positive_control",
        method=PRORM_PLUS,
        weight=_fixture_head(0.2, 0.3),
        initial_objective=0.9,
        final_objective=0.2,
        initial_head_sha256=initial_head_sha,
        pcg=True,
        rank_evidence=moment_map,
        objective_name="exact_margin_prorm_plus",
    )
    exact_soft_bt = _serialized_head(
        seed=seed,
        arm="exact_soft_label_bt_secondary_diagnostic",
        method=BT_MLE,
        weight=_fixture_head(0.4, 0.2),
        initial_objective=0.9,
        final_objective=0.25,
        initial_head_sha256=initial_head_sha,
        pcg=False,
        rank_evidence=identifiability,
        objective_name="exact_soft_label_bt_cross_entropy",
    )
    low = _serialized_head(
        seed=seed,
        arm="low_dimensional_tangent_positive_control",
        method=PRORM_PLUS,
        weight=_fixture_head(0.1, 0.4),
        initial_objective=0.8,
        final_objective=0.3,
        initial_head_sha256=initial_head_sha,
        pcg=False,
        rank_evidence=projected_moment_map,
        objective_name="low_dimensional_prorm_plus",
    )
    label_payload: dict[str, object] = {
        "namespace": "prorm-common-beta-r4-labels-v1",
        "base_seed": seed,
        "derived_seed": seed + 123,
        "derivation_sha256": _token_sha256(f"{seed}:derivation"),
        "initial_state_sha256": _token_sha256(f"{seed}:initial-generator"),
        "final_state_sha256": _token_sha256(f"{seed}:final-generator"),
        "probability_sha256": _token_sha256(f"{seed}:probability"),
        "replicate_count_sha256": _token_sha256(f"{seed}:counts"),
        "replicate_win_sha256": _token_sha256(f"{seed}:wins"),
        "replicate_h_sha256": _token_sha256(f"{seed}:replicate-h"),
        "mean_h_sha256": _token_sha256(f"{seed}:mean-h"),
        "realized_total_annotations": train_prompts * 4,
    }
    label_stream_sha = _canonical_sha256(label_payload)
    label_stream = {
        "namespace": label_payload["namespace"],
        "base_seed": seed,
        "derived_seed": label_payload["derived_seed"],
        "derivation_sha256": label_payload["derivation_sha256"],
        "generator_device": "cpu",
        "initial_state_sha256": label_payload["initial_state_sha256"],
        "final_state_sha256": label_payload["final_state_sha256"],
        "oracle_reward_sha256": train_oracle_reward_sha,
        "canonical_probability_sha256": label_payload["probability_sha256"],
        "replicate_count_sha256": label_payload["replicate_count_sha256"],
        "replicate_win_sha256": label_payload["replicate_win_sha256"],
        "replicate_h_sha256": label_payload["replicate_h_sha256"],
        "mean_h_sha256": label_payload["mean_h_sha256"],
        "label_stream_sha256": label_stream_sha,
        "realized_total_annotations": label_payload["realized_total_annotations"],
        "realized_annotations_per_edge": 4.0,
        "expected_annotations_per_edge": 40.0,
        "num_edges": train_prompts,
        "num_replicates": 4,
        "gamma": 0.9,
        "bt_target": "pooled_raw_wins_and_totals",
        "prorm_target": "mean_of_per_replicate_h",
        "raw_labels_retained": False,
        "raw_node_rewards_retained": False,
    }
    direct_direction_sha = _token_sha256(f"{seed}:direct-direction")
    settings_sha = _token_sha256(f"{seed}:settings")
    input_training_sha = _token_sha256(f"{seed}:input-training")
    training_instance = {
        "schema_version": "phase2-training-instance/v1",
        "phase2_config_hash": design_sha,
        "settings_sha256": settings_sha,
        "input_training_sha256": input_training_sha,
        "oracle_reward_sha256": train_oracle_reward_sha,
        "seed": seed,
        "label_stream_sha256": label_stream_sha,
        "reward_head_identifiability_sha256": _canonical_sha256(identifiability),
        "prorm_moment_map_identifiability_sha256": _canonical_sha256(moment_map),
        "bt_head_sha256": bt["head_sha256"],
        "prorm_plus_head_sha256": prorm["head_sha256"],
        "low_dimensional_head_sha256": low["head_sha256"],
        "low_dimensional_projection_sha256": projection_sha,
        "low_dimensional_moment_map_identifiability_sha256": _canonical_sha256(
            projected_moment_map
        ),
        "exact_margin_head_sha256": exact["head_sha256"],
        "exact_soft_label_bt_head_sha256": exact_soft_bt["head_sha256"],
        "direct_oracle_direction_sha256": direct_direction_sha,
    }
    return {
        "schema_version": "phase2-fresh-head-training/v2",
        "training_design_sha256": design_sha,
        "training_settings_sha256": settings_sha,
        "training_instance_sha256": _canonical_sha256(training_instance),
        "input_training_sha256": input_training_sha,
        "training_arm": "r4_independent_gamma_0.9",
        "absolute_damping": 0.001,
        "label_stream": label_stream,
        "primary_heads": {BT_MLE: bt, PRORM_PLUS: prorm},
        "primary_optimization_audit": {
            "geometry": {
                "split": "train",
                "num_edges": train_prompts,
                "reward_head_dimension": _FIXTURE_REWARD_DIMENSION,
                "policy_tangent_dimension": 512,
                "absolute_damping": 0.001,
            },
            "learners": {
                BT_MLE: {
                    "objective": 0.5,
                    "gradient_l2_norm": 1.0e-6,
                    "gradient": "full_data_unclipped",
                    "objective_definition": "exact_repeated_label_bt_nll",
                    "label_weighting": "each_annotation",
                },
                PRORM_PLUS: _fresh_inner_audit(objective=0.4),
            },
            "optimizer_constructed": False,
            "optimizer_step_called": False,
            "saved_heads_mutated": False,
            "reward_head_identifiability": identifiability,
            "prorm_moment_map_identifiability": moment_map,
        },
        "low_dimensional_control": {
            "schema_version": "low-dimensional-tangent-training-control/v1",
            "interpretation": "positive_control_only;ineligible_for_primary_claim",
            "enabled": True,
            "eligible_for_primary_claim": False,
            "training_arm": "r4_independent_gamma_0.9",
            "label_stream_sha256": label_stream_sha,
            "target": "same_r4_mean_h_as_primary_prorm_plus",
            "bt_head": {
                "head_sha256": bt["head_sha256"],
                "retrained": False,
                "reason": "bt_objective_is_independent_of_policy_tangent_geometry",
            },
            "projection": {
                "schema_version": "seeded-orthonormal-tangent/v1",
                "source_layout_id": "training-policy-score-flatten-order/v1",
                "source_dimension": 512,
                "selected_dimension": low_dimension,
                "num_fisher_nodes": train_prompts * candidates,
                "namespace": config["positive_controls"]["low_dimensional_tangent"][
                    "seed_namespace"
                ],
                "declared_seed": seed,
                "effective_seed": seed + 456,
                "algorithm": "gaussian_qr_sign_canonical_v1",
                "projection_sha256": projection_sha,
                "projection_dtype": "torch.float64",
                "score_construction": "S_low = cast_fp32(cast_fp64(S_full) @ P_fp64)",
                "deployment_scatter": "u_full = P @ u_low",
                "orthonormal_columns": True,
                "orthonormality_max_absolute_error": 1.0e-12,
                "strictly_below_fisher_node_count": True,
            },
            "projected_prorm_moment_map_identifiability": projected_moment_map,
            "geometry": {
                "regularization": "moore_penrose_pseudoinverse",
                "ridge_enabled": False,
                "ridge_coefficient": 0.0,
                "solver": "torch.linalg.eigh_truncated_moore_penrose",
                "solver_dtype": "float64",
                "selected_dimension": low_dimension,
                "numerical_rank": low_dimension,
                "relative_eigenvalue_tolerance": 1.0e-10,
                "smallest_retained_eigenvalue": 0.1,
                "largest_retained_eigenvalue": 2.0,
                "fisher_sha256": low_fisher_sha,
                "pseudoinverse_sha256": low_pseudoinverse_sha,
                "pcg_used": False,
            },
            "head": low,
            "final_full_data_audit": {
                "optimizer_constructed": False,
                "optimizer_step_called": False,
                "saved_head_mutated": False,
                "objective": 0.3,
                "gradient": "full_data_unclipped",
                "gradient_l2_norm": 1.0e-6,
                "pseudoinverse_solve_relative_residual": 1.0e-9,
                "selected_direction_sha256": _token_sha256(f"{seed}:low-direction"),
            },
            "deployment_score_identity": {
                "formula": "(S_full @ P) @ u_low == S_full @ (P @ u_low)",
                "selected_direction_sha256": _token_sha256(f"{seed}:low-direction"),
                "scattered_full_direction_sha256": _token_sha256(f"{seed}:scattered-direction"),
                "low_projected_score_sha256": _token_sha256(f"{seed}:low-score"),
                "full_projected_score_sha256": _token_sha256(f"{seed}:full-score"),
                "max_absolute_error": 1.0e-8,
                "l2_error": 1.0e-8,
                "absolute_tolerance": 1.0e-5,
                "passed": True,
            },
            "fresh_zero_initialized": True,
            "raw_labels_retained": False,
            "raw_node_rewards_retained": False,
        },
        "exact_margin_control": {
            "head": exact,
            "target_audit": {
                "schema_version": "exact-margin-audit/v1",
                "source_node_rewards_sha256": train_oracle_reward_sha,
                "exact_margin_sha256": _token_sha256(f"{seed}:exact-margin"),
                "source_shape": [train_prompts, candidates],
                "orientation": "candidate_0_minus_candidate_1",
                "raw_node_rewards_retained": False,
                "bt_counts_source": "input_training_passthrough",
                "purpose": "zero_label_noise_reward_head_training_control",
                "reward_head_fit_required": True,
                "oracle_direction_identity_expected": False,
            },
            "optimization_audit": {
                "learner": _fresh_inner_audit(objective=0.2),
                "bt_audit_discarded": True,
                "optimizer_constructed": False,
                "optimizer_step_called": False,
            },
            "reward_class_and_optimizer_gap": {
                "interpretation": "restricted_reward_class_and_finite_optimizer_gap",
                "trained_direction_pcg": _pcg(),
                "algebraic_identity_claimed": False,
                "raw_node_rewards_retained": False,
            },
        },
        "exact_soft_label_bt_control": {
            "head": exact_soft_bt,
            "target_audit": {
                "schema_version": "exact-soft-label-bt-target/v1",
                "split": "train",
                "orientation": "candidate_0_minus_candidate_1",
                "input": "sigmoid_of_train_transformed_oracle_margin",
                "target_construction": "p_star = sigmoid(delta_r_star)",
                "source_node_rewards_sha256": train_oracle_reward_sha,
                "canonical_margin_sha256": _token_sha256(f"{seed}:exact-margin"),
                "target_probability_sha256": _token_sha256(f"{seed}:exact-soft-target-probability"),
                "reward_feature_difference_sha256": _token_sha256(
                    f"{seed}:exact-soft-feature-difference"
                ),
                "num_canonical_edges": train_prompts,
                "reward_dimension": _FIXTURE_REWARD_DIMENSION,
                "same_reward_features_and_canonical_edges_as": "exact_margin_prorm_plus",
                "noise_free": True,
                "bernoulli_sampling_used": False,
                "sampled_label_stream_accessed": False,
                "raw_target_probabilities_retained": False,
                "raw_oracle_margins_retained": False,
                "raw_node_rewards_retained": False,
                "test_or_validation_data_accessed": False,
                "role": "noise_free_positive_control_and_secondary_misspecification_diagnostic",
                "eligible_for_primary_claim": False,
            },
            "optimization_audit": {
                "schema_version": "exact-soft-label-bt-optimization/v1",
                "objective": "mean(softplus(delta_r_phi) - p_star * delta_r_phi)",
                "objective_name": "exact_soft_label_bt_cross_entropy",
                "optimizer": "adamw",
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "microbatch_size": 64,
                "max_grad_norm": 1.0,
                "fresh_zero_initialized_bias_free_linear_head": True,
                "head_sha256": exact_soft_bt["head_sha256"],
                "target_probability_sha256": _token_sha256(f"{seed}:exact-soft-target-probability"),
                "reward_feature_difference_sha256": _token_sha256(
                    f"{seed}:exact-soft-feature-difference"
                ),
                "initial_objective": 0.9,
                "final_objective": 0.25,
                "objective_change_final_minus_initial": -0.65,
                "final_full_data_unclipped_gradient_l2_norm": 1.0e-6,
                "final_gradient_ratio_to_zero_initialization": 1.0e-6,
                "first_order_convergence_passed": True,
                "fixed_720_step_checkpoint_role": "compute_matched_and_pilot_diagnostic_only",
                "fixed_720_step_checkpoint_used_for_head_selection": False,
                "favorable_ordering_gate_applied": False,
                "pilot_measure_only": config["design"]["stage"] == "pilot",
                "eligible_for_primary_claim": False,
                "sampled_label_stream_accessed": False,
                "test_or_validation_data_accessed": False,
                "saved_head_mutated_by_audit": False,
            },
        },
        "direct_oracle_identity": {
            "schema_version": "direct-oracle-exact-moment-identity/v1",
            "interpretation": "algebraic_identity_bypasses_reward_class_and_optimizer",
            "source_node_rewards_sha256": train_oracle_reward_sha,
            "num_prompts": train_prompts,
            "num_candidates": candidates,
            "policy_dimension": 512,
            "canonical_margin_sha256": _token_sha256(f"{seed}:canonical-margin"),
            "canonical_pair_moment_sha256": _token_sha256(f"{seed}:canonical-moment"),
            "complete_pair_u_stat_moment_sha256": _token_sha256(f"{seed}:complete-pair"),
            "all_node_covariance_moment_sha256": _token_sha256(f"{seed}:all-node"),
            "complete_pair_identity_absolute_error": 1.0e-12,
            "complete_pair_identity_relative_error": 1.0e-12,
            "complete_pair_identity_is_algebraic": True,
            "reward_head_bypassed": True,
            "optimizer_bypassed": True,
            "trained_exact_margin_head_required_to_match": False,
            "raw_node_rewards_retained": False,
            "native_oracle_direction": {
                "direction_sha256": direct_direction_sha,
                "absolute_damping": 0.001,
                "moment_norm": 1.0,
                "pcg": _pcg(),
            },
        },
        "isolation": {
            "test_data_accessed": False,
            "old_phase1_comparison_heads_used": False,
            "raw_node_rewards_retained": False,
            "raw_labels_retained": False,
            "primary_heads_are_fresh_zero_initialized": True,
        },
    }


def _heldout_fixed_beta(
    *,
    seed: int,
    config: dict[str, Any],
    design_sha: str,
    runtime_sha: str,
    beta: float,
    head_weights: dict[str, list[float]],
    bt_minus_prorm_regret: float = 0.2,
) -> tuple[dict[str, object], str]:
    validation_prompts = int(config["run"]["split_sizes"]["validation"])
    test_prompts = int(config["run"]["split_sizes"]["test"])
    candidates = int(config["data"]["num_candidates"])
    heads_sha = _canonical_sha256(head_weights)

    def split_payload(split_name: str, prompts: int) -> dict[str, object]:
        input_identity = {
            "split": split_name,
            "num_prompts": prompts,
            "num_candidates": candidates,
            "policy_dimension": 512,
            "reward_dimension": _FIXTURE_REWARD_DIMENSION,
            "prompt_ids_sha256": _token_sha256(f"{seed}:{split_name}:prompt-ids"),
            "policy_scores_sha256": _token_sha256(f"{seed}:{split_name}:scores"),
            "reward_features_sha256": _token_sha256(f"{seed}:{split_name}:features"),
            "candidates_sha256": _token_sha256(f"{seed}:{split_name}:candidates"),
            "contains_oracle_targets": False,
        }
        prorm_regret = 0.2
        bt_regret = prorm_regret + bt_minus_prorm_regret
        learners = {
            BT_MLE: {
                "head_sha256": _canonical_sha256(head_weights[BT_MLE]),
                "local_regret_at_frozen_global_beta": bt_regret,
                "native_beta1_squared_fisher_direction_error": 0.4,
                "native_beta1_fisher_cosine": 0.8,
                "native_beta1_predicted_fisher_norm": 1.0,
                "native_beta1_target_fisher_norm": 1.1,
                "direction_vectors_serialized": False,
            },
            PRORM_PLUS: {
                "head_sha256": _canonical_sha256(head_weights[PRORM_PLUS]),
                "local_regret_at_frozen_global_beta": prorm_regret,
                "native_beta1_squared_fisher_direction_error": 0.2,
                "native_beta1_fisher_cosine": 0.9,
                "native_beta1_predicted_fisher_norm": 1.0,
                "native_beta1_target_fisher_norm": 1.1,
                "direction_vectors_serialized": False,
            },
        }
        return {
            "input_identity": input_identity,
            "input_identity_sha256": _canonical_sha256(input_identity),
            "transformed_oracle_rewards_sha256": _token_sha256(
                f"{seed}:{split_name}:oracle-rewards"
            ),
            "raw_oracle_logits_serialized": False,
            "node_fisher_estimator": "mean_all_saved_split_nodes",
            "moment_estimator": "per_prompt_unbiased_candidate_covariance",
            "relative_damping": 0.001,
            "absolute_damping": 0.002,
            "fixed_beta": beta,
            "fixed_beta_source": ("pilot_selected_global_beta_frozen_in_confirmatory_design"),
            "learners": learners,
            "prorm_plus_minus_bt_mle": {
                "local_regret_at_frozen_global_beta": (prorm_regret - bt_regret),
                "native_beta1_squared_fisher_direction_error": -0.2,
                "native_beta1_fisher_cosine": 0.1,
            },
        }

    validation = split_payload("validation", validation_prompts)
    test = split_payload("test", test_prompts)
    deferred_payload = {
        "schema_version": "phase2-deferred-heldout-input/v1",
        "split_order": ["validation", "test"],
        "validation": validation["input_identity"],
        "test": test["input_identity"],
        "contains_oracle_targets": False,
    }
    frozen_state = {
        "schema_version": "phase2-heldout-frozen-state/v1",
        "source_config_hash": config["design"]["source_config_hash"],
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract_sha256": runtime_sha,
        "seed": seed,
        "heads_sha256": heads_sha,
        "training_design_sha256": design_sha,
        "beta_common": beta,
        "deployment_identity_sha256": _token_sha256(f"{seed}:deployment"),
        "heads_frozen": True,
        "beta_common_frozen": True,
        "deployed_directions_frozen": True,
    }
    payload = {
        "schema_version": "phase2-heldout-fixed-beta/v1",
        "estimand": "frozen_global_common_beta_local_regret",
        "formal_gate_split": "test",
        "descriptive_split": "validation",
        "split_order": ["validation", "test"],
        "beta_common": beta,
        "frozen_state": frozen_state,
        "frozen_state_sha256": _canonical_sha256(frozen_state),
        "deferred_input_sha256": _canonical_sha256(deferred_payload),
        "oracle_rescore": {
            "source": "saved_validation_and_test_candidates_rescored_after_policy_freeze",
            "oracle_chat_template_sha256": _token_sha256("oracle-chat-template"),
            "transform": {"b": 0.0, "tau": 1.0},
            "combined_transformed_rewards_sha256": _token_sha256(f"{seed}:heldout-oracle-rewards"),
            "raw_oracle_logits_serialized": False,
        },
        "solver": {
            "pcg_dtype": "float64",
            "pcg_max_iterations": 8192,
            "pcg_tolerance": 1.0e-5,
            "relative_damping": 0.001,
            "split_specific_node_fisher_and_damping": True,
        },
        "splits": {"validation": validation, "test": test},
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
    return payload, _canonical_sha256(payload)


def _rollout_summary(num_trajectories: int, cap_count: int) -> dict[str, object]:
    return {
        "num_trajectories": num_trajectories,
        "terminated_by_eos_count": num_trajectories - cap_count,
        "terminated_by_eos_rate": (num_trajectories - cap_count) / num_trajectories,
        "reached_max_length_count": cap_count,
        "reached_max_length_rate": cap_count / num_trajectories,
        "response_token_count": {
            "mean": 80.0,
            "minimum": 1,
            "maximum": 256,
        },
    }


def _kl_tail_summary(
    *,
    num_prompts: int,
    candidates: int,
    mean_kl: float,
    formal_gate_applied: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "on-policy-kl-tail-summary/v1",
        "unit": "prompt_mean_over_candidates",
        "num_prompts": num_prompts,
        "candidates_per_prompt": candidates,
        "mean": mean_kl,
        "p50": mean_kl,
        "p90": mean_kl,
        "p95": mean_kl,
        "p99": mean_kl,
        "maximum": mean_kl,
        "per_sequence_maximum": mean_kl,
        "pilot_selection_role": "locality_tail_measurement",
        "formal_gate_applied": formal_gate_applied,
    }


def _pre_oracle_safety_gate(
    *,
    runtime: Phase2Design,
    arms: dict[str, object],
) -> dict[str, object]:
    thresholds = {
        "mean_policy_to_reference_kl_cap": runtime.measured_kl_safety_cap,
        "prompt_mean_p95_kl_cap": runtime.prompt_mean_p95_kl_cap,
        "prompt_mean_p99_kl_cap": runtime.prompt_mean_p99_kl_cap,
        "prompt_mean_maximum_kl_cap": runtime.prompt_mean_maximum_kl_cap,
        "per_sequence_maximum_kl_cap": runtime.per_sequence_maximum_kl_cap,
        "reached_max_length_rate_cap": runtime.max_length_formal_threshold,
    }
    observed_by_arm: dict[str, dict[str, float]] = {}
    violations: list[str] = []
    for arm_name in PHASE2_ARM_ORDER:
        arm = arms[arm_name]
        tail = arm["on_policy_kl_tail"]
        rollout = arm["rollout"]
        observed = {
            "mean_policy_to_reference_kl": arm["mean_on_policy_kl_pi_updated_to_pi0"],
            "prompt_mean_p95_kl": tail["p95"],
            "prompt_mean_p99_kl": tail["p99"],
            "prompt_mean_maximum_kl": tail["maximum"],
            "per_sequence_maximum_kl": tail["per_sequence_maximum"],
            "reached_max_length_rate": rollout["reached_max_length_rate"],
        }
        observed_by_arm[arm_name] = observed
        for metric, value in observed.items():
            if value > thresholds[f"{metric}_cap"]:
                violations.append(f"{arm_name}:{metric}")
    formal = runtime.stage == "confirmatory"
    return {
        "schema_version": "phase2-pre-oracle-safety-gate/v1",
        "design_stage": runtime.stage,
        "pilot_phase": runtime.pilot_phase,
        "measure_only": not formal,
        "formal_gate": formal,
        "thresholds": thresholds,
        "observed_by_arm": observed_by_arm,
        "violations": violations,
        "passed": not violations,
        "beta_retuned": False,
        "on_violation": (
            "fail_before_final_oracle_and_heldout"
            if formal
            else "publish_target_free_diagnostics_without_final_oracle"
        ),
    }


def _utility(
    *,
    target_utility: float,
    zero_utility: float,
    oracle_utility: float,
    kl: float,
    beta: float,
    num_prompts: int,
    candidates: int,
) -> dict[str, object]:
    return {
        "schema_version": "downstream-policy-utility/v1",
        "beta_common": beta,
        "num_prompts": num_prompts,
        "candidates_per_prompt": candidates,
        "mean_target_reward": target_utility + beta * kl,
        "mean_on_policy_kl_pi_updated_to_pi0": kl,
        "mean_target_utility": target_utility,
        "target_utility_sample_standard_error": 0.01,
        "improvement_over_zero_b": {
            "mean": target_utility - zero_utility,
            "sample_standard_error": 0.02,
        },
        "oracle_step_reference_gap": {
            "mean": oracle_utility - target_utility,
            "sample_standard_error": 0.02,
        },
        "oracle_step_is_global_optimum": False,
    }


def _write_rollouts(
    path: Path,
    *,
    beta: float,
    arm_values: dict[str, tuple[float, float]],
    count_per_arm: int,
) -> None:
    prompt = "synthetic full prompt"
    prompt_token_ids = [101, 102]
    prompt_semantics = {
        "schema_version": "full-policy-prompt-semantics/v1",
        "raw_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "policy_chat_token_count": len(prompt_token_ids),
        "policy_prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "max_prompt_tokens": 1024,
        "truncated": False,
        "raw_prompt_preserved": True,
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for arm_name in PHASE2_ARM_ORDER:
            target_utility, kl = arm_values[arm_name]
            reward = target_utility + beta * kl
            row = {
                "schema_version": "common-beta-trajectory/v2",
                "arm": arm_name,
                "beta_common": beta,
                "prompt_semantics": prompt_semantics,
                "kl_orientation": "pi_updated_to_pi0",
                "kl_history_source": "updated_policy",
                "on_policy_kl_pi_updated_to_pi0": kl,
                "transformed_oracle_reward": reward,
                "target_utility": target_utility,
                "raw_oracle_logit_serialized": False,
            }
            serialized = json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True)
            for _ in range(count_per_arm):
                handle.write(serialized)
                handle.write("\n")


def _seed_result(
    root: Path,
    config: dict[str, Any],
    seed: int,
    *,
    bt_improvement: float = 0.1,
    prorm_advantage: float = 0.2,
    oracle_improvement: float = 0.6,
    heldout_bt_minus_prorm_regret: float = 0.2,
) -> Path:
    seed_dir = root / f"seed-{seed}"
    seed_dir.mkdir(parents=True)
    result_path = seed_dir / "result.json"
    rollouts_path = seed_dir / "result.rollouts.jsonl"
    design_sha = phase2_design_identity(config)
    runtime = Phase2Design.from_phase2_config(config)
    beta = (
        float(config["objective"]["common_beta"]["frozen_global_beta"])
        if config["design"]["stage"] == "confirmatory"
        else 2.0
    )
    num_prompts = int(config["run"]["split_sizes"]["test"])
    candidates = int(config["data"]["num_candidates"])
    count = num_prompts * candidates
    zero = 1.0 + 0.001 * (seed - int(config["run"]["seeds"][0]))
    bt = zero + bt_improvement
    prorm = bt + prorm_advantage
    oracle = zero + oracle_improvement
    utilities = {
        "zero_b": zero,
        BT_MLE: bt,
        PRORM_PLUS: prorm,
        "oracle_step": oracle,
    }
    kl = {"zero_b": 0.0, BT_MLE: 0.002, PRORM_PLUS: 0.003, "oracle_step": 0.004}
    _write_rollouts(
        rollouts_path,
        beta=beta,
        arm_values={arm: (utilities[arm], kl[arm]) for arm in PHASE2_ARM_ORDER},
        count_per_arm=count,
    )

    head_weights = {
        BT_MLE: _fixture_head(1.0, 0.0),
        PRORM_PLUS: _fixture_head(0.0, 1.0),
    }
    arms: dict[str, object] = {}
    for arm_name in PHASE2_ARM_ORDER:
        arms[arm_name] = {
            "deployment": (
                {
                    "schema_version": "zero-b-deployment/v1",
                    "beta_common": beta,
                    "displacement_is_exact_zero": True,
                    "learner_specific_rescaling": False,
                }
                if arm_name == "zero_b"
                else _direction(arm_name, beta)
            ),
            "rollout": _rollout_summary(count, cap_count=0),
            "mean_on_policy_kl_pi_updated_to_pi0": kl[arm_name],
            "on_policy_kl_tail": _kl_tail_summary(
                num_prompts=num_prompts,
                candidates=candidates,
                mean_kl=kl[arm_name],
                formal_gate_applied=config["design"]["stage"] == "confirmatory",
            ),
            "utility": _utility(
                target_utility=utilities[arm_name],
                zero_utility=zero,
                oracle_utility=oracle,
                kl=kl[arm_name],
                beta=beta,
                num_prompts=num_prompts,
                candidates=candidates,
            ),
        }
    environment = _environment()
    train_oracle_reward_sha = _token_sha256(f"{seed}:train-oracle-rewards")
    training_audit = _head_training_audit(
        seed=seed,
        config=config,
        design_sha=design_sha,
        train_oracle_reward_sha=train_oracle_reward_sha,
        head_weights=head_weights,
    )
    heldout, heldout_sha = _heldout_fixed_beta(
        seed=seed,
        config=config,
        design_sha=design_sha,
        runtime_sha=runtime.sha256,
        beta=beta,
        head_weights=head_weights,
        bt_minus_prorm_regret=heldout_bt_minus_prorm_regret,
    )
    if config["design"]["stage"] == "confirmatory":
        current_seed_curvature = 4.0 + 0.01 * (seed - int(config["run"]["seeds"][0]))
        calibration_evidence: dict[str, object] = {
            "schema_version": "common-beta-frozen-global/v1",
            "rule": "single_pilot_frozen_global_beta_scalar",
            "beta_selection_split": "excluded_pilot",
            "beta_source": ("frozen_pilot_global_beta_in_confirmatory_design_identity"),
            "beta_common": beta,
            "frozen_global_beta": beta,
            "beta_matches_frozen_global_beta": True,
            "beta_selected_from_current_seed_curvature": False,
            "current_seed_oracle_natural_curvature": current_seed_curvature,
            "reference_target_oracle_quadratic_kl": 0.003,
            "predicted_current_seed_oracle_quadratic_kl": (
                0.5 * current_seed_curvature / (beta * beta)
            ),
            "current_seed_curvature_role": "predicted_kl_diagnostic_only",
            "frozen_in_phase2_design_identity": True,
            "learner_specific_rescaling": False,
            "post_evaluation_retuning": False,
        }
        information_boundary: dict[str, object] = {
            "beta_selection_split": "excluded_pilot",
            "current_seed_train_curvature_role": "predicted_kl_diagnostic_only",
            "new_rollout_prompts_used_for_calibration": False,
            "source_materialization_heldout_scores_used_for_calibration": False,
            "new_rollout_oracle_scoring_after_heads_beta_and_directions_frozen": True,
            "heldout_candidate_rescore_after_heads_beta_and_directions_frozen": True,
            "heldout_directions_used_for_policy": False,
        }
    else:
        calibration_evidence = {
            "schema_version": "common-beta-calibration/v1",
            "beta_common": beta,
            "target_oracle_quadratic_kl": 0.003,
            "predicted_oracle_quadratic_kl": 0.003,
            "calibration_split": "train_only",
            "learner_specific_rescaling": False,
        }
        information_boundary = {
            "calibration_split": "train_only",
            "new_rollout_prompts_used_for_calibration": False,
            "source_materialization_heldout_scores_used_for_calibration": False,
            "new_rollout_oracle_scoring_after_heads_beta_and_directions_frozen": True,
            "heldout_candidate_rescore_after_heads_beta_and_directions_frozen": True,
            "heldout_directions_used_for_policy": False,
        }
    payload: dict[str, object] = {
        "schema_version": "common-beta-finite-policy/v2",
        "design_stage": config["design"]["stage"],
        "formal_eligibility": config["design"]["formal_eligibility"],
        "per_seed_supports_formal_claim": False,
        "source_config_hash": config["design"]["source_config_hash"],
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract": runtime.to_dict(),
        "phase2_runtime_contract_sha256": runtime.sha256,
        "seed": seed,
        "artifact_metadata_sha256": "d" * 64,
        "run_manifest_sha256": "e" * 64,
        "environment_identity": environment,
        "current_process_identity": environment,
        "train_oracle_rescore": {
            "source": "saved_train_candidates_rescored_with_pinned_oracle",
            "num_prompts": int(config["run"]["split_sizes"]["train"]),
            "num_candidates": candidates,
            "transformed_rewards_sha256": train_oracle_reward_sha,
            "oracle_chat_template_sha256": _token_sha256("oracle-chat-template"),
            "frozen_transform": {"b": 0.0, "tau": 1.0},
            "raw_oracle_logits_serialized": False,
        },
        "head_training": {
            "training_arm": "r4_independent_gamma_0.9",
            "training_design_sha256": design_sha,
            "heads_sha256": _canonical_sha256(head_weights),
            "head_weights": head_weights,
            "audit": training_audit,
            "source": "trained_after_train_oracle_rescore",
            "old_phase1_comparison_heads_reused": False,
            "test_data_accessed": False,
        },
        "common_beta_calibration": calibration_evidence,
        "train_oracle_direction": {
            "schema_version": "policy-direction/v1",
            "pcg": {"converged": True},
        },
        "measured_kl_safety": {
            "schema_version": "measured-kl-safety/v1",
            "cap": 0.02,
            "passed": True,
            "measured_by_policy": kl,
            "violations": [],
            "beta_retuned": False,
        },
        "pre_oracle_safety_gate": _pre_oracle_safety_gate(
            runtime=runtime,
            arms=arms,
        ),
        "arms": arms,
        "heldout_fixed_beta": heldout,
        "heldout_fixed_beta_sha256": heldout_sha,
        "information_boundary": information_boundary,
        "common_random_numbers": {
            "named_stream": "rollout",
            "same_per_prompt_seed_reset_across_arms": True,
            "candidate_index_alignment": True,
        },
        "policy_and_oracle_co_resident": False,
        "learner_specific_line_search": False,
        "rollouts_jsonl": rollouts_path.name,
        "rollouts_sha256": _sha256(rollouts_path),
    }
    _write_json(result_path, payload)
    return result_path


def _campaign(
    tmp_path: Path,
    *,
    bt_improvement: float = 0.1,
    prorm_advantage: float = 0.2,
    oracle_improvement: float = 0.6,
    heldout_bt_minus_prorm_regret: float = 0.2,
) -> tuple[dict[str, Any], list[Path]]:
    config = load_phase2_config(ROOT / "configs" / "common_beta_pilot.yaml")
    paths = [
        _seed_result(
            tmp_path,
            config,
            int(seed),
            bt_improvement=bt_improvement,
            prorm_advantage=prorm_advantage,
            oracle_improvement=oracle_improvement,
            heldout_bt_minus_prorm_regret=heldout_bt_minus_prorm_regret,
        )
        for seed in config["run"]["seeds"]
    ]
    return config, paths


def _confirmatory_campaign(
    tmp_path: Path,
    *,
    frozen_global_beta: float = 2.5,
) -> tuple[dict[str, Any], list[Path]]:
    config = copy.deepcopy(load_phase2_config(ROOT / "configs" / "common_beta_pilot.yaml"))
    seeds = list(
        range(
            20260901,
            20260901 + PHASE2_MIN_CONFIRMATORY_SEEDS,
        )
    )
    config["design"].update(
        {
            "name": "common-beta-confirmatory-test",
            "stage": "confirmatory",
            "pilot_phase": None,
            "formal_eligibility": True,
            "evidence_role": "confirmatory_evidence",
        }
    )
    config["run"].update(
        {
            "seeds": seeds,
            "confirmatory": True,
            "formal_eligibility": True,
            "excluded_from_confirmatory_evidence": False,
        }
    )
    config["reward_model"]["identifiability"].update(
        {
            "role": "confirmatory_frozen_identifiability_contract",
            "confirmatory_freeze_requirement": "satisfied_by_current_confirmatory_identity",
        }
    )
    config["objective"]["common_beta"].update(
        {
            "rule": "single_pilot_frozen_global_beta_scalar",
            "calibration_split": "excluded_pilot",
            "calibration_source": ("frozen_pilot_global_beta_in_confirmatory_design_identity"),
            "frozen_global_beta": frozen_global_beta,
            "beta_source_aggregate_sha256": "b" * 64,
            "sensitivity_k_cal": None,
            "sensitivity_frozen_global_beta_multipliers": [0.5, 2.0],
            "primary_execution_role": "confirmatory_primary",
            "sensitivity_execution_role": (
                "required_separate_frozen_global_beta_multiplier_sensitivity"
            ),
        }
    )
    config["objective"]["full_tangent"]["ridge"].update(
        {
            "primary_execution_role": "confirmatory_primary",
            "sensitivity_execution_role": "required_separate_confirmatory_sensitivity",
        }
    )
    config["evaluation"]["decision_gates"].update(
        {
            "application": "confirmatory_evidence_decision",
            "supports_formal_claim": True,
        }
    )
    config["evaluation"]["max_length"].update(
        {
            "role": "confirmatory_truncation_safety_gate",
            "measure_only": False,
            "formal_gate": True,
            "formal_threshold": 0.05,
            "parent_pilot_aggregate_sha256": "b" * 64,
            "post_pilot_requirement": "satisfied_by_new_confirmatory_design_identity",
        }
    )
    return config, [_seed_result(tmp_path, config, seed) for seed in seeds]


def _mutate(path: Path, callback: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    callback(value)
    _write_json(path, value)


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(value)
        for child in value.values():
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def test_finite_policy_aggregate_rejects_pilot_diagnostic_schema(
    tmp_path: Path,
) -> None:
    config, paths = _campaign(tmp_path)
    _mutate(
        paths[0],
        lambda value: value.__setitem__(
            "schema_version",
            "common-beta-pilot-diagnostics/v2",
        ),
    )

    with pytest.raises(ValueError, match="common-beta-finite-policy/v2"):
        build_common_beta_seed_aggregate(
            config,
            paths,
            reference_base=tmp_path,
        )


def test_complete_pilot_seed_arithmetic_evidence_sources_and_new_file_only(
    tmp_path: Path,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    output = tmp_path / "aggregate" / "common-beta.json"
    aggregate = write_common_beta_seed_aggregate(config, paths, output)
    payload = aggregate.to_dict()

    assert payload["schema_version"] == PHASE2_AGGREGATE_SCHEMA
    assert payload["seeds"] == sorted(config["run"]["seeds"])
    assert payload["num_seeds"] == 3
    assert payload["experimental_unit"] == "seed"
    assert payload["prompt_or_candidate_pseudo_replication"] is False
    paired = payload["paired_prorm_plus_minus_bt"]["metrics"]
    assert paired["mean_target_utility"]["paired_mean"] == pytest.approx(0.2)
    assert paired["mean_target_utility"]["bootstrap_ci"]["lower"] == pytest.approx(0.2)
    assert paired["improvement_over_zero_b"]["paired_mean"] == pytest.approx(0.2)
    assert paired["oracle_step_reference_gap"]["paired_mean"] == pytest.approx(-0.2)
    assert paired["mean_target_reward"]["paired_mean"] == pytest.approx(0.202)
    assert paired["mean_on_policy_kl_pi_updated_to_pi0"]["paired_mean"] == pytest.approx(0.001)
    oracle = payload["oracle_step_positive_control"]["aggregate"]["metrics"][
        "oracle_step_improvement_over_zero_b"
    ]
    assert oracle["paired_mean"] == pytest.approx(0.6)
    assert oracle["bootstrap_ci"]["lower"] == pytest.approx(0.6)
    prorm_zero = payload["prorm_plus_over_zero_b"]["aggregate"]["metrics"][
        "prorm_plus_improvement_over_zero_b"
    ]
    assert prorm_zero["paired_mean"] == pytest.approx(0.3)
    assert prorm_zero["bootstrap_ci"]["lower"] == pytest.approx(0.3)
    heldout = payload["heldout_bt_minus_prorm_plus"]["aggregate"]["metrics"][
        "bt_mle_minus_prorm_plus_heldout_contrast"
    ]
    assert heldout["paired_mean"] == pytest.approx(0.2)
    assert heldout["bootstrap_ci"]["lower"] == pytest.approx(0.2)
    evidence = payload["pre_registered_evidence"]
    assert evidence["status"] == "pilot_gates_passed_formal_ineligible"
    assert evidence["supports_pre_registered_claim"] is False
    assert evidence["criteria_passed_under_current_gate_contract"] is True
    assert (
        evidence["criteria"]["prorm_plus_improvement_over_zero_b_paired_seed_ci_lower_positive"]
        is True
    )
    gates = payload["control_gate_evidence"]
    assert gates["all_seed_gates_passed"] is True
    assert gates["numeric_gate_contract"]["design_bound"] is True
    assert gates["numeric_gate_contract"]["design_stage"] == "pilot"
    tail = gates["on_policy_kl_tail_diagnostics"]
    assert tail["formal_gate_applied"] is False
    assert tail["formal_threshold"] is None
    assert tail["across_seed_by_arm"][PRORM_PLUS]["p95"]["seed_mean"] == pytest.approx(0.003)
    assert len(gates["per_seed"]) == 3
    assert all(item["gates"]["passed"] is True for item in gates["per_seed"])
    assert all(
        item["gates"]["exact_soft_label_bt_secondary_diagnostic"]["favorable_ordering_gate_applied"]
        is False
        for item in gates["per_seed"]
    )
    assert all(
        item["gates"]["prorm_moment_map_identifiability"][
            "unique_ridge_prorm_quadratic_head_iff_full_column_rank"
        ]
        is True
        for item in gates["per_seed"]
    )
    assert all(
        item["gates"]["prorm_moment_map_identifiability"][
            "population_identifiability_theorem_claimed"
        ]
        is False
        for item in gates["per_seed"]
    )
    assert all(item["max_length"]["measure_only"] is True for item in gates["per_seed"])
    assert all(item["max_length"]["formal_threshold"] == 0.05 for item in gates["per_seed"])
    assert all(
        item["max_length"]["unified_pre_oracle_gate_passed"] is True for item in gates["per_seed"]
    )
    assert all(
        item["on_policy_kl_tail_by_arm"][PRORM_PLUS]["formal_gate_applied"] is False
        for item in gates["per_seed"]
    )
    beta_contract = payload["global_beta_contract"]
    assert beta_contract["role"] == "pilot_per_seed_candidates_only"
    assert beta_contract["frozen_global_beta"] is None
    assert beta_contract["beta_selected_from_current_seed_curvature"] is True
    assert beta_contract["all_confirmatory_seeds_match_frozen_global_beta"] is None
    assert [item["seed"] for item in beta_contract["per_seed_beta_common"]] == sorted(
        config["run"]["seeds"]
    )
    assert payload["integrity"]["rollout_rows_used_for_inference"] is False
    assert payload["integrity"]["all_confirmatory_seeds_used_config_frozen_global_beta"] is None
    assert len(payload["sources"]) == 3
    assert all(len(source["result_sha256"]) == 64 for source in payload["sources"])
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert "p_value" not in _all_keys(payload)
    assert "significant" not in _all_keys(payload)
    assert "all_identity_safety_and_control_gates_passed" not in _all_keys(payload)

    with pytest.raises(FileExistsError, match="overwrite"):
        write_common_beta_seed_aggregate(config, paths, output)


def test_confirmatory_aggregate_proves_one_frozen_beta_across_all_seeds(
    tmp_path: Path,
) -> None:
    config, paths = _confirmatory_campaign(tmp_path / "campaign")
    aggregate = build_common_beta_seed_aggregate(
        config,
        paths,
        reference_base=tmp_path,
    )
    payload = aggregate.to_dict()
    contract = payload["global_beta_contract"]

    assert contract["design_stage"] == "confirmatory"
    assert contract["rule"] == "single_pilot_frozen_global_beta_scalar"
    assert contract["beta_selection_split"] == "excluded_pilot"
    assert contract["beta_source"] == ("frozen_pilot_global_beta_in_confirmatory_design_identity")
    assert contract["role"] == "confirmatory_config_frozen_scalar_verification"
    assert contract["frozen_global_beta"] == 2.5
    assert contract["beta_selected_from_current_seed_curvature"] is False
    assert contract["all_confirmatory_seeds_match_frozen_global_beta"] is True
    assert {item["beta_common"] for item in contract["per_seed_beta_common"]} == {2.5}
    assert payload["integrity"]["all_confirmatory_seeds_used_config_frozen_global_beta"] is True
    assert payload["bootstrap"] == {
        "seed": config["evaluation"]["paired_bootstrap_seed"],
        "resamples": config["evaluation"]["paired_bootstrap_resamples"],
        "method": "paired_seed_percentile_bootstrap",
        "confidence_level": 0.95,
        "interval_sidedness": "two_sided",
        "effective_component_one_sided_alpha": 0.025,
        "interpretation": (
            "frequentist uncertainty for the RNG expectation of the paired contrast "
            "conditional on the frozen prompt pool, models, oracle, and design; not "
            "a claim about an unrestricted human-prompt population"
        ),
    }
    inference = payload["formal_inference_contract"]
    assert inference["test_structure"] == "intersection_union_single_conjunctive_claim"
    assert inference["effective_component_one_sided_alpha"] == 0.025
    assert inference["multiplicity_adjustment"] == ("none_for_intersection_union_conjunctive_claim")
    assert inference["separate_endpoint_claims_without_adjustment_allowed"] is False
    kl_summary = payload["control_gate_evidence"]["on_policy_kl_tail_diagnostics"]
    assert kl_summary["formal_gate_applied"] is True
    assert kl_summary["formal_threshold"] == {
        "mean_policy_to_reference_kl_cap": 0.02,
        "prompt_mean_p95_kl_cap": 0.02,
        "prompt_mean_p99_kl_cap": 0.05,
        "prompt_mean_maximum_kl_cap": 0.10,
        "per_sequence_maximum_kl_cap": 0.20,
    }

    def change_curvature_only(result: dict[str, object]) -> None:
        evidence = result["common_beta_calibration"]
        evidence["current_seed_oracle_natural_curvature"] = 40.0
        evidence["predicted_current_seed_oracle_quadratic_kl"] = 0.5 * 40.0 / (2.5 * 2.5)

    _mutate(paths[0], change_curvature_only)
    curvature_changed = build_common_beta_seed_aggregate(
        config,
        paths,
        reference_base=tmp_path,
    ).to_dict()
    assert {
        item["beta_common"]
        for item in curvature_changed["global_beta_contract"]["per_seed_beta_common"]
    } == {2.5}

    def change_seed_beta(result: dict[str, object]) -> None:
        result["common_beta_calibration"]["beta_common"] = 2.75

    _mutate(paths[0], change_seed_beta)
    with pytest.raises(ValueError, match="exact global beta frozen"):
        build_common_beta_seed_aggregate(
            config,
            paths,
            reference_base=tmp_path,
        )


def test_missing_or_swapped_seed_fails_exact_overlay_set(tmp_path: Path) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    with pytest.raises(ValueError, match="exactly equal"):
        build_common_beta_seed_aggregate(
            config,
            paths[:-1],
            reference_base=tmp_path,
        )

    first_seed = int(config["run"]["seeds"][0])

    def swap_to_duplicate(value: dict[str, object]) -> None:
        value["seed"] = int(config["run"]["seeds"][1])

    _mutate(paths[0], swap_to_duplicate)
    with pytest.raises(ValueError, match="duplicate|exactly match|base_seed"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)
    assert first_seed not in {
        json.loads(path.read_text(encoding="utf-8"))["seed"] for path in paths
    }


def test_cross_seed_environment_identity_mismatch_fails(tmp_path: Path) -> None:
    config, paths = _campaign(tmp_path / "campaign")

    def change_identity(value: dict[str, object]) -> None:
        identity = _environment(gpu="NVIDIA A100")
        value["environment_identity"] = identity
        value["current_process_identity"] = identity

    _mutate(paths[-1], change_identity)
    with pytest.raises(ValueError, match="all overlay seeds must share"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("design_stage", "confirmatory", "design_stage"),
        ("formal_eligibility", True, "formal_eligibility"),
        ("per_seed_supports_formal_claim", True, "per_seed_supports_formal_claim"),
    ],
)
def test_result_v2_stage_and_formal_unit_contract_is_fail_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[0], lambda payload: payload.update({field: value}))
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["arms"][BT_MLE]["on_policy_kl_tail"].update(
                {"formal_gate_applied": True}
            ),
            "formal_gate_applied",
        ),
        (
            lambda payload: payload["arms"][BT_MLE]["on_policy_kl_tail"].update({"mean": 0.5}),
            "does not match",
        ),
        (
            lambda payload: payload["arms"][BT_MLE]["on_policy_kl_tail"].update({"p90": 0.001}),
            "not monotone",
        ),
    ],
)
def test_pilot_kl_tail_contract_rejects_tampering(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[0], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


def test_kl_safety_failure_is_rejected_before_aggregation(tmp_path: Path) -> None:
    config, paths = _campaign(tmp_path / "campaign")

    def fail_safety(value: dict[str, object]) -> None:
        safety = value["measured_kl_safety"]
        safety["passed"] = False
        safety["violations"] = [PRORM_PLUS]

    _mutate(paths[-1], fail_safety)
    with pytest.raises(ValueError, match="KL safety gate"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


def test_pre_oracle_safety_gate_is_mandatory_for_outcome_aggregation(
    tmp_path: Path,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[0], lambda payload: payload.pop("pre_oracle_safety_gate"))

    with pytest.raises(ValueError, match="pre_oracle_safety_gate"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    "metric",
    [
        "mean_policy_to_reference_kl",
        "prompt_mean_p95_kl",
        "prompt_mean_p99_kl",
        "prompt_mean_maximum_kl",
        "per_sequence_maximum_kl",
        "reached_max_length_rate",
    ],
)
def test_pre_oracle_gate_recomputes_all_six_metrics_from_arm_summaries(
    tmp_path: Path,
    metric: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")

    def tamper(value: dict[str, object]) -> None:
        observed = value["pre_oracle_safety_gate"]["observed_by_arm"][PRORM_PLUS]
        observed[metric] = float(observed[metric]) + 1.0e-3

    _mutate(paths[0], tamper)
    with pytest.raises(ValueError, match="serialized tail/length evidence"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


def test_pre_oracle_gate_thresholds_must_equal_runtime_contract(
    tmp_path: Path,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")

    def tamper(value: dict[str, object]) -> None:
        value["pre_oracle_safety_gate"]["thresholds"]["prompt_mean_p99_kl_cap"] = 0.051

    _mutate(paths[0], tamper)
    with pytest.raises(ValueError, match="differs from the runtime contract"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


def test_confirmatory_pre_oracle_violation_cannot_reach_outcome_aggregation(
    tmp_path: Path,
) -> None:
    config, paths = _confirmatory_campaign(tmp_path / "campaign")

    def record_formal_failure(value: dict[str, object]) -> None:
        value["arms"][PRORM_PLUS]["on_policy_kl_tail"]["per_sequence_maximum"] = 0.21
        gate = value["pre_oracle_safety_gate"]
        gate["observed_by_arm"][PRORM_PLUS]["per_sequence_maximum_kl"] = 0.21
        gate["violations"] = [f"{PRORM_PLUS}:per_sequence_maximum_kl"]
        gate["passed"] = False

    _mutate(paths[0], record_formal_failure)
    with pytest.raises(ValueError, match="frozen confirmatory safety gate"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


def test_rollout_path_and_hash_tampering_are_rejected(tmp_path: Path) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    rollouts = paths[0].with_name("result.rollouts.jsonl")
    with rollouts.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)

    config, paths = _campaign(tmp_path / "second-campaign")

    def escape_sibling(value: dict[str, object]) -> None:
        value["rollouts_jsonl"] = "../result.rollouts.jsonl"

    _mutate(paths[0], escape_sibling)
    with pytest.raises(ValueError, match="exact sibling"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["head_training"].update(
                {"old_phase1_comparison_heads_reused": True}
            ),
            "reused old Phase-1 heads",
        ),
        (
            lambda value: value["arms"].pop("oracle_step"),
            "exactly the four",
        ),
    ],
)
def test_r4_fresh_head_and_exact_arm_gates(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[-1], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["head_training"]["audit"]["label_stream"].update(
                {"num_replicates": 3}
            ),
            "num_replicates",
        ),
        (
            lambda value: value["head_training"]["audit"]["label_stream"].update(
                {"clipping": True}
            ),
            "forbidden label clipping",
        ),
        (
            lambda value: value["head_training"]["audit"]["label_stream"].update(
                {"bt_target": "mean_h"}
            ),
            "bt_target",
        ),
    ],
)
def test_r4_independence_no_clipping_and_raw_bt_label_gates_reject_tampering(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[0], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["head_training"]["audit"]["primary_optimization_audit"][
                "reward_head_identifiability"
            ].update({"role": "diagnostic_only"}),
            "role",
        ),
        (
            lambda payload: payload["head_training"]["audit"]["primary_optimization_audit"][
                "reward_head_identifiability"
            ].update(
                {
                    "numerical_rank": 1,
                    "full_column_rank": False,
                    "acceptance_gate_passed": False,
                }
            ),
            "acceptance_gate_passed",
        ),
        (
            lambda payload: payload["head_training"]["audit"]["primary_heads"][BT_MLE][
                "first_order_convergence"
            ]["solution_identification"]["optional_objective_rank_diagnostic"]["evidence"].update(
                {"design_matrix_sha256": "0" * 64}
            ),
            "rank_diagnostic_sha256",
        ),
    ],
)
def test_reward_head_identifiability_is_bound_across_all_head_evidence(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[0], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["head_training"]["audit"]["primary_optimization_audit"][
                "prorm_moment_map_identifiability"
            ]["computation"].update({"randomized_rank_approximation_used": True}),
            "randomized_rank_approximation_used",
        ),
        (
            lambda payload: payload["head_training"]["audit"]["primary_optimization_audit"][
                "prorm_moment_map_identifiability"
            ]["singular_values_descending"].__setitem__(1, 1.499),
            "singular_values_sha256",
        ),
        (
            lambda payload: payload["head_training"]["audit"]["primary_heads"][PRORM_PLUS][
                "first_order_convergence"
            ]["solution_identification"]["optional_objective_rank_diagnostic"]["evidence"].update(
                {"moment_map_sha256": "0" * 64}
            ),
            "rank_diagnostic_sha256",
        ),
        (
            lambda payload: payload["head_training"]["audit"]["exact_margin_control"]["head"][
                "first_order_convergence"
            ]["solution_identification"]["optional_objective_rank_diagnostic"]["evidence"].update(
                {"moment_map_sha256": "0" * 64}
            ),
            "rank_diagnostic_sha256",
        ),
    ],
)
def test_prorm_moment_map_rank_is_exact_and_bound_to_each_prorm_head(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[0], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


def test_moment_map_evidence_is_bound_into_training_instance_hash(
    tmp_path: Path,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")

    def mutate_every_bound_copy(payload: dict[str, object]) -> None:
        audit = payload["head_training"]["audit"]
        replacement = "0" * 64
        audit["primary_optimization_audit"]["prorm_moment_map_identifiability"][
            "moment_map_sha256"
        ] = replacement
        audit["primary_heads"][PRORM_PLUS]["first_order_convergence"]["solution_identification"][
            "optional_objective_rank_diagnostic"
        ]["evidence"]["moment_map_sha256"] = replacement
        audit["exact_margin_control"]["head"]["first_order_convergence"]["solution_identification"][
            "optional_objective_rank_diagnostic"
        ]["evidence"]["moment_map_sha256"] = replacement

    _mutate(paths[0], mutate_every_bound_copy)
    with pytest.raises(ValueError, match="training_instance_sha256"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["head_training"]["audit"]["primary_heads"][BT_MLE][
                "first_order_convergence"
            ].update({"consecutive_threshold_passes_at_selection": 2}),
            "consecutive_threshold_passes_at_selection",
        ),
        (
            lambda payload: payload["head_training"]["audit"]["primary_heads"][PRORM_PLUS][
                "first_order_convergence"
            ]["checks"][-1].update({"gradient_clipping_applied": True}),
            "consecutive full-data checks",
        ),
    ],
)
def test_adaptive_sustained_convergence_evidence_is_not_a_boolean_shortcut(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[0], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["head_training"]["audit"]["direct_oracle_identity"].update(
                {"complete_pair_identity_absolute_error": 1.0e-2}
            ),
            "pair-node moment identity",
        ),
        (
            lambda value: value["head_training"]["audit"]["direct_oracle_identity"][
                "native_oracle_direction"
            ]["pcg"].update({"converged": False}),
            "did not converge",
        ),
    ],
)
def test_direct_pair_node_identity_and_pcg_gates_reject_tampering(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[1], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["head_training"]["audit"]["exact_margin_control"][
                "target_audit"
            ].update({"orientation": "candidate_1_minus_candidate_0"}),
            "orientation",
        ),
        (
            lambda value: value["head_training"]["audit"]["exact_margin_control"]["head"].update(
                {"final_objective": 1.0}
            ),
            "not bound|did not decrease",
        ),
        (
            lambda value: (
                value["head_training"]["audit"]["exact_margin_control"]["optimization_audit"][
                    "learner"
                ].update({"gradient_l2_norm": 1.0}),
                value["head_training"]["audit"]["exact_margin_control"]["head"][
                    "first_order_convergence"
                ]["final_gate"]["measurement"].update({"gradient_l2_norm": 1.0}),
                value["head_training"]["audit"]["exact_margin_control"]["head"][
                    "first_order_convergence"
                ]["final_gate"].update({"gradient_ratio_to_zero_initialization": 1.0}),
            ),
            "outer-convergence gate",
        ),
    ],
)
def test_exact_margin_target_objective_and_outer_gate_reject_tampering(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[2], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["head_training"]["audit"]["exact_soft_label_bt_control"][
                "target_audit"
            ].update({"sampled_label_stream_accessed": True}),
            "sampled_label_stream_accessed",
        ),
        (
            lambda payload: payload["head_training"]["audit"]["exact_soft_label_bt_control"][
                "optimization_audit"
            ].update({"favorable_ordering_gate_applied": True}),
            "favorable_ordering_gate_applied",
        ),
        (
            lambda payload: payload["head_training"]["audit"]["exact_soft_label_bt_control"][
                "optimization_audit"
            ].update({"target_probability_sha256": "0" * 64}),
            "target_probability_sha256",
        ),
        (
            lambda payload: payload["head_training"]["audit"]["exact_soft_label_bt_control"].update(
                {"unexpected": True}
            ),
            "must contain exactly",
        ),
    ],
)
def test_exact_soft_label_bt_is_strict_noise_free_secondary_evidence(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[0], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["head_training"]["audit"]["low_dimensional_control"][
                "geometry"
            ].update({"numerical_rank": 255}),
            "numerical_rank",
        ),
        (
            lambda value: value["head_training"]["audit"]["low_dimensional_control"][
                "projection"
            ].update({"orthonormal_columns": False}),
            "orthonormal_columns",
        ),
        (
            lambda value: value["head_training"]["audit"]["low_dimensional_control"][
                "final_full_data_audit"
            ].update({"pseudoinverse_solve_relative_residual": 1.0e-2}),
            "pseudoinverse residual",
        ),
        (
            lambda value: value["head_training"]["audit"]["low_dimensional_control"][
                "deployment_score_identity"
            ].update({"max_absolute_error": 1.0e-2}),
            "projection/scatter identity",
        ),
        (
            lambda value: value["head_training"]["audit"]["low_dimensional_control"][
                "projected_prorm_moment_map_identifiability"
            ].update({"projection_sha256": "0" * 64}),
            "projection_sha256",
        ),
        (
            lambda value: value["head_training"]["audit"]["low_dimensional_control"][
                "projected_prorm_moment_map_identifiability"
            ]["projected_geometry"].update({"fisher_sha256": "0" * 64}),
            "projected_geometry",
        ),
        (
            lambda value: value["head_training"]["audit"]["low_dimensional_control"]["head"][
                "first_order_convergence"
            ]["solution_identification"]["optional_objective_rank_diagnostic"]["evidence"].update(
                {"moment_map_sha256": "0" * 64}
            ),
            "rank_diagnostic_sha256",
        ),
    ],
)
def test_low_dimensional_rank_orthonormal_pseudoinverse_and_scatter_gates(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")
    _mutate(paths[-1], mutation)
    with pytest.raises(ValueError, match=message):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


def test_low_dimensional_projected_moment_map_is_bound_into_training_instance(
    tmp_path: Path,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")

    def mutate_all_projected_copies(payload: dict[str, object]) -> None:
        low = payload["head_training"]["audit"]["low_dimensional_control"]
        replacement = "0" * 64
        low["projected_prorm_moment_map_identifiability"]["moment_map_sha256"] = replacement
        low["head"]["first_order_convergence"]["solution_identification"][
            "optional_objective_rank_diagnostic"
        ]["evidence"]["moment_map_sha256"] = replacement

    _mutate(paths[-1], mutate_all_projected_copies)
    with pytest.raises(ValueError, match="training_instance_sha256"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


@pytest.mark.parametrize("learner", [BT_MLE, PRORM_PLUS])
def test_primary_outer_convergence_gate_rejects_each_learner(
    tmp_path: Path,
    learner: str,
) -> None:
    config, paths = _campaign(tmp_path / "campaign")

    def fail_outer_gate(value: dict[str, object]) -> None:
        audit = value["head_training"]["audit"]["primary_optimization_audit"]
        audit["learners"][learner]["gradient_l2_norm"] = 1.0
        convergence = value["head_training"]["audit"]["primary_heads"][learner][
            "first_order_convergence"
        ]
        convergence["final_gate"]["measurement"]["gradient_l2_norm"] = 1.0
        convergence["final_gate"]["gradient_ratio_to_zero_initialization"] = 1.0

    _mutate(paths[-1], fail_outer_gate)
    with pytest.raises(ValueError, match="outer-convergence gate"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


def test_inconsistent_summary_arithmetic_is_rejected(tmp_path: Path) -> None:
    config, paths = _campaign(tmp_path / "campaign")

    def break_arithmetic(value: dict[str, object]) -> None:
        improvement = value["arms"][PRORM_PLUS]["utility"]["improvement_over_zero_b"]
        improvement["mean"] += 0.5

    _mutate(paths[2], break_arithmetic)
    with pytest.raises(ValueError, match="improvement-over-zero arithmetic"):
        build_common_beta_seed_aggregate(config, paths, reference_base=tmp_path)


def test_valid_negative_effect_emits_not_passed_without_changing_endpoint(
    tmp_path: Path,
) -> None:
    config, paths = _campaign(
        tmp_path / "campaign",
        prorm_advantage=-0.1,
        oracle_improvement=0.6,
    )
    aggregate = build_common_beta_seed_aggregate(
        config,
        paths,
        reference_base=tmp_path,
    )
    evidence = aggregate.to_dict()["pre_registered_evidence"]

    assert evidence["status"] == "not_passed"
    assert evidence["supports_pre_registered_claim"] is False
    assert (
        evidence["criteria"]["oracle_step_improvement_over_zero_b_paired_seed_ci_lower_positive"]
        is True
    )
    assert evidence["criteria"]["prorm_plus_minus_bt_target_utility_paired_mean_positive"] is False


def test_prorm_plus_must_beat_zero_b_at_the_paired_seed_ci_gate(
    tmp_path: Path,
) -> None:
    config, paths = _campaign(
        tmp_path / "campaign",
        bt_improvement=-0.3,
        prorm_advantage=0.2,
        oracle_improvement=0.6,
    )
    aggregate = build_common_beta_seed_aggregate(
        config,
        paths,
        reference_base=tmp_path,
    )
    payload = aggregate.to_dict()
    evidence = payload["pre_registered_evidence"]

    assert payload["paired_prorm_plus_minus_bt"]["metrics"]["mean_target_utility"]["bootstrap_ci"][
        "lower"
    ] == pytest.approx(0.2)
    assert payload["prorm_plus_over_zero_b"]["aggregate"]["metrics"][
        "prorm_plus_improvement_over_zero_b"
    ]["bootstrap_ci"]["lower"] == pytest.approx(-0.1)
    assert (
        evidence["criteria"]["prorm_plus_improvement_over_zero_b_paired_seed_ci_lower_positive"]
        is False
    )
    assert evidence["criteria_passed_under_current_gate_contract"] is False
    assert evidence["status"] == "not_passed"


def _failure_spec(
    seed: int,
    *,
    attempts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "seed": seed,
        "run_manifest_sha256": "e" * 64,
        "artifact_metadata_sha256": "d" * 64,
        "environment_identity": _environment(),
        "failure_stage": "pre_oracle_safety",
        "failure_class": "safety",
        "failure_type": "prompt_mean_p99_kl_cap_exceeded",
        "failure_message_sha256": _token_sha256("frozen safety gate failed"),
        "final_outcome_reveal_started": False,
        "attempt_ledger": {
            "schema_version": PHASE2_ATTEMPT_LEDGER_SCHEMA,
            "retry_policy": "single_predeclared_attempt_no_retry",
            "replacement_seed_allowed": False,
            "attempts": (
                [
                    {
                        "attempt_index": 1,
                        "cluster_name": "hpc4",
                        "array_job_id": "9001",
                        "array_task_id": seed - 20260901,
                        "slurm_job_id": "9001_0",
                        "status": "terminal_failure",
                        "final_outcome_reveal_started": False,
                        "log_sha256": _token_sha256("terminal log"),
                    }
                ]
                if attempts is None
                else attempts
            ),
        },
        "evidence_sha256_by_role": {
            "phase2_run_log": _token_sha256("phase2 run log"),
            "pre_oracle_gate": _token_sha256("pre-oracle gate evidence"),
        },
    }


def _unavailable_failure_spec(seed: int) -> dict[str, object]:
    spec = _failure_spec(seed)
    for key in (
        "run_manifest_sha256",
        "artifact_metadata_sha256",
        "environment_identity",
    ):
        del spec[key]
    spec["failure_stage"] = "scheduler_reconciliation"
    spec["failure_class"] = "infrastructure"
    spec["failure_type"] = "hard_termination_before_compute_trap"
    spec["capture_method"] = "scheduler_terminal_reconciliation"
    spec["evidence_availability"] = {
        "schema_version": PHASE2_FAILURE_EVIDENCE_AVAILABILITY_SCHEMA,
        "run_manifest": {
            "status": "unavailable",
            "reason": "not_published_before_hard_termination",
        },
        "artifact_metadata": {
            "status": "unavailable",
            "reason": "not_produced_before_failure",
        },
        "environment_identity": {
            "status": "unavailable",
            "reason": "not_recoverable_from_scheduler_evidence",
        },
    }
    spec["evidence_sha256_by_role"] = {
        "scheduler_terminal_attestation": _token_sha256(
            "Slurm terminal state and exit-code attestation"
        )
    }
    return spec


def _success_attempt_ledger(
    seed: int,
) -> dict[str, object]:
    attempts = [
        {
            "attempt_index": 1,
            "cluster_name": "hpc4",
            "array_job_id": str(seed * 100 + 1),
            "array_task_id": seed - 20260901,
            "slurm_job_id": f"{seed}01",
            "status": "success_result",
            "final_outcome_reveal_started": True,
            "log_sha256": _token_sha256(f"seed {seed} attempt 1"),
        }
    ]
    return {
        "schema_version": PHASE2_ATTEMPT_LEDGER_SCHEMA,
        "retry_policy": "single_predeclared_attempt_no_retry",
        "replacement_seed_allowed": False,
        "attempts": attempts,
    }


def _success_terminal(
    config: dict[str, Any],
    result_path: Path,
) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    seed = int(result["seed"])
    output = result_path.with_name("SUCCESS.json")
    write_phase2_seed_success_manifest(
        config,
        result_path,
        _success_attempt_ledger(seed),
        output,
    )
    return output


def _success_terminals(
    config: dict[str, Any],
    result_paths: list[Path],
) -> list[Path]:
    return [_success_terminal(config, path) for path in result_paths]


def test_failed_seed_manifest_is_identity_bound_and_immutable(tmp_path: Path) -> None:
    config, _ = _confirmatory_campaign(tmp_path / "campaign")
    seed = int(config["run"]["seeds"][0])
    output = tmp_path / "failed-seed.json"

    manifest = write_phase2_seed_failure_manifest(
        config,
        _failure_spec(seed),
        output,
    )

    assert manifest["schema_version"] == PHASE2_SEED_FAILURE_SCHEMA
    assert manifest["terminal_status"] == "failed"
    assert manifest["seed"] == seed
    assert manifest["seed_replacement_allowed"] is False
    assert manifest["phase2_design_sha256"] == phase2_design_identity(config)
    assert (
        manifest["phase2_runtime_contract_sha256"] == Phase2Design.from_phase2_config(config).sha256
    )
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError, match="overwrite"):
        write_phase2_seed_failure_manifest(config, _failure_spec(seed), output)


def test_failure_v2_records_unavailable_early_stage_evidence_without_fake_hashes(
    tmp_path: Path,
) -> None:
    config, _ = _confirmatory_campaign(tmp_path / "campaign")
    seed = int(config["run"]["seeds"][0])
    manifest = write_phase2_seed_failure_manifest(
        config,
        _unavailable_failure_spec(seed),
        tmp_path / "hard-termination.json",
    )

    assert manifest["schema_version"] == PHASE2_SEED_FAILURE_SCHEMA
    availability = manifest["evidence_availability"]
    assert availability["schema_version"] == PHASE2_FAILURE_EVIDENCE_AVAILABILITY_SCHEMA
    assert availability["run_manifest"] == {
        "status": "unavailable",
        "reason": "not_published_before_hard_termination",
    }
    assert "sha256" not in availability["run_manifest"]
    assert availability["artifact_metadata"]["status"] == "unavailable"
    assert availability["environment_identity"]["status"] == "unavailable"
    assert set(manifest["evidence_sha256_by_role"]) == {"scheduler_terminal_attestation"}


@pytest.mark.parametrize(
    ("slot_name", "slot_value", "message"),
    [
        (
            "run_manifest",
            {"status": "unavailable", "reason": "", "sha256": "a" * 64},
            "fields differ",
        ),
        (
            "artifact_metadata",
            {"status": "available", "reason": "missing"},
            "fields differ",
        ),
        (
            "run_manifest",
            {"status": "unavailable", "reason": "/secret/job/error text"},
            "must be one of",
        ),
        (
            "environment_identity",
            {"status": "available", "value": None},
            "must be a string-keyed mapping",
        ),
    ],
)
def test_failure_v2_rejects_ambiguous_or_fabricated_availability_slots(
    tmp_path: Path,
    slot_name: str,
    slot_value: dict[str, object],
    message: str,
) -> None:
    config, _ = _confirmatory_campaign(tmp_path / "campaign")
    seed = int(config["run"]["seeds"][0])
    spec = _unavailable_failure_spec(seed)
    spec["evidence_availability"][slot_name] = slot_value
    with pytest.raises((TypeError, ValueError), match=message):
        write_phase2_seed_failure_manifest(
            config,
            spec,
            tmp_path / f"invalid-{slot_name}.json",
        )


def test_campaign_finalizer_rejects_legacy_v1_failure_manifest(
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    seed = int(config["run"]["seeds"][0])
    current = build_phase2_seed_failure_manifest(
        config,
        **_failure_spec(seed),
    )
    availability = current.pop("evidence_availability")
    current.pop("capture_method")
    current["schema_version"] = PHASE2_SEED_FAILURE_SCHEMA_V1
    current["run_manifest_sha256"] = availability["run_manifest"]["sha256"]
    current["artifact_metadata_sha256"] = availability["artifact_metadata"]["sha256"]
    current["environment_identity"] = availability["environment_identity"]["value"]
    legacy = result_paths[0].with_name("legacy-v1-failure.json")
    _write_json(legacy, current)
    terminals[0] = legacy

    with pytest.raises(ValueError, match="neither a success terminal"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            tmp_path / "legacy-campaign.json",
            aggregate_output_json=tmp_path / "legacy-aggregate.json",
        )


def test_failure_manifest_rejects_all_formal_retries(
    tmp_path: Path,
) -> None:
    config, _ = _confirmatory_campaign(tmp_path / "campaign")
    seed = int(config["run"]["seeds"][0])
    valid_attempts = [
        {
            "attempt_index": 1,
            "cluster_name": "hpc4",
            "array_job_id": "9001",
            "array_task_id": seed - 20260901,
            "slurm_job_id": "9001_0",
            "status": "infrastructure_failure_pre_outcome",
            "final_outcome_reveal_started": False,
            "log_sha256": _token_sha256("pre-outcome infrastructure failure"),
        },
        {
            "attempt_index": 2,
            "cluster_name": "hpc4",
            "array_job_id": "9002",
            "array_task_id": seed - 20260901,
            "slurm_job_id": "9002_0",
            "status": "terminal_failure",
            "final_outcome_reveal_started": False,
            "log_sha256": _token_sha256("terminal failure"),
        },
    ]
    with pytest.raises(ValueError, match="exactly one"):
        write_phase2_seed_failure_manifest(
            config,
            _failure_spec(seed, attempts=valid_attempts),
            tmp_path / "invalid-retry.json",
        )
    with pytest.raises(ValueError, match="not predeclared"):
        write_phase2_seed_failure_manifest(
            config,
            _failure_spec(99999999),
            tmp_path / "replacement-seed.json",
        )


def test_success_manifest_binds_result_and_single_attempt(tmp_path: Path) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    result_path = result_paths[0]
    seed = int(config["run"]["seeds"][0])
    output = result_path.with_name("SUCCESS.json")

    manifest = write_phase2_seed_success_manifest(
        config,
        result_path,
        _success_attempt_ledger(seed),
        output,
    )

    assert manifest["schema_version"] == PHASE2_SEED_SUCCESS_SCHEMA
    assert manifest["terminal_status"] == "success_result"
    assert manifest["terminal"] is True
    assert manifest["supports_formal_claim"] is False
    assert manifest["seed"] == seed
    assert manifest["result"] == {
        "path": "result.json",
        "sha256": _sha256(result_path),
        "schema_version": "common-beta-finite-policy/v2",
    }
    rollout_path = result_path.with_name("result.rollouts.jsonl")
    assert manifest["rollout"] == {
        "path": "result.rollouts.jsonl",
        "sha256": _sha256(rollout_path),
        "schema_version": "common-beta-trajectory/v2",
    }
    assert [attempt["attempt_index"] for attempt in manifest["attempt_ledger"]["attempts"]] == [1]
    assert [attempt["status"] for attempt in manifest["attempt_ledger"]["attempts"]] == [
        "success_result",
    ]
    assert manifest["attempt_ledger"]["attempts"][-1]["final_outcome_reveal_started"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError, match="overwrite"):
        write_phase2_seed_success_manifest(
            config,
            result_path,
            _success_attempt_ledger(seed),
            output,
        )


@pytest.mark.parametrize("failure_mode", ["mutated", "deleted"])
def test_success_manifest_rejects_changed_or_missing_rollout_jsonl(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    result_path = result_paths[0]
    rollout_path = result_path.with_name("result.rollouts.jsonl")
    seed = int(config["run"]["seeds"][0])
    if failure_mode == "mutated":
        with rollout_path.open("ab") as handle:
            handle.write(b" ")
        message = "rollout JSONL SHA-256 changed"
    else:
        rollout_path.unlink()
        message = "must be a regular file"

    with pytest.raises(ValueError, match=message):
        write_phase2_seed_success_manifest(
            config,
            result_path,
            _success_attempt_ledger(seed),
            result_path.with_name("SUCCESS.json"),
        )
    assert not result_path.with_name("SUCCESS.json").exists()


def test_success_manifest_rejects_rollout_symlink_substitution(tmp_path: Path) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    result_path = result_paths[0]
    rollout_path = result_path.with_name("result.rollouts.jsonl")
    preserved = rollout_path.with_name("preserved-rollouts.jsonl")
    rollout_path.rename(preserved)
    try:
        rollout_path.symlink_to(preserved.name)
    except OSError as error:
        preserved.rename(rollout_path)
        pytest.skip(f"symbolic links are unavailable: {error}")

    seed = int(config["run"]["seeds"][0])
    with pytest.raises(ValueError, match="must not be a symbolic link"):
        write_phase2_seed_success_manifest(
            config,
            result_path,
            _success_attempt_ledger(seed),
            result_path.with_name("SUCCESS.json"),
        )
    assert not result_path.with_name("SUCCESS.json").exists()


def test_success_manifest_rejects_rollout_with_unbound_row_schema(tmp_path: Path) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    result_path = result_paths[0]
    rollout_path = result_path.with_name("result.rollouts.jsonl")
    lines = rollout_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["schema_version"] = "common-beta-trajectory/v999"
    lines[0] = json.dumps(first, sort_keys=True)
    rollout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["rollouts_sha256"] = _sha256(rollout_path)
    _write_json(result_path, result)

    seed = int(config["run"]["seeds"][0])
    with pytest.raises(ValueError, match="unsupported rollout trajectory schema"):
        write_phase2_seed_success_manifest(
            config,
            result_path,
            _success_attempt_ledger(seed),
            result_path.with_name("SUCCESS.json"),
        )
    assert not result_path.with_name("SUCCESS.json").exists()


def test_success_manifest_rejects_result_rollout_path_escape(tmp_path: Path) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    result_path = result_paths[0]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["rollouts_jsonl"] = "../result.rollouts.jsonl"
    _write_json(result_path, result)
    seed = int(config["run"]["seeds"][0])

    with pytest.raises(ValueError, match="one POSIX basename"):
        write_phase2_seed_success_manifest(
            config,
            result_path,
            _success_attempt_ledger(seed),
            result_path.with_name("SUCCESS.json"),
        )
    assert not result_path.with_name("SUCCESS.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda attempts: attempts[0].__setitem__("attempt_index", 2),
            "contiguous and one-based",
        ),
        (
            lambda attempts: attempts[0].__setitem__(
                "status",
                "terminal_failure",
            ),
            "single formal attempt",
        ),
        (
            lambda attempts: attempts[-1].__setitem__(
                "final_outcome_reveal_started",
                False,
            ),
            "must reveal its final outcome",
        ),
    ],
)
def test_success_manifest_rejects_attempt_ledger_tampering(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    seed = int(config["run"]["seeds"][0])
    ledger = _success_attempt_ledger(seed)
    mutation(ledger["attempts"])

    with pytest.raises(ValueError, match=message):
        build_phase2_seed_success_manifest(
            config,
            result_paths[0],
            ledger,
            reference_base=result_paths[0].parent,
        )


def test_success_spec_is_duplicate_key_free_and_has_no_ambient_fields(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"attempt_ledger":{},"attempt_ledger":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_phase2_seed_success_spec(duplicate)

    unknown = tmp_path / "unknown.json"
    _write_json(
        unknown,
        {
            "attempt_ledger": {},
            "result_json": "ambient-result-path.json",
        },
    )
    with pytest.raises(ValueError, match="fields differ"):
        load_phase2_seed_success_spec(unknown)


def test_formal_campaign_rejects_bare_success_results_without_ledgers(
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    with pytest.raises(ValueError, match="bare result"):
        write_phase2_campaign_terminal(
            config,
            result_paths,
            tmp_path / "bare-terminal.json",
            aggregate_output_json=tmp_path / "bare-aggregate.json",
        )
    assert not (tmp_path / "bare-terminal.json").exists()
    assert not (tmp_path / "bare-aggregate.json").exists()


def test_campaign_failure_finalizer_keeps_exact_30_slots_and_emits_no_ci(
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    failed_index = 7
    failed_seed = int(config["run"]["seeds"][failed_index])
    failure_path = tmp_path / "campaign" / f"seed-{failed_seed}" / "FAILED.json"
    write_phase2_seed_failure_manifest(
        config,
        _failure_spec(failed_seed),
        failure_path,
    )
    terminals[failed_index] = failure_path
    terminal_output = tmp_path / "campaign-terminal.json"
    aggregate_output = tmp_path / "primary-aggregate.json"

    payload = write_phase2_campaign_terminal(
        config,
        terminals,
        terminal_output,
        aggregate_output_json=aggregate_output,
    )

    assert payload["schema_version"] == PHASE2_CAMPAIGN_TERMINAL_SCHEMA
    assert payload["status"] == "not_passed_due_to_seed_failure"
    assert payload["declared_seeds"] == config["run"]["seeds"]
    assert payload["num_declared_seeds"] == 30
    assert payload["terminal_seed_set_complete"] is True
    assert payload["failed_seeds"] == [failed_seed]
    assert payload["seed_replacement_allowed"] is False
    assert payload["primary_ci_computed"] is False
    assert payload["primary_aggregate"] is None
    assert len(payload["entries"]) == 30
    assert not aggregate_output.exists()
    assert json.loads(terminal_output.read_text(encoding="utf-8")) == payload


def test_campaign_finalizer_rejects_missing_duplicate_and_replacement_seed(
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)

    with pytest.raises(ValueError, match="exactly one terminal input"):
        write_phase2_campaign_terminal(
            config,
            terminals[:-1],
            tmp_path / "missing.json",
            aggregate_output_json=tmp_path / "missing-aggregate.json",
        )
    duplicated = [*terminals[:-1], terminals[0]]
    with pytest.raises(ValueError, match="duplicate terminal input"):
        write_phase2_campaign_terminal(
            config,
            duplicated,
            tmp_path / "duplicate.json",
            aggregate_output_json=tmp_path / "duplicate-aggregate.json",
        )

    _mutate(terminals[-1], lambda value: value.__setitem__("seed", 99999999))
    with pytest.raises(ValueError, match="disagrees with its bound result"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            tmp_path / "replacement.json",
            aggregate_output_json=tmp_path / "replacement-aggregate.json",
        )
    assert not (tmp_path / "missing.json").exists()
    assert not (tmp_path / "duplicate.json").exists()
    assert not (tmp_path / "replacement.json").exists()


def test_campaign_finalizer_revalidates_every_success_sidecar_field(
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    original = json.loads(terminals[0].read_text(encoding="utf-8"))
    cases = [
        (
            "identity",
            lambda value: value.__setitem__("phase2_design_sha256", "0" * 64),
            "invalid success terminal identity",
        ),
        (
            "result-sha",
            lambda value: value["result"].__setitem__("sha256", "0" * 64),
            "SHA-256 changed",
        ),
        (
            "absolute-result-path",
            lambda value: value["result"].__setitem__("path", "C:/forged/result.json"),
            "one POSIX basename",
        ),
        (
            "traversal-result-path",
            lambda value: value["result"].__setitem__("path", "../result.json"),
            "one POSIX basename",
        ),
        (
            "nested-result-path",
            lambda value: value["result"].__setitem__("path", "nested/result.json"),
            "one POSIX basename",
        ),
        (
            "backslash-result-path",
            lambda value: value["result"].__setitem__("path", r"nested\result.json"),
            "POSIX separators",
        ),
        (
            "result-copy",
            lambda value: value.__setitem__("run_manifest_sha256", "0" * 64),
            "disagrees with its bound result",
        ),
        (
            "absolute-rollout-path",
            lambda value: value["rollout"].__setitem__(
                "path",
                "C:/forged/result.rollouts.jsonl",
            ),
            "one POSIX basename",
        ),
        (
            "traversal-rollout-path",
            lambda value: value["rollout"].__setitem__(
                "path",
                "../result.rollouts.jsonl",
            ),
            "one POSIX basename",
        ),
        (
            "backslash-rollout-path",
            lambda value: value["rollout"].__setitem__(
                "path",
                r"nested\result.rollouts.jsonl",
            ),
            "POSIX separators",
        ),
        (
            "rollout-sha",
            lambda value: value["rollout"].__setitem__("sha256", "0" * 64),
            "rollout sidecar disagrees with its bound result",
        ),
        (
            "rollout-schema",
            lambda value: value["rollout"].__setitem__(
                "schema_version",
                "common-beta-trajectory/v999",
            ),
            "rollout sidecar disagrees with its bound result",
        ),
        (
            "attempt-index",
            lambda value: value["attempt_ledger"]["attempts"][0].__setitem__(
                "attempt_index",
                2,
            ),
            "contiguous and one-based",
        ),
        (
            "terminal-status",
            lambda value: value["attempt_ledger"]["attempts"][0].__setitem__(
                "status",
                "terminal_failure",
            ),
            "single formal attempt",
        ),
        (
            "success-without-reveal",
            lambda value: value["attempt_ledger"]["attempts"][0].__setitem__(
                "final_outcome_reveal_started",
                False,
            ),
            "must reveal its final outcome",
        ),
    ]
    for name, mutation, message in cases:
        tampered = copy.deepcopy(original)
        mutation(tampered)
        tampered_path = terminals[0].parent / f"{name}-SUCCESS.json"
        _write_json(tampered_path, tampered)
        inputs = [tampered_path, *terminals[1:]]
        terminal_output = tmp_path / f"{name}-campaign.json"
        aggregate_output = tmp_path / f"{name}-aggregate.json"
        with pytest.raises(ValueError, match=message):
            write_phase2_campaign_terminal(
                config,
                inputs,
                terminal_output,
                aggregate_output_json=aggregate_output,
            )
        assert not terminal_output.exists()
        assert not aggregate_output.exists()


def test_campaign_finalizer_rejects_slurm_job_reuse_across_seed_ledgers(
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    first = json.loads(terminals[0].read_text(encoding="utf-8"))
    reused_job_id = first["attempt_ledger"]["attempts"][-1]["slurm_job_id"]

    def reuse_job(value: dict[str, object]) -> None:
        value["attempt_ledger"]["attempts"][-1]["slurm_job_id"] = reused_job_id

    _mutate(terminals[1], reuse_job)
    with pytest.raises(ValueError, match="repeat Slurm job identity"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            tmp_path / "reused-job-campaign.json",
            aggregate_output_json=tmp_path / "reused-job-aggregate.json",
        )
    assert not (tmp_path / "reused-job-campaign.json").exists()
    assert not (tmp_path / "reused-job-aggregate.json").exists()


def test_campaign_finalizer_rejects_symlink_substitution_for_bound_result(
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    result_path = result_paths[0]
    preserved = result_path.with_name("preserved-result.json")
    result_path.rename(preserved)
    try:
        result_path.symlink_to(preserved.name)
    except OSError as error:
        preserved.rename(result_path)
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            tmp_path / "symlink-campaign.json",
            aggregate_output_json=tmp_path / "symlink-aggregate.json",
        )
    assert not (tmp_path / "symlink-campaign.json").exists()
    assert not (tmp_path / "symlink-aggregate.json").exists()


def test_campaign_finalizer_rehashes_all_terminal_manifests_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    original_builder = phase2_campaign_module.build_common_beta_seed_aggregate

    def mutate_after_aggregate(*args: Any, **kwargs: Any) -> Any:
        aggregate = original_builder(*args, **kwargs)
        terminal = json.loads(terminals[0].read_text(encoding="utf-8"))
        terminal["pre_oracle_safety_gate_passed"] = False
        _write_json(terminals[0], terminal)
        return aggregate

    monkeypatch.setattr(
        phase2_campaign_module,
        "build_common_beta_seed_aggregate",
        mutate_after_aggregate,
    )
    with pytest.raises(ValueError, match="terminal manifest changed during finalization"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            tmp_path / "toctou-campaign.json",
            aggregate_output_json=tmp_path / "toctou-aggregate.json",
        )
    assert not (tmp_path / "toctou-campaign.json").exists()
    assert not (tmp_path / "toctou-aggregate.json").exists()


def test_campaign_finalizer_rehashes_rollouts_before_success_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    rollout_path = result_paths[0].with_name("result.rollouts.jsonl")
    original_builder = phase2_campaign_module.build_common_beta_seed_aggregate

    def mutate_after_aggregate(*args: Any, **kwargs: Any) -> Any:
        aggregate = original_builder(*args, **kwargs)
        with rollout_path.open("ab") as handle:
            handle.write(b" ")
        return aggregate

    monkeypatch.setattr(
        phase2_campaign_module,
        "build_common_beta_seed_aggregate",
        mutate_after_aggregate,
    )
    terminal_output = tmp_path / "rollout-toctou-campaign.json"
    aggregate_output = tmp_path / "rollout-toctou-aggregate.json"
    with pytest.raises(ValueError, match="rollout JSONL changed during finalization"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            terminal_output,
            aggregate_output_json=aggregate_output,
        )
    assert not terminal_output.exists()
    assert not aggregate_output.exists()


def test_failed_seed_terminal_branch_rehashes_surviving_success_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    failed_index = 7
    failed_seed = int(config["run"]["seeds"][failed_index])
    failure_path = result_paths[failed_index].with_name("FAILED.json")
    write_phase2_seed_failure_manifest(
        config,
        _failure_spec(failed_seed),
        failure_path,
    )
    terminals[failed_index] = failure_path
    original_validator = phase2_campaign_module._validate_campaign_job_id_uniqueness

    def mutate_after_ledger_validation(*args: Any, **kwargs: Any) -> None:
        original_validator(*args, **kwargs)
        _mutate(result_paths[0], lambda value: value.__setitem__("toctou", True))

    monkeypatch.setattr(
        phase2_campaign_module,
        "_validate_campaign_job_id_uniqueness",
        mutate_after_ledger_validation,
    )
    with pytest.raises(ValueError, match="successful result changed during finalization"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            tmp_path / "failed-seed-toctou-campaign.json",
            aggregate_output_json=tmp_path / "failed-seed-toctou-aggregate.json",
        )
    assert not (tmp_path / "failed-seed-toctou-campaign.json").exists()
    assert not (tmp_path / "failed-seed-toctou-aggregate.json").exists()


def test_failed_seed_terminal_branch_rehashes_surviving_rollouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    failed_index = 7
    failed_seed = int(config["run"]["seeds"][failed_index])
    failure_path = result_paths[failed_index].with_name("FAILED.json")
    write_phase2_seed_failure_manifest(
        config,
        _failure_spec(failed_seed),
        failure_path,
    )
    terminals[failed_index] = failure_path
    rollout_path = result_paths[0].with_name("result.rollouts.jsonl")
    original_validator = phase2_campaign_module._validate_campaign_job_id_uniqueness

    def mutate_after_ledger_validation(*args: Any, **kwargs: Any) -> None:
        original_validator(*args, **kwargs)
        with rollout_path.open("ab") as handle:
            handle.write(b" ")

    monkeypatch.setattr(
        phase2_campaign_module,
        "_validate_campaign_job_id_uniqueness",
        mutate_after_ledger_validation,
    )
    terminal_output = tmp_path / "failed-seed-rollout-toctou-campaign.json"
    aggregate_output = tmp_path / "failed-seed-rollout-toctou-aggregate.json"
    with pytest.raises(ValueError, match="rollout JSONL changed during finalization"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            terminal_output,
            aggregate_output_json=aggregate_output,
        )
    assert not terminal_output.exists()
    assert not aggregate_output.exists()


def test_aggregate_exception_branch_rehashes_success_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)

    def mutate_then_fail(*args: Any, **kwargs: Any) -> Any:
        _mutate(result_paths[0], lambda value: value.__setitem__("toctou", True))
        raise ValueError("forced aggregate validation failure")

    monkeypatch.setattr(
        phase2_campaign_module,
        "build_common_beta_seed_aggregate",
        mutate_then_fail,
    )
    with pytest.raises(ValueError, match="successful result changed during finalization"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            tmp_path / "aggregate-exception-toctou-campaign.json",
            aggregate_output_json=tmp_path / "aggregate-exception-toctou-aggregate.json",
        )
    assert not (tmp_path / "aggregate-exception-toctou-campaign.json").exists()
    assert not (tmp_path / "aggregate-exception-toctou-aggregate.json").exists()


def test_aggregate_exception_branch_rehashes_rollouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    rollout_path = result_paths[0].with_name("result.rollouts.jsonl")

    def mutate_then_fail(*args: Any, **kwargs: Any) -> Any:
        with rollout_path.open("ab") as handle:
            handle.write(b" ")
        raise ValueError("forced aggregate validation failure")

    monkeypatch.setattr(
        phase2_campaign_module,
        "build_common_beta_seed_aggregate",
        mutate_then_fail,
    )
    terminal_output = tmp_path / "aggregate-exception-rollout-campaign.json"
    aggregate_output = tmp_path / "aggregate-exception-rollout-aggregate.json"
    with pytest.raises(ValueError, match="rollout JSONL changed during finalization"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            terminal_output,
            aggregate_output_json=aggregate_output,
        )
    assert not terminal_output.exists()
    assert not aggregate_output.exists()


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (OSError, "transient aggregate filesystem failure"),
        (ValueError, "deterministic aggregate validation failure"),
    ],
)
def test_aggregate_errors_propagate_without_publishing_or_guessing_failed_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: type[Exception],
    message: str,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)

    def fail_aggregate(*args: Any, **kwargs: Any) -> Any:
        raise error_type(message)

    monkeypatch.setattr(
        phase2_campaign_module,
        "build_common_beta_seed_aggregate",
        fail_aggregate,
    )
    terminal_output = tmp_path / "aggregate-error-campaign.json"
    aggregate_output = tmp_path / "aggregate-error-primary.json"
    with pytest.raises(error_type, match=message):
        write_phase2_campaign_terminal(
            config,
            terminals,
            terminal_output,
            aggregate_output_json=aggregate_output,
        )
    assert not terminal_output.exists()
    assert not aggregate_output.exists()


def test_positive_branch_rejects_disappeared_success_result_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    original_builder = phase2_campaign_module.build_common_beta_seed_aggregate

    def remove_after_aggregate(*args: Any, **kwargs: Any) -> Any:
        aggregate = original_builder(*args, **kwargs)
        result_paths[0].unlink()
        return aggregate

    monkeypatch.setattr(
        phase2_campaign_module,
        "build_common_beta_seed_aggregate",
        remove_after_aggregate,
    )
    with pytest.raises(ValueError, match="successful result is no longer a regular file"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            tmp_path / "missing-result-campaign.json",
            aggregate_output_json=tmp_path / "missing-result-aggregate.json",
        )
    assert not (tmp_path / "missing-result-campaign.json").exists()
    assert not (tmp_path / "missing-result-aggregate.json").exists()


@pytest.mark.parametrize("failure_kind", ["identity", "safety"])
def test_identity_or_safety_tamper_after_success_manifest_is_rejected(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)

    def invalidate(value: dict[str, object]) -> None:
        if failure_kind == "identity":
            value["phase2_design_sha256"] = "0" * 64
        else:
            gate = value["pre_oracle_safety_gate"]
            gate["passed"] = False
            gate["violations"] = [f"{PRORM_PLUS}:prompt_mean_p99_kl"]

    _mutate(result_paths[2], invalidate)
    terminal_output = tmp_path / f"{failure_kind}-terminal.json"
    aggregate_output = tmp_path / f"{failure_kind}-aggregate.json"
    with pytest.raises(ValueError, match="SHA-256 changed"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            terminal_output,
            aggregate_output_json=aggregate_output,
        )
    assert not terminal_output.exists()
    assert not aggregate_output.exists()


def test_all_success_campaign_finalizer_is_the_only_path_that_computes_primary_ci(
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")
    terminals = _success_terminals(config, result_paths)
    terminal_output = tmp_path / "campaign-terminal.json"
    aggregate_output = tmp_path / "primary-aggregate.json"

    payload = write_phase2_campaign_terminal(
        config,
        terminals,
        terminal_output,
        aggregate_output_json=aggregate_output,
    )

    assert payload["status"] == "primary_aggregate_completed"
    assert payload["failed_seeds"] == []
    assert payload["successful_result_seeds"] == config["run"]["seeds"]
    assert payload["primary_ci_computed"] is True
    assert payload["primary_aggregate"]["sha256"] == _sha256(aggregate_output)
    assert aggregate_output.exists()


def test_invalid_success_result_aborts_finalization_without_guessing_seed_failure(
    tmp_path: Path,
) -> None:
    config, result_paths = _confirmatory_campaign(tmp_path / "campaign")

    def fail_numeric_gate(value: dict[str, object]) -> None:
        convergence = value["head_training"]["audit"]["primary_heads"][BT_MLE][
            "first_order_convergence"
        ]
        convergence["final_gate"]["measurement"]["gradient_l2_norm"] = 1.0
        convergence["final_gate"]["gradient_ratio_to_zero_initialization"] = 1.0

    _mutate(result_paths[4], fail_numeric_gate)
    terminals = _success_terminals(config, result_paths)
    terminal_output = tmp_path / "invalid-terminal.json"
    aggregate_output = tmp_path / "invalid-aggregate.json"
    with pytest.raises(ValueError, match="failed the sustained outer-convergence gate"):
        write_phase2_campaign_terminal(
            config,
            terminals,
            terminal_output,
            aggregate_output_json=aggregate_output,
        )

    assert not terminal_output.exists()
    assert not aggregate_output.exists()
