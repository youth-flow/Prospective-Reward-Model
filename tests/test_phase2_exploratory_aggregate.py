from __future__ import annotations

import copy
import hashlib
import json
import math
import struct

import pytest
import torch

from smart_reward.phase2_config import (
    PHASE2_FROZEN_ORACLE_B,
    PHASE2_FROZEN_ORACLE_TAU,
    PHASE2_RECOVERY_LR_SCHEDULE_SHA256,
)
from smart_reward.phase2_exploratory_aggregate import (
    FIXED_THREE_EXPLORATORY_SCHEMA,
    FIXED_THREE_EXPLORATORY_SEEDS,
    assert_exploratory_payload_has_no_inferential_fields,
    build_fixed_three_exploratory_aggregate,
    normalize_budgeted_end_to_end_seed_result,
    validate_fixed_three_exploratory_aggregate,
)
from smart_reward.phase2_heldout import heldout_evaluation_sha256
from smart_reward.phase2_rollout import BUDGETED_COMMON_BETA_RULE, Phase2Design


def _records() -> list[dict[str, object]]:
    bt_regret = [5.0, 6.0, 7.0]
    prorm_regret = [4.0, 5.0, 8.0]
    bt_utility = [1.0, 1.2, 0.9]
    prorm_utility = [1.2, 1.1, 1.3]
    return [
        {
            "seed": seed,
            "admissible": True,
            "phase2_design_sha256": "a" * 64,
            "phase2_runtime_contract_sha256": "b" * 64,
            "beta_source_aggregate_sha256": "c" * 64,
            "frozen_global_beta": 2.5,
            "endpoints": {
                "heldout_local_regret": {
                    "bt_mle": bt_regret[index],
                    "prorm_plus": prorm_regret[index],
                },
                "finite_policy_utility": {
                    "bt_mle": bt_utility[index],
                    "prorm_plus": prorm_utility[index],
                },
                "oracle_pairwise_cross_entropy": {
                    "bt_mle": 0.72 + 0.01 * index,
                    "prorm_plus": 0.68 + 0.005 * index,
                },
                "oracle_probability_mae": {
                    "bt_mle": 0.20 + 0.01 * index,
                    "prorm_plus": 0.16 + 0.005 * index,
                },
                "pairwise_order_accuracy": {
                    "bt_mle": 0.60 + 0.01 * index,
                    "prorm_plus": 0.66 + 0.01 * index,
                },
            },
        }
        for index, seed in enumerate(FIXED_THREE_EXPLORATORY_SEEDS)
    ]


def _aggregate(
    records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return build_fixed_three_exploratory_aggregate(
        _records() if records is None else records,
        bootstrap_seed=934,
        bootstrap_resamples=2_000,
    )


def _all_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(key)
            keys.extend(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_all_keys(child))
    return keys


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _float32_head_sha256(values: list[float]) -> str:
    digest = hashlib.sha256()
    digest.update(b"torch.float32")
    digest.update(repr((len(values),)).encode("ascii"))
    digest.update(struct.pack(f"<{len(values)}f", *values))
    return digest.hexdigest()


def _budgeted_protocol() -> dict[str, object]:
    return {
        "schema_version": "deterministic-adamw-lr-decay/v1",
        "first_order_audit_dtype": "float64",
        "legacy_constant_lr_boundary_snapshot_steps": 5760,
        "validation_or_test_selection": False,
        "learning_rate_schedule": {
            "schedule_sha256": PHASE2_RECOVERY_LR_SCHEDULE_SHA256,
        },
    }


def _budgeted_first_order(head_sha256: str | None = None) -> dict[str, object]:
    protocol = _budgeted_protocol()

    def check(step: int) -> dict[str, object]:
        return {
            "step": step,
            "threshold_passed": True,
            "post_update": True,
            "full_data": True,
            "gradient_clipping_applied": False,
            "measurement": {"audit_dtype": "float64", "gradient_l2_norm": 0.001},
        }

    result = {
        "converged": True,
        "fail_closed": True,
        "test_or_validation_data_accessed": False,
        "spec": {
            "fail_closed": True,
            "gradient": "full_data_post_update_unclipped",
            "validation_or_test_selection": False,
            "schema_version": "objective-first-order-convergence-spec/v2",
            "gradient_ratio_tolerance": 0.001,
            "gradient_norm_denominator_floor": 1.0e-30,
            "min_steps": 100,
            "max_steps": 12760,
            "check_interval": 20,
            "consecutive_checks": 3,
            "optimizer_protocol": protocol,
            "denominator": "exact_zero_initialization_gradient_l2_norm",
        },
        "schema_version": "objective-first-order-convergence/v2",
        "selected_primary_step": 5760,
        "consecutive_threshold_passes_at_selection": 3,
        "initial_zero_head_measurement": {"gradient_l2_norm": 2.0},
        "final_gate": {
            "step": 5760,
            "threshold_passed": True,
            "fresh_post_restore_audit": True,
            "gradient_ratio_to_zero_initialization": 0.0005,
            "measurement": {"audit_dtype": "float64", "gradient_l2_norm": 0.001},
        },
        "checks": [
            {**check(5720), "gradient_ratio_to_zero_initialization": 0.0005},
            {**check(5740), "gradient_ratio_to_zero_initialization": 0.0005},
            {**check(5760), "gradient_ratio_to_zero_initialization": 0.0005},
        ],
        "fixed_step_snapshot_steps": 720,
        "fixed_step_snapshot_is_not_primary_selection": True,
        "fixed_step_compute_matched_snapshot": {
            "step": 720,
            "used_as_primary_selection_rule": False,
        },
        "legacy_constant_lr_boundary_snapshot": {
            "step": 5760,
            "used_as_primary_selection_rule": False,
            "test_or_validation_data_accessed": False,
        },
        "optimizer_protocol_execution": {
            "schema_version": "deterministic-adamw-lr-decay-execution/v2",
            "protocol": protocol,
            "completed_updates_observed": 5760,
            "per_update_state_checks": {"present": True},
            "test_or_validation_data_accessed": False,
        },
    }
    if head_sha256 is not None:
        result["selected_primary_head_sha256"] = head_sha256
        result["optimizer_protocol_execution"]["selected_head_sha256"] = head_sha256
    return result


def _serialized_head(
    *,
    arm: str,
    method: str,
    weights: list[float],
) -> dict[str, object]:
    head_sha256 = _float32_head_sha256(weights)
    return {
        "arm": arm,
        "method": method,
        "head_weight": list(weights),
        "head_dtype": "torch.float32",
        "initial_head_sha256": _float32_head_sha256([0.0] * len(weights)),
        "head_sha256": head_sha256,
        "first_order_convergence": _budgeted_first_order(head_sha256),
    }


def _duplicate_selected_first_order_step(value: dict[str, object]) -> None:
    convergence = value["head_training"]["audit"]["primary_heads"]["prorm_plus"][
        "first_order_convergence"
    ]
    convergence["checks"][1]["step"] = convergence["checks"][0]["step"]


def _rebind_primary_head_to_wrong_digest(value: dict[str, object]) -> None:
    head = value["head_training"]["audit"]["primary_heads"]["bt_mle"]
    wrong = "9" * 64
    head["head_sha256"] = wrong
    convergence = head["first_order_convergence"]
    convergence["selected_primary_head_sha256"] = wrong
    convergence["optimizer_protocol_execution"]["selected_head_sha256"] = wrong


def _rebind_control_head_to_wrong_digest(value: dict[str, object]) -> None:
    head = value["head_training"]["audit"]["exact_margin_control"]["head"]
    wrong = "8" * 64
    head["head_sha256"] = wrong
    convergence = head["first_order_convergence"]
    convergence["selected_primary_head_sha256"] = wrong
    convergence["optimizer_protocol_execution"]["selected_head_sha256"] = wrong


def _budgeted_heldout() -> dict[str, object]:
    def pcg() -> dict[str, object]:
        return {
            "iterations": 3,
            "residual_norm": 1.0e-8,
            "relative_residual": 1.0e-8,
            "converged": True,
            "reason": "converged",
            "cold_start": True,
            "true_residual_reported": True,
        }

    weights = {"bt_mle": [0.1], "prorm_plus": [0.2]}

    def split(offset: float) -> dict[str, object]:
        return {
            "fixed_beta": 2.5,
            "fixed_beta_source": (
                "accepted_freeze_global_beta_frozen_in_budgeted_end_to_end_design"
            ),
            "learners": {
                "bt_mle": {
                    "local_regret_at_frozen_global_beta": 0.4 + offset,
                    "head_sha256": _canonical_sha256(weights["bt_mle"]),
                },
                "prorm_plus": {
                    "local_regret_at_frozen_global_beta": 0.2 + offset,
                    "head_sha256": _canonical_sha256(weights["prorm_plus"]),
                },
            },
            "preference_fit": {
                "bt_mle": {
                    "oracle_pairwise_cross_entropy": 0.70,
                    "oracle_probability_mae": 0.20,
                    "pairwise_order_accuracy": 0.60,
                },
                "prorm_plus": {
                    "oracle_pairwise_cross_entropy": 0.65,
                    "oracle_probability_mae": 0.15,
                    "pairwise_order_accuracy": 0.66,
                },
            },
            "heldout_pcg_evidence": {
                "schema_version": "heldout-pcg-evidence/v1",
                "operator": "node_empirical_fisher_plus_split_specific_isotropic_damping",
                "pcg_dtype": "float64",
                "pcg_max_iterations": 100,
                "pcg_tolerance": 1.0e-5,
                "preconditioner": "none",
                "residual_recompute_interval": 20,
                "all_solves_cold_start": True,
                "all_solves_converged": True,
                "target_direction_shared_across_learners": True,
                "target_direction": pcg(),
                "learners": {
                    learner: {
                        "predicted_direction": pcg(),
                        "reward_error_direction": pcg(),
                    }
                    for learner in ("bt_mle", "prorm_plus")
                },
            },
        }

    return {
        "schema_version": "phase2-heldout-fixed-beta/v2",
        "estimand": "frozen_global_common_beta_local_regret",
        "formal_gate_split": None,
        "descriptive_split": "validation",
        "split_order": ["validation", "test"],
        "beta_common": 2.5,
        "frozen_state": {},
        "frozen_state_sha256": "d" * 64,
        "deferred_input_sha256": "e" * 64,
        "oracle_rescore": {},
        "solver": {},
        "splits": {"validation": split(0.1), "test": split(0.0)},
        "information_boundary": {
            "fresh_targets_created_after_heads_beta_and_deployments_frozen": True,
            "validation_or_test_targets_available_to_head_trainer": False,
            "validation_or_test_targets_available_to_beta_calibration": False,
            "validation_or_test_targets_available_to_policy_deployment": False,
            "heldout_direction_used_for_policy": False,
        },
        "raw_oracle_logits_serialized": False,
        "heldout_direction_vectors_serialized": False,
        "evaluation_evidence_role": "budgeted_end_to_end_exploratory_heldout_evidence",
        "formal_claim_eligible": False,
        "primary_descriptive_split": "test",
        "operational_oracle_preference_fit": {
            "schema_version": "operational-oracle-preference-fit-contract/v1",
            "pair_definition": "all_unordered_candidate_pairs_within_prompt",
            "expected_pairs_per_prompt_for_four_candidates": 6,
            "aggregation": "mean_pairs_within_prompt_then_mean_prompts",
            "cross_entropy_and_probability_mae_include_oracle_ties": True,
            "oracle_or_predicted_tie_accuracy_credit": 0.5,
        },
    }


def _budgeted_result() -> dict[str, object]:
    heldout = _budgeted_heldout()
    source_digest = "c" * 64
    source_config = _digest("source-config")
    weights = {"bt_mle": [0.1], "prorm_plus": [0.2]}
    heads_sha = _canonical_sha256(weights)
    runtime = Phase2Design(
        stage="budgeted_end_to_end",
        formal_eligibility=False,
        pilot_phase=None,
        common_beta_rule=BUDGETED_COMMON_BETA_RULE,
        common_beta_calibration_split="excluded_pilot",
        common_beta_source=("accepted_freeze_global_beta_in_budgeted_end_to_end_design_identity"),
        frozen_global_beta=2.5,
        beta_source_aggregate_sha256=source_digest,
        parent_pilot_aggregate_sha256=source_digest,
        rollout_candidates_per_prompt=4,
        pcg_max_iterations=100,
        pcg_tolerance=1.0e-5,
        k_cal_sensitivity_values=None,
        frozen_global_beta_sensitivity_multipliers=None,
    ).to_dict()
    runtime_sha = _canonical_sha256(runtime)
    frozen = {
        "schema_version": "common-beta-frozen-global-budgeted/v1",
        "evidence_role": "budgeted_end_to_end_fixed_three_exploratory_only",
        "formal_eligibility": False,
        "supports_formal_claim": False,
        "beta_common": 2.5,
        "frozen_global_beta": 2.5,
        "beta_source_aggregate_sha256": source_digest,
        "beta_matches_frozen_global_beta": True,
        "beta_selected_from_current_seed_curvature": False,
        "accepted_freeze_beta_reused_without_recalibration": True,
        "current_seed_can_change_beta": False,
        "frozen_in_phase2_design_identity": True,
        "learner_specific_rescaling": False,
        "post_evaluation_retuning": False,
    }
    result: dict[str, object] = {
        "schema_version": "common-beta-budgeted-end-to-end/v1",
        "design_stage": "budgeted_end_to_end",
        "formal_eligibility": False,
        "formal_claim_eligible": False,
        "supports_formal_claim": False,
        "per_seed_supports_formal_claim": False,
        "excluded_from_confirmatory_evidence": True,
        "confirmatory_authorization_created": False,
        "evidence_role": "budgeted_end_to_end_fixed_three_exploratory_only",
        "seed": FIXED_THREE_EXPLORATORY_SEEDS[0],
        "phase2_design_sha256": "a" * 64,
        "phase2_runtime_contract_sha256": runtime_sha,
        "phase2_runtime_contract": runtime,
        "source_config_hash": source_config,
        "common_beta_frozen_evidence": frozen,
        "head_training": {
            "test_data_accessed": False,
            "old_phase1_comparison_heads_reused": False,
            "source": "trained_after_train_oracle_rescore",
            "heads_sha256": heads_sha,
            "head_weights": weights,
            "audit": {
                "schema_version": "phase2-fresh-head-training/v3",
                "training_design_sha256": "a" * 64,
                "training_settings_sha256": _digest("settings"),
                "training_instance_sha256": _digest("instance"),
                "input_training_sha256": _digest("input"),
                "training_arm": "r4_independent_gamma_0.9",
                "primary_heads": {
                    learner: _serialized_head(
                        arm="r4_independent_gamma_0.9",
                        method=learner,
                        weights=weights[learner],
                    )
                    for learner in ("bt_mle", "prorm_plus")
                },
                "low_dimensional_control": {
                    "head": _serialized_head(
                        arm="low_dimensional_tangent_positive_control",
                        method="prorm_plus",
                        weights=[0.3],
                    ),
                    "bt_head": {
                        "head_sha256": _float32_head_sha256(weights["bt_mle"]),
                        "retrained": False,
                    },
                },
                "exact_margin_control": {
                    "head": _serialized_head(
                        arm="exact_margin_positive_control",
                        method="prorm_plus",
                        weights=[0.4],
                    ),
                },
                "exact_soft_label_bt_control": {
                    "head": _serialized_head(
                        arm="exact_soft_label_bt_secondary_diagnostic",
                        method="bt_mle",
                        weights=[0.5],
                    ),
                },
                "primary_optimization_audit": {"proven": True},
                "direct_oracle_identity": {"proven": True},
                "isolation": {
                    "test_data_accessed": False,
                    "old_phase1_comparison_heads_used": False,
                    "raw_node_rewards_retained": False,
                    "raw_labels_retained": False,
                    "primary_heads_are_fresh_zero_initialized": True,
                },
            },
        },
        "pre_oracle_safety_gate": {
            "schema_version": "phase2-pre-oracle-safety-gate/v2",
            "design_stage": "budgeted_end_to_end",
            "measure_only": False,
            "formal_gate": False,
            "enforced_before_final_oracle": True,
            "supports_formal_claim": False,
            "evidence_role": "budgeted_end_to_end_fixed_three_exploratory_only",
            "passed": True,
            "beta_retuned": False,
            "on_violation": "fail_before_final_oracle_and_heldout",
            "violations": [],
        },
        "heldout_fixed_beta": heldout,
        "heldout_fixed_beta_sha256": heldout_evaluation_sha256(heldout),
    }
    frozen_state = {
        "schema_version": "phase2-heldout-frozen-state/v1",
        "source_config_hash": source_config,
        "phase2_design_sha256": result["phase2_design_sha256"],
        "phase2_runtime_contract_sha256": runtime_sha,
        "seed": result["seed"],
        "heads_sha256": heads_sha,
        "training_design_sha256": result["phase2_design_sha256"],
        "beta_common": 2.5,
        "deployment_identity_sha256": _digest("deployment"),
        "heads_frozen": True,
        "beta_common_frozen": True,
        "deployed_directions_frozen": True,
    }
    heldout["frozen_state"] = frozen_state
    heldout["frozen_state_sha256"] = _canonical_sha256(frozen_state)
    heldout["oracle_rescore"] = {
        "source": "saved_validation_and_test_candidates_rescored_after_policy_freeze",
        "oracle_chat_template_sha256": _digest("template"),
        "transform": {"b": PHASE2_FROZEN_ORACLE_B, "tau": PHASE2_FROZEN_ORACLE_TAU},
        "combined_transformed_rewards_sha256": _digest("heldout-rewards"),
        "raw_oracle_logits_serialized": False,
    }
    heldout["solver"] = {
        "pcg_dtype": "float64",
        "pcg_max_iterations": 100,
        "pcg_tolerance": 1.0e-5,
        "relative_damping": 0.001,
        "split_specific_node_fisher_and_damping": True,
        "explicit_pcg_evidence_serialized_per_split": True,
        "all_direction_and_regret_solves_audited": True,
    }
    result["heldout_fixed_beta_sha256"] = heldout_evaluation_sha256(heldout)

    def arm(utility: float) -> dict[str, object]:
        kl = 0.01
        return {
            "mean_on_policy_kl_pi_updated_to_pi0": kl,
            "on_policy_kl_tail": {
                "p95": kl,
                "p99": kl,
                "maximum": kl,
                "per_sequence_maximum": kl,
            },
            "rollout": {"reached_max_length_rate": 0.0},
            "utility": {
                "beta_common": 2.5,
                "mean_target_reward": utility + 2.5 * kl,
                "mean_on_policy_kl_pi_updated_to_pi0": kl,
                "mean_target_utility": utility,
            },
        }

    result["arms"] = {
        "zero_b": arm(0.8),
        "bt_mle": arm(1.2),
        "prorm_plus": arm(1.5),
        "oracle_step": arm(1.8),
    }
    observed = {
        name: {
            "mean_policy_to_reference_kl": 0.01,
            "prompt_mean_p95_kl": 0.01,
            "prompt_mean_p99_kl": 0.01,
            "prompt_mean_maximum_kl": 0.01,
            "per_sequence_maximum_kl": 0.01,
            "reached_max_length_rate": 0.0,
        }
        for name in result["arms"]
    }
    result["pre_oracle_safety_gate"].update(
        {
            "thresholds": {
                "mean_policy_to_reference_kl_cap": 0.02,
                "prompt_mean_p95_kl_cap": 0.02,
                "prompt_mean_p99_kl_cap": 0.05,
                "prompt_mean_maximum_kl_cap": 0.1,
                "per_sequence_maximum_kl_cap": 0.2,
                "reached_max_length_rate_cap": 0.05,
            },
            "observed_by_arm": observed,
        }
    )
    return result


def test_complete_fixed_three_emits_only_descriptive_oriented_effects() -> None:
    payload = _aggregate()

    assert payload["schema_version"] == FIXED_THREE_EXPLORATORY_SCHEMA
    assert payload["analysis_role"] == "fixed_three_exploratory_descriptive_only"
    assert payload["seeds"] == list(FIXED_THREE_EXPLORATORY_SEEDS)
    assert payload["observed_seeds"] == list(FIXED_THREE_EXPLORATORY_SEEDS)
    assert payload["formal_claim_eligible"] is False
    assert payload["aggregation_state"] == "complete_descriptive_aggregate"
    assert payload["contrast_orientation"] == "prorm_plus_minus_bt"

    effects = payload["effect_summaries"]
    regret = effects["heldout_local_regret"]
    assert [row["prorm_plus_minus_bt"] for row in regret["per_seed"]] == [
        -1.0,
        -1.0,
        1.0,
    ]
    assert regret["n"] == 3
    assert regret["mean"] == pytest.approx(-1.0 / 3.0)
    assert regret["sd_ddof1"] == pytest.approx(math.sqrt(4.0 / 3.0))
    assert regret["min"] == -1.0
    assert regret["median"] == -1.0
    assert regret["max"] == 1.0
    assert regret["direction"] == "lower_is_better"
    assert regret["favorable_prorm_plus_minus_bt_sign"] == "negative"

    utility = effects["finite_policy_utility"]
    assert utility["direction"] == "higher_is_better"
    assert utility["favorable_prorm_plus_minus_bt_sign"] == "positive"
    assert utility["n"] == 3
    assert utility["per_seed"][0]["prorm_plus_minus_bt"] == pytest.approx(0.2)

    cross_entropy = effects["oracle_pairwise_cross_entropy"]
    assert cross_entropy["direction"] == "lower_is_better"
    assert cross_entropy["favorable_prorm_plus_minus_bt_sign"] == "negative"
    assert cross_entropy["per_seed"][0]["prorm_plus_minus_bt"] == pytest.approx(-0.04)

    probability_mae = effects["oracle_probability_mae"]
    assert probability_mae["direction"] == "lower_is_better"
    assert probability_mae["favorable_prorm_plus_minus_bt_sign"] == "negative"

    order_accuracy = effects["pairwise_order_accuracy"]
    assert order_accuracy["direction"] == "higher_is_better"
    assert order_accuracy["favorable_prorm_plus_minus_bt_sign"] == "positive"
    assert order_accuracy["per_seed"][0]["prorm_plus_minus_bt"] == pytest.approx(0.06)
    assert validate_fixed_three_exploratory_aggregate(payload) == payload


def test_deterministic_paired_bootstrap_does_not_touch_global_torch_rng() -> None:
    torch.manual_seed(817)
    state = torch.random.get_rng_state().clone()

    first = _aggregate()
    second = _aggregate()

    assert first == second
    assert torch.equal(torch.random.get_rng_state(), state)
    interval = first["effect_summaries"]["heldout_local_regret"][
        "paired_seed_bootstrap_descriptive_interval"
    ]
    assert interval["method"] == "paired_seed_percentile_bootstrap"
    assert interval["confidence_level"] == 0.95
    assert first["bootstrap"] == {
        "method": "paired_seed_percentile_bootstrap",
        "unit": "seed",
        "resamples": 2_000,
        "seed": 934,
        "confidence_level": 0.95,
        "interpretation": "descriptive_only",
    }


@pytest.mark.parametrize("missing_index", range(3))
def test_any_missing_seed_withholds_all_effect_summaries(missing_index: int) -> None:
    records = _records()
    del records[missing_index]

    payload = _aggregate(records)

    assert "effect_summaries" not in payload
    assert payload["aggregation_state"] == "effect_summaries_withheld"
    assert payload["missing_seeds"] == [FIXED_THREE_EXPLORATORY_SEEDS[missing_index]]
    assert validate_fixed_three_exploratory_aggregate(payload) == payload


@pytest.mark.parametrize("inadmissible_index", range(3))
def test_any_inadmissible_seed_withholds_all_effect_summaries(
    inadmissible_index: int,
) -> None:
    records = _records()
    records[inadmissible_index]["admissible"] = False
    del records[inadmissible_index]["endpoints"]

    payload = _aggregate(records)

    assert "effect_summaries" not in payload
    assert payload["missing_seeds"] == []
    assert payload["inadmissible_seeds"] == [FIXED_THREE_EXPLORATORY_SEEDS[inadmissible_index]]
    assert validate_fixed_three_exploratory_aggregate(payload) == payload


@pytest.mark.parametrize(
    "records",
    [
        lambda values: list(reversed(values)),
        lambda values: values + [copy.deepcopy(values[-1])],
        lambda values: [values[0], values[0], *values[2:]],
    ],
)
def test_seed_order_duplicates_and_extras_fail_closed(records: object) -> None:
    malformed = records(_records())
    with pytest.raises(ValueError, match="fixed|order|three"):
        _aggregate(malformed)


@pytest.mark.parametrize("historical_seed", [20261004, 20261005])
def test_historical_fixed_five_only_seeds_are_rejected(historical_seed: int) -> None:
    records = _records()
    records[-1]["seed"] = historical_seed

    with pytest.raises(ValueError, match="fixed exploratory seeds"):
        _aggregate(records)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("phase2_design_sha256", "d" * 64),
        ("phase2_runtime_contract_sha256", "e" * 64),
        ("beta_source_aggregate_sha256", "f" * 64),
        ("frozen_global_beta", 2.75),
    ],
)
def test_all_seed_beta_design_and_runtime_identities_must_match(
    field: str,
    replacement: object,
) -> None:
    records = _records()
    records[2][field] = replacement

    with pytest.raises(ValueError, match="identity"):
        _aggregate(records)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("phase2_design_sha256", "A" * 64),
        ("phase2_runtime_contract_sha256", "short"),
        ("beta_source_aggregate_sha256", None),
        ("frozen_global_beta", 0.0),
        ("frozen_global_beta", float("nan")),
    ],
)
def test_identity_fields_are_strictly_validated(
    field: str,
    replacement: object,
) -> None:
    records = _records()
    records[0][field] = replacement

    with pytest.raises((TypeError, ValueError), match="hexadecimal|positive|finite"):
        _aggregate(records)


@pytest.mark.parametrize(
    "field",
    [
        "passed",
        "not_passed",
        "p-value",
        "p_value",
        "significant",
        "significance",
        "supports_formal_claim",
        "nested_formal_decision",
    ],
)
def test_inferential_and_formal_fields_are_recursively_forbidden(field: str) -> None:
    records = _records()
    endpoints = records[0]["endpoints"]
    endpoints["heldout_local_regret"]["audit"] = {field: False}

    with pytest.raises(ValueError, match="forbidden"):
        _aggregate(records)


def test_only_root_false_formal_marker_is_allowed() -> None:
    payload = _aggregate()
    forbidden_keys = {
        "passed",
        "not_passed",
        "p_value",
        "p-value",
        "significant",
        "significance",
    }
    assert not forbidden_keys.intersection(_all_keys(payload))
    assert json.dumps(payload).count('"formal_claim_eligible"') == 1
    assert_exploratory_payload_has_no_inferential_fields(payload)

    nested = copy.deepcopy(payload)
    nested["identity"]["formal_claim_eligible"] = False
    with pytest.raises(ValueError, match="forbidden"):
        assert_exploratory_payload_has_no_inferential_fields(nested)

    root_true = copy.deepcopy(payload)
    root_true["formal_claim_eligible"] = True
    with pytest.raises(ValueError, match="must be false"):
        assert_exploratory_payload_has_no_inferential_fields(root_true)


def test_fixed_endpoint_set_and_learner_pair_cannot_drift() -> None:
    records = _records()
    records[0]["endpoints"]["extra_metric"] = {"bt_mle": 1.0, "prorm_plus": 2.0}
    with pytest.raises(ValueError, match="fixed endpoint set"):
        _aggregate(records)

    records = _records()
    records[0]["endpoints"]["heldout_local_regret"]["other"] = 1.0
    with pytest.raises(ValueError, match="exactly"):
        _aggregate(records)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_endpoint_values_fail_closed(invalid: float) -> None:
    records = _records()
    records[1]["endpoints"]["finite_policy_utility"]["prorm_plus"] = invalid
    with pytest.raises(ValueError, match="finite"):
        _aggregate(records)


def test_validator_rejects_tampered_orientation_statistics_and_withholding() -> None:
    payload = _aggregate()

    tampered = copy.deepcopy(payload)
    tampered["verdict"] = "descriptive"
    with pytest.raises(ValueError, match="root fields"):
        validate_fixed_three_exploratory_aggregate(tampered)

    tampered = copy.deepcopy(payload)
    row = tampered["effect_summaries"]["heldout_local_regret"]["per_seed"][0]
    row["prorm_plus_minus_bt"] = 1.0
    with pytest.raises(ValueError, match="ProRM\\+ minus BT"):
        validate_fixed_three_exploratory_aggregate(tampered)

    tampered = copy.deepcopy(payload)
    tampered["effect_summaries"]["heldout_local_regret"]["sd_ddof1"] = 0.0
    with pytest.raises(ValueError, match="sd_ddof1"):
        validate_fixed_three_exploratory_aggregate(tampered)

    incomplete = _aggregate(_records()[:-1])
    incomplete["effect_summaries"] = payload["effect_summaries"]
    with pytest.raises(ValueError, match="exactly when"):
        validate_fixed_three_exploratory_aggregate(incomplete)


def test_historical_fixed_five_aggregate_does_not_validate_as_current() -> None:
    historical = _aggregate()
    historical["schema_version"] = "prorm-phase2-fixed-five-exploratory-descriptive-aggregate/v1"
    historical["analysis_role"] = "fixed_five_exploratory_descriptive_only"
    historical["seeds"] = [20261001, 20261002, 20261003, 20261004, 20261005]

    with pytest.raises(ValueError, match="schema_version"):
        validate_fixed_three_exploratory_aggregate(historical)


def test_budgeted_normalizer_extracts_the_five_fixed_endpoints() -> None:
    result = _budgeted_result()
    normalized = normalize_budgeted_end_to_end_seed_result(result)

    assert normalized["seed"] == FIXED_THREE_EXPLORATORY_SEEDS[0]
    assert normalized["phase2_design_sha256"] == "a" * 64
    assert normalized["phase2_runtime_contract_sha256"] == result["phase2_runtime_contract_sha256"]
    assert normalized["beta_source_aggregate_sha256"] == "c" * 64
    assert normalized["frozen_global_beta"] == 2.5
    assert normalized["admissible"] is True
    assert normalized["endpoints"] == {
        "heldout_local_regret": {"bt_mle": 0.4, "prorm_plus": 0.2},
        "finite_policy_utility": {"bt_mle": 1.2, "prorm_plus": 1.5},
        "oracle_pairwise_cross_entropy": {"bt_mle": 0.70, "prorm_plus": 0.65},
        "oracle_probability_mae": {"bt_mle": 0.20, "prorm_plus": 0.15},
        "pairwise_order_accuracy": {"bt_mle": 0.60, "prorm_plus": 0.66},
    }


def test_historical_raw_result_evidence_role_is_not_currently_admissible() -> None:
    result = _budgeted_result()
    result["evidence_role"] = "budgeted_end_to_end_exploratory_only"

    normalized = normalize_budgeted_end_to_end_seed_result(result)

    assert normalized["admissible"] is False
    assert "endpoints" not in normalized


def test_budgeted_normalizer_accepts_canonical_sorted_json_round_trip() -> None:
    result = json.loads(
        json.dumps(
            _budgeted_result(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    normalized = normalize_budgeted_end_to_end_seed_result(result)

    assert normalized["admissible"] is True
    assert set(normalized["endpoints"]) == {
        "heldout_local_regret",
        "finite_policy_utility",
        "oracle_pairwise_cross_entropy",
        "oracle_probability_mae",
        "pairwise_order_accuracy",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["pre_oracle_safety_gate"].update({"passed": False}),
        lambda value: value["pre_oracle_safety_gate"].update(
            {"enforced_before_final_oracle": False}
        ),
        lambda value: value["head_training"]["audit"]["primary_heads"]["bt_mle"][
            "first_order_convergence"
        ]["final_gate"].update({"gradient_ratio_to_zero_initialization": 0.01}),
        lambda value: value["heldout_fixed_beta"]["splits"]["test"]["heldout_pcg_evidence"][
            "learners"
        ]["prorm_plus"]["reward_error_direction"].update({"cold_start": False}),
        lambda value: value["heldout_fixed_beta"]["splits"]["validation"]["heldout_pcg_evidence"][
            "target_direction"
        ].update({"relative_residual": 1.0e-3}),
    ],
)
def test_budgeted_normalizer_withholds_all_endpoints_on_gate_or_pcg_tamper(
    mutate: object,
) -> None:
    result = _budgeted_result()
    mutate(result)
    # The held-out hash is deliberately not recomputed: both payload-content
    # and identity tampering must fail closed through the same public bridge.

    normalized = normalize_budgeted_end_to_end_seed_result(result)

    assert normalized["seed"] == FIXED_THREE_EXPLORATORY_SEEDS[0]
    assert normalized["phase2_design_sha256"] == "a" * 64
    assert normalized["phase2_runtime_contract_sha256"] == result["phase2_runtime_contract_sha256"]
    assert normalized["beta_source_aggregate_sha256"] == "c" * 64
    assert normalized["frozen_global_beta"] == 2.5
    assert normalized["admissible"] is False
    assert "endpoints" not in normalized


def test_budgeted_normalizer_rejects_formal_or_nonfixed_raw_results_but_keeps_identity() -> None:
    result = _budgeted_result()
    result["excluded_from_confirmatory_evidence"] = False
    result["formal_claim_eligible"] = True

    normalized = normalize_budgeted_end_to_end_seed_result(result)

    assert normalized["admissible"] is False
    assert "endpoints" not in normalized
    assert normalized["seed"] == FIXED_THREE_EXPLORATORY_SEEDS[0]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: (
            value["head_training"]["audit"]["primary_heads"]["bt_mle"]["first_order_convergence"][
                "spec"
            ].update({"gradient_ratio_tolerance": 10.0}),
            value["head_training"]["audit"]["primary_heads"]["bt_mle"]["first_order_convergence"][
                "final_gate"
            ].update({"gradient_ratio_to_zero_initialization": 5.0}),
        ),
        lambda value: value["head_training"]["audit"].update({"direct_oracle_identity": {}}),
        lambda value: value.update({"phase2_runtime_contract_sha256": "b" * 64}),
        lambda value: value["arms"]["prorm_plus"]["utility"].update({"mean_target_utility": 1.0e9}),
        _duplicate_selected_first_order_step,
        _rebind_primary_head_to_wrong_digest,
        _rebind_control_head_to_wrong_digest,
    ],
)
def test_budgeted_normalizer_rejects_authority_and_endpoint_tampering(
    mutate: object,
) -> None:
    result = _budgeted_result()
    mutate(result)

    normalized = normalize_budgeted_end_to_end_seed_result(result)

    assert normalized["admissible"] is False
    assert "endpoints" not in normalized
    assert normalized["seed"] == FIXED_THREE_EXPLORATORY_SEEDS[0]


def test_budgeted_normalizer_rejects_cross_seed_heldout_splice_even_after_rehash() -> None:
    result = _budgeted_result()
    heldout = result["heldout_fixed_beta"]
    frozen = heldout["frozen_state"]
    frozen["seed"] = FIXED_THREE_EXPLORATORY_SEEDS[1]
    heldout["frozen_state_sha256"] = _canonical_sha256(frozen)
    result["heldout_fixed_beta_sha256"] = heldout_evaluation_sha256(heldout)

    normalized = normalize_budgeted_end_to_end_seed_result(result)

    assert normalized["admissible"] is False
    assert "endpoints" not in normalized
