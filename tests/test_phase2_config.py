from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from smart_reward.config import ConfigError, config_hash, load_config
from smart_reward.phase2_config import (
    PHASE1_MAIN_CONFIG_HASH,
    PHASE1_MAIN_SEEDS,
    PHASE2_CONFIRMATORY_EXCLUDED_SEEDS,
    PHASE2_CONFIRMATORY_NUM_SEEDS,
    PHASE2_CONFIRMATORY_SEEDS,
    PHASE2_FIXED_LORA_A_INITIALIZATION_SEED,
    PHASE2_FIXED_LORA_A_SHA256,
    PHASE2_FIXED_LORA_A_SOURCE_METADATA_SHA256,
    PHASE2_FIXED_LORA_A_SOURCE_SEED,
    PHASE2_FROZEN_ORACLE_B,
    PHASE2_FROZEN_ORACLE_SOURCE_ARTIFACTS,
    PHASE2_FROZEN_ORACLE_TAU,
    PHASE2_PILOT_BASE_CONFIG,
    PHASE2_PILOT_CONFIG,
    PHASE2_PILOT_SEEDS,
    PHASE2_SCHEMA_VERSION,
    load_phase2_config,
    load_phase2_config_bundle,
    phase2_design_identity,
    validate_phase2_config,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / PHASE2_PILOT_CONFIG
BASE_PATH = ROOT / PHASE2_PILOT_BASE_CONFIG
MAIN_PATH = ROOT / "configs" / "main.yaml"
EXPECTED_BASE_IDENTITY = "81ccbd3bc9d745d3792d1834116ba2480d34a7201d6f4e463a61d8d8ab0baefa"
EXPECTED_PILOT_IDENTITY = "0c8820b67b8ca85c23cd5c31b8d25001018b31a7271f6a861d88cfae2f85d7ca"


@pytest.fixture
def pilot_config() -> dict[str, Any]:
    return load_phase2_config(PILOT_PATH)


def _set_path(config: dict[str, Any], path: str, value: object) -> None:
    target: dict[str, Any] = config
    components = path.split(".")
    for component in components[:-1]:
        target = target[component]
    target[components[-1]] = value


def _future_confirmatory(
    pilot: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = copy.deepcopy(pilot)
    base = load_config(BASE_PATH)
    seeds = list(PHASE2_CONFIRMATORY_SEEDS)

    overlay["design"].update(
        {
            "name": "common-beta-confirmatory-future",
            "stage": "confirmatory",
            "pilot_phase": None,
            "formal_eligibility": True,
            "evidence_role": "confirmatory_evidence",
            "source_config": "configs/future_confirmatory_base.yaml",
        }
    )
    overlay["run"].update(
        {
            "seeds": seeds,
            "confirmatory": True,
            "formal_eligibility": True,
            "excluded_from_confirmatory_evidence": False,
        }
    )
    base["run"]["name"] = "future-confirmatory-materialization"
    base["run"]["seeds"] = seeds
    overlay["reward_model"]["identifiability"].update(
        {
            "role": "confirmatory_frozen_identifiability_contract",
            "confirmatory_freeze_requirement": ("satisfied_by_current_confirmatory_identity"),
        }
    )
    common_beta = overlay["objective"]["common_beta"]
    common_beta.update(
        {
            "rule": "single_pilot_frozen_global_beta_scalar",
            "calibration_split": "excluded_pilot",
            "calibration_source": ("frozen_pilot_global_beta_in_confirmatory_design_identity"),
            "frozen_global_beta": 2.5,
            "beta_source_aggregate_sha256": "a" * 64,
            "sensitivity_k_cal": None,
            "sensitivity_frozen_global_beta_multipliers": [0.5, 2.0],
            "primary_execution_role": "confirmatory_primary",
            "sensitivity_execution_role": (
                "required_separate_frozen_global_beta_multiplier_sensitivity"
            ),
        }
    )
    ridge = overlay["objective"]["full_tangent"]["ridge"]
    ridge["primary_execution_role"] = "confirmatory_primary"
    ridge["sensitivity_execution_role"] = "required_separate_confirmatory_sensitivity"
    overlay["evaluation"]["decision_gates"].update(
        {
            "application": "confirmatory_evidence_decision",
            "supports_formal_claim": True,
        }
    )
    overlay["evaluation"]["max_length"].update(
        {
            "role": "confirmatory_truncation_safety_gate",
            "measure_only": False,
            "formal_gate": True,
            "formal_threshold": 0.05,
            "parent_pilot_aggregate_sha256": "a" * 64,
            "post_pilot_requirement": ("satisfied_by_new_confirmatory_design_identity"),
        }
    )
    overlay["design"]["source_config_hash"] = config_hash(base)
    return overlay, base


def _pilot_freeze(
    pilot: dict[str, Any],
    *,
    beta: float = 2.5,
    calibration_aggregate_sha256: str = "b" * 64,
) -> dict[str, Any]:
    overlay = copy.deepcopy(pilot)
    overlay["design"].update(
        {
            "name": "common-beta-pilot-freeze-test",
            "pilot_phase": "freeze",
        }
    )
    overlay["objective"]["common_beta"].update(
        {
            "rule": "pilot_fixed_global_beta_target_free_safety_rehearsal",
            "calibration_split": "excluded_pilot_calibration",
            "calibration_source": (
                "frozen_calibration_aggregate_candidate_in_pilot_freeze_design_identity"
            ),
            "frozen_global_beta": beta,
            "beta_source_aggregate_sha256": calibration_aggregate_sha256,
            "sensitivity_k_cal": None,
            "sensitivity_frozen_global_beta_multipliers": None,
            "primary_execution_role": "pilot_frozen_global_beta_safety_rehearsal",
            "sensitivity_execution_role": ("new_pilot_freeze_design_identity_double_beta_grid"),
        }
    )
    overlay["evaluation"]["decision_gates"]["application"] = (
        "pilot_freeze_target_free_safety_selection"
    )
    overlay["evaluation"]["max_length"].update(
        {
            "role": "pilot_frozen_global_beta_safety_selection",
            "measure_only": True,
            "formal_gate": False,
            "formal_threshold": 0.05,
            "parent_pilot_aggregate_sha256": calibration_aggregate_sha256,
            "post_pilot_requirement": (
                "freeze_confirmatory_identity_if_passed_else_double_beta_new_identity"
            ),
        }
    )
    return overlay


def test_pilot_bundle_binds_overlay_and_declared_base() -> None:
    bundle = load_phase2_config_bundle(PILOT_PATH)

    assert bundle.config["schema_version"] == PHASE2_SCHEMA_VERSION
    assert bundle.design_identity == EXPECTED_PILOT_IDENTITY
    assert phase2_design_identity(bundle.config) == EXPECTED_PILOT_IDENTITY
    assert bundle.design_identity != PHASE1_MAIN_CONFIG_HASH
    assert config_hash(bundle.base_config) == EXPECTED_BASE_IDENTITY
    assert bundle.config["design"]["source_config_hash"] == EXPECTED_BASE_IDENTITY
    assert bundle.config["design"]["source_config"] == PHASE2_PILOT_BASE_CONFIG
    assert bundle.base_config_path.resolve() == BASE_PATH.resolve()
    assert bundle.source_path.resolve() == PILOT_PATH.resolve()


def test_pilot_is_permanently_ineligible_and_has_only_three_registered_seeds(
    pilot_config: dict[str, Any],
) -> None:
    assert pilot_config["design"] == {
        "name": "common-beta-pilot-v2",
        "stage": "pilot",
        "pilot_phase": "calibration",
        "formal_eligibility": False,
        "evidence_role": "pilot_design_selection_only",
        "pilot_results_permanently_excluded_from_confirmatory": True,
        "source_config": PHASE2_PILOT_BASE_CONFIG,
        "source_config_hash": EXPECTED_BASE_IDENTITY,
        "predecessor_config": "configs/main.yaml",
        "predecessor_config_hash": PHASE1_MAIN_CONFIG_HASH,
        "predecessor_evidence_role": "exploratory_audit_only",
        "estimand": "fixed_beta_downstream_policy_regret",
    }
    assert pilot_config["run"] == {
        "seeds": [20260801, 20260802, 20260803],
        "num_prompts": 2048,
        "split_sizes": {"train": 1536, "validation": 256, "test": 256},
        "confirmatory": False,
        "formal_eligibility": False,
        "excluded_from_confirmatory_evidence": True,
    }
    assert frozenset(pilot_config["run"]["seeds"]) == PHASE2_PILOT_SEEDS
    assert not PHASE2_PILOT_SEEDS & PHASE1_MAIN_SEEDS
    assert PHASE2_CONFIRMATORY_EXCLUDED_SEEDS == (PHASE1_MAIN_SEEDS | PHASE2_PILOT_SEEDS)


def test_old_fake_confirmatory_candidate_files_are_absent() -> None:
    assert not (ROOT / "configs" / "common_beta.yaml").exists()
    assert not (ROOT / "configs" / "common_beta_base.yaml").exists()


def test_pilot_base_remains_readable_by_strict_eight_section_loader() -> None:
    base = load_config(BASE_PATH)

    assert config_hash(base) == EXPECTED_BASE_IDENTITY
    assert set(base) == {
        "run",
        "data",
        "policy",
        "oracle",
        "annotations",
        "objective",
        "reward_model",
        "evaluation",
    }
    assert base["run"]["name"] == "common-beta-pilot-materialization-v2"
    assert base["run"]["seeds"] == [20260801, 20260802, 20260803]
    assert base["policy"]["max_prompt_tokens"] == 1024
    assert base["policy"]["max_response_tokens"] == 256


def test_phase2_freezes_one_policy_tangent_basis_from_an_excluded_phase1_seed(
    pilot_config: dict[str, Any],
) -> None:
    fixed_a = pilot_config["policy"]["fixed_lora_a"]

    assert fixed_a == {
        "mode": "frozen_global",
        "initialization_seed": PHASE2_FIXED_LORA_A_INITIALIZATION_SEED,
        "expected_sha256": PHASE2_FIXED_LORA_A_SHA256,
        "source_seed": PHASE2_FIXED_LORA_A_SOURCE_SEED,
        "source_named_stream": "policy_lora_a",
        "source_config": "configs/main.yaml",
        "source_config_hash": PHASE1_MAIN_CONFIG_HASH,
        "source_artifact_metadata_sha256": (PHASE2_FIXED_LORA_A_SOURCE_METADATA_SHA256),
        "source_seed_excluded_from_phase2": True,
    }
    assert pilot_config["policy"] == load_config(BASE_PATH)["policy"]
    assert "fixed_lora_a" not in load_config(MAIN_PATH)["policy"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("mode", "current_seed_derived", r"fixed_lora_a\.mode"),
        ("initialization_seed", 1, r"fixed_lora_a\.initialization_seed"),
        ("expected_sha256", "0" * 64, r"fixed_lora_a\.expected_sha256"),
        ("source_seed", 20260723, r"fixed_lora_a\.source_seed"),
        ("source_named_stream", "rollout", r"source_named_stream"),
        ("source_config_hash", "0" * 64, r"source_config_hash"),
        (
            "source_artifact_metadata_sha256",
            "0" * 64,
            r"source_artifact_metadata_sha256",
        ),
        ("source_seed_excluded_from_phase2", False, r"must be true"),
    ],
)
def test_phase2_fixed_lora_a_contract_fails_closed(
    pilot_config: dict[str, Any],
    field: str,
    value: object,
    match: str,
) -> None:
    pilot_config["policy"]["fixed_lora_a"][field] = value
    with pytest.raises(ConfigError, match=match):
        validate_phase2_config(pilot_config)


def test_phase2_rejects_missing_fixed_lora_a_contract(
    pilot_config: dict[str, Any],
) -> None:
    del pilot_config["policy"]["fixed_lora_a"]
    with pytest.raises(ConfigError, match=r"policy.*missing keys.*fixed_lora_a"):
        validate_phase2_config(pilot_config)


def test_phase2_freezes_one_global_oracle_transform_from_excluded_phase1_artifacts(
    pilot_config: dict[str, Any],
) -> None:
    calibration = pilot_config["oracle"]["transform_calibration"]

    assert calibration["mode"] == "frozen_global"
    assert calibration["b"] == PHASE2_FROZEN_ORACLE_B
    assert calibration["tau"] == PHASE2_FROZEN_ORACLE_TAU
    assert calibration["aggregation_rule"] == "componentwise_median"
    assert calibration["source_split"] == "train"
    assert calibration["source_config"] == "configs/main.yaml"
    assert calibration["source_config_hash"] == PHASE1_MAIN_CONFIG_HASH
    assert calibration["source_seeds_excluded_from_phase2"] is True
    assert (
        tuple(
            (artifact["seed"], artifact["metadata_sha256"])
            for artifact in calibration["source_artifacts"]
        )
        == PHASE2_FROZEN_ORACLE_SOURCE_ARTIFACTS
    )
    assert pilot_config["oracle"] == load_config(BASE_PATH)["oracle"]

    # Existing Phase-1 semantics remain the per-seed train-only fit.
    assert "transform_calibration" not in load_config(MAIN_PATH)["oracle"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda calibration: calibration.update({"mode": "current_seed_train_fit"}),
            r"transform_calibration\.mode",
        ),
        (
            lambda calibration: calibration.update({"b": -4.0}),
            r"transform_calibration\.b",
        ),
        (
            lambda calibration: calibration.update({"tau": 3.0}),
            r"transform_calibration\.tau",
        ),
        (
            lambda calibration: calibration.update({"aggregation_rule": "mean"}),
            r"aggregation_rule",
        ),
        (
            lambda calibration: calibration.update({"source_config_hash": "0" * 64}),
            r"source_config_hash",
        ),
        (
            lambda calibration: calibration["source_artifacts"].reverse(),
            r"five ordered.*metadata identities",
        ),
        (
            lambda calibration: calibration.update({"silent_typo": True}),
            r"unknown keys.*silent_typo",
        ),
    ],
)
def test_frozen_oracle_transform_contract_fails_closed(
    pilot_config: dict[str, Any],
    mutation: Any,
    match: str,
) -> None:
    mutation(pilot_config["oracle"]["transform_calibration"])
    with pytest.raises(ConfigError, match=match):
        validate_phase2_config(pilot_config)


def test_phase2_rejects_missing_frozen_oracle_transform_contract(
    pilot_config: dict[str, Any],
) -> None:
    del pilot_config["oracle"]["transform_calibration"]
    with pytest.raises(ConfigError, match=r"oracle.*missing keys.*transform_calibration"):
        validate_phase2_config(pilot_config)


def test_phase2_prompt_cap_is_frozen_at_1024(
    pilot_config: dict[str, Any],
) -> None:
    pilot_config["policy"]["max_prompt_tokens"] = 384
    with pytest.raises(ConfigError, match=r"policy\.max_prompt_tokens must equal 1024"):
        validate_phase2_config(pilot_config)


def test_adaptive_convergence_is_frozen_and_720_is_only_compute_checkpoint(
    pilot_config: dict[str, Any],
) -> None:
    reward = pilot_config["reward_model"]
    assert reward["outer_steps"] == 720
    assert reward["adaptive_convergence"] == {
        "relative_gradient_ratio_tolerance": 1.0e-3,
        "minimum_steps": 100,
        "maximum_steps": 5760,
        "check_interval_steps": 20,
        "consecutive_passing_checks": 3,
        "compute_matched_checkpoint_steps": 720,
        "gradient_measurement": "post_update_full_data_unclipped",
        "denominator": "exact_zero_initialization_gradient_l2_norm",
        "denominator_floor": 1.0e-30,
        "prorm_pcg_audit_initialization": "cold_start_zero",
        "fail_closed": True,
        "solution_tie_break": "zero_initialized_adamw_implicit_bias",
        "unique_solution_claim": False,
        "validation_or_test_selection": False,
        "primary_heads_required_to_converge": ["bt_mle", "prorm_plus"],
    }


def test_rank_identifiability_is_train_only_measurement_not_minimum_norm_claim(
    pilot_config: dict[str, Any],
) -> None:
    assert pilot_config["reward_model"]["identifiability"] == {
        "design_matrix": "reward_feature_difference_design_matrix",
        "split": "train",
        "relative_rank_tolerance": 1.0e-10,
        "role": "pilot_measure_only",
        "require_full_column_rank": False,
        "algorithmic_tie_break": "zero_initialized_adamw_implicit_bias",
        "minimum_norm_claim": False,
        "confirmatory_freeze_requirement": (
            "decide_gate_from_train_only_pilot_then_issue_new_identity"
        ),
    }


def test_r4_integrity_is_a_fail_closed_per_seed_contract(
    pilot_config: dict[str, Any],
) -> None:
    annotations = pilot_config["annotations"]
    assert annotations["independent_replicates_per_edge"] == 4
    assert annotations["replicate_reduction"] == "arithmetic_mean"
    assert annotations["prohibit_clipping"] is True
    assert annotations["bt_label_use"] == "all_underlying_bernoulli_labels"
    assert annotations["replicate_rng"] == (
        "single_named_generator_sequential_independent_draws_with_preserved_boundaries"
    )
    assert annotations["decision_gates"]["action"] == "fail_closed"
    assert annotations["decision_gates"]["require_all"] == [
        "exactly_four_replicate_boundaries",
        "single_generator_initial_final_state_and_draw_count",
        "replicate_tensor_hashes_preserve_boundaries",
        "no_label_clipping",
        "bt_uses_all_raw_bernoulli_labels",
        "prorm_uses_arithmetic_mean_of_four_unclipped_estimators",
    ]


def test_controls_tolerances_and_execution_roles_are_design_bound(
    pilot_config: dict[str, Any],
) -> None:
    controls = pilot_config["positive_controls"]
    assert controls["direct_oracle_geometry"]["enabled"] is True
    assert controls["exact_margin"]["enabled"] is True
    assert controls["exact_soft_label_bt"] == {
        "enabled": True,
        "role": "noise_free_positive_control_and_secondary_misspecification_diagnostic",
        "noise_free": True,
        "input": "sigmoid_of_train_transformed_oracle_margin",
        "eligible_for_primary_claim": False,
    }
    assert controls["low_dimensional_tangent"]["dimension"] == 256
    assert controls["numeric_gate_tolerances"] == {
        "direct_identity_absolute_error": 1.0e-10,
        "direct_identity_relative_error": 1.0e-10,
        "objective_binding_relative_error": 2.0e-5,
        "objective_binding_absolute_error": 2.0e-7,
        "outer_relative_gradient_ratio": 1.0e-3,
        "low_dimensional_orthonormality_max_absolute_error": 1.0e-10,
        "low_dimensional_pseudoinverse_relative_residual": 1.0e-6,
        "low_dimensional_scatter_max_absolute_error": 1.0e-4,
        "low_dimensional_score_identity_max_absolute_error": 1.0e-4,
    }
    assert controls["decision_gates"]["action"] == "fail_closed"
    assert controls["decision_gates"]["unit"] == "per_seed"

    common = pilot_config["objective"]["common_beta"]
    ridge = pilot_config["objective"]["full_tangent"]["ridge"]
    assert common["primary_execution_role"] == "pilot_global_beta_calibration_candidate"
    assert common["sensitivity_execution_role"] == (
        "required_separate_global_beta_candidate_sensitivity"
    )
    assert common["rule"] == (
        "pilot_seed_candidate_from_oracle_train_fisher_quadratic_for_future_global_beta"
    )
    assert common["frozen_global_beta"] is None
    assert common["sensitivity_k_cal"] == [0.001, 0.01]
    assert common["sensitivity_frozen_global_beta_multipliers"] is None
    assert ridge["primary_execution_role"] == "pilot_candidate_primary"
    assert ridge["sensitivity_execution_role"] == "required_separate_pilot_sensitivity"
    for section in (common, ridge):
        assert section["sensitivity_executed_separately"] is True
        assert section["sensitivity_eligible_for_primary_claim"] is False
    all_six = pilot_config["secondary_experiments"]["all_six_pairs"]
    assert all_six["execution_role"] == "separate_secondary_efficiency_experiment"
    assert all_six["executed_in_primary_four_arm_run"] is False
    assert all_six["eligible_for_primary_claim"] is False


def test_test_endpoints_seed_interval_and_effect_gates_are_explicit(
    pilot_config: dict[str, Any],
) -> None:
    evaluation = pilot_config["evaluation"]
    assert evaluation["experimental_unit"] == "seed"
    assert evaluation["endpoints"] == {
        "heldout_local_regret": {
            "split": "test",
            "metric": "local_regret_at_frozen_global_beta",
            "contrast": "bt_mle_minus_prorm_plus",
            "direction": "higher_is_better",
        },
        "finite_policy_utility": {
            "split": "test",
            "metric": "operational_oracle_reward_minus_beta_common_on_policy_kl",
            "contrast": "prorm_plus_minus_bt_mle",
            "direction": "higher_is_better",
        },
    }
    assert evaluation["seed_level_interval"] == {
        "method": "paired_seed_percentile_bootstrap",
        "confidence_level": 0.95,
        "interval_sidedness": "two_sided",
        "effective_component_one_sided_alpha": 0.025,
        "lower_bound_rule": "strictly_greater_than_zero",
        "prompt_rows_used_as_independent_replicates": False,
        "estimand": (
            "rng_expectation_of_paired_contrast_conditioned_on_frozen_prompt_pool_"
            "models_oracle_and_design"
        ),
        "test_structure": "intersection_union_single_conjunctive_claim",
        "component_null": "mean_paired_contrast_lte_zero",
        "component_alternative": "mean_paired_contrast_gt_zero",
        "multiplicity_adjustment": "none_for_intersection_union_conjunctive_claim",
    }
    required = evaluation["decision_gates"]["require_all"]
    assert "heldout_bt_minus_prorm_plus_interval_lower_positive" in required
    assert "finite_prorm_plus_minus_bt_mle_interval_lower_positive" in required
    assert "finite_prorm_plus_minus_zero_b_interval_lower_positive" in required
    assert "finite_oracle_step_minus_zero_b_interval_lower_positive" in required
    assert evaluation["decision_gates"]["supports_formal_claim"] is False


def test_256_token_horizon_is_pilot_selection_input_only(
    pilot_config: dict[str, Any],
) -> None:
    assert pilot_config["policy"]["max_response_tokens"] == 256
    assert pilot_config["evaluation"]["max_length"] == {
        "candidate_horizon_tokens": 256,
        "role": "pilot_horizon_selection_input",
        "measure_only": True,
        "formal_gate": False,
        "formal_threshold": 0.05,
        "allowed_horizon_sequence": [256, 512, 1024],
        "horizon_grid_index": 0,
        "parent_pilot_aggregate_sha256": None,
        "previous_horizon_failed_length_gate": False,
        "post_pilot_requirement": "issue_new_pilot_freeze_design_identity",
    }


def test_failed_length_gate_can_only_escalate_horizon_in_a_new_bound_identity(
    pilot_config: dict[str, Any],
) -> None:
    escalated = copy.deepcopy(pilot_config)
    escalated["design"]["name"] = "common-beta-pilot-calibration-h512"
    escalated["policy"]["max_response_tokens"] = 512
    escalated["evaluation"]["max_length"].update(
        {
            "candidate_horizon_tokens": 512,
            "horizon_grid_index": 1,
            "parent_pilot_aggregate_sha256": "9" * 64,
            "previous_horizon_failed_length_gate": True,
        }
    )

    validated = validate_phase2_config(escalated)
    assert validated["policy"]["max_response_tokens"] == 512
    assert validated["evaluation"]["max_length"]["horizon_grid_index"] == 1
    assert phase2_design_identity(escalated) != phase2_design_identity(pilot_config)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda config: config["evaluation"]["max_length"].update(
                {"parent_pilot_aggregate_sha256": None}
            ),
            "parent_pilot_aggregate_sha256",
        ),
        (
            lambda config: config["evaluation"]["max_length"].update(
                {"previous_horizon_failed_length_gate": False}
            ),
            "requires previous_horizon_failed_length_gate=true",
        ),
        (
            lambda config: config["evaluation"]["max_length"].update({"horizon_grid_index": 0}),
            "allowed_horizon_sequence",
        ),
        (
            lambda config: config["evaluation"]["max_length"].update(
                {"allowed_horizon_sequence": [256, 1024]}
            ),
            "allowed_horizon_sequence",
        ),
    ],
)
def test_horizon_escalation_contract_fails_closed(
    pilot_config: dict[str, Any],
    mutation: Any,
    match: str,
) -> None:
    escalated = copy.deepcopy(pilot_config)
    escalated["design"]["name"] = "common-beta-pilot-calibration-h512"
    escalated["policy"]["max_response_tokens"] = 512
    escalated["evaluation"]["max_length"].update(
        {
            "candidate_horizon_tokens": 512,
            "horizon_grid_index": 1,
            "parent_pilot_aggregate_sha256": "9" * 64,
            "previous_horizon_failed_length_gate": True,
        }
    )
    mutation(escalated)
    with pytest.raises(ConfigError, match=match):
        validate_phase2_config(escalated)


def test_schema_accepts_a_future_confirmatory_stage_only_with_new_base_identity(
    pilot_config: dict[str, Any],
) -> None:
    overlay, base = _future_confirmatory(pilot_config)

    validated = validate_phase2_config(overlay, base_config=base)
    assert validated["design"]["stage"] == "confirmatory"
    assert validated["run"]["confirmatory"] is True
    assert validated["objective"]["common_beta"]["frozen_global_beta"] == 2.5
    assert validated["objective"]["common_beta"]["calibration_split"] == "excluded_pilot"
    assert validated["objective"]["common_beta"]["calibration_source"] == (
        "frozen_pilot_global_beta_in_confirmatory_design_identity"
    )
    assert validated["objective"]["common_beta"]["sensitivity_k_cal"] is None
    assert validated["objective"]["common_beta"]["sensitivity_frozen_global_beta_multipliers"] == [
        0.5,
        2.0,
    ]
    assert tuple(validated["run"]["seeds"]) == PHASE2_CONFIRMATORY_SEEDS
    assert len(validated["run"]["seeds"]) == PHASE2_CONFIRMATORY_NUM_SEEDS
    assert not set(validated["run"]["seeds"]) & PHASE2_CONFIRMATORY_EXCLUDED_SEEDS
    assert validated["design"]["source_config_hash"] == config_hash(base)


def test_frozen_global_beta_is_part_of_the_confirmatory_design_identity(
    pilot_config: dict[str, Any],
) -> None:
    first, _ = _future_confirmatory(pilot_config)
    second = copy.deepcopy(first)
    second["objective"]["common_beta"]["frozen_global_beta"] = 2.75

    assert phase2_design_identity(first) != phase2_design_identity(second)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda config: config["run"].update({"seeds": list(PHASE2_CONFIRMATORY_SEEDS[:-1])}),
            "exact preregistered ordered",
        ),
        (
            lambda config: config["run"].update(
                {
                    "seeds": [20260722] + list(PHASE2_CONFIRMATORY_SEEDS[1:]),
                }
            ),
            "exact preregistered ordered",
        ),
        (
            lambda config: config["run"].update(
                {
                    "seeds": [20260801] + list(PHASE2_CONFIRMATORY_SEEDS[1:]),
                }
            ),
            "exact preregistered ordered",
        ),
        (
            lambda config: config["run"].update(
                {"seeds": list(reversed(PHASE2_CONFIRMATORY_SEEDS))}
            ),
            "exact preregistered ordered",
        ),
        (
            lambda config: config["run"].update({"confirmatory": False}),
            "confirmatory runs require",
        ),
        (
            lambda config: config["design"].update({"formal_eligibility": False}),
            "confirmatory design.formal_eligibility",
        ),
    ],
)
def test_future_confirmatory_seed_and_eligibility_contract_fails_closed(
    pilot_config: dict[str, Any],
    mutation: Any,
    match: str,
) -> None:
    overlay, _ = _future_confirmatory(pilot_config)
    mutation(overlay)
    with pytest.raises(ConfigError, match=match):
        validate_phase2_config(overlay)


@pytest.mark.parametrize("value", [None, 0.0, -1.0, float("inf"), float("nan")])
def test_future_confirmatory_requires_finite_positive_frozen_global_beta(
    pilot_config: dict[str, Any],
    value: object,
) -> None:
    overlay, _ = _future_confirmatory(pilot_config)
    overlay["objective"]["common_beta"]["frozen_global_beta"] = value
    with pytest.raises(ConfigError, match="frozen_global_beta"):
        validate_phase2_config(overlay)


def test_pilot_forbids_a_prefilled_frozen_global_beta(
    pilot_config: dict[str, Any],
) -> None:
    pilot_config["objective"]["common_beta"]["frozen_global_beta"] = 2.5
    with pytest.raises(ConfigError, match="must be null"):
        validate_phase2_config(pilot_config)


def test_pilot_freeze_binds_one_candidate_from_the_calibration_aggregate(
    pilot_config: dict[str, Any],
) -> None:
    overlay = _pilot_freeze(pilot_config)
    validated = validate_phase2_config(overlay)

    assert validated["design"]["pilot_phase"] == "freeze"
    common = validated["objective"]["common_beta"]
    assert common["frozen_global_beta"] == 2.5
    assert common["beta_source_aggregate_sha256"] == "b" * 64
    assert common["sensitivity_k_cal"] is None
    assert common["sensitivity_frozen_global_beta_multipliers"] is None
    assert validated["evaluation"]["max_length"]["measure_only"] is True
    assert validated["evaluation"]["max_length"]["formal_threshold"] == 0.05
    assert validated["design"]["formal_eligibility"] is False


def test_pilot_freeze_beta_and_source_aggregate_are_identity_bound(
    pilot_config: dict[str, Any],
) -> None:
    first = _pilot_freeze(pilot_config)
    second = _pilot_freeze(pilot_config, beta=5.0)
    third = _pilot_freeze(pilot_config, calibration_aggregate_sha256="c" * 64)

    assert phase2_design_identity(first) != phase2_design_identity(second)
    assert phase2_design_identity(first) != phase2_design_identity(third)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("objective.common_beta.frozen_global_beta", None, "frozen_global_beta"),
        ("objective.common_beta.frozen_global_beta", 0.0, "frozen_global_beta"),
        (
            "objective.common_beta.beta_source_aggregate_sha256",
            None,
            "beta_source_aggregate_sha256",
        ),
        (
            "objective.common_beta.sensitivity_k_cal",
            [0.001, 0.01],
            "sensitivity_k_cal must be null",
        ),
        (
            "objective.common_beta.sensitivity_frozen_global_beta_multipliers",
            [2.0],
            "multipliers must be null",
        ),
        ("evaluation.max_length.formal_threshold", None, "must be a real number"),
    ],
)
def test_pilot_freeze_contract_fails_closed(
    pilot_config: dict[str, Any],
    path: str,
    value: object,
    match: str,
) -> None:
    overlay = _pilot_freeze(pilot_config)
    _set_path(overlay, path, value)
    with pytest.raises(ConfigError, match=match):
        validate_phase2_config(overlay)


def test_confirmatory_sensitivity_can_only_multiply_the_frozen_global_beta(
    pilot_config: dict[str, Any],
) -> None:
    overlay, _ = _future_confirmatory(pilot_config)
    overlay["objective"]["common_beta"]["sensitivity_k_cal"] = [0.001, 0.01]
    with pytest.raises(ConfigError, match="sensitivity_k_cal must be null"):
        validate_phase2_config(overlay)

    overlay, _ = _future_confirmatory(pilot_config)
    overlay["objective"]["common_beta"]["sensitivity_frozen_global_beta_multipliers"] = [
        1.0,
        2.0,
        4.0,
    ]
    with pytest.raises(ConfigError, match="must equal"):
        validate_phase2_config(overlay)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("schema_version", "prorm-common-beta-config/v1", "schema_version"),
        ("design.stage", "confirmatory", "confirmatory design.pilot_phase"),
        ("design.formal_eligibility", True, "pilot design.formal_eligibility"),
        (
            "design.pilot_results_permanently_excluded_from_confirmatory",
            False,
            "must be true",
        ),
        ("design.predecessor_config_hash", "0" * 64, "predecessor_config_hash"),
        ("run.seeds", [20260801, 20260802], "exactly the permanently excluded"),
        ("run.confirmatory", True, "pilot runs require"),
        ("run.formal_eligibility", True, "pilot runs require"),
        ("run.excluded_from_confirmatory_evidence", False, "pilot runs require"),
        ("policy.max_response_tokens", 128, "candidate_horizon_tokens"),
        ("annotations.independent_replicates_per_edge", 1, "must equal 4"),
        ("annotations.prohibit_clipping", False, "must be true"),
        (
            "annotations.replicate_rng",
            "one_shared_stream",
            "replicate_rng",
        ),
        (
            "reward_model.adaptive_convergence.relative_gradient_ratio_tolerance",
            0.01,
            "relative_gradient_ratio_tolerance",
        ),
        (
            "reward_model.adaptive_convergence.maximum_steps",
            720,
            "maximum_steps",
        ),
        (
            "reward_model.adaptive_convergence.compute_matched_checkpoint_steps",
            719,
            "compute_matched_checkpoint_steps",
        ),
        (
            "reward_model.adaptive_convergence.gradient_measurement",
            "pre_update_minibatch_clipped",
            "gradient_measurement",
        ),
        (
            "reward_model.adaptive_convergence.prorm_pcg_audit_initialization",
            "warm_start",
            "pcg_audit_initialization",
        ),
        ("reward_model.adaptive_convergence.fail_closed", False, "must be true"),
        (
            "reward_model.adaptive_convergence.unique_solution_claim",
            True,
            "must be false",
        ),
        (
            "reward_model.identifiability.require_full_column_rank",
            True,
            "measure-only",
        ),
        ("reward_model.identifiability.minimum_norm_claim", True, "must be false"),
        (
            "objective.common_beta.sensitivity_executed_separately",
            False,
            "must be true",
        ),
        (
            "objective.full_tangent.ridge.sensitivity_eligible_for_primary_claim",
            True,
            "must be false",
        ),
        (
            "positive_controls.numeric_gate_tolerances.direct_identity_absolute_error",
            1.0e-5,
            "direct_identity_absolute_error",
        ),
        (
            "positive_controls.exact_soft_label_bt.enabled",
            False,
            "must be true",
        ),
        (
            "positive_controls.exact_soft_label_bt.noise_free",
            False,
            "must be true",
        ),
        (
            "positive_controls.exact_soft_label_bt.input",
            "sampled_bernoulli_labels",
            "exact_soft_label_bt.input",
        ),
        (
            "positive_controls.exact_soft_label_bt.eligible_for_primary_claim",
            True,
            "must be false",
        ),
        (
            "evaluation.endpoints.heldout_local_regret.split",
            "validation",
            "heldout_local_regret.split",
        ),
        ("evaluation.experimental_unit", "prompt", "must equal 'seed'"),
        (
            "evaluation.seed_level_interval.prompt_rows_used_as_independent_replicates",
            True,
            "must be false",
        ),
        ("evaluation.max_length.measure_only", False, "must be true"),
        (
            "secondary_experiments.all_six_pairs.executed_in_primary_four_arm_run",
            True,
            "must be false",
        ),
    ],
)
def test_locked_pilot_contract_rejects_scientific_tampering(
    pilot_config: dict[str, Any],
    path: str,
    value: object,
    match: str,
) -> None:
    _set_path(pilot_config, path, value)
    with pytest.raises(ConfigError, match=match):
        validate_phase2_config(pilot_config)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "design",
        "run",
        "annotations",
        "annotations.decision_gates",
        "reward_model",
        "reward_model.adaptive_convergence",
        "reward_model.identifiability",
        "objective.common_beta",
        "objective.full_tangent.ridge",
        "positive_controls.numeric_gate_tolerances",
        "positive_controls.exact_soft_label_bt",
        "evaluation.endpoints.heldout_local_regret",
        "evaluation.seed_level_interval",
        "evaluation.max_length",
        "secondary_experiments.all_six_pairs",
    ],
)
def test_unknown_fields_are_rejected_recursively(
    pilot_config: dict[str, Any],
    path: str,
) -> None:
    target: dict[str, Any] = pilot_config
    if path:
        for component in path.split("."):
            target = target[component]
    target["silent_typo"] = True
    with pytest.raises(ConfigError, match="unknown keys"):
        validate_phase2_config(pilot_config)


def test_dynamic_source_hash_rejects_tampered_base(
    pilot_config: dict[str, Any],
) -> None:
    tampered_base = load_config(BASE_PATH)
    tampered_base["run"]["name"] = "tampered"

    with pytest.raises(ConfigError, match="design.source_config_hash"):
        validate_phase2_config(pilot_config, base_config=tampered_base)

    tampered_overlay = copy.deepcopy(pilot_config)
    tampered_overlay["design"]["source_config_hash"] = "0" * 64
    with pytest.raises(ConfigError, match="design.source_config_hash"):
        validate_phase2_config(tampered_overlay, base_config=load_config(BASE_PATH))


def test_declared_source_path_drives_default_base_resolution(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    configs = repository / "configs"
    configs.mkdir(parents=True)
    base = configs / "renamed_dynamic_base.yaml"
    base.write_bytes(BASE_PATH.read_bytes())
    overlay = load_phase2_config(PILOT_PATH)
    overlay["design"]["source_config"] = "configs/renamed_dynamic_base.yaml"
    yaml = pytest.importorskip("yaml")
    overlay_path = configs / "renamed_overlay.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")

    bundle = load_phase2_config_bundle(overlay_path)
    assert bundle.base_config_path.resolve() == base.resolve()
    assert bundle.config["design"]["source_config"] == ("configs/renamed_dynamic_base.yaml")


def test_explicit_base_must_resolve_to_exact_declared_path() -> None:
    with pytest.raises(ConfigError, match="exact overlay-declared"):
        load_phase2_config(PILOT_PATH, base_config_path=MAIN_PATH)


def test_declared_source_path_rejects_absolute_parent_and_backslash(
    pilot_config: dict[str, Any],
) -> None:
    for invalid in (
        "/tmp/base.yaml",
        "../configs/base.yaml",
        "configs/../base.yaml",
        r"configs\base.yaml",
        "other/base.yaml",
    ):
        mutated = copy.deepcopy(pilot_config)
        mutated["design"]["source_config"] = invalid
        with pytest.raises(ConfigError, match="normalized relative POSIX"):
            validate_phase2_config(mutated)


def test_duplicate_yaml_keys_and_nonfinite_values_are_rejected(tmp_path: Path) -> None:
    text = PILOT_PATH.read_text(encoding="utf-8")
    duplicate = text.replace(
        "    primary_k_cal: 0.003",
        "    primary_k_cal: 0.003\n    primary_k_cal: 0.003",
    )
    duplicate_path = tmp_path / "common_beta_pilot.yaml"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate YAML key 'primary_k_cal'"):
        load_phase2_config(duplicate_path, base_config_path=BASE_PATH)

    nonfinite = text.replace(
        "    mean_policy_to_reference_kl_cap: 0.02",
        "    mean_policy_to_reference_kl_cap: .nan",
    )
    nonfinite_path = tmp_path / "common_beta_pilot.yaml"
    nonfinite_path.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(ConfigError, match="must be finite"):
        load_phase2_config(nonfinite_path, base_config_path=BASE_PATH)


def test_validation_and_loading_return_independent_mappings(
    pilot_config: dict[str, Any],
) -> None:
    original = copy.deepcopy(pilot_config)
    validated = validate_phase2_config(pilot_config)
    validated["run"]["seeds"][0] = 1
    assert pilot_config == original

    bundle = load_phase2_config_bundle(PILOT_PATH)
    bundle.config["run"]["seeds"][0] = 1
    assert bundle.base_config["run"]["seeds"][0] == 20260801
    assert load_phase2_config(PILOT_PATH)["run"]["seeds"][0] == 20260801


def test_expected_pilot_file_sha_is_exact() -> None:
    identities = {
        "overlay": hashlib.sha256(PILOT_PATH.read_bytes()).hexdigest(),
        "base": hashlib.sha256(BASE_PATH.read_bytes()).hexdigest(),
    }
    assert identities == {
        "overlay": "b855883b744ed87c998e8771fe8c4f736ed132c97977ddcf672c5eeed143fb29",
        "base": "e32cf5ad2a7bb2f6fa27180aa2fa6e05e2b457cfe032bab1c33f86646af1beb1",
    }
