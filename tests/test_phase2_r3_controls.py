from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import yaml

from smart_reward.phase2_r3_controls import (
    R3_CONTROL_FAMILY_RESULT_SCHEMA,
    R3_CONTROL_FIRST_ORDER_GATE_SCHEMA,
    R3_CONTROLS_CONFIG_SCHEMA,
    R3_GATE_C_AUDIT_INTERVAL,
    R3_GATE_C_FAMILIES,
    R3_GATE_C_MAX_UPDATES,
    R3_GATE_C_MIN_UPDATES,
    R3_GATE_C_PROFILE_UPDATES,
    R3_GATE_C_SEEDS,
    R3ControlResultError,
    R3ControlsConfigError,
    adapt_exact_margin_prorm_plus_result,
    adapt_exact_soft_label_bt_result,
    adapt_low_dimensional_prorm_plus_result,
    load_r3_controls_config,
    validate_r3_control_family_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase2_recovery_r3_controls.yaml"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rehash(result: dict[str, object]) -> dict[str, object]:
    unsigned = copy.deepcopy(result)
    unsigned.pop("result_sha256", None)
    unsigned["result_sha256"] = _canonical_sha(unsigned)
    return unsigned


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


@pytest.fixture(scope="module")
def controls_config():
    return load_r3_controls_config(CONFIG)


def _dimensions() -> dict[str, int]:
    # The tiny evidence fixture contains no tensors.  These dimensions exercise
    # the formal d=256 strict-subspace and d<n_F validators on CPU.
    return {
        "num_train_prompts": 129,
        "num_candidates": 2,
        "policy_dimension": 257,
        "reward_dimension": 2,
    }


def _first_order(objective: str) -> dict[str, object]:
    initial_gradient = 2.0
    denominator = 2.0
    check_gradients = (1.8e-3, 1.6e-3, 1.4e-3)
    return {
        "schema_version": R3_CONTROL_FIRST_ORDER_GATE_SCHEMA,
        "objective": objective,
        "learning_rate_schedule_sha256": (
            "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
        ),
        "initial_full_data_unclipped_gradient_l2_norm": initial_gradient,
        "gradient_norm_denominator": denominator,
        "final_full_data_unclipped_gradient_l2_norm": 1.2e-3,
        "gradient_ratio_to_zero_initialization": 1.2e-3 / denominator,
        "selected_step": 140,
        "consecutive_passing_checks": 3,
        "sustained_checks": [
            {
                "step": step,
                "gradient_l2_norm": gradient,
                "gradient_ratio_to_zero_initialization": gradient / denominator,
                "threshold_passed": True,
            }
            for step, gradient in zip((100, 120, 140), check_gradients, strict=True)
        ],
        "full_data_post_update_unclipped": True,
        "fresh_zero_initialized": True,
        "fresh_post_restore_audit": True,
        "test_or_validation_data_accessed": False,
        "passed": True,
    }


def _head(objective: str, method: str) -> dict[str, object]:
    return {
        "schema_version": "phase2-r3-control-head-audit/v1",
        "method": method,
        "objective": objective,
        "initial_head_sha256": _sha(f"{objective}:zero"),
        "head_sha256": _sha(f"{objective}:head"),
        "initial_objective": 1.0,
        "final_objective": 0.5,
        "cold_full_data_audit_objective": 0.5,
        "cold_full_data_audit_gradient_l2_norm": 1.2e-3,
        "objective_decrease_passed": True,
        "objective_binding_passed": True,
        "first_order_gate": _first_order(objective),
        "fresh_zero_initialized": True,
        "raw_head_weight_retained": False,
    }


def _pcg(label: str) -> dict[str, object]:
    return {
        "schema_version": "phase2-r3-control-pcg/v1",
        "iterations": 7,
        "residual_norm": 2.0e-7,
        "relative_residual": 1.0e-6,
        "converged": True,
        "cold_start": True,
        "warm_start_used": False,
    }


def _target_common(schema: str) -> dict[str, object]:
    dimensions = _dimensions()
    return {
        "schema_version": schema,
        "split": "train",
        "orientation": "candidate_0_minus_candidate_1",
        "source_node_rewards_sha256": _sha("oracle"),
        "canonical_margin_sha256": _sha("margin"),
        "reward_feature_difference_sha256": _sha("feature-difference"),
        "num_train_prompts": dimensions["num_train_prompts"],
        "num_candidates": dimensions["num_candidates"],
        "reward_dimension": dimensions["reward_dimension"],
        "raw_node_rewards_retained": False,
    }


def _exact_margin_evidence() -> dict[str, object]:
    target = {
        **_target_common("phase2-r3-exact-margin-target/v1"),
        "target": "transformed_oracle_reward_difference",
        "sampled_label_stream_accessed": False,
    }
    return {
        "schema_version": "phase2-r3-exact-margin-prorm-plus-evidence/v1",
        "target_audit": target,
        "head_audit": _head("exact_margin_prorm_plus", "prorm_plus"),
        "pcg_audits": {
            "selected_head_final_inner": _pcg("selected"),
            "cold_saved_head_audit": _pcg("audit"),
            "trained_direction": _pcg("trained-direction"),
        },
        "direct_identity": {
            "schema_version": "direct-oracle-exact-moment-identity/v1",
            "interpretation": "algebraic_identity_bypasses_reward_class_and_optimizer",
            "source_node_rewards_sha256": _sha("oracle"),
            "canonical_margin_sha256": _sha("margin"),
            "canonical_pair_moment_sha256": _sha("canonical-moment"),
            "complete_pair_u_stat_moment_sha256": _sha("u-stat-moment"),
            "all_node_covariance_moment_sha256": _sha("node-moment"),
            "complete_pair_identity_absolute_error": 1.0e-12,
            "complete_pair_identity_relative_error": 2.0e-12,
            "complete_pair_identity_is_algebraic": True,
            "reward_head_bypassed": True,
            "optimizer_bypassed": True,
            "trained_exact_margin_head_required_to_match": False,
            "native_oracle_direction_sha256": _sha("oracle-direction"),
            "native_oracle_direction_pcg": _pcg("oracle-direction"),
            "raw_node_rewards_retained": False,
        },
        "gates": {
            "exact_margin_objective_decrease": True,
            "exact_margin_first_order_convergence": True,
            "direct_oracle_moment_identity": True,
            "all_required_pcg_solves_converged": True,
        },
    }


def _exact_soft_evidence() -> dict[str, object]:
    target = {
        **_target_common("phase2-r3-exact-soft-label-bt-target/v1"),
        "target": "sigmoid_of_train_transformed_oracle_margin",
        "target_probability_sha256": _sha("probability"),
        "noise_free": True,
        "bernoulli_sampling_used": False,
        "sampled_label_stream_accessed": False,
        "raw_target_probabilities_retained": False,
        "raw_oracle_margins_retained": False,
    }
    return {
        "schema_version": "phase2-r3-exact-soft-label-bt-evidence/v1",
        "target_audit": target,
        "head_audit": _head("exact_soft_label_bt_cross_entropy", "bt_mle"),
        "gates": {
            "exact_soft_label_objective_decrease": True,
            "exact_soft_label_first_order_convergence": True,
            "saved_head_objective_binding": True,
        },
    }


def _low_dimensional_evidence() -> dict[str, object]:
    target = {
        **_target_common("phase2-r3-low-dimensional-r4-target/v1"),
        "target": "family_local_r4_mean_h_regenerated_from_train_oracle",
        "family_local_label_stream_sha256": _sha("family-local-label-stream"),
        "annotation_scheme": "geometric_randomized_truncation",
        "annotation_gamma": 0.9,
        "independent_replicates_per_edge": 4,
        "replicate_reduction": "arithmetic_mean",
        "label_rng_namespace": "prorm-common-beta-r4-labels-v1",
        "primary_label_stream_accessed": False,
        "raw_labels_retained": False,
    }
    selected_direction = _sha("low-direction")
    scattered_direction = _sha("scattered-direction")
    return {
        "schema_version": "phase2-r3-low-dimensional-prorm-plus-evidence/v1",
        "target_audit": target,
        "projection": {
            "schema_version": "seeded-orthonormal-tangent/v1",
            "algorithm": "gaussian_qr_sign_canonical_v1",
            "namespace": "prorm-common-beta-low-dimensional-tangent-v1",
            "source_layout_id": "training-policy-score-flatten-order/v1",
            "declared_seed": 20260801,
            "source_dimension": 257,
            "selected_dimension": 256,
            "num_fisher_nodes": 258,
            "projection_sha256": _sha("projection"),
            "projection_dtype": "torch.float64",
            "orthonormality_max_absolute_error": 5.0e-12,
            "orthonormality_absolute_tolerance": 1.0e-10,
            "orthonormality_passed": True,
        },
        "geometry": {
            "schema_version": "phase2-r3-low-dimensional-pseudoinverse-geometry/v1",
            "regularization": "moore_penrose_pseudoinverse",
            "ridge_enabled": False,
            "solver": "torch.linalg.eigh_truncated_moore_penrose",
            "solver_dtype": "float64",
            "selected_dimension": 256,
            "numerical_rank": 256,
            "relative_eigenvalue_tolerance": 1.0e-10,
            "fisher_sha256": _sha("low-fisher"),
            "pseudoinverse_sha256": _sha("low-pseudoinverse"),
            "pseudoinverse_solve_relative_residual": 8.0e-7,
            "pseudoinverse_relative_residual_tolerance": 1.0e-6,
            "exact_rank_passed": True,
            "pseudoinverse_residual_passed": True,
        },
        "head_audit": _head("low_dimensional_prorm_plus", "prorm_plus"),
        "scatter_identity": {
            "schema_version": "phase2-r3-low-dimensional-scatter-identity/v1",
            "formula": "u_full = P @ u_low",
            "selected_direction_sha256": selected_direction,
            "scattered_full_direction_sha256": scattered_direction,
            "reference_scattered_full_direction_sha256": _sha("reference-scatter"),
            "max_absolute_error": 7.0e-5,
            "l2_error": 8.0e-5,
            "absolute_tolerance": 1.0e-4,
            "passed": True,
        },
        "score_identity": {
            "schema_version": "phase2-r3-low-dimensional-score-identity/v1",
            "formula": "(S_full @ P) @ u_low == S_full @ (P @ u_low)",
            "selected_direction_sha256": selected_direction,
            "scattered_full_direction_sha256": scattered_direction,
            "low_projected_score_sha256": _sha("low-score"),
            "full_projected_score_sha256": _sha("full-score"),
            "max_absolute_error": 9.0e-5,
            "l2_error": 1.1e-4,
            "absolute_tolerance": 1.0e-4,
            "passed": True,
        },
        "gates": {
            "low_dimensional_objective_decrease": True,
            "low_dimensional_first_order_convergence": True,
            "low_dimensional_exact_rank": True,
            "low_dimensional_orthonormality": True,
            "low_dimensional_pseudoinverse_residual": True,
            "low_dimensional_scatter_identity": True,
            "low_dimensional_score_identity": True,
        },
    }


def _adapt(family: str, controls_config, *, seed: int = 20260801) -> dict[str, object]:
    evidence = {
        "exact_margin_prorm_plus": _exact_margin_evidence,
        "exact_soft_label_bt": _exact_soft_evidence,
        "low_dimensional_prorm_plus": _low_dimensional_evidence,
    }[family]()
    if family == "low_dimensional_prorm_plus":
        evidence["projection"]["declared_seed"] = seed
    adapter = {
        "exact_margin_prorm_plus": adapt_exact_margin_prorm_plus_result,
        "exact_soft_label_bt": adapt_exact_soft_label_bt_result,
        "low_dimensional_prorm_plus": adapt_low_dimensional_prorm_plus_result,
    }[family]
    return adapter(
        seed=seed,
        config=controls_config,
        input_training_sha256=_sha("training"),
        train_oracle_rewards_sha256=_sha("oracle"),
        input_dimensions=_dimensions(),
        family_evidence=evidence,
    )


def test_frozen_controls_config_is_exact_three_by_three_and_reuses_existing_gates(
    controls_config,
) -> None:
    assert controls_config.normalized["schema_version"] == R3_CONTROLS_CONFIG_SCHEMA
    assert controls_config.seeds == R3_GATE_C_SEEDS
    assert controls_config.families == R3_GATE_C_FAMILIES
    assert len(controls_config.seeds) == len(controls_config.families) == 3
    assert controls_config.profile_updates == R3_GATE_C_PROFILE_UPDATES == 100
    assert controls_config.minimum_updates == R3_GATE_C_MIN_UPDATES == 100
    assert controls_config.maximum_updates == R3_GATE_C_MAX_UPDATES == 12760
    assert controls_config.audit_interval_updates == R3_GATE_C_AUDIT_INTERVAL == 20

    normalized = controls_config.normalized
    assert normalized["first_order_gate"]["relative_gradient_ratio_tolerance"] == 1.0e-3
    assert normalized["numeric_gate_tolerances"] == {
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
    assert normalized["execution_boundary"]["calibration_beta_access_allowed"] is False
    assert normalized["profiling"]["reusable_as_formal_control_result"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("run", "seeds"), [20260801, 20260802]),
        (
            ("numeric_gate_tolerances", "low_dimensional_scatter_max_absolute_error"),
            2.0e-4,
        ),
        (("execution_boundary", "policy_rollout_allowed"), True),
        (("profiling", "reusable_as_formal_control_result"), True),
    ],
)
def test_controls_config_rejects_science_or_boundary_tampering(
    tmp_path: Path,
    path: tuple[str, str],
    replacement: object,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = replacement
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(R3ControlsConfigError, match="frozen contract"):
        load_r3_controls_config(changed)


def test_loaded_config_detects_later_byte_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "controls.yaml"
    copied.write_bytes(CONFIG.read_bytes())
    bundle = load_r3_controls_config(copied)
    copied.write_bytes(copied.read_bytes() + b"\n")
    with pytest.raises(R3ControlsConfigError, match="bytes changed"):
        bundle.validate_integrity()


def test_all_three_family_adapters_emit_head_free_self_hashed_results(
    controls_config,
) -> None:
    for family in R3_GATE_C_FAMILIES:
        result = _adapt(family, controls_config)
        assert result["schema_version"] == R3_CONTROL_FAMILY_RESULT_SCHEMA
        assert result["family"] == family
        assert result["seed"] == 20260801
        assert result["completion"] == {
            "status": "completed",
            "completed_updates": 140,
            "stop_reason": "sustained_first_order_gate",
            "formal_family_result": True,
            "profile_only": False,
            "head_or_optimizer_state_retained": False,
        }
        json.dumps(result, sort_keys=True, allow_nan=False)
        result_keys = _all_keys(result)
        for forbidden in (
            "head_weight",
            "optimizer_state",
            "checkpoint_state",
            "primary_bt_head_sha256",
            "heldout_metric",
            "policy_utility",
            "frozen_global_beta",
        ):
            assert forbidden not in result_keys
        assert validate_r3_control_family_result(result, controls_config) == result


def test_exact_three_seed_matrix_is_constructible_without_cross_family_state(
    controls_config,
) -> None:
    results = [
        _adapt(family, controls_config, seed=seed)
        for seed in R3_GATE_C_SEEDS
        for family in R3_GATE_C_FAMILIES
    ]
    assert len(results) == 9
    assert {(result["seed"], result["family"]) for result in results} == {
        (seed, family) for seed in R3_GATE_C_SEEDS for family in R3_GATE_C_FAMILIES
    }
    assert len({result["result_sha256"] for result in results}) == 9


def test_family_adapter_signatures_have_no_forbidden_channels() -> None:
    forbidden = {
        "primary_head",
        "primary_state",
        "optimizer_state",
        "checkpoint",
        "heldout",
        "validation",
        "policy",
        "beta",
    }
    for adapter in (
        adapt_exact_margin_prorm_plus_result,
        adapt_exact_soft_label_bt_result,
        adapt_low_dimensional_prorm_plus_result,
    ):
        assert forbidden.isdisjoint(inspect.signature(adapter).parameters)


def test_exact_margin_validates_direct_identity_and_all_pcg_audits(controls_config) -> None:
    result = _adapt("exact_margin_prorm_plus", controls_config)
    evidence = result["family_evidence"]
    assert set(evidence["pcg_audits"]) == {
        "selected_head_final_inner",
        "cold_saved_head_audit",
        "trained_direction",
    }
    assert evidence["direct_identity"]["reward_head_bypassed"] is True

    tampered = copy.deepcopy(result)
    tampered["family_evidence"]["pcg_audits"]["trained_direction"]["converged"] = False
    with pytest.raises(R3ControlResultError, match="converged"):
        validate_r3_control_family_result(_rehash(tampered), controls_config)

    tampered = copy.deepcopy(result)
    tampered["family_evidence"]["direct_identity"]["complete_pair_identity_absolute_error"] = (
        1.1e-10
    )
    with pytest.raises(R3ControlResultError, match="moment identity"):
        validate_r3_control_family_result(_rehash(tampered), controls_config)


def test_exact_soft_label_is_noise_free_and_rejects_sampled_label_access(
    controls_config,
) -> None:
    result = _adapt("exact_soft_label_bt", controls_config)
    target = result["family_evidence"]["target_audit"]
    assert target["noise_free"] is True
    assert target["bernoulli_sampling_used"] is False
    assert target["sampled_label_stream_accessed"] is False

    tampered = copy.deepcopy(result)
    tampered["family_evidence"]["target_audit"]["sampled_label_stream_accessed"] = True
    with pytest.raises(R3ControlResultError, match="sampled_label_stream_accessed"):
        validate_r3_control_family_result(_rehash(tampered), controls_config)


def test_low_dimensional_has_independent_fixed_scatter_and_score_gates(
    controls_config,
) -> None:
    result = _adapt("low_dimensional_prorm_plus", controls_config)
    evidence = result["family_evidence"]
    assert "bt_head" not in evidence
    assert evidence["target_audit"]["primary_label_stream_accessed"] is False
    assert evidence["scatter_identity"]["absolute_tolerance"] == 1.0e-4
    assert evidence["score_identity"]["absolute_tolerance"] == 1.0e-4
    assert (
        evidence["scatter_identity"]["schema_version"]
        != (evidence["score_identity"]["schema_version"])
    )

    scatter_failure = copy.deepcopy(result)
    scatter_failure["family_evidence"]["scatter_identity"]["max_absolute_error"] = 1.1e-4
    with pytest.raises(R3ControlResultError, match="scatter identity gate"):
        validate_r3_control_family_result(_rehash(scatter_failure), controls_config)

    score_failure = copy.deepcopy(result)
    score_failure["family_evidence"]["score_identity"]["max_absolute_error"] = 1.1e-4
    with pytest.raises(R3ControlResultError, match="score identity gate"):
        validate_r3_control_family_result(_rehash(score_failure), controls_config)


def test_low_dimensional_rejects_primary_bt_binding_even_with_valid_self_hash(
    controls_config,
) -> None:
    result = _adapt("low_dimensional_prorm_plus", controls_config)
    tampered = copy.deepcopy(result)
    tampered["family_evidence"]["primary_bt_head_sha256"] = _sha("primary-bt")
    with pytest.raises(R3ControlResultError, match="invalid fields"):
        validate_r3_control_family_result(_rehash(tampered), controls_config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda result: result["information_boundary"].__setitem__(
                "heldout_or_validation_accessed", True
            ),
            "information_boundary",
        ),
        (
            lambda result: result["completion"].__setitem__("profile_only", True),
            "completion.profile_only",
        ),
        (
            lambda result: result["family_evidence"]["head_audit"].__setitem__(
                "final_objective", 1.1
            ),
            "objective did not decrease",
        ),
        (
            lambda result: result["family_evidence"]["head_audit"]["first_order_gate"].__setitem__(
                "gradient_ratio_to_zero_initialization", 2.0e-3
            ),
            "gradient-ratio arithmetic",
        ),
    ],
)
def test_result_validator_fails_closed_after_semantic_tampering(
    controls_config,
    mutate,
    message: str,
) -> None:
    result = _adapt("exact_soft_label_bt", controls_config)
    tampered = copy.deepcopy(result)
    mutate(tampered)
    with pytest.raises(R3ControlResultError, match=message):
        validate_r3_control_family_result(_rehash(tampered), controls_config)


def test_result_validator_rejects_stale_self_hash(controls_config) -> None:
    result = _adapt("exact_soft_label_bt", controls_config)
    result["completion"]["completed_updates"] = 160
    with pytest.raises(R3ControlResultError, match="completion.completed_updates"):
        validate_r3_control_family_result(result, controls_config)
