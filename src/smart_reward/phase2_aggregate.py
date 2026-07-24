"""Strict seed-level aggregation for the common-beta finite-policy experiment.

The experimental unit in this module is always one frozen Phase-2 seed.  A
rollout JSONL is read only to verify its sibling path, byte hash, schema, and
arm counts; prompt and candidate rows never enter a bootstrap table.

An aggregate is publishable only when every declared overlay seed shares the exact
source/design/runtime/environment identity, every KL safety record passed,
fresh R=4 heads were trained after the train-oracle rescore, and every result
contains exactly the four preregistered policy arms.  Scientific evidence can
still be ``not_passed``: hard integrity gates reject inputs, whereas the two
frozen effect criteria determine the reported evidence status.
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

from .contracts import BT_MLE, CANONICAL_LEARNERS, PRORM_PLUS
from .paths import relative_posix_reference
from .phase2_config import phase2_design_identity, validate_phase2_config
from .phase2_rollout import (
    KL_HISTORY_SOURCE,
    KL_ORIENTATION,
    PHASE2_ARM_ORDER,
    PHASE2_RESULT_SCHEMA,
    PHASE2_ROLLOUT_SCHEMA,
    Phase2Design,
)
from .repro import atomic_write_json
from .statistics import PairedMetricsAggregate, aggregate_paired_metrics

PHASE2_AGGREGATE_SCHEMA = "common-beta-seed-aggregate/v3"

_NUMERIC_GATE_CONFIG_PATH = "positive_controls.numeric_gate_tolerances"
_ADAPTIVE_CONVERGENCE_CONFIG_PATH = "reward_model.adaptive_convergence"

_HEX = frozenset("0123456789abcdef")
_ENVIRONMENT_KEYS = frozenset(
    {
        "formal",
        "git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "account",
        "partition",
        "gpu_models",
    }
)
_PAIR_METRIC_DIRECTIONS: dict[str, str] = {
    "improvement_over_zero_b": "higher_is_better",
    "mean_on_policy_kl_pi_updated_to_pi0": "lower_is_better",
    "mean_response_token_count": "lower_is_better",
    "mean_target_reward": "higher_is_better",
    "mean_target_utility": "higher_is_better",
    "on_policy_kl_prompt_maximum": "lower_is_better",
    "on_policy_kl_prompt_p50": "lower_is_better",
    "on_policy_kl_prompt_p90": "lower_is_better",
    "on_policy_kl_prompt_p95": "lower_is_better",
    "on_policy_kl_prompt_p99": "lower_is_better",
    "on_policy_kl_sequence_maximum": "lower_is_better",
    "oracle_step_reference_gap": "lower_is_better",
    "reached_max_length_rate": "lower_is_better",
}
_METRIC_ROLES: dict[str, str] = {
    "mean_target_utility": "primary",
    "improvement_over_zero_b": "primary_corollary",
    "oracle_step_reference_gap": "positive_control_distance",
    "mean_target_reward": "secondary",
    "mean_on_policy_kl_pi_updated_to_pi0": "diagnostic",
    "on_policy_kl_prompt_p50": "pilot_locality_tail_diagnostic",
    "on_policy_kl_prompt_p90": "pilot_locality_tail_diagnostic",
    "on_policy_kl_prompt_p95": "pilot_locality_tail_diagnostic",
    "on_policy_kl_prompt_p99": "pilot_locality_tail_diagnostic",
    "on_policy_kl_prompt_maximum": "pilot_locality_tail_diagnostic",
    "on_policy_kl_sequence_maximum": "pilot_locality_tail_diagnostic",
    "reached_max_length_rate": "diagnostic",
    "mean_response_token_count": "diagnostic",
}


def _digest(value: object, *, name: str, lengths: set[int] | None = None) -> str:
    allowed = {64} if lengths is None else lengths
    if (
        not isinstance(value, str)
        or len(value) not in allowed
        or any(character not in _HEX for character in value)
    ):
        rendered = " or ".join(str(length) for length in sorted(allowed))
        raise ValueError(f"{name} must be a lowercase hexadecimal digest of length {rendered}")
    return value


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _required(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    missing = keys - set(result)
    if missing:
        raise ValueError(f"{name} is missing required fields {sorted(missing)!r}")
    return result


def _canonical_sha256(value: object) -> str:
    payload = dict(value) if isinstance(value, Mapping) else value
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_bytes(raw: bytes, *, path: Path) -> Mapping[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{path} contains non-finite JSON number {value}")

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not strict UTF-8 JSON: {error}") from error
    return _mapping(value, name=str(path))


def _read_regular_file(path: Path, *, name: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist as a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name} {path}: {error}") from error


def _close(left: float, right: float, *, tolerance: float = 2.0e-6) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _numeric_gate_contract(
    numeric_tolerances: Mapping[str, object],
    adaptive_convergence: Mapping[str, object],
    *,
    stage: str,
    formal_eligibility: bool,
) -> dict[str, object]:
    """Expose only thresholds already bound into the normalized overlay."""

    return {
        "schema_version": "phase2-design-bound-numeric-gate-contract/v1",
        "status": "design_bound",
        "design_bound": True,
        "overlay_paths": [
            _NUMERIC_GATE_CONFIG_PATH,
            _ADAPTIVE_CONVERGENCE_CONFIG_PATH,
        ],
        "design_stage": stage,
        "design_formal_eligibility": formal_eligibility,
        "tolerances": dict(numeric_tolerances),
        "adaptive_convergence": dict(adaptive_convergence),
    }


def _expect(value: object, expected: object, *, name: str) -> None:
    if value != expected:
        raise ValueError(f"{name} must equal {expected!r}")


def _validate_reward_head_identifiability(
    value: object,
    *,
    config: Mapping[str, object],
    expected_train_prompts: int,
    expected_reward_dimension: int,
    name: str,
) -> tuple[dict[str, object], str]:
    evidence = _mapping(value, name=name)
    expected_fields = {
        "schema_version": "reward-head-identifiability/v1",
        "design_matrix": "canonical_edge_reward_feature_differences",
        "split": "train",
        "shape": [expected_train_prompts, expected_reward_dimension],
        "audit_dtype": "torch.float64",
        "role": config["role"],
        "require_full_column_rank": config["require_full_column_rank"],
        "bt_unique_finite_optimum_sufficient_condition_only": True,
        "prorm_moment_map_full_rank_proved": False,
        "algorithmic_tie_break": config["algorithmic_tie_break"],
        "minimum_norm_solution_claimed": config["minimum_norm_claim"],
        "test_or_validation_data_accessed": False,
    }
    for field, expected in expected_fields.items():
        _expect(evidence.get(field), expected, name=f"{name}.{field}")
    source_dtype = evidence.get("source_dtype")
    if source_dtype not in {"torch.float32", "torch.float64"}:
        raise ValueError(f"{name}.source_dtype is invalid")
    _digest(evidence.get("design_matrix_sha256"), name=f"{name}.design_matrix_sha256")
    relative_tolerance = _finite(
        evidence.get("relative_rank_tolerance"),
        name=f"{name}.relative_rank_tolerance",
    )
    if not _close(
        relative_tolerance,
        _finite(
            config["relative_rank_tolerance"],
            name="reward_model.identifiability.relative_rank_tolerance",
        ),
        tolerance=1.0e-15,
    ):
        raise ValueError(f"{name} rank tolerance differs from the overlay")
    rank = _integer(evidence.get("numerical_rank"), name=f"{name}.numerical_rank")
    column_dimension = _integer(
        evidence.get("column_dimension"),
        name=f"{name}.column_dimension",
        minimum=1,
    )
    _expect(column_dimension, expected_reward_dimension, name=f"{name}.column_dimension")
    if rank > min(expected_train_prompts, column_dimension):
        raise ValueError(f"{name}.numerical_rank exceeds matrix dimensions")
    full_column_rank = evidence.get("full_column_rank")
    _expect(full_column_rank, rank == column_dimension, name=f"{name}.full_column_rank")
    largest = _finite(
        evidence.get("largest_singular_value"),
        name=f"{name}.largest_singular_value",
    )
    if largest <= 0.0:
        raise ValueError(f"{name}.largest_singular_value must be positive")
    threshold = _finite(
        evidence.get("absolute_singular_value_threshold"),
        name=f"{name}.absolute_singular_value_threshold",
    )
    if not math.isclose(
        threshold,
        relative_tolerance * largest,
        rel_tol=1.0e-10,
        abs_tol=1.0e-14,
    ):
        raise ValueError(f"{name} singular-value threshold arithmetic failed")
    smallest = _finite(
        evidence.get("smallest_singular_value"),
        name=f"{name}.smallest_singular_value",
        nonnegative=True,
    )
    if smallest > largest:
        raise ValueError(f"{name} singular-value ordering is invalid")
    smallest_retained = evidence.get("smallest_retained_singular_value")
    condition = evidence.get("retained_condition_number")
    if rank == 0:
        if smallest_retained is not None or condition is not None:
            raise ValueError(f"{name} rank-zero diagnostic has retained-spectrum values")
    else:
        retained = _finite(
            smallest_retained,
            name=f"{name}.smallest_retained_singular_value",
        )
        if retained <= threshold or retained > largest:
            raise ValueError(f"{name} smallest retained singular value is invalid")
        observed_condition = _finite(
            condition,
            name=f"{name}.retained_condition_number",
        )
        if not math.isclose(
            observed_condition,
            largest / retained,
            rel_tol=1.0e-10,
            abs_tol=1.0e-14,
        ):
            raise ValueError(f"{name} retained condition-number arithmetic failed")
    required = config["require_full_column_rank"] is True
    expected_acceptance = bool(full_column_rank) if required else True
    _expect(
        evidence.get("acceptance_gate_passed"),
        expected_acceptance,
        name=f"{name}.acceptance_gate_passed",
    )
    if required and not full_column_rank:
        raise ValueError(f"{name} failed the design-bound full-column-rank gate")
    normalized = dict(evidence)
    return normalized, _canonical_sha256(normalized)


def _validate_prorm_moment_map_identifiability(
    value: object,
    *,
    config: Mapping[str, object],
    expected_train_prompts: int,
    expected_policy_dimension: int,
    expected_reward_dimension: int,
    expected_projection_sha256: str | None = None,
    expected_projected_geometry: Mapping[str, object] | None = None,
    name: str,
) -> tuple[dict[str, object], str]:
    projected = expected_projection_sha256 is not None
    if projected != (expected_projected_geometry is not None):
        raise ValueError(
            "expected projection and projected geometry must either both be "
            "supplied or both omitted"
        )
    required_keys = {
        "schema_version",
        "design_matrix",
        "formula",
        "split",
        "orientation",
        "shape",
        "num_edges",
        "policy_dimension",
        "column_dimension",
        "source_policy_score_dtype",
        "source_reward_feature_dtype",
        "audit_dtype",
        "edge_policy_score_difference_sha256",
        "edge_reward_feature_difference_sha256",
        "moment_map_sha256",
        "computation",
        "relative_rank_tolerance",
        "absolute_singular_value_threshold",
        "singular_values_descending",
        "singular_values_sha256",
        "singular_spectrum_summary",
        "numerical_rank",
        "full_column_rank",
        "retained_condition_number",
        "population_identifiability_theorem_claimed",
        "role",
        "require_full_column_rank",
        "acceptance_gate_passed",
        "algorithmic_tie_break",
        "minimum_norm_solution_claimed",
        "test_or_validation_data_accessed",
    }
    required_keys.add("projected_geometry" if projected else "ridge_geometry")
    uniqueness_key = (
        "unique_projected_prorm_quadratic_head_iff_full_column_rank"
        if projected
        else "unique_ridge_prorm_quadratic_head_iff_full_column_rank"
    )
    observed_uniqueness_key = (
        "observed_unique_projected_prorm_quadratic_head"
        if projected
        else "observed_unique_ridge_prorm_quadratic_head"
    )
    required_keys.update({uniqueness_key, observed_uniqueness_key})
    if projected:
        required_keys.update(
            {
                "projection_sha256",
                "row_dimension",
                "full_row_rank",
                "require_full_row_rank",
                "acceptance_gate_definition",
            }
        )
    evidence = _mapping(value, name=name)
    if set(evidence) != required_keys:
        missing = sorted(required_keys - set(evidence))
        extra = sorted(set(evidence) - required_keys)
        raise ValueError(f"{name} fields differ; missing={missing!r}, extra={extra!r}")
    expected_fields = {
        "schema_version": (
            "projected-prorm-moment-map-identifiability/v1"
            if projected
            else "prorm-moment-map-identifiability/v1"
        ),
        "design_matrix": (
            "canonical_train_edge_projected_moment_jacobian"
            if projected
            else "canonical_train_edge_moment_jacobian"
        ),
        "formula": "J_m = Z^T D / (2 n_edges)",
        "split": "train",
        "orientation": "candidate_0_minus_candidate_1",
        "shape": [expected_policy_dimension, expected_reward_dimension],
        "num_edges": expected_train_prompts,
        "policy_dimension": expected_policy_dimension,
        "column_dimension": expected_reward_dimension,
        "audit_dtype": "torch.float64",
        "role": config["role"],
        "require_full_column_rank": (False if projected else config["require_full_column_rank"]),
        uniqueness_key: True,
        "population_identifiability_theorem_claimed": False,
        "algorithmic_tie_break": config["algorithmic_tie_break"],
        "minimum_norm_solution_claimed": config["minimum_norm_claim"],
        "test_or_validation_data_accessed": False,
    }
    if expected_projection_sha256 is not None:
        expected_fields["projection_sha256"] = expected_projection_sha256
        expected_fields["row_dimension"] = expected_policy_dimension
        expected_fields["require_full_row_rank"] = True
        expected_fields["acceptance_gate_definition"] = (
            "full_row_rank_for_projected_policy_moment_coverage"
        )
    for field, expected in expected_fields.items():
        _expect(evidence[field], expected, name=f"{name}.{field}")
    for field in ("source_policy_score_dtype", "source_reward_feature_dtype"):
        if evidence[field] not in {"torch.float32", "torch.float64"}:
            raise ValueError(f"{name}.{field} is invalid")
    for field in (
        "edge_policy_score_difference_sha256",
        "edge_reward_feature_difference_sha256",
        "moment_map_sha256",
        "singular_values_sha256",
    ):
        _digest(evidence[field], name=f"{name}.{field}")

    computation = _mapping(evidence["computation"], name=f"{name}.computation")
    expected_computation_keys = {
        "algorithm",
        "row_block_size",
        "num_row_blocks",
        "full_moment_map_materialized",
        "randomized_rank_approximation_used",
    }
    if set(computation) != expected_computation_keys:
        raise ValueError(f"{name}.computation has an invalid field set")
    _expect(
        computation["algorithm"],
        "deterministic_blocked_fp64_tsqr",
        name=f"{name}.computation.algorithm",
    )
    _expect(
        computation["full_moment_map_materialized"],
        False,
        name=f"{name}.computation.full_moment_map_materialized",
    )
    _expect(
        computation["randomized_rank_approximation_used"],
        False,
        name=f"{name}.computation.randomized_rank_approximation_used",
    )
    block_rows = _integer(
        computation["row_block_size"],
        name=f"{name}.computation.row_block_size",
        minimum=1,
    )
    expected_block_rows = min(
        expected_policy_dimension,
        max(
            1,
            min(
                4096,
                (16 * 1024 * 1024) // (8 * expected_reward_dimension),
            ),
        ),
    )
    _expect(
        block_rows,
        expected_block_rows,
        name=f"{name}.computation.row_block_size",
    )
    expected_blocks = (expected_policy_dimension + block_rows - 1) // block_rows
    _expect(
        computation["num_row_blocks"],
        expected_blocks,
        name=f"{name}.computation.num_row_blocks",
    )

    relative_tolerance = _finite(
        evidence["relative_rank_tolerance"],
        name=f"{name}.relative_rank_tolerance",
        nonnegative=True,
    )
    configured_tolerance = _finite(
        config["relative_rank_tolerance"],
        name="reward_model.identifiability.relative_rank_tolerance",
        nonnegative=True,
    )
    if not _close(relative_tolerance, configured_tolerance, tolerance=1.0e-15):
        raise ValueError(f"{name} rank tolerance differs from the overlay")
    raw_spectrum = evidence["singular_values_descending"]
    if not isinstance(raw_spectrum, list):
        raise TypeError(f"{name}.singular_values_descending must be a list")
    expected_spectrum_size = min(expected_policy_dimension, expected_reward_dimension)
    if len(raw_spectrum) != expected_spectrum_size:
        raise ValueError(f"{name}.singular_values_descending has the wrong length")
    spectrum = [
        _finite(
            value,
            name=f"{name}.singular_values_descending[{index}]",
            nonnegative=True,
        )
        for index, value in enumerate(raw_spectrum)
    ]
    if any(left < right for left, right in zip(spectrum, spectrum[1:], strict=False)):
        raise ValueError(f"{name}.singular_values_descending is not descending")
    spectrum_payload = {
        "dtype": "torch.float64",
        "shape": [len(spectrum)],
        "values_descending": spectrum,
    }
    _expect(
        evidence["singular_values_sha256"],
        _canonical_sha256(spectrum_payload),
        name=f"{name}.singular_values_sha256",
    )
    largest = spectrum[0]
    smallest = spectrum[-1]
    threshold = _finite(
        evidence["absolute_singular_value_threshold"],
        name=f"{name}.absolute_singular_value_threshold",
        nonnegative=True,
    )
    if not math.isclose(
        threshold,
        relative_tolerance * largest,
        rel_tol=1.0e-10,
        abs_tol=1.0e-14,
    ):
        raise ValueError(f"{name} singular-value threshold arithmetic failed")
    retained = [value for value in spectrum if value > threshold]
    numerical_rank = _integer(
        evidence["numerical_rank"],
        name=f"{name}.numerical_rank",
    )
    _expect(numerical_rank, len(retained), name=f"{name}.numerical_rank")
    full_column_rank = numerical_rank == expected_reward_dimension
    full_row_rank = numerical_rank == expected_policy_dimension
    _expect(evidence["full_column_rank"], full_column_rank, name=f"{name}.full_column_rank")
    if projected:
        _expect(evidence["full_row_rank"], full_row_rank, name=f"{name}.full_row_rank")
    _expect(
        evidence[observed_uniqueness_key],
        full_column_rank,
        name=f"{name}.{observed_uniqueness_key}",
    )
    smallest_retained = retained[-1] if retained else None
    expected_condition = (
        largest / smallest_retained
        if smallest_retained is not None and smallest_retained > 0.0
        else None
    )
    condition = evidence["retained_condition_number"]
    if expected_condition is None:
        _expect(condition, None, name=f"{name}.retained_condition_number")
    elif not math.isclose(
        _finite(condition, name=f"{name}.retained_condition_number"),
        expected_condition,
        rel_tol=1.0e-10,
        abs_tol=1.0e-14,
    ):
        raise ValueError(f"{name} retained condition-number arithmetic failed")

    summary = _mapping(
        evidence["singular_spectrum_summary"],
        name=f"{name}.singular_spectrum_summary",
    )
    if set(summary) != {"count", "largest", "smallest", "smallest_retained"}:
        raise ValueError(f"{name}.singular_spectrum_summary has an invalid field set")
    expected_summary = {
        "count": len(spectrum),
        "largest": largest,
        "smallest": smallest,
        "smallest_retained": smallest_retained,
    }
    for field, expected in expected_summary.items():
        _expect(summary[field], expected, name=f"{name}.singular_spectrum_summary.{field}")

    if projected:
        if expected_projected_geometry is None:
            raise RuntimeError("projected geometry expectation was not resolved")
        geometry = _mapping(
            evidence["projected_geometry"],
            name=f"{name}.projected_geometry",
        )
        expected_geometry = {
            "matrix": "H_low = F_hat_low",
            "positive_definite": True,
            "reason": "projected_fisher_numerical_rank_equals_selected_dimension",
            "regularization": "moore_penrose_pseudoinverse_on_full_rank_H_low",
            "fisher_sha256": expected_projected_geometry["fisher_sha256"],
            "pseudoinverse_sha256": expected_projected_geometry["pseudoinverse_sha256"],
            "relative_eigenvalue_tolerance": expected_projected_geometry[
                "relative_eigenvalue_tolerance"
            ],
            "head_hessian": "J_m^T H_low^{-1} J_m / beta",
            "rank_identity": "rank(J_m^T H_low^{-1} J_m) = rank(J_m)",
        }
        if dict(geometry) != expected_geometry:
            raise ValueError(
                f"{name}.projected_geometry does not establish the locked rank identity"
            )
    else:
        geometry = _mapping(evidence["ridge_geometry"], name=f"{name}.ridge_geometry")
        expected_geometry = {
            "matrix": "H = F_hat + lambda I",
            "positive_definite": True,
            "reason": "configured_relative_damping_is_strictly_positive",
            "head_hessian": "J_m^T H^{-1} J_m / beta",
            "rank_identity": "rank(J_m^T H^{-1} J_m) = rank(J_m)",
        }
        if dict(geometry) != expected_geometry:
            raise ValueError(f"{name}.ridge_geometry does not establish the locked rank identity")
    required = False if projected else config["require_full_column_rank"] is True
    expected_acceptance = full_row_rank if projected else (full_column_rank if required else True)
    _expect(
        evidence["acceptance_gate_passed"],
        expected_acceptance,
        name=f"{name}.acceptance_gate_passed",
    )
    if projected and not full_row_rank:
        raise ValueError(f"{name} failed the projected policy-moment full-row-rank gate")
    if required and not full_column_rank:
        raise ValueError(f"{name} failed the design-bound full-column-rank gate")
    normalized = dict(evidence)
    return normalized, _canonical_sha256(normalized)


def _validate_first_order_convergence(
    value: object,
    *,
    expected_objective_name: str,
    expected_initial_objective: float,
    expected_final_objective: float,
    expected_head_sha256: str,
    expected_fixed_snapshot_steps: int,
    numeric_tolerances: Mapping[str, object],
    adaptive_convergence: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    ratio_tolerance = _finite(
        numeric_tolerances["outer_relative_gradient_ratio"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.outer_relative_gradient_ratio",
    )
    objective_relative_tolerance = _finite(
        numeric_tolerances["objective_binding_relative_error"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.objective_binding_relative_error",
    )
    objective_absolute_tolerance = _finite(
        numeric_tolerances["objective_binding_absolute_error"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.objective_binding_absolute_error",
    )
    min_steps = _integer(
        adaptive_convergence["minimum_steps"],
        name=f"{_ADAPTIVE_CONVERGENCE_CONFIG_PATH}.minimum_steps",
        minimum=1,
    )
    max_steps = _integer(
        adaptive_convergence["maximum_steps"],
        name=f"{_ADAPTIVE_CONVERGENCE_CONFIG_PATH}.maximum_steps",
        minimum=min_steps,
    )
    check_interval = _integer(
        adaptive_convergence["check_interval_steps"],
        name=f"{_ADAPTIVE_CONVERGENCE_CONFIG_PATH}.check_interval_steps",
        minimum=1,
    )
    consecutive_checks = _integer(
        adaptive_convergence["consecutive_passing_checks"],
        name=f"{_ADAPTIVE_CONVERGENCE_CONFIG_PATH}.consecutive_passing_checks",
        minimum=1,
    )
    denominator_floor = _finite(
        adaptive_convergence["denominator_floor"],
        name=f"{_ADAPTIVE_CONVERGENCE_CONFIG_PATH}.denominator_floor",
    )
    if denominator_floor <= 0.0:
        raise ValueError("adaptive convergence denominator floor must be positive")
    if not _close(
        ratio_tolerance,
        _finite(
            adaptive_convergence["relative_gradient_ratio_tolerance"],
            name=f"{_ADAPTIVE_CONVERGENCE_CONFIG_PATH}.relative_gradient_ratio_tolerance",
        ),
        tolerance=1.0e-15,
    ):
        raise ValueError("numeric and adaptive outer gradient-ratio tolerances disagree")
    convergence = _required(
        value,
        name=name,
        keys={
            "schema_version",
            "objective",
            "converged",
            "fail_closed",
            "spec",
            "initial_zero_head_measurement",
            "checks",
            "selected_primary_step",
            "selected_primary_head_sha256",
            "consecutive_threshold_passes_at_selection",
            "final_gate",
            "fixed_step_compute_matched_snapshot",
            "fixed_step_snapshot_steps",
            "fixed_step_snapshot_is_not_primary_selection",
            "solution_identification",
            "test_or_validation_data_accessed",
        },
    )
    expected_scalars = {
        "schema_version": "objective-first-order-convergence/v1",
        "objective": expected_objective_name,
        "converged": True,
        "fail_closed": True,
        "fixed_step_snapshot_steps": expected_fixed_snapshot_steps,
        "fixed_step_snapshot_is_not_primary_selection": True,
        "test_or_validation_data_accessed": False,
    }
    for field, expected in expected_scalars.items():
        _expect(convergence[field], expected, name=f"{name}.{field}")
    _expect(
        convergence["selected_primary_head_sha256"],
        expected_head_sha256,
        name=f"{name}.selected_primary_head_sha256",
    )
    spec = _mapping(convergence["spec"], name=f"{name}.spec")
    expected_spec = {
        "schema_version": "objective-first-order-convergence-spec/v1",
        "gradient_ratio_tolerance": ratio_tolerance,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "check_interval": check_interval,
        "consecutive_checks": consecutive_checks,
        "gradient_norm_denominator_floor": denominator_floor,
        "fail_closed": True,
        "gradient": "full_data_post_update_unclipped",
        "denominator": "exact_zero_initialization_gradient_l2_norm",
        "validation_or_test_selection": False,
    }
    for field, expected in expected_spec.items():
        _expect(spec.get(field), expected, name=f"{name}.spec.{field}")
    selected_step = _integer(
        convergence["selected_primary_step"],
        name=f"{name}.selected_primary_step",
        minimum=min_steps,
    )
    if selected_step > max_steps or selected_step % check_interval != 0:
        raise ValueError(f"{name}.selected_primary_step is not a scheduled gate check")
    _expect(
        convergence["consecutive_threshold_passes_at_selection"],
        consecutive_checks,
        name=f"{name}.consecutive_threshold_passes_at_selection",
    )

    initial = _mapping(
        convergence["initial_zero_head_measurement"],
        name=f"{name}.initial_zero_head_measurement",
    )
    initial_objective = _finite(
        initial.get("objective"),
        name=f"{name}.initial_zero_head_measurement.objective",
        nonnegative=True,
    )
    initial_gradient = _finite(
        initial.get("gradient_l2_norm"),
        name=f"{name}.initial_zero_head_measurement.gradient_l2_norm",
        nonnegative=True,
    )
    if not math.isclose(
        initial_objective,
        expected_initial_objective,
        rel_tol=objective_relative_tolerance,
        abs_tol=objective_absolute_tolerance,
    ):
        raise ValueError(f"{name} initial objective is not bound to the zero-head audit")
    final_gate = _mapping(convergence["final_gate"], name=f"{name}.final_gate")
    _expect(final_gate.get("step"), selected_step, name=f"{name}.final_gate.step")
    _expect(
        final_gate.get("threshold_passed"),
        True,
        name=f"{name}.final_gate.threshold_passed",
    )
    _expect(
        final_gate.get("fresh_post_restore_audit"),
        True,
        name=f"{name}.final_gate.fresh_post_restore_audit",
    )
    final_measurement = _mapping(
        final_gate.get("measurement"),
        name=f"{name}.final_gate.measurement",
    )
    final_objective = _finite(
        final_measurement.get("objective"),
        name=f"{name}.final_gate.measurement.objective",
        nonnegative=True,
    )
    final_gradient = _finite(
        final_measurement.get("gradient_l2_norm"),
        name=f"{name}.final_gate.measurement.gradient_l2_norm",
        nonnegative=True,
    )
    if not math.isclose(
        final_objective,
        expected_final_objective,
        rel_tol=objective_relative_tolerance,
        abs_tol=objective_absolute_tolerance,
    ):
        raise ValueError(f"{name} final objective is not bound to its selected iterate")
    denominator = max(initial_gradient, denominator_floor)
    expected_ratio = final_gradient / denominator
    recorded_ratio = _finite(
        final_gate.get("gradient_ratio_to_zero_initialization"),
        name=f"{name}.final_gate.gradient_ratio_to_zero_initialization",
        nonnegative=True,
    )
    if not math.isclose(recorded_ratio, expected_ratio, rel_tol=1.0e-10, abs_tol=1.0e-14):
        raise ValueError(f"{name} final gradient-ratio arithmetic failed")
    if recorded_ratio > ratio_tolerance:
        raise ValueError(f"{name} failed the sustained outer-convergence gate")

    checks = convergence["checks"]
    if not isinstance(checks, list) or len(checks) < consecutive_checks:
        raise ValueError(f"{name}.checks lacks sustained convergence evidence")
    selected_steps = {
        selected_step - offset * check_interval for offset in range(consecutive_checks)
    }
    selected_checks = [
        check
        for check in checks
        if isinstance(check, Mapping) and check.get("step") in selected_steps
    ]
    if len(selected_checks) != consecutive_checks or any(
        check.get("threshold_passed") is not True
        or check.get("post_update") is not True
        or check.get("full_data") is not True
        or check.get("gradient_clipping_applied") is not False
        for check in selected_checks
    ):
        raise ValueError(f"{name} does not contain the required consecutive full-data checks")

    fixed = _mapping(
        convergence["fixed_step_compute_matched_snapshot"],
        name=f"{name}.fixed_step_compute_matched_snapshot",
    )
    expected_fixed = {
        "schema_version": "fixed-step-compute-matched-snapshot/v1",
        "step": expected_fixed_snapshot_steps,
        "role": "compute_matched_and_pilot_diagnostic_only",
        "used_as_primary_selection_rule": False,
    }
    for field, expected in expected_fixed.items():
        _expect(fixed.get(field), expected, name=f"{name}.fixed_step.{field}")
    fixed_history = _mapping(
        fixed.get("history_summary"),
        name=f"{name}.fixed_step.history_summary",
    )
    _expect(
        fixed_history.get("num_steps"),
        expected_fixed_snapshot_steps,
        name=f"{name}.fixed_step.history_summary.num_steps",
    )
    identification = _mapping(
        convergence["solution_identification"],
        name=f"{name}.solution_identification",
    )
    expected_identification = {
        "initialization": "exact_zero_head",
        "tie_break": "zero_initialized_adamw_implicit_bias",
        "validation_or_test_checkpoint_selection": False,
        "objective_value_checkpoint_selection": False,
        "minimum_norm_projection_applied": False,
        "minimum_norm_solution_claimed": False,
        "unique_reward_head_solution_claimed": False,
    }
    for field, expected in expected_identification.items():
        _expect(
            identification.get(field),
            expected,
            name=f"{name}.solution_identification.{field}",
        )
    rank_diagnostic = _mapping(
        identification.get("optional_objective_rank_diagnostic"),
        name=f"{name}.solution_identification.optional_objective_rank_diagnostic",
    )
    _expect(
        rank_diagnostic.get("evaluated"),
        True,
        name=f"{name}.solution_identification.optional_objective_rank_diagnostic.evaluated",
    )
    rank_evidence = _mapping(
        rank_diagnostic.get("evidence"),
        name=f"{name}.solution_identification.optional_objective_rank_diagnostic.evidence",
    )
    return {
        "passed": True,
        "selected_primary_step": selected_step,
        "initial_gradient_l2_norm": initial_gradient,
        "final_gradient_l2_norm": final_gradient,
        "gradient_ratio_to_zero_initialization": recorded_ratio,
        "design_bound_gradient_ratio_tolerance": ratio_tolerance,
        "sustained_checks": consecutive_checks,
        "fixed_720_step_snapshot_is_diagnostic_only": True,
        "numeric_tolerance_design_bound": True,
        "rank_diagnostic_sha256": _canonical_sha256(rank_evidence),
    }


def _objective_decrease_gate(
    *,
    initial: object,
    final: object,
    audit: object,
    gradient: object,
    convergence_gate: Mapping[str, object],
    numeric_tolerances: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    objective_relative_tolerance = _finite(
        numeric_tolerances["objective_binding_relative_error"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.objective_binding_relative_error",
    )
    objective_absolute_tolerance = _finite(
        numeric_tolerances["objective_binding_absolute_error"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.objective_binding_absolute_error",
    )
    initial_value = _finite(initial, name=f"{name}.initial_objective", nonnegative=True)
    final_value = _finite(final, name=f"{name}.final_objective", nonnegative=True)
    audit_value = _finite(audit, name=f"{name}.audit_objective", nonnegative=True)
    gradient_value = _finite(
        gradient,
        name=f"{name}.gradient_l2_norm",
        nonnegative=True,
    )
    if not math.isclose(
        audit_value,
        final_value,
        rel_tol=objective_relative_tolerance,
        abs_tol=objective_absolute_tolerance,
    ):
        raise ValueError(f"{name} final objective is not bound to its cold full-data audit")
    if not final_value < initial_value:
        raise ValueError(f"{name} objective did not decrease from the zero initialization")
    converged_gradient = _finite(
        convergence_gate.get("final_gradient_l2_norm"),
        name=f"{name}.convergence_final_gradient_l2_norm",
        nonnegative=True,
    )
    if not math.isclose(
        gradient_value,
        converged_gradient,
        rel_tol=objective_relative_tolerance,
        abs_tol=objective_absolute_tolerance,
    ):
        raise ValueError(f"{name} cold audit gradient does not bind the convergence gate")
    return {
        "passed": True,
        "initial_objective": initial_value,
        "final_objective": final_value,
        "objective_decreased": True,
        "cold_full_data_audit_objective": audit_value,
        "objective_binding_absolute_error": abs(audit_value - final_value),
        "gradient_l2_norm": gradient_value,
        "first_order_convergence": dict(convergence_gate),
        "numeric_tolerance_design_bound": True,
    }


def _pcg_gate(
    value: object,
    *,
    expected_relative_tolerance: float,
    name: str,
) -> dict[str, object]:
    pcg = _required(
        value,
        name=name,
        keys={"iterations", "relative_residual", "converged", "reason"},
    )
    iterations = _integer(pcg["iterations"], name=f"{name}.iterations")
    relative_residual = _finite(
        pcg["relative_residual"],
        name=f"{name}.relative_residual",
        nonnegative=True,
    )
    if pcg["converged"] is not True:
        raise ValueError(f"{name} did not converge")
    if relative_residual > expected_relative_tolerance:
        raise ValueError(f"{name} relative residual exceeds the design-bound PCG tolerance")
    reason = pcg["reason"]
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"{name}.reason must be non-empty")
    return {
        "passed": True,
        "iterations": iterations,
        "relative_residual": relative_residual,
        "design_bound_relative_tolerance": expected_relative_tolerance,
        "reason": reason,
    }


def _validate_environment(value: object, *, name: str) -> dict[str, object]:
    identity = _mapping(value, name=name)
    if set(identity) != _ENVIRONMENT_KEYS:
        raise ValueError(f"{name} must contain the exact formal environment identity fields")
    if identity["formal"] is not True:
        raise ValueError(f"{name} must be formal and clean")
    commit = _digest(identity["git_commit"], name=f"{name}.git_commit", lengths={40, 64})
    image = _digest(identity["image_sha256"], name=f"{name}.image_sha256")
    inventory = _digest(
        identity["hf_inventory_sha256"],
        name=f"{name}.hf_inventory_sha256",
    )
    if identity["account"] != "sigroup":
        raise ValueError(f"{name}.account must equal sigroup")
    partition = identity["partition"]
    if not isinstance(partition, str) or not partition:
        raise ValueError(f"{name}.partition must be non-empty")
    gpu_models = identity["gpu_models"]
    if (
        not isinstance(gpu_models, list)
        or len(gpu_models) != 1
        or not isinstance(gpu_models[0], str)
        or not gpu_models[0]
    ):
        raise ValueError(f"{name} must record exactly one GPU model")
    return {
        "formal": True,
        "git_commit": commit,
        "image_sha256": image,
        "hf_inventory_sha256": inventory,
        "account": "sigroup",
        "partition": partition,
        "gpu_models": list(gpu_models),
    }


def _validate_serialized_head(
    value: object,
    *,
    expected_arm: str,
    expected_method: str,
    expected_objective_name: str,
    expected_weight: Sequence[float] | None,
    expected_outer_steps: int,
    expected_pcg_tolerance: float,
    pcg_required: bool,
    numeric_tolerances: Mapping[str, object],
    adaptive_convergence: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    evidence = _required(
        value,
        name=name,
        keys={
            "arm",
            "method",
            "head_weight",
            "head_dtype",
            "initial_head_sha256",
            "head_sha256",
            "initial_objective",
            "final_objective",
            "history_summary",
            "final_pcg",
            "first_order_convergence",
        },
    )
    _expect(evidence["arm"], expected_arm, name=f"{name}.arm")
    _expect(evidence["method"], expected_method, name=f"{name}.method")
    raw_weight = evidence["head_weight"]
    if not isinstance(raw_weight, list) or not raw_weight:
        raise ValueError(f"{name}.head_weight must be a non-empty list")
    weight = [
        _finite(item, name=f"{name}.head_weight[{index}]") for index, item in enumerate(raw_weight)
    ]
    if expected_weight is not None and weight != list(expected_weight):
        raise ValueError(f"{name}.head_weight does not match the deployed fresh head")
    dtype = evidence["head_dtype"]
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"{name}.head_dtype must be non-empty")
    initial_sha = _digest(
        evidence["initial_head_sha256"],
        name=f"{name}.initial_head_sha256",
    )
    final_sha = _digest(evidence["head_sha256"], name=f"{name}.head_sha256")
    initial_objective = _finite(
        evidence["initial_objective"],
        name=f"{name}.initial_objective",
        nonnegative=True,
    )
    final_objective = _finite(
        evidence["final_objective"],
        name=f"{name}.final_objective",
        nonnegative=True,
    )
    history = _mapping(evidence["history_summary"], name=f"{name}.history_summary")
    if history.get("history_objective_timing") != "pre_update":
        raise ValueError(f"{name} has an invalid optimization-history timing")
    final_pcg: dict[str, object] | None
    if not pcg_required:
        if evidence["final_pcg"] is not None:
            raise ValueError(f"{name} must not contain iterative PCG evidence")
        final_pcg = None
    else:
        final_pcg = _pcg_gate(
            evidence["final_pcg"],
            expected_relative_tolerance=expected_pcg_tolerance,
            name=f"{name}.final_pcg",
        )
    convergence_gate = _validate_first_order_convergence(
        evidence["first_order_convergence"],
        expected_objective_name=expected_objective_name,
        expected_initial_objective=initial_objective,
        expected_final_objective=final_objective,
        expected_head_sha256=final_sha,
        expected_fixed_snapshot_steps=expected_outer_steps,
        numeric_tolerances=numeric_tolerances,
        adaptive_convergence=adaptive_convergence,
        name=f"{name}.first_order_convergence",
    )
    if history.get("num_steps") != convergence_gate["selected_primary_step"]:
        raise ValueError(f"{name} history does not end at the selected primary iterate")
    return {
        "weight": weight,
        "initial_head_sha256": initial_sha,
        "head_sha256": final_sha,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "final_pcg": final_pcg,
        "convergence_gate": convergence_gate,
    }


def _validate_head_training(
    value: object,
    *,
    design_sha256: str,
    seed: int,
    train_oracle_reward_sha256: str,
    expected_train_prompts: int,
    expected_candidates: int,
    expected_outer_steps: int,
    expected_low_dimension: int,
    expected_projection_namespace: str,
    expected_eigenvalue_tolerance: float,
    expected_pcg_tolerance: float,
    prohibit_label_clipping: bool,
    numeric_tolerances: Mapping[str, object],
    adaptive_convergence: Mapping[str, object],
    identifiability_config: Mapping[str, object],
    reward_model_config: Mapping[str, object],
    exact_soft_label_bt_config: Mapping[str, object],
    design_stage: str,
    name: str,
) -> dict[str, object]:
    """Validate every fresh-head/control field and return auditable gate evidence."""

    head = _required(
        value,
        name=name,
        keys={
            "training_arm",
            "training_design_sha256",
            "heads_sha256",
            "head_weights",
            "audit",
            "source",
            "old_phase1_comparison_heads_reused",
            "test_data_accessed",
        },
    )
    _expect(
        head["training_arm"],
        "r4_independent_gamma_0.9",
        name=f"{name}.training_arm",
    )
    _expect(
        head["training_design_sha256"],
        design_sha256,
        name=f"{name}.training_design_sha256",
    )
    _expect(
        head["source"],
        "trained_after_train_oracle_rescore",
        name=f"{name}.source",
    )
    if head["old_phase1_comparison_heads_reused"] is not False:
        raise ValueError(f"{name} reused old Phase-1 heads")
    if head["test_data_accessed"] is not False:
        raise ValueError(f"{name} accessed test data")

    weights = _mapping(head["head_weights"], name=f"{name}.head_weights")
    if set(weights) != set(CANONICAL_LEARNERS):
        raise ValueError(f"{name}.head_weights must contain exactly {CANONICAL_LEARNERS!r}")
    normalized: dict[str, list[float]] = {}
    dimensions: set[int] = set()
    for learner in CANONICAL_LEARNERS:
        raw_weight = weights[learner]
        if not isinstance(raw_weight, list) or not raw_weight:
            raise ValueError(f"{name}.head_weights.{learner} must be a non-empty list")
        weight = [
            _finite(item, name=f"{name}.head_weights.{learner}[{index}]")
            for index, item in enumerate(raw_weight)
        ]
        normalized[learner] = weight
        dimensions.add(len(weight))
    if len(dimensions) != 1:
        raise ValueError(f"{name} learner heads must have the same dimension")
    expected_heads_sha = _canonical_sha256(normalized)
    recorded_heads_sha = _digest(head["heads_sha256"], name=f"{name}.heads_sha256")
    if recorded_heads_sha != expected_heads_sha:
        raise ValueError(f"{name}.heads_sha256 does not match the fresh head weights")

    audit = _required(
        head["audit"],
        name=f"{name}.audit",
        keys={
            "schema_version",
            "training_design_sha256",
            "training_settings_sha256",
            "training_instance_sha256",
            "input_training_sha256",
            "training_arm",
            "absolute_damping",
            "label_stream",
            "primary_heads",
            "primary_optimization_audit",
            "low_dimensional_control",
            "exact_margin_control",
            "exact_soft_label_bt_control",
            "direct_oracle_identity",
            "isolation",
        },
    )
    _expect(
        audit["schema_version"],
        "phase2-fresh-head-training/v2",
        name=f"{name}.audit.schema_version",
    )
    _expect(
        audit["training_design_sha256"],
        design_sha256,
        name=f"{name}.audit.training_design_sha256",
    )
    _expect(
        audit["training_arm"],
        "r4_independent_gamma_0.9",
        name=f"{name}.audit.training_arm",
    )
    training_settings_sha = _digest(
        audit["training_settings_sha256"],
        name=f"{name}.audit.training_settings_sha256",
    )
    training_instance_sha = _digest(
        audit["training_instance_sha256"],
        name=f"{name}.audit.training_instance_sha256",
    )
    input_training_sha = _digest(
        audit["input_training_sha256"],
        name=f"{name}.audit.input_training_sha256",
    )
    absolute_damping = _finite(
        audit["absolute_damping"],
        name=f"{name}.audit.absolute_damping",
    )
    if absolute_damping <= 0.0:
        raise ValueError(f"{name}.audit.absolute_damping must be positive")

    isolation = _mapping(audit["isolation"], name=f"{name}.audit.isolation")
    expected_isolation = {
        "test_data_accessed": False,
        "old_phase1_comparison_heads_used": False,
        "raw_node_rewards_retained": False,
        "raw_labels_retained": False,
        "primary_heads_are_fresh_zero_initialized": True,
    }
    for key, expected in expected_isolation.items():
        _expect(isolation.get(key), expected, name=f"{name}.audit.isolation.{key}")

    label = _required(
        audit["label_stream"],
        name=f"{name}.audit.label_stream",
        keys={
            "namespace",
            "base_seed",
            "derived_seed",
            "derivation_sha256",
            "generator_device",
            "initial_state_sha256",
            "final_state_sha256",
            "oracle_reward_sha256",
            "canonical_probability_sha256",
            "replicate_count_sha256",
            "replicate_win_sha256",
            "replicate_h_sha256",
            "mean_h_sha256",
            "label_stream_sha256",
            "realized_total_annotations",
            "realized_annotations_per_edge",
            "expected_annotations_per_edge",
            "num_edges",
            "num_replicates",
            "gamma",
            "bt_target",
            "prorm_target",
            "raw_labels_retained",
            "raw_node_rewards_retained",
        },
    )
    _expect(
        label["namespace"],
        "prorm-common-beta-r4-labels-v1",
        name=f"{name}.audit.label_stream.namespace",
    )
    _expect(label["base_seed"], seed, name=f"{name}.audit.label_stream.base_seed")
    _integer(
        label["derived_seed"],
        name=f"{name}.audit.label_stream.derived_seed",
    )
    generator_device = label["generator_device"]
    if not isinstance(generator_device, str) or not generator_device:
        raise ValueError(f"{name}.audit.label_stream.generator_device must be non-empty")
    label_digests: dict[str, str] = {}
    for field in (
        "derivation_sha256",
        "initial_state_sha256",
        "final_state_sha256",
        "oracle_reward_sha256",
        "canonical_probability_sha256",
        "replicate_count_sha256",
        "replicate_win_sha256",
        "replicate_h_sha256",
        "mean_h_sha256",
        "label_stream_sha256",
    ):
        label_digests[field] = _digest(
            label[field],
            name=f"{name}.audit.label_stream.{field}",
        )
    _expect(
        label_digests["oracle_reward_sha256"],
        train_oracle_reward_sha256,
        name=f"{name}.audit.label_stream.oracle_reward_sha256",
    )
    if label_digests["initial_state_sha256"] == label_digests["final_state_sha256"]:
        raise ValueError(f"{name}.audit.label_stream generator state did not advance")
    _expect(label["num_replicates"], 4, name=f"{name}.audit.label_stream.num_replicates")
    if not _close(
        _finite(label["gamma"], name=f"{name}.audit.label_stream.gamma"),
        0.9,
        tolerance=1.0e-12,
    ):
        raise ValueError(f"{name}.audit.label_stream.gamma must equal 0.9")
    _expect(
        label["bt_target"],
        "pooled_raw_wins_and_totals",
        name=f"{name}.audit.label_stream.bt_target",
    )
    _expect(
        label["prorm_target"],
        "mean_of_per_replicate_h",
        name=f"{name}.audit.label_stream.prorm_target",
    )
    _expect(
        label["raw_labels_retained"],
        False,
        name=f"{name}.audit.label_stream.raw_labels_retained",
    )
    _expect(
        label["raw_node_rewards_retained"],
        False,
        name=f"{name}.audit.label_stream.raw_node_rewards_retained",
    )
    if "clipping" in label and label["clipping"] is not False:
        raise ValueError(f"{name}.audit.label_stream applies forbidden label clipping")
    if prohibit_label_clipping is not True:
        raise ValueError("the overlay must prohibit repeated-label clipping")
    num_edges = _integer(label["num_edges"], name=f"{name}.audit.label_stream.num_edges")
    _expect(num_edges, expected_train_prompts, name=f"{name}.audit.label_stream.num_edges")
    total_annotations = _integer(
        label["realized_total_annotations"],
        name=f"{name}.audit.label_stream.realized_total_annotations",
        minimum=4 * expected_train_prompts,
    )
    realized_per_edge = _finite(
        label["realized_annotations_per_edge"],
        name=f"{name}.audit.label_stream.realized_annotations_per_edge",
    )
    if not _close(
        realized_per_edge,
        total_annotations / expected_train_prompts,
        tolerance=1.0e-12,
    ):
        raise ValueError(f"{name}.audit.label_stream annotation-cost arithmetic failed")
    if not _close(
        _finite(
            label["expected_annotations_per_edge"],
            name=f"{name}.audit.label_stream.expected_annotations_per_edge",
        ),
        40.0,
        tolerance=1.0e-12,
    ):
        raise ValueError(f"{name}.audit.label_stream expected R=4 cost must equal 40")
    label_payload = {
        "namespace": label["namespace"],
        "base_seed": label["base_seed"],
        "derived_seed": label["derived_seed"],
        "derivation_sha256": label["derivation_sha256"],
        "initial_state_sha256": label["initial_state_sha256"],
        "final_state_sha256": label["final_state_sha256"],
        "probability_sha256": label["canonical_probability_sha256"],
        "replicate_count_sha256": label["replicate_count_sha256"],
        "replicate_win_sha256": label["replicate_win_sha256"],
        "replicate_h_sha256": label["replicate_h_sha256"],
        "mean_h_sha256": label["mean_h_sha256"],
        "realized_total_annotations": total_annotations,
    }
    if _canonical_sha256(label_payload) != label_digests["label_stream_sha256"]:
        raise ValueError(f"{name}.audit.label_stream_sha256 does not bind its payload")

    primary_heads = _mapping(
        audit["primary_heads"],
        name=f"{name}.audit.primary_heads",
    )
    if set(primary_heads) != set(CANONICAL_LEARNERS):
        raise ValueError(f"{name}.audit.primary_heads must contain both learners")
    primary: dict[str, dict[str, object]] = {}
    for learner in CANONICAL_LEARNERS:
        primary[learner] = _validate_serialized_head(
            primary_heads[learner],
            expected_arm="r4_independent_gamma_0.9",
            expected_method=learner,
            expected_objective_name=learner,
            expected_weight=normalized[learner],
            expected_outer_steps=expected_outer_steps,
            expected_pcg_tolerance=expected_pcg_tolerance,
            pcg_required=(learner == PRORM_PLUS),
            numeric_tolerances=numeric_tolerances,
            adaptive_convergence=adaptive_convergence,
            name=f"{name}.audit.primary_heads.{learner}",
        )
    if primary[BT_MLE]["initial_head_sha256"] != primary[PRORM_PLUS]["initial_head_sha256"]:
        raise ValueError(f"{name} primary learners do not share the zero initialization")

    primary_audit = _required(
        audit["primary_optimization_audit"],
        name=f"{name}.audit.primary_optimization_audit",
        keys={
            "geometry",
            "learners",
            "optimizer_constructed",
            "optimizer_step_called",
            "saved_heads_mutated",
            "reward_head_identifiability",
            "prorm_moment_map_identifiability",
        },
    )
    for field in ("optimizer_constructed", "optimizer_step_called", "saved_heads_mutated"):
        _expect(
            primary_audit[field],
            False,
            name=f"{name}.audit.primary_optimization_audit.{field}",
        )
    primary_geometry = _mapping(
        primary_audit["geometry"],
        name=f"{name}.audit.primary_optimization_audit.geometry",
    )
    _expect(
        primary_geometry.get("split"),
        "train",
        name=f"{name}.audit.primary_optimization_audit.geometry.split",
    )
    _expect(
        primary_geometry.get("num_edges"),
        expected_train_prompts,
        name=f"{name}.audit.primary_optimization_audit.geometry.num_edges",
    )
    _expect(
        primary_geometry.get("absolute_damping"),
        absolute_damping,
        name=f"{name}.audit.primary_optimization_audit.geometry.absolute_damping",
    )
    reward_dimension = _integer(
        primary_geometry.get("reward_head_dimension"),
        name=f"{name}.audit.primary_optimization_audit.geometry.reward_head_dimension",
        minimum=1,
    )
    policy_dimension = _integer(
        primary_geometry.get("policy_tangent_dimension"),
        name=f"{name}.audit.primary_optimization_audit.geometry.policy_tangent_dimension",
        minimum=1,
    )
    identifiability, identifiability_sha = _validate_reward_head_identifiability(
        primary_audit["reward_head_identifiability"],
        config=identifiability_config,
        expected_train_prompts=expected_train_prompts,
        expected_reward_dimension=reward_dimension,
        name=f"{name}.audit.primary_optimization_audit.reward_head_identifiability",
    )
    moment_map, moment_map_sha = _validate_prorm_moment_map_identifiability(
        primary_audit["prorm_moment_map_identifiability"],
        config=identifiability_config,
        expected_train_prompts=expected_train_prompts,
        expected_policy_dimension=policy_dimension,
        expected_reward_dimension=reward_dimension,
        name=f"{name}.audit.primary_optimization_audit.prorm_moment_map_identifiability",
    )
    for learner in CANONICAL_LEARNERS:
        expected_rank_sha = identifiability_sha if learner == BT_MLE else moment_map_sha
        _expect(
            _mapping(
                primary[learner]["convergence_gate"],
                name=f"{name}.primary_heads.{learner}.convergence_gate",
            ).get("rank_diagnostic_sha256"),
            expected_rank_sha,
            name=f"{name}.primary_heads.{learner}.rank_diagnostic_sha256",
        )
    primary_learner_audits = _mapping(
        primary_audit["learners"],
        name=f"{name}.audit.primary_optimization_audit.learners",
    )
    if set(primary_learner_audits) != set(CANONICAL_LEARNERS):
        raise ValueError(f"{name} primary optimization audit must contain both learners")
    primary_outer_gates: dict[str, object] = {}
    for learner in CANONICAL_LEARNERS:
        learner_audit = _mapping(
            primary_learner_audits[learner],
            name=f"{name}.audit.primary_optimization_audit.learners.{learner}",
        )
        _expect(
            learner_audit.get("gradient"),
            "full_data_unclipped",
            name=f"{name}.audit.primary_optimization_audit.learners.{learner}.gradient",
        )
        if learner == BT_MLE:
            _expect(
                learner_audit.get("objective_definition"),
                "exact_repeated_label_bt_nll",
                name=f"{name}.audit.primary_optimization_audit.learners.{learner}"
                ".objective_definition",
            )
            _expect(
                learner_audit.get("label_weighting"),
                "each_annotation",
                name=f"{name}.audit.primary_optimization_audit.learners.{learner}.label_weighting",
            )
        else:
            _expect(
                learner_audit.get("objective_definition"),
                "damped_fisher_gmm_dual_loss",
                name=f"{name}.audit.primary_optimization_audit.learners.{learner}"
                ".objective_definition",
            )
            _expect(
                learner_audit.get("gradient_definition"),
                "fresh_dual_envelope_gradient",
                name=f"{name}.audit.primary_optimization_audit.learners.{learner}"
                ".gradient_definition",
            )
            inner_pcg = _mapping(
                learner_audit.get("fresh_inner_pcg"),
                name=f"{name}.audit.primary_optimization_audit.learners.{learner}.fresh_inner_pcg",
            )
            _expect(
                inner_pcg.get("warm_start_used"),
                False,
                name=f"{name}.audit.primary_optimization_audit.learners.{learner}"
                ".fresh_inner_pcg.warm_start_used",
            )
            _pcg_gate(
                inner_pcg,
                expected_relative_tolerance=expected_pcg_tolerance,
                name=f"{name}.audit.primary_optimization_audit.learners.{learner}.fresh_inner_pcg",
            )
        primary_outer_gates[learner] = _objective_decrease_gate(
            initial=primary[learner]["initial_objective"],
            final=primary[learner]["final_objective"],
            audit=learner_audit.get("objective"),
            gradient=learner_audit.get("gradient_l2_norm"),
            convergence_gate=_mapping(
                primary[learner]["convergence_gate"],
                name=f"{name}.primary_heads.{learner}.convergence_gate",
            ),
            numeric_tolerances=numeric_tolerances,
            name=f"{name}.primary_outer_convergence.{learner}",
        )

    exact = _required(
        audit["exact_margin_control"],
        name=f"{name}.audit.exact_margin_control",
        keys={
            "head",
            "target_audit",
            "optimization_audit",
            "reward_class_and_optimizer_gap",
        },
    )
    exact_target = _mapping(
        exact["target_audit"],
        name=f"{name}.audit.exact_margin_control.target_audit",
    )
    expected_exact_target = {
        "schema_version": "exact-margin-audit/v1",
        "orientation": "candidate_0_minus_candidate_1",
        "raw_node_rewards_retained": False,
        "bt_counts_source": "input_training_passthrough",
        "purpose": "zero_label_noise_reward_head_training_control",
        "reward_head_fit_required": True,
        "oracle_direction_identity_expected": False,
    }
    for field, expected in expected_exact_target.items():
        _expect(
            exact_target.get(field),
            expected,
            name=f"{name}.audit.exact_margin_control.target_audit.{field}",
        )
    exact_source_sha = _digest(
        exact_target.get("source_node_rewards_sha256"),
        name=f"{name}.audit.exact_margin_control.target_audit.source_node_rewards_sha256",
    )
    _expect(
        exact_source_sha,
        train_oracle_reward_sha256,
        name=f"{name}.audit.exact_margin_control.target_audit.source_node_rewards_sha256",
    )
    exact_margin_sha = _digest(
        exact_target.get("exact_margin_sha256"),
        name=f"{name}.audit.exact_margin_control.target_audit.exact_margin_sha256",
    )
    _expect(
        exact_target.get("source_shape"),
        [expected_train_prompts, expected_candidates],
        name=f"{name}.audit.exact_margin_control.target_audit.source_shape",
    )
    exact_head = _validate_serialized_head(
        exact["head"],
        expected_arm="exact_margin_positive_control",
        expected_method=PRORM_PLUS,
        expected_objective_name="exact_margin_prorm_plus",
        expected_weight=None,
        expected_outer_steps=expected_outer_steps,
        expected_pcg_tolerance=expected_pcg_tolerance,
        pcg_required=True,
        numeric_tolerances=numeric_tolerances,
        adaptive_convergence=adaptive_convergence,
        name=f"{name}.audit.exact_margin_control.head",
    )
    _expect(
        exact_head["initial_head_sha256"],
        primary[PRORM_PLUS]["initial_head_sha256"],
        name=f"{name}.audit.exact_margin_control.head.initial_head_sha256",
    )
    exact_optimization = _mapping(
        exact["optimization_audit"],
        name=f"{name}.audit.exact_margin_control.optimization_audit",
    )
    _expect(
        exact_optimization.get("bt_audit_discarded"),
        True,
        name=f"{name}.audit.exact_margin_control.optimization_audit.bt_audit_discarded",
    )
    for field in ("optimizer_constructed", "optimizer_step_called"):
        _expect(
            exact_optimization.get(field),
            False,
            name=f"{name}.audit.exact_margin_control.optimization_audit.{field}",
        )
    exact_learner_audit = _mapping(
        exact_optimization.get("learner"),
        name=f"{name}.audit.exact_margin_control.optimization_audit.learner",
    )
    _expect(
        exact_learner_audit.get("gradient"),
        "full_data_unclipped",
        name=f"{name}.audit.exact_margin_control.optimization_audit.learner.gradient",
    )
    _expect(
        exact_learner_audit.get("gradient_definition"),
        "fresh_dual_envelope_gradient",
        name=f"{name}.audit.exact_margin_control.optimization_audit.learner.gradient_definition",
    )
    exact_inner_pcg = _mapping(
        exact_learner_audit.get("fresh_inner_pcg"),
        name=f"{name}.audit.exact_margin_control.optimization_audit.learner.fresh_inner_pcg",
    )
    _expect(
        exact_inner_pcg.get("warm_start_used"),
        False,
        name=f"{name}.audit.exact_margin_control.optimization_audit.learner"
        ".fresh_inner_pcg.warm_start_used",
    )
    _pcg_gate(
        exact_inner_pcg,
        expected_relative_tolerance=expected_pcg_tolerance,
        name=f"{name}.audit.exact_margin_control.optimization_audit.learner.fresh_inner_pcg",
    )
    exact_outer_gate = _objective_decrease_gate(
        initial=exact_head["initial_objective"],
        final=exact_head["final_objective"],
        audit=exact_learner_audit.get("objective"),
        gradient=exact_learner_audit.get("gradient_l2_norm"),
        convergence_gate=_mapping(
            exact_head["convergence_gate"],
            name=f"{name}.exact_margin_control.head.convergence_gate",
        ),
        numeric_tolerances=numeric_tolerances,
        name=f"{name}.exact_margin_outer_convergence",
    )
    _expect(
        _mapping(
            exact_head["convergence_gate"],
            name=f"{name}.exact_margin_control.head.convergence_gate",
        ).get("rank_diagnostic_sha256"),
        moment_map_sha,
        name=f"{name}.exact_margin_control.head.rank_diagnostic_sha256",
    )
    exact_gap = _mapping(
        exact["reward_class_and_optimizer_gap"],
        name=f"{name}.audit.exact_margin_control.reward_class_and_optimizer_gap",
    )
    if "restricted_reward_class_and_finite_optimizer_gap" not in str(
        exact_gap.get("interpretation")
    ):
        raise ValueError(f"{name} exact-margin control misstates its interpretation")
    _expect(
        exact_gap.get("algebraic_identity_claimed"),
        False,
        name=f"{name}.audit.exact_margin_control.reward_class_and_optimizer_gap"
        ".algebraic_identity_claimed",
    )
    _expect(
        exact_gap.get("raw_node_rewards_retained"),
        False,
        name=f"{name}.audit.exact_margin_control.reward_class_and_optimizer_gap"
        ".raw_node_rewards_retained",
    )
    exact_direction_pcg = _pcg_gate(
        exact_gap.get("trained_direction_pcg"),
        expected_relative_tolerance=expected_pcg_tolerance,
        name=f"{name}.audit.exact_margin_control.reward_class_and_optimizer_gap"
        ".trained_direction_pcg",
    )

    direct = _mapping(
        audit["direct_oracle_identity"],
        name=f"{name}.audit.direct_oracle_identity",
    )
    expected_direct = {
        "schema_version": "direct-oracle-exact-moment-identity/v1",
        "interpretation": "algebraic_identity_bypasses_reward_class_and_optimizer",
        "num_prompts": expected_train_prompts,
        "num_candidates": expected_candidates,
        "complete_pair_identity_is_algebraic": True,
        "reward_head_bypassed": True,
        "optimizer_bypassed": True,
        "trained_exact_margin_head_required_to_match": False,
        "raw_node_rewards_retained": False,
    }
    for field, expected in expected_direct.items():
        _expect(
            direct.get(field),
            expected,
            name=f"{name}.audit.direct_oracle_identity.{field}",
        )
    direct_source_sha = _digest(
        direct.get("source_node_rewards_sha256"),
        name=f"{name}.audit.direct_oracle_identity.source_node_rewards_sha256",
    )
    _expect(
        direct_source_sha,
        train_oracle_reward_sha256,
        name=f"{name}.audit.direct_oracle_identity.source_node_rewards_sha256",
    )
    for field in (
        "canonical_margin_sha256",
        "canonical_pair_moment_sha256",
        "complete_pair_u_stat_moment_sha256",
        "all_node_covariance_moment_sha256",
    ):
        _digest(direct.get(field), name=f"{name}.audit.direct_oracle_identity.{field}")
    direct_absolute_error = _finite(
        direct.get("complete_pair_identity_absolute_error"),
        name=f"{name}.audit.direct_oracle_identity.complete_pair_identity_absolute_error",
        nonnegative=True,
    )
    direct_relative_error = _finite(
        direct.get("complete_pair_identity_relative_error"),
        name=f"{name}.audit.direct_oracle_identity.complete_pair_identity_relative_error",
        nonnegative=True,
    )
    direct_absolute_tolerance = _finite(
        numeric_tolerances["direct_identity_absolute_error"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.direct_identity_absolute_error",
    )
    direct_relative_tolerance = _finite(
        numeric_tolerances["direct_identity_relative_error"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.direct_identity_relative_error",
    )
    if direct_absolute_error > direct_absolute_tolerance or (
        direct_relative_error > direct_relative_tolerance
    ):
        raise ValueError(f"{name} direct pair-node moment identity failed")
    native_direction = _mapping(
        direct.get("native_oracle_direction"),
        name=f"{name}.audit.direct_oracle_identity.native_oracle_direction",
    )
    direct_direction_sha = _digest(
        native_direction.get("direction_sha256"),
        name=f"{name}.audit.direct_oracle_identity.native_oracle_direction.direction_sha256",
    )
    if not _close(
        _finite(
            native_direction.get("absolute_damping"),
            name=f"{name}.audit.direct_oracle_identity.native_oracle_direction.absolute_damping",
        ),
        absolute_damping,
        tolerance=1.0e-12,
    ):
        raise ValueError(f"{name} direct-oracle direction uses a different Fisher damping")
    if (
        _finite(
            native_direction.get("moment_norm"),
            name=f"{name}.audit.direct_oracle_identity.native_oracle_direction.moment_norm",
        )
        <= 0.0
    ):
        raise ValueError(f"{name} direct-oracle moment must be non-zero")
    direct_pcg = _pcg_gate(
        native_direction.get("pcg"),
        expected_relative_tolerance=expected_pcg_tolerance,
        name=f"{name}.audit.direct_oracle_identity.native_oracle_direction.pcg",
    )

    low = _mapping(
        audit["low_dimensional_control"],
        name=f"{name}.audit.low_dimensional_control",
    )
    expected_low = {
        "schema_version": "low-dimensional-tangent-training-control/v1",
        "enabled": True,
        "eligible_for_primary_claim": False,
        "training_arm": "r4_independent_gamma_0.9",
        "target": "same_r4_mean_h_as_primary_prorm_plus",
        "fresh_zero_initialized": True,
        "raw_labels_retained": False,
        "raw_node_rewards_retained": False,
    }
    for field, expected in expected_low.items():
        _expect(low.get(field), expected, name=f"{name}.audit.low_dimensional_control.{field}")
    if "positive_control_only" not in str(low.get("interpretation")):
        raise ValueError(f"{name} low-dimensional arm is not marked positive-control-only")
    _expect(
        low.get("label_stream_sha256"),
        label_digests["label_stream_sha256"],
        name=f"{name}.audit.low_dimensional_control.label_stream_sha256",
    )
    low_bt = _mapping(low.get("bt_head"), name=f"{name}.audit.low_dimensional_control.bt_head")
    _expect(
        low_bt.get("head_sha256"),
        primary[BT_MLE]["head_sha256"],
        name=f"{name}.audit.low_dimensional_control.bt_head.head_sha256",
    )
    _expect(
        low_bt.get("retrained"),
        False,
        name=f"{name}.audit.low_dimensional_control.bt_head.retrained",
    )
    projection = _mapping(
        low.get("projection"),
        name=f"{name}.audit.low_dimensional_control.projection",
    )
    expected_projection = {
        "schema_version": "seeded-orthonormal-tangent/v1",
        "selected_dimension": expected_low_dimension,
        "namespace": expected_projection_namespace,
        "declared_seed": seed,
        "algorithm": "gaussian_qr_sign_canonical_v1",
        "projection_dtype": "torch.float64",
        "score_construction": "S_low = cast_fp32(cast_fp64(S_full) @ P_fp64)",
        "deployment_scatter": "u_full = P @ u_low",
        "orthonormal_columns": True,
        "strictly_below_fisher_node_count": True,
    }
    for field, expected in expected_projection.items():
        _expect(
            projection.get(field),
            expected,
            name=f"{name}.audit.low_dimensional_control.projection.{field}",
        )
    orthonormality_error = _finite(
        projection.get("orthonormality_max_absolute_error"),
        name=f"{name}.audit.low_dimensional_control.projection.orthonormality_max_absolute_error",
        nonnegative=True,
    )
    orthonormality_tolerance = _finite(
        numeric_tolerances["low_dimensional_orthonormality_max_absolute_error"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.low_dimensional_orthonormality_max_absolute_error",
    )
    if orthonormality_error > orthonormality_tolerance:
        raise ValueError(f"{name} low-dimensional orthonormality numeric gate failed")
    source_dimension = _integer(
        projection.get("source_dimension"),
        name=f"{name}.audit.low_dimensional_control.projection.source_dimension",
        minimum=expected_low_dimension + 1,
    )
    num_fisher_nodes = _integer(
        projection.get("num_fisher_nodes"),
        name=f"{name}.audit.low_dimensional_control.projection.num_fisher_nodes",
        minimum=expected_low_dimension + 1,
    )
    _expect(
        num_fisher_nodes,
        expected_train_prompts * expected_candidates,
        name=f"{name}.audit.low_dimensional_control.projection.num_fisher_nodes",
    )
    _integer(
        projection.get("effective_seed"),
        name=f"{name}.audit.low_dimensional_control.projection.effective_seed",
    )
    projection_sha = _digest(
        projection.get("projection_sha256"),
        name=f"{name}.audit.low_dimensional_control.projection.projection_sha256",
    )
    source_layout_id = projection.get("source_layout_id")
    if not isinstance(source_layout_id, str) or not source_layout_id:
        raise ValueError(f"{name} low-dimensional source layout ID is missing")

    low_geometry = _mapping(
        low.get("geometry"),
        name=f"{name}.audit.low_dimensional_control.geometry",
    )
    expected_low_geometry = {
        "regularization": "moore_penrose_pseudoinverse",
        "ridge_enabled": False,
        "ridge_coefficient": 0.0,
        "solver": "torch.linalg.eigh_truncated_moore_penrose",
        "solver_dtype": "float64",
        "selected_dimension": expected_low_dimension,
        "numerical_rank": expected_low_dimension,
        "relative_eigenvalue_tolerance": expected_eigenvalue_tolerance,
        "pcg_used": False,
    }
    for field, expected in expected_low_geometry.items():
        _expect(
            low_geometry.get(field),
            expected,
            name=f"{name}.audit.low_dimensional_control.geometry.{field}",
        )
    smallest_retained = _finite(
        low_geometry.get("smallest_retained_eigenvalue"),
        name=f"{name}.audit.low_dimensional_control.geometry.smallest_retained_eigenvalue",
    )
    largest_retained = _finite(
        low_geometry.get("largest_retained_eigenvalue"),
        name=f"{name}.audit.low_dimensional_control.geometry.largest_retained_eigenvalue",
    )
    if smallest_retained <= 0.0 or largest_retained < smallest_retained:
        raise ValueError(f"{name} low-dimensional retained Fisher spectrum is invalid")
    low_fisher_sha = _digest(
        low_geometry.get("fisher_sha256"),
        name=f"{name}.audit.low_dimensional_control.geometry.fisher_sha256",
    )
    low_pseudoinverse_sha = _digest(
        low_geometry.get("pseudoinverse_sha256"),
        name=f"{name}.audit.low_dimensional_control.geometry.pseudoinverse_sha256",
    )
    projected_moment_map, projected_moment_map_sha = _validate_prorm_moment_map_identifiability(
        low.get("projected_prorm_moment_map_identifiability"),
        config=identifiability_config,
        expected_train_prompts=expected_train_prompts,
        expected_policy_dimension=expected_low_dimension,
        expected_reward_dimension=reward_dimension,
        expected_projection_sha256=projection_sha,
        expected_projected_geometry={
            "fisher_sha256": low_fisher_sha,
            "pseudoinverse_sha256": low_pseudoinverse_sha,
            "relative_eigenvalue_tolerance": expected_eigenvalue_tolerance,
        },
        name=f"{name}.audit.low_dimensional_control.projected_prorm_moment_map_identifiability",
    )
    low_head = _validate_serialized_head(
        low.get("head"),
        expected_arm="low_dimensional_tangent_positive_control",
        expected_method=PRORM_PLUS,
        expected_objective_name="low_dimensional_prorm_plus",
        expected_weight=None,
        expected_outer_steps=expected_outer_steps,
        expected_pcg_tolerance=expected_pcg_tolerance,
        pcg_required=False,
        numeric_tolerances=numeric_tolerances,
        adaptive_convergence=adaptive_convergence,
        name=f"{name}.audit.low_dimensional_control.head",
    )
    _expect(
        low_head["initial_head_sha256"],
        primary[PRORM_PLUS]["initial_head_sha256"],
        name=f"{name}.audit.low_dimensional_control.head.initial_head_sha256",
    )
    low_audit = _mapping(
        low.get("final_full_data_audit"),
        name=f"{name}.audit.low_dimensional_control.final_full_data_audit",
    )
    for field in ("optimizer_constructed", "optimizer_step_called", "saved_head_mutated"):
        _expect(
            low_audit.get(field),
            False,
            name=f"{name}.audit.low_dimensional_control.final_full_data_audit.{field}",
        )
    _expect(
        low_audit.get("gradient"),
        "full_data_unclipped",
        name=f"{name}.audit.low_dimensional_control.final_full_data_audit.gradient",
    )
    low_outer_gate = _objective_decrease_gate(
        initial=low_head["initial_objective"],
        final=low_head["final_objective"],
        audit=low_audit.get("objective"),
        gradient=low_audit.get("gradient_l2_norm"),
        convergence_gate=_mapping(
            low_head["convergence_gate"],
            name=f"{name}.low_dimensional_control.head.convergence_gate",
        ),
        numeric_tolerances=numeric_tolerances,
        name=f"{name}.low_dimensional_outer_convergence",
    )
    _expect(
        _mapping(
            low_head["convergence_gate"],
            name=f"{name}.low_dimensional_control.head.convergence_gate",
        ).get("rank_diagnostic_sha256"),
        projected_moment_map_sha,
        name=f"{name}.low_dimensional_control.head.rank_diagnostic_sha256",
    )
    low_relative_residual = _finite(
        low_audit.get("pseudoinverse_solve_relative_residual"),
        name=f"{name}.audit.low_dimensional_control.final_full_data_audit"
        ".pseudoinverse_solve_relative_residual",
        nonnegative=True,
    )
    pseudoinverse_tolerance = _finite(
        numeric_tolerances["low_dimensional_pseudoinverse_relative_residual"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.low_dimensional_pseudoinverse_relative_residual",
    )
    if low_relative_residual > pseudoinverse_tolerance:
        raise ValueError(f"{name} low-dimensional pseudoinverse residual gate failed")
    low_selected_direction_sha = _digest(
        low_audit.get("selected_direction_sha256"),
        name=f"{name}.audit.low_dimensional_control.final_full_data_audit"
        ".selected_direction_sha256",
    )
    score_identity = _mapping(
        low.get("deployment_score_identity"),
        name=f"{name}.audit.low_dimensional_control.deployment_score_identity",
    )
    _expect(
        score_identity.get("formula"),
        "(S_full @ P) @ u_low == S_full @ (P @ u_low)",
        name=f"{name}.audit.low_dimensional_control.deployment_score_identity.formula",
    )
    _expect(
        score_identity.get("passed"),
        True,
        name=f"{name}.audit.low_dimensional_control.deployment_score_identity.passed",
    )
    _expect(
        score_identity.get("selected_direction_sha256"),
        low_selected_direction_sha,
        name=f"{name}.audit.low_dimensional_control.deployment_score_identity"
        ".selected_direction_sha256",
    )
    for field in (
        "scattered_full_direction_sha256",
        "low_projected_score_sha256",
        "full_projected_score_sha256",
    ):
        _digest(
            score_identity.get(field),
            name=f"{name}.audit.low_dimensional_control.deployment_score_identity.{field}",
        )
    scatter_error = _finite(
        score_identity.get("max_absolute_error"),
        name=f"{name}.audit.low_dimensional_control.deployment_score_identity.max_absolute_error",
        nonnegative=True,
    )
    serialized_scatter_tolerance = _finite(
        score_identity.get("absolute_tolerance"),
        name=f"{name}.audit.low_dimensional_control.deployment_score_identity.absolute_tolerance",
    )
    scatter_tolerance = _finite(
        numeric_tolerances["low_dimensional_scatter_max_absolute_error"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.low_dimensional_scatter_max_absolute_error",
    )
    score_identity_tolerance = _finite(
        numeric_tolerances["low_dimensional_score_identity_max_absolute_error"],
        name=f"{_NUMERIC_GATE_CONFIG_PATH}.low_dimensional_score_identity_max_absolute_error",
    )
    if serialized_scatter_tolerance <= 0.0 or scatter_error > min(
        serialized_scatter_tolerance,
        scatter_tolerance,
        score_identity_tolerance,
    ):
        raise ValueError(f"{name} low-dimensional projection/scatter identity failed")

    exact_soft = _mapping(
        audit["exact_soft_label_bt_control"],
        name=f"{name}.audit.exact_soft_label_bt_control",
    )
    if set(exact_soft) != {"head", "target_audit", "optimization_audit"}:
        raise ValueError(
            f"{name}.audit.exact_soft_label_bt_control must contain exactly "
            "head, target_audit, and optimization_audit"
        )
    exact_soft_head = _validate_serialized_head(
        exact_soft["head"],
        expected_arm="exact_soft_label_bt_secondary_diagnostic",
        expected_method=BT_MLE,
        expected_objective_name="exact_soft_label_bt_cross_entropy",
        expected_weight=None,
        expected_outer_steps=expected_outer_steps,
        expected_pcg_tolerance=expected_pcg_tolerance,
        pcg_required=False,
        numeric_tolerances=numeric_tolerances,
        adaptive_convergence=adaptive_convergence,
        name=f"{name}.audit.exact_soft_label_bt_control.head",
    )
    _expect(
        exact_soft_head["initial_head_sha256"],
        primary[BT_MLE]["initial_head_sha256"],
        name=f"{name}.audit.exact_soft_label_bt_control.head.initial_head_sha256",
    )
    _expect(
        _mapping(
            exact_soft_head["convergence_gate"],
            name=f"{name}.exact_soft_label_bt_control.head.convergence_gate",
        ).get("rank_diagnostic_sha256"),
        identifiability_sha,
        name=f"{name}.exact_soft_label_bt_control.head.rank_diagnostic_sha256",
    )

    exact_soft_target_keys = {
        "schema_version",
        "split",
        "orientation",
        "input",
        "target_construction",
        "source_node_rewards_sha256",
        "canonical_margin_sha256",
        "target_probability_sha256",
        "reward_feature_difference_sha256",
        "num_canonical_edges",
        "reward_dimension",
        "same_reward_features_and_canonical_edges_as",
        "noise_free",
        "bernoulli_sampling_used",
        "sampled_label_stream_accessed",
        "raw_target_probabilities_retained",
        "raw_oracle_margins_retained",
        "raw_node_rewards_retained",
        "test_or_validation_data_accessed",
        "role",
        "eligible_for_primary_claim",
    }
    exact_soft_target = _mapping(
        exact_soft["target_audit"],
        name=f"{name}.audit.exact_soft_label_bt_control.target_audit",
    )
    if set(exact_soft_target) != exact_soft_target_keys:
        raise ValueError(f"{name} exact-soft-label BT target audit has an invalid field set")
    expected_exact_soft_target = {
        "schema_version": "exact-soft-label-bt-target/v1",
        "split": "train",
        "orientation": "candidate_0_minus_candidate_1",
        "input": exact_soft_label_bt_config["input"],
        "target_construction": "p_star = sigmoid(delta_r_star)",
        "source_node_rewards_sha256": train_oracle_reward_sha256,
        "canonical_margin_sha256": exact_margin_sha,
        "num_canonical_edges": expected_train_prompts,
        "reward_dimension": reward_dimension,
        "same_reward_features_and_canonical_edges_as": "exact_margin_prorm_plus",
        "noise_free": exact_soft_label_bt_config["noise_free"],
        "bernoulli_sampling_used": False,
        "sampled_label_stream_accessed": False,
        "raw_target_probabilities_retained": False,
        "raw_oracle_margins_retained": False,
        "raw_node_rewards_retained": False,
        "test_or_validation_data_accessed": False,
        "role": exact_soft_label_bt_config["role"],
        "eligible_for_primary_claim": exact_soft_label_bt_config["eligible_for_primary_claim"],
    }
    for field, expected in expected_exact_soft_target.items():
        _expect(
            exact_soft_target[field],
            expected,
            name=f"{name}.audit.exact_soft_label_bt_control.target_audit.{field}",
        )
    exact_soft_target_probability_sha = _digest(
        exact_soft_target["target_probability_sha256"],
        name=f"{name}.audit.exact_soft_label_bt_control.target_probability_sha256",
    )
    exact_soft_feature_sha = _digest(
        exact_soft_target["reward_feature_difference_sha256"],
        name=f"{name}.audit.exact_soft_label_bt_control.reward_feature_difference_sha256",
    )

    exact_soft_optimization_keys = {
        "schema_version",
        "objective",
        "objective_name",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "microbatch_size",
        "max_grad_norm",
        "fresh_zero_initialized_bias_free_linear_head",
        "head_sha256",
        "target_probability_sha256",
        "reward_feature_difference_sha256",
        "initial_objective",
        "final_objective",
        "objective_change_final_minus_initial",
        "final_full_data_unclipped_gradient_l2_norm",
        "final_gradient_ratio_to_zero_initialization",
        "first_order_convergence_passed",
        "fixed_720_step_checkpoint_role",
        "fixed_720_step_checkpoint_used_for_head_selection",
        "favorable_ordering_gate_applied",
        "pilot_measure_only",
        "eligible_for_primary_claim",
        "sampled_label_stream_accessed",
        "test_or_validation_data_accessed",
        "saved_head_mutated_by_audit",
    }
    exact_soft_optimization = _mapping(
        exact_soft["optimization_audit"],
        name=f"{name}.audit.exact_soft_label_bt_control.optimization_audit",
    )
    if set(exact_soft_optimization) != exact_soft_optimization_keys:
        raise ValueError(f"{name} exact-soft-label BT optimization audit has an invalid field set")
    expected_exact_soft_optimization = {
        "schema_version": "exact-soft-label-bt-optimization/v1",
        "objective": "mean(softplus(delta_r_phi) - p_star * delta_r_phi)",
        "objective_name": "exact_soft_label_bt_cross_entropy",
        "optimizer": reward_model_config["optimizer"],
        "learning_rate": reward_model_config["learning_rate"],
        "weight_decay": reward_model_config["weight_decay"],
        "microbatch_size": reward_model_config["microbatch_size"],
        "max_grad_norm": reward_model_config["max_grad_norm"],
        "fresh_zero_initialized_bias_free_linear_head": True,
        "head_sha256": exact_soft_head["head_sha256"],
        "target_probability_sha256": exact_soft_target_probability_sha,
        "reward_feature_difference_sha256": exact_soft_feature_sha,
        "first_order_convergence_passed": True,
        "fixed_720_step_checkpoint_role": "compute_matched_and_pilot_diagnostic_only",
        "fixed_720_step_checkpoint_used_for_head_selection": False,
        "favorable_ordering_gate_applied": False,
        "pilot_measure_only": design_stage == "pilot",
        "eligible_for_primary_claim": False,
        "sampled_label_stream_accessed": False,
        "test_or_validation_data_accessed": False,
        "saved_head_mutated_by_audit": False,
    }
    for field, expected in expected_exact_soft_optimization.items():
        _expect(
            exact_soft_optimization[field],
            expected,
            name=f"{name}.audit.exact_soft_label_bt_control.optimization_audit.{field}",
        )
    exact_soft_initial = _finite(
        exact_soft_optimization["initial_objective"],
        name=f"{name}.audit.exact_soft_label_bt_control.initial_objective",
        nonnegative=True,
    )
    exact_soft_final = _finite(
        exact_soft_optimization["final_objective"],
        name=f"{name}.audit.exact_soft_label_bt_control.final_objective",
        nonnegative=True,
    )
    _expect(
        exact_soft_initial,
        exact_soft_head["initial_objective"],
        name=f"{name}.audit.exact_soft_label_bt_control.initial_objective",
    )
    _expect(
        exact_soft_final,
        exact_soft_head["final_objective"],
        name=f"{name}.audit.exact_soft_label_bt_control.final_objective",
    )
    exact_soft_objective_change = _finite(
        exact_soft_optimization["objective_change_final_minus_initial"],
        name=f"{name}.audit.exact_soft_label_bt_control.objective_change",
    )
    if not math.isclose(
        exact_soft_objective_change,
        exact_soft_final - exact_soft_initial,
        rel_tol=1.0e-10,
        abs_tol=1.0e-14,
    ):
        raise ValueError(f"{name} exact-soft-label BT objective-change arithmetic failed")
    exact_soft_outer_gate = _objective_decrease_gate(
        initial=exact_soft_initial,
        final=exact_soft_final,
        audit=exact_soft_final,
        gradient=exact_soft_optimization["final_full_data_unclipped_gradient_l2_norm"],
        convergence_gate=_mapping(
            exact_soft_head["convergence_gate"],
            name=f"{name}.exact_soft_label_bt_control.head.convergence_gate",
        ),
        numeric_tolerances=numeric_tolerances,
        name=f"{name}.exact_soft_label_bt_outer_convergence",
    )
    recorded_exact_soft_ratio = _finite(
        exact_soft_optimization["final_gradient_ratio_to_zero_initialization"],
        name=f"{name}.audit.exact_soft_label_bt_control.final_gradient_ratio",
        nonnegative=True,
    )
    if not math.isclose(
        recorded_exact_soft_ratio,
        float(exact_soft_head["convergence_gate"]["gradient_ratio_to_zero_initialization"]),
        rel_tol=1.0e-10,
        abs_tol=1.0e-14,
    ):
        raise ValueError(f"{name} exact-soft-label BT gradient-ratio audit is inconsistent")

    recomputed_training_instance_sha = _canonical_sha256(
        {
            "schema_version": "phase2-training-instance/v1",
            "phase2_config_hash": design_sha256,
            "settings_sha256": training_settings_sha,
            "input_training_sha256": input_training_sha,
            "oracle_reward_sha256": train_oracle_reward_sha256,
            "seed": seed,
            "label_stream_sha256": label_digests["label_stream_sha256"],
            "reward_head_identifiability_sha256": identifiability_sha,
            "prorm_moment_map_identifiability_sha256": moment_map_sha,
            "bt_head_sha256": primary[BT_MLE]["head_sha256"],
            "prorm_plus_head_sha256": primary[PRORM_PLUS]["head_sha256"],
            "low_dimensional_head_sha256": low_head["head_sha256"],
            "low_dimensional_projection_sha256": projection_sha,
            "low_dimensional_moment_map_identifiability_sha256": (projected_moment_map_sha),
            "exact_margin_head_sha256": exact_head["head_sha256"],
            "exact_soft_label_bt_head_sha256": exact_soft_head["head_sha256"],
            "direct_oracle_direction_sha256": direct_direction_sha,
        }
    )
    if recomputed_training_instance_sha != training_instance_sha:
        raise ValueError(f"{name}.audit.training_instance_sha256 does not bind all controls")

    gate_evidence = {
        "schema_version": "phase2-head-control-gates/v1",
        "passed": True,
        "r4_repeated_labels": {
            "passed": True,
            "replicates": 4,
            "gamma": 0.9,
            "independence_evidence": (
                "named_generator_stream_plus_phase2_training_schema_r4_sequential_draw_contract"
            ),
            "replicate_tensor_digests_present": True,
            "label_clipping_prohibited_by_overlay": True,
            "label_clipping_operator_in_audit": False,
            "prorm_reduction": "arithmetic_mean_of_four_per_replicate_h",
            "bt_label_routing": "pooled_raw_wins_and_totals_each_annotation",
            "replicate_boundaries_reused_as_one_truncation": False,
            "label_stream_sha256": label_digests["label_stream_sha256"],
        },
        "direct_pair_node_identity_and_solver": {
            "passed": True,
            "absolute_error": direct_absolute_error,
            "relative_error": direct_relative_error,
            "design_bound_absolute_tolerance": direct_absolute_tolerance,
            "design_bound_relative_tolerance": direct_relative_tolerance,
            "numeric_tolerance_design_bound": True,
            "pcg": direct_pcg,
        },
        "exact_margin_target_and_optimization": {
            "passed": True,
            "target": "transformed_oracle_reward_difference_candidate_0_minus_candidate_1",
            "source_node_rewards_sha256": exact_source_sha,
            "exact_margin_sha256": exact_margin_sha,
            "objective_and_outer_convergence": exact_outer_gate,
            "trained_direction_pcg": exact_direction_pcg,
            "restricted_reward_class_gap_is_not_identity_failure": True,
        },
        "exact_soft_label_bt_secondary_diagnostic": {
            "passed": True,
            "noise_free": True,
            "sampled_label_stream_accessed": False,
            "source_node_rewards_sha256": train_oracle_reward_sha256,
            "canonical_margin_sha256": exact_margin_sha,
            "target_probability_sha256": exact_soft_target_probability_sha,
            "reward_feature_difference_sha256": exact_soft_feature_sha,
            "objective_and_outer_convergence": exact_soft_outer_gate,
            "favorable_ordering_gate_applied": False,
            "eligible_for_primary_claim": False,
        },
        "low_dimensional_geometry": {
            "passed": True,
            "selected_dimension": expected_low_dimension,
            "source_dimension": source_dimension,
            "numerical_rank": expected_low_dimension,
            "orthonormal_columns": True,
            "orthonormality_max_absolute_error": orthonormality_error,
            "design_bound_orthonormality_tolerance": orthonormality_tolerance,
            "regularization": "moore_penrose_pseudoinverse",
            "pseudoinverse_relative_residual": low_relative_residual,
            "design_bound_pseudoinverse_relative_residual_tolerance": (pseudoinverse_tolerance),
            "scatter_max_absolute_error": scatter_error,
            "design_bound_scatter_max_absolute_error_tolerance": scatter_tolerance,
            "design_bound_score_identity_max_absolute_error_tolerance": (score_identity_tolerance),
            "numeric_tolerance_design_bound": True,
            "objective_and_outer_convergence": low_outer_gate,
            "projected_moment_map_identifiability": {
                "evidence_sha256": projected_moment_map_sha,
                "projection_sha256": projection_sha,
                "shape": projected_moment_map["shape"],
                "numerical_rank": projected_moment_map["numerical_rank"],
                "column_dimension": projected_moment_map["column_dimension"],
                "full_column_rank": projected_moment_map["full_column_rank"],
                "row_dimension": projected_moment_map["row_dimension"],
                "full_row_rank": projected_moment_map["full_row_rank"],
                "require_full_row_rank": True,
                "acceptance_gate_passed": projected_moment_map["acceptance_gate_passed"],
                "unique_projected_prorm_quadratic_head_iff_full_column_rank": True,
                "observed_unique_projected_prorm_quadratic_head": (
                    projected_moment_map["observed_unique_projected_prorm_quadratic_head"]
                ),
                "population_identifiability_theorem_claimed": False,
                "minimum_norm_solution_claimed": False,
            },
        },
        "primary_outer_convergence": {
            "passed": True,
            "gradient": "cold_full_data_unclipped",
            "learners": primary_outer_gates,
            "numeric_tolerance_design_bound": True,
        },
        "training_instance_hash_chain": {
            "passed": True,
            "training_instance_sha256": training_instance_sha,
            "binds_primary_heads_labels_and_all_controls": True,
        },
        "reward_head_identifiability": {
            "passed": True,
            "evidence_sha256": identifiability_sha,
            "role": identifiability["role"],
            "numerical_rank": identifiability["numerical_rank"],
            "column_dimension": identifiability["column_dimension"],
            "full_column_rank": identifiability["full_column_rank"],
            "require_full_column_rank": identifiability["require_full_column_rank"],
            "acceptance_gate_passed": identifiability["acceptance_gate_passed"],
            "prorm_moment_map_full_rank_proved": False,
        },
        "prorm_moment_map_identifiability": {
            "passed": True,
            "evidence_sha256": moment_map_sha,
            "role": moment_map["role"],
            "shape": moment_map["shape"],
            "numerical_rank": moment_map["numerical_rank"],
            "column_dimension": moment_map["column_dimension"],
            "full_column_rank": moment_map["full_column_rank"],
            "require_full_column_rank": moment_map["require_full_column_rank"],
            "acceptance_gate_passed": moment_map["acceptance_gate_passed"],
            "unique_ridge_prorm_quadratic_head_iff_full_column_rank": True,
            "observed_unique_ridge_prorm_quadratic_head": moment_map[
                "observed_unique_ridge_prorm_quadratic_head"
            ],
            "population_identifiability_theorem_claimed": False,
            "minimum_norm_solution_claimed": False,
        },
    }
    return {
        "training_design_sha256": design_sha256,
        "heads_sha256": recorded_heads_sha,
        "individual_head_sha256": {
            learner: primary[learner]["head_sha256"] for learner in CANONICAL_LEARNERS
        },
        "heldout_head_sha256": {
            learner: _canonical_sha256(normalized[learner]) for learner in CANONICAL_LEARNERS
        },
        "gate_evidence": gate_evidence,
    }


def _validate_heldout_fixed_beta(
    value: object,
    *,
    recorded_sha256: object,
    seed: int,
    source_config_hash: str,
    design_sha256: str,
    runtime_sha256: str,
    beta_common: float,
    heads_sha256: str,
    training_design_sha256: str,
    expected_head_sha256: Mapping[str, object],
    expected_validation_prompts: int,
    expected_test_prompts: int,
    expected_candidates: int,
    expected_pcg_tolerance: float,
    expected_pcg_max_iterations: int,
    expected_relative_damping: float,
    expected_transform: Mapping[str, object],
    name: str,
) -> tuple[float, dict[str, object]]:
    heldout = _required(
        value,
        name=name,
        keys={
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
        },
    )
    recorded_sha = _digest(recorded_sha256, name=f"{name}_sha256")
    if _canonical_sha256(heldout) != recorded_sha:
        raise ValueError(f"{name} canonical SHA256 mismatch")
    expected_top = {
        "schema_version": "phase2-heldout-fixed-beta/v1",
        "estimand": "frozen_global_common_beta_local_regret",
        "formal_gate_split": "test",
        "descriptive_split": "validation",
        "split_order": ["validation", "test"],
        "raw_oracle_logits_serialized": False,
        "heldout_direction_vectors_serialized": False,
    }
    for field, expected in expected_top.items():
        _expect(heldout[field], expected, name=f"{name}.{field}")
    heldout_beta = _finite(heldout["beta_common"], name=f"{name}.beta_common")
    if not _close(heldout_beta, beta_common, tolerance=1.0e-12):
        raise ValueError(f"{name} does not use the frozen global beta_common")

    frozen = _mapping(heldout["frozen_state"], name=f"{name}.frozen_state")
    frozen_sha = _digest(
        heldout["frozen_state_sha256"],
        name=f"{name}.frozen_state_sha256",
    )
    if _canonical_sha256(frozen) != frozen_sha:
        raise ValueError(f"{name}.frozen_state_sha256 mismatch")
    expected_frozen = {
        "schema_version": "phase2-heldout-frozen-state/v1",
        "source_config_hash": source_config_hash,
        "phase2_design_sha256": design_sha256,
        "phase2_runtime_contract_sha256": runtime_sha256,
        "seed": seed,
        "heads_sha256": heads_sha256,
        "training_design_sha256": training_design_sha256,
        "heads_frozen": True,
        "beta_common_frozen": True,
        "deployed_directions_frozen": True,
    }
    for field, expected in expected_frozen.items():
        _expect(frozen.get(field), expected, name=f"{name}.frozen_state.{field}")
    frozen_beta = _finite(
        frozen.get("beta_common"),
        name=f"{name}.frozen_state.beta_common",
    )
    if not _close(frozen_beta, beta_common, tolerance=1.0e-12):
        raise ValueError(f"{name}.frozen_state beta does not match")
    _digest(
        frozen.get("deployment_identity_sha256"),
        name=f"{name}.frozen_state.deployment_identity_sha256",
    )

    oracle = _mapping(heldout["oracle_rescore"], name=f"{name}.oracle_rescore")
    _expect(
        oracle.get("source"),
        "saved_validation_and_test_candidates_rescored_after_policy_freeze",
        name=f"{name}.oracle_rescore.source",
    )
    _digest(
        oracle.get("oracle_chat_template_sha256"),
        name=f"{name}.oracle_rescore.oracle_chat_template_sha256",
    )
    _digest(
        oracle.get("combined_transformed_rewards_sha256"),
        name=f"{name}.oracle_rescore.combined_transformed_rewards_sha256",
    )
    _expect(
        oracle.get("raw_oracle_logits_serialized"),
        False,
        name=f"{name}.oracle_rescore.raw_oracle_logits_serialized",
    )
    transform = _mapping(oracle.get("transform"), name=f"{name}.oracle_rescore.transform")
    for field in ("b", "tau"):
        if not _close(
            _finite(transform.get(field), name=f"{name}.oracle_rescore.transform.{field}"),
            _finite(
                expected_transform.get(field),
                name=f"{name}.expected_transform.{field}",
            ),
            tolerance=1.0e-12,
        ):
            raise ValueError(f"{name} held-out oracle transform differs from train rescore")

    solver = _mapping(heldout["solver"], name=f"{name}.solver")
    expected_solver = {
        "pcg_dtype": "float64",
        "pcg_max_iterations": expected_pcg_max_iterations,
        "split_specific_node_fisher_and_damping": True,
    }
    for field, expected in expected_solver.items():
        _expect(solver.get(field), expected, name=f"{name}.solver.{field}")
    if not _close(
        _finite(solver.get("pcg_tolerance"), name=f"{name}.solver.pcg_tolerance"),
        expected_pcg_tolerance,
        tolerance=1.0e-15,
    ) or not _close(
        _finite(solver.get("relative_damping"), name=f"{name}.solver.relative_damping"),
        expected_relative_damping,
        tolerance=1.0e-15,
    ):
        raise ValueError(f"{name} held-out solver differs from the overlay")

    boundary = _mapping(heldout["information_boundary"], name=f"{name}.information_boundary")
    expected_boundary = {
        "fresh_targets_created_after_heads_beta_and_deployments_frozen": True,
        "validation_or_test_targets_available_to_head_trainer": False,
        "validation_or_test_targets_available_to_beta_calibration": False,
        "validation_or_test_targets_available_to_policy_deployment": False,
        "heldout_direction_used_for_policy": False,
    }
    for field, expected in expected_boundary.items():
        _expect(boundary.get(field), expected, name=f"{name}.information_boundary.{field}")

    splits = _mapping(heldout["splits"], name=f"{name}.splits")
    if set(splits) != {"validation", "test"}:
        raise ValueError(f"{name} must contain exactly validation and test splits")
    input_identities: dict[str, object] = {}
    test_bt_regret: float | None = None
    test_prorm_regret: float | None = None
    split_gate_evidence: dict[str, object] = {}
    for split_name, expected_prompts in (
        ("validation", expected_validation_prompts),
        ("test", expected_test_prompts),
    ):
        split = _mapping(splits[split_name], name=f"{name}.splits.{split_name}")
        input_identity = _mapping(
            split.get("input_identity"),
            name=f"{name}.splits.{split_name}.input_identity",
        )
        expected_input = {
            "split": split_name,
            "num_prompts": expected_prompts,
            "num_candidates": expected_candidates,
            "contains_oracle_targets": False,
        }
        for field, expected in expected_input.items():
            _expect(
                input_identity.get(field),
                expected,
                name=f"{name}.splits.{split_name}.input_identity.{field}",
            )
        for field in (
            "prompt_ids_sha256",
            "policy_scores_sha256",
            "reward_features_sha256",
            "candidates_sha256",
        ):
            _digest(
                input_identity.get(field),
                name=f"{name}.splits.{split_name}.input_identity.{field}",
            )
        _integer(
            input_identity.get("policy_dimension"),
            name=f"{name}.splits.{split_name}.input_identity.policy_dimension",
            minimum=1,
        )
        _integer(
            input_identity.get("reward_dimension"),
            name=f"{name}.splits.{split_name}.input_identity.reward_dimension",
            minimum=1,
        )
        input_identity_sha = _digest(
            split.get("input_identity_sha256"),
            name=f"{name}.splits.{split_name}.input_identity_sha256",
        )
        if _canonical_sha256(input_identity) != input_identity_sha:
            raise ValueError(f"{name} {split_name} input identity SHA256 mismatch")
        input_identities[split_name] = dict(input_identity)
        _digest(
            split.get("transformed_oracle_rewards_sha256"),
            name=f"{name}.splits.{split_name}.transformed_oracle_rewards_sha256",
        )
        expected_split = {
            "raw_oracle_logits_serialized": False,
            "node_fisher_estimator": "mean_all_saved_split_nodes",
            "moment_estimator": "per_prompt_unbiased_candidate_covariance",
            "fixed_beta_source": ("pilot_selected_global_beta_frozen_in_confirmatory_design"),
        }
        for field, expected in expected_split.items():
            _expect(split.get(field), expected, name=f"{name}.splits.{split_name}.{field}")
        if not _close(
            _finite(split.get("fixed_beta"), name=f"{name}.splits.{split_name}.fixed_beta"),
            beta_common,
            tolerance=1.0e-12,
        ):
            raise ValueError(f"{name} {split_name} changed beta")
        if not _close(
            _finite(
                split.get("relative_damping"),
                name=f"{name}.splits.{split_name}.relative_damping",
            ),
            expected_relative_damping,
            tolerance=1.0e-15,
        ):
            raise ValueError(f"{name} {split_name} changed relative damping")
        if (
            _finite(
                split.get("absolute_damping"),
                name=f"{name}.splits.{split_name}.absolute_damping",
            )
            <= 0.0
        ):
            raise ValueError(f"{name} {split_name} absolute damping must be positive")
        learners = _mapping(split.get("learners"), name=f"{name}.splits.{split_name}.learners")
        if set(learners) != set(CANONICAL_LEARNERS):
            raise ValueError(f"{name} {split_name} must contain exactly both learners")
        regrets: dict[str, float] = {}
        for learner in CANONICAL_LEARNERS:
            learner_result = _mapping(
                learners[learner],
                name=f"{name}.splits.{split_name}.learners.{learner}",
            )
            _expect(
                learner_result.get("head_sha256"),
                expected_head_sha256[learner],
                name=f"{name}.splits.{split_name}.learners.{learner}.head_sha256",
            )
            regret = _finite(
                learner_result.get("local_regret_at_frozen_global_beta"),
                name=f"{name}.splits.{split_name}.learners.{learner}.local_regret",
                nonnegative=True,
            )
            regrets[learner] = regret
            _finite(
                learner_result.get("native_beta1_squared_fisher_direction_error"),
                name=f"{name}.splits.{split_name}.learners.{learner}.fisher_error",
                nonnegative=True,
            )
            cosine = learner_result.get("native_beta1_fisher_cosine")
            if (
                cosine is not None
                and not -1.0
                <= _finite(
                    cosine,
                    name=f"{name}.splits.{split_name}.learners.{learner}.fisher_cosine",
                )
                <= 1.0
            ):
                raise ValueError(f"{name} {split_name} Fisher cosine is outside [-1,1]")
            for field in (
                "native_beta1_predicted_fisher_norm",
                "native_beta1_target_fisher_norm",
            ):
                _finite(
                    learner_result.get(field),
                    name=f"{name}.splits.{split_name}.learners.{learner}.{field}",
                    nonnegative=True,
                )
            _expect(
                learner_result.get("direction_vectors_serialized"),
                False,
                name=f"{name}.splits.{split_name}.learners.{learner}.direction_vectors_serialized",
            )
        contrast = _mapping(
            split.get("prorm_plus_minus_bt_mle"),
            name=f"{name}.splits.{split_name}.prorm_plus_minus_bt_mle",
        )
        stored_prorm_minus_bt = _finite(
            contrast.get("local_regret_at_frozen_global_beta"),
            name=f"{name}.splits.{split_name}.prorm_plus_minus_bt_mle.local_regret",
        )
        expected_prorm_minus_bt = regrets[PRORM_PLUS] - regrets[BT_MLE]
        if not _close(stored_prorm_minus_bt, expected_prorm_minus_bt):
            raise ValueError(f"{name} {split_name} held-out regret contrast arithmetic failed")
        if split_name == "test":
            test_bt_regret = regrets[BT_MLE]
            test_prorm_regret = regrets[PRORM_PLUS]
        split_gate_evidence[split_name] = {
            "input_identity_sha256": input_identity_sha,
            "bt_mle_local_regret": regrets[BT_MLE],
            "prorm_plus_local_regret": regrets[PRORM_PLUS],
            "bt_mle_minus_prorm_plus_local_regret": (regrets[BT_MLE] - regrets[PRORM_PLUS]),
        }

    deferred_payload = {
        "schema_version": "phase2-deferred-heldout-input/v1",
        "split_order": ["validation", "test"],
        "validation": input_identities["validation"],
        "test": input_identities["test"],
        "contains_oracle_targets": False,
    }
    deferred_sha = _digest(
        heldout["deferred_input_sha256"],
        name=f"{name}.deferred_input_sha256",
    )
    if _canonical_sha256(deferred_payload) != deferred_sha:
        raise ValueError(f"{name}.deferred_input_sha256 mismatch")
    if test_bt_regret is None or test_prorm_regret is None:
        raise RuntimeError("held-out test regret was not assembled")
    bt_minus_prorm = test_bt_regret - test_prorm_regret
    return bt_minus_prorm, {
        "schema_version": "phase2-heldout-aggregate-gate-input/v1",
        "passed_integrity": True,
        "heldout_fixed_beta_sha256": recorded_sha,
        "frozen_state_sha256": frozen_sha,
        "deferred_input_sha256": deferred_sha,
        "formal_gate_split": "test",
        "contrast": "bt_mle_minus_prorm_plus",
        "direction": "higher_is_better",
        "test_bt_mle_minus_prorm_plus_local_regret": bt_minus_prorm,
        "splits": split_gate_evidence,
    }


def _validate_prompt_estimate(value: object, *, name: str) -> tuple[float, float]:
    estimate = _required(
        value,
        name=name,
        keys={"mean", "sample_standard_error"},
    )
    mean = _finite(estimate["mean"], name=f"{name}.mean")
    standard_error = _finite(
        estimate["sample_standard_error"],
        name=f"{name}.sample_standard_error",
        nonnegative=True,
    )
    return mean, standard_error


def _validate_rollout_summary(
    value: object,
    *,
    name: str,
    expected_trajectories: int,
    max_response_tokens: int,
) -> tuple[float, float]:
    rollout = _required(
        value,
        name=name,
        keys={
            "num_trajectories",
            "terminated_by_eos_count",
            "terminated_by_eos_rate",
            "reached_max_length_count",
            "reached_max_length_rate",
            "response_token_count",
        },
    )
    count = _integer(rollout["num_trajectories"], name=f"{name}.num_trajectories", minimum=1)
    if count != expected_trajectories:
        raise ValueError(f"{name}.num_trajectories does not match the prompt/candidate design")
    for prefix in ("terminated_by_eos", "reached_max_length"):
        event_count = _integer(rollout[f"{prefix}_count"], name=f"{name}.{prefix}_count")
        if event_count > count:
            raise ValueError(f"{name}.{prefix}_count exceeds num_trajectories")
        rate = _finite(rollout[f"{prefix}_rate"], name=f"{name}.{prefix}_rate")
        if not 0.0 <= rate <= 1.0 or not _close(rate, event_count / count, tolerance=1.0e-12):
            raise ValueError(f"{name}.{prefix}_rate is inconsistent with its count")
    lengths = _required(
        rollout["response_token_count"],
        name=f"{name}.response_token_count",
        keys={"mean", "minimum", "maximum"},
    )
    mean_length = _finite(lengths["mean"], name=f"{name}.response_token_count.mean")
    minimum = _integer(
        lengths["minimum"],
        name=f"{name}.response_token_count.minimum",
        minimum=1,
    )
    maximum = _integer(
        lengths["maximum"],
        name=f"{name}.response_token_count.maximum",
        minimum=1,
    )
    if minimum > maximum or maximum > max_response_tokens:
        raise ValueError(f"{name}.response_token_count violates the response horizon")
    if not float(minimum) <= mean_length <= float(maximum):
        raise ValueError(f"{name}.response_token_count.mean lies outside its range")
    return float(rollout["reached_max_length_rate"]), mean_length


def _validate_on_policy_kl_tail(
    value: object,
    *,
    expected_num_prompts: int,
    expected_candidates_per_prompt: int,
    expected_mean: float,
    design_stage: str,
    name: str,
) -> dict[str, object]:
    tail = _required(
        value,
        name=name,
        keys={
            "schema_version",
            "unit",
            "num_prompts",
            "candidates_per_prompt",
            "mean",
            "p50",
            "p90",
            "p95",
            "p99",
            "maximum",
            "per_sequence_maximum",
            "pilot_selection_role",
            "formal_gate_applied",
        },
    )
    expected = {
        "schema_version": "on-policy-kl-tail-summary/v1",
        "unit": "prompt_mean_over_candidates",
        "num_prompts": expected_num_prompts,
        "candidates_per_prompt": expected_candidates_per_prompt,
        "pilot_selection_role": "locality_tail_measurement",
    }
    for field, expected_value in expected.items():
        _expect(tail[field], expected_value, name=f"{name}.{field}")
    if design_stage == "pilot":
        _expect(tail["formal_gate_applied"], False, name=f"{name}.formal_gate_applied")
    else:
        _expect(tail["formal_gate_applied"], True, name=f"{name}.formal_gate_applied")

    observed = {
        field: _finite(tail[field], name=f"{name}.{field}", nonnegative=True)
        for field in ("mean", "p50", "p90", "p95", "p99", "maximum", "per_sequence_maximum")
    }
    if not _close(observed["mean"], expected_mean):
        raise ValueError(f"{name}.mean does not match the arm mean on-policy KL")
    ordered = [
        observed["p50"],
        observed["p90"],
        observed["p95"],
        observed["p99"],
        observed["maximum"],
        observed["per_sequence_maximum"],
    ]
    if any(ordered[index + 1] < ordered[index] for index in range(len(ordered) - 1)):
        raise ValueError(f"{name} quantiles/maxima are not monotone")
    return {
        "schema_version": tail["schema_version"],
        "unit": tail["unit"],
        "num_prompts": expected_num_prompts,
        "candidates_per_prompt": expected_candidates_per_prompt,
        **observed,
        "pilot_selection_role": tail["pilot_selection_role"],
        "formal_gate_applied": tail["formal_gate_applied"],
    }


def _validate_direction_deployment(
    value: object,
    *,
    arm_name: str,
    beta_common: float,
    name: str,
) -> None:
    deployment = _mapping(value, name=name)
    if arm_name == "zero_b":
        if (
            deployment.get("schema_version") != "zero-b-deployment/v1"
            or deployment.get("displacement_is_exact_zero") is not True
            or deployment.get("learner_specific_rescaling") is not False
            or not _close(
                _finite(deployment.get("beta_common"), name=f"{name}.beta_common"),
                beta_common,
                tolerance=1.0e-12,
            )
        ):
            raise ValueError(f"{name} is not the exact zero-B deployment")
        return

    direction = _required(
        deployment.get("direction"),
        name=f"{name}.direction",
        keys={"schema_version", "beta", "pcg"},
    )
    if direction["schema_version"] != "policy-direction/v1":
        raise ValueError(f"{name}.direction has the wrong schema")
    if not _close(
        _finite(direction["beta"], name=f"{name}.direction.beta"),
        1.0,
        tolerance=1.0e-12,
    ):
        raise ValueError(f"{name}.direction must be the native beta=1 direction")
    pcg = _mapping(direction["pcg"], name=f"{name}.direction.pcg")
    if pcg.get("converged") is not True:
        raise ValueError(f"{name}.direction PCG did not converge")

    common = _required(
        deployment.get("common_beta_direction"),
        name=f"{name}.common_beta_direction",
        keys={
            "schema_version",
            "name",
            "beta_common",
            "learner_specific_rescaling",
        },
    )
    if (
        common["schema_version"] != "common-beta-direction/v1"
        or common["name"] != arm_name
        or common["learner_specific_rescaling"] is not False
        or not _close(
            _finite(common["beta_common"], name=f"{name}.common_beta_direction.beta_common"),
            beta_common,
            tolerance=1.0e-12,
        )
    ):
        raise ValueError(f"{name} did not use the frozen common beta directly")


def _validate_arm(
    value: object,
    *,
    arm_name: str,
    beta_common: float,
    num_prompts: int,
    candidates_per_prompt: int,
    max_response_tokens: int,
    design_stage: str,
    name: str,
) -> tuple[dict[str, float], int, dict[str, object]]:
    arm = _required(
        value,
        name=name,
        keys={
            "deployment",
            "rollout",
            "mean_on_policy_kl_pi_updated_to_pi0",
            "on_policy_kl_tail",
            "utility",
        },
    )
    _validate_direction_deployment(
        arm["deployment"],
        arm_name=arm_name,
        beta_common=beta_common,
        name=f"{name}.deployment",
    )
    expected_trajectories = num_prompts * candidates_per_prompt
    cap_rate, mean_length = _validate_rollout_summary(
        arm["rollout"],
        name=f"{name}.rollout",
        expected_trajectories=expected_trajectories,
        max_response_tokens=max_response_tokens,
    )
    arm_kl = _finite(
        arm["mean_on_policy_kl_pi_updated_to_pi0"],
        name=f"{name}.mean_on_policy_kl_pi_updated_to_pi0",
        nonnegative=True,
    )
    kl_tail = _validate_on_policy_kl_tail(
        arm["on_policy_kl_tail"],
        expected_num_prompts=num_prompts,
        expected_candidates_per_prompt=candidates_per_prompt,
        expected_mean=arm_kl,
        design_stage=design_stage,
        name=f"{name}.on_policy_kl_tail",
    )

    utility = _required(
        arm["utility"],
        name=f"{name}.utility",
        keys={
            "schema_version",
            "beta_common",
            "num_prompts",
            "candidates_per_prompt",
            "mean_target_reward",
            "mean_on_policy_kl_pi_updated_to_pi0",
            "mean_target_utility",
            "target_utility_sample_standard_error",
            "improvement_over_zero_b",
            "oracle_step_reference_gap",
            "oracle_step_is_global_optimum",
        },
    )
    if utility["schema_version"] != "downstream-policy-utility/v1":
        raise ValueError(f"{name}.utility has the wrong schema")
    if utility["num_prompts"] != num_prompts or (
        utility["candidates_per_prompt"] != candidates_per_prompt
    ):
        raise ValueError(f"{name}.utility does not use the frozen prompt/candidate shape")
    if utility["oracle_step_is_global_optimum"] is not False:
        raise ValueError(f"{name}.utility mislabels the oracle step as a global optimum")
    utility_beta = _finite(utility["beta_common"], name=f"{name}.utility.beta_common")
    if not _close(utility_beta, beta_common, tolerance=1.0e-12):
        raise ValueError(f"{name}.utility used a different beta")
    target_reward = _finite(
        utility["mean_target_reward"],
        name=f"{name}.utility.mean_target_reward",
    )
    utility_kl = _finite(
        utility["mean_on_policy_kl_pi_updated_to_pi0"],
        name=f"{name}.utility.mean_on_policy_kl_pi_updated_to_pi0",
        nonnegative=True,
    )
    target_utility = _finite(
        utility["mean_target_utility"],
        name=f"{name}.utility.mean_target_utility",
    )
    _finite(
        utility["target_utility_sample_standard_error"],
        name=f"{name}.utility.target_utility_sample_standard_error",
        nonnegative=True,
    )
    if not _close(arm_kl, utility_kl):
        raise ValueError(f"{name} arm and utility KL summaries disagree")
    if not _close(target_utility, target_reward - beta_common * utility_kl):
        raise ValueError(f"{name}.utility violates reward - beta_common * KL")
    improvement, _ = _validate_prompt_estimate(
        utility["improvement_over_zero_b"],
        name=f"{name}.utility.improvement_over_zero_b",
    )
    gap, _ = _validate_prompt_estimate(
        utility["oracle_step_reference_gap"],
        name=f"{name}.utility.oracle_step_reference_gap",
    )
    return (
        {
            "mean_target_utility": target_utility,
            "improvement_over_zero_b": improvement,
            "oracle_step_reference_gap": gap,
            "mean_target_reward": target_reward,
            "mean_on_policy_kl_pi_updated_to_pi0": utility_kl,
            "on_policy_kl_prompt_p50": float(kl_tail["p50"]),
            "on_policy_kl_prompt_p90": float(kl_tail["p90"]),
            "on_policy_kl_prompt_p95": float(kl_tail["p95"]),
            "on_policy_kl_prompt_p99": float(kl_tail["p99"]),
            "on_policy_kl_prompt_maximum": float(kl_tail["maximum"]),
            "on_policy_kl_sequence_maximum": float(kl_tail["per_sequence_maximum"]),
            "reached_max_length_rate": cap_rate,
            "mean_response_token_count": mean_length,
        },
        expected_trajectories,
        kl_tail,
    )


def _validate_pre_oracle_safety_gate(
    value: object,
    *,
    expected_runtime_contract: Mapping[str, object],
    expected_design_stage: str,
    metrics_by_arm: Mapping[str, Mapping[str, float]],
    name: str,
) -> dict[str, object]:
    """Recompute every serialized pre-oracle gate input from arm summaries."""

    required_keys = {
        "schema_version",
        "design_stage",
        "pilot_phase",
        "measure_only",
        "formal_gate",
        "thresholds",
        "observed_by_arm",
        "violations",
        "passed",
        "beta_retuned",
        "on_violation",
    }
    gate = _required(value, name=name, keys=required_keys)
    if set(gate) != required_keys:
        raise ValueError(f"{name} has unknown fields")
    expected_pilot_phase = expected_runtime_contract.get("pilot_phase")
    expected_formal_gate = expected_design_stage == "confirmatory"
    expected_measure_only = not expected_formal_gate
    if (
        gate["schema_version"] != "phase2-pre-oracle-safety-gate/v1"
        or gate["design_stage"] != expected_design_stage
        or gate["pilot_phase"] != expected_pilot_phase
        or gate["measure_only"] is not expected_measure_only
        or gate["formal_gate"] is not expected_formal_gate
        or not isinstance(gate["passed"], bool)
        or gate["beta_retuned"] is not False
        or gate["on_violation"]
        != (
            "fail_before_final_oracle_and_heldout"
            if expected_formal_gate
            else "publish_target_free_diagnostics_without_final_oracle"
        )
    ):
        raise ValueError(f"{name} has invalid stage or enforcement semantics")

    max_length = _required(
        expected_runtime_contract.get("max_length_gate"),
        name=f"{name}.runtime_max_length_gate",
        keys={"formal_gate", "formal_threshold", "measure_only"},
    )
    expected_thresholds = {
        "mean_policy_to_reference_kl_cap": _finite(
            expected_runtime_contract.get("measured_kl_safety_cap"),
            name=f"{name}.runtime.mean_policy_to_reference_kl_cap",
        ),
        "prompt_mean_p95_kl_cap": _finite(
            expected_runtime_contract.get("prompt_mean_p95_kl_cap"),
            name=f"{name}.runtime.prompt_mean_p95_kl_cap",
        ),
        "prompt_mean_p99_kl_cap": _finite(
            expected_runtime_contract.get("prompt_mean_p99_kl_cap"),
            name=f"{name}.runtime.prompt_mean_p99_kl_cap",
        ),
        "prompt_mean_maximum_kl_cap": _finite(
            expected_runtime_contract.get("prompt_mean_maximum_kl_cap"),
            name=f"{name}.runtime.prompt_mean_maximum_kl_cap",
        ),
        "per_sequence_maximum_kl_cap": _finite(
            expected_runtime_contract.get("per_sequence_maximum_kl_cap"),
            name=f"{name}.runtime.per_sequence_maximum_kl_cap",
        ),
        "reached_max_length_rate_cap": _finite(
            max_length.get("formal_threshold"),
            name=f"{name}.runtime.reached_max_length_rate_cap",
        ),
    }
    thresholds = _mapping(gate["thresholds"], name=f"{name}.thresholds")
    if set(thresholds) != set(expected_thresholds):
        raise ValueError(f"{name}.thresholds must contain exactly the six frozen caps")
    for metric, expected in expected_thresholds.items():
        observed = _finite(thresholds[metric], name=f"{name}.thresholds.{metric}")
        if not _close(observed, expected, tolerance=1.0e-12):
            raise ValueError(f"{name}.thresholds.{metric} differs from the runtime contract")

    observed_by_arm = _mapping(gate["observed_by_arm"], name=f"{name}.observed_by_arm")
    if set(observed_by_arm) != set(PHASE2_ARM_ORDER):
        raise ValueError(f"{name}.observed_by_arm must contain exactly the frozen arms")
    metric_sources = {
        "mean_policy_to_reference_kl": "mean_on_policy_kl_pi_updated_to_pi0",
        "prompt_mean_p95_kl": "on_policy_kl_prompt_p95",
        "prompt_mean_p99_kl": "on_policy_kl_prompt_p99",
        "prompt_mean_maximum_kl": "on_policy_kl_prompt_maximum",
        "per_sequence_maximum_kl": "on_policy_kl_sequence_maximum",
        "reached_max_length_rate": "reached_max_length_rate",
    }
    expected_violations: list[str] = []
    normalized_observed: dict[str, dict[str, float]] = {}
    for arm_name in PHASE2_ARM_ORDER:
        arm_observed = _mapping(
            observed_by_arm[arm_name],
            name=f"{name}.observed_by_arm.{arm_name}",
        )
        if set(arm_observed) != set(metric_sources):
            raise ValueError(f"{name}.observed_by_arm.{arm_name} must contain exactly six metrics")
        arm_metrics = metrics_by_arm[arm_name]
        normalized_observed[arm_name] = {}
        for metric, source in metric_sources.items():
            observed = _finite(
                arm_observed[metric],
                name=f"{name}.observed_by_arm.{arm_name}.{metric}",
                nonnegative=True,
            )
            recomputed = _finite(
                arm_metrics[source],
                name=f"{name}.recomputed.{arm_name}.{metric}",
                nonnegative=True,
            )
            if not _close(observed, recomputed):
                raise ValueError(
                    f"{name} disagrees with serialized tail/length evidence for {arm_name}:{metric}"
                )
            normalized_observed[arm_name][metric] = observed
            cap = expected_thresholds[f"{metric}_cap"]
            if recomputed > cap:
                expected_violations.append(f"{arm_name}:{metric}")

    violations = gate["violations"]
    if not isinstance(violations, list) or any(not isinstance(item, str) for item in violations):
        raise TypeError(f"{name}.violations must be a list of strings")
    expected_passed = not expected_violations
    if violations != expected_violations or gate["passed"] is not expected_passed:
        raise ValueError(f"{name} threshold arithmetic is inconsistent")
    if expected_formal_gate and not expected_passed:
        raise ValueError(f"{name} did not pass the frozen confirmatory safety gate")
    return {
        "schema_version": gate["schema_version"],
        "design_stage": expected_design_stage,
        "pilot_phase": expected_pilot_phase,
        "measure_only": expected_measure_only,
        "formal_gate": expected_formal_gate,
        "thresholds": expected_thresholds,
        "observed_by_arm": normalized_observed,
        "violations": expected_violations,
        "passed": expected_passed,
        "beta_retuned": False,
        "on_violation": gate["on_violation"],
    }


def _validate_rollout_jsonl(
    result_path: Path,
    *,
    recorded_reference: object,
    recorded_sha256: object,
    expected_counts: Mapping[str, int],
    beta_common: float,
) -> tuple[Path, str]:
    expected_name = f"{result_path.stem}.rollouts.jsonl"
    if not isinstance(recorded_reference, str) or recorded_reference != expected_name:
        raise ValueError(
            f"{result_path}:rollouts_jsonl must name the exact sibling {expected_name!r}"
        )
    pure = PurePosixPath(recorded_reference)
    if pure.is_absolute() or len(pure.parts) != 1 or pure.name != recorded_reference:
        raise ValueError(f"{result_path}:rollouts_jsonl must be a sibling POSIX filename")
    rollout_path = result_path.parent / recorded_reference
    if rollout_path.parent.resolve() != result_path.parent.resolve():
        raise ValueError(f"{result_path}:rollouts_jsonl escapes its result directory")
    raw = _read_regular_file(rollout_path, name="rollout JSONL")
    actual_sha = hashlib.sha256(raw).hexdigest()
    expected_sha = _digest(
        recorded_sha256,
        name=f"{result_path}:rollouts_sha256",
    )
    if actual_sha != expected_sha:
        raise ValueError(f"{result_path}: sibling rollout JSONL SHA256 mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{rollout_path} is not UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"{rollout_path} must be a non-empty JSONL without blank rows")

    observed_arms: list[str] = []
    for index, line in enumerate(lines):
        row = _strict_json_bytes(line.encode(), path=rollout_path)
        if row.get("schema_version") != PHASE2_ROLLOUT_SCHEMA:
            raise ValueError(f"{rollout_path}:{index + 1} has the wrong trajectory schema")
        arm_name = row.get("arm")
        if arm_name not in PHASE2_ARM_ORDER:
            raise ValueError(f"{rollout_path}:{index + 1} has an invalid arm")
        observed_arms.append(str(arm_name))
        if (
            row.get("kl_orientation") != KL_ORIENTATION
            or row.get("kl_history_source") != KL_HISTORY_SOURCE
            or row.get("raw_oracle_logit_serialized") is not False
        ):
            raise ValueError(f"{rollout_path}:{index + 1} violates the KL/oracle contract")
        row_beta = _finite(row.get("beta_common"), name=f"{rollout_path}:{index + 1}.beta")
        if not _close(row_beta, beta_common, tolerance=1.0e-12):
            raise ValueError(f"{rollout_path}:{index + 1} used a different beta")
        reward = _finite(
            row.get("transformed_oracle_reward"),
            name=f"{rollout_path}:{index + 1}.reward",
        )
        kl = _finite(
            row.get("on_policy_kl_pi_updated_to_pi0"),
            name=f"{rollout_path}:{index + 1}.kl",
            nonnegative=True,
        )
        target = _finite(
            row.get("target_utility"),
            name=f"{rollout_path}:{index + 1}.target_utility",
        )
        if not _close(target, reward - beta_common * kl):
            raise ValueError(f"{rollout_path}:{index + 1} has inconsistent target utility")

    expected_arms = [
        arm_name for arm_name in PHASE2_ARM_ORDER for _ in range(expected_counts[arm_name])
    ]
    if observed_arms != expected_arms:
        raise ValueError(f"{rollout_path} does not contain exact arm-major trajectory counts")
    return rollout_path, actual_sha


@dataclass(frozen=True, slots=True)
class Phase2AggregateSource:
    """Byte-level provenance for one accepted seed result and rollout file."""

    seed: int
    result_path: str
    result_sha256: str
    rollouts_path: str
    rollouts_sha256: str
    artifact_metadata_sha256: str
    run_manifest_sha256: str
    heads_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "result_path": self.result_path,
            "result_sha256": self.result_sha256,
            "rollouts_path": self.rollouts_path,
            "rollouts_sha256": self.rollouts_sha256,
            "artifact_metadata_sha256": self.artifact_metadata_sha256,
            "run_manifest_sha256": self.run_manifest_sha256,
            "heads_sha256": self.heads_sha256,
        }


@dataclass(frozen=True, slots=True)
class _LoadedSeed:
    seed: int
    beta_common: float
    metrics: Mapping[str, Mapping[str, float]]
    oracle_step_improvement: float
    prorm_plus_improvement: float
    heldout_bt_minus_prorm_plus_local_regret: float
    environment_identity: Mapping[str, object]
    control_gate_evidence: Mapping[str, object]
    heldout_gate_evidence: Mapping[str, object]
    max_length_gate_evidence: Mapping[str, object]
    on_policy_kl_tail_by_arm: Mapping[str, Mapping[str, object]]
    source: Phase2AggregateSource


def _load_seed_result(
    raw_path: str | os.PathLike[str],
    *,
    expected_source_config_hash: str,
    expected_design_sha256: str,
    expected_design_stage: str,
    expected_formal_eligibility: bool,
    expected_runtime_contract: Mapping[str, object],
    expected_runtime_sha256: str,
    expected_train_prompts: int,
    expected_validation_prompts: int,
    expected_num_prompts: int,
    expected_candidates: int,
    expected_outer_steps: int,
    expected_low_dimension: int,
    expected_projection_namespace: str,
    expected_eigenvalue_tolerance: float,
    expected_pcg_tolerance: float,
    expected_pcg_max_iterations: int,
    expected_relative_damping: float,
    prohibit_label_clipping: bool,
    numeric_tolerances: Mapping[str, object],
    adaptive_convergence: Mapping[str, object],
    identifiability_config: Mapping[str, object],
    reward_model_config: Mapping[str, object],
    exact_soft_label_bt_config: Mapping[str, object],
    reference_base: Path,
) -> _LoadedSeed:
    path = Path(raw_path).resolve()
    raw = _read_regular_file(path, name="Phase-2 result JSON")
    result_sha = hashlib.sha256(raw).hexdigest()
    value = _required(
        _strict_json_bytes(raw, path=path),
        name=str(path),
        keys={
            "schema_version",
            "design_stage",
            "formal_eligibility",
            "per_seed_supports_formal_claim",
            "source_config_hash",
            "phase2_design_sha256",
            "phase2_runtime_contract",
            "phase2_runtime_contract_sha256",
            "seed",
            "artifact_metadata_sha256",
            "run_manifest_sha256",
            "environment_identity",
            "current_process_identity",
            "train_oracle_rescore",
            "head_training",
            "common_beta_calibration",
            "train_oracle_direction",
            "measured_kl_safety",
            "pre_oracle_safety_gate",
            "arms",
            "heldout_fixed_beta",
            "heldout_fixed_beta_sha256",
            "information_boundary",
            "common_random_numbers",
            "policy_and_oracle_co_resident",
            "learner_specific_line_search",
            "rollouts_jsonl",
            "rollouts_sha256",
        },
    )
    if value["schema_version"] != PHASE2_RESULT_SCHEMA:
        raise ValueError(f"{path} is not a {PHASE2_RESULT_SCHEMA} result")
    _expect(value["design_stage"], expected_design_stage, name=f"{path}:design_stage")
    if value["formal_eligibility"] is not expected_formal_eligibility:
        raise ValueError(f"{path}:formal_eligibility does not match the overlay")
    if value["per_seed_supports_formal_claim"] is not False:
        raise ValueError(f"{path}:per_seed_supports_formal_claim must be false")
    if value["source_config_hash"] != expected_source_config_hash:
        raise ValueError(f"{path} source_config_hash does not match the overlay")
    if value["phase2_design_sha256"] != expected_design_sha256:
        raise ValueError(f"{path} phase2_design_sha256 does not match the overlay")
    runtime = _mapping(value["phase2_runtime_contract"], name=f"{path}:runtime")
    runtime_sha = _digest(
        value["phase2_runtime_contract_sha256"],
        name=f"{path}:phase2_runtime_contract_sha256",
    )
    if _canonical_sha256(runtime) != runtime_sha:
        raise ValueError(f"{path} runtime contract hash does not match its payload")
    if runtime_sha != expected_runtime_sha256 or dict(runtime) != dict(expected_runtime_contract):
        raise ValueError(f"{path} runtime contract does not match the overlay")

    seed = _integer(value["seed"], name=f"{path}:seed")
    environment = _validate_environment(
        value["environment_identity"],
        name=f"{path}:environment_identity",
    )
    current = _validate_environment(
        value["current_process_identity"],
        name=f"{path}:current_process_identity",
    )
    if current != environment:
        raise ValueError(f"{path} process and run environment identities differ")

    train_oracle = _required(
        value["train_oracle_rescore"],
        name=f"{path}:train_oracle",
        keys={
            "source",
            "num_prompts",
            "num_candidates",
            "transformed_rewards_sha256",
            "oracle_chat_template_sha256",
            "frozen_transform",
            "raw_oracle_logits_serialized",
        },
    )
    if (
        train_oracle.get("source") != "saved_train_candidates_rescored_with_pinned_oracle"
        or train_oracle.get("raw_oracle_logits_serialized") is not False
    ):
        raise ValueError(f"{path} violates the train-oracle rescore boundary")
    if (
        train_oracle["num_prompts"] != expected_train_prompts
        or train_oracle["num_candidates"] != expected_candidates
    ):
        raise ValueError(f"{path} train-oracle rescore has the wrong train geometry")
    train_oracle_reward_sha = _digest(
        train_oracle["transformed_rewards_sha256"],
        name=f"{path}:train_oracle.transformed_rewards_sha256",
    )
    _digest(
        train_oracle["oracle_chat_template_sha256"],
        name=f"{path}:train_oracle.oracle_chat_template_sha256",
    )
    frozen_transform = _mapping(
        train_oracle["frozen_transform"],
        name=f"{path}:train_oracle.frozen_transform",
    )
    for field in ("b", "tau"):
        _finite(
            frozen_transform.get(field),
            name=f"{path}:train_oracle.frozen_transform.{field}",
        )
    head = _validate_head_training(
        value["head_training"],
        design_sha256=expected_design_sha256,
        seed=seed,
        train_oracle_reward_sha256=train_oracle_reward_sha,
        expected_train_prompts=expected_train_prompts,
        expected_candidates=expected_candidates,
        expected_outer_steps=expected_outer_steps,
        expected_low_dimension=expected_low_dimension,
        expected_projection_namespace=expected_projection_namespace,
        expected_eigenvalue_tolerance=expected_eigenvalue_tolerance,
        expected_pcg_tolerance=expected_pcg_tolerance,
        prohibit_label_clipping=prohibit_label_clipping,
        numeric_tolerances=numeric_tolerances,
        adaptive_convergence=adaptive_convergence,
        identifiability_config=identifiability_config,
        reward_model_config=reward_model_config,
        exact_soft_label_bt_config=exact_soft_label_bt_config,
        design_stage=expected_design_stage,
        name=f"{path}:head_training",
    )

    if expected_design_stage == "confirmatory":
        calibration_keys = {
            "schema_version",
            "rule",
            "beta_selection_split",
            "beta_source",
            "beta_common",
            "frozen_global_beta",
            "beta_matches_frozen_global_beta",
            "beta_selected_from_current_seed_curvature",
            "current_seed_oracle_natural_curvature",
            "reference_target_oracle_quadratic_kl",
            "predicted_current_seed_oracle_quadratic_kl",
            "current_seed_curvature_role",
            "frozen_in_phase2_design_identity",
            "learner_specific_rescaling",
            "post_evaluation_retuning",
        }
        calibration = _required(
            value["common_beta_calibration"],
            name=f"{path}:common_beta_calibration",
            keys=calibration_keys,
        )
        if set(calibration) != calibration_keys:
            raise ValueError(f"{path} confirmatory common-beta evidence has unknown fields")
        expected_beta = _finite(
            expected_runtime_contract.get("frozen_global_beta"),
            name=f"{path}:runtime.frozen_global_beta",
        )
        beta_common = _finite(
            calibration["beta_common"],
            name=f"{path}:common_beta_calibration.beta_common",
        )
        recorded_frozen_beta = _finite(
            calibration["frozen_global_beta"],
            name=f"{path}:common_beta_calibration.frozen_global_beta",
        )
        if (
            beta_common <= 0.0
            or beta_common != expected_beta
            or recorded_frozen_beta != expected_beta
        ):
            raise ValueError(
                f"{path} did not use the exact global beta frozen in the design identity"
            )
        expected_calibration = {
            "schema_version": "common-beta-frozen-global/v1",
            "rule": expected_runtime_contract.get("common_beta_rule"),
            "beta_selection_split": expected_runtime_contract.get("common_beta_calibration_split"),
            "beta_source": expected_runtime_contract.get("common_beta_source"),
            "beta_matches_frozen_global_beta": True,
            "beta_selected_from_current_seed_curvature": False,
            "current_seed_curvature_role": "predicted_kl_diagnostic_only",
            "frozen_in_phase2_design_identity": True,
            "learner_specific_rescaling": False,
            "post_evaluation_retuning": False,
        }
        for field, expected in expected_calibration.items():
            _expect(
                calibration[field],
                expected,
                name=f"{path}:common_beta_calibration.{field}",
            )
        target_k = _finite(
            calibration["reference_target_oracle_quadratic_kl"],
            name=f"{path}:common_beta_calibration.reference_target",
        )
        curvature = _finite(
            calibration["current_seed_oracle_natural_curvature"],
            name=f"{path}:common_beta_calibration.current_seed_curvature",
        )
        predicted_k = _finite(
            calibration["predicted_current_seed_oracle_quadratic_kl"],
            name=f"{path}:common_beta_calibration.predicted",
        )
        if (
            curvature <= 0.0
            or not _close(
                target_k,
                float(expected_runtime_contract["target_oracle_quadratic_kl"]),
                tolerance=1.0e-12,
            )
            or not _close(
                predicted_k,
                0.5 * curvature / (expected_beta * expected_beta),
                tolerance=1.0e-10,
            )
        ):
            raise ValueError(f"{path} frozen-beta predicted-KL evidence is inconsistent")
    else:
        calibration = _required(
            value["common_beta_calibration"],
            name=f"{path}:common_beta_calibration",
            keys={
                "schema_version",
                "beta_common",
                "target_oracle_quadratic_kl",
                "predicted_oracle_quadratic_kl",
                "calibration_split",
                "learner_specific_rescaling",
            },
        )
        if (
            calibration["schema_version"] != "common-beta-calibration/v1"
            or calibration["calibration_split"] != "train_only"
            or calibration["learner_specific_rescaling"] is not False
        ):
            raise ValueError(f"{path} has an invalid common-beta calibration contract")
        beta_common = _finite(
            calibration["beta_common"],
            name=f"{path}:common_beta_calibration.beta_common",
        )
        if beta_common <= 0.0:
            raise ValueError(f"{path} beta_common must be strictly positive")
        target_k = _finite(
            calibration["target_oracle_quadratic_kl"],
            name=f"{path}:common_beta_calibration.target",
        )
        predicted_k = _finite(
            calibration["predicted_oracle_quadratic_kl"],
            name=f"{path}:common_beta_calibration.predicted",
        )
        if not _close(
            target_k,
            float(expected_runtime_contract["target_oracle_quadratic_kl"]),
            tolerance=1.0e-12,
        ) or not _close(predicted_k, target_k, tolerance=1.0e-10):
            raise ValueError(f"{path} common-beta calibration target is inconsistent")
    heldout_contrast, heldout_gate_evidence = _validate_heldout_fixed_beta(
        value["heldout_fixed_beta"],
        recorded_sha256=value["heldout_fixed_beta_sha256"],
        seed=seed,
        source_config_hash=expected_source_config_hash,
        design_sha256=expected_design_sha256,
        runtime_sha256=expected_runtime_sha256,
        beta_common=beta_common,
        heads_sha256=str(head["heads_sha256"]),
        training_design_sha256=str(head["training_design_sha256"]),
        expected_head_sha256=_mapping(
            head["heldout_head_sha256"],
            name=f"{path}:head_training.heldout_head_sha256",
        ),
        expected_validation_prompts=expected_validation_prompts,
        expected_test_prompts=expected_num_prompts,
        expected_candidates=expected_candidates,
        expected_pcg_tolerance=expected_pcg_tolerance,
        expected_pcg_max_iterations=expected_pcg_max_iterations,
        expected_relative_damping=expected_relative_damping,
        expected_transform=frozen_transform,
        name=f"{path}:heldout_fixed_beta",
    )

    oracle_direction = _mapping(
        value["train_oracle_direction"],
        name=f"{path}:train_oracle_direction",
    )
    oracle_pcg = _mapping(
        oracle_direction.get("pcg"),
        name=f"{path}:train_oracle_direction.pcg",
    )
    if oracle_direction.get("schema_version") != "policy-direction/v1" or (
        oracle_pcg.get("converged") is not True
    ):
        raise ValueError(f"{path} train-oracle direction did not converge")

    if (
        value["policy_and_oracle_co_resident"] is not False
        or value["learner_specific_line_search"] is not False
    ):
        raise ValueError(f"{path} violates the model-memory/common-beta deployment contract")
    boundary = _mapping(value["information_boundary"], name=f"{path}:information_boundary")
    required_boundary = {
        "new_rollout_prompts_used_for_calibration": False,
        "source_materialization_heldout_scores_used_for_calibration": False,
        "new_rollout_oracle_scoring_after_heads_beta_and_directions_frozen": True,
        "heldout_candidate_rescore_after_heads_beta_and_directions_frozen": True,
        "heldout_directions_used_for_policy": False,
    }
    if expected_design_stage == "confirmatory":
        required_boundary.update(
            {
                "beta_selection_split": expected_runtime_contract.get(
                    "common_beta_calibration_split"
                ),
                "current_seed_train_curvature_role": ("predicted_kl_diagnostic_only"),
            }
        )
    else:
        required_boundary["calibration_split"] = "train_only"
    if any(boundary.get(key) != expected for key, expected in required_boundary.items()):
        raise ValueError(f"{path} violates the train/test information boundary")
    crn = _mapping(value["common_random_numbers"], name=f"{path}:common_random_numbers")
    if (
        crn.get("named_stream") != "rollout"
        or crn.get("same_per_prompt_seed_reset_across_arms") is not True
        or crn.get("candidate_index_alignment") is not True
    ):
        raise ValueError(f"{path} lacks the frozen common-random-number contract")

    arms = _mapping(value["arms"], name=f"{path}:arms")
    if set(arms) != set(PHASE2_ARM_ORDER):
        raise ValueError(f"{path} must contain exactly the four Phase-2 arms")
    metrics: dict[str, Mapping[str, float]] = {}
    expected_counts: dict[str, int] = {}
    kl_tail_by_arm: dict[str, Mapping[str, object]] = {}
    for arm_name in PHASE2_ARM_ORDER:
        (
            metrics[arm_name],
            expected_counts[arm_name],
            kl_tail_by_arm[arm_name],
        ) = _validate_arm(
            arms[arm_name],
            arm_name=arm_name,
            beta_common=beta_common,
            num_prompts=expected_num_prompts,
            candidates_per_prompt=expected_candidates,
            max_response_tokens=int(expected_runtime_contract["max_response_tokens"]),
            design_stage=expected_design_stage,
            name=f"{path}:arms.{arm_name}",
        )

    pre_oracle_safety_gate = _validate_pre_oracle_safety_gate(
        value["pre_oracle_safety_gate"],
        expected_runtime_contract=expected_runtime_contract,
        expected_design_stage=expected_design_stage,
        metrics_by_arm=metrics,
        name=f"{path}:pre_oracle_safety_gate",
    )

    max_length_contract = _required(
        expected_runtime_contract["max_length_gate"],
        name=f"{path}:runtime.max_length_gate",
        keys={"formal_gate", "formal_threshold", "measure_only"},
    )
    observed_max_length_rates = {
        arm_name: metrics[arm_name]["reached_max_length_rate"] for arm_name in PHASE2_ARM_ORDER
    }
    if expected_design_stage == "pilot":
        if (
            max_length_contract["formal_gate"] is not False
            or max_length_contract["measure_only"] is not True
        ):
            raise ValueError(f"{path} pilot max-length contract must be measure-only")
        max_length_threshold = _finite(
            max_length_contract["formal_threshold"],
            name=f"{path}:runtime.max_length_gate.formal_threshold",
        )
        max_length_passed = all(
            rate <= max_length_threshold for rate in observed_max_length_rates.values()
        )
    else:
        if (
            max_length_contract["formal_gate"] is not True
            or max_length_contract["measure_only"] is not False
        ):
            raise ValueError(f"{path} confirmatory max-length contract must be a formal gate")
        max_length_threshold = _finite(
            max_length_contract["formal_threshold"],
            name=f"{path}:runtime.max_length_gate.formal_threshold",
        )
        max_length_passed = all(
            rate <= max_length_threshold for rate in observed_max_length_rates.values()
        )
        if not max_length_passed:
            raise ValueError(f"{path} failed the frozen confirmatory max-length gate")
    max_length_gate_evidence = {
        "schema_version": "phase2-max-length-gate-evidence/v1",
        "design_stage": expected_design_stage,
        "measure_only": max_length_contract["measure_only"],
        "formal_gate": max_length_contract["formal_gate"],
        "formal_threshold": max_length_contract["formal_threshold"],
        "observed_reached_max_length_rate_by_arm": observed_max_length_rates,
        "passed": max_length_passed,
        "unified_pre_oracle_gate_passed": pre_oracle_safety_gate["passed"],
    }

    zero_utility = metrics["zero_b"]["mean_target_utility"]
    oracle_utility = metrics["oracle_step"]["mean_target_utility"]
    for arm_name in PHASE2_ARM_ORDER:
        expected_improvement = metrics[arm_name]["mean_target_utility"] - zero_utility
        expected_gap = oracle_utility - metrics[arm_name]["mean_target_utility"]
        if not _close(metrics[arm_name]["improvement_over_zero_b"], expected_improvement):
            raise ValueError(f"{path}:arms.{arm_name} improvement-over-zero arithmetic failed")
        if not _close(metrics[arm_name]["oracle_step_reference_gap"], expected_gap):
            raise ValueError(f"{path}:arms.{arm_name} oracle-step gap arithmetic failed")

    safety = _required(
        value["measured_kl_safety"],
        name=f"{path}:measured_kl_safety",
        keys={
            "schema_version",
            "cap",
            "passed",
            "measured_by_policy",
            "violations",
            "beta_retuned",
        },
    )
    cap = _finite(safety["cap"], name=f"{path}:measured_kl_safety.cap")
    if (
        safety["schema_version"] != "measured-kl-safety/v1"
        or safety["passed"] is not True
        or safety["violations"] != []
        or safety["beta_retuned"] is not False
        or not _close(
            cap,
            float(expected_runtime_contract["measured_kl_safety_cap"]),
            tolerance=1.0e-12,
        )
    ):
        raise ValueError(f"{path} did not pass the frozen KL safety gate")
    measured = _mapping(
        safety["measured_by_policy"],
        name=f"{path}:measured_kl_safety.measured_by_policy",
    )
    if set(measured) != set(PHASE2_ARM_ORDER):
        raise ValueError(f"{path} safety record must contain exactly four arms")
    for arm_name in PHASE2_ARM_ORDER:
        measured_kl = _finite(
            measured[arm_name],
            name=f"{path}:measured_kl_safety.{arm_name}",
            nonnegative=True,
        )
        if measured_kl > cap or not _close(
            measured_kl,
            metrics[arm_name]["mean_on_policy_kl_pi_updated_to_pi0"],
        ):
            raise ValueError(f"{path} safety and arm KL disagree for {arm_name}")

    rollout_path, rollout_sha = _validate_rollout_jsonl(
        path,
        recorded_reference=value["rollouts_jsonl"],
        recorded_sha256=value["rollouts_sha256"],
        expected_counts=expected_counts,
        beta_common=beta_common,
    )
    artifact_sha = _digest(
        value["artifact_metadata_sha256"],
        name=f"{path}:artifact_metadata_sha256",
    )
    manifest_sha = _digest(
        value["run_manifest_sha256"],
        name=f"{path}:run_manifest_sha256",
    )
    return _LoadedSeed(
        seed=seed,
        beta_common=beta_common,
        metrics={learner: metrics[learner] for learner in CANONICAL_LEARNERS},
        oracle_step_improvement=metrics["oracle_step"]["improvement_over_zero_b"],
        prorm_plus_improvement=metrics[PRORM_PLUS]["improvement_over_zero_b"],
        heldout_bt_minus_prorm_plus_local_regret=heldout_contrast,
        environment_identity=environment,
        control_gate_evidence=_mapping(
            head["gate_evidence"],
            name=f"{path}:head_training.gate_evidence",
        ),
        heldout_gate_evidence=heldout_gate_evidence,
        max_length_gate_evidence=max_length_gate_evidence,
        on_policy_kl_tail_by_arm=kl_tail_by_arm,
        source=Phase2AggregateSource(
            seed=seed,
            result_path=relative_posix_reference(path, base=reference_base),
            result_sha256=result_sha,
            rollouts_path=relative_posix_reference(rollout_path, base=reference_base),
            rollouts_sha256=rollout_sha,
            artifact_metadata_sha256=artifact_sha,
            run_manifest_sha256=manifest_sha,
            heads_sha256=head["heads_sha256"],
        ),
    )


@dataclass(frozen=True, slots=True)
class CommonBetaEvidence:
    """Design-bound gates over the overlay's complete seed-level observations."""

    all_validated_seed_control_gates_passed: bool
    numeric_control_tolerances_design_bound: bool
    design_stage: str
    design_formal_eligibility: bool
    heldout_bt_minus_prorm_plus_local_regret_ci_lower_positive: bool
    oracle_step_improvement_ci_lower_positive: bool
    prorm_plus_improvement_over_zero_b_ci_lower_positive: bool
    prorm_plus_minus_bt_target_utility_mean_positive: bool
    prorm_plus_minus_bt_target_utility_ci_lower_positive: bool

    @property
    def criteria_passed_under_current_gate_contract(self) -> bool:
        return (
            self.all_validated_seed_control_gates_passed
            and self.heldout_bt_minus_prorm_plus_local_regret_ci_lower_positive
            and self.oracle_step_improvement_ci_lower_positive
            and self.prorm_plus_improvement_over_zero_b_ci_lower_positive
            and self.prorm_plus_minus_bt_target_utility_mean_positive
            and self.prorm_plus_minus_bt_target_utility_ci_lower_positive
        )

    @property
    def passed(self) -> bool:
        return (
            self.criteria_passed_under_current_gate_contract
            and self.numeric_control_tolerances_design_bound
            and self.design_formal_eligibility
        )

    def to_dict(self) -> dict[str, object]:
        if self.passed:
            status = "passed"
        elif self.criteria_passed_under_current_gate_contract and (
            not self.design_formal_eligibility
        ):
            status = "pilot_gates_passed_formal_ineligible"
        elif self.criteria_passed_under_current_gate_contract:
            status = "numeric_gates_not_design_bound"
        else:
            status = "not_passed"
        return {
            "status": status,
            "supports_pre_registered_claim": self.passed,
            "design_stage": self.design_stage,
            "design_formal_eligibility": self.design_formal_eligibility,
            "criteria_passed_under_current_gate_contract": (
                self.criteria_passed_under_current_gate_contract
            ),
            "criteria": {
                "all_validated_seed_control_gates_passed": (
                    self.all_validated_seed_control_gates_passed
                ),
                "numeric_control_tolerances_bound_into_design_identity": (
                    self.numeric_control_tolerances_design_bound
                ),
                "design_is_formally_eligible": self.design_formal_eligibility,
                "heldout_bt_minus_prorm_plus_local_regret_paired_seed_ci_lower_positive": (
                    self.heldout_bt_minus_prorm_plus_local_regret_ci_lower_positive
                ),
                "oracle_step_improvement_over_zero_b_paired_seed_ci_lower_positive": (
                    self.oracle_step_improvement_ci_lower_positive
                ),
                "prorm_plus_improvement_over_zero_b_paired_seed_ci_lower_positive": (
                    self.prorm_plus_improvement_over_zero_b_ci_lower_positive
                ),
                "prorm_plus_minus_bt_target_utility_paired_mean_positive": (
                    self.prorm_plus_minus_bt_target_utility_mean_positive
                ),
                "prorm_plus_minus_bt_target_utility_paired_seed_ci_lower_positive": (
                    self.prorm_plus_minus_bt_target_utility_ci_lower_positive
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class CommonBetaSeedAggregate:
    """Strict JSON representation of one complete overlay-defined campaign."""

    source_config_hash: str
    phase2_design_sha256: str
    runtime_contract: Mapping[str, object]
    runtime_contract_sha256: str
    environment_identity: Mapping[str, object]
    seeds: tuple[int, ...]
    paired_prorm_plus_minus_bt: PairedMetricsAggregate
    heldout_bt_minus_prorm_plus: PairedMetricsAggregate
    oracle_step_positive_control: PairedMetricsAggregate
    prorm_plus_over_zero_b: PairedMetricsAggregate
    global_beta_contract: Mapping[str, object]
    control_gate_evidence: Mapping[str, object]
    evidence: CommonBetaEvidence
    sources: tuple[Phase2AggregateSource, ...]
    bootstrap_seed: int
    bootstrap_resamples: int

    def __post_init__(self) -> None:
        _digest(self.source_config_hash, name="source_config_hash")
        _digest(self.phase2_design_sha256, name="phase2_design_sha256")
        _digest(self.runtime_contract_sha256, name="runtime_contract_sha256")
        if _canonical_sha256(self.runtime_contract) != self.runtime_contract_sha256:
            raise ValueError("runtime contract digest mismatch")
        if not self.seeds or self.seeds != tuple(sorted(set(self.seeds))):
            raise ValueError("aggregate seeds must be a non-empty unique sorted tuple")
        if self.paired_prorm_plus_minus_bt.seeds != self.seeds:
            raise ValueError("paired learner aggregate uses the wrong seeds")
        if self.heldout_bt_minus_prorm_plus.seeds != self.seeds:
            raise ValueError("held-out local-regret aggregate uses the wrong seeds")
        if self.oracle_step_positive_control.seeds != self.seeds:
            raise ValueError("oracle control aggregate uses the wrong seeds")
        if self.prorm_plus_over_zero_b.seeds != self.seeds:
            raise ValueError("ProRM+-over-zero aggregate uses the wrong seeds")
        if tuple(source.seed for source in self.sources) != self.seeds:
            raise ValueError("aggregate sources must be seed-sorted and complete")
        if (
            self.bootstrap_seed != self.paired_prorm_plus_minus_bt.bootstrap_seed
            or self.bootstrap_seed != self.heldout_bt_minus_prorm_plus.bootstrap_seed
            or (self.bootstrap_seed != self.oracle_step_positive_control.bootstrap_seed)
            or self.bootstrap_seed != self.prorm_plus_over_zero_b.bootstrap_seed
        ):
            raise ValueError("all aggregates must use the overlay bootstrap seed")
        if self.bootstrap_resamples != (
            self.paired_prorm_plus_minus_bt.bootstrap_resamples
        ) or self.bootstrap_resamples != (self.oracle_step_positive_control.bootstrap_resamples):
            raise ValueError("all aggregates must use the overlay bootstrap resample count")
        if self.bootstrap_resamples != self.prorm_plus_over_zero_b.bootstrap_resamples:
            raise ValueError("all aggregates must use the overlay bootstrap resample count")
        if self.bootstrap_resamples != self.heldout_bt_minus_prorm_plus.bootstrap_resamples:
            raise ValueError("all aggregates must use the overlay bootstrap resample count")
        beta_contract_keys = {
            "schema_version",
            "design_stage",
            "rule",
            "beta_selection_split",
            "beta_source",
            "role",
            "frozen_global_beta",
            "beta_selected_from_current_seed_curvature",
            "all_confirmatory_seeds_match_frozen_global_beta",
            "per_seed_beta_common",
        }
        beta_contract = _required(
            self.global_beta_contract,
            name="global_beta_contract",
            keys=beta_contract_keys,
        )
        if set(beta_contract) != beta_contract_keys:
            raise ValueError("global_beta_contract contains unknown fields")
        runtime_stage = self.runtime_contract.get("stage")
        runtime_rule = self.runtime_contract.get("common_beta_rule")
        _expect(
            beta_contract["schema_version"],
            "phase2-campaign-global-beta-contract/v1",
            name="global_beta_contract.schema_version",
        )
        _expect(
            beta_contract["design_stage"],
            runtime_stage,
            name="global_beta_contract.design_stage",
        )
        _expect(
            beta_contract["rule"],
            runtime_rule,
            name="global_beta_contract.rule",
        )
        _expect(
            beta_contract["beta_selection_split"],
            self.runtime_contract.get("common_beta_calibration_split"),
            name="global_beta_contract.beta_selection_split",
        )
        _expect(
            beta_contract["beta_source"],
            self.runtime_contract.get("common_beta_source"),
            name="global_beta_contract.beta_source",
        )
        per_seed_beta = beta_contract["per_seed_beta_common"]
        if not isinstance(per_seed_beta, list) or [
            item.get("seed") if isinstance(item, Mapping) else None for item in per_seed_beta
        ] != list(self.seeds):
            raise ValueError("global-beta evidence must be complete and seed-sorted")
        if runtime_stage == "confirmatory":
            frozen_beta = _finite(
                self.runtime_contract.get("frozen_global_beta"),
                name="runtime_contract.frozen_global_beta",
            )
            if (
                beta_contract["role"] != "confirmatory_config_frozen_scalar_verification"
                or beta_contract["frozen_global_beta"] != frozen_beta
                or beta_contract["beta_selected_from_current_seed_curvature"] is not False
                or beta_contract["all_confirmatory_seeds_match_frozen_global_beta"] is not True
            ):
                raise ValueError("confirmatory global-beta aggregate contract is invalid")
            for item in per_seed_beta:
                if (
                    not isinstance(item, Mapping)
                    or _finite(
                        item.get("beta_common"),
                        name="global_beta_contract.per_seed.beta_common",
                    )
                    != frozen_beta
                ):
                    raise ValueError("a confirmatory seed did not use the frozen global beta")
        else:
            if (
                beta_contract["role"] != "pilot_per_seed_candidates_only"
                or beta_contract["frozen_global_beta"] is not None
                or beta_contract["beta_selected_from_current_seed_curvature"] is not True
                or beta_contract["all_confirmatory_seeds_match_frozen_global_beta"] is not None
            ):
                raise ValueError("pilot global-beta candidate contract is invalid")
        gates = _required(
            self.control_gate_evidence,
            name="control_gate_evidence",
            keys={
                "schema_version",
                "numeric_gate_contract",
                "on_policy_kl_tail_diagnostics",
                "per_seed",
                "all_seed_gates_passed",
            },
        )
        if gates["all_seed_gates_passed"] is not True:
            raise ValueError("accepted aggregate contains a failed seed control gate")
        numeric_contract = _mapping(
            gates["numeric_gate_contract"],
            name="control_gate_evidence.numeric_gate_contract",
        )
        if numeric_contract.get("design_bound") is not True:
            raise ValueError("aggregate numeric gates must come from the normalized overlay")
        per_seed = gates["per_seed"]
        if not isinstance(per_seed, list) or [
            item.get("seed") if isinstance(item, Mapping) else None for item in per_seed
        ] != list(self.seeds):
            raise ValueError("control gate evidence must be complete and seed-sorted")
        _validate_environment(self.environment_identity, name="environment_identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PHASE2_AGGREGATE_SCHEMA,
            "source_config_hash": self.source_config_hash,
            "phase2_design_sha256": self.phase2_design_sha256,
            "phase2_runtime_contract": dict(self.runtime_contract),
            "phase2_runtime_contract_sha256": self.runtime_contract_sha256,
            "environment_identity": dict(self.environment_identity),
            "seeds": list(self.seeds),
            "num_seeds": len(self.seeds),
            "experimental_unit": "seed",
            "prompt_or_candidate_pseudo_replication": False,
            "bootstrap": {
                "seed": self.bootstrap_seed,
                "resamples": self.bootstrap_resamples,
                "method": "paired_seed_percentile_bootstrap",
                "confidence_level": 0.95,
                "interval_sidedness": "two_sided",
                "effective_component_one_sided_alpha": 0.025,
                "interpretation": (
                    "frequentist uncertainty for the RNG expectation of the paired "
                    "contrast conditional on the frozen prompt pool, models, oracle, "
                    "and design; not a claim about an unrestricted human-prompt population"
                ),
            },
            "formal_inference_contract": {
                "estimand": (
                    "rng_expectation_of_paired_contrast_conditioned_on_frozen_prompt_pool_"
                    "models_oracle_and_design"
                ),
                "experimental_unit": "seed",
                "test_structure": "intersection_union_single_conjunctive_claim",
                "global_null": "at_least_one_required_mean_contrast_lte_zero",
                "global_alternative": "all_required_mean_contrasts_gt_zero",
                "component_interval_rule": (
                    "two_sided_95_percent_percentile_ci_lower_strictly_gt_zero"
                ),
                "effective_component_one_sided_alpha": 0.025,
                "multiplicity_adjustment": ("none_for_intersection_union_conjunctive_claim"),
                "separate_endpoint_claims_without_adjustment_allowed": False,
                "prompt_or_candidate_rows_as_independent_replicates": False,
            },
            "metric_roles": dict(_METRIC_ROLES),
            "paired_prorm_plus_minus_bt": self.paired_prorm_plus_minus_bt.to_dict(),
            "heldout_bt_minus_prorm_plus": {
                "formal_gate_split": "test",
                "reported_difference": "bt_mle_minus_prorm_plus",
                "direction": "higher_is_better",
                "aggregate": self.heldout_bt_minus_prorm_plus.to_dict(),
            },
            "oracle_step_positive_control": {
                "left_operand": "zero_b",
                "right_operand": "oracle_step",
                "reported_difference": "oracle_step_minus_zero_b",
                "aggregate": self.oracle_step_positive_control.to_dict(),
            },
            "prorm_plus_over_zero_b": {
                "left_operand": "zero_b",
                "right_operand": PRORM_PLUS,
                "reported_difference": "prorm_plus_minus_zero_b",
                "aggregate": self.prorm_plus_over_zero_b.to_dict(),
            },
            "global_beta_contract": dict(self.global_beta_contract),
            "control_gate_evidence": dict(self.control_gate_evidence),
            "pre_registered_evidence": self.evidence.to_dict(),
            "integrity": {
                "exact_overlay_seed_set": True,
                "same_source_design_runtime_and_environment": True,
                "all_kl_safety_gates_passed": True,
                "all_heads_fresh_r4": True,
                "all_head_control_fields_revalidated": True,
                "all_results_have_exact_four_arms": True,
                "all_sibling_rollout_paths_and_hashes_verified": True,
                "all_confirmatory_seeds_used_config_frozen_global_beta": (
                    self.global_beta_contract["all_confirmatory_seeds_match_frozen_global_beta"]
                ),
                "rollout_rows_used_for_inference": False,
            },
            "sources": [source.to_dict() for source in self.sources],
        }


def build_common_beta_seed_aggregate(
    overlay_config: Mapping[str, object],
    result_jsons: Sequence[str | os.PathLike[str]],
    *,
    reference_base: str | os.PathLike[str],
) -> CommonBetaSeedAggregate:
    """Validate and aggregate the complete seed set declared by the overlay."""

    validated = validate_phase2_config(overlay_config)
    run = _mapping(validated["run"], name="overlay.run")
    raw_seeds = run.get("seeds")
    if not isinstance(raw_seeds, list):
        raise ValueError("Phase-2 aggregation requires overlay.run.seeds")
    declared_seeds = tuple(_integer(seed, name="overlay.run.seeds") for seed in raw_seeds)
    if not declared_seeds or len(set(declared_seeds)) != len(declared_seeds):
        raise ValueError("Phase-2 aggregation requires a non-empty unique overlay seed set")
    expected_seeds = tuple(sorted(declared_seeds))
    if isinstance(result_jsons, (str, bytes, bytearray)) or not isinstance(
        result_jsons,
        Sequence,
    ):
        raise TypeError("result_jsons must be an explicit sequence of result paths")
    if len(result_jsons) != len(expected_seeds):
        raise ValueError("Phase-2 result path count must exactly equal the overlay seed count")
    path_keys = tuple(str(Path(path).resolve()) for path in result_jsons)
    if len(set(path_keys)) != len(path_keys):
        raise ValueError("Phase-2 result JSON paths must be unique")

    design = _mapping(validated["design"], name="overlay.design")
    stage = design["stage"]
    if stage not in {"pilot", "confirmatory"}:
        raise ValueError("overlay.design.stage is invalid after normalization")
    formal_eligibility = design["formal_eligibility"] is True
    source_config_hash = _digest(
        design["source_config_hash"],
        name="overlay.design.source_config_hash",
    )
    design_sha256 = phase2_design_identity(validated)
    runtime = Phase2Design.from_phase2_config(validated)
    runtime_contract = runtime.to_dict()
    runtime_sha256 = runtime.sha256
    split_sizes = _mapping(run["split_sizes"], name="overlay.run.split_sizes")
    train_prompts = _integer(
        split_sizes["train"],
        name="overlay.run.split_sizes.train",
        minimum=2,
    )
    validation_prompts = _integer(
        split_sizes["validation"],
        name="overlay.run.split_sizes.validation",
        minimum=2,
    )
    num_prompts = _integer(split_sizes["test"], name="overlay.run.split_sizes.test", minimum=2)
    data = _mapping(validated["data"], name="overlay.data")
    candidates = _integer(data["num_candidates"], name="overlay.data.num_candidates", minimum=1)
    reward_model = _mapping(validated["reward_model"], name="overlay.reward_model")
    adaptive_convergence = _mapping(
        reward_model["adaptive_convergence"],
        name="overlay.reward_model.adaptive_convergence",
    )
    identifiability_config = _mapping(
        reward_model["identifiability"],
        name="overlay.reward_model.identifiability",
    )
    outer_steps = _integer(
        reward_model["outer_steps"],
        name="overlay.reward_model.outer_steps",
        minimum=1,
    )
    annotations = _mapping(validated["annotations"], name="overlay.annotations")
    prohibit_label_clipping = annotations.get("prohibit_clipping") is True
    controls = _mapping(validated["positive_controls"], name="overlay.positive_controls")
    numeric_tolerances = _mapping(
        controls["numeric_gate_tolerances"],
        name="overlay.positive_controls.numeric_gate_tolerances",
    )
    exact_soft_label_bt = _mapping(
        controls["exact_soft_label_bt"],
        name="overlay.positive_controls.exact_soft_label_bt",
    )
    low_dimensional = _mapping(
        controls["low_dimensional_tangent"],
        name="overlay.positive_controls.low_dimensional_tangent",
    )
    low_dimension = _integer(
        low_dimensional["dimension"],
        name="overlay.positive_controls.low_dimensional_tangent.dimension",
        minimum=1,
    )
    projection_namespace = low_dimensional["seed_namespace"]
    if not isinstance(projection_namespace, str) or not projection_namespace:
        raise ValueError("overlay low-dimensional projection namespace must be non-empty")
    eigenvalue_tolerance = _finite(
        low_dimensional["relative_eigenvalue_tolerance"],
        name="overlay.positive_controls.low_dimensional_tangent.relative_eigenvalue_tolerance",
    )
    objective = _mapping(validated["objective"], name="overlay.objective")
    full_tangent = _mapping(objective["full_tangent"], name="overlay.objective.full_tangent")
    ridge = _mapping(full_tangent["ridge"], name="overlay.objective.full_tangent.ridge")
    pcg_tolerance = _finite(
        ridge["pcg_tolerance"],
        name="overlay.objective.full_tangent.ridge.pcg_tolerance",
    )
    pcg_max_iterations = _integer(
        ridge["pcg_max_iterations"],
        name="overlay.objective.full_tangent.ridge.pcg_max_iterations",
        minimum=1,
    )
    relative_damping = _finite(
        ridge["relative_coefficient"],
        name="overlay.objective.full_tangent.ridge.relative_coefficient",
    )
    evaluation = _mapping(validated["evaluation"], name="overlay.evaluation")
    bootstrap_seed = _integer(
        evaluation["paired_bootstrap_seed"],
        name="overlay.evaluation.paired_bootstrap_seed",
    )
    bootstrap_resamples = _integer(
        evaluation["paired_bootstrap_resamples"],
        name="overlay.evaluation.paired_bootstrap_resamples",
        minimum=1,
    )
    base = Path(reference_base).resolve()

    loaded: dict[int, _LoadedSeed] = {}
    for raw_path in result_jsons:
        item = _load_seed_result(
            raw_path,
            expected_source_config_hash=source_config_hash,
            expected_design_sha256=design_sha256,
            expected_design_stage=str(stage),
            expected_formal_eligibility=formal_eligibility,
            expected_runtime_contract=runtime_contract,
            expected_runtime_sha256=runtime_sha256,
            expected_train_prompts=train_prompts,
            expected_validation_prompts=validation_prompts,
            expected_num_prompts=num_prompts,
            expected_candidates=candidates,
            expected_outer_steps=outer_steps,
            expected_low_dimension=low_dimension,
            expected_projection_namespace=projection_namespace,
            expected_eigenvalue_tolerance=eigenvalue_tolerance,
            expected_pcg_tolerance=pcg_tolerance,
            expected_pcg_max_iterations=pcg_max_iterations,
            expected_relative_damping=relative_damping,
            prohibit_label_clipping=prohibit_label_clipping,
            numeric_tolerances=numeric_tolerances,
            adaptive_convergence=adaptive_convergence,
            identifiability_config=identifiability_config,
            reward_model_config=reward_model,
            exact_soft_label_bt_config=exact_soft_label_bt,
            reference_base=base,
        )
        if item.seed in loaded:
            raise ValueError(f"duplicate Phase-2 result for seed {item.seed}")
        loaded[item.seed] = item
    if set(loaded) != set(expected_seeds):
        raise ValueError(
            "Phase-2 result seeds must exactly match the complete overlay seed set; "
            f"missing={sorted(set(expected_seeds) - set(loaded))!r}, "
            f"unexpected={sorted(set(loaded) - set(expected_seeds))!r}"
        )

    per_seed_beta_common = [
        {"seed": seed, "beta_common": loaded[seed].beta_common} for seed in expected_seeds
    ]
    if stage == "confirmatory":
        frozen_global_beta = _finite(
            runtime_contract.get("frozen_global_beta"),
            name="overlay runtime frozen_global_beta",
        )
        if any(loaded[seed].beta_common != frozen_global_beta for seed in expected_seeds):
            raise ValueError(
                "all confirmatory seeds must use the exact global beta frozen in the overlay"
            )
        global_beta_contract: dict[str, object] = {
            "schema_version": "phase2-campaign-global-beta-contract/v1",
            "design_stage": "confirmatory",
            "rule": runtime_contract["common_beta_rule"],
            "beta_selection_split": runtime_contract["common_beta_calibration_split"],
            "beta_source": runtime_contract["common_beta_source"],
            "role": "confirmatory_config_frozen_scalar_verification",
            "frozen_global_beta": frozen_global_beta,
            "beta_selected_from_current_seed_curvature": False,
            "all_confirmatory_seeds_match_frozen_global_beta": True,
            "per_seed_beta_common": per_seed_beta_common,
        }
    else:
        global_beta_contract = {
            "schema_version": "phase2-campaign-global-beta-contract/v1",
            "design_stage": "pilot",
            "rule": runtime_contract["common_beta_rule"],
            "beta_selection_split": runtime_contract["common_beta_calibration_split"],
            "beta_source": runtime_contract["common_beta_source"],
            "role": "pilot_per_seed_candidates_only",
            "frozen_global_beta": None,
            "beta_selected_from_current_seed_curvature": True,
            "all_confirmatory_seeds_match_frozen_global_beta": None,
            "per_seed_beta_common": per_seed_beta_common,
        }

    shared_environment = dict(loaded[expected_seeds[0]].environment_identity)
    if any(
        dict(loaded[seed].environment_identity) != shared_environment for seed in expected_seeds[1:]
    ):
        raise ValueError(
            "all overlay seeds must share Git, image, HF inventory, account, partition, "
            "and one GPU model"
        )

    bt_by_seed = {seed: dict(loaded[seed].metrics[BT_MLE]) for seed in expected_seeds}
    prorm_by_seed = {seed: dict(loaded[seed].metrics[PRORM_PLUS]) for seed in expected_seeds}
    paired = aggregate_paired_metrics(
        bt_by_seed,
        prorm_by_seed,
        directions=_PAIR_METRIC_DIRECTIONS,
        bootstrap_seed=bootstrap_seed,
        num_resamples=bootstrap_resamples,
    )
    zero_by_seed = {seed: {"oracle_step_improvement_over_zero_b": 0.0} for seed in expected_seeds}
    oracle_by_seed = {
        seed: {"oracle_step_improvement_over_zero_b": loaded[seed].oracle_step_improvement}
        for seed in expected_seeds
    }
    oracle_control = aggregate_paired_metrics(
        zero_by_seed,
        oracle_by_seed,
        directions={"oracle_step_improvement_over_zero_b": "higher_is_better"},
        bootstrap_seed=bootstrap_seed,
        num_resamples=bootstrap_resamples,
    )
    zero_for_prorm_by_seed = {
        seed: {"prorm_plus_improvement_over_zero_b": 0.0} for seed in expected_seeds
    }
    prorm_over_zero_by_seed = {
        seed: {"prorm_plus_improvement_over_zero_b": loaded[seed].prorm_plus_improvement}
        for seed in expected_seeds
    }
    prorm_over_zero = aggregate_paired_metrics(
        zero_for_prorm_by_seed,
        prorm_over_zero_by_seed,
        directions={"prorm_plus_improvement_over_zero_b": "higher_is_better"},
        bootstrap_seed=bootstrap_seed,
        num_resamples=bootstrap_resamples,
    )
    heldout_zero_by_seed = {
        seed: {"bt_mle_minus_prorm_plus_heldout_contrast": 0.0} for seed in expected_seeds
    }
    heldout_contrast_by_seed = {
        seed: {
            "bt_mle_minus_prorm_plus_heldout_contrast": (
                loaded[seed].heldout_bt_minus_prorm_plus_local_regret
            )
        }
        for seed in expected_seeds
    }
    heldout_local_regret = aggregate_paired_metrics(
        heldout_zero_by_seed,
        heldout_contrast_by_seed,
        directions={"bt_mle_minus_prorm_plus_heldout_contrast": "higher_is_better"},
        bootstrap_seed=bootstrap_seed,
        num_resamples=bootstrap_resamples,
    )
    per_seed_control_gates = [
        {
            "seed": seed,
            "result_sha256": loaded[seed].source.result_sha256,
            "gates": dict(loaded[seed].control_gate_evidence),
            "heldout_fixed_beta": dict(loaded[seed].heldout_gate_evidence),
            "max_length": dict(loaded[seed].max_length_gate_evidence),
            "on_policy_kl_tail_by_arm": {
                arm_name: dict(loaded[seed].on_policy_kl_tail_by_arm[arm_name])
                for arm_name in PHASE2_ARM_ORDER
            },
        }
        for seed in expected_seeds
    ]
    all_seed_control_gates_passed = all(
        item["gates"].get("passed") is True
        and item["heldout_fixed_beta"].get("passed_integrity") is True
        and item["max_length"].get("passed") is True
        for item in per_seed_control_gates
    )
    kl_tail_fields = ("mean", "p50", "p90", "p95", "p99", "maximum", "per_sequence_maximum")
    kl_tail_across_seeds = {
        arm_name: {
            field: {
                "seed_mean": sum(
                    float(loaded[seed].on_policy_kl_tail_by_arm[arm_name][field])
                    for seed in expected_seeds
                )
                / len(expected_seeds),
                "seed_maximum": max(
                    float(loaded[seed].on_policy_kl_tail_by_arm[arm_name][field])
                    for seed in expected_seeds
                ),
            }
            for field in kl_tail_fields
        }
        for arm_name in PHASE2_ARM_ORDER
    }
    control_gate_evidence = {
        "schema_version": "phase2-campaign-control-gates/v1",
        "numeric_gate_contract": _numeric_gate_contract(
            numeric_tolerances,
            adaptive_convergence,
            stage=str(stage),
            formal_eligibility=formal_eligibility,
        ),
        "on_policy_kl_tail_diagnostics": {
            "schema_version": "phase2-campaign-kl-tail-diagnostics/v1",
            "role": (
                "pilot_locality_tail_measurement"
                if stage == "pilot"
                else "confirmatory_pre_oracle_safety_gate"
            ),
            "formal_gate_applied": stage == "confirmatory",
            "formal_threshold": (
                None
                if stage == "pilot"
                else {
                    "mean_policy_to_reference_kl_cap": 0.02,
                    "prompt_mean_p95_kl_cap": 0.02,
                    "prompt_mean_p99_kl_cap": 0.05,
                    "prompt_mean_maximum_kl_cap": 0.10,
                    "per_sequence_maximum_kl_cap": 0.20,
                }
            ),
            "experimental_unit_for_summary": "seed",
            "across_seed_by_arm": kl_tail_across_seeds,
        },
        "per_seed": per_seed_control_gates,
        "all_seed_gates_passed": all_seed_control_gates_passed,
    }

    paired_summaries = {summary.metric: summary for summary in paired.metrics}
    target = paired_summaries["mean_target_utility"]
    oracle_summary = oracle_control.metrics[0]
    prorm_zero_summary = prorm_over_zero.metrics[0]
    heldout_summary = heldout_local_regret.metrics[0]
    evidence = CommonBetaEvidence(
        all_validated_seed_control_gates_passed=all_seed_control_gates_passed,
        numeric_control_tolerances_design_bound=True,
        design_stage=str(stage),
        design_formal_eligibility=formal_eligibility,
        heldout_bt_minus_prorm_plus_local_regret_ci_lower_positive=(
            heldout_summary.bootstrap_ci.lower > 0.0
        ),
        oracle_step_improvement_ci_lower_positive=(oracle_summary.bootstrap_ci.lower > 0.0),
        prorm_plus_improvement_over_zero_b_ci_lower_positive=(
            prorm_zero_summary.bootstrap_ci.lower > 0.0
        ),
        prorm_plus_minus_bt_target_utility_mean_positive=(target.paired_mean > 0.0),
        prorm_plus_minus_bt_target_utility_ci_lower_positive=(target.bootstrap_ci.lower > 0.0),
    )
    return CommonBetaSeedAggregate(
        source_config_hash=source_config_hash,
        phase2_design_sha256=design_sha256,
        runtime_contract=runtime_contract,
        runtime_contract_sha256=runtime_sha256,
        environment_identity=shared_environment,
        seeds=expected_seeds,
        paired_prorm_plus_minus_bt=paired,
        heldout_bt_minus_prorm_plus=heldout_local_regret,
        oracle_step_positive_control=oracle_control,
        prorm_plus_over_zero_b=prorm_over_zero,
        global_beta_contract=global_beta_contract,
        control_gate_evidence=control_gate_evidence,
        evidence=evidence,
        sources=tuple(loaded[seed].source for seed in expected_seeds),
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )


def write_common_beta_seed_aggregate(
    overlay_config: Mapping[str, object],
    result_jsons: Sequence[str | os.PathLike[str]],
    output_json: str | os.PathLike[str],
) -> CommonBetaSeedAggregate:
    """Build and atomically publish a new aggregate, never overwrite one."""

    destination = Path(output_json)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing aggregate: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    aggregate = build_common_beta_seed_aggregate(
        overlay_config,
        result_jsons,
        reference_base=destination.parent,
    )
    atomic_write_json(destination, aggregate.to_dict(), overwrite=False)
    return aggregate


__all__ = [
    "PHASE2_AGGREGATE_SCHEMA",
    "CommonBetaEvidence",
    "CommonBetaSeedAggregate",
    "Phase2AggregateSource",
    "build_common_beta_seed_aggregate",
    "write_common_beta_seed_aggregate",
]
