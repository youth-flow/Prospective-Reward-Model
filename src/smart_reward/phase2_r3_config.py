"""Closed, standalone science contract for the Phase-2 R3 recovery campaign.

R3 does not inherit or compile the legacy R2 overlay.  The legacy file is read
only to prove equality of an explicit list of migrated scientific fields.  Its
design identity is deliberately absent from the normalized R3 document and
therefore cannot contribute to the R3 semantic identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .phase2_training import (
    AdamWRecoveryProtocol,
    FirstOrderConvergenceSpec,
    LearningRateStage,
    Phase2TrainingSettings,
)

R3_SCIENCE_CONFIG_SCHEMA = "phase2-recovery-r3-science-config/v1"
R3_SCIENCE_CONFIG_PATH = "configs/phase2_recovery_r3_science.yaml"
R3_PRIMARY_SEEDS = (20260801, 20260802, 20260803)
R3_PRIMARY_HEADS = ("bt_mle", "prorm_plus")
R3_RECOVERY_SCHEDULE_SHA256 = "46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"
_R2_RECOVERY_DESIGN_SHA256 = "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"
_MATERIALIZATION_SOURCE_SHA256 = "81ccbd3bc9d745d3792d1834116ba2480d34a7201d6f4e463a61d8d8ab0baefa"
_R2_SOURCE_PATH = "configs/common_beta_recovery_pilot.yaml"
_R2_SOURCE_SCHEMA = "prorm-common-beta-recovery-config/v1"
_LABEL_RNG_NAMESPACE = "prorm-common-beta-r4-labels-v1"
_RECOVERY_TIE_BREAK = "exact_zero_initialized_deterministic_adamw_lr_decay_path"

_EXPECTED_CAMPAIGN = {
    "name": "phase2-recovery-r3-primary-only",
    "campaign_kind": "phase2_recovery_revision3_primary_only",
    "execution_revision": 3,
    "execution_scope": "primary_only",
    "evidence_role": "train_only_nonconfirmatory_recovery",
    "confirmatory": False,
    "formal_eligibility": False,
    "inherited_r2_design": False,
    "primary_heads": list(R3_PRIMARY_HEADS),
}
_EXPECTED_MATERIALIZATION = {
    "source_config_hash": _MATERIALIZATION_SOURCE_SHA256,
    "reuse_scope": "immutable_parent_materialization_only",
    "reward_or_label_artifact_reuse": False,
    "optimizer_state_reuse": False,
}
_EXPECTED_ANNOTATION_GATES = {
    "action": "fail_closed",
    "require_all": [
        "exactly_four_replicate_boundaries",
        "single_generator_initial_final_state_and_draw_count",
        "replicate_tensor_hashes_preserve_boundaries",
        "no_label_clipping",
        "bt_uses_all_raw_bernoulli_labels",
        "prorm_uses_arithmetic_mean_of_four_unclipped_estimators",
    ],
}
_EXPECTED_ANNOTATIONS = {
    "scheme": "geometric_randomized_truncation",
    "gamma": 0.9,
    "independent_replicates_per_edge": 4,
    "replicate_reduction": "arithmetic_mean",
    "prohibit_clipping": True,
    "bt_label_use": "all_underlying_bernoulli_labels",
    "replicate_rng": (
        "single_named_generator_sequential_independent_draws_with_preserved_boundaries"
    ),
    "named_rng_namespace": _LABEL_RNG_NAMESPACE,
    "decision_gates": _EXPECTED_ANNOTATION_GATES,
}
_EXPECTED_LR_STAGES = [
    {"first_update": 1, "last_update": 5760, "learning_rate": 1.0e-3},
    {"first_update": 5761, "last_update": 6760, "learning_rate": 3.0e-4},
    {"first_update": 6761, "last_update": 8760, "learning_rate": 1.0e-4},
    {"first_update": 8761, "last_update": 10760, "learning_rate": 3.0e-5},
    {"first_update": 10761, "last_update": 12760, "learning_rate": 1.0e-5},
]
_EXPECTED_OPTIMIZER_PROTOCOL = {
    "schema_version": "deterministic-adamw-lr-decay-recovery/v1",
    "one_time_recovery": True,
    "scope": "every_phase2_first_order_convergence_trainer",
    "initialization": "exact_zero_head_and_fresh_optimizer_state",
    "learning_rate_schedule": {
        "update_indexing": "one_indexed_inclusive",
        "application": "set_learning_rate_immediately_before_optimizer_update",
        "stages": _EXPECTED_LR_STAGES,
        "schedule_sha256": R3_RECOVERY_SCHEDULE_SHA256,
    },
    "legacy_constant_lr_boundary_snapshot_steps": 5760,
    "state_transition": "preserve_all_adamw_moments_across_learning_rate_boundaries",
    "adamw": {
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "differentiable": False,
    },
    "reward_head_dtype": "float32",
    "first_order_audit_dtype": "float64",
    "microbatch_order": "canonical_edge_order_contiguous_ascending_no_shuffle",
    "optimizer_state_reset_at_lr_milestone": False,
    "one_optimizer_update_per_step": True,
    "tie_break": _RECOVERY_TIE_BREAK,
    "validation_or_test_selection": False,
}
_EXPECTED_CONVERGENCE = {
    "relative_gradient_ratio_tolerance": 1.0e-3,
    "minimum_steps": 100,
    "maximum_steps": 12760,
    "check_interval_steps": 20,
    "consecutive_passing_checks": 3,
    "compute_matched_checkpoint_steps": 720,
    "gradient_measurement": "post_update_full_data_unclipped",
    "denominator": "exact_zero_initialization_gradient_l2_norm",
    "denominator_floor": 1.0e-30,
    "prorm_pcg_audit_initialization": "cold_start_zero",
    "fail_closed": True,
    "solution_tie_break": _RECOVERY_TIE_BREAK,
    "unique_solution_claim": False,
    "validation_or_test_selection": False,
    "primary_heads_required_to_converge": list(R3_PRIMARY_HEADS),
}
_EXPECTED_IDENTIFIABILITY = {
    "design_matrix": "reward_feature_difference_design_matrix",
    "split": "train",
    "relative_rank_tolerance": 1.0e-10,
    "role": "pilot_measure_only",
    "require_full_column_rank": False,
    "algorithmic_tie_break": _RECOVERY_TIE_BREAK,
    "minimum_norm_claim": False,
    "confirmatory_freeze_requirement": (
        "decide_gate_from_train_only_pilot_then_issue_new_identity"
    ),
}
_EXPECTED_REWARD_MODEL = {
    "dtype": "float32",
    "outer_steps": 720,
    "refresh_dual_every_steps": 1,
    "optimizer": "adamw",
    "learning_rate": 1.0e-3,
    "weight_decay": 0.0,
    "microbatch_size": 64,
    "max_grad_norm": 1.0,
    "optimizer_protocol": _EXPECTED_OPTIMIZER_PROTOCOL,
    "adaptive_convergence": _EXPECTED_CONVERGENCE,
    "identifiability": _EXPECTED_IDENTIFIABILITY,
}
_EXPECTED_RIDGE = {
    "enabled": True,
    "rule": "relative_to_mean_fisher_diagonal",
    "relative_coefficient": 0.001,
    "sensitivity_multipliers": [0.1, 1.0, 10.0],
    "primary_execution_role": "pilot_candidate_primary",
    "sensitivity_execution_role": "required_separate_pilot_sensitivity",
    "sensitivity_executed_separately": True,
    "sensitivity_eligible_for_primary_claim": False,
    "solver_dtype": "float64",
    "pcg_tolerance": 1.0e-5,
    "pcg_max_iterations": 8192,
}
_EXPECTED_OBJECTIVE = {
    "training_beta": 1.0,
    "probability_floor": 0.25,
    "full_tangent_ridge": _EXPECTED_RIDGE,
    "pcg_absolute_tolerance": 0.0,
    "pcg_residual_recompute_interval": 20,
    "require_pcg_convergence": True,
}
_EXPECTED_DIAGNOSTICS = {
    "snapshots": [
        {"update": 720, "role": "compute_matched_diagnostic_only"},
        {"update": 5760, "role": "legacy_boundary_diagnostic_only"},
    ],
    "may_select_primary_iterate": False,
    "may_change_science_config": False,
}
_EXPECTED_EXECUTION_BOUNDARY = {
    "execution_scope": "train_only",
    "policy_rollout_allowed": False,
    "validation_or_test_access_allowed": False,
    "final_oracle_allowed": False,
    "downstream_utility_allowed": False,
    "controls_executed_in_primary_run": False,
    "profile_evidence_reusable": False,
}
_EXPECTED_LOW_DIMENSIONAL = {
    "enabled": True,
    "construction": "seeded_orthonormal_projection",
    "dimension": 256,
    "seed_namespace": "prorm-common-beta-low-dimensional-tangent-v1",
    "regularization": "moore_penrose_pseudoinverse",
    "relative_eigenvalue_tolerance": 1.0e-10,
    "eligible_for_primary_claim": False,
}
_EXPECTED_EXACT_SOFT_BT = {
    "enabled": True,
    "role": "noise_free_positive_control_and_secondary_misspecification_diagnostic",
    "noise_free": True,
    "input": "sigmoid_of_train_transformed_oracle_margin",
    "eligible_for_primary_claim": False,
}
_EXPECTED_SETTINGS_TYPE_CONTRACT = {
    "role": "nonexecuted_phase2_training_settings_compatibility",
    "low_dimensional_tangent": _EXPECTED_LOW_DIMENSIONAL,
    "exact_soft_label_bt": _EXPECTED_EXACT_SOFT_BT,
    "low_dimensional_source_layout_id": "training-policy-score-flatten-order/v1",
    "max_total_annotations": None,
}
_EXPECTED_MIGRATION_PAIRS = {
    "run.seeds": "run.seeds",
    "campaign.primary_heads": (
        "reward_model.adaptive_convergence.primary_heads_required_to_converge"
    ),
    "campaign.confirmatory": "run.confirmatory",
    "campaign.formal_eligibility": "run.formal_eligibility",
    "materialization.source_config_hash": "design.source_config_hash",
    "annotations.scheme": "annotations.scheme",
    "annotations.gamma": "annotations.gamma",
    "annotations.independent_replicates_per_edge": ("annotations.independent_replicates_per_edge"),
    "annotations.replicate_reduction": "annotations.replicate_reduction",
    "annotations.prohibit_clipping": "annotations.prohibit_clipping",
    "annotations.bt_label_use": "annotations.bt_label_use",
    "annotations.replicate_rng": "annotations.replicate_rng",
    "annotations.decision_gates": "annotations.decision_gates",
    "reward_model.dtype": "reward_model.dtype",
    "reward_model.outer_steps": "reward_model.outer_steps",
    "reward_model.refresh_dual_every_steps": "reward_model.refresh_dual_every_steps",
    "reward_model.optimizer": "reward_model.optimizer",
    "reward_model.learning_rate": "reward_model.learning_rate",
    "reward_model.weight_decay": "reward_model.weight_decay",
    "reward_model.microbatch_size": "reward_model.microbatch_size",
    "reward_model.max_grad_norm": "reward_model.max_grad_norm",
    "reward_model.optimizer_protocol": "reward_model.optimizer_protocol",
    "reward_model.adaptive_convergence": "reward_model.adaptive_convergence",
    "reward_model.identifiability": "reward_model.identifiability",
    "objective.probability_floor": "oracle.probability_floor",
    "objective.full_tangent_ridge": "objective.full_tangent.ridge",
    "execution_boundary.execution_scope": "recovery_control.execution_scope",
    "execution_boundary.policy_rollout_allowed": "recovery_control.policy_rollout_allowed",
    "execution_boundary.validation_or_test_access_allowed": (
        "recovery_control.validation_or_test_access_allowed"
    ),
    "execution_boundary.final_oracle_allowed": "recovery_control.final_oracle_allowed",
    "execution_boundary.downstream_utility_allowed": (
        "recovery_control.downstream_utility_allowed"
    ),
    "settings_type_contract.low_dimensional_tangent": ("positive_controls.low_dimensional_tangent"),
    "settings_type_contract.exact_soft_label_bt": ("positive_controls.exact_soft_label_bt"),
}
_EXPECTED_MIGRATION = {
    "source_path": _R2_SOURCE_PATH,
    "source_schema_version": _R2_SOURCE_SCHEMA,
    "mode": "explicit_field_equality_only_no_inheritance",
    "r2_design_identity_part_of_r3_semantic_identity": False,
    "field_pairs": _EXPECTED_MIGRATION_PAIRS,
}
_EXPECTED_ROOT_KEYS = {
    "schema_version",
    "campaign",
    "materialization",
    "run",
    "annotations",
    "reward_model",
    "objective",
    "diagnostics",
    "execution_boundary",
    "settings_type_contract",
    "migration_equality",
}


class R3ScienceConfigError(ValueError):
    """The standalone R3 science contract is absent, changed, or malformed."""


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if set(left) != set(right):  # type: ignore[arg-type]
            return False
        return all(_strict_equal(left[key], right[key]) for key in left)  # type: ignore[index]
    if isinstance(left, list):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    return bool(left == right)


def _strict_json_copy(value: object, *, path: str = "config") -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise R3ScienceConfigError(f"{path} must not contain a non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if type(value) is not dict:
            value = dict(value)
        if any(type(key) is not str for key in value):
            raise R3ScienceConfigError(f"{path} must contain only string keys")
        return {key: _strict_json_copy(value[key], path=f"{path}.{key}") for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _strict_json_copy(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise R3ScienceConfigError(f"{path} contains unsupported value type {type(value).__name__}")


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise R3ScienceConfigError(f"{name} must be a lowercase SHA256 digest")
    return value


def _parse_yaml(raw: bytes, *, source: Path) -> object:
    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[object, object]:
        loader.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as error:
                raise R3ScienceConfigError(
                    f"{source} contains an unhashable YAML mapping key"
                ) from error
            if duplicate:
                raise R3ScienceConfigError(f"{source} contains duplicate key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        unique_mapping,
    )
    try:
        text = raw.decode("utf-8")
        return yaml.load(text, Loader=UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise R3ScienceConfigError(f"{source} is not UTF-8 YAML") from error
    except yaml.YAMLError as error:
        raise R3ScienceConfigError(f"failed to parse R3 science config {source}") from error


def _read_stable_regular_bytes(path: Path, *, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise R3ScienceConfigError(f"cannot stat {name} {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise R3ScienceConfigError(f"{name} must be a non-symlink regular file: {path}")
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise R3ScienceConfigError(f"cannot read {name} {path}: {error}") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if (
        path.is_symlink()
        or not stat.S_ISREG(after.st_mode)
        or before_identity != after_identity
        or len(raw) != after.st_size
    ):
        raise R3ScienceConfigError(f"{name} changed while it was being read: {path}")
    return raw


def _mapping(value: object, *, path: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise R3ScienceConfigError(f"{path} must be a mapping with string keys")
    return value


def _require_exact(value: object, expected: object, *, path: str) -> None:
    if not _strict_equal(value, expected):
        raise R3ScienceConfigError(f"{path} differs from the locked R3 science contract")


def _value_at_path(root: Mapping[str, object], dotted_path: str) -> object:
    value: object = root
    traversed: list[str] = []
    for component in dotted_path.split("."):
        traversed.append(component)
        mapping = _mapping(value, path=".".join(traversed[:-1]) or "config")
        if component not in mapping:
            raise R3ScienceConfigError(f"migration source is missing field {dotted_path!r}")
        value = mapping[component]
    return value


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_migration_equality(normalized: Mapping[str, object]) -> None:
    source_path = _repository_root().joinpath(*_R2_SOURCE_PATH.split("/"))
    raw = _read_stable_regular_bytes(source_path, name="R2 migration equality source")
    parsed = _parse_yaml(raw, source=source_path)
    r2 = _mapping(parsed, path="R2 migration equality source")
    if r2.get("schema_version") != _R2_SOURCE_SCHEMA:
        raise R3ScienceConfigError(
            "R2 migration equality source has the wrong schema; R3 does not accept "
            "legacy schemas as its own configuration"
        )
    for r3_path, r2_path in _EXPECTED_MIGRATION_PAIRS.items():
        r3_value = _value_at_path(normalized, r3_path)
        r2_value = _value_at_path(r2, r2_path)
        if not _strict_equal(r3_value, r2_value):
            raise R3ScienceConfigError(
                f"R3 migrated field {r3_path!r} differs from R2 source field {r2_path!r}"
            )


def _validate_and_normalize(parsed: object) -> dict[str, object]:
    root = _mapping(parsed, path="R3 science config")
    if set(root) != _EXPECTED_ROOT_KEYS:
        missing = sorted(_EXPECTED_ROOT_KEYS - set(root))
        unknown = sorted(set(root) - _EXPECTED_ROOT_KEYS)
        raise R3ScienceConfigError(
            f"R3 science config is not closed: missing={missing!r}, unknown={unknown!r}"
        )
    if root["schema_version"] != R3_SCIENCE_CONFIG_SCHEMA:
        raise R3ScienceConfigError(
            f"schema_version must equal {R3_SCIENCE_CONFIG_SCHEMA!r}; "
            "legacy Phase-2 overlays are not accepted"
        )
    _require_exact(root["campaign"], _EXPECTED_CAMPAIGN, path="campaign")
    _require_exact(
        root["materialization"],
        _EXPECTED_MATERIALIZATION,
        path="materialization",
    )
    _require_exact(root["run"], {"seeds": list(R3_PRIMARY_SEEDS)}, path="run")
    _require_exact(root["annotations"], _EXPECTED_ANNOTATIONS, path="annotations")
    _require_exact(root["reward_model"], _EXPECTED_REWARD_MODEL, path="reward_model")
    _require_exact(root["objective"], _EXPECTED_OBJECTIVE, path="objective")
    _require_exact(root["diagnostics"], _EXPECTED_DIAGNOSTICS, path="diagnostics")
    _require_exact(
        root["execution_boundary"],
        _EXPECTED_EXECUTION_BOUNDARY,
        path="execution_boundary",
    )
    _require_exact(
        root["settings_type_contract"],
        _EXPECTED_SETTINGS_TYPE_CONTRACT,
        path="settings_type_contract",
    )
    _require_exact(
        root["migration_equality"],
        _EXPECTED_MIGRATION,
        path="migration_equality",
    )
    normalized = _strict_json_copy(root)
    if type(normalized) is not dict:
        raise RuntimeError("internal R3 normalization did not return a mapping")
    _validate_migration_equality(normalized)
    return normalized


def _build_settings(
    normalized: Mapping[str, object],
    *,
    semantic_sha256: str,
) -> Phase2TrainingSettings:
    campaign = _mapping(normalized["campaign"], path="campaign")
    materialization = _mapping(normalized["materialization"], path="materialization")
    run = _mapping(normalized["run"], path="run")
    annotations = _mapping(normalized["annotations"], path="annotations")
    reward_model = _mapping(normalized["reward_model"], path="reward_model")
    optimizer_protocol = _mapping(
        reward_model["optimizer_protocol"],
        path="reward_model.optimizer_protocol",
    )
    schedule = _mapping(
        optimizer_protocol["learning_rate_schedule"],
        path="reward_model.optimizer_protocol.learning_rate_schedule",
    )
    convergence = _mapping(
        reward_model["adaptive_convergence"],
        path="reward_model.adaptive_convergence",
    )
    identifiability = _mapping(
        reward_model["identifiability"],
        path="reward_model.identifiability",
    )
    objective = _mapping(normalized["objective"], path="objective")
    ridge = _mapping(objective["full_tangent_ridge"], path="objective.full_tangent_ridge")
    type_contract = _mapping(
        normalized["settings_type_contract"],
        path="settings_type_contract",
    )
    low_dimensional = _mapping(
        type_contract["low_dimensional_tangent"],
        path="settings_type_contract.low_dimensional_tangent",
    )
    exact_soft_bt = _mapping(
        type_contract["exact_soft_label_bt"],
        path="settings_type_contract.exact_soft_label_bt",
    )
    adamw = _mapping(
        optimizer_protocol["adamw"],
        path="reward_model.optimizer_protocol.adamw",
    )
    protocol = AdamWRecoveryProtocol(
        stages=tuple(
            LearningRateStage(
                first_update=int(_mapping(stage, path="schedule stage")["first_update"]),
                last_update=int(_mapping(stage, path="schedule stage")["last_update"]),
                learning_rate=float(_mapping(stage, path="schedule stage")["learning_rate"]),
            )
            for stage in schedule["stages"]  # type: ignore[union-attr]
        ),
        schedule_sha256=str(schedule["schedule_sha256"]),
        mode="recovery",
        legacy_boundary_snapshot_steps=int(
            optimizer_protocol["legacy_constant_lr_boundary_snapshot_steps"]
        ),
        betas=tuple(float(value) for value in adamw["betas"]),  # type: ignore[arg-type]
        eps=float(adamw["eps"]),
        amsgrad=bool(adamw["amsgrad"]),
        maximize=bool(adamw["maximize"]),
        foreach=bool(adamw["foreach"]),
        fused=bool(adamw["fused"]),
        capturable=bool(adamw["capturable"]),
        differentiable=bool(adamw["differentiable"]),
        reward_head_dtype=str(optimizer_protocol["reward_head_dtype"]),
        first_order_audit_dtype=str(optimizer_protocol["first_order_audit_dtype"]),
        microbatch_order=str(optimizer_protocol["microbatch_order"]),
        optimizer_state_reset_at_lr_milestone=bool(
            optimizer_protocol["optimizer_state_reset_at_lr_milestone"]
        ),
        one_optimizer_update_per_step=bool(optimizer_protocol["one_optimizer_update_per_step"]),
        tie_break=str(optimizer_protocol["tie_break"]),
    )
    return Phase2TrainingSettings(
        phase2_config_hash=semantic_sha256,
        source_config_hash=str(materialization["source_config_hash"]),
        stage="pilot",
        formal_eligibility=bool(campaign["formal_eligibility"]),
        seeds=tuple(int(seed) for seed in run["seeds"]),  # type: ignore[arg-type]
        outer_steps=int(reward_model["outer_steps"]),
        learning_rate=float(reward_model["learning_rate"]),
        optimizer=str(reward_model["optimizer"]),
        weight_decay=float(reward_model["weight_decay"]),
        microbatch_size=int(reward_model["microbatch_size"]),
        max_grad_norm=float(reward_model["max_grad_norm"]),
        training_beta=float(objective["training_beta"]),
        relative_damping=float(ridge["relative_coefficient"]),
        pcg_dtype=str(ridge["solver_dtype"]),  # type: ignore[arg-type]
        pcg_max_iterations=int(ridge["pcg_max_iterations"]),
        pcg_tolerance=float(ridge["pcg_tolerance"]),
        pcg_absolute_tolerance=float(objective["pcg_absolute_tolerance"]),
        pcg_residual_recompute_interval=int(objective["pcg_residual_recompute_interval"]),
        require_pcg_convergence=bool(objective["require_pcg_convergence"]),
        num_label_replicates=int(annotations["independent_replicates_per_edge"]),
        annotation_gamma=float(annotations["gamma"]),
        probability_floor=float(objective["probability_floor"]),
        label_rng_namespace=str(annotations["named_rng_namespace"]),
        max_total_annotations=type_contract["max_total_annotations"],  # type: ignore[arg-type]
        low_dimensional_enabled=bool(low_dimensional["enabled"]),
        low_dimensional_selected_dimension=int(low_dimensional["dimension"]),
        low_dimensional_namespace=str(low_dimensional["seed_namespace"]),
        low_dimensional_regularization=str(low_dimensional["regularization"]),
        low_dimensional_relative_eigenvalue_tolerance=float(
            low_dimensional["relative_eigenvalue_tolerance"]
        ),
        low_dimensional_source_layout_id=str(type_contract["low_dimensional_source_layout_id"]),
        exact_soft_label_bt_enabled=bool(exact_soft_bt["enabled"]),
        exact_soft_label_bt_role=str(exact_soft_bt["role"]),
        exact_soft_label_bt_noise_free=bool(exact_soft_bt["noise_free"]),
        exact_soft_label_bt_input=str(exact_soft_bt["input"]),
        exact_soft_label_bt_eligible_for_primary_claim=bool(
            exact_soft_bt["eligible_for_primary_claim"]
        ),
        convergence=FirstOrderConvergenceSpec(
            gradient_ratio_tolerance=float(convergence["relative_gradient_ratio_tolerance"]),
            min_steps=int(convergence["minimum_steps"]),
            max_steps=int(convergence["maximum_steps"]),
            check_interval=int(convergence["check_interval_steps"]),
            consecutive_checks=int(convergence["consecutive_passing_checks"]),
            gradient_norm_denominator_floor=float(convergence["denominator_floor"]),
            fail_closed=bool(convergence["fail_closed"]),
            optimizer_protocol=protocol,
        ),
        identifiability_relative_rank_tolerance=float(identifiability["relative_rank_tolerance"]),
        identifiability_role=str(identifiability["role"]),
        identifiability_require_full_column_rank=bool(identifiability["require_full_column_rank"]),
    )


@dataclass(frozen=True, slots=True)
class _LoadedR3Science:
    settings: Phase2TrainingSettings
    semantic_sha256: str
    file_sha256: str
    normalized: dict[str, object]


def _load_components(source_path: Path) -> _LoadedR3Science:
    raw = _read_stable_regular_bytes(source_path, name="R3 science config")
    file_sha256 = hashlib.sha256(raw).hexdigest()
    parsed = _parse_yaml(raw, source=source_path)
    normalized = _validate_and_normalize(parsed)
    semantic_sha256 = _canonical_sha256(normalized)
    if semantic_sha256 == _R2_RECOVERY_DESIGN_SHA256:
        raise R3ScienceConfigError("R3 semantic identity must differ from the R2 design identity")
    settings = _build_settings(normalized, semantic_sha256=semantic_sha256)
    if settings.phase2_config_hash != semantic_sha256:
        raise RuntimeError("compiled settings lost the R3 semantic identity")
    return _LoadedR3Science(
        settings=settings,
        semantic_sha256=semantic_sha256,
        file_sha256=file_sha256,
        normalized=normalized,
    )


@dataclass(frozen=True, slots=True)
class R3ScienceConfigBundle:
    """Validated R3 science plus hashes bound to one regular source file."""

    settings: Phase2TrainingSettings
    semantic_sha256: str
    file_sha256: str
    source_path: Path
    normalized: dict[str, object]

    def __post_init__(self) -> None:
        if type(self.settings) is not Phase2TrainingSettings:
            raise TypeError("settings must be exactly Phase2TrainingSettings")
        _require_digest(self.semantic_sha256, name="semantic_sha256")
        _require_digest(self.file_sha256, name="file_sha256")
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be pathlib.Path")
        if type(self.normalized) is not dict:
            raise TypeError("normalized must be exactly dict")
        self.validate_integrity()

    def validate_integrity(self) -> None:
        """Re-read the source and reconstruct every binding on every call."""

        current = _load_components(self.source_path)
        if self.file_sha256 != current.file_sha256:
            raise R3ScienceConfigError("R3 science source file bytes changed")
        if self.semantic_sha256 != current.semantic_sha256:
            raise R3ScienceConfigError("R3 science semantic identity changed")
        if not _strict_equal(self.normalized, current.normalized):
            raise R3ScienceConfigError("R3 normalized science document changed")
        if self.settings != current.settings:
            raise R3ScienceConfigError("R3 compiled Phase2TrainingSettings changed")


def load_r3_science_config(path: str | os.PathLike[str]) -> R3ScienceConfigBundle:
    """Load only a path to the standalone R3 schema; mappings/bundles are rejected."""

    if isinstance(path, bool) or not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a filesystem path, not a config object")
    unresolved = Path(os.fspath(path))
    if unresolved.is_symlink():
        raise R3ScienceConfigError(f"R3 science config must not be a symbolic link: {unresolved}")
    source_path = unresolved.absolute()
    loaded = _load_components(source_path)
    return R3ScienceConfigBundle(
        settings=loaded.settings,
        semantic_sha256=loaded.semantic_sha256,
        file_sha256=loaded.file_sha256,
        source_path=source_path,
        normalized=loaded.normalized,
    )


__all__ = [
    "R3_PRIMARY_HEADS",
    "R3_PRIMARY_SEEDS",
    "R3_RECOVERY_SCHEDULE_SHA256",
    "R3_SCIENCE_CONFIG_PATH",
    "R3_SCIENCE_CONFIG_SCHEMA",
    "R3ScienceConfigBundle",
    "R3ScienceConfigError",
    "load_r3_science_config",
]
