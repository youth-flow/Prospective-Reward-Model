"""Head-free scientific contract for the separated Phase-2 R3 Gate-C controls.

The module deliberately stops at the local science/evidence boundary.  It
freezes the three-by-three family/seed matrix, validates compact per-family
gate observations, and emits self-hashed results containing no reward-head
vectors, optimizer state, checkpoints, primary-head references, held-out
measurements, policy outcomes, or calibration beta.

Scheduler admission, checkpoint continuation, terminal capture, aggregation,
and Gate-C authorization belong to the HPC execution layer.  That layer must
call :func:`validate_r3_control_family_result` before accepting any result.
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
from typing import Any, Final, Literal, TypeAlias

import yaml

from .phase2_r3_config import (
    R3_RECOVERY_SCHEDULE_SHA256,
    R3_SCIENCE_CONFIG_PATH,
    R3_SCIENCE_CONFIG_SCHEMA,
    load_r3_science_config,
)

R3_CONTROLS_CONFIG_SCHEMA: Final = "phase2-recovery-r3-controls-config/v1"
R3_CONTROLS_CONFIG_PATH: Final = "configs/phase2_recovery_r3_controls.yaml"
R3_CONTROL_FAMILY_RESULT_SCHEMA: Final = "phase2-recovery-r3-control-family-result/v1"
R3_CONTROL_FIRST_ORDER_GATE_SCHEMA: Final = "phase2-recovery-r3-control-first-order-gate/v1"
R3_GATE_C_CONFIG_SCHEMA: Final = R3_CONTROLS_CONFIG_SCHEMA
R3_GATE_C_FAMILY_RESULT_SCHEMA: Final = R3_CONTROL_FAMILY_RESULT_SCHEMA

R3_GATE_C_SEEDS: Final = (20260801, 20260802, 20260803)
R3_GATE_C_FAMILIES: Final = (
    "exact_margin_prorm_plus",
    "exact_soft_label_bt",
    "low_dimensional_prorm_plus",
)
R3_GATE_C_PROFILE_UPDATES: Final = 100
R3_GATE_C_MIN_UPDATES: Final = 100
R3_GATE_C_MAX_UPDATES: Final = 12760
R3_GATE_C_AUDIT_INTERVAL: Final = 20
R3_GATE_C_CONSECUTIVE_CHECKS: Final = 3

R3ControlFamily: TypeAlias = Literal[
    "exact_margin_prorm_plus",
    "exact_soft_label_bt",
    "low_dimensional_prorm_plus",
]

_HEX_DIGITS = frozenset("0123456789abcdef")
_SOURCE_SCIENCE_SEMANTIC_SHA256 = "782195dc35c3b40639a6aa2316b70b4aa7aacb7012f2ceb4855efecec6e7d5b2"
_SOURCE_SCIENCE_FILE_SHA256 = "2afa8ec36e634c37caf2d1f73e361c6a654afe728a57b4528cf3b58a70d76f96"
_RECOVERY_TIE_BREAK = "exact_zero_initialized_deterministic_adamw_lr_decay_path"
_LABEL_RNG_NAMESPACE = "prorm-common-beta-r4-labels-v1"
_LOW_DIMENSIONAL_NAMESPACE = "prorm-common-beta-low-dimensional-tangent-v1"
_LOW_DIMENSIONAL_LAYOUT = "training-policy-score-flatten-order/v1"
_LOW_DIMENSIONAL_DIMENSION = 256

_EXPECTED_CAMPAIGN = {
    "name": "phase2-recovery-r3-separated-controls",
    "campaign_kind": "phase2_recovery_revision3_separated_controls",
    "execution_revision": 3,
    "execution_scope": "separated_mechanism_controls",
    "evidence_role": "train_only_nonconfirmatory_gate_c",
    "confirmatory": False,
    "formal_eligibility": False,
    "families": list(R3_GATE_C_FAMILIES),
}
_EXPECTED_SCIENCE_BINDING = {
    "source_path": R3_SCIENCE_CONFIG_PATH,
    "source_schema_version": R3_SCIENCE_CONFIG_SCHEMA,
    "source_semantic_sha256": _SOURCE_SCIENCE_SEMANTIC_SHA256,
    "source_file_sha256": _SOURCE_SCIENCE_FILE_SHA256,
    "reuse_scope": "frozen_science_constants_only",
    "primary_head_artifact_reuse": False,
    "primary_optimizer_state_reuse": False,
    "primary_checkpoint_reuse": False,
    "primary_label_artifact_reuse": False,
}
_EXPECTED_RUN = {
    "seeds": list(R3_GATE_C_SEEDS),
    "matrix": "every_family_for_every_seed",
    "partial_seed_or_family_success_allowed": False,
}
_EXPECTED_PROFILING = {
    "fixed_updates_per_family": R3_GATE_C_PROFILE_UPDATES,
    "evidence_role": "throughput_projection_only",
    "reusable_as_formal_control_result": False,
    "head_or_optimizer_state_reusable": False,
}
_EXPECTED_LR_STAGES = [
    {"first_update": 1, "last_update": 5760, "learning_rate": 1.0e-3},
    {"first_update": 5761, "last_update": 6760, "learning_rate": 3.0e-4},
    {"first_update": 6761, "last_update": 8760, "learning_rate": 1.0e-4},
    {"first_update": 8761, "last_update": 10760, "learning_rate": 3.0e-5},
    {"first_update": 10761, "last_update": 12760, "learning_rate": 1.0e-5},
]
_EXPECTED_OPTIMIZER = {
    "dtype": "float32",
    "optimizer": "adamw",
    "learning_rate": 1.0e-3,
    "weight_decay": 0.0,
    "microbatch_size": 64,
    "max_grad_norm": 1.0,
    "initialization": "exact_zero_head_and_fresh_optimizer_state",
    "learning_rate_schedule": {
        "update_indexing": "one_indexed_inclusive",
        "application": "set_learning_rate_immediately_before_optimizer_update",
        "stages": _EXPECTED_LR_STAGES,
        "schedule_sha256": R3_RECOVERY_SCHEDULE_SHA256,
    },
    "state_transition": "preserve_all_adamw_moments_across_learning_rate_boundaries",
    "first_order_audit_dtype": "float64",
    "microbatch_order": "canonical_edge_order_contiguous_ascending_no_shuffle",
    "optimizer_state_reset_at_lr_milestone": False,
    "one_optimizer_update_per_step": True,
    "tie_break": _RECOVERY_TIE_BREAK,
    "validation_or_test_selection": False,
}
_EXPECTED_FIRST_ORDER_GATE = {
    "relative_gradient_ratio_tolerance": 1.0e-3,
    "minimum_steps": R3_GATE_C_MIN_UPDATES,
    "maximum_steps": R3_GATE_C_MAX_UPDATES,
    "check_interval_steps": R3_GATE_C_AUDIT_INTERVAL,
    "consecutive_passing_checks": R3_GATE_C_CONSECUTIVE_CHECKS,
    "gradient_measurement": "post_update_full_data_unclipped",
    "denominator": "exact_zero_initialization_gradient_l2_norm",
    "denominator_floor": 1.0e-30,
    "fail_closed": True,
    "solution_tie_break": _RECOVERY_TIE_BREAK,
    "unique_solution_claim": False,
    "validation_or_test_selection": False,
}
_EXPECTED_PCG = {
    "kind": "pcg",
    "dtype": "float64",
    "relative_tolerance": 1.0e-5,
    "absolute_tolerance": 0.0,
    "max_iterations": 8192,
    "residual_recompute_interval": 20,
    "require_convergence": True,
    "initialization": "cold_start_zero",
}
_EXPECTED_FAMILIES = {
    "exact_margin_prorm_plus": {
        "target": "transformed_oracle_reward_difference",
        "objective": "prorm_plus",
        "fixed_training_objective_scale": 1.0,
        "full_tangent_ridge": {
            "enabled": True,
            "rule": "relative_to_mean_fisher_diagonal",
            "relative_coefficient": 0.001,
        },
        "direct_oracle_identity_required": True,
        "solver": _EXPECTED_PCG,
        "eligible_for_primary_claim": False,
    },
    "exact_soft_label_bt": {
        "target": "sigmoid_of_train_transformed_oracle_margin",
        "objective": "exact_expected_bernoulli_cross_entropy",
        "noise_free": True,
        "bernoulli_sampling_used": False,
        "sampled_label_stream_accessed": False,
        "eligible_for_primary_claim": False,
    },
    "low_dimensional_prorm_plus": {
        "target": "family_local_r4_mean_h_regenerated_from_train_oracle",
        "objective": "prorm_plus",
        "fixed_training_objective_scale": 1.0,
        "annotation_scheme": "geometric_randomized_truncation",
        "annotation_gamma": 0.9,
        "independent_replicates_per_edge": 4,
        "replicate_reduction": "arithmetic_mean",
        "prohibit_clipping": True,
        "label_rng_namespace": _LABEL_RNG_NAMESPACE,
        "primary_label_stream_accessed": False,
        "construction": "seeded_orthonormal_projection",
        "dimension": _LOW_DIMENSIONAL_DIMENSION,
        "projection_seed_namespace": _LOW_DIMENSIONAL_NAMESPACE,
        "source_layout_id": _LOW_DIMENSIONAL_LAYOUT,
        "regularization": "moore_penrose_pseudoinverse",
        "relative_eigenvalue_tolerance": 1.0e-10,
        "ridge_enabled": False,
        "eligible_for_primary_claim": False,
    },
}
_EXPECTED_TOLERANCES = {
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
_EXPECTED_DECISION_GATES = {
    "action": "fail_closed",
    "unit": "per_family_per_seed",
    "exact_margin_prorm_plus": [
        "exact_margin_objective_decrease",
        "exact_margin_first_order_convergence",
        "direct_oracle_moment_identity",
        "all_required_pcg_solves_converged",
    ],
    "exact_soft_label_bt": [
        "exact_soft_label_objective_decrease",
        "exact_soft_label_first_order_convergence",
        "saved_head_objective_binding",
    ],
    "low_dimensional_prorm_plus": [
        "low_dimensional_objective_decrease",
        "low_dimensional_first_order_convergence",
        "low_dimensional_exact_rank",
        "low_dimensional_orthonormality",
        "low_dimensional_pseudoinverse_residual",
        "low_dimensional_scatter_identity",
        "low_dimensional_score_identity",
    ],
}
_EXPECTED_EXECUTION_BOUNDARY = {
    "information_boundary": "train_only",
    "primary_head_access_allowed": False,
    "primary_optimizer_state_access_allowed": False,
    "primary_checkpoint_access_allowed": False,
    "heldout_or_validation_access_allowed": False,
    "policy_optimization_allowed": False,
    "policy_rollout_allowed": False,
    "downstream_utility_access_allowed": False,
    "calibration_beta_access_allowed": False,
    "final_oracle_access_allowed": False,
    "family_jobs_must_be_independent": True,
}
_EXPECTED_RESULT_CONTRACT = {
    "vectors_retained": False,
    "head_weights_retained": False,
    "optimizer_state_retained": False,
    "checkpoint_state_retained": False,
    "raw_oracle_rewards_retained": False,
    "raw_labels_retained": False,
    "hashes_and_gate_evidence_only": True,
    "self_hash_required": True,
}
_EXPECTED_ROOT = {
    "schema_version": R3_CONTROLS_CONFIG_SCHEMA,
    "campaign": _EXPECTED_CAMPAIGN,
    "science_binding": _EXPECTED_SCIENCE_BINDING,
    "run": _EXPECTED_RUN,
    "profiling": _EXPECTED_PROFILING,
    "optimizer": _EXPECTED_OPTIMIZER,
    "first_order_gate": _EXPECTED_FIRST_ORDER_GATE,
    "families": _EXPECTED_FAMILIES,
    "numeric_gate_tolerances": _EXPECTED_TOLERANCES,
    "decision_gates": _EXPECTED_DECISION_GATES,
    "execution_boundary": _EXPECTED_EXECUTION_BOUNDARY,
    "result_contract": _EXPECTED_RESULT_CONTRACT,
}

_COMMON_RESULT_KEYS = {
    "schema_version",
    "family",
    "seed",
    "controls_config_semantic_sha256",
    "controls_config_file_sha256",
    "source_science_semantic_sha256",
    "input_training_sha256",
    "train_oracle_rewards_sha256",
    "input_dimensions",
    "information_boundary",
    "completion",
    "family_evidence",
    "result_sha256",
}
_RESULT_INFORMATION_BOUNDARY = {
    "information_boundary": "train_only",
    "primary_head_accessed": False,
    "primary_optimizer_state_accessed": False,
    "primary_checkpoint_accessed": False,
    "heldout_or_validation_accessed": False,
    "policy_optimization_executed": False,
    "policy_rollout_executed": False,
    "downstream_utility_accessed": False,
    "calibration_beta_accessed": False,
    "final_oracle_accessed": False,
}


class R3ControlsConfigError(ValueError):
    """The frozen Gate-C science contract is absent, changed, or malformed."""


class R3ControlResultError(ValueError):
    """A compact Gate-C family result is incomplete, changed, or malformed."""


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


def _strict_json_copy(
    value: object,
    *,
    path: str,
    error_type: type[ValueError],
) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise error_type(f"{path} must not contain a non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise error_type(f"{path} must contain only string keys")
        return {
            key: _strict_json_copy(value[key], path=f"{path}.{key}", error_type=error_type)
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _strict_json_copy(
                item,
                path=f"{path}[{index}]",
                error_type=error_type,
            )
            for index, item in enumerate(value)
        ]
    raise error_type(f"{path} contains unsupported value type {type(value).__name__}")


def _canonical_sha256(value: object, *, error_type: type[ValueError]) -> str:
    normalized = _strict_json_copy(value, path="hash payload", error_type=error_type)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(
    value: object,
    *,
    path: str,
    keys: set[str] | None = None,
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise R3ControlResultError(f"{path} must be an exact mapping with string keys")
    if keys is not None and set(value) != keys:
        missing = sorted(keys - set(value))
        unknown = sorted(set(value) - keys)
        raise R3ControlResultError(
            f"{path} has invalid fields: missing={missing!r}, unknown={unknown!r}"
        )
    return value


def _digest(value: object, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise R3ControlResultError(f"{path} must be a lowercase SHA256 digest")
    return value


def _integer(
    value: object,
    *,
    path: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise R3ControlResultError(f"{path} must be an integer {bounds}")
    return value


def _finite(value: object, *, path: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R3ControlResultError(f"{path} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise R3ControlResultError(f"{path} must be {qualifier}")
    return result


def _expect(value: object, expected: object, *, path: str) -> None:
    if not _strict_equal(value, expected):
        raise R3ControlResultError(f"{path} differs from the frozen Gate-C contract")


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
                raise R3ControlsConfigError(
                    f"{source} contains an unhashable YAML mapping key"
                ) from error
            if duplicate:
                raise R3ControlsConfigError(f"{source} contains duplicate key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        unique_mapping,
    )
    try:
        return yaml.load(raw.decode("utf-8"), Loader=UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise R3ControlsConfigError(f"{source} is not UTF-8 YAML") from error
    except yaml.YAMLError as error:
        raise R3ControlsConfigError(f"failed to parse Gate-C config {source}") from error


def _read_stable_regular_bytes(path: Path, *, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise R3ControlsConfigError(f"cannot stat {name} {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise R3ControlsConfigError(f"{name} must be a non-symlink regular file: {path}")
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise R3ControlsConfigError(f"cannot read {name} {path}: {error}") from error
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
        raise R3ControlsConfigError(f"{name} changed while it was being read: {path}")
    return raw


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_config_components(source_path: Path) -> tuple[dict[str, object], str, str]:
    raw = _read_stable_regular_bytes(source_path, name="Gate-C controls config")
    parsed = _parse_yaml(raw, source=source_path)
    normalized = _strict_json_copy(
        parsed,
        path="Gate-C controls config",
        error_type=R3ControlsConfigError,
    )
    if type(normalized) is not dict:
        raise R3ControlsConfigError("Gate-C controls config must be a mapping")
    if not _strict_equal(normalized, _EXPECTED_ROOT):
        expected_keys = set(_EXPECTED_ROOT)
        observed_keys = set(normalized)
        if expected_keys != observed_keys:
            raise R3ControlsConfigError(
                "Gate-C controls config is not closed: "
                f"missing={sorted(expected_keys - observed_keys)!r}, "
                f"unknown={sorted(observed_keys - expected_keys)!r}"
            )
        raise R3ControlsConfigError("Gate-C controls config differs from its frozen contract")

    science_path = _repository_root().joinpath(*R3_SCIENCE_CONFIG_PATH.split("/"))
    science = load_r3_science_config(science_path)
    if (
        science.semantic_sha256 != _SOURCE_SCIENCE_SEMANTIC_SHA256
        or science.file_sha256 != _SOURCE_SCIENCE_FILE_SHA256
    ):
        raise R3ControlsConfigError("Gate-C source R3 science binding changed")

    return (
        normalized,
        hashlib.sha256(raw).hexdigest(),
        _canonical_sha256(normalized, error_type=R3ControlsConfigError),
    )


@dataclass(frozen=True, slots=True)
class R3ControlsConfigBundle:
    """Validated frozen Gate-C config and its stable file/semantic identities."""

    source_path: Path
    file_sha256: str
    semantic_sha256: str
    normalized: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be pathlib.Path")
        for name in ("file_sha256", "semantic_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in _HEX_DIGITS for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest")
        if type(self.normalized) is not dict:
            raise TypeError("normalized must be exactly dict")
        self.validate_integrity()

    @property
    def seeds(self) -> tuple[int, int, int]:
        return R3_GATE_C_SEEDS

    @property
    def families(self) -> tuple[str, str, str]:
        return R3_GATE_C_FAMILIES

    @property
    def profile_updates(self) -> int:
        return R3_GATE_C_PROFILE_UPDATES

    @property
    def minimum_updates(self) -> int:
        return R3_GATE_C_MIN_UPDATES

    @property
    def maximum_updates(self) -> int:
        return R3_GATE_C_MAX_UPDATES

    @property
    def audit_interval_updates(self) -> int:
        return R3_GATE_C_AUDIT_INTERVAL

    def validate_integrity(self) -> None:
        normalized, file_sha256, semantic_sha256 = _load_config_components(self.source_path)
        if self.file_sha256 != file_sha256:
            raise R3ControlsConfigError("Gate-C controls config bytes changed")
        if self.semantic_sha256 != semantic_sha256:
            raise R3ControlsConfigError("Gate-C controls semantic identity changed")
        if not _strict_equal(self.normalized, normalized):
            raise R3ControlsConfigError("Gate-C normalized config changed")


def load_r3_controls_config(path: str | os.PathLike[str]) -> R3ControlsConfigBundle:
    """Load the frozen Gate-C config from one non-symlink regular file."""

    if isinstance(path, bool) or not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a filesystem path")
    unresolved = Path(os.fspath(path))
    if unresolved.is_symlink():
        raise R3ControlsConfigError(f"Gate-C config must not be a symbolic link: {unresolved}")
    source_path = unresolved.absolute()
    normalized, file_sha256, semantic_sha256 = _load_config_components(source_path)
    return R3ControlsConfigBundle(
        source_path=source_path,
        file_sha256=file_sha256,
        semantic_sha256=semantic_sha256,
        normalized=normalized,
    )


def _require_bundle(value: object) -> R3ControlsConfigBundle:
    if type(value) is not R3ControlsConfigBundle:
        raise TypeError("config must be an exact R3ControlsConfigBundle")
    value.validate_integrity()
    return value


def _family(value: object) -> R3ControlFamily:
    if value not in R3_GATE_C_FAMILIES:
        raise R3ControlResultError(f"family must be one of {R3_GATE_C_FAMILIES!r}")
    return value  # type: ignore[return-value]


def _dimensions(value: object, *, family: R3ControlFamily) -> dict[str, int]:
    dimensions = _mapping(
        value,
        path="input_dimensions",
        keys={
            "num_train_prompts",
            "num_candidates",
            "policy_dimension",
            "reward_dimension",
        },
    )
    result = {
        "num_train_prompts": _integer(
            dimensions["num_train_prompts"],
            path="input_dimensions.num_train_prompts",
            minimum=1,
        ),
        "num_candidates": _integer(
            dimensions["num_candidates"],
            path="input_dimensions.num_candidates",
            minimum=2,
        ),
        "policy_dimension": _integer(
            dimensions["policy_dimension"],
            path="input_dimensions.policy_dimension",
            minimum=1,
        ),
        "reward_dimension": _integer(
            dimensions["reward_dimension"],
            path="input_dimensions.reward_dimension",
            minimum=1,
        ),
    }
    if family == "low_dimensional_prorm_plus":
        if result["policy_dimension"] <= _LOW_DIMENSIONAL_DIMENSION:
            raise R3ControlResultError(
                "low-dimensional family requires policy_dimension > selected dimension"
            )
        if result["num_train_prompts"] * result["num_candidates"] <= _LOW_DIMENSIONAL_DIMENSION:
            raise R3ControlResultError(
                "low-dimensional family requires train Fisher nodes > selected dimension"
            )
    return result


def _first_order_gate(value: object, *, expected_objective: str) -> dict[str, object]:
    keys = {
        "schema_version",
        "objective",
        "learning_rate_schedule_sha256",
        "initial_full_data_unclipped_gradient_l2_norm",
        "gradient_norm_denominator",
        "final_full_data_unclipped_gradient_l2_norm",
        "gradient_ratio_to_zero_initialization",
        "selected_step",
        "consecutive_passing_checks",
        "sustained_checks",
        "full_data_post_update_unclipped",
        "fresh_zero_initialized",
        "fresh_post_restore_audit",
        "test_or_validation_data_accessed",
        "passed",
    }
    gate = _mapping(value, path="first_order_gate", keys=keys)
    _expect(
        gate["schema_version"],
        R3_CONTROL_FIRST_ORDER_GATE_SCHEMA,
        path="first_order_gate.schema_version",
    )
    _expect(gate["objective"], expected_objective, path="first_order_gate.objective")
    _expect(
        gate["learning_rate_schedule_sha256"],
        R3_RECOVERY_SCHEDULE_SHA256,
        path="first_order_gate.learning_rate_schedule_sha256",
    )
    initial_gradient = _finite(
        gate["initial_full_data_unclipped_gradient_l2_norm"],
        path="first_order_gate.initial_full_data_unclipped_gradient_l2_norm",
        nonnegative=True,
    )
    denominator = _finite(
        gate["gradient_norm_denominator"],
        path="first_order_gate.gradient_norm_denominator",
        nonnegative=True,
    )
    expected_denominator = max(initial_gradient, _EXPECTED_FIRST_ORDER_GATE["denominator_floor"])
    if denominator != expected_denominator:
        raise R3ControlResultError("first_order_gate denominator arithmetic failed")
    final_gradient = _finite(
        gate["final_full_data_unclipped_gradient_l2_norm"],
        path="first_order_gate.final_full_data_unclipped_gradient_l2_norm",
        nonnegative=True,
    )
    final_ratio = _finite(
        gate["gradient_ratio_to_zero_initialization"],
        path="first_order_gate.gradient_ratio_to_zero_initialization",
        nonnegative=True,
    )
    if not math.isclose(
        final_ratio,
        final_gradient / denominator,
        rel_tol=1.0e-10,
        abs_tol=1.0e-14,
    ):
        raise R3ControlResultError("first_order_gate final gradient-ratio arithmetic failed")
    if final_ratio > _EXPECTED_TOLERANCES["outer_relative_gradient_ratio"]:
        raise R3ControlResultError("first_order_gate final gradient ratio exceeds the frozen gate")

    selected_step = _integer(
        gate["selected_step"],
        path="first_order_gate.selected_step",
        minimum=R3_GATE_C_MIN_UPDATES,
        maximum=R3_GATE_C_MAX_UPDATES,
    )
    if selected_step % R3_GATE_C_AUDIT_INTERVAL != 0:
        raise R3ControlResultError("first_order_gate selected_step is not a scheduled audit")
    _expect(
        gate["consecutive_passing_checks"],
        R3_GATE_C_CONSECUTIVE_CHECKS,
        path="first_order_gate.consecutive_passing_checks",
    )
    checks = gate["sustained_checks"]
    if type(checks) is not list or len(checks) != R3_GATE_C_CONSECUTIVE_CHECKS:
        raise R3ControlResultError("first_order_gate must retain exactly three sustained checks")
    expected_steps = [
        selected_step - (R3_GATE_C_CONSECUTIVE_CHECKS - 1 - index) * R3_GATE_C_AUDIT_INTERVAL
        for index in range(R3_GATE_C_CONSECUTIVE_CHECKS)
    ]
    if expected_steps[0] < R3_GATE_C_MIN_UPDATES:
        raise R3ControlResultError("first_order_gate selected before three eligible checks")
    normalized_checks: list[dict[str, object]] = []
    for index, (item, expected_step) in enumerate(zip(checks, expected_steps, strict=True)):
        check = _mapping(
            item,
            path=f"first_order_gate.sustained_checks[{index}]",
            keys={
                "step",
                "gradient_l2_norm",
                "gradient_ratio_to_zero_initialization",
                "threshold_passed",
            },
        )
        _expect(
            check["step"],
            expected_step,
            path=f"first_order_gate.sustained_checks[{index}].step",
        )
        gradient = _finite(
            check["gradient_l2_norm"],
            path=f"first_order_gate.sustained_checks[{index}].gradient_l2_norm",
            nonnegative=True,
        )
        ratio = _finite(
            check["gradient_ratio_to_zero_initialization"],
            path=(
                f"first_order_gate.sustained_checks[{index}].gradient_ratio_to_zero_initialization"
            ),
            nonnegative=True,
        )
        if not math.isclose(
            ratio,
            gradient / denominator,
            rel_tol=1.0e-10,
            abs_tol=1.0e-14,
        ):
            raise R3ControlResultError(
                f"first_order_gate.sustained_checks[{index}] ratio arithmetic failed"
            )
        if ratio > _EXPECTED_TOLERANCES["outer_relative_gradient_ratio"]:
            raise R3ControlResultError(
                f"first_order_gate.sustained_checks[{index}] exceeds the frozen threshold"
            )
        _expect(
            check["threshold_passed"],
            True,
            path=f"first_order_gate.sustained_checks[{index}].threshold_passed",
        )
        normalized_checks.append(dict(check))
    for field, expected in {
        "full_data_post_update_unclipped": True,
        "fresh_zero_initialized": True,
        "fresh_post_restore_audit": True,
        "test_or_validation_data_accessed": False,
        "passed": True,
    }.items():
        _expect(gate[field], expected, path=f"first_order_gate.{field}")
    return {**dict(gate), "sustained_checks": normalized_checks}


def _pcg(value: object, *, path: str) -> dict[str, object]:
    pcg = _mapping(
        value,
        path=path,
        keys={
            "schema_version",
            "iterations",
            "residual_norm",
            "relative_residual",
            "converged",
            "cold_start",
            "warm_start_used",
        },
    )
    _expect(pcg["schema_version"], "phase2-r3-control-pcg/v1", path=f"{path}.schema_version")
    _integer(
        pcg["iterations"],
        path=f"{path}.iterations",
        minimum=0,
        maximum=int(_EXPECTED_PCG["max_iterations"]),
    )
    _finite(pcg["residual_norm"], path=f"{path}.residual_norm", nonnegative=True)
    relative_residual = _finite(
        pcg["relative_residual"],
        path=f"{path}.relative_residual",
        nonnegative=True,
    )
    if relative_residual > _EXPECTED_PCG["relative_tolerance"]:
        raise R3ControlResultError(f"{path} exceeds the frozen relative-residual threshold")
    for field, expected in {
        "converged": True,
        "cold_start": True,
        "warm_start_used": False,
    }.items():
        _expect(pcg[field], expected, path=f"{path}.{field}")
    return dict(pcg)


def _objective_and_head(
    value: object,
    *,
    path: str,
    method: str,
    objective: str,
) -> tuple[dict[str, object], dict[str, object]]:
    head = _mapping(
        value,
        path=path,
        keys={
            "schema_version",
            "method",
            "objective",
            "initial_head_sha256",
            "head_sha256",
            "initial_objective",
            "final_objective",
            "cold_full_data_audit_objective",
            "cold_full_data_audit_gradient_l2_norm",
            "objective_decrease_passed",
            "objective_binding_passed",
            "first_order_gate",
            "fresh_zero_initialized",
            "raw_head_weight_retained",
        },
    )
    _expect(head["schema_version"], "phase2-r3-control-head-audit/v1", path=f"{path}.schema")
    _expect(head["method"], method, path=f"{path}.method")
    _expect(head["objective"], objective, path=f"{path}.objective")
    _digest(head["initial_head_sha256"], path=f"{path}.initial_head_sha256")
    _digest(head["head_sha256"], path=f"{path}.head_sha256")
    initial_objective = _finite(
        head["initial_objective"],
        path=f"{path}.initial_objective",
        nonnegative=True,
    )
    final_objective = _finite(
        head["final_objective"],
        path=f"{path}.final_objective",
        nonnegative=True,
    )
    audit_objective = _finite(
        head["cold_full_data_audit_objective"],
        path=f"{path}.cold_full_data_audit_objective",
        nonnegative=True,
    )
    audit_gradient = _finite(
        head["cold_full_data_audit_gradient_l2_norm"],
        path=f"{path}.cold_full_data_audit_gradient_l2_norm",
        nonnegative=True,
    )
    if not final_objective < initial_objective:
        raise R3ControlResultError(f"{path} objective did not decrease")
    if not math.isclose(
        audit_objective,
        final_objective,
        rel_tol=_EXPECTED_TOLERANCES["objective_binding_relative_error"],
        abs_tol=_EXPECTED_TOLERANCES["objective_binding_absolute_error"],
    ):
        raise R3ControlResultError(f"{path} saved-head objective binding failed")
    gate = _first_order_gate(head["first_order_gate"], expected_objective=objective)
    final_gradient = _finite(
        gate["final_full_data_unclipped_gradient_l2_norm"],
        path=f"{path}.first_order_gate.final_gradient",
        nonnegative=True,
    )
    if not math.isclose(
        audit_gradient,
        final_gradient,
        rel_tol=_EXPECTED_TOLERANCES["objective_binding_relative_error"],
        abs_tol=_EXPECTED_TOLERANCES["objective_binding_absolute_error"],
    ):
        raise R3ControlResultError(f"{path} cold gradient does not bind its first-order gate")
    for field, expected in {
        "objective_decrease_passed": True,
        "objective_binding_passed": True,
        "fresh_zero_initialized": True,
        "raw_head_weight_retained": False,
    }.items():
        _expect(head[field], expected, path=f"{path}.{field}")
    return ({**dict(head), "first_order_gate": gate}, gate)


def _target_common(
    value: object,
    *,
    path: str,
    oracle_sha256: str,
    dimensions: Mapping[str, int],
    extra_keys: set[str],
) -> dict[str, object]:
    keys = {
        "schema_version",
        "split",
        "orientation",
        "source_node_rewards_sha256",
        "canonical_margin_sha256",
        "reward_feature_difference_sha256",
        "num_train_prompts",
        "num_candidates",
        "reward_dimension",
        "raw_node_rewards_retained",
    } | extra_keys
    target = _mapping(value, path=path, keys=keys)
    _expect(target["split"], "train", path=f"{path}.split")
    _expect(
        target["orientation"],
        "candidate_0_minus_candidate_1",
        path=f"{path}.orientation",
    )
    _expect(
        _digest(target["source_node_rewards_sha256"], path=f"{path}.source_node_rewards_sha256"),
        oracle_sha256,
        path=f"{path}.source_node_rewards_sha256",
    )
    _digest(target["canonical_margin_sha256"], path=f"{path}.canonical_margin_sha256")
    _digest(
        target["reward_feature_difference_sha256"],
        path=f"{path}.reward_feature_difference_sha256",
    )
    for field in ("num_train_prompts", "num_candidates", "reward_dimension"):
        _expect(target[field], dimensions[field], path=f"{path}.{field}")
    _expect(target["raw_node_rewards_retained"], False, path=f"{path}.raw_node_rewards_retained")
    return dict(target)


def _exact_margin_evidence(
    value: object,
    *,
    oracle_sha256: str,
    dimensions: Mapping[str, int],
) -> tuple[dict[str, object], int]:
    evidence = _mapping(
        value,
        path="family_evidence",
        keys={
            "schema_version",
            "target_audit",
            "head_audit",
            "pcg_audits",
            "direct_identity",
            "gates",
        },
    )
    _expect(
        evidence["schema_version"],
        "phase2-r3-exact-margin-prorm-plus-evidence/v1",
        path="family_evidence.schema_version",
    )
    target = _target_common(
        evidence["target_audit"],
        path="family_evidence.target_audit",
        oracle_sha256=oracle_sha256,
        dimensions=dimensions,
        extra_keys={"target", "sampled_label_stream_accessed"},
    )
    _expect(
        target["schema_version"],
        "phase2-r3-exact-margin-target/v1",
        path="family_evidence.target_audit.schema_version",
    )
    _expect(
        target["target"],
        "transformed_oracle_reward_difference",
        path="family_evidence.target_audit.target",
    )
    _expect(
        target["sampled_label_stream_accessed"],
        False,
        path="family_evidence.target_audit.sampled_label_stream_accessed",
    )
    head, first_order = _objective_and_head(
        evidence["head_audit"],
        path="family_evidence.head_audit",
        method="prorm_plus",
        objective="exact_margin_prorm_plus",
    )
    pcg_audits = _mapping(
        evidence["pcg_audits"],
        path="family_evidence.pcg_audits",
        keys={"selected_head_final_inner", "cold_saved_head_audit", "trained_direction"},
    )
    normalized_pcg_audits = {
        name: _pcg(pcg_audits[name], path=f"family_evidence.pcg_audits.{name}")
        for name in (
            "selected_head_final_inner",
            "cold_saved_head_audit",
            "trained_direction",
        )
    }

    direct = _mapping(
        evidence["direct_identity"],
        path="family_evidence.direct_identity",
        keys={
            "schema_version",
            "interpretation",
            "source_node_rewards_sha256",
            "canonical_margin_sha256",
            "canonical_pair_moment_sha256",
            "complete_pair_u_stat_moment_sha256",
            "all_node_covariance_moment_sha256",
            "complete_pair_identity_absolute_error",
            "complete_pair_identity_relative_error",
            "complete_pair_identity_is_algebraic",
            "reward_head_bypassed",
            "optimizer_bypassed",
            "trained_exact_margin_head_required_to_match",
            "native_oracle_direction_sha256",
            "native_oracle_direction_pcg",
            "raw_node_rewards_retained",
        },
    )
    _expect(
        direct["schema_version"],
        "direct-oracle-exact-moment-identity/v1",
        path="family_evidence.direct_identity.schema_version",
    )
    _expect(
        direct["interpretation"],
        "algebraic_identity_bypasses_reward_class_and_optimizer",
        path="family_evidence.direct_identity.interpretation",
    )
    _expect(
        _digest(
            direct["source_node_rewards_sha256"],
            path="family_evidence.direct_identity.source_node_rewards_sha256",
        ),
        oracle_sha256,
        path="family_evidence.direct_identity.source_node_rewards_sha256",
    )
    _expect(
        direct["canonical_margin_sha256"],
        target["canonical_margin_sha256"],
        path="family_evidence.direct_identity.canonical_margin_sha256",
    )
    for field in (
        "canonical_pair_moment_sha256",
        "complete_pair_u_stat_moment_sha256",
        "all_node_covariance_moment_sha256",
        "native_oracle_direction_sha256",
    ):
        _digest(direct[field], path=f"family_evidence.direct_identity.{field}")
    absolute_error = _finite(
        direct["complete_pair_identity_absolute_error"],
        path="family_evidence.direct_identity.complete_pair_identity_absolute_error",
        nonnegative=True,
    )
    relative_error = _finite(
        direct["complete_pair_identity_relative_error"],
        path="family_evidence.direct_identity.complete_pair_identity_relative_error",
        nonnegative=True,
    )
    if (
        absolute_error > _EXPECTED_TOLERANCES["direct_identity_absolute_error"]
        or relative_error > _EXPECTED_TOLERANCES["direct_identity_relative_error"]
    ):
        raise R3ControlResultError("family_evidence direct-oracle moment identity failed")
    for field, expected in {
        "complete_pair_identity_is_algebraic": True,
        "reward_head_bypassed": True,
        "optimizer_bypassed": True,
        "trained_exact_margin_head_required_to_match": False,
        "raw_node_rewards_retained": False,
    }.items():
        _expect(direct[field], expected, path=f"family_evidence.direct_identity.{field}")
    direct_pcg = _pcg(
        direct["native_oracle_direction_pcg"],
        path="family_evidence.direct_identity.native_oracle_direction_pcg",
    )

    gates = _mapping(
        evidence["gates"],
        path="family_evidence.gates",
        keys={
            "exact_margin_objective_decrease",
            "exact_margin_first_order_convergence",
            "direct_oracle_moment_identity",
            "all_required_pcg_solves_converged",
        },
    )
    for field in gates:
        _expect(gates[field], True, path=f"family_evidence.gates.{field}")
    normalized = {
        **dict(evidence),
        "target_audit": target,
        "head_audit": head,
        "pcg_audits": normalized_pcg_audits,
        "direct_identity": {**dict(direct), "native_oracle_direction_pcg": direct_pcg},
        "gates": dict(gates),
    }
    return normalized, int(first_order["selected_step"])


def _exact_soft_evidence(
    value: object,
    *,
    oracle_sha256: str,
    dimensions: Mapping[str, int],
) -> tuple[dict[str, object], int]:
    evidence = _mapping(
        value,
        path="family_evidence",
        keys={"schema_version", "target_audit", "head_audit", "gates"},
    )
    _expect(
        evidence["schema_version"],
        "phase2-r3-exact-soft-label-bt-evidence/v1",
        path="family_evidence.schema_version",
    )
    target = _target_common(
        evidence["target_audit"],
        path="family_evidence.target_audit",
        oracle_sha256=oracle_sha256,
        dimensions=dimensions,
        extra_keys={
            "target",
            "target_probability_sha256",
            "noise_free",
            "bernoulli_sampling_used",
            "sampled_label_stream_accessed",
            "raw_target_probabilities_retained",
            "raw_oracle_margins_retained",
        },
    )
    _expect(
        target["schema_version"],
        "phase2-r3-exact-soft-label-bt-target/v1",
        path="family_evidence.target_audit.schema_version",
    )
    _expect(
        target["target"],
        "sigmoid_of_train_transformed_oracle_margin",
        path="family_evidence.target_audit.target",
    )
    _digest(
        target["target_probability_sha256"],
        path="family_evidence.target_audit.target_probability_sha256",
    )
    for field, expected in {
        "noise_free": True,
        "bernoulli_sampling_used": False,
        "sampled_label_stream_accessed": False,
        "raw_target_probabilities_retained": False,
        "raw_oracle_margins_retained": False,
    }.items():
        _expect(target[field], expected, path=f"family_evidence.target_audit.{field}")
    head, first_order = _objective_and_head(
        evidence["head_audit"],
        path="family_evidence.head_audit",
        method="bt_mle",
        objective="exact_soft_label_bt_cross_entropy",
    )
    gates = _mapping(
        evidence["gates"],
        path="family_evidence.gates",
        keys={
            "exact_soft_label_objective_decrease",
            "exact_soft_label_first_order_convergence",
            "saved_head_objective_binding",
        },
    )
    for field in gates:
        _expect(gates[field], True, path=f"family_evidence.gates.{field}")
    return (
        {
            **dict(evidence),
            "target_audit": target,
            "head_audit": head,
            "gates": dict(gates),
        },
        int(first_order["selected_step"]),
    )


def _low_dimensional_evidence(
    value: object,
    *,
    seed: int,
    oracle_sha256: str,
    dimensions: Mapping[str, int],
) -> tuple[dict[str, object], int]:
    evidence = _mapping(
        value,
        path="family_evidence",
        keys={
            "schema_version",
            "target_audit",
            "projection",
            "geometry",
            "head_audit",
            "scatter_identity",
            "score_identity",
            "gates",
        },
    )
    _expect(
        evidence["schema_version"],
        "phase2-r3-low-dimensional-prorm-plus-evidence/v1",
        path="family_evidence.schema_version",
    )
    target = _target_common(
        evidence["target_audit"],
        path="family_evidence.target_audit",
        oracle_sha256=oracle_sha256,
        dimensions=dimensions,
        extra_keys={
            "target",
            "family_local_label_stream_sha256",
            "annotation_scheme",
            "annotation_gamma",
            "independent_replicates_per_edge",
            "replicate_reduction",
            "label_rng_namespace",
            "primary_label_stream_accessed",
            "raw_labels_retained",
        },
    )
    _expect(
        target["schema_version"],
        "phase2-r3-low-dimensional-r4-target/v1",
        path="family_evidence.target_audit.schema_version",
    )
    expected_target = _EXPECTED_FAMILIES["low_dimensional_prorm_plus"]
    for field, expected in {
        "target": expected_target["target"],
        "annotation_scheme": expected_target["annotation_scheme"],
        "annotation_gamma": expected_target["annotation_gamma"],
        "independent_replicates_per_edge": expected_target["independent_replicates_per_edge"],
        "replicate_reduction": expected_target["replicate_reduction"],
        "label_rng_namespace": expected_target["label_rng_namespace"],
        "primary_label_stream_accessed": False,
        "raw_labels_retained": False,
    }.items():
        _expect(target[field], expected, path=f"family_evidence.target_audit.{field}")
    _digest(
        target["family_local_label_stream_sha256"],
        path="family_evidence.target_audit.family_local_label_stream_sha256",
    )

    projection = _mapping(
        evidence["projection"],
        path="family_evidence.projection",
        keys={
            "schema_version",
            "algorithm",
            "namespace",
            "source_layout_id",
            "declared_seed",
            "source_dimension",
            "selected_dimension",
            "num_fisher_nodes",
            "projection_sha256",
            "projection_dtype",
            "orthonormality_max_absolute_error",
            "orthonormality_absolute_tolerance",
            "orthonormality_passed",
        },
    )
    for field, expected in {
        "schema_version": "seeded-orthonormal-tangent/v1",
        "algorithm": "gaussian_qr_sign_canonical_v1",
        "namespace": _LOW_DIMENSIONAL_NAMESPACE,
        "source_layout_id": _LOW_DIMENSIONAL_LAYOUT,
        "declared_seed": seed,
        "source_dimension": dimensions["policy_dimension"],
        "selected_dimension": _LOW_DIMENSIONAL_DIMENSION,
        "num_fisher_nodes": dimensions["num_train_prompts"] * dimensions["num_candidates"],
        "projection_dtype": "torch.float64",
        "orthonormality_absolute_tolerance": _EXPECTED_TOLERANCES[
            "low_dimensional_orthonormality_max_absolute_error"
        ],
        "orthonormality_passed": True,
    }.items():
        _expect(projection[field], expected, path=f"family_evidence.projection.{field}")
    _digest(projection["projection_sha256"], path="family_evidence.projection.projection_sha256")
    orthonormality_error = _finite(
        projection["orthonormality_max_absolute_error"],
        path="family_evidence.projection.orthonormality_max_absolute_error",
        nonnegative=True,
    )
    if (
        orthonormality_error
        > _EXPECTED_TOLERANCES["low_dimensional_orthonormality_max_absolute_error"]
    ):
        raise R3ControlResultError("low-dimensional orthonormality gate failed")

    geometry = _mapping(
        evidence["geometry"],
        path="family_evidence.geometry",
        keys={
            "schema_version",
            "regularization",
            "ridge_enabled",
            "solver",
            "solver_dtype",
            "selected_dimension",
            "numerical_rank",
            "relative_eigenvalue_tolerance",
            "fisher_sha256",
            "pseudoinverse_sha256",
            "pseudoinverse_solve_relative_residual",
            "pseudoinverse_relative_residual_tolerance",
            "exact_rank_passed",
            "pseudoinverse_residual_passed",
        },
    )
    for field, expected in {
        "schema_version": "phase2-r3-low-dimensional-pseudoinverse-geometry/v1",
        "regularization": "moore_penrose_pseudoinverse",
        "ridge_enabled": False,
        "solver": "torch.linalg.eigh_truncated_moore_penrose",
        "solver_dtype": "float64",
        "selected_dimension": _LOW_DIMENSIONAL_DIMENSION,
        "numerical_rank": _LOW_DIMENSIONAL_DIMENSION,
        "relative_eigenvalue_tolerance": 1.0e-10,
        "pseudoinverse_relative_residual_tolerance": _EXPECTED_TOLERANCES[
            "low_dimensional_pseudoinverse_relative_residual"
        ],
        "exact_rank_passed": True,
        "pseudoinverse_residual_passed": True,
    }.items():
        _expect(geometry[field], expected, path=f"family_evidence.geometry.{field}")
    _digest(geometry["fisher_sha256"], path="family_evidence.geometry.fisher_sha256")
    _digest(
        geometry["pseudoinverse_sha256"],
        path="family_evidence.geometry.pseudoinverse_sha256",
    )
    pseudoinverse_residual = _finite(
        geometry["pseudoinverse_solve_relative_residual"],
        path="family_evidence.geometry.pseudoinverse_solve_relative_residual",
        nonnegative=True,
    )
    if (
        pseudoinverse_residual
        > _EXPECTED_TOLERANCES["low_dimensional_pseudoinverse_relative_residual"]
    ):
        raise R3ControlResultError("low-dimensional pseudoinverse residual gate failed")

    head, first_order = _objective_and_head(
        evidence["head_audit"],
        path="family_evidence.head_audit",
        method="prorm_plus",
        objective="low_dimensional_prorm_plus",
    )
    selected_direction_sha = _digest(
        _mapping(
            evidence["scatter_identity"],
            path="family_evidence.scatter_identity",
        ).get("selected_direction_sha256"),
        path="family_evidence.scatter_identity.selected_direction_sha256",
    )

    scatter = _mapping(
        evidence["scatter_identity"],
        path="family_evidence.scatter_identity",
        keys={
            "schema_version",
            "formula",
            "selected_direction_sha256",
            "scattered_full_direction_sha256",
            "reference_scattered_full_direction_sha256",
            "max_absolute_error",
            "l2_error",
            "absolute_tolerance",
            "passed",
        },
    )
    for field, expected in {
        "schema_version": "phase2-r3-low-dimensional-scatter-identity/v1",
        "formula": "u_full = P @ u_low",
        "absolute_tolerance": _EXPECTED_TOLERANCES["low_dimensional_scatter_max_absolute_error"],
        "passed": True,
    }.items():
        _expect(scatter[field], expected, path=f"family_evidence.scatter_identity.{field}")
    for field in (
        "scattered_full_direction_sha256",
        "reference_scattered_full_direction_sha256",
    ):
        _digest(scatter[field], path=f"family_evidence.scatter_identity.{field}")
    scatter_error = _finite(
        scatter["max_absolute_error"],
        path="family_evidence.scatter_identity.max_absolute_error",
        nonnegative=True,
    )
    _finite(
        scatter["l2_error"],
        path="family_evidence.scatter_identity.l2_error",
        nonnegative=True,
    )
    if scatter_error > _EXPECTED_TOLERANCES["low_dimensional_scatter_max_absolute_error"]:
        raise R3ControlResultError("low-dimensional scatter identity gate failed")

    score = _mapping(
        evidence["score_identity"],
        path="family_evidence.score_identity",
        keys={
            "schema_version",
            "formula",
            "selected_direction_sha256",
            "scattered_full_direction_sha256",
            "low_projected_score_sha256",
            "full_projected_score_sha256",
            "max_absolute_error",
            "l2_error",
            "absolute_tolerance",
            "passed",
        },
    )
    for field, expected in {
        "schema_version": "phase2-r3-low-dimensional-score-identity/v1",
        "formula": "(S_full @ P) @ u_low == S_full @ (P @ u_low)",
        "selected_direction_sha256": selected_direction_sha,
        "scattered_full_direction_sha256": scatter["scattered_full_direction_sha256"],
        "absolute_tolerance": _EXPECTED_TOLERANCES[
            "low_dimensional_score_identity_max_absolute_error"
        ],
        "passed": True,
    }.items():
        _expect(score[field], expected, path=f"family_evidence.score_identity.{field}")
    for field in ("low_projected_score_sha256", "full_projected_score_sha256"):
        _digest(score[field], path=f"family_evidence.score_identity.{field}")
    score_error = _finite(
        score["max_absolute_error"],
        path="family_evidence.score_identity.max_absolute_error",
        nonnegative=True,
    )
    _finite(
        score["l2_error"],
        path="family_evidence.score_identity.l2_error",
        nonnegative=True,
    )
    if score_error > _EXPECTED_TOLERANCES["low_dimensional_score_identity_max_absolute_error"]:
        raise R3ControlResultError("low-dimensional score identity gate failed")

    gates = _mapping(
        evidence["gates"],
        path="family_evidence.gates",
        keys={
            "low_dimensional_objective_decrease",
            "low_dimensional_first_order_convergence",
            "low_dimensional_exact_rank",
            "low_dimensional_orthonormality",
            "low_dimensional_pseudoinverse_residual",
            "low_dimensional_scatter_identity",
            "low_dimensional_score_identity",
        },
    )
    for field in gates:
        _expect(gates[field], True, path=f"family_evidence.gates.{field}")
    return (
        {
            **dict(evidence),
            "target_audit": target,
            "projection": dict(projection),
            "geometry": dict(geometry),
            "head_audit": head,
            "scatter_identity": dict(scatter),
            "score_identity": dict(score),
            "gates": dict(gates),
        },
        int(first_order["selected_step"]),
    )


def _validated_family_evidence(
    family: R3ControlFamily,
    value: object,
    *,
    seed: int,
    oracle_sha256: str,
    dimensions: Mapping[str, int],
) -> tuple[dict[str, object], int]:
    if family == "exact_margin_prorm_plus":
        return _exact_margin_evidence(
            value,
            oracle_sha256=oracle_sha256,
            dimensions=dimensions,
        )
    if family == "exact_soft_label_bt":
        return _exact_soft_evidence(
            value,
            oracle_sha256=oracle_sha256,
            dimensions=dimensions,
        )
    return _low_dimensional_evidence(
        value,
        seed=seed,
        oracle_sha256=oracle_sha256,
        dimensions=dimensions,
    )


def _build_family_result(
    *,
    family: R3ControlFamily,
    seed: int,
    config: R3ControlsConfigBundle,
    input_training_sha256: str,
    train_oracle_rewards_sha256: str,
    input_dimensions: Mapping[str, object],
    family_evidence: Mapping[str, object],
) -> dict[str, object]:
    bundle = _require_bundle(config)
    method = _family(family)
    if type(seed) is not int or seed not in R3_GATE_C_SEEDS:
        raise R3ControlResultError(f"seed must be one of {R3_GATE_C_SEEDS!r}")
    training_sha = _digest(input_training_sha256, path="input_training_sha256")
    oracle_sha = _digest(
        train_oracle_rewards_sha256,
        path="train_oracle_rewards_sha256",
    )
    dimensions = _dimensions(dict(input_dimensions), family=method)
    normalized_evidence, completed_updates = _validated_family_evidence(
        method,
        dict(family_evidence),
        seed=seed,
        oracle_sha256=oracle_sha,
        dimensions=dimensions,
    )
    payload = {
        "schema_version": R3_CONTROL_FAMILY_RESULT_SCHEMA,
        "family": method,
        "seed": seed,
        "controls_config_semantic_sha256": bundle.semantic_sha256,
        "controls_config_file_sha256": bundle.file_sha256,
        "source_science_semantic_sha256": _SOURCE_SCIENCE_SEMANTIC_SHA256,
        "input_training_sha256": training_sha,
        "train_oracle_rewards_sha256": oracle_sha,
        "input_dimensions": dimensions,
        "information_boundary": dict(_RESULT_INFORMATION_BOUNDARY),
        "completion": {
            "status": "completed",
            "completed_updates": completed_updates,
            "stop_reason": "sustained_first_order_gate",
            "formal_family_result": True,
            "profile_only": False,
            "head_or_optimizer_state_retained": False,
        },
        "family_evidence": normalized_evidence,
    }
    normalized = _strict_json_copy(
        payload,
        path="Gate-C family result",
        error_type=R3ControlResultError,
    )
    if type(normalized) is not dict:
        raise RuntimeError("internal Gate-C result normalization failed")
    result = {
        **normalized,
        "result_sha256": _canonical_sha256(
            normalized,
            error_type=R3ControlResultError,
        ),
    }
    return validate_r3_control_family_result(result, bundle)


def adapt_exact_margin_prorm_plus_result(
    *,
    seed: int,
    config: R3ControlsConfigBundle,
    input_training_sha256: str,
    train_oracle_rewards_sha256: str,
    input_dimensions: Mapping[str, object],
    family_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Adapt one independent exact-margin/direct-identity observation."""

    return _build_family_result(
        family="exact_margin_prorm_plus",
        seed=seed,
        config=config,
        input_training_sha256=input_training_sha256,
        train_oracle_rewards_sha256=train_oracle_rewards_sha256,
        input_dimensions=input_dimensions,
        family_evidence=family_evidence,
    )


def adapt_exact_soft_label_bt_result(
    *,
    seed: int,
    config: R3ControlsConfigBundle,
    input_training_sha256: str,
    train_oracle_rewards_sha256: str,
    input_dimensions: Mapping[str, object],
    family_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Adapt one independent exact-soft-label BT observation."""

    return _build_family_result(
        family="exact_soft_label_bt",
        seed=seed,
        config=config,
        input_training_sha256=input_training_sha256,
        train_oracle_rewards_sha256=train_oracle_rewards_sha256,
        input_dimensions=input_dimensions,
        family_evidence=family_evidence,
    )


def adapt_low_dimensional_prorm_plus_result(
    *,
    seed: int,
    config: R3ControlsConfigBundle,
    input_training_sha256: str,
    train_oracle_rewards_sha256: str,
    input_dimensions: Mapping[str, object],
    family_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Adapt one independent low-dimensional ridge-free ProRM+ observation."""

    return _build_family_result(
        family="low_dimensional_prorm_plus",
        seed=seed,
        config=config,
        input_training_sha256=input_training_sha256,
        train_oracle_rewards_sha256=train_oracle_rewards_sha256,
        input_dimensions=input_dimensions,
        family_evidence=family_evidence,
    )


def validate_r3_control_family_result(
    value: object,
    config: R3ControlsConfigBundle,
) -> dict[str, object]:
    """Revalidate one compact result, including its self-hash and every gate."""

    bundle = _require_bundle(config)
    normalized = _strict_json_copy(
        value,
        path="Gate-C family result",
        error_type=R3ControlResultError,
    )
    result = _mapping(
        normalized,
        path="Gate-C family result",
        keys=_COMMON_RESULT_KEYS,
    )
    _expect(
        result["schema_version"],
        R3_CONTROL_FAMILY_RESULT_SCHEMA,
        path="schema_version",
    )
    family = _family(result["family"])
    seed = _integer(result["seed"], path="seed", minimum=0)
    if seed not in R3_GATE_C_SEEDS:
        raise R3ControlResultError(f"seed must be one of {R3_GATE_C_SEEDS!r}")
    for field, expected in {
        "controls_config_semantic_sha256": bundle.semantic_sha256,
        "controls_config_file_sha256": bundle.file_sha256,
        "source_science_semantic_sha256": _SOURCE_SCIENCE_SEMANTIC_SHA256,
    }.items():
        _expect(result[field], expected, path=field)
    _digest(result["input_training_sha256"], path="input_training_sha256")
    oracle_sha = _digest(
        result["train_oracle_rewards_sha256"],
        path="train_oracle_rewards_sha256",
    )
    dimensions = _dimensions(result["input_dimensions"], family=family)
    _expect(
        result["information_boundary"],
        _RESULT_INFORMATION_BOUNDARY,
        path="information_boundary",
    )
    evidence, completed_updates = _validated_family_evidence(
        family,
        result["family_evidence"],
        seed=seed,
        oracle_sha256=oracle_sha,
        dimensions=dimensions,
    )
    completion = _mapping(
        result["completion"],
        path="completion",
        keys={
            "status",
            "completed_updates",
            "stop_reason",
            "formal_family_result",
            "profile_only",
            "head_or_optimizer_state_retained",
        },
    )
    for field, expected in {
        "status": "completed",
        "completed_updates": completed_updates,
        "stop_reason": "sustained_first_order_gate",
        "formal_family_result": True,
        "profile_only": False,
        "head_or_optimizer_state_retained": False,
    }.items():
        _expect(completion[field], expected, path=f"completion.{field}")

    claimed_sha = _digest(result["result_sha256"], path="result_sha256")
    unsigned = dict(result)
    unsigned.pop("result_sha256")
    expected_sha = _canonical_sha256(unsigned, error_type=R3ControlResultError)
    if claimed_sha != expected_sha:
        raise R3ControlResultError("Gate-C family result self-hash mismatch")
    return {
        **dict(result),
        "input_dimensions": dimensions,
        "completion": dict(completion),
        "family_evidence": evidence,
    }


__all__ = [
    "R3_CONTROLS_CONFIG_PATH",
    "R3_CONTROLS_CONFIG_SCHEMA",
    "R3_CONTROL_FAMILY_RESULT_SCHEMA",
    "R3_CONTROL_FIRST_ORDER_GATE_SCHEMA",
    "R3_GATE_C_AUDIT_INTERVAL",
    "R3_GATE_C_CONFIG_SCHEMA",
    "R3_GATE_C_CONSECUTIVE_CHECKS",
    "R3_GATE_C_FAMILIES",
    "R3_GATE_C_FAMILY_RESULT_SCHEMA",
    "R3_GATE_C_MAX_UPDATES",
    "R3_GATE_C_MIN_UPDATES",
    "R3_GATE_C_PROFILE_UPDATES",
    "R3_GATE_C_SEEDS",
    "R3ControlFamily",
    "R3ControlResultError",
    "R3ControlsConfigBundle",
    "R3ControlsConfigError",
    "adapt_exact_margin_prorm_plus_result",
    "adapt_exact_soft_label_bt_result",
    "adapt_low_dimensional_prorm_plus_result",
    "load_r3_controls_config",
    "validate_r3_control_family_result",
]
