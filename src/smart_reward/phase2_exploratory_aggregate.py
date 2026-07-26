"""Strict descriptive aggregation for the fixed-three Phase-2 exploratory wave.

This module is deliberately independent of :mod:`smart_reward.phase2_aggregate`.
It accepts already-normalized in-memory seed records and performs no file or
HPC access.  A later publication layer may load and authenticate artifacts,
normalize them into this small mapping contract, and call
:func:`build_fixed_three_exploratory_aggregate`.

The scientific contract is intentionally narrow:

* the experimental units are exactly seeds ``20261001`` through ``20261003``;
* every effect is oriented as ``ProRM+ - BT``;
* the downstream and operational-oracle preference endpoint definitions and
  their favorable signs are fixed here;
* all three records must be present and admissible before any effect summary is
  emitted;
* intervals are deterministic paired-seed bootstrap descriptions, not
  hypothesis tests; and
* inferential/formal-decision fields are forbidden recursively, except for the
  required root marker ``formal_claim_eligible: false``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from statistics import median, stdev

from .phase2_config import (
    PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE as _CONFIG_BUDGETED_EVIDENCE_ROLE,
)
from .phase2_config import (
    PHASE2_BUDGETED_END_TO_END_SEEDS as _CONFIG_BUDGETED_SEEDS,
)
from .phase2_config import (
    PHASE2_BUDGETED_END_TO_END_STAGE as _CONFIG_BUDGETED_STAGE,
)
from .phase2_config import (
    PHASE2_FROZEN_ORACLE_B,
    PHASE2_FROZEN_ORACLE_TAU,
    PHASE2_RECOVERY_LR_SCHEDULE_SHA256,
)
from .phase2_heldout import (
    PHASE2_HELDOUT_SCHEMA_V2,
    verify_heldout_evaluation_payload,
)
from .phase2_rollout import (
    BUDGETED_COMMON_BETA_RULE,
    MEAN_POLICY_TO_REFERENCE_KL_CAP,
    PER_SEQUENCE_MAXIMUM_KL_CAP,
    PHASE2_ARM_ORDER,
    PHASE2_DESIGN_SCHEMA,
    PROMPT_MEAN_MAXIMUM_KL_CAP,
    PROMPT_MEAN_P95_KL_CAP,
    PROMPT_MEAN_P99_KL_CAP,
    REACHED_MAX_LENGTH_RATE_CAP,
)
from .phase2_training import (
    PHASE2_RECOVERY_TRAINING_SCHEMA,
    PRIMARY_TRAINING_ARM,
)
from .statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    paired_bootstrap_ci,
)

FIXED_THREE_EXPLORATORY_SEEDS = _CONFIG_BUDGETED_SEEDS
FIXED_THREE_EXPLORATORY_SCHEMA = "prorm-phase2-fixed-three-exploratory-descriptive-aggregate/v1"
CONTRAST_ORIENTATION = "prorm_plus_minus_bt"
PHASE2_BUDGETED_END_TO_END_RESULT_SCHEMA = "common-beta-budgeted-end-to-end/v1"
PHASE2_BUDGETED_END_TO_END_STAGE = _CONFIG_BUDGETED_STAGE
PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE = _CONFIG_BUDGETED_EVIDENCE_ROLE
_MAX_TORCH_SEED = 2**63 - 1
_ARM_ORDER = ("zero_b", "bt_mle", "prorm_plus", "oracle_step")
_PREFERENCE_FIT_METRICS = (
    "oracle_pairwise_cross_entropy",
    "oracle_probability_mae",
    "pairwise_order_accuracy",
)
_TRAINING_SCHEMAS = {
    "phase2-fresh-head-training/v2": "objective-first-order-convergence/v1",
    "phase2-fresh-head-training/v3": "objective-first-order-convergence/v2",
}

# These are not payload-selected values.  They are the locked recovery
# protocol, exported by the authoritative Phase-2 config/training contracts.
_FIRST_ORDER_RATIO_TOLERANCE = 1.0e-3
_FIRST_ORDER_MINIMUM_STEPS = 100
_FIRST_ORDER_MAXIMUM_STEPS = 12760
_FIRST_ORDER_CHECK_INTERVAL = 20
_FIRST_ORDER_CONSECUTIVE_CHECKS = 3
_FIRST_ORDER_FIXED_SNAPSHOT_STEPS = 720
_FIRST_ORDER_LEGACY_BOUNDARY_STEPS = 5760
_FIRST_ORDER_DENOMINATOR_FLOOR = 1.0e-30
_EXPECTED_SAFETY_THRESHOLDS = {
    "mean_policy_to_reference_kl_cap": MEAN_POLICY_TO_REFERENCE_KL_CAP,
    "prompt_mean_p95_kl_cap": PROMPT_MEAN_P95_KL_CAP,
    "prompt_mean_p99_kl_cap": PROMPT_MEAN_P99_KL_CAP,
    "prompt_mean_maximum_kl_cap": PROMPT_MEAN_MAXIMUM_KL_CAP,
    "per_sequence_maximum_kl_cap": PER_SEQUENCE_MAXIMUM_KL_CAP,
    "reached_max_length_rate_cap": REACHED_MAX_LENGTH_RATE_CAP,
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _float32_head_sha256(value: object, *, name: str) -> tuple[list[float], str]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
        or not value
    ):
        raise TypeError(f"{name} must be a non-empty sequence")
    weights = [_finite(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    try:
        raw = struct.pack(f"<{len(weights)}f", *weights)
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name} cannot be represented as torch.float32") from error
    digest = hashlib.sha256()
    digest.update(b"torch.float32")
    digest.update(repr((len(weights),)).encode("ascii"))
    digest.update(raw)
    return weights, digest.hexdigest()


def _budgeted_identity_from_result(result: Mapping[str, object]) -> dict[str, object]:
    """Extract the aggregatable identity before judging a seed's evidence.

    Keeping this separate from the admission gates is intentional: a failed
    seed must still be attributable to the immutable design, runtime contract,
    and accepted global-beta freeze that produced it.
    """

    seed = _seed(_required(result, "seed", name="budgeted result"), name="budgeted result.seed")
    frozen = _mapping(
        _required(result, "common_beta_frozen_evidence", name=f"budgeted seed {seed}"),
        name=f"budgeted seed {seed}.common_beta_frozen_evidence",
    )
    identity = {
        "phase2_design_sha256": _digest(
            _required(result, "phase2_design_sha256", name=f"budgeted seed {seed}"),
            name=f"budgeted seed {seed}.phase2_design_sha256",
        ),
        "phase2_runtime_contract_sha256": _digest(
            _required(
                result,
                "phase2_runtime_contract_sha256",
                name=f"budgeted seed {seed}",
            ),
            name=f"budgeted seed {seed}.phase2_runtime_contract_sha256",
        ),
        "beta_source_aggregate_sha256": _digest(
            _required(
                frozen,
                "beta_source_aggregate_sha256",
                name=f"budgeted seed {seed}.common_beta_frozen_evidence",
            ),
            name=(f"budgeted seed {seed}.common_beta_frozen_evidence.beta_source_aggregate_sha256"),
        ),
        "frozen_global_beta": _finite(
            _required(
                frozen,
                "frozen_global_beta",
                name=f"budgeted seed {seed}.common_beta_frozen_evidence",
            ),
            name=(f"budgeted seed {seed}.common_beta_frozen_evidence.frozen_global_beta"),
            positive=True,
        ),
    }
    return {"seed": seed, **identity}


def _budgeted_gate(condition: bool, message: str) -> None:
    """Raise a local admission failure, never a partially usable endpoint."""

    if not condition:
        raise ValueError(message)


def _validate_budgeted_frozen_beta_evidence(
    result: Mapping[str, object],
    *,
    identity: Mapping[str, object],
) -> None:
    seed = identity["seed"]
    frozen = _mapping(
        result["common_beta_frozen_evidence"],
        name=f"budgeted seed {seed}.common_beta_frozen_evidence",
    )
    expected = {
        "schema_version": "common-beta-frozen-global-budgeted/v1",
        "evidence_role": PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
        "formal_eligibility": False,
        "supports_formal_claim": False,
        "beta_matches_frozen_global_beta": True,
        "beta_selected_from_current_seed_curvature": False,
        "accepted_freeze_beta_reused_without_recalibration": True,
        "current_seed_can_change_beta": False,
        "frozen_in_phase2_design_identity": True,
        "learner_specific_rescaling": False,
        "post_evaluation_retuning": False,
    }
    for key, expected_value in expected.items():
        _budgeted_gate(
            frozen.get(key) == expected_value,
            f"budgeted seed {seed} frozen-beta evidence field {key!r} is invalid",
        )
    beta_common = _finite(
        frozen.get("beta_common"),
        name=f"budgeted seed {seed}.common_beta_frozen_evidence.beta_common",
        positive=True,
    )
    _budgeted_gate(
        beta_common == identity["frozen_global_beta"],
        f"budgeted seed {seed} beta_common differs from frozen_global_beta",
    )

    runtime = _mapping(
        _required(result, "phase2_runtime_contract", name=f"budgeted seed {seed}"),
        name=f"budgeted seed {seed}.phase2_runtime_contract",
    )
    required_runtime = {
        "schema_version": PHASE2_DESIGN_SCHEMA,
        "stage": PHASE2_BUDGETED_END_TO_END_STAGE,
        "formal_eligibility": False,
        "pilot_phase": None,
        "common_beta_rule": BUDGETED_COMMON_BETA_RULE,
        "common_beta_calibration_split": "excluded_pilot",
        "common_beta_source": (
            "accepted_freeze_global_beta_in_budgeted_end_to_end_design_identity"
        ),
        "frozen_global_beta": identity["frozen_global_beta"],
        "beta_source_aggregate_sha256": identity["beta_source_aggregate_sha256"],
        "learner_specific_line_search": False,
        "arm_order": list(PHASE2_ARM_ORDER),
    }
    for key, expected_value in required_runtime.items():
        _budgeted_gate(
            runtime.get(key) == expected_value,
            f"budgeted seed {seed} runtime contract field {key!r} disagrees with the freeze",
        )
    _budgeted_gate(
        _canonical_sha256(runtime) == identity["phase2_runtime_contract_sha256"],
        f"budgeted seed {seed} runtime-contract SHA256 does not bind its payload",
    )


def _validate_budgeted_first_order_head(value: object, *, name: str) -> None:
    """Verify the non-negotiable full-data first-order selection gate.

    This is deliberately independent of an optimizer schedule revision.  The
    aggregate only needs the evidence that both primary heads were selected by
    a sustained, full-data, unclipped first-order gate without held-out access.
    """

    convergence = _mapping(value, name=name)
    _budgeted_gate(
        convergence.get("schema_version") == "objective-first-order-convergence/v2"
        and convergence.get("converged") is True
        and convergence.get("fail_closed") is True,
        f"{name} does not prove fail-closed convergence",
    )
    _budgeted_gate(
        convergence.get("test_or_validation_data_accessed") is False,
        f"{name} used held-out data for training selection",
    )
    spec = _mapping(convergence.get("spec"), name=f"{name}.spec")
    _budgeted_gate(
        spec.get("schema_version") == "objective-first-order-convergence-spec/v2"
        and spec.get("fail_closed") is True
        and spec.get("gradient") == "full_data_post_update_unclipped"
        and spec.get("denominator") == "exact_zero_initialization_gradient_l2_norm"
        and spec.get("validation_or_test_selection") is False,
        f"{name} first-order specification is not a full-data fail-closed gate",
    )
    _budgeted_gate(
        spec.get("gradient_ratio_tolerance") == _FIRST_ORDER_RATIO_TOLERANCE
        and spec.get("gradient_norm_denominator_floor") == _FIRST_ORDER_DENOMINATOR_FLOOR
        and spec.get("min_steps") == _FIRST_ORDER_MINIMUM_STEPS
        and spec.get("max_steps") == _FIRST_ORDER_MAXIMUM_STEPS
        and spec.get("check_interval") == _FIRST_ORDER_CHECK_INTERVAL
        and spec.get("consecutive_checks") == _FIRST_ORDER_CONSECUTIVE_CHECKS,
        f"{name} self-reported first-order thresholds differ from the locked protocol",
    )
    protocol = _mapping(spec.get("optimizer_protocol"), name=f"{name}.spec.optimizer_protocol")
    schedule = _mapping(
        protocol.get("learning_rate_schedule"), name=f"{name}.spec.optimizer_protocol.schedule"
    )
    _budgeted_gate(
        protocol.get("schema_version") == "deterministic-adamw-lr-decay/v1"
        and protocol.get("first_order_audit_dtype") == "float64"
        and protocol.get("legacy_constant_lr_boundary_snapshot_steps")
        == _FIRST_ORDER_LEGACY_BOUNDARY_STEPS
        and protocol.get("validation_or_test_selection") is False
        and schedule.get("schedule_sha256") == PHASE2_RECOVERY_LR_SCHEDULE_SHA256,
        f"{name} is not bound to the adopted post-recovery optimizer protocol",
    )
    min_steps = _FIRST_ORDER_MINIMUM_STEPS
    max_steps = _FIRST_ORDER_MAXIMUM_STEPS
    interval = _FIRST_ORDER_CHECK_INTERVAL
    sustained = _FIRST_ORDER_CONSECUTIVE_CHECKS
    selected = _positive_integer(
        convergence.get("selected_primary_step"), name=f"{name}.selected_primary_step"
    )
    _budgeted_gate(
        min_steps <= selected <= max_steps and selected % interval == 0,
        f"{name} selected first-order iterate is invalid",
    )
    _budgeted_gate(
        convergence.get("consecutive_threshold_passes_at_selection") == sustained,
        f"{name} lacks the required sustained first-order checks",
    )
    final_gate = _mapping(convergence.get("final_gate"), name=f"{name}.final_gate")
    measurement = _mapping(final_gate.get("measurement"), name=f"{name}.final_gate.measurement")
    _budgeted_gate(
        final_gate.get("step") == selected
        and final_gate.get("threshold_passed") is True
        and final_gate.get("fresh_post_restore_audit") is True
        and measurement.get("audit_dtype") == "float64",
        f"{name} final first-order audit is incomplete",
    )
    ratio = _finite(
        final_gate.get("gradient_ratio_to_zero_initialization"),
        name=f"{name}.final_gate.gradient_ratio_to_zero_initialization",
    )
    initial = _mapping(
        convergence.get("initial_zero_head_measurement"),
        name=f"{name}.initial_zero_head_measurement",
    )
    initial_gradient = _finite(
        initial.get("gradient_l2_norm"),
        name=f"{name}.initial_zero_head_measurement.gradient_l2_norm",
    )
    final_gradient = _finite(
        measurement.get("gradient_l2_norm"),
        name=f"{name}.final_gate.measurement.gradient_l2_norm",
    )
    _budgeted_gate(
        initial_gradient >= 0.0 and final_gradient >= 0.0,
        f"{name} first-order gradient norms must be non-negative",
    )
    expected_ratio = final_gradient / max(initial_gradient, _FIRST_ORDER_DENOMINATOR_FLOOR)
    _budgeted_gate(
        0.0 <= ratio <= _FIRST_ORDER_RATIO_TOLERANCE
        and math.isclose(ratio, expected_ratio, rel_tol=1.0e-10, abs_tol=1.0e-14),
        f"{name} final gradient ratio is not tolerance-valid full-data evidence",
    )
    checks = convergence.get("checks")
    _budgeted_gate(
        isinstance(checks, list) and len(checks) >= sustained,
        f"{name} lacks first-order check history",
    )
    history_steps = [
        _positive_integer(check.get("step"), name=f"{name}.checks[{index}].step")
        if isinstance(check, Mapping)
        else -1
        for index, check in enumerate(checks)
    ]
    _budgeted_gate(
        all(step > 0 and step <= selected and step % interval == 0 for step in history_steps)
        and history_steps == sorted(history_steps)
        and len(history_steps) == len(set(history_steps)),
        f"{name} first-order check history is duplicated or out of order",
    )
    selected_steps = {selected - offset * interval for offset in range(sustained)}
    selected_checks = [
        check
        for check in checks
        if isinstance(check, Mapping) and check.get("step") in selected_steps
    ]
    _budgeted_gate(
        {check.get("step") for check in selected_checks} == selected_steps
        and len(selected_checks) == sustained
        and all(
            check.get("threshold_passed") is True
            and check.get("post_update") is True
            and check.get("full_data") is True
            and check.get("gradient_clipping_applied") is False
            and isinstance(check.get("measurement"), Mapping)
            and check["measurement"].get("audit_dtype") == "float64"
            and 0.0
            <= _finite(
                check.get("gradient_ratio_to_zero_initialization"),
                name=f"{name}.checks[{check.get('step')}].gradient_ratio",
            )
            <= _FIRST_ORDER_RATIO_TOLERANCE
            for check in selected_checks
        ),
        f"{name} lacks sustained full-data unclipped first-order checks",
    )
    fixed = _mapping(
        convergence.get("fixed_step_compute_matched_snapshot"),
        name=f"{name}.fixed_step_compute_matched_snapshot",
    )
    legacy = _mapping(
        convergence.get("legacy_constant_lr_boundary_snapshot"),
        name=f"{name}.legacy_constant_lr_boundary_snapshot",
    )
    _budgeted_gate(
        convergence.get("fixed_step_snapshot_steps") == _FIRST_ORDER_FIXED_SNAPSHOT_STEPS
        and convergence.get("fixed_step_snapshot_is_not_primary_selection") is True
        and fixed.get("step") == _FIRST_ORDER_FIXED_SNAPSHOT_STEPS
        and fixed.get("used_as_primary_selection_rule") is False
        and legacy.get("step") == _FIRST_ORDER_LEGACY_BOUNDARY_STEPS
        and legacy.get("used_as_primary_selection_rule") is False
        and legacy.get("test_or_validation_data_accessed") is False,
        f"{name} lacks required 720/5760 diagnostic evidence",
    )
    execution = _mapping(
        convergence.get("optimizer_protocol_execution"),
        name=f"{name}.optimizer_protocol_execution",
    )
    _budgeted_gate(
        execution.get("schema_version") == "deterministic-adamw-lr-decay-execution/v2"
        and execution.get("protocol") == protocol
        and execution.get("completed_updates_observed")
        >= max(selected, _FIRST_ORDER_LEGACY_BOUNDARY_STEPS)
        and execution.get("per_update_state_checks") is not None
        and execution.get("test_or_validation_data_accessed") is False,
        f"{name} optimizer execution evidence is incomplete",
    )


def _validate_serialized_training_head(
    value: object,
    *,
    name: str,
    expected_arm: str,
    expected_method: str,
    expected_weights: object | None = None,
) -> str:
    head = _mapping(value, name=name)
    weights, computed_sha256 = _float32_head_sha256(
        head.get("head_weight"),
        name=f"{name}.head_weight",
    )
    if expected_weights is not None:
        normalized_expected, _ = _float32_head_sha256(
            expected_weights,
            name=f"{name}.expected_head_weight",
        )
        _budgeted_gate(
            weights == normalized_expected,
            f"{name} differs from the serialized runner head",
        )
    zero_sha256 = _float32_head_sha256(
        [0.0] * len(weights),
        name=f"{name}.zero_head",
    )[1]
    _budgeted_gate(
        head.get("arm") == expected_arm
        and head.get("method") == expected_method
        and head.get("head_dtype") == "torch.float32"
        and head.get("head_sha256") == computed_sha256
        and head.get("initial_head_sha256") == zero_sha256,
        f"{name} weight/dtype/zero-initialization identity is invalid",
    )
    convergence = _mapping(
        head.get("first_order_convergence"),
        name=f"{name}.first_order_convergence",
    )
    execution = _mapping(
        convergence.get("optimizer_protocol_execution"),
        name=f"{name}.optimizer_protocol_execution",
    )
    _budgeted_gate(
        convergence.get("selected_primary_head_sha256") == computed_sha256
        and execution.get("selected_head_sha256") == computed_sha256,
        f"{name} selected first-order checkpoint is not bound to its weights",
    )
    _validate_budgeted_first_order_head(
        convergence,
        name=f"{name}.first_order_convergence",
    )
    return computed_sha256


def _validate_budgeted_first_order_evidence(result: Mapping[str, object], *, seed: int) -> None:
    training = _mapping(
        _required(result, "head_training", name=f"budgeted seed {seed}"),
        name=f"budgeted seed {seed}.head_training",
    )
    _budgeted_gate(
        training.get("test_data_accessed") is False
        and training.get("old_phase1_comparison_heads_reused") is False
        and training.get("source") == "trained_after_train_oracle_rescore",
        f"budgeted seed {seed} training isolation evidence is invalid",
    )
    audit = _mapping(training.get("audit"), name=f"budgeted seed {seed}.head_training.audit")
    _budgeted_gate(
        audit.get("schema_version") == PHASE2_RECOVERY_TRAINING_SCHEMA
        and audit.get("training_design_sha256") == result.get("phase2_design_sha256")
        and audit.get("training_arm") == PRIMARY_TRAINING_ARM,
        f"budgeted seed {seed} training schema/settings identity is invalid",
    )
    for field in ("training_settings_sha256", "training_instance_sha256", "input_training_sha256"):
        _digest(audit.get(field), name=f"budgeted seed {seed}.audit.{field}")
    weights = _mapping(training.get("head_weights"), name=f"budgeted seed {seed}.head_weights")
    _budgeted_gate(
        set(weights) == {"bt_mle", "prorm_plus"}
        and _canonical_sha256(
            {learner: list(weights[learner]) for learner in ("bt_mle", "prorm_plus")}
        )
        == training.get("heads_sha256"),
        f"budgeted seed {seed} serialized heads do not bind heads_sha256",
    )
    _digest(training.get("heads_sha256"), name=f"budgeted seed {seed}.heads_sha256")
    primary = _mapping(
        audit.get("primary_heads"),
        name=f"budgeted seed {seed}.head_training.audit.primary_heads",
    )
    _budgeted_gate(
        set(primary) == {"bt_mle", "prorm_plus"},
        f"budgeted seed {seed} primary first-order heads are incomplete",
    )
    primary_hashes: dict[str, str] = {}
    for learner in ("bt_mle", "prorm_plus"):
        primary_hashes[learner] = _validate_serialized_training_head(
            primary[learner],
            name=f"budgeted seed {seed}.{learner}",
            expected_arm=PRIMARY_TRAINING_ARM,
            expected_method=learner,
            expected_weights=weights[learner],
        )
    for control, expected_arm, expected_method in (
        (
            "low_dimensional_control",
            "low_dimensional_tangent_positive_control",
            "prorm_plus",
        ),
        ("exact_margin_control", "exact_margin_positive_control", "prorm_plus"),
        (
            "exact_soft_label_bt_control",
            "exact_soft_label_bt_secondary_diagnostic",
            "bt_mle",
        ),
    ):
        control_value = _mapping(audit.get(control), name=f"budgeted seed {seed}.{control}")
        control_head = control_value.get("head")
        _budgeted_gate(
            bool(control_value) and isinstance(control_head, Mapping),
            f"budgeted seed {seed} {control} evidence is empty or incomplete",
        )
        _validate_serialized_training_head(
            control_head,
            name=f"budgeted seed {seed}.{control}.head",
            expected_arm=expected_arm,
            expected_method=expected_method,
        )
        if control == "low_dimensional_control":
            bt_reference = _mapping(
                control_value.get("bt_head"),
                name=f"budgeted seed {seed}.{control}.bt_head",
            )
            _budgeted_gate(
                bt_reference.get("head_sha256") == primary_hashes["bt_mle"]
                and bt_reference.get("retrained") is False,
                f"budgeted seed {seed} low-dimensional BT reference is cross-bound",
            )
    for key in ("primary_optimization_audit", "direct_oracle_identity"):
        nested = _mapping(audit.get(key), name=f"budgeted seed {seed}.{key}")
        _budgeted_gate(bool(nested), f"budgeted seed {seed} {key} evidence is empty")
    isolation = _mapping(audit.get("isolation"), name=f"budgeted seed {seed}.audit.isolation")
    _budgeted_gate(
        isolation
        == {
            "test_data_accessed": False,
            "old_phase1_comparison_heads_used": False,
            "raw_node_rewards_retained": False,
            "raw_labels_retained": False,
            "primary_heads_are_fresh_zero_initialized": True,
        },
        f"budgeted seed {seed} audit isolation evidence is invalid",
    )


def _validate_budgeted_pre_oracle_safety(result: Mapping[str, object], *, seed: int) -> None:
    gate = _mapping(
        _required(result, "pre_oracle_safety_gate", name=f"budgeted seed {seed}"),
        name=f"budgeted seed {seed}.pre_oracle_safety_gate",
    )
    required = {
        "schema_version": "phase2-pre-oracle-safety-gate/v2",
        "design_stage": PHASE2_BUDGETED_END_TO_END_STAGE,
        "measure_only": False,
        "formal_gate": False,
        "enforced_before_final_oracle": True,
        "supports_formal_claim": False,
        "evidence_role": PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
        "passed": True,
        "beta_retuned": False,
        "on_violation": "fail_before_final_oracle_and_heldout",
    }
    for key, expected in required.items():
        _budgeted_gate(
            gate.get(key) == expected,
            f"budgeted seed {seed} pre-oracle safety field {key!r} is invalid",
        )
    _budgeted_gate(
        gate.get("violations") == [],
        f"budgeted seed {seed} pre-oracle safety gate reports violations",
    )
    thresholds = _mapping(gate.get("thresholds"), name=f"budgeted seed {seed}.safety.thresholds")
    _budgeted_gate(
        dict(thresholds) == _EXPECTED_SAFETY_THRESHOLDS,
        f"budgeted seed {seed} pre-oracle thresholds drift from the runtime contract",
    )
    arms = _mapping(result.get("arms"), name=f"budgeted seed {seed}.arms")
    observed = _mapping(gate.get("observed_by_arm"), name=f"budgeted seed {seed}.safety.observed")
    expected_arms = set(PHASE2_ARM_ORDER)
    _budgeted_gate(
        set(observed) == expected_arms and set(arms) == expected_arms,
        f"budgeted seed {seed} pre-oracle arm set is invalid",
    )
    observed_metric_names = tuple(key.removesuffix("_cap") for key in _EXPECTED_SAFETY_THRESHOLDS)
    violations: list[str] = []
    for arm_name in PHASE2_ARM_ORDER:
        arm = _mapping(arms[arm_name], name=f"budgeted seed {seed}.arms.{arm_name}")
        tail = _mapping(
            arm.get("on_policy_kl_tail"), name=f"budgeted seed {seed}.arms.{arm_name}.tail"
        )
        rollout = _mapping(arm.get("rollout"), name=f"budgeted seed {seed}.arms.{arm_name}.rollout")
        expected_observed = {
            "mean_policy_to_reference_kl": _finite(
                arm.get("mean_on_policy_kl_pi_updated_to_pi0"),
                name=f"budgeted seed {seed}.{arm_name}.mean_kl",
            ),
            "prompt_mean_p95_kl": _finite(
                tail.get("p95"), name=f"budgeted seed {seed}.{arm_name}.p95"
            ),
            "prompt_mean_p99_kl": _finite(
                tail.get("p99"), name=f"budgeted seed {seed}.{arm_name}.p99"
            ),
            "prompt_mean_maximum_kl": _finite(
                tail.get("maximum"), name=f"budgeted seed {seed}.{arm_name}.maximum"
            ),
            "per_sequence_maximum_kl": _finite(
                tail.get("per_sequence_maximum"),
                name=f"budgeted seed {seed}.{arm_name}.sequence_maximum",
            ),
            "reached_max_length_rate": _finite(
                rollout.get("reached_max_length_rate"),
                name=f"budgeted seed {seed}.{arm_name}.length_rate",
            ),
        }
        actual_observed = _mapping(
            observed[arm_name], name=f"budgeted seed {seed}.safety.{arm_name}"
        )
        _budgeted_gate(
            set(actual_observed) == set(observed_metric_names)
            and all(
                math.isclose(
                    _finite(
                        actual_observed[metric],
                        name=f"budgeted seed {seed}.safety.{arm_name}.{metric}",
                    ),
                    expected_observed[metric],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
                for metric in observed_metric_names
            ),
            f"budgeted seed {seed} pre-oracle observations do not bind arm summaries",
        )
        violations.extend(
            f"{arm_name}:{metric}"
            for metric, value in expected_observed.items()
            if value > _EXPECTED_SAFETY_THRESHOLDS[f"{metric}_cap"]
        )
    _budgeted_gate(
        gate.get("violations") == violations and gate.get("passed") is (not violations),
        f"budgeted seed {seed} pre-oracle pass arithmetic is invalid",
    )


def _extract_budgeted_endpoints(
    result: Mapping[str, object],
    *,
    identity: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    seed = identity["seed"]
    heldout = _mapping(
        _required(result, "heldout_fixed_beta", name=f"budgeted seed {seed}"),
        name=f"budgeted seed {seed}.heldout_fixed_beta",
    )
    verify_heldout_evaluation_payload(
        heldout,
        expected_sha256=_required(
            result,
            "heldout_fixed_beta_sha256",
            name=f"budgeted seed {seed}",
        ),
    )
    _budgeted_gate(
        heldout.get("schema_version") == PHASE2_HELDOUT_SCHEMA_V2,
        f"budgeted seed {seed} must use held-out v2 evidence",
    )
    _budgeted_gate(
        _finite(
            heldout.get("beta_common"),
            name=f"budgeted seed {seed}.heldout.beta_common",
            positive=True,
        )
        == identity["frozen_global_beta"],
        f"budgeted seed {seed} held-out beta differs from the accepted freeze",
    )
    training = _mapping(result.get("head_training"), name=f"budgeted seed {seed}.head_training")
    heads_sha256 = _digest(training.get("heads_sha256"), name=f"budgeted seed {seed}.heads_sha256")
    source_config_hash = _digest(
        result.get("source_config_hash"), name=f"budgeted seed {seed}.source_config_hash"
    )
    frozen_state = _mapping(
        heldout.get("frozen_state"), name=f"budgeted seed {seed}.heldout.frozen_state"
    )
    _budgeted_gate(
        _canonical_sha256(frozen_state)
        == _digest(
            heldout.get("frozen_state_sha256"),
            name=f"budgeted seed {seed}.heldout.frozen_state_sha256",
        ),
        f"budgeted seed {seed} held-out frozen-state hash is invalid",
    )
    expected_frozen = {
        "schema_version": "phase2-heldout-frozen-state/v1",
        "source_config_hash": source_config_hash,
        "phase2_design_sha256": identity["phase2_design_sha256"],
        "phase2_runtime_contract_sha256": identity["phase2_runtime_contract_sha256"],
        "seed": seed,
        "heads_sha256": heads_sha256,
        "training_design_sha256": identity["phase2_design_sha256"],
        "beta_common": identity["frozen_global_beta"],
        "heads_frozen": True,
        "beta_common_frozen": True,
        "deployed_directions_frozen": True,
    }
    _budgeted_gate(
        all(frozen_state.get(key) == expected for key, expected in expected_frozen.items())
        and _digest(
            frozen_state.get("deployment_identity_sha256"),
            name=f"budgeted seed {seed}.heldout.deployment_identity_sha256",
        )
        is not None,
        f"budgeted seed {seed} held-out frozen state is not bound to this seed",
    )
    runtime = _mapping(result.get("phase2_runtime_contract"), name=f"budgeted seed {seed}.runtime")
    solver = _mapping(heldout.get("solver"), name=f"budgeted seed {seed}.heldout.solver")
    _budgeted_gate(
        solver.get("pcg_dtype") == "float64"
        and solver.get("pcg_max_iterations") == runtime.get("pcg_max_iterations")
        and solver.get("pcg_tolerance") == runtime.get("pcg_tolerance")
        and solver.get("relative_damping") == runtime.get("relative_damping")
        and solver.get("split_specific_node_fisher_and_damping") is True
        and solver.get("explicit_pcg_evidence_serialized_per_split") is True
        and solver.get("all_direction_and_regret_solves_audited") is True,
        f"budgeted seed {seed} held-out solver is not bound to the runtime",
    )
    oracle_rescore = _mapping(
        heldout.get("oracle_rescore"), name=f"budgeted seed {seed}.heldout.oracle_rescore"
    )
    transform = _mapping(
        oracle_rescore.get("transform"),
        name=f"budgeted seed {seed}.heldout.oracle_rescore.transform",
    )
    _budgeted_gate(
        oracle_rescore.get("source")
        == "saved_validation_and_test_candidates_rescored_after_policy_freeze"
        and oracle_rescore.get("raw_oracle_logits_serialized") is False
        and transform == {"b": PHASE2_FROZEN_ORACLE_B, "tau": PHASE2_FROZEN_ORACLE_TAU}
        and _digest(
            oracle_rescore.get("oracle_chat_template_sha256"),
            name=f"budgeted seed {seed}.heldout.oracle_template",
        )
        is not None
        and _digest(
            oracle_rescore.get("combined_transformed_rewards_sha256"),
            name=f"budgeted seed {seed}.heldout.oracle_rewards",
        )
        is not None,
        f"budgeted seed {seed} held-out oracle rescore is invalid",
    )
    splits = _mapping(heldout.get("splits"), name=f"budgeted seed {seed}.heldout.splits")
    test = _mapping(splits.get("test"), name=f"budgeted seed {seed}.heldout.splits.test")
    _budgeted_gate(
        _finite(
            test.get("fixed_beta"),
            name=f"budgeted seed {seed}.heldout.splits.test.fixed_beta",
            positive=True,
        )
        == identity["frozen_global_beta"]
        and test.get("fixed_beta_source")
        == "accepted_freeze_global_beta_frozen_in_budgeted_end_to_end_design",
        f"budgeted seed {seed} test split is not bound to the accepted freeze",
    )
    learners = _mapping(
        test.get("learners"), name=f"budgeted seed {seed}.heldout.splits.test.learners"
    )
    preference_fit = _mapping(
        test.get("preference_fit"), name=f"budgeted seed {seed}.heldout.splits.test.preference_fit"
    )
    arms = _mapping(
        _required(result, "arms", name=f"budgeted seed {seed}"),
        name=f"budgeted seed {seed}.arms",
    )
    endpoints = {key: {} for key in _ENDPOINT_KEYS}
    head_weights = _mapping(training.get("head_weights"), name=f"budgeted seed {seed}.head_weights")
    for learner in ("bt_mle", "prorm_plus"):
        heldout_learner = _mapping(
            learners.get(learner), name=f"budgeted seed {seed}.heldout.test.learners.{learner}"
        )
        arm = _mapping(arms.get(learner), name=f"budgeted seed {seed}.arms.{learner}")
        utility = _mapping(arm.get("utility"), name=f"budgeted seed {seed}.arms.{learner}.utility")
        fit = _mapping(
            preference_fit.get(learner),
            name=f"budgeted seed {seed}.heldout.test.preference_fit.{learner}",
        )
        endpoints["heldout_local_regret"][learner] = _finite(
            heldout_learner.get("local_regret_at_frozen_global_beta"),
            name=f"budgeted seed {seed}.heldout.test.{learner}.local_regret",
        )
        endpoints["finite_policy_utility"][learner] = _finite(
            utility.get("mean_target_utility"),
            name=f"budgeted seed {seed}.arms.{learner}.utility.mean_target_utility",
        )
        mean_reward = _finite(
            utility.get("mean_target_reward"),
            name=f"budgeted seed {seed}.arms.{learner}.utility.mean_target_reward",
        )
        mean_kl = _finite(
            utility.get("mean_on_policy_kl_pi_updated_to_pi0"),
            name=f"budgeted seed {seed}.arms.{learner}.utility.mean_on_policy_kl",
        )
        _budgeted_gate(
            utility.get("beta_common") == identity["frozen_global_beta"]
            and math.isclose(
                endpoints["finite_policy_utility"][learner],
                mean_reward - identity["frozen_global_beta"] * mean_kl,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                mean_kl,
                _finite(
                    arm.get("mean_on_policy_kl_pi_updated_to_pi0"),
                    name=f"budgeted seed {seed}.arms.{learner}.mean_kl",
                ),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
            f"budgeted seed {seed} {learner} utility endpoint does not bind reward and KL",
        )
        _budgeted_gate(
            heldout_learner.get("head_sha256")
            == _canonical_sha256([float(value) for value in head_weights[learner]]),
            f"budgeted seed {seed} {learner} held-out head identity is spliced",
        )
        for metric in _PREFERENCE_FIT_METRICS:
            endpoints[metric][learner] = _finite(
                fit.get(metric),
                name=f"budgeted seed {seed}.heldout.test.preference_fit.{learner}.{metric}",
            )
    return endpoints


def normalize_budgeted_end_to_end_seed_result(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Normalize one raw budgeted E2E result for the fixed-three aggregate.

    The returned mapping is deliberately the *only* bridge from raw result
    artifacts to descriptive aggregation.  Immutable identity is extracted
    first.  Every downstream endpoint is withheld if any non-formal but
    enforced safety, held-out-PCG, frozen-beta, or first-order training gate
    fails.  No preference-only side summary is created: the three held-out
    preference-fit values enter the existing fixed endpoint set directly.
    """

    raw = _mapping(result, name="budgeted result")
    identity = _budgeted_identity_from_result(raw)
    normalized: dict[str, object] = {**identity, "admissible": False}
    seed = identity["seed"]
    try:
        root_requirements = {
            "schema_version": PHASE2_BUDGETED_END_TO_END_RESULT_SCHEMA,
            "design_stage": PHASE2_BUDGETED_END_TO_END_STAGE,
            "formal_eligibility": False,
            "formal_claim_eligible": False,
            "supports_formal_claim": False,
            "per_seed_supports_formal_claim": False,
            "excluded_from_confirmatory_evidence": True,
            "confirmatory_authorization_created": False,
            "evidence_role": PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
        }
        for key, expected in root_requirements.items():
            _budgeted_gate(
                raw.get(key) == expected,
                f"budgeted seed {seed} result field {key!r} is invalid",
            )
        _validate_budgeted_frozen_beta_evidence(raw, identity=identity)
        _validate_budgeted_first_order_evidence(raw, seed=seed)
        _validate_budgeted_pre_oracle_safety(raw, seed=seed)
        normalized["endpoints"] = _extract_budgeted_endpoints(raw, identity=identity)
        normalized["admissible"] = True
    except (KeyError, TypeError, ValueError):
        # A normalized false record remains aggregatable and auditable, while
        # deliberately exposing no outcome value from a failed seed.
        normalized.pop("endpoints", None)
    return normalized


@dataclass(frozen=True, slots=True)
class _EndpointDefinition:
    key: str
    split: str
    metric: str
    direction: str
    favorable_sign: str

    def to_dict(self) -> dict[str, str]:
        return {
            "split": self.split,
            "metric": self.metric,
            "contrast_orientation": CONTRAST_ORIENTATION,
            "direction": self.direction,
            "favorable_prorm_plus_minus_bt_sign": self.favorable_sign,
        }


_ENDPOINTS = (
    _EndpointDefinition(
        key="heldout_local_regret",
        split="test",
        metric="local_regret_at_frozen_global_beta",
        direction="lower_is_better",
        favorable_sign="negative",
    ),
    _EndpointDefinition(
        key="finite_policy_utility",
        split="test",
        metric="operational_oracle_reward_minus_beta_common_on_policy_kl",
        direction="higher_is_better",
        favorable_sign="positive",
    ),
    _EndpointDefinition(
        key="oracle_pairwise_cross_entropy",
        split="test",
        metric="prompt_mean_operational_oracle_pairwise_cross_entropy",
        direction="lower_is_better",
        favorable_sign="negative",
    ),
    _EndpointDefinition(
        key="oracle_probability_mae",
        split="test",
        metric="prompt_mean_operational_oracle_pairwise_probability_mae",
        direction="lower_is_better",
        favorable_sign="negative",
    ),
    _EndpointDefinition(
        key="pairwise_order_accuracy",
        split="test",
        metric="prompt_mean_operational_oracle_pairwise_order_accuracy_ties_half",
        direction="higher_is_better",
        favorable_sign="positive",
    ),
)
_ENDPOINT_BY_KEY = {endpoint.key: endpoint for endpoint in _ENDPOINTS}
_ENDPOINT_KEYS = tuple(endpoint.key for endpoint in _ENDPOINTS)

_IDENTITY_KEYS = (
    "phase2_design_sha256",
    "phase2_runtime_contract_sha256",
    "beta_source_aggregate_sha256",
    "frozen_global_beta",
)
_INFERENTIAL_EXACT_KEYS = frozenset(
    {
        "passed",
        "not_passed",
        "p_value",
        "pvalue",
        "significance",
        "significant",
        "statistically_significant",
    }
)


def _normalized_key(key: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", key.strip().lower())).strip("_")


def _forbidden_field(key: str) -> bool:
    normalized = _normalized_key(key)
    tokens = tuple(part for part in normalized.split("_") if part)
    if "formal" in tokens:
        return True
    if normalized in _INFERENTIAL_EXACT_KEYS:
        return True
    if "significance" in tokens or "significant" in tokens:
        return True
    if normalized.startswith("passed_") or normalized.endswith("_passed"):
        return True
    return len(tokens) >= 2 and tokens[-2:] == ("p", "value")


def _assert_no_forbidden_fields(
    value: object,
    *,
    path: str,
    allow_root_formal_marker: bool,
    at_root: bool = True,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            child_path = f"{path}.{key}"
            root_formal_marker = (
                at_root and allow_root_formal_marker and key == "formal_claim_eligible"
            )
            if root_formal_marker:
                if child is not False:
                    raise ValueError(f"{child_path} must be false")
            elif _forbidden_field(key):
                raise ValueError(f"{child_path} is a forbidden inferential/formal field")
            _assert_no_forbidden_fields(
                child,
                path=child_path,
                allow_root_formal_marker=False,
                at_root=False,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_forbidden_fields(
                child,
                path=f"{path}[{index}]",
                allow_root_formal_marker=False,
                at_root=False,
            )


def assert_exploratory_payload_has_no_inferential_fields(
    payload: Mapping[str, object],
) -> None:
    """Reject inferential/formal fields recursively.

    The sole exception is the root-level marker
    ``formal_claim_eligible: false``.  This check covers both obvious names
    (``p-value``, ``significant``) and compound decision keys ending in
    ``_passed``.
    """

    _mapping(payload, name="payload")
    _assert_no_forbidden_fields(
        payload,
        path="payload",
        allow_root_formal_marker=True,
    )


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
    return value


def _required(mapping: Mapping[str, object], key: str, *, name: str) -> object:
    if key not in mapping:
        raise ValueError(f"{name} is missing required field {key!r}")
    return mapping[key]


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _bootstrap_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("bootstrap_seed must be an integer")
    if not 0 <= value <= _MAX_TORCH_SEED:
        raise ValueError(f"bootstrap_seed must lie in [0, {_MAX_TORCH_SEED}]")
    return value


def _confidence(value: object) -> float:
    result = _finite(value, name="confidence_level")
    if not 0.0 < result < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    return result


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _seed(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value not in FIXED_THREE_EXPLORATORY_SEEDS:
        raise ValueError(
            f"{name} must be one of the fixed exploratory seeds {FIXED_THREE_EXPLORATORY_SEEDS!r}"
        )
    return value


def _seed_records(
    value: object,
) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        raise TypeError("seed_records must be an ordered sequence of mappings")
    records = tuple(
        _mapping(record, name=f"seed_records[{index}]") for index, record in enumerate(value)
    )
    if len(records) > len(FIXED_THREE_EXPLORATORY_SEEDS):
        raise ValueError("seed_records cannot contain more than the fixed three seeds")
    return records


def _validate_endpoint_values(
    value: object,
    *,
    seed: int,
) -> dict[str, tuple[float, float]]:
    endpoints = _mapping(value, name=f"seed {seed} endpoints")
    if set(endpoints) != set(_ENDPOINT_KEYS):
        raise ValueError(
            f"seed {seed} endpoints must be exactly the fixed endpoint set {_ENDPOINT_KEYS!r}"
        )
    normalized: dict[str, tuple[float, float]] = {}
    for endpoint in _ENDPOINTS:
        learners = _mapping(
            endpoints[endpoint.key],
            name=f"seed {seed} endpoints.{endpoint.key}",
        )
        if set(learners) != {"bt_mle", "prorm_plus"}:
            raise ValueError(
                f"seed {seed} endpoints.{endpoint.key} must contain exactly "
                "'bt_mle' and 'prorm_plus'"
            )
        bt = _finite(
            learners["bt_mle"],
            name=f"seed {seed} endpoints.{endpoint.key}.bt_mle",
        )
        prorm_plus = _finite(
            learners["prorm_plus"],
            name=f"seed {seed} endpoints.{endpoint.key}.prorm_plus",
        )
        normalized[endpoint.key] = (bt, prorm_plus)
    return normalized


def _identity_from_record(
    record: Mapping[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    return {
        "phase2_design_sha256": _digest(
            _required(record, "phase2_design_sha256", name=f"seed {seed}"),
            name=f"seed {seed} phase2_design_sha256",
        ),
        "phase2_runtime_contract_sha256": _digest(
            _required(
                record,
                "phase2_runtime_contract_sha256",
                name=f"seed {seed}",
            ),
            name=f"seed {seed} phase2_runtime_contract_sha256",
        ),
        "beta_source_aggregate_sha256": _digest(
            _required(
                record,
                "beta_source_aggregate_sha256",
                name=f"seed {seed}",
            ),
            name=f"seed {seed} beta_source_aggregate_sha256",
        ),
        "frozen_global_beta": _finite(
            _required(record, "frozen_global_beta", name=f"seed {seed}"),
            name=f"seed {seed} frozen_global_beta",
            positive=True,
        ),
    }


def _empty_identity() -> dict[str, None]:
    return {key: None for key in _IDENTITY_KEYS}


def _endpoint_definitions() -> dict[str, dict[str, str]]:
    return {endpoint.key: endpoint.to_dict() for endpoint in _ENDPOINTS}


def _bootstrap_contract(
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, object]:
    return {
        "method": "paired_seed_percentile_bootstrap",
        "unit": "seed",
        "resamples": resamples,
        "seed": seed,
        "confidence_level": confidence_level,
        "interpretation": "descriptive_only",
    }


def _effect_summary(
    *,
    endpoint: _EndpointDefinition,
    records: Sequence[tuple[int, float, float]],
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence_level: float,
) -> dict[str, object]:
    per_seed: list[dict[str, object]] = []
    differences: list[float] = []
    for seed, bt, prorm_plus in records:
        difference = prorm_plus - bt
        differences.append(difference)
        per_seed.append(
            {
                "seed": seed,
                "bt_mle": bt,
                "prorm_plus": prorm_plus,
                "prorm_plus_minus_bt": difference,
            }
        )

    interval = paired_bootstrap_ci(
        differences,
        bootstrap_seed=bootstrap_seed,
        num_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
    )
    return {
        "metric": endpoint.metric,
        "direction": endpoint.direction,
        "favorable_prorm_plus_minus_bt_sign": endpoint.favorable_sign,
        "contrast_orientation": CONTRAST_ORIENTATION,
        "n": len(differences),
        "per_seed": per_seed,
        "mean": math.fsum(differences) / len(differences),
        "sd_ddof1": stdev(differences),
        "min": min(differences),
        "median": median(differences),
        "max": max(differences),
        "paired_seed_bootstrap_descriptive_interval": interval.to_dict(),
    }


def build_fixed_three_exploratory_aggregate(
    seed_records: Sequence[Mapping[str, object]],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, object]:
    """Build the fixed-three exploratory descriptive aggregate.

    Each normalized record has these required fields:

    ``seed``, ``admissible``, ``phase2_design_sha256``,
    ``phase2_runtime_contract_sha256``, ``beta_source_aggregate_sha256``, and
    ``frozen_global_beta``.  An admissible record must additionally contain
    ``endpoints`` with exactly the five locked endpoint keys: held-out local
    regret, finite-policy utility, operational-oracle pairwise cross-entropy,
    oracle-probability MAE, and pairwise order accuracy.  Each endpoint maps
    ``bt_mle`` and ``prorm_plus`` to finite scalars.

    Records must occur in the fixed seed order.  A missing or inadmissible seed
    produces an auditable withheld payload with no ``effect_summaries`` key.
    Identity disagreement is always an error, including for a partial wave.
    """

    records = _seed_records(seed_records)
    checked_bootstrap_seed = _bootstrap_seed(bootstrap_seed)
    checked_resamples = _positive_integer(
        bootstrap_resamples,
        name="bootstrap_resamples",
    )
    checked_confidence = _confidence(confidence_level)

    observed: list[int] = []
    inadmissible: list[int] = []
    shared_identity: dict[str, object] | None = None
    endpoint_rows: dict[str, list[tuple[int, float, float]]] = {
        endpoint: [] for endpoint in _ENDPOINT_KEYS
    }

    for index, record in enumerate(records):
        _assert_no_forbidden_fields(
            record,
            path=f"seed_records[{index}]",
            allow_root_formal_marker=False,
        )
        seed = _seed(
            _required(record, "seed", name=f"seed_records[{index}]"),
            name=f"seed_records[{index}].seed",
        )
        observed.append(seed)
        identity = _identity_from_record(record, seed=seed)
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            differing = [key for key in _IDENTITY_KEYS if identity[key] != shared_identity[key]]
            raise ValueError(
                f"seed {seed} does not share the fixed beta/design/runtime identity: {differing!r}"
            )

        admissible = _required(record, "admissible", name=f"seed {seed}")
        if not isinstance(admissible, bool):
            raise TypeError(f"seed {seed} admissible must be a boolean")
        if not admissible:
            inadmissible.append(seed)
            continue

        endpoints = _validate_endpoint_values(
            _required(record, "endpoints", name=f"seed {seed}"),
            seed=seed,
        )
        for endpoint, (bt, prorm_plus) in endpoints.items():
            endpoint_rows[endpoint].append((seed, bt, prorm_plus))

    expected_observed_order = tuple(
        seed for seed in FIXED_THREE_EXPLORATORY_SEEDS if seed in set(observed)
    )
    if tuple(observed) != expected_observed_order:
        raise ValueError(
            "seed_records must be unique and follow the exact fixed exploratory seed order"
        )

    missing = [seed for seed in FIXED_THREE_EXPLORATORY_SEEDS if seed not in set(observed)]
    complete_and_admissible = not missing and not inadmissible
    payload: dict[str, object] = {
        "schema_version": FIXED_THREE_EXPLORATORY_SCHEMA,
        "analysis_role": "fixed_three_exploratory_descriptive_only",
        "formal_claim_eligible": False,
        "contrast_orientation": CONTRAST_ORIENTATION,
        "seeds": list(FIXED_THREE_EXPLORATORY_SEEDS),
        "observed_seeds": observed,
        "missing_seeds": missing,
        "inadmissible_seeds": inadmissible,
        "aggregation_state": (
            "complete_descriptive_aggregate"
            if complete_and_admissible
            else "effect_summaries_withheld"
        ),
        "identity": shared_identity if shared_identity is not None else _empty_identity(),
        "endpoint_definitions": _endpoint_definitions(),
        "bootstrap": _bootstrap_contract(
            seed=checked_bootstrap_seed,
            resamples=checked_resamples,
            confidence_level=checked_confidence,
        ),
    }
    if complete_and_admissible:
        payload["effect_summaries"] = {
            endpoint.key: _effect_summary(
                endpoint=endpoint,
                records=endpoint_rows[endpoint.key],
                bootstrap_seed=checked_bootstrap_seed,
                bootstrap_resamples=checked_resamples,
                confidence_level=checked_confidence,
            )
            for endpoint in _ENDPOINTS
        }

    assert_exploratory_payload_has_no_inferential_fields(payload)
    return payload


def validate_fixed_three_exploratory_aggregate(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate a serialized aggregate and return a shallow normalized copy."""

    value = _mapping(payload, name="payload")
    assert_exploratory_payload_has_no_inferential_fields(value)
    required_root_keys = {
        "schema_version",
        "analysis_role",
        "formal_claim_eligible",
        "contrast_orientation",
        "seeds",
        "observed_seeds",
        "missing_seeds",
        "inadmissible_seeds",
        "aggregation_state",
        "identity",
        "endpoint_definitions",
        "bootstrap",
    }
    allowed_root_keys = required_root_keys | {"effect_summaries"}
    if not required_root_keys.issubset(value) or not set(value).issubset(allowed_root_keys):
        raise ValueError("payload root fields are invalid")
    if value.get("schema_version") != FIXED_THREE_EXPLORATORY_SCHEMA:
        raise ValueError("payload schema_version is invalid")
    if value.get("analysis_role") != "fixed_three_exploratory_descriptive_only":
        raise ValueError("payload analysis_role is invalid")
    if value.get("formal_claim_eligible") is not False:
        raise ValueError("payload formal_claim_eligible must be false")
    if value.get("contrast_orientation") != CONTRAST_ORIENTATION:
        raise ValueError("payload contrast_orientation is invalid")
    if value.get("seeds") != list(FIXED_THREE_EXPLORATORY_SEEDS):
        raise ValueError("payload seeds do not match the exact fixed-three order")

    observed_value = value.get("observed_seeds")
    if not isinstance(observed_value, list):
        raise TypeError("payload observed_seeds must be a list")
    observed = tuple(_seed(seed, name="payload observed seed") for seed in observed_value)
    expected_order = tuple(seed for seed in FIXED_THREE_EXPLORATORY_SEEDS if seed in set(observed))
    if observed != expected_order:
        raise ValueError("payload observed_seeds are not unique and in fixed order")

    missing = [seed for seed in FIXED_THREE_EXPLORATORY_SEEDS if seed not in set(observed)]
    if value.get("missing_seeds") != missing:
        raise ValueError("payload missing_seeds is inconsistent with observed_seeds")
    inadmissible_value = value.get("inadmissible_seeds")
    if not isinstance(inadmissible_value, list):
        raise TypeError("payload inadmissible_seeds must be a list")
    inadmissible = tuple(
        _seed(seed, name="payload inadmissible seed") for seed in inadmissible_value
    )
    if inadmissible != tuple(
        seed for seed in FIXED_THREE_EXPLORATORY_SEEDS if seed in set(inadmissible)
    ) or not set(inadmissible).issubset(observed):
        raise ValueError("payload inadmissible_seeds must be an ordered observed subset")

    complete = not missing and not inadmissible
    expected_state = "complete_descriptive_aggregate" if complete else "effect_summaries_withheld"
    if value.get("aggregation_state") != expected_state:
        raise ValueError("payload aggregation_state is inconsistent")
    if complete != ("effect_summaries" in value):
        raise ValueError("effect_summaries must exist exactly when all three seeds are admissible")

    identity = _mapping(value.get("identity"), name="payload identity")
    if set(identity) != set(_IDENTITY_KEYS):
        raise ValueError("payload identity fields are invalid")
    if observed:
        _digest(
            identity["phase2_design_sha256"],
            name="payload identity.phase2_design_sha256",
        )
        _digest(
            identity["phase2_runtime_contract_sha256"],
            name="payload identity.phase2_runtime_contract_sha256",
        )
        _digest(
            identity["beta_source_aggregate_sha256"],
            name="payload identity.beta_source_aggregate_sha256",
        )
        _finite(
            identity["frozen_global_beta"],
            name="payload identity.frozen_global_beta",
            positive=True,
        )
    elif dict(identity) != _empty_identity():
        raise ValueError("an empty observed wave must have a null identity")

    if value.get("endpoint_definitions") != _endpoint_definitions():
        raise ValueError("payload endpoint definitions do not match the fixed contract")

    bootstrap = _mapping(value.get("bootstrap"), name="payload bootstrap")
    expected_bootstrap_keys = {
        "method",
        "unit",
        "resamples",
        "seed",
        "confidence_level",
        "interpretation",
    }
    if set(bootstrap) != expected_bootstrap_keys:
        raise ValueError("payload bootstrap fields are invalid")
    if (
        bootstrap["method"] != "paired_seed_percentile_bootstrap"
        or bootstrap["unit"] != "seed"
        or bootstrap["interpretation"] != "descriptive_only"
    ):
        raise ValueError("payload bootstrap contract is invalid")
    checked_resamples = _positive_integer(
        bootstrap["resamples"],
        name="payload bootstrap.resamples",
    )
    checked_bootstrap_seed = _bootstrap_seed(bootstrap["seed"])
    checked_confidence = _confidence(bootstrap["confidence_level"])

    if complete:
        summaries = _mapping(value["effect_summaries"], name="payload effect_summaries")
        if set(summaries) != set(_ENDPOINT_KEYS):
            raise ValueError("payload effect_summaries has the wrong endpoint set")
        for endpoint in _ENDPOINTS:
            _validate_effect_summary(
                summaries[endpoint.key],
                endpoint=endpoint,
                bootstrap_seed=checked_bootstrap_seed,
                bootstrap_resamples=checked_resamples,
                confidence_level=checked_confidence,
            )
    return dict(value)


def _validate_effect_summary(
    value: object,
    *,
    endpoint: _EndpointDefinition,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence_level: float,
) -> None:
    summary = _mapping(value, name=f"effect_summaries.{endpoint.key}")
    expected_keys = {
        "metric",
        "direction",
        "favorable_prorm_plus_minus_bt_sign",
        "contrast_orientation",
        "n",
        "per_seed",
        "mean",
        "sd_ddof1",
        "min",
        "median",
        "max",
        "paired_seed_bootstrap_descriptive_interval",
    }
    if set(summary) != expected_keys:
        raise ValueError(f"effect_summaries.{endpoint.key} fields are invalid")
    if (
        summary["metric"] != endpoint.metric
        or summary["direction"] != endpoint.direction
        or summary["favorable_prorm_plus_minus_bt_sign"] != endpoint.favorable_sign
        or summary["contrast_orientation"] != CONTRAST_ORIENTATION
        or summary["n"] != len(FIXED_THREE_EXPLORATORY_SEEDS)
    ):
        raise ValueError(f"effect_summaries.{endpoint.key} contract is invalid")

    per_seed_value = summary["per_seed"]
    if not isinstance(per_seed_value, list) or len(per_seed_value) != len(
        FIXED_THREE_EXPLORATORY_SEEDS
    ):
        raise ValueError(f"effect_summaries.{endpoint.key}.per_seed must contain three rows")
    differences: list[float] = []
    for expected_seed, raw_row in zip(
        FIXED_THREE_EXPLORATORY_SEEDS,
        per_seed_value,
        strict=True,
    ):
        row = _mapping(raw_row, name=f"effect_summaries.{endpoint.key}.per_seed")
        if set(row) != {
            "seed",
            "bt_mle",
            "prorm_plus",
            "prorm_plus_minus_bt",
        }:
            raise ValueError(f"effect_summaries.{endpoint.key}.per_seed fields are invalid")
        if row["seed"] != expected_seed:
            raise ValueError(f"effect_summaries.{endpoint.key}.per_seed order is invalid")
        bt = _finite(row["bt_mle"], name=f"{endpoint.key} bt_mle")
        prorm_plus = _finite(row["prorm_plus"], name=f"{endpoint.key} prorm_plus")
        difference = _finite(
            row["prorm_plus_minus_bt"],
            name=f"{endpoint.key} prorm_plus_minus_bt",
        )
        if difference != prorm_plus - bt:
            raise ValueError(f"{endpoint.key} contrast is not ProRM+ minus BT")
        differences.append(difference)

    expected_statistics = {
        "mean": math.fsum(differences) / len(differences),
        "sd_ddof1": stdev(differences),
        "min": min(differences),
        "median": median(differences),
        "max": max(differences),
    }
    for key, expected in expected_statistics.items():
        actual = _finite(summary[key], name=f"effect_summaries.{endpoint.key}.{key}")
        if actual != expected:
            raise ValueError(f"effect_summaries.{endpoint.key}.{key} is inconsistent")

    interval = _mapping(
        summary["paired_seed_bootstrap_descriptive_interval"],
        name=f"effect_summaries.{endpoint.key}.bootstrap interval",
    )
    expected_interval = paired_bootstrap_ci(
        differences,
        bootstrap_seed=bootstrap_seed,
        num_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
    ).to_dict()
    if dict(interval) != expected_interval:
        raise ValueError(f"effect_summaries.{endpoint.key} bootstrap interval is inconsistent")


__all__ = [
    "CONTRAST_ORIENTATION",
    "FIXED_THREE_EXPLORATORY_SCHEMA",
    "FIXED_THREE_EXPLORATORY_SEEDS",
    "normalize_budgeted_end_to_end_seed_result",
    "assert_exploratory_payload_has_no_inferential_fields",
    "build_fixed_three_exploratory_aggregate",
    "validate_fixed_three_exploratory_aggregate",
]
