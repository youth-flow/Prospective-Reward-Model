"""Strict target-free aggregation for the two-stage Phase-2 pilot.

The calibration pilot and the frozen-beta safety pilot use the same three
permanently excluded seeds, but answer different engineering questions:

* ``calibration`` computes one train-only beta candidate per seed and exposes
  their maximum as the first global-beta grid point;
* ``freeze`` reruns all seeds and all arms at one identity-bound beta and
  accepts that scalar only when every pre-oracle KL/tail/length gate passes.

Neither stage is efficacy evidence.  This module therefore accepts only the
target-free pilot schemas, verifies the diagnostic sidecars byte-for-byte, and
rejects any held-out/final-oracle payload before producing a selection record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path, PurePosixPath

from .paths import relative_posix_reference
from .phase2_config import (
    PHASE2_PILOT_SEEDS,
    phase2_design_identity,
    validate_phase2_config,
)
from .phase2_rollout import (
    KL_HISTORY_SOURCE,
    KL_ORIENTATION,
    PHASE2_ARM_ORDER,
    PHASE2_PILOT_DIAGNOSTIC_SCHEMA,
    PHASE2_PILOT_RESULT_SCHEMA,
    Phase2Design,
)
from .repro import atomic_write_json

PHASE2_PILOT_AGGREGATE_SCHEMA = "common-beta-pilot-selection-aggregate/v1"

_HEX = frozenset("0123456789abcdef")
_PILOT_RESULT_COMMON_KEYS = frozenset(
    {
        "schema_version",
        "design_stage",
        "pilot_phase",
        "formal_eligibility",
        "evidence_role",
        "per_seed_supports_formal_claim",
        "source_config_hash",
        "phase2_design_sha256",
        "phase2_runtime_contract",
        "phase2_runtime_contract_sha256",
        "seed",
        "artifact_dir",
        "diagnostics_jsonl",
        "artifact_metadata_sha256",
        "run_manifest",
        "run_manifest_sha256",
        "environment_identity",
        "current_process_identity",
        "train_oracle_rescore",
        "head_training",
        "deployment_hashes",
        "measured_kl_safety",
        "pre_oracle_safety_gate",
        "pilot_kl_safety_gate",
        "arms",
        "information_boundary",
        "common_random_numbers",
        "memory_schedule",
        "policy_and_oracle_co_resident",
        "learner_specific_line_search",
        "diagnostics_sha256",
    }
)
_PILOT_BETA_EVIDENCE_KEYS = {
    "calibration": "train_only_global_beta_calibration_candidate",
    "freeze": "pilot_fixed_global_beta_rehearsal",
}
_DIAGNOSTIC_ROW_KEYS = frozenset(
    {
        "schema_version",
        "pilot_phase",
        "arm",
        "beta_common",
        "beta_role",
        "prompt_id",
        "candidate_index",
        "response_token_count",
        "terminated_by_eos",
        "reached_max_length",
        "prompt_rollout_seed",
        "kl_orientation",
        "kl_history_source",
        "on_policy_kl_pi_updated_to_pi0",
        "contains_prompt_text",
        "contains_response_text",
        "contains_token_ids",
        "contains_oracle_outcome",
    }
)
_FORBIDDEN_PILOT_KEYS = frozenset(
    {
        "prompt",
        "response",
        "token_ids",
        "response_mask",
        "head_weights",
        "transformed_oracle_reward",
        "target_utility",
        "mean_target_reward",
        "mean_target_utility",
        "heldout_fixed_beta",
        "heldout_fixed_beta_sha256",
        "rollouts_jsonl",
        "rollouts_sha256",
        "final_oracle_rewards",
    }
)
_SAFETY_THRESHOLDS = {
    "mean_policy_to_reference_kl_cap": 0.02,
    "prompt_mean_p95_kl_cap": 0.02,
    "prompt_mean_p99_kl_cap": 0.05,
    "prompt_mean_maximum_kl_cap": 0.10,
    "per_sequence_maximum_kl_cap": 0.20,
    "reached_max_length_rate_cap": 0.05,
}
_SAFETY_METRICS = tuple(key.removesuffix("_cap") for key in _SAFETY_THRESHOLDS)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _exact_keys(
    value: object,
    *,
    name: str,
    expected: set[str] | frozenset[str],
) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    if set(result) != set(expected):
        raise ValueError(
            f"{name} keys differ from the target-free schema; "
            f"missing={sorted(set(expected) - set(result))!r}, "
            f"unknown={sorted(set(result) - set(expected))!r}"
        )
    return result


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _close(left: float, right: float, *, tolerance: float = 2.0e-6) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _read_regular(path: Path, *, name: str, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a regular file: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{name} exceeds {max_bytes} bytes: {path}")
    return path.read_bytes()


def _strict_json_bytes(raw: bytes, *, path: Path) -> Mapping[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{path} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not strict UTF-8 JSON: {error}") from error
    return _mapping(value, name=str(path))


def _strict_jsonl_bytes(raw: bytes, *, path: Path) -> list[Mapping[str, object]]:
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{path} must be non-empty newline-terminated JSONL")
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"{path}:{line_number} is blank")
        rows.append(_strict_json_bytes(line, path=Path(f"{path}:{line_number}")))
    return rows


def _reject_forbidden_keys(value: object, *, path: str = "pilot_result") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_PILOT_KEYS:
                raise ValueError(f"{path} contains forbidden pilot field {key!r}")
            _reject_forbidden_keys(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, path=f"{path}[{index}]")


def _resolved_sidecar(result_path: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{result_path}:diagnostics_jsonl must be a non-empty string")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or "\\" in reference or ".." in pure.parts or len(pure.parts) != 1:
        raise ValueError(f"{result_path}:diagnostics_jsonl must name a sibling JSONL file")
    sidecar = result_path.parent / reference
    if sidecar.resolve().parent != result_path.parent.resolve():
        raise ValueError(f"{result_path}:diagnostics_jsonl escapes its result directory")
    return sidecar


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile input must not be empty")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sidecar_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    pilot_phase: str,
    beta_common: float,
    prompts: int,
    candidates: int,
    max_response_tokens: int,
    name: str,
) -> dict[str, dict[str, float]]:
    expected_count = len(PHASE2_ARM_ORDER) * prompts * candidates
    if len(rows) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} diagnostic rows")
    expected_role = (
        "seed_calibration_candidate"
        if pilot_phase == "calibration"
        else "frozen_global_beta_candidate"
    )
    summaries: dict[str, dict[str, float]] = {}
    reference_coordinates: list[tuple[object, int, int]] | None = None
    offset = 0
    for arm_name in PHASE2_ARM_ORDER:
        arm_rows = rows[offset : offset + prompts * candidates]
        offset += prompts * candidates
        coordinates: list[tuple[object, int, int]] = []
        kl_values: list[float] = []
        prompt_means: list[float] = []
        lengths: list[int] = []
        reached_count = 0
        eos_count = 0
        prompt_ids: set[object] = set()
        for prompt_index in range(prompts):
            group = arm_rows[prompt_index * candidates : (prompt_index + 1) * candidates]
            prompt_id = group[0].get("prompt_id")
            if (
                isinstance(prompt_id, bool)
                or not isinstance(prompt_id, (str, int))
                or (isinstance(prompt_id, str) and not prompt_id)
            ):
                raise ValueError(f"{name}:{arm_name} has an invalid prompt_id")
            if prompt_id in prompt_ids:
                raise ValueError(f"{name}:{arm_name} repeats prompt_id {prompt_id!r}")
            prompt_ids.add(prompt_id)
            group_kl: list[float] = []
            for candidate_index, raw_row in enumerate(group):
                row = _exact_keys(
                    raw_row,
                    name=f"{name}:{arm_name}:{prompt_index}:{candidate_index}",
                    expected=_DIAGNOSTIC_ROW_KEYS,
                )
                if (
                    row["schema_version"] != PHASE2_PILOT_DIAGNOSTIC_SCHEMA
                    or row["pilot_phase"] != pilot_phase
                    or row["arm"] != arm_name
                    or row["beta_role"] != expected_role
                    or row["candidate_index"] != candidate_index
                    or row["prompt_id"] != prompt_id
                    or row["kl_orientation"] != KL_ORIENTATION
                    or row["kl_history_source"] != KL_HISTORY_SOURCE
                ):
                    raise ValueError(f"{name} has a diagnostic row identity mismatch")
                if not _close(
                    _finite(row["beta_common"], name=f"{name}:row.beta_common"),
                    beta_common,
                    tolerance=1.0e-12,
                ):
                    raise ValueError(f"{name} diagnostic rows disagree on beta_common")
                for flag in (
                    "contains_prompt_text",
                    "contains_response_text",
                    "contains_token_ids",
                    "contains_oracle_outcome",
                ):
                    if row[flag] is not False:
                        raise ValueError(f"{name} leaked forbidden pilot content via {flag}")
                length = _integer(
                    row["response_token_count"],
                    name=f"{name}:response_token_count",
                )
                if length > max_response_tokens:
                    raise ValueError(f"{name} response length exceeds the frozen horizon")
                eos = row["terminated_by_eos"]
                reached = row["reached_max_length"]
                if not isinstance(eos, bool) or not isinstance(reached, bool) or (eos and reached):
                    raise ValueError(f"{name} has invalid termination flags")
                if reached and length != max_response_tokens:
                    raise ValueError(f"{name} max-length flag disagrees with token count")
                rollout_seed = _integer(
                    row["prompt_rollout_seed"],
                    name=f"{name}:prompt_rollout_seed",
                )
                kl = _finite(
                    row["on_policy_kl_pi_updated_to_pi0"],
                    name=f"{name}:on_policy_kl",
                )
                if arm_name == "zero_b" and kl != 0.0:
                    raise ValueError(f"{name} zero_b KL must be exactly zero")
                coordinates.append((prompt_id, candidate_index, rollout_seed))
                group_kl.append(kl)
                kl_values.append(kl)
                lengths.append(length)
                reached_count += int(reached)
                eos_count += int(eos)
            prompt_means.append(sum(group_kl) / candidates)
        if reference_coordinates is None:
            reference_coordinates = coordinates
        elif coordinates != reference_coordinates:
            raise ValueError(f"{name} violates common-random prompt/candidate alignment")
        summaries[arm_name] = {
            "num_trajectories": float(len(arm_rows)),
            "mean_policy_to_reference_kl": sum(kl_values) / len(kl_values),
            "prompt_mean_p50_kl": _linear_quantile(prompt_means, 0.50),
            "prompt_mean_p90_kl": _linear_quantile(prompt_means, 0.90),
            "prompt_mean_p95_kl": _linear_quantile(prompt_means, 0.95),
            "prompt_mean_p99_kl": _linear_quantile(prompt_means, 0.99),
            "prompt_mean_maximum_kl": max(prompt_means),
            "per_sequence_maximum_kl": max(kl_values),
            "reached_max_length_rate": reached_count / len(arm_rows),
            "terminated_by_eos_rate": eos_count / len(arm_rows),
            "mean_response_token_count": sum(lengths) / len(lengths),
            "minimum_response_token_count": float(min(lengths)),
            "maximum_response_token_count": float(max(lengths)),
        }
    return summaries


def _validate_arm_summaries(
    value: object,
    *,
    recomputed: Mapping[str, Mapping[str, float]],
    beta_common: float,
    prompts: int,
    candidates: int,
    formal_gate_applied: bool,
    name: str,
) -> None:
    arms = _mapping(value, name=name)
    if set(arms) != set(PHASE2_ARM_ORDER):
        raise ValueError(f"{name} must contain exactly the frozen arms")
    for arm_name in PHASE2_ARM_ORDER:
        arm = _exact_keys(
            arms[arm_name],
            name=f"{name}.{arm_name}",
            expected={
                "deployment_hashes",
                "rollout_length",
                "mean_on_policy_kl_pi_updated_to_pi0",
                "on_policy_kl_tail",
            },
        )
        deployment = _exact_keys(
            arm["deployment_hashes"],
            name=f"{name}.{arm_name}.deployment_hashes",
            expected={
                "beta_common",
                "displacement_sha256",
                "direction_evidence_sha256",
                "common_beta_evidence_sha256",
            },
        )
        if not _close(
            _finite(deployment["beta_common"], name=f"{name}:deployment.beta_common"),
            beta_common,
            tolerance=1.0e-12,
        ):
            raise ValueError(f"{name} deployment beta differs from beta_common")
        _digest(
            deployment["displacement_sha256"],
            name=f"{name}.{arm_name}.displacement_sha256",
        )
        for field in ("direction_evidence_sha256", "common_beta_evidence_sha256"):
            digest = deployment[field]
            if digest is not None:
                _digest(digest, name=f"{name}.{arm_name}.{field}")
        observed = recomputed[arm_name]
        recorded_mean = _finite(
            arm["mean_on_policy_kl_pi_updated_to_pi0"],
            name=f"{name}.{arm_name}.mean_kl",
        )
        if not _close(recorded_mean, observed["mean_policy_to_reference_kl"]):
            raise ValueError(f"{name}.{arm_name} mean KL disagrees with its sidecar")
        length = _exact_keys(
            arm["rollout_length"],
            name=f"{name}.{arm_name}.rollout_length",
            expected={
                "num_trajectories",
                "terminated_by_eos_count",
                "terminated_by_eos_rate",
                "reached_max_length_count",
                "reached_max_length_rate",
                "response_token_count",
            },
        )
        expected_trajectories = prompts * candidates
        if length["num_trajectories"] != expected_trajectories:
            raise ValueError(f"{name}.{arm_name} has the wrong rollout count")
        for field, summary_key in (
            ("terminated_by_eos_rate", "terminated_by_eos_rate"),
            ("reached_max_length_rate", "reached_max_length_rate"),
        ):
            if not _close(
                _finite(length[field], name=f"{name}.{arm_name}.{field}"),
                observed[summary_key],
            ):
                raise ValueError(f"{name}.{arm_name}.{field} disagrees with its sidecar")
        response = _exact_keys(
            length["response_token_count"],
            name=f"{name}.{arm_name}.response_token_count",
            expected={"mean", "minimum", "maximum"},
        )
        for field, summary_key in (
            ("mean", "mean_response_token_count"),
            ("minimum", "minimum_response_token_count"),
            ("maximum", "maximum_response_token_count"),
        ):
            if not _close(
                _finite(response[field], name=f"{name}.{arm_name}.response.{field}"),
                observed[summary_key],
            ):
                raise ValueError(f"{name}.{arm_name} response-length summary mismatch")
        tail = _exact_keys(
            arm["on_policy_kl_tail"],
            name=f"{name}.{arm_name}.on_policy_kl_tail",
            expected={
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
        if (
            tail["schema_version"] != "on-policy-kl-tail-summary/v1"
            or tail["unit"] != "prompt_mean_over_candidates"
            or tail["num_prompts"] != prompts
            or tail["candidates_per_prompt"] != candidates
            or tail["pilot_selection_role"] != "locality_tail_measurement"
            or tail["formal_gate_applied"] is not formal_gate_applied
        ):
            raise ValueError(f"{name}.{arm_name} has an invalid KL-tail contract")
        for field, summary_key in (
            ("mean", "mean_policy_to_reference_kl"),
            ("p50", "prompt_mean_p50_kl"),
            ("p90", "prompt_mean_p90_kl"),
            ("p95", "prompt_mean_p95_kl"),
            ("p99", "prompt_mean_p99_kl"),
            ("maximum", "prompt_mean_maximum_kl"),
            ("per_sequence_maximum", "per_sequence_maximum_kl"),
        ):
            if not _close(
                _finite(tail[field], name=f"{name}.{arm_name}.tail.{field}"),
                observed[summary_key],
            ):
                raise ValueError(f"{name}.{arm_name} KL-tail summary mismatch")


def _expected_violations(
    observed_by_arm: Mapping[str, Mapping[str, float]],
) -> list[str]:
    return [
        f"{arm_name}:{metric}"
        for arm_name in PHASE2_ARM_ORDER
        for metric in _SAFETY_METRICS
        if observed_by_arm[arm_name][metric] > _SAFETY_THRESHOLDS[f"{metric}_cap"]
    ]


def _validate_pre_oracle_gate(
    value: object,
    *,
    pilot_phase: str,
    recomputed: Mapping[str, Mapping[str, float]],
    name: str,
) -> bool:
    gate = _exact_keys(
        value,
        name=name,
        expected={
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
        },
    )
    if (
        gate["schema_version"] != "phase2-pre-oracle-safety-gate/v1"
        or gate["design_stage"] != "pilot"
        or gate["pilot_phase"] != pilot_phase
        or gate["measure_only"] is not True
        or gate["formal_gate"] is not False
        or gate["beta_retuned"] is not False
        or gate["on_violation"] != "publish_target_free_diagnostics_without_final_oracle"
    ):
        raise ValueError(f"{name} is not a target-free pilot gate")
    thresholds = _mapping(gate["thresholds"], name=f"{name}.thresholds")
    if set(thresholds) != set(_SAFETY_THRESHOLDS):
        raise ValueError(f"{name} threshold fields differ from the preregistration")
    for field, expected in _SAFETY_THRESHOLDS.items():
        if not _close(
            _finite(thresholds[field], name=f"{name}.thresholds.{field}"),
            expected,
            tolerance=1.0e-12,
        ):
            raise ValueError(f"{name}.{field} differs from the preregistered threshold")
    observed = _mapping(gate["observed_by_arm"], name=f"{name}.observed_by_arm")
    if set(observed) != set(PHASE2_ARM_ORDER):
        raise ValueError(f"{name} must contain exactly the frozen observed arms")
    for arm_name in PHASE2_ARM_ORDER:
        arm = _mapping(observed[arm_name], name=f"{name}.{arm_name}")
        if set(arm) != set(_SAFETY_METRICS):
            raise ValueError(f"{name}.{arm_name} safety metrics are incomplete")
        for metric in _SAFETY_METRICS:
            if not _close(
                _finite(arm[metric], name=f"{name}.{arm_name}.{metric}"),
                recomputed[arm_name][metric],
            ):
                raise ValueError(f"{name}.{arm_name}.{metric} disagrees with the sidecar")
    violations = _expected_violations(recomputed)
    if gate["violations"] != violations:
        raise ValueError(f"{name} violations disagree with threshold arithmetic")
    passed = not violations
    if gate["passed"] is not passed:
        raise ValueError(f"{name}.passed disagrees with threshold arithmetic")
    return passed


def _validate_information_boundary(value: object, *, name: str) -> None:
    boundary = _mapping(value, name=name)
    required_false = {
        "new_rollout_prompts_used_for_calibration",
        "final_oracle_session_opened",
        "rollout_responses_oracle_scored",
        "heldout_evaluator_called",
        "oracle_outcomes_serialized",
        "prompt_or_response_text_serialized",
        "token_ids_or_response_masks_serialized",
        "source_artifact_heldout_targets_exposed_by_phase2_prepared_inputs",
    }
    missing = required_false - set(boundary)
    if missing or any(boundary[field] is not False for field in required_false):
        raise ValueError(f"{name} does not prove the target-free information boundary")


def _validate_beta_evidence(
    value: object,
    *,
    pilot_phase: str,
    design: Phase2Design,
    name: str,
) -> tuple[float, dict[str, object]]:
    if pilot_phase == "calibration":
        evidence = _exact_keys(
            value,
            name=name,
            expected={
                "schema_version",
                "rule",
                "candidate_beta",
                "frozen_global_beta",
                "oracle_natural_curvature",
                "target_oracle_quadratic_kl",
                "predicted_oracle_quadratic_kl",
                "calibration_split",
                "formal_beta_selected",
                "formal_selection_rule",
                "learner_specific_rescaling",
            },
        )
        if (
            evidence["schema_version"] != "global-beta-calibration-candidate/v1"
            or evidence["rule"] != design.common_beta_rule
            or evidence["frozen_global_beta"] is not None
            or evidence["calibration_split"] != "train_only"
            or evidence["formal_beta_selected"] is not False
            or evidence["learner_specific_rescaling"] is not False
        ):
            raise ValueError(f"{name} has an invalid calibration-candidate contract")
        beta = _finite(evidence["candidate_beta"], name=f"{name}.candidate_beta")
        if beta <= 0.0:
            raise ValueError(f"{name}.candidate_beta must be strictly positive")
        curvature = _finite(
            evidence["oracle_natural_curvature"],
            name=f"{name}.oracle_natural_curvature",
        )
        target = _finite(
            evidence["target_oracle_quadratic_kl"],
            name=f"{name}.target_oracle_quadratic_kl",
        )
        predicted = _finite(
            evidence["predicted_oracle_quadratic_kl"],
            name=f"{name}.predicted_oracle_quadratic_kl",
        )
    else:
        evidence = _exact_keys(
            value,
            name=name,
            expected={
                "schema_version",
                "rule",
                "beta_common",
                "frozen_global_beta",
                "beta_matches_frozen_global_beta",
                "beta_source_aggregate_sha256",
                "current_seed_oracle_natural_curvature",
                "reference_target_oracle_quadratic_kl",
                "predicted_current_seed_oracle_quadratic_kl",
                "current_seed_curvature_role",
                "beta_selected_from_current_seed_curvature",
                "frozen_in_phase2_design_identity",
                "learner_specific_rescaling",
                "post_evaluation_retuning",
            },
        )
        beta = _finite(evidence["beta_common"], name=f"{name}.beta_common")
        frozen = _finite(
            evidence["frozen_global_beta"],
            name=f"{name}.frozen_global_beta",
        )
        if (
            evidence["schema_version"] != "pilot-frozen-global-beta-rehearsal/v1"
            or evidence["rule"] != design.common_beta_rule
            or beta <= 0.0
            or beta != frozen
            or beta != design.frozen_global_beta
            or evidence["beta_matches_frozen_global_beta"] is not True
            or evidence["beta_source_aggregate_sha256"] != design.beta_source_aggregate_sha256
            or evidence["current_seed_curvature_role"] != "predicted_kl_diagnostic_only"
            or evidence["beta_selected_from_current_seed_curvature"] is not False
            or evidence["frozen_in_phase2_design_identity"] is not True
            or evidence["learner_specific_rescaling"] is not False
            or evidence["post_evaluation_retuning"] is not False
        ):
            raise ValueError(f"{name} has an invalid frozen-beta rehearsal contract")
        curvature = _finite(
            evidence["current_seed_oracle_natural_curvature"],
            name=f"{name}.current_seed_oracle_natural_curvature",
        )
        target = _finite(
            evidence["reference_target_oracle_quadratic_kl"],
            name=f"{name}.reference_target_oracle_quadratic_kl",
        )
        predicted = _finite(
            evidence["predicted_current_seed_oracle_quadratic_kl"],
            name=f"{name}.predicted_current_seed_oracle_quadratic_kl",
        )
    if (
        curvature <= 0.0
        or not _close(
            target,
            design.target_oracle_quadratic_kl,
            tolerance=1.0e-12,
        )
        or not _close(predicted, 0.5 * curvature / (beta * beta), tolerance=1.0e-10)
    ):
        raise ValueError(f"{name} has inconsistent curvature/beta/predicted-KL arithmetic")
    return beta, dict(evidence)


def _validate_environment(value: object, *, name: str) -> dict[str, object]:
    environment = dict(_mapping(value, name=name))
    if environment.get("formal") is not True:
        raise ValueError(f"{name} must be a formal clean HPC environment identity")
    for field in ("git_commit", "image_sha256", "hf_inventory_sha256"):
        raw = environment.get(field)
        if field == "git_commit":
            if (
                not isinstance(raw, str)
                or len(raw) not in {40, 64}
                or any(character not in _HEX for character in raw)
            ):
                raise ValueError(f"{name}.{field} is invalid")
        else:
            _digest(raw, name=f"{name}.{field}")
    if environment.get("account") != "sigroup":
        raise ValueError(f"{name}.account must equal 'sigroup'")
    gpu_models = environment.get("gpu_models")
    if (
        not isinstance(gpu_models, list)
        or len(gpu_models) != 1
        or not isinstance(gpu_models[0], str)
        or not gpu_models[0]
    ):
        raise ValueError(f"{name} must bind exactly one GPU model")
    return environment


def _load_seed(
    raw_path: str | os.PathLike[str],
    *,
    expected_design_sha256: str,
    expected_runtime: Mapping[str, object],
    expected_runtime_sha256: str,
    expected_source_config_hash: str,
    expected_pilot_phase: str,
    design: Phase2Design,
    prompts: int,
    candidates: int,
) -> dict[str, object]:
    path = Path(raw_path).resolve()
    raw = _read_regular(path, name="pilot result", max_bytes=64 * 1024 * 1024)
    result_sha = _sha256_bytes(raw)
    beta_key = _PILOT_BETA_EVIDENCE_KEYS[expected_pilot_phase]
    value = _exact_keys(
        _strict_json_bytes(raw, path=path),
        name=str(path),
        expected=set(_PILOT_RESULT_COMMON_KEYS) | {beta_key},
    )
    _reject_forbidden_keys(value)
    if (
        value["schema_version"] != PHASE2_PILOT_RESULT_SCHEMA
        or value["design_stage"] != "pilot"
        or value["pilot_phase"] != expected_pilot_phase
        or value["formal_eligibility"] is not False
        or value["per_seed_supports_formal_claim"] is not False
        or value["source_config_hash"] != expected_source_config_hash
        or value["phase2_design_sha256"] != expected_design_sha256
        or value["policy_and_oracle_co_resident"] is not False
        or value["learner_specific_line_search"] is not False
    ):
        raise ValueError(f"{path} is not a matching, formally ineligible pilot result")
    runtime = _mapping(value["phase2_runtime_contract"], name=f"{path}:runtime")
    runtime_sha = _digest(
        value["phase2_runtime_contract_sha256"],
        name=f"{path}:runtime_sha256",
    )
    if (
        dict(runtime) != dict(expected_runtime)
        or runtime_sha != expected_runtime_sha256
        or _canonical_sha256(runtime) != runtime_sha
    ):
        raise ValueError(f"{path} runtime contract differs from the overlay")
    seed = _integer(value["seed"], name=f"{path}:seed")
    for field in (
        "artifact_metadata_sha256",
        "run_manifest_sha256",
        "diagnostics_sha256",
    ):
        _digest(value[field], name=f"{path}:{field}")
    environment = _validate_environment(
        value["environment_identity"],
        name=f"{path}:environment_identity",
    )
    current = _validate_environment(
        value["current_process_identity"],
        name=f"{path}:current_process_identity",
    )
    if current != environment:
        raise ValueError(f"{path} process identity differs from the run identity")
    beta, beta_evidence = _validate_beta_evidence(
        value[beta_key],
        pilot_phase=expected_pilot_phase,
        design=design,
        name=f"{path}:{beta_key}",
    )
    sidecar = _resolved_sidecar(path, value["diagnostics_jsonl"])
    sidecar_raw = _read_regular(
        sidecar,
        name="pilot diagnostic sidecar",
        max_bytes=256 * 1024 * 1024,
    )
    sidecar_sha = _sha256_bytes(sidecar_raw)
    if sidecar_sha != value["diagnostics_sha256"]:
        raise ValueError(f"{path} diagnostic sidecar SHA256 mismatch")
    rows = _strict_jsonl_bytes(sidecar_raw, path=sidecar)
    recomputed = _sidecar_summary(
        rows,
        pilot_phase=expected_pilot_phase,
        beta_common=beta,
        prompts=prompts,
        candidates=candidates,
        max_response_tokens=design.max_response_tokens,
        name=str(sidecar),
    )
    _validate_arm_summaries(
        value["arms"],
        recomputed=recomputed,
        beta_common=beta,
        prompts=prompts,
        candidates=candidates,
        formal_gate_applied=False,
        name=f"{path}:arms",
    )
    safety_passed = _validate_pre_oracle_gate(
        value["pre_oracle_safety_gate"],
        pilot_phase=expected_pilot_phase,
        recomputed=recomputed,
        name=f"{path}:pre_oracle_safety_gate",
    )
    _validate_information_boundary(
        value["information_boundary"],
        name=f"{path}:information_boundary",
    )
    head = _mapping(value["head_training"], name=f"{path}:head_training")
    if (
        head.get("training_design_sha256") != expected_design_sha256
        or head.get("head_weights_serialized") is not False
        or head.get("old_phase1_comparison_heads_reused") is not False
        or head.get("test_data_accessed") is not False
    ):
        raise ValueError(f"{path} head-training boundary is invalid")
    return {
        "seed": seed,
        "beta_common": beta,
        "beta_evidence": beta_evidence,
        "safety_passed": safety_passed,
        "observed_by_arm": {arm: dict(recomputed[arm]) for arm in PHASE2_ARM_ORDER},
        "environment_identity": environment,
        "result_path": path,
        "result_sha256": result_sha,
        "sidecar_path": sidecar,
        "sidecar_sha256": sidecar_sha,
    }


def _load_source_aggregate(
    raw_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> tuple[Path, str, Mapping[str, object]]:
    path = Path(raw_path).resolve()
    raw = _read_regular(path, name="beta source aggregate", max_bytes=16 * 1024 * 1024)
    observed_sha = _sha256_bytes(raw)
    if observed_sha != expected_sha256:
        raise ValueError("beta source aggregate SHA256 differs from the design identity")
    value = _strict_json_bytes(raw, path=path)
    if value.get("schema_version") != PHASE2_PILOT_AGGREGATE_SCHEMA:
        raise ValueError("beta source aggregate has the wrong schema")
    return path, observed_sha, value


def _power_of_two_grid_index(value: float, base: float) -> int:
    if value <= 0.0 or base <= 0.0 or value < base:
        raise ValueError("frozen pilot beta is outside the preregistered beta*=2 grid")
    ratio = value / base
    exponent = round(math.log2(ratio))
    if exponent < 0 or not _close(ratio, 2.0**exponent, tolerance=1.0e-12):
        raise ValueError("frozen pilot beta is outside the preregistered beta*=2 grid")
    return exponent


def verify_beta_source_aggregate(
    overlay_config: Mapping[str, object],
    source_aggregate: str | os.PathLike[str] | None,
) -> dict[str, object] | None:
    """Verify the predecessor selection record bound into a freeze/formal design."""

    config = validate_phase2_config(overlay_config)
    design = Phase2Design.from_phase2_config(config)
    expected_sha = design.beta_source_aggregate_sha256
    if design.pilot_phase == "calibration":
        if source_aggregate is not None or expected_sha is not None:
            raise ValueError("pilot calibration must not consume a beta source aggregate")
        return None
    if source_aggregate is None or expected_sha is None:
        raise ValueError("this design requires its identity-bound beta source aggregate")
    path, observed_sha, value = _load_source_aggregate(
        source_aggregate,
        expected_sha256=expected_sha,
    )
    design_config = _mapping(config["design"], name="design")
    if (
        value.get("formal_eligibility") is not False
        or value.get("supports_formal_claim") is not False
        or value.get("evidence_role") != "target_free_design_selection_only"
        or value.get("source_config_hash") != design_config["source_config_hash"]
    ):
        raise ValueError(
            "beta source aggregate is not a target-free pilot artifact for the same base config"
        )
    selection = _mapping(value.get("selection"), name=f"{path}:selection")
    if design.pilot_phase == "freeze":
        horizon = _mapping(value.get("horizon"), name=f"{path}:horizon")
        if (
            horizon.get("all_seed_length_gates_passed") is not True
            or horizon.get("candidate_horizon_tokens") != design.max_response_tokens
            or horizon.get("horizon_grid_index") != design.horizon_grid_index
        ):
            raise ValueError(
                "pilot freeze requires an immediate beta parent at the same accepted horizon"
            )
        source_phase = value.get("pilot_phase")
        current_beta = float(design.frozen_global_beta)
        if source_phase == "calibration":
            base = _finite(
                selection.get("recommended_pilot_freeze_beta"),
                name=f"{path}:selection.recommended_pilot_freeze_beta",
            )
            if not _close(current_beta, base, tolerance=1.0e-12):
                raise ValueError(
                    "the initial freeze identity must use exactly the calibration "
                    "aggregate's recommended beta"
                )
            grid_index = 0
        elif source_phase == "freeze":
            previous_beta = _finite(
                selection.get("frozen_global_beta"),
                name=f"{path}:selection.frozen_global_beta",
            )
            previous_index = _integer(
                selection.get("beta_grid_index"),
                name=f"{path}:selection.beta_grid_index",
            )
            next_beta = _finite(
                selection.get("next_global_beta"),
                name=f"{path}:selection.next_global_beta",
            )
            if (
                selection.get("schema_version") != "pilot-freeze-selection/v1"
                or selection.get("selection_accepted") is not False
                or selection.get("accepted_for_confirmatory_identity") is not False
                or selection.get("all_length_gates_passed") is not True
                or selection.get("all_non_length_safety_gates_passed") is not False
                or selection.get("next_action") != "issue_new_pilot_freeze_identity_at_double_beta"
                or not _close(next_beta, 2.0 * previous_beta, tolerance=1.0e-12)
                or not _close(current_beta, next_beta, tolerance=1.0e-12)
            ):
                raise ValueError(
                    "a retry freeze identity must bind the immediately preceding "
                    "non-length safety failure and its exact doubled beta"
                )
            grid_index = previous_index + 1
            base = current_beta / (2.0**grid_index)
            _power_of_two_grid_index(current_beta, base)
        else:
            raise ValueError(
                "pilot freeze beta source must be calibration for grid index zero "
                "or the immediately preceding failed freeze"
            )
        return {
            "path": path,
            "sha256": observed_sha,
            "source_pilot_phase": source_phase,
            "base_beta": base,
            "beta_grid_index": grid_index,
        }
    if value.get("pilot_phase") != "freeze":
        raise ValueError("confirmatory beta must be sourced from a freeze aggregate")
    if selection.get("selection_accepted") is not True:
        raise ValueError("confirmatory beta source aggregate did not pass pilot safety")
    frozen = _finite(
        selection.get("frozen_global_beta"),
        name=f"{path}:selection.frozen_global_beta",
    )
    if frozen != design.frozen_global_beta:
        raise ValueError("confirmatory beta differs from the accepted freeze aggregate")
    return {
        "path": path,
        "sha256": observed_sha,
        "source_pilot_phase": "freeze",
        "accepted_beta": frozen,
    }


def verify_horizon_parent_aggregate(
    overlay_config: Mapping[str, object],
    parent_aggregate: str | os.PathLike[str] | None,
) -> dict[str, object] | None:
    """Verify the aggregate that authorized the current response horizon."""

    config = validate_phase2_config(overlay_config)
    design = Phase2Design.from_phase2_config(config)
    expected_sha = design.parent_pilot_aggregate_sha256
    if expected_sha is None:
        if parent_aggregate is not None:
            raise ValueError("the initial calibration horizon has no parent aggregate")
        return None
    if parent_aggregate is None:
        raise ValueError("this horizon requires its identity-bound parent pilot aggregate")
    path, observed_sha, value = _load_source_aggregate(
        parent_aggregate,
        expected_sha256=expected_sha,
    )
    horizon = _mapping(value.get("horizon"), name=f"{path}:horizon")
    source_tokens = _integer(
        horizon.get("candidate_horizon_tokens"),
        name=f"{path}:horizon.candidate_horizon_tokens",
        minimum=1,
    )
    source_index = _integer(
        horizon.get("horizon_grid_index"),
        name=f"{path}:horizon.horizon_grid_index",
    )
    if design.pilot_phase == "calibration":
        selection = _mapping(value.get("selection"), name=f"{path}:selection")
        if (
            design.horizon_grid_index < 1
            or value.get("pilot_phase") not in {"calibration", "freeze"}
            or horizon.get("all_seed_length_gates_passed") is not False
            or source_index != design.horizon_grid_index - 1
            or source_tokens != design.allowed_horizon_sequence[source_index]
            or selection.get("next_horizon_tokens") != design.max_response_tokens
        ):
            raise ValueError(
                "escalated calibration horizon is not authorized by the immediately "
                "preceding failed length-gate aggregate"
            )
    else:
        expected_phase = "calibration" if design.pilot_phase == "freeze" else "freeze"
        if (
            value.get("pilot_phase") != expected_phase
            or source_tokens != design.max_response_tokens
            or source_index != design.horizon_grid_index
        ):
            raise ValueError("horizon parent aggregate does not bind the same accepted horizon")
        if design.pilot_phase is None:
            selection = _mapping(value.get("selection"), name=f"{path}:selection")
            if selection.get("selection_accepted") is not True:
                raise ValueError("confirmatory horizon parent was not accepted by the freeze pilot")
    return {
        "path": path,
        "sha256": observed_sha,
        "source_pilot_phase": value.get("pilot_phase"),
        "source_horizon_tokens": source_tokens,
        "source_horizon_grid_index": source_index,
    }


def build_phase2_pilot_aggregate(
    overlay_config: Mapping[str, object],
    result_jsons: Sequence[str | os.PathLike[str]],
    *,
    reference_base: str | os.PathLike[str] | None = None,
    beta_source_aggregate: str | os.PathLike[str] | None = None,
    horizon_parent_aggregate: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Validate all excluded pilot seeds and build one target-free selection record."""

    config = validate_phase2_config(overlay_config)
    design_mapping = _mapping(config["design"], name="design")
    if design_mapping["stage"] != "pilot" or design_mapping["formal_eligibility"] is not False:
        raise ValueError("pilot aggregation accepts only formally ineligible pilot overlays")
    pilot_phase = design_mapping["pilot_phase"]
    if pilot_phase not in {"calibration", "freeze"}:
        raise ValueError("pilot aggregation requires calibration or freeze phase")
    design = Phase2Design.from_phase2_config(config)
    runtime = design.to_dict()
    runtime_sha = design.sha256
    design_sha = phase2_design_identity(config)
    run = _mapping(config["run"], name="run")
    declared_seeds = tuple(int(seed) for seed in run["seeds"])
    if set(declared_seeds) != set(PHASE2_PILOT_SEEDS) or len(declared_seeds) != 3:
        raise ValueError("pilot aggregate requires the three permanently excluded seeds")
    if len(result_jsons) != len(declared_seeds):
        raise ValueError("pilot aggregate requires exactly one result for every declared seed")
    data = _mapping(config["data"], name="data")
    split_sizes = _mapping(run["split_sizes"], name="run.split_sizes")
    loaded: dict[int, dict[str, object]] = {}
    for path in result_jsons:
        seed_result = _load_seed(
            path,
            expected_design_sha256=design_sha,
            expected_runtime=runtime,
            expected_runtime_sha256=runtime_sha,
            expected_source_config_hash=str(design_mapping["source_config_hash"]),
            expected_pilot_phase=str(pilot_phase),
            design=design,
            prompts=int(split_sizes["test"]),
            candidates=int(data["num_candidates"]),
        )
        seed = int(seed_result["seed"])
        if seed not in declared_seeds:
            raise ValueError(f"pilot result seed {seed} is not declared by the overlay")
        if seed in loaded:
            raise ValueError(f"duplicate pilot result for seed {seed}")
        loaded[seed] = seed_result
    if tuple(sorted(loaded)) != tuple(sorted(declared_seeds)):
        raise ValueError("pilot result set does not match the declared seeds")
    environment = loaded[declared_seeds[0]]["environment_identity"]
    if any(loaded[seed]["environment_identity"] != environment for seed in declared_seeds):
        raise ValueError("pilot seeds do not share one execution environment identity")

    source_binding = verify_beta_source_aggregate(config, beta_source_aggregate)
    parent_input = (
        beta_source_aggregate
        if horizon_parent_aggregate is None and design.pilot_phase == "freeze"
        else horizon_parent_aggregate
    )
    horizon_binding = verify_horizon_parent_aggregate(config, parent_input)
    beta_by_seed = {str(seed): float(loaded[seed]["beta_common"]) for seed in declared_seeds}
    all_safety_passed = all(bool(loaded[seed]["safety_passed"]) for seed in declared_seeds)
    all_length_passed = all(
        float(loaded[seed]["observed_by_arm"][arm]["reached_max_length_rate"])
        <= _SAFETY_THRESHOLDS["reached_max_length_rate_cap"]
        for seed in declared_seeds
        for arm in PHASE2_ARM_ORDER
    )
    all_non_length_safety_passed = all(
        float(loaded[seed]["observed_by_arm"][arm][metric]) <= _SAFETY_THRESHOLDS[f"{metric}_cap"]
        for seed in declared_seeds
        for arm in PHASE2_ARM_ORDER
        for metric in _SAFETY_METRICS
        if metric != "reached_max_length_rate"
    )
    next_horizon = (
        design.allowed_horizon_sequence[design.horizon_grid_index + 1]
        if design.horizon_grid_index + 1 < len(design.allowed_horizon_sequence)
        else None
    )
    if pilot_phase == "calibration":
        recommended = max(beta_by_seed.values())
        selection = {
            "schema_version": "pilot-calibration-selection/v1",
            "candidate_beta_by_seed": beta_by_seed,
            "aggregation_rule": "maximum_across_excluded_pilot_seeds",
            "recommended_pilot_freeze_beta": recommended,
            "freeze_validation_required": all_length_passed,
            "beta_grid": "recommended_beta_times_two_to_nonnegative_integer",
            "calibration_safety_diagnostics_passed": all_safety_passed,
            "calibration_safety_diagnostics_are_measure_only": True,
            "horizon_accepted": all_length_passed,
            "next_horizon_tokens": (
                design.max_response_tokens if all_length_passed else next_horizon
            ),
            "selection_accepted": None,
            "next_action": (
                "issue_pilot_freeze_identity_at_recommended_beta"
                if all_length_passed
                else (
                    "issue_new_calibration_identity_at_next_horizon"
                    if next_horizon is not None
                    else "stop_and_revise_horizon_protocol"
                )
            ),
        }
    else:
        frozen = float(design.frozen_global_beta)
        if any(beta != frozen for beta in beta_by_seed.values()):
            raise ValueError("pilot freeze seeds did not all use the exact same global beta")
        if source_binding is None:
            raise RuntimeError("pilot freeze lost its source aggregate binding")
        selection_accepted = all_length_passed and all_non_length_safety_passed
        if not all_length_passed:
            next_action = (
                "issue_new_calibration_identity_at_next_horizon"
                if next_horizon is not None
                else "stop_and_revise_horizon_protocol"
            )
            next_global_beta: float | None = None
        elif not all_non_length_safety_passed:
            next_action = "issue_new_pilot_freeze_identity_at_double_beta"
            next_global_beta = 2.0 * frozen
        else:
            next_action = "freeze_confirmatory_design_identity"
            next_global_beta = frozen
        selection = {
            "schema_version": "pilot-freeze-selection/v1",
            "frozen_global_beta": frozen,
            "all_seeds_and_arms_used_same_beta": True,
            "beta_grid_index": source_binding["beta_grid_index"],
            "all_pre_oracle_safety_gates_passed": all_safety_passed,
            "all_length_gates_passed": all_length_passed,
            "all_non_length_safety_gates_passed": all_non_length_safety_passed,
            "selection_accepted": selection_accepted,
            "accepted_for_confirmatory_identity": selection_accepted,
            "next_horizon_tokens": (
                design.max_response_tokens if all_length_passed else next_horizon
            ),
            "next_global_beta": next_global_beta,
            "next_action": next_action,
        }
    base = Path(reference_base).resolve() if reference_base is not None else Path.cwd().resolve()
    sources = [
        {
            "seed": seed,
            "result": relative_posix_reference(
                loaded[seed]["result_path"],
                base=base,
            ),
            "result_sha256": loaded[seed]["result_sha256"],
            "diagnostics_jsonl": relative_posix_reference(
                loaded[seed]["sidecar_path"],
                base=base,
            ),
            "diagnostics_sha256": loaded[seed]["sidecar_sha256"],
        }
        for seed in declared_seeds
    ]
    return {
        "schema_version": PHASE2_PILOT_AGGREGATE_SCHEMA,
        "pilot_phase": pilot_phase,
        "formal_eligibility": False,
        "supports_formal_claim": False,
        "evidence_role": "target_free_design_selection_only",
        "source_config_hash": design_mapping["source_config_hash"],
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract_sha256": runtime_sha,
        "beta_source_aggregate_sha256": design.beta_source_aggregate_sha256,
        "seeds": list(declared_seeds),
        "environment_identity": environment,
        "thresholds": dict(_SAFETY_THRESHOLDS),
        "horizon": {
            "schema_version": "pilot-horizon-selection/v1",
            "candidate_horizon_tokens": design.max_response_tokens,
            "allowed_horizon_sequence": list(design.allowed_horizon_sequence),
            "horizon_grid_index": design.horizon_grid_index,
            "parent_pilot_aggregate_sha256": design.parent_pilot_aggregate_sha256,
            "previous_horizon_failed_length_gate": (design.previous_horizon_failed_length_gate),
            "all_seed_length_gates_passed": all_length_passed,
            "parent_binding_verified": (
                design.parent_pilot_aggregate_sha256 is None or horizon_binding is not None
            ),
        },
        "per_seed": {
            str(seed): {
                "beta_common": loaded[seed]["beta_common"],
                "pre_oracle_safety_passed": loaded[seed]["safety_passed"],
                "observed_by_arm": loaded[seed]["observed_by_arm"],
            }
            for seed in declared_seeds
        },
        "selection": selection,
        "information_boundary": {
            "heldout_evaluator_called": False,
            "final_oracle_session_opened": False,
            "oracle_outcomes_consumed": False,
            "prompt_or_response_text_consumed": False,
            "formal_efficacy_evidence_produced": False,
        },
        "sources": sources,
    }


def write_phase2_pilot_aggregate(
    overlay_config: Mapping[str, object],
    result_jsons: Sequence[str | os.PathLike[str]],
    output_json: str | os.PathLike[str],
    *,
    beta_source_aggregate: str | os.PathLike[str] | None = None,
    horizon_parent_aggregate: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Build and atomically publish a new pilot selection aggregate."""

    destination = Path(output_json)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite pilot aggregate: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_phase2_pilot_aggregate(
        overlay_config,
        result_jsons,
        reference_base=destination.parent,
        beta_source_aggregate=beta_source_aggregate,
        horizon_parent_aggregate=horizon_parent_aggregate,
    )
    atomic_write_json(destination, payload, overwrite=False)
    return payload


__all__ = [
    "PHASE2_PILOT_AGGREGATE_SCHEMA",
    "build_phase2_pilot_aggregate",
    "verify_beta_source_aggregate",
    "verify_horizon_parent_aggregate",
    "write_phase2_pilot_aggregate",
]
