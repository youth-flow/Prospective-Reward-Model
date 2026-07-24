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

PHASE2_PILOT_AGGREGATE_SCHEMA = "common-beta-pilot-selection-aggregate/v2"
PHASE2_PILOT_AGGREGATION_IDENTITY_SCHEMA = "phase2-pilot-aggregation-identity/v1"
_CALIBRATION_FORMAL_SELECTION_RULE = (
    "maximum_pilot_seed_candidate_then_smallest_passing_frozen_kl_only_grid"
)

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
_AGGREGATION_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "aggregator_git_commit",
        "producer_git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "validator_source_sha256",
    }
)
_PILOT_AGGREGATE_KEYS = frozenset(
    {
        "schema_version",
        "pilot_phase",
        "formal_eligibility",
        "supports_formal_claim",
        "evidence_role",
        "source_config_hash",
        "phase2_design_sha256",
        "phase2_runtime_contract",
        "phase2_runtime_contract_sha256",
        "beta_source_aggregate_sha256",
        "seeds",
        "environment_identity",
        "aggregation_identity",
        "rollout_geometry",
        "thresholds",
        "horizon",
        "predecessors",
        "per_seed",
        "selection",
        "information_boundary",
        "sources",
    }
)
_AGGREGATE_SOURCE_KEYS = frozenset(
    {
        "seed",
        "result",
        "result_sha256",
        "diagnostics_jsonl",
        "diagnostics_sha256",
        "artifact_metadata",
        "artifact_metadata_sha256",
        "run_manifest",
        "run_manifest_sha256",
        "output_verification",
        "output_verification_sha256",
        "success_receipt",
        "success_receipt_sha256",
    }
)
_AGGREGATE_PREDECESSOR_KEYS = frozenset({"beta_source_aggregate", "horizon_parent_aggregate"})
_AGGREGATE_PREDECESSOR_REFERENCE_KEYS = frozenset({"path", "sha256"})
_AGGREGATE_HORIZON_KEYS = frozenset(
    {
        "schema_version",
        "candidate_horizon_tokens",
        "allowed_horizon_sequence",
        "horizon_grid_index",
        "parent_pilot_aggregate_sha256",
        "previous_horizon_failed_length_gate",
        "all_seed_length_gates_passed",
        "parent_binding_verified",
    }
)
_AGGREGATE_INFORMATION_BOUNDARY_KEYS = frozenset(
    {
        "heldout_evaluator_called",
        "final_oracle_session_opened",
        "oracle_outcomes_consumed",
        "prompt_or_response_text_consumed",
        "formal_efficacy_evidence_produced",
    }
)
_PROMPT_MATERIALIZATION_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "policy_chat_template_sha256",
        "encoding",
        "add_generation_prompt",
        "truncation",
        "fail_closed_above_max_prompt_tokens",
        "max_prompt_tokens",
        "num_prompts",
        "minimum_policy_chat_token_count",
        "maximum_policy_chat_token_count",
        "mean_policy_chat_token_count",
        "over_limit_prompt_count",
        "truncated_prompt_count",
        "raw_prompt_preserved_count",
        "records_sha256",
        "candidate_prefixes_verified",
    }
)
_PROMPT_ROLLOUT_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "num_prompts",
        "max_prompt_tokens",
        "minimum_policy_chat_token_count",
        "maximum_policy_chat_token_count",
        "mean_policy_chat_token_count",
        "over_limit_prompt_count",
        "truncated_prompt_count",
        "raw_prompt_preserved_count",
        "matches_materialization_token_prefix_evidence",
        "same_evidence_across_policy_arms",
    }
)
_PROMPT_ORACLE_SUMMARY_KEYS = frozenset(
    {
        "input_text",
        "rerendered_with_independent_oracle_chat_template",
        "policy_chat_tokens_reused_by_oracle",
        "policy_and_oracle_chat_template_sha256_distinct",
        "policy_chat_template_sha256",
        "oracle_chat_template_sha256",
    }
)
_PROMPT_SEMANTICS_RECORD_KEYS = frozenset(
    {
        "prompt_id",
        "raw_prompt_sha256",
        "policy_chat_token_count",
        "policy_prompt_token_ids_sha256",
        "max_prompt_tokens",
        "truncated",
        "raw_prompt_preserved",
    }
)


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


def _git_commit(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git object ID")
    return value


def _validator_source_sha256() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def _validate_aggregation_identity(
    value: object,
    *,
    producer_environment: Mapping[str, object],
    name: str,
    require_current_validator: bool,
) -> dict[str, object]:
    identity = _exact_keys(
        value,
        name=name,
        expected=_AGGREGATION_IDENTITY_KEYS,
    )
    if identity["schema_version"] != PHASE2_PILOT_AGGREGATION_IDENTITY_SCHEMA:
        raise ValueError(f"{name}.schema_version is invalid")
    _git_commit(identity["aggregator_git_commit"], name=f"{name}.aggregator_git_commit")
    producer_commit = _git_commit(
        identity["producer_git_commit"],
        name=f"{name}.producer_git_commit",
    )
    image_sha = _digest(identity["image_sha256"], name=f"{name}.image_sha256")
    inventory_sha = _digest(
        identity["hf_inventory_sha256"],
        name=f"{name}.hf_inventory_sha256",
    )
    validator_sha = _digest(
        identity["validator_source_sha256"],
        name=f"{name}.validator_source_sha256",
    )
    if (
        producer_commit != producer_environment.get("git_commit")
        or image_sha != producer_environment.get("image_sha256")
        or inventory_sha != producer_environment.get("hf_inventory_sha256")
    ):
        raise ValueError(f"{name} does not bind the shared seed producer identity")
    if require_current_validator and validator_sha != _validator_source_sha256():
        raise ValueError(f"{name} does not bind the loaded pilot aggregate validator source")
    return dict(identity)


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


def _validate_artifact_prompt_semantics(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    semantics = _exact_keys(
        value,
        name=name,
        expected={
            "schema_version",
            "encoding",
            "add_generation_prompt",
            "truncation",
            "fail_closed_above_max_prompt_tokens",
            "max_prompt_tokens",
            "num_prompts",
            "records_sha256",
            "records",
        },
    )
    max_prompt_tokens = _integer(
        semantics["max_prompt_tokens"],
        name=f"{name}.max_prompt_tokens",
        minimum=1,
    )
    num_prompts = _integer(
        semantics["num_prompts"],
        name=f"{name}.num_prompts",
        minimum=1,
    )
    records_sha256 = _digest(
        semantics["records_sha256"],
        name=f"{name}.records_sha256",
    )
    records = semantics["records"]
    if (
        not isinstance(records, list)
        or len(records) != num_prompts
        or semantics["schema_version"] != "full-policy-prompt-semantics/v1"
        or semantics["encoding"] != "policy_tokenizer_apply_chat_template"
        or semantics["add_generation_prompt"] is not True
        or semantics["truncation"] is not False
        or semantics["fail_closed_above_max_prompt_tokens"] is not True
    ):
        raise ValueError(f"{name} does not contain the exact full-prompt record stream")
    if _canonical_sha256(records) != records_sha256:
        raise ValueError(f"{name}.records_sha256 differs from the canonical record bytes")

    token_counts: list[int] = []
    prompt_ids: set[str] = set()
    raw_preserved = 0
    for index, raw_record in enumerate(records):
        record = _exact_keys(
            raw_record,
            name=f"{name}.records[{index}]",
            expected=_PROMPT_SEMANTICS_RECORD_KEYS,
        )
        prompt_id = record["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in prompt_ids:
            raise ValueError(f"{name}.records[{index}].prompt_id is invalid or duplicated")
        prompt_ids.add(prompt_id)
        _digest(
            record["raw_prompt_sha256"],
            name=f"{name}.records[{index}].raw_prompt_sha256",
        )
        _digest(
            record["policy_prompt_token_ids_sha256"],
            name=f"{name}.records[{index}].policy_prompt_token_ids_sha256",
        )
        token_count = _integer(
            record["policy_chat_token_count"],
            name=f"{name}.records[{index}].policy_chat_token_count",
            minimum=1,
        )
        record_cap = _integer(
            record["max_prompt_tokens"],
            name=f"{name}.records[{index}].max_prompt_tokens",
            minimum=1,
        )
        if (
            token_count > max_prompt_tokens
            or record_cap != max_prompt_tokens
            or record["truncated"] is not False
            or record["raw_prompt_preserved"] is not True
        ):
            raise ValueError(f"{name}.records[{index}] violates the full-prompt contract")
        token_counts.append(token_count)
        raw_preserved += 1
    return {
        "schema_version": semantics["schema_version"],
        "encoding": semantics["encoding"],
        "add_generation_prompt": semantics["add_generation_prompt"],
        "truncation": semantics["truncation"],
        "fail_closed_above_max_prompt_tokens": semantics["fail_closed_above_max_prompt_tokens"],
        "max_prompt_tokens": max_prompt_tokens,
        "num_prompts": num_prompts,
        "minimum_policy_chat_token_count": min(token_counts),
        "maximum_policy_chat_token_count": max(token_counts),
        "mean_policy_chat_token_count": sum(token_counts) / len(token_counts),
        "over_limit_prompt_count": 0,
        "truncated_prompt_count": 0,
        "raw_prompt_preserved_count": raw_preserved,
        "records_sha256": records_sha256,
    }


def _runtime_sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return tuple(value)


def _optional_runtime_sequence(
    value: object,
    *,
    name: str,
) -> tuple[object, ...] | None:
    if value is None:
        return None
    return _runtime_sequence(value, name=name)


def _phase2_design_from_runtime(
    value: object,
    *,
    expected_phase: str,
    expected_sha256: str,
    name: str,
) -> Phase2Design:
    runtime = _mapping(value, name=name)
    scope = _exact_keys(
        runtime.get("sensitivity_scope"),
        name=f"{name}.sensitivity_scope",
        expected={
            "pilot_k_cal_candidates",
            "frozen_global_beta_multipliers",
            "sensitivity_step_rule",
            "ridge_multipliers_configured",
            "executed_by_this_runner_invocation",
            "result_role",
        },
    )
    max_length = _exact_keys(
        runtime.get("max_length_gate"),
        name=f"{name}.max_length_gate",
        expected={"formal_gate", "formal_threshold", "measure_only"},
    )
    try:
        design = Phase2Design(
            stage=runtime["stage"],  # type: ignore[arg-type]
            formal_eligibility=runtime["formal_eligibility"],  # type: ignore[arg-type]
            pilot_phase=runtime["pilot_phase"],  # type: ignore[arg-type]
            common_beta_rule=runtime["common_beta_rule"],  # type: ignore[arg-type]
            common_beta_calibration_split=runtime[  # type: ignore[arg-type]
                "common_beta_calibration_split"
            ],
            common_beta_source=runtime["common_beta_source"],  # type: ignore[arg-type]
            frozen_global_beta=runtime["frozen_global_beta"],  # type: ignore[arg-type]
            beta_source_aggregate_sha256=runtime[  # type: ignore[arg-type]
                "beta_source_aggregate_sha256"
            ],
            target_oracle_quadratic_kl=runtime[  # type: ignore[arg-type]
                "target_oracle_quadratic_kl"
            ],
            measured_kl_safety_cap=runtime["measured_kl_safety_cap"],  # type: ignore[arg-type]
            prompt_mean_p95_kl_cap=runtime["prompt_mean_p95_kl_cap"],  # type: ignore[arg-type]
            prompt_mean_p99_kl_cap=runtime["prompt_mean_p99_kl_cap"],  # type: ignore[arg-type]
            prompt_mean_maximum_kl_cap=runtime[  # type: ignore[arg-type]
                "prompt_mean_maximum_kl_cap"
            ],
            per_sequence_maximum_kl_cap=runtime[  # type: ignore[arg-type]
                "per_sequence_maximum_kl_cap"
            ],
            max_response_tokens=runtime["max_response_tokens"],  # type: ignore[arg-type]
            allowed_horizon_sequence=_runtime_sequence(
                runtime["allowed_horizon_sequence"],
                name=f"{name}.allowed_horizon_sequence",
            ),  # type: ignore[arg-type]
            horizon_grid_index=runtime["horizon_grid_index"],  # type: ignore[arg-type]
            parent_pilot_aggregate_sha256=runtime[  # type: ignore[arg-type]
                "parent_pilot_aggregate_sha256"
            ],
            previous_horizon_failed_length_gate=runtime[  # type: ignore[arg-type]
                "previous_horizon_failed_length_gate"
            ],
            rollout_candidates_per_prompt=runtime[  # type: ignore[arg-type]
                "rollout_candidates_per_prompt"
            ],
            relative_damping=runtime["relative_damping"],  # type: ignore[arg-type]
            pcg_dtype=runtime["pcg_dtype"],  # type: ignore[arg-type]
            pcg_max_iterations=runtime["pcg_max_iterations"],  # type: ignore[arg-type]
            pcg_tolerance=runtime["pcg_tolerance"],  # type: ignore[arg-type]
            oracle_batch_size=runtime["oracle_batch_size"],  # type: ignore[arg-type]
            kl_token_chunk_size=runtime["kl_token_chunk_size"],  # type: ignore[arg-type]
            k_cal_sensitivity_values=_optional_runtime_sequence(
                scope["pilot_k_cal_candidates"],
                name=f"{name}.sensitivity_scope.pilot_k_cal_candidates",
            ),  # type: ignore[arg-type]
            frozen_global_beta_sensitivity_multipliers=_optional_runtime_sequence(
                scope["frozen_global_beta_multipliers"],
                name=f"{name}.sensitivity_scope.frozen_global_beta_multipliers",
            ),  # type: ignore[arg-type]
            ridge_sensitivity_multipliers=_runtime_sequence(
                scope["ridge_multipliers_configured"],
                name=f"{name}.sensitivity_scope.ridge_multipliers_configured",
            ),  # type: ignore[arg-type]
            max_length_formal_gate=max_length["formal_gate"],  # type: ignore[arg-type]
            max_length_formal_threshold=max_length["formal_threshold"],  # type: ignore[arg-type]
        )
    except KeyError as error:
        raise ValueError(f"{name} is missing runtime field {error.args[0]!r}") from error
    if (
        dict(runtime) != design.to_dict()
        or design.stage != "pilot"
        or design.formal_eligibility
        or design.pilot_phase != expected_phase
        or design.sha256 != expected_sha256
    ):
        raise ValueError(f"{name} is not the exact declared pilot runtime contract")
    return design


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


def _validate_information_boundary(
    value: object,
    *,
    pilot_phase: str,
    materialized_prompts: int,
    rollout_prompts: int,
    max_prompt_tokens: int,
    name: str,
) -> None:
    boundary = _exact_keys(
        value,
        name=name,
        expected={
            "calibration_split",
            "new_rollout_prompts_used_for_calibration",
            "final_oracle_session_opened",
            "rollout_responses_oracle_scored",
            "heldout_evaluator_called",
            "oracle_outcomes_serialized",
            "prompt_or_response_text_serialized",
            "token_ids_or_response_masks_serialized",
            "source_artifact_format",
            "source_artifact_may_contain_prior_heldout_candidate_scores",
            "source_artifact_heldout_targets_exposed_by_phase2_prepared_inputs",
            "prompt_semantics",
        },
    )
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
    expected_split = (
        "train_only" if pilot_phase == "calibration" else "excluded_pilot_calibration_outputs_only"
    )
    if (
        any(boundary[field] is not False for field in required_false)
        or boundary["calibration_split"] != expected_split
        or boundary["source_artifact_format"] != "phase1_bridge"
        or boundary["source_artifact_may_contain_prior_heldout_candidate_scores"] is not True
    ):
        raise ValueError(f"{name} does not prove the target-free information boundary")
    prompt_semantics = _exact_keys(
        boundary["prompt_semantics"],
        name=f"{name}.prompt_semantics",
        expected={"schema_version", "materialization", "rollout", "oracle"},
    )
    if prompt_semantics["schema_version"] != "phase2-full-prompt-continuity/v1":
        raise ValueError(f"{name}.prompt_semantics has the wrong schema")
    materialization = _exact_keys(
        prompt_semantics["materialization"],
        name=f"{name}.prompt_semantics.materialization",
        expected=_PROMPT_MATERIALIZATION_SUMMARY_KEYS,
    )
    rollout = _exact_keys(
        prompt_semantics["rollout"],
        name=f"{name}.prompt_semantics.rollout",
        expected=_PROMPT_ROLLOUT_SUMMARY_KEYS,
    )
    oracle = _exact_keys(
        prompt_semantics["oracle"],
        name=f"{name}.prompt_semantics.oracle",
        expected=_PROMPT_ORACLE_SUMMARY_KEYS,
    )

    materialized_count = _integer(
        materialization["num_prompts"],
        name=f"{name}.prompt_semantics.materialization.num_prompts",
        minimum=1,
    )
    materialized_cap = _integer(
        materialization["max_prompt_tokens"],
        name=f"{name}.prompt_semantics.materialization.max_prompt_tokens",
        minimum=1,
    )
    materialized_minimum = _integer(
        materialization["minimum_policy_chat_token_count"],
        name=f"{name}.prompt_semantics.materialization.minimum_policy_chat_token_count",
        minimum=1,
    )
    materialized_maximum = _integer(
        materialization["maximum_policy_chat_token_count"],
        name=f"{name}.prompt_semantics.materialization.maximum_policy_chat_token_count",
        minimum=1,
    )
    materialized_mean = _finite(
        materialization["mean_policy_chat_token_count"],
        name=f"{name}.prompt_semantics.materialization.mean_policy_chat_token_count",
    )
    if (
        materialization["schema_version"] != "full-policy-prompt-semantics/v1"
        or materialization["encoding"] != "policy_tokenizer_apply_chat_template"
        or materialization["add_generation_prompt"] is not True
        or materialization["truncation"] is not False
        or materialization["fail_closed_above_max_prompt_tokens"] is not True
        or materialization["candidate_prefixes_verified"] is not True
        or materialized_count != materialized_prompts
        or materialized_cap != max_prompt_tokens
        or _integer(
            materialization["over_limit_prompt_count"],
            name=f"{name}.prompt_semantics.materialization.over_limit_prompt_count",
        )
        != 0
        or _integer(
            materialization["truncated_prompt_count"],
            name=f"{name}.prompt_semantics.materialization.truncated_prompt_count",
        )
        != 0
        or _integer(
            materialization["raw_prompt_preserved_count"],
            name=f"{name}.prompt_semantics.materialization.raw_prompt_preserved_count",
        )
        != materialized_count
        or not (
            materialized_minimum <= materialized_mean <= materialized_maximum <= materialized_cap
        )
    ):
        raise ValueError(f"{name}.prompt_semantics.materialization is invalid")
    materialization_policy_template = _digest(
        materialization["policy_chat_template_sha256"],
        name=f"{name}.prompt_semantics.materialization.policy_chat_template_sha256",
    )
    _digest(
        materialization["records_sha256"],
        name=f"{name}.prompt_semantics.materialization.records_sha256",
    )

    rollout_count = _integer(
        rollout["num_prompts"],
        name=f"{name}.prompt_semantics.rollout.num_prompts",
        minimum=1,
    )
    rollout_cap = _integer(
        rollout["max_prompt_tokens"],
        name=f"{name}.prompt_semantics.rollout.max_prompt_tokens",
        minimum=1,
    )
    rollout_minimum = _integer(
        rollout["minimum_policy_chat_token_count"],
        name=f"{name}.prompt_semantics.rollout.minimum_policy_chat_token_count",
        minimum=1,
    )
    rollout_maximum = _integer(
        rollout["maximum_policy_chat_token_count"],
        name=f"{name}.prompt_semantics.rollout.maximum_policy_chat_token_count",
        minimum=1,
    )
    rollout_mean = _finite(
        rollout["mean_policy_chat_token_count"],
        name=f"{name}.prompt_semantics.rollout.mean_policy_chat_token_count",
    )
    if (
        rollout["schema_version"] != materialization["schema_version"]
        or rollout_count != rollout_prompts
        or rollout_cap != materialized_cap
        or _integer(
            rollout["over_limit_prompt_count"],
            name=f"{name}.prompt_semantics.rollout.over_limit_prompt_count",
        )
        != 0
        or _integer(
            rollout["truncated_prompt_count"],
            name=f"{name}.prompt_semantics.rollout.truncated_prompt_count",
        )
        != 0
        or _integer(
            rollout["raw_prompt_preserved_count"],
            name=f"{name}.prompt_semantics.rollout.raw_prompt_preserved_count",
        )
        != rollout_count
        or rollout["matches_materialization_token_prefix_evidence"] is not True
        or rollout["same_evidence_across_policy_arms"] is not True
        or not (rollout_minimum <= rollout_mean <= rollout_maximum <= rollout_cap)
    ):
        raise ValueError(f"{name}.prompt_semantics.rollout is invalid")

    oracle_policy_template = _digest(
        oracle["policy_chat_template_sha256"],
        name=f"{name}.prompt_semantics.oracle.policy_chat_template_sha256",
    )
    oracle_template = _digest(
        oracle["oracle_chat_template_sha256"],
        name=f"{name}.prompt_semantics.oracle.oracle_chat_template_sha256",
    )
    if (
        oracle["input_text"] != "same_raw_prompt_plus_assistant_response"
        or oracle["rerendered_with_independent_oracle_chat_template"] is not True
        or oracle["policy_chat_tokens_reused_by_oracle"] is not False
        or oracle["policy_and_oracle_chat_template_sha256_distinct"] is not True
        or oracle_policy_template != materialization_policy_template
        or oracle_template == oracle_policy_template
    ):
        raise ValueError(f"{name}.prompt_semantics.oracle is invalid")


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
            or evidence["formal_selection_rule"] != _CALIBRATION_FORMAL_SELECTION_RULE
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
        if target <= 0.0:
            raise ValueError(f"{name}.target_oracle_quadratic_kl must be strictly positive")
        expected_beta = math.sqrt(curvature / (2.0 * target))
        if not math.isclose(
            predicted,
            target,
            rel_tol=2.0e-12,
            abs_tol=1.0e-15,
        ) or not math.isclose(
            beta,
            expected_beta,
            rel_tol=2.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"{name} violates the calibration target/beta closed-form identity")
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


def _fixed_sibling(
    result_path: Path,
    reference: object,
    *,
    field: str,
    expected_name: str,
) -> Path:
    if reference != expected_name:
        raise ValueError(
            f"{result_path}:{field} must equal the fixed sibling name {expected_name!r}"
        )
    sibling = result_path.parent / expected_name
    if sibling.parent.resolve() != result_path.parent.resolve():
        raise ValueError(f"{result_path}:{field} escapes its result directory")
    return sibling


def _parse_success_receipt(path: Path) -> tuple[Mapping[str, object], bytes]:
    raw = _read_regular(path, name="pilot SUCCESS receipt", max_bytes=64 * 1024)
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{path} must be a non-empty newline-terminated receipt")
    fields: dict[str, object] = {}
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not UTF-8") from error
    for line_number, line in enumerate(lines, start=1):
        if not line or "=" not in line:
            raise ValueError(f"{path}:{line_number} is not a key=value receipt line")
        key, value = line.split("=", 1)
        if not key or key in fields:
            raise ValueError(f"{path}:{line_number} has an invalid or duplicate key")
        fields[key] = value
    return (
        _exact_keys(
            fields,
            name=str(path),
            expected={
                "schema_version",
                "status",
                "workload_exit_code",
                "final_exit_code",
                "array_job_id",
                "array_task_id",
                "seed",
                "phase2_design_sha256",
                "base_config_hash",
                "git_commit",
                "beta_source_aggregate_present",
                "beta_source_aggregate_sha256",
                "horizon_parent_aggregate_present",
                "horizon_parent_aggregate_sha256",
                "created_at_utc",
            },
        ),
        raw,
    )


def _manifest_environment(
    value: Mapping[str, object],
    *,
    expected_source_config_hash: str,
    seed: int,
    name: str,
) -> dict[str, object]:
    manifest = _exact_keys(
        value,
        name=name,
        expected={
            "schema_version",
            "created_at_utc",
            "config_hash",
            "normalized_config",
            "seed",
            "selected_seed",
            "named_seeds",
            "git",
            "python",
            "platform",
            "torch",
            "revisions",
            "packages",
            "slurm",
        },
    )
    if (
        manifest["schema_version"] != "smart-reward-run/v1"
        or manifest["config_hash"] != expected_source_config_hash
        or manifest["selected_seed"] != seed
    ):
        raise ValueError(f"{name} does not bind the pilot base identity and selected seed")
    git = _exact_keys(
        manifest["git"],
        name=f"{name}.git",
        expected={"commit", "dirty"},
    )
    commit = git["commit"]
    if (
        not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or any(character not in _HEX for character in commit)
        or git["dirty"] is not False
    ):
        raise ValueError(f"{name} does not bind a clean producer Git commit")
    slurm = _mapping(manifest["slurm"], name=f"{name}.slurm")
    torch_state = _mapping(manifest["torch"], name=f"{name}.torch")
    gpus = torch_state.get("gpus")
    gpu_models = (
        [gpu.get("name") for gpu in gpus if isinstance(gpu, Mapping)]
        if isinstance(gpus, list)
        else []
    )
    environment = {
        "formal": (
            slurm.get("PRORM_GIT_COMMIT") == commit
            and slurm.get("SLURM_JOB_ACCOUNT") == "sigroup"
            and slurm.get("SLURM_JOB_PARTITION") == "gpu-l20"
            and torch_state.get("cuda_available") is True
            and torch_state.get("gpu_count") == 1
            and len(gpu_models) == 1
            and isinstance(gpu_models[0], str)
            and bool(gpu_models[0])
        ),
        "git_commit": commit,
        "image_sha256": slurm.get("PRORM_IMAGE_SHA256"),
        "hf_inventory_sha256": slurm.get("PRORM_HF_INVENTORY_SHA256"),
        "account": slurm.get("SLURM_JOB_ACCOUNT"),
        "partition": slurm.get("SLURM_JOB_PARTITION"),
        "gpu_models": gpu_models,
    }
    return _validate_environment(environment, name=f"{name}.environment_identity")


def _validate_seed_provenance(
    result_path: Path,
    result: Mapping[str, object],
    *,
    seed: int,
    expected_design_sha256: str,
    expected_source_config_hash: str,
    pilot_phase: str,
    environment: Mapping[str, object],
    diagnostics_sha256: str,
    diagnostic_records: int,
    safety_passed: bool,
    design: Phase2Design,
) -> dict[str, object]:
    artifact_reference = result["artifact_dir"]
    if artifact_reference != "artifact":
        raise ValueError(f"{result_path}:artifact_dir must equal the fixed sibling 'artifact'")
    artifact_dir = result_path.parent / "artifact"
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"{result_path}:artifact_dir is not a directory")
    metadata_path = artifact_dir / "metadata.json"
    metadata_raw = _read_regular(
        metadata_path,
        name="pilot artifact metadata",
        max_bytes=16 * 1024 * 1024,
    )
    metadata_sha256 = _sha256_bytes(metadata_raw)
    if metadata_sha256 != result["artifact_metadata_sha256"]:
        raise ValueError(f"{result_path} artifact metadata SHA256 mismatch")
    metadata = _exact_keys(
        _strict_json_bytes(metadata_raw, path=metadata_path),
        name=str(metadata_path),
        expected={
            "schema",
            "config_hash",
            "seed",
            "splits",
            "tensors",
            "tensor_sha256",
            "evidence",
        },
    )
    evidence = _mapping(metadata["evidence"], name=f"{metadata_path}:evidence")
    producer = _exact_keys(
        evidence.get("producer"),
        name=f"{metadata_path}:evidence.producer",
        expected={"git_commit", "image_sha256", "hf_inventory_sha256"},
    )
    expected_producer = {
        "git_commit": environment["git_commit"],
        "image_sha256": environment["image_sha256"],
        "hf_inventory_sha256": environment["hf_inventory_sha256"],
    }
    if (
        metadata["schema"] != "controlled-feature-artifact/v1"
        or metadata["config_hash"] != expected_source_config_hash
        or metadata["seed"] != seed
        or dict(producer) != expected_producer
    ):
        raise ValueError(f"{metadata_path} does not bind the seed/base/producer identity")
    prompt_semantics = _mapping(
        _mapping(
            result["information_boundary"],
            name=f"{result_path}:information_boundary",
        )["prompt_semantics"],
        name=f"{result_path}:information_boundary.prompt_semantics",
    )
    materialization_semantics = _mapping(
        prompt_semantics["materialization"],
        name=f"{result_path}:information_boundary.prompt_semantics.materialization",
    )
    oracle_semantics = _mapping(
        prompt_semantics["oracle"],
        name=f"{result_path}:information_boundary.prompt_semantics.oracle",
    )
    artifact_policy_template = _digest(
        evidence.get("chat_template_sha256"),
        name=f"{metadata_path}:evidence.chat_template_sha256",
    )
    artifact_oracle_template = _digest(
        evidence.get("oracle_chat_template_sha256"),
        name=f"{metadata_path}:evidence.oracle_chat_template_sha256",
    )
    artifact_prompt_semantics = _validate_artifact_prompt_semantics(
        evidence.get("policy_prompt_semantics"),
        name=f"{metadata_path}:evidence.policy_prompt_semantics",
    )
    if (
        materialization_semantics["policy_chat_template_sha256"] != artifact_policy_template
        or oracle_semantics["policy_chat_template_sha256"] != artifact_policy_template
        or oracle_semantics["oracle_chat_template_sha256"] != artifact_oracle_template
        or any(
            materialization_semantics[field] != expected
            for field, expected in artifact_prompt_semantics.items()
        )
    ):
        raise ValueError(
            f"{result_path} prompt-continuity evidence differs from its bound artifact metadata"
        )

    manifest_path = _fixed_sibling(
        result_path,
        result["run_manifest"],
        field="run_manifest",
        expected_name="run-manifest.json",
    )
    manifest_raw = _read_regular(
        manifest_path,
        name="pilot run manifest",
        max_bytes=64 * 1024 * 1024,
    )
    manifest_sha256 = _sha256_bytes(manifest_raw)
    if manifest_sha256 != result["run_manifest_sha256"]:
        raise ValueError(f"{result_path} run manifest SHA256 mismatch")
    manifest = _strict_json_bytes(manifest_raw, path=manifest_path)
    manifest_environment = _manifest_environment(
        manifest,
        expected_source_config_hash=expected_source_config_hash,
        seed=seed,
        name=str(manifest_path),
    )
    if manifest_environment != environment:
        raise ValueError(f"{result_path} environment identity differs from its run manifest")

    verification_path = result_path.parent / "phase2-output-verification.json"
    verification_raw = _read_regular(
        verification_path,
        name="pilot output verification",
        max_bytes=16 * 1024 * 1024,
    )
    verification = _exact_keys(
        _strict_json_bytes(verification_raw, path=verification_path),
        name=str(verification_path),
        expected={
            "schema_version",
            "status",
            "seed",
            "source_config_hash",
            "phase2_design_sha256",
            "pilot_phase",
            "diagnostic_records",
            "diagnostics_sha256",
            "kl_gate_passed",
            "kl_measure_only",
            "kl_violations",
            "pre_oracle_gate_passed",
            "pre_oracle_violations",
            "environment_identity",
        },
    )
    measured = _mapping(
        result["measured_kl_safety"],
        name=f"{result_path}:measured_kl_safety",
    )
    pre_oracle = _mapping(
        result["pre_oracle_safety_gate"],
        name=f"{result_path}:pre_oracle_safety_gate",
    )
    if (
        verification["schema_version"] != "prorm-phase2-output-verification/v1"
        or verification["status"] != "passed"
        or verification["seed"] != seed
        or verification["source_config_hash"] != expected_source_config_hash
        or verification["phase2_design_sha256"] != expected_design_sha256
        or verification["pilot_phase"] != pilot_phase
        or verification["diagnostic_records"] != diagnostic_records
        or verification["diagnostics_sha256"] != diagnostics_sha256
        or verification["kl_gate_passed"] is not measured.get("passed")
        or verification["kl_measure_only"] is not True
        or verification["kl_violations"] != measured.get("violations")
        or verification["pre_oracle_gate_passed"] is not safety_passed
        or verification["pre_oracle_violations"] != pre_oracle.get("violations")
        or verification["environment_identity"] != environment
    ):
        raise ValueError(f"{verification_path} does not bind the verified pilot result")

    success_path = result_path.parent / "SUCCESS"
    success, success_raw = _parse_success_receipt(success_path)
    beta_source = design.beta_source_aggregate_sha256
    horizon_parent = design.parent_pilot_aggregate_sha256
    expected_beta_present = "1" if beta_source is not None else "0"
    expected_horizon_present = "1" if horizon_parent is not None else "0"
    array_job_id = str(success["array_job_id"])
    array_task_id = str(success["array_task_id"])
    if (
        success["schema_version"] != "prorm-phase2-run-status/v1"
        or success["status"] != "SUCCESS"
        or success["workload_exit_code"] != "0"
        or success["final_exit_code"] != "0"
        or success["seed"] != str(seed)
        or success["phase2_design_sha256"] != expected_design_sha256
        or success["base_config_hash"] != expected_source_config_hash
        or success["git_commit"] != environment["git_commit"]
        or success["beta_source_aggregate_present"] != expected_beta_present
        or success["beta_source_aggregate_sha256"] != (beta_source or "none")
        or success["horizon_parent_aggregate_present"] != expected_horizon_present
        or success["horizon_parent_aggregate_sha256"] != (horizon_parent or "none")
        or not array_job_id.isdigit()
        or array_job_id.startswith("0")
        or not array_task_id.isdigit()
        or int(array_task_id) != seed - min(PHASE2_PILOT_SEEDS)
        or result_path.parent.name != f"job-{array_job_id}_{array_task_id}"
        or not success["created_at_utc"]
    ):
        raise ValueError(f"{success_path} does not bind the successful pilot attempt")

    return {
        "artifact_metadata_path": metadata_path,
        "artifact_metadata_sha256": metadata_sha256,
        "run_manifest_path": manifest_path,
        "run_manifest_sha256": manifest_sha256,
        "output_verification_path": verification_path,
        "output_verification_sha256": _sha256_bytes(verification_raw),
        "success_receipt_path": success_path,
        "success_receipt_sha256": _sha256_bytes(success_raw),
    }


def _load_seed(
    raw_path: str | os.PathLike[str],
    *,
    expected_design_sha256: str,
    expected_runtime: Mapping[str, object],
    expected_runtime_sha256: str,
    expected_source_config_hash: str,
    expected_pilot_phase: str,
    design: Phase2Design,
    materialized_prompts: int,
    prompts: int,
    candidates: int,
    max_prompt_tokens: int,
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
        pilot_phase=expected_pilot_phase,
        materialized_prompts=materialized_prompts,
        rollout_prompts=prompts,
        max_prompt_tokens=max_prompt_tokens,
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
    provenance = _validate_seed_provenance(
        path,
        value,
        seed=seed,
        expected_design_sha256=expected_design_sha256,
        expected_source_config_hash=expected_source_config_hash,
        pilot_phase=expected_pilot_phase,
        environment=environment,
        diagnostics_sha256=sidecar_sha,
        diagnostic_records=len(rows),
        safety_passed=safety_passed,
        design=design,
    )
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
        **provenance,
    }


def _selection_from_loaded_seeds(
    *,
    pilot_phase: str,
    design: Phase2Design,
    declared_seeds: Sequence[int],
    loaded: Mapping[int, Mapping[str, object]],
    source_binding: Mapping[str, object] | None,
) -> tuple[dict[str, float], bool, bool, bool, dict[str, object]]:
    beta_by_seed = {str(seed): float(loaded[seed]["beta_common"]) for seed in declared_seeds}
    all_safety_passed = all(bool(loaded[seed]["safety_passed"]) for seed in declared_seeds)
    all_length_passed = all(
        float(
            _mapping(
                _mapping(
                    loaded[seed]["observed_by_arm"],
                    name=f"seed-{seed}.observed_by_arm",
                )[arm],
                name=f"seed-{seed}.observed_by_arm.{arm}",
            )["reached_max_length_rate"]
        )
        <= _SAFETY_THRESHOLDS["reached_max_length_rate_cap"]
        for seed in declared_seeds
        for arm in PHASE2_ARM_ORDER
    )
    all_non_length_safety_passed = all(
        float(
            _mapping(
                _mapping(
                    loaded[seed]["observed_by_arm"],
                    name=f"seed-{seed}.observed_by_arm",
                )[arm],
                name=f"seed-{seed}.observed_by_arm.{arm}",
            )[metric]
        )
        <= _SAFETY_THRESHOLDS[f"{metric}_cap"]
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
        selection: dict[str, object] = {
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
            raise ValueError("pilot freeze lacks its validated beta-source aggregate binding")
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
    return (
        beta_by_seed,
        all_safety_passed,
        all_length_passed,
        all_non_length_safety_passed,
        selection,
    )


def _resolved_aggregate_reference(
    aggregate_path: Path,
    reference: object,
    *,
    name: str,
) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{name} must be a non-empty relative POSIX path")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or "\\" in reference or not pure.parts:
        raise ValueError(f"{name} must be a relative POSIX path")
    resolved = (aggregate_path.parent / Path(*pure.parts)).resolve()
    if relative_posix_reference(resolved, base=aggregate_path.parent) != reference:
        raise ValueError(f"{name} is not a canonical relative reference")
    return resolved


def _validate_thresholds(value: object, *, name: str) -> dict[str, float]:
    thresholds = _exact_keys(
        value,
        name=name,
        expected=set(_SAFETY_THRESHOLDS),
    )
    validated: dict[str, float] = {}
    for field, expected in _SAFETY_THRESHOLDS.items():
        observed = _finite(thresholds[field], name=f"{name}.{field}")
        if not _close(observed, expected, tolerance=1.0e-12):
            raise ValueError(f"{name}.{field} differs from the preregistered threshold")
        validated[field] = observed
    return validated


def _validate_aggregate_information_boundary(value: object, *, name: str) -> None:
    boundary = _exact_keys(
        value,
        name=name,
        expected=_AGGREGATE_INFORMATION_BOUNDARY_KEYS,
    )
    if any(boundary[field] is not False for field in _AGGREGATE_INFORMATION_BOUNDARY_KEYS):
        raise ValueError(f"{name} is not a strictly target-free aggregate boundary")


def _load_predecessor_reference(
    aggregate_path: Path,
    value: object,
    *,
    expected_sha256: str | None,
    name: str,
    cache: dict[tuple[Path, str], Mapping[str, object]],
    active: set[Path],
) -> tuple[Path, str, Mapping[str, object]] | None:
    if expected_sha256 is None:
        if value is not None:
            raise ValueError(f"{name} must be null when the runtime declares no predecessor")
        return None
    reference = _exact_keys(
        value,
        name=name,
        expected=_AGGREGATE_PREDECESSOR_REFERENCE_KEYS,
    )
    digest = _digest(reference["sha256"], name=f"{name}.sha256")
    if digest != expected_sha256:
        raise ValueError(f"{name}.sha256 differs from the runtime predecessor identity")
    path = _resolved_aggregate_reference(
        aggregate_path,
        reference["path"],
        name=f"{name}.path",
    )
    return _load_source_aggregate(
        path,
        expected_sha256=digest,
        _cache=cache,
        _active=active,
    )


def _validate_phase2_pilot_aggregate(
    path: Path,
    value: Mapping[str, object],
    *,
    cache: dict[tuple[Path, str], Mapping[str, object]],
    active: set[Path],
) -> None:
    aggregate = _exact_keys(
        value,
        name=str(path),
        expected=_PILOT_AGGREGATE_KEYS,
    )
    pilot_phase = aggregate["pilot_phase"]
    if (
        aggregate["schema_version"] != PHASE2_PILOT_AGGREGATE_SCHEMA
        or pilot_phase not in {"calibration", "freeze"}
        or aggregate["formal_eligibility"] is not False
        or aggregate["supports_formal_claim"] is not False
        or aggregate["evidence_role"] != "target_free_design_selection_only"
    ):
        raise ValueError(f"{path} is not a formally ineligible Phase-2 pilot aggregate")
    source_config_hash = _digest(
        aggregate["source_config_hash"],
        name=f"{path}:source_config_hash",
    )
    design_sha = _digest(
        aggregate["phase2_design_sha256"],
        name=f"{path}:phase2_design_sha256",
    )
    runtime_sha = _digest(
        aggregate["phase2_runtime_contract_sha256"],
        name=f"{path}:phase2_runtime_contract_sha256",
    )
    design = _phase2_design_from_runtime(
        aggregate["phase2_runtime_contract"],
        expected_phase=str(pilot_phase),
        expected_sha256=runtime_sha,
        name=f"{path}:phase2_runtime_contract",
    )
    if aggregate["beta_source_aggregate_sha256"] != design.beta_source_aggregate_sha256:
        raise ValueError(f"{path} beta-source identity differs from its runtime contract")
    if design.beta_source_aggregate_sha256 is not None:
        _digest(
            aggregate["beta_source_aggregate_sha256"],
            name=f"{path}:beta_source_aggregate_sha256",
        )

    declared_seeds = aggregate["seeds"]
    if declared_seeds != list(PHASE2_PILOT_SEEDS):
        raise ValueError(f"{path} must declare the exact ordered excluded pilot seeds")
    environment = _validate_environment(
        aggregate["environment_identity"],
        name=f"{path}:environment_identity",
    )
    _validate_aggregation_identity(
        aggregate["aggregation_identity"],
        producer_environment=environment,
        name=f"{path}:aggregation_identity",
        require_current_validator=False,
    )
    geometry = _exact_keys(
        aggregate["rollout_geometry"],
        name=f"{path}:rollout_geometry",
        expected={
            "materialized_prompts",
            "test_prompts",
            "candidates_per_prompt",
            "max_prompt_tokens",
        },
    )
    materialized_prompts = _integer(
        geometry["materialized_prompts"],
        name=f"{path}:rollout_geometry.materialized_prompts",
        minimum=1,
    )
    prompts = _integer(
        geometry["test_prompts"],
        name=f"{path}:rollout_geometry.test_prompts",
        minimum=1,
    )
    candidates = _integer(
        geometry["candidates_per_prompt"],
        name=f"{path}:rollout_geometry.candidates_per_prompt",
        minimum=1,
    )
    max_prompt_tokens = _integer(
        geometry["max_prompt_tokens"],
        name=f"{path}:rollout_geometry.max_prompt_tokens",
        minimum=1,
    )
    if (
        materialized_prompts != 2048
        or prompts != 256
        or candidates != design.rollout_candidates_per_prompt
        or max_prompt_tokens != 1024
    ):
        raise ValueError(f"{path} rollout geometry differs from its runtime contract")
    _validate_thresholds(aggregate["thresholds"], name=f"{path}:thresholds")

    predecessors = _exact_keys(
        aggregate["predecessors"],
        name=f"{path}:predecessors",
        expected=_AGGREGATE_PREDECESSOR_KEYS,
    )
    beta_predecessor = _load_predecessor_reference(
        path,
        predecessors["beta_source_aggregate"],
        expected_sha256=design.beta_source_aggregate_sha256,
        name=f"{path}:predecessors.beta_source_aggregate",
        cache=cache,
        active=active,
    )
    horizon_predecessor = _load_predecessor_reference(
        path,
        predecessors["horizon_parent_aggregate"],
        expected_sha256=design.parent_pilot_aggregate_sha256,
        name=f"{path}:predecessors.horizon_parent_aggregate",
        cache=cache,
        active=active,
    )
    source_binding = _beta_source_binding_for_design(
        design,
        expected_source_config_hash=source_config_hash,
        predecessor=beta_predecessor,
    )
    horizon_binding = _horizon_parent_binding_for_design(
        design,
        predecessor=horizon_predecessor,
    )

    raw_sources = aggregate["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != len(PHASE2_PILOT_SEEDS):
        raise ValueError(f"{path}:sources must contain exactly three ordered seed records")
    loaded: dict[int, dict[str, object]] = {}
    for expected_seed, raw_source in zip(PHASE2_PILOT_SEEDS, raw_sources, strict=True):
        source = _exact_keys(
            raw_source,
            name=f"{path}:sources.seed-{expected_seed}",
            expected=_AGGREGATE_SOURCE_KEYS,
        )
        if source["seed"] != expected_seed:
            raise ValueError(f"{path}:sources are not in the exact declared seed order")
        result_path = _resolved_aggregate_reference(
            path,
            source["result"],
            name=f"{path}:sources.seed-{expected_seed}.result",
        )
        loaded_seed = _load_seed(
            result_path,
            expected_design_sha256=design_sha,
            expected_runtime=design.to_dict(),
            expected_runtime_sha256=runtime_sha,
            expected_source_config_hash=source_config_hash,
            expected_pilot_phase=str(pilot_phase),
            design=design,
            materialized_prompts=materialized_prompts,
            prompts=prompts,
            candidates=candidates,
            max_prompt_tokens=max_prompt_tokens,
        )
        if loaded_seed["seed"] != expected_seed:
            raise ValueError(f"{path}:source result seed differs from its source record")
        expected_source = {
            "seed": expected_seed,
            "result": relative_posix_reference(
                loaded_seed["result_path"],
                base=path.parent,
            ),
            "result_sha256": loaded_seed["result_sha256"],
            "diagnostics_jsonl": relative_posix_reference(
                loaded_seed["sidecar_path"],
                base=path.parent,
            ),
            "diagnostics_sha256": loaded_seed["sidecar_sha256"],
            "artifact_metadata": relative_posix_reference(
                loaded_seed["artifact_metadata_path"],
                base=path.parent,
            ),
            "artifact_metadata_sha256": loaded_seed["artifact_metadata_sha256"],
            "run_manifest": relative_posix_reference(
                loaded_seed["run_manifest_path"],
                base=path.parent,
            ),
            "run_manifest_sha256": loaded_seed["run_manifest_sha256"],
            "output_verification": relative_posix_reference(
                loaded_seed["output_verification_path"],
                base=path.parent,
            ),
            "output_verification_sha256": loaded_seed["output_verification_sha256"],
            "success_receipt": relative_posix_reference(
                loaded_seed["success_receipt_path"],
                base=path.parent,
            ),
            "success_receipt_sha256": loaded_seed["success_receipt_sha256"],
        }
        if dict(source) != expected_source:
            raise ValueError(f"{path}:source provenance differs from the referenced seed bytes")
        if loaded_seed["environment_identity"] != environment:
            raise ValueError(f"{path}:source seed producer identity is inconsistent")
        loaded[expected_seed] = loaded_seed

    (
        _,
        _,
        all_length_passed,
        _,
        expected_selection,
    ) = _selection_from_loaded_seeds(
        pilot_phase=str(pilot_phase),
        design=design,
        declared_seeds=PHASE2_PILOT_SEEDS,
        loaded=loaded,
        source_binding=source_binding,
    )
    if aggregate["selection"] != expected_selection:
        raise ValueError(f"{path}:selection differs from recomputed source evidence")
    expected_per_seed = {
        str(seed): {
            "beta_common": loaded[seed]["beta_common"],
            "pre_oracle_safety_passed": loaded[seed]["safety_passed"],
            "observed_by_arm": loaded[seed]["observed_by_arm"],
        }
        for seed in PHASE2_PILOT_SEEDS
    }
    if aggregate["per_seed"] != expected_per_seed:
        raise ValueError(f"{path}:per_seed differs from recomputed source evidence")
    expected_horizon = {
        "schema_version": "pilot-horizon-selection/v1",
        "candidate_horizon_tokens": design.max_response_tokens,
        "allowed_horizon_sequence": list(design.allowed_horizon_sequence),
        "horizon_grid_index": design.horizon_grid_index,
        "parent_pilot_aggregate_sha256": design.parent_pilot_aggregate_sha256,
        "previous_horizon_failed_length_gate": design.previous_horizon_failed_length_gate,
        "all_seed_length_gates_passed": all_length_passed,
        "parent_binding_verified": (
            design.parent_pilot_aggregate_sha256 is None or horizon_binding is not None
        ),
    }
    if aggregate["horizon"] != expected_horizon:
        raise ValueError(f"{path}:horizon differs from recomputed source evidence")
    _validate_aggregate_information_boundary(
        aggregate["information_boundary"],
        name=f"{path}:information_boundary",
    )


def _load_source_aggregate(
    raw_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    _cache: dict[tuple[Path, str], Mapping[str, object]] | None = None,
    _active: set[Path] | None = None,
) -> tuple[Path, str, Mapping[str, object]]:
    path = Path(raw_path).resolve()
    raw = _read_regular(path, name="pilot predecessor aggregate", max_bytes=64 * 1024 * 1024)
    observed_sha = _sha256_bytes(raw)
    if observed_sha != expected_sha256:
        raise ValueError("pilot predecessor aggregate SHA256 differs from the design identity")
    cache = {} if _cache is None else _cache
    active = set() if _active is None else _active
    key = (path, observed_sha)
    if key in cache:
        return path, observed_sha, cache[key]
    if path in active:
        raise ValueError("pilot predecessor aggregate graph contains a cycle")
    value = _strict_json_bytes(raw, path=path)
    active.add(path)
    try:
        _validate_phase2_pilot_aggregate(
            path,
            value,
            cache=cache,
            active=active,
        )
    finally:
        active.remove(path)
    cache[key] = value
    return path, observed_sha, value


def _power_of_two_grid_index(value: float, base: float) -> int:
    if value <= 0.0 or base <= 0.0 or value < base:
        raise ValueError("frozen pilot beta is outside the preregistered beta*=2 grid")
    ratio = value / base
    exponent = round(math.log2(ratio))
    if exponent < 0 or not _close(ratio, 2.0**exponent, tolerance=1.0e-12):
        raise ValueError("frozen pilot beta is outside the preregistered beta*=2 grid")
    return exponent


def _beta_source_binding_for_design(
    design: Phase2Design,
    *,
    expected_source_config_hash: str,
    predecessor: tuple[Path, str, Mapping[str, object]] | None,
) -> dict[str, object] | None:
    expected_sha = design.beta_source_aggregate_sha256
    if design.pilot_phase == "calibration":
        if predecessor is not None or expected_sha is not None:
            raise ValueError("pilot calibration must not consume a beta source aggregate")
        return None
    if predecessor is None or expected_sha is None:
        raise ValueError("this design requires its identity-bound beta source aggregate")

    path, observed_sha, value = predecessor
    if observed_sha != expected_sha:
        raise ValueError("pilot predecessor aggregate SHA256 differs from the design identity")
    if (
        value.get("formal_eligibility") is not False
        or value.get("supports_formal_claim") is not False
        or value.get("evidence_role") != "target_free_design_selection_only"
        or (
            design.pilot_phase == "freeze"
            and value.get("source_config_hash") != expected_source_config_hash
        )
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


def verify_beta_source_aggregate(
    overlay_config: Mapping[str, object],
    source_aggregate: str | os.PathLike[str] | None,
) -> dict[str, object] | None:
    """Verify the predecessor selection record bound into a freeze/formal design."""

    config = validate_phase2_config(overlay_config)
    design = Phase2Design.from_phase2_config(config)
    design_config = _mapping(config["design"], name="design")
    expected_sha = design.beta_source_aggregate_sha256
    if source_aggregate is None:
        predecessor = None
    elif expected_sha is None:
        raise ValueError("pilot calibration must not consume a beta source aggregate")
    else:
        predecessor = _load_source_aggregate(
            source_aggregate,
            expected_sha256=expected_sha,
        )
    return _beta_source_binding_for_design(
        design,
        expected_source_config_hash=str(design_config["source_config_hash"]),
        predecessor=predecessor,
    )


def _horizon_parent_binding_for_design(
    design: Phase2Design,
    *,
    predecessor: tuple[Path, str, Mapping[str, object]] | None,
) -> dict[str, object] | None:
    expected_sha = design.parent_pilot_aggregate_sha256
    if expected_sha is None:
        if predecessor is not None:
            raise ValueError("the initial calibration horizon has no parent aggregate")
        return None
    if predecessor is None:
        raise ValueError("this horizon requires its identity-bound parent pilot aggregate")

    path, observed_sha, value = predecessor
    if observed_sha != expected_sha:
        raise ValueError("pilot predecessor aggregate SHA256 differs from the design identity")
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


def verify_horizon_parent_aggregate(
    overlay_config: Mapping[str, object],
    parent_aggregate: str | os.PathLike[str] | None,
) -> dict[str, object] | None:
    """Verify the aggregate that authorized the current response horizon."""

    config = validate_phase2_config(overlay_config)
    design = Phase2Design.from_phase2_config(config)
    expected_sha = design.parent_pilot_aggregate_sha256
    if parent_aggregate is None:
        predecessor = None
    elif expected_sha is None:
        raise ValueError("the initial calibration horizon has no parent aggregate")
    else:
        predecessor = _load_source_aggregate(
            parent_aggregate,
            expected_sha256=expected_sha,
        )
    return _horizon_parent_binding_for_design(
        design,
        predecessor=predecessor,
    )


def build_phase2_pilot_aggregate(
    overlay_config: Mapping[str, object],
    result_jsons: Sequence[str | os.PathLike[str]],
    *,
    aggregation_identity: Mapping[str, object],
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
    policy = _mapping(config["policy"], name="policy")
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
            materialized_prompts=int(run["num_prompts"]),
            prompts=int(split_sizes["test"]),
            candidates=int(data["num_candidates"]),
            max_prompt_tokens=int(policy["max_prompt_tokens"]),
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
    validated_aggregation_identity = _validate_aggregation_identity(
        aggregation_identity,
        producer_environment=_mapping(
            environment,
            name="shared seed producer environment identity",
        ),
        name="aggregation_identity",
        require_current_validator=True,
    )

    source_binding = verify_beta_source_aggregate(config, beta_source_aggregate)
    parent_input = (
        beta_source_aggregate
        if horizon_parent_aggregate is None and design.pilot_phase == "freeze"
        else horizon_parent_aggregate
    )
    horizon_binding = verify_horizon_parent_aggregate(config, parent_input)
    _, _, all_length_passed, _, selection = _selection_from_loaded_seeds(
        pilot_phase=str(pilot_phase),
        design=design,
        declared_seeds=declared_seeds,
        loaded=loaded,
        source_binding=source_binding,
    )
    base = Path(reference_base).resolve() if reference_base is not None else Path.cwd().resolve()
    predecessors = {
        "beta_source_aggregate": (
            None
            if source_binding is None
            else {
                "path": relative_posix_reference(source_binding["path"], base=base),
                "sha256": source_binding["sha256"],
            }
        ),
        "horizon_parent_aggregate": (
            None
            if horizon_binding is None
            else {
                "path": relative_posix_reference(horizon_binding["path"], base=base),
                "sha256": horizon_binding["sha256"],
            }
        ),
    }
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
            "artifact_metadata": relative_posix_reference(
                loaded[seed]["artifact_metadata_path"],
                base=base,
            ),
            "artifact_metadata_sha256": loaded[seed]["artifact_metadata_sha256"],
            "run_manifest": relative_posix_reference(
                loaded[seed]["run_manifest_path"],
                base=base,
            ),
            "run_manifest_sha256": loaded[seed]["run_manifest_sha256"],
            "output_verification": relative_posix_reference(
                loaded[seed]["output_verification_path"],
                base=base,
            ),
            "output_verification_sha256": loaded[seed]["output_verification_sha256"],
            "success_receipt": relative_posix_reference(
                loaded[seed]["success_receipt_path"],
                base=base,
            ),
            "success_receipt_sha256": loaded[seed]["success_receipt_sha256"],
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
        "phase2_runtime_contract": runtime,
        "phase2_runtime_contract_sha256": runtime_sha,
        "beta_source_aggregate_sha256": design.beta_source_aggregate_sha256,
        "seeds": list(declared_seeds),
        "environment_identity": environment,
        "aggregation_identity": validated_aggregation_identity,
        "rollout_geometry": {
            "materialized_prompts": int(run["num_prompts"]),
            "test_prompts": int(split_sizes["test"]),
            "candidates_per_prompt": int(data["num_candidates"]),
            "max_prompt_tokens": int(policy["max_prompt_tokens"]),
        },
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
        "predecessors": predecessors,
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
    aggregation_identity: Mapping[str, object],
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
        aggregation_identity=aggregation_identity,
        reference_base=destination.parent,
        beta_source_aggregate=beta_source_aggregate,
        horizon_parent_aggregate=horizon_parent_aggregate,
    )
    atomic_write_json(destination, payload, overwrite=False)
    return payload


__all__ = [
    "PHASE2_PILOT_AGGREGATE_SCHEMA",
    "PHASE2_PILOT_AGGREGATION_IDENTITY_SCHEMA",
    "build_phase2_pilot_aggregate",
    "verify_beta_source_aggregate",
    "verify_horizon_parent_aggregate",
    "write_phase2_pilot_aggregate",
]
