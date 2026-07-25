"""Per-seed verification for fresh post-recovery calibration outputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .phase2_aggregate import validate_post_recovery_pilot_head_training
from .phase2_config import load_phase2_config_bundle
from .phase2_post_recovery_control import (
    OPTIMIZER_SCHEDULE_SHA256,
    ORDERED_SEEDS,
    POST_RECOVERY_OUTPUT_VERIFICATION_SCHEMA,
    POST_RECOVERY_PILOT_PHASES,
    verify_recovery_authorization_config_binding,
)
from .repro import atomic_write_json

_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_VECTOR_KEYS = frozenset(
    {
        "head_weight",
        "head_weights",
        "direction",
        "natural_direction",
        "displacement",
        "oracle_displacement",
        "moment",
        "operator_direction",
        "projection_matrix",
        "true_rewards",
    }
)
_HEAD_NAMES = (
    "primary_bt_mle",
    "primary_prorm_plus",
    "low_dimensional_prorm_plus",
    "exact_margin_prorm_plus",
    "exact_soft_label_bt",
)


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(item: str) -> object:
    raise ValueError(f"non-finite JSON constant {item!r}")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _digest(value: object, *, name: str, lengths: frozenset[int] = frozenset({64})) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_nested_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_nested_keys(item))
        return keys
    return set()


def verify_and_write_post_recovery_output(
    *,
    overlay_path: str | os.PathLike[str],
    authorization_path: str | os.PathLike[str],
    authorization_sha256: str,
    result_path: str | os.PathLike[str],
    diagnostics_path: str | os.PathLike[str],
    phase2_output_verification_path: str | os.PathLike[str],
    post_recovery_output_verification_path: str | os.PathLike[str],
    seed: int,
    expected_design_sha256: str,
    expected_base_config_hash: str,
    expected_git_commit: str,
    expected_image_sha256: str,
    expected_hf_inventory_sha256: str,
    expected_artifact_metadata_sha256: str,
    expected_slurm_job_id_raw: str,
    expected_array_job_id: str,
    expected_array_task_id: int,
    expected_pilot_phase: str,
) -> dict[str, object]:
    """Verify one target-free result and publish old-reader plus strict receipts."""

    if seed not in ORDERED_SEEDS:
        raise ValueError("seed is not in the locked post-recovery calibration order")
    for value, name, lengths in (
        (authorization_sha256, "authorization_sha256", frozenset({64})),
        (expected_design_sha256, "expected_design_sha256", frozenset({64})),
        (expected_base_config_hash, "expected_base_config_hash", frozenset({64})),
        (expected_git_commit, "expected_git_commit", frozenset({40, 64})),
        (expected_image_sha256, "expected_image_sha256", frozenset({64})),
        (
            expected_hf_inventory_sha256,
            "expected_hf_inventory_sha256",
            frozenset({64}),
        ),
        (
            expected_artifact_metadata_sha256,
            "expected_artifact_metadata_sha256",
            frozenset({64}),
        ),
    ):
        _digest(value, name=name, lengths=lengths)
    if re.fullmatch(r"[1-9][0-9]*", expected_slurm_job_id_raw) is None:
        raise ValueError("expected_slurm_job_id_raw must be a positive integer")
    if re.fullmatch(r"[1-9][0-9]*", expected_array_job_id) is None:
        raise ValueError("expected_array_job_id must be a positive integer")
    if expected_pilot_phase not in POST_RECOVERY_PILOT_PHASES:
        raise ValueError("expected_pilot_phase must be calibration or freeze")
    if (
        isinstance(expected_array_task_id, bool)
        or not isinstance(expected_array_task_id, int)
        or not 0 <= expected_array_task_id < len(ORDERED_SEEDS)
        or ORDERED_SEEDS[expected_array_task_id] != seed
    ):
        raise ValueError("array task/seed mapping is not the locked calibration order")
    slurm_array_task_job_id = f"{expected_array_job_id}_{expected_array_task_id}"
    binding = verify_recovery_authorization_config_binding(
        authorization_path,
        overlay_path,
        expected_sha256=authorization_sha256,
        expected_pilot_phase=expected_pilot_phase,
    )
    if (
        binding["phase2_design_sha256"] != expected_design_sha256
        or binding["base_config_hash"] != expected_base_config_hash
    ):
        raise ValueError("post-recovery config identity differs from the job binding")
    config = load_phase2_config_bundle(overlay_path).config
    design = _mapping(config.get("design"), name="post-recovery design")
    if design.get("stage") != "pilot" or design.get("pilot_phase") != expected_pilot_phase:
        raise ValueError("post-recovery config pilot phase differs from the job binding")
    result_file = Path(result_path)
    diagnostics_file = Path(diagnostics_path)
    result = _load_json(result_file, name="post-recovery Phase-2 result")
    if _FORBIDDEN_VECTOR_KEYS.intersection(_nested_keys(result)):
        raise ValueError("post-recovery result contains a serialized training vector")
    boundary = _mapping(result.get("information_boundary"), name="information_boundary")
    environment = _mapping(result.get("environment_identity"), name="environment_identity")
    current = _mapping(
        result.get("current_process_identity"),
        name="current_process_identity",
    )
    training = _mapping(result.get("head_training"), name="head_training")
    audit = _mapping(training.get("audit"), name="head_training.audit")
    rescore = _mapping(result.get("train_oracle_rescore"), name="train_oracle_rescore")
    if (
        result.get("schema_version") != "common-beta-pilot-diagnostics/v2"
        or result.get("design_stage") != "pilot"
        or result.get("pilot_phase") != expected_pilot_phase
        or result.get("formal_eligibility") is not False
        or result.get("per_seed_supports_formal_claim") is not False
        or result.get("source_config_hash") != expected_base_config_hash
        or result.get("phase2_design_sha256") != expected_design_sha256
        or result.get("seed") != seed
        or result.get("artifact_dir") != "artifact"
        or result.get("artifact_metadata_sha256") != expected_artifact_metadata_sha256
        or result.get("diagnostics_jsonl") != diagnostics_file.name
        or result.get("run_manifest") != "run-manifest.json"
        or environment != current
        or environment.get("formal") is not True
        or environment.get("git_commit") != expected_git_commit
        or environment.get("image_sha256") != expected_image_sha256
        or environment.get("hf_inventory_sha256") != expected_hf_inventory_sha256
        or environment.get("account") != "sigroup"
        or environment.get("partition") != "gpu-l20"
        or environment.get("gpu_models") != ["NVIDIA L20"]
        or rescore.get("raw_oracle_logits_serialized") is not False
        or training.get("training_design_sha256") != expected_design_sha256
        or training.get("head_weights_serialized") is not False
        or training.get("old_phase1_comparison_heads_reused") is not False
        or training.get("test_data_accessed") is not False
        or training.get("source") != "trained_after_train_oracle_rescore"
        or audit.get("schema_version") != "phase2-fresh-head-training/v3"
        or audit.get("training_design_sha256") != expected_design_sha256
    ):
        raise ValueError("post-recovery result identity or fresh-training contract is invalid")
    isolation = _mapping(audit.get("isolation"), name="head_training.audit.isolation")
    if (
        isolation.get("primary_heads_are_fresh_zero_initialized") is not True
        or isolation.get("old_phase1_comparison_heads_used") is not False
        or isolation.get("test_data_accessed") is not False
    ):
        raise ValueError("post-recovery training isolation evidence is invalid")
    for key in (
        "final_oracle_session_opened",
        "rollout_responses_oracle_scored",
        "heldout_evaluator_called",
        "oracle_outcomes_serialized",
        "prompt_or_response_text_serialized",
        "token_ids_or_response_masks_serialized",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"post-recovery result crossed information boundary {key}")

    train_oracle_reward_sha256 = _digest(
        rescore.get("transformed_rewards_sha256"),
        name="train_oracle_rescore.transformed_rewards_sha256",
    )
    deep_gate = validate_post_recovery_pilot_head_training(
        training,
        config=config,
        design_sha256=expected_design_sha256,
        seed=seed,
        train_oracle_reward_sha256=train_oracle_reward_sha256,
        name="head_training",
    )
    if (
        deep_gate.get("passed") is not True
        or deep_gate.get("five_head_adopted_schedule_verified") is not True
        or deep_gate.get("vectors_redacted") is not True
        or deep_gate.get("adopted_optimizer_schedule_sha256") != OPTIMIZER_SCHEDULE_SHA256
    ):
        raise ValueError("post-recovery deep five-head gate did not pass")
    deep_head_checks = _mapping(
        deep_gate.get("five_head_training"),
        name="post-recovery deep gate.five_head_training",
    )
    if tuple(deep_head_checks) != _HEAD_NAMES:
        raise ValueError("post-recovery result does not expose the exact five trainer audits")
    head_checks = {name: dict(_mapping(deep_head_checks[name], name=name)) for name in _HEAD_NAMES}

    if not diagnostics_file.is_file() or diagnostics_file.is_symlink():
        raise ValueError("post-recovery diagnostics sidecar is missing or unsafe")
    diagnostics_sha256 = _sha256_file(diagnostics_file)
    if result.get("diagnostics_sha256") != diagnostics_sha256:
        raise ValueError("post-recovery diagnostics SHA256 differs from the result")
    records_list: list[Mapping[str, object]] = []
    with diagnostics_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"post-recovery diagnostics row {line_number} is blank")
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_pairs,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"post-recovery diagnostics row {line_number} is not strict JSON"
                ) from error
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != "common-beta-pilot-diagnostic-row/v2"
                or row.get("pilot_phase") != expected_pilot_phase
                or any(
                    row.get(key) is not False
                    for key in (
                        "contains_prompt_text",
                        "contains_response_text",
                        "contains_token_ids",
                        "contains_oracle_outcome",
                    )
                )
                or _FORBIDDEN_VECTOR_KEYS.intersection(_nested_keys(row))
            ):
                raise ValueError(f"post-recovery diagnostics row {line_number} is not target-free")
            records_list.append(row)
    if not records_list:
        raise ValueError("post-recovery diagnostics sidecar is empty")

    # Reuse the authoritative pilot sidecar validator so this strict receipt
    # proves the exact row schema, cardinality, ordering, CRN coordinates,
    # horizon, beta, and KL semantics instead of merely counting records.
    from .phase2_pilot_aggregate import _sidecar_summary

    beta_key = (
        "train_only_global_beta_calibration_candidate"
        if expected_pilot_phase == "calibration"
        else "pilot_fixed_global_beta_rehearsal"
    )
    beta_evidence = _mapping(result.get(beta_key), name=beta_key)
    beta_field = "candidate_beta" if expected_pilot_phase == "calibration" else "beta_common"
    beta_value = beta_evidence.get(beta_field)
    if (
        isinstance(beta_value, bool)
        or not isinstance(beta_value, (int, float))
        or not float("-inf") < float(beta_value) < float("inf")
        or float(beta_value) <= 0.0
    ):
        raise ValueError("post-recovery result beta evidence is invalid")
    run = _mapping(config.get("run"), name="run")
    split_sizes = _mapping(run.get("split_sizes"), name="run.split_sizes")
    data = _mapping(config.get("data"), name="data")
    policy = _mapping(config.get("policy"), name="policy")
    _sidecar_summary(
        records_list,
        pilot_phase=expected_pilot_phase,
        beta_common=float(beta_value),
        prompts=int(split_sizes["test"]),
        candidates=int(data["num_candidates"]),
        max_response_tokens=int(policy["max_response_tokens"]),
        name="post-recovery diagnostics",
    )
    records = len(records_list)

    measured = _mapping(result.get("measured_kl_safety"), name="measured_kl_safety")
    pre_oracle = _mapping(
        result.get("pre_oracle_safety_gate"),
        name="pre_oracle_safety_gate",
    )
    old_reader_receipt = {
        "schema_version": "prorm-phase2-output-verification/v1",
        "status": "passed",
        "seed": seed,
        "source_config_hash": expected_base_config_hash,
        "phase2_design_sha256": expected_design_sha256,
        "pilot_phase": expected_pilot_phase,
        "diagnostic_records": records,
        "diagnostics_sha256": diagnostics_sha256,
        "kl_gate_passed": measured.get("passed"),
        "kl_measure_only": True,
        "kl_violations": measured.get("violations"),
        "pre_oracle_gate_passed": pre_oracle.get("passed"),
        "pre_oracle_violations": pre_oracle.get("violations"),
        "environment_identity": dict(environment),
    }
    phase2_receipt_path = Path(phase2_output_verification_path)
    strict_receipt_path = Path(post_recovery_output_verification_path)
    if (
        phase2_receipt_path.exists()
        or phase2_receipt_path.is_symlink()
        or strict_receipt_path.exists()
        or strict_receipt_path.is_symlink()
    ):
        raise FileExistsError("refusing to overwrite post-recovery output verification")
    atomic_write_json(phase2_receipt_path, old_reader_receipt, overwrite=False)
    try:
        strict_receipt = {
            "schema_version": POST_RECOVERY_OUTPUT_VERIFICATION_SCHEMA,
            "status": "passed",
            "pilot_phase": expected_pilot_phase,
            "slurm_job_id_raw": expected_slurm_job_id_raw,
            "allocation_job_id_raw": expected_slurm_job_id_raw,
            "slurm_array_task_job_id": slurm_array_task_job_id,
            "array_job_id": expected_array_job_id,
            "array_task_id": str(expected_array_task_id),
            "seed": seed,
            "phase2_design_sha256": expected_design_sha256,
            "source_config_hash": expected_base_config_hash,
            "result_sha256": _sha256_file(result_file),
            "phase2_output_verification_sha256": _sha256_file(phase2_receipt_path),
            "diagnostics_sha256": diagnostics_sha256,
            "recovery_authorization_sha256": authorization_sha256,
            "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
            "materialization_mode": "fresh",
            "recovery_outputs_reused": False,
            "five_head_adopted_schedule_verified": True,
            "five_head_training": head_checks,
            "target_free_information_boundary_verified": True,
        }
        atomic_write_json(strict_receipt_path, strict_receipt, overwrite=False)
    except BaseException:
        # The two receipts form one publication unit.  The strict receipt is
        # written second, so a failed second publication must not leave the
        # legacy-looking success receipt behind.  atomic_write_json itself
        # guarantees that a failed no-overwrite publication never replaces an
        # existing strict path.
        try:
            phase2_receipt_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            raise RuntimeError(
                "strict receipt publication failed and legacy receipt cleanup failed"
            ) from cleanup_error
        raise
    return strict_receipt


__all__ = ["verify_and_write_post_recovery_output"]
