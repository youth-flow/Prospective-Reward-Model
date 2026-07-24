from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from smart_reward.phase2_config import (
    PHASE2_PILOT_SEEDS,
    load_phase2_config,
    phase2_design_identity,
)
from smart_reward.phase2_pilot_aggregate import (
    PHASE2_PILOT_AGGREGATE_SCHEMA,
    build_phase2_pilot_aggregate,
    verify_beta_source_aggregate,
    write_phase2_pilot_aggregate,
)
from smart_reward.phase2_rollout import (
    PHASE2_ARM_ORDER,
    PHASE2_PILOT_DIAGNOSTIC_SCHEMA,
    PHASE2_PILOT_RESULT_SCHEMA,
    Phase2Design,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "configs" / "common_beta_pilot.yaml"
THRESHOLDS = {
    "mean_policy_to_reference_kl_cap": 0.02,
    "prompt_mean_p95_kl_cap": 0.02,
    "prompt_mean_p99_kl_cap": 0.05,
    "prompt_mean_maximum_kl_cap": 0.10,
    "per_sequence_maximum_kl_cap": 0.20,
    "reached_max_length_rate_cap": 0.05,
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_config(
    pilot: dict[str, Any],
    *,
    beta: float,
    source_sha256: str,
    horizon_parent_sha256: str | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(pilot)
    config["design"].update(
        {
            "name": "common-beta-pilot-freeze-test",
            "pilot_phase": "freeze",
        }
    )
    config["objective"]["common_beta"].update(
        {
            "rule": "pilot_fixed_global_beta_target_free_safety_rehearsal",
            "calibration_split": "excluded_pilot_calibration",
            "calibration_source": (
                "frozen_calibration_aggregate_candidate_in_pilot_freeze_design_identity"
            ),
            "frozen_global_beta": beta,
            "beta_source_aggregate_sha256": source_sha256,
            "sensitivity_k_cal": None,
            "sensitivity_frozen_global_beta_multipliers": None,
            "primary_execution_role": "pilot_frozen_global_beta_safety_rehearsal",
            "sensitivity_execution_role": ("new_pilot_freeze_design_identity_double_beta_grid"),
        }
    )
    config["evaluation"]["decision_gates"]["application"] = (
        "pilot_freeze_target_free_safety_selection"
    )
    config["evaluation"]["max_length"].update(
        {
            "role": "pilot_frozen_global_beta_safety_selection",
            "measure_only": True,
            "formal_gate": False,
            "formal_threshold": 0.05,
            "parent_pilot_aggregate_sha256": (
                source_sha256 if horizon_parent_sha256 is None else horizon_parent_sha256
            ),
            "post_pilot_requirement": (
                "freeze_confirmatory_identity_if_passed_else_double_beta_new_identity"
            ),
        }
    )
    return config


def _escalated_calibration_config(
    pilot: dict[str, Any],
    *,
    horizon: int,
    horizon_grid_index: int,
    parent_sha256: str,
) -> dict[str, Any]:
    config = copy.deepcopy(pilot)
    config["design"]["name"] = f"common-beta-pilot-calibration-h{horizon}"
    config["policy"]["max_response_tokens"] = horizon
    config["evaluation"]["max_length"].update(
        {
            "candidate_horizon_tokens": horizon,
            "horizon_grid_index": horizon_grid_index,
            "parent_pilot_aggregate_sha256": parent_sha256,
            "previous_horizon_failed_length_gate": True,
        }
    )
    return config


def _environment() -> dict[str, object]:
    return {
        "formal": True,
        "git_commit": "a" * 40,
        "image_sha256": "b" * 64,
        "hf_inventory_sha256": "c" * 64,
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_models": ["NVIDIA L20"],
    }


def _observed(kl: float, reached_max_length_rate: float = 0.0) -> dict[str, float]:
    return {
        "mean_policy_to_reference_kl": kl,
        "prompt_mean_p95_kl": kl,
        "prompt_mean_p99_kl": kl,
        "prompt_mean_maximum_kl": kl,
        "per_sequence_maximum_kl": kl,
        "reached_max_length_rate": reached_max_length_rate,
    }


def _violations(
    kl_by_arm: dict[str, float],
    reached_by_arm: dict[str, float],
) -> list[str]:
    return [
        f"{arm}:{metric}"
        for arm in PHASE2_ARM_ORDER
        for metric in (
            "mean_policy_to_reference_kl",
            "prompt_mean_p95_kl",
            "prompt_mean_p99_kl",
            "prompt_mean_maximum_kl",
            "per_sequence_maximum_kl",
            "reached_max_length_rate",
        )
        if _observed(kl_by_arm[arm], reached_by_arm[arm])[metric] > THRESHOLDS[f"{metric}_cap"]
    ]


def _seed_result(
    directory: Path,
    config: dict[str, Any],
    *,
    seed: int,
    beta: float,
    unsafe: bool = False,
    length_unsafe: bool = False,
) -> Path:
    pilot_phase = str(config["design"]["pilot_phase"])
    design = Phase2Design.from_phase2_config(config)
    design_sha = phase2_design_identity(config)
    sidecar_path = directory / f"seed-{seed}.diagnostics.jsonl"
    kl_by_arm = {
        "zero_b": 0.0,
        "bt_mle": 0.005,
        "prorm_plus": 0.03 if unsafe else 0.006,
        "oracle_step": 0.007,
    }
    reached_by_arm = {
        arm: (1.0 if length_unsafe and arm == "zero_b" else 0.0) for arm in PHASE2_ARM_ORDER
    }
    rows: list[dict[str, object]] = []
    for arm in PHASE2_ARM_ORDER:
        for prompt_index in range(256):
            for candidate_index in range(4):
                rows.append(
                    {
                        "schema_version": PHASE2_PILOT_DIAGNOSTIC_SCHEMA,
                        "pilot_phase": pilot_phase,
                        "arm": arm,
                        "beta_common": beta,
                        "beta_role": (
                            "seed_calibration_candidate"
                            if pilot_phase == "calibration"
                            else "frozen_global_beta_candidate"
                        ),
                        "prompt_id": f"prompt-{prompt_index}",
                        "candidate_index": candidate_index,
                        "response_token_count": (
                            design.max_response_tokens if reached_by_arm[arm] == 1.0 else 10
                        ),
                        "terminated_by_eos": reached_by_arm[arm] == 0.0,
                        "reached_max_length": reached_by_arm[arm] == 1.0,
                        "prompt_rollout_seed": 100000 + prompt_index,
                        "kl_orientation": "pi_updated_to_pi0",
                        "kl_history_source": "updated_policy",
                        "on_policy_kl_pi_updated_to_pi0": kl_by_arm[arm],
                        "contains_prompt_text": False,
                        "contains_response_text": False,
                        "contains_token_ids": False,
                        "contains_oracle_outcome": False,
                    }
                )
    sidecar_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    arms: dict[str, object] = {}
    for arm in PHASE2_ARM_ORDER:
        kl = kl_by_arm[arm]
        reached = reached_by_arm[arm]
        response_tokens = design.max_response_tokens if reached == 1.0 else 10
        arms[arm] = {
            "deployment_hashes": {
                "beta_common": beta,
                "displacement_sha256": hashlib.sha256(f"{seed}:{arm}".encode()).hexdigest(),
                "direction_evidence_sha256": None if arm == "zero_b" else "d" * 64,
                "common_beta_evidence_sha256": None if arm == "zero_b" else "e" * 64,
            },
            "rollout_length": {
                "num_trajectories": 1024,
                "terminated_by_eos_count": 0 if reached == 1.0 else 1024,
                "terminated_by_eos_rate": 1.0 - reached,
                "reached_max_length_count": 1024 if reached == 1.0 else 0,
                "reached_max_length_rate": reached,
                "response_token_count": {
                    "mean": float(response_tokens),
                    "minimum": response_tokens,
                    "maximum": response_tokens,
                },
            },
            "mean_on_policy_kl_pi_updated_to_pi0": kl,
            "on_policy_kl_tail": {
                "schema_version": "on-policy-kl-tail-summary/v1",
                "unit": "prompt_mean_over_candidates",
                "num_prompts": 256,
                "candidates_per_prompt": 4,
                "mean": kl,
                "p50": kl,
                "p90": kl,
                "p95": kl,
                "p99": kl,
                "maximum": kl,
                "per_sequence_maximum": kl,
                "pilot_selection_role": "locality_tail_measurement",
                "formal_gate_applied": False,
            },
        }
    violations = _violations(kl_by_arm, reached_by_arm)
    mean_violations = sorted(
        arm
        for arm, value in kl_by_arm.items()
        if value > THRESHOLDS["mean_policy_to_reference_kl_cap"]
    )
    curvature = 2.0 * design.target_oracle_quadratic_kl * beta * beta
    if pilot_phase == "calibration":
        beta_key = "train_only_global_beta_calibration_candidate"
        beta_evidence: dict[str, object] = {
            "schema_version": "global-beta-calibration-candidate/v1",
            "rule": design.common_beta_rule,
            "candidate_beta": beta,
            "frozen_global_beta": None,
            "oracle_natural_curvature": curvature,
            "target_oracle_quadratic_kl": design.target_oracle_quadratic_kl,
            "predicted_oracle_quadratic_kl": design.target_oracle_quadratic_kl,
            "calibration_split": "train_only",
            "formal_beta_selected": False,
            "formal_selection_rule": (
                "maximum_pilot_seed_candidate_then_smallest_passing_frozen_kl_only_grid"
            ),
            "learner_specific_rescaling": False,
        }
    else:
        beta_key = "pilot_fixed_global_beta_rehearsal"
        beta_evidence = {
            "schema_version": "pilot-frozen-global-beta-rehearsal/v1",
            "rule": design.common_beta_rule,
            "beta_common": beta,
            "frozen_global_beta": beta,
            "beta_matches_frozen_global_beta": True,
            "beta_source_aggregate_sha256": design.beta_source_aggregate_sha256,
            "current_seed_oracle_natural_curvature": curvature,
            "reference_target_oracle_quadratic_kl": design.target_oracle_quadratic_kl,
            "predicted_current_seed_oracle_quadratic_kl": (design.target_oracle_quadratic_kl),
            "current_seed_curvature_role": "predicted_kl_diagnostic_only",
            "beta_selected_from_current_seed_curvature": False,
            "frozen_in_phase2_design_identity": True,
            "learner_specific_rescaling": False,
            "post_evaluation_retuning": False,
        }
    environment = _environment()
    result_path = directory / f"seed-{seed}.json"
    payload = {
        "schema_version": PHASE2_PILOT_RESULT_SCHEMA,
        "design_stage": "pilot",
        "pilot_phase": pilot_phase,
        "formal_eligibility": False,
        "evidence_role": "optimization_horizon_and_kl_design_selection_only",
        "per_seed_supports_formal_claim": False,
        "source_config_hash": config["design"]["source_config_hash"],
        "phase2_design_sha256": design_sha,
        "phase2_runtime_contract": design.to_dict(),
        "phase2_runtime_contract_sha256": design.sha256,
        "seed": seed,
        "artifact_dir": f"artifact-{seed}",
        "diagnostics_jsonl": sidecar_path.name,
        "artifact_metadata_sha256": "1" * 64,
        "run_manifest": f"manifest-{seed}.json",
        "run_manifest_sha256": "2" * 64,
        "environment_identity": environment,
        "current_process_identity": environment,
        "train_oracle_rescore": {
            "raw_oracle_logits_serialized": False,
            "frozen_transform": {"b": -4.500244140625, "tau": 2.7715682983398438},
        },
        "head_training": {
            "training_design_sha256": design_sha,
            "head_weights_serialized": False,
            "old_phase1_comparison_heads_reused": False,
            "test_data_accessed": False,
        },
        "deployment_hashes": {arm: arms[arm]["deployment_hashes"] for arm in PHASE2_ARM_ORDER},
        "measured_kl_safety": {
            "schema_version": "measured-kl-safety/v1",
            "cap": 0.02,
            "passed": not mean_violations,
            "measured_by_policy": dict(sorted(kl_by_arm.items())),
            "violations": mean_violations,
            "beta_retuned": False,
        },
        "pre_oracle_safety_gate": {
            "schema_version": "phase2-pre-oracle-safety-gate/v1",
            "design_stage": "pilot",
            "pilot_phase": pilot_phase,
            "measure_only": True,
            "formal_gate": False,
            "thresholds": THRESHOLDS,
            "observed_by_arm": {
                arm: _observed(kl_by_arm[arm], reached_by_arm[arm]) for arm in PHASE2_ARM_ORDER
            },
            "violations": violations,
            "passed": not violations,
            "beta_retuned": False,
            "on_violation": "publish_target_free_diagnostics_without_final_oracle",
        },
        "pilot_kl_safety_gate": {
            "gate_passed": not mean_violations,
            "measure_only": True,
        },
        "arms": arms,
        "information_boundary": {
            "new_rollout_prompts_used_for_calibration": False,
            "final_oracle_session_opened": False,
            "rollout_responses_oracle_scored": False,
            "heldout_evaluator_called": False,
            "oracle_outcomes_serialized": False,
            "prompt_or_response_text_serialized": False,
            "token_ids_or_response_masks_serialized": False,
            "source_artifact_heldout_targets_exposed_by_phase2_prepared_inputs": False,
        },
        "common_random_numbers": {"candidate_index_alignment": True},
        "memory_schedule": ["stop_before_final_oracle_and_heldout_evaluation"],
        "policy_and_oracle_co_resident": False,
        "learner_specific_line_search": False,
        "diagnostics_sha256": _sha256(sidecar_path),
        beta_key: beta_evidence,
    }
    _write_json(result_path, payload)
    return result_path


def _calibration_campaign(tmp_path: Path) -> tuple[dict[str, Any], list[Path]]:
    config = load_phase2_config(PILOT_PATH)
    results = [
        _seed_result(
            tmp_path,
            config,
            seed=seed,
            beta=beta,
        )
        for seed, beta in zip(sorted(PHASE2_PILOT_SEEDS), (1.5, 2.0, 1.75), strict=True)
    ]
    return config, results


def test_calibration_aggregate_selects_maximum_and_remains_target_free(
    tmp_path: Path,
) -> None:
    config, results = _calibration_campaign(tmp_path)
    payload = build_phase2_pilot_aggregate(
        config,
        results,
        reference_base=tmp_path,
    )

    assert payload["schema_version"] == PHASE2_PILOT_AGGREGATE_SCHEMA
    assert payload["pilot_phase"] == "calibration"
    assert payload["formal_eligibility"] is False
    assert payload["supports_formal_claim"] is False
    assert payload["selection"]["recommended_pilot_freeze_beta"] == 2.0
    assert payload["selection"]["freeze_validation_required"] is True
    assert payload["selection"]["selection_accepted"] is None
    assert payload["information_boundary"]["oracle_outcomes_consumed"] is False


def test_failed_length_gate_requires_next_horizon_calibration_with_parent_hash(
    tmp_path: Path,
) -> None:
    config = load_phase2_config(PILOT_PATH)
    failed_results = [
        _seed_result(
            tmp_path,
            config,
            seed=seed,
            beta=beta,
            length_unsafe=seed == min(PHASE2_PILOT_SEEDS),
        )
        for seed, beta in zip(sorted(PHASE2_PILOT_SEEDS), (1.5, 2.0, 1.75), strict=True)
    ]
    failed_path = tmp_path / "calibration-h256-failed.json"
    failed = write_phase2_pilot_aggregate(config, failed_results, failed_path)

    assert failed["horizon"]["all_seed_length_gates_passed"] is False
    assert failed["selection"]["horizon_accepted"] is False
    assert failed["selection"]["freeze_validation_required"] is False
    assert failed["selection"]["next_horizon_tokens"] == 512
    assert failed["selection"]["next_action"] == "issue_new_calibration_identity_at_next_horizon"

    escalated = _escalated_calibration_config(
        config,
        horizon=512,
        horizon_grid_index=1,
        parent_sha256=_sha256(failed_path),
    )
    escalated_results = [
        _seed_result(tmp_path, escalated, seed=seed, beta=beta)
        for seed, beta in zip(sorted(PHASE2_PILOT_SEEDS), (1.6, 2.1, 1.8), strict=True)
    ]
    with pytest.raises(ValueError, match="horizon requires"):
        build_phase2_pilot_aggregate(
            escalated,
            escalated_results,
            reference_base=tmp_path,
        )
    accepted = build_phase2_pilot_aggregate(
        escalated,
        escalated_results,
        reference_base=tmp_path,
        horizon_parent_aggregate=failed_path,
    )

    assert accepted["horizon"]["candidate_horizon_tokens"] == 512
    assert accepted["horizon"]["horizon_grid_index"] == 1
    assert accepted["horizon"]["parent_binding_verified"] is True
    assert accepted["selection"]["horizon_accepted"] is True
    assert accepted["selection"]["next_action"] == (
        "issue_pilot_freeze_identity_at_recommended_beta"
    )


def test_freeze_aggregate_validates_one_beta_and_accepts_only_after_all_gates(
    tmp_path: Path,
) -> None:
    calibration, calibration_results = _calibration_campaign(tmp_path)
    calibration_path = tmp_path / "calibration-aggregate.json"
    calibration_payload = write_phase2_pilot_aggregate(
        calibration,
        calibration_results,
        calibration_path,
    )
    beta = calibration_payload["selection"]["recommended_pilot_freeze_beta"]
    freeze = _freeze_config(
        calibration,
        beta=beta,
        source_sha256=_sha256(calibration_path),
    )
    freeze_results = [
        _seed_result(tmp_path, freeze, seed=seed, beta=beta) for seed in sorted(PHASE2_PILOT_SEEDS)
    ]

    payload = build_phase2_pilot_aggregate(
        freeze,
        freeze_results,
        reference_base=tmp_path,
        beta_source_aggregate=calibration_path,
    )

    assert payload["pilot_phase"] == "freeze"
    assert payload["selection"]["frozen_global_beta"] == 2.0
    assert payload["selection"]["all_seeds_and_arms_used_same_beta"] is True
    assert payload["selection"]["beta_grid_index"] == 0
    assert payload["selection"]["selection_accepted"] is True
    assert payload["selection"]["accepted_for_confirmatory_identity"] is True
    assert payload["selection"]["next_action"] == "freeze_confirmatory_design_identity"


def test_failed_freeze_recommends_exact_double_in_a_new_identity(
    tmp_path: Path,
) -> None:
    calibration, calibration_results = _calibration_campaign(tmp_path)
    calibration_path = tmp_path / "calibration-aggregate.json"
    calibration_payload = write_phase2_pilot_aggregate(
        calibration,
        calibration_results,
        calibration_path,
    )
    beta = calibration_payload["selection"]["recommended_pilot_freeze_beta"]
    freeze = _freeze_config(
        calibration,
        beta=beta,
        source_sha256=_sha256(calibration_path),
    )
    freeze_results = [
        _seed_result(
            tmp_path,
            freeze,
            seed=seed,
            beta=beta,
            unsafe=seed == min(PHASE2_PILOT_SEEDS),
        )
        for seed in sorted(PHASE2_PILOT_SEEDS)
    ]

    payload = build_phase2_pilot_aggregate(
        freeze,
        freeze_results,
        reference_base=tmp_path,
        beta_source_aggregate=calibration_path,
    )

    assert payload["selection"]["selection_accepted"] is False
    assert payload["selection"]["accepted_for_confirmatory_identity"] is False
    assert payload["selection"]["next_global_beta"] == 2.0 * beta
    assert payload["selection"]["next_action"] == "issue_new_pilot_freeze_identity_at_double_beta"

    failed_path = tmp_path / "failed-freeze-aggregate.json"
    _write_json(failed_path, payload)
    retry_beta = 2.0 * beta
    retry = _freeze_config(
        calibration,
        beta=retry_beta,
        source_sha256=_sha256(failed_path),
        horizon_parent_sha256=_sha256(calibration_path),
    )
    retry_results = [
        _seed_result(tmp_path, retry, seed=seed, beta=retry_beta)
        for seed in sorted(PHASE2_PILOT_SEEDS)
    ]
    retry_payload = build_phase2_pilot_aggregate(
        retry,
        retry_results,
        reference_base=tmp_path,
        beta_source_aggregate=failed_path,
        horizon_parent_aggregate=calibration_path,
    )
    assert retry_payload["selection"]["beta_grid_index"] == 1
    assert retry_payload["selection"]["selection_accepted"] is True

    skipped_retry = _freeze_config(
        calibration,
        beta=4.0 * beta,
        source_sha256=_sha256(failed_path),
        horizon_parent_sha256=_sha256(calibration_path),
    )
    with pytest.raises(ValueError, match="immediately preceding"):
        verify_beta_source_aggregate(skipped_retry, failed_path)


def test_freeze_source_hash_and_power_of_two_grid_fail_closed(tmp_path: Path) -> None:
    calibration, calibration_results = _calibration_campaign(tmp_path)
    calibration_path = tmp_path / "calibration-aggregate.json"
    calibration_payload = write_phase2_pilot_aggregate(
        calibration,
        calibration_results,
        calibration_path,
    )
    base = calibration_payload["selection"]["recommended_pilot_freeze_beta"]
    skipped_initial_grid_point = _freeze_config(
        calibration,
        beta=2.0 * base,
        source_sha256=_sha256(calibration_path),
    )
    with pytest.raises(ValueError, match="initial freeze identity"):
        verify_beta_source_aggregate(skipped_initial_grid_point, calibration_path)

    wrong_hash = _freeze_config(
        calibration,
        beta=base,
        source_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="SHA256"):
        verify_beta_source_aggregate(wrong_hash, calibration_path)


def test_pilot_aggregate_rejects_missing_seed_and_sidecar_leak(tmp_path: Path) -> None:
    config, results = _calibration_campaign(tmp_path)
    with pytest.raises(ValueError, match="exactly one result"):
        build_phase2_pilot_aggregate(config, results[:-1], reference_base=tmp_path)

    sidecar = results[0].with_name(f"{results[0].stem}.diagnostics.jsonl")
    first, *remaining = sidecar.read_text(encoding="utf-8").splitlines()
    row = json.loads(first)
    row["response"] = "leaked"
    sidecar.write_text(
        "\n".join([json.dumps(row, sort_keys=True), *remaining]) + "\n",
        encoding="utf-8",
    )
    result = json.loads(results[0].read_text(encoding="utf-8"))
    result["diagnostics_sha256"] = _sha256(sidecar)
    _write_json(results[0], result)
    with pytest.raises(ValueError, match="target-free schema"):
        build_phase2_pilot_aggregate(config, results, reference_base=tmp_path)


def test_pilot_aggregate_never_overwrites(tmp_path: Path) -> None:
    config, results = _calibration_campaign(tmp_path)
    output = tmp_path / "pilot-aggregate.json"
    write_phase2_pilot_aggregate(config, results, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_phase2_pilot_aggregate(config, results, output)
