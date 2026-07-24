from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import smart_reward.phase2_pilot_aggregate as pilot_aggregate_module
from smart_reward.phase2_config import (
    PHASE2_PILOT_SEEDS,
    load_phase2_config,
    phase2_design_identity,
)
from smart_reward.phase2_pilot_aggregate import (
    PHASE2_PILOT_AGGREGATE_SCHEMA,
    PHASE2_PILOT_AGGREGATION_IDENTITY_SCHEMA,
    verify_beta_source_aggregate,
)
from smart_reward.phase2_pilot_aggregate import (
    build_phase2_pilot_aggregate as _build_phase2_pilot_aggregate,
)
from smart_reward.phase2_pilot_aggregate import (
    write_phase2_pilot_aggregate as _write_phase2_pilot_aggregate,
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
PROMPT_SEMANTICS_RECORDS = [
    {
        "prompt_id": f"prompt-{index}",
        "raw_prompt_sha256": hashlib.sha256(f"raw-prompt-{index}".encode()).hexdigest(),
        "policy_chat_token_count": 24,
        "policy_prompt_token_ids_sha256": hashlib.sha256(
            f"policy-prefix-{index}".encode()
        ).hexdigest(),
        "max_prompt_tokens": 1024,
        "truncated": False,
        "raw_prompt_preserved": True,
    }
    for index in range(2048)
]
PROMPT_SEMANTICS_RECORDS_SHA256 = hashlib.sha256(
    json.dumps(
        PROMPT_SEMANTICS_RECORDS,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


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


def _aggregation_identity(**overrides: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": PHASE2_PILOT_AGGREGATION_IDENTITY_SCHEMA,
        "aggregator_git_commit": "d" * 40,
        "producer_git_commit": _environment()["git_commit"],
        "image_sha256": _environment()["image_sha256"],
        "hf_inventory_sha256": _environment()["hf_inventory_sha256"],
        "validator_source_sha256": hashlib.sha256(
            Path(pilot_aggregate_module.__file__).read_bytes()
        ).hexdigest(),
    }
    identity.update(overrides)
    return identity


def build_phase2_pilot_aggregate(
    config: dict[str, Any],
    result_jsons: list[Path],
    **kwargs: Any,
) -> dict[str, object]:
    kwargs.setdefault("aggregation_identity", _aggregation_identity())
    return _build_phase2_pilot_aggregate(config, result_jsons, **kwargs)


def write_phase2_pilot_aggregate(
    config: dict[str, Any],
    result_jsons: list[Path],
    output_json: Path,
    **kwargs: Any,
) -> dict[str, object]:
    kwargs.setdefault("aggregation_identity", _aggregation_identity())
    return _write_phase2_pilot_aggregate(
        config,
        result_jsons,
        output_json,
        **kwargs,
    )


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
    task_id = seed - min(PHASE2_PILOT_SEEDS)
    run_dir = directory / design_sha / f"seed-{seed}" / f"job-1000_{task_id}"
    run_dir.mkdir(parents=True)
    sidecar_path = run_dir / "phase2-pilot-diagnostics.diagnostics.jsonl"
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
    artifact_dir = run_dir / "artifact"
    artifact_dir.mkdir()
    artifact_metadata = {
        "schema": "controlled-feature-artifact/v1",
        "config_hash": config["design"]["source_config_hash"],
        "seed": seed,
        "splits": {
            "train": {"prompt_ids": ["train"]},
            "validation": {"prompt_ids": ["validation"]},
            "test": {"prompt_ids": ["test"]},
        },
        "tensors": {},
        "tensor_sha256": "4" * 64,
        "evidence": {
            "chat_template_sha256": "1" * 64,
            "oracle_chat_template_sha256": "3" * 64,
            "policy_prompt_semantics": {
                "schema_version": "full-policy-prompt-semantics/v1",
                "encoding": "policy_tokenizer_apply_chat_template",
                "add_generation_prompt": True,
                "truncation": False,
                "fail_closed_above_max_prompt_tokens": True,
                "max_prompt_tokens": 1024,
                "num_prompts": 2048,
                "records_sha256": PROMPT_SEMANTICS_RECORDS_SHA256,
                "records": PROMPT_SEMANTICS_RECORDS,
            },
            "producer": {
                "git_commit": environment["git_commit"],
                "image_sha256": environment["image_sha256"],
                "hf_inventory_sha256": environment["hf_inventory_sha256"],
            },
        },
    }
    artifact_metadata_path = artifact_dir / "metadata.json"
    _write_json(artifact_metadata_path, artifact_metadata)

    manifest = {
        "schema_version": "smart-reward-run/v1",
        "created_at_utc": "2026-07-25T00:00:00Z",
        "config_hash": config["design"]["source_config_hash"],
        "normalized_config": {},
        "seed": sorted(PHASE2_PILOT_SEEDS),
        "selected_seed": seed,
        "named_seeds": {},
        "git": {"commit": environment["git_commit"], "dirty": False},
        "python": {},
        "platform": {},
        "torch": {
            "cuda_available": True,
            "gpu_count": 1,
            "gpus": [{"index": 0, "name": "NVIDIA L20"}],
        },
        "revisions": {},
        "packages": {},
        "slurm": {
            "PRORM_GIT_COMMIT": environment["git_commit"],
            "PRORM_IMAGE_SHA256": environment["image_sha256"],
            "PRORM_HF_INVENTORY_SHA256": environment["hf_inventory_sha256"],
            "SLURM_JOB_ACCOUNT": "sigroup",
            "SLURM_JOB_PARTITION": "gpu-l20",
        },
    }
    manifest_path = run_dir / "run-manifest.json"
    _write_json(manifest_path, manifest)

    result_path = run_dir / "phase2-pilot-diagnostics.json"
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
        "artifact_dir": "artifact",
        "diagnostics_jsonl": sidecar_path.name,
        "artifact_metadata_sha256": _sha256(artifact_metadata_path),
        "run_manifest": "run-manifest.json",
        "run_manifest_sha256": _sha256(manifest_path),
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
            "calibration_split": (
                "train_only"
                if pilot_phase == "calibration"
                else "excluded_pilot_calibration_outputs_only"
            ),
            "new_rollout_prompts_used_for_calibration": False,
            "final_oracle_session_opened": False,
            "rollout_responses_oracle_scored": False,
            "heldout_evaluator_called": False,
            "oracle_outcomes_serialized": False,
            "prompt_or_response_text_serialized": False,
            "token_ids_or_response_masks_serialized": False,
            "source_artifact_format": "phase1_bridge",
            "source_artifact_may_contain_prior_heldout_candidate_scores": True,
            "source_artifact_heldout_targets_exposed_by_phase2_prepared_inputs": False,
            "prompt_semantics": {
                "schema_version": "phase2-full-prompt-continuity/v1",
                "materialization": {
                    "schema_version": "full-policy-prompt-semantics/v1",
                    "policy_chat_template_sha256": "1" * 64,
                    "encoding": "policy_tokenizer_apply_chat_template",
                    "add_generation_prompt": True,
                    "truncation": False,
                    "fail_closed_above_max_prompt_tokens": True,
                    "max_prompt_tokens": 1024,
                    "num_prompts": 2048,
                    "minimum_policy_chat_token_count": 24,
                    "maximum_policy_chat_token_count": 24,
                    "mean_policy_chat_token_count": 24.0,
                    "over_limit_prompt_count": 0,
                    "truncated_prompt_count": 0,
                    "raw_prompt_preserved_count": 2048,
                    "records_sha256": PROMPT_SEMANTICS_RECORDS_SHA256,
                    "candidate_prefixes_verified": True,
                },
                "rollout": {
                    "schema_version": "full-policy-prompt-semantics/v1",
                    "num_prompts": 256,
                    "max_prompt_tokens": 1024,
                    "minimum_policy_chat_token_count": 24,
                    "maximum_policy_chat_token_count": 24,
                    "mean_policy_chat_token_count": 24.0,
                    "over_limit_prompt_count": 0,
                    "truncated_prompt_count": 0,
                    "raw_prompt_preserved_count": 256,
                    "matches_materialization_token_prefix_evidence": True,
                    "same_evidence_across_policy_arms": True,
                },
                "oracle": {
                    "input_text": "same_raw_prompt_plus_assistant_response",
                    "rerendered_with_independent_oracle_chat_template": True,
                    "policy_chat_tokens_reused_by_oracle": False,
                    "policy_and_oracle_chat_template_sha256_distinct": True,
                    "policy_chat_template_sha256": "1" * 64,
                    "oracle_chat_template_sha256": "3" * 64,
                },
            },
        },
        "common_random_numbers": {"candidate_index_alignment": True},
        "memory_schedule": ["stop_before_final_oracle_and_heldout_evaluation"],
        "policy_and_oracle_co_resident": False,
        "learner_specific_line_search": False,
        "diagnostics_sha256": _sha256(sidecar_path),
        beta_key: beta_evidence,
    }
    _write_json(result_path, payload)

    output_verification = {
        "schema_version": "prorm-phase2-output-verification/v1",
        "status": "passed",
        "seed": seed,
        "source_config_hash": config["design"]["source_config_hash"],
        "phase2_design_sha256": design_sha,
        "pilot_phase": pilot_phase,
        "diagnostic_records": len(rows),
        "diagnostics_sha256": _sha256(sidecar_path),
        "kl_gate_passed": not mean_violations,
        "kl_measure_only": True,
        "kl_violations": mean_violations,
        "pre_oracle_gate_passed": not violations,
        "pre_oracle_violations": violations,
        "environment_identity": environment,
    }
    _write_json(
        run_dir / "phase2-output-verification.json",
        output_verification,
    )

    beta_source_sha = design.beta_source_aggregate_sha256
    horizon_parent_sha = design.parent_pilot_aggregate_sha256
    success = {
        "schema_version": "prorm-phase2-run-status/v1",
        "status": "SUCCESS",
        "workload_exit_code": "0",
        "final_exit_code": "0",
        "array_job_id": "1000",
        "array_task_id": str(task_id),
        "seed": str(seed),
        "phase2_design_sha256": design_sha,
        "base_config_hash": config["design"]["source_config_hash"],
        "git_commit": environment["git_commit"],
        "beta_source_aggregate_present": "1" if beta_source_sha is not None else "0",
        "beta_source_aggregate_sha256": beta_source_sha or "none",
        "horizon_parent_aggregate_present": "1" if horizon_parent_sha is not None else "0",
        "horizon_parent_aggregate_sha256": horizon_parent_sha or "none",
        "created_at_utc": "2026-07-25T00:00:00Z",
    }
    (run_dir / "SUCCESS").write_text(
        "".join(f"{key}={value}\n" for key, value in success.items()),
        encoding="utf-8",
    )
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
    source = payload["sources"][0]
    for role in (
        "artifact_metadata",
        "run_manifest",
        "output_verification",
        "success_receipt",
    ):
        assert isinstance(source[role], str)
        assert len(source[f"{role}_sha256"]) == 64


def test_aggregation_identity_is_required_and_binds_producer_and_validator(
    tmp_path: Path,
) -> None:
    config, results = _calibration_campaign(tmp_path)
    with pytest.raises(TypeError, match="aggregation_identity"):
        _build_phase2_pilot_aggregate(config, results)

    with pytest.raises(ValueError, match="shared seed producer identity"):
        _build_phase2_pilot_aggregate(
            config,
            results,
            aggregation_identity=_aggregation_identity(producer_git_commit="f" * 40),
        )
    with pytest.raises(ValueError, match="loaded pilot aggregate validator source"):
        _build_phase2_pilot_aggregate(
            config,
            results,
            aggregation_identity=_aggregation_identity(validator_source_sha256="f" * 64),
        )


def test_predecessor_rejects_schema_incomplete_aggregate_even_when_sha_matches(
    tmp_path: Path,
) -> None:
    calibration, results = _calibration_campaign(tmp_path)
    valid_path = tmp_path / "calibration-valid.json"
    payload = write_phase2_pilot_aggregate(calibration, results, valid_path)
    incomplete_path = tmp_path / "calibration-schema-incomplete.json"
    incomplete = copy.deepcopy(payload)
    del incomplete["aggregation_identity"]
    _write_json(incomplete_path, incomplete)
    freeze = _freeze_config(
        calibration,
        beta=payload["selection"]["recommended_pilot_freeze_beta"],
        source_sha256=_sha256(incomplete_path),
    )

    with pytest.raises(ValueError, match="keys differ from the target-free schema"):
        verify_beta_source_aggregate(freeze, incomplete_path)


def test_predecessor_recomputes_selection_and_source_provenance(
    tmp_path: Path,
) -> None:
    calibration, results = _calibration_campaign(tmp_path)
    aggregate_path = tmp_path / "calibration-valid.json"
    payload = write_phase2_pilot_aggregate(calibration, results, aggregate_path)
    tampered_path = tmp_path / "calibration-selection-tampered.json"
    tampered = copy.deepcopy(payload)
    tampered["selection"]["recommended_pilot_freeze_beta"] = 3.0
    _write_json(tampered_path, tampered)
    tampered_freeze = _freeze_config(
        calibration,
        beta=3.0,
        source_sha256=_sha256(tampered_path),
    )
    with pytest.raises(ValueError, match="selection differs from recomputed source evidence"):
        verify_beta_source_aggregate(tampered_freeze, tampered_path)

    valid_freeze = _freeze_config(
        calibration,
        beta=payload["selection"]["recommended_pilot_freeze_beta"],
        source_sha256=_sha256(aggregate_path),
    )
    results[0].write_bytes(results[0].read_bytes() + b" ")
    with pytest.raises(ValueError, match="source provenance differs"):
        verify_beta_source_aggregate(valid_freeze, aggregate_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "predicted_oracle_quadratic_kl",
            0.004,
            "calibration target/beta closed-form identity",
        ),
        (
            "candidate_beta",
            1.6,
            "calibration target/beta closed-form identity",
        ),
        (
            "formal_selection_rule",
            "maximum_candidate_without_freeze_grid",
            "invalid calibration-candidate contract",
        ),
    ],
)
def test_calibration_beta_candidate_contract_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    config, results = _calibration_campaign(tmp_path)
    result = json.loads(results[0].read_text(encoding="utf-8"))
    result["train_only_global_beta_calibration_candidate"][field] = value
    _write_json(results[0], result)

    with pytest.raises(ValueError, match=match):
        build_phase2_pilot_aggregate(
            config,
            results,
            reference_base=tmp_path,
        )


def test_pilot_information_boundary_requires_the_exact_real_schema(
    tmp_path: Path,
) -> None:
    config, results = _calibration_campaign(tmp_path)
    result = json.loads(results[0].read_text(encoding="utf-8"))
    result["information_boundary"]["unregistered_boundary_claim"] = False
    _write_json(results[0], result)

    with pytest.raises(ValueError, match="keys differ from the target-free schema"):
        build_phase2_pilot_aggregate(
            config,
            results,
            reference_base=tmp_path,
        )

    config, results = _calibration_campaign(tmp_path / "count")
    result = json.loads(results[0].read_text(encoding="utf-8"))
    result["information_boundary"]["prompt_semantics"]["materialization"]["num_prompts"] = 1
    result["information_boundary"]["prompt_semantics"]["materialization"][
        "raw_prompt_preserved_count"
    ] = 1
    _write_json(results[0], result)

    with pytest.raises(ValueError, match="materialization is invalid"):
        build_phase2_pilot_aggregate(
            config,
            results,
            reference_base=tmp_path,
        )

    config, results = _calibration_campaign(tmp_path / "records")
    result = json.loads(results[0].read_text(encoding="utf-8"))
    result["information_boundary"]["prompt_semantics"]["materialization"]["records_sha256"] = (
        "f" * 64
    )
    _write_json(results[0], result)

    with pytest.raises(ValueError, match="prompt-continuity evidence differs"):
        build_phase2_pilot_aggregate(
            config,
            results,
            reference_base=tmp_path,
        )

    config, results = _calibration_campaign(tmp_path / "coupled-records")
    result_path = results[0]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata_path = result_path.parent / "artifact" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evidence"]["policy_prompt_semantics"]["records_sha256"] = "f" * 64
    _write_json(metadata_path, metadata)
    result["artifact_metadata_sha256"] = _sha256(metadata_path)
    result["information_boundary"]["prompt_semantics"]["materialization"]["records_sha256"] = (
        "f" * 64
    )
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="canonical record bytes"):
        build_phase2_pilot_aggregate(
            config,
            results,
            reference_base=tmp_path,
        )

    config, results = _calibration_campaign(tmp_path / "nested")
    result = json.loads(results[0].read_text(encoding="utf-8"))
    result["information_boundary"]["prompt_semantics"]["oracle"]["oracle_scores"] = [1.0]
    _write_json(results[0], result)

    with pytest.raises(ValueError, match="keys differ from the target-free schema"):
        build_phase2_pilot_aggregate(
            config,
            results,
            reference_base=tmp_path,
        )


def test_pilot_provenance_hashes_and_producer_identity_fail_closed(
    tmp_path: Path,
) -> None:
    config, results = _calibration_campaign(tmp_path)
    result_path = results[0]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata_path = result_path.parent / "artifact" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evidence"]["producer"]["git_commit"] = "f" * 40
    _write_json(metadata_path, metadata)
    result["artifact_metadata_sha256"] = _sha256(metadata_path)
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="seed/base/producer identity"):
        build_phase2_pilot_aggregate(
            config,
            results,
            reference_base=tmp_path,
        )

    config, results = _calibration_campaign(tmp_path / "template")
    result_path = results[0]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata_path = result_path.parent / "artifact" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evidence"]["chat_template_sha256"] = "5" * 64
    _write_json(metadata_path, metadata)
    result["artifact_metadata_sha256"] = _sha256(metadata_path)
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="prompt-continuity evidence differs"):
        build_phase2_pilot_aggregate(
            config,
            results,
            reference_base=tmp_path,
        )


def test_pilot_manifest_output_verification_and_success_receipt_fail_closed(
    tmp_path: Path,
) -> None:
    config, results = _calibration_campaign(tmp_path)
    result_path = results[0]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = result_path.parent / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git"]["commit"] = "f" * 40
    manifest["slurm"]["PRORM_GIT_COMMIT"] = "f" * 40
    _write_json(manifest_path, manifest)
    result["run_manifest_sha256"] = _sha256(manifest_path)
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="differs from its run manifest"):
        build_phase2_pilot_aggregate(
            config,
            results,
            reference_base=tmp_path,
        )

    result, results = _calibration_campaign(tmp_path / "verification")
    result_path = results[0]
    verification_path = result_path.parent / "phase2-output-verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["status"] = "failed"
    _write_json(verification_path, verification)
    with pytest.raises(ValueError, match="does not bind the verified pilot result"):
        build_phase2_pilot_aggregate(
            result,
            results,
            reference_base=tmp_path,
        )

    result, results = _calibration_campaign(tmp_path / "success")
    result_path = results[0]
    success_path = result_path.parent / "SUCCESS"
    success = success_path.read_text(encoding="utf-8").replace(
        "status=SUCCESS\n",
        "status=FAILED\n",
    )
    success_path.write_text(success, encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind the successful pilot attempt"):
        build_phase2_pilot_aggregate(
            result,
            results,
            reference_base=tmp_path,
        )


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


def test_accepted_freeze_can_authorize_a_new_confirmatory_base_identity(
    tmp_path: Path,
) -> None:
    calibration, calibration_results = _calibration_campaign(tmp_path)
    calibration_path = tmp_path / "calibration-aggregate.json"
    calibration_payload = write_phase2_pilot_aggregate(
        calibration,
        calibration_results,
        calibration_path,
    )
    beta = float(calibration_payload["selection"]["recommended_pilot_freeze_beta"])
    freeze = _freeze_config(
        calibration,
        beta=beta,
        source_sha256=_sha256(calibration_path),
    )
    freeze_results = [
        _seed_result(tmp_path, freeze, seed=seed, beta=beta) for seed in sorted(PHASE2_PILOT_SEEDS)
    ]
    freeze_path = tmp_path / "freeze-aggregate.json"
    write_phase2_pilot_aggregate(
        freeze,
        freeze_results,
        freeze_path,
        beta_source_aggregate=calibration_path,
    )
    freeze_sha = _sha256(freeze_path)
    freeze_design = Phase2Design.from_phase2_config(freeze)
    confirmatory_design = replace(
        freeze_design,
        stage="confirmatory",
        formal_eligibility=True,
        pilot_phase=None,
        common_beta_rule="single_pilot_frozen_global_beta_scalar",
        common_beta_calibration_split="excluded_pilot",
        common_beta_source="frozen_pilot_global_beta_in_confirmatory_design_identity",
        beta_source_aggregate_sha256=freeze_sha,
        parent_pilot_aggregate_sha256=freeze_sha,
        frozen_global_beta_sensitivity_multipliers=(0.5, 2.0),
        max_length_formal_gate=True,
    )
    predecessor = pilot_aggregate_module._load_source_aggregate(
        freeze_path,
        expected_sha256=freeze_sha,
    )

    binding = pilot_aggregate_module._beta_source_binding_for_design(
        confirmatory_design,
        expected_source_config_hash="9" * 64,
        predecessor=predecessor,
    )

    assert binding is not None
    assert binding["accepted_beta"] == beta
    assert binding["sha256"] == freeze_sha


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
