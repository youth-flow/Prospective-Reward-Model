from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smart_reward.config import config_hash
from smart_reward.phase2_config import PHASE2_BUDGETED_END_TO_END_SEEDS
from smart_reward.phase2_rollout import BUDGETED_COMMON_BETA_RULE, Phase2Design
from smart_reward.seeding import SeedBundle, derive_seed

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "hpc4" / "verify_phase2_budgeted_end_to_end_seed_output.py"

SEED = PHASE2_BUDGETED_END_TO_END_SEEDS[0]
DESIGN = "a" * 64
GIT = "b" * 40
IMAGE = "c" * 64
INVENTORY = "d" * 64
FREEZE = "e" * 64
POLICY_TEMPLATE = "1" * 64
ORACLE_TEMPLATE = "2" * 64
BETA = 2.5
JOB_ID = "900001"
ARRAY_JOB_ID = "900000"
TASK_ID = 0
EVIDENCE_ROLE = "budgeted_end_to_end_fixed_three_exploratory_only"
ARMS = ("zero_b", "bt_mle", "prorm_plus", "oracle_step")


def _runtime_contract() -> dict[str, object]:
    return Phase2Design(
        stage="budgeted_end_to_end",
        formal_eligibility=False,
        pilot_phase=None,
        common_beta_rule=BUDGETED_COMMON_BETA_RULE,
        common_beta_calibration_split="excluded_pilot",
        common_beta_source=("accepted_freeze_global_beta_in_budgeted_end_to_end_design_identity"),
        frozen_global_beta=BETA,
        beta_source_aggregate_sha256=FREEZE,
        parent_pilot_aggregate_sha256=FREEZE,
        rollout_candidates_per_prompt=2,
        k_cal_sensitivity_values=None,
        frozen_global_beta_sensitivity_multipliers=None,
    ).to_dict()


def _module():
    spec = importlib.util.spec_from_file_location("_budgeted_seed_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    raw = _canonical_bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> str:
    raw = b"".join(_canonical_bytes(record) for record in records)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _normalized(seed: int = SEED) -> dict[str, object]:
    return {
        "seed": seed,
        "phase2_design_sha256": DESIGN,
        "phase2_runtime_contract_sha256": "f" * 64,
        "beta_source_aggregate_sha256": FREEZE,
        "frozen_global_beta": BETA,
        "admissible": True,
        "endpoints": {
            "heldout_local_regret": {"bt_mle": 0.2, "prorm_plus": 0.1},
            "finite_policy_utility": {"bt_mle": 0.4, "prorm_plus": 0.5},
            "oracle_pairwise_cross_entropy": {"bt_mle": 0.6, "prorm_plus": 0.5},
            "oracle_probability_mae": {"bt_mle": 0.3, "prorm_plus": 0.2},
            "pairwise_order_accuracy": {"bt_mle": 0.7, "prorm_plus": 0.8},
        },
    }


def _row(
    arm: str,
    prompt_id: str,
    prompt: str,
    candidate_index: int,
) -> dict[str, object]:
    token_ids = [10, 20 + candidate_index]
    prompt_token_sha = hashlib.sha256(b"[10]").hexdigest()
    kl = 0.01 * (candidate_index + 1)
    reward = 1.0 + candidate_index
    prompt_seed = derive_seed(
        SeedBundle.from_base_seed(SEED).rollout,
        f"phase2-test-prompt:{prompt_id}",
    )
    return {
        "schema_version": "common-beta-budgeted-trajectory/v1",
        "design_stage": "budgeted_end_to_end",
        "evidence_role": EVIDENCE_ROLE,
        "formal_claim_eligible": False,
        "supports_formal_claim": False,
        "arm": arm,
        "policy_source": (
            "zero_b_reference" if arm == "zero_b" else "direct_common_beta_displacement"
        ),
        "beta_common": BETA,
        "prompt_id": prompt_id,
        "candidate_index": candidate_index,
        "prompt": prompt,
        "prompt_semantics": {
            "schema_version": "full-policy-prompt-semantics/v1",
            "raw_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "policy_chat_token_count": 1,
            "policy_prompt_token_ids_sha256": prompt_token_sha,
            "max_prompt_tokens": 16,
            "truncated": False,
            "raw_prompt_preserved": True,
        },
        "response": f"{arm}-{candidate_index}",
        "token_ids": token_ids,
        "response_mask": [0, 1],
        "response_token_count": 1,
        "terminated_by_eos": True,
        "reached_max_length": False,
        "prompt_rollout_seed": prompt_seed,
        "kl_orientation": "pi_updated_to_pi0",
        "kl_history_source": "updated_policy",
        "on_policy_kl_pi_updated_to_pi0": kl,
        "transformed_oracle_reward": reward,
        "target_utility": reward - BETA * kl,
        "raw_oracle_logit_serialized": False,
    }


def _records() -> list[dict[str, object]]:
    return [
        _row(arm, prompt_id, prompt, candidate)
        for arm in ARMS
        for prompt_id, prompt in (("test-0", "Test prompt"),)
        for candidate in range(2)
    ]


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    verifier = _module()
    job_dir = tmp_path.resolve() / "job"
    artifact_dir = job_dir / "artifact"
    artifact_dir.mkdir(parents=True)
    overlay = tmp_path.resolve() / "overlay.yaml"
    overlay.write_text("test: overlay\n", encoding="utf-8", newline="\n")
    result_path = job_dir / "phase2-result.json"
    rollouts_path = job_dir / "phase2-result.rollouts.jsonl"
    manifest_path = job_dir / "run-manifest.json"
    metadata_path = artifact_dir / "metadata.json"
    materialization_path = job_dir / "artifact-materialization.json"
    output_path = job_dir / "phase2-budgeted-output-verification.json"

    base = {
        "run": {
            "name": "test",
            "seeds": list(PHASE2_BUDGETED_END_TO_END_SEEDS),
        }
    }
    base_hash = config_hash(base)
    runtime_contract = _runtime_contract()
    config = {
        "design": {
            "stage": "budgeted_end_to_end",
            "formal_eligibility": False,
            "pilot_phase": None,
            "evidence_role": EVIDENCE_ROLE,
            "source_config_hash": base_hash,
        },
        "policy": {
            "max_response_tokens": runtime_contract["max_response_tokens"],
        },
        "data": {},
        "reward_model": {"microbatch_size": runtime_contract["oracle_batch_size"]},
        "run": {
            "seeds": list(PHASE2_BUDGETED_END_TO_END_SEEDS),
            "confirmatory": False,
        },
        "objective": {
            "common_beta": {
                "rule": runtime_contract["common_beta_rule"],
                "calibration_split": runtime_contract["common_beta_calibration_split"],
                "calibration_source": runtime_contract["common_beta_source"],
                "beta_source_aggregate_sha256": FREEZE,
                "frozen_global_beta": BETA,
                "primary_k_cal": runtime_contract["target_oracle_quadratic_kl"],
                "sensitivity_k_cal": None,
                "sensitivity_frozen_global_beta_multipliers": None,
            },
            "full_tangent": {
                "ridge": {
                    "relative_coefficient": runtime_contract["relative_damping"],
                    "solver_dtype": runtime_contract["pcg_dtype"],
                    "pcg_max_iterations": runtime_contract["pcg_max_iterations"],
                    "pcg_tolerance": runtime_contract["pcg_tolerance"],
                    "sensitivity_multipliers": runtime_contract["sensitivity_scope"][
                        "ridge_multipliers_configured"
                    ],
                }
            },
        },
        "evaluation": {
            "rollout_candidates_per_prompt": 2,
            "safety": {
                "mean_policy_to_reference_kl_cap": runtime_contract["measured_kl_safety_cap"],
                "prompt_mean_p95_kl_cap": runtime_contract["prompt_mean_p95_kl_cap"],
                "prompt_mean_p99_kl_cap": runtime_contract["prompt_mean_p99_kl_cap"],
                "prompt_mean_maximum_kl_cap": runtime_contract["prompt_mean_maximum_kl_cap"],
                "per_sequence_maximum_kl_cap": runtime_contract["per_sequence_maximum_kl_cap"],
            },
            "max_length": {
                "allowed_horizon_sequence": runtime_contract["allowed_horizon_sequence"],
                "horizon_grid_index": runtime_contract["horizon_grid_index"],
                "parent_pilot_aggregate_sha256": FREEZE,
                "previous_horizon_failed_length_gate": False,
                "formal_gate": runtime_contract["max_length_gate"]["formal_gate"],
                "formal_threshold": runtime_contract["max_length_gate"]["formal_threshold"],
            },
        },
    }
    monkeypatch.setattr(
        verifier,
        "load_phase2_config_bundle",
        lambda path: SimpleNamespace(
            config=copy.deepcopy(config),
            base_config=copy.deepcopy(base),
            design_identity=DESIGN,
        ),
    )

    def normalize(value: dict[str, object]) -> dict[str, object]:
        normalized = _normalized(int(value["seed"]))
        normalized["phase2_design_sha256"] = value["phase2_design_sha256"]
        normalized["phase2_runtime_contract_sha256"] = value["phase2_runtime_contract_sha256"]
        frozen = value["common_beta_frozen_evidence"]
        normalized["beta_source_aggregate_sha256"] = frozen["beta_source_aggregate_sha256"]
        normalized["frozen_global_beta"] = frozen["frozen_global_beta"]
        return normalized

    monkeypatch.setattr(
        verifier,
        "normalize_budgeted_end_to_end_seed_result",
        normalize,
    )

    environment = {
        "formal": True,
        "git_commit": GIT,
        "image_sha256": IMAGE,
        "hf_inventory_sha256": INVENTORY,
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_models": ["NVIDIA L20"],
    }
    manifest = {
        "schema_version": "smart-reward-run/v1",
        "created_at_utc": "2026-07-26T00:00:00Z",
        "config_hash": base_hash,
        "normalized_config": base,
        "seed": list(PHASE2_BUDGETED_END_TO_END_SEEDS),
        "selected_seed": SEED,
        "named_seeds": {},
        "git": {"commit": GIT, "dirty": False},
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
            "SLURM_JOB_ID": JOB_ID,
            "SLURM_ARRAY_JOB_ID": ARRAY_JOB_ID,
            "SLURM_ARRAY_TASK_ID": str(TASK_ID),
            "SLURM_JOB_ACCOUNT": "sigroup",
            "SLURM_JOB_PARTITION": "gpu-l20",
            "SLURM_CLUSTER_NAME": "hpc4",
            "PRORM_GIT_COMMIT": GIT,
            "PRORM_IMAGE_SHA256": IMAGE,
            "PRORM_HF_INVENTORY_SHA256": INVENTORY,
        },
    }
    manifest_hash = _write_json(manifest_path, manifest)
    metadata = {
        "schema": "controlled-feature-artifact/v1",
        "config_hash": base_hash,
        "seed": SEED,
        "splits": {
            "train": {"prompt_ids": ["train-0"]},
            "validation": {"prompt_ids": ["validation-0"]},
            "test": {"prompt_ids": ["test-0"]},
        },
        "tensors": {},
        "tensor_sha256": "9" * 64,
        "evidence": {
            "producer": {
                "git_commit": GIT,
                "image_sha256": IMAGE,
                "hf_inventory_sha256": INVENTORY,
            }
        },
    }
    metadata_hash = _write_json(metadata_path, metadata)
    materialization = {
        "schema_version": "prorm-phase2-budgeted-artifact-binding/v1",
        "mode": "fresh",
        "seed": SEED,
        "phase2_design_sha256": DESIGN,
        "base_config_hash": base_hash,
        "artifact_metadata_sha256": metadata_hash,
        "recovery_artifact_reused": False,
        "recovery_reward_heads_reused": False,
        "recovery_optimizer_state_reused": False,
    }
    _write_json(materialization_path, materialization)
    records = _records()
    rollouts_hash = _write_jsonl(rollouts_path, records)
    runtime = copy.deepcopy(runtime_contract)
    runtime_hash = hashlib.sha256(
        json.dumps(
            runtime,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    def arm_summary(arm_name: str) -> dict[str, object]:
        arm_rows = [row for row in records if row["arm"] == arm_name]
        count = len(arm_rows)
        mean_kl = (
            math.fsum(float(row["on_policy_kl_pi_updated_to_pi0"]) for row in arm_rows) / count
        )
        mean_reward = math.fsum(float(row["transformed_oracle_reward"]) for row in arm_rows) / count
        mean_utility = math.fsum(float(row["target_utility"]) for row in arm_rows) / count
        prompt_mean_kl = mean_kl
        return {
            "mean_on_policy_kl_pi_updated_to_pi0": mean_kl,
            "on_policy_kl_tail": {
                "schema_version": "on-policy-kl-tail-summary/v1",
                "unit": "prompt_mean_over_candidates",
                "num_prompts": 1,
                "candidates_per_prompt": 2,
                "mean": prompt_mean_kl,
                "p50": prompt_mean_kl,
                "p90": prompt_mean_kl,
                "p95": prompt_mean_kl,
                "p99": prompt_mean_kl,
                "maximum": prompt_mean_kl,
                "per_sequence_maximum": 0.02,
                "pilot_selection_role": "locality_tail_measurement",
                "formal_gate_applied": False,
            },
            "rollout": {
                "num_trajectories": count,
                "terminated_by_eos_count": count,
                "terminated_by_eos_rate": 1.0,
                "reached_max_length_count": 0,
                "reached_max_length_rate": 0.0,
                "response_token_count": {
                    "mean": 1.0,
                    "minimum": 1,
                    "maximum": 1,
                },
            },
            "utility": {
                "mean_on_policy_kl_pi_updated_to_pi0": mean_kl,
                "mean_target_reward": mean_reward,
                "mean_target_utility": mean_utility,
            },
        }

    result = {
        "schema_version": "common-beta-budgeted-end-to-end/v1",
        "design_stage": "budgeted_end_to_end",
        "formal_eligibility": False,
        "formal_claim_eligible": False,
        "supports_formal_claim": False,
        "per_seed_supports_formal_claim": False,
        "excluded_from_confirmatory_evidence": True,
        "confirmatory_authorization_created": False,
        "evidence_role": EVIDENCE_ROLE,
        "seed": SEED,
        "phase2_design_sha256": DESIGN,
        "source_config_hash": base_hash,
        "phase2_runtime_contract": runtime,
        "phase2_runtime_contract_sha256": runtime_hash,
        "common_beta_frozen_evidence": {
            "schema_version": "common-beta-frozen-global-budgeted/v1",
            "evidence_role": EVIDENCE_ROLE,
            "formal_eligibility": False,
            "supports_formal_claim": False,
            "beta_source_aggregate_sha256": FREEZE,
            "frozen_global_beta": BETA,
            "beta_common": BETA,
        },
        "common_random_numbers": {
            "named_stream": "rollout",
            "seed": SeedBundle.from_base_seed(SEED).rollout,
            "same_per_prompt_seed_reset_across_arms": True,
            "candidate_index_alignment": True,
        },
        "numerical_event_sequence": [
            "freeze_heldout_evaluation_state",
            "policy_rollouts_and_on_policy_kl",
            "enforced_nonformal_pre_oracle_safety",
            "final_operational_oracle_rollout_scoring",
            "deferred_heldout_oracle_scoring_and_metrics",
        ],
        "numerical_event_sequence_matches_confirmatory": True,
        "information_boundary": {
            "beta_selection_split": "excluded_pilot",
            "current_seed_train_curvature_role": "predicted_kl_diagnostic_only",
            "new_rollout_prompts_used_for_calibration": False,
            "source_materialization_heldout_scores_used_for_calibration": False,
            "new_rollout_oracle_scoring_after_heads_beta_and_directions_frozen": True,
            "heldout_candidate_rescore_after_heads_beta_and_directions_frozen": True,
            "heldout_directions_used_for_policy": False,
            "source_artifact_may_contain_prior_heldout_candidate_scores": True,
            "prompt_semantics": {
                "oracle": {
                    "input_text": "same_raw_prompt_plus_assistant_response",
                    "rerendered_with_independent_oracle_chat_template": True,
                    "policy_chat_tokens_reused_by_oracle": False,
                    "policy_and_oracle_chat_template_sha256_distinct": True,
                    "policy_chat_template_sha256": POLICY_TEMPLATE,
                    "oracle_chat_template_sha256": ORACLE_TEMPLATE,
                }
            },
        },
        "train_oracle_rescore": {
            "oracle_chat_template_sha256": ORACLE_TEMPLATE,
        },
        "heldout_fixed_beta": {
            "oracle_rescore": {
                "oracle_chat_template_sha256": ORACLE_TEMPLATE,
            }
        },
        "rollouts_jsonl": "phase2-result.rollouts.jsonl",
        "rollouts_sha256": rollouts_hash,
        "run_manifest": "run-manifest.json",
        "run_manifest_sha256": manifest_hash,
        "artifact_dir": "artifact",
        "artifact_metadata_sha256": metadata_hash,
        "environment_identity": environment,
        "current_process_identity": environment,
        "arms": {arm: arm_summary(arm) for arm in ARMS},
    }
    _write_json(result_path, result)
    prepare_calls: list[dict[str, object]] = []

    def prepare_inputs(path: Path, **arguments: object) -> SimpleNamespace:
        prepare_calls.append({"path": path, **arguments})
        prompt_token_sha = hashlib.sha256(b"[10]").hexdigest()
        return SimpleNamespace(
            seed=SEED,
            phase2_config_hash=DESIGN,
            source_config_hash=base_hash,
            artifact_metadata_sha256=metadata_hash,
            run_manifest_sha256=manifest_hash,
            environment_identity=environment,
            artifact_dir=artifact_dir,
            run_manifest=manifest_path,
            test_prompts=(
                SimpleNamespace(
                    prompt_id="test-0",
                    messages=(SimpleNamespace(role="user", content="Test prompt"),),
                ),
            ),
            materialization_prompt_semantics={
                "records": [
                    {
                        "prompt_id": "test-0",
                        "raw_prompt_sha256": hashlib.sha256(b"Test prompt").hexdigest(),
                        "policy_chat_token_count": 1,
                        "policy_prompt_token_ids_sha256": prompt_token_sha,
                        "max_prompt_tokens": 16,
                        "truncated": False,
                        "raw_prompt_preserved": True,
                    }
                ]
            },
            policy_chat_template_sha256=POLICY_TEMPLATE,
            oracle_chat_template_sha256=ORACLE_TEMPLATE,
        )

    monkeypatch.setattr(verifier, "prepare_phase2_inputs", prepare_inputs)
    kwargs = {
        "seed": SEED,
        "design_sha256": DESIGN,
        "base_config_hash": base_hash,
        "git_commit": GIT,
        "image_sha256": IMAGE,
        "hf_inventory_sha256": INVENTORY,
        "artifact_metadata_sha256": metadata_hash,
        "freeze_evidence_sha256": FREEZE,
        "slurm_job_id_raw": JOB_ID,
        "array_job_id": ARRAY_JOB_ID,
        "array_task_id": TASK_ID,
    }
    return {
        "verifier": verifier,
        "overlay": overlay,
        "result_path": result_path,
        "rollouts_path": rollouts_path,
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "materialization_path": materialization_path,
        "output_path": output_path,
        "result": result,
        "records": records,
        "manifest": manifest,
        "metadata": metadata,
        "materialization": materialization,
        "prepare_calls": prepare_calls,
        "kwargs": kwargs,
    }


def _verify(fixture: dict[str, Any]) -> dict[str, object]:
    return fixture["verifier"].verify_seed_output(
        fixture["overlay"],
        fixture["result_path"],
        fixture["rollouts_path"],
        fixture["output_path"],
        **fixture["kwargs"],
    )


@pytest.mark.parametrize("seed", [20261004, 20261005])
def test_seed_four_and_five_are_outside_fixed_three(seed: int, tmp_path: Path) -> None:
    verifier = _module()
    with pytest.raises(ValueError, match="outside the fixed-three"):
        verifier.verify_seed_output(
            tmp_path / "overlay.yaml",
            tmp_path / "phase2-result.json",
            tmp_path / "phase2-result.rollouts.jsonl",
            tmp_path / "verification.json",
            seed=seed,
            design_sha256=DESIGN,
            base_config_hash="0" * 64,
            git_commit=GIT,
            image_sha256=IMAGE,
            hf_inventory_sha256=INVENTORY,
            artifact_metadata_sha256="3" * 64,
            freeze_evidence_sha256=FREEZE,
            slurm_job_id_raw=JOB_ID,
            array_job_id=ARRAY_JOB_ID,
            array_task_id=3,
        )


def test_verifies_complete_closure_and_exclusively_writes_canonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    verification = _verify(fixture)

    assert verification["schema_version"] == (
        "prorm-phase2-budgeted-fixed-three-seed-output-verification/v1"
    )
    assert verification["status"] == "verified"
    assert verification["formal_claim_eligible"] is False
    assert verification["seed"] == SEED
    assert verification["normalized_seed_record"]["admissible"] is True
    assert verification["rollout_geometry"] == {
        "arm_order": list(ARMS),
        "candidates_per_prompt": 2,
        "row_count": 8,
        "rows_per_arm": 2,
        "test_prompt_count": 1,
    }
    assert fixture["prepare_calls"] == [
        {
            "path": fixture["overlay"],
            "seed": SEED,
            "artifact_dir": fixture["metadata_path"].parent,
            "run_manifest": fixture["manifest_path"],
            "require_formal": True,
            "match_current_environment": True,
        }
    ]
    output_raw = fixture["output_path"].read_bytes()
    assert output_raw == _canonical_bytes(json.loads(output_raw))
    with pytest.raises(FileExistsError, match="overwrite"):
        _verify(fixture)


def test_result_formal_flip_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["result"]["formal_claim_eligible"] = True
    _write_json(fixture["result_path"], fixture["result"])

    with pytest.raises(ValueError, match="formal_claim_eligible"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


@pytest.mark.parametrize("mutation", ["splice", "reorder", "duplicate", "wrong_count"])
def test_rollout_semantic_tampering_is_rejected_even_with_replayed_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    records = copy.deepcopy(fixture["records"])
    if mutation == "splice":
        records[0]["prompt_id"] = "test-from-another-seed"
    elif mutation == "reorder":
        records[0], records[1] = records[1], records[0]
    elif mutation == "duplicate":
        records[1] = copy.deepcopy(records[0])
    else:
        records.pop()
    new_hash = _write_jsonl(fixture["rollouts_path"], records)
    fixture["result"]["rollouts_sha256"] = new_hash
    _write_json(fixture["result_path"], fixture["result"])

    with pytest.raises(ValueError, match="rollout"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


@pytest.mark.parametrize("target", ["manifest_seed", "manifest_slurm", "artifact_seed", "config"])
def test_cross_seed_config_manifest_and_artifact_binding_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if target == "manifest_seed":
        fixture["manifest"]["selected_seed"] = PHASE2_BUDGETED_END_TO_END_SEEDS[1]
        new_hash = _write_json(fixture["manifest_path"], fixture["manifest"])
        fixture["result"]["run_manifest_sha256"] = new_hash
    elif target == "manifest_slurm":
        fixture["manifest"]["slurm"]["SLURM_ARRAY_TASK_ID"] = "1"
        new_hash = _write_json(fixture["manifest_path"], fixture["manifest"])
        fixture["result"]["run_manifest_sha256"] = new_hash
    elif target == "artifact_seed":
        fixture["metadata"]["seed"] = PHASE2_BUDGETED_END_TO_END_SEEDS[1]
        metadata_hash = _write_json(fixture["metadata_path"], fixture["metadata"])
        fixture["materialization"]["artifact_metadata_sha256"] = metadata_hash
        _write_json(fixture["materialization_path"], fixture["materialization"])
        fixture["result"]["artifact_metadata_sha256"] = metadata_hash
        fixture["kwargs"]["artifact_metadata_sha256"] = metadata_hash
    else:
        fixture["result"]["source_config_hash"] = "8" * 64
    _write_json(fixture["result_path"], fixture["result"])

    with pytest.raises(ValueError):
        _verify(fixture)
    assert not fixture["output_path"].exists()


def test_normalizer_inadmissible_gate_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fixture["verifier"],
        "normalize_budgeted_end_to_end_seed_result",
        lambda value: {"seed": value["seed"], "admissible": False},
    )

    with pytest.raises(ValueError, match="normalized"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


@pytest.mark.parametrize(
    "target",
    [
        "prompt_rollout_seed",
        "common_random_numbers",
        "numerical_event_sequence",
        "information_boundary",
        "prepared_inputs",
    ],
)
def test_cross_seed_rng_complete_input_and_information_boundary_tampering_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if target == "prompt_rollout_seed":
        records = copy.deepcopy(fixture["records"])
        records[0]["prompt_rollout_seed"] += 1
        new_hash = _write_jsonl(fixture["rollouts_path"], records)
        fixture["result"]["rollouts_sha256"] = new_hash
    elif target == "common_random_numbers":
        fixture["result"]["common_random_numbers"]["seed"] += 1
    elif target == "numerical_event_sequence":
        fixture["result"]["numerical_event_sequence"].reverse()
    elif target == "information_boundary":
        fixture["result"]["information_boundary"]["new_rollout_prompts_used_for_calibration"] = True
    else:
        monkeypatch.setattr(
            fixture["verifier"],
            "prepare_phase2_inputs",
            lambda *args, **kwargs: SimpleNamespace(
                seed=PHASE2_BUDGETED_END_TO_END_SEEDS[1],
                phase2_config_hash=DESIGN,
                source_config_hash=fixture["kwargs"]["base_config_hash"],
                artifact_metadata_sha256=fixture["kwargs"]["artifact_metadata_sha256"],
                run_manifest_sha256=fixture["result"]["run_manifest_sha256"],
                environment_identity=fixture["result"]["environment_identity"],
                artifact_dir=fixture["metadata_path"].parent,
                run_manifest=fixture["manifest_path"],
                test_prompts=(SimpleNamespace(prompt_id="test-0"),),
            ),
        )
    _write_json(fixture["result_path"], fixture["result"])

    with pytest.raises(ValueError, match="random|event|boundary|rollout|input"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


@pytest.mark.parametrize("target", ["duplicate_result_key", "nonfinite_rollout"])
def test_duplicate_keys_and_nonfinite_numbers_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if target == "duplicate_result_key":
        raw = fixture["result_path"].read_bytes().rstrip(b"\n")
        assert raw.endswith(b"}")
        fixture["result_path"].write_bytes(raw[:-1] + b',"seed":20261001}\n')
    else:
        records = copy.deepcopy(fixture["records"])
        records[0]["target_utility"] = float("nan")
        raw = b"".join(
            (
                json.dumps(
                    record,
                    allow_nan=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            for record in records
        )
        fixture["rollouts_path"].write_bytes(raw)
        fixture["result"]["rollouts_sha256"] = hashlib.sha256(raw).hexdigest()
        _write_json(fixture["result_path"], fixture["result"])

    with pytest.raises(ValueError, match="duplicate|non-finite"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


def test_runtime_contract_must_equal_the_validated_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    runtime = fixture["result"]["phase2_runtime_contract"]
    runtime["max_response_tokens"] = 512
    fixture["result"]["phase2_runtime_contract_sha256"] = hashlib.sha256(
        json.dumps(
            runtime,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(fixture["result_path"], fixture["result"])

    with pytest.raises(ValueError, match="validated overlay"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


def test_rollout_prompt_must_equal_materialization_prompt_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    records = copy.deepcopy(fixture["records"])
    replacement = "Cross-wired prompt"
    replacement_sha = hashlib.sha256(replacement.encode("utf-8")).hexdigest()
    for row in records:
        row["prompt"] = replacement
        row["prompt_semantics"]["raw_prompt_sha256"] = replacement_sha
    new_hash = _write_jsonl(fixture["rollouts_path"], records)
    fixture["result"]["rollouts_sha256"] = new_hash
    _write_json(fixture["result_path"], fixture["result"])

    with pytest.raises(ValueError, match="materialization prompt semantics"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


@pytest.mark.parametrize("location", ("train", "heldout", "continuity"))
def test_oracle_template_must_equal_prepared_qwen3_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if location == "train":
        fixture["result"]["train_oracle_rescore"]["oracle_chat_template_sha256"] = "3" * 64
    elif location == "heldout":
        fixture["result"]["heldout_fixed_beta"]["oracle_rescore"]["oracle_chat_template_sha256"] = (
            "3" * 64
        )
    else:
        fixture["result"]["information_boundary"]["prompt_semantics"]["oracle"][
            "policy_chat_tokens_reused_by_oracle"
        ] = True
    _write_json(fixture["result_path"], fixture["result"])

    with pytest.raises(ValueError, match="template closure"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


@pytest.mark.parametrize("target", ("utility", "kl_tail", "length"))
def test_result_arm_summary_must_equal_rollout_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if target == "utility":
        fixture["result"]["arms"]["bt_mle"]["utility"]["mean_target_utility"] += 1.0
    elif target == "kl_tail":
        fixture["result"]["arms"]["bt_mle"]["on_policy_kl_tail"]["p99"] += 1.0
    else:
        fixture["result"]["arms"]["bt_mle"]["rollout"]["reached_max_length_rate"] = 1.0
    _write_json(fixture["result_path"], fixture["result"])

    with pytest.raises(ValueError, match="summary|summaries|KL tail"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


def test_symlink_input_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    target = tmp_path.resolve() / "rollouts-target.jsonl"
    target.write_bytes(fixture["rollouts_path"].read_bytes())
    fixture["rollouts_path"].unlink()
    try:
        fixture["rollouts_path"].symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(ValueError, match="non-symlink"):
        _verify(fixture)
    assert not fixture["output_path"].exists()


def test_existing_output_symlink_is_rejected_without_touching_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    target = tmp_path.resolve() / "unrelated.json"
    target.write_text("unchanged\n", encoding="utf-8", newline="\n")
    try:
        fixture["output_path"].symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(FileExistsError, match="overwrite"):
        _verify(fixture)
    assert target.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.parametrize(
    "relative_output",
    (
        "wrong-name.json",
        "../phase2-budgeted-output-verification.json",
    ),
)
def test_verification_output_path_is_locked_to_job_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_output: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["output_path"] = fixture["result_path"].parent / relative_output

    with pytest.raises(ValueError, match="locked filename"):
        _verify(fixture)
    assert not fixture["output_path"].exists()
