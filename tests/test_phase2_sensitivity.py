from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from smart_reward.config import load_config
from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.phase2_aggregate import PHASE2_AGGREGATE_SCHEMA
from smart_reward.phase2_config import (
    PHASE2_CONFIRMATORY_SEEDS,
    load_phase2_config,
    phase2_design_identity,
    validate_phase2_config,
)
from smart_reward.phase2_rollout import PHASE2_ARM_ORDER, Phase2Design
from smart_reward.phase2_sensitivity import (
    BETA_SENSITIVITY_GRID,
    PHASE2_SENSITIVITY_AGGREGATE_SCHEMA,
    PHASE2_SENSITIVITY_SEED_SCHEMA,
    RIDGE_SENSITIVITY_GRID,
    build_phase2_sensitivity_aggregate,
    load_primary_sensitivity_binding,
)
from smart_reward.repeated_label_diagnostics import build_repeated_label_tail_diagnostics

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "configs" / "common_beta_pilot.yaml"
BASE = ROOT / "configs" / "common_beta_pilot_base.yaml"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _confirmatory() -> dict[str, Any]:
    overlay = copy.deepcopy(load_phase2_config(PILOT))
    base = load_config(BASE)
    seeds = list(PHASE2_CONFIRMATORY_SEEDS)
    overlay["design"].update(
        {
            "name": "common-beta-confirmatory-sensitivity-test",
            "stage": "confirmatory",
            "pilot_phase": None,
            "formal_eligibility": True,
            "evidence_role": "confirmatory_evidence",
            "source_config": "configs/future_confirmatory_base.yaml",
        }
    )
    overlay["run"].update(
        {
            "seeds": seeds,
            "confirmatory": True,
            "formal_eligibility": True,
            "excluded_from_confirmatory_evidence": False,
        }
    )
    base["run"]["name"] = "future-confirmatory-sensitivity-test"
    base["run"]["seeds"] = seeds
    from smart_reward.config import config_hash

    overlay["design"]["source_config_hash"] = config_hash(base)
    overlay["reward_model"]["identifiability"].update(
        {
            "role": "confirmatory_frozen_identifiability_contract",
            "confirmatory_freeze_requirement": "satisfied_by_current_confirmatory_identity",
        }
    )
    overlay["objective"]["common_beta"].update(
        {
            "rule": "single_pilot_frozen_global_beta_scalar",
            "calibration_split": "excluded_pilot",
            "calibration_source": ("frozen_pilot_global_beta_in_confirmatory_design_identity"),
            "frozen_global_beta": 2.5,
            "beta_source_aggregate_sha256": "a" * 64,
            "sensitivity_k_cal": None,
            "sensitivity_frozen_global_beta_multipliers": [0.5, 2.0],
            "primary_execution_role": "confirmatory_primary",
            "sensitivity_execution_role": (
                "required_separate_frozen_global_beta_multiplier_sensitivity"
            ),
        }
    )
    ridge = overlay["objective"]["full_tangent"]["ridge"]
    ridge["primary_execution_role"] = "confirmatory_primary"
    ridge["sensitivity_execution_role"] = "required_separate_confirmatory_sensitivity"
    overlay["evaluation"]["decision_gates"].update(
        {
            "application": "confirmatory_evidence_decision",
            "supports_formal_claim": True,
        }
    )
    overlay["evaluation"]["max_length"].update(
        {
            "role": "confirmatory_truncation_safety_gate",
            "measure_only": False,
            "formal_gate": True,
            "formal_threshold": 0.05,
            "parent_pilot_aggregate_sha256": "a" * 64,
            "post_pilot_requirement": "satisfied_by_new_confirmatory_design_identity",
        }
    )
    return validate_phase2_config(overlay)


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


def _primary_sha(seed: int) -> str:
    return hashlib.sha256(f"primary:{seed}".encode()).hexdigest()


def _primary_aggregate(
    tmp_path: Path,
    config: dict[str, Any],
    *,
    status: str = "passed",
) -> Path:
    runtime = Phase2Design.from_phase2_config(config)
    path = tmp_path / "primary-aggregate.json"
    payload = {
        "schema_version": PHASE2_AGGREGATE_SCHEMA,
        "source_config_hash": config["design"]["source_config_hash"],
        "phase2_design_sha256": phase2_design_identity(config),
        "phase2_runtime_contract_sha256": runtime.sha256,
        "environment_identity": _environment(),
        "seeds": list(PHASE2_CONFIRMATORY_SEEDS),
        "pre_registered_evidence": {
            "status": status,
            "supports_pre_registered_claim": status == "passed",
        },
        "sources": [
            {"seed": seed, "result_sha256": _primary_sha(seed)}
            for seed in PHASE2_CONFIRMATORY_SEEDS
        ],
    }
    _write(path, payload)
    return path


def _seed_artifact(
    path: Path,
    config: dict[str, Any],
    seed: int,
    *,
    unsafe_beta: float | None = None,
) -> None:
    runtime = Phase2Design.from_phase2_config(config)
    environment = _environment()
    primary_sha = _primary_sha(seed)
    primary_path = path.parent / "primary" / f"{seed}.json"
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_bytes(f"primary:{seed}".encode())
    assert hashlib.sha256(primary_path.read_bytes()).hexdigest() == primary_sha
    ridge_cells: list[dict[str, object]] = []
    for multiplier in RIDGE_SENSITIVITY_GRID:
        ridge_cells.append(
            {
                "schema_version": "phase2-ridge-sensitivity-cell/v1",
                "multiplier": multiplier,
                "relative_ridge": 0.001 * multiplier,
                "status": "primary_reference" if multiplier == 1.0 else "completed",
                "execution": {
                    "prorm_plus_head_retrained": multiplier != 1.0,
                    "bt_mle_head_retrained": False,
                    "source_primary_result_sha256": primary_sha,
                    "source_heads_sha256": "f" * 64,
                    **(
                        {}
                        if multiplier == 1.0
                        else {
                            "replayed_label_stream_sha256": "1" * 64,
                            "trained_prorm_plus_head_sha256": "2" * 64,
                            "training_evidence_sha256": "3" * 64,
                        }
                    ),
                },
                "heldout_test": {
                    BT_MLE: {"local_regret": 1.0 + 0.01 * (seed % 7)},
                    PRORM_PLUS: {"local_regret": 0.6 + 0.005 * (seed % 7)},
                },
                "eligible_for_primary_claim": False,
                "primary_result_modified": False,
            }
        )
    beta_cells: list[dict[str, object]] = []
    for multiplier in BETA_SENSITIVITY_GRID:
        unsafe = multiplier == unsafe_beta
        status = (
            "primary_reference"
            if multiplier == 1.0
            else "pre_oracle_safety_failed"
            if unsafe
            else "completed"
        )
        arms = (
            None
            if unsafe
            else {
                arm: {
                    "mean_target_reward": 1.0,
                    "mean_on_policy_kl": 0.01 if arm != "zero_b" else 0.0,
                    "mean_target_utility": (
                        0.0
                        if arm == "zero_b"
                        else 0.2
                        if arm == BT_MLE
                        else 0.4
                        if arm == PRORM_PLUS
                        else 0.5
                    ),
                }
                for arm in PHASE2_ARM_ORDER
            }
        )
        cell: dict[str, object] = {
            "schema_version": "phase2-beta-sensitivity-cell/v1",
            "multiplier": multiplier,
            "beta": 2.5 * multiplier,
            "status": status,
            "execution": {
                "reward_heads_reused": True,
                "reward_heads_retrained": False,
                "policy_redeployed": multiplier != 1.0,
                "rollout_reexecuted": multiplier != 1.0,
                "source_primary_result_sha256": primary_sha,
                **(
                    {"source_primary_rollouts_sha256": "4" * 64}
                    if multiplier == 1.0
                    else {
                        "source_heads_sha256": "f" * 64,
                        "rollout_identity_sha256": "5" * 64,
                    }
                ),
            },
            "pre_oracle_safety": {"passed": not unsafe},
            "arms": arms,
            "eligible_for_primary_claim": False,
            "primary_result_modified": False,
        }
        if unsafe:
            cell["outcome_oracle_called"] = False
        beta_cells.append(cell)
    payload: dict[str, object] = {
        "schema_version": PHASE2_SENSITIVITY_SEED_SCHEMA,
        "seed": seed,
        "source_config_hash": config["design"]["source_config_hash"],
        "phase2_design_sha256": phase2_design_identity(config),
        "phase2_runtime_contract_sha256": runtime.sha256,
        "environment_identity": environment,
        "environment_identity_sha256": _canonical_sha256(environment),
        "primary_result": {
            "path": f"primary/{seed}.json",
            "sha256": primary_sha,
            "artifact_metadata_sha256": "d" * 64,
            "run_manifest_sha256": "e" * 64,
            "heads_sha256": "f" * 64,
            "label_stream_sha256": "1" * 64,
            "beta0": 2.5,
            "relative_ridge0": 0.001,
            "read_only": True,
            "modified": False,
        },
        "grid_contract": {
            "ridge_multipliers": list(RIDGE_SENSITIVITY_GRID),
            "beta_multipliers": list(BETA_SENSITIVITY_GRID),
            "one_factor_at_a_time": True,
            "primary_reference_multiplier": 1.0,
            "all_cells_retained": True,
            "failed_cells_may_not_be_dropped": True,
        },
        "ridge_cells": ridge_cells,
        "beta_cells": beta_cells,
        "role": "required_secondary_sensitivity_only",
        "eligible_for_primary_claim": False,
        "primary_efficacy_status_read": False,
        "primary_efficacy_status_modified": False,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    _write(path, payload)


def _campaign(
    tmp_path: Path,
    config: dict[str, Any],
    *,
    unsafe_seed: int | None = None,
    unsafe_beta: float | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for seed in PHASE2_CONFIRMATORY_SEEDS:
        path = tmp_path / f"sensitivity-{seed}.json"
        _seed_artifact(
            path,
            config,
            seed,
            unsafe_beta=unsafe_beta if seed == unsafe_seed else None,
        )
        paths.append(path)
    return paths


def test_primary_binding_accepts_sorted_json_but_requires_exact_frozen_heads(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    runtime = Phase2Design.from_phase2_config(config)
    seed = PHASE2_CONFIRMATORY_SEEDS[0]
    heads = {
        BT_MLE: [0.1, -0.2],
        PRORM_PLUS: [0.3, -0.4],
    }
    heads_sha = _canonical_sha256(heads)
    train_prompts = int(config["run"]["split_sizes"]["train"])
    replicate_count_sha = "1" * 64
    replicate_h_sha = "2" * 64
    mean_h_sha = "3" * 64
    tail_diagnostics = build_repeated_label_tail_diagnostics(
        replicate_counts=torch.ones((4, train_prompts), dtype=torch.int64),
        replicate_h=torch.zeros((4, train_prompts), dtype=torch.float64),
        mean_h=torch.zeros(train_prompts, dtype=torch.float64),
        replicate_count_sha256=replicate_count_sha,
        replicate_h_sha256=replicate_h_sha,
        mean_h_sha256=mean_h_sha,
    )
    result = {
        "schema_version": "common-beta-finite-policy/v2",
        "design_stage": "confirmatory",
        "formal_eligibility": True,
        "per_seed_supports_formal_claim": False,
        "source_config_hash": config["design"]["source_config_hash"],
        "phase2_design_sha256": phase2_design_identity(config),
        "phase2_runtime_contract_sha256": runtime.sha256,
        "seed": seed,
        "artifact_metadata_sha256": "a" * 64,
        "run_manifest_sha256": "b" * 64,
        "environment_identity": _environment(),
        "train_oracle_rescore": {"transformed_rewards_sha256": "c" * 64},
        "head_training": {
            "head_weights": heads,
            "heads_sha256": heads_sha,
            "training_design_sha256": phase2_design_identity(config),
            "audit": {
                "label_stream": {
                    "label_stream_sha256": "d" * 64,
                    "replicate_count_sha256": replicate_count_sha,
                    "replicate_h_sha256": replicate_h_sha,
                    "mean_h_sha256": mean_h_sha,
                    "repeated_label_tail_diagnostics": tail_diagnostics,
                }
            },
        },
        "common_beta_calibration": {"beta_common": 2.5},
        "pre_oracle_safety_gate": {
            "schema_version": "phase2-pre-oracle-safety-gate/v1",
            "formal_gate": True,
            "measure_only": False,
            "passed": True,
            "violations": [],
            "beta_retuned": False,
        },
        "arms": {
            arm: {
                "utility": {
                    "mean_target_reward": 0.5,
                    "mean_on_policy_kl_pi_updated_to_pi0": (0.0 if arm == "zero_b" else 0.01),
                    "mean_target_utility": 0.4,
                }
            }
            for arm in PHASE2_ARM_ORDER
        },
        "heldout_fixed_beta": {
            "splits": {
                "test": {
                    "learners": {
                        learner: {"local_regret_at_frozen_global_beta": 0.2}
                        for learner in (BT_MLE, PRORM_PLUS)
                    }
                }
            }
        },
        "rollouts_sha256": "e" * 64,
    }
    path = tmp_path / "primary.json"
    _write(path, result)
    binding = load_primary_sensitivity_binding(config, path)
    assert binding.seed == seed
    assert binding.heads == {
        BT_MLE: (0.1, -0.2),
        PRORM_PLUS: (0.3, -0.4),
    }
    assert binding.heads_sha256 == heads_sha
    assert binding.repeated_label_tail_diagnostics_sha256 == tail_diagnostics["diagnostics_sha256"]
    assert binding.beta0 == 2.5
    assert tuple(binding.primary_arms) == PHASE2_ARM_ORDER

    result["head_training"]["head_weights"][BT_MLE][0] = 9.0
    _write(path, result)
    with pytest.raises(ValueError, match="heads_sha256"):
        load_primary_sensitivity_binding(config, path)


def test_sensitivity_aggregate_requires_and_uses_the_full_fixed_grid(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config, status="not_passed")
    paths = _campaign(tmp_path, config)
    payload = build_phase2_sensitivity_aggregate(
        config,
        paths,
        primary_aggregate_json=primary,
        reference_base=tmp_path,
    )

    assert payload["schema_version"] == PHASE2_SENSITIVITY_AGGREGATE_SCHEMA
    assert payload["seeds"] == list(PHASE2_CONFIRMATORY_SEEDS)
    assert payload["grid_contract"]["ridge_multipliers"] == [0.1, 1.0, 10.0]
    assert payload["grid_contract"]["beta_multipliers"] == [0.5, 1.0, 2.0]
    assert set(payload["ridge"]) == {"0.1", "1.0", "10.0"}
    assert set(payload["beta"]) == {"0.5", "1.0", "2.0"}
    assert payload["primary_aggregate"]["efficacy_status"] == "not_passed"
    assert payload["primary_aggregate"]["modified"] is False
    assert payload["claim_contract"]["eligible_to_modify_primary_efficacy_status"] is False
    assert all(cell["subset_interval_computed"] is False for cell in payload["beta"].values())


def test_sensitivity_aggregate_rejects_missing_seed_and_selected_subsets(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    with pytest.raises(ValueError, match="exactly 30"):
        build_phase2_sensitivity_aggregate(
            config,
            paths[:-1],
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


def test_sensitivity_aggregate_rejects_missing_or_reordered_grid_cell(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["ridge_cells"] = value["ridge_cells"][1:]
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = _canonical_sha256(unsigned)
    _write(paths[0], value)

    with pytest.raises(ValueError, match="omits, duplicates, or reorders"):
        build_phase2_sensitivity_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


def test_unsafe_beta_cell_is_retained_without_subset_interval(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    failed_seed = PHASE2_CONFIRMATORY_SEEDS[4]
    paths = _campaign(
        tmp_path,
        config,
        unsafe_seed=failed_seed,
        unsafe_beta=0.5,
    )
    payload = build_phase2_sensitivity_aggregate(
        config,
        paths,
        primary_aggregate_json=primary,
        reference_base=tmp_path,
    )
    cell = payload["beta"]["0.5"]
    assert cell["cell_status"] == "not_estimable_due_to_pre_oracle_safety_failure"
    assert cell["failed_seeds"] == [failed_seed]
    assert cell["paired_prorm_plus_minus_bt"] is None
    assert cell["subset_interval_computed"] is False
    assert payload["beta"]["2.0"]["cell_status"] == "completed_exact_30"


def test_unsafe_beta_cell_cannot_claim_outcome_oracle_was_called(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    failed_seed = PHASE2_CONFIRMATORY_SEEDS[0]
    paths = _campaign(
        tmp_path,
        config,
        unsafe_seed=failed_seed,
        unsafe_beta=0.5,
    )
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["beta_cells"][0]["outcome_oracle_called"] = True
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = _canonical_sha256(unsigned)
    _write(paths[0], value)
    with pytest.raises(ValueError, match="leaked outcome metrics"):
        build_phase2_sensitivity_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


def test_sensitivity_source_must_match_primary_aggregate_seed_binding(
    tmp_path: Path,
) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["primary_result"]["sha256"] = "9" * 64
    for cell in value["ridge_cells"]:
        cell["execution"]["source_primary_result_sha256"] = "9" * 64
    for cell in value["beta_cells"]:
        cell["execution"]["source_primary_result_sha256"] = "9" * 64
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = _canonical_sha256(unsigned)
    _write(paths[0], value)
    with pytest.raises(ValueError, match="byte hash does not match"):
        build_phase2_sensitivity_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


def test_sensitivity_artifact_hash_tampering_is_rejected(tmp_path: Path) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["ridge_cells"][0]["heldout_test"][BT_MLE]["local_regret"] = 999.0
    _write(paths[0], value)
    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        build_phase2_sensitivity_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )


def test_sensitivity_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    config = _confirmatory()
    primary = _primary_aggregate(tmp_path, config)
    paths = _campaign(tmp_path, config)
    paths[0].write_text(
        '{"schema_version":"phase2-confirmatory-sensitivity-seed/v1",'
        '"schema_version":"phase2-confirmatory-sensitivity-seed/v1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid strict JSON"):
        build_phase2_sensitivity_aggregate(
            config,
            paths,
            primary_aggregate_json=primary,
            reference_base=tmp_path,
        )
