from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smart_reward import phase2_recovery_aggregate as aggregate
from smart_reward.phase2_training import _tensor_sha256
from tests import test_phase2_aggregate as full_schema_fixture

AGGREGATOR_COMMIT = "b" * 40
REAL_DEEP_GATE = aggregate._deep_validate_recovery_training
_TEST_LIVE_USER_ID = "researcher(4242)"
_TEST_LIVE_COMMAND = "/home/researcher/Smart-Reward-Model/scripts/hpc4/phase2_recovery_pilot.sbatch"
_TEST_LIVE_WORK_DIR = "/home/researcher/Smart-Reward-Model"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _live_raw() -> bytes:
    rows: list[str] = []
    for task, (job_id, state) in enumerate(
        zip(aggregate._LIVE_JOB_IDS, aggregate._LIVE_STATES, strict=True)
    ):
        nodes = "1-1" if task == 2 else "1"
        tres = "cpu=8,mem=96G,node=1,billing=8,gres/gpu=1"
        if task != 2:
            tres += ",gres/gpu:l20=1"
        rows.append(
            " ".join(
                [
                    f"JobId={job_id}",
                    f"ArrayJobId={aggregate.SOURCE_ARRAY_JOB_ID}",
                    f"ArrayTaskId={task}",
                    "ArrayTaskThrottle=3",
                    "JobName=prorm-p2-recovery",
                    f"UserId={_TEST_LIVE_USER_ID}",
                    "Account=sigroup",
                    "QOS=l20_qos",
                    f"JobState={state}",
                    "Requeue=0",
                    "Restarts=0",
                    "BatchFlag=1",
                    "Reboot=0",
                    "TimeLimit=12:00:00",
                    "Partition=gpu-l20",
                    f"NumNodes={nodes}",
                    "NumCPUs=8",
                    "NumTasks=1",
                    "CPUs/Task=8",
                    f"TRES={tres}",
                    "TresPerNode=gres:gpu:1",
                    f"Command={_TEST_LIVE_COMMAND}",
                    f"WorkDir={_TEST_LIVE_WORK_DIR}",
                ]
            )
        )
    return (
        "\n".join(
            [
                f"schema={aggregate.RECOVERY_LIVE_CONTROL_SCHEMA}",
                f"captured_at={aggregate._LIVE_CONTROL_CAPTURED_AT}",
                f"command={aggregate._LIVE_CONTROL_COMMAND}",
                *rows,
            ]
        )
        + "\n"
    ).encode()


def _patch_test_live_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        aggregate,
        "_LIVE_USER_ID_SHA256",
        hashlib.sha256(_TEST_LIVE_USER_ID.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        aggregate,
        "_LIVE_COMMAND_SHA256",
        hashlib.sha256(_TEST_LIVE_COMMAND.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        aggregate,
        "_LIVE_WORK_DIR_SHA256",
        hashlib.sha256(_TEST_LIVE_WORK_DIR.encode()).hexdigest(),
    )


def _sacct_raw(
    *,
    job_ids: tuple[str, str, str] | None = None,
    states: tuple[str, str, str] = ("COMPLETED", "COMPLETED", "COMPLETED"),
) -> bytes:
    raw_ids = aggregate._LIVE_JOB_IDS if job_ids is None else job_ids
    return "".join(
        (
            f"{aggregate.SOURCE_ARRAY_JOB_ID}_{task}|{raw_ids[task]}|{states[task]}|"
            "0:0|0:0|hpc4|sigroup|gpu-l20|1|8|"
            f"{aggregate._SACCT_REQUEST_TRES}|{aggregate._SACCT_ALLOCATED_TRES}\n"
        )
        for task in range(3)
    ).encode()


def _scheduler_evidence(root: Path) -> Path:
    path = root / aggregate._SCHEDULER_EVIDENCE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_name(f"{path.stem}.sacct.psv")
    raw = _sacct_raw()
    raw_path.write_bytes(raw)
    _write_json(
        path,
        {
            "schema_version": aggregate.RECOVERY_SCHEDULER_EVIDENCE_SCHEMA,
            "source_command": list(aggregate._SACCT_COMMAND),
            "array_job_id": aggregate.SOURCE_ARRAY_JOB_ID,
            "captured_at_utc": "2026-07-25T08:00:00Z",
            "raw_sacct": {
                "filename": raw_path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            },
            "rows": aggregate._parse_sacct_raw(raw),
        },
    )
    return path


def _token_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _solver_measurement(
    solver: str,
    *,
    objective: float,
    gradient: float,
) -> dict[str, object]:
    inner: dict[str, object] | None
    if solver == "none":
        inner = None
    elif solver == "pcg":
        inner = {
            "method": "pcg",
            "dtype": "float64",
            "cold_start": True,
            "warm_start_used": False,
            "iterations": 7,
            "residual_norm": 1.0e-8,
            "relative_residual": 1.0e-8,
            "converged": True,
        }
    elif solver == "pseudoinverse":
        inner = {
            "method": "truncated_moore_penrose_pseudoinverse",
            "dtype": "float64",
            "cold_start": True,
            "warm_start_used": False,
            "numerical_rank": 256,
            "relative_eigenvalue_tolerance": 1.0e-10,
            "solve_residual_norm": 1.0e-9,
            "solve_relative_residual": 1.0e-9,
            "converged": True,
        }
    else:
        raise AssertionError(f"unknown test solver {solver}")
    return {
        "objective": objective,
        "gradient_l2_norm": gradient,
        "inner_solver": inner,
        "audit_dtype": "float64",
    }


def _history_summary(steps: int, *, solver: str) -> dict[str, object]:
    checkpoint_steps = sorted(
        {
            1,
            max(1, steps // 4),
            max(1, steps // 2),
            max(1, (3 * steps) // 4),
            steps,
        }
    )

    def checkpoint(step: int) -> dict[str, object]:
        dual = solver in {"pcg", "pseudoinverse"}
        iterative = solver == "pcg"
        return {
            "step": step,
            "objective": 0.5,
            "gradient_norm": 0.1,
            "dual_loss": 0.5 if dual else None,
            "dual_saddle_value": 0.5 if dual else None,
            "dual_refresh": step if dual else None,
            "pcg_iterations": 7 if iterative else None,
            "pcg_residual_norm": 1.0e-8 if iterative else None,
            "pcg_relative_residual": 1.0e-8 if iterative else None,
            "pcg_converged": True if iterative else None,
        }

    return {
        "num_steps": steps,
        "history_objective_timing": "pre_update",
        "stored_checkpoint_steps": checkpoint_steps,
        "checkpoints": [checkpoint(step) for step in checkpoint_steps],
        "objective": {
            "first": 0.9,
            "last_pre_update": 0.5,
            "minimum": 0.5,
            "maximum": 0.9,
        },
        "gradient_l2_norm": {
            "first": 1.0,
            "last_pre_update": 0.1,
            "minimum": 1.0e-6,
            "maximum": 1.0,
        },
        "pcg": (
            {
                "num_fresh_solves": steps,
                "all_converged": True,
                "maximum_relative_residual": 1.0e-8,
                "maximum_iterations": 7,
            }
            if solver == "pcg"
            else None
        ),
    }


def _strict_recovery_head(
    head: dict[str, object],
    *,
    seed: int,
    objective_name: str,
    solver: str,
    rank_evidence: dict[str, object],
    protocol: dict[str, object],
    zero_sha256: str,
) -> None:
    weight = head["head_weight"]
    assert isinstance(weight, list)
    head_sha256 = _tensor_sha256(torch.tensor(weight, dtype=torch.float32))
    head["head_dtype"] = "torch.float32"
    head["initial_head_sha256"] = zero_sha256
    head["head_sha256"] = head_sha256
    selected_step = 140
    initial_objective = float(head["initial_objective"])
    final_objective = float(head["final_objective"])
    checks: list[dict[str, object]] = []
    consecutive = 0
    for step in range(20, selected_step + 1, 20):
        gradient = 0.1 if step < 100 else 1.0e-6
        eligible = step >= 100
        passed = eligible and gradient <= 1.0e-3
        consecutive = consecutive + 1 if passed else 0
        checks.append(
            {
                "step": step,
                "post_update": True,
                "full_data": True,
                "gradient_clipping_applied": False,
                "measurement": _solver_measurement(
                    solver,
                    objective=final_objective,
                    gradient=gradient,
                ),
                "gradient_ratio_to_zero_initialization": gradient,
                "eligible_after_min_steps": eligible,
                "threshold_passed": passed,
                "consecutive_threshold_passes": consecutive,
                "learning_rate_used_for_update": 1.0e-3,
                "learning_rate_schedule_sha256": (aggregate.OPTIMIZER_SCHEDULE_SHA256),
            }
        )
    optimizer_state = _token_sha256(f"{seed}:{objective_name}:optimizer-state")
    state_dict = _token_sha256(f"{seed}:{objective_name}:optimizer-state-dict")
    checkpoint_sha = aggregate._canonical_sha256(
        {
            "schema_version": "selected-recovery-state-binding/v1",
            "completed_updates": selected_step,
            "head_sha256": head_sha256,
            "optimizer_state_sha256": optimizer_state,
            "optimizer_state_dict_sha256": state_dict,
        }
    )
    head["history_summary"] = _history_summary(selected_step, solver=solver)
    head["final_pcg"] = (
        {
            "method": "pcg",
            "dtype": "float64",
            "cold_start": True,
            "warm_start_used": False,
            "iterations": 7,
            "residual_norm": 1.0e-8,
            "relative_residual": 1.0e-8,
            "converged": True,
        }
        if solver == "pcg"
        else None
    )
    head["first_order_convergence"] = {
        "schema_version": "objective-first-order-convergence/v2",
        "objective": objective_name,
        "converged": True,
        "fail_closed": True,
        "spec": {
            "schema_version": "objective-first-order-convergence-spec/v2",
            "gradient_ratio_tolerance": 1.0e-3,
            "min_steps": 100,
            "max_steps": 12760,
            "check_interval": 20,
            "consecutive_checks": 3,
            "gradient_norm_denominator_floor": 1.0e-30,
            "fail_closed": True,
            "gradient": "full_data_post_update_unclipped",
            "denominator": "exact_zero_initialization_gradient_l2_norm",
            "validation_or_test_selection": False,
            "optimizer_protocol": copy.deepcopy(protocol),
        },
        "gradient_ratio_formula": (
            "||full_data_unclipped_gradient(w_t)||_2 / "
            "max(||full_data_unclipped_gradient(w_zero)||_2, denominator_floor)"
        ),
        "initial_zero_head_measurement": _solver_measurement(
            solver,
            objective=initial_objective,
            gradient=1.0,
        ),
        "checks": checks,
        "selected_primary_step": selected_step,
        "selected_primary_head_sha256": head_sha256,
        "consecutive_threshold_passes_at_selection": 3,
        "final_gate": {
            "step": selected_step,
            "measurement": _solver_measurement(
                solver,
                objective=final_objective,
                gradient=1.0e-6,
            ),
            "gradient_ratio_to_zero_initialization": 1.0e-6,
            "threshold_passed": True,
            "fresh_post_restore_audit": True,
            "learning_rate_at_selected_iterate": 1.0e-3,
        },
        "fixed_step_compute_matched_snapshot": {
            "schema_version": "fixed-step-compute-matched-snapshot/v1",
            "step": 720,
            "head_sha256": _token_sha256(f"{seed}:{objective_name}:fixed-720"),
            "measurement": _solver_measurement(
                solver,
                objective=final_objective,
                gradient=1.0e-6,
            ),
            "gradient_ratio_to_zero_initialization": 1.0e-6,
            "history_summary": _history_summary(720, solver=solver),
            "role": "compute_matched_and_pilot_diagnostic_only",
            "used_as_primary_selection_rule": False,
            "coincides_with_selected_primary_iterate": False,
        },
        "fixed_step_snapshot_steps": 720,
        "fixed_step_snapshot_is_not_primary_selection": True,
        "solution_identification": {
            "initialization": "exact_zero_head",
            "tie_break": aggregate._RECOVERY_TIE_BREAK,
            "primary_iterate_selection": (
                "first_scheduled_iterate_completing_the_sustained_first_order_gate"
            ),
            "validation_or_test_checkpoint_selection": False,
            "objective_value_checkpoint_selection": False,
            "minimum_norm_projection_applied": False,
            "minimum_norm_solution_claimed": False,
            "unique_reward_head_solution_claimed": False,
            "optional_objective_rank_diagnostic": {
                "evaluated": True,
                "evidence": copy.deepcopy(rank_evidence),
            },
            "minimum_norm_note": (
                "exact_zero_initialization_and_the_AdamW_path_are_reported; "
                "zero initialization alone does not prove an Euclidean "
                "minimum-norm solution under adaptive preconditioning"
            ),
        },
        "test_or_validation_data_accessed": False,
        "legacy_constant_lr_boundary_snapshot": {
            "schema_version": "legacy-constant-lr-boundary-snapshot/v1",
            "step": 5760,
            "head_sha256": _token_sha256(f"{seed}:{objective_name}:legacy-5760"),
            "measurement": _solver_measurement(
                solver,
                objective=final_objective,
                gradient=1.0e-6,
            ),
            "gradient_ratio_to_zero_initialization": 1.0e-6,
            "history_summary": _history_summary(5760, solver=solver),
            "learning_rate_used_for_update": 1.0e-3,
            "learning_rate_schedule_sha256": aggregate.OPTIMIZER_SCHEDULE_SHA256,
            "role": "immutable_legacy_constant_lr_failure_boundary_diagnostic",
            "used_as_primary_selection_rule": False,
            "coincides_with_selected_primary_iterate": False,
            "test_or_validation_data_accessed": False,
        },
        "optimizer_protocol_execution": {
            "schema_version": "deterministic-adamw-lr-decay-execution/v2",
            "protocol": copy.deepcopy(protocol),
            "optimizer_class": "torch.optim.AdamW",
            "parameter_count": 1,
            "fresh_optimizer_state_before_first_update": True,
            "reward_head_dtype_observed": "torch.float32",
            "first_order_audit_dtype_required": "float64",
            "microbatch_order": "canonical_edge_order_contiguous_ascending_no_shuffle",
            "one_optimizer_update_per_step": True,
            "learning_rate_set_immediately_before_every_update": True,
            "single_optimizer_instance_for_all_updates": True,
            "optimizer_state_reset_at_lr_milestone": False,
            "adamw_moments_preserved_at_learning_rate_boundaries": True,
            "boundary_transitions": [],
            "completed_updates_observed": 5760,
            "per_update_state_checks": {
                "schema_version": "recovery-adamw-per-update-state-checks/v1",
                "before_update_checks": 5760,
                "after_update_checks": 5760,
                "first_pre_update_state_empty": True,
                "completed_updates_covered": 5760,
                "check_sequence_sha256": _token_sha256(f"{seed}:{objective_name}:state-checks"),
                "all_updates_checked_before_and_after": True,
                "all_subsequent_pre_update_scalar_steps_exact": True,
                "all_post_update_scalar_steps_exact": True,
                "exp_avg_and_exp_avg_sq_shape_dtype_device_valid": True,
            },
            "selected_primary_optimizer_state_restored_without_reconstruction": True,
            "selected_primary_optimizer_state_restored_and_verified": True,
            "selected_optimizer_object_identity_preserved": True,
            "selected_optimizer_moments_restored_and_verified": True,
            "selected_head_sha256": head_sha256,
            "restored_head_sha256": head_sha256,
            "selected_optimizer_state_sha256": optimizer_state,
            "restored_optimizer_state_sha256": optimizer_state,
            "selected_checkpoint_optimizer_state_dict_sha256": state_dict,
            "restored_optimizer_state_dict_sha256": state_dict,
            "selected_checkpoint_sha256": checkpoint_sha,
            "test_or_validation_data_accessed": False,
        },
    }


def _production_schema_head_training(
    seed: int,
    *,
    transformed_reward_sha256: str,
) -> dict[str, object]:
    fixture_config = full_schema_fixture._post_recovery_config()
    recovery_config = aggregate._load_frozen_recovery_config()
    primary_weights = {
        "bt_mle": full_schema_fixture._fixture_head(0.1, 0.2),
        "prorm_plus": full_schema_fixture._fixture_head(0.3, 0.4),
    }
    audit = full_schema_fixture._head_training_audit(
        seed=seed,
        config=fixture_config,
        design_sha=aggregate.RECOVERY_DESIGN_SHA256,
        train_oracle_reward_sha=transformed_reward_sha256,
        head_weights=primary_weights,
        include_tail_diagnostics=False,
    )
    protocol = copy.deepcopy(recovery_config["reward_model"]["optimizer_protocol"])
    zero_sha256 = _tensor_sha256(torch.zeros(256, dtype=torch.float32))
    primary = audit["primary_heads"]
    low = audit["low_dimensional_control"]
    exact = audit["exact_margin_control"]
    exact_soft = audit["exact_soft_label_bt_control"]
    assert isinstance(primary, dict)
    assert isinstance(low, dict)
    assert isinstance(exact, dict)
    assert isinstance(exact_soft, dict)
    heads = (
        (
            primary["bt_mle"],
            "bt_mle",
            "none",
            audit["primary_optimization_audit"]["reward_head_identifiability"],
        ),
        (
            primary["prorm_plus"],
            "prorm_plus",
            "pcg",
            audit["primary_optimization_audit"]["prorm_moment_map_identifiability"],
        ),
        (
            low["head"],
            "low_dimensional_prorm_plus",
            "pseudoinverse",
            low["projected_prorm_moment_map_identifiability"],
        ),
        (
            exact["head"],
            "exact_margin_prorm_plus",
            "pcg",
            audit["primary_optimization_audit"]["prorm_moment_map_identifiability"],
        ),
        (
            exact_soft["head"],
            "exact_soft_label_bt_cross_entropy",
            "none",
            audit["primary_optimization_audit"]["reward_head_identifiability"],
        ),
    )
    for raw_head, objective, solver, rank in heads:
        assert isinstance(raw_head, dict)
        assert isinstance(rank, dict)
        _strict_recovery_head(
            raw_head,
            seed=seed,
            objective_name=objective,
            solver=solver,
            rank_evidence=rank,
            protocol=protocol,
            zero_sha256=zero_sha256,
        )
    audit["schema_version"] = "phase2-fresh-head-training/v3"
    audit["training_design_sha256"] = aggregate.RECOVERY_DESIGN_SHA256
    audit["training_settings_sha256"] = aggregate.TRAINING_SETTINGS_SHA256
    label = audit["label_stream"]
    assert isinstance(label, dict)
    label["oracle_reward_sha256"] = transformed_reward_sha256
    if seed == aggregate.ORDERED_SEEDS[0]:
        label.update(
            {
                "namespace": "prorm-common-beta-r4-labels-v1",
                "base_seed": 20260801,
                "derived_seed": 2443486425476852717,
                "derivation_sha256": (
                    "a1e901f534096bedb757ba978e2ba9838031aeacab4e86265557279953b236ae"
                ),
                "initial_state_sha256": (
                    "6f8e7260e641e4f52990e6f28c6558f333133bd0286549dbca3de426bf51a3d1"
                ),
                "final_state_sha256": (
                    "7dbc6a6143c98a995abda1baab73de8867267121c1299d80b6c10f8165f6ce83"
                ),
                "mean_h_sha256": (
                    "524eb0c9936dffe8d0ef807d4b4181cb3c4bbad556a51c7578a16c06b1e13cf0"
                ),
                "replicate_count_sha256": (
                    "b905ce98d6ec87a03bbda10405e3ffc766ff913bf9613e58bb50bf1ffa7b63c8"
                ),
                "replicate_win_sha256": (
                    "de68ae122cfe40a09146fdd00f367b640bdcc03a12aff46e6713fe7e88cafb13"
                ),
                "replicate_h_sha256": (
                    "92a99a227ce5679049c856b3fbd92005d0d9a8460760bc346a5ef266d7d3350d"
                ),
                "realized_total_annotations": 61011,
                "realized_annotations_per_edge": 61011 / 1536,
            }
        )
    label_payload = {
        "namespace": label["namespace"],
        "base_seed": label["base_seed"],
        "derived_seed": label["derived_seed"],
        "derivation_sha256": label["derivation_sha256"],
        "initial_state_sha256": label["initial_state_sha256"],
        "final_state_sha256": label["final_state_sha256"],
        "probability_sha256": label["canonical_probability_sha256"],
        "replicate_count_sha256": label["replicate_count_sha256"],
        "replicate_win_sha256": label["replicate_win_sha256"],
        "replicate_h_sha256": label["replicate_h_sha256"],
        "mean_h_sha256": label["mean_h_sha256"],
        "realized_total_annotations": label["realized_total_annotations"],
    }
    label["label_stream_sha256"] = aggregate._canonical_sha256(label_payload)
    low["label_stream_sha256"] = label["label_stream_sha256"]
    primary_audit = audit["primary_optimization_audit"]
    assert isinstance(primary_audit, dict)
    projection = low["projection"]
    direct = audit["direct_oracle_identity"]
    assert isinstance(projection, dict)
    assert isinstance(direct, dict)
    native = direct["native_oracle_direction"]
    assert isinstance(native, dict)
    input_training_sha256 = audit["input_training_sha256"]
    head_hashes = {
        "bt": primary["bt_mle"]["head_sha256"],
        "prorm": primary["prorm_plus"]["head_sha256"],
        "low": low["head"]["head_sha256"],
        "exact": exact["head"]["head_sha256"],
        "soft": exact_soft["head"]["head_sha256"],
    }
    low["bt_head"]["head_sha256"] = head_hashes["bt"]
    exact_soft["optimization_audit"]["head_sha256"] = head_hashes["soft"]
    training_instance = {
        "schema_version": "phase2-training-instance/v1",
        "phase2_config_hash": aggregate.RECOVERY_DESIGN_SHA256,
        "settings_sha256": aggregate.TRAINING_SETTINGS_SHA256,
        "input_training_sha256": input_training_sha256,
        "oracle_reward_sha256": transformed_reward_sha256,
        "seed": seed,
        "label_stream_sha256": label["label_stream_sha256"],
        "reward_head_identifiability_sha256": aggregate._canonical_sha256(
            primary_audit["reward_head_identifiability"]
        ),
        "prorm_moment_map_identifiability_sha256": aggregate._canonical_sha256(
            primary_audit["prorm_moment_map_identifiability"]
        ),
        "bt_head_sha256": head_hashes["bt"],
        "prorm_plus_head_sha256": head_hashes["prorm"],
        "low_dimensional_head_sha256": head_hashes["low"],
        "low_dimensional_projection_sha256": projection["projection_sha256"],
        "low_dimensional_moment_map_identifiability_sha256": (
            aggregate._canonical_sha256(low["projected_prorm_moment_map_identifiability"])
        ),
        "exact_margin_head_sha256": head_hashes["exact"],
        "exact_soft_label_bt_head_sha256": head_hashes["soft"],
        "direct_oracle_direction_sha256": native["direction_sha256"],
    }
    audit["training_instance_sha256"] = aggregate._canonical_sha256(training_instance)
    return {
        "schema_version": "phase2-fresh-head-training/v3",
        "heads": {
            "bt_mle": primary["bt_mle"]["head_weight"],
            "prorm_plus": primary["prorm_plus"]["head_weight"],
        },
        "audit": audit,
        "test_data_accessed": False,
    }


def _refresh_training_instance(
    training: dict[str, object],
    *,
    seed: int,
    oracle_reward_sha256: str,
) -> None:
    audit = training["audit"]
    assert isinstance(audit, dict)
    primary = audit["primary_heads"]
    primary_audit = audit["primary_optimization_audit"]
    low = audit["low_dimensional_control"]
    exact = audit["exact_margin_control"]
    exact_soft = audit["exact_soft_label_bt_control"]
    direct = audit["direct_oracle_identity"]
    assert all(
        isinstance(value, dict)
        for value in (primary, primary_audit, low, exact, exact_soft, direct)
    )
    projection = low["projection"]
    native = direct["native_oracle_direction"]
    assert isinstance(projection, dict)
    assert isinstance(native, dict)
    audit["training_instance_sha256"] = aggregate._canonical_sha256(
        {
            "schema_version": "phase2-training-instance/v1",
            "phase2_config_hash": aggregate.RECOVERY_DESIGN_SHA256,
            "settings_sha256": aggregate.TRAINING_SETTINGS_SHA256,
            "input_training_sha256": audit["input_training_sha256"],
            "oracle_reward_sha256": oracle_reward_sha256,
            "seed": seed,
            "label_stream_sha256": audit["label_stream"]["label_stream_sha256"],
            "reward_head_identifiability_sha256": aggregate._canonical_sha256(
                primary_audit["reward_head_identifiability"]
            ),
            "prorm_moment_map_identifiability_sha256": aggregate._canonical_sha256(
                primary_audit["prorm_moment_map_identifiability"]
            ),
            "bt_head_sha256": primary["bt_mle"]["head_sha256"],
            "prorm_plus_head_sha256": primary["prorm_plus"]["head_sha256"],
            "low_dimensional_head_sha256": low["head"]["head_sha256"],
            "low_dimensional_projection_sha256": projection["projection_sha256"],
            "low_dimensional_moment_map_identifiability_sha256": (
                aggregate._canonical_sha256(low["projected_prorm_moment_map_identifiability"])
            ),
            "exact_margin_head_sha256": exact["head"]["head_sha256"],
            "exact_soft_label_bt_head_sha256": exact_soft["head"]["head_sha256"],
            "direct_oracle_direction_sha256": native["direction_sha256"],
        }
    )


def _selected_parent(seed: int, task: int, root: Path) -> dict[str, object]:
    artifact_hashes = {
        "metadata.json": "a" * 64,
        "tensors.safetensors": "b" * 64,
        "candidates.jsonl": "c" * 64,
        "prompts.jsonl": "d" * 64,
        "training_edges.jsonl": "e" * 64,
        "evaluation_edges.jsonl": "f" * 64,
        "policy_prompt_semantics_records": "1" * 64,
        "selected_prompt_ids": "2" * 64,
    }
    evidence_hashes = {
        "FAILED": "3" * 64,
        "run-manifest.json": "4" * 64,
        "artifact-materialization.json": "5" * 64,
        "artifact-verification.json": "6" * 64,
        "phase2-run.log": "7" * 64,
    }
    source_run = f"parent-runs/seed-{seed}"
    source_artifact = f"parent-artifacts/seed-{seed}"
    return {
        "seed": seed,
        "array_task_id": task,
        "source_run": source_run,
        "source_artifact": source_artifact,
        "evidence_sha256": evidence_hashes,
        "artifact_sha256": artifact_hashes,
        "source_run_resolved": str(root / source_run),
        "source_artifact_resolved": str(root / source_artifact),
    }


def _manifest(seed: int, task: int) -> dict[str, object]:
    return {
        "schema_version": "smart-reward-run/v1",
        "created_at_utc": "2026-07-25T08:00:00Z",
        "config_hash": aggregate.SOURCE_CONFIG_HASH,
        "normalized_config": {},
        "seed": list(aggregate.ORDERED_SEEDS),
        "selected_seed": seed,
        "named_seeds": {},
        "git": {"commit": aggregate.RECOVERY_GIT_COMMIT, "dirty": False},
        "python": {},
        "platform": {},
        "torch": {
            "installed": True,
            "version": "2.7.1+cu126",
            "cuda_available": True,
            "cuda_version": "12.6",
            "cudnn_version": 90501,
            "gpu_count": 1,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA L20",
                    "total_memory_bytes": 47676129280,
                    "compute_capability": "8.9",
                }
            ],
        },
        "revisions": {},
        "packages": {},
        "slurm": {
            "PRORM_GIT_COMMIT": aggregate.RECOVERY_GIT_COMMIT,
            "PRORM_IMAGE_SHA256": aggregate.IMAGE_SHA256,
            "PRORM_HF_INVENTORY_SHA256": aggregate.HF_INVENTORY_SHA256,
            "SLURM_ARRAY_JOB_ID": aggregate.SOURCE_ARRAY_JOB_ID,
            "SLURM_ARRAY_TASK_ID": str(task),
            "SLURM_JOB_ID": aggregate._LIVE_JOB_IDS[task],
            "SLURM_CLUSTER_NAME": "hpc4",
            "SLURM_JOB_ACCOUNT": "sigroup",
            "SLURM_JOB_PARTITION": "gpu-l20",
            "SLURM_JOB_NAME": "prorm-p2-recovery",
            "SLURM_JOB_NODELIST": "gpu-l20-01",
            "SLURM_NNODES": "1",
            "SLURM_NTASKS": "1",
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_GPUS_ON_NODE": "1",
            "SLURM_PROCID": "0",
            "SLURM_LOCALID": "0",
            "SLURM_NODEID": "0",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    }


def _result(
    seed: int,
    manifest_sha256: str,
    selected: dict[str, object],
) -> dict[str, object]:
    environment = {
        "formal": True,
        "git_commit": aggregate.RECOVERY_GIT_COMMIT,
        "image_sha256": aggregate.IMAGE_SHA256,
        "hf_inventory_sha256": aggregate.HF_INVENTORY_SHA256,
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_models": ["NVIDIA L20"],
    }
    transformed = (
        "7a7d7b005ec7e377205d6f40743bed950ad38154dec6f54516f7ced8ffca0b1a"
        if seed == aggregate.ORDERED_SEEDS[0]
        else "8" * 64
    )
    head_training = _production_schema_head_training(
        seed,
        transformed_reward_sha256=transformed,
    )
    parent_entry = {
        key: value
        for key, value in selected.items()
        if key not in {"source_run_resolved", "source_artifact_resolved"}
    }
    artifact_hashes = selected["artifact_sha256"]
    assert isinstance(artifact_hashes, dict)
    return {
        "schema_version": "prorm-phase2-recovery-train-only-result/v1",
        "status": "SUCCESS",
        "design_stage": "pilot",
        "evidence_role": "one_shot_optimizer_recovery_train_only",
        "formal_eligibility": False,
        "per_seed_supports_formal_claim": False,
        "seed": seed,
        "source_config_hash": aggregate.SOURCE_CONFIG_HASH,
        "recovery_design_sha256": aggregate.RECOVERY_DESIGN_SHA256,
        "recovery_execution_identity": environment,
        "recovery_run_manifest_sha256": manifest_sha256,
        "parent_failure_binding": {
            "registry_sha256": aggregate.PARENT_REGISTRY_SHA256,
            "parent_phase2_design_sha256": aggregate.PARENT_DESIGN_SHA256,
            "parent_source_job_array_id": "1647491",
            "parent_seed_entry": parent_entry,
            "parent_artifact_producer": {
                "git_commit": aggregate.PARENT_PRODUCER_GIT_COMMIT,
                "image_sha256": aggregate.IMAGE_SHA256,
                "hf_inventory_sha256": aggregate.HF_INVENTORY_SHA256,
            },
            "parent_failure_aggregate_present": False,
            "exact_three_seed_failure_registry_used": True,
            "optimizer_diagnostic": {},
        },
        "artifact_reuse": {
            "mode": "immutable_parent_materialization_only",
            "metadata_sha256": artifact_hashes["metadata.json"],
            "tensor_file_sha256": artifact_hashes["tensors.safetensors"],
            "candidate_file_sha256": artifact_hashes["candidates.jsonl"],
            "producer_identity_separate_from_recovery_training_identity": True,
            "materialized_or_mutated_by_recovery": False,
        },
        "train_oracle_rescore": {
            "source": "saved_train_candidate_prefix_only",
            "num_prompts": 1536,
            "num_candidates": 4,
            "transformed_rewards_sha256": transformed,
            "oracle_chat_template_sha256": "a" * 64,
            "frozen_transform": {"b": 0.0, "tau": 1.0},
            "raw_oracle_logits_serialized": False,
        },
        "head_training": head_training,
        "information_boundary": dict(aggregate._RESULT_BOUNDARY),
        "one_shot_no_further_adaptation": True,
        "failure_action": "hard_fail_no_second_recovery",
    }


def _success(seed: int, task: int) -> str:
    return (
        "schema_version=prorm-phase2-recovery-run-status/v1\n"
        "status=SUCCESS\n"
        "workload_exit_code=0\n"
        "final_exit_code=0\n"
        f"array_job_id={aggregate.SOURCE_ARRAY_JOB_ID}\n"
        f"array_task_id={task}\n"
        f"seed={seed}\n"
        f"execution_revision={aggregate.EXECUTION_REVISION}\n"
        f"retry_reason={aggregate.RETRY_REASON}\n"
        f"recovery_design_sha256={aggregate.RECOVERY_DESIGN_SHA256}\n"
        f"base_config_hash={aggregate.SOURCE_CONFIG_HASH}\n"
        f"recovery_git_commit={aggregate.RECOVERY_GIT_COMMIT}\n"
        f"parent_design_sha256={aggregate.PARENT_DESIGN_SHA256}\n"
        f"parent_registry_sha256={aggregate.PARENT_REGISTRY_SHA256}\n"
        f"parent_producer_git_commit={aggregate.PARENT_PRODUCER_GIT_COMMIT}\n"
        "one_shot_no_further_adaptation=true\n"
        "created_at_utc=2026-07-25T08:00:00Z\n"
    )


def _campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Path], Path, Path, dict[int, dict[str, object]]]:
    root = tmp_path.resolve()
    monkeypatch.setattr(aggregate, "PRODUCTION_PROJECT_ROOT", root)
    if os.name == "nt":
        # Keep Windows test paths below MAX_PATH without changing any
        # production identity value.
        short_execution = Path("runs/phase2-recovery-pilot/design/execution-2")
        monkeypatch.setattr(
            aggregate,
            "_RECOVERY_EXECUTION_RELATIVE",
            short_execution,
        )
        monkeypatch.setattr(
            aggregate,
            "_LIVE_CONTROL_RELATIVE",
            short_execution / "live.txt",
        )

        def validate_short_run(path: Path, *, seed: int, array_task_id: int) -> None:
            expected = (
                root
                / short_execution
                / f"seed-{seed}"
                / f"job-{aggregate.SOURCE_ARRAY_JOB_ID}_{array_task_id}"
            )
            if path != expected:
                raise ValueError("ordered run directory has invalid test namespace")

        monkeypatch.setattr(aggregate, "_validate_run_path", validate_short_run)
    monkeypatch.setattr(
        aggregate,
        "_validate_claimed_aggregator_git_identity",
        lambda **_: None,
    )
    monkeypatch.setattr(aggregate, "_verify_cli_checkout", lambda _: None)
    monkeypatch.setattr(
        aggregate,
        "_deep_validate_recovery_training",
        lambda _value, *, seed, train_oracle_reward_sha256: {
            name: 140 for name in aggregate._FIVE_HEAD_NAMES
        },
    )

    live_path = root / aggregate._LIVE_CONTROL_RELATIVE
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live = _live_raw()
    live_path.write_bytes(live)
    live_path.chmod(0o440)
    monkeypatch.setattr(aggregate, "LIVE_CONTROL_SHA256", hashlib.sha256(live).hexdigest())
    monkeypatch.setattr(aggregate, "LIVE_CONTROL_SIZE_BYTES", len(live))
    _patch_test_live_identity(monkeypatch)

    selected = {
        seed: _selected_parent(seed, task, root)
        for task, seed in enumerate(aggregate.ORDERED_SEEDS)
    }
    monkeypatch.setattr(
        aggregate,
        "_validate_parent_receipts",
        lambda _path, *, seed: selected[seed],
    )

    original_is_symlink = Path.is_symlink
    original_readlink = os.readlink

    def synthetic_is_symlink(path: Path) -> bool:
        if path.name == aggregate._REQUIRED_REFERENCE and path.is_file():
            return True
        return original_is_symlink(path)

    def synthetic_readlink(path: str | os.PathLike[str]) -> str:
        candidate = Path(path)
        if candidate.name == aggregate._REQUIRED_REFERENCE and candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        return original_readlink(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_is_symlink)
    monkeypatch.setattr(os, "readlink", synthetic_readlink)

    execution = root / aggregate._RECOVERY_EXECUTION_RELATIVE
    paths: list[Path] = []
    for task, seed in enumerate(aggregate.ORDERED_SEEDS):
        run = execution / f"seed-{seed}" / f"job-{aggregate.SOURCE_ARRAY_JOB_ID}_{task}"
        run.mkdir(parents=True)
        for name in aggregate._REQUIRED_EVIDENCE_FILES:
            (run / name).write_text(f"{name}\n", encoding="utf-8", newline="\n")
        (run / aggregate._REQUIRED_REFERENCE).write_text(
            f"../../../../parent-artifacts/seed-{seed}",
            encoding="utf-8",
        )
        snapshot = b'{"same":true}\n'
        (run / "artifact-snapshot-before.json").write_bytes(snapshot)
        (run / "artifact-snapshot-after.json").write_bytes(snapshot)
        (run / "parent-run-snapshot-before.json").write_bytes(snapshot)
        (run / "parent-run-snapshot-after.json").write_bytes(snapshot)
        (run / "SUCCESS").write_text(
            _success(seed, task),
            encoding="utf-8",
            newline="\n",
        )
        manifest_path = run / "run-manifest.json"
        _write_json(manifest_path, _manifest(seed, task))
        _write_json(
            run / "gpu-check.json",
            {
                "status": "ok",
                "gpu_model": "NVIDIA L20",
                "cuda_device_count": 1,
            },
        )
        result_path = run / "recovery-result.json"
        result = _result(
            seed,
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            selected[seed],
        )
        _write_json(result_path, result)
        result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
        _write_json(
            run / "recovery-output-verification.json",
            aggregate._derive_recovery_output_verification(
                result,
                result_sha256=result_sha,
                seed=seed,
            ),
        )
        paths.append(run)
    scheduler = _scheduler_evidence(root)
    output = root / aggregate._AUTHORIZATION_RELATIVE
    return paths, scheduler, output, selected


def _build(
    paths: list[Path],
    scheduler: Path,
) -> dict[str, object]:
    return aggregate.build_phase2_recovery_authorization(
        paths,
        scheduler_evidence=scheduler,
        aggregator_git_commit=AGGREGATOR_COMMIT,
    )


def _refresh_result_receipt(run: Path) -> None:
    result_path = run / "recovery-result.json"
    verification_path = run / "recovery-output-verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    _write_json(verification_path, verification)


def _all_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [key for name, item in value.items() for key in [name, *_all_keys(item)]]
    if isinstance(value, list):
        return [key for item in value for key in _all_keys(item)]
    return []


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError:
        pytest.skip("host does not permit symlink creation")


def test_builds_head_free_schedule_only_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    payload = _build(paths, scheduler)

    assert payload["campaign_namespace"] == aggregate._campaign_namespace_identity()
    assert payload["ordered_seeds"] == [20260801, 20260802, 20260803]
    assert payload["full_calibration_authorized"] is True
    assert payload["authorized_information"] == "optimizer_schedule_only"
    assert payload["recovery_outputs_reusable"] is False
    assert payload["recovery_output_reuse"] == {
        "beta": False,
        "reward_model_parameters": False,
        "policy": False,
    }
    live = payload["supplementary_submission_control"]
    assert live["terminal_status_authority"] is False
    assert [row["state_at_capture"] for row in live["rows"]] == [
        "RUNNING",
        "RUNNING",
        "PENDING",
    ]
    assert [row["job_id_raw"] for row in payload["scheduler_terminal"]["rows"]] == [
        "1648126",
        "1648203",
        "1648125",
    ]
    assert all(
        forbidden not in key.lower()
        for key in _all_keys(payload)
        for forbidden in ("head", "vector", "path")
    )
    assert all(len(source["evidence"]) == 18 for source in payload["sources"])


def test_rejects_mirrored_recovery_tree_outside_production_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path / "production", monkeypatch)
    mirror = (tmp_path / "mirror").resolve()
    mirrored_paths = (
        [mirror / path.parent.name / path.name for path in paths]
        if os.name == "nt"
        else [
            mirror
            / "phase2-recovery-pilot"
            / aggregate.RECOVERY_DESIGN_SHA256
            / f"execution-{aggregate.EXECUTION_REVISION}"
            / path.parent.name
            / path.name
            for path in paths
        ]
    )
    for source, destination in zip(paths, mirrored_paths, strict=True):
        destination.mkdir(parents=True)
        for child in source.iterdir():
            if child.is_file():
                (destination / child.name).write_bytes(child.read_bytes())
    with pytest.raises(ValueError, match="frozen production execution"):
        _build(mirrored_paths, scheduler)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("SLURM_JOB_ID", "999999999"),
        ("SLURM_CLUSTER_NAME", "not-hpc4"),
        ("SLURM_NNODES", "999"),
        ("SLURM_NTASKS", "999"),
        ("SLURM_GPUS_ON_NODE", "0"),
    ],
)
def test_manifest_binds_terminal_job_id_and_runtime_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    manifest_path = paths[1] / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["slurm"][field] = replacement
    _write_json(manifest_path, manifest)
    result_path = paths[1] / "recovery-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["recovery_run_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _write_json(result_path, result)
    _refresh_result_receipt(paths[1])
    with pytest.raises(ValueError, match="run-manifest"):
        _build(paths, scheduler)


def test_deep_five_head_verification_rejects_empty_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        aggregate,
        "_deep_validate_recovery_training",
        REAL_DEEP_GATE,
    )
    result_path = paths[0] / "recovery-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["head_training"] = {}
    _write_json(result_path, result)
    _refresh_result_receipt(paths[0])
    with pytest.raises((TypeError, ValueError), match="head_training"):
        _build(paths, scheduler)


def test_ad7613_rich_final_pcg_schema_passes_unmocked_five_head_deep_gate() -> None:
    oracle_sha256 = "8" * 64
    training = _production_schema_head_training(
        20260802,
        transformed_reward_sha256=oracle_sha256,
    )
    raw_primary_pcg = copy.deepcopy(training["audit"]["primary_heads"]["prorm_plus"]["final_pcg"])
    assert set(raw_primary_pcg) == {
        "method",
        "dtype",
        "cold_start",
        "warm_start_used",
        "iterations",
        "residual_norm",
        "relative_residual",
        "converged",
    }

    assert aggregate._deep_validate_recovery_training(
        training,
        seed=20260802,
        train_oracle_reward_sha256=oracle_sha256,
    ) == {name: 140 for name in aggregate._FIVE_HEAD_NAMES}
    assert training["audit"]["primary_heads"]["prorm_plus"]["final_pcg"] == raw_primary_pcg


@pytest.mark.parametrize("mutation", ["future_five_field_projection", "fresh_audit_drift"])
def test_ad7613_raw_final_pcg_must_remain_exact_rich_fresh_audit(
    mutation: str,
) -> None:
    oracle_sha256 = "8" * 64
    training = _production_schema_head_training(
        20260802,
        transformed_reward_sha256=oracle_sha256,
    )
    final_pcg = training["audit"]["primary_heads"]["prorm_plus"]["final_pcg"]
    assert isinstance(final_pcg, dict)
    if mutation == "future_five_field_projection":
        final_pcg.pop("method")
        final_pcg.pop("dtype")
        final_pcg.pop("warm_start_used")
    else:
        final_pcg["iterations"] = 8

    with pytest.raises(ValueError, match="final_pcg"):
        aggregate._deep_validate_recovery_training(
            training,
            seed=20260802,
            train_oracle_reward_sha256=oracle_sha256,
        )


@pytest.mark.parametrize(
    ("location", "field", "replacement"),
    [
        ("checks", "step", 21),
        ("checks", "eligible_after_min_steps", True),
        ("checks", "consecutive_threshold_passes", 2),
        ("checks", "learning_rate_used_for_update", 3.0e-4),
        ("execution", "completed_updates_observed", 5761),
    ],
)
def test_real_deep_gate_rejects_selection_schedule_and_counter_tamper(
    location: str,
    field: str,
    replacement: object,
) -> None:
    oracle_sha256 = "8" * 64
    training = _production_schema_head_training(
        20260802,
        transformed_reward_sha256=oracle_sha256,
    )
    audit = training["audit"]
    assert isinstance(audit, dict)
    head = audit["primary_heads"]["bt_mle"]
    convergence = head["first_order_convergence"]
    if location == "checks":
        convergence["checks"][0][field] = replacement
    else:
        convergence["optimizer_protocol_execution"][field] = replacement

    with pytest.raises(ValueError):
        aggregate._deep_validate_recovery_training(
            training,
            seed=20260802,
            train_oracle_reward_sha256=oracle_sha256,
        )


def test_coordinated_result_receipt_and_claimed_head_hash_tamper_still_fails(
    tmp_path: Path,
) -> None:
    seed = 20260802
    oracle_sha256 = "8" * 64
    training = _production_schema_head_training(
        seed,
        transformed_reward_sha256=oracle_sha256,
    )
    audit = training["audit"]
    assert isinstance(audit, dict)
    primary = audit["primary_heads"]
    low = audit["low_dimensional_control"]
    assert isinstance(primary, dict)
    assert isinstance(low, dict)
    head = primary["bt_mle"]
    assert isinstance(head, dict)
    tampered_weight = copy.deepcopy(head["head_weight"])
    tampered_weight[0] += 0.25
    forged_sha256 = _token_sha256("coordinated-attacker-claimed-head")
    head["head_weight"] = tampered_weight
    training["heads"]["bt_mle"] = copy.deepcopy(tampered_weight)
    head["head_sha256"] = forged_sha256
    convergence = head["first_order_convergence"]
    convergence["selected_primary_head_sha256"] = forged_sha256
    execution = convergence["optimizer_protocol_execution"]
    execution["selected_head_sha256"] = forged_sha256
    execution["restored_head_sha256"] = forged_sha256
    execution["selected_checkpoint_sha256"] = aggregate._canonical_sha256(
        {
            "schema_version": "selected-recovery-state-binding/v1",
            "completed_updates": convergence["selected_primary_step"],
            "head_sha256": forged_sha256,
            "optimizer_state_sha256": execution["selected_optimizer_state_sha256"],
            "optimizer_state_dict_sha256": execution[
                "selected_checkpoint_optimizer_state_dict_sha256"
            ],
        }
    )
    low["bt_head"]["head_sha256"] = forged_sha256
    _refresh_training_instance(
        training,
        seed=seed,
        oracle_reward_sha256=oracle_sha256,
    )
    result = _result(
        seed,
        "a" * 64,
        _selected_parent(seed, 1, tmp_path.resolve()),
    )
    result["head_training"] = training
    forged_result_sha256 = hashlib.sha256(_canonical(result)).hexdigest()
    forged_receipt = {
        "status": "ok",
        "result_sha256": forged_result_sha256,
        "five_head_recovery_protocol_verified": True,
        "selected_primary_steps": {name: 140 for name in aggregate._FIVE_HEAD_NAMES},
        "diagnostic_seed_reproduction": {
            "anchor_seed": 20260801,
            "applicable": False,
            "passed": None,
        },
    }

    with pytest.raises(ValueError, match="tensor SHA256"):
        aggregate._validate_output_verification(
            forged_receipt,
            result_sha256=forged_result_sha256,
            seed=seed,
            result=result,
        )


def test_rederived_receipt_rejects_forged_selected_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    verification_path = paths[2] / "recovery-output-verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["selected_primary_steps"]["primary_bt_mle"] += 20
    _write_json(verification_path, verification)
    with pytest.raises(ValueError, match="rederived five-head receipt"):
        _build(paths, scheduler)


def test_result_parent_artifact_hashes_must_match_fresh_registry_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    result_path = paths[2] / "recovery-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["artifact_reuse"]["metadata_sha256"] = "0" * 64
    _write_json(result_path, result)
    _refresh_result_receipt(paths[2])
    with pytest.raises(ValueError, match="train-only contract"):
        _build(paths, scheduler)


def test_parent_receipts_must_equal_fresh_frozen_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path.resolve()
    derived = {
        "status": "ok",
        "schema_version": "prorm-phase2-recovery-parent-failures/v1",
        "registry_sha256": aggregate.PARENT_REGISTRY_SHA256,
        "campaign": {},
        "selected_seed": {"seed": 20260801},
        "all_three_sources_verified": True,
    }
    _write_json(run / "parent-verification-before.json", derived)
    changed = copy.deepcopy(derived)
    changed["status"] = "changed"
    _write_json(run / "parent-verification-after.json", changed)
    monkeypatch.setattr(
        aggregate,
        "_derive_parent_verification",
        lambda _seed: (derived, _canonical(derived)),
    )
    monkeypatch.setattr(aggregate, "_validate_parent_artifact_reference", lambda *_a, **_k: None)
    monkeypatch.setattr(aggregate, "_validate_parent_snapshots", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="fresh frozen revalidation"):
        aggregate._validate_parent_receipts(run, seed=20260801)


def test_frozen_parent_registry_and_validator_match_ad7613_git_blobs() -> None:
    registry, validator = aggregate._verify_frozen_parent_support()
    assert hashlib.sha256(registry.read_bytes()).hexdigest() == aggregate.PARENT_REGISTRY_SHA256
    assert hashlib.sha256(validator.read_bytes()).hexdigest() == aggregate._PARENT_VALIDATOR_SHA256


def test_parent_artifact_reference_must_resolve_to_selected_registry_artifact(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    artifact = tmp_path / "artifact"
    wrong = tmp_path / "wrong"
    run.mkdir()
    artifact.mkdir()
    wrong.mkdir()
    link = run / "parent-artifact"
    _symlink_or_skip(link, wrong, target_is_directory=True)
    selected = {"source_artifact_resolved": str(artifact.resolve())}
    with pytest.raises(ValueError, match="frozen registry artifact"):
        aggregate._validate_parent_artifact_reference(run.resolve(), selected=selected)


def test_required_parent_artifact_cannot_be_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    (paths[0] / aggregate._REQUIRED_REFERENCE).unlink()
    with pytest.raises(ValueError, match="missing required evidence"):
        _build(paths, scheduler)


@pytest.mark.parametrize(
    ("field", "source", "replacement"),
    [
        ("UserId", _TEST_LIVE_USER_ID, "other-researcher(4242)"),
        ("Command", _TEST_LIVE_COMMAND, "/srv/other/phase2_recovery_pilot.sbatch"),
        ("WorkDir", _TEST_LIVE_WORK_DIR, "/srv/other"),
    ],
)
def test_live_receipt_identity_fields_are_exact_digest_bound(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    source: str,
    replacement: str,
) -> None:
    _patch_test_live_identity(monkeypatch)
    raw = _live_raw()
    mutated = raw.replace(
        f"{field}={source}".encode(),
        f"{field}={replacement}".encode(),
    )
    assert mutated != raw
    with pytest.raises(ValueError, match="frozen submission receipt"):
        aggregate._parse_live_control(mutated)


def test_live_receipt_is_required_hash_bound_and_not_terminal_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    live_path = tmp_path.resolve() / aggregate._LIVE_CONTROL_RELATIVE
    live_path.chmod(0o640)
    live_path.write_bytes(live_path.read_bytes() + b"tampered\n")
    with pytest.raises(ValueError, match="bytes changed"):
        _build(paths, scheduler)


def test_terminal_sacct_must_bind_nonconsecutive_live_job_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    raw_path = scheduler.with_name(f"{scheduler.stem}.sacct.psv")
    raw = _sacct_raw(job_ids=("1648126", "1648204", "1648125"))
    raw_path.write_bytes(raw)
    evidence = json.loads(scheduler.read_text(encoding="utf-8"))
    evidence["raw_sacct"]["sha256"] = hashlib.sha256(raw).hexdigest()
    evidence["raw_sacct"]["size_bytes"] = len(raw)
    evidence["rows"] = aggregate._parse_sacct_raw(raw)
    _write_json(scheduler, evidence)
    with pytest.raises(ValueError, match="allocation IDs differ"):
        _build(paths, scheduler)


@pytest.mark.parametrize(
    "raw",
    [
        _sacct_raw().replace(b"1648125_0|", b"1648125_[0-2]|", 1),
        _sacct_raw().replace(b"1648125_0|", b"1648125|", 1),
        _sacct_raw().replace(b"1648125_0|", b"1648125_0.batch|", 1),
        _sacct_raw(states=("FAILED", "COMPLETED", "COMPLETED")),
    ],
)
def test_sacct_parser_rejects_range_parent_step_and_failure(raw: bytes) -> None:
    with pytest.raises(ValueError, match="exact successful task allocation"):
        aggregate._parse_sacct_raw(raw)


def test_sacct_command_matches_supported_hpc4_terminal_query_exactly() -> None:
    assert aggregate._SACCT_COMMAND == (
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        "1648125",
        (
            "--format=JobID,JobIDRaw,State,ExitCode,DerivedExitCode,Cluster,"
            "Account,Partition,NNodes,NCPUS,ReqTRES,AllocTRES"
        ),
    )
    assert "--array" not in aggregate._SACCT_COMMAND
    assert "Restarts" not in aggregate._SACCT_FORMAT


@pytest.mark.parametrize(
    "raw",
    [
        _sacct_raw().replace(b"\n", b"|\n", 1),
        _sacct_raw().replace(b"|1|8|", b"|2|8|", 1),
        _sacct_raw().replace(b"|1|8|", b"|1|16|", 1),
        _sacct_raw().replace(b"gres/gpu:l20=1,", b"", 1),
    ],
)
def test_sacct_parser_rejects_trailing_field_and_resource_drift(raw: bytes) -> None:
    with pytest.raises(ValueError):
        aggregate._parse_sacct_raw(raw)


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        (
            "gpu-check.json",
            lambda value: value.update({"gpu_model": "NVIDIA A100-SXM4-80GB"}),
        ),
        (
            "run-manifest.json",
            lambda value: value["torch"]["gpus"][0].update({"total_memory_bytes": 1}),
        ),
        (
            "run-manifest.json",
            lambda value: value["slurm"].update({"UNRECOGNIZED_ENV": "unsafe"}),
        ),
    ],
)
def test_gpu_check_and_manifest_real_schema_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    mutation: object,
) -> None:
    paths, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    target = paths[1] / filename
    value = json.loads(target.read_text(encoding="utf-8"))
    mutation(value)
    _write_json(target, value)
    if filename == "run-manifest.json":
        result_path = paths[1] / "recovery-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["recovery_run_manifest_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        _write_json(result_path, result)
        _refresh_result_receipt(paths[1])

    with pytest.raises(ValueError, match="gpu-check|run-manifest"):
        _build(paths, scheduler)


def test_writes_canonical_exact_namespace_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, output, _ = _campaign(tmp_path, monkeypatch)
    payload = aggregate.write_phase2_recovery_authorization(
        paths,
        output,
        scheduler_evidence=scheduler,
        aggregator_git_commit=AGGREGATOR_COMMIT,
    )
    assert output.read_bytes() == _canonical(payload)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        aggregate.write_phase2_recovery_authorization(
            paths,
            output,
            scheduler_evidence=scheduler,
            aggregator_git_commit=AGGREGATOR_COMMIT,
        )
    with pytest.raises(ValueError, match="frozen campaign namespace"):
        aggregate.write_phase2_recovery_authorization(
            paths,
            tmp_path / "elsewhere.json",
            scheduler_evidence=scheduler,
            aggregator_git_commit=AGGREGATOR_COMMIT,
        )


def test_public_verifier_rechecks_external_sha_canonical_and_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, output, _ = _campaign(tmp_path, monkeypatch)
    aggregate.write_phase2_recovery_authorization(
        paths,
        output,
        scheduler_evidence=scheduler,
        aggregator_git_commit=AGGREGATOR_COMMIT,
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert aggregate.verify_phase2_recovery_authorization(output, digest)[
        "full_calibration_authorized"
    ]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        aggregate.verify_phase2_recovery_authorization(output, "f" * 64)
    noncanonical = json.loads(output.read_text(encoding="utf-8"))
    output.write_text(json.dumps(noncanonical, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical JSON"):
        aggregate.verify_phase2_recovery_authorization(
            output,
            hashlib.sha256(output.read_bytes()).hexdigest(),
        )


def test_public_verifier_uses_embedded_live_receipt_without_source_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, output, _ = _campaign(tmp_path, monkeypatch)
    aggregate.write_phase2_recovery_authorization(
        paths,
        output,
        scheduler_evidence=scheduler,
        aggregator_git_commit=AGGREGATOR_COMMIT,
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    live_path = tmp_path.resolve() / aggregate._LIVE_CONTROL_RELATIVE
    live_path.chmod(0o600)
    live_path.unlink()

    assert aggregate.verify_phase2_recovery_authorization(output, digest)[
        "full_calibration_authorized"
    ]


def test_claimed_nonexistent_aggregation_commit_is_rejected() -> None:
    with pytest.raises((RuntimeError, ValueError)):
        aggregate._validate_claimed_aggregator_git_identity(
            aggregator_git_commit="b" * 40,
            validator_sha256=aggregate._validator_source_sha256(),
            deep_gate_source_sha256="a" * 64,
            tensor_hash_source_sha256="b" * 64,
            config_validator_source_sha256="c" * 64,
            recovery_config_sha256=aggregate._RECOVERY_CONFIG_SHA256,
        )


def test_cli_checkout_checks_head_clean_status_and_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        "src/smart_reward/phase2_recovery_aggregate.py": Path(aggregate.__file__).read_bytes(),
        aggregate._DEEP_GATE_SOURCE_RELATIVE.as_posix(): (
            aggregate._repository_root() / aggregate._DEEP_GATE_SOURCE_RELATIVE
        ).read_bytes(),
        aggregate._TENSOR_HASH_SOURCE_RELATIVE.as_posix(): (
            aggregate._repository_root() / aggregate._TENSOR_HASH_SOURCE_RELATIVE
        ).read_bytes(),
        aggregate._CONFIG_VALIDATOR_SOURCE_RELATIVE.as_posix(): (
            aggregate._repository_root() / aggregate._CONFIG_VALIDATOR_SOURCE_RELATIVE
        ).read_bytes(),
    }

    def clean_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        arguments = command[3:]
        text = bool(kwargs.get("text"))
        if arguments[:2] == ["rev-parse", "--verify"]:
            stdout: str | bytes = AGGREGATOR_COMMIT + "\n"
        elif arguments[:2] == ["status", "--porcelain"]:
            stdout = ""
        else:
            specification = str(arguments[-1])
            relative = specification.split(":", maxsplit=1)[1]
            stdout = sources[relative]
        return SimpleNamespace(
            returncode=0,
            stdout=stdout,
            stderr="" if text else b"",
        )

    monkeypatch.setattr(aggregate.subprocess, "run", clean_run)
    aggregate._verify_cli_checkout(AGGREGATOR_COMMIT)

    def dirty_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        completed = clean_run(command, **kwargs)
        if command[3:5] == ["status", "--porcelain"]:
            completed.stdout = " M src/smart_reward/phase2_recovery_aggregate.py\n"
        return completed

    monkeypatch.setattr(aggregate.subprocess, "run", dirty_run)
    with pytest.raises(ValueError, match="exact clean committed"):
        aggregate._verify_cli_checkout(AGGREGATOR_COMMIT)


def test_cli_rechecks_checkout_immediately_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths, scheduler, output, _ = _campaign(tmp_path, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        aggregate,
        "_verify_cli_checkout",
        lambda commit: calls.append(commit),
    )
    assert (
        aggregate.main(
            [
                str(output),
                *(str(path) for path in paths),
                "--scheduler-evidence",
                str(scheduler),
                "--aggregator-git-commit",
                AGGREGATOR_COMMIT,
            ]
        )
        == 0
    )
    assert calls == [AGGREGATOR_COMMIT, AGGREGATOR_COMMIT]
    assert json.loads(capsys.readouterr().out)["status"] == "authorized"


def test_cli_reports_precomputed_canonical_digest_without_rehashing_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths, scheduler, output, _ = _campaign(tmp_path, monkeypatch)
    original = aggregate._sha256_file

    def guarded_sha256(path: Path, *, name: str = "source file") -> str:
        if path.absolute() == output.absolute():
            raise AssertionError("CLI attempted to rehash the published output path")
        return original(path, name=name)

    monkeypatch.setattr(aggregate, "_sha256_file", guarded_sha256)
    assert (
        aggregate.main(
            [
                str(output),
                *(str(path) for path in paths),
                "--scheduler-evidence",
                str(scheduler),
                "--aggregator-git-commit",
                AGGREGATOR_COMMIT,
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert (
        report["sha256"]
        == hashlib.sha256(_canonical(json.loads(output.read_text(encoding="utf-8")))).hexdigest()
    )


def test_exclusive_publication_rejects_published_inode_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    original_fstat = os.fstat
    first_regular_inode: tuple[int, int] | None = None

    def mismatching_fstat(descriptor: int) -> os.stat_result:
        nonlocal first_regular_inode
        observed = original_fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            return observed
        identity = (observed.st_dev, observed.st_ino)
        if first_regular_inode is None:
            first_regular_inode = identity
            return observed
        if identity == first_regular_inode:
            fields = list(observed)
            fields[1] = observed.st_ino + 1
            return os.stat_result(fields)
        return observed

    monkeypatch.setattr(aggregate.os, "fstat", mismatching_fstat)
    with pytest.raises(ValueError, match="inode"):
        aggregate._write_exclusive_bytes(
            destination,
            b'{"status":"ok"}\n',
            label="test artifact",
        )
    assert not destination.exists()


def test_scheduler_json_failure_unlinks_raw_and_fsyncs_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    raw_path = scheduler.with_name(f"{scheduler.stem}.sacct.psv")
    scheduler.unlink()
    raw_path.unlink()
    calls = 0
    synced: list[Path] = []

    def write_then_fail(path: Path, payload: bytes, *, label: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            path.write_bytes(payload)
            return hashlib.sha256(payload).hexdigest()
        raise OSError("simulated canonical JSON publication failure")

    monkeypatch.setattr(aggregate, "_write_exclusive_bytes", write_then_fail)
    monkeypatch.setattr(
        aggregate,
        "_fsync_directory",
        lambda path: synced.append(path),
    )
    monkeypatch.setattr(
        aggregate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=_sacct_raw(),
            stderr=b"",
        ),
    )

    with pytest.raises(OSError, match="simulated"):
        aggregate.capture_phase2_recovery_scheduler_evidence(scheduler)
    assert not raw_path.exists()
    assert synced == [raw_path.parent]


def test_checked_in_capture_requires_exact_namespace_and_preserves_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, scheduler, _, _ = _campaign(tmp_path, monkeypatch)
    raw_path = scheduler.with_name(f"{scheduler.stem}.sacct.psv")
    scheduler.unlink()
    raw_path.unlink()
    raw = _sacct_raw()
    monkeypatch.setattr(
        aggregate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=raw,
            stderr=b"",
        ),
    )
    payload = aggregate.capture_phase2_recovery_scheduler_evidence(
        scheduler,
        now=datetime(2026, 7, 25, 8, 0, tzinfo=timezone.utc),
    )
    assert raw_path.read_bytes() == raw
    assert scheduler.read_bytes() == _canonical(payload)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        aggregate.capture_phase2_recovery_scheduler_evidence(scheduler)
    with pytest.raises(ValueError, match="frozen campaign namespace"):
        aggregate.capture_phase2_recovery_scheduler_evidence(tmp_path / "wrong.json")


def test_source_change_during_publication_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scheduler, output, _ = _campaign(tmp_path, monkeypatch)
    original = aggregate._snapshot_evidence
    calls = 0

    def changing_snapshot(path: Path) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        value = original(path)
        if calls > 6 and path == paths[0]:
            value = copy.deepcopy(value)
            value[0]["sha256"] = "f" * 64
        return value

    monkeypatch.setattr(aggregate, "_snapshot_evidence", changing_snapshot)
    with pytest.raises(ValueError, match="changed"):
        aggregate.write_phase2_recovery_authorization(
            paths,
            output,
            scheduler_evidence=scheduler,
            aggregator_git_commit=AGGREGATOR_COMMIT,
        )
