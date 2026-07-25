from __future__ import annotations

import copy
import inspect
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import smart_reward.phase2_aggregate as phase2_aggregate
import smart_reward.phase2_training as phase2_training
from smart_reward.config import load_config
from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_config import (
    PHASE2_BUDGETED_END_TO_END_SEEDS,
    PHASE2_BUDGETED_END_TO_END_STAGE,
    PHASE2_CONFIRMATORY_SEEDS,
    PHASE2_POST_RECOVERY_CALIBRATION_SCHEMA_VERSION,
    PHASE2_RECOVERY_LR_SCHEDULE_SHA256,
    PHASE2_RECOVERY_PILOT_CONFIG,
    load_phase2_config_bundle,
)
from smart_reward.phase2_controls import (
    build_exact_margin_canonical_arm,
    sample_canonical_r4_noisy_arm,
)
from smart_reward.phase2_rollout import Phase2HeadTrainingResult
from smart_reward.phase2_training import (
    EXACT_SOFT_BT_ARM,
    EXACT_SOFT_BT_INPUT,
    EXACT_SOFT_BT_ROLE,
    LABEL_RNG_NAMESPACE,
    PRIMARY_TRAINING_ARM,
    AdamWRecoveryProtocol,
    FirstOrderConvergenceSpec,
    FreshPhase2HeadTrainer,
    LearningRateStage,
    OptimizationConvergenceError,
    Phase2TrainingResult,
    Phase2TrainingSettings,
    compile_phase2_training_settings,
    train_phase2_heads,
)

ROOT = Path(__file__).resolve().parents[1]


def test_final_pcg_evidence_projects_rich_inner_solver_to_locked_schema() -> None:
    rich_inner_solver = {
        "method": "pcg",
        "dtype": "float64",
        "cold_start": True,
        "warm_start_used": False,
        "iterations": 7,
        "residual_norm": 1.0e-8,
        "relative_residual": 1.0e-8,
        "converged": True,
    }

    assert phase2_training._pcg_evidence(rich_inner_solver) == {
        "iterations": 7,
        "residual_norm": 1.0e-8,
        "relative_residual": 1.0e-8,
        "converged": True,
        "cold_start": True,
    }


def _training() -> TrainingTensorData:
    num_prompts, num_candidates, policy_dimension, reward_dimension = 4, 4, 3, 2
    node = torch.arange(
        num_prompts * num_candidates,
        dtype=torch.float32,
    ).reshape(num_prompts, num_candidates)
    policy_scores = torch.stack(
        [
            torch.sin((coordinate + 1.0) * 0.11 * node) + 0.1 * coordinate
            for coordinate in range(policy_dimension)
        ],
        dim=-1,
    )
    reward_features = torch.stack(
        [
            torch.cos((coordinate + 1.0) * 0.13 * node) - 0.05 * coordinate
            for coordinate in range(reward_dimension)
        ],
        dim=-1,
    )
    return TrainingTensorData(
        prompt_ids=tuple(f"train-{index}" for index in range(num_prompts)),
        policy_scores=policy_scores,
        reward_features=reward_features,
        # These legacy source fields must be ignored by the R=4 primary arm.
        h=torch.linspace(-0.4, 0.3, num_prompts),
        left_wins=torch.tensor([1, 2, 1, 3], dtype=torch.int64),
        num_annotations=torch.tensor([3, 4, 2, 5], dtype=torch.int64),
    )


def _oracle_rewards(training: TrainingTensorData) -> torch.Tensor:
    node = torch.arange(
        training.num_prompts * training.num_candidates,
        dtype=training.policy_scores.dtype,
        device=training.policy_scores.device,
    ).reshape(training.num_prompts, training.num_candidates)
    # All canonical margins are well within +/-log(3), hence p is in [0.25, 0.75].
    return 0.2 * torch.sin(0.3 * node)


@pytest.fixture(scope="module")
def config_bundle():
    return load_phase2_config_bundle(ROOT / "configs" / "common_beta_pilot.yaml")


@pytest.fixture(scope="module")
def compiled_settings(config_bundle) -> Phase2TrainingSettings:
    return compile_phase2_training_settings(config_bundle)


@pytest.fixture(scope="module")
def toy_settings(
    compiled_settings: Phase2TrainingSettings,
) -> Phase2TrainingSettings:
    # The formal projected dimension is 256.  This explicitly non-formal
    # identity is required for the three-dimensional unit-test tangent.
    return replace(
        compiled_settings,
        phase2_config_hash="f" * 64,
        low_dimensional_selected_dimension=2,
        convergence=FirstOrderConvergenceSpec(
            gradient_ratio_tolerance=1.0e6,
            min_steps=20,
            max_steps=720,
            check_interval=20,
            consecutive_checks=2,
        ),
    )


@pytest.fixture(scope="module")
def trained_result(
    toy_settings: Phase2TrainingSettings,
) -> tuple[TrainingTensorData, torch.Tensor, Phase2TrainingResult]:
    training = _training()
    rewards = _oracle_rewards(training)
    frozen_input = {
        "policy_scores": training.policy_scores.clone(),
        "reward_features": training.reward_features.clone(),
        "h": training.h.clone(),
        "left_wins": training.left_wins.clone(),
        "num_annotations": training.num_annotations.clone(),
        "rewards": rewards.clone(),
    }
    result = train_phase2_heads(
        training,
        rewards,
        seed=20260801,
        settings=toy_settings,
    )
    for name, expected in frozen_input.items():
        observed = rewards if name == "rewards" else getattr(training, name)
        assert torch.equal(observed, expected)
    return training, rewards, result


def test_settings_compile_from_overlay_bundle_and_explicit_mapping(config_bundle) -> None:
    from_bundle = compile_phase2_training_settings(config_bundle)
    from_overlay = compile_phase2_training_settings(config_bundle.config)
    from_explicit = compile_phase2_training_settings(
        {
            "config": config_bundle.config,
            "base_config": config_bundle.base_config,
        }
    )

    assert from_bundle == from_overlay == from_explicit
    assert from_bundle.phase2_config_hash == config_bundle.design_identity
    assert from_bundle.source_config_hash == config_bundle.config["design"]["source_config_hash"]
    assert from_bundle.stage == "pilot"
    assert from_bundle.formal_eligibility is False
    assert from_bundle.identifiability_role == "pilot_measure_only"
    assert from_bundle.identifiability_require_full_column_rank is False
    assert from_bundle.outer_steps == 720
    assert from_bundle.num_label_replicates == 4
    assert from_bundle.annotation_gamma == 0.9
    assert from_bundle.relative_damping == 0.001
    assert from_bundle.require_pcg_convergence is True
    assert from_bundle.low_dimensional_enabled is True
    assert from_bundle.low_dimensional_selected_dimension == 256
    assert from_bundle.low_dimensional_regularization == "moore_penrose_pseudoinverse"
    assert from_bundle.exact_soft_label_bt_enabled is True
    assert from_bundle.exact_soft_label_bt_role == EXACT_SOFT_BT_ROLE
    assert from_bundle.exact_soft_label_bt_noise_free is True
    assert from_bundle.exact_soft_label_bt_input == EXACT_SOFT_BT_INPUT
    assert from_bundle.exact_soft_label_bt_eligible_for_primary_claim is False
    assert from_bundle.convergence == FirstOrderConvergenceSpec()
    assert from_bundle.convergence.gradient_ratio_tolerance == 1.0e-3
    assert from_bundle.convergence.min_steps == 100
    assert from_bundle.convergence.max_steps == 5760
    assert from_bundle.convergence.check_interval == 20
    assert from_bundle.convergence.consecutive_checks == 3
    assert len(from_bundle.sha256) == 64


def test_budgeted_training_stage_changes_provenance_not_numerical_algorithm(
    compiled_settings: Phase2TrainingSettings,
) -> None:
    budgeted = replace(
        compiled_settings,
        phase2_config_hash="e" * 64,
        stage=PHASE2_BUDGETED_END_TO_END_STAGE,
        formal_eligibility=False,
        seeds=PHASE2_BUDGETED_END_TO_END_SEEDS,
        identifiability_role=("budgeted_end_to_end_exploratory_frozen_identifiability_audit"),
    )

    assert budgeted.stage == PHASE2_BUDGETED_END_TO_END_STAGE
    assert budgeted.formal_eligibility is False
    assert budgeted.seeds == tuple(range(20261001, 20261006))
    assert budgeted.outer_steps == compiled_settings.outer_steps
    assert budgeted.learning_rate == compiled_settings.learning_rate
    assert budgeted.optimizer == compiled_settings.optimizer
    assert budgeted.weight_decay == compiled_settings.weight_decay
    assert budgeted.microbatch_size == compiled_settings.microbatch_size
    assert budgeted.max_grad_norm == compiled_settings.max_grad_norm
    assert budgeted.training_beta == compiled_settings.training_beta
    assert budgeted.relative_damping == compiled_settings.relative_damping
    assert budgeted.pcg_dtype == compiled_settings.pcg_dtype
    assert budgeted.pcg_max_iterations == compiled_settings.pcg_max_iterations
    assert budgeted.pcg_tolerance == compiled_settings.pcg_tolerance
    assert budgeted.convergence == compiled_settings.convergence
    assert budgeted.num_label_replicates == compiled_settings.num_label_replicates
    assert budgeted.annotation_gamma == compiled_settings.annotation_gamma

    with pytest.raises(ValueError, match="exact ordered seed list"):
        replace(budgeted, seeds=tuple(reversed(PHASE2_BUDGETED_END_TO_END_SEEDS)))
    with pytest.raises(ValueError, match="cannot be formally eligible"):
        replace(budgeted, formal_eligibility=True)
    with pytest.raises(ValueError, match="independent exploratory"):
        replace(budgeted, identifiability_role="confirmatory_frozen_identifiability_contract")
    with pytest.raises(ValueError, match="exact preregistered"):
        replace(
            compiled_settings,
            stage="confirmatory",
            formal_eligibility=True,
            seeds=PHASE2_CONFIRMATORY_SEEDS[:-1],
            identifiability_role="confirmatory_frozen_identifiability_contract",
        )


def test_recovery_settings_compile_hash_bound_schedule_for_every_controller() -> None:
    bundle = load_phase2_config_bundle(ROOT / PHASE2_RECOVERY_PILOT_CONFIG)
    settings = compile_phase2_training_settings(bundle)
    protocol = settings.convergence.optimizer_protocol

    assert protocol is not None
    assert settings.to_dict()["schema_version"] == "phase2-training-settings/v3"
    assert settings.convergence.max_steps == 12760
    assert protocol.schedule_sha256 == PHASE2_RECOVERY_LR_SCHEDULE_SHA256
    assert protocol.legacy_boundary_snapshot_steps == 5760
    assert [
        protocol.learning_rate_for_update(update)
        for update in (1, 5760, 5761, 6760, 6761, 8760, 8761, 10760, 10761, 12760)
    ] == [
        1.0e-3,
        1.0e-3,
        3.0e-4,
        3.0e-4,
        1.0e-4,
        1.0e-4,
        3.0e-5,
        3.0e-5,
        1.0e-5,
        1.0e-5,
    ]


def test_post_recovery_calibration_compiles_the_identical_audited_schedule() -> None:
    recovery_bundle = load_phase2_config_bundle(ROOT / PHASE2_RECOVERY_PILOT_CONFIG)
    post_recovery = copy.deepcopy(recovery_bundle.config)
    post_recovery["schema_version"] = PHASE2_POST_RECOVERY_CALIBRATION_SCHEMA_VERSION
    post_recovery["design"]["name"] = "common-beta-post-recovery-calibration-v1"
    del post_recovery["recovery_control"]
    declared_protocol = post_recovery["reward_model"]["optimizer_protocol"]
    declared_protocol["schema_version"] = "deterministic-adamw-lr-decay/v1"
    del declared_protocol["one_time_recovery"]
    declared_protocol["role"] = "frozen_post_recovery_phase2_optimizer"
    declared_protocol["source_recovery_authorization_sha256"] = "a" * 64
    post_recovery["recovery_success_reference"] = {
        "schema_version": "prorm-phase2-recovery-success-reference/v1",
        "artifact_sha256": "a" * 64,
        "authorization_projection": {
            "schema_version": "prorm-phase2-recovery-success-projection/v1",
            "source_schema_version": ("prorm-phase2-recovery-success-authorization/v1"),
            "recovery_design_sha256": (
                "9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4"
            ),
            "optimizer_schedule_sha256": PHASE2_RECOVERY_LR_SCHEDULE_SHA256,
            "source_array_job_id": "1648125",
            "execution_revision": 2,
            "ordered_seeds": [20260801, 20260802, 20260803],
            "recovery_status": "all_three_seeds_success",
            "full_calibration_authorized": True,
            "authorized_information": "optimizer_schedule_only",
            "recovery_outputs_reusable": False,
            "validation_or_heldout_access_authorized": False,
            "policy_or_final_utility_access_authorized": False,
        },
    }

    settings = compile_phase2_training_settings(
        {
            "config": post_recovery,
            "base_config": recovery_bundle.base_config,
        }
    )
    protocol = settings.convergence.optimizer_protocol

    assert protocol is not None
    assert settings.to_dict()["schema_version"] == "phase2-training-settings/v3"
    assert settings.convergence.max_steps == 12760
    assert protocol.schedule_sha256 == PHASE2_RECOVERY_LR_SCHEDULE_SHA256
    assert protocol.mode == "adopted"
    assert protocol.source_recovery_authorization_sha256 == "a" * 64
    serialized = protocol.to_dict()
    assert serialized["schema_version"] == "deterministic-adamw-lr-decay/v1"
    assert serialized["role"] == "frozen_post_recovery_phase2_optimizer"
    assert serialized["source_recovery_authorization_sha256"] == "a" * 64
    assert "one_time_recovery" not in serialized

    recovery_protocol = compile_phase2_training_settings(
        recovery_bundle
    ).convergence.optimizer_protocol
    assert recovery_protocol is not None
    assert recovery_protocol.mode == "recovery"
    assert recovery_protocol.source_recovery_authorization_sha256 is None
    assert recovery_protocol.to_dict()["schema_version"] == (
        "deterministic-adamw-lr-decay-recovery/v1"
    )
    assert recovery_protocol.to_dict()["one_time_recovery"] is True
    assert [
        protocol.learning_rate_for_update(update)
        for update in (1, 5760, 5761, 6760, 6761, 8760, 8761, 10760, 10761, 12760)
    ] == [
        recovery_protocol.learning_rate_for_update(update)
        for update in (1, 5760, 5761, 6760, 6761, 8760, 8761, 10760, 10761, 12760)
    ]


def test_recovery_identifiability_is_tie_break_bound_and_separation_aware() -> None:
    settings = compile_phase2_training_settings(
        load_phase2_config_bundle(ROOT / PHASE2_RECOVERY_PILOT_CONFIG)
    )
    protocol = settings.convergence.optimizer_protocol
    assert protocol is not None
    training = _training()

    reward_rank = phase2_training._reward_head_identifiability(training, settings)
    assert reward_rank["schema_version"] == "reward-head-identifiability/v2"
    assert reward_rank["algorithmic_tie_break"] == protocol.tie_break
    assert reward_rank["full_design_rank_proves_finite_bt_minimizer_exists"] is False
    mixed = reward_rank["mixed_outcome_edge_coercivity_diagnostic"]
    assert mixed["num_mixed_outcome_edges"] == training.num_prompts
    assert mixed["shape"] == [training.num_prompts, training.reward_dimension]
    assert mixed["full_column_rank"] is True
    assert mixed["sufficient_condition_observed"] is True
    assert mixed["acceptance_gate_applied"] is False
    assert mixed["raw_outcomes_serialized"] is False
    assert reward_rank["bt_unique_finite_minimizer_sufficient_condition_observed"] is True

    all_left_wins = TrainingTensorData(
        prompt_ids=training.prompt_ids,
        policy_scores=training.policy_scores,
        reward_features=training.reward_features,
        h=training.h,
        left_wins=training.num_annotations.clone(),
        num_annotations=training.num_annotations,
    )
    separated_rank = phase2_training._reward_head_identifiability(
        all_left_wins,
        settings,
    )
    assert separated_rank["full_column_rank"] is True
    separated_mixed = separated_rank["mixed_outcome_edge_coercivity_diagnostic"]
    assert separated_mixed["num_mixed_outcome_edges"] == 0
    assert separated_mixed["shape"] == [0, training.reward_dimension]
    assert separated_mixed["numerical_rank"] == 0
    assert separated_mixed["full_column_rank"] is False
    assert separated_mixed["sufficient_condition_observed"] is False
    assert separated_rank["bt_unique_finite_minimizer_sufficient_condition_observed"] is False

    moment_rank = phase2_training._prorm_moment_map_identifiability(
        training,
        settings,
    )
    assert moment_rank["schema_version"] == "prorm-moment-map-identifiability/v2"
    assert moment_rank["algorithmic_tie_break"] == protocol.tie_break


def test_settings_reject_nonformal_or_unbound_values(
    compiled_settings: Phase2TrainingSettings,
    config_bundle,
) -> None:
    with pytest.raises(ValueError, match="exactly 720"):
        replace(compiled_settings, outer_steps=719)
    with pytest.raises(ValueError, match="must reach"):
        replace(
            compiled_settings,
            convergence=FirstOrderConvergenceSpec(max_steps=700),
        )
    with pytest.raises(ValueError, match="exactly four"):
        replace(compiled_settings, num_label_replicates=1)
    with pytest.raises(ValueError, match="gamma=0.9"):
        replace(compiled_settings, annotation_gamma=0.95)
    with pytest.raises(ValueError, match="fail on PCG"):
        replace(compiled_settings, require_pcg_convergence=False)
    with pytest.raises(ValueError, match="must be enabled"):
        replace(compiled_settings, low_dimensional_enabled=False)
    with pytest.raises(ValueError, match="positive integer"):
        replace(compiled_settings, low_dimensional_selected_dimension=0)
    with pytest.raises(ValueError, match="Moore-Penrose"):
        replace(compiled_settings, low_dimensional_regularization="ridge")
    with pytest.raises(ValueError, match="must be enabled"):
        replace(compiled_settings, exact_soft_label_bt_enabled=False)
    with pytest.raises(ValueError, match="noise-free"):
        replace(compiled_settings, exact_soft_label_bt_noise_free=False)
    with pytest.raises(ValueError, match="cannot be eligible"):
        replace(compiled_settings, exact_soft_label_bt_eligible_for_primary_claim=True)
    with pytest.raises(ValueError, match="exactly config and base_config"):
        compile_phase2_training_settings(
            {
                "config": config_bundle.config,
                "base_config": config_bundle.base_config,
                "typo": True,
            }
        )
    with pytest.raises(ValueError, match="base_config semantic identity"):
        compile_phase2_training_settings(
            {
                "config": config_bundle.config,
                "base_config": load_config(ROOT / "configs" / "main.yaml"),
            }
        )


def test_primary_training_is_fresh_r4_and_runner_compatible(trained_result) -> None:
    training, _, result = trained_result

    assert result.training_design_sha256 == result.settings.phase2_config_hash
    assert result.training_settings_sha256 == result.settings.sha256
    assert result.training_arm == PRIMARY_TRAINING_ARM
    assert set(result.heads) == {BT_MLE, PRORM_PLUS}
    assert result.bt_head == result.bt_mle.head_weight
    assert result.prorm_plus_head == result.prorm_plus.head_weight
    assert len(result.bt_head) == len(result.prorm_plus_head) == training.reward_dimension
    assert all(math.isfinite(value) for head in result.heads.values() for value in head)
    assert any(value != 0.0 for value in result.bt_head)
    assert any(value != 0.0 for value in result.prorm_plus_head)
    assert result.bt_mle.initial_head_sha256 == result.prorm_plus.initial_head_sha256
    assert result.bt_mle.arm == result.prorm_plus.arm == PRIMARY_TRAINING_ARM
    assert result.old_phase1_comparison_heads_used is False
    assert result.test_data_accessed is False
    assert result.raw_node_rewards_retained is False

    adapted = result.to_runner_head_result()
    assert isinstance(adapted, Phase2HeadTrainingResult)
    assert adapted.heads == result.heads
    assert adapted.training_design_sha256 == result.training_design_sha256
    assert adapted.training_arm == PRIMARY_TRAINING_ARM
    assert adapted.test_data_accessed is False


def test_named_label_stream_records_rng_hash_routing_and_cost(trained_result) -> None:
    training, _, result = trained_result
    evidence = result.label_stream

    assert evidence.namespace == LABEL_RNG_NAMESPACE
    assert evidence.base_seed == 20260801
    assert evidence.derived_seed != evidence.base_seed
    assert evidence.initial_state_sha256 != evidence.final_state_sha256
    assert evidence.num_edges == training.num_prompts
    assert evidence.num_replicates == 4
    assert evidence.gamma == 0.9
    assert evidence.realized_total_annotations >= training.num_prompts * 4
    assert evidence.realized_annotations_per_edge == pytest.approx(
        evidence.realized_total_annotations / training.num_prompts
    )
    assert evidence.expected_annotations_per_edge == 40.0
    assert evidence.bt_target == "pooled_raw_wins_and_totals"
    assert evidence.prorm_target == "mean_of_per_replicate_h"
    assert evidence.raw_labels_retained is False
    assert evidence.raw_node_rewards_retained is False
    tail = evidence.repeated_label_tail_diagnostics
    assert tail["schema_version"] == "repeated-label-tail-diagnostics/v1"
    assert tail["split"] == "train"
    assert tail["gamma"] == 0.9
    assert tail["num_replicates"] == 4
    assert tail["scalar_only"] is True
    assert tail["descriptive_only"] is True
    assert tail["used_for_clipping"] is False
    assert tail["used_for_selection"] is False
    assert tail["used_for_gating"] is False
    assert tail["source_tensor_sha256"] == {
        "replicate_count_sha256": evidence.replicate_count_sha256,
        "replicate_h_sha256": evidence.replicate_h_sha256,
        "mean_h_sha256": evidence.mean_h_sha256,
    }
    assert tail["metrics"]["replicate_count"]["sample_size"] == 4 * training.num_prompts
    assert tail["metrics"]["abs_replicate_h"]["sample_size"] == 4 * training.num_prompts
    assert tail["metrics"]["abs_mean_h"]["sample_size"] == training.num_prompts
    assert len(tail["diagnostics_sha256"]) == 64
    label_payload = {
        "namespace": evidence.namespace,
        "base_seed": evidence.base_seed,
        "derived_seed": evidence.derived_seed,
        "derivation_sha256": evidence.derivation_sha256,
        "initial_state_sha256": evidence.initial_state_sha256,
        "final_state_sha256": evidence.final_state_sha256,
        "probability_sha256": evidence.canonical_probability_sha256,
        "replicate_count_sha256": evidence.replicate_count_sha256,
        "replicate_win_sha256": evidence.replicate_win_sha256,
        "replicate_h_sha256": evidence.replicate_h_sha256,
        "mean_h_sha256": evidence.mean_h_sha256,
        "repeated_label_tail_diagnostics_sha256": tail["diagnostics_sha256"],
        "realized_total_annotations": evidence.realized_total_annotations,
    }
    assert phase2_training._canonical_sha256(label_payload) == evidence.label_stream_sha256
    for name in (
        "derivation_sha256",
        "oracle_reward_sha256",
        "canonical_probability_sha256",
        "replicate_count_sha256",
        "replicate_win_sha256",
        "replicate_h_sha256",
        "mean_h_sha256",
        "label_stream_sha256",
    ):
        assert len(getattr(evidence, name)) == 64


def test_label_stream_rejects_tail_diagnostic_semantic_tampering(trained_result) -> None:
    _, _, result = trained_result
    tampered = copy.deepcopy(result.label_stream.repeated_label_tail_diagnostics)
    tampered["used_for_selection"] = True

    with pytest.raises(ValueError, match="used_for_selection"):
        replace(result.label_stream, repeated_label_tail_diagnostics=tampered)


def test_objective_specific_convergence_and_fixed_snapshot_are_bound(
    trained_result,
) -> None:
    training, _, result = trained_result
    for head in (
        result.bt_mle,
        result.prorm_plus,
        result.exact_margin_control.head,
        result.exact_soft_label_bt_control.head,
    ):
        history = head.history_summary
        assert history["num_steps"] == 40
        assert history["history_objective_timing"] == "pre_update"
        assert history["stored_checkpoint_steps"] == [1, 10, 20, 30, 40]
        assert len(history["checkpoints"]) == 5
        assert math.isfinite(head.initial_objective)
        assert math.isfinite(head.final_objective)
        convergence = head.first_order_convergence
        assert convergence["converged"] is True
        assert convergence["fail_closed"] is True
        assert convergence["selected_primary_step"] == 40
        assert convergence["test_or_validation_data_accessed"] is False
        assert convergence["fixed_step_snapshot_is_not_primary_selection"] is True
        fixed = convergence["fixed_step_compute_matched_snapshot"]
        assert fixed["step"] == 720
        assert fixed["history_summary"]["num_steps"] == 720
        assert fixed["role"] == "compute_matched_and_pilot_diagnostic_only"
        assert fixed["used_as_primary_selection_rule"] is False
        assert fixed["coincides_with_selected_primary_iterate"] is False
        identification = convergence["solution_identification"]
        assert identification["initialization"] == "exact_zero_head"
        assert identification["tie_break"] == "zero_initialized_adamw_implicit_bias"
        assert identification["minimum_norm_solution_claimed"] is False
        assert identification["unique_reward_head_solution_claimed"] is False
        rank = identification["optional_objective_rank_diagnostic"]
        assert rank["evaluated"] is True
        assert rank["evidence"]["role"] == "pilot_measure_only"
        assert rank["evidence"]["minimum_norm_solution_claimed"] is False

        initial_gradient = convergence["initial_zero_head_measurement"]["gradient_l2_norm"]
        final_gate = convergence["final_gate"]
        denominator = max(
            initial_gradient,
            convergence["spec"]["gradient_norm_denominator_floor"],
        )
        assert final_gate["gradient_ratio_to_zero_initialization"] == pytest.approx(
            final_gate["measurement"]["gradient_l2_norm"] / denominator
        )
        assert (
            final_gate["gradient_ratio_to_zero_initialization"]
            <= (convergence["spec"]["gradient_ratio_tolerance"])
        )
        assert final_gate["fresh_post_restore_audit"] is True
    assert result.bt_mle.final_pcg is None
    assert result.exact_soft_label_bt_control.head.final_pcg is None
    assert result.prorm_plus.final_pcg["converged"] is True
    assert result.prorm_plus.final_pcg["cold_start"] is True
    assert set(result.prorm_plus.final_pcg) == {
        "iterations",
        "residual_norm",
        "relative_residual",
        "converged",
        "cold_start",
    }
    assert result.exact_margin_control.head.final_pcg["converged"] is True

    primary = result.primary_optimization_audit
    rank = primary["reward_head_identifiability"]
    assert rank["design_matrix"] == "canonical_edge_reward_feature_differences"
    assert rank["shape"] == [4, 2]
    assert rank["numerical_rank"] <= rank["column_dimension"] == 2
    assert rank["acceptance_gate_passed"] is True
    assert rank["prorm_moment_map_full_rank_proved"] is False
    moment_map = primary["prorm_moment_map_identifiability"]
    assert moment_map["schema_version"] == "prorm-moment-map-identifiability/v1"
    assert moment_map["formula"] == "J_m = Z^T D / (2 n_edges)"
    assert moment_map["shape"] == [3, 2]
    assert moment_map["num_edges"] == 4
    assert moment_map["audit_dtype"] == "torch.float64"
    assert moment_map["computation"]["algorithm"] == "deterministic_blocked_fp64_tsqr"
    assert moment_map["computation"]["randomized_rank_approximation_used"] is False
    assert moment_map["unique_ridge_prorm_quadratic_head_iff_full_column_rank"] is True
    assert moment_map["population_identifiability_theorem_claimed"] is False
    assert moment_map["minimum_norm_solution_claimed"] is False
    assert moment_map["role"] == "pilot_measure_only"
    assert moment_map["acceptance_gate_passed"] is True
    assert (
        moment_map["observed_unique_ridge_prorm_quadratic_head"] is moment_map["full_column_rank"]
    )
    spectrum = moment_map["singular_values_descending"]
    assert spectrum == sorted(spectrum, reverse=True)
    assert len(spectrum) == 2
    expected_j = (
        (training.policy_scores[:, 0] - training.policy_scores[:, 1]).to(torch.float64).mT
        @ (training.reward_features[:, 0] - training.reward_features[:, 1]).to(torch.float64)
    ) * (1.0 / (2.0 * training.num_prompts))
    assert moment_map["moment_map_sha256"] == phase2_training._tensor_sha256(expected_j)
    assert spectrum == pytest.approx(
        torch.linalg.svdvals(expected_j).tolist(),
        rel=1.0e-12,
        abs=1.0e-14,
    )
    d_rank_sha = phase2_training._canonical_sha256(rank)
    moment_rank_sha = phase2_training._canonical_sha256(moment_map)
    assert (
        result.bt_mle.first_order_convergence["solution_identification"][
            "optional_objective_rank_diagnostic"
        ]["evidence"]
        == rank
    )
    assert (
        result.prorm_plus.first_order_convergence["solution_identification"][
            "optional_objective_rank_diagnostic"
        ]["evidence"]
        == moment_map
    )
    assert (
        result.exact_margin_control.head.first_order_convergence["solution_identification"][
            "optional_objective_rank_diagnostic"
        ]["evidence"]
        == moment_map
    )
    assert (
        result.exact_soft_label_bt_control.head.first_order_convergence["solution_identification"][
            "optional_objective_rank_diagnostic"
        ]["evidence"]
        == rank
    )
    assert d_rank_sha != moment_rank_sha
    assert primary["optimizer_constructed"] is False
    assert primary["optimizer_step_called"] is False
    assert primary["saved_heads_mutated"] is False
    assert primary["learners"][BT_MLE]["gradient"] == "full_data_unclipped"
    assert primary["learners"][PRORM_PLUS]["gradient_definition"] == "fresh_dual_envelope_gradient"
    assert primary["learners"][PRORM_PLUS]["fresh_inner_pcg"]["warm_start_used"] is False
    assert primary["learners"][PRORM_PLUS]["fresh_inner_pcg"]["converged"] is True
    assert primary["learners"][BT_MLE]["objective"] == pytest.approx(
        result.bt_mle.final_objective,
        rel=2.0e-5,
    )
    assert primary["learners"][PRORM_PLUS]["objective"] == pytest.approx(
        result.prorm_plus.final_objective,
        rel=2.0e-5,
    )


def test_exact_trained_head_gap_is_separate_from_direct_identity(trained_result) -> None:
    _, _, result = trained_result
    exact = result.exact_margin_control
    direct = result.direct_oracle_identity

    assert exact.target_audit["raw_node_rewards_retained"] is False
    assert exact.target_audit["orientation"] == "candidate_0_minus_candidate_1"
    assert exact.optimization_audit["bt_audit_discarded"] is True
    assert exact.optimization_audit["learner"]["gradient"] == "full_data_unclipped"
    gap = exact.reward_class_and_optimizer_gap
    assert "restricted_reward_class_and_finite_optimizer_gap" in gap["interpretation"]
    assert gap["algebraic_identity_claimed"] is False
    assert gap["finite_optimizer_steps"] == 40
    assert gap["all_node_moment_error_l2"] >= 0.0
    assert gap["direction_error_l2"] >= 0.0
    assert gap["squared_fisher_direction_error"] >= 0.0
    assert gap["trained_direction_pcg"]["converged"] is True

    assert direct["interpretation"] == ("algebraic_identity_bypasses_reward_class_and_optimizer")
    assert direct["complete_pair_identity_is_algebraic"] is True
    assert direct["reward_head_bypassed"] is True
    assert direct["optimizer_bypassed"] is True
    assert direct["trained_exact_margin_head_required_to_match"] is False
    assert direct["raw_node_rewards_retained"] is False
    assert direct["complete_pair_identity_absolute_error"] < 1.0e-12
    assert direct["native_oracle_direction"]["pcg"]["converged"] is True
    assert "direction" not in direct["native_oracle_direction"]
    assert "complete_pair_u_stat_moment" not in direct
    assert "all_node_covariance_moment" not in direct


def test_exact_soft_label_bt_uses_exact_train_probability_and_is_diagnostic_only(
    trained_result,
) -> None:
    training, rewards, result = trained_result
    exact_arm = build_exact_margin_canonical_arm(training, rewards)
    transient = phase2_training._ExactSoftLabelBTBatch.from_exact_margin_training(
        exact_arm.training
    )
    expected_probability = torch.sigmoid(rewards[:, 0] - rewards[:, 1])
    expected_features = training.reward_features[:, 0] - training.reward_features[:, 1]
    assert torch.equal(transient.target_probabilities, expected_probability)
    assert torch.equal(transient.feature_differences, expected_features)

    control = result.exact_soft_label_bt_control
    head = control.head
    target = control.target_audit
    optimization = control.optimization_audit
    assert head.arm == EXACT_SOFT_BT_ARM
    assert head.method == BT_MLE
    assert head.final_pcg is None
    assert head.first_order_convergence["converged"] is True
    assert head.first_order_convergence["fixed_step_compute_matched_snapshot"]["step"] == 720
    assert target == {
        "schema_version": "exact-soft-label-bt-target/v1",
        "split": "train",
        "orientation": "candidate_0_minus_candidate_1",
        "input": EXACT_SOFT_BT_INPUT,
        "target_construction": "p_star = sigmoid(delta_r_star)",
        "source_node_rewards_sha256": result.label_stream.oracle_reward_sha256,
        "canonical_margin_sha256": phase2_training._tensor_sha256(rewards[:, 0] - rewards[:, 1]),
        "target_probability_sha256": phase2_training._tensor_sha256(expected_probability),
        "reward_feature_difference_sha256": phase2_training._tensor_sha256(expected_features),
        "num_canonical_edges": training.num_prompts,
        "reward_dimension": training.reward_dimension,
        "same_reward_features_and_canonical_edges_as": "exact_margin_prorm_plus",
        "noise_free": True,
        "bernoulli_sampling_used": False,
        "sampled_label_stream_accessed": False,
        "raw_target_probabilities_retained": False,
        "raw_oracle_margins_retained": False,
        "raw_node_rewards_retained": False,
        "test_or_validation_data_accessed": False,
        "role": EXACT_SOFT_BT_ROLE,
        "eligible_for_primary_claim": False,
    }
    assert optimization["objective"] == ("mean(softplus(delta_r_phi) - p_star * delta_r_phi)")
    assert optimization["head_sha256"] == head.head_sha256
    assert optimization["first_order_convergence_passed"] is True
    assert optimization["fixed_720_step_checkpoint_used_for_head_selection"] is False
    assert optimization["favorable_ordering_gate_applied"] is False
    assert optimization["pilot_measure_only"] is True
    assert optimization["eligible_for_primary_claim"] is False
    assert optimization["sampled_label_stream_accessed"] is False
    assert optimization["test_or_validation_data_accessed"] is False


def test_exact_soft_label_bt_is_invariant_to_distinct_sampled_label_streams(
    toy_settings: Phase2TrainingSettings,
) -> None:
    training = _training()
    rewards = _oracle_rewards(training)
    probabilities = torch.sigmoid(rewards[:, 0] - rewards[:, 1])
    first_noisy = sample_canonical_r4_noisy_arm(
        training,
        probabilities,
        generator=torch.Generator().manual_seed(101),
    )
    second_noisy = sample_canonical_r4_noisy_arm(
        training,
        probabilities,
        generator=torch.Generator().manual_seed(202),
    )
    assert not torch.equal(
        first_noisy.repeated_labels.wins,
        second_noisy.repeated_labels.wins,
    )

    exact_arm = build_exact_margin_canonical_arm(training, rewards)
    rank = phase2_training._reward_head_identifiability(
        exact_arm.training,
        toy_settings,
    )
    signature = inspect.signature(phase2_training._train_exact_soft_label_bt_control)
    assert set(signature.parameters) == {
        "exact_margin_training",
        "source_node_rewards_sha256",
        "exact_margin_sha256",
        "settings",
        "rank_diagnostic",
    }
    assert not {"labels", "generator", "seed", "validation", "test"} & set(signature.parameters)
    arguments = {
        "source_node_rewards_sha256": exact_arm.audit.source_node_rewards_sha256,
        "exact_margin_sha256": exact_arm.audit.exact_margin_sha256,
        "settings": toy_settings,
        "rank_diagnostic": rank,
    }
    first_control = phase2_training._train_exact_soft_label_bt_control(
        exact_arm.training,
        **arguments,
    )
    second_control = phase2_training._train_exact_soft_label_bt_control(
        exact_arm.training,
        **arguments,
    )
    assert first_control == second_control
    assert first_control.target_audit["sampled_label_stream_accessed"] is False


def test_prorm_moment_map_rank_audit_is_deterministic_across_multiple_blocks(
    toy_settings: Phase2TrainingSettings,
) -> None:
    source = _training()
    coordinates = torch.arange(4097, dtype=torch.float32)
    node = torch.arange(
        source.num_prompts * source.num_candidates,
        dtype=torch.float32,
    ).reshape(source.num_prompts, source.num_candidates, 1)
    policy_scores = torch.sin(node * 0.03 * (coordinates + 1.0)) + 0.001 * coordinates
    training = TrainingTensorData(
        prompt_ids=source.prompt_ids,
        policy_scores=policy_scores,
        reward_features=source.reward_features,
        h=source.h,
        left_wins=source.left_wins,
        num_annotations=source.num_annotations,
    )

    first = phase2_training._prorm_moment_map_identifiability(training, toy_settings)
    second = phase2_training._prorm_moment_map_identifiability(training, toy_settings)
    assert first == second
    assert first["computation"]["row_block_size"] == 4096
    assert first["computation"]["num_row_blocks"] == 2
    edge_scores = (policy_scores[:, 0] - policy_scores[:, 1]).to(torch.float64)
    edge_features = (source.reward_features[:, 0] - source.reward_features[:, 1]).to(torch.float64)
    expected = (edge_scores.mT @ edge_features) * (1.0 / (2.0 * source.num_prompts))
    assert first["moment_map_sha256"] == phase2_training._tensor_sha256(expected)
    assert first["singular_values_descending"] == pytest.approx(
        torch.linalg.svdvals(expected).tolist(),
        rel=1.0e-11,
        abs=1.0e-13,
    )


def test_low_dimensional_positive_control_is_fresh_ridge_free_and_deployable(
    trained_result,
) -> None:
    _, _, result = trained_result
    control = result.low_dimensional_control

    assert control["enabled"] is True
    assert control["eligible_for_primary_claim"] is False
    assert "positive_control_only" in control["interpretation"]
    assert control["training_arm"] == PRIMARY_TRAINING_ARM
    assert control["label_stream_sha256"] == result.label_stream.label_stream_sha256
    assert control["target"] == "same_r4_mean_h_as_primary_prorm_plus"
    assert control["bt_head"] == {
        "head_sha256": result.bt_mle.head_sha256,
        "retrained": False,
        "reason": "bt_objective_is_independent_of_policy_tangent_geometry",
    }

    projection = control["projection"]
    assert projection["source_dimension"] == 3
    assert projection["selected_dimension"] == 2
    assert projection["num_fisher_nodes"] == 16
    assert projection["orthonormal_columns"] is True
    assert projection["strictly_below_fisher_node_count"] is True
    assert projection["score_construction"] == ("S_low = cast_fp32(cast_fp64(S_full) @ P_fp64)")
    assert projection["projection_dtype"] == "torch.float64"
    assert projection["orthonormality_max_absolute_error"] <= 1.0e-10
    assert len(projection["projection_sha256"]) == 64

    geometry = control["geometry"]
    assert geometry["regularization"] == "moore_penrose_pseudoinverse"
    assert geometry["ridge_enabled"] is False
    assert geometry["ridge_coefficient"] == 0.0
    assert geometry["selected_dimension"] == 2
    assert 1 <= geometry["numerical_rank"] <= 2
    assert geometry["relative_eigenvalue_tolerance"] == 1.0e-10
    assert geometry["pcg_used"] is False
    assert geometry["smallest_retained_eigenvalue"] > 0.0

    head = control["head"]
    assert head["arm"] == "low_dimensional_tangent_positive_control"
    assert head["method"] == PRORM_PLUS
    assert head["initial_head_sha256"] == result.prorm_plus.initial_head_sha256
    assert head["history_summary"]["num_steps"] == 40
    assert head["history_summary"]["pcg"] is None
    convergence = head["first_order_convergence"]
    assert convergence["converged"] is True
    assert convergence["selected_primary_step"] == 40
    assert convergence["fixed_step_compute_matched_snapshot"]["step"] == 720
    assert convergence["fixed_step_compute_matched_snapshot"]["history_summary"]["num_steps"] == 720


def test_real_low_dimensional_training_path_emits_aggregate_valid_projected_rank(
    trained_result,
    config_bundle,
) -> None:
    training, _, result = trained_result
    control = result.low_dimensional_control
    evidence = control["projected_prorm_moment_map_identifiability"]
    projection = control["projection"]
    geometry = control["geometry"]
    head = control["head"]
    convergence_rank = control["head"]["first_order_convergence"]["solution_identification"][
        "optional_objective_rank_diagnostic"
    ]

    assert convergence_rank["evaluated"] is True
    assert convergence_rank["evidence"] == evidence
    assert evidence["schema_version"] == ("projected-prorm-moment-map-identifiability/v1")
    assert evidence["design_matrix"] == ("canonical_train_edge_projected_moment_jacobian")
    assert evidence["shape"] == [2, training.reward_dimension]
    assert evidence["projection_sha256"] == projection["projection_sha256"]
    assert evidence["projected_geometry"]["fisher_sha256"] == geometry["fisher_sha256"]
    assert (
        evidence["projected_geometry"]["pseudoinverse_sha256"] == (geometry["pseudoinverse_sha256"])
    )
    assert evidence["unique_projected_prorm_quadratic_head_iff_full_column_rank"] is True
    assert evidence["require_full_column_rank"] is False
    assert evidence["full_row_rank"] is True
    assert evidence["require_full_row_rank"] is True
    assert evidence["acceptance_gate_definition"] == (
        "full_row_rank_for_projected_policy_moment_coverage"
    )
    assert evidence["population_identifiability_theorem_claimed"] is False
    assert evidence["minimum_norm_solution_claimed"] is False

    normalized, evidence_sha = phase2_aggregate._validate_prorm_moment_map_identifiability(
        evidence,
        config=config_bundle.config["reward_model"]["identifiability"],
        expected_train_prompts=training.num_prompts,
        expected_policy_dimension=2,
        expected_reward_dimension=training.reward_dimension,
        expected_projection_sha256=projection["projection_sha256"],
        expected_projected_geometry={
            "fisher_sha256": geometry["fisher_sha256"],
            "pseudoinverse_sha256": geometry["pseudoinverse_sha256"],
            "relative_eigenvalue_tolerance": geometry["relative_eigenvalue_tolerance"],
        },
        name="real_low_dimensional_training_path",
    )
    assert normalized == evidence
    assert evidence_sha == phase2_training._canonical_sha256(evidence)
    assert all(math.isfinite(value) for value in head["head_weight"])

    audit = control["final_full_data_audit"]
    assert audit["optimizer_constructed"] is False
    assert audit["optimizer_step_called"] is False
    assert audit["saved_head_mutated"] is False
    assert audit["gradient"] == "full_data_unclipped"
    assert audit["objective"] == pytest.approx(head["final_objective"])
    assert audit["gradient_l2_norm"] >= 0.0

    identity = control["deployment_score_identity"]
    assert identity["passed"] is True
    assert identity["max_absolute_error"] <= identity["absolute_tolerance"]
    assert identity["l2_error"] >= 0.0
    for name in (
        "selected_direction_sha256",
        "scattered_full_direction_sha256",
        "low_projected_score_sha256",
        "full_projected_score_sha256",
    ):
        assert len(identity[name]) == 64


def test_formal_low_dimension_fails_closed_when_toy_tangent_cannot_execute(
    compiled_settings: Phase2TrainingSettings,
) -> None:
    training = _training()
    with pytest.raises(
        ValueError,
        match="configured low-dimensional control cannot execute",
    ):
        train_phase2_heads(
            training,
            _oracle_rewards(training),
            seed=20260801,
            settings=compiled_settings,
        )


def test_result_is_strict_json_and_contains_no_raw_reward_or_label_values(
    trained_result,
) -> None:
    _, _, result = trained_result
    payload = result.to_dict()
    rendered = json.dumps(payload, allow_nan=False, sort_keys=True)

    assert payload["schema_version"] == "phase2-fresh-head-training/v2"
    assert payload["audit"]["schema_version"] == "phase2-fresh-head-training/v2"
    for head in (result.bt_mle, result.prorm_plus):
        convergence = head.first_order_convergence
        measurements = [
            convergence["initial_zero_head_measurement"],
            convergence["final_gate"]["measurement"],
            convergence["fixed_step_compute_matched_snapshot"]["measurement"],
            *(check["measurement"] for check in convergence["checks"]),
        ]
        assert all("audit_dtype" not in measurement for measurement in measurements)
    assert payload["heads"] == {
        BT_MLE: list(result.bt_head),
        PRORM_PLUS: list(result.prorm_plus_head),
    }
    forbidden_keys = {
        "train_oracle_rewards",
        "raw_labels",
        "canonical_margins",
        "direction",
        "node_rewards",
        "replicate_h",
        "counts",
        "wins",
        "target_probabilities",
        "sampled_labels",
    }

    def assert_no_forbidden_keys(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for nested in value.values():
                assert_no_forbidden_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_no_forbidden_keys(nested)

    assert_no_forbidden_keys(payload)
    assert "train_oracle_rewards" not in rendered
    assert result.audit["isolation"]["raw_node_rewards_retained"] is False
    assert result.audit["isolation"]["raw_labels_retained"] is False

    with pytest.raises(FrozenInstanceError):
        result.training_arm = "phase1"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.bt_mle.head_weight = (0.0,)  # type: ignore[misc]


def test_recovery_result_uses_fresh_head_training_v3_only(
    trained_result,
) -> None:
    _, _, legacy_result = trained_result
    recovery_settings = compile_phase2_training_settings(
        load_phase2_config_bundle(ROOT / PHASE2_RECOVERY_PILOT_CONFIG)
    )
    recovery_result = replace(
        legacy_result,
        settings=recovery_settings,
        training_design_sha256=recovery_settings.phase2_config_hash,
        training_settings_sha256=recovery_settings.sha256,
    )

    assert recovery_result.schema_version == "phase2-fresh-head-training/v3"
    assert recovery_result.to_dict()["schema_version"] == "phase2-fresh-head-training/v3"
    assert recovery_result.audit["schema_version"] == "phase2-fresh-head-training/v3"
    assert legacy_result.schema_version == "phase2-fresh-head-training/v2"


def test_source_phase1_label_fields_cannot_change_primary_or_control_heads(
    trained_result,
    toy_settings: Phase2TrainingSettings,
) -> None:
    training, rewards, first = trained_result
    altered = TrainingTensorData(
        prompt_ids=training.prompt_ids,
        policy_scores=training.policy_scores.clone(),
        reward_features=training.reward_features.clone(),
        h=torch.full_like(training.h, 99.0),
        left_wins=torch.zeros_like(training.left_wins),
        num_annotations=torch.ones_like(training.num_annotations),
    )
    second = train_phase2_heads(
        altered,
        rewards,
        seed=20260801,
        settings=toy_settings,
    )

    assert second.label_stream.label_stream_sha256 == first.label_stream.label_stream_sha256
    assert second.heads == first.heads
    assert (
        second.exact_margin_control.head.head_weight == first.exact_margin_control.head.head_weight
    )
    assert second.exact_soft_label_bt_control == first.exact_soft_label_bt_control
    assert second.low_dimensional_control == first.low_dimensional_control
    assert second.direct_oracle_identity == first.direct_oracle_identity
    # Provenance still detects that a different source object was supplied,
    # even though fields excluded from Phase-2 training cannot change heads.
    assert second.input_training_sha256 != first.input_training_sha256
    assert second.training_instance_sha256 != first.training_instance_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_shape", "shape"),
        ("integer", "floating-point"),
        ("nan", "finite"),
        ("requires_grad", "frozen"),
        ("outside_transform", r"\[0.25, 0.75\]"),
    ],
)
def test_invalid_transient_oracle_rewards_fail_before_training(
    mutation: str,
    message: str,
    toy_settings: Phase2TrainingSettings,
) -> None:
    training = _training()
    rewards = _oracle_rewards(training)
    if mutation == "wrong_shape":
        rewards = rewards[:, :2]
    elif mutation == "integer":
        rewards = rewards.to(torch.int64)
    elif mutation == "nan":
        rewards = rewards.clone()
        rewards[0, 0] = float("nan")
    elif mutation == "requires_grad":
        rewards = rewards.clone().requires_grad_(True)
    elif mutation == "outside_transform":
        rewards = rewards.clone()
        rewards[0, 0] = 3.0
        rewards[0, 1] = -3.0

    with pytest.raises((TypeError, ValueError), match=message):
        train_phase2_heads(
            training,
            rewards,
            seed=20260801,
            settings=toy_settings,
        )


def test_seed_cost_and_pcg_fail_closed(
    toy_settings: Phase2TrainingSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _training()
    rewards = _oracle_rewards(training)
    with pytest.raises(ValueError, match="configured Phase-2"):
        train_phase2_heads(
            training,
            rewards,
            seed=20260722,
            settings=toy_settings,
        )
    with pytest.raises(RuntimeError, match="max_total_annotations"):
        train_phase2_heads(
            training,
            rewards,
            seed=20260801,
            settings=replace(toy_settings, max_total_annotations=1),
        )

    def nonconverged(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(pcg_converged=False)

    monkeypatch.setattr(phase2_training.ProRMPlusTrainer, "evaluate", nonconverged)
    with pytest.raises(RuntimeError, match="fresh cold-start FP64 ProRM\\+ PCG audit"):
        train_phase2_heads(
            training,
            rewards,
            seed=20260801,
            settings=toy_settings,
        )


class _FakeFirstOrderTrainer:
    """Tiny stateful trainer exercising the convergence controller only."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)))
        self.completed_steps = 0
        self.history: list[phase2_training.TrainingStepDiagnostics] = []

    def step(self):
        with torch.no_grad():
            self.model.weight.add_(1.0)
        self.completed_steps += 1
        diagnostic = phase2_training.TrainingStepDiagnostics(
            step=self.completed_steps,
            objective=float(10 - self.completed_steps),
            gradient_norm=1.0,
        )
        self.history.append(diagnostic)
        return diagnostic

    def state_dict(self):
        return {
            "weight": self.model.weight.detach().clone(),
            "completed_steps": self.completed_steps,
            "history": tuple(self.history),
        }

    def load_state_dict(self, state):
        with torch.no_grad():
            self.model.weight.copy_(state["weight"])
        self.completed_steps = state["completed_steps"]
        self.history = list(state["history"])


class _RecoveryFirstOrderTrainer:
    """One-parameter trainer whose recorded LR exposes controller ordering."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(1, dtype=torch.float32)))
        self.optimizer = torch.optim.AdamW(
            [self.model.weight],
            lr=1.0e-3,
            weight_decay=0.0,
        )
        self.completed_steps = 0
        self.history: list[phase2_training.TrainingStepDiagnostics] = []
        self.observed_learning_rates: list[float] = []

    def step(self):
        self.observed_learning_rates.append(float(self.optimizer.param_groups[0]["lr"]))
        self.optimizer.zero_grad(set_to_none=True)
        self.model.weight.grad = torch.ones_like(self.model.weight)
        self.optimizer.step()
        self.completed_steps += 1
        diagnostic = phase2_training.TrainingStepDiagnostics(
            step=self.completed_steps,
            objective=1.0,
            gradient_norm=1.0,
        )
        self.history.append(diagnostic)
        return diagnostic

    def state_dict(self):
        return {
            "weight": self.model.weight.detach().clone(),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "completed_steps": self.completed_steps,
            "history": tuple(self.history),
        }

    def load_state_dict(self, state):
        with torch.no_grad():
            self.model.weight.copy_(state["weight"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.completed_steps = state["completed_steps"]
        self.history = list(state["history"])


class _RecoveryDoubleStepTrainer(_RecoveryFirstOrderTrainer):
    """Failure injection: two AdamW updates are hidden behind one trainer step."""

    def step(self):
        self.observed_learning_rates.append(float(self.optimizer.param_groups[0]["lr"]))
        self.optimizer.zero_grad(set_to_none=True)
        self.model.weight.grad = torch.ones_like(self.model.weight)
        self.optimizer.step()
        self.optimizer.step()
        self.completed_steps += 1
        diagnostic = phase2_training.TrainingStepDiagnostics(
            step=self.completed_steps,
            objective=1.0,
            gradient_norm=1.0,
        )
        self.history.append(diagnostic)
        return diagnostic


class _RecoveryResetMomentsTrainer(_RecoveryFirstOrderTrainer):
    """Failure injection: erase moments immediately before update two."""

    def step(self):
        if self.completed_steps == 1:
            self.optimizer.state.clear()
        return super().step()


class _RecoveryDoesNotRestoreOptimizerTrainer(_RecoveryFirstOrderTrainer):
    """Failure injection: restore the selected head/history but not AdamW."""

    def load_state_dict(self, state):
        with torch.no_grad():
            self.model.weight.copy_(state["weight"])
        self.completed_steps = state["completed_steps"]
        self.history = list(state["history"])


class _RecoveryReplacesOptimizerTrainer(_RecoveryFirstOrderTrainer):
    """Failure injection: reconstruct AdamW while restoring a checkpoint."""

    def load_state_dict(self, state):
        super().load_state_dict(state)
        replacement = torch.optim.AdamW(
            [self.model.weight],
            lr=float(self.optimizer.param_groups[0]["lr"]),
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )
        replacement.load_state_dict(state["optimizer"])
        self.optimizer = replacement


def _recovery_optimizer_protocol() -> AdamWRecoveryProtocol:
    return AdamWRecoveryProtocol(
        stages=(
            LearningRateStage(1, 5760, 1.0e-3),
            LearningRateStage(5761, 6760, 3.0e-4),
            LearningRateStage(6761, 8760, 1.0e-4),
            LearningRateStage(8761, 10760, 3.0e-5),
            LearningRateStage(10761, 12760, 1.0e-5),
        ),
        schedule_sha256=PHASE2_RECOVERY_LR_SCHEDULE_SHA256,
    )


def test_first_order_controller_early_stops_restores_and_keeps_snapshot() -> None:
    trainer = _FakeFirstOrderTrainer()
    gradients = {0: 10.0, 1: 3.0, 2: 1.0, 3: 0.5, 4: 0.25}

    def audit():
        return phase2_training._FirstOrderMeasurement(
            objective=float(10 - trainer.completed_steps),
            gradient_l2_norm=gradients[trainer.completed_steps],
        )

    observed = phase2_training._run_trainer_to_first_order_convergence(
        trainer,
        audit=audit,
        spec=FirstOrderConvergenceSpec(
            gradient_ratio_tolerance=0.1,
            min_steps=1,
            max_steps=4,
            check_interval=1,
            consecutive_checks=2,
        ),
        fixed_snapshot_steps=4,
        objective_name="fake_convex_objective",
    )

    assert trainer.completed_steps == 3
    assert float(trainer.model.weight.item()) == 3.0
    assert len(observed.history) == 3
    assert observed.evidence["selected_primary_step"] == 3
    assert observed.evidence["final_gate"][
        "gradient_ratio_to_zero_initialization"
    ] == pytest.approx(0.05)
    snapshot = observed.evidence["fixed_step_compute_matched_snapshot"]
    assert snapshot["step"] == 4
    assert snapshot["gradient_ratio_to_zero_initialization"] == pytest.approx(0.025)
    assert snapshot["head_sha256"] != observed.evidence["selected_primary_head_sha256"]
    assert snapshot["used_as_primary_selection_rule"] is False


def test_first_order_controller_fails_closed_with_complete_evidence() -> None:
    trainer = _FakeFirstOrderTrainer()

    def audit():
        return phase2_training._FirstOrderMeasurement(
            objective=1.0,
            gradient_l2_norm=1.0,
        )

    with pytest.raises(OptimizationConvergenceError) as caught:
        phase2_training._run_trainer_to_first_order_convergence(
            trainer,
            audit=audit,
            spec=FirstOrderConvergenceSpec(
                gradient_ratio_tolerance=0.01,
                min_steps=1,
                max_steps=4,
                check_interval=1,
                consecutive_checks=2,
            ),
            fixed_snapshot_steps=4,
            objective_name="never_converges",
        )

    evidence = caught.value.evidence
    assert evidence["converged"] is False
    assert evidence["fail_closed"] is True
    assert evidence["selected_primary_step"] is None
    assert evidence["final_gate"] is None
    assert len(evidence["checks"]) == 4
    assert evidence["fixed_step_compute_matched_snapshot"]["step"] == 4
    assert (
        evidence["fixed_step_compute_matched_snapshot"]["used_as_primary_selection_rule"] is False
    )


def test_recovery_controller_sets_lr_before_update_preserves_moments_and_snapshots() -> None:
    trainer = _RecoveryFirstOrderTrainer()
    protocol = _recovery_optimizer_protocol()

    def audit():
        # Force selection only after the second LR transition so the test
        # exercises both moment-preserving boundary assignments.
        gradient = (
            10.0
            if trainer.completed_steps == 0
            else (1.0 if trainer.completed_steps <= 6760 else 0.005)
        )
        return phase2_training._FirstOrderMeasurement(
            objective=1.0,
            gradient_l2_norm=gradient,
            audit_dtype="float64",
        )

    observed = phase2_training._run_trainer_to_first_order_convergence(
        trainer,
        audit=audit,
        spec=FirstOrderConvergenceSpec(
            gradient_ratio_tolerance=1.0e-3,
            min_steps=100,
            max_steps=12760,
            check_interval=20,
            consecutive_checks=3,
            optimizer_protocol=protocol,
        ),
        fixed_snapshot_steps=720,
        objective_name="recovery_test_objective",
    )

    assert trainer.completed_steps == 6820
    assert observed.evidence["selected_primary_step"] == 6820
    assert trainer.observed_learning_rates[5759] == pytest.approx(1.0e-3)
    assert trainer.observed_learning_rates[5760] == pytest.approx(3.0e-4)
    assert trainer.observed_learning_rates[6759] == pytest.approx(3.0e-4)
    assert trainer.observed_learning_rates[6760] == pytest.approx(1.0e-4)
    assert observed.evidence["fixed_step_compute_matched_snapshot"]["step"] == 720
    legacy = observed.evidence["legacy_constant_lr_boundary_snapshot"]
    assert legacy["step"] == 5760
    assert legacy["learning_rate_used_for_update"] == pytest.approx(1.0e-3)
    execution = observed.evidence["optimizer_protocol_execution"]
    assert execution["schema_version"] == "deterministic-adamw-lr-decay-execution/v2"
    assert execution["completed_updates_observed"] == 6820
    assert execution["single_optimizer_instance_for_all_updates"] is True
    assert execution["optimizer_state_reset_at_lr_milestone"] is False
    assert execution["one_optimizer_update_per_step"] is True
    transitions = execution["boundary_transitions"]
    assert [item["next_update"] for item in transitions] == [5761, 6761]
    assert all(item["moments_preserved"] is True for item in transitions)
    assert all(
        item["moment_state_sha256_before_lr_assignment"]
        == item["moment_state_sha256_after_lr_assignment"]
        for item in transitions
    )
    state_checks = execution["per_update_state_checks"]
    assert state_checks["before_update_checks"] == 6820
    assert state_checks["after_update_checks"] == 6820
    assert state_checks["completed_updates_covered"] == 6820
    assert state_checks["first_pre_update_state_empty"] is True
    assert state_checks["all_updates_checked_before_and_after"] is True
    assert state_checks["all_subsequent_pre_update_scalar_steps_exact"] is True
    assert state_checks["all_post_update_scalar_steps_exact"] is True
    assert state_checks["exp_avg_and_exp_avg_sq_shape_dtype_device_valid"] is True
    assert len(state_checks["check_sequence_sha256"]) == 64
    assert execution["selected_primary_optimizer_state_restored_and_verified"] is True
    assert execution["selected_optimizer_object_identity_preserved"] is True
    assert execution["selected_optimizer_moments_restored_and_verified"] is True
    assert execution["selected_head_sha256"] == execution["restored_head_sha256"]
    assert (
        execution["selected_optimizer_state_sha256"] == execution["restored_optimizer_state_sha256"]
    )
    assert (
        execution["selected_checkpoint_optimizer_state_dict_sha256"]
        == execution["restored_optimizer_state_dict_sha256"]
    )
    for name in (
        "selected_head_sha256",
        "selected_optimizer_state_sha256",
        "selected_checkpoint_optimizer_state_dict_sha256",
        "selected_checkpoint_sha256",
    ):
        assert len(execution[name]) == 64
    measurements = [
        observed.evidence["initial_zero_head_measurement"],
        observed.evidence["final_gate"]["measurement"],
        observed.evidence["fixed_step_compute_matched_snapshot"]["measurement"],
        observed.evidence["legacy_constant_lr_boundary_snapshot"]["measurement"],
        *(check["measurement"] for check in observed.evidence["checks"]),
    ]
    assert measurements
    assert all(measurement["audit_dtype"] == "float64" for measurement in measurements)
    group = trainer.optimizer.param_groups[0]
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == pytest.approx(1.0e-8)
    assert group["amsgrad"] is False
    assert group["maximize"] is False
    assert group["foreach"] is False
    assert group["fused"] is False


def test_recovery_controller_rejects_hidden_double_optimizer_step() -> None:
    trainer = _RecoveryDoubleStepTrainer()

    with pytest.raises(RuntimeError, match="scalar step must equal 1"):
        phase2_training._run_trainer_to_first_order_convergence(
            trainer,
            audit=lambda: phase2_training._FirstOrderMeasurement(
                objective=1.0,
                gradient_l2_norm=1.0,
                audit_dtype="float64",
            ),
            spec=FirstOrderConvergenceSpec(
                min_steps=100,
                max_steps=12760,
                check_interval=20,
                consecutive_checks=3,
                optimizer_protocol=_recovery_optimizer_protocol(),
            ),
            fixed_snapshot_steps=720,
            objective_name="double_step_failure_injection",
        )


def test_recovery_controller_rejects_optimizer_moment_reset() -> None:
    trainer = _RecoveryResetMomentsTrainer()

    with pytest.raises(RuntimeError, match="scalar step must equal 2"):
        phase2_training._run_trainer_to_first_order_convergence(
            trainer,
            audit=lambda: phase2_training._FirstOrderMeasurement(
                objective=1.0,
                gradient_l2_norm=1.0,
                audit_dtype="float64",
            ),
            spec=FirstOrderConvergenceSpec(
                min_steps=100,
                max_steps=12760,
                check_interval=20,
                consecutive_checks=3,
                optimizer_protocol=_recovery_optimizer_protocol(),
            ),
            fixed_snapshot_steps=720,
            objective_name="moment_reset_failure_injection",
        )


def test_recovery_controller_rejects_missing_optimizer_restore() -> None:
    trainer = _RecoveryDoesNotRestoreOptimizerTrainer()

    def audit():
        gradient = (
            10.0
            if trainer.completed_steps == 0
            else (0.005 if trainer.completed_steps >= 100 else 1.0)
        )
        return phase2_training._FirstOrderMeasurement(
            objective=1.0,
            gradient_l2_norm=gradient,
            audit_dtype="float64",
        )

    with pytest.raises(RuntimeError, match="scalar step must equal 100"):
        phase2_training._run_trainer_to_first_order_convergence(
            trainer,
            audit=audit,
            spec=FirstOrderConvergenceSpec(
                gradient_ratio_tolerance=1.0e-3,
                min_steps=100,
                max_steps=12760,
                check_interval=20,
                consecutive_checks=1,
                optimizer_protocol=_recovery_optimizer_protocol(),
            ),
            fixed_snapshot_steps=720,
            objective_name="missing_optimizer_restore_failure_injection",
        )


def test_recovery_controller_rejects_optimizer_object_reconstruction() -> None:
    trainer = _RecoveryReplacesOptimizerTrainer()

    def audit():
        gradient = (
            10.0
            if trainer.completed_steps == 0
            else (0.005 if trainer.completed_steps >= 100 else 1.0)
        )
        return phase2_training._FirstOrderMeasurement(
            objective=1.0,
            gradient_l2_norm=gradient,
            audit_dtype="float64",
        )

    with pytest.raises(RuntimeError, match="replaced the optimizer object"):
        phase2_training._run_trainer_to_first_order_convergence(
            trainer,
            audit=audit,
            spec=FirstOrderConvergenceSpec(
                gradient_ratio_tolerance=1.0e-3,
                min_steps=100,
                max_steps=12760,
                check_interval=20,
                consecutive_checks=1,
                optimizer_protocol=_recovery_optimizer_protocol(),
            ),
            fixed_snapshot_steps=720,
            objective_name="optimizer_object_reconstruction_failure_injection",
        )


def test_api_structurally_forbids_heldout_and_old_comparison_heads(
    compiled_settings: Phase2TrainingSettings,
) -> None:
    signature = inspect.signature(train_phase2_heads)
    assert tuple(signature.parameters) == (
        "training",
        "train_oracle_rewards",
        "seed",
        "settings",
    )
    assert "validation" not in signature.parameters
    assert "test" not in signature.parameters
    assert "heads" not in signature.parameters
    assert "comparison" not in signature.parameters

    with pytest.raises(TypeError, match="unexpected keyword"):
        train_phase2_heads(
            _training(),
            _oracle_rewards(_training()),
            seed=20260801,
            settings=compiled_settings,
            comparison_heads={BT_MLE: (1.0,), PRORM_PLUS: (1.0,)},  # type: ignore[call-arg]
        )


def test_runner_adapter_always_delegates_to_fresh_training(
    compiled_settings: Phase2TrainingSettings,
    monkeypatch: pytest.MonkeyPatch,
    trained_result,
) -> None:
    _, _, expected = trained_result
    calls: list[dict[str, object]] = []

    def fake_train(training, rewards, *, seed, settings):
        calls.append(
            {
                "training": training,
                "rewards": rewards,
                "seed": seed,
                "settings": settings,
            }
        )
        return expected

    monkeypatch.setattr(phase2_training, "train_phase2_heads", fake_train)
    adapter = FreshPhase2HeadTrainer(compiled_settings)
    training = _training()
    rewards = _oracle_rewards(training)
    observed = adapter.train_heads(training, rewards, seed=20260801)

    assert isinstance(observed, Phase2HeadTrainingResult)
    assert observed.heads == expected.heads
    assert adapter.last_result is expected
    assert calls == [
        {
            "training": training,
            "rewards": rewards,
            "seed": 20260801,
            "settings": compiled_settings,
        }
    ]
