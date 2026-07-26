from __future__ import annotations

import copy
import json
import random
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

import smart_reward.phase2_primary as phase2_primary
import smart_reward.phase2_training as phase2_training
from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_checkpoint import CheckpointInterruption, DurableCheckpointStore
from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_primary import (
    PHASE2_PRIMARY_CORE_CAMPAIGN_KIND,
    PHASE2_PRIMARY_CORE_EXECUTION_REVISION,
    PHASE2_PRIMARY_CORE_EXECUTION_ROLE,
    PHASE2_PRIMARY_CORE_ROLE,
    NeutralPhase2TrainingContext,
    prepare_neutral_phase2_context,
    primary_core_checkpoint_binding,
    train_primary_head_core,
)
from smart_reward.phase2_training import (
    FirstOrderConvergenceSpec,
    Phase2TrainingSettings,
    compile_phase2_training_settings,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260801


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
    return 0.2 * torch.sin(0.3 * node)


@pytest.fixture(scope="module")
def toy_settings() -> Phase2TrainingSettings:
    compiled = compile_phase2_training_settings(
        load_phase2_config_bundle(ROOT / "configs" / "common_beta_pilot.yaml")
    )
    return replace(
        compiled,
        phase2_config_hash="e" * 64,
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
def primary_context(toy_settings: Phase2TrainingSettings) -> NeutralPhase2TrainingContext:
    training = _training()
    return prepare_neutral_phase2_context(
        training,
        _oracle_rewards(training),
        seed=SEED,
        settings=toy_settings,
    )


def _one_step_controller(trainer, *, audit, objective_name, **_kwargs):
    initial = audit()
    diagnostic = trainer.step()
    final = audit()
    return phase2_training._ConvergedTrainingRun(
        history=(diagnostic,),
        initial=initial,
        final=final,
        evidence={
            "schema_version": "unit-test-one-step-controller/v1",
            "objective": objective_name,
            "converged": True,
            "test_or_validation_data_accessed": False,
        },
    )


def test_primary_context_and_head_evidence_match_original_primary_prefix(
    primary_context: NeutralPhase2TrainingContext,
    toy_settings: Phase2TrainingSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    captured_heads: dict[str, object] = {}
    execution_roles: list[str | None] = []
    original_label_evidence = phase2_training.LabelStreamEvidence
    original_make_head_evidence = phase2_training._make_head_evidence

    def capture_label_evidence(**kwargs):
        evidence = original_label_evidence(**kwargs)
        captured["label_stream"] = evidence
        return evidence

    def capture_head_evidence(**kwargs):
        evidence = original_make_head_evidence(**kwargs)
        if kwargs["arm"] == phase2_training.PRIMARY_TRAINING_ARM:
            captured_heads[kwargs["method"]] = evidence
        return evidence

    def one_step_with_role_capture(trainer, **kwargs):
        execution_roles.append(kwargs.get("execution_role"))
        return _one_step_controller(trainer, **kwargs)

    class StopBeforeControls(RuntimeError):
        pass

    def forbid_control(*_args, **_kwargs):
        raise StopBeforeControls

    monkeypatch.setattr(phase2_training, "LabelStreamEvidence", capture_label_evidence)
    monkeypatch.setattr(phase2_training, "_make_head_evidence", capture_head_evidence)
    monkeypatch.setattr(
        phase2_training,
        "_run_trainer_to_first_order_convergence",
        one_step_with_role_capture,
    )
    monkeypatch.setattr(
        phase2_training,
        "select_seeded_orthonormal_tangent",
        forbid_control,
    )
    training = _training()
    with pytest.raises(StopBeforeControls):
        phase2_training.train_phase2_heads(
            training,
            _oracle_rewards(training),
            seed=SEED,
            settings=toy_settings,
        )
    assert set(captured_heads) == {BT_MLE, PRORM_PLUS}
    assert captured["label_stream"].to_dict() == primary_context.label_stream.to_dict()

    monkeypatch.setattr(phase2_training, "LabelStreamEvidence", original_label_evidence)
    monkeypatch.setattr(
        phase2_training,
        "_make_head_evidence",
        original_make_head_evidence,
    )
    for learner in (BT_MLE, PRORM_PLUS):
        result = train_primary_head_core(primary_context, learner)
        assert result.head.to_dict() == captured_heads[learner].to_dict()
        assert result.label_stream.to_dict() == captured["label_stream"].to_dict()
        assert result.to_dict()["information_boundary"]["controls_executed"] is False
    assert execution_roles[-2:] == [
        PHASE2_PRIMARY_CORE_EXECUTION_ROLE,
        PHASE2_PRIMARY_CORE_EXECUTION_ROLE,
    ]


@pytest.mark.parametrize(
    ("learner", "forbidden_trainer"),
    ((BT_MLE, "ProRMPlusTrainer"), (PRORM_PLUS, "BTMLETrainer")),
)
def test_each_primary_call_instantiates_no_other_trainer_or_control(
    primary_context: NeutralPhase2TrainingContext,
    learner: str,
    forbidden_trainer: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unrequested trainer or control was instantiated")

    monkeypatch.setattr(phase2_primary, forbidden_trainer, forbidden)
    for name in (
        "_train_low_dimensional_control",
        "_train_exact_soft_label_bt_control",
        "build_direct_oracle_geometry_control",
        "build_exact_margin_canonical_arm",
        "select_seeded_orthonormal_tangent",
    ):
        monkeypatch.setattr(phase2_training, name, forbidden)
    monkeypatch.setattr(
        phase2_training,
        "_run_trainer_to_first_order_convergence",
        _one_step_controller,
    )

    result = train_primary_head_core(primary_context, learner)

    assert result.learner == learner
    assert result.head.method == learner
    assert result.to_dict()["information_boundary"]["controls_executed"] is False


def test_neutral_context_and_core_result_hashes_cover_all_serialized_content(
    primary_context: NeutralPhase2TrainingContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phase2_training,
        "_run_trainer_to_first_order_convergence",
        _one_step_controller,
    )
    result = train_primary_head_core(primary_context, BT_MLE)

    changed_label = replace(
        primary_context.label_stream,
        generator_device="changed-device",
    )
    with pytest.raises(ValueError, match="context identity"):
        replace(primary_context, label_stream=changed_label)

    mutations = (
        {"training_design_sha256": "f" * 64},
        {"absolute_damping": result.absolute_damping * 2.0},
        {"label_stream": changed_label},
        {
            "reward_head_identifiability": {
                **result.reward_head_identifiability,
                "test_or_validation_data_accessed": True,
            }
        },
        {"checkpoint_store_used": not result.checkpoint_store_used},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="result identity"):
            replace(result, **mutation)

    with pytest.raises(ValueError, match="head_sha256"):
        replace(
            result.head,
            head_weight=(result.head.head_weight[0] + 1.0, *result.head.head_weight[1:]),
        )


def test_checkpoint_binding_mechanically_rejects_profile_and_every_identity_change(
    primary_context: NeutralPhase2TrainingContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = primary_core_checkpoint_binding(primary_context, BT_MLE)
    assert PHASE2_PRIMARY_CORE_CAMPAIGN_KIND == "phase2_r4_primary_core_unclaimed"
    assert PHASE2_PRIMARY_CORE_EXECUTION_REVISION == 0
    assert PHASE2_PRIMARY_CORE_ROLE == "unclaimed_train_only_core"
    assert expected == {
        "schema_version": phase2_primary.PHASE2_PRIMARY_CORE_CHECKPOINT_BINDING_SCHEMA,
        "campaign_kind": PHASE2_PRIMARY_CORE_CAMPAIGN_KIND,
        "execution_revision": PHASE2_PRIMARY_CORE_EXECUTION_REVISION,
        "role": PHASE2_PRIMARY_CORE_ROLE,
        "objective": BT_MLE,
        "context_sha256": primary_context.context_sha256,
        "settings_sha256": primary_context.settings.sha256,
        "input_training_sha256": primary_context.input_training_sha256,
        "oracle_reward_sha256": primary_context.oracle_reward_sha256,
        "label_stream_sha256": primary_context.label_stream.label_stream_sha256,
        "seed": SEED,
        "formal_r3_evidence": False,
        "active_named_rng_states": [],
    }

    def must_not_build(*_args, **_kwargs):
        raise AssertionError("binding rejection must happen before trainer construction")

    monkeypatch.setattr(phase2_primary, "build_primary_core_trainer", must_not_build)
    mutations = {
        "campaign_kind": "phase2_recovery_r3_profile",
        "execution_revision": 2,
        "role": "phase2_recovery_r3_profile_nonreusable",
        "objective": PRORM_PLUS,
        "context_sha256": "0" * 64,
        "settings_sha256": "1" * 64,
        "input_training_sha256": "2" * 64,
        "oracle_reward_sha256": "3" * 64,
        "label_stream_sha256": "4" * 64,
        "seed": SEED + 1,
        "profile_nonreusable": True,
        "active_named_rng_states": ["profile_generator"],
    }
    for index, (field, mutated_value) in enumerate(mutations.items()):
        bad_binding = copy.deepcopy(expected)
        bad_binding[field] = mutated_value
        store = DurableCheckpointStore(
            tmp_path / f"bad-{index:02d}",
            objective=BT_MLE,
            binding=bad_binding,
        )
        with pytest.raises(ValueError, match="binding"):
            train_primary_head_core(
                primary_context,
                BT_MLE,
                checkpoint_store=store,
            )


def test_primary_context_rejects_non_frozen_check_cadence(
    toy_settings: Phase2TrainingSettings,
) -> None:
    training = _training()
    bad_settings = replace(
        toy_settings,
        convergence=replace(
            toy_settings.convergence,
            check_interval=10,
        ),
    )

    with pytest.raises(ValueError, match="frozen 20-update cadence"):
        prepare_neutral_phase2_context(
            training,
            _oracle_rewards(training),
            seed=SEED,
            settings=bad_settings,
        )


def test_terminal_generation_audit_fails_before_completed_result(
    primary_context: NeutralPhase2TrainingContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DurableCheckpointStore(
        tmp_path / "terminal-audit-failure",
        objective=BT_MLE,
        binding=primary_core_checkpoint_binding(primary_context, BT_MLE),
    )
    original_audit_generations = DurableCheckpointStore.audit_generations
    audit_calls = 0

    def fail_terminal_audit(self, *, verify_all_checkpoint_bytes):
        nonlocal audit_calls
        audit_calls += 1
        if audit_calls == 2:
            raise RuntimeError("injected terminal checkpoint audit failure")
        return original_audit_generations(
            self,
            verify_all_checkpoint_bytes=verify_all_checkpoint_bytes,
        )

    def one_step_with_checkpoint(
        trainer,
        *,
        audit,
        objective_name,
        checkpoint_hook,
        **_kwargs,
    ):
        initial = audit()
        diagnostic = trainer.step()
        final = audit()
        checkpoint_hook(
            {"controller_state": {"completed_steps": trainer.completed_steps}},
            reason="interval",
        )
        return phase2_training._ConvergedTrainingRun(
            history=(diagnostic,),
            initial=initial,
            final=final,
            evidence={
                "schema_version": "unit-test-one-step-controller/v1",
                "objective": objective_name,
                "converged": True,
                "test_or_validation_data_accessed": False,
            },
        )

    monkeypatch.setattr(
        DurableCheckpointStore,
        "audit_generations",
        fail_terminal_audit,
    )
    monkeypatch.setattr(
        phase2_training,
        "_run_trainer_to_first_order_convergence",
        one_step_with_checkpoint,
    )

    with pytest.raises(RuntimeError, match="injected terminal checkpoint audit failure"):
        train_primary_head_core(
            primary_context,
            BT_MLE,
            checkpoint_store=store,
        )

    progress_statuses = [
        json.loads(path.read_text(encoding="utf-8"))["status"]
        for path in sorted(store.progress_directory.glob("event-*.json"))
    ]
    assert audit_calls == 2
    assert "completed" not in progress_statuses


@pytest.mark.parametrize("learner", (BT_MLE, PRORM_PLUS))
def test_real_primary_disk_signal_resume_matches_uninterrupted(
    primary_context: NeutralPhase2TrainingContext,
    learner: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninterrupted = train_primary_head_core(primary_context, learner)
    binding = primary_core_checkpoint_binding(primary_context, learner)
    store = DurableCheckpointStore(
        tmp_path / learner,
        objective=learner,
        binding=binding,
    )
    original_builder = phase2_primary.build_primary_core_trainer
    active_trainer: dict[str, object] = {}

    def observed_builder(context, method):
        trainer = original_builder(context, method)
        active_trainer["value"] = trainer
        return trainer

    monkeypatch.setattr(
        phase2_primary,
        "build_primary_core_trainer",
        observed_builder,
    )
    signal_requested = False

    def request_signal_at_step_twenty() -> str | None:
        nonlocal signal_requested
        trainer = active_trainer.get("value")
        completed = getattr(trainer, "completed_steps", 0)
        if not signal_requested and completed >= 20:
            signal_requested = True
            return "SIGUSR1"
        return None

    with pytest.raises(CheckpointInterruption, match="SIGUSR1"):
        train_primary_head_core(
            primary_context,
            learner,
            checkpoint_store=store,
            stop_requested=request_signal_at_step_twenty,
        )
    first_generation = store.audit_generations(verify_all_checkpoint_bytes=True)
    assert len(first_generation) == 1
    assert first_generation[0]["completed_steps"] == 20
    assert first_generation[0]["save_reason"] == "signal"

    resumed = train_primary_head_core(
        primary_context,
        learner,
        checkpoint_store=store,
    )

    assert resumed.resumed_from_checkpoint is True
    assert resumed.head.to_dict() == uninterrupted.head.to_dict()
    assert resumed.primary_head_result_sha256 != uninterrupted.primary_head_result_sha256
    generations = store.audit_generations(verify_all_checkpoint_bytes=True)
    assert [item["generation"] for item in generations] == list(range(1, len(generations) + 1))
    assert generations[-1]["completed_steps"] == 720
    assert generations[-1]["save_reason"] == "interval"
    progress_events = sorted(store.progress_directory.glob("event-*.json"))
    terminal_progress = json.loads(progress_events[-1].read_text(encoding="utf-8"))
    selected_step = resumed.head.first_order_convergence["selected_primary_step"]
    assert selected_step < 720
    assert terminal_progress["status"] == "completed"
    assert terminal_progress["completed_steps"] == 720
    assert terminal_progress["details"]["controller_updates_executed"] == 720
    assert terminal_progress["details"]["selected_primary_step"] == selected_step


def test_named_rng_is_consumed_before_training_and_never_active(
    toy_settings: Phase2TrainingSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_before = torch.get_rng_state().clone()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    training = _training()

    context = prepare_neutral_phase2_context(
        training,
        _oracle_rewards(training),
        seed=SEED,
        settings=toy_settings,
    )

    assert torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert all(
        not isinstance(getattr(context, field.name), torch.Generator) for field in fields(context)
    )
    binding = primary_core_checkpoint_binding(context, BT_MLE)
    assert binding["active_named_rng_states"] == []
    assert binding["formal_r3_evidence"] is False

    monkeypatch.setattr(
        phase2_training,
        "_run_trainer_to_first_order_convergence",
        _one_step_controller,
    )
    result = train_primary_head_core(context, BT_MLE)
    serialized = result.to_dict()
    assert serialized["named_rng_contract"] == {
        "label_rng_consumed_during_context_preparation": True,
        "label_rng_state_retained": False,
        "active_named_rng_states": [],
    }
    assert serialized["information_boundary"] == {
        "train_only": True,
        "raw_oracle_rewards_retained": False,
        "raw_repeated_labels_retained": False,
        "validation_or_test_data_accessed": False,
        "policy_session_opened": False,
        "policy_rollout_performed": False,
        "beta_outcome_computed": False,
        "controls_executed": False,
    }
    assert not any(isinstance(value, torch.Tensor) for value in serialized.values())
