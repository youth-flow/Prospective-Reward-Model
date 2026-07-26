from __future__ import annotations

import copy
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import smart_reward.phase2_r3_primary as r3_primary
from smart_reward import phase2_training as training
from smart_reward.contracts import BT_MLE, PRORM_PLUS
from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_checkpoint import (
    CheckpointInterruption,
    CheckpointSignal,
    DurableCheckpointStore,
)
from smart_reward.phase2_primary import (
    build_primary_core_trainer,
)
from smart_reward.phase2_r3_identity import (
    ArtifactRef,
    admit_primary_segment,
    authorize_gate_p,
    create_r3_primary_design,
    validate_continuation_evidence,
)
from smart_reward.phase2_r3_primary import (
    FORMAL_PRIMARY_EXECUTION_ROLE,
    _formal_checkpoint_payload,
    capture_slurm_segment_runtime,
    continuation_checkpoint_artifact_ref,
    formal_primary_checkpoint_binding,
    run_formal_r3_primary_head_segment,
)
from tests import conftest as r3_test_support
from tests import test_phase2_r3_profile as profile_test_support

ROOT = Path(__file__).resolve().parents[1]
SCIENCE_PATH = ROOT / "configs" / "phase2_recovery_r3_science.yaml"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ref(schema: str, role: str, value: str) -> ArtifactRef:
    return ArtifactRef(
        schema_version=schema,
        artifact_sha256=_digest(value),
        role=role,
    )


def _training() -> TrainingTensorData:
    node = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    return TrainingTensorData(
        prompt_ids=tuple(f"r3-primary-{index}" for index in range(4)),
        policy_scores=torch.stack(
            [torch.sin(0.11 * node), torch.cos(0.17 * node), 0.03 * node],
            dim=-1,
        ),
        reward_features=torch.stack(
            [torch.cos(0.13 * node), torch.sin(0.19 * node)],
            dim=-1,
        ),
        h=torch.linspace(-0.4, 0.3, 4),
        left_wins=torch.tensor([1, 2, 1, 3], dtype=torch.int64),
        num_annotations=torch.tensor([3, 4, 2, 5], dtype=torch.int64),
    )


def _admitted_design():
    evidence = r3_test_support.make_shared_gate_p_evidence()
    materialization_capability = evidence.materialization_capability
    authorization = authorize_gate_p(
        operational_bundle=evidence.operational_bundle,
        successful_terminal=evidence.successful_terminal,
    )
    design = create_r3_primary_design(
        science=evidence.science,
        gate0_capability=evidence.gate0_capability,
        gate1_capabilities=evidence.gate1_capabilities,
        profile_authorization=authorization,
        operational_bundle=evidence.operational_bundle,
    )
    segment = admit_primary_segment(
        design=design,
        materialization_capability=materialization_capability,
        task_id=0,
        seed=20260801,
        segment_index=1,
    )
    return materialization_capability.materialization.context, segment


def _admitted_design_for_operational_bundle(bundle):
    evidence = r3_test_support.make_shared_gate_p_evidence()
    terminal = evidence.successful_terminal
    if bundle.file_sha256 != evidence.operational_bundle.file_sha256:
        terminal = profile_test_support._successful_profile_terminal(
            bundle,
            r3_test_support._PersistentTempPathFactory().mktemp("primary-profile-terminal"),
        )
    authorization = authorize_gate_p(
        operational_bundle=bundle,
        successful_terminal=terminal,
    )
    design = create_r3_primary_design(
        science=evidence.science,
        gate0_capability=evidence.gate0_capability,
        gate1_capabilities=evidence.gate1_capabilities,
        profile_authorization=authorization,
        operational_bundle=bundle,
    )
    return admit_primary_segment(
        design=design,
        materialization_capability=evidence.materialization_capability,
        task_id=0,
        seed=20260801,
        segment_index=1,
    )


def _published_operational_bundle(tmp_path_factory: pytest.TempPathFactory):
    del tmp_path_factory
    evidence = r3_test_support.make_shared_gate_p_evidence()
    return (
        evidence.operational_bundle,
        evidence.formal_result,
        evidence.resource_plan,
    )


def _slurm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "SLURM_CLUSTER_NAME": "hpc4",
        "SLURM_JOB_ID": "200001",
        "SLURM_ARRAY_JOB_ID": "200000",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ACCOUNT": "sigroup",
        "SLURM_JOB_PARTITION": "gpu-l20",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_checkpoint_binding_is_stable_across_segments_and_distinct_by_head(
    tmp_path: Path,
) -> None:
    del tmp_path
    _, segment = _admitted_design()
    bt = formal_primary_checkpoint_binding(segment, BT_MLE)
    prorm = formal_primary_checkpoint_binding(segment, PRORM_PLUS)
    assert bt["logical_run_id"] == prorm["logical_run_id"]
    assert bt["head_run_id"] != prorm["head_run_id"]
    assert bt["formal_r3_evidence"] is True
    assert bt["information_boundary"] == "train_only"

    shared = r3_test_support.make_shared_gate_p_evidence()
    terminal = r3_test_support.make_continuable_primary_terminal(
        predecessor=segment,
        operational_bundle=shared.operational_bundle,
    )
    evidence = validate_continuation_evidence(
        predecessor=segment,
        continuable_terminal=terminal,
    )
    continuation = admit_primary_segment(
        design=segment.design,
        materialization_capability=shared.materialization_capability,
        task_id=0,
        seed=20260801,
        segment_index=2,
        continuation_evidence=evidence,
    )
    assert formal_primary_checkpoint_binding(continuation, BT_MLE) == bt


def test_slurm_runtime_is_environment_captured_and_tamper_evident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, segment = _admitted_design()
    _slurm_environment(monkeypatch)
    runtime = capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=43200,
    )
    assert runtime.account == "sigroup"
    assert runtime.array_task_id == segment.task_id
    with pytest.raises((FrozenInstanceError, AttributeError)):
        runtime.job_id = "tampered"  # type: ignore[misc]
    with pytest.raises(TypeError, match="captured from the process environment"):
        replace(runtime, _factory_token=object())
    forged_payload = runtime.to_dict()
    forged_payload.pop("runtime_sha256")
    forged_payload["job_id"] = "forged-job"
    with pytest.raises(TypeError, match="captured from the process environment"):
        replace(
            runtime,
            job_id="forged-job",
            runtime_sha256=r3_primary._canonical_sha256(forged_payload),
        )


def test_slurm_runtime_rejects_wrong_task_or_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, segment = _admitted_design()
    _slurm_environment(monkeypatch)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    with pytest.raises(ValueError, match="array task"):
        capture_slurm_segment_runtime(segment, requested_walltime_seconds=43200)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "other")
    with pytest.raises(ValueError, match="sigroup"):
        capture_slurm_segment_runtime(segment, requested_walltime_seconds=43200)


def test_primary_accepts_only_reopened_sealed_operational_bundle(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _, live_plan = _published_operational_bundle(tmp_path_factory)
    segment = _admitted_design_for_operational_bundle(bundle)
    _slurm_environment(monkeypatch)
    runtime = capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=(bundle.requested_walltime_seconds_per_segment),
    )

    assert (
        r3_primary._validate_primary_resource_plan(
            admission=segment,
            runtime=runtime,
            resource_plan=bundle,
        )
        is bundle
    )
    with pytest.raises(TypeError, match="VerifiedGatePOperationalBundle"):
        r3_primary._validate_primary_resource_plan(
            admission=segment,
            runtime=runtime,
            resource_plan=bundle.to_dict(),
        )
    with pytest.raises(TypeError, match="VerifiedGatePOperationalBundle"):
        r3_primary._validate_primary_resource_plan(
            admission=segment,
            runtime=runtime,
            resource_plan=live_plan,
        )
    with pytest.raises(TypeError, match="produced by reopen"):
        replace(bundle, _factory_token=object())


def test_formal_runner_rejects_cpu_before_training_or_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, segment = _admitted_design()
    _slurm_environment(monkeypatch)
    runtime = capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=43200,
    )
    store = DurableCheckpointStore(
        tmp_path,
        objective=BT_MLE,
        binding=formal_primary_checkpoint_binding(segment, BT_MLE),
    )
    signal = CheckpointSignal()
    monkeypatch.setattr(
        r3_primary,
        "_validate_primary_resource_plan",
        lambda **_kwargs: SimpleNamespace(
            durable_checkpoint_cadence_updates=200,
            gpu_name="NVIDIA L20",
        ),
    )
    monkeypatch.setattr(
        r3_primary,
        "_validate_head_execution_slice",
        lambda *_args, **_kwargs: {
            "slice_sha256": _digest("slice"),
            "fresh_or_resume": "fresh",
        },
    )
    with signal, pytest.raises(RuntimeError, match="requires one coherent CUDA"):
        run_formal_r3_primary_head_segment(
            segment,
            BT_MLE,
            runtime=runtime,
            resource_plan=object(),  # intercepted before any formal promotion
            checkpoint_store=store,
            checkpoint_signal=signal,
            head_execution_slice={},
        )
    assert store.audit_generations(verify_all_checkpoint_bytes=True) == ()


def test_formal_runner_requires_installed_signal_handlers_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, segment = _admitted_design()
    _slurm_environment(monkeypatch)
    runtime = capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=43200,
    )
    store = DurableCheckpointStore(
        tmp_path,
        objective=BT_MLE,
        binding=formal_primary_checkpoint_binding(segment, BT_MLE),
    )
    monkeypatch.setattr(
        r3_primary,
        "_validate_primary_resource_plan",
        lambda **_kwargs: SimpleNamespace(
            durable_checkpoint_cadence_updates=200,
            gpu_name="NVIDIA L20",
        ),
    )
    monkeypatch.setattr(
        r3_primary,
        "_validate_head_execution_slice",
        lambda *_args, **_kwargs: {
            "slice_sha256": _digest("slice"),
            "fresh_or_resume": "fresh",
        },
    )

    with pytest.raises(RuntimeError, match="installed checkpoint signal handlers"):
        run_formal_r3_primary_head_segment(
            segment,
            BT_MLE,
            runtime=runtime,
            resource_plan=object(),
            checkpoint_store=store,
            checkpoint_signal=CheckpointSignal(),
            head_execution_slice={},
        )
    assert store.audit_generations(verify_all_checkpoint_bytes=True) == ()


def test_primary_strictly_validates_self_hashed_head_execution_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, segment = _admitted_design()
    _slurm_environment(monkeypatch)
    runtime = capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=43200,
    )
    start_cursor = {
        "global_safe_block": 0,
        "bt_mle_completed_updates": 0,
        "prorm_plus_completed_updates": 0,
        "next_head": BT_MLE,
    }
    end_cursor = {
        "global_safe_block": 2,
        "bt_mle_completed_updates": 40,
        "prorm_plus_completed_updates": 0,
        "next_head": BT_MLE,
    }
    plan = SimpleNamespace(
        resource_plan_sha256=_digest("plan"),
        formal_profile_sha256=_digest("formal-profile"),
        profile_run_sha256=_digest("profile-run"),
        segment_boundaries=(
            {
                "start_boundary": start_cursor,
                "end_boundary": end_cursor,
                "max_safe_update_blocks_to_execute": 2,
            },
        ),
    )
    payload = {
        "schema_version": r3_primary.HEAD_EXECUTION_SLICE_SCHEMA,
        "resource_plan_sha256": plan.resource_plan_sha256,
        "formal_profile_sha256": plan.formal_profile_sha256,
        "profile_run_sha256": plan.profile_run_sha256,
        "design_sha256": segment.design.design_sha256,
        "admission_sha256": segment.admission_sha256,
        "logical_run_id": segment.logical_run_id,
        "head_run_id": segment.head_run_ids[0],
        "scheduler_segment_id": segment.scheduler_segment_id,
        "runtime_sha256": runtime.runtime_sha256,
        "segment_index": 1,
        "task_id": segment.task_id,
        "seed": segment.seed,
        "head": BT_MLE,
        "fresh_or_resume": "fresh",
        "science_audit_cadence_updates": 20,
        "maximum_updates_per_head": 12760,
        "max_safe_update_blocks_to_execute": 2,
        "safe_update_blocks_consumed_before_head": 0,
        "safe_update_blocks_available_to_head": 2,
        "start_completed_updates": 0,
        "end_completed_updates_inclusive": 40,
        "nominal_segment_start_cursor": start_cursor,
        "nominal_segment_start_cursor_sha256": (
            r3_primary._orchestrator_evidence_sha256(start_cursor)
        ),
        "nominal_segment_end_cursor": end_cursor,
        "nominal_segment_end_cursor_sha256": (r3_primary._orchestrator_evidence_sha256(end_cursor)),
        "actual_cursor_before_head": {
            "bt_mle_completed_updates": 0,
            "bt_mle_complete": False,
            "prorm_plus_completed_updates": 0,
            "prorm_plus_complete": False,
            "next_head": BT_MLE,
        },
        "predecessor_checkpoint": None,
        "information_boundary": ("operational_cursor_only_no_scientific_adaptation"),
    }
    payload["actual_cursor_before_head_sha256"] = r3_primary._orchestrator_evidence_sha256(
        payload["actual_cursor_before_head"]
    )
    value = {
        **payload,
        "slice_sha256": r3_primary._orchestrator_evidence_sha256(payload),
    }

    validated = r3_primary._validate_head_execution_slice(
        value,
        admission=segment,
        runtime=runtime,
        plan=plan,
        learner=BT_MLE,
    )
    assert validated["end_completed_updates_inclusive"] == 40

    tampered = copy.deepcopy(value)
    tampered["safe_update_blocks_available_to_head"] = 1
    unsigned = {key: item for key, item in tampered.items() if key != "slice_sha256"}
    tampered["slice_sha256"] = r3_primary._orchestrator_evidence_sha256(unsigned)
    with pytest.raises(ValueError, match="available update budget"):
        r3_primary._validate_head_execution_slice(
            tampered,
            admission=segment,
            runtime=runtime,
            plan=plan,
            learner=BT_MLE,
        )


def test_progress_hook_preserves_audit_grid_and_forces_terminal_checkpoint() -> None:
    context, _ = _admitted_design()
    toy_spec = replace(
        context.settings.convergence,
        gradient_ratio_tolerance=1.0e9,
        min_steps=20,
        max_steps=40,
        consecutive_checks=1,
        optimizer_protocol=None,
    )
    trainer = build_primary_core_trainer(context, BT_MLE)
    checkpoints: list[tuple[int, str]] = []
    progress: list[dict[str, object]] = []

    def checkpoint_hook(payload, *, reason):
        checkpoints.append((payload["controller_state"]["completed_steps"], reason))

    convergence = training._run_trainer_to_first_order_convergence(
        trainer,
        audit=lambda: training._bt_first_order_measurement(trainer),
        spec=toy_spec,
        fixed_snapshot_steps=40,
        objective_name=BT_MLE,
        checkpoint_hook=checkpoint_hook,
        checkpoint_interval_steps=30,
        progress_hook=lambda value: progress.append(dict(value)),
        execution_role=FORMAL_PRIMARY_EXECUTION_ROLE,
    )
    assert convergence.evidence["converged"] is True
    assert [event["completed_steps"] for event in progress] == [20, 30, 40]
    assert checkpoints == [
        (20, "stage_boundary"),
        (30, "interval"),
        (40, "stage_boundary"),
    ]


def test_interruption_progress_retains_the_next_logical_update() -> None:
    context, _ = _admitted_design()
    toy_spec = replace(
        context.settings.convergence,
        gradient_ratio_tolerance=1.0e-30,
        min_steps=20,
        max_steps=40,
        consecutive_checks=3,
        optimizer_protocol=None,
    )
    trainer = build_primary_core_trainer(context, BT_MLE)
    progress: list[dict[str, object]] = []
    saved: list[tuple[int, str]] = []

    def checkpoint_hook(payload, *, reason):
        saved.append((payload["controller_state"]["completed_steps"], reason))

    with pytest.raises(CheckpointInterruption):
        training._run_trainer_to_first_order_convergence(
            trainer,
            audit=lambda: training._bt_first_order_measurement(trainer),
            spec=toy_spec,
            fixed_snapshot_steps=40,
            objective_name=BT_MLE,
            checkpoint_hook=checkpoint_hook,
            checkpoint_interval_steps=30,
            stop_requested=(lambda: "USR1" if trainer.completed_steps >= 20 else None),
            progress_hook=lambda value: progress.append(dict(value)),
            execution_role=FORMAL_PRIMARY_EXECUTION_ROLE,
        )
    assert saved == [(20, "signal")]
    assert progress[-1]["completed_steps"] == 20
    assert progress[-1]["next_update"] == 21
    assert progress[-1]["interruption_requested"] is True


def test_continuation_artifact_rejects_non_controller_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, segment = _admitted_design()
    _slurm_environment(monkeypatch)
    runtime = capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=43200,
    )
    store = DurableCheckpointStore(
        tmp_path,
        objective=BT_MLE,
        binding=formal_primary_checkpoint_binding(segment, BT_MLE),
    )
    store.save(
        {
            "schema_version": "phase2-recovery-r3-primary-checkpoint-payload/v1",
            "design_sha256": segment.design.design_sha256,
            "admission_sha256": segment.admission_sha256,
            "logical_run_id": segment.logical_run_id,
            "head_run_id": segment.head_run_ids[0],
            "scheduler_segment_id": segment.scheduler_segment_id,
            "segment_index": 1,
            "task_id": 0,
            "seed": 20260801,
            "objective": BT_MLE,
            "runtime_sha256": runtime.runtime_sha256,
            "controller_checkpoint_sha256": _digest("fake-controller"),
            "controller_checkpoint": {"checkpoint_sha256": _digest("fake-controller")},
            "information_boundary": "train_only",
        },
        completed_steps=20,
        reason="manual",
    )
    with pytest.raises(RuntimeError, match="signal or planned-boundary receipt"):
        continuation_checkpoint_artifact_ref(
            store,
            predecessor=segment,
            learner=BT_MLE,
        )


def test_formal_checkpoint_payload_binds_current_segment_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, segment = _admitted_design()
    _slurm_environment(monkeypatch)
    runtime = capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=43200,
    )
    controller = {"checkpoint_sha256": _digest("controller")}
    payload = _formal_checkpoint_payload(
        admission=segment,
        runtime=runtime,
        learner=BT_MLE,
        head_execution_slice_sha256=_digest("slice"),
        controller_checkpoint=controller,
    )
    assert payload["admission_sha256"] == segment.admission_sha256
    assert payload["runtime_sha256"] == runtime.runtime_sha256
    assert payload["head_execution_slice_sha256"] == _digest("slice")
    assert payload["controller_checkpoint"] is not controller


def test_selected_terminal_recovers_result_after_publication_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, segment = _admitted_design()
    _slurm_environment(monkeypatch)
    runtime = capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=43200,
    )
    selected_step = 20
    controller_updates = 40
    weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
    history = [
        training.TrainingStepDiagnostics(
            step=step,
            objective=float(100 - step),
            gradient_norm=1.0,
        )
        for step in range(1, selected_step + 1)
    ]

    class TerminalTrainer:
        def __init__(self) -> None:
            self.model = SimpleNamespace(weight=weight)
            self.completed_steps = selected_step
            self.history = history

        def state_dict(self):
            return {
                "weight": self.model.weight.detach().clone(),
                "completed_steps": self.completed_steps,
                "history": [training.asdict(item) for item in self.history],
            }

    trainer = TerminalTrainer()
    initial = training._FirstOrderMeasurement(
        objective=1.0,
        gradient_l2_norm=10.0,
        audit_dtype="float64",
    )
    final = training._FirstOrderMeasurement(
        objective=0.5,
        gradient_l2_norm=0.001,
        audit_dtype="float64",
    )
    head_sha = training._tensor_sha256(weight)
    evidence = {
        "schema_version": "objective-first-order-convergence/v2",
        "objective": BT_MLE,
        "converged": True,
        "fail_closed": True,
        "spec": context.settings.convergence.to_dict(),
        "initial_zero_head_measurement": initial.to_dict(include_audit_dtype=True),
        "selected_primary_step": selected_step,
        "selected_primary_head_sha256": head_sha,
        "final_gate": {
            "step": selected_step,
            "measurement": final.to_dict(include_audit_dtype=True),
            "gradient_ratio_to_zero_initialization": 0.0001,
            "threshold_passed": True,
            "fresh_post_restore_audit": True,
        },
        "test_or_validation_data_accessed": False,
    }
    identity = training._first_order_controller_identity(
        objective_name=BT_MLE,
        execution_role=r3_primary.FORMAL_PRIMARY_EXECUTION_ROLE,
        spec=context.settings.convergence,
        fixed_snapshot_steps=context.settings.outer_steps,
        rank_diagnostic=context.reward_head_identifiability,
    )
    selected_terminal = training._build_selected_primary_terminal_checkpoint(
        trainer,
        identity=identity,
        selected_primary_step=selected_step,
        controller_updates_executed=controller_updates,
        initial=initial,
        final=final,
        evidence=evidence,
    )
    head = training.TrainedHeadEvidence(
        arm=training.PRIMARY_TRAINING_ARM,
        method=BT_MLE,
        head_weight=(0.5,),
        head_dtype="torch.float64",
        initial_head_sha256=training._tensor_sha256(torch.zeros(1, dtype=torch.float64)),
        head_sha256=head_sha,
        initial_objective=1.0,
        final_objective=0.5,
        history_summary=training._history_summary(history),
        final_pcg=None,
        first_order_convergence=evidence,
    )
    slice_sha = _digest("terminal-slice")
    core = r3_primary._terminal_result_core_payload(
        admission=segment,
        runtime=runtime,
        learner=BT_MLE,
        context=context,
        head=head,
        selected_primary_step=selected_step,
        controller_updates_executed=controller_updates,
        head_execution_slice_sha256=slice_sha,
        resumed_from_predecessor=False,
    )
    outer = r3_primary._formal_selected_terminal_payload(
        admission=segment,
        runtime=runtime,
        learner=BT_MLE,
        head_execution_slice_sha256=slice_sha,
        selected_terminal_checkpoint=selected_terminal,
        terminal_result_core=core,
    )
    store = DurableCheckpointStore(
        tmp_path,
        objective=BT_MLE,
        binding=formal_primary_checkpoint_binding(segment, BT_MLE),
    )
    store.save(
        outer,
        completed_steps=selected_step,
        reason="stage_boundary",
    )

    # Simulate process death here: no result file or completion receipt exists.
    recovered = r3_primary.recover_formal_result_from_selected_terminal(
        store,
        admission=segment,
        learner=BT_MLE,
    )

    assert recovered.selected_primary_step == selected_step
    assert recovered.controller_updates_executed == controller_updates
    assert recovered.head.to_dict() == head.to_dict()
    assert recovered.head_execution_slice_sha256 == slice_sha
    assert (
        store.audit_generations(verify_all_checkpoint_bytes=True)[-1]["completed_steps"]
        == selected_step
    )
