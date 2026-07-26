from __future__ import annotations

import copy
import hashlib
import inspect
import io
import math
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import smart_reward.phase2_r3_profile as formal_profile
import smart_reward.phase2_r3_profile_artifacts as profile_artifacts
import smart_reward.phase2_r3_terminal as terminal_evidence
from smart_reward import phase2_r3_inputs, phase2_training
from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_primary import (
    build_primary_core_trainer,
    prepare_neutral_phase2_context,
)
from smart_reward.phase2_profile import (
    PHASE2_PROFILE_AUDIT_UPDATES,
    PHASE2_PROFILE_BINDING_SCHEMA,
    PHASE2_PROFILE_CAMPAIGN_KIND,
    PHASE2_PROFILE_EXECUTION_REVISION,
    PHASE2_PROFILE_LEARNER_ORDER,
    PHASE2_PROFILE_RESULT_SCHEMA,
    PHASE2_PROFILE_ROLE,
    PHASE2_PROFILE_SEED,
    PHASE2_PROFILE_STOP_REASON,
    PHASE2_PROFILE_UPDATES,
    profile_core_binding,
    validate_gate_p_profile_core_result,
)
from smart_reward.phase2_r3_artifacts import canonical_json_bytes
from smart_reward.phase2_r3_config import (
    R3ScienceConfigBundle,
    load_r3_science_config,
)
from smart_reward.phase2_r3_gate0 import R3Gate0Capability
from smart_reward.phase2_r3_gate1 import R3Gate1Capabilities
from smart_reward.phase2_r3_identity import (
    FORMAL_CUDA_PROFILE_RESULT_ROLE,
    FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
    RESOURCE_PLAN_ROLE,
    RESOURCE_PLAN_SCHEMA,
    ArtifactRef,
    ValidatedGatePRun,
    authorize_gate_p,
    create_gate_p_admission,
    create_validated_gate_p_run,
)
from smart_reward.phase2_r3_inputs import R3TrainMaterializationCapability
from smart_reward.phase2_r3_materialization import (
    TrainMaterializationProvenance,
    ValidatedR3Materialization,
    validate_r3_materialization,
)
from smart_reward.phase2_r3_profile import (
    GatePResourcePlan,
    ProfilePreparationTimings,
    ProfileSafetyMarginPolicy,
    SchedulerResourceEnvelope,
    build_gate_p_resource_plan,
    formal_cuda_profile_artifact_ref,
    freeze_profile_safety_margin_policy,
    record_profile_preparation_from_train_input,
    record_profile_preparation_timings,
    resource_plan_artifact_ref,
    run_formal_gate_p_cuda_profile,
    validate_formal_cuda_profile_result,
    validate_gate_p_resource_plan,
    validate_scheduler_resource_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
SCIENCE_CONFIG = ROOT / "configs" / "phase2_recovery_r3_science.yaml"
GPU_NAME = "NVIDIA L20"
GPU_MEMORY_BYTES = 48 * 1024**3


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ref(schema: str, role: str, token: str) -> ArtifactRef:
    return ArtifactRef(
        schema_version=schema,
        artifact_sha256=_digest(token),
        role=role,
    )


def _training() -> TrainingTensorData:
    num_prompts, num_candidates = 4, 4
    node = torch.arange(
        num_prompts * num_candidates,
        dtype=torch.float32,
    ).reshape(num_prompts, num_candidates)
    return TrainingTensorData(
        prompt_ids=tuple(f"r3-profile-{index}" for index in range(num_prompts)),
        policy_scores=torch.stack(
            [
                torch.sin((coordinate + 1.0) * 0.11 * node) + 0.1 * coordinate
                for coordinate in range(3)
            ],
            dim=-1,
        ),
        reward_features=torch.stack(
            [
                torch.cos((coordinate + 1.0) * 0.13 * node) - 0.05 * coordinate
                for coordinate in range(2)
            ],
            dim=-1,
        ),
        h=torch.linspace(-0.4, 0.3, num_prompts),
        left_wins=torch.tensor([1, 2, 1, 3], dtype=torch.int64),
        num_annotations=torch.tensor([3, 4, 2, 5], dtype=torch.int64),
    )


def _validated_profile_run(
    science: R3ScienceConfigBundle,
    *,
    gate0_capability: R3Gate0Capability,
    gate1_capabilities: R3Gate1Capabilities,
    seal_materialization: Callable[
        [ValidatedR3Materialization],
        R3TrainMaterializationCapability,
    ],
) -> ValidatedGatePRun:
    training = _training()
    node = torch.arange(
        training.num_prompts * training.num_candidates,
        dtype=training.policy_scores.dtype,
    ).reshape(training.num_prompts, training.num_candidates)
    context = prepare_neutral_phase2_context(
        training,
        0.2 * torch.sin(0.3 * node),
        seed=PHASE2_PROFILE_SEED,
        settings=science.settings,
    )
    provenance = TrainMaterializationProvenance.from_context(
        context,
        parent_artifact_registry_sha256=_digest("parent-registry"),
        artifact_metadata_sha256=_digest("metadata"),
        artifact_tensors_sha256=_digest("tensors"),
        artifact_candidates_sha256=_digest("candidates"),
        artifact_materialization_sha256=_digest("materialization"),
        artifact_verification_sha256=_digest("verification"),
        source_run_manifest_sha256=_digest("manifest"),
        source_producer_identity_sha256=_digest("producer"),
        candidate_train_prefix_sha256=_digest("candidate-prefix"),
        candidate_train_prefix_count=(
            context.training.num_prompts * context.training.num_candidates
        ),
    )
    materialization = validate_r3_materialization(
        context,
        science_bundle=science,
        provenance=provenance,
    )
    admission = create_gate_p_admission(
        gate0_capability=gate0_capability,
        gate1_capabilities=gate1_capabilities,
        science=science,
    )
    return create_validated_gate_p_run(
        materialization_capability=seal_materialization(materialization),
        science=science,
        admission=admission,
    )


def _cuda_memory(current: int = 256, peak: int = 512) -> dict[str, object]:
    return {
        "measurement": "cuda_allocator",
        "current_bytes": current,
        "peak_bytes": peak,
    }


def _core_profile(profile_run: ValidatedGatePRun) -> dict[str, object]:
    learners: list[dict[str, object]] = []
    for learner_name, step_seconds in (
        ("bt_mle", 0.01),
        ("prorm_plus", 0.02),
    ):
        steps: list[dict[str, object]] = []
        for update in range(1, PHASE2_PROFILE_UPDATES + 1):
            step: dict[str, object] = {
                "update": update,
                "wall_seconds": step_seconds,
                "cuda_memory": _cuda_memory(),
            }
            if learner_name == "prorm_plus":
                step["pcg"] = {
                    "iterations": 3,
                    "residual_norm": 1.0e-6,
                    "relative_residual": 1.0e-6,
                    "converged": True,
                    "reason": "converged",
                }
            steps.append(step)
        learners.append(
            {
                "learner": learner_name,
                "updates_executed": PHASE2_PROFILE_UPDATES,
                "stop_reason": PHASE2_PROFILE_STOP_REASON,
                "build_wall_seconds": 0.1,
                "phase_wall_seconds": 3.0,
                "gradient_selection_applied": False,
                "steps": steps,
                "audits": [
                    {
                        "update": update,
                        "wall_seconds": 0.03,
                        "trainer_state_unchanged": True,
                    }
                    for update in PHASE2_PROFILE_AUDIT_UPDATES
                ],
                "ephemeral_checkpoint_io": [
                    {
                        "update": update,
                        "serialized_bytes": 1_000 + 10 * update,
                        "serialize_wall_seconds": 0.01,
                        "fsync_wall_seconds": 0.01,
                        "reload_wall_seconds": 0.01,
                        "roundtrip_verified": True,
                        "artifact_retained": False,
                        "reusable": False,
                        "filesystem_scope": "declared_profile_directory",
                    }
                    for update in PHASE2_PROFILE_AUDIT_UPDATES
                ],
            }
        )
    context = profile_run.materialization.context
    payload: dict[str, object] = {
        "schema_version": PHASE2_PROFILE_RESULT_SCHEMA,
        "campaign_kind": PHASE2_PROFILE_CAMPAIGN_KIND,
        "execution_revision": PHASE2_PROFILE_EXECUTION_REVISION,
        "role": PHASE2_PROFILE_ROLE,
        "profile_nonreusable": True,
        "seed": PHASE2_PROFILE_SEED,
        "context_sha256": context.context_sha256,
        "settings_sha256": context.settings.sha256,
        "input_training_sha256": context.input_training_sha256,
        "binding_sha256": phase2_training._canonical_sha256(profile_core_binding(context)),
        "learner_order": list(PHASE2_PROFILE_LEARNER_ORDER),
        "update_cap_per_learner": PHASE2_PROFILE_UPDATES,
        "audit_update_indices": list(PHASE2_PROFILE_AUDIT_UPDATES),
        "stop_reason": PHASE2_PROFILE_STOP_REASON,
        "device_type": "cuda",
        "formal_cuda_profile": True,
        "setup": {
            "wall_seconds": 0.05,
            "cuda_memory": _cuda_memory(),
        },
        "learners": learners,
        "information_boundary": {
            "train_only": True,
            "validation_or_test_data_accessed": False,
            "policy_session_opened": False,
            "policy_rollout_performed": False,
            "controls_executed": False,
            "serialized_training_state_retained": False,
            "profile_consumable_as_primary_evidence": False,
        },
    }
    payload["profile_sha256"] = phase2_training._canonical_sha256(payload)
    validate_gate_p_profile_core_result(payload)
    return payload


def _production_checkpoint_io_evidence(
    core: dict[str, object],
) -> dict[str, object]:
    learners: list[dict[str, object]] = []
    for learner_index, learner in enumerate(core["learners"]):
        fixed = 10_000 + learner_index * 2_000
        slope = 100 + learner_index * 20
        samples: list[dict[str, object]] = []
        for update in (*PHASE2_PROFILE_AUDIT_UPDATES, 12_760):
            size = fixed + slope * update
            samples.append(
                {
                    "update": update,
                    "serialized_bytes": size,
                    "serialize_wall_seconds": size / 100_000_000.0,
                    "fsync_wall_seconds": size / 50_000_000.0,
                    "reload_and_verify_wall_seconds": size / 80_000_000.0,
                    "progress_receipt_publish_fsync_verify_wall_seconds": (
                        0.001 + update / 100_000_000.0
                    ),
                    "signal_or_planned_boundary_receipt_publish_fsync_verify_wall_seconds": (
                        0.002 + update / 100_000_000.0
                    ),
                    "final_restore_load_terminal_payload_wall_seconds": (
                        0.003 + update / 100_000_000.0
                    ),
                    "production_outer_payload_isomorphic_profile_schema": True,
                    "durable_outer_envelope_exact_schema": True,
                    "profile_role": True,
                    "reusable_as_primary_state": False,
                    "same_pass_live_trainer_state": True,
                    "full_live_history_records": update,
                    "worst_case_selected_state_copy_included": True,
                    "selected_state_history_records": update,
                    "convergence_check_records": update // 20,
                    "recovery_state_check_records": 2 * update,
                    "fixed_tensor_components_included": True,
                    "fixed_snapshot_included": True,
                    "legacy_boundary_snapshot_included": True,
                    "optimizer_protocol_stage_records": 5,
                    "optimizer_protocol_transition_records": 4,
                    "primary_outer_exact_key_contract_verified": True,
                    "production_controller_builder_used": True,
                    "convergence_check_exact_key_contract_verified": True,
                    "snapshot_exact_key_contract_verified": True,
                    "optimizer_protocol_execution_exact_key_contract_verified": (True),
                    "selected_terminal_production_builder_used": update > 0,
                    "fresh_trainer_restore_load_measured": True,
                    "atomic_publication_fsync_included": True,
                    "committed_byte_verification_included": True,
                }
            )
        learners.append(
            {
                "learner": learner["learner"],
                "samples": samples,
                "fixed_serialized_bytes_upper_bound": fixed,
                "per_update_serialized_byte_slope_upper_bound": slope,
                "target_serialized_bytes_upper_bound": fixed + slope * 12_760,
                "minimum_serialize_throughput_bytes_per_second": 100_000_000.0,
                "minimum_fsync_throughput_bytes_per_second": 50_000_000.0,
                "minimum_reload_verify_throughput_bytes_per_second": (80_000_000.0),
                "maximum_progress_receipt_wall_seconds": (0.001 + 12_760 / 100_000_000.0),
                "maximum_boundary_receipt_wall_seconds": (0.002 + 12_760 / 100_000_000.0),
                "maximum_finalization_noncheckpoint_wall_seconds": (0.003 + 12_760 / 100_000_000.0),
            }
        )
    payload: dict[str, object] = {
        "schema_version": formal_profile.PRODUCTION_CHECKPOINT_IO_PROFILE_SCHEMA,
        "capture_mode": ("same_pass_live_production_outer_envelope_worst_case_selected_copy"),
        "core_profile_sha256": core["profile_sha256"],
        "production_outer_payload_schema": (
            formal_profile.PRODUCTION_OUTER_CHECKPOINT_PAYLOAD_SCHEMA
        ),
        "benchmark_payload_schema": (formal_profile.PRODUCTION_PROFILE_BENCHMARK_PAYLOAD_SCHEMA),
        "durable_checkpoint_envelope_schema": (
            formal_profile.PRODUCTION_DURABLE_CHECKPOINT_ENVELOPE_SCHEMA
        ),
        "controller_checkpoint_schema": (formal_profile.PRODUCTION_CONTROLLER_CHECKPOINT_SCHEMA),
        "sample_updates": [*PHASE2_PROFILE_AUDIT_UPDATES, 12_760],
        "schema_growth_model": {
            "formula": (
                "diagnostic_bytes(step<=100)=fixed+slope*step;"
                "projection_bound(step>100)=measured_target_12760_envelope"
            ),
            "live_trainer_history_records_per_update": 1,
            "selected_state_history_records_per_update": 1,
            "convergence_check_records_per_audit_block": 1,
            "recovery_state_check_records_per_update": 2,
            "only_declared_linear_fields_extrapolated": True,
            "fixed_tensor_components_measured_at_every_sample": True,
            "worst_case_selected_state_copy_measured_at_every_sample": True,
            "nonlinear_or_unbounded_fields_absent": True,
            "simple_fixed_io_times_event_count_forbidden": True,
            "target_12760_envelope_actually_serialized": True,
            "projection_uses_measured_target_not_linear_extrapolation": True,
        },
        "learners": learners,
        "information_boundary": formal_profile.FORMAL_PROFILE_INFORMATION_BOUNDARY,
    }
    payload["evidence_sha256"] = formal_profile._canonical_sha256(payload)
    return payload


def _rehash_core(core: dict[str, object]) -> dict[str, object]:
    copied = copy.deepcopy(core)
    copied.pop("profile_sha256")
    copied["profile_sha256"] = phase2_training._canonical_sha256(copied)
    return copied


def _cuda_identity() -> dict[str, object]:
    return {
        "logical_device_index": 0,
        "name": GPU_NAME,
        "total_memory_bytes": GPU_MEMORY_BYTES,
        "compute_capability_major": 8,
        "compute_capability_minor": 9,
        "torch_cuda_version": "12.4",
        "cuda_visible_devices": "0",
    }


def _gpu_sample(sample_index: int) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "wall_time_ns": 100 + sample_index,
        "monotonic_time_ns": 1_000 + sample_index,
        "uuid": "GPU-test-uuid",
        "name": GPU_NAME,
        "total_memory_bytes": GPU_MEMORY_BYTES,
        "gpu_utilization_percent": 50.0 + sample_index,
        "memory_utilization_percent": 25.0 + sample_index,
    }


@pytest.fixture(scope="module")
def science() -> R3ScienceConfigBundle:
    return load_r3_science_config(SCIENCE_CONFIG)


@pytest.fixture(scope="module")
def profile_run(
    science: R3ScienceConfigBundle,
    sealed_r3_gate0_capability: R3Gate0Capability,
    sealed_r3_gate1_capabilities: R3Gate1Capabilities,
    seal_r3_train_materialization: Callable[
        [ValidatedR3Materialization],
        R3TrainMaterializationCapability,
    ],
) -> ValidatedGatePRun:
    return _validated_profile_run(
        science,
        gate0_capability=sealed_r3_gate0_capability,
        gate1_capabilities=sealed_r3_gate1_capabilities,
        seal_materialization=seal_r3_train_materialization,
    )


@pytest.fixture(scope="module")
def safety_policy(profile_run: ValidatedGatePRun) -> ProfileSafetyMarginPolicy:
    return freeze_profile_safety_margin_policy(
        profile_run,
        walltime_margin_fraction=0.2,
        fixed_walltime_margin_seconds=10.0,
        memory_margin_fraction=0.25,
        signal_margin_seconds=5.0,
        durable_checkpoint_cadence_updates=200,
    )


@pytest.fixture(scope="module")
def envelope(profile_run: ValidatedGatePRun) -> SchedulerResourceEnvelope:
    return validate_scheduler_resource_envelope(
        profile_run,
        scheduler_raw_evidence_sha256=_digest("scheduler-raw"),
        resource_raw_evidence_sha256=_digest("resource-raw"),
        partition="gpu-l20",
        gpu_name=GPU_NAME,
        gpu_total_memory_bytes=GPU_MEMORY_BYTES,
        max_allocation_wall_seconds=300,
        max_array_concurrency=3,
        max_scheduler_segments=4,
        max_gpus_per_task=1,
        max_cpus_per_task=64,
        max_memory_bytes=64 * 1024**3,
    )


@pytest.fixture(scope="module")
def preparation(profile_run: ValidatedGatePRun) -> ProfilePreparationTimings:
    return record_profile_preparation_timings(
        profile_run,
        artifact_verification_wall_seconds=0.1,
        oracle_rescore_wall_seconds=0.3,
        label_reconstruction_wall_seconds=0.2,
    )


def _patch_formal_runtime(
    monkeypatch: pytest.MonkeyPatch,
    core: dict[str, object],
) -> None:
    clock = iter((0, 1_000_000, 2_000_000, 10_000_000_000))
    monkeypatch.setattr(
        formal_profile.time,
        "perf_counter_ns",
        lambda: next(clock),
    )
    monkeypatch.setattr(
        formal_profile,
        "_require_live_cuda",
        lambda _run: _cuda_identity(),
    )
    monkeypatch.setattr(
        formal_profile,
        "_sample_gpu_utilization",
        lambda _identity, *, sample_index: _gpu_sample(sample_index),
    )
    memory_samples = iter(
        (
            {
                "current_rss_bytes": 512 * 1024**2,
                "peak_rss_bytes": 768 * 1024**2,
                "measurement": "linux_proc_status_vmrss_vmhwm",
            },
            {
                "current_rss_bytes": 640 * 1024**2,
                "peak_rss_bytes": 1024 * 1024**2,
                "measurement": "linux_proc_status_vmrss_vmhwm",
            },
        )
    )
    monkeypatch.setattr(
        formal_profile,
        "_read_process_memory",
        lambda: next(memory_samples),
    )
    monkeypatch.setattr(
        formal_profile,
        "_run_core_with_production_checkpoint_io",
        lambda _run, *, io_probe_directory: (
            copy.deepcopy(core),
            _production_checkpoint_io_evidence(core),
        ),
    )


@pytest.fixture
def formal_result(
    profile_run: ValidatedGatePRun,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    preparation: ProfilePreparationTimings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    core = _core_profile(profile_run)
    _patch_formal_runtime(monkeypatch, core)
    return run_formal_gate_p_cuda_profile(
        profile_run,
        safety_policy=safety_policy,
        envelope=envelope,
        preparation=preparation,
        io_probe_directory=tmp_path,
    )


@pytest.fixture(scope="module")
def sealed_operational_bundle(
    profile_run: ValidatedGatePRun,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    preparation: ProfilePreparationTimings,
    tmp_path_factory: pytest.TempPathFactory,
):
    directory = tmp_path_factory.mktemp("sealed-operational-bundle")
    patcher = pytest.MonkeyPatch()
    try:
        core = _core_profile(profile_run)
        _patch_formal_runtime(patcher, core)
        result = run_formal_gate_p_cuda_profile(
            profile_run,
            safety_policy=safety_policy,
            envelope=envelope,
            preparation=preparation,
            io_probe_directory=directory,
        )
    finally:
        patcher.undo()
    plan = build_gate_p_resource_plan(
        result,
        safety_policy=safety_policy,
        envelope=envelope,
        requested_walltime_seconds_per_segment=300,
        array_concurrency=2,
        cpus_per_task=4,
        memory_bytes=2 * 1024**3,
    )
    bundle = profile_artifacts.publish_verified_gate_p_operational_bundle(
        (directory / "gate-p-operational-bundle.json").resolve(),
        profile_run=profile_run,
        safety_policy=safety_policy,
        envelope=envelope,
        formal_result=result,
        resource_plan=plan,
    )
    return bundle, result, plan


def _successful_profile_terminal(
    bundle: profile_artifacts.VerifiedGatePOperationalBundle,
    directory: Path,
) -> terminal_evidence.SuccessfulProfileTerminalCapability:
    intent = terminal_evidence.publish_profile_allocation_intent(
        (directory / "profile-allocation-intent.json").resolve(),
        cluster="hpc4",
        account="sigroup",
        partition=bundle.partition,
        gpu_name=bundle.gpu_name,
        gpus_per_task=bundle.gpus_per_task,
        cpus_per_task=bundle.cpus_per_task,
        memory_bytes=bundle.memory_bytes,
        requested_walltime_seconds=(bundle.requested_walltime_seconds_per_segment),
    )
    environment = {
        "SLURM_CLUSTER_NAME": "hpc4",
        "SLURM_JOB_ID": "510001",
        "SLURM_ARRAY_JOB_ID": "510000",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ACCOUNT": "sigroup",
        "SLURM_JOB_PARTITION": bundle.partition,
        "SLURM_CPUS_PER_TASK": str(bundle.cpus_per_task),
        "SLURM_GPUS_PER_TASK": str(bundle.gpus_per_task),
        "SLURM_MEM_PER_NODE": str(bundle.memory_bytes // 1024**2),
    }
    with patch.dict(os.environ, environment, clear=False):
        receipt = terminal_evidence.capture_profile_slurm_runtime_receipt(
            bundle,
            intent,
            (directory / "profile-runtime-receipt.json").resolve(),
        )
    gpu_token = bundle.partition.removeprefix("gpu-").lower()
    memory_mib = bundle.memory_bytes // 1024**2
    common_tres = (
        f"billing={bundle.cpus_per_task},cpu={bundle.cpus_per_task},"
        f"gres/gpu={bundle.gpus_per_task},mem={memory_mib}M,node=1"
    )
    raw = (
        "|".join(
            (
                receipt.sacct_job_selector,
                receipt.job_id,
                "COMPLETED",
                "0:0",
                "0:0",
                "hpc4",
                "sigroup",
                bundle.partition,
                "test_qos",
                "1",
                str(bundle.cpus_per_task),
                common_tres,
                f"{common_tres},gres/gpu:{gpu_token}={bundle.gpus_per_task}",
                "137",
            )
        )
        + "\n"
    ).encode()
    inspection = terminal_evidence.inspect_sacct_terminal_bytes(
        raw,
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return terminal_evidence.produce_successful_profile_terminal(
        bundle,
        runtime_receipt=receipt,
        inspection=inspection,
        evidence_directory=(directory / "profile-terminal").resolve(),
    )


@pytest.fixture(scope="module")
def successful_profile_terminal(
    sealed_operational_bundle,
    tmp_path_factory: pytest.TempPathFactory,
) -> terminal_evidence.SuccessfulProfileTerminalCapability:
    bundle, _, _ = sealed_operational_bundle
    return _successful_profile_terminal(
        bundle,
        tmp_path_factory.mktemp("successful-profile-terminal"),
    )


def test_safety_policy_is_positive_prefrozen_and_self_hashed(
    profile_run: ValidatedGatePRun,
    safety_policy: ProfileSafetyMarginPolicy,
) -> None:
    assert safety_policy.declared_before_profile is True
    assert safety_policy.durable_checkpoint_cadence_updates == 200
    assert safety_policy.mandatory_checkpoint_updates == (
        5_760,
        6_760,
        8_760,
        10_760,
        12_760,
    )
    assert safety_policy.checkpoint_on_selection is True
    assert safety_policy.checkpoint_before_head_transition is True
    assert safety_policy.checkpoint_on_signal_safe_boundary is True
    assert safety_policy.checkpoint_at_segment_terminal is True
    assert safety_policy.checkpoint_before_resume is True
    assert safety_policy.to_dict()["checkpoint_before_resume"] is True
    safety_policy.validate_integrity()
    with pytest.raises(ValueError):
        freeze_profile_safety_margin_policy(
            profile_run,
            walltime_margin_fraction=0.0,
            fixed_walltime_margin_seconds=10.0,
            memory_margin_fraction=0.25,
            signal_margin_seconds=5.0,
            durable_checkpoint_cadence_updates=200,
        )
    with pytest.raises(ValueError):
        replace(
            safety_policy,
            signal_margin_seconds=0.0,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    with pytest.raises(ValueError):
        replace(
            safety_policy,
            checkpoint_on_selection=False,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )


def test_scheduler_envelope_binds_raw_evidence_and_hpc_limits(
    envelope: SchedulerResourceEnvelope,
) -> None:
    assert envelope.slurm_account == "sigroup"
    assert envelope.partition == "gpu-l20"
    assert envelope.scheduler_raw_evidence_sha256 == _digest("scheduler-raw")
    assert envelope.resource_raw_evidence_sha256 == _digest("resource-raw")
    envelope.validate_integrity()
    with pytest.raises(ValueError):
        replace(
            envelope,
            scheduler_raw_evidence_sha256="0" * 64,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )


def test_preparation_timings_are_train_only_and_positive(
    profile_run: ValidatedGatePRun,
    preparation: ProfilePreparationTimings,
) -> None:
    assert preparation.heldout_bytes_decoded is False
    assert preparation.source_artifacts_reverified is True
    assert preparation.oracle_rescore_wall_seconds == 0.3
    preparation.validate_integrity()
    with pytest.raises(ValueError):
        record_profile_preparation_timings(
            profile_run,
            artifact_verification_wall_seconds=0.0,
            oracle_rescore_wall_seconds=1.0,
            label_reconstruction_wall_seconds=1.0,
        )
    with pytest.raises(ValueError):
        record_profile_preparation_timings(
            profile_run,
            artifact_verification_wall_seconds=1.0,
            oracle_rescore_wall_seconds=0.0,
            label_reconstruction_wall_seconds=1.0,
        )


def test_input_preparation_timings_flow_into_profile_without_omission(
    profile_run: ValidatedGatePRun,
) -> None:
    payload = phase2_r3_inputs._timing_payload(
        materialization_attestation_sha256=(profile_run.materialization.attestation_sha256),
        artifact_verification_wall_seconds=0.1,
        oracle_rescore_wall_seconds=0.3,
        label_reconstruction_wall_seconds=0.2,
    )
    input_timings = phase2_r3_inputs.R3InputPreparationTimings(
        **payload,
        timings_sha256=phase2_r3_inputs._canonical_sha256(payload),
    )

    preparation = record_profile_preparation_from_train_input(
        profile_run,
        input_timings,
    )

    assert preparation.artifact_verification_wall_seconds == 0.1
    assert preparation.oracle_rescore_wall_seconds == 0.3
    assert preparation.label_reconstruction_wall_seconds == 0.2


def test_real_cpu_context_cannot_be_promoted_as_formal_cuda(
    profile_run: ValidatedGatePRun,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    preparation: ProfilePreparationTimings,
    tmp_path: Path,
) -> None:
    assert profile_run.materialization.context.training.policy_scores.device.type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA-resident"):
        run_formal_gate_p_cuda_profile(
            profile_run,
            safety_policy=safety_policy,
            envelope=envelope,
            preparation=preparation,
            io_probe_directory=tmp_path,
        )


def test_formal_entry_has_no_formal_bool_or_caller_core_payload() -> None:
    parameters = inspect.signature(run_formal_gate_p_cuda_profile).parameters
    assert "formal" not in parameters
    assert "core_profile" not in parameters
    assert list(parameters)[0] == "profile_run"


def test_core_exposes_same_pass_read_only_boundary_probe_interface() -> None:
    parameters = inspect.signature(formal_profile.run_gate_p_profile_core).parameters
    assert "live_boundary_probe" in parameters


def test_target_checkpoint_uses_production_controller_shape_and_dominates_100(
    profile_run: ValidatedGatePRun,
) -> None:
    context = profile_run.materialization.context
    trainer = build_primary_core_trainer(context, "bt_mle")
    trainer.step()
    live_state = trainer.state_dict()
    state_100 = formal_profile._expanded_profile_trainer_state(
        live_state,
        target_update=100,
    )
    state_target = formal_profile._expanded_profile_trainer_state(
        live_state,
        target_update=formal_profile.R3_MAXIMUM_UPDATES_PER_HEAD,
    )
    identity = phase2_training._first_order_controller_identity(
        objective_name="bt_mle",
        execution_role="phase2_recovery_r3_primary",
        spec=context.settings.convergence,
        fixed_snapshot_steps=context.settings.outer_steps,
        rank_diagnostic=context.reward_head_identifiability,
    )
    payload_100 = formal_profile._profile_benchmark_outer_payload(
        learner="bt_mle",
        update=100,
        trainer_state=state_100,
        identity=identity,
        spec=context.settings.convergence,
    )
    payload_target = formal_profile._profile_benchmark_outer_payload(
        learner="bt_mle",
        update=formal_profile.R3_MAXIMUM_UPDATES_PER_HEAD,
        trainer_state=state_target,
        identity=identity,
        spec=context.settings.convergence,
    )
    assert set(payload_target) == formal_profile._PRIMARY_OUTER_PAYLOAD_FIELDS
    controller = payload_target["controller_checkpoint"]
    assert controller["schema_version"] == (formal_profile.PRODUCTION_CONTROLLER_CHECKPOINT_SCHEMA)
    assert set(controller) == formal_profile._CONTROLLER_CHECKPOINT_FIELDS
    controller_state = controller["controller_state"]
    assert set(controller_state) == formal_profile._CONTROLLER_STATE_FIELDS
    assert len(controller_state["checks"]) == 638
    assert set(controller_state["checks"][0]) == (formal_profile._RECOVERY_CONVERGENCE_CHECK_FIELDS)
    assert set(controller_state["fixed_snapshot"]) == (formal_profile._FIXED_SNAPSHOT_FIELDS)
    assert set(controller_state["legacy_boundary_snapshot"]) == (
        formal_profile._LEGACY_SNAPSHOT_FIELDS
    )
    execution = controller_state["optimizer_protocol_execution"]
    assert set(execution) == formal_profile._OPTIMIZER_PROTOCOL_EXECUTION_FIELDS
    assert len(execution["protocol"]["learning_rate_schedule"]["stages"]) == 5
    assert len(execution["boundary_transitions"]) == 4
    assert len(controller_state["recovery_state_check_transcript"]) == 25_520

    target_buffer = io.BytesIO()
    torch.save(payload_target, target_buffer)
    hundred_buffer = io.BytesIO()
    torch.save(payload_100, hundred_buffer)
    assert target_buffer.tell() >= hundred_buffer.tell()


def test_checkpoint_benchmark_measures_fresh_restore_and_terminal_builder(
    profile_run: ValidatedGatePRun,
    tmp_path: Path,
) -> None:
    context = profile_run.materialization.context
    trainer = build_primary_core_trainer(context, "bt_mle")
    trainer.step()
    state = trainer.state_dict()
    identity = phase2_training._first_order_controller_identity(
        objective_name="bt_mle",
        execution_role="phase2_recovery_r3_primary",
        spec=context.settings.convergence,
        fixed_snapshot_steps=context.settings.outer_steps,
        rank_diagnostic=context.reward_head_identifiability,
    )
    sample = formal_profile._benchmark_production_checkpoint_envelope(
        learner="bt_mle",
        update=1,
        trainer_state=state,
        identity=identity,
        spec=context.settings.convergence,
        fresh_trainer_factory=lambda: build_primary_core_trainer(
            context,
            "bt_mle",
        ),
        directory=tmp_path,
    )
    assert sample["fresh_trainer_restore_load_measured"] is True
    assert sample["selected_terminal_production_builder_used"] is True
    assert sample["final_restore_load_terminal_payload_wall_seconds"] > 0
    assert sample["primary_outer_exact_key_contract_verified"] is True
    with pytest.raises(RuntimeError, match="selected-terminal schema"):
        formal_profile._profile_selected_terminal_outer_payload(
            learner="bt_mle",
            selected_terminal_checkpoint={
                "schema_version": "non-production-terminal/v1",
                "terminal_checkpoint_sha256": "f" * 64,
            },
        )


def test_complete_mocked_cuda_profile_promotes_and_builds_exact_artifact_ref(
    formal_result,
    profile_run: ValidatedGatePRun,
    envelope: SchedulerResourceEnvelope,
) -> None:
    validated = validate_formal_cuda_profile_result(formal_result)
    assert validated.profile_run is profile_run
    assert validated.core_profile["formal_cuda_profile"] is True
    assert validated.core_profile["update_cap_per_learner"] == 100
    assert validated.stop_reason == PHASE2_PROFILE_STOP_REASON
    assert validated.information_boundary["train_only"] is True
    assert validated.scheduler_raw_evidence_sha256 == (envelope.scheduler_raw_evidence_sha256)
    assert len(validated.gpu_utilization_samples) == 2
    assert validated.cpu_memory["peak_rss_bytes"] == 1024 * 1024**2
    production_io = validated.production_checkpoint_io_evidence
    assert production_io["sample_updates"] == [
        0,
        20,
        40,
        60,
        80,
        100,
        12_760,
    ]
    assert (
        production_io["learners"][0]["target_serialized_bytes_upper_bound"]
        == production_io["learners"][0]["samples"][-1]["serialized_bytes"]
    )
    ref = formal_cuda_profile_artifact_ref(validated)
    assert ref.schema_version == FORMAL_CUDA_PROFILE_RESULT_SCHEMA
    assert ref.role == FORMAL_CUDA_PROFILE_RESULT_ROLE
    assert ref.artifact_sha256 == validated.formal_profile_sha256


def test_formal_result_replace_and_nested_core_tamper_fail_closed(
    formal_result,
) -> None:
    with pytest.raises(ValueError):
        replace(
            formal_result,
            formal_profile_sha256="1" * 64,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    with pytest.raises(ValueError):
        replace(
            formal_result,
            scheduler_raw_evidence_sha256="2" * 64,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    production_io = copy.deepcopy(formal_result.production_checkpoint_io_evidence)
    production_io["learners"][0]["samples"][1]["serialized_bytes"] += 1
    with pytest.raises(ValueError):
        replace(
            formal_result,
            production_checkpoint_io_evidence=production_io,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    production_io = copy.deepcopy(formal_result.production_checkpoint_io_evidence)
    production_io["learners"][0]["samples"][-1]["fresh_trainer_restore_load_measured"] = False
    production_io.pop("evidence_sha256")
    production_io["evidence_sha256"] = formal_profile._canonical_sha256(production_io)
    with pytest.raises(ValueError, match="omitted required outer state"):
        replace(
            formal_result,
            production_checkpoint_io_evidence=production_io,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    core = copy.deepcopy(formal_result.core_profile)
    core["learners"][0]["steps"][0]["wall_seconds"] = 99.0
    core = _rehash_core(core)
    with pytest.raises(ValueError):
        replace(
            formal_result,
            core_profile=core,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )


@pytest.mark.parametrize("failure_kind", ["cpu_flag", "zero_time", "system_tmp", "pcg"])
def test_incomplete_or_non_cuda_core_cannot_be_promoted(
    failure_kind: str,
    profile_run: ValidatedGatePRun,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    preparation: ProfilePreparationTimings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _core_profile(profile_run)
    if failure_kind == "cpu_flag":
        core["device_type"] = "cpu"
        core["formal_cuda_profile"] = False
        core["setup"]["cuda_memory"] = {
            "measurement": "nonformal_cpu",
            "current_bytes": None,
            "peak_bytes": None,
        }
        for learner in core["learners"]:
            for step in learner["steps"]:
                step["cuda_memory"] = {
                    "measurement": "nonformal_cpu",
                    "current_bytes": None,
                    "peak_bytes": None,
                }
    elif failure_kind == "zero_time":
        core["learners"][0]["steps"][0]["wall_seconds"] = 0.0
    elif failure_kind == "system_tmp":
        core["learners"][0]["ephemeral_checkpoint_io"][0]["filesystem_scope"] = (
            "system_temporary_directory"
        )
    else:
        del core["learners"][1]["steps"][0]["pcg"]["reason"]
    core = _rehash_core(core)
    _patch_formal_runtime(monkeypatch, core)

    with pytest.raises((TypeError, ValueError)):
        run_formal_gate_p_cuda_profile(
            profile_run,
            safety_policy=safety_policy,
            envelope=envelope,
            preparation=preparation,
            io_probe_directory=tmp_path,
        )


def test_live_cuda_identity_must_match_resource_envelope(
    profile_run: ValidatedGatePRun,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    preparation: ProfilePreparationTimings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _cuda_identity()
    identity["name"] = "Different GPU"
    monkeypatch.setattr(formal_profile, "_require_live_cuda", lambda _run: identity)
    with pytest.raises(ValueError, match="resource envelope"):
        run_formal_gate_p_cuda_profile(
            profile_run,
            safety_policy=safety_policy,
            envelope=envelope,
            preparation=preparation,
            io_probe_directory=tmp_path,
        )


def test_resource_projection_is_transparent_and_covers_both_12760_heads(
    formal_result,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
) -> None:
    plan = build_gate_p_resource_plan(
        formal_result,
        safety_policy=safety_policy,
        envelope=envelope,
        requested_walltime_seconds_per_segment=300,
        array_concurrency=2,
        cpus_per_task=4,
        memory_bytes=2 * 1024**3,
    )
    assert type(plan) is GatePResourcePlan
    assert plan.audit_cadence_updates == 20
    assert plan.durable_checkpoint_cadence_updates == 200
    assert plan.advance_signal_lead_seconds > 0
    assert plan.max_scheduler_segments == 3
    assert plan.total_effective_capacity_seconds >= (plan.projected_required_wall_seconds)
    learners = plan.projection["learners"]
    assert plan.projection["setup_wall_seconds"] == pytest.approx(0.651)
    assert [learner["target_updates"] for learner in learners] == [12_760, 12_760]
    assert [learner["target_audits"] for learner in learners] == [639, 639]
    assert [learner["base_target_checkpoint_events"] for learner in learners] == [
        68,
        68,
    ]
    assert [learner["base_target_checkpoint_events"] for learner in learners] != [
        639,
        639,
    ]
    assert [learner["selection_checkpoint_events"] for learner in learners] == [1, 1]
    assert [learner["head_transition_checkpoint_events"] for learner in learners] == [1, 0]
    assert [learner["target_checkpoint_events_before_segmentation"] for learner in learners] == [
        70,
        69,
    ]
    assert all(
        learner["target_serialized_bytes_upper_bound"]
        == learner["production_checkpoint_io_samples"][-1]["serialized_bytes"]
        for learner in learners
    )
    assert all(
        learner["projection_rule"].startswith("sum_each_event_using_next_measured")
        for learner in learners
    )
    assert plan.checkpoint_on_selection is True
    assert plan.checkpoint_before_head_transition is True
    assert plan.checkpoint_on_signal_safe_boundary is True
    assert plan.checkpoint_at_segment_terminal is True
    assert plan.checkpoint_before_resume is True
    assert plan.checkpoint_event_projection == {
        "cadence_and_learning_rate_events": 136,
        "selection_events": 2,
        "head_transition_events": 1,
        "signal_safe_boundary_events": 3,
        "segment_terminal_events": 3,
        "before_resume_events": 2,
        "total_events": 147,
        "coalescing_credit_assumed": False,
    }
    assert plan.projected_head_agnostic_block_pricing_reserve_seconds > 0
    assert plan.projected_early_head_transition_reserve_seconds > 0
    assert plan.segment_execution_contract["every_transferable_block_is_head_agnostic"] is True
    assert plan.projected_signal_safe_boundary_checkpoint_overhead_seconds > 0
    assert plan.projected_before_resume_checkpoint_overhead_seconds > 0
    assert plan.segment_boundaries[0]["start_boundary"]["global_safe_block"] == 0
    assert (
        plan.segment_boundaries[-1]["end_boundary"]["global_safe_block"]
        == formal_profile.R3_TOTAL_SAFE_UPDATE_BLOCKS
    )
    for previous, current in zip(
        plan.segment_boundaries,
        plan.segment_boundaries[1:],
        strict=False,
    ):
        assert previous["end_boundary"] == current["start_boundary"]
        assert previous["continuation_required"] is True
        assert previous["projected_before_resume_checkpoint_events"] == 1
    assert (
        sum(segment["max_safe_update_blocks_to_execute"] for segment in plan.segment_boundaries)
        == formal_profile.R3_TOTAL_SAFE_UPDATE_BLOCKS
    )
    assert all(
        segment["fixed_ordered_head_transition_allowed"] == ["bt_mle", "prorm_plus"]
        and segment["head_agnostic_block_prices"] is True
        and segment["safe_block_pricing_rule"]
        == "pairwise_max_bt_mle_prorm_plus_by_local_block_index"
        and segment["journal_actual_cursor_required"] is True
        and segment["nominal_boundaries_are_worst_case_projection_only"] is True
        and segment["actual_cursor_must_reach_nominal_end"] is False
        for segment in plan.segment_boundaries
    )
    assert (
        plan.segment_execution_contract["early_convergence_transition_within_segment_allowed"]
        is True
    )
    assert plan.segment_execution_contract["per_segment_block_budget_must_not_be_exceeded"] is True
    assert plan.segment_boundaries[-1]["continuation_required"] is False
    assert plan.segment_boundaries[-1]["projected_before_resume_checkpoint_events"] == 0
    projection = plan.projection
    serial_signal_path = math.fsum(
        (
            projection["maximum_measured_update_wall_seconds"],
            projection["maximum_measured_audit_wall_seconds"],
            projection["maximum_finalization_noncheckpoint_wall_seconds"],
            projection["maximum_projected_safe_boundary_evidence_chain_wall_seconds"],
            safety_policy.signal_margin_seconds,
        )
    )
    old_unsafe_parallelized_path = math.fsum(
        (
            max(
                projection["maximum_measured_update_wall_seconds"],
                projection["maximum_measured_audit_wall_seconds"],
                projection["maximum_finalization_noncheckpoint_wall_seconds"],
            ),
            projection["maximum_projected_safe_boundary_evidence_chain_wall_seconds"],
            safety_policy.signal_margin_seconds,
        )
    )
    assert serial_signal_path > old_unsafe_parallelized_path
    assert plan.advance_signal_lead_seconds == math.ceil(serial_signal_path)
    assert projection["advance_signal_lead_formula"] == (
        "ceil(max_update+max_audit+max_finalization_noncheckpoint+"
        "max_safe_boundary_evidence_chain+signal_margin)"
    )
    validate_gate_p_resource_plan(plan)
    ref = resource_plan_artifact_ref(plan)
    assert ref.schema_version == RESOURCE_PLAN_SCHEMA
    assert ref.role == RESOURCE_PLAN_ROLE
    assert ref.artifact_sha256 == plan.resource_plan_sha256


def test_resource_plan_artifacts_are_accepted_by_identity_authorization(
    sealed_operational_bundle,
    successful_profile_terminal: (terminal_evidence.SuccessfulProfileTerminalCapability),
) -> None:
    bundle, formal_result, plan = sealed_operational_bundle
    authorization = authorize_gate_p(
        operational_bundle=bundle,
        successful_terminal=successful_profile_terminal,
    )
    assert authorization.formal_cuda_profile_result.artifact_sha256 == (
        formal_result.formal_profile_sha256
    )
    assert authorization.resource_plan.artifact_sha256 == plan.resource_plan_sha256


@pytest.mark.parametrize(
    "overrides",
    [
        {"requested_walltime_seconds_per_segment": 301},
        {"requested_walltime_seconds_per_segment": 5},
        {"array_concurrency": 4},
        {"cpus_per_task": 65},
        {"memory_bytes": 1024},
    ],
)
def test_resource_plan_fails_closed_when_limits_do_not_cover_projection(
    overrides: dict[str, int],
    formal_result,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
) -> None:
    arguments = {
        "requested_walltime_seconds_per_segment": 300,
        "array_concurrency": 2,
        "cpus_per_task": 4,
        "memory_bytes": 2 * 1024**3,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        build_gate_p_resource_plan(
            formal_result,
            safety_policy=safety_policy,
            envelope=envelope,
            **arguments,
        )


def test_resource_plan_replace_tamper_fails_closed(
    formal_result,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
) -> None:
    plan = build_gate_p_resource_plan(
        formal_result,
        safety_policy=safety_policy,
        envelope=envelope,
        requested_walltime_seconds_per_segment=300,
        array_concurrency=2,
        cpus_per_task=4,
        memory_bytes=2 * 1024**3,
    )
    with pytest.raises(ValueError):
        replace(
            plan,
            advance_signal_lead_seconds=plan.advance_signal_lead_seconds + 1,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    with pytest.raises(ValueError):
        replace(
            plan,
            max_scheduler_segments=plan.max_scheduler_segments + 1,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    with pytest.raises(ValueError):
        replace(
            plan,
            checkpoint_before_resume=False,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    tampered_counts = dict(plan.checkpoint_event_projection)
    tampered_counts["before_resume_events"] = 0
    with pytest.raises(ValueError):
        replace(
            plan,
            checkpoint_event_projection=tampered_counts,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    tampered_contract = dict(plan.segment_execution_contract)
    tampered_contract["actual_cursor_must_reach_nominal_end"] = True
    with pytest.raises(ValueError):
        replace(
            plan,
            segment_execution_contract=tampered_contract,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )
    with pytest.raises(ValueError):
        replace(
            plan,
            resource_plan_sha256="3" * 64,
            _factory_token=formal_profile._FACTORY_TOKEN,
        )


def test_profile_authority_seals_are_not_inherited_by_dataclass_replace(
    formal_result,
    safety_policy: ProfileSafetyMarginPolicy,
    envelope: SchedulerResourceEnvelope,
    preparation: ProfilePreparationTimings,
) -> None:
    plan = build_gate_p_resource_plan(
        formal_result,
        safety_policy=safety_policy,
        envelope=envelope,
        requested_walltime_seconds_per_segment=300,
        array_concurrency=2,
        cpus_per_task=4,
        memory_bytes=2 * 1024**3,
    )

    for authority in (
        safety_policy,
        envelope,
        preparation,
        formal_result,
        plan,
    ):
        with pytest.raises(TypeError, match="validating factory"):
            replace(authority)


def test_claim_free_core_binding_remains_distinct_from_formal_schema(
    profile_run: ValidatedGatePRun,
) -> None:
    binding = profile_core_binding(profile_run.materialization.context)
    assert binding["schema_version"] == PHASE2_PROFILE_BINDING_SCHEMA
    assert binding["schema_version"] != FORMAL_CUDA_PROFILE_RESULT_SCHEMA
    assert binding["profile_nonreusable"] is True


def _write_operational_payload(
    path: Path,
    payload: dict[str, object],
) -> str:
    raw = canonical_json_bytes(payload)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _rehash_operational_bundle(payload: dict[str, object]) -> None:
    unhashed = dict(payload)
    del unhashed["bundle_semantic_sha256"]
    payload["bundle_semantic_sha256"] = formal_profile._canonical_sha256(unhashed)


def test_sealed_operational_bundle_exposes_only_verified_read_only_evidence(
    sealed_operational_bundle,
) -> None:
    bundle, result, plan = sealed_operational_bundle
    bundle.validate_integrity()

    assert bundle.profile_run_sha256 == result.profile_run.profile_run_sha256
    assert bundle.formal_profile_sha256 == result.formal_profile_sha256
    assert bundle.resource_plan_sha256 == plan.resource_plan_sha256
    assert bundle.to_dict()["formal_cuda_profile_result"] == result.to_dict()
    assert bundle.to_dict()["resource_plan"] == plan.to_dict()
    assert bundle.to_dict()["resource_plan"]["projection"] == plan.projection
    assert bundle.slurm_account == plan.slurm_account
    assert bundle.partition == plan.partition
    assert bundle.gpu_name == plan.gpu_name
    assert bundle.gpus_per_task == plan.gpus_per_task
    assert bundle.cpus_per_task == plan.cpus_per_task
    assert bundle.memory_bytes == plan.memory_bytes
    assert (
        bundle.requested_walltime_seconds_per_segment == plan.requested_walltime_seconds_per_segment
    )
    assert bundle.checkpoint_on_selection is True
    assert bundle.checkpoint_before_head_transition is True
    assert bundle.checkpoint_on_signal_safe_boundary is True
    assert bundle.checkpoint_at_segment_terminal is True
    assert bundle.checkpoint_before_resume is True
    assert bundle.to_dict()["resource_plan"]["segment_boundaries"] == list(plan.segment_boundaries)
    assert not hasattr(bundle, "authorized")
    assert not hasattr(bundle, "authorization_sha256")
    with pytest.raises(TypeError):
        bundle.resource_plan["partition"] = "cpu"  # type: ignore[index]

    formal_ref = profile_artifacts.formal_profile_artifact_ref(bundle)
    plan_ref = profile_artifacts.resource_plan_artifact_ref(bundle)
    assert formal_ref.to_dict() == {
        "schema_version": FORMAL_CUDA_PROFILE_RESULT_SCHEMA,
        "artifact_sha256": result.formal_profile_sha256,
        "role": FORMAL_CUDA_PROFILE_RESULT_ROLE,
    }
    assert plan_ref.to_dict() == {
        "schema_version": RESOURCE_PLAN_SCHEMA,
        "artifact_sha256": plan.resource_plan_sha256,
        "role": RESOURCE_PLAN_ROLE,
    }


def test_sealed_bundle_factory_token_cannot_be_inherited_by_replace(
    sealed_operational_bundle,
) -> None:
    bundle, _, _ = sealed_operational_bundle
    invalid_seal = copy.copy(bundle)
    object.__setattr__(invalid_seal, "_seal", object())
    with pytest.raises(TypeError, match="factory seal"):
        invalid_seal.validate_integrity()

    with pytest.raises(ValueError, match="InitVar"):
        replace(bundle)

    payload = bundle.to_dict()
    payload["role"] = "forged_operational_bundle"
    _rehash_operational_bundle(payload)
    canonical = canonical_json_bytes(payload)
    with pytest.raises(ValueError, match="InitVar"):
        replace(
            bundle,
            bundle_semantic_sha256=str(payload["bundle_semantic_sha256"]),
            file_sha256=hashlib.sha256(canonical).hexdigest(),
            size_bytes=len(canonical),
            _canonical_bytes=canonical,
        )


def test_generic_artifact_refs_cannot_construct_or_issue_sealed_refs(
    sealed_operational_bundle,
) -> None:
    bundle, result, plan = sealed_operational_bundle
    generic_formal = formal_cuda_profile_artifact_ref(result)
    generic_plan = resource_plan_artifact_ref(plan)
    with pytest.raises(TypeError, match="VerifiedGatePOperationalBundle"):
        profile_artifacts.formal_profile_artifact_ref(generic_formal)
    with pytest.raises(TypeError, match="VerifiedGatePOperationalBundle"):
        profile_artifacts.resource_plan_artifact_ref(generic_plan)
    with pytest.raises(TypeError, match="produced by reopen"):
        profile_artifacts.VerifiedGatePOperationalBundle(
            artifact_path=bundle.artifact_path,
            file_sha256=bundle.file_sha256,
            size_bytes=bundle.size_bytes,
            profile_run_sha256=bundle.profile_run_sha256,
            formal_profile_sha256=bundle.formal_profile_sha256,
            resource_plan_sha256=bundle.resource_plan_sha256,
            bundle_semantic_sha256=bundle.bundle_semantic_sha256,
            _canonical_bytes=canonical_json_bytes(bundle.to_dict()),
            _factory_token=None,
        )


def test_operational_bundle_reopen_is_context_oracle_and_cuda_independent(
    sealed_operational_bundle,
    profile_run: ValidatedGatePRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _, plan = sealed_operational_bundle
    expected_projection = copy.deepcopy(plan.to_dict()["projection"])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure-data reopen touched a forbidden live path")

    monkeypatch.setattr(
        type(profile_run.materialization),
        "validate_integrity",
        forbidden,
    )
    monkeypatch.setattr(
        type(profile_run.materialization.context),
        "validate_integrity",
        forbidden,
    )
    monkeypatch.setattr(formal_profile, "_validated_run", forbidden)
    monkeypatch.setattr(formal_profile, "_require_live_cuda", forbidden)
    monkeypatch.setattr(formal_profile, "_sample_gpu_utilization", forbidden)
    monkeypatch.setattr(
        formal_profile,
        "_run_core_with_production_checkpoint_io",
        forbidden,
    )
    monkeypatch.setattr(
        formal_profile,
        "run_formal_gate_p_cuda_profile",
        forbidden,
    )
    monkeypatch.setattr(formal_profile, "build_gate_p_resource_plan", forbidden)
    monkeypatch.setattr(
        phase2_training,
        "_validate_frozen_oracle_rewards",
        forbidden,
    )
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "memory_allocated", forbidden)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", forbidden)

    reopened = profile_artifacts.reopen_verified_gate_p_operational_bundle(
        bundle.artifact_path,
        expected_file_sha256=bundle.file_sha256,
    )
    assert reopened.to_dict()["resource_plan"]["projection"] == expected_projection

    copied = (tmp_path / "standalone-operational-bundle.json").resolve()
    copied_sha = _write_operational_payload(copied, bundle.to_dict())
    standalone = profile_artifacts.reopen_verified_gate_p_operational_bundle(
        copied,
        expected_file_sha256=copied_sha,
    )
    assert standalone.resource_plan_sha256 == plan.resource_plan_sha256


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":"x","schema_version":"y"}\n',
        b'{ "schema_version": "x" }\n',
    ],
)
def test_operational_bundle_rejects_duplicate_or_noncanonical_transport(
    raw: bytes,
    tmp_path: Path,
) -> None:
    path = (tmp_path / f"invalid-{hashlib.sha256(raw).hexdigest()}.json").resolve()
    path.write_bytes(raw)
    with pytest.raises((TypeError, ValueError)):
        profile_artifacts.reopen_verified_gate_p_operational_bundle(
            path,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_operational_bundle_rejects_file_nested_and_formal_sha_tampering(
    sealed_operational_bundle,
    tmp_path: Path,
) -> None:
    bundle, _, _ = sealed_operational_bundle
    with pytest.raises(ValueError, match="does not match"):
        profile_artifacts.reopen_verified_gate_p_operational_bundle(
            bundle.artifact_path,
            expected_file_sha256="f" * 64,
        )

    nested = bundle.to_dict()
    nested["safety_margin_policy"]["policy_sha256"] = "e" * 64
    _rehash_operational_bundle(nested)
    nested_path = (tmp_path / "nested-sha-tamper.json").resolve()
    nested_file_sha = _write_operational_payload(nested_path, nested)
    with pytest.raises(ValueError, match="safety policy semantic"):
        profile_artifacts.reopen_verified_gate_p_operational_bundle(
            nested_path,
            expected_file_sha256=nested_file_sha,
        )

    formal = bundle.to_dict()
    formal["formal_cuda_profile_result"]["formal_profile_sha256"] = "d" * 64
    _rehash_operational_bundle(formal)
    formal_path = (tmp_path / "formal-sha-tamper.json").resolve()
    formal_file_sha = _write_operational_payload(formal_path, formal)
    with pytest.raises(ValueError, match="formal profile semantic"):
        profile_artifacts.reopen_verified_gate_p_operational_bundle(
            formal_path,
            expected_file_sha256=formal_file_sha,
        )


def test_operational_bundle_recomputes_projection_and_rejects_mixed_dependencies(
    sealed_operational_bundle,
    tmp_path: Path,
) -> None:
    bundle, _, _ = sealed_operational_bundle

    projection = bundle.to_dict()
    resource_plan = projection["resource_plan"]
    resource_plan["projection"]["required_cpu_memory_bytes"] += 1
    unhashed_plan = dict(resource_plan)
    del unhashed_plan["resource_plan_sha256"]
    resource_plan["resource_plan_sha256"] = formal_profile._canonical_sha256(unhashed_plan)
    _rehash_operational_bundle(projection)
    projection_path = (tmp_path / "projection-tamper.json").resolve()
    projection_file_sha = _write_operational_payload(
        projection_path,
        projection,
    )
    with pytest.raises(ValueError, match="pure-data projection"):
        profile_artifacts.reopen_verified_gate_p_operational_bundle(
            projection_path,
            expected_file_sha256=projection_file_sha,
        )

    mixed = bundle.to_dict()
    safety = mixed["safety_margin_policy"]
    safety["profile_run_sha256"] = _digest("another-profile-run")
    unhashed_safety = dict(safety)
    del unhashed_safety["policy_sha256"]
    safety["policy_sha256"] = formal_profile._canonical_sha256(unhashed_safety)
    _rehash_operational_bundle(mixed)
    mixed_path = (tmp_path / "mixed-dependencies.json").resolve()
    mixed_file_sha = _write_operational_payload(mixed_path, mixed)
    with pytest.raises(ValueError, match="another profile run"):
        profile_artifacts.reopen_verified_gate_p_operational_bundle(
            mixed_path,
            expected_file_sha256=mixed_file_sha,
        )
