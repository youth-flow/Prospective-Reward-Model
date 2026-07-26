from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from smart_reward import phase2_r3_orchestrator as orchestrator
from smart_reward import phase2_r3_profile as formal_profile
from smart_reward import phase2_r3_profile_artifacts as profile_artifacts
from smart_reward.experiment import TrainingTensorData
from smart_reward.phase2_checkpoint import CheckpointInterruption, CheckpointSignal
from smart_reward.phase2_r3_identity import (
    VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
    ArtifactRef,
    admit_primary_segment,
    authorize_gate_p,
    create_r3_primary_design,
    validate_continuation_evidence,
)
from smart_reward.phase2_r3_primary import capture_slurm_segment_runtime
from tests import conftest as r3_test_support
from tests import test_phase2_r3_profile as profile_test_support

ROOT = Path(__file__).resolve().parents[1]
SCIENCE_PATH = ROOT / "configs" / "phase2_recovery_r3_science.yaml"


@pytest.mark.skipif(os.name != "posix", reason="requires real POSIX directory modes")
def test_orchestrator_nested_directories_are_writable_0750_with_optional_setgid(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "orchestrator-modes").resolve()
    root.mkdir(mode=0o750)
    os.chmod(root, 0o750)
    heads = orchestrator._ensure_child_directory(root, "heads")
    learner = orchestrator._ensure_child_directory(heads, "prorm_plus")
    probe = learner / "no-overwrite-probe"
    with probe.open("xb") as stream:
        stream.write(b"writable\n")
    assert probe.read_bytes() == b"writable\n"
    assert stat.S_IMODE(heads.stat().st_mode) == 0o750
    assert stat.S_IMODE(learner.stat().st_mode) == 0o750

    retained_bad = root / "retained-bad"
    retained_bad.mkdir(mode=0o550)
    os.chmod(retained_bad, 0o550)
    with pytest.raises(ValueError, match="mode 0750"):
        orchestrator._ensure_child_directory(root, retained_bad.name)

    retained_setgid = root / "retained-setgid"
    retained_setgid.mkdir(mode=0o750)
    os.chmod(retained_setgid, 0o2750)
    assert orchestrator._ensure_child_directory(root, retained_setgid.name) == retained_setgid


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
        prompt_ids=tuple(f"r3-orchestrator-{index}" for index in range(4)),
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


def _segment_and_policy(
    *,
    checkpoint_multiplier: int = 2,
    resource_plan_token: str = "resource-plan",
    split_after_bt: bool = False,
):
    del checkpoint_multiplier, resource_plan_token, split_after_bt
    evidence = r3_test_support.make_shared_gate_p_evidence()
    science = evidence.science
    bundle = evidence.operational_bundle
    authorization = authorize_gate_p(
        operational_bundle=bundle,
        successful_terminal=evidence.successful_terminal,
    )
    design = create_r3_primary_design(
        science=science,
        gate0_capability=evidence.gate0_capability,
        gate1_capabilities=evidence.gate1_capabilities,
        profile_authorization=authorization,
        operational_bundle=bundle,
    )
    segment = admit_primary_segment(
        design=design,
        materialization_capability=evidence.materialization_capability,
        task_id=0,
        seed=20260801,
        segment_index=1,
    )
    return segment, bundle


def _segment_for_operational_bundle(bundle):
    evidence = r3_test_support.make_shared_gate_p_evidence()
    terminal = evidence.successful_terminal
    if bundle.file_sha256 != evidence.operational_bundle.file_sha256:
        terminal = profile_test_support._successful_profile_terminal(
            bundle,
            r3_test_support._PersistentTempPathFactory().mktemp("orchestrator-profile-terminal"),
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


def _independent_operational_bundle():
    evidence = r3_test_support.make_shared_gate_p_evidence()
    formal_result = evidence.formal_result
    original_plan = evidence.resource_plan
    array_concurrency = 1 if original_plan.array_concurrency != 1 else 2
    independent_plan = formal_profile.build_gate_p_resource_plan(
        formal_result,
        safety_policy=formal_result.safety_policy,
        envelope=formal_result.envelope,
        requested_walltime_seconds_per_segment=(
            original_plan.requested_walltime_seconds_per_segment
        ),
        array_concurrency=array_concurrency,
        cpus_per_task=original_plan.cpus_per_task,
        memory_bytes=original_plan.memory_bytes,
    )
    directory = r3_test_support._PersistentTempPathFactory().mktemp(
        "independent-orchestrator-operational-bundle"
    )
    bundle = profile_artifacts.publish_verified_gate_p_operational_bundle(
        (directory / "gate-p-operational-bundle.json").resolve(),
        profile_run=formal_result.profile_run,
        safety_policy=formal_result.safety_policy,
        envelope=formal_result.envelope,
        formal_result=formal_result,
        resource_plan=independent_plan,
    )
    assert bundle.resource_plan_sha256 != evidence.operational_bundle.resource_plan_sha256
    assert bundle.bundle_semantic_sha256 != evidence.operational_bundle.bundle_semantic_sha256
    return bundle


def _slurm_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_id: str = "300001",
) -> None:
    values = {
        "SLURM_CLUSTER_NAME": "hpc4",
        "SLURM_JOB_ID": job_id,
        "SLURM_ARRAY_JOB_ID": "300000",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ACCOUNT": "sigroup",
        "SLURM_JOB_PARTITION": "gpu-l20",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _runtime(segment, monkeypatch: pytest.MonkeyPatch, *, job_id: str = "300001"):
    _slurm_environment(monkeypatch, job_id=job_id)
    bundle = r3_test_support.make_shared_gate_p_evidence().operational_bundle
    return capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=bundle.requested_walltime_seconds_per_segment,
    )


def _checkpoint_ref(learner: str) -> ArtifactRef:
    return _ref(
        VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
        VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
        f"checkpoint-{learner}",
    )


class _FakeFormalResult:
    def __init__(
        self,
        admission,
        runtime,
        learner: str,
        head_execution_slice: dict[str, object],
    ) -> None:
        self.admission = admission
        self.runtime = runtime
        self.learner = learner
        self.terminal_checkpoint_artifact_sha256 = _checkpoint_ref(learner).artifact_sha256
        context = admission.materialization.context
        start = int(head_execution_slice["start_completed_updates"])
        cadence = int(head_execution_slice["science_audit_cadence_updates"])
        cap = int(head_execution_slice["end_completed_updates_inclusive"])
        completed_updates = min(cap, start + 5 * cadence)
        payload: dict[str, object] = {
            "schema_version": orchestrator.FORMAL_PRIMARY_HEAD_RESULT_SCHEMA,
            "campaign_kind": admission.design.campaign_kind,
            "execution_revision": admission.design.execution_revision,
            "campaign_role": admission.design.campaign_role,
            "execution_role": "phase2_recovery_r3_primary",
            "design_sha256": admission.design.design_sha256,
            "admission_sha256": admission.admission_sha256,
            "logical_run_id": admission.logical_run_id,
            "head_run_id": admission.head_run_ids[("bt_mle", "prorm_plus").index(learner)],
            "scheduler_segment_id": admission.scheduler_segment_id,
            "runtime_sha256": runtime.runtime_sha256,
            "segment_index": admission.segment_index,
            "task_id": admission.task_id,
            "seed": admission.seed,
            "learner": learner,
            "science_semantic_sha256": admission.design.science.semantic_sha256,
            "science_file_sha256": admission.design.science.file_sha256,
            "materialization_attestation_sha256": (admission.materialization.attestation_sha256),
            "context_sha256": context.context_sha256,
            "input_training_sha256": context.input_training_sha256,
            "prepared_training_sha256": context.primary_training_sha256,
            "oracle_reward_sha256": context.oracle_reward_sha256,
            "label_stream_sha256": context.label_stream.label_stream_sha256,
            "selected_primary_step": completed_updates,
            "controller_updates_executed": completed_updates,
            "head_execution_slice_sha256": head_execution_slice["slice_sha256"],
            "head": {
                "test_only_non_cuda_placeholder": learner,
                "not_a_formal_cuda_claim": True,
            },
            "terminal_checkpoint_artifact_sha256": (self.terminal_checkpoint_artifact_sha256),
            "resumed_from_predecessor": (head_execution_slice["fresh_or_resume"] == "resume"),
            "information_boundary": {
                "train_only": True,
                "validation_or_test_data_accessed": False,
                "policy_session_opened": False,
                "policy_rollout_performed": False,
                "beta_outcome_computed": False,
                "controls_executed": False,
            },
            "external_scheduler_terminal_validated": False,
            "formal_r3_evidence": False,
        }
        self.result_sha256 = orchestrator._canonical_sha256(payload)
        self._payload = {**payload, "result_sha256": self.result_sha256}

    def validate_integrity(self) -> None:
        unsigned = dict(self._payload)
        observed = unsigned.pop("result_sha256")
        if observed != orchestrator._canonical_sha256(unsigned):
            raise ValueError("fake result was tampered")

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(self._payload))


def _patch_non_cuda_formal_boundary(
    monkeypatch: pytest.MonkeyPatch,
    runner,
) -> None:
    # These tests exercise only orchestration.  They replace the exact formal
    # runner/result type and continuation auditor rather than pretending a CPU
    # tensor context is CUDA evidence.
    monkeypatch.setattr(
        orchestrator,
        "FormalR3PrimaryHeadResult",
        _FakeFormalResult,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_formal_r3_primary_head_segment",
        runner,
    )
    bundle_type = type(r3_test_support.make_shared_gate_p_evidence().operational_bundle)
    sealed_boundaries = bundle_type.segment_boundaries.fget
    assert sealed_boundaries is not None

    def exact_dict_boundaries(bundle):
        return tuple(
            {
                **dict(boundary),
                "start_boundary": dict(boundary["start_boundary"]),
                "end_boundary": dict(boundary["end_boundary"]),
            }
            for boundary in sealed_boundaries(bundle)
        )

    # The sealed accessor intentionally exposes immutable Mapping values while
    # the orchestration cursor parser requires exact dicts.  Preserve every
    # sealed value and change only that representation at this non-CUDA test
    # boundary; all production identity, policy, and cursor validation still runs.
    monkeypatch.setattr(
        bundle_type,
        "segment_boundaries",
        property(exact_dict_boundaries),
    )
    monkeypatch.setattr(
        orchestrator,
        "_terminal_checkpoint_artifact_sha256",
        lambda store, *, admission, learner: _checkpoint_ref(learner).artifact_sha256,
    )
    monkeypatch.setattr(
        orchestrator,
        "continuation_checkpoint_artifact_ref",
        lambda store, *, predecessor, learner: _checkpoint_ref(learner),
    )


def _task_root(tmp_path: Path) -> Path:
    root = (tmp_path / "task").resolve()
    root.mkdir()
    return root


def _journal_events(task_root: Path) -> list[dict[str, Any]]:
    paths = sorted((task_root / "task-journal").glob("event-*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_fixed_order_independent_stores_and_head_free_complete_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, policy = _segment_and_policy(split_after_bt=True)
    runtime = _runtime(segment, monkeypatch)
    calls: list[tuple[str, Path, object, dict[str, object]]] = []

    def runner(admission, learner, **kwargs):
        calls.append(
            (
                learner,
                kwargs["checkpoint_store"].root,
                kwargs["resource_plan"],
                kwargs["head_execution_slice"],
            )
        )
        return _FakeFormalResult(
            admission,
            kwargs["runtime"],
            learner,
            kwargs["head_execution_slice"],
        )

    _patch_non_cuda_formal_boundary(monkeypatch, runner)
    root = _task_root(tmp_path)
    outcome = orchestrator.run_r3_primary_task_segment(
        segment,
        runtime=runtime,
        task_root=root,
        checkpoint_signal=CheckpointSignal(),
        operational_policy=policy,
    )

    assert [item[0] for item in calls] == ["bt_mle", "prorm_plus"]
    assert calls[0][1] != calls[1][1]
    assert all(item[2] is policy for item in calls)
    assert [item[3]["fresh_or_resume"] for item in calls] == [
        "fresh",
        "fresh",
    ]
    assert calls[0][3]["safe_update_blocks_consumed_before_head"] == 0
    assert calls[1][3]["safe_update_blocks_consumed_before_head"] > 0
    assert calls[0][3]["nominal_segment_start_cursor_sha256"]
    assert calls[0][3]["nominal_segment_end_cursor_sha256"]
    for item in calls:
        execution_slice = dict(item[3])
        observed = execution_slice.pop("slice_sha256")
        assert observed == orchestrator._canonical_sha256(execution_slice)
    assert (
        policy.durable_checkpoint_cadence_updates
        % segment.design.science.settings.convergence.check_interval
        == 0
    )
    assert outcome.status == "compute_complete_pending_external_scheduler_terminal"
    assert outcome.external_scheduler_success_claimed is False
    assert outcome.continuation_checkpoint is None
    outcome_payload = json.loads(outcome.artifact_path.read_text(encoding="utf-8"))
    assert outcome_payload["external_scheduler_success_claimed"] is False
    assert outcome_payload["r3_success_authorization_created"] is False
    assert "head" not in outcome_payload
    assert [event["event_type"] for event in _journal_events(root)] == [
        "head_started",
        "head_completed",
        "head_started",
        "head_completed",
        "segment_compute_complete",
    ]

    for learner in ("bt_mle", "prorm_plus"):
        result_path = root / "heads" / learner / "internal-head-result.json"
        receipt_path = root / "heads" / learner / "head-completion.json"
        result_raw = result_path.read_bytes()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["internal_result"]["file_sha256"] == hashlib.sha256(result_raw).hexdigest()
        assert (
            receipt["terminal_checkpoint_artifact_sha256"]
            == _checkpoint_ref(learner).artifact_sha256
        )
        unsigned = dict(receipt)
        observed = unsigned.pop("receipt_sha256")
        assert observed == orchestrator._canonical_sha256(unsigned)


def test_boundary_signal_enters_next_head_before_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, policy = _segment_and_policy(split_after_bt=True)
    runtime = _runtime(segment, monkeypatch)
    checkpoint_signal = CheckpointSignal()
    calls: list[str] = []

    def runner(admission, learner, **kwargs):
        calls.append(learner)
        if learner == "bt_mle":
            checkpoint_signal.requested = True
            checkpoint_signal.signal_name = "SIGUSR1"
            checkpoint_signal.received_at_utc = "2026-07-26T04:05:06Z"
            return _FakeFormalResult(
                admission,
                kwargs["runtime"],
                learner,
                kwargs["head_execution_slice"],
            )
        assert checkpoint_signal.requested is True
        raise CheckpointInterruption("test safe-boundary interruption")

    _patch_non_cuda_formal_boundary(monkeypatch, runner)
    root = _task_root(tmp_path)
    outcome = orchestrator.run_r3_primary_task_segment(
        segment,
        runtime=runtime,
        task_root=root,
        checkpoint_signal=checkpoint_signal,
        operational_policy=policy,
    )

    assert calls == ["bt_mle", "prorm_plus"]
    assert outcome.status == "continuation_required_after_safe_checkpoint"
    assert outcome.continuation_checkpoint == _checkpoint_ref("prorm_plus")
    assert outcome.external_scheduler_success_claimed is False
    assert not (root / "heads" / "prorm_plus" / "internal-head-result.json").exists()
    assert [event["event_type"] for event in _journal_events(root)] == [
        "head_started",
        "head_completed",
        "head_started",
        "continuation_required",
    ]


def test_result_then_receipt_crash_is_recovered_without_rerunning_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, policy = _segment_and_policy()
    runtime = _runtime(segment, monkeypatch)
    calls: list[str] = []

    def runner(admission, learner, **kwargs):
        calls.append(learner)
        return _FakeFormalResult(
            admission,
            kwargs["runtime"],
            learner,
            kwargs["head_execution_slice"],
        )

    _patch_non_cuda_formal_boundary(monkeypatch, runner)
    root = _task_root(tmp_path)
    original_publish = orchestrator._publish_no_overwrite
    injected = False

    def crash_before_first_receipt(path, raw, *, name):
        nonlocal injected
        if path.name == "head-completion.json" and not injected:
            injected = True
            raise OSError("injected receipt publication crash")
        return original_publish(path, raw, name=name)

    monkeypatch.setattr(
        orchestrator,
        "_publish_no_overwrite",
        crash_before_first_receipt,
    )
    with pytest.raises(OSError, match="injected"):
        orchestrator.run_r3_primary_task_segment(
            segment,
            runtime=runtime,
            task_root=root,
            checkpoint_signal=CheckpointSignal(),
            operational_policy=policy,
        )
    assert (root / "heads" / "bt_mle" / "internal-head-result.json").is_file()
    assert not (root / "heads" / "bt_mle" / "head-completion.json").exists()

    monkeypatch.setattr(orchestrator, "_publish_no_overwrite", original_publish)
    outcome = orchestrator.run_r3_primary_task_segment(
        segment,
        runtime=runtime,
        task_root=root,
        checkpoint_signal=CheckpointSignal(),
        operational_policy=policy,
    )

    assert calls == ["bt_mle", "prorm_plus"]
    assert outcome.status == "compute_complete_pending_external_scheduler_terminal"
    assert "head_completion_recovered_after_crash" in [
        event["event_type"] for event in _journal_events(root)
    ]


def test_terminal_commit_then_result_crash_recovers_without_rerunning_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, policy = _segment_and_policy()
    runtime = _runtime(segment, monkeypatch)
    calls: list[str] = []
    committed_terminals: dict[str, _FakeFormalResult] = {}

    def runner(admission, learner, **kwargs):
        calls.append(learner)
        result = _FakeFormalResult(
            admission,
            kwargs["runtime"],
            learner,
            kwargs["head_execution_slice"],
        )
        if learner == "bt_mle" and learner not in committed_terminals:
            committed_terminals[learner] = result
            raise OSError("injected crash after selected terminal commit")
        return result

    _patch_non_cuda_formal_boundary(monkeypatch, runner)
    monkeypatch.setattr(
        orchestrator,
        "latest_generation_is_selected_terminal",
        lambda store, *, admission, learner: learner in committed_terminals,
    )
    monkeypatch.setattr(
        orchestrator,
        "recover_formal_result_from_selected_terminal",
        lambda store, *, admission, learner: committed_terminals[learner],
    )
    root = _task_root(tmp_path)

    with pytest.raises(OSError, match="after selected terminal commit"):
        orchestrator.run_r3_primary_task_segment(
            segment,
            runtime=runtime,
            task_root=root,
            checkpoint_signal=CheckpointSignal(),
            operational_policy=policy,
        )
    assert not (root / "heads" / "bt_mle" / "internal-head-result.json").exists()

    outcome = orchestrator.run_r3_primary_task_segment(
        segment,
        runtime=runtime,
        task_root=root,
        checkpoint_signal=CheckpointSignal(),
        operational_policy=policy,
    )

    assert calls == ["bt_mle", "prorm_plus"]
    assert outcome.status == "compute_complete_pending_external_scheduler_terminal"
    assert (root / "heads" / "bt_mle" / "internal-head-result.json").is_file()
    assert (root / "heads" / "bt_mle" / "head-completion.json").is_file()
    assert "head_completion_recovered_after_crash" in [
        event["event_type"] for event in _journal_events(root)
    ]


def test_internal_result_tampering_is_detected_even_with_existing_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, policy = _segment_and_policy()
    runtime = _runtime(segment, monkeypatch)

    def runner(admission, learner, **kwargs):
        return _FakeFormalResult(
            admission,
            kwargs["runtime"],
            learner,
            kwargs["head_execution_slice"],
        )

    _patch_non_cuda_formal_boundary(monkeypatch, runner)
    root = _task_root(tmp_path)
    orchestrator.run_r3_primary_task_segment(
        segment,
        runtime=runtime,
        task_root=root,
        checkpoint_signal=CheckpointSignal(),
        operational_policy=policy,
    )
    path = root / "heads" / "bt_mle" / "internal-head-result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["controller_updates_executed"] = 101
    path.chmod(0o640)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical JSON|self-hash"):
        orchestrator.run_r3_primary_task_segment(
            segment,
            runtime=runtime,
            task_root=root,
            checkpoint_signal=CheckpointSignal(),
            operational_policy=policy,
        )


def test_completion_receipt_and_journal_tampering_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, policy = _segment_and_policy()
    runtime = _runtime(segment, monkeypatch)

    def runner(admission, learner, **kwargs):
        return _FakeFormalResult(
            admission,
            kwargs["runtime"],
            learner,
            kwargs["head_execution_slice"],
        )

    _patch_non_cuda_formal_boundary(monkeypatch, runner)
    root = _task_root(tmp_path)
    orchestrator.run_r3_primary_task_segment(
        segment,
        runtime=runtime,
        task_root=root,
        checkpoint_signal=CheckpointSignal(),
        operational_policy=policy,
    )
    receipt = root / "heads" / "bt_mle" / "head-completion.json"
    raw = receipt.read_bytes()
    receipt.chmod(0o640)
    receipt.write_bytes(raw.replace(b'"learner":"bt_mle"', b'"learner":"prorm_plus"'))
    with pytest.raises((ValueError, FileExistsError)):
        orchestrator.run_r3_primary_task_segment(
            segment,
            runtime=runtime,
            task_root=root,
            checkpoint_signal=CheckpointSignal(),
            operational_policy=policy,
        )

    # Use a separate clean run for the journal-chain mutation.
    other = (tmp_path / "other-task").resolve()
    other.mkdir()
    orchestrator.run_r3_primary_task_segment(
        segment,
        runtime=runtime,
        task_root=other,
        checkpoint_signal=CheckpointSignal(),
        operational_policy=policy,
    )
    event = other / "task-journal" / "event-00000001.json"
    event.chmod(0o640)
    event.write_bytes(event.read_bytes().replace(b"head_started", b"head_starteD"))
    with pytest.raises(ValueError, match="self-hash"):
        orchestrator.run_r3_primary_task_segment(
            segment,
            runtime=runtime,
            task_root=other,
            checkpoint_signal=CheckpointSignal(),
            operational_policy=policy,
        )


def test_later_segment_skips_completed_bt_and_starts_prorm_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, policy = _segment_and_policy(split_after_bt=True)
    runtime1 = _runtime(first, monkeypatch, job_id="300001")

    def first_runner(admission, learner, **kwargs):
        if learner == "prorm_plus":
            raise CheckpointInterruption("planned test boundary before first PR update")
        return _FakeFormalResult(
            admission,
            kwargs["runtime"],
            learner,
            kwargs["head_execution_slice"],
        )

    _patch_non_cuda_formal_boundary(monkeypatch, first_runner)
    root = _task_root(tmp_path)
    first_outcome = orchestrator.run_r3_primary_task_segment(
        first,
        runtime=runtime1,
        task_root=root,
        checkpoint_signal=CheckpointSignal(),
        operational_policy=policy,
    )
    assert first_outcome.status == "continuation_required_after_safe_checkpoint"
    continuable_terminal = r3_test_support.make_continuable_primary_terminal(
        predecessor=first,
        operational_bundle=policy,
    )
    continuation = validate_continuation_evidence(
        predecessor=first,
        continuable_terminal=continuable_terminal,
    )
    second = admit_primary_segment(
        design=first.design,
        materialization_capability=(
            r3_test_support.make_shared_gate_p_evidence().materialization_capability
        ),
        task_id=first.task_id,
        seed=first.seed,
        segment_index=2,
        continuation_evidence=continuation,
    )
    runtime2 = _runtime(second, monkeypatch, job_id="300002")

    second_calls: list[tuple[str, dict[str, object]]] = []

    def second_runner(admission, learner, **kwargs):
        second_calls.append((learner, kwargs["head_execution_slice"]))
        return _FakeFormalResult(
            admission,
            kwargs["runtime"],
            learner,
            kwargs["head_execution_slice"],
        )

    monkeypatch.setattr(
        orchestrator,
        "run_formal_r3_primary_head_segment",
        second_runner,
    )
    outcome = orchestrator.run_r3_primary_task_segment(
        second,
        runtime=runtime2,
        task_root=root,
        checkpoint_signal=CheckpointSignal(),
        operational_policy=policy,
    )

    assert outcome.status == "compute_complete_pending_external_scheduler_terminal"
    assert [learner for learner, _ in second_calls] == ["prorm_plus"]
    assert second_calls[0][1]["fresh_or_resume"] == "fresh"
    assert second_calls[0][1]["start_completed_updates"] == 0
    assert second_calls[0][1]["predecessor_checkpoint"] is None
    segment2_events = [
        event["event_type"] for event in _journal_events(root) if event["segment_index"] == 2
    ]
    assert segment2_events == [
        "head_completion_revalidated_and_skipped",
        "head_started",
        "head_completed",
        "segment_compute_complete",
    ]


def test_current_segment_checkpoint_discovery_returns_continuation_not_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, policy = _segment_and_policy(split_after_bt=True)
    runtime = _runtime(segment, monkeypatch)
    root = _task_root(tmp_path)

    def crash_after_head_start(*args, **kwargs):
        raise RuntimeError("injected process crash after durable head start")

    _patch_non_cuda_formal_boundary(monkeypatch, crash_after_head_start)
    with pytest.raises(RuntimeError, match="injected process crash"):
        orchestrator.run_r3_primary_task_segment(
            segment,
            runtime=runtime,
            task_root=root,
            checkpoint_signal=CheckpointSignal(),
            operational_policy=policy,
        )

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("discovered current checkpoint must not be rerun")

    monkeypatch.setattr(
        orchestrator,
        "run_formal_r3_primary_head_segment",
        forbidden_runner,
    )
    monkeypatch.setattr(
        orchestrator,
        "_checkpoint_position",
        lambda **kwargs: (
            "current_segment_checkpoint",
            _checkpoint_ref(kwargs["learner"]),
        ),
    )
    outcome = orchestrator.run_r3_primary_task_segment(
        segment,
        runtime=runtime,
        task_root=root,
        checkpoint_signal=CheckpointSignal(),
        operational_policy=policy,
    )

    assert outcome.status == "continuation_required_after_safe_checkpoint"
    assert outcome.external_scheduler_success_claimed is False
    payload = json.loads(outcome.artifact_path.read_text(encoding="utf-8"))
    assert payload["continuation_reason"] == ("checkpoint_discovered_after_process_crash")
    assert payload["external_scheduler_terminal_required"] is True


def test_crash_before_any_safe_checkpoint_cannot_restart_same_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, policy = _segment_and_policy()
    runtime = _runtime(segment, monkeypatch)
    root = _task_root(tmp_path)
    calls = 0

    def crashing_runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("crash before checkpoint")

    _patch_non_cuda_formal_boundary(monkeypatch, crashing_runner)
    with pytest.raises(RuntimeError, match="crash before checkpoint"):
        orchestrator.run_r3_primary_task_segment(
            segment,
            runtime=runtime,
            task_root=root,
            checkpoint_signal=CheckpointSignal(),
            operational_policy=policy,
        )
    with pytest.raises(RuntimeError, match="before any recoverable"):
        orchestrator.run_r3_primary_task_segment(
            segment,
            runtime=runtime,
            task_root=root,
            checkpoint_signal=CheckpointSignal(),
            operational_policy=policy,
        )
    assert calls == 1


def test_plan_cursor_mismatch_fails_before_any_task_artifact(
    tmp_path: Path,
) -> None:
    segment, policy = _segment_and_policy()
    boundary = dict(policy.segment_boundaries[0])
    end = dict(boundary["end_boundary"])
    end["global_safe_block"] = int(end["global_safe_block"]) + 1
    root = _task_root(tmp_path)
    with pytest.raises(ValueError, match="canonical R3 plan cursor"):
        orchestrator._validate_plan_cursor(
            end,
            name="tampered Gate-P segment end cursor",
            maximum_updates_per_head=(segment.design.science.settings.convergence.max_steps),
            audit_cadence_updates=policy.audit_cadence_updates,
        )
    assert list(root.iterdir()) == []


def test_operational_policy_is_exact_reopened_sealed_bundle(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _, live_plan = _published_operational_bundle(tmp_path_factory)
    segment = _segment_for_operational_bundle(policy)
    _slurm_environment(monkeypatch)
    runtime = capture_slurm_segment_runtime(
        segment,
        requested_walltime_seconds=(policy.requested_walltime_seconds_per_segment),
    )
    assert (
        orchestrator._validate_operational_policy(
            policy,
            admission=segment,
            runtime=runtime,
        )
        is policy
    )
    assert not hasattr(policy, "formal")
    science_interval = segment.design.science.settings.convergence.check_interval
    assert policy.audit_cadence_updates == science_interval
    assert policy.durable_checkpoint_cadence_updates % science_interval == 0
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        policy.durable_checkpoint_cadence_updates += 1  # type: ignore[misc]

    with pytest.raises(TypeError, match="VerifiedGatePOperationalBundle"):
        orchestrator._validate_operational_policy(
            policy.to_dict(),
            admission=segment,
            runtime=runtime,
        )
    with pytest.raises(TypeError, match="VerifiedGatePOperationalBundle"):
        orchestrator._validate_operational_policy(
            live_plan,
            admission=segment,
            runtime=runtime,
        )
    with pytest.raises(TypeError, match="produced by reopen"):
        replace(policy, _factory_token=object())

    independent_policy = _independent_operational_bundle()
    mismatched_segment = _segment_for_operational_bundle(independent_policy)
    mismatched_runtime = capture_slurm_segment_runtime(
        mismatched_segment,
        requested_walltime_seconds=(policy.requested_walltime_seconds_per_segment),
    )
    with pytest.raises(ValueError, match="authorized artifact"):
        orchestrator._validate_operational_policy(
            policy,
            admission=mismatched_segment,
            runtime=mismatched_runtime,
        )


def test_later_segment_cannot_be_admitted_without_continuation_capability() -> None:
    first, _ = _segment_and_policy(split_after_bt=True)
    with pytest.raises(
        (TypeError, ValueError),
        match="Continuation|continuation",
    ):
        admit_primary_segment(
            design=first.design,
            materialization_capability=(
                r3_test_support.make_shared_gate_p_evidence().materialization_capability
            ),
            task_id=first.task_id,
            seed=first.seed,
            segment_index=2,
        )


def test_segment_outcome_cannot_be_caller_constructed(tmp_path: Path) -> None:
    artifact = tmp_path / "outcome.json"
    artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(TypeError, match="durable evidence"):
        orchestrator.R3PrimarySegmentOutcome(
            status="compute_complete_pending_external_scheduler_terminal",
            outcome_sha256="a" * 64,
            file_sha256="b" * 64,
            artifact_path=artifact,
            continuation_checkpoint=None,
        )


def test_segment_outcome_seal_is_not_inherited_by_dataclass_replace(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "outcome.json"
    artifact.write_text("{}\n", encoding="utf-8")
    outcome = orchestrator.R3PrimarySegmentOutcome(
        status="compute_complete_pending_external_scheduler_terminal",
        outcome_sha256="a" * 64,
        file_sha256="b" * 64,
        artifact_path=artifact,
        continuation_checkpoint=None,
        _factory_token=orchestrator._FACTORY_TOKEN,
    )

    with pytest.raises(TypeError, match="durable evidence"):
        replace(outcome, outcome_sha256="c" * 64)


def test_head_execution_slice_cannot_be_caller_constructed() -> None:
    with pytest.raises(TypeError, match="derived from a Gate-P plan"):
        orchestrator.PrimaryHeadExecutionSlice(
            head="bt_mle",
            fresh_or_resume="fresh",
            start_completed_updates=0,
            end_completed_updates_inclusive=1,
            max_safe_update_blocks_to_execute=1,
            safe_update_blocks_consumed_before_head=0,
            safe_update_blocks_available_to_head=1,
            slice_sha256="a" * 64,
            _payload={},
        )


def test_head_execution_slice_seal_is_not_inherited_by_dataclass_replace() -> None:
    payload: dict[str, object] = {}
    execution_slice = orchestrator.PrimaryHeadExecutionSlice(
        head="bt_mle",
        fresh_or_resume="fresh",
        start_completed_updates=0,
        end_completed_updates_inclusive=20,
        max_safe_update_blocks_to_execute=1,
        safe_update_blocks_consumed_before_head=0,
        safe_update_blocks_available_to_head=1,
        slice_sha256=orchestrator._canonical_sha256(payload),
        _payload=payload,
        _factory_token=orchestrator._FACTORY_TOKEN,
    )
    forged_payload = {"forged": True}

    with pytest.raises(TypeError, match="derived from a Gate-P plan"):
        replace(
            execution_slice,
            _payload=forged_payload,
            slice_sha256=orchestrator._canonical_sha256(forged_payload),
        )
