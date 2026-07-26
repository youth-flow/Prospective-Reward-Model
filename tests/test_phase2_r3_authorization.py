from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from smart_reward import phase2_r3_authorization as authorization
from smart_reward import phase2_r3_terminal as terminal
from smart_reward.phase2_r3_artifacts import canonical_json_bytes, publish_canonical_artifact
from smart_reward.phase2_r3_identity import (
    R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA,
    VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
)
from smart_reward.phase2_r3_orchestrator import (
    SEGMENT_OUTCOME_SCHEMA,
    primary_outcome_semantic_sha256,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _raw_sacct(*, task_id: int, array_job_id: str, job_id: str) -> bytes:
    return (
        "|".join(
            (
                f"{array_job_id}_{task_id}",
                job_id,
                "COMPLETED",
                "0:0",
                "0:0",
                "hpc4",
                "sigroup",
                "gpu-l20",
                "l20_qos",
                "1",
                "4",
                "billing=4,cpu=4,gres/gpu=1,mem=2G,node=1",
                "billing=4,cpu=4,gres/gpu=1,mem=2G,node=1,gres/gpu:l20=1",
                "137",
            )
        )
        + "\n"
    ).encode("utf-8")


def _fake_bundle(root: Path) -> SimpleNamespace:
    bundle_parent = root / "runs" / "phase2-recovery-r3" / "gatep" / "fixture" / "gatep-attempt-001"
    bundle_parent.mkdir(parents=True, exist_ok=True)
    artifact = publish_canonical_artifact(
        (bundle_parent / "gatep-operational-bundle.json").resolve(),
        {"fixture": "authorization transport only"},
    )
    return SimpleNamespace(
        artifact_path=artifact.artifact_path,
        file_sha256=artifact.file_sha256,
        size_bytes=artifact.size_bytes,
        partition="gpu-l20",
        gpu_name="NVIDIA L20",
        gpus_per_task=1,
        cpus_per_task=4,
        memory_bytes=2 * 1024**3,
        requested_walltime_seconds_per_segment=3600,
        resource_plan_sha256=_digest("resource-plan"),
        bundle_semantic_sha256=_digest("bundle-semantic"),
        profile_run_sha256=_digest("profile-run"),
        formal_profile_sha256=_digest("formal-profile"),
        max_scheduler_segments=3,
        validate_integrity=lambda: None,
    )


def _closure(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundle: SimpleNamespace,
    task_id: int,
    seed: int,
    segment_index: int,
    status: str,
    logical_run_id: str,
    continuation_evidence_sha256: str | None = None,
) -> terminal.PrimarySegmentRuntimeClosure:
    head_runs = [
        {"head": "bt_mle", "head_run_id": _digest(f"{task_id}:bt")},
        {"head": "prorm_plus", "head_run_id": _digest(f"{task_id}:prorm-plus")},
    ]
    scheduler_segment_id = authorization._expected_scheduler_segment_id(
        logical_run_id=logical_run_id,
        segment_index=segment_index,
    )
    admission_unsigned: dict[str, object] = {
        "schema_version": R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA,
        "design_sha256": _digest("shared-r3-design"),
        "materialization_attestation_sha256": _digest(f"materialization:{task_id}"),
        "task_id": task_id,
        "seed": seed,
        "segment_index": segment_index,
        "logical_run_id": logical_run_id,
        "head_runs": head_runs,
        "scheduler_segment_id": scheduler_segment_id,
        "start_mode": (
            "fresh_zero_head_fresh_adamw"
            if segment_index == 1
            else "verified_state_complete_continuation"
        ),
        "continuation_evidence_sha256": continuation_evidence_sha256,
    }
    admission = {
        **admission_unsigned,
        "admission_sha256": terminal._canonical_sha256(admission_unsigned),
    }
    array_job_id = str(510000 + segment_index)
    job_id = str(520000 + task_id * 10 + segment_index)
    runtime_unsigned: dict[str, object] = {
        "schema_version": "phase2-recovery-r3-slurm-segment-runtime/v1",
        "design_sha256": admission["design_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "scheduler_segment_id": scheduler_segment_id,
        "segment_index": segment_index,
        "task_id": task_id,
        "seed": seed,
        "cluster": "hpc4",
        "job_id": job_id,
        "array_job_id": array_job_id,
        "array_task_id": task_id,
        "account": "sigroup",
        "partition": "gpu-l20",
        "requested_walltime_seconds": 3600,
        "captured_monotonic_ns": 1000 + segment_index,
    }
    runtime = {
        **runtime_unsigned,
        "runtime_sha256": terminal._canonical_sha256(runtime_unsigned),
    }
    continuable = status == "continuation_required_after_safe_checkpoint"
    checkpoint = (
        {
            "schema_version": VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
            "artifact_sha256": _digest(f"checkpoint:{task_id}:{segment_index}"),
            "role": VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
        }
        if continuable
        else None
    )
    completed = (
        []
        if continuable
        else [
            {
                "learner": run["head"],
                "head_run_id": run["head_run_id"],
                "completion_receipt_sha256": _digest(f"completion:{task_id}:{run['head']}"),
            }
            for run in head_runs
        ]
    )
    outcome_unsigned: dict[str, object] = {
        "schema_version": SEGMENT_OUTCOME_SCHEMA,
        "status": status,
        "design_sha256": admission["design_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "logical_run_id": logical_run_id,
        "scheduler_segment_id": scheduler_segment_id,
        "runtime_sha256": runtime["runtime_sha256"],
        "segment_index": segment_index,
        "task_id": task_id,
        "seed": seed,
        "gate_p_resource_plan_sha256": bundle.resource_plan_sha256,
        "completed_heads": completed,
        "active_learner": "bt_mle" if continuable else None,
        "continuation_checkpoint": checkpoint,
        "continuation_reason": "safe scheduler boundary" if continuable else None,
        "all_primary_heads_compute_complete": not continuable,
        "external_scheduler_terminal_required": True,
        "external_scheduler_success_claimed": False,
        "r3_success_authorization_created": False,
        "information_boundary": "train_only_head_free_segment_outcome",
    }
    outcome = {
        **outcome_unsigned,
        "outcome_sha256": primary_outcome_semantic_sha256(outcome_unsigned),
    }
    monkeypatch.setattr(terminal, "_validated_operational_bundle", lambda value: value)
    monkeypatch.setattr(
        terminal,
        "_validate_primary_dependencies",
        lambda admitted, *, runtime, outcome, operational_bundle: (
            admitted,
            runtime,
            outcome,
            operational_bundle,
            globals()["_CURRENT_OUTCOME"],
        ),
    )
    globals()["_CURRENT_OUTCOME"] = outcome
    attempt = (
        root / "runs" / "phase2-recovery-r3" / f"attempt-task-{task_id}-segment-{segment_index}"
    )
    (attempt / "runtime-closures").mkdir(parents=True, exist_ok=True)
    (attempt / "terminal-evidence").mkdir(exist_ok=True)
    return terminal.publish_primary_segment_runtime_closure(
        (attempt / "runtime-closures" / f"task-{task_id}.json").resolve(),
        admission=SimpleNamespace(to_dict=lambda: admission),  # type: ignore[arg-type]
        runtime=SimpleNamespace(to_dict=lambda: runtime),  # type: ignore[arg-type]
        outcome=SimpleNamespace(),  # type: ignore[arg-type]
        operational_bundle=bundle,  # type: ignore[arg-type]
    )


def _terminal(
    root: Path,
    *,
    bundle: SimpleNamespace,
    closure: terminal.PrimarySegmentRuntimeClosure,
    task_id: int,
    segment_index: int,
) -> authorization.PrimaryTerminalCapability:
    runtime = closure.runtime_payload
    raw = _raw_sacct(
        task_id=task_id,
        array_job_id=str(runtime["array_job_id"]),
        job_id=str(runtime["job_id"]),
    )
    inspection = terminal.inspect_sacct_terminal_bytes(
        raw,
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
    )
    directory = (
        closure.artifact_path.parent.parent
        / "terminal-evidence"
        / f"task-{task_id}-segment-{segment_index}"
    )
    if closure.continuation_required:
        return terminal.produce_continuable_primary_terminal(
            bundle,  # type: ignore[arg-type]
            runtime_closure=closure,
            inspection=inspection,
            evidence_directory=directory,
        )
    return terminal.produce_completed_primary_terminal(
        bundle,  # type: ignore[arg-type]
        runtime_closure=closure,
        inspection=inspection,
        evidence_directory=directory,
    )


def _histories(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    tuple[authorization.PrimaryTerminalCapability, ...],
    tuple[authorization.PrimaryTerminalCapability, ...],
    tuple[authorization.PrimaryTerminalCapability, ...],
]:
    bundle = _fake_bundle(root)
    first_closure = _closure(
        root,
        monkeypatch,
        bundle=bundle,
        task_id=0,
        seed=20260801,
        segment_index=1,
        status="continuation_required_after_safe_checkpoint",
        logical_run_id=_digest("logical-run:0"),
    )
    first = _terminal(
        root,
        bundle=bundle,
        closure=first_closure,
        task_id=0,
        segment_index=1,
    )
    assert type(first) is terminal.ContinuablePrimaryTerminalCapability
    second_closure = _closure(
        root,
        monkeypatch,
        bundle=bundle,
        task_id=0,
        seed=20260801,
        segment_index=2,
        status="compute_complete_pending_external_scheduler_terminal",
        logical_run_id=_digest("logical-run:0"),
        continuation_evidence_sha256=(authorization._expected_continuation_evidence_sha256(first)),
    )
    second = _terminal(
        root,
        bundle=bundle,
        closure=second_closure,
        task_id=0,
        segment_index=2,
    )
    histories: list[tuple[authorization.PrimaryTerminalCapability, ...]] = [(first, second)]
    for task_id, seed in ((1, 20260802), (2, 20260803)):
        closure = _closure(
            root,
            monkeypatch,
            bundle=bundle,
            task_id=task_id,
            seed=seed,
            segment_index=1,
            status="compute_complete_pending_external_scheduler_terminal",
            logical_run_id=_digest(f"logical-run:{task_id}"),
        )
        histories.append(
            (
                _terminal(
                    root,
                    bundle=bundle,
                    closure=closure,
                    task_id=task_id,
                    segment_index=1,
                ),
            )
        )
    return histories[0], histories[1], histories[2]


def test_three_seed_authorization_revalidates_every_segment_and_is_no_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    output_parent = root / authorization.R3_SUCCESS_AUTHORIZATION_RELATIVE.parent
    output_parent.mkdir(parents=True)
    histories = _histories(root, monkeypatch)
    bundle = histories[0][0].operational_bundle
    monkeypatch.setattr(
        authorization,
        "reopen_verified_gate_p_operational_bundle",
        lambda _path, *, expected_file_sha256: bundle,
    )

    artifact = authorization.publish_r3_success_authorization(
        histories,
        project_root=root,
    )
    verified = authorization.verify_r3_success_authorization(
        artifact.artifact_path,
        expected_sha256=artifact.file_sha256,
        project_root=root,
    )

    assert verified["schema_version"] == authorization.R3_SUCCESS_AUTHORIZATION_SCHEMA
    assert verified["ordered_seeds"] == [20260801, 20260802, 20260803]
    assert [len(source["segments"]) for source in verified["sources"]] == [2, 1, 1]
    assert verified["transport_boundary"] == authorization._TRANSPORT_BOUNDARY
    assert verified["recovery_outputs_reusable"] is False
    assert verified["gate_r_passed"] is True
    assert verified["fresh_calibration_authorized"] is False
    assert verified["authorized_next_action"] == "await_separate_gate_c_authorization"
    with pytest.raises(FileExistsError, match="overwrite"):
        authorization.publish_r3_success_authorization(
            histories,
            project_root=root,
        )
    with pytest.raises(ValueError, match="exactly three"):
        authorization.build_r3_success_authorization(
            histories[:2],
            project_root=root,
        )


def test_authorization_rejects_broken_continuation_partial_and_old_r2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    (root / authorization.R3_SUCCESS_AUTHORIZATION_RELATIVE.parent).mkdir(parents=True)
    histories = _histories(root, monkeypatch)
    bundle = histories[0][0].operational_bundle

    bad_root = (root / "broken").resolve()
    bad_root.mkdir()
    bad_bundle = _fake_bundle(bad_root)
    bad_first_closure = _closure(
        bad_root,
        monkeypatch,
        bundle=bad_bundle,
        task_id=0,
        seed=20260801,
        segment_index=1,
        status="continuation_required_after_safe_checkpoint",
        logical_run_id=_digest("bad-logical"),
    )
    bad_first = _terminal(
        bad_root,
        bundle=bad_bundle,
        closure=bad_first_closure,
        task_id=0,
        segment_index=1,
    )
    bad_second_closure = _closure(
        bad_root,
        monkeypatch,
        bundle=bad_bundle,
        task_id=0,
        seed=20260801,
        segment_index=2,
        status="compute_complete_pending_external_scheduler_terminal",
        logical_run_id=_digest("bad-logical"),
        continuation_evidence_sha256="0" * 64,
    )
    bad_second = _terminal(
        bad_root,
        bundle=bad_bundle,
        closure=bad_second_closure,
        task_id=0,
        segment_index=2,
    )
    with pytest.raises(ValueError, match="continuation chain"):
        authorization.build_r3_success_authorization(
            ((bad_first, bad_second), histories[1], histories[2]),
            project_root=root,
        )

    monkeypatch.setattr(
        authorization,
        "reopen_verified_gate_p_operational_bundle",
        lambda _path, *, expected_file_sha256: bundle,
    )
    artifact = authorization.publish_r3_success_authorization(
        histories,
        project_root=root,
    )
    tampered = artifact.payload
    tampered["fresh_calibration_authorized"] = True
    unsigned = dict(tampered)
    unsigned.pop("authorization_sha256")
    tampered["authorization_sha256"] = authorization._artifact_semantic_sha256(unsigned)
    if os.name == "posix":
        os.chmod(artifact.artifact_path, stat.S_IRUSR | stat.S_IWUSR)
    artifact.artifact_path.write_bytes(canonical_json_bytes(tampered))
    if os.name == "posix":
        os.chmod(artifact.artifact_path, stat.S_IRUSR | stat.S_IRGRP)
    tampered_file_sha = hashlib.sha256(artifact.artifact_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Gate-R-only boundary"):
        authorization.verify_r3_success_authorization(
            artifact.artifact_path,
            expected_sha256=tampered_file_sha,
            project_root=root,
        )

    legacy_root = (root / "legacy").resolve()
    legacy_output = legacy_root / authorization.R3_SUCCESS_AUTHORIZATION_RELATIVE
    legacy_output.parent.mkdir(parents=True)
    legacy = publish_canonical_artifact(
        legacy_output,
        {
            "schema_version": "prorm-phase2-recovery-success-authorization/v1",
            "full_calibration_authorized": True,
        },
    )
    with pytest.raises(ValueError, match="closed field set"):
        authorization.verify_r3_success_authorization(
            legacy.artifact_path,
            expected_sha256=legacy.file_sha256,
            project_root=legacy_root,
        )


def test_authorization_reopen_rejects_noncanonical_terminal_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    output = root / authorization.R3_SUCCESS_AUTHORIZATION_RELATIVE
    output.parent.mkdir(parents=True)
    histories = _histories(root, monkeypatch)
    payload = authorization.build_r3_success_authorization(
        histories,
        project_root=root,
    )
    bundle = histories[0][0].operational_bundle
    monkeypatch.setattr(
        authorization,
        "reopen_verified_gate_p_operational_bundle",
        lambda _path, *, expected_file_sha256: bundle,
    )
    first_segment = payload["sources"][0]["segments"][0]  # type: ignore[index]
    original = root / first_segment["terminal_evidence_directory"]  # type: ignore[index]
    wrong = root / "runs" / "phase2-recovery-r3" / "terminal-evidence" / "wrong-name"
    wrong.parent.mkdir(parents=True)
    shutil.copytree(original, wrong)
    first_segment["terminal_evidence_directory"] = wrong.relative_to(root).as_posix()  # type: ignore[index]
    payload["terminal_set_sha256"] = authorization._artifact_semantic_sha256(
        {"sources": payload["sources"]}
    )
    unsigned = dict(payload)
    unsigned.pop("authorization_sha256")
    payload["authorization_sha256"] = authorization._artifact_semantic_sha256(unsigned)
    artifact = publish_canonical_artifact(output, payload)
    with pytest.raises(ValueError, match="terminal evidence must be"):
        authorization.verify_r3_success_authorization(
            artifact.artifact_path,
            expected_sha256=artifact.file_sha256,
            project_root=root,
        )
